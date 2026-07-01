from __future__ import annotations

import csv
import json
import math
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


META_ROOT = Path(r"A:\FDT_TO _PROCESS\meta_training")
OCR_CACHE_ROOT = Path(r"A:\RealForm\processed\meta_training_ocr_cache")
OUTPUT_ROOT = Path(r"A:\RealForm\processed\meta_training_bm25_benchmark")
DICTIONARY_PATH = Path(r"A:\RealForm\processed\synthetic_fill_images\dictionary_500_words.json")
CORRUPTION_RATE = 0.25
BM25_K1 = 1.5
BM25_B = 0.75


@dataclass(frozen=True)
class ImageRow:
    template_name: str
    image_path: Path
    role: str


@dataclass(frozen=True)
class Bm25Index:
    template_names: list[str]
    doc_tokens: list[list[str]]
    doc_lengths: list[int]
    average_doc_length: float
    doc_frequencies: dict[str, int]
    term_frequencies: list[Counter[str]]


TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def tokenize_words(words: list[str]) -> list[str]:
    tokens: list[str] = []
    for word in words:
        tokens.extend(match.group(0).lower() for match in TOKEN_RE.finditer(str(word)))
    return tokens or ["empty"]


def load_word_bank() -> list[str]:
    fallback = [
        "name",
        "date",
        "address",
        "phone",
        "account",
        "signature",
        "office",
        "total",
        "state",
        "number",
    ]
    if not DICTIONARY_PATH.exists():
        return fallback
    payload = read_json(DICTIONARY_PATH)
    if isinstance(payload, list):
        words = [str(item).strip() for item in payload if str(item).strip()]
    elif isinstance(payload, dict) and isinstance(payload.get("words"), list):
        words = [str(item).strip() for item in payload["words"] if str(item).strip()]
    elif isinstance(payload, dict):
        words = [str(key).strip() for key in payload.keys() if str(key).strip()]
    else:
        words = []
    clean_words = [word for word in words if TOKEN_RE.search(word)]
    return clean_words or fallback


def discover_meta_images() -> tuple[list[ImageRow], list[ImageRow]]:
    originals: list[ImageRow] = []
    queries: list[ImageRow] = []
    for template_dir in sorted([path for path in META_ROOT.iterdir() if path.is_dir()], key=lambda path: path.name.lower()):
        original_path = template_dir / "original_template.png"
        if not original_path.exists():
            continue
        page_paths = sorted(template_dir.glob("page_*.png"), key=lambda path: path.name.lower())
        if not page_paths:
            continue
        originals.append(ImageRow(template_dir.name, original_path.resolve(), "original"))
        for page_path in page_paths:
            queries.append(ImageRow(template_dir.name, page_path.resolve(), "query"))
    if not originals or not queries:
        raise RuntimeError(f"No usable meta_training images found under {META_ROOT}")
    return originals, queries


def ocr_path(row: ImageRow) -> Path:
    return OCR_CACHE_ROOT / row.template_name / f"{row.image_path.stem}.ocr.json"


def load_ocr_words(row: ImageRow) -> list[str]:
    path = ocr_path(row)
    if not path.exists():
        raise FileNotFoundError(f"Missing OCR cache: {path}")
    payload = read_json(path)
    return [str(word) for word in payload.get("words", [])]


def corrupt_words(words: list[str], word_bank: list[str], seed_key: str) -> tuple[list[str], int]:
    if not words:
        return words, 0
    rng = random.Random(seed_key)
    count = max(1, int(round(len(words) * CORRUPTION_RATE)))
    count = min(count, len(words))
    indices = rng.sample(range(len(words)), k=count)
    corrupted = list(words)
    for index in indices:
        corrupted[index] = rng.choice(word_bank)
    return corrupted, count


def build_index(originals: list[ImageRow]) -> Bm25Index:
    template_names: list[str] = []
    doc_tokens: list[list[str]] = []
    term_frequencies: list[Counter[str]] = []
    doc_frequencies: dict[str, int] = {}

    for row in originals:
        tokens = tokenize_words(load_ocr_words(row))
        template_names.append(row.template_name)
        doc_tokens.append(tokens)
        counter = Counter(tokens)
        term_frequencies.append(counter)
        for token in counter.keys():
            doc_frequencies[token] = doc_frequencies.get(token, 0) + 1

    doc_lengths = [len(tokens) for tokens in doc_tokens]
    average_doc_length = sum(doc_lengths) / max(1, len(doc_lengths))
    return Bm25Index(
        template_names=template_names,
        doc_tokens=doc_tokens,
        doc_lengths=doc_lengths,
        average_doc_length=average_doc_length,
        doc_frequencies=doc_frequencies,
        term_frequencies=term_frequencies,
    )


def bm25_scores(index: Bm25Index, query_tokens: list[str]) -> list[float]:
    query_terms = Counter(query_tokens)
    total_docs = len(index.doc_tokens)
    scores: list[float] = []
    for doc_index, term_frequency in enumerate(index.term_frequencies):
        doc_length = index.doc_lengths[doc_index]
        score = 0.0
        length_factor = BM25_K1 * (1.0 - BM25_B + BM25_B * doc_length / max(index.average_doc_length, 1e-12))
        for token, query_count in query_terms.items():
            frequency = term_frequency.get(token, 0)
            if frequency == 0:
                continue
            document_frequency = index.doc_frequencies.get(token, 0)
            idf = math.log(1.0 + (total_docs - document_frequency + 0.5) / (document_frequency + 0.5))
            score += query_count * idf * (frequency * (BM25_K1 + 1.0)) / (frequency + length_factor)
        scores.append(score)
    return scores


def evaluate(
    index: Bm25Index,
    queries: list[ImageRow],
    *,
    corrupt: bool,
    word_bank: list[str],
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    corruption_rows: list[dict[str, Any]] = []
    top1 = 0
    top5 = 0
    top10 = 0
    reciprocal_ranks: list[float] = []

    for query in queries:
        words = load_ocr_words(query)
        corrupted_count = 0
        if corrupt:
            words, corrupted_count = corrupt_words(words, word_bank, f"{query.template_name}:{query.image_path.name}")
            corruption_rows.append(
                {
                    "template_name": query.template_name,
                    "image_path": str(query.image_path),
                    "word_count": len(words),
                    "corrupted_words": corrupted_count,
                }
            )
        query_tokens = tokenize_words(words)
        scores = bm25_scores(index, query_tokens)
        order = sorted(range(len(scores)), key=lambda index_id: scores[index_id], reverse=True)
        ranked_names = [index.template_names[index_id] for index_id in order]
        rank = ranked_names.index(query.template_name) + 1 if query.template_name in ranked_names else len(ranked_names) + 1
        top1 += int(rank == 1)
        top5 += int(rank <= 5)
        top10 += int(rank <= 10)
        reciprocal_ranks.append(1.0 / rank)
        rows.append(
            {
                "template_name": query.template_name,
                "query_image": str(query.image_path),
                "rank": rank,
                "top1": int(rank == 1),
                "top5": int(rank <= 5),
                "top10": int(rank <= 10),
                "best_match": ranked_names[0],
                "best_score": scores[order[0]],
                "true_score": scores[index.template_names.index(query.template_name)],
                "query_token_count": len(query_tokens),
                "corrupted_words": corrupted_count,
            }
        )

    with (output_dir / "per_query_rankings.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    per_template: list[dict[str, Any]] = []
    for template_name in sorted({row["template_name"] for row in rows}, key=str.lower):
        template_rows = [row for row in rows if row["template_name"] == template_name]
        per_template.append(
            {
                "template_name": template_name,
                "query_count": len(template_rows),
                "retrieval_at_1": sum(row["top1"] for row in template_rows) / len(template_rows),
                "retrieval_at_5": sum(row["top5"] for row in template_rows) / len(template_rows),
                "retrieval_at_10": sum(row["top10"] for row in template_rows) / len(template_rows),
                "mean_rank": sum(row["rank"] for row in template_rows) / len(template_rows),
            }
        )
    with (output_dir / "per_template_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_template[0].keys()))
        writer.writeheader()
        writer.writerows(per_template)

    if corrupt:
        write_json(output_dir / "corruption_manifest.json", corruption_rows)

    summary = {
        "mode": "corrupt25" if corrupt else "clean_ocr",
        "gallery_templates": len(index.template_names),
        "query_images": len(queries),
        "retrieval_at_1": top1 / len(queries),
        "retrieval_at_5": top5 / len(queries),
        "retrieval_at_10": top10 / len(queries),
        "mrr": sum(reciprocal_ranks) / len(reciprocal_ranks),
        "bm25_k1": BM25_K1,
        "bm25_b": BM25_B,
        "corruption_rate": CORRUPTION_RATE if corrupt else 0.0,
        "meta_root": str(META_ROOT),
        "ocr_cache_root": str(OCR_CACHE_ROOT),
    }
    write_json(output_dir / "benchmark_summary.json", summary)
    return summary


def main() -> None:
    originals, queries = discover_meta_images()
    missing = [str(ocr_path(row)) for row in [*originals, *queries] if not ocr_path(row).exists()]
    if missing:
        write_json(OUTPUT_ROOT / "missing_ocr_files.json", missing)
        raise FileNotFoundError(f"Missing {len(missing)} OCR cache file(s). See {OUTPUT_ROOT / 'missing_ocr_files.json'}")

    word_bank = load_word_bank()
    index = build_index(originals)
    clean_summary = evaluate(index, queries, corrupt=False, word_bank=word_bank, output_dir=OUTPUT_ROOT / "clean_ocr")
    corrupt_summary = evaluate(index, queries, corrupt=True, word_bank=word_bank, output_dir=OUTPUT_ROOT / "corrupt25")
    contrast = {
        "clean_ocr": clean_summary,
        "corrupt25": corrupt_summary,
        "delta": {
            key: corrupt_summary[key] - clean_summary[key]
            for key in ["retrieval_at_1", "retrieval_at_5", "retrieval_at_10", "mrr"]
        },
    }
    write_json(OUTPUT_ROOT / "contrast_summary.json", contrast)
    print(json.dumps(contrast, indent=2), flush=True)


if __name__ == "__main__":
    main()
