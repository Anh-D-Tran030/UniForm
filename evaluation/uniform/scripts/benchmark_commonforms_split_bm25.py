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

import numpy as np
from scipy import sparse

SPLIT_OCR_ROOT = Path(r"A:\RealForm\processed\synthetic_fill_images_10k_folders_degraded_ocr_split_9400_300_300_augmented")
ORIGINAL_OCR_ROOT = Path(r"A:\RealForm\processed\CommonFormsEnglish_original_ocr_cache_projection")
OUTPUT_ROOT = Path(r"A:\RealForm\processed\commonforms_split_bm25_corrupt25_benchmark")
DICTIONARY_PATH = Path(r"A:\RealForm\processed\synthetic_fill_images\dictionary_500_words.json")
CORRUPTION_RATE = 0.25
BM25_K1 = 1.5
BM25_B = 0.75
TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


@dataclass(frozen=True)
class QueryRow:
    template_name: str
    ocr_path: Path
    image_name: str


@dataclass
class Bm25Index:
    template_names: list[str]
    template_to_id: dict[str, int]
    vocab: dict[str, int]
    bm25_matrix: sparse.csr_matrix


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
    fallback = ["name", "date", "address", "phone", "account", "signature", "state", "total"]
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
    return [word for word in words if TOKEN_RE.search(word)] or fallback


def corrupt_words(words: list[str], word_bank: list[str], seed_key: str) -> tuple[list[str], int]:
    if not words:
        return words, 0
    rng = random.Random(seed_key)
    count = max(1, int(round(len(words) * CORRUPTION_RATE)))
    count = min(count, len(words))
    corrupted = list(words)
    for index in rng.sample(range(len(corrupted)), k=count):
        corrupted[index] = rng.choice(word_bank)
    return corrupted, count


def load_template_names(split_name: str) -> list[str]:
    path = SPLIT_OCR_ROOT / f"{split_name}_templates.json"
    if path.exists():
        return [str(item) for item in read_json(path)]
    split_dir = SPLIT_OCR_ROOT / split_name
    return sorted([path.name for path in split_dir.iterdir() if path.is_dir()], key=str.lower)


def original_ocr_path(template_name: str) -> Path:
    return ORIGINAL_OCR_ROOT / f"{template_name}.ocr.json"


def load_words_from_ocr(path: Path) -> list[str]:
    payload = read_json(path)
    return [str(word) for word in payload.get("words", [])]


def discover_queries(split_name: str, template_names: list[str]) -> list[QueryRow]:
    queries: list[QueryRow] = []
    for template_name in template_names:
        query_dir = SPLIT_OCR_ROOT / split_name / template_name
        if not query_dir.exists():
            continue
        for ocr_path in sorted(query_dir.glob("fill_*.ocr.json"), key=lambda path: path.name.lower()):
            queries.append(QueryRow(template_name=template_name, ocr_path=ocr_path, image_name=ocr_path.stem.replace(".ocr", "")))
    return queries


def build_index(template_names: list[str]) -> Bm25Index:
    token_counts: list[Counter[str]] = []
    doc_lengths: list[int] = []
    kept_templates: list[str] = []
    vocab: dict[str, int] = {}

    for template_name in template_names:
        path = original_ocr_path(template_name)
        if not path.exists():
            continue
        tokens = tokenize_words(load_words_from_ocr(path))
        kept_templates.append(template_name)
        doc_lengths.append(len(tokens))
        counts = Counter(tokens)
        token_counts.append(counts)
        for token in counts:
            if token not in vocab:
                vocab[token] = len(vocab)

    if not kept_templates:
        raise RuntimeError("No original OCR documents were available for BM25 indexing.")

    total_docs = len(kept_templates)
    average_doc_length = sum(doc_lengths) / total_docs
    doc_frequencies: Counter[str] = Counter()
    for counts in token_counts:
        doc_frequencies.update(counts.keys())

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for doc_id, counts in enumerate(token_counts):
        length_factor = BM25_K1 * (1.0 - BM25_B + BM25_B * doc_lengths[doc_id] / max(average_doc_length, 1e-12))
        for token, frequency in counts.items():
            document_frequency = doc_frequencies[token]
            idf = math.log(1.0 + (total_docs - document_frequency + 0.5) / (document_frequency + 0.5))
            weight = idf * (frequency * (BM25_K1 + 1.0)) / (frequency + length_factor)
            rows.append(doc_id)
            cols.append(vocab[token])
            data.append(weight)

    matrix = sparse.csr_matrix((data, (rows, cols)), shape=(total_docs, len(vocab)), dtype=np.float32)
    return Bm25Index(
        template_names=kept_templates,
        template_to_id={template_name: index for index, template_name in enumerate(kept_templates)},
        vocab=vocab,
        bm25_matrix=matrix,
    )


def build_query_matrix(index: Bm25Index, query_token_rows: list[list[str]]) -> sparse.csr_matrix:
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for row_id, tokens in enumerate(query_token_rows):
        for token, count in Counter(tokens).items():
            token_id = index.vocab.get(token)
            if token_id is None:
                continue
            rows.append(row_id)
            cols.append(token_id)
            data.append(float(count))
    return sparse.csr_matrix((data, (rows, cols)), shape=(len(query_token_rows), len(index.vocab)), dtype=np.float32)


def evaluate_split(split_name: str, *, corrupt: bool, word_bank: list[str]) -> dict[str, Any]:
    template_names = load_template_names(split_name)
    index = build_index(template_names)
    queries = [query for query in discover_queries(split_name, index.template_names) if query.template_name in index.template_names]
    output_dir = OUTPUT_ROOT / split_name / ("corrupt25" if corrupt else "clean_ocr")
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    top1 = top5 = top10 = 0
    reciprocal_ranks: list[float] = []
    total_corrupted = 0

    batch_size = 256
    for batch_start in range(0, len(queries), batch_size):
        batch_queries = queries[batch_start : batch_start + batch_size]
        batch_tokens: list[list[str]] = []
        batch_words: list[list[str]] = []
        batch_corrupted_counts: list[int] = []
        for query in batch_queries:
            words = load_words_from_ocr(query.ocr_path)
            corrupted_count = 0
            if corrupt:
                words, corrupted_count = corrupt_words(words, word_bank, f"{split_name}:{query.template_name}:{query.image_name}")
                total_corrupted += corrupted_count
            batch_words.append(words)
            batch_corrupted_counts.append(corrupted_count)
            batch_tokens.append(tokenize_words(words))

        query_matrix = build_query_matrix(index, batch_tokens)
        score_matrix = (query_matrix @ index.bm25_matrix.T).toarray()
        for offset, query in enumerate(batch_queries):
            scores = score_matrix[offset]
            true_doc_id = index.template_to_id[query.template_name]
            true_score = float(scores[true_doc_id])
            rank = int(1 + np.count_nonzero(scores > true_score))
            best_doc_id = int(np.argmax(scores))
            best_score = float(scores[best_doc_id])
            best_match = index.template_names[best_doc_id]
            top1 += int(rank == 1)
            top5 += int(rank <= 5)
            top10 += int(rank <= 10)
            reciprocal_ranks.append(1.0 / rank)
            rows.append(
                {
                    "split": split_name,
                    "template_name": query.template_name,
                    "query_ocr": str(query.ocr_path),
                    "rank": rank,
                    "top1": int(rank == 1),
                    "top5": int(rank <= 5),
                    "top10": int(rank <= 10),
                    "best_match": best_match,
                    "best_score": best_score,
                    "true_score": true_score,
                    "word_count": len(batch_words[offset]),
                    "corrupted_words": batch_corrupted_counts[offset],
                }
            )

        processed = min(batch_start + len(batch_queries), len(queries))
        if processed == len(batch_queries) or processed % 5000 < batch_size or processed == len(queries):
            print(f"{split_name} {'corrupt25' if corrupt else 'clean'}: {processed}/{len(queries)} queries", flush=True)

    with (output_dir / "per_query_rankings.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "split": split_name,
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
        "total_corrupted_words": total_corrupted,
        "split_ocr_root": str(SPLIT_OCR_ROOT),
        "original_ocr_root": str(ORIGINAL_OCR_ROOT),
    }
    write_json(output_dir / "benchmark_summary.json", summary)
    return summary


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    word_bank = load_word_bank()
    summaries: dict[str, dict[str, Any]] = {}
    for split_name in ["train", "test"]:
        clean = evaluate_split(split_name, corrupt=False, word_bank=word_bank)
        corrupt = evaluate_split(split_name, corrupt=True, word_bank=word_bank)
        summaries[split_name] = {
            "clean_ocr": clean,
            "corrupt25": corrupt,
            "delta": {
                key: corrupt[key] - clean[key]
                for key in ["retrieval_at_1", "retrieval_at_5", "retrieval_at_10", "mrr"]
            },
        }
    write_json(OUTPUT_ROOT / "contrast_summary.json", summaries)
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == "__main__":
    main()
