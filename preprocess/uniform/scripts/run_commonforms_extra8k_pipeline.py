from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from scipy import sparse


SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

EXPERIMENT_ROOT = Path(r"C:\Users\thanh\OneDrive - UTS\Personal_code\LayoutLMv3 Experiment")
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

import benchmark_commonforms_projection_retrieval as projection_bench
import build_tesseract_ocr_cache_degraded as ocr_cache
import degrade_synthetic_fill_10k as degrade
import generate_commonforms_synthetic_fills as synth_fill
import train_commonforms_template_projection as commontrain
import visualize_layoutlmv3_template_embeddings as viz


OLD_DEGRADED_ROOT = Path(r"A:\RealForm\processed\synthetic_fill_images_10k_folders_degraded")
FILL_ROOT = Path(r"A:\RealForm\processed\synthetic_fill_images_8k_extra_3fills")
DEGRADED_ROOT = Path(r"A:\RealForm\processed\synthetic_fill_images_8k_extra_3fills_degraded")
CLEAN_OCR_ROOT = Path(r"A:\RealForm\processed\synthetic_fill_images_8k_extra_3fills_degraded_ocr_clean")
CORRUPT_OCR_ROOT = Path(r"A:\RealForm\processed\synthetic_fill_images_8k_extra_3fills_degraded_ocr_corrupt25")
BENCHMARK_ROOT = Path(r"A:\RealForm\processed\commonforms_8k_extra_3fills_retrieval_benchmark")
ORIGINAL_ROOT = Path(r"A:\RealForm\processed\CommonFormsEnglish\images")
ORIGINAL_OCR_ROOT = Path(r"A:\RealForm\processed\CommonFormsEnglish_original_ocr_cache_projection")
CHECKPOINT_DIR = Path(r"A:\RealForm\outputs\layoutlmv3_projection_commonforms_masked_split_10ep_rerun_285trainretrieval")
DICTIONARY_PATH = Path(r"A:\RealForm\processed\synthetic_fill_images\dictionary_500_words.json")
TESSERACT_CMD = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")

TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
CORRUPTION_RATE = 0.25
BM25_K1 = 1.5
BM25_B = 0.75


@dataclass(frozen=True)
class SelectedTemplate:
    stem: str
    metadata_path: Path
    original_image_path: Path


@dataclass
class Bm25Index:
    documents: list[viz.DocumentSample]
    vocab: dict[str, int]
    matrix: sparse.csr_matrix


class Logger:
    def info(self, message: str, *args: Any) -> None:
        log(message % args if args else message)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def log(message: str) -> None:
    print(f"[{utc_now()}] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate, degrade, OCR, and benchmark 8k extra CommonForms English templates with 3 fills each."
    )
    parser.add_argument("--template-count", type=int, default=8000)
    parser.add_argument("--fills-per-template", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260417)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--force-cpu", action="store_true")
    parser.add_argument("--skip-layoutlm", action="store_true", help="Only run fill/degrade/OCR/BM25.")
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def count_files(root: Path, pattern: str) -> int:
    if not root.exists():
        return 0
    return sum(1 for _ in root.rglob(pattern))


def configure_synthetic_fill_output() -> None:
    synth_fill.OUTPUT_ROOT = FILL_ROOT
    synth_fill.STATE_PATH = FILL_ROOT / "synthetic_fill_state.json"
    synth_fill.LOG_PATH = FILL_ROOT / "synthetic_fill.log"
    synth_fill.DICTIONARY_PATH = FILL_ROOT / "dictionary_500_words.json"


def load_or_copy_dictionary() -> list[str]:
    configure_synthetic_fill_output()
    if DICTIONARY_PATH.exists():
        payload = read_json(DICTIONARY_PATH)
        write_json(FILL_ROOT / "dictionary_500_words.json", payload)
        words = list(payload.get("words") if isinstance(payload, dict) else payload)
        return [str(word) for word in words][:500]
    return synth_fill.load_or_build_dictionary()


def old_degraded_template_names() -> set[str]:
    if not OLD_DEGRADED_ROOT.exists():
        return set()
    return {path.name for path in OLD_DEGRADED_ROOT.iterdir() if path.is_dir()}


def select_templates(template_count: int) -> list[SelectedTemplate]:
    manifest_path = BENCHMARK_ROOT / "selected_templates.json"
    if manifest_path.exists():
        rows = read_json(manifest_path)
        return [
            SelectedTemplate(
                stem=str(row["stem"]),
                metadata_path=Path(row["metadata_path"]),
                original_image_path=Path(row["original_image_path"]),
            )
            for row in rows
        ]

    excluded = old_degraded_template_names()
    selected: list[SelectedTemplate] = []
    for metadata_path in synth_fill.list_template_metadata():
        if metadata_path.stem in excluded:
            continue
        image_path = synth_fill.template_image_path(metadata_path)
        if image_path is None:
            continue
        selected.append(SelectedTemplate(metadata_path.stem, metadata_path, image_path))
        if len(selected) >= template_count:
            break

    if len(selected) < template_count:
        raise RuntimeError(f"Only found {len(selected)} extra English templates outside {OLD_DEGRADED_ROOT}; need {template_count}.")

    write_json(
        manifest_path,
        [
            {
                "stem": item.stem,
                "metadata_path": str(item.metadata_path),
                "original_image_path": str(item.original_image_path),
            }
            for item in selected
        ],
    )
    return selected


def run_fill(selected: list[SelectedTemplate], fills_per_template: int, seed: int, words: list[str]) -> None:
    configure_synthetic_fill_output()
    FILL_ROOT.mkdir(parents=True, exist_ok=True)
    state = synth_fill.load_json(synth_fill.STATE_PATH, {"templates": {}})
    handwriting_fonts = synth_fill.available_fonts()
    digital_fonts = synth_fill.available_digital_fonts()
    log(f"Fill stage: {len(selected)} templates, {fills_per_template} fills each -> {FILL_ROOT}")
    for index, item in enumerate(selected, start=1):
        created, _attempted = synth_fill.process_template(
            metadata_path=item.metadata_path,
            image_path=item.original_image_path,
            words=words,
            handwriting_fonts=handwriting_fonts,
            digital_fonts=digital_fonts,
            fills_per_template=fills_per_template,
            seed=seed,
            state=state,
        )
        if created or index == 1 or index % 100 == 0 or index == len(selected):
            log(f"Fill progress {index}/{len(selected)} templates. created={created}")
    write_json(FILL_ROOT / "fill_summary.json", {"templates": len(selected), "fills_per_template": fills_per_template})


def run_degrade(selected: list[SelectedTemplate]) -> None:
    DEGRADED_ROOT.mkdir(parents=True, exist_ok=True)
    log(f"Degrade stage: source={FILL_ROOT} output={DEGRADED_ROOT}")
    for index, item in enumerate(selected, start=1):
        source_dir = FILL_ROOT / item.stem
        if not source_dir.exists():
            raise FileNotFoundError(f"Missing synthetic fill folder: {source_dir}")
        folder = degrade.TemplateFolder(
            stem=item.stem,
            path=source_dir,
            image_count=len(list(source_dir.glob("fill_*.png"))),
        )
        degrade.process_template_folder(folder=folder, output_root=DEGRADED_ROOT, overwrite=False)
        if index == 1 or index % 100 == 0 or index == len(selected):
            log(f"Degrade progress {index}/{len(selected)} templates.")
    write_json(DEGRADED_ROOT / "degradation_summary.json", {"source_root": str(FILL_ROOT), "output_root": str(DEGRADED_ROOT)})


def run_clean_ocr() -> None:
    log(f"Clean OCR stage: degraded={DEGRADED_ROOT} cache={CLEAN_OCR_ROOT}")
    args = SimpleNamespace(
        degraded_root=DEGRADED_ROOT,
        cache_root=CLEAN_OCR_ROOT,
        tesseract_cmd=TESSERACT_CMD,
        skip_existing=True,
        limit_templates=None,
        limit_images=None,
        overlay_count=8,
        psm=11,
        oem=3,
        min_confidence=30.0,
        watch=False,
        poll_seconds=30,
        idle_polls=3,
    )
    ocr_cache.build_cache(args)


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


def run_corrupt_ocr(word_bank: list[str]) -> None:
    clean_root = CLEAN_OCR_ROOT / "ocr_json"
    corrupt_root = CORRUPT_OCR_ROOT / "ocr_json"
    log(f"Corrupt OCR stage: clean={clean_root} corrupt={corrupt_root}")
    total = 0
    written = 0
    reused = 0
    total_corrupted = 0
    for clean_path in sorted(clean_root.rglob("*.ocr.json"), key=lambda path: str(path).lower()):
        total += 1
        relative_path = clean_path.relative_to(clean_root)
        target_path = corrupt_root / relative_path
        if target_path.exists():
            reused += 1
            continue
        payload = read_json(clean_path)
        corrupted_words, corrupted_count = corrupt_words(
            [str(word) for word in payload.get("words", [])],
            word_bank,
            str(relative_path).lower(),
        )
        payload["words"] = corrupted_words
        payload["word_count"] = len(corrupted_words)
        payload["ocr_corruption"] = {
            "rate": CORRUPTION_RATE,
            "corrupted_words": corrupted_count,
            "source_clean_ocr": str(clean_path),
        }
        write_json(target_path, payload)
        written += 1
        total_corrupted += corrupted_count
        if total == 1 or total % 1000 == 0:
            log(f"Corrupt OCR progress {total} files. written={written} reused={reused}")
    write_json(
        CORRUPT_OCR_ROOT / "corrupt_ocr_summary.json",
        {
            "clean_ocr_root": str(clean_root),
            "corrupt_ocr_root": str(corrupt_root),
            "total_seen": total,
            "written": written,
            "reused": reused,
            "total_corrupted_words": total_corrupted,
            "corruption_rate": CORRUPTION_RATE,
        },
    )


def ensure_original_ocr(selected: list[SelectedTemplate]) -> list[commontrain.TemplateSelection]:
    selections = [
        commontrain.TemplateSelection(
            stem=item.stem,
            query_dir=DEGRADED_ROOT / item.stem,
            original_image_path=item.original_image_path,
        )
        for item in selected
    ]
    commontrain.ensure_original_ocr_cache(
        selections=selections,
        original_ocr_root=ORIGINAL_OCR_ROOT,
        tesseract_cmd=TESSERACT_CMD,
        psm=11,
        oem=3,
        min_confidence=30.0,
        logger=Logger(),
    )
    return selections


def build_documents(
    selections: list[commontrain.TemplateSelection],
    query_ocr_root: Path,
) -> tuple[list[viz.DocumentSample], list[viz.DocumentSample]]:
    return commontrain.build_documents_for_split(
        selections=selections,
        query_ocr_root=query_ocr_root / "ocr_json",
        original_ocr_root=ORIGINAL_OCR_ROOT,
    )


def tokenize_words(words: list[str]) -> list[str]:
    tokens: list[str] = []
    for word in words:
        tokens.extend(match.group(0).lower() for match in TOKEN_RE.finditer(str(word)))
    return tokens or ["empty"]


def build_bm25_index(original_documents: list[viz.DocumentSample]) -> Bm25Index:
    token_counts: list[Counter[str]] = []
    doc_lengths: list[int] = []
    vocab: dict[str, int] = {}
    for document in original_documents:
        tokens = tokenize_words([str(word) for word in document.words])
        counts = Counter(tokens)
        token_counts.append(counts)
        doc_lengths.append(len(tokens))
        for token in counts:
            if token not in vocab:
                vocab[token] = len(vocab)

    total_docs = len(original_documents)
    average_doc_length = sum(doc_lengths) / max(total_docs, 1)
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
    return Bm25Index(documents=original_documents, vocab=vocab, matrix=matrix)


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


def evaluate_bm25(
    query_documents: list[viz.DocumentSample],
    original_documents: list[viz.DocumentSample],
    mode: str,
) -> dict[str, Any]:
    output_dir = BENCHMARK_ROOT / "bm25" / mode
    output_dir.mkdir(parents=True, exist_ok=True)
    index = build_bm25_index(original_documents)
    original_labels = [document.stem for document in index.documents]
    label_to_doc_id = {document.stem: index_id for index_id, document in enumerate(index.documents)}
    rows: list[dict[str, Any]] = []
    top1 = top5 = top10 = 0
    reciprocal_ranks: list[float] = []

    batch_size = 256
    for batch_start in range(0, len(query_documents), batch_size):
        batch = query_documents[batch_start : batch_start + batch_size]
        query_matrix = build_query_matrix(index, [tokenize_words([str(word) for word in document.words]) for document in batch])
        score_matrix = (query_matrix @ index.matrix.T).toarray()
        for offset, document in enumerate(batch):
            scores = score_matrix[offset]
            true_doc_id = label_to_doc_id[document.stem]
            true_score = float(scores[true_doc_id])
            rank = int(1 + np.count_nonzero(scores > true_score))
            best_doc_id = int(np.argmax(scores))
            top1 += int(rank == 1)
            top5 += int(rank <= 5)
            top10 += int(rank <= 10)
            reciprocal_ranks.append(1.0 / rank)
            rows.append(
                {
                    "template_name": document.stem,
                    "query_image": str(document.image_path),
                    "rank": rank,
                    "top1": int(rank == 1),
                    "top5": int(rank <= 5),
                    "top10": int(rank <= 10),
                    "best_match": original_labels[best_doc_id],
                    "best_score": float(scores[best_doc_id]),
                    "true_score": true_score,
                }
            )
        if batch_start == 0 or (batch_start + len(batch)) % 5000 < batch_size or batch_start + len(batch) == len(query_documents):
            log(f"BM25 {mode}: {batch_start + len(batch)}/{len(query_documents)} queries")

    with (output_dir / "per_query_rankings.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    query_count = len(query_documents)
    summary = {
        "method": "bm25",
        "mode": mode,
        "gallery_templates": len(original_documents),
        "query_images": query_count,
        "retrieval_at_1": top1 / max(query_count, 1),
        "retrieval_at_5": top5 / max(query_count, 1),
        "retrieval_at_10": top10 / max(query_count, 1),
        "mrr": sum(reciprocal_ranks) / max(len(reciprocal_ranks), 1),
        "correct_top1_queries": top1,
        "bm25_k1": BM25_K1,
        "bm25_b": BM25_B,
    }
    write_json(output_dir / "retrieval_summary.json", summary)
    return summary


def evaluate_layoutlm(
    query_documents: list[viz.DocumentSample],
    original_documents: list[viz.DocumentSample],
    mode: str,
    batch_size: int,
    force_cpu: bool,
) -> dict[str, Any]:
    output_dir = BENCHMARK_ROOT / "layoutlmfc" / mode
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu" if force_cpu or not torch.cuda.is_available() else "cuda")
    model, processor, training_config = projection_bench.load_projection_model(CHECKPOINT_DIR, device)
    model.eval()
    log(f"LayoutLM+FC {mode}: device={device} gallery={len(original_documents)} queries={len(query_documents)}")
    original_embeddings = projection_bench.collect_projection_embeddings(
        model=model,
        processor=processor,
        documents=original_documents,
        batch_size=batch_size,
        max_length=int(training_config["max_length"]),
        device=device,
    )
    query_embeddings = projection_bench.collect_projection_embeddings(
        model=model,
        processor=processor,
        documents=query_documents,
        batch_size=batch_size,
        max_length=int(training_config["max_length"]),
        device=device,
    )
    metrics = commontrain.compute_retrieval_vs_original(query_documents, query_embeddings, original_documents, original_embeddings)
    summary = {
        "method": "layoutlmv3_base_plus_fc_projection",
        "mode": mode,
        "checkpoint_dir": str(CHECKPOINT_DIR),
        "device": str(device),
        "gallery_templates": len(original_documents),
        "query_images": len(query_documents),
        **{key: value for key, value in metrics.items() if key not in {"positive_scores", "nearest_negative_scores", "per_template_top1"}},
    }
    write_json(output_dir / "retrieval_summary.json", summary)
    write_json(output_dir / "per_template_top1.json", metrics.get("per_template_top1", {}))
    return summary


def run_benchmarks(selections: list[commontrain.TemplateSelection], batch_size: int, force_cpu: bool, skip_layoutlm: bool) -> None:
    clean_queries, originals = build_documents(selections, CLEAN_OCR_ROOT)
    corrupt_queries, _ = build_documents(selections, CORRUPT_OCR_ROOT)
    log(f"Benchmark stage: gallery={len(originals)} clean_queries={len(clean_queries)} corrupt_queries={len(corrupt_queries)}")

    bm25_clean = evaluate_bm25(clean_queries, originals, "clean_ocr")
    bm25_corrupt = evaluate_bm25(corrupt_queries, originals, "corrupt25")
    layout_clean = None
    layout_corrupt = None
    if not skip_layoutlm:
        layout_clean = evaluate_layoutlm(clean_queries, originals, "clean_ocr", batch_size, force_cpu)
        layout_corrupt = evaluate_layoutlm(corrupt_queries, originals, "corrupt25", batch_size, force_cpu)

    write_json(
        BENCHMARK_ROOT / "comparison_summary.json",
        {
            "template_count": len(selections),
            "fills_per_template": 3,
            "gallery_templates": len(originals),
            "clean_query_images": len(clean_queries),
            "corrupt_query_images": len(corrupt_queries),
            "bm25": {"clean_ocr": bm25_clean, "corrupt25": bm25_corrupt},
            "layoutlmfc": {"clean_ocr": layout_clean, "corrupt25": layout_corrupt},
        },
    )


def main() -> None:
    args = parse_args()
    BENCHMARK_ROOT.mkdir(parents=True, exist_ok=True)
    log(f"Starting extra 8k pipeline with args={vars(args)}")
    words = load_or_copy_dictionary()
    selected = select_templates(args.template_count)
    log(f"Selected {len(selected)} templates excluding {OLD_DEGRADED_ROOT}.")

    run_fill(selected, args.fills_per_template, args.seed, words)
    run_degrade(selected)
    run_clean_ocr()
    run_corrupt_ocr(words)
    selections = ensure_original_ocr(selected)
    run_benchmarks(selections, args.batch_size, args.force_cpu, args.skip_layoutlm)

    summary = {
        "selected_templates": len(selected),
        "filled_images": count_files(FILL_ROOT, "fill_*.png"),
        "degraded_images": count_files(DEGRADED_ROOT, "fill_*.png"),
        "clean_ocr_json": count_files(CLEAN_OCR_ROOT / "ocr_json", "*.ocr.json"),
        "corrupt_ocr_json": count_files(CORRUPT_OCR_ROOT / "ocr_json", "*.ocr.json"),
        "benchmark_root": str(BENCHMARK_ROOT),
    }
    write_json(BENCHMARK_ROOT / "pipeline_summary.json", summary)
    log(f"Finished extra 8k pipeline: {summary}")


if __name__ == "__main__":
    main()
