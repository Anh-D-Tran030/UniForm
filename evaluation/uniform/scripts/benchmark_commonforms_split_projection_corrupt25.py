from __future__ import annotations

import argparse
import json
import random
import re
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch


SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

EXPERIMENT_ROOT = Path(r"C:\Users\thanh\OneDrive - UTS\Personal_code\LayoutLMv3 Experiment")
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

import benchmark_commonforms_projection_retrieval as projection_bench
import train_commonforms_projection_masked_split as split_train
import train_commonforms_template_projection as commontrain
import train_layoutlmv3_template_projection as baseproj
import visualize_layoutlmv3_template_embeddings as viz


CHECKPOINT_DIR = Path(r"A:\RealForm\outputs\layoutlmv3_projection_commonforms_masked_split_10ep_rerun_285trainretrieval")
SPLIT_OCR_ROOT = Path(r"A:\RealForm\processed\synthetic_fill_images_10k_folders_degraded_ocr_split_9400_300_300_augmented")
DEGRADED_ROOT = Path(r"A:\RealForm\processed\synthetic_fill_images_10k_folders_degraded")
ORIGINAL_ROOT = Path(r"A:\RealForm\processed\CommonFormsEnglish\images")
ORIGINAL_OCR_ROOT = Path(r"A:\RealForm\processed\CommonFormsEnglish_original_ocr_cache_projection")
DICTIONARY_PATH = Path(r"A:\RealForm\processed\synthetic_fill_images\dictionary_500_words.json")
OUTPUT_ROOT = Path(r"A:\RealForm\processed\commonforms_split_layoutlmfc_corrupt25_benchmark")
CORRUPTION_RATE = 0.25
TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark the CommonForms LayoutLMv3+FC projection model with 25% OCR word corruption."
    )
    parser.add_argument("--checkpoint-dir", type=Path, default=CHECKPOINT_DIR)
    parser.add_argument("--split-ocr-root", type=Path, default=SPLIT_OCR_ROOT)
    parser.add_argument("--degraded-root", type=Path, default=DEGRADED_ROOT)
    parser.add_argument("--original-root", type=Path, default=ORIGINAL_ROOT)
    parser.add_argument("--original-ocr-root", type=Path, default=ORIGINAL_OCR_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--dictionary-path", type=Path, default=DICTIONARY_PATH)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--train-query-sample-count", type=int, default=285)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--splits", nargs="+", choices=["train", "test"], default=["train", "test"])
    parser.add_argument("--force-cpu", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_word_bank(dictionary_path: Path) -> list[str]:
    fallback = ["name", "date", "address", "phone", "account", "signature", "state", "total"]
    if not dictionary_path.exists():
        return fallback
    payload = read_json(dictionary_path)
    if isinstance(payload, list):
        words = [str(item).strip() for item in payload if str(item).strip()]
    elif isinstance(payload, dict) and isinstance(payload.get("words"), list):
        words = [str(item).strip() for item in payload["words"] if str(item).strip()]
    elif isinstance(payload, dict):
        words = [str(key).strip() for key in payload.keys() if str(key).strip()]
    else:
        words = []
    return [word for word in words if TOKEN_RE.search(word)] or fallback


def corrupt_document_words(
    document: viz.DocumentSample,
    word_bank: list[str],
    seed_key: str,
) -> tuple[viz.DocumentSample, int]:
    if not document.words:
        return document, 0
    rng = random.Random(seed_key)
    count = max(1, int(round(len(document.words) * CORRUPTION_RATE)))
    count = min(count, len(document.words))
    words = list(document.words)
    for index in rng.sample(range(len(words)), k=count):
        words[index] = rng.choice(word_bank)
    return replace(document, words=words), count


def corrupt_documents(
    documents: list[viz.DocumentSample],
    split_name: str,
    word_bank: list[str],
) -> tuple[list[viz.DocumentSample], int]:
    corrupted: list[viz.DocumentSample] = []
    total_corrupted = 0
    for document in documents:
        corrupted_document, corrupted_count = corrupt_document_words(
            document,
            word_bank,
            f"{split_name}:{document.stem}:{document.image_path.name}:{document.record_id}",
        )
        corrupted.append(corrupted_document)
        total_corrupted += corrupted_count
    return corrupted, total_corrupted


def discover_documents(
    split_name: str,
    split_ocr_root: Path,
    degraded_root: Path,
    original_root: Path,
    original_ocr_root: Path,
    train_query_sample_count: int,
    seed: int,
) -> tuple[list[viz.DocumentSample], list[viz.DocumentSample], int, int]:
    original_image_index = commontrain.build_original_image_index(original_root)
    selections = split_train.discover_split_selections(split_name, split_ocr_root, degraded_root, original_image_index)
    query_documents, original_documents = split_train.build_documents_for_split(
        selections,
        split_ocr_root,
        split_name,
        original_ocr_root,
    )
    full_query_count = len(query_documents)
    if split_name == "train" and train_query_sample_count > 0:
        query_documents = split_train.sample_train_query_documents(query_documents, train_query_sample_count, seed)
    return query_documents, original_documents, len(selections), full_query_count


def collect_embeddings(
    model: baseproj.LayoutLMv3TemplateProjectionModel,
    processor: Any,
    documents: list[viz.DocumentSample],
    batch_size: int,
    max_length: int,
    device: torch.device,
) -> np.ndarray:
    return projection_bench.collect_projection_embeddings(
        model=model,
        processor=processor,
        documents=documents,
        batch_size=batch_size,
        max_length=max_length,
        device=device,
    )


def evaluate_split(
    split_name: str,
    args: argparse.Namespace,
    model: baseproj.LayoutLMv3TemplateProjectionModel,
    processor: Any,
    training_config: dict[str, Any],
    device: torch.device,
    word_bank: list[str],
) -> dict[str, Any]:
    query_documents, original_documents, template_count, full_query_count = discover_documents(
        split_name=split_name,
        split_ocr_root=args.split_ocr_root.resolve(),
        degraded_root=args.degraded_root.resolve(),
        original_root=args.original_root.resolve(),
        original_ocr_root=args.original_ocr_root.resolve(),
        train_query_sample_count=args.train_query_sample_count,
        seed=args.seed,
    )
    corrupted_query_documents, total_corrupted_words = corrupt_documents(query_documents, split_name, word_bank)

    print(
        f"{split_name}: templates={template_count} gallery={len(original_documents)} "
        f"queries={len(corrupted_query_documents)} full_queries={full_query_count} corrupt_words={total_corrupted_words}",
        flush=True,
    )
    original_embeddings = collect_embeddings(
        model=model,
        processor=processor,
        documents=original_documents,
        batch_size=args.batch_size,
        max_length=int(training_config["max_length"]),
        device=device,
    )
    query_embeddings = collect_embeddings(
        model=model,
        processor=processor,
        documents=corrupted_query_documents,
        batch_size=args.batch_size,
        max_length=int(training_config["max_length"]),
        device=device,
    )

    metrics = commontrain.compute_retrieval_vs_original(
        corrupted_query_documents,
        query_embeddings,
        original_documents,
        original_embeddings,
    )
    metrics_summary = {key: value for key, value in metrics.items() if key not in {"positive_scores", "nearest_negative_scores", "per_template_top1"}}
    summary = {
        "split": split_name,
        "mode": "layoutlmfc_corrupt25",
        "checkpoint_dir": str(args.checkpoint_dir.resolve()),
        "device": str(device),
        "gallery_templates": len(original_documents),
        "split_templates": template_count,
        "query_images": len(corrupted_query_documents),
        "full_split_query_images": full_query_count,
        "train_query_sample_count": args.train_query_sample_count if split_name == "train" else None,
        "corruption_rate": CORRUPTION_RATE,
        "total_corrupted_words": total_corrupted_words,
        **metrics_summary,
    }
    split_output = args.output_root.resolve() / split_name / "corrupt25"
    write_json(split_output / "retrieval_vs_original_summary.json", summary)
    write_json(
        split_output / "query_rows.json",
        [
            {
                "template_stem": document.stem,
                "record_id": document.record_id,
                "image_path": str(document.image_path),
                "word_count": len(document.words),
            }
            for document in corrupted_query_documents
        ],
    )
    write_json(split_output / "per_template_top1.json", metrics.get("per_template_top1", {}))
    return summary


def main() -> None:
    args = parse_args()
    args.output_root.resolve().mkdir(parents=True, exist_ok=True)
    word_bank = load_word_bank(args.dictionary_path.resolve())

    device = torch.device("cpu" if args.force_cpu or not torch.cuda.is_available() else "cuda")
    model, processor, training_config = projection_bench.load_projection_model(args.checkpoint_dir.resolve(), device)
    model.eval()
    print(f"Loaded LayoutLM+FC checkpoint on {device}: {args.checkpoint_dir.resolve()}", flush=True)

    split_summaries: dict[str, Any] = {}
    for split_name in args.splits:
        split_summaries[split_name] = evaluate_split(
            split_name,
            args,
            model,
            processor,
            training_config,
            device,
            word_bank,
        )

    combined = {
        "model": "layoutlmv3_base_plus_fc_projection",
        "checkpoint_dir": str(args.checkpoint_dir.resolve()),
        "output_root": str(args.output_root.resolve()),
        "corruption_rate": CORRUPTION_RATE,
        "word_bank_size": len(word_bank),
        "summaries": split_summaries,
    }
    write_json(args.output_root.resolve() / "contrast_summary.json", combined)
    print(json.dumps(combined, indent=2), flush=True)


if __name__ == "__main__":
    main()
