from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pytesseract
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

EXPERIMENT_ROOT = Path(r"C:\Users\thanh\OneDrive - UTS\Personal_code\LayoutLMv3 Experiment")
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

import train_layoutlmv3_template_projection as baseproj
import visualize_layoutlmv3_template_embeddings as viz


DEFAULT_MODEL_NAME = "microsoft/layoutlmv3-base"
DEFAULT_QUERY_ROOT = Path(r"A:\RealForm\processed\synthetic_fill_images_10k_folders_degraded")
DEFAULT_QUERY_OCR_ROOT = Path(r"A:\RealForm\processed\synthetic_fill_images_10k_folders_degraded_ocr_cache\ocr_json")
DEFAULT_ORIGINAL_ROOT = Path(r"A:\RealForm\processed\CommonFormsEnglish\images")
DEFAULT_ORIGINAL_OCR_ROOT = Path(r"A:\RealForm\processed\CommonFormsEnglish_original_ocr_cache_projection")
DEFAULT_OUTPUT_DIR = Path(r"A:\RealForm\outputs\layoutlmv3_projection_commonforms_2k_300_300_5ep")
DEFAULT_TESSERACT_CMD = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
DEFAULT_TRAIN_TEMPLATES = 2000
DEFAULT_VAL_TEMPLATES = 300
DEFAULT_TEST_TEMPLATES = 300
DEFAULT_EPOCHS = 5
DEFAULT_SEEN_VAL_FRACTION = 0.1
DEFAULT_BATCH_SIZE = 8
DEFAULT_EVAL_BATCH_SIZE = 8
DEFAULT_PROJECTION_DIM = 128
DEFAULT_CLASSES_PER_BATCH = 4
DEFAULT_SAMPLES_PER_CLASS = 2
DEFAULT_SEED = 42
IMAGE_GLOB = "fill_*.png"


@dataclass(frozen=True)
class TemplateSelection:
    stem: str
    query_dir: Path
    original_image_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a LayoutLMv3 projection model on OCR-backed CommonForms degraded synthetic fills "
            "with template-level train/val/test splits and retrieval-vs-original diagnostics."
        )
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--query-root", type=Path, default=DEFAULT_QUERY_ROOT)
    parser.add_argument("--query-ocr-root", type=Path, default=DEFAULT_QUERY_OCR_ROOT)
    parser.add_argument("--original-root", type=Path, default=DEFAULT_ORIGINAL_ROOT)
    parser.add_argument("--original-ocr-root", type=Path, default=DEFAULT_ORIGINAL_OCR_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tesseract-cmd", type=Path, default=DEFAULT_TESSERACT_CMD)
    parser.add_argument("--train-templates", type=int, default=DEFAULT_TRAIN_TEMPLATES)
    parser.add_argument("--val-templates", type=int, default=DEFAULT_VAL_TEMPLATES)
    parser.add_argument("--test-templates", type=int, default=DEFAULT_TEST_TEMPLATES)
    parser.add_argument("--seen-val-fraction", type=float, default=DEFAULT_SEEN_VAL_FRACTION)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--eval-batch-size", type=int, default=DEFAULT_EVAL_BATCH_SIZE)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--pooling", choices=["cls", "mean"], default="cls")
    parser.add_argument("--projection-dim", type=int, default=DEFAULT_PROJECTION_DIM)
    parser.add_argument("--classes-per-batch", type=int, default=DEFAULT_CLASSES_PER_BATCH)
    parser.add_argument("--samples-per-class", type=int, default=DEFAULT_SAMPLES_PER_CLASS)
    parser.add_argument("--batches-per-epoch", type=int)
    parser.add_argument("--train-backbone-lr", type=float, default=2e-5)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--supcon-temperature", type=float, default=0.07)
    parser.add_argument("--arcface-margin", type=float, default=0.2)
    parser.add_argument("--arcface-scale", type=float, default=30.0)
    parser.add_argument("--arcface-weight", type=float, default=0.2)
    parser.add_argument("--supcon-weight", type=float, default=0.8)
    parser.add_argument("--unfreeze-last-n", type=int, default=4)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--psm", type=int, default=11)
    parser.add_argument("--oem", type=int, default=3)
    parser.add_argument("--min-confidence", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict[str, Any] | list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def configure_tesseract(executable_path: Path) -> None:
    if not executable_path.exists():
        raise FileNotFoundError(f"Tesseract executable was not found at {executable_path}.")
    pytesseract.pytesseract.tesseract_cmd = str(executable_path)


def normalize_box(box: list[int], width: int, height: int) -> list[int]:
    return viz.normalize_bbox([int(box[0]), int(box[1]), int(box[2]), int(box[3])], width, height)


def build_original_image_index(original_root: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for image_path in sorted(
        [path for path in original_root.rglob("*") if path.is_file()],
        key=lambda path: str(path).lower(),
    ):
        index.setdefault(image_path.stem, image_path)
    return index


def count_matching_files(path: Path, pattern: str) -> int:
    if not path.exists():
        return 0
    return len(list(path.glob(pattern)))


def select_ready_templates(
    query_root: Path,
    query_ocr_root: Path,
    original_image_index: dict[str, Path],
    total_templates: int,
    seed: int,
    logger: Any | None = None,
) -> list[TemplateSelection]:
    selections: list[TemplateSelection] = []
    query_dirs = sorted([path for path in query_root.iterdir() if path.is_dir()], key=lambda path: path.name.lower())
    for index, query_dir in enumerate(query_dirs, start=1):
        if count_matching_files(query_dir, IMAGE_GLOB) != 10:
            continue
        if count_matching_files(query_ocr_root / query_dir.name, "*.ocr.json") != 10:
            continue
        original_image_path = original_image_index.get(query_dir.name)
        if original_image_path is None:
            continue
        selections.append(TemplateSelection(stem=query_dir.name, query_dir=query_dir, original_image_path=original_image_path))
        if logger and (len(selections) == 1 or len(selections) % 250 == 0):
            logger.info("Template selection progress: ready=%s scanned=%s/%s", len(selections), index, len(query_dirs))
        if len(selections) >= total_templates:
            break

    rng = random.Random(seed)
    rng.shuffle(selections)
    if len(selections) < total_templates:
        raise RuntimeError(f"Only found {len(selections)} ready templates; need {total_templates}.")
    return selections[:total_templates]


def split_template_selections(
    selections: list[TemplateSelection],
    train_count: int,
    val_count: int,
    test_count: int,
) -> tuple[list[TemplateSelection], list[TemplateSelection], list[TemplateSelection]]:
    total = train_count + val_count + test_count
    if len(selections) < total:
        raise ValueError(f"Need {total} selections but only have {len(selections)}.")
    return (
        selections[:train_count],
        selections[train_count : train_count + val_count],
        selections[train_count + val_count : total],
    )


def make_original_cache_path(original_ocr_root: Path, stem: str) -> Path:
    return original_ocr_root / f"{stem}.ocr.json"


def run_tesseract(image_path: Path, psm: int, oem: int, min_confidence: float) -> dict[str, Any]:
    with Image.open(image_path) as image:
        rgb_image = image.convert("RGB")
        width, height = rgb_image.size
        scale_factor = 2.0
        preprocessed_image = ImageOps.autocontrast(rgb_image.convert("L")).resize(
            (int(width * scale_factor), int(height * scale_factor)),
            resample=Image.Resampling.BICUBIC,
        )
        data = pytesseract.image_to_data(
            preprocessed_image,
            output_type=pytesseract.Output.DICT,
            config=f"--psm {psm} --oem {oem}",
            lang="eng",
        )

    words: list[str] = []
    boxes: list[list[int]] = []
    confidences: list[float] = []
    for index in range(len(data.get("text", []))):
        text = str(data["text"][index]).strip()
        if not text:
            continue
        try:
            confidence = float(data["conf"][index])
        except (TypeError, ValueError):
            confidence = -1.0
        if confidence < min_confidence:
            continue
        left = int(round(int(data["left"][index]) / scale_factor))
        top = int(round(int(data["top"][index]) / scale_factor))
        box_width = max(1, int(round(int(data["width"][index]) / scale_factor)))
        box_height = max(1, int(round(int(data["height"][index]) / scale_factor)))
        words.append(text)
        boxes.append([left, top, left + box_width, top + box_height])
        confidences.append(confidence)

    if not words:
        words = ["[EMPTY]"]
        boxes = [[0, 0, 1, 1]]
        confidences = [0.0]

    return {
        "image_path": str(image_path.resolve()),
        "image_name": image_path.name,
        "image_size": {"width": width, "height": height},
        "word_count": len(words),
        "words": words,
        "boxes": boxes,
        "confidences": confidences,
        "engine": "tesseract",
        "psm": psm,
        "oem": oem,
    }


def ensure_original_ocr_cache(
    selections: list[TemplateSelection],
    original_ocr_root: Path,
    tesseract_cmd: Path,
    psm: int,
    oem: int,
    min_confidence: float,
    logger: Any,
) -> None:
    configure_tesseract(tesseract_cmd.resolve())
    built = 0
    reused = 0
    for index, selection in enumerate(selections, start=1):
        cache_path = make_original_cache_path(original_ocr_root, selection.stem)
        if cache_path.exists():
            reused += 1
            continue
        payload = run_tesseract(selection.original_image_path, psm=psm, oem=oem, min_confidence=min_confidence)
        payload.update({"template_stem": selection.stem, "is_original": True})
        write_json(cache_path, payload)
        built += 1
        if index == 1 or index % 100 == 0 or index == len(selections):
            logger.info("Built original OCR cache for %s/%s templates.", index, len(selections))
    logger.info("Original OCR cache ready: built=%s reused=%s", built, reused)


def build_documents_for_split(
    selections: list[TemplateSelection],
    query_ocr_root: Path,
    original_ocr_root: Path,
) -> tuple[list[viz.DocumentSample], list[viz.DocumentSample]]:
    query_documents: list[viz.DocumentSample] = []
    original_documents: list[viz.DocumentSample] = []
    for selection in selections:
        original_payload = read_json(make_original_cache_path(original_ocr_root, selection.stem))
        original_width = int(original_payload["image_size"]["width"])
        original_height = int(original_payload["image_size"]["height"])
        original_documents.append(
            viz.DocumentSample(
                subset="commonforms_original",
                stem=selection.stem,
                class_label=selection.stem,
                image_path=selection.original_image_path.resolve(),
                record_id=0,
                words=list(original_payload["words"]),
                boxes=[normalize_box(box, original_width, original_height) for box in original_payload["boxes"]],
                is_original=True,
            )
        )

        query_cache_dir = query_ocr_root / selection.stem
        for record_id, image_path in enumerate(sorted(selection.query_dir.glob(IMAGE_GLOB), key=lambda path: path.name.lower()), start=1):
            payload = read_json(query_cache_dir / f"{image_path.stem}.ocr.json")
            width = int(payload["image_size"]["width"])
            height = int(payload["image_size"]["height"])
            query_documents.append(
                viz.DocumentSample(
                    subset="commonforms_degraded",
                    stem=selection.stem,
                    class_label=selection.stem,
                    image_path=image_path.resolve(),
                    record_id=record_id,
                    words=list(payload["words"]),
                    boxes=[normalize_box(box, width, height) for box in payload["boxes"]],
                    is_original=False,
                )
            )
    return query_documents, original_documents


def build_classifier_examples(
    documents: list[viz.DocumentSample],
    class_to_train_label: dict[str, int],
) -> list[baseproj.TemplateExample]:
    return baseproj.build_examples_for_documents(documents, class_to_train_label.keys(), class_to_train_label)


def build_inference_loader(
    documents: list[viz.DocumentSample],
    collator: baseproj.ProcessorCollator,
    batch_size: int,
) -> DataLoader:
    dataset = baseproj.TemplateDocumentDataset(baseproj.build_inference_examples(documents))
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collator)


def compute_retrieval_vs_original(
    query_documents: list[viz.DocumentSample],
    query_embeddings: np.ndarray,
    original_documents: list[viz.DocumentSample],
    original_embeddings: np.ndarray,
    chunk_size: int = 1024,
) -> dict[str, Any]:
    if not query_documents or not original_documents:
        return {
            "retrieval_at_1": 0.0,
            "retrieval_at_5": 0.0,
            "retrieval_at_10": 0.0,
            "mrr": 0.0,
            "correct_top1_queries": 0,
            "query_count": len(query_documents),
            "positive_scores": [],
            "nearest_negative_scores": [],
            "per_template_top1": {},
        }

    normalized_queries = query_embeddings / np.maximum(np.linalg.norm(query_embeddings, axis=1, keepdims=True), 1e-12)
    normalized_originals = original_embeddings / np.maximum(np.linalg.norm(original_embeddings, axis=1, keepdims=True), 1e-12)
    original_labels = np.array([document.stem for document in original_documents], dtype=object)
    top1 = 0
    top5 = 0
    top10 = 0
    reciprocal_ranks: list[float] = []
    positive_scores: list[float] = []
    nearest_negative_scores: list[float] = []
    per_template_hits: dict[str, list[int]] = {}

    for start in range(0, len(query_documents), chunk_size):
        scores = normalized_queries[start : start + chunk_size] @ normalized_originals.T
        for local_index, similarities in enumerate(scores):
            query_document = query_documents[start + local_index]
            order = np.argsort(-similarities)
            ranked_labels = original_labels[order]
            true_label = query_document.stem
            top1_hit = ranked_labels[0] == true_label
            top1 += int(top1_hit)
            top5 += int(true_label in ranked_labels[:5])
            top10 += int(true_label in ranked_labels[:10])
            per_template_hits.setdefault(true_label, []).append(int(top1_hit))

            reciprocal_rank = 0.0
            for rank, label in enumerate(ranked_labels, start=1):
                if label == true_label:
                    reciprocal_rank = 1.0 / rank
                    positive_scores.append(float(similarities[order[rank - 1]]))
                    break
            reciprocal_ranks.append(reciprocal_rank)
            for index in order:
                if original_labels[index] != true_label:
                    nearest_negative_scores.append(float(similarities[index]))
                    break

    per_template_top1 = {template: float(sum(hits) / max(1, len(hits))) for template, hits in sorted(per_template_hits.items())}
    query_count = len(query_documents)
    return {
        "retrieval_at_1": top1 / max(query_count, 1),
        "retrieval_at_5": top5 / max(query_count, 1),
        "retrieval_at_10": top10 / max(query_count, 1),
        "mrr": float(sum(reciprocal_ranks) / max(query_count, 1)),
        "correct_top1_queries": top1,
        "query_count": query_count,
        "positive_scores": positive_scores,
        "nearest_negative_scores": nearest_negative_scores,
        "per_template_top1": per_template_top1,
    }


def save_history_csv(output_dir: Path, history_rows: list[dict[str, Any]]) -> None:
    if not history_rows:
        return
    csv_path = output_dir / "training_history.csv"
    fieldnames = list(history_rows[0].keys())
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history_rows)


def plot_history(output_dir: Path, history_rows: list[dict[str, Any]]) -> None:
    if not history_rows:
        return
    epochs = [row["epoch"] for row in history_rows]

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, [row["train_loss"] for row in history_rows], label="train_loss")
    plt.plot(epochs, [row["train_arcface_loss"] for row in history_rows], label="train_arcface_loss")
    plt.plot(epochs, [row["train_supcon_loss"] for row in history_rows], label="train_supcon_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curves")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "training_loss_curves.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, [row["train_classifier_accuracy"] for row in history_rows], label="train_classifier_accuracy")
    plt.plot(epochs, [row["seen_val_classifier_accuracy"] for row in history_rows], label="seen_val_classifier_accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Classifier Accuracy Curves")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "classifier_accuracy_curves.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, [row["train_retrieval_at_1"] for row in history_rows], label="train_retrieval_at_1")
    plt.plot(epochs, [row["val_retrieval_at_1"] for row in history_rows], label="val_retrieval_at_1")
    plt.plot(epochs, [row["val_retrieval_at_5"] for row in history_rows], label="val_retrieval_at_5")
    plt.xlabel("Epoch")
    plt.ylabel("Retrieval")
    plt.title("Retrieval vs Original Curves")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "retrieval_curves.png", dpi=200)
    plt.close()


def plot_similarity_hist(output_path: Path, positive_scores: list[float], nearest_negative_scores: list[float], title: str) -> None:
    plt.figure(figsize=(10, 6))
    plt.hist(positive_scores, bins=40, alpha=0.65, label="positive_similarity")
    plt.hist(nearest_negative_scores, bins=40, alpha=0.65, label="nearest_negative_similarity")
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Count")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_pca_projection(
    output_path: Path,
    query_documents: list[viz.DocumentSample],
    query_embeddings: np.ndarray,
    original_documents: list[viz.DocumentSample],
    original_embeddings: np.ndarray,
    title: str,
    max_queries: int = 1200,
) -> None:
    rng = np.random.default_rng(42)
    if len(query_documents) > max_queries:
        chosen_indices = np.sort(rng.choice(len(query_documents), size=max_queries, replace=False))
        query_documents = [query_documents[index] for index in chosen_indices]
        query_embeddings = query_embeddings[chosen_indices]

    combined_embeddings = np.concatenate([original_embeddings, query_embeddings], axis=0)
    points = viz.reduce_embeddings(combined_embeddings, method="pca", seed=42)
    original_count = len(original_documents)

    plt.figure(figsize=(10, 8))
    plt.scatter(points[:original_count, 0], points[:original_count, 1], s=18, c="#d62728", label="original", alpha=0.8)
    plt.scatter(points[original_count:, 0], points[original_count:, 1], s=10, c="#1f77b4", label="degraded_query", alpha=0.45)
    plt.title(title)
    plt.xlabel("PCA-1")
    plt.ylabel("PCA-2")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_retrieval_artifacts(
    output_dir: Path,
    split_name: str,
    metrics: dict[str, Any],
    query_documents: list[viz.DocumentSample],
    query_embeddings: np.ndarray,
    original_documents: list[viz.DocumentSample],
    original_embeddings: np.ndarray,
) -> None:
    split_dir = output_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    write_json(split_dir / "retrieval_vs_original_summary.json", metrics)
    plot_similarity_hist(
        split_dir / "retrieval_similarity_hist.png",
        metrics["positive_scores"],
        metrics["nearest_negative_scores"],
        title=f"{split_name} retrieval similarity distribution",
    )
    plot_pca_projection(
        split_dir / "retrieval_pca_projection.png",
        query_documents=query_documents,
        query_embeddings=query_embeddings,
        original_documents=original_documents,
        original_embeddings=original_embeddings,
        title=f"{split_name} originals vs degraded queries (PCA)",
    )
    with (split_dir / "per_template_top1.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["template_stem", "top1_accuracy"])
        writer.writeheader()
        for template, score in metrics["per_template_top1"].items():
            writer.writerow({"template_stem": template, "top1_accuracy": score})


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = baseproj.build_logger(output_dir)
    baseproj.set_seed(args.seed)

    total_templates = args.train_templates + args.val_templates + args.test_templates
    logger.info("Selecting %s ready templates from degraded+OCR CommonForms.", total_templates)
    original_root = args.original_root.resolve()
    logger.info("Indexing original template images under %s", original_root)
    original_image_index = build_original_image_index(original_root)
    logger.info("Original image index ready with %s stems.", len(original_image_index))
    selections = select_ready_templates(
        query_root=args.query_root.resolve(),
        query_ocr_root=args.query_ocr_root.resolve(),
        original_image_index=original_image_index,
        total_templates=total_templates,
        seed=args.seed,
        logger=logger,
    )
    train_selections, val_selections, test_selections = split_template_selections(
        selections,
        train_count=args.train_templates,
        val_count=args.val_templates,
        test_count=args.test_templates,
    )
    write_json(
        output_dir / "split_summary.json",
        {
            "train_templates": [selection.stem for selection in train_selections],
            "val_templates": [selection.stem for selection in val_selections],
            "test_templates": [selection.stem for selection in test_selections],
        },
    )

    ensure_original_ocr_cache(
        selections=selections,
        original_ocr_root=args.original_ocr_root.resolve(),
        tesseract_cmd=args.tesseract_cmd.resolve(),
        psm=args.psm,
        oem=args.oem,
        min_confidence=args.min_confidence,
        logger=logger,
    )

    logger.info("Loading OCR-backed documents for train/val/test splits.")
    train_query_documents, train_original_documents = build_documents_for_split(
        train_selections,
        query_ocr_root=args.query_ocr_root.resolve(),
        original_ocr_root=args.original_ocr_root.resolve(),
    )
    val_query_documents, val_original_documents = build_documents_for_split(
        val_selections,
        query_ocr_root=args.query_ocr_root.resolve(),
        original_ocr_root=args.original_ocr_root.resolve(),
    )
    test_query_documents, test_original_documents = build_documents_for_split(
        test_selections,
        query_ocr_root=args.query_ocr_root.resolve(),
        original_ocr_root=args.original_ocr_root.resolve(),
    )
    logger.info(
        "Loaded documents: train_query=%s train_original=%s val_query=%s val_original=%s test_query=%s test_original=%s",
        len(train_query_documents),
        len(train_original_documents),
        len(val_query_documents),
        len(val_original_documents),
        len(test_query_documents),
        len(test_original_documents),
    )

    train_class_labels = sorted({document.class_label for document in train_query_documents})
    train_class_to_id = {class_label: index for index, class_label in enumerate(train_class_labels)}
    all_train_examples = build_classifier_examples(train_query_documents, train_class_to_id)
    train_examples, seen_val_examples = baseproj.split_train_val_examples(all_train_examples, args.seen_val_fraction, args.seed)

    processor = viz.load_processor(args.model_name)
    collator = baseproj.ProcessorCollator(processor, args.max_length)

    train_dataset = baseproj.TemplateDocumentDataset(train_examples)
    seen_val_dataset = baseproj.TemplateDocumentDataset(seen_val_examples)
    train_sampler = baseproj.ClassBalancedBatchSampler(
        train_examples,
        classes_per_batch=args.classes_per_batch,
        samples_per_class=args.samples_per_class,
        batches_per_epoch=args.batches_per_epoch,
        seed=args.seed,
    )
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=0, collate_fn=collator)
    train_accuracy_loader = DataLoader(train_dataset, batch_size=args.eval_batch_size, shuffle=False, num_workers=0, collate_fn=collator)
    seen_val_loader = DataLoader(seen_val_dataset, batch_size=args.eval_batch_size, shuffle=False, num_workers=0, collate_fn=collator)
    train_retrieval_loader = build_inference_loader(train_query_documents + train_original_documents, collator, args.eval_batch_size)
    val_retrieval_loader = build_inference_loader(val_query_documents + val_original_documents, collator, args.eval_batch_size)
    test_retrieval_loader = build_inference_loader(test_query_documents + test_original_documents, collator, args.eval_batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)
    model = baseproj.LayoutLMv3TemplateProjectionModel(
        model_name=args.model_name,
        projection_dim=args.projection_dim,
        pooling=args.pooling,
        num_train_classes=len(train_class_labels),
        arcface_margin=args.arcface_margin,
        arcface_scale=args.arcface_scale,
    )
    baseproj.freeze_backbone_except_top_layers(model, args.unfreeze_last_n)
    model.to(device)

    parameter_summary = baseproj.summarize_trainable_parameters(model)
    write_json(output_dir / "model_parameter_summary.json", parameter_summary)
    optimizer = baseproj.create_optimizer(model, args)
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    training_config = vars(args).copy()
    training_config["device"] = str(device)
    training_config["total_steps"] = total_steps
    training_config["warmup_steps"] = warmup_steps
    write_json(output_dir / "training_config.json", training_config)

    best_metric = float("-inf")
    best_val_metrics: dict[str, Any] | None = None
    history_rows: list[dict[str, Any]] = []

    for epoch_index in range(1, args.epochs + 1):
        epoch_start = time.time()
        model.train()
        total_loss = 0.0
        total_arcface_loss = 0.0
        total_supcon_loss = 0.0
        step_count = 0

        for batch_index, batch in enumerate(train_loader, start=1):
            batch = baseproj.move_batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    bbox=batch["bbox"],
                    pixel_values=batch["pixel_values"],
                )
                arcface_logits = model.arcface_head(outputs["embeddings"], batch["labels"])
                arcface_loss = F.cross_entropy(arcface_logits, batch["labels"])
                supcon_loss = baseproj.supervised_contrastive_loss(outputs["embeddings"], batch["labels"], temperature=args.supcon_temperature)
                total_batch_loss = args.arcface_weight * arcface_loss + args.supcon_weight * supcon_loss

            scaler.scale(total_batch_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += float(total_batch_loss.item())
            total_arcface_loss += float(arcface_loss.item())
            total_supcon_loss += float(supcon_loss.item())
            step_count += 1
            if batch_index == 1 or batch_index % 20 == 0 or batch_index == len(train_loader):
                logger.info(
                    "Epoch %s step %s/%s loss=%.4f arcface=%.4f supcon=%.4f",
                    epoch_index,
                    batch_index,
                    len(train_loader),
                    total_batch_loss.item(),
                    arcface_loss.item(),
                    supcon_loss.item(),
                )

        train_classifier_accuracy = baseproj.evaluate_classifier_accuracy(model, train_accuracy_loader, device)
        seen_val_classifier_accuracy = baseproj.evaluate_classifier_accuracy(model, seen_val_loader, device)

        train_embeddings_all, _ = baseproj.collect_embeddings(model, train_retrieval_loader, device)
        train_query_embeddings = train_embeddings_all[: len(train_query_documents)]
        train_original_embeddings = train_embeddings_all[len(train_query_documents) :]
        train_metrics = compute_retrieval_vs_original(train_query_documents, train_query_embeddings, train_original_documents, train_original_embeddings)

        val_embeddings_all, _ = baseproj.collect_embeddings(model, val_retrieval_loader, device)
        val_query_embeddings = val_embeddings_all[: len(val_query_documents)]
        val_original_embeddings = val_embeddings_all[len(val_query_documents) :]
        val_metrics = compute_retrieval_vs_original(val_query_documents, val_query_embeddings, val_original_documents, val_original_embeddings)

        history_rows.append(
            {
                "epoch": epoch_index,
                "train_loss": total_loss / max(1, step_count),
                "train_arcface_loss": total_arcface_loss / max(1, step_count),
                "train_supcon_loss": total_supcon_loss / max(1, step_count),
                "train_classifier_accuracy": train_classifier_accuracy,
                "seen_val_classifier_accuracy": seen_val_classifier_accuracy,
                "train_retrieval_at_1": train_metrics["retrieval_at_1"],
                "train_retrieval_at_5": train_metrics["retrieval_at_5"],
                "val_retrieval_at_1": val_metrics["retrieval_at_1"],
                "val_retrieval_at_5": val_metrics["retrieval_at_5"],
                "val_retrieval_at_10": val_metrics["retrieval_at_10"],
                "epoch_minutes": (time.time() - epoch_start) / 60.0,
            }
        )
        save_history_csv(output_dir, history_rows)
        plot_history(output_dir, history_rows)
        logger.info(
            "Epoch %s complete: train_loss=%.4f train_cls_acc=%.4f seen_val_cls_acc=%.4f train_r@1=%.4f val_r@1=%.4f val_r@5=%.4f",
            epoch_index,
            history_rows[-1]["train_loss"],
            train_classifier_accuracy,
            seen_val_classifier_accuracy,
            train_metrics["retrieval_at_1"],
            val_metrics["retrieval_at_1"],
            val_metrics["retrieval_at_5"],
        )

        if val_metrics["retrieval_at_1"] > best_metric:
            best_metric = float(val_metrics["retrieval_at_1"])
            best_val_metrics = val_metrics
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "epoch": epoch_index,
                    "best_metric": best_metric,
                    "args": training_config,
                },
                output_dir / "best_projection_model.pt",
            )
            logger.info("Saved new best checkpoint at epoch %s with val_retrieval_at_1=%.4f.", epoch_index, best_metric)

    checkpoint = torch.load(output_dir / "best_projection_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info("Loaded best checkpoint from epoch %s.", checkpoint["epoch"])

    train_embeddings_all, _ = baseproj.collect_embeddings(model, train_retrieval_loader, device)
    train_query_embeddings = train_embeddings_all[: len(train_query_documents)]
    train_original_embeddings = train_embeddings_all[len(train_query_documents) :]
    final_train_metrics = compute_retrieval_vs_original(train_query_documents, train_query_embeddings, train_original_documents, train_original_embeddings)
    save_retrieval_artifacts(output_dir, "train_retrieval", final_train_metrics, train_query_documents, train_query_embeddings, train_original_documents, train_original_embeddings)

    val_embeddings_all, _ = baseproj.collect_embeddings(model, val_retrieval_loader, device)
    val_query_embeddings = val_embeddings_all[: len(val_query_documents)]
    val_original_embeddings = val_embeddings_all[len(val_query_documents) :]
    final_val_metrics = compute_retrieval_vs_original(val_query_documents, val_query_embeddings, val_original_documents, val_original_embeddings)
    save_retrieval_artifacts(output_dir, "val_retrieval", final_val_metrics, val_query_documents, val_query_embeddings, val_original_documents, val_original_embeddings)

    test_embeddings_all, _ = baseproj.collect_embeddings(model, test_retrieval_loader, device)
    test_query_embeddings = test_embeddings_all[: len(test_query_documents)]
    test_original_embeddings = test_embeddings_all[len(test_query_documents) :]
    final_test_metrics = compute_retrieval_vs_original(test_query_documents, test_query_embeddings, test_original_documents, test_original_embeddings)
    save_retrieval_artifacts(output_dir, "test_retrieval", final_test_metrics, test_query_documents, test_query_embeddings, test_original_documents, test_original_embeddings)

    final_summary = {
        "best_epoch": checkpoint["epoch"],
        "best_val_retrieval_at_1": best_metric,
        "best_val_metrics": best_val_metrics,
        "final_train_metrics": final_train_metrics,
        "final_val_metrics": final_val_metrics,
        "final_test_metrics": final_test_metrics,
    }
    write_json(output_dir / "final_metrics_summary.json", final_summary)
    logger.info("Final train retrieval@1=%.4f", final_train_metrics["retrieval_at_1"])
    logger.info("Final val retrieval@1=%.4f", final_val_metrics["retrieval_at_1"])
    logger.info("Final test retrieval@1=%.4f", final_test_metrics["retrieval_at_1"])
    logger.info("Saved final metrics summary to %s", output_dir / "final_metrics_summary.json")


if __name__ == "__main__":
    main()
