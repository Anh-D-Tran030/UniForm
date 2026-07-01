from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytesseract
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader


EXPERIMENT_ROOT = Path(r"C:\Users\thanh\OneDrive - UTS\Personal_code\LayoutLMv3 Experiment")
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

import benchmark_image_data_projection as image_eval
import train_layoutlmv3_template_projection as proj
import visualize_layoutlmv3_template_embeddings as viz


DEFAULT_CHECKPOINT_DIR = EXPERIMENT_ROOT / "outputs" / "layoutlmv3_projection_eccv_tesseract_5ep_bs8"
DEFAULT_TESSERACT_CMD = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
DEFAULT_TEMPLATE_COUNT = 100
DEFAULT_QUERY_ROOT = Path(r"A:\RealForm\processed\synthetic_fill_images_10k_folders_degraded")
DEFAULT_QUERY_OCR_ROOT = Path(r"A:\RealForm\processed\synthetic_fill_images_10k_folders_degraded_ocr_cache\ocr_json")
DEFAULT_ORIGINAL_ROOT = Path(r"A:\RealForm\processed\CommonFormsEnglish\images")
DEFAULT_ORIGINAL_OCR_ROOT = Path(r"A:\RealForm\processed\CommonFormsEnglish_original_ocr_cache_projection")
DEFAULT_OUTPUT_DIR = Path(r"A:\RealForm\processed\projection_retrieval_benchmark_commonforms")


@dataclass(frozen=True)
class TemplateSelection:
    stem: str
    query_dir: Path
    original_image_path: Path


def log(message: str) -> None:
    print(message, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark retrieval@1 for degraded synthetic CommonForms queries against original template "
            "embeddings using the trained LayoutLMv3 projection checkpoint."
        )
    )
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--query-root", type=Path, default=DEFAULT_QUERY_ROOT)
    parser.add_argument("--query-ocr-root", type=Path, default=DEFAULT_QUERY_OCR_ROOT)
    parser.add_argument("--original-root", type=Path, default=DEFAULT_ORIGINAL_ROOT)
    parser.add_argument("--original-ocr-root", type=Path, default=DEFAULT_ORIGINAL_OCR_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--template-count", type=int, default=DEFAULT_TEMPLATE_COUNT)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--psm", type=int, default=11)
    parser.add_argument("--oem", type=int, default=3)
    parser.add_argument("--min-confidence", type=float, default=30.0)
    parser.add_argument("--tesseract-cmd", type=Path, default=DEFAULT_TESSERACT_CMD)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict[str, Any] | list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)


def configure_tesseract(executable_path: Path) -> None:
    if not executable_path.exists():
        raise FileNotFoundError(f"Tesseract executable was not found at {executable_path}.")
    pytesseract.pytesseract.tesseract_cmd = str(executable_path)


def normalize_box(box: list[int], width: int, height: int) -> list[int]:
    return viz.normalize_bbox([int(box[0]), int(box[1]), int(box[2]), int(box[3])], width, height)


def discover_original_image(original_root: Path, stem: str) -> Path | None:
    matches = sorted(
        [path for path in original_root.rglob("*") if path.is_file() and path.stem == stem],
        key=lambda path: str(path).lower(),
    )
    return matches[0] if matches else None


def count_matching_files(path: Path, pattern: str) -> int:
    if not path.exists():
        return 0
    return len(list(path.glob(pattern)))


def select_ready_templates(
    query_root: Path,
    query_ocr_root: Path,
    original_root: Path,
    template_count: int,
) -> list[TemplateSelection]:
    selections: list[TemplateSelection] = []
    for query_dir in sorted([path for path in query_root.iterdir() if path.is_dir()], key=lambda path: path.name.lower()):
        image_count = count_matching_files(query_dir, "fill_*.png")
        if image_count != 10:
            continue
        ocr_dir = query_ocr_root / query_dir.name
        ocr_count = count_matching_files(ocr_dir, "*.ocr.json")
        if ocr_count != 10:
            continue
        original_image_path = discover_original_image(original_root, query_dir.name)
        if original_image_path is None:
            continue
        selections.append(
            TemplateSelection(
                stem=query_dir.name,
                query_dir=query_dir,
                original_image_path=original_image_path,
            )
        )
        if len(selections) >= template_count:
            break
    if len(selections) < template_count:
        raise RuntimeError(f"Only found {len(selections)} ready templates; need {template_count}.")
    return selections


def make_original_cache_path(original_ocr_root: Path, stem: str) -> Path:
    return original_ocr_root / f"{stem}.ocr.json"


def run_tesseract(
    image_path: Path,
    psm: int,
    oem: int,
    min_confidence: float,
    timeout: float | None = None,
) -> dict[str, Any]:
    with Image.open(image_path) as image:
        rgb_image = image.convert("RGB")
        width, height = rgb_image.size
        scale_factor = 2.0
        preprocessed_image = ImageOps.autocontrast(rgb_image.convert("L")).resize(
            (int(width * scale_factor), int(height * scale_factor)),
            resample=Image.Resampling.BICUBIC,
        )
        config = f"--psm {psm} --oem {oem}"
        data = pytesseract.image_to_data(
            preprocessed_image,
            output_type=pytesseract.Output.DICT,
            config=config,
            lang="eng",
            timeout=timeout,
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
) -> None:
    configure_tesseract(tesseract_cmd.resolve())
    built = 0
    reused = 0
    for selection in selections:
        cache_path = make_original_cache_path(original_ocr_root, selection.stem)
        if cache_path.exists():
            reused += 1
            continue
        payload = run_tesseract(selection.original_image_path, psm=psm, oem=oem, min_confidence=min_confidence)
        payload.update(
            {
                "template_stem": selection.stem,
                "is_original": True,
            }
        )
        write_json(cache_path, payload)
        built += 1
    log(f"Original OCR cache ready: built={built}, reused={reused}")


def build_documents(
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
        query_image_paths = sorted(selection.query_dir.glob("fill_*.png"), key=lambda path: path.name.lower())
        for record_id, image_path in enumerate(query_image_paths, start=1):
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


def load_projection_model(checkpoint_dir: Path, device: torch.device) -> tuple[proj.LayoutLMv3TemplateProjectionModel, Any, dict[str, Any]]:
    checkpoint_dir = checkpoint_dir.resolve()
    training_config = read_json(checkpoint_dir / "training_config.json")
    checkpoint = torch.load(checkpoint_dir / "best_projection_model.pt", map_location=device, weights_only=False)
    state_dict = checkpoint["model_state_dict"]
    num_train_classes = int(state_dict["arcface_head.weight"].shape[0])

    model = proj.LayoutLMv3TemplateProjectionModel(
        model_name=training_config["model_name"],
        projection_dim=int(training_config["projection_dim"]),
        pooling=training_config["pooling"],
        num_train_classes=num_train_classes,
        arcface_margin=float(training_config["arcface_margin"]),
        arcface_scale=float(training_config["arcface_scale"]),
    )
    model.load_state_dict(state_dict)
    model.to(device)
    processor = viz.load_processor(training_config["model_name"])
    return model, processor, training_config


def collect_projection_embeddings(
    model: proj.LayoutLMv3TemplateProjectionModel,
    processor: Any,
    documents: list[viz.DocumentSample],
    batch_size: int,
    max_length: int,
    device: torch.device,
) -> np.ndarray:
    examples = []
    class_to_id = {label: index for index, label in enumerate(sorted({document.class_label for document in documents}))}
    for document in documents:
        examples.append(
            proj.TemplateExample(
                document=document,
                class_label=document.class_label,
                train_label_id=class_to_id[document.class_label],
            )
        )

    collator = proj.ProcessorCollator(processor, max_length=max_length)
    data_loader = DataLoader(
        proj.TemplateDocumentDataset(examples),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collator,
    )
    embeddings, ordered_documents = proj.collect_embeddings(model, data_loader, device)
    if [str(document.image_path) for document in ordered_documents] != [str(document.image_path) for document in documents]:
        raise RuntimeError("Projection dataloader changed document ordering unexpectedly.")
    return embeddings


def compute_retrieval_metrics(
    query_documents: list[viz.DocumentSample],
    query_embeddings: np.ndarray,
    original_documents: list[viz.DocumentSample],
    original_embeddings: np.ndarray,
) -> dict[str, Any]:
    normalized_queries = query_embeddings / np.maximum(np.linalg.norm(query_embeddings, axis=1, keepdims=True), 1e-12)
    normalized_originals = original_embeddings / np.maximum(np.linalg.norm(original_embeddings, axis=1, keepdims=True), 1e-12)
    scores = normalized_queries @ normalized_originals.T

    top1 = 0
    top5 = 0
    top10 = 0
    reciprocal_ranks: list[float] = []
    failures: list[dict[str, Any]] = []

    original_labels = [document.stem for document in original_documents]
    for query_index, (query_document, similarities) in enumerate(zip(query_documents, scores)):
        order = np.argsort(-similarities)
        ranked_labels = [original_labels[index] for index in order]
        true_label = query_document.stem
        top1_hit = ranked_labels[0] == true_label
        top1 += int(top1_hit)
        top5 += int(true_label in ranked_labels[:5])
        top10 += int(true_label in ranked_labels[:10])

        reciprocal_rank = 0.0
        for rank, label in enumerate(ranked_labels, start=1):
            if label == true_label:
                reciprocal_rank = 1.0 / rank
                break
        reciprocal_ranks.append(reciprocal_rank)

        if not top1_hit and len(failures) < 25:
            failures.append(
                {
                    "query_image": str(query_document.image_path),
                    "true_template": true_label,
                    "predicted_template_top1": ranked_labels[0],
                    "top5_templates": ranked_labels[:5],
                    "top5_scores": [float(similarities[index]) for index in order[:5]],
                }
            )

    query_count = len(query_documents)
    return {
        "gallery_templates": len(original_documents),
        "query_images": query_count,
        "retrieval_at_1": top1 / max(query_count, 1),
        "retrieval_at_5": top5 / max(query_count, 1),
        "retrieval_at_10": top10 / max(query_count, 1),
        "mrr": float(sum(reciprocal_ranks) / max(query_count, 1)),
        "correct_top1_queries": top1,
        "failures_preview": failures,
    }


def serialize_documents(documents: list[viz.DocumentSample]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, document in enumerate(documents):
        rows.append(
            {
                "row_index": index,
                "template_stem": document.stem,
                "class_label": document.class_label,
                "image_path": str(document.image_path),
                "record_id": int(document.record_id),
                "is_original": bool(document.is_original),
                "word_count": len(document.words),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    selections = select_ready_templates(
        query_root=args.query_root.resolve(),
        query_ocr_root=args.query_ocr_root.resolve(),
        original_root=args.original_root.resolve(),
        template_count=args.template_count,
    )
    log(f"Selected {len(selections)} ready templates for benchmarking.")
    write_json(output_dir / "selected_templates.json", [selection.stem for selection in selections])

    ensure_original_ocr_cache(
        selections=selections,
        original_ocr_root=args.original_ocr_root.resolve(),
        tesseract_cmd=args.tesseract_cmd.resolve(),
        psm=args.psm,
        oem=args.oem,
        min_confidence=args.min_confidence,
    )

    query_documents, original_documents = build_documents(
        selections=selections,
        query_ocr_root=args.query_ocr_root.resolve(),
        original_ocr_root=args.original_ocr_root.resolve(),
    )
    log(f"Prepared {len(query_documents)} degraded query images and {len(original_documents)} original gallery images.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, processor, training_config = load_projection_model(args.checkpoint_dir.resolve(), device)
    log(f"Loaded checkpoint on device={device}. pooling={training_config['pooling']} projection_dim={training_config['projection_dim']}")

    original_embeddings = collect_projection_embeddings(
        model=model,
        processor=processor,
        documents=original_documents,
        batch_size=args.batch_size,
        max_length=int(training_config["max_length"]),
        device=device,
    )
    query_embeddings = collect_projection_embeddings(
        model=model,
        processor=processor,
        documents=query_documents,
        batch_size=args.batch_size,
        max_length=int(training_config["max_length"]),
        device=device,
    )

    write_npy(output_dir / "original_embeddings.npy", original_embeddings)
    write_npy(output_dir / "query_embeddings.npy", query_embeddings)
    write_json(output_dir / "original_embedding_rows.json", serialize_documents(original_documents))
    write_json(output_dir / "query_embedding_rows.json", serialize_documents(query_documents))

    metrics = compute_retrieval_metrics(
        query_documents=query_documents,
        query_embeddings=query_embeddings,
        original_documents=original_documents,
        original_embeddings=original_embeddings,
    )
    summary = {
        "checkpoint_dir": str(args.checkpoint_dir.resolve()),
        "device": str(device),
        "template_count": len(selections),
        "query_count": len(query_documents),
        "gallery_count": len(original_documents),
        "assumption": "Queries are degraded synthetic fill images; gallery is the unfilled original template image for each template.",
        "metrics": metrics,
    }
    write_json(output_dir / "retrieval_benchmark_summary.json", summary)
    log(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
