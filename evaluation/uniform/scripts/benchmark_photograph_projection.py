from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytesseract
import torch
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener
from torch.utils.data import DataLoader


register_heif_opener()

EXPERIMENT_ROOT = Path(r"C:\Users\thanh\OneDrive - UTS\Personal_code\LayoutLMv3 Experiment")
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import benchmark_commonforms_projection_retrieval as bench
import train_layoutlmv3_template_projection as proj
import visualize_layoutlmv3_template_embeddings as viz


DEFAULT_PHOTO_ROOT = Path(r"A:\FDT_TO _PROCESS\Photograph")
DEFAULT_IMAGE_ROOT = Path(r"A:\RealForm\processed\Photograph_image_data")
DEFAULT_OCR_CACHE_ROOT = Path(r"A:\RealForm\processed\Photograph_ocr_cache")
DEFAULT_OUTPUT_DIR = Path(r"A:\RealForm\processed\Photograph_projection_benchmark")
DEFAULT_TESSERACT_CMD = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".heic", ".heif"}
PHOTO_OCR_MAX_SIDE = 2200
PHOTO_OCR_TIMEOUT_SECONDS = 45


@dataclass(frozen=True)
class PhotographPage:
    template_name: str
    role: str
    record_id: int
    source_path: Path
    image_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark projection retrieval on photographed forms.")
    parser.add_argument("--photo-root", type=Path, default=DEFAULT_PHOTO_ROOT)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--ocr-cache-root", type=Path, default=DEFAULT_OCR_CACHE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--tesseract-cmd", type=Path, default=DEFAULT_TESSERACT_CMD)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--psm", type=int, default=11)
    parser.add_argument("--oem", type=int, default=3)
    parser.add_argument("--min-confidence", type=float, default=30.0)
    return parser.parse_args()


def write_json(path: Path, payload: dict[str, Any] | list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)


def resolve_checkpoint_dir(explicit_checkpoint_dir: Path | None) -> Path:
    if explicit_checkpoint_dir is not None:
        checkpoint_dir = explicit_checkpoint_dir.resolve()
        if not (checkpoint_dir / "best_projection_model.pt").exists():
            raise FileNotFoundError(f"No best_projection_model.pt in {checkpoint_dir}")
        return checkpoint_dir

    candidates = sorted(
        Path(r"A:\RealForm\outputs").glob("**/best_projection_model.pt"),
        key=lambda candidate: candidate.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError("Could not find any best_projection_model.pt under A:\\RealForm\\outputs.")
    return candidates[0].parent.resolve()


def convert_to_png(source_path: Path, output_path: Path) -> None:
    if output_path.exists():
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source_path) as image:
        fixed = ImageOps.exif_transpose(image).convert("RGB")
        fixed.save(output_path)
        fixed.close()


def valid_ocr_cache(cache_path: Path) -> bool:
    if not cache_path.exists():
        return False
    try:
        payload = read_json(cache_path)
    except Exception:
        return False
    return isinstance(payload, dict) and "image_size" in payload and "words" in payload and "boxes" in payload


def run_photo_tesseract(image_path: Path, psm: int, oem: int, min_confidence: float) -> dict[str, Any]:
    with Image.open(image_path) as image:
        rgb_image = ImageOps.exif_transpose(image).convert("RGB")
        width, height = rgb_image.size
        scale = min(1.0, PHOTO_OCR_MAX_SIDE / max(width, height))
        if scale < 1.0:
            ocr_image = rgb_image.resize(
                (max(1, int(width * scale)), max(1, int(height * scale))),
                resample=Image.Resampling.BICUBIC,
            )
        else:
            ocr_image = rgb_image.copy()
        ocr_image = ImageOps.autocontrast(ocr_image.convert("L"))
        data = pytesseract.image_to_data(
            ocr_image,
            output_type=pytesseract.Output.DICT,
            config=f"--psm {psm} --oem {oem}",
            lang="eng",
            timeout=PHOTO_OCR_TIMEOUT_SECONDS,
        )
        ocr_image.close()
        rgb_image.close()

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

        left = int(round(int(data["left"][index]) / max(scale, 1e-6)))
        top = int(round(int(data["top"][index]) / max(scale, 1e-6)))
        box_width = max(1, int(round(int(data["width"][index]) / max(scale, 1e-6))))
        box_height = max(1, int(round(int(data["height"][index]) / max(scale, 1e-6))))
        words.append(text)
        boxes.append([left, top, left + box_width, top + box_height])
        confidences.append(confidence)

    if not words:
        words = ["[EMPTY]"]
        boxes = [[0, 0, width, height]]
        confidences = [0.0]

    return {
        "image_path": str(image_path.resolve()),
        "image_size": {"width": width, "height": height},
        "words": words,
        "boxes": boxes,
        "confidences": confidences,
        "ocr_max_side": PHOTO_OCR_MAX_SIDE,
        "ocr_timeout_seconds": PHOTO_OCR_TIMEOUT_SECONDS,
    }


def prepare_photograph_image_data(photo_root: Path, image_root: Path) -> tuple[list[PhotographPage], list[PhotographPage]]:
    original_pages: list[PhotographPage] = []
    query_pages: list[PhotographPage] = []
    manifest: list[dict[str, Any]] = []

    template_dirs = sorted([path for path in photo_root.iterdir() if path.is_dir()], key=lambda path: path.name.lower())
    for template_dir in template_dirs:
        image_paths = sorted(
            [
                path
                for path in template_dir.iterdir()
                if path.is_file() and path.suffix.lower() in VALID_IMAGE_EXTS
            ],
            key=lambda path: path.name.lower(),
        )
        if len(image_paths) < 2:
            continue

        template_output_dir = image_root / template_dir.name
        page_rows: list[dict[str, Any]] = []
        for index, source_path in enumerate(image_paths, start=1):
            role = "original" if index == 1 else "query"
            output_path = template_output_dir / f"image_{index:03d}_{role}.png"
            convert_to_png(source_path, output_path)
            page = PhotographPage(
                template_name=template_dir.name,
                role=role,
                record_id=index,
                source_path=source_path.resolve(),
                image_path=output_path.resolve(),
            )
            if role == "original":
                original_pages.append(page)
            else:
                query_pages.append(page)
            page_rows.append(
                {
                    "record_id": index,
                    "role": role,
                    "source_path": str(source_path.resolve()),
                    "image_path": str(output_path.resolve()),
                }
            )

        manifest.append(
            {
                "template_name": template_dir.name,
                "original_source": str(image_paths[0].resolve()),
                "query_count": len(image_paths) - 1,
                "pages": page_rows,
            }
        )

    if not original_pages or not query_pages:
        raise RuntimeError(f"No usable photograph template folders found under {photo_root}.")
    write_json(image_root / "photograph_manifest.json", manifest)
    return original_pages, query_pages


def build_ocr_cache(
    pages: list[PhotographPage],
    ocr_cache_root: Path,
    tesseract_cmd: Path,
    psm: int,
    oem: int,
    min_confidence: float,
) -> None:
    ocr_cache_root.mkdir(parents=True, exist_ok=True)
    bench.configure_tesseract(tesseract_cmd.resolve())
    for index, page in enumerate(pages, start=1):
        cache_path = ocr_cache_root / page.template_name / f"{page.image_path.stem}.ocr.json"
        if valid_ocr_cache(cache_path):
            continue
        try:
            payload = run_photo_tesseract(page.image_path, psm=psm, oem=oem, min_confidence=min_confidence)
        except RuntimeError as exc:
            # Tesseract timeout or a single damaged photo should not block the full benchmark.
            with Image.open(page.image_path) as image:
                width, height = image.size
            payload = {
                "image_path": str(page.image_path.resolve()),
                "image_size": {"width": width, "height": height},
                "words": ["[EMPTY]"],
                "boxes": [[0, 0, width, height]],
                "confidences": [0.0],
                "ocr_error": str(exc),
            }
        payload.update(
            {
                "template_name": page.template_name,
                "role": page.role,
                "record_id": page.record_id,
                "source_path": str(page.source_path),
                "image_path": str(page.image_path),
            }
        )
        write_json(cache_path, payload)
        if index == 1 or index % 25 == 0 or index == len(pages):
            print(f"OCR cached {index}/{len(pages)} photograph images.", flush=True)


def build_documents(pages: list[PhotographPage], ocr_cache_root: Path) -> list[viz.DocumentSample]:
    documents: list[viz.DocumentSample] = []
    for page in pages:
        payload = read_json(ocr_cache_root / page.template_name / f"{page.image_path.stem}.ocr.json")
        width = int(payload["image_size"]["width"])
        height = int(payload["image_size"]["height"])
        documents.append(
            viz.DocumentSample(
                subset="photograph",
                stem=page.template_name,
                class_label=page.template_name,
                image_path=page.image_path,
                record_id=page.record_id,
                words=list(payload["words"]),
                boxes=[bench.normalize_box(box, width, height) for box in payload["boxes"]],
                is_original=page.role == "original",
            )
        )
    return documents


def collect_projection_embeddings(
    model: proj.LayoutLMv3TemplateProjectionModel,
    processor: Any,
    documents: list[viz.DocumentSample],
    batch_size: int,
    max_length: int,
    device: torch.device,
) -> np.ndarray:
    class_to_id = {label: index for index, label in enumerate(sorted({document.class_label for document in documents}))}
    examples = [
        proj.TemplateExample(document=document, class_label=document.class_label, train_label_id=class_to_id[document.class_label])
        for document in documents
    ]
    collator = proj.ProcessorCollator(processor, max_length=max_length)
    loader = DataLoader(
        proj.TemplateDocumentDataset(examples),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collator,
    )
    embeddings, ordered_documents = proj.collect_embeddings(model, loader, device)
    if [str(document.image_path) for document in ordered_documents] != [str(document.image_path) for document in documents]:
        raise RuntimeError("Document ordering changed during embedding collection.")
    return embeddings


def write_rows(path: Path, documents: list[viz.DocumentSample]) -> None:
    rows = [
        {
            "template_name": document.stem,
            "record_id": document.record_id,
            "image_path": str(document.image_path),
            "is_original": bool(document.is_original),
        }
        for document in documents
    ]
    write_json(path, rows)


def save_rankings(
    output_dir: Path,
    query_documents: list[viz.DocumentSample],
    original_documents: list[viz.DocumentSample],
    query_embeddings: np.ndarray,
    original_embeddings: np.ndarray,
) -> dict[str, Any]:
    normalized_queries = query_embeddings / np.maximum(np.linalg.norm(query_embeddings, axis=1, keepdims=True), 1e-12)
    normalized_originals = original_embeddings / np.maximum(np.linalg.norm(original_embeddings, axis=1, keepdims=True), 1e-12)
    scores = normalized_queries @ normalized_originals.T
    original_labels = [document.stem for document in original_documents]

    top1 = 0
    top5 = 0
    top10 = 0
    reciprocal_ranks: list[float] = []
    rows: list[dict[str, Any]] = []

    for query_document, similarities in zip(query_documents, scores, strict=False):
        order = np.argsort(-similarities)
        ranked_labels = [original_labels[index] for index in order]
        true_label = query_document.stem
        top1 += int(ranked_labels[0] == true_label)
        top5 += int(true_label in ranked_labels[:5])
        top10 += int(true_label in ranked_labels[:10])

        reciprocal_rank = 0.0
        for rank, label in enumerate(ranked_labels, start=1):
            if label == true_label:
                reciprocal_rank = 1.0 / rank
                break
        reciprocal_ranks.append(reciprocal_rank)

        row: dict[str, Any] = {
            "query_image": str(query_document.image_path),
            "true_template": true_label,
            "predicted_template_top1": ranked_labels[0],
            "top1_score": float(similarities[order[0]]),
            "hit_at_1": ranked_labels[0] == true_label,
        }
        for slot in range(5):
            if slot < len(order):
                row[f"rank_{slot + 1}_template"] = ranked_labels[slot]
                row[f"rank_{slot + 1}_score"] = float(similarities[order[slot]])
        rows.append(row)

    with (output_dir / "per_query_rankings.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    per_template: dict[str, list[int]] = {}
    for row in rows:
        per_template.setdefault(str(row["true_template"]), []).append(int(bool(row["hit_at_1"])))
    with (output_dir / "per_template_top1.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["template_name", "top1_accuracy"])
        writer.writeheader()
        for template_name, hits in sorted(per_template.items()):
            writer.writerow({"template_name": template_name, "top1_accuracy": sum(hits) / max(1, len(hits))})

    return {
        "gallery_templates": len(original_documents),
        "query_images": len(query_documents),
        "retrieval_at_1": top1 / max(1, len(query_documents)),
        "retrieval_at_5": top5 / max(1, len(query_documents)),
        "retrieval_at_10": top10 / max(1, len(query_documents)),
        "mrr": float(sum(reciprocal_ranks) / max(1, len(query_documents))),
    }


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = resolve_checkpoint_dir(args.checkpoint_dir)

    original_pages, query_pages = prepare_photograph_image_data(args.photo_root.resolve(), args.image_root.resolve())
    all_pages = original_pages + query_pages
    print(
        f"Prepared photograph benchmark: originals={len(original_pages)} queries={len(query_pages)} checkpoint={checkpoint_dir}",
        flush=True,
    )

    build_ocr_cache(
        pages=all_pages,
        ocr_cache_root=args.ocr_cache_root.resolve(),
        tesseract_cmd=args.tesseract_cmd.resolve(),
        psm=args.psm,
        oem=args.oem,
        min_confidence=args.min_confidence,
    )

    original_documents = build_documents(original_pages, args.ocr_cache_root.resolve())
    query_documents = build_documents(query_pages, args.ocr_cache_root.resolve())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, processor, training_config = bench.load_projection_model(checkpoint_dir, device)
    model.eval()
    print(f"Loaded checkpoint on device={device}. originals={len(original_documents)} queries={len(query_documents)}", flush=True)

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

    summary = save_rankings(output_dir, query_documents, original_documents, query_embeddings, original_embeddings)
    summary.update(
        {
            "checkpoint_dir": str(checkpoint_dir),
            "device": str(device),
            "template_count": len(original_documents),
            "query_count": len(query_documents),
            "photo_root": str(args.photo_root.resolve()),
            "image_root": str(args.image_root.resolve()),
            "ocr_cache_root": str(args.ocr_cache_root.resolve()),
            "original_selection_rule": "first filename after case-insensitive sort inside each Photograph template folder",
            "ocr_corruption": False,
        }
    )
    write_json(output_dir / "benchmark_summary.json", summary)
    write_npy(output_dir / "original_embeddings.npy", original_embeddings)
    write_npy(output_dir / "query_embeddings.npy", query_embeddings)
    write_rows(output_dir / "original_embedding_rows.json", original_documents)
    write_rows(output_dir / "query_embedding_rows.json", query_documents)
    print(f"Saved benchmark outputs to {output_dir}", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
