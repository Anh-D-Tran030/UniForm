from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader


EXPERIMENT_ROOT = Path(r"C:\Users\thanh\OneDrive - UTS\Personal_code\LayoutLMv3 Experiment")
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import benchmark_commonforms_projection_retrieval as bench
import train_layoutlmv3_template_projection as proj
import visualize_layoutlmv3_template_embeddings as viz


DEFAULT_META_ROOT = Path(r"A:\FDT_TO _PROCESS\meta_training")
DEFAULT_OCR_CACHE_ROOT = Path(r"A:\RealForm\processed\meta_training_ocr_cache")
DEFAULT_OUTPUT_DIR = Path(r"A:\RealForm\processed\meta_training_projection_benchmark")
DEFAULT_TESSERACT_CMD = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")


@dataclass(frozen=True)
class TemplateImage:
    template_name: str
    image_path: Path
    role: str
    record_id: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark projection retrieval on meta_training templates.")
    parser.add_argument("--meta-root", type=Path, default=DEFAULT_META_ROOT)
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


def write_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


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


def discover_meta_images(meta_root: Path, output_dir: Path) -> tuple[list[TemplateImage], list[TemplateImage]]:
    original_images: list[TemplateImage] = []
    query_images: list[TemplateImage] = []
    manifests: list[dict[str, Any]] = []

    template_dirs = sorted([path for path in meta_root.iterdir() if path.is_dir()], key=lambda path: path.name.lower())
    if not template_dirs:
        raise FileNotFoundError(f"No template directories found under {meta_root}.")

    for template_dir in template_dirs:
        original_path = template_dir / "original_template.png"
        if not original_path.exists():
            continue
        page_paths = sorted(template_dir.glob("page_*.png"), key=lambda path: path.name.lower())
        if not page_paths:
            continue

        original_images.append(
            TemplateImage(
                template_name=template_dir.name,
                image_path=original_path.resolve(),
                role="original",
                record_id=1,
            )
        )
        page_records: list[dict[str, Any]] = [{"role": "original", "image_path": str(original_path.resolve())}]
        for index, page_path in enumerate(page_paths, start=2):
            query_images.append(
                TemplateImage(
                    template_name=template_dir.name,
                    image_path=page_path.resolve(),
                    role="query",
                    record_id=index,
                )
            )
            page_records.append({"role": "query", "image_path": str(page_path.resolve())})

        manifests.append(
            {
                "template_name": template_dir.name,
                "original_image": str(original_path.resolve()),
                "query_count": len(page_paths),
                "pages": page_records,
            }
        )

    if not original_images or not query_images:
        raise RuntimeError("Need at least one original_template.png and one page_*.png query image.")

    write_json(output_dir / "discovery_manifest.json", manifests)
    return original_images, query_images


def build_ocr_cache(images: list[TemplateImage], ocr_cache_root: Path, tesseract_cmd: Path, psm: int, oem: int, min_confidence: float) -> None:
    ocr_cache_root.mkdir(parents=True, exist_ok=True)
    bench.configure_tesseract(tesseract_cmd.resolve())
    for index, image_row in enumerate(images, start=1):
        cache_path = ocr_cache_root / image_row.template_name / f"{image_row.image_path.stem}.ocr.json"
        if cache_path.exists():
            continue
        payload = bench.run_tesseract(image_row.image_path, psm=psm, oem=oem, min_confidence=min_confidence)
        payload.update(
            {
                "template_name": image_row.template_name,
                "role": image_row.role,
                "record_id": image_row.record_id,
                "source_image": str(image_row.image_path),
            }
        )
        write_json(cache_path, payload)
        if index == 1 or index % 25 == 0 or index == len(images):
            print(f"OCR cached {index}/{len(images)} images.", flush=True)


def build_documents(images: list[TemplateImage], ocr_cache_root: Path) -> list[viz.DocumentSample]:
    documents: list[viz.DocumentSample] = []
    for image_row in images:
        payload = read_json(ocr_cache_root / image_row.template_name / f"{image_row.image_path.stem}.ocr.json")
        width = int(payload["image_size"]["width"])
        height = int(payload["image_size"]["height"])
        documents.append(
            viz.DocumentSample(
                subset="meta_training",
                stem=image_row.template_name,
                class_label=image_row.template_name,
                image_path=image_row.image_path,
                record_id=image_row.record_id,
                words=list(payload["words"]),
                boxes=[bench.normalize_box(box, width, height) for box in payload["boxes"]],
                is_original=image_row.role == "original",
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
    checkpoint_dir = resolve_checkpoint_dir(args.checkpoint_dir)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    original_images, query_images = discover_meta_images(args.meta_root.resolve(), output_dir)
    all_images = original_images + query_images
    print(
        f"Discovered templates={len(original_images)} original_images={len(original_images)} query_images={len(query_images)} checkpoint={checkpoint_dir}",
        flush=True,
    )

    build_ocr_cache(
        images=all_images,
        ocr_cache_root=args.ocr_cache_root.resolve(),
        tesseract_cmd=args.tesseract_cmd.resolve(),
        psm=args.psm,
        oem=args.oem,
        min_confidence=args.min_confidence,
    )

    original_documents = build_documents(original_images, args.ocr_cache_root.resolve())
    query_documents = build_documents(query_images, args.ocr_cache_root.resolve())

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
            "meta_root": str(args.meta_root.resolve()),
        }
    )
    write_json(output_dir / "benchmark_summary.json", summary)
    write_npy(output_dir / "original_embeddings.npy", original_embeddings)
    write_npy(output_dir / "query_embeddings.npy", query_embeddings)
    write_rows(output_dir / "original_embedding_rows.json", original_documents)
    write_rows(output_dir / "query_embedding_rows.json", query_documents)
    print(f"Saved benchmark outputs to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
