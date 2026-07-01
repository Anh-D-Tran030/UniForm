from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import zlib
from io import BytesIO
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pypdfium2 as pdfium
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


DEFAULT_PDF_ROOT = Path(r"A:\RealForm\Test_data_pdf")
DEFAULT_IMAGE_ROOT = Path(r"A:\RealForm\processed\Test_data_pdf_image_data")
DEFAULT_OCR_CACHE_ROOT = Path(r"A:\RealForm\processed\Test_data_pdf_ocr_cache")
DEFAULT_OUTPUT_DIR = Path(r"A:\RealForm\processed\Test_data_pdf_projection_benchmark")
DEFAULT_CHECKPOINT_DIR = Path(r"A:\RealForm\outputs\layoutlmv3_projection_commonforms_2k_300_300_5ep_20260403-003909")
DEFAULT_TESSERACT_CMD = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")


@dataclass(frozen=True)
class TemplatePage:
    template_name: str
    page_index: int
    role: str
    image_path: Path
    pdf_name: str

    @property
    def cache_path(self) -> Path:
        return DEFAULT_OCR_CACHE_ROOT / self.template_name / f"{self.image_path.stem}.ocr.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract scanned test PDFs and benchmark projection retrieval.")
    parser.add_argument("--pdf-root", type=Path, default=DEFAULT_PDF_ROOT)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--ocr-cache-root", type=Path, default=DEFAULT_OCR_CACHE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--tesseract-cmd", type=Path, default=DEFAULT_TESSERACT_CMD)
    parser.add_argument("--render-scale", type=float, default=2.5)
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


def slugify(text: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", text.strip())
    normalized = normalized.strip("_")
    return normalized or "template"


def discover_pdfs(pdf_root: Path) -> list[Path]:
    pdfs: list[Path] = []
    for path in sorted([candidate for candidate in pdf_root.iterdir() if candidate.is_file()], key=lambda candidate: candidate.name.lower()):
        with path.open("rb") as handle:
            signature = handle.read(5)
        if signature == b"%PDF-":
            pdfs.append(path)
    if not pdfs:
        raise FileNotFoundError(f"No PDFs found under {pdf_root}.")
    return pdfs


def extract_image_streams_from_broken_pdf(pdf_path: Path, template_dir: Path) -> list[Path]:
    data = pdf_path.read_bytes()
    pattern = re.compile(
        rb"<<(?P<dict>.*?/Type\s*/XObject\s*/Subtype\s*/Image.*?/Width\s+(?P<width>\d+).*?/Height\s+(?P<height>\d+).*?/BitsPerComponent\s+(?P<bits>\d+).*?/Length\s+(?P<length>\d+).*?/Filter\s*/(?P<filter>[A-Za-z0-9]+).*?)>>\s*stream\r?\n",
        re.DOTALL,
    )
    extracted_paths: list[Path] = []
    for image_index, match in enumerate(pattern.finditer(data), start=1):
        width = int(match.group("width"))
        height = int(match.group("height"))
        bits = int(match.group("bits"))
        stream_length = int(match.group("length"))
        filter_name = match.group("filter").decode("ascii", errors="ignore")
        stream_start = match.end()
        stream_end = min(len(data), stream_start + stream_length)
        raw_stream = data[stream_start:stream_end]
        if width < 100 or height < 100:
            continue
        if bits != 8:
            continue

        image: Image.Image | None = None
        if filter_name == "FlateDecode":
            try:
                decoded = zlib.decompress(raw_stream)
            except zlib.error:
                decoded = zlib.decompressobj().decompress(raw_stream)
            pixel_count = width * height
            if len(decoded) == pixel_count * 3:
                image = Image.frombytes("RGB", (width, height), decoded)
            elif len(decoded) == pixel_count:
                image = Image.frombytes("L", (width, height), decoded).convert("RGB")
            elif len(decoded) == pixel_count * 4:
                image = Image.frombytes("RGBA", (width, height), decoded).convert("RGB")
            elif len(decoded) < pixel_count * 3 and len(decoded) >= width * 3:
                expected = pixel_count * 3
                padded = decoded + (b"\xff" * (expected - len(decoded)))
                image = Image.frombytes("RGB", (width, height), padded)
        elif filter_name == "DCTDecode":
            try:
                image = Image.open(BytesIO(raw_stream)).convert("RGB")
            except Exception:
                image = None

        if image is None:
            continue

        output_path = template_dir / f"page_{image_index:02d}_{'original' if image_index == 1 else 'filled'}.png"
        image.save(output_path)
        image.close()
        extracted_paths.append(output_path)
    return extracted_paths


def extract_pdfs_to_images(pdf_paths: list[Path], image_root: Path, render_scale: float) -> list[TemplatePage]:
    image_root.mkdir(parents=True, exist_ok=True)
    seen_names: dict[str, int] = {}
    pages: list[TemplatePage] = []
    extraction_manifest: list[dict[str, Any]] = []

    for pdf_path in pdf_paths:
        base_name = slugify(pdf_path.stem)
        seen_names[base_name] = seen_names.get(base_name, 0) + 1
        template_name = base_name if seen_names[base_name] == 1 else f"{base_name}_{seen_names[base_name]}"
        template_dir = image_root / template_name
        template_dir.mkdir(parents=True, exist_ok=True)

        page_records: list[dict[str, Any]] = []
        extracted_paths: list[Path] = []
        try:
            document = pdfium.PdfDocument(str(pdf_path))
            page_count = len(document)
            for page_index in range(page_count):
                role = "original" if page_index == 0 else "filled"
                image_name = f"page_{page_index + 1:02d}_{role}.png"
                image_path = template_dir / image_name
                if not image_path.exists():
                    page = document.get_page(page_index)
                    bitmap = page.render(scale=render_scale)
                    pil_image = bitmap.to_pil()
                    pil_image.save(image_path)
                    pil_image.close()
                    page.close()
                extracted_paths.append(image_path)
            document.close()
        except Exception:
            extracted_paths = extract_image_streams_from_broken_pdf(pdf_path, template_dir)
            page_count = len(extracted_paths)

        if page_count == 0:
            raise RuntimeError(f"Could not extract any page images from {pdf_path.name}.")

        for page_index, image_path in enumerate(extracted_paths):
            role = "original" if page_index == 0 else "filled"
            pages.append(
                TemplatePage(
                    template_name=template_name,
                    page_index=page_index,
                    role=role,
                    image_path=image_path,
                    pdf_name=pdf_path.name,
                )
            )
            with Image.open(image_path) as image:
                width, height = image.size
            page_records.append(
                {
                    "page_index": page_index + 1,
                    "role": role,
                    "image_path": str(image_path),
                    "size": {"width": width, "height": height},
                }
            )
        extraction_manifest.append(
            {
                "template_name": template_name,
                "pdf_name": pdf_path.name,
                "page_count": page_count,
                "pages": page_records,
            }
        )

    write_json(image_root / "extraction_manifest.json", extraction_manifest)
    return pages


def build_ocr_cache(pages: list[TemplatePage], ocr_cache_root: Path, tesseract_cmd: Path, psm: int, oem: int, min_confidence: float) -> None:
    ocr_cache_root.mkdir(parents=True, exist_ok=True)
    bench.configure_tesseract(tesseract_cmd.resolve())
    for index, page in enumerate(pages, start=1):
        cache_path = ocr_cache_root / page.template_name / f"{page.image_path.stem}.ocr.json"
        if cache_path.exists():
            continue
        payload = bench.run_tesseract(page.image_path, psm=psm, oem=oem, min_confidence=min_confidence)
        payload.update(
            {
                "template_name": page.template_name,
                "page_index": page.page_index + 1,
                "role": page.role,
                "source_pdf": page.pdf_name,
            }
        )
        write_json(cache_path, payload)
        if index == 1 or index % 10 == 0 or index == len(pages):
            print(f"OCR cached {index}/{len(pages)} extracted pages.", flush=True)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_documents(pages: list[TemplatePage], ocr_cache_root: Path) -> tuple[list[viz.DocumentSample], list[viz.DocumentSample]]:
    query_documents: list[viz.DocumentSample] = []
    original_documents: list[viz.DocumentSample] = []

    for page in pages:
        payload = read_json(ocr_cache_root / page.template_name / f"{page.image_path.stem}.ocr.json")
        width = int(payload["image_size"]["width"])
        height = int(payload["image_size"]["height"])
        document = viz.DocumentSample(
            subset="test_pdf",
            stem=page.template_name,
            class_label=page.template_name,
            image_path=page.image_path.resolve(),
            record_id=page.page_index + 1,
            words=list(payload["words"]),
            boxes=[bench.normalize_box(box, width, height) for box in payload["boxes"]],
            is_original=page.role == "original",
        )
        if page.role == "original":
            original_documents.append(document)
        else:
            query_documents.append(document)
    return query_documents, original_documents


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


def save_rankings(output_dir: Path, query_documents: list[viz.DocumentSample], original_documents: list[viz.DocumentSample], query_embeddings: np.ndarray, original_embeddings: np.ndarray) -> dict[str, Any]:
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
    pdf_paths = discover_pdfs(args.pdf_root.resolve())
    pages = extract_pdfs_to_images(pdf_paths, args.image_root.resolve(), args.render_scale)
    build_ocr_cache(
        pages=pages,
        ocr_cache_root=args.ocr_cache_root.resolve(),
        tesseract_cmd=args.tesseract_cmd.resolve(),
        psm=args.psm,
        oem=args.oem,
        min_confidence=args.min_confidence,
    )
    query_documents, original_documents = build_documents(pages, args.ocr_cache_root.resolve())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, processor, training_config = bench.load_projection_model(args.checkpoint_dir.resolve(), device)
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

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        output_dir / "benchmark_summary.json",
        save_rankings(output_dir, query_documents, original_documents, query_embeddings, original_embeddings)
        | {
            "checkpoint_dir": str(args.checkpoint_dir.resolve()),
            "device": str(device),
            "pdf_count": len(pdf_paths),
            "template_names": [document.stem for document in original_documents],
        },
    )
    write_npy(output_dir / "original_embeddings.npy", original_embeddings)
    write_npy(output_dir / "query_embeddings.npy", query_embeddings)
    write_rows(output_dir / "original_embedding_rows.json", original_documents)
    write_rows(output_dir / "query_embedding_rows.json", query_documents)
    print(f"Saved benchmark outputs to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
