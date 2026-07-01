from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from PIL import Image
from reportlab.pdfgen import canvas


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy 600 English CommonForms images into a separate folder and "
            "create 6 PDFs with 100 one-form pages each."
        )
    )
    parser.add_argument(
        "--source-dir",
        default=str(Path("A:/RealForm/processed/CommonFormsEnglish")),
        help="English-only source folder.",
    )
    parser.add_argument(
        "--selected-dir",
        default=str(Path("A:/RealForm/processed/CommonFormsEnglishSelected600")),
        help="Destination folder for the selected 600 image/json pairs.",
    )
    parser.add_argument(
        "--pdf-dir",
        default=str(Path("A:/RealForm/processed/CommonFormsEnglishPDFs")),
        help="Destination folder for the PDF bundles.",
    )
    parser.add_argument(
        "--selection-count",
        type=int,
        default=600,
        help="Number of English forms to select.",
    )
    parser.add_argument(
        "--images-per-pdf",
        type=int,
        default=100,
        help="Number of forms per PDF.",
    )
    return parser.parse_args()


def image_files(source_dir: Path) -> list[Path]:
    return sorted((source_dir / "images").glob("*/*"))


def metadata_path_for(image_path: Path, source_dir: Path) -> Path:
    split = image_path.parent.name
    return source_dir / "metadata" / split / f"{image_path.stem}.json"


def copy_selection(selected_images: list[Path], source_dir: Path, selected_dir: Path) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []

    for image_path in selected_images:
        split = image_path.parent.name
        metadata_path = metadata_path_for(image_path, source_dir)
        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing metadata for {image_path}")

        target_image_dir = selected_dir / "images" / split
        target_metadata_dir = selected_dir / "metadata" / split
        target_image_dir.mkdir(parents=True, exist_ok=True)
        target_metadata_dir.mkdir(parents=True, exist_ok=True)

        target_image_path = target_image_dir / image_path.name
        target_metadata_path = target_metadata_dir / metadata_path.name

        shutil.copy2(image_path, target_image_path)
        shutil.copy2(metadata_path, target_metadata_path)

        records.append(
            {
                "image_path": str(target_image_path),
                "metadata_path": str(target_metadata_path),
                "split": split,
                "file_name": image_path.name,
            }
        )

    return records


def create_pdf(pdf_path: Path, image_paths: list[Path]) -> None:
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pdf = None

    for index, image_path in enumerate(image_paths):
        with Image.open(image_path) as img:
            width, height = img.size

        if pdf is None:
            pdf = canvas.Canvas(str(pdf_path), pagesize=(width, height))
        else:
            pdf.setPageSize((width, height))

        pdf.drawImage(str(image_path), 0, 0, width=width, height=height, preserveAspectRatio=True, anchor="c")
        pdf.showPage()

    if pdf is None:
        raise ValueError(f"No images supplied for {pdf_path}")
    pdf.save()


def write_manifest(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def main() -> None:
    args = parse_args()
    source_dir = Path(args.source_dir)
    selected_dir = Path(args.selected_dir)
    pdf_dir = Path(args.pdf_dir)

    all_images = image_files(source_dir)
    if len(all_images) < args.selection_count:
        raise RuntimeError(
            f"Needed {args.selection_count} English images but only found {len(all_images)} in {source_dir}"
        )

    selected_images = all_images[: args.selection_count]
    records = copy_selection(selected_images, source_dir, selected_dir)

    pdf_count = args.selection_count // args.images_per_pdf
    if pdf_count * args.images_per_pdf != args.selection_count:
        raise RuntimeError("selection-count must be divisible by images-per-pdf")

    pdf_records: list[dict[str, object]] = []
    for pdf_index in range(pdf_count):
        start = pdf_index * args.images_per_pdf
        end = start + args.images_per_pdf
        pdf_images = [Path(record["image_path"]) for record in records[start:end]]
        pdf_path = pdf_dir / f"english_forms_{pdf_index + 1:02d}.pdf"
        create_pdf(pdf_path, pdf_images)
        pdf_records.append(
            {
                "pdf_path": str(pdf_path),
                "page_count": len(pdf_images),
                "first_image": str(pdf_images[0]),
                "last_image": str(pdf_images[-1]),
            }
        )
        print(f"created {pdf_path.name} pages={len(pdf_images)}", flush=True)

    write_manifest(
        selected_dir / "selection_manifest.json",
        {
            "selection_count": len(records),
            "records": records,
        },
    )
    write_manifest(
        pdf_dir / "pdf_manifest.json",
        {
            "pdf_count": len(pdf_records),
            "images_per_pdf": args.images_per_pdf,
            "pdfs": pdf_records,
        },
    )

    print(
        f"done selection_count={len(records)} pdf_count={len(pdf_records)} images_per_pdf={args.images_per_pdf}",
        flush=True,
    )


if __name__ == "__main__":
    main()
