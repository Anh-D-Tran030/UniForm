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
            "Select English CommonForms templates that were not used in the 10k synthetic-fill set "
            "and package them into 100-page PDFs."
        )
    )
    parser.add_argument(
        "--source-dir",
        default=str(Path("A:/RealForm/processed/CommonFormsEnglish")),
        help="English-only source folder.",
    )
    parser.add_argument(
        "--exclude-dir",
        default=str(Path("A:/RealForm/processed/synthetic_fill_images_10k_folders")),
        help="Folder containing the 10k synthetic template directories to exclude.",
    )
    parser.add_argument(
        "--selected-dir",
        default=str(Path("A:/RealForm/processed/CommonFormsEnglishBeyond10kSelected300")),
        help="Destination folder for the selected images.",
    )
    parser.add_argument(
        "--pdf-dir",
        default=str(Path("A:/RealForm/processed/CommonFormsEnglishBeyond10kPDFs")),
        help="Destination folder for the PDF bundles.",
    )
    parser.add_argument(
        "--selection-count",
        type=int,
        default=300,
        help="Number of English forms to select.",
    )
    parser.add_argument(
        "--images-per-pdf",
        type=int,
        default=100,
        help="Number of forms per PDF.",
    )
    parser.add_argument(
        "--skip-first-n",
        type=int,
        default=0,
        help="Skip the first N eligible templates after exclusions.",
    )
    parser.add_argument(
        "--exclude-selection-manifests",
        nargs="*",
        default=[],
        help="Optional prior selection manifests whose template names should also be excluded.",
    )
    return parser.parse_args()


def image_files(source_dir: Path) -> list[Path]:
    return sorted(
        [path for path in (source_dir / "images").glob("*/*") if path.is_file()],
        key=lambda path: str(path).lower(),
    )


def excluded_template_names(exclude_dir: Path) -> set[str]:
    return {path.name for path in exclude_dir.iterdir() if path.is_dir()}


def excluded_from_manifests(manifest_paths: list[str]) -> set[str]:
    excluded: set[str] = set()
    for manifest_arg in manifest_paths:
        manifest_path = Path(manifest_arg)
        if not manifest_path.exists():
            continue
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        for record in payload.get("records", []):
            template_name = record.get("template_name")
            if template_name:
                excluded.add(str(template_name))
    return excluded


def write_manifest(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def create_pdf(pdf_path: Path, image_paths: list[Path]) -> None:
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pdf = None

    for image_path in image_paths:
        with Image.open(image_path) as image:
            width, height = image.size

        if pdf is None:
            pdf = canvas.Canvas(str(pdf_path), pagesize=(width, height))
        else:
            pdf.setPageSize((width, height))

        pdf.drawImage(str(image_path), 0, 0, width=width, height=height, preserveAspectRatio=True, anchor="c")
        pdf.showPage()

    if pdf is None:
        raise ValueError(f"No images supplied for {pdf_path}")
    pdf.save()


def main() -> None:
    args = parse_args()
    source_dir = Path(args.source_dir)
    exclude_dir = Path(args.exclude_dir)
    selected_dir = Path(args.selected_dir)
    pdf_dir = Path(args.pdf_dir)

    if selected_dir.exists():
        shutil.rmtree(selected_dir)
    if pdf_dir.exists():
        shutil.rmtree(pdf_dir)

    all_images = image_files(source_dir)
    excluded = excluded_template_names(exclude_dir)
    excluded.update(excluded_from_manifests(args.exclude_selection_manifests))
    eligible_images = [path for path in all_images if path.stem not in excluded]
    eligible_images = eligible_images[args.skip_first_n :]

    if len(eligible_images) < args.selection_count:
        raise RuntimeError(
            f"Needed {args.selection_count} English images outside the 10k set, "
            f"but only found {len(eligible_images)} eligible images."
        )

    if args.selection_count % args.images_per_pdf != 0:
        raise RuntimeError("selection-count must be divisible by images-per-pdf")

    selected_images = eligible_images[: args.selection_count]
    selected_records: list[dict[str, str]] = []

    for image_path in selected_images:
        split = image_path.parent.name
        target_dir = selected_dir / "images" / split
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / image_path.name
        shutil.copy2(image_path, target_path)
        selected_records.append(
            {
                "split": split,
                "file_name": image_path.name,
                "template_name": image_path.stem,
                "source_image_path": str(image_path),
                "selected_image_path": str(target_path),
            }
        )

    pdf_count = args.selection_count // args.images_per_pdf
    pdf_records: list[dict[str, object]] = []
    for pdf_index in range(pdf_count):
        start = pdf_index * args.images_per_pdf
        end = start + args.images_per_pdf
        pdf_images = [Path(record["selected_image_path"]) for record in selected_records[start:end]]
        pdf_path = pdf_dir / f"english_forms_beyond_10k_{pdf_index + 1:02d}.pdf"
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
            "selection_count": len(selected_records),
            "eligible_count": len(eligible_images),
            "excluded_template_count": len(excluded),
            "skip_first_n": args.skip_first_n,
            "records": selected_records,
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
        f"done eligible_count={len(eligible_images)} selection_count={len(selected_records)} "
        f"pdf_count={len(pdf_records)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
