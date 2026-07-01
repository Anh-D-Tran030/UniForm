from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pypdfium2 as pdfium


DEFAULT_PDF_ROOT = Path(r"A:\RealForm\ExtraPdf")
DEFAULT_OUTPUT_ROOT = Path(r"A:\RealForm\processed\ExtraPdf_first_pages")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract the first page from each PDF in ExtraPdf.")
    parser.add_argument("--pdf-root", type=Path, default=DEFAULT_PDF_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--render-scale", type=float, default=2.5)
    parser.add_argument("--clear-output", action="store_true")
    return parser.parse_args()


def slugify(text: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", text.strip())
    normalized = normalized.strip("_")
    return normalized or "template"


def write_json(path: Path, payload: dict[str, Any] | list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def discover_pdfs(pdf_root: Path) -> list[Path]:
    if not pdf_root.exists():
        raise FileNotFoundError(f"PDF root not found: {pdf_root}")
    pdfs = sorted([path for path in pdf_root.iterdir() if path.is_file() and path.suffix.lower() == ".pdf"], key=lambda path: path.name.lower())
    if not pdfs:
        raise FileNotFoundError(f"No PDF files found under {pdf_root}")
    return pdfs


def extract_first_pages(pdf_paths: list[Path], output_root: Path, render_scale: float) -> list[dict[str, Any]]:
    output_root.mkdir(parents=True, exist_ok=True)
    seen_names: dict[str, int] = {}
    manifest: list[dict[str, Any]] = []

    for index, pdf_path in enumerate(pdf_paths, start=1):
        base_name = slugify(pdf_path.stem)
        seen_names[base_name] = seen_names.get(base_name, 0) + 1
        template_name = base_name if seen_names[base_name] == 1 else f"{base_name}_{seen_names[base_name]}"
        template_dir = output_root / template_name
        template_dir.mkdir(parents=True, exist_ok=True)
        image_path = template_dir / "page_01_original.png"

        document = pdfium.PdfDocument(str(pdf_path))
        if len(document) == 0:
            document.close()
            raise RuntimeError(f"PDF has no pages: {pdf_path}")
        if not image_path.exists():
            page = document.get_page(0)
            bitmap = page.render(scale=render_scale)
            pil_image = bitmap.to_pil()
            pil_image.save(image_path)
            pil_image.close()
            page.close()
        document.close()

        manifest.append(
            {
                "template_name": template_name,
                "pdf_name": pdf_path.name,
                "image_path": str(image_path),
            }
        )
        if index == 1 or index % 10 == 0 or index == len(pdf_paths):
            print(f"extracted {index}/{len(pdf_paths)} first pages", flush=True)

    return manifest


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    if args.clear_output and output_root.exists():
        for child in output_root.iterdir():
            if child.is_dir():
                for nested in sorted(child.rglob("*"), reverse=True):
                    if nested.is_file():
                        nested.unlink()
                    elif nested.is_dir():
                        nested.rmdir()
                child.rmdir()
            else:
                child.unlink()

    pdf_paths = discover_pdfs(args.pdf_root.resolve())
    manifest = extract_first_pages(pdf_paths, output_root, args.render_scale)
    write_json(output_root / "extraction_manifest.json", manifest)
    print(f"done extracted_first_pages={len(manifest)} output_root={output_root}", flush=True)


if __name__ == "__main__":
    main()
