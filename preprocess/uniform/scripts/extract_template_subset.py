from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy a fixed-size subset of template images and matching JSON metadata into a new folder."
    )
    parser.add_argument(
        "--source-dir",
        required=True,
        help="Template source root containing images/ and metadata/ subfolders.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Destination root for the subset.",
    )
    parser.add_argument(
        "--count",
        type=int,
        required=True,
        help="Number of templates to copy.",
    )
    return parser.parse_args()


def image_files(source_dir: Path) -> list[Path]:
    return sorted(path for path in (source_dir / "images").glob("*/*") if path.is_file())


def metadata_path_for(image_path: Path, source_dir: Path) -> Path:
    split = image_path.parent.name
    return source_dir / "metadata" / split / f"{image_path.stem}.json"


def main() -> None:
    args = parse_args()
    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)

    images = image_files(source_dir)
    if len(images) < args.count:
        raise RuntimeError(f"Requested {args.count} templates, but only found {len(images)} under {source_dir}")

    selected = images[: args.count]
    records: list[dict[str, str]] = []

    for image_path in selected:
        metadata_path = metadata_path_for(image_path, source_dir)
        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing metadata for {image_path}")

        split = image_path.parent.name
        out_image_dir = output_dir / "images" / split
        out_meta_dir = output_dir / "metadata" / split
        out_image_dir.mkdir(parents=True, exist_ok=True)
        out_meta_dir.mkdir(parents=True, exist_ok=True)

        out_image_path = out_image_dir / image_path.name
        out_meta_path = out_meta_dir / metadata_path.name

        shutil.copy2(image_path, out_image_path)
        shutil.copy2(metadata_path, out_meta_path)

        records.append(
            {
                "image_path": str(out_image_path),
                "metadata_path": str(out_meta_path),
                "split": split,
                "file_name": image_path.name,
            }
        )

    manifest = {
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "selection_count": len(records),
        "records": records,
    }
    (output_dir / "subset_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    print(f"done selection_count={len(records)} output_dir={output_dir}", flush=True)


if __name__ == "__main__":
    main()
