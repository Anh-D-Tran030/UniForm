from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy a fixed number of filled-template folders, keeping only fill images and excluding fill JSON files."
    )
    parser.add_argument(
        "--source-dir",
        required=True,
        help="Root folder containing synthetic filled template folders.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Destination folder for the selected filled-template folders.",
    )
    parser.add_argument(
        "--count",
        type=int,
        required=True,
        help="Number of filled-template folders to extract.",
    )
    return parser.parse_args()


def template_dirs_from_fill_images(source_dir: Path) -> list[Path]:
    return sorted({path.parent for path in source_dir.rglob("fill_*.png")})


def main() -> None:
    args = parse_args()
    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)

    template_dirs = template_dirs_from_fill_images(source_dir)
    if len(template_dirs) < args.count:
        raise RuntimeError(
            f"Requested {args.count} filled-template folders, but only found {len(template_dirs)} in {source_dir}"
        )

    selected = template_dirs[: args.count]
    copied = 0

    for template_dir in selected:
        relative_dir = template_dir.relative_to(source_dir)
        target_dir = output_dir / relative_dir
        target_dir.mkdir(parents=True, exist_ok=True)

        for image_path in sorted(template_dir.glob("fill_*.png")):
            shutil.copy2(image_path, target_dir / image_path.name)

        copied += 1

    print(f"done copied_template_dirs={copied} output_dir={output_dir}", flush=True)


if __name__ == "__main__":
    main()
