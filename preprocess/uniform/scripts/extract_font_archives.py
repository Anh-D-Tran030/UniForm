from __future__ import annotations

import json
import re
import shutil
import zipfile
from pathlib import Path


FONTS_DIR = Path(r"A:\RealForm\Fonts")
EXTRACTED_DIR = FONTS_DIR / "extracted"
USABLE_DIR = FONTS_DIR / "usable_fonts"
MANIFEST_PATH = FONTS_DIR / "font_manifest.json"


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")


def unique_target_path(directory: Path, file_name: str) -> Path:
    candidate = directory / file_name
    if not candidate.exists():
        return candidate

    stem = Path(file_name).stem
    suffix = Path(file_name).suffix
    index = 2
    while True:
        candidate = directory / f"{stem}_{index}{suffix}"
        if not candidate.exists():
            return candidate
        index += 1


def main() -> None:
    EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)
    USABLE_DIR.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, object]] = []

    for zip_path in sorted(FONTS_DIR.glob("*.zip")):
        archive_name = zip_path.stem
        archive_dir = EXTRACTED_DIR / safe_name(archive_name)
        archive_dir.mkdir(parents=True, exist_ok=True)

        extracted_files: list[str] = []
        usable_files: list[str] = []

        with zipfile.ZipFile(zip_path) as zf:
            for member in zf.infolist():
                if member.is_dir():
                    continue

                member_name = Path(member.filename).name
                if not member_name:
                    continue

                target_name = safe_name(member_name)
                extracted_target = unique_target_path(archive_dir, target_name)

                with zf.open(member) as src, extracted_target.open("wb") as dst:
                    shutil.copyfileobj(src, dst)

                extracted_files.append(str(extracted_target))

                if extracted_target.suffix.lower() in {".ttf", ".otf"}:
                    usable_name = safe_name(f"{safe_name(archive_name)}__{member_name}")
                    usable_target = unique_target_path(USABLE_DIR, usable_name)
                    shutil.copy2(extracted_target, usable_target)
                    usable_files.append(str(usable_target))

        records.append(
            {
                "archive": str(zip_path),
                "archive_name": archive_name,
                "extract_dir": str(archive_dir),
                "font_files": usable_files,
                "all_extracted_files": extracted_files,
            }
        )
        print(f"extracted {zip_path.name} font_files={len(usable_files)}", flush=True)

    MANIFEST_PATH.write_text(
        json.dumps(
            {
                "fonts_dir": str(FONTS_DIR),
                "usable_fonts_dir": str(USABLE_DIR),
                "archive_count": len(records),
                "archives": records,
            },
            indent=2,
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )
    print(f"done archive_count={len(records)} manifest={MANIFEST_PATH}", flush=True)


if __name__ == "__main__":
    main()
