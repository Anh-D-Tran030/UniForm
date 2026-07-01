from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any

from langdetect import DetectorFactory, LangDetectException, detect_langs
from rapidocr_onnxruntime import RapidOCR


DetectorFactory.seed = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "OCR extracted CommonForms images, detect the dominant language, and "
            "copy English images plus their metadata JSON into a separate folder."
        )
    )
    parser.add_argument(
        "--source-dir",
        default=str(Path("A:/RealForm/processed/CommonFormsFillable")),
        help="Directory created by the fillable-area extractor.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path("A:/RealForm/processed/CommonFormsEnglish")),
        help="Directory where English-only images and metadata will be copied.",
    )
    parser.add_argument(
        "--poll-interval-seconds",
        type=int,
        default=20,
        help="Seconds to wait between scans when no new metadata is ready.",
    )
    parser.add_argument(
        "--idle-rounds-before-exit",
        type=int,
        default=3,
        help="Number of idle polls before exit.",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.60,
        help="Minimum confidence required for the dominant language to count.",
    )
    parser.add_argument(
        "--min-text-length",
        type=int,
        default=40,
        help="Minimum OCR text length before language detection is trusted.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Optional limit for testing. Zero means no limit.",
    )
    return parser.parse_args()


def load_json(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def metadata_files(source_dir: Path) -> list[Path]:
    return sorted((source_dir / "metadata").glob("*/*.json"))


def detect_language(text: str) -> tuple[str | None, float]:
    try:
        langs = detect_langs(text)
    except LangDetectException:
        return None, 0.0
    if not langs:
        return None, 0.0
    top = langs[0]
    return top.lang, float(top.prob)


def relative_split_path(metadata_path: Path, metadata_root: Path) -> tuple[str, str]:
    relative = metadata_path.relative_to(metadata_root)
    split = relative.parts[0]
    file_name = relative.name
    return split, file_name


def normalize_ocr_text(result: Any) -> str:
    if not result:
        return ""
    lines: list[str] = []
    for item in result:
        try:
            text = item[1]
        except Exception:
            continue
        if text:
            lines.append(str(text))
    return "\n".join(lines).strip()


def process_metadata(
    metadata_path: Path,
    source_dir: Path,
    output_dir: Path,
    engine: RapidOCR,
    state: dict[str, Any],
    manifest_path: Path,
    min_confidence: float,
    min_text_length: int,
) -> bool:
    metadata_root = source_dir / "metadata"
    image_root = source_dir / "images"
    split, file_name = relative_split_path(metadata_path, metadata_root)
    image_name = f"{metadata_path.stem}.jpeg"

    candidate_paths = list((image_root / split).glob(f"{metadata_path.stem}.*"))
    if not candidate_paths:
        raise FileNotFoundError(f"Missing image for metadata: {metadata_path}")
    image_path = candidate_paths[0]

    result, _ = engine(str(image_path))
    ocr_text = normalize_ocr_text(result)
    text_length = len(ocr_text)
    language, confidence = detect_language(ocr_text) if text_length >= min_text_length else (None, 0.0)
    is_english = language == "en" and confidence >= min_confidence

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["ocr_language"] = language
    metadata["ocr_language_confidence"] = confidence
    metadata["ocr_text_length"] = text_length

    processed_items: dict[str, Any] = state.setdefault("processed_items", {})
    summary: dict[str, Any] = state.setdefault(
        "summary",
        {
            "processed_files": 0,
            "english_files": 0,
        },
    )

    record = {
        "metadata_path": str(metadata_path),
        "image_path": str(image_path),
        "output_metadata_path": None,
        "output_image_path": None,
        "language": language,
        "confidence": confidence,
        "is_english": is_english,
        "ocr_text_length": text_length,
        "ocr_preview": ocr_text[:500],
        "fillable_count": metadata.get("fillable_count"),
        "split": split,
        "file_name": metadata.get("file_name", file_name),
    }

    if is_english:
        output_image_dir = output_dir / "images" / split
        output_metadata_dir = output_dir / "metadata" / split
        output_image_dir.mkdir(parents=True, exist_ok=True)
        output_metadata_dir.mkdir(parents=True, exist_ok=True)

        output_image_path = output_image_dir / image_path.name
        output_metadata_path = output_metadata_dir / metadata_path.name

        shutil.copy2(image_path, output_image_path)
        output_metadata_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )

        record["output_image_path"] = str(output_image_path)
        record["output_metadata_path"] = str(output_metadata_path)
        summary["english_files"] += 1

    processed_items[str(metadata_path)] = record
    summary["processed_files"] += 1

    with manifest_path.open("a", encoding="utf-8") as manifest_file:
        manifest_file.write(json.dumps(record, ensure_ascii=True) + "\n")

    return is_english


def main() -> None:
    args = parse_args()
    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    state_path = output_dir / "language_filter_state.json"
    manifest_path = output_dir / "ocr_manifest.jsonl"

    output_dir.mkdir(parents=True, exist_ok=True)
    state = load_json(
        state_path,
        default={
            "processed_items": {},
            "summary": {
                "processed_files": 0,
                "english_files": 0,
            },
        },
    )

    print(f"Watching extracted metadata in: {source_dir}", flush=True)
    print(f"Copying English results to: {output_dir}", flush=True)
    print(
        f"English rule: dominant lang='en' and confidence >= {args.min_confidence}",
        flush=True,
    )

    engine = RapidOCR()
    idle_round = 0
    processed_now = 0

    while True:
        all_metadata = metadata_files(source_dir)
        pending = [path for path in all_metadata if str(path) not in state.get("processed_items", {})]

        if args.max_files > 0:
            remaining = args.max_files - processed_now
            if remaining <= 0:
                print("Reached max-files limit. Exiting.", flush=True)
                break
            pending = pending[:remaining]

        if pending:
            idle_round = 0
            for metadata_path in pending:
                try:
                    is_english = process_metadata(
                        metadata_path=metadata_path,
                        source_dir=source_dir,
                        output_dir=output_dir,
                        engine=engine,
                        state=state,
                        manifest_path=manifest_path,
                        min_confidence=args.min_confidence,
                        min_text_length=args.min_text_length,
                    )
                    processed_now += 1
                    save_json(state_path, state)
                    print(
                        f"processed {metadata_path.name} english={is_english}",
                        flush=True,
                    )
                except Exception as exc:
                    print(f"skip {metadata_path.name} reason={exc}", flush=True)
        else:
            idle_round += 1
            save_json(state_path, state)
            print(
                f"idle_wait {idle_round}/{args.idle_rounds_before_exit} "
                f"sleeping {args.poll_interval_seconds}s",
                flush=True,
            )
            if idle_round >= args.idle_rounds_before_exit:
                break
            time.sleep(args.poll_interval_seconds)

    save_json(state_path, state)
    print(
        "done "
        f"processed_files={state['summary']['processed_files']} "
        f"english_files={state['summary']['english_files']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
