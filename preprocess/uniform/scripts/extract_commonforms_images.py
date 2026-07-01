from __future__ import annotations

import argparse
import json
import time
from io import BytesIO
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract CommonForms images that contain at least N fillable areas "
            "with area greater than the configured threshold."
        )
    )
    parser.add_argument(
        "--input-dir",
        default=str(Path("A:/RealForm/data/CommonForms/data")),
        help="Directory containing CommonForms parquet shards.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path("A:/RealForm/processed/CommonFormsFillable")),
        help="Directory where extracted images and metadata will be written.",
    )
    parser.add_argument(
        "--min-fillable-count",
        type=int,
        default=3,
        help="Minimum number of fillable regions required to keep an image.",
    )
    parser.add_argument(
        "--min-area",
        type=float,
        default=3.0,
        help="Minimum object area for a region to count as fillable.",
    )
    parser.add_argument(
        "--poll-interval-seconds",
        type=int,
        default=20,
        help="Seconds to wait between directory scans when no new stable shards are ready.",
    )
    parser.add_argument(
        "--idle-rounds-before-exit",
        type=int,
        default=3,
        help="Number of idle polls before the watcher exits.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Parquet batch size.",
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


def get_split_name(parquet_path: Path) -> str:
    return parquet_path.name.split("-")[0]


def is_readable_parquet(parquet_path: Path) -> bool:
    try:
        pq.ParquetFile(parquet_path)
        return True
    except Exception:
        return False


def stable_parquet_files(input_dir: Path, known_sizes: dict[str, int]) -> tuple[list[Path], dict[str, int]]:
    ready: list[Path] = []
    new_sizes: dict[str, int] = {}

    for parquet_path in sorted(input_dir.glob("*.parquet")):
        size = parquet_path.stat().st_size
        key = str(parquet_path)
        new_sizes[key] = size
        if known_sizes.get(key) == size or is_readable_parquet(parquet_path):
            ready.append(parquet_path)

    return ready, new_sizes


def qualifying_objects(objects: dict[str, Any], min_area: float) -> list[dict[str, Any]]:
    object_ids = list(objects.get("id") or [])
    areas = list(objects.get("area") or [])
    limit = min(len(object_ids), len(areas))

    valid: list[dict[str, Any]] = []
    for index in range(limit):
        object_id = object_ids[index]
        area = areas[index]
        if object_id is None or area is None:
            continue
        if float(area) > min_area:
            valid.append(
                {
                    "object_id": int(object_id),
                    "area": float(area),
                }
            )

    return valid


def image_extension(file_name: str) -> str:
    suffix = Path(file_name).suffix.lower()
    return suffix if suffix else ".png"


def write_image(image_bytes: bytes, target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(BytesIO(image_bytes)) as img:
        img.save(target_path)


def safe_stem(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value)


def process_parquet(
    parquet_path: Path,
    output_dir: Path,
    state: dict[str, Any],
    manifest_path: Path,
    min_fillable_count: int,
    min_area: float,
    batch_size: int,
) -> int:
    split_name = get_split_name(parquet_path)
    images_dir = output_dir / "images" / split_name
    metadata_dir = output_dir / "metadata" / split_name
    images_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    parquet_key = str(parquet_path)
    processed_rows: dict[str, dict[str, Any]] = state.setdefault("processed_rows", {})
    parquet_stats: dict[str, Any] = state.setdefault("parquet_stats", {})
    summary: dict[str, Any] = state.setdefault(
        "summary",
        {
            "saved_images": 0,
            "processed_rows": 0,
            "processed_parquet_files": 0,
        },
    )

    saved_from_file = 0
    parquet_file = pq.ParquetFile(parquet_path)

    with manifest_path.open("a", encoding="utf-8") as manifest_file:
        for batch in parquet_file.iter_batches(batch_size=batch_size):
            for row in batch.to_pylist():
                row_key = f"{parquet_key}::{row['id']}"
                if row_key in processed_rows:
                    continue

                summary["processed_rows"] += 1
                valid_objects = qualifying_objects(row["objects"], min_area=min_area)
                fillable_count = len(valid_objects)

                record: dict[str, Any] = {
                    "row_id": int(row["id"]),
                    "image_id": int(row["image_id"]),
                    "split": split_name,
                    "source_parquet": parquet_path.name,
                    "file_name": row["file_name"],
                    "width": int(row["width"]),
                    "height": int(row["height"]),
                    "object_count": len(list(row["objects"].get("id") or [])),
                    "fillable_count": fillable_count,
                    "min_area_threshold": min_area,
                    "qualifying_objects": valid_objects,
                }

                if fillable_count >= min_fillable_count:
                    original_stem = Path(row["file_name"]).stem
                    base_name = safe_stem(f"{split_name}__{row['image_id']}__{original_stem}")
                    image_path = images_dir / f"{base_name}{image_extension(row['file_name'])}"
                    metadata_path = metadata_dir / f"{base_name}.json"

                    write_image(row["image"]["bytes"], image_path)
                    record["image_path"] = str(image_path)
                    record["metadata_path"] = str(metadata_path)

                    metadata_path.write_text(
                        json.dumps(record, indent=2, ensure_ascii=True),
                        encoding="utf-8",
                    )
                    manifest_file.write(json.dumps(record, ensure_ascii=True) + "\n")

                    summary["saved_images"] += 1
                    saved_from_file += 1

                processed_rows[row_key] = {
                    "fillable_count": fillable_count,
                    "qualified": fillable_count >= min_fillable_count,
                }

    summary["processed_parquet_files"] = len({key.split("::", 1)[0] for key in processed_rows})
    parquet_stats[parquet_key] = {
        "saved_images": saved_from_file,
        "last_processed_epoch": int(time.time()),
    }
    return saved_from_file


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    state_path = output_dir / "extract_state.json"
    manifest_path = output_dir / "manifest.jsonl"

    output_dir.mkdir(parents=True, exist_ok=True)
    state = load_json(
        state_path,
        default={
            "processed_rows": {},
            "parquet_stats": {},
            "summary": {
                "saved_images": 0,
                "processed_rows": 0,
                "processed_parquet_files": 0,
            },
        },
    )

    print(f"Watching parquet dir: {input_dir}", flush=True)
    print(f"Writing extracted images to: {output_dir}", flush=True)
    print(
        f"Keeping images with >= {args.min_fillable_count} fillable regions "
        f"and area > {args.min_area}",
        flush=True,
    )

    known_sizes: dict[str, int] = {}
    idle_round = 0
    processed_files = 0

    while True:
        ready_files, known_sizes = stable_parquet_files(input_dir, known_sizes)
        unprocessed_ready = [
            parquet_path
            for parquet_path in ready_files
            if str(parquet_path) not in state.get("parquet_stats", {})
        ]

        if args.max_files > 0:
            remaining = args.max_files - processed_files
            if remaining <= 0:
                print("Reached max-files limit. Exiting.", flush=True)
                break
            unprocessed_ready = unprocessed_ready[:remaining]

        if unprocessed_ready:
            idle_round = 0
            for parquet_path in unprocessed_ready:
                try:
                    saved_count = process_parquet(
                        parquet_path=parquet_path,
                        output_dir=output_dir,
                        state=state,
                        manifest_path=manifest_path,
                        min_fillable_count=args.min_fillable_count,
                        min_area=args.min_area,
                        batch_size=args.batch_size,
                    )
                    processed_files += 1
                    save_json(state_path, state)
                    print(
                        f"processed {parquet_path.name} saved_images={saved_count}",
                        flush=True,
                    )
                except Exception as exc:  # pragma: no cover - defensive retry path
                    print(f"skip {parquet_path.name} reason={exc}", flush=True)
                    state.get("parquet_stats", {}).pop(str(parquet_path), None)
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
        f"saved_images={state['summary']['saved_images']} "
        f"processed_rows={state['summary']['processed_rows']} "
        f"processed_parquet_files={state['summary']['processed_parquet_files']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
