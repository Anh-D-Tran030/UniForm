from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


COMMONFORMS_ENGLISH_ROOT = Path(r"A:\RealForm\processed\CommonFormsEnglish")
METADATA_ROOT = COMMONFORMS_ENGLISH_ROOT / "metadata"
IMAGES_ROOT = COMMONFORMS_ENGLISH_ROOT / "images"
REGIONS_ROOT = COMMONFORMS_ENGLISH_ROOT / "regions"
PARQUET_ROOT = Path(r"A:\RealForm\data\CommonForms\data")
SUMMARY_PATH = REGIONS_ROOT / "regions_summary.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def log(message: str) -> None:
    print(f"[{utc_now()}] {message}", flush=True)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def list_metadata_paths() -> list[Path]:
    return sorted(METADATA_ROOT.glob("*/*.json"), key=lambda path: str(path).lower())


def group_metadata_by_parquet(metadata_paths: list[Path]) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = defaultdict(list)
    for metadata_path in metadata_paths:
        payload = read_json(metadata_path)
        source_parquet = str(payload.get("source_parquet") or "")
        if source_parquet:
            groups[source_parquet].append(metadata_path)
    return dict(sorted(groups.items()))


def parquet_object_rows(parquet_name: str) -> dict[int, dict[str, Any]]:
    parquet_path = PARQUET_ROOT / parquet_name
    if not parquet_path.exists():
        raise FileNotFoundError(f"Missing source parquet: {parquet_path}")
    table = pq.read_table(parquet_path, columns=["id", "objects"])
    rows: dict[int, dict[str, Any]] = {}
    for row in table.to_pylist():
        rows[int(row["id"])] = row["objects"]
    return rows


def object_bbox_map(objects: dict[str, Any]) -> dict[int, list[float]]:
    object_ids = list(objects.get("id") or [])
    boxes = list(objects.get("bbox") or [])
    result: dict[int, list[float]] = {}
    for object_id, bbox in zip(object_ids, boxes):
        result[int(object_id)] = [float(value) for value in bbox]
    return result


def find_english_image(split: str, stem: str) -> Path | None:
    image_dir = IMAGES_ROOT / split
    matches = sorted(image_dir.glob(f"{stem}.*"), key=lambda path: path.name.lower())
    return matches[0] if matches else None


def build_region_payload(metadata_path: Path, row_objects: dict[int, dict[str, Any]]) -> tuple[dict[str, Any], int]:
    metadata = read_json(metadata_path)
    split = metadata_path.parent.name
    stem = metadata_path.stem
    row_id = int(metadata["row_id"])
    objects = row_objects.get(row_id)
    if objects is None:
        raise KeyError(f"row_id={row_id} not found for {metadata_path}")

    boxes_by_id = object_bbox_map(objects)
    regions: list[dict[str, Any]] = []
    missing = 0
    for item in metadata.get("qualifying_objects", []):
        object_id = int(item["object_id"])
        bbox = boxes_by_id.get(object_id)
        if bbox is None:
            missing += 1
            continue
        regions.append(
            {
                "object_id": object_id,
                "bbox": bbox,
                "bbox_format": "xywh",
                "area": float(item.get("area") or 0.0),
            }
        )

    image_path = find_english_image(split, stem)
    payload = {
        "template_name": stem,
        "split": split,
        "image_path": str(image_path) if image_path else None,
        "metadata_path": str(metadata_path),
        "source_parquet": metadata.get("source_parquet"),
        "row_id": row_id,
        "image_id": metadata.get("image_id"),
        "file_name": metadata.get("file_name"),
        "width": metadata.get("width"),
        "height": metadata.get("height"),
        "fillable_count": len(regions),
        "original_fillable_count": metadata.get("fillable_count"),
        "object_count": metadata.get("object_count"),
        "bbox_source": "source_parquet.objects.bbox",
        "bbox_format": "xywh",
        "regions": regions,
    }
    return payload, missing


def main() -> None:
    metadata_paths = list_metadata_paths()
    groups = group_metadata_by_parquet(metadata_paths)
    log(f"Found {len(metadata_paths)} metadata files across {len(groups)} parquet files.")

    total_seen = 0
    total_written = 0
    total_regions = 0
    total_missing = 0
    failures: list[dict[str, Any]] = []

    for parquet_index, (parquet_name, paths) in enumerate(groups.items(), start=1):
        log(f"Reading {parquet_name} for {len(paths)} metadata files ({parquet_index}/{len(groups)}).")
        row_objects = parquet_object_rows(parquet_name)
        for metadata_path in paths:
            total_seen += 1
            split = metadata_path.parent.name
            output_path = REGIONS_ROOT / split / f"{metadata_path.stem}.json"
            try:
                payload, missing = build_region_payload(metadata_path, row_objects)
            except Exception as exc:  # noqa: BLE001
                failures.append({"metadata_path": str(metadata_path), "error": repr(exc)})
                continue
            write_json(output_path, payload)
            total_written += 1
            total_regions += len(payload["regions"])
            total_missing += missing
            if total_seen == 1 or total_seen % 1000 == 0:
                log(
                    f"Progress {total_seen}/{len(metadata_paths)} metadata files. "
                    f"written={total_written} regions={total_regions} missing={total_missing}"
                )

    summary = {
        "metadata_root": str(METADATA_ROOT),
        "images_root": str(IMAGES_ROOT),
        "regions_root": str(REGIONS_ROOT),
        "parquet_root": str(PARQUET_ROOT),
        "metadata_seen": total_seen,
        "region_files_written": total_written,
        "regions_written": total_regions,
        "missing_bboxes": total_missing,
        "failure_count": len(failures),
        "failures_preview": failures[:50],
        "finished_at": utc_now(),
    }
    write_json(SUMMARY_PATH, summary)
    log(f"Finished writing regions: {summary}")


if __name__ == "__main__":
    main()
