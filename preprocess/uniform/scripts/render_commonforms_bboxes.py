from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select English CommonForms images and render qualifying object bounding boxes "
            "onto annotated copies in a separate folder."
        )
    )
    parser.add_argument(
        "--source-dir",
        default=str(Path("A:/RealForm/processed/CommonFormsEnglish")),
        help="English-form source directory.",
    )
    parser.add_argument(
        "--dataset-dir",
        default=str(Path("A:/RealForm/data/CommonForms/data")),
        help="Directory containing the source parquet shards.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path("A:/RealForm/processed/CommonFormsEnglishBBox100")),
        help="Directory for annotated output.",
    )
    parser.add_argument(
        "--selection-count",
        type=int,
        default=100,
        help="Number of English forms to annotate.",
    )
    return parser.parse_args()


def metadata_files(source_dir: Path) -> list[Path]:
    return sorted((source_dir / "metadata").glob("*/*.json"))


def image_files(source_dir: Path) -> list[Path]:
    return sorted(path for path in (source_dir / "images").glob("*/*") if path.is_file())


def load_parquet_row(parquet_path: Path, row_id: int) -> dict:
    table = pq.read_table(parquet_path, columns=["id", "objects"])
    for row in table.to_pylist():
        if int(row["id"]) == row_id:
            return row
    raise KeyError(f"row_id={row_id} not found in {parquet_path}")


def object_bbox_map(objects: dict) -> dict[int, dict]:
    object_ids = list(objects.get("id") or [])
    boxes = list(objects.get("bbox") or [])
    areas = list(objects.get("area") or [])
    result: dict[int, dict] = {}
    for index, object_id in enumerate(object_ids):
        bbox = boxes[index] if index < len(boxes) else None
        area = areas[index] if index < len(areas) else None
        if bbox is None:
            continue
        result[int(object_id)] = {
            "bbox": [float(v) for v in bbox],
            "area": float(area) if area is not None else None,
        }
    return result


def draw_boxes(image_path: Path, target_path: Path, boxes: list[dict]) -> None:
    with Image.open(image_path) as img:
        draw = ImageDraw.Draw(img)
        font = ImageFont.load_default()

        for item in boxes:
            x, y, w, h = item["bbox"]
            x2 = x + w
            y2 = y + h
            draw.rectangle((x, y, x2, y2), outline=(255, 0, 0), width=4)
            label = str(item["object_id"])
            text_box = draw.textbbox((0, 0), label, font=font)
            text_w = text_box[2] - text_box[0]
            text_h = text_box[3] - text_box[1]
            text_x = x
            text_y = max(0, y - text_h - 6)
            draw.rectangle((text_x, text_y, text_x + text_w + 6, text_y + text_h + 4), fill=(255, 0, 0))
            draw.text((text_x + 3, text_y + 2), label, fill=(255, 255, 255), font=font)

        target_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(target_path)


def main() -> None:
    args = parse_args()
    source_dir = Path(args.source_dir)
    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)

    all_images = image_files(source_dir)
    if len(all_images) < args.selection_count:
        raise RuntimeError(
            f"Needed {args.selection_count} English forms but only found {len(all_images)} in {source_dir}"
        )

    records: list[dict] = []

    for image_path in all_images:
        if len(records) >= args.selection_count:
            break

        split = image_path.parent.name
        metadata_path = source_dir / "metadata" / split / f"{image_path.stem}.json"
        if not metadata_path.exists():
            continue

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        parquet_path = dataset_dir / metadata["source_parquet"]
        row = load_parquet_row(parquet_path, int(metadata["row_id"]))
        bbox_lookup = object_bbox_map(row["objects"])

        qualifying_boxes: list[dict] = []
        for item in metadata.get("qualifying_objects", []):
            object_id = int(item["object_id"])
            if object_id not in bbox_lookup:
                continue
            qualifying_boxes.append(
                {
                    "object_id": object_id,
                    "area": float(item["area"]),
                    "bbox": bbox_lookup[object_id]["bbox"],
                }
            )

        annotated_image_dir = output_dir / "annotated_images" / split
        copied_metadata_dir = output_dir / "metadata" / split
        original_image_dir = output_dir / "original_images" / split
        annotated_image_dir.mkdir(parents=True, exist_ok=True)
        copied_metadata_dir.mkdir(parents=True, exist_ok=True)
        original_image_dir.mkdir(parents=True, exist_ok=True)

        annotated_image_path = annotated_image_dir / image_path.name
        copied_image_path = original_image_dir / image_path.name
        copied_metadata_path = copied_metadata_dir / metadata_path.name

        shutil.copy2(image_path, copied_image_path)
        shutil.copy2(metadata_path, copied_metadata_path)
        draw_boxes(image_path, annotated_image_path, qualifying_boxes)

        record = {
            "row_id": metadata["row_id"],
            "image_id": metadata["image_id"],
            "file_name": metadata["file_name"],
            "split": split,
            "source_parquet": metadata["source_parquet"],
            "annotated_image_path": str(annotated_image_path),
            "original_image_path": str(copied_image_path),
            "metadata_path": str(copied_metadata_path),
            "bbox_count": len(qualifying_boxes),
            "boxes": qualifying_boxes,
        }
        records.append(record)
        print(f"annotated {image_path.name} bbox_count={len(qualifying_boxes)}", flush=True)

    if len(records) < args.selection_count:
        raise RuntimeError(
            f"Annotated only {len(records)} files; needed {args.selection_count}. "
            "There may be missing metadata or source parquet rows."
        )

    manifest_path = output_dir / "bbox_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "selection_count": len(records),
                "records": records,
            },
            indent=2,
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )
    print(f"done selection_count={len(records)} manifest={manifest_path}", flush=True)


if __name__ == "__main__":
    main()
