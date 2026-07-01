from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from huggingface_hub import snapshot_download
from langdetect import DetectorFactory, LangDetectException, detect_langs
from PIL import Image
from rapidocr_onnxruntime import RapidOCR


DetectorFactory.seed = 0


REPO_ID = "jbarrow/CommonForms"
DATASET_ROOT = Path(r"A:\RealForm\data\CommonForms")
PARQUET_DIR = DATASET_ROOT / "data"
OUTPUT_ROOT = Path(r"A:\RealForm\processed\CommonFormsEnglish")

MIN_FILLABLE_COUNT = 3
MIN_AREA = 3.0
MIN_TEXT_LENGTH = 40
MIN_ENGLISH_CONFIDENCE = 0.60


def safe_name(text):
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in text)


def image_suffix(file_name):
    suffix = Path(file_name).suffix.lower()
    return suffix if suffix else ".png"


def detect_language(text):
    if len(text.strip()) < MIN_TEXT_LENGTH:
        return None, 0.0
    try:
        langs = detect_langs(text)
    except LangDetectException:
        return None, 0.0
    if not langs:
        return None, 0.0
    top = langs[0]
    return top.lang, float(top.prob)


def normalize_ocr_text(result):
    if not result:
        return ""
    lines: list[str] = []
    for item in result:
        if len(item) >= 2 and item[1]:
            lines.append(str(item[1]))
    return "\n".join(lines).strip()


def get_valid_fillable_objects(objects):
    object_ids = list(objects.get("id") or [])
    areas = list(objects.get("area") or [])
    bboxes = list(objects.get("bbox") or [])

    limit = min(len(object_ids), len(areas), len(bboxes))
    valid = []

    for index in range(limit):
        object_id = object_ids[index]
        area = areas[index]
        bbox = bboxes[index]

        if object_id is None or area is None or not bbox:
            continue
        if float(area) <= MIN_AREA:
            continue

        valid.append(
            {
                "object_id": int(object_id),
                "area": float(area),
                "bbox": bbox,
            }
        )

    return valid


def save_image(image_bytes, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(BytesIO(image_bytes)) as image:
        image.save(output_path)


def save_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def download_commonforms():
    DATASET_ROOT.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=DATASET_ROOT,
        force_download=False,
    )


def process_parquet_file(parquet_path, ocr_engine):
    split = parquet_path.name.split("-")[0]
    image_dir = OUTPUT_ROOT / "images" / split
    meta_dir = OUTPUT_ROOT / "metadata" / split

    saved = 0
    seen = 0

    parquet_file = pq.ParquetFile(parquet_path)
    for batch in parquet_file.iter_batches(batch_size=128):
        for row in batch.to_pylist():
            seen += 1

            valid_objects = get_valid_fillable_objects(row["objects"])
            if len(valid_objects) < MIN_FILLABLE_COUNT:
                continue

            if not row["image"] or not row["image"].get("bytes"):
                continue

            base_name = safe_name(f"{split}__{row['image_id']}__{Path(row['file_name']).stem}")
            output_image = image_dir / f"{base_name}{image_suffix(row['file_name'])}"
            output_json = meta_dir / f"{base_name}.json"

            if output_image.exists() and output_json.exists():
                saved += 1
                continue

            image_bytes = row["image"]["bytes"]
            result, _ = ocr_engine(image_bytes)
            ocr_text = normalize_ocr_text(result)
            language, confidence = detect_language(ocr_text)

            if language != "en" or confidence < MIN_ENGLISH_CONFIDENCE:
                continue

            save_image(image_bytes, output_image)

            metadata = {
                "row_id": int(row["id"]),
                "image_id": int(row["image_id"]),
                "split": split,
                "file_name": row["file_name"],
                "width": int(row["width"]),
                "height": int(row["height"]),
                "fillable_count": len(valid_objects),
                "qualifying_objects": valid_objects,
                "ocr_language": language,
                "ocr_language_confidence": confidence,
                "ocr_text_preview": ocr_text[:500],
                "image_path": str(output_image),
            }
            save_json(output_json, metadata)
            saved += 1

    return seen, saved

if __name__ == "__main__":
    download_commonforms()
    
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    ocr_engine = RapidOCR()

    total_seen = 0
    total_saved = 0

    for parquet_path in sorted(PARQUET_DIR.glob("*.parquet")):
        seen, saved = process_parquet_file(parquet_path, ocr_engine)
        total_seen += seen
        total_saved += saved
