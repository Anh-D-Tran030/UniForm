import json
import os
import re
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from statistics import mean
from typing import Any

from fastapi import FastAPI, HTTPException
from minio import Minio
from minio.error import S3Error
import uvicorn


MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin123")
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "document-extractions")
MINIO_SECURE = os.getenv("MINIO_SECURE", "false").lower() in {"1", "true", "yes"}

app = FastAPI(title="Gold Tier Transform Service API")

client = Minio(
    MINIO_ENDPOINT,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=MINIO_SECURE,
)


def _clean_key_part(value: Any, fallback: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value or "").strip()).strip("-")
    return cleaned or fallback


def _ensure_bucket() -> None:
    try:
        if not client.bucket_exists(MINIO_BUCKET):
            client.make_bucket(MINIO_BUCKET)
    except S3Error as exc:
        raise HTTPException(status_code=502, detail=f"MinIO bucket check failed: {exc}") from exc


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json_object(object_name: str) -> dict[str, Any]:
    response = None
    try:
        response = client.get_object(MINIO_BUCKET, object_name)
        payload = json.loads(response.read().decode("utf-8"))
    except S3Error as exc:
        if exc.code in {"NoSuchKey", "NoSuchObject"}:
            raise HTTPException(status_code=404, detail=f"Silver object not found: {object_name}") from exc
        raise HTTPException(status_code=502, detail=f"MinIO read failed: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=502, detail=f"Silver JSON is invalid: {exc}") from exc
    finally:
        if response:
            response.close()
            response.release_conn()

    if not isinstance(payload, dict):
        raise HTTPException(status_code=422, detail="Silver JSON must be an object")
    return payload


def _put_json(object_name: str, payload: dict[str, Any]) -> None:
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    try:
        client.put_object(
            MINIO_BUCKET,
            object_name,
            BytesIO(data),
            length=len(data),
            content_type="application/json",
        )
    except S3Error as exc:
        raise HTTPException(status_code=502, detail=f"MinIO write failed: {exc}") from exc


def _put_jsonl(object_name: str, rows: list[dict[str, Any]]) -> None:
    body = "".join(
        f"{json.dumps(row, ensure_ascii=False, separators=(',', ':'))}\n" for row in rows
    ).encode("utf-8")
    try:
        client.put_object(
            MINIO_BUCKET,
            object_name,
            BytesIO(body),
            length=len(body),
            content_type="application/x-ndjson",
        )
    except S3Error as exc:
        raise HTTPException(status_code=502, detail=f"MinIO write failed: {exc}") from exc


def _parse_silver_path(object_name: str) -> tuple[str, str] | None:
    parts = object_name.split("/")
    if len(parts) < 4 or parts[:2] != ["silver", "kvp-json"] or not object_name.endswith(".json"):
        return None
    return parts[2], Path(parts[-1]).stem


def _silver_path(template_id: str, run_id: str) -> str:
    return f"silver/kvp-json/{_clean_key_part(template_id, 'unknown')}/{_clean_key_part(run_id, '')}.json"


def _as_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:
        return None
    return number


def _bbox_parts(prefix: str, value: Any) -> dict[str, float | None]:
    if not isinstance(value, list) or len(value) < 4:
        coords: list[float | None] = [None, None, None, None]
    else:
        coords = [_as_number(item) for item in value[:4]]

    return {
        f"{prefix}_bbox_x0": coords[0],
        f"{prefix}_bbox_y0": coords[1],
        f"{prefix}_bbox_x1": coords[2],
        f"{prefix}_bbox_y1": coords[3],
    }


def _extract_pairs(silver_payload: dict[str, Any]) -> list[dict[str, Any]]:
    kvp_payload = silver_payload.get("kvp")
    if isinstance(kvp_payload, dict) and isinstance(kvp_payload.get("key_values"), list):
        return [item for item in kvp_payload["key_values"] if isinstance(item, dict)]
    if isinstance(silver_payload.get("key_values"), list):
        return [item for item in silver_payload["key_values"] if isinstance(item, dict)]
    return []


def _payload_metadata(silver_payload: dict[str, Any], fallback_template_id: str, fallback_run_id: str) -> dict[str, str | None]:
    kvp_payload = silver_payload.get("kvp") if isinstance(silver_payload.get("kvp"), dict) else {}
    return {
        "run_id": str(silver_payload.get("run_id") or kvp_payload.get("run_id") or fallback_run_id),
        "template_id": str(silver_payload.get("template_id") or kvp_payload.get("template_id") or fallback_template_id),
        "source_file_name": silver_payload.get("source_file_name") or kvp_payload.get("source_file_name"),
        "created_at": silver_payload.get("created_at") or kvp_payload.get("created_at"),
        "bronze_path": silver_payload.get("image_object") or silver_payload.get("bronze_path"),
    }


def transform_silver_object(silver_object: str) -> dict[str, Any]:
    parsed = _parse_silver_path(silver_object)
    if not parsed:
        raise HTTPException(status_code=400, detail="Object must be under silver/kvp-json/{template_id}/{run_id}.json")

    fallback_template_id, fallback_run_id = parsed
    silver_payload = _read_json_object(silver_object)
    metadata = _payload_metadata(silver_payload, fallback_template_id, fallback_run_id)
    run_id = _clean_key_part(metadata["run_id"], fallback_run_id)
    template_id = _clean_key_part(metadata["template_id"], fallback_template_id)
    transformed_at = _now()
    pairs = _extract_pairs(silver_payload)

    field_rows: list[dict[str, Any]] = []
    confidences: list[float] = []

    for index, pair in enumerate(pairs):
        confidence = _as_number(pair.get("score"))
        if confidence is not None:
            confidences.append(confidence)

        field_rows.append(
            {
                "run_id": run_id,
                "template_id": template_id,
                "field_index": index,
                "field_key": str(pair.get("key") or ""),
                "field_value": str(pair.get("value") or ""),
                "confidence": confidence,
                "source_file_name": metadata["source_file_name"],
                "silver_path": silver_object,
                "created_at": metadata["created_at"],
                "transformed_at": transformed_at,
                **_bbox_parts("key", pair.get("key_bbox")),
                **_bbox_parts("value", pair.get("value_bbox")),
            }
        )

    field_count = len(field_rows)
    gold_form_run_path = f"gold/form-runs/{template_id}/{run_id}.json"
    gold_fields_path = f"gold/extracted-fields/{template_id}/{run_id}.jsonl"
    gold_manifest_path = f"gold/manifests/{template_id}/{run_id}.json"

    form_run = {
        "run_id": run_id,
        "template_id": template_id,
        "source_file_name": metadata["source_file_name"],
        "bronze_path": metadata["bronze_path"],
        "silver_path": silver_object,
        "gold_form_run_path": gold_form_run_path,
        "gold_fields_path": gold_fields_path,
        "gold_manifest_path": gold_manifest_path,
        "field_count": field_count,
        "avg_confidence": mean(confidences) if confidences else None,
        "min_confidence": min(confidences) if confidences else None,
        "max_confidence": max(confidences) if confidences else None,
        "created_at": metadata["created_at"],
        "transformed_at": transformed_at,
        "status": "gold_ready",
    }
    manifest = {
        **form_run,
    }

    _put_json(gold_form_run_path, form_run)
    _put_jsonl(gold_fields_path, field_rows)
    _put_json(gold_manifest_path, manifest)

    return {
        "run_id": run_id,
        "template_id": template_id,
        "field_count": field_count,
        "silver_path": silver_object,
        "gold_form_run_path": gold_form_run_path,
        "gold_fields_path": gold_fields_path,
        "gold_manifest_path": gold_manifest_path,
    }


def _list_silver_objects(template_id: str | None = None, run_id: str | None = None) -> list[str]:
    if template_id and run_id:
        return [_silver_path(template_id, run_id)]
    prefix = "silver/kvp-json/"
    if template_id:
        prefix = f"silver/kvp-json/{_clean_key_part(template_id, 'unknown')}/"

    try:
        objects = [
            item.object_name
            for item in client.list_objects(MINIO_BUCKET, prefix=prefix, recursive=True)
            if _parse_silver_path(item.object_name)
        ]
    except S3Error as exc:
        raise HTTPException(status_code=502, detail=f"MinIO list failed: {exc}") from exc

    if run_id:
        safe_run_id = _clean_key_part(run_id, "")
        objects = [name for name in objects if Path(name).stem == safe_run_id]
    return objects


@app.get("/health")
async def health():
    return {"ok": True, "service": "gold-tier", "bucket": MINIO_BUCKET}


@app.get("/gold/summary")
async def gold_summary():
    _ensure_bucket()
    counts = {"form_runs": 0, "extracted_fields": 0, "manifests": 0}
    try:
        for item in client.list_objects(MINIO_BUCKET, prefix="gold/", recursive=True):
            if item.object_name.startswith("gold/form-runs/"):
                counts["form_runs"] += 1
            elif item.object_name.startswith("gold/extracted-fields/"):
                counts["extracted_fields"] += 1
            elif item.object_name.startswith("gold/manifests/"):
                counts["manifests"] += 1
    except S3Error as exc:
        raise HTTPException(status_code=502, detail=f"MinIO list failed: {exc}") from exc

    return {"bucket": MINIO_BUCKET, "counts": counts}


@app.post("/transform")
async def transform(template_id: str | None = None, run_id: str | None = None):
    _ensure_bucket()
    silver_objects = _list_silver_objects(template_id=template_id, run_id=run_id)
    results = [transform_silver_object(object_name) for object_name in silver_objects]
    return {"count": len(results), "results": results}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8009)
