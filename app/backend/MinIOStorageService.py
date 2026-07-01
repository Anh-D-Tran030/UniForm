import json
import os
import re
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import Response as FastAPIResponse
from minio import Minio
from minio.error import S3Error
import uvicorn


MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin123")
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "document-extractions")
MINIO_SECURE = os.getenv("MINIO_SECURE", "false").lower() in {"1", "true", "yes"}

app = FastAPI(title="MinIO Storage Service API")

client = Minio(
    MINIO_ENDPOINT,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=MINIO_SECURE,
)


def _clean_key_part(value, fallback) :
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value or "").strip()).strip("-")
    return cleaned or fallback


def _image_extension(filename, content_type):
    suffix = Path(filename or "").suffix.lower().lstrip(".")
    if suffix in {"png", "jpg", "jpeg", "webp", "tif", "tiff"}:
        return suffix

    content_type = (content_type or "").lower()
    if content_type == "image/png":
        return "png"
    if content_type in {"image/jpeg", "image/jpg"}:
        return "jpg"
    if content_type == "image/webp":
        return "webp"
    return "bin"


def _ensure_bucket() -> None:
    try:
        if not client.bucket_exists(MINIO_BUCKET):
            client.make_bucket(MINIO_BUCKET)
    except S3Error as exc:
        raise HTTPException(status_code=502, detail=f"MinIO bucket check failed: {exc}") from exc


def _isoformat(value):
    if not value:
        return None
    return value.astimezone(timezone.utc).isoformat()


def _parse_bronze_path(object_name):
    parts = object_name.split("/")
    if len(parts) < 4 or parts[:2] != ["bronze", "raw-images"]:
        return None
    return {
        "template_id": parts[2],
        "run_id": Path(parts[-1]).stem,
    }


def _parse_silver_path(object_name):
    parts = object_name.split("/")
    if len(parts) < 4 or parts[:2] != ["silver", "kvp-json"] or not object_name.endswith(".json"):
        return None
    return {
        "template_id": parts[2],
        "run_id": Path(parts[-1]).stem,
    }


def _read_json_object(object_name):
    response = None
    try:
        response = client.get_object(MINIO_BUCKET, object_name)
        return json.loads(response.read().decode("utf-8"))
    except S3Error as exc:
        if exc.code == "NoSuchKey":
            raise HTTPException(status_code=404, detail="Stored JSON was not found") from exc
        raise HTTPException(status_code=502, detail=f"MinIO read failed: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=502, detail=f"Stored JSON is invalid: {exc}") from exc
    finally:
        if response:
            response.close()
            response.release_conn()


@app.get("/forms")
async def list_forms():
    _ensure_bucket()

    forms = {}

    try:
        bronze_objects = client.list_objects(MINIO_BUCKET, prefix="bronze/raw-images/", recursive=True)
        for item in bronze_objects:
            parsed = _parse_bronze_path(item.object_name)
            if not parsed:
                continue
            key = (parsed["template_id"], parsed["run_id"])
            record = forms.setdefault(
                key,
                {
                    "run_id": parsed["run_id"],
                    "template_id": parsed["template_id"],
                    "status": "pending",
                    "bronze_path": None,
                    "silver_path": None,
                    "source_file_name": None,
                    "created_at": None,
                },
            )
            record["bronze_path"] = item.object_name
            record["created_at"] = record["created_at"] or _isoformat(item.last_modified)

        silver_objects = client.list_objects(MINIO_BUCKET, prefix="silver/kvp-json/", recursive=True)
        for item in silver_objects:
            parsed = _parse_silver_path(item.object_name)
            if not parsed:
                continue
            key = (parsed["template_id"], parsed["run_id"])
            record = forms.setdefault(
                key,
                {
                    "run_id": parsed["run_id"],
                    "template_id": parsed["template_id"],
                    "status": "pending",
                    "bronze_path": None,
                    "silver_path": None,
                    "source_file_name": None,
                    "created_at": None,
                },
            )
            record["silver_path"] = item.object_name
            record["status"] = "processed"
            record["created_at"] = record["created_at"] or _isoformat(item.last_modified)
            try:
                payload = _read_json_object(item.object_name)
            except HTTPException:
                payload = {}
            if isinstance(payload, dict):
                record["source_file_name"] = payload.get("source_file_name")
                record["created_at"] = payload.get("created_at") or record["created_at"]
    except S3Error as exc:
        raise HTTPException(status_code=502, detail=f"MinIO list failed: {exc}") from exc

    sorted_forms = sorted(
        forms.values(),
        key=lambda item: item.get("created_at") or "",
        reverse=True,
    )
    return {"forms": sorted_forms}


@app.get("/forms/{template_id}/{run_id}/json")
async def get_form_json(template_id: str, run_id: str):
    _ensure_bucket()
    safe_template_id = _clean_key_part(template_id, "unknown")
    safe_run_id = _clean_key_part(run_id, "")
    if not safe_run_id:
        raise HTTPException(status_code=400, detail="Missing run_id")
    return _read_json_object(f"silver/kvp-json/{safe_template_id}/{safe_run_id}.json")


@app.post("/ingest")
async def ingest(
    image: UploadFile = File(...),
    kvp_json: str = Form(...),
    run_id: str | None = Form(None),
    template_id: str | None = Form(None),
):
    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded image is empty")

    try:
        kvp_payload = json.loads(kvp_json)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid kvp_json: {exc}") from exc

    _ensure_bucket()

    safe_run_id = _clean_key_part(run_id, uuid4().hex)
    safe_template_id = _clean_key_part(template_id, "unknown")
    image_ext = _image_extension(image.filename, image.content_type)
    image_content_type = image.content_type or "application/octet-stream"

    bronze_path = f"bronze/raw-images/{safe_template_id}/{safe_run_id}.{image_ext}"
    silver_path = f"silver/kvp-json/{safe_template_id}/{safe_run_id}.json"

    envelope = {
        "run_id": safe_run_id,
        "template_id": safe_template_id,
        "source_file_name": image.filename,
        "image_object": bronze_path,
        "kvp": kvp_payload,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    silver_bytes = json.dumps(envelope, indent=2, ensure_ascii=False).encode("utf-8")

    try:
        client.put_object(
            MINIO_BUCKET,
            bronze_path,
            BytesIO(image_bytes),
            length=len(image_bytes),
            content_type=image_content_type,
        )
        client.put_object(
            MINIO_BUCKET,
            silver_path,
            BytesIO(silver_bytes),
            length=len(silver_bytes),
            content_type="application/json",
        )
    except S3Error as exc:
        raise HTTPException(status_code=502, detail=f"MinIO write failed: {exc}") from exc

    return {
        "run_id": safe_run_id,
        "bucket": MINIO_BUCKET,
        "bronze_path": bronze_path,
        "silver_path": silver_path,
    }


def _object_kind(object_name):
    if object_name.startswith("bronze/raw-images/"):
        return "bronze"
    if object_name.startswith("silver/kvp-json/"):
        return "silver"
    return "other"


@app.get("/objects")
async def list_objects(prefix: str = Query("", description="Optional key prefix filter")):
    """Raw object listing for the in-app object browser. Returns every object in
    the bucket with size / last-modified / content-type metadata."""
    _ensure_bucket()

    objects = []
    try:
        for item in client.list_objects(MINIO_BUCKET, prefix=prefix or None, recursive=True):
            objects.append(
                {
                    "name": item.object_name,
                    "kind": _object_kind(item.object_name),
                    "size": item.size,
                    "last_modified": _isoformat(item.last_modified),
                    "etag": (item.etag or "").strip('"') or None,
                }
            )
    except S3Error as exc:
        raise HTTPException(status_code=502, detail=f"MinIO list failed: {exc}") from exc

    objects.sort(key=lambda obj: obj.get("last_modified") or "", reverse=True)
    return {"bucket": MINIO_BUCKET, "count": len(objects), "objects": objects}


@app.get("/object")
async def get_object(key: str = Query(..., description="Full object key to fetch")):
    """Stream a single object's bytes so the UI can preview images / download files."""
    _ensure_bucket()
    if not key:
        raise HTTPException(status_code=400, detail="Missing object key")

    response = None
    try:
        stat = client.stat_object(MINIO_BUCKET, key)
        response = client.get_object(MINIO_BUCKET, key)
        data = response.read()
    except S3Error as exc:
        if exc.code in {"NoSuchKey", "NoSuchObject"}:
            raise HTTPException(status_code=404, detail="Object not found") from exc
        raise HTTPException(status_code=502, detail=f"MinIO read failed: {exc}") from exc
    finally:
        if response:
            response.close()
            response.release_conn()

    content_type = getattr(stat, "content_type", None) or "application/octet-stream"
    filename = Path(key).name
    return FastAPIResponse(
        content=data,
        media_type=content_type,
        headers={"Content-Disposition": f'inline; filename="{filename}"'},
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8007)
