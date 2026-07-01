import fastapi
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
import torch
import json
import os
from PIL import Image, ImageOps
import pytesseract
from pathlib import Path
from uuid import uuid4

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")

from transformers import (
    AutoProcessor,
    LayoutLMv3ImageProcessor,
    LayoutLMv3Model,
    LayoutLMv3Processor,
    LayoutLMv3TokenizerFast,
)
import torch.nn.functional as F
import torch.nn as nn
import psycopg
import math
import base64
import requests
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_TESSERACT_CMD = Path(os.getenv("TESSERACT_CMD", r"C:\Program Files\Tesseract-OCR\tesseract.exe"))
MODEL_PATH = Path(os.getenv("ODC_MODEL_PATH", str(PROJECT_ROOT / "models" / "odc_projection_scripted.pt")))
DOCUMENT_AI_ENDPOINT = os.getenv(
    "DOCUMENT_AI_ENDPOINT",
    "https://us-documentai.googleapis.com/v1/projects/971280404964/locations/us/processors/18e2688f80888cc9:process",
)
GCLOUD_CMD = Path(os.getenv("GCLOUD_CMD", "gcloud"))
REALFORM_DSN = os.getenv("REALFORM_DSN", "postgresql://postgres:postgres@localhost:5432/realform")
import uvicorn
app = FastAPI(title="ODC Service API")
GALLERY_DIR = PROJECT_ROOT / "gallery"
GALLERY_DIR.mkdir(parents=True, exist_ok=True)

UPLOAD_DIR = PROJECT_ROOT / "uploaded_imgs"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def connect_db():
    conn = psycopg.connect(REALFORM_DSN)
    return conn

def get_gcloud_token():
    import subprocess

    result = subprocess.run(
        [str(GCLOUD_CMD), "auth", "print-access-token"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()

def request_document_ai(image_bytes, mime_type="image/png", endpoint=DOCUMENT_AI_ENDPOINT):
    payload = {
        "rawDocument": {
            "content": base64.b64encode(image_bytes).decode("ascii"),
            "mimeType": mime_type,
        }
    }
    response = requests.post(
        endpoint,
        headers={
            "Authorization": f"Bearer {get_gcloud_token()}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=180,
    )
    if response.status_code >= 400:
        raise HTTPException(status_code=response.status_code, detail=response.text)
    return response.json()

def load_model(model_path = MODEL_PATH):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # chekcpoint_dict = torch.load(model_path, map_location=device, weights_only=False)
    # model = chekcpoint_dict["model"]
    model = torch.jit.load(model_path, map_location="cpu")
    model.eval()
    # model.to(device)
    return model

def tesserac_tocr_image(image, psm=6, oem=3, min_confidence=30):
    pytesseract.pytesseract.tesseract_cmd = str(DEFAULT_TESSERACT_CMD)
    image_width, image_height = image.size
    scale = 2.0
    processed = ImageOps.autocontrast(
        image.convert("L").resize(
            (int(image_width * scale), int(image_height * scale)),
            resample=Image.Resampling.BICUBIC,
        )
    )
    config = f"--psm {psm} --oem {oem}"
    data = pytesseract.image_to_data(
        processed,
        config=config,
        output_type=pytesseract.Output.DICT,
        lang="eng",
    )
    words = []
    bboxes = []
    confidences = []

    for i in range(len(data.get("text", []))):
        text = str(data["text"][i]).strip()
        if not text:
            continue
        try:
            confidence = float(data["conf"][i])
        except (TypeError, ValueError):
            confidence = -1.0
        if confidence < min_confidence:
            continue
        left = int(round(int(data["left"][i]) / scale))
        top = int(round(int(data["top"][i]) / scale))
        width = max(1, int(round(int(data["width"][i]) / scale)))
        height = max(1, int(round(int(data["height"][i]) / scale)))
        x1 = max(0, left)
        y1 = max(0, top)
        x2 = min(image_width, left + width)
        y2 = min(image_height, top + height)
        bboxes.append([
            max(0, min(1000, round(1000 * x1 / image_width))),
            max(0, min(1000, round(1000 * y1 / image_height))),
            max(0, min(1000, round(1000 * x2 / image_width))),
            max(0, min(1000, round(1000 * y2 / image_height))),
        ])
        words.append(text)
        confidences.append(confidence)
    if not words:
        words = ["[EMPTY]"]
        bboxes = [[0, 0, 1, 1]]
        confidences = [0.0]
    return {
        "image_size": {"width": image_width, "height": image_height},
        "word_count": len(words),
        "words": words,
        "bboxes": bboxes,
        "confidences": confidences,
        "engine": "tesseract",
        "psm": psm,
        "oem": oem,
    }
class ProcessorApplier:
    def __init__(self, processor , max_length = 512):
        self.processor = processor
        self.max_length = max_length
    def apply(self, image, payload):
        encoding = self.processor(
            images=[image],
            text=[payload["words"]],
            boxes=[payload["bboxes"]],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        ) 
        return{
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "bbox": encoding["bbox"],
            "pixel_values": encoding["pixel_values"]
        }

def get_mebed(model, payload):
    with torch.no_grad():
        # device = "cuda" if torch.cuda.is_available() else "cpu"
        device = "cpu"
        model.eval()
        model.to(device)
        input_ids = payload["input_ids"].to(device)
        attention_mask = payload["attention_mask"].to(device)
        bbox = payload["bbox"].to(device)
        pixel_values = payload["pixel_values"].to(device)
        with torch.no_grad():
            outputs = model(input_ids, attention_mask, bbox, pixel_values)
        if isinstance(outputs, dict):
            return outputs["embeddings"]
        return outputs

def _prepare_embedding_vector(raw_embedding, expected_dim=128):
    tensor = raw_embedding.detach().float().cpu()

    if tensor.ndim == 3:
        tensor = tensor.mean(dim=1)
    if tensor.ndim == 2:
        tensor = tensor[0]

    vector = tensor.flatten()
    if vector.numel() > expected_dim:
        vector = vector[:expected_dim]
    elif vector.numel() < expected_dim:
        vector = F.pad(vector, (0, expected_dim - vector.numel()))

    norm = torch.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm

    return vector.tolist()


def _to_pgvector(vector_values):
    return "[" + ",".join(f"{x:.8f}" for x in vector_values) + "]"


def embed_and_write_db(image, model, applier, conn, template_id, display_name=None, image_path=None,):
    if isinstance(image, (str, Path)):
        image = Image.open(image).convert("RGB")

    ocr_payload = tesserac_tocr_image(image)
    model_payload = applier.apply(image, ocr_payload)
    raw_embedding = get_mebed(model, model_payload)

    embedding_vector = _prepare_embedding_vector(raw_embedding, expected_dim=128)
    embedding_literal = _to_pgvector(embedding_vector)

    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO templates (template_id, display_name, image_path, ocr_json, embedding)
            VALUES (%s, %s, %s, %s::jsonb, %s::vector)
            """,
            (
                template_id,
                display_name,
                str(image_path) if image_path is not None else None,
                json.dumps(ocr_payload),
                embedding_literal,
            ),
        )
    conn.commit()


processor = AutoProcessor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
applier = ProcessorApplier(processor, max_length=512)
model = load_model()

@app.post("/embed")
async def process_embed(
    template_id: str = Form(...),
    image: UploadFile = File(...),
    display_name: str = Form(""),
):
    content = await image.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded image is empty")

    ext = Path(image.filename or "upload.png").suffix or ".png"
    image_file = GALLERY_DIR / f"{template_id}_{uuid4().hex}{ext}"
    image_file.write_bytes(content)

    conn = None
    try:
        conn = connect_db()
        embed_and_write_db(
            image=str(image_file),
            model=model,
            applier=applier,
            conn=conn,
            template_id=template_id,
            display_name=display_name,
            image_path=str(image_file),
        )
        return {"ok": True, "template_id": template_id, "image_path": str(image_file)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if conn is not None:
            conn.close()
        
        
@app.post("/query")
async def query(image:UploadFile, top_k = 10):
    content = await image.read()
    if not content:
        raise HTTPException(status_code=400)
    ext = Path(image.filename or "query.png").suffix or ".png"
    image_file = UPLOAD_DIR / f"query_{uuid4().hex}{ext}"
    image_file.write_bytes(content)
    conn = None
    image = Image.open(image_file).convert("RGB")
    ocr_payload = tesserac_tocr_image(image)
    model_payload = applier.apply(image, ocr_payload)
    raw_embedding = get_mebed(model, model_payload)
    embedding_vector = _prepare_embedding_vector(raw_embedding, expected_dim=128)
    embedding_literal = _to_pgvector(embedding_vector)
    try:
        conn = connect_db()
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    id,
                    template_id,
                    display_name,
                    image_path,
                    ocr_json->>'word_count' AS word_count,
                    1 - (embedding <=> %s::vector) AS cosine_similarity
                FROM templates
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (embedding_literal, embedding_literal,top_k),
            )
            rows = cur.fetchall()
            matches = []
            for row in rows:
                matches.append({
                    "id": row[0],
                    "template_id": row[1],
                    "display_name": row[2],
                    "image_path": row[3],
                    "word_count": row[4],
                    "cosine_similarity": float(row[5]),
                })

            return {
                "ok": True,
                "query_image_path": str(image_file),
                "query_word_count": ocr_payload["word_count"],
                "matches": matches,
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if conn is not None:
            conn.close()

@app.get("/templates")
async def get_all_template():
    conn = None
    try:
        conn = connect_db()
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    id,
                    template_id,
                    display_name,
                    image_path,
                    ocr_json->>'word_count' AS word_count
                FROM templates
                ORDER BY id DESC
                """,
            )
            rows = cur.fetchall()
            templates = []
            for row in rows:
                templates.append({
                    "id": row[0],
                    "template_id": row[1],
                    "display_name": row[2],
                    "image_path": row[3],
                    "word_count": row[4],
                })

            return {
                "ok": True,
                "templates": templates,
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if conn is not None:
            conn.close()

@app.delete("/templates/{template_id}")
async def remove_template(template_id: str):
    conn = None
    try:
        conn = connect_db()
        with conn.cursor() as cur:
            cur.execute(
                """
                DELETE FROM templates
                WHERE template_id = %s
                """,
                (template_id,),
            )
            deleted_count = cur.rowcount
            conn.commit()
            return {"ok": True, "deleted_count": deleted_count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if conn is not None:
            conn.close()

@app.put("/templates/{template_id}")
async def update_and_embed_template(template_id: str, image_file: UploadFile):
    content = await image_file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded image is empty")

    ext = Path(image_file.filename or f"{template_id}.png").suffix or ".png"
    new_image_path = GALLERY_DIR / f"{template_id}_{uuid4().hex}{ext}"
    new_image_path.write_bytes(content)

    conn = None
    try:
        conn = connect_db()
        embed_and_write_db(
            image=str(new_image_path),
            model=model,
            applier=applier,
            conn=conn,
            template_id=template_id,
            display_name=None,
            image_path=str(new_image_path),
        )
        return {"ok": True, "template_id": template_id, "new_image_path": str(new_image_path)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if conn is not None:
            conn.close()

@app.post("/documentai/process")
async def documentai_process(
    image_file: UploadFile = File(...),
    endpoint: str = Form(DOCUMENT_AI_ENDPOINT),
):
    content = await image_file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded image is empty")
    mime_type = image_file.content_type or "image/png"
    return request_document_ai(content, mime_type=mime_type, endpoint=endpoint)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8005)
