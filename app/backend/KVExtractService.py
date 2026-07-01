import json
import os
from io import BytesIO
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image, ImageDraw, ImageFont
import requests
import torch
from transformers import AutoModelForTokenClassification, AutoProcessor
from transformers.utils import generic as hf_generic
from transformers.utils import import_utils as hf_import_utils
import transformers.image_transforms as hf_image_transforms
import uvicorn

hf_import_utils._tf_available = False
hf_generic._is_tensorflow = lambda _x: False
hf_image_transforms.is_tf_tensor = lambda _x: False

os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["USE_TF"] = "0"

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR = Path(os.getenv("KVP_MODEL_DIR", str(PROJECT_ROOT / "models" / "Key-Value-Pair")))
ODC_DOCUMENTAI_URL = os.getenv("ODC_DOCUMENTAI_URL", "http://localhost:8005/documentai/process")
OUTPUT_DIR = PROJECT_ROOT / "uploaded_imgs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="KV Extract Service API")


class ProcessorApplier:
    def __init__(self, processor, max_length=512):
        self.processor = processor
        self.max_length = max_length

    def apply(self, image, payload):
        return self.processor(
            images=[image],
            text=[payload["words"]],
            boxes=[payload["bboxes"]],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )


def _clamp01(value):
    return max(0.0, min(1.0, float(value)))


def _extract_text_from_anchor(anchor, full_text):
    if not anchor:
        return ""
    segments = anchor.get("textSegments", [])
    if not segments:
        return ""

    text_parts = []
    text_len = len(full_text)
    for seg in segments:
        start = int(seg.get("startIndex", 0) or 0)
        end = int(seg.get("endIndex", 0) or 0)
        if end <= start:
            continue
        if start >= text_len:
            continue
        text_parts.append(full_text[start:min(end, text_len)])
    return "".join(text_parts).strip()


def _layout_to_norm_box(layout, image_width, image_height):
    layout = layout or {}
    poly = layout.get("boundingPoly", {}) or {}

    normalized_vertices = poly.get("normalizedVertices", []) or []
    if normalized_vertices:
        xs = [_clamp01(v.get("x", 0.0) or 0.0) for v in normalized_vertices]
        ys = [_clamp01(v.get("y", 0.0) or 0.0) for v in normalized_vertices]
        if xs and ys:
            x1 = min(xs)
            y1 = min(ys)
            x2 = max(xs)
            y2 = max(ys)
            return x1, y1, x2, y2

    vertices = poly.get("vertices", []) or []
    if vertices and image_width > 0 and image_height > 0:
        xs = [float(v.get("x", 0.0) or 0.0) / float(image_width) for v in vertices]
        ys = [float(v.get("y", 0.0) or 0.0) / float(image_height) for v in vertices]
        if xs and ys:
            x1 = _clamp01(min(xs))
            y1 = _clamp01(min(ys))
            x2 = _clamp01(max(xs))
            y2 = _clamp01(max(ys))
            return x1, y1, x2, y2

    return None


def _documentai_to_payload(ocr_response, image_width, image_height):
    document = ocr_response.get("document", {})
    full_text = document.get("text", "")

    words = []
    bboxes = []
    pixel_bboxes = []

    for page in document.get("pages", []):
        for token in page.get("tokens", []):
            layout = token.get("layout", {})
            token_text = _extract_text_from_anchor(layout.get("textAnchor", {}), full_text)
            if not token_text:
                continue

            norm_box = _layout_to_norm_box(layout, image_width, image_height)
            if norm_box is None:
                continue

            nx1, ny1, nx2, ny2 = norm_box
            x1 = max(0, min(1000, int(round(nx1 * 1000))))
            y1 = max(0, min(1000, int(round(ny1 * 1000))))
            x2 = max(0, min(1000, int(round(nx2 * 1000))))
            y2 = max(0, min(1000, int(round(ny2 * 1000))))

            px1 = max(0, min(image_width - 1, int(round(nx1 * image_width))))
            py1 = max(0, min(image_height - 1, int(round(ny1 * image_height))))
            px2 = max(px1 + 1, min(image_width, int(round(nx2 * image_width))))
            py2 = max(py1 + 1, min(image_height, int(round(ny2 * image_height))))

            words.append(token_text)
            bboxes.append([x1, y1, x2, y2])
            pixel_bboxes.append([px1, py1, px2, py2])

    if not words:
        raise HTTPException(status_code=422, detail="No OCR tokens returned from ODC DocumentAI")

    return {
        "image_size": {"width": image_width, "height": image_height},
        "word_count": len(words),
        "words": words,
        "bboxes": bboxes,
        "pixel_bboxes": pixel_bboxes,
        "engine": "odc-documentai",
    }


def _call_odc_documentai(image_bytes, filename, content_type):
    files = {
        "image_file": (filename, image_bytes, content_type or "image/png"),
    }

    try:
        response = requests.post(ODC_DOCUMENTAI_URL, files=files, timeout=180)
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"ODC service unreachable: {e}")

    if response.status_code >= 400:
        raise HTTPException(status_code=response.status_code, detail=f"ODC OCR failed: {response.text}")

    return response.json()


def _id_to_label(label_id):
    if label_id in model.config.id2label:
        return model.config.id2label[label_id]
    return model.config.id2label.get(str(label_id), "O")


def _predict_word_labels(image, payload):
    encoding = applier.apply(image, payload)

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    bbox = encoding["bbox"].to(device)
    pixel_values = encoding["pixel_values"].to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            bbox=bbox,
            pixel_values=pixel_values,
        )

    logits = outputs.logits
    probs = torch.softmax(logits, dim=-1)
    pred_ids = torch.argmax(logits, dim=-1)[0].detach().cpu().tolist()
    pred_scores = torch.max(probs, dim=-1).values[0].detach().cpu().tolist()

    word_ids = encoding.word_ids(batch_index=0)
    best_by_word = {}
    for token_idx, word_idx in enumerate(word_ids):
        if word_idx is None:
            continue
        if word_idx < 0 or word_idx >= payload["word_count"]:
            continue

        score = float(pred_scores[token_idx])
        label_id = int(pred_ids[token_idx])
        previous = best_by_word.get(word_idx)
        if previous is None or score > previous[1]:
            best_by_word[word_idx] = (label_id, score)

    labels = ["O"] * payload["word_count"]
    scores = [0.0] * payload["word_count"]

    for word_idx, (label_id, score) in best_by_word.items():
        labels[word_idx] = _id_to_label(label_id)
        scores[word_idx] = score

    return labels, scores


def _overlay_predictions(image, payload, labels):
    overlay = image.convert("RGB").copy()
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.load_default()

    color_map = {
        "B-KEY": "#1e40ff",
        "I-KEY": "#1e40ff",
        "B-VAL": "#11883a",
        "I-VAL": "#11883a",
    }

    for idx, label in enumerate(labels):
        if label == "O":
            continue

        box = payload["pixel_bboxes"][idx]
        color = color_map.get(label, "#333333")
        draw.rectangle(box, outline=color, width=2)

        text_x = box[0]
        text_y = max(0, box[1] - 12)
        try:
            text_box = draw.textbbox((text_x, text_y), label, font=font)
            draw.rectangle(text_box, fill="white")
        except Exception:
            pass
        draw.text((text_x, text_y), label, fill=color, font=font)

    return overlay


def _group_labeled_spans(payload, labels, scores, prefix):
    spans = []
    current = None

    for index, label in enumerate(labels):
        if label == f"B-{prefix}":
            if current:
                spans.append(current)
            current = {
                "label": prefix,
                "indices": [index],
                "text_parts": [payload["words"][index]],
                "scores": [float(scores[index])],
            }
            continue

        if label == f"I-{prefix}" and current:
            current["indices"].append(index)
            current["text_parts"].append(payload["words"][index])
            current["scores"].append(float(scores[index]))
            continue

        if current:
            spans.append(current)
            current = None

    if current:
        spans.append(current)

    normalized = []
    for span in spans:
        indices = span["indices"]
        pixel_boxes = [payload["pixel_bboxes"][idx] for idx in indices]
        norm_boxes = [payload["bboxes"][idx] for idx in indices]

        text = " ".join(part.strip() for part in span["text_parts"] if part.strip()).strip()
        if not text:
            continue

        normalized.append(
            {
                "label": prefix,
                "text": text,
                "score": round(sum(span["scores"]) / max(1, len(span["scores"])), 4),
                "pixel_bbox": [
                    min(box[0] for box in pixel_boxes),
                    min(box[1] for box in pixel_boxes),
                    max(box[2] for box in pixel_boxes),
                    max(box[3] for box in pixel_boxes),
                ],
                "bbox": [
                    min(box[0] for box in norm_boxes),
                    min(box[1] for box in norm_boxes),
                    max(box[2] for box in norm_boxes),
                    max(box[3] for box in norm_boxes),
                ],
            }
        )

    return normalized


def _dedupe_spans(spans):
    unique = []
    seen = set()

    for span in spans:
        key = (span["text"].casefold(), tuple(span["bbox"]))
        if key in seen:
            continue
        seen.add(key)
        unique.append(span)

    return unique


def _fallback_value_spans(payload):
    words = payload["words"]
    pixel_boxes = payload["pixel_bboxes"]
    norm_boxes = payload["bboxes"]

    items = []
    for idx, word in enumerate(words):
        cleaned = str(word).strip()
        if not cleaned:
            continue
        box = pixel_boxes[idx]
        items.append(
            {
                "bbox": norm_boxes[idx],
                "pixel_bbox": box,
                "text": cleaned,
                "x1": box[0],
                "x2": box[2],
                "y1": box[1],
                "y2": box[3],
                "y_center": (box[1] + box[3]) / 2.0,
                "height": max(1, box[3] - box[1]),
            }
        )

    items.sort(key=lambda item: (item["y_center"], item["x1"]))

    lines = []
    current = []
    current_center = None
    current_height = None

    for item in items:
        if not current:
            current = [item]
            current_center = item["y_center"]
            current_height = item["height"]
            continue

        tolerance = max(12.0, float(current_height or 0) * 0.8)
        if abs(item["y_center"] - float(current_center or 0.0)) <= tolerance:
            current.append(item)
            current_center = sum(entry["y_center"] for entry in current) / len(current)
            current_height = max(entry["height"] for entry in current)
            continue

        lines.append(current)
        current = [item]
        current_center = item["y_center"]
        current_height = item["height"]

    if current:
        lines.append(current)

    spans = []
    for line in lines:
        line.sort(key=lambda item: item["x1"])
        text = " ".join(entry["text"] for entry in line).strip()
        if len(text) < 2:
            continue

        spans.append(
            {
                "label": "VAL",
                "text": text,
                "score": 0.0,
                "pixel_bbox": [
                    min(entry["pixel_bbox"][0] for entry in line),
                    min(entry["pixel_bbox"][1] for entry in line),
                    max(entry["pixel_bbox"][2] for entry in line),
                    max(entry["pixel_bbox"][3] for entry in line),
                ],
                "bbox": [
                    min(entry["bbox"][0] for entry in line),
                    min(entry["bbox"][1] for entry in line),
                    max(entry["bbox"][2] for entry in line),
                    max(entry["bbox"][3] for entry in line),
                ],
            }
        )

    return spans


def _analyze_document(image_bytes, filename, content_type):
    try:
        pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    ocr_response = _call_odc_documentai(
        image_bytes=image_bytes,
        filename=filename,
        content_type=content_type,
    )
    ocr_payload = _documentai_to_payload(ocr_response, pil_image.width, pil_image.height)
    labels, scores = _predict_word_labels(pil_image, ocr_payload)
    overlay_image = _overlay_predictions(pil_image, ocr_payload, labels)

    output_path = OUTPUT_DIR / f"kv_overlay_{uuid4().hex}.png"
    overlay_image.save(output_path, format="PNG")

    value_spans = _dedupe_spans(_group_labeled_spans(ocr_payload, labels, scores, "VAL"))
    key_spans = _dedupe_spans(_group_labeled_spans(ocr_payload, labels, scores, "KEY"))
    if not value_spans:
        value_spans = _dedupe_spans(_fallback_value_spans(ocr_payload))

    return {
        "image_size": {"width": pil_image.width, "height": pil_image.height},
        "ocr_word_count": ocr_payload["word_count"],
        "overlay_image_path": str(output_path),
        "detected_values": value_spans,
        "detected_keys": key_spans,
        "ocr_payload": {
            "word_count": ocr_payload["word_count"],
            "words": ocr_payload["words"],
            "bboxes": ocr_payload["bboxes"],
            "pixel_bboxes": ocr_payload["pixel_bboxes"],
        },
    }


def _load_model_assets():
    active_device = "cuda" if torch.cuda.is_available() else "cpu"
    active_processor = AutoProcessor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
    active_model = AutoModelForTokenClassification.from_pretrained(str(MODEL_DIR))
    active_model.eval()
    active_model.to(active_device)
    return active_processor, active_model, active_device


processor, model, device = _load_model_assets()
applier = ProcessorApplier(processor, max_length=512)


@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded image is empty")
    analysis = _analyze_document(
        image_bytes=image_bytes,
        filename=image.filename or "upload.png",
        content_type=image.content_type or "image/png",
    )
    output_path = analysis["overlay_image_path"]
    return FileResponse(str(output_path), media_type="image/png", filename=Path(output_path).name)


@app.post("/analyze")
async def analyze(image: UploadFile = File(...)):
    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded image is empty")

    analysis = _analyze_document(
        image_bytes=image_bytes,
        filename=image.filename or "upload.png",
        content_type=image.content_type or "image/png",
    )

    return json.loads(json.dumps(analysis))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8006)
