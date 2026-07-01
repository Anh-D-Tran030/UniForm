# KVPService.py — port 8007
import os
import requests
import torch
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from io import BytesIO
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoModelForTokenClassification, AutoProcessor
from transformers.utils import generic as hf_generic
from transformers.utils import import_utils as hf_import_utils
import transformers.image_transforms as hf_image_transforms
from uuid import uuid4


hf_import_utils._tf_available       = False
hf_generic._is_tensorflow            = lambda _x: False
hf_image_transforms.is_tf_tensor     = lambda _x: False

os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["USE_TF"]             = "0"


MODEL_DIR          = Path("./layoutlmv3-finetuned-kvp-100ep/best_model")
ODC_DOCUMENTAI_URL = os.getenv("ODC_DOCUMENTAI_URL", "http://localhost:8005/documentai/process")
OUTPUT_DIR         = Path("uploaded_imgs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PORT = 8006

app = FastAPI(title="KVP Service API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ProcessorApplier:

    def __init__(self, processor, max_length=512):
        self.processor  = processor
        self.max_length = max_length

    def apply(self, image, ocr_payload):
        return self.processor(
            images=[image],
            text=[ocr_payload["words"]],
            boxes=[ocr_payload["bboxes"]],
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

    text_len  = len(full_text)
    text_parts = []
    for seg in segments:
        start = int(seg.get("startIndex", 0) or 0)
        end   = int(seg.get("endIndex",   0) or 0)
        if end <= start or start >= text_len:
            continue
        text_parts.append(full_text[start : min(end, text_len)])
    return "".join(text_parts).strip()


def _layout_to_norm_box(layout, image_width, image_height):

    layout = layout or {}
    poly   = layout.get("boundingPoly", {}) or {}

    norm_verts = poly.get("normalizedVertices", []) or []
    if norm_verts:
        xs = [_clamp01(v.get("x", 0.0) or 0.0) for v in norm_verts]
        ys = [_clamp01(v.get("y", 0.0) or 0.0) for v in norm_verts]
        if xs and ys:
            return min(xs), min(ys), max(xs), max(ys)

    abs_verts = poly.get("vertices", []) or []
    if abs_verts and image_width > 0 and image_height > 0:
        xs = [_clamp01(float(v.get("x", 0.0) or 0.0) / image_width)  for v in abs_verts]
        ys = [_clamp01(float(v.get("y", 0.0) or 0.0) / image_height) for v in abs_verts]
        if xs and ys:
            return min(xs), min(ys), max(xs), max(ys)

    return None


def _documentai_to_payload(ocr_response, image_width, image_height):

    document  = ocr_response.get("document", {})
    full_text = document.get("text", "")

    words, bboxes, pixel_bboxes = [], [], []

    for page in document.get("pages", []):
        for token in page.get("tokens", []):
            layout     = token.get("layout", {})
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

            px1 = max(0, min(image_width  - 1, int(round(nx1 * image_width))))
            py1 = max(0, min(image_height - 1, int(round(ny1 * image_height))))
            px2 = max(px1 + 1, min(image_width,  int(round(nx2 * image_width))))
            py2 = max(py1 + 1, min(image_height, int(round(ny2 * image_height))))

            words.append(token_text)
            bboxes.append([x1, y1, x2, y2])
            pixel_bboxes.append([px1, py1, px2, py2])

    if not words:
        raise HTTPException(status_code=422, detail="No OCR tokens returned from ODC DocumentAI")

    return {
        "image_size":   {"width": image_width, "height": image_height},
        "word_count":   len(words),
        "words":        words,
        "bboxes":       bboxes,
        "pixel_bboxes": pixel_bboxes,
        "engine":       "odc-documentai",
    }


def _call_odc_documentai(image_bytes, filename, content_type, odc_url):

    try:
        response = requests.post(
            odc_url,
            files={"image_file": (filename, image_bytes, content_type or "image/png")},
            timeout=180,
        )
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"ODC service unreachable: {e}")

    if response.status_code >= 400:
        raise HTTPException(
            status_code=response.status_code,
            detail=f"ODC OCR failed: {response.text}",
        )
    return response.json()




def _id_to_label(label_id, model):

    if label_id in model.config.id2label:
        return model.config.id2label[label_id]
    return model.config.id2label.get(str(label_id), "O")


def _predict_word_labels(image, ocr_payload, model, device, applier,
                         chunk_size=400, overlap=50):
    total_words = ocr_payload["word_count"]
    all_words   = ocr_payload["words"]
    all_bboxes  = ocr_payload["bboxes"]

    
    best_by_word = {}

    start = 0
    while start < total_words:
        end = min(start + chunk_size, total_words)

        chunk_payload = {
            "words":      all_words[start:end],
            "bboxes":     all_bboxes[start:end],
            "word_count": end - start,
        }

        encoding = applier.apply(image, chunk_payload)

        with torch.no_grad():
            outputs = model(
                input_ids=      encoding["input_ids"].to(device),
                attention_mask= encoding["attention_mask"].to(device),
                bbox=           encoding["bbox"].to(device),
                pixel_values=   encoding["pixel_values"].to(device),
            )

        logits      = outputs.logits                                            
        probs       = torch.softmax(logits, dim=-1)
        pred_ids    = torch.argmax(logits, dim=-1)[0].detach().cpu().tolist()  
        pred_scores = torch.max(probs, dim=-1).values[0].detach().cpu().tolist()

        word_ids = encoding.word_ids(batch_index=0)

        for token_idx, chunk_word_idx in enumerate(word_ids):
            if chunk_word_idx is None:
                continue
            global_word_idx = start + chunk_word_idx
            if global_word_idx >= total_words:
                continue
            score    = float(pred_scores[token_idx])
            label_id = int(pred_ids[token_idx])
            prev     = best_by_word.get(global_word_idx)
            if prev is None or score > prev[1]:
                best_by_word[global_word_idx] = (label_id, score)

        start += chunk_size - overlap

    labels = ["O"]  * total_words
    scores = [0.0]  * total_words
    for word_idx, (label_id, score) in best_by_word.items():
        labels[word_idx] = _id_to_label(label_id, model)
        scores[word_idx] = score

    return labels, scores


def _overlay_predictions(image, ocr_payload, labels):
    overlay = image.convert("RGB").copy()
    draw    = ImageDraw.Draw(overlay)
    font    = ImageFont.load_default()

    role_color = {"KEY": "blue", "VAL": "green"}

    for idx, label in enumerate(labels):
        if label == "O":
            continue

        box   = ocr_payload["pixel_bboxes"][idx]
        role  = label.split("-", 1)[-1] if "-" in label else "other"
        color = role_color.get(role, "violet")

        draw.rectangle(box, outline=color, width=2)

        display_label = label.split("-", 1)[-1] if "-" in label else label

        text_x = box[0]
        text_y = max(0, box[1] - 12)
        try:
            text_bbox = draw.textbbox((text_x, text_y), display_label, font=font)
            draw.rectangle(text_bbox, fill="white")
        except Exception:
            pass
        draw.text((text_x, text_y), display_label, fill=color, font=font)

    return overlay


def _group_labeled_spans(ocr_payload, labels, scores, prefix):
    spans   = []
    current = None

    for index, label in enumerate(labels):
        if label == f"B-{prefix}":
            if current:
                spans.append(current)
            current = {
                "label":      prefix,
                "indices":    [index],
                "text_parts": [ocr_payload["words"][index]],
                "scores":     [float(scores[index])],
            }
        elif label == f"I-{prefix}" and current:
            current["indices"].append(index)
            current["text_parts"].append(ocr_payload["words"][index])
            current["scores"].append(float(scores[index]))
        else:
            if current:
                spans.append(current)
                current = None

    if current:
        spans.append(current)

    result = []
    for span in spans:
        indices     = span["indices"]
        pixel_boxes = [ocr_payload["pixel_bboxes"][i] for i in indices]
        norm_boxes  = [ocr_payload["bboxes"][i]       for i in indices]
        text        = " ".join(p.strip() for p in span["text_parts"] if p.strip()).strip()
        if not text:
            continue
        result.append({
            "label": prefix,
            "text":  text,
            "score": round(sum(span["scores"]) / max(1, len(span["scores"])), 4),
            "pixel_bbox": [
                min(b[0] for b in pixel_boxes), min(b[1] for b in pixel_boxes),
                max(b[2] for b in pixel_boxes), max(b[3] for b in pixel_boxes),
            ],
            "bbox": [
                min(b[0] for b in norm_boxes), min(b[1] for b in norm_boxes),
                max(b[2] for b in norm_boxes), max(b[3] for b in norm_boxes),
            ],
        })
    return result


def _dedupe_spans(spans):
    seen   = set()
    unique = []
    for span in spans:
        key = (span["text"].casefold(), tuple(span["bbox"]))
        if key not in seen:
            seen.add(key)
            unique.append(span)
    return unique


def _fallback_value_spans(ocr_payload):
    lines = _group_words_into_lines(ocr_payload)
    spans = []
    for line in lines:
        text = " ".join(e["text"] for e in line).strip()
        if len(text) < 2:
            continue
        spans.append({
            "label": "VAL",
            "text":  text,
            "score": 0.0,
            "pixel_bbox": [
                min(e["pixel_bbox"][0] for e in line), min(e["pixel_bbox"][1] for e in line),
                max(e["pixel_bbox"][2] for e in line), max(e["pixel_bbox"][3] for e in line),
            ],
            "bbox": [
                min(e["bbox"][0] for e in line), min(e["bbox"][1] for e in line),
                max(e["bbox"][2] for e in line), max(e["bbox"][3] for e in line),
            ],
        })
    return spans


def _group_words_into_lines(ocr_payload):
    words       = ocr_payload["words"]
    pixel_boxes = ocr_payload["pixel_bboxes"]
    norm_boxes  = ocr_payload["bboxes"]

    items = []
    for idx, word in enumerate(words):
        cleaned = str(word).strip()
        if not cleaned:
            continue
        box = pixel_boxes[idx]
        items.append({
            "bbox":       norm_boxes[idx],
            "pixel_bbox": box,
            "text":       cleaned,
            "x1":         box[0], "x2": box[2],
            "y1":         box[1], "y2": box[3],
            "y_center":   (box[1] + box[3]) / 2.0,
            "height":     max(1, box[3] - box[1]),
        })

    items.sort(key=lambda it: (it["y_center"], it["x1"]))

    lines          = []
    current        = []
    current_center = None
    current_height = None

    for item in items:
        if not current:
            current        = [item]
            current_center = item["y_center"]
            current_height = item["height"]
            continue

        tolerance = max(12.0, float(current_height or 0) * 0.8)
        if abs(item["y_center"] - float(current_center or 0.0)) <= tolerance:
            current.append(item)
            current_center = sum(e["y_center"] for e in current) / len(current)
            current_height = max(e["height"] for e in current)
        else:
            lines.append(current)
            current        = [item]
            current_center = item["y_center"]
            current_height = item["height"]

    if current:
        lines.append(current)

    for line in lines:
        line.sort(key=lambda it: it["x1"])

    return lines


def _rule_based_kvp_fallback(ocr_payload):
    lines = _group_words_into_lines(ocr_payload)

    def _line_pixel_bbox(line):
        return [
            min(e["pixel_bbox"][0] for e in line), min(e["pixel_bbox"][1] for e in line),
            max(e["pixel_bbox"][2] for e in line), max(e["pixel_bbox"][3] for e in line),
        ]

    def _words_pixel_bbox(words_in_line):
        return [
            min(e["pixel_bbox"][0] for e in words_in_line),
            min(e["pixel_bbox"][1] for e in words_in_line),
            max(e["pixel_bbox"][2] for e in words_in_line),
            max(e["pixel_bbox"][3] for e in words_in_line),
        ]

    pairs       = []
    last_key    = None
    last_key_bb = None

    for line in lines:
        text = " ".join(e["text"] for e in line).strip()
        if len(text) < 2:
            continue

        first_word = line[0]["text"].strip()
        is_key = first_word.endswith(":") or (
            first_word.isupper() and len(first_word) <= 20
        )

        if is_key and len(line) >= 2:
            key_text    = first_word
            key_bb      = line[0]["pixel_bbox"]
            val_words   = line[1:]
            val_text    = " ".join(e["text"] for e in val_words).strip()
            val_bb      = _words_pixel_bbox(val_words) if val_words else None
            last_key    = key_text
            last_key_bb = key_bb
            pairs.append({
                "key":        key_text,
                "value":      val_text,
                "key_bbox":   key_bb,
                "value_bbox": val_bb,
                "score":      0.0,
                "source":     "rule_based_fallback",
            })
        elif is_key and len(line) == 1:
            last_key    = first_word
            last_key_bb = line[0]["pixel_bbox"]
            pairs.append({
                "key":        first_word,
                "value":      "",
                "key_bbox":   last_key_bb,
                "value_bbox": None,
                "score":      0.0,
                "source":     "rule_based_fallback",
            })
        else:
            pairs.append({
                "key":        last_key or "",
                "value":      text,
                "key_bbox":   last_key_bb,
                "value_bbox": _line_pixel_bbox(line),
                "score":      0.0,
                "source":     "rule_based_fallback",
            })

    if len(pairs) > 15:
        return []
    return pairs


def _bbox_centroid(bbox):
    return (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0


def _box_center_xy(box):
    return (float(box[0] + box[2]) / 2.0, float(box[1] + box[3]) / 2.0)


def _box_height(box):
    return max(1.0, float(box[3] - box[1]))


def _vertical_overlap_ratio(a, b):
    overlap = max(0.0, min(float(a[3]), float(b[3])) - max(float(a[1]), float(b[1])))
    return overlap / max(1.0, min(_box_height(a), _box_height(b)))


def _nearest_key_distance(value_span, key_span):
    value_box = value_span["pixel_bbox"]
    key_box   = key_span["pixel_bbox"]

    value_cx, value_cy = _box_center_xy(value_box)
    key_cx,   key_cy   = _box_center_xy(key_box)
    y_delta            = abs(value_cy - key_cy)
    max_height         = max(_box_height(value_box), _box_height(key_box))

    same_line = (
        _vertical_overlap_ratio(value_box, key_box) >= 0.35
        or y_delta <= max(12.0, max_height * 0.75)
    )

    if key_box[2] <= value_box[0]:
        x_gap = float(value_box[0] - key_box[2])
    elif value_box[2] <= key_box[0]:
        x_gap = float(key_box[0] - value_box[2])
    else:
        x_gap = 0.0

    if same_line:
        distance = x_gap + (y_delta * 2.0)
    else:
        x_delta  = abs(value_cx - key_cx)
        distance = ((x_delta ** 2) + (y_delta ** 2)) ** 0.5

    key_is_left = key_cx <= value_cx
    if not key_is_left:
        distance += 250.0
    if not same_line:
        distance += 100.0

    return distance


def _pair_kvps_geo(key_spans, value_spans):
    if not key_spans:
        return []

    pairs = []
    for val in value_spans:
        best_key = min(key_spans, key=lambda k: _nearest_key_distance(val, k))
        distance = _nearest_key_distance(val, best_key)
        pairs.append({
            "key":        best_key["text"],
            "value":      val["text"],
            "key_bbox":   best_key["pixel_bbox"],
            "value_bbox": val["pixel_bbox"],
            "score":      round(1.0 / (1.0 + distance / 100.0), 4),
        })

    return pairs


def build_key_values_response(kvp_pairs):
    return {
        "key_values": [
            {
                "key":       pair["key"],
                "value":     pair["value"],
                "score":     pair["score"],
                "key_bbox":  pair.get("key_bbox"),
                "value_bbox": pair.get("value_bbox"),
            }
            for pair in kvp_pairs
        ]
    }


def _centroid_dist(bbox_a, bbox_b):
    cx_a, cy_a = _bbox_centroid(bbox_a)
    cx_b, cy_b = _bbox_centroid(bbox_b)
    return ((cx_a - cx_b) ** 2 + (cy_a - cy_b) ** 2) ** 0.5


def _pair_kvps(key_spans, value_spans):
    sorted_keys = sorted(
        enumerate(key_spans),
        key=lambda t: (t[1]["pixel_bbox"][1], t[1]["pixel_bbox"][0]),
    )
    claimed = set()
    pairs   = []

    for _ki, key in sorted_keys:
        kx1, ky1, kx2, ky2 = key["pixel_bbox"]
        key_height    = max(1, ky2 - ky1)
        key_y_center  = (ky1 + ky2) / 2.0
        row_tolerance = 1.5 * key_height

        same_row = [
            (vi, val, val["pixel_bbox"][0],
             abs((val["pixel_bbox"][1] + val["pixel_bbox"][3]) / 2.0 - key_y_center))
            for vi, val in enumerate(value_spans)
            if vi not in claimed
            and abs((val["pixel_bbox"][1] + val["pixel_bbox"][3]) / 2.0 - key_y_center) <= row_tolerance
            and val["pixel_bbox"][0] > kx2
        ]

        if same_row:
            same_row.sort(key=lambda t: (t[3], t[2]))   # dy first, then x position
            vi, val, _, _ = same_row[0]
            claimed.add(vi)
            pairs.append({
                "key":        key["text"],
                "value":      val["text"],
                "key_bbox":   key["pixel_bbox"],
                "value_bbox": val["pixel_bbox"],
                "score":      round((key["score"] + val["score"]) / 2.0, 4),
                "source":     "primary",
            })
            continue

        key_cx  = (kx1 + kx2) / 2.0
        key_width = max(1, kx2 - kx1)

        below = [
            (vi, val, (val["pixel_bbox"][1] + val["pixel_bbox"][3]) / 2.0 - key_y_center)
            for vi, val in enumerate(value_spans)
            if vi not in claimed
            and (val["pixel_bbox"][1] + val["pixel_bbox"][3]) / 2.0 > key_y_center
            and abs((val["pixel_bbox"][0] + val["pixel_bbox"][2]) / 2.0 - key_cx) <= 1.5 * key_width
        ]

        if below:
            below.sort(key=lambda t: t[2])
            vi, val, _ = below[0]
            claimed.add(vi)
            pairs.append({
                "key":        key["text"],
                "value":      val["text"],
                "key_bbox":   key["pixel_bbox"],
                "value_bbox": val["pixel_bbox"],
                "score":      round((key["score"] + val["score"]) / 2.0, 4),
                "source":     "primary",
            })
            continue

        pairs.append({
            "key":        key["text"],
            "value":      "",
            "key_bbox":   key["pixel_bbox"],
            "value_bbox": None,
            "score":      round(key["score"], 4),
            "source":     "primary",
        })

    if key_spans:
        for vi, val in enumerate(value_spans):
            if vi in claimed:
                continue

            nearest_key = min(
                key_spans,
                key=lambda k: _centroid_dist(k["pixel_bbox"], val["pixel_bbox"]),
            )
            pairs.append({
                "key":        nearest_key["text"],
                "value":      val["text"],
                "key_bbox":   nearest_key["pixel_bbox"],
                "value_bbox": val["pixel_bbox"],
                "score":      round((nearest_key["score"] + val["score"]) / 2.0, 4),
                "source":     "nearest_key_fallback",
            })

    seen_keys = {}   # normalised key text → index into deduped list
    deduped   = []
    for pair in pairs:
        norm = pair["key"].casefold()
        if norm not in seen_keys:
            seen_keys[norm] = len(deduped)
            deduped.append(pair)
        else:
            idx           = seen_keys[norm]
            existing      = deduped[idx]
            pair_has_val  = bool(pair["value"])
            exist_has_val = bool(existing["value"])
            if pair_has_val and not exist_has_val:
                # Candidate has a value; existing does not — always prefer candidate.
                deduped[idx] = pair
            elif pair_has_val == exist_has_val and pair["score"] > existing["score"]:
                # Same value-presence; use score as tiebreaker.
                deduped[idx] = pair

    return deduped


def _analyze_document(image_bytes, filename, content_type, make_overlay=True):
    try:
        pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    ocr_response = _call_odc_documentai(image_bytes, filename, content_type, ODC_DOCUMENTAI_URL)
    ocr_payload  = _documentai_to_payload(ocr_response, pil_image.width, pil_image.height)

    labels, scores = _predict_word_labels(pil_image, ocr_payload, model, device, applier)

    output_path = None
    if make_overlay:
        overlay     = _overlay_predictions(pil_image, ocr_payload, labels)
        output_path = OUTPUT_DIR / f"kvp_overlay_{uuid4().hex}.png"
        overlay.save(output_path, format="PNG")

    value_spans = _dedupe_spans(_group_labeled_spans(ocr_payload, labels, scores, "VAL"))
    key_spans   = _dedupe_spans(_group_labeled_spans(ocr_payload, labels, scores, "KEY"))
    if not value_spans:
        value_spans = _dedupe_spans(_fallback_value_spans(ocr_payload))

    kvp_pairs = _pair_kvps_geo(key_spans, value_spans)
    if not kvp_pairs:
        kvp_pairs = _rule_based_kvp_fallback(ocr_payload)

    return {
        "image_size":         {"width": pil_image.width, "height": pil_image.height},
        "ocr_word_count":     ocr_payload["word_count"],
        "overlay_image_path": str(output_path) if output_path else None,
        "detected_keys":      key_spans,
        "detected_values":    value_spans,
        "kvp_pairs":          kvp_pairs,
        "ocr_payload": {
            "word_count":   ocr_payload["word_count"],
            "words":        ocr_payload["words"],
            "bboxes":       ocr_payload["bboxes"],
            "pixel_bboxes": ocr_payload["pixel_bboxes"],
        },
    }


def _load_model_assets():
    active_device    = "cuda" if torch.cuda.is_available() else "cpu"
    active_processor = AutoProcessor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
    active_model     = AutoModelForTokenClassification.from_pretrained(str(MODEL_DIR))
    active_model.eval()
    active_model.to(active_device)
    return active_processor, active_model, active_device


processor, model, device = _load_model_assets()
applier = ProcessorApplier(processor, max_length=512)


def _stream_image(path):
    return StreamingResponse(
        open(path, "rb"),
        media_type="image/png",
        headers={"Content-Disposition": f"inline; filename={Path(path).name}"},
    )


@app.post("/predict", response_class=StreamingResponse)
async def predict(image: UploadFile = File(...)):
    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded image is empty")
    result = _analyze_document(
        image_bytes,
        image.filename or "upload.png",
        image.content_type or "image/png",
    )
    return _stream_image(result["overlay_image_path"])


@app.post("/analyze", response_class=StreamingResponse)
async def analyze(image: UploadFile = File(...)):
    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded image is empty")
    result = _analyze_document(
        image_bytes,
        image.filename or "upload.png",
        image.content_type or "image/png",
    )
    return _stream_image(result["overlay_image_path"])


@app.post("/key-values")
async def key_values(image: UploadFile = File(...)):
    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded image is empty")
    result = _analyze_document(
        image_bytes,
        image.filename or "upload.png",
        image.content_type or "image/png",
        make_overlay=False,
    )
    return JSONResponse(content={
        "key_values": [
            {
                "key":        pair["key"],
                "value":      pair["value"],
                "score":      pair["score"],
                "key_bbox":   pair.get("key_bbox"),
                "value_bbox": pair.get("value_bbox"),
            }
            for pair in result["kvp_pairs"]
        ],
        "ocr_word_count": result["ocr_word_count"],
        "image_size":     result["image_size"],
    })


@app.post("/json")
async def analyze_json(image: UploadFile = File(...)):
    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded image is empty")
    result = _analyze_document(
        image_bytes,
        image.filename or "upload.png",
        image.content_type or "image/png",
        make_overlay=False,
    )
    result.pop("overlay_image_path", None)
    return JSONResponse(content=result)


@app.get("/health")
async def health():
    try:
        label_count = len(model.config.id2label)
        loaded      = True
    except Exception:
        label_count = 0
        loaded      = False
    return {
        "status":       "ok",
        "model_dir":    str(MODEL_DIR),
        "device":       device,
        "model_loaded": loaded,
        "label_count":  label_count,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
