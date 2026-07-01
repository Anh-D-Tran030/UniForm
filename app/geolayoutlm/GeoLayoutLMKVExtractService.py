import io
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from uuid import uuid4

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import types

import requests
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from omegaconf import OmegaConf
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel
import uvicorn


PROJECT_ROOT = Path(__file__).resolve().parent
GEOLAYOUTLM_DIR = PROJECT_ROOT / "GeoLayoutLM"
DATASET_ROOT = PROJECT_ROOT / "geolayoutlm_data" / "funsd_geo"
WORKSPACE = PROJECT_ROOT / "geolayoutlm_workspace"
CHECKPOINT_DIR = WORKSPACE / "checkpoints"
UPLOAD_DIR = PROJECT_ROOT / "service_uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR = PROJECT_ROOT / "uploaded_imgs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_SIZE = "large"
MODEL_BACKBONE = f"alibaba-damo/geolayoutlm-{MODEL_SIZE}-uncased"
MODEL_CONFIG_JSON = GEOLAYOUTLM_DIR / "configs" / "GeoLayoutLM" / f"GeoLayoutLM_{MODEL_SIZE}_model_config.json"
MODEL_CKPT = GEOLAYOUTLM_DIR / f"geolayoutlm_{MODEL_SIZE}_pretrain.pt"

OCR_SERVICE_URL = os.getenv("ODC_DOCUMENTAI_URL", "http://localhost:8005/documentai/process")
SERVICE_PORT = int(os.getenv("GEOLAYOUTLM_SERVICE_PORT", "8006"))
USE_DOCUMENT_AI_LINES = os.getenv("GEOLAYOUTLM_USE_DOCUMENT_AI_LINES", "1").lower() not in {"0", "false", "no"}

MAX_SEQ_LENGTH = 512
MAX_BLOCK_NUM = 150
IMG_H = 768
IMG_W = 768
CLASS_NAMES = ["O", "HEADER", "QUESTION", "ANSWER"]
BIO_CLASS_NAMES = ["O", "B-HEADER", "I-HEADER", "B-QUESTION", "I-QUESTION", "B-ANSWER", "I-ANSWER"]


def _install_runtime_patches():
    class _DummySummaryWriter:
        def __init__(self, *args, **kwargs):
            pass

        def add_scalar(self, *args, **kwargs):
            pass

        def add_hparams(self, *args, **kwargs):
            pass

        def flush(self):
            pass

        def close(self):
            pass

    def _dummy_hparams(*args, **kwargs):
        return [], [], []

    tensorboard_stub = types.ModuleType("torch.utils.tensorboard")
    tensorboard_stub.__path__ = []
    tensorboard_stub.SummaryWriter = _DummySummaryWriter
    summary_stub = types.ModuleType("torch.utils.tensorboard.summary")
    summary_stub.hparams = _dummy_hparams
    sys.modules["torch.utils.tensorboard"] = tensorboard_stub
    sys.modules["torch.utils.tensorboard.summary"] = summary_stub

    for name in list(sys.modules):
        if name == "pytorch_lightning" or name.startswith("pytorch_lightning."):
            del sys.modules[name]

    import transformers.modeling_utils as modeling_utils
    import transformers.pytorch_utils as pytorch_utils

    for name in [
        "apply_chunking_to_forward",
        "find_pruneable_heads_and_indices",
        "prune_linear_layer",
    ]:
        if not hasattr(modeling_utils, name) and hasattr(pytorch_utils, name):
            setattr(modeling_utils, name, getattr(pytorch_utils, name))


_install_runtime_patches()

for import_path in [PROJECT_ROOT, GEOLAYOUTLM_DIR, GEOLAYOUTLM_DIR / "bros"]:
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from GeoLayoutLM.lightning_modules.data_modules.vie_dataset import VIEDataset
from GeoLayoutLM.lightning_modules.geolayoutlm_vie_module import GeoLayoutLMVIEModule


app = FastAPI(title="GeoLayoutLM KV Extract Service API")


class OCRWord(BaseModel):
    text: str
    bbox: List[int]
    confidence: Optional[float] = None


class OCRPayload(BaseModel):
    image_size: Dict[str, int]
    words: List[str]
    bboxes: List[List[int]]
    confidences: Optional[List[float]] = None
    engine: str = "google_document_ai"


class SingleExampleVIEDataset(VIEDataset):
    def __init__(self, example: Dict, *args, **kwargs):
        self._single_example = example
        super().__init__(*args, **kwargs)

    def _load_examples(self):
        return [self._single_example]


def find_default_checkpoint() -> Path:
    env_path = os.environ.get("GEOLAYOUTLM_CHECKPOINT")
    if env_path:
        path = Path(env_path)
        if path.exists():
            return path
        raise FileNotFoundError(f"GEOLAYOUTLM_CHECKPOINT does not exist: {path}")

    linking_ckpts = sorted(CHECKPOINT_DIR.glob("*f1_linking=*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if linking_ckpts:
        return linking_ckpts[0]

    any_ckpts = sorted(CHECKPOINT_DIR.glob("*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if any_ckpts:
        return any_ckpts[0]

    raise FileNotFoundError(f"No GeoLayoutLM checkpoints found in {CHECKPOINT_DIR}")


def build_cfg(checkpoint_path: Path):
    if not MODEL_CONFIG_JSON.exists():
        raise FileNotFoundError(f"Missing model config JSON: {MODEL_CONFIG_JSON}")
    if not MODEL_CKPT.exists():
        raise FileNotFoundError(f"Missing GeoLayoutLM pretrain checkpoint: {MODEL_CKPT}")
    if not DATASET_ROOT.exists():
        raise FileNotFoundError(f"Missing GeoLayoutLM dataset root: {DATASET_ROOT}")

    cfg = OmegaConf.create({
        "workspace": str(WORKSPACE),
        "dataset": "funsd_plus_arrow_as_funsd_geo",
        "dataset_root_path": str(DATASET_ROOT),
        "task": "ee_el",
        "img_h": IMG_H,
        "img_w": IMG_W,
        "seed": 42,
        "cudnn_deterministic": False,
        "cudnn_benchmark": True,
        "model": {
            "backbone": MODEL_BACKBONE,
            "config_json": str(MODEL_CONFIG_JSON),
            "model_ckpt": str(MODEL_CKPT),
            "head": "vie",
            "use_inner_id": True,
            "max_prob_as_father": True,
            "max_prob_as_father_upperbound": False,
            "n_classes": 7,
        },
        "train": {
            "batch_size": 1,
            "num_samples_per_epoch": 1,
            "max_seq_length": MAX_SEQ_LENGTH,
            "max_block_num": MAX_BLOCK_NUM,
            "max_epochs": 1,
            "use_fp16": torch.cuda.is_available(),
            "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
            "strategy": {"type": ""},
            "clip_gradient_algorithm": "norm",
            "clip_gradient_value": 1.0,
            "num_workers": 0,
            "optimizer": {
                "method": "adamw",
                "params": {"lr": 2e-5, "weight_decay": 0.01, "eps": 1e-8},
                "lr_schedule": {"method": "linear", "params": {"warmup_steps": 200}},
            },
            "val_interval": 1,
        },
        "val": {
            "batch_size": 1,
            "num_workers": 0,
            "limit_val_batches": 1.0,
            "dump_dir": str(WORKSPACE / "service_result"),
            "pretrained_best_type": "linking",
        },
        "pretrained_model_file": str(checkpoint_path),
    })
    cfg.save_weight_dir = str(WORKSPACE / "checkpoints")
    cfg.tensorboard_dir = str(WORKSPACE / "tensorboard_logs")
    return cfg


def load_model_bundle():
    checkpoint_path = find_default_checkpoint()
    cfg = build_cfg(checkpoint_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pl_module = GeoLayoutLMVIEModule(cfg)
    model = pl_module.net.to(device)
    model.eval()
    return {
        "cfg": cfg,
        "device": device,
        "tokenizer": model.tokenizer,
        "model": model,
        "checkpoint_path": checkpoint_path,
    }


MODEL_BUNDLE = load_model_bundle()


def request_ocr_service(image_bytes: bytes, filename: str, mime_type: str = "image/png"):
    files = {
        "image_file": (filename, image_bytes, mime_type or "image/png"),
    }

    try:
        response = requests.post(OCR_SERVICE_URL, files=files, timeout=180)
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"OCR service unreachable: {exc}") from exc

    if response.status_code >= 400:
        raise HTTPException(status_code=response.status_code, detail=f"OCR service failed: {response.text}")

    return response.json()


def _anchor_text(full_text: str, text_anchor: Dict) -> str:
    pieces = []
    for segment in text_anchor.get("textSegments", []):
        start = int(segment.get("startIndex", 0))
        end = int(segment.get("endIndex", 0))
        pieces.append(full_text[start:end])
    return "".join(pieces).strip()


def _poly_to_box(poly: Dict, width: int, height: int) -> List[int]:
    vertices = poly.get("normalizedVertices") or []
    if vertices:
        xs = [float(v.get("x", 0.0)) * width for v in vertices]
        ys = [float(v.get("y", 0.0)) * height for v in vertices]
    else:
        vertices = poly.get("vertices") or []
        xs = [float(v.get("x", 0.0)) for v in vertices]
        ys = [float(v.get("y", 0.0)) for v in vertices]
    if not xs or not ys:
        return [0, 0, max(1, width), max(1, height)]
    x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
    return [
        int(max(0, min(width, round(x0)))),
        int(max(0, min(height, round(y0)))),
        int(max(0, min(width, round(x1)))),
        int(max(0, min(height, round(y1)))),
    ]


def parse_google_document_ai(google_response: Dict, fallback_size: Tuple[int, int]) -> Dict:
    document = google_response.get("document", google_response)
    full_text = document.get("text", "")
    fallback_width, fallback_height = fallback_size
    words: List[str] = []
    bboxes: List[List[int]] = []
    confidences: List[float] = []
    used_lines = False

    for page in document.get("pages", []):
        dimension = page.get("dimension", {})
        page_width = int(round(float(dimension.get("width") or fallback_width)))
        page_height = int(round(float(dimension.get("height") or fallback_height)))
        page_lines = page.get("lines") or []
        use_lines = USE_DOCUMENT_AI_LINES and bool(page_lines)
        layout_items = page_lines if use_lines else page.get("tokens", [])
        used_lines = used_lines or use_lines

        for item in layout_items:
            layout = item.get("layout", {})
            text = _anchor_text(full_text, layout.get("textAnchor", {}))
            if not text:
                continue
            text = " ".join(text.split())
            if not text:
                continue
            box = _poly_to_box(layout.get("boundingPoly", {}), page_width, page_height)
            words.append(text)
            bboxes.append(box)
            confidences.append(float(layout.get("confidence", item.get("confidence", 0.0)) or 0.0))

    if not words:
        words = ["[EMPTY]"]
        bboxes = [[0, 0, 1, 1]]
        confidences = [0.0]

    return {
        "image_size": {"width": fallback_width, "height": fallback_height},
        "word_count": len(words),
        "words": words,
        "bboxes": bboxes,
        "confidences": confidences,
        "engine": "google_document_ai_lines" if used_lines else "google_document_ai",
    }


def normalize_ocr_payload(ocr_payload: Dict, image_size: Tuple[int, int]) -> Dict:
    width, height = image_size
    if "document" in ocr_payload or "pages" in ocr_payload:
        return parse_google_document_ai(ocr_payload, image_size)

    words = ocr_payload.get("words", [])
    bboxes = ocr_payload.get("bboxes", [])
    confidences = ocr_payload.get("confidences") or [0.0] * len(words)
    if len(words) != len(bboxes):
        raise HTTPException(status_code=400, detail="OCR payload words and bboxes lengths do not match")

    clean_words = []
    clean_bboxes = []
    clean_confidences = []
    for text, box, confidence in zip(words, bboxes, confidences):
        text = str(text).strip()
        if not text:
            continue
        x0, y0, x1, y1 = box
        clean_words.append(text)
        clean_bboxes.append([
            int(max(0, min(width, round(x0)))),
            int(max(0, min(height, round(y0)))),
            int(max(0, min(width, round(x1)))),
            int(max(0, min(height, round(y1)))),
        ])
        clean_confidences.append(float(confidence or 0.0))

    if not clean_words:
        clean_words = ["[EMPTY]"]
        clean_bboxes = [[0, 0, 1, 1]]
        clean_confidences = [0.0]

    return {
        "image_size": {"width": width, "height": height},
        "word_count": len(clean_words),
        "words": clean_words,
        "bboxes": clean_bboxes,
        "confidences": clean_confidences,
        "engine": ocr_payload.get("engine", "provided_google_ocr"),
    }


def box_to_points(box: List[int]) -> List[List[int]]:
    x0, y0, x1, y1 = box
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def build_geo_example(image_path: Path, image: Image.Image, ocr_payload: Dict, tokenizer) -> Dict:
    image_path = image_path.resolve()
    width, height = image.size
    words_json = []
    first_token_idx_list = []
    block_boxes = []
    token_cursor = 1

    for text, box in zip(ocr_payload["words"], ocr_payload["bboxes"]):
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if not token_ids:
            token_ids = [tokenizer.unk_token_id]
        words_json.append({
            "text": text,
            "tokens": token_ids,
            "boundingBox": box_to_points(box),
        })
        first_token_idx_list.append(token_cursor)
        block_boxes.append(box)
        token_cursor += len(token_ids)
        if len(first_token_idx_list) >= MAX_BLOCK_NUM:
            break

    word_count = len(words_json)
    return {
        "words": words_json,
        "blocks": {
            "first_token_idx_list": first_token_idx_list,
            "boxes": block_boxes,
        },
        "parse": {
            "class": {"O": [[idx] for idx in range(word_count)]},
            "relations": [],
        },
        "meta": {
            "image_path": str(image_path),
            "imageSize": {"width": width, "height": height},
            "voca": "bert-base-uncased",
        },
    }


def make_batch(image_path: Path, image: Image.Image, ocr_payload: Dict):
    bundle = MODEL_BUNDLE
    example = build_geo_example(image_path, image, ocr_payload, bundle["tokenizer"])
    dataset = SingleExampleVIEDataset(
        example,
        "funsd_plus_arrow_as_funsd_geo",
        "ee_el",
        "geolayoutlm",
        "vie",
        str(DATASET_ROOT),
        bundle["tokenizer"],
        MAX_SEQ_LENGTH,
        MAX_BLOCK_NUM,
        IMG_H,
        IMG_W,
        mode="infer",
    )
    item = dataset[0]
    batch = {}
    for key, value in item.items():
        if isinstance(value, torch.Tensor):
            if not torch.is_floating_point(value) and value.dtype != torch.bool:
                value = value.long()
            batch[key] = value.unsqueeze(0).to(bundle["device"])
        else:
            batch[key] = [value]
    return batch, example


def union_box(boxes: List[List[int]]) -> List[int]:
    return [
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    ]


def bio_type(label: str) -> Tuple[str, str]:
    if label == "O":
        return "O", "O"
    prefix, entity_type = label.split("-", 1)
    return prefix, entity_type


def build_entity_spans(labels: List[str], words: List[str], boxes: List[List[int]], probs: List[float]) -> List[Dict]:
    spans = []
    current = None
    for idx, label in enumerate(labels):
        prefix, entity_type = bio_type(label)
        if entity_type == "O":
            if current is not None:
                spans.append(current)
                current = None
            continue

        should_start = current is None or prefix == "B" or current["type"] != entity_type
        if should_start:
            if current is not None:
                spans.append(current)
            current = {"type": entity_type, "word_indices": [], "label_probs": []}

        current["word_indices"].append(idx)
        current["label_probs"].append(probs[idx])

    if current is not None:
        spans.append(current)

    for span_idx, span in enumerate(spans):
        span_boxes = [boxes[idx] for idx in span["word_indices"]]
        span["id"] = f"e{span_idx}"
        span["text"] = " ".join(words[idx] for idx in span["word_indices"])
        span["bbox"] = union_box(span_boxes)
        span["confidence"] = sum(span["label_probs"]) / max(1, len(span["label_probs"]))
        del span["label_probs"]
    return spans


def _box_center_xy(box: List[int]) -> Tuple[float, float]:
    return (float(box[0] + box[2]) / 2.0, float(box[1] + box[3]) / 2.0)


def _box_height(box: List[int]) -> float:
    return max(1.0, float(box[3] - box[1]))


def _vertical_overlap_ratio(a: List[int], b: List[int]) -> float:
    overlap = max(0.0, min(float(a[3]), float(b[3])) - max(float(a[1]), float(b[1])))
    return overlap / max(1.0, min(_box_height(a), _box_height(b)))


def _nearest_key_distance(value_entity: Dict, key_entity: Dict) -> float:
    value_box = value_entity["bbox"]
    key_box = key_entity["bbox"]
    value_cx, value_cy = _box_center_xy(value_box)
    key_cx, key_cy = _box_center_xy(key_box)
    y_delta = abs(value_cy - key_cy)
    same_line = (
        _vertical_overlap_ratio(value_box, key_box) >= 0.35
        or y_delta <= max(12.0, max(_box_height(value_box), _box_height(key_box)) * 0.75)
    )

    key_is_left = key_cx <= value_cx
    if key_box[2] <= value_box[0]:
        x_gap = float(value_box[0] - key_box[2])
    elif value_box[2] <= key_box[0]:
        x_gap = float(key_box[0] - value_box[2])
    else:
        x_gap = 0.0

    if same_line:
        distance = x_gap + (y_delta * 2.0)
    else:
        x_delta = abs(value_cx - key_cx)
        distance = ((x_delta ** 2) + (y_delta ** 2)) ** 0.5

    if not key_is_left:
        distance += 250.0
    if not same_line:
        distance += 100.0
    return distance


def build_nearest_key_values(entities: List[Dict]) -> List[Dict]:
    value_entities = [entity for entity in entities if entity["type"] == "ANSWER"]
    key_entities = [entity for entity in entities if entity["type"] == "QUESTION"]
    key_values = []

    for value_entity in value_entities:
        if not key_entities:
            break

        best_key = min(
            key_entities,
            key=lambda key_entity: _nearest_key_distance(value_entity, key_entity),
        )
        distance = _nearest_key_distance(value_entity, best_key)
        key_values.append({
            "key": best_key["text"],
            "value": value_entity["text"],
            "key_entity_id": best_key["id"],
            "value_entity_id": value_entity["id"],
            "score": round(1.0 / (1.0 + distance / 100.0), 4),
            "key_bbox": best_key["bbox"],
            "value_bbox": value_entity["bbox"],
            "linking_method": "nearest_key_geometry",
        })

    return key_values


def decode_predictions(batch: Dict, ocr_payload: Dict, head_outputs: Dict, re_threshold: float):
    first_token_idxes = batch["first_token_idxes"][0].detach().cpu().tolist()
    block_mask = batch["block_mask"][0].detach().cpu().tolist()
    valid_blocks = min(sum(int(x) for x in block_mask), len(ocr_payload["words"]))

    label_probs = torch.softmax(head_outputs["logits4labeling"][0], dim=-1).detach().cpu()
    word_labels = []
    word_label_scores = []
    for block_idx in range(valid_blocks):
        token_idx = int(first_token_idxes[block_idx])
        label_id = int(label_probs[token_idx].argmax().item())
        word_labels.append(BIO_CLASS_NAMES[label_id])
        word_label_scores.append(float(label_probs[token_idx, label_id].item()))

    words = ocr_payload["words"][:valid_blocks]
    boxes = ocr_payload["bboxes"][:valid_blocks]
    entities = build_entity_spans(word_labels, words, boxes, word_label_scores)
    key_values = build_nearest_key_values(entities)

    return {
        "tokens": [
            {
                "text": words[idx],
                "bbox": boxes[idx],
                "label": word_labels[idx],
                "confidence": word_label_scores[idx],
            }
            for idx in range(valid_blocks)
        ],
        "entities": entities,
        "key_values": key_values,
    }


def run_inference(image_path: Path, ocr_payload: Dict, re_threshold: float):
    image_path = image_path.resolve()
    image = Image.open(image_path).convert("RGB")
    batch, _ = make_batch(image_path, image, ocr_payload)
    model = MODEL_BUNDLE["model"]
    with torch.inference_mode():
        head_outputs, _ = model(batch)
    return decode_predictions(batch, ocr_payload, head_outputs, re_threshold)


def save_upload(content: bytes, filename: Optional[str]) -> Path:
    ext = Path(filename or "upload.png").suffix or ".png"
    image_path = UPLOAD_DIR / f"{uuid4().hex}{ext}"
    image_path.write_bytes(content)
    return image_path


def _draw_label(draw: ImageDraw.ImageDraw, box: List[int], text: str, color: str, font):
    text_x = box[0]
    text_y = max(0, box[1] - 13)
    try:
        text_box = draw.textbbox((text_x, text_y), text, font=font)
        draw.rectangle(text_box, fill="white")
    except Exception:
        pass
    draw.text((text_x, text_y), text, fill=color, font=font)


def overlay_predictions(image_path: Path, prediction: Dict) -> Path:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    color_map = {
        "HEADER": "#7c3aed",
        "KEY": "#1e40ff",
        "VALUE": "#11883a",
    }
    label_map = {
        "HEADER": "HEADER",
        "QUESTION": "KEY",
        "ANSWER": "VALUE",
    }

    for entity in prediction.get("entities", []):
        entity_type = entity.get("type", "")
        overlay_label = label_map.get(entity_type)
        if overlay_label is None:
            continue
        box = [int(value) for value in entity.get("bbox", [0, 0, 1, 1])]
        color = color_map.get(overlay_label, "#333333")
        draw.rectangle(box, outline=color, width=3)
        _draw_label(draw, box, overlay_label, color, font)

    output_path = OUTPUT_DIR / f"geolayoutlm_overlay_{uuid4().hex}.png"
    image.save(output_path, format="PNG")
    return output_path


def analyze_image_bytes(
    content: bytes,
    filename: Optional[str],
    content_type: Optional[str],
    ocr_json: Optional[str],
    re_threshold: float,
    make_overlay: bool = True,
) -> Dict:
    image_path = save_upload(content, filename)
    try:
        with Image.open(io.BytesIO(content)) as pil_image:
            image_size = pil_image.convert("RGB").size
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid image file") from exc

    if ocr_json:
        try:
            google_response = json.loads(ocr_json)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail=f"Invalid ocr_json: {exc}") from exc
    else:
        google_response = request_ocr_service(
            image_bytes=content,
            filename=filename or "upload.png",
            mime_type=content_type or "image/png",
        )

    ocr_payload = normalize_ocr_payload(google_response, image_size)
    prediction = run_inference(image_path, ocr_payload, re_threshold)
    analysis = {
        "ok": True,
        "image_path": str(image_path),
        "ocr": {
            "engine": ocr_payload["engine"],
            "word_count": ocr_payload["word_count"],
            "image_size": ocr_payload["image_size"],
        },
        "checkpoint": str(MODEL_BUNDLE["checkpoint_path"]),
        **prediction,
    }
    if make_overlay:
        overlay_path = overlay_predictions(image_path, prediction)
        analysis["overlay_image_path"] = str(overlay_path)
    return analysis


@app.get("/health")
async def health():
    return {
        "ok": True,
        "device": str(MODEL_BUNDLE["device"]),
        "checkpoint": str(MODEL_BUNDLE["checkpoint_path"]),
        "model_size": MODEL_SIZE,
        "ocr_service_url": OCR_SERVICE_URL,
    }


@app.get("/")
async def index():
    return {
        "service": "GeoLayoutLM KV Extract Service",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict",
        "key_values": "/key-values",
    }


@app.post("/predict")
async def predict(
    image: UploadFile = File(...),
    ocr_json: Optional[str] = Form(None),
    re_threshold: float = Form(0.5),
):
    content = await image.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded image is empty")

    analysis = analyze_image_bytes(
        content=content,
        filename=image.filename,
        content_type=image.content_type,
        ocr_json=ocr_json,
        re_threshold=re_threshold,
    )
    output_path = Path(analysis["overlay_image_path"])
    return FileResponse(str(output_path), media_type="image/png", filename=output_path.name)


@app.post("/analyze")
async def analyze(
    image: UploadFile = File(...),
    ocr_json: Optional[str] = Form(None),
    re_threshold: float = Form(0.5),
):
    content = await image.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded image is empty")

    return analyze_image_bytes(
        content=content,
        filename=image.filename,
        content_type=image.content_type,
        ocr_json=ocr_json,
        re_threshold=re_threshold,
    )


@app.post("/key-values")
async def key_values(
    image: UploadFile = File(...),
    ocr_json: Optional[str] = Form(None),
    re_threshold: float = Form(0.5),
):
    content = await image.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded image is empty")

    analysis = analyze_image_bytes(
        content=content,
        filename=image.filename,
        content_type=image.content_type,
        ocr_json=ocr_json,
        re_threshold=re_threshold,
        make_overlay=False,
    )
    return {
        "key_values": [
            {
                "key": item["key"],
                "value": item["value"],
                "score": item["score"],
                "key_bbox": item["key_bbox"],
                "value_bbox": item["value_bbox"],
            }
            for item in analysis.get("key_values", [])
        ]
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=SERVICE_PORT)
