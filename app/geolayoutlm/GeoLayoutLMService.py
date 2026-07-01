import base64
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
from fastapi.responses import HTMLResponse
from omegaconf import OmegaConf
from PIL import Image
from pydantic import BaseModel
import uvicorn


PROJECT_ROOT = Path(__file__).resolve().parent
GEOLAYOUTLM_DIR = PROJECT_ROOT / "GeoLayoutLM"
DATASET_ROOT = PROJECT_ROOT / "geolayoutlm_data" / "funsd_geo"
WORKSPACE = PROJECT_ROOT / "geolayoutlm_workspace"
CHECKPOINT_DIR = WORKSPACE / "checkpoints"
UPLOAD_DIR = PROJECT_ROOT / "service_uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

MODEL_SIZE = "large"
MODEL_BACKBONE = f"alibaba-damo/geolayoutlm-{MODEL_SIZE}-uncased"
MODEL_CONFIG_JSON = GEOLAYOUTLM_DIR / "configs" / "GeoLayoutLM" / f"GeoLayoutLM_{MODEL_SIZE}_model_config.json"
MODEL_CKPT = GEOLAYOUTLM_DIR / f"geolayoutlm_{MODEL_SIZE}_pretrain.pt"

DOCUMENT_AI_ENDPOINT = (
    "https://us-documentai.googleapis.com/v1/projects/971280404964/"
    "locations/us/processors/18e2688f80888cc9:process"
)
GCLOUD_CMD = Path(
    r"C:\Users\thanh\AppData\Local\Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd"
)

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


app = FastAPI(title="GeoLayoutLM KVP Service API")


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


def get_gcloud_token():
    import subprocess

    result = subprocess.run(
        [str(GCLOUD_CMD), "auth", "print-access-token"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def request_document_ai(image_bytes: bytes, mime_type: str = "image/png", endpoint: str = DOCUMENT_AI_ENDPOINT):
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

    for page in document.get("pages", []):
        dimension = page.get("dimension", {})
        page_width = int(round(float(dimension.get("width") or fallback_width)))
        page_height = int(round(float(dimension.get("height") or fallback_height)))

        for token in page.get("tokens", []):
            layout = token.get("layout", {})
            text = _anchor_text(full_text, layout.get("textAnchor", {}))
            if not text:
                continue
            text = " ".join(text.split())
            if not text:
                continue
            box = _poly_to_box(layout.get("boundingPoly", {}), page_width, page_height)
            words.append(text)
            bboxes.append(box)
            confidences.append(float(layout.get("confidence", token.get("confidence", 0.0)) or 0.0))

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
        "engine": "google_document_ai",
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

    word_to_entity = {}
    for entity in entities:
        for word_idx in entity["word_indices"]:
            word_to_entity[word_idx] = entity

    link_probs = torch.sigmoid(head_outputs["logits4linking_list"][-1][0]).detach().cpu()
    key_values = []
    answer_entities = [entity for entity in entities if entity["type"] == "ANSWER"]
    question_entities = [entity for entity in entities if entity["type"] == "QUESTION"]

    for answer in answer_entities:
        best = None
        answer_blocks = answer["word_indices"]
        for question in question_entities:
            question_blocks = question["word_indices"]
            scores = [
                float(link_probs[answer_block, question_block].item())
                for answer_block in answer_blocks
                for question_block in question_blocks
                if answer_block < link_probs.shape[0] and question_block < link_probs.shape[1]
            ]
            if not scores:
                continue
            score = max(scores)
            if best is None or score > best["score"]:
                best = {"question": question, "answer": answer, "score": score}
        if best is not None and best["score"] >= re_threshold:
            key_values.append({
                "key": best["question"]["text"],
                "value": best["answer"]["text"],
                "key_entity_id": best["question"]["id"],
                "value_entity_id": best["answer"]["id"],
                "score": best["score"],
                "key_bbox": best["question"]["bbox"],
                "value_bbox": best["answer"]["bbox"],
            })

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


@app.get("/health")
async def health():
    return {
        "ok": True,
        "device": str(MODEL_BUNDLE["device"]),
        "checkpoint": str(MODEL_BUNDLE["checkpoint_path"]),
        "model_size": MODEL_SIZE,
    }


@app.get("/", response_class=HTMLResponse)
async def index():
    checkpoint = MODEL_BUNDLE["checkpoint_path"]
    device = MODEL_BUNDLE["device"]
    return f"""
    <!doctype html>
    <html>
      <head>
        <meta charset="utf-8" />
        <title>GeoLayoutLM KVP Service</title>
        <style>
          body {{
            margin: 0;
            font-family: Arial, sans-serif;
            background: #f6f7f9;
            color: #162033;
          }}
          main {{
            max-width: 920px;
            margin: 40px auto;
            padding: 0 20px;
          }}
          h1 {{
            margin: 0 0 8px;
            font-size: 28px;
          }}
          .panel {{
            background: white;
            border: 1px solid #d9dee8;
            border-radius: 8px;
            padding: 20px;
            margin-top: 18px;
          }}
          label {{
            display: block;
            font-weight: 700;
            margin: 14px 0 6px;
          }}
          input, textarea, button {{
            font: inherit;
          }}
          input[type="file"], input[type="number"], textarea {{
            width: 100%;
            box-sizing: border-box;
          }}
          textarea {{
            min-height: 160px;
            resize: vertical;
          }}
          button {{
            margin-top: 16px;
            padding: 10px 14px;
            border: 0;
            border-radius: 6px;
            background: #174ea6;
            color: white;
            cursor: pointer;
          }}
          button:disabled {{
            background: #7c8aa3;
            cursor: wait;
          }}
          code {{
            word-break: break-all;
          }}
          pre {{
            white-space: pre-wrap;
            overflow-wrap: anywhere;
            background: #101828;
            color: #e7eefc;
            border-radius: 8px;
            padding: 16px;
            min-height: 120px;
          }}
          .hint {{
            color: #526070;
            font-size: 14px;
          }}
        </style>
      </head>
      <body>
        <main>
          <h1>GeoLayoutLM KVP Service</h1>
          <div>Device: <code>{device}</code></div>
          <div>Checkpoint: <code>{checkpoint}</code></div>

          <section class="panel">
            <h2>Infer</h2>
            <form id="infer-form">
              <label>Image</label>
              <input name="image_file" type="file" accept="image/*" required />

              <label>Google OCR JSON (optional)</label>
              <textarea name="ocr_json" placeholder="Leave blank to call Google Document AI first"></textarea>
              <div class="hint">If this is blank, the service calls Google Document AI before model inference.</div>

              <label>RE threshold</label>
              <input name="re_threshold" type="number" min="0" max="1" step="0.01" value="0.5" />

              <button id="infer-button" type="submit">Run Inference</button>
            </form>
          </section>

          <section class="panel">
            <h2>Result</h2>
            <pre id="result">Ready.</pre>
          </section>

          <section class="panel">
            <h2>API</h2>
            <div><a href="/docs">Open FastAPI docs</a></div>
            <div><a href="/health">Health check</a></div>
          </section>
        </main>
        <script>
          const form = document.getElementById("infer-form");
          const button = document.getElementById("infer-button");
          const result = document.getElementById("result");

          form.addEventListener("submit", async (event) => {{
            event.preventDefault();
            button.disabled = true;
            result.textContent = "Running OCR + GeoLayoutLM inference... this can take a while on the first request.";

            try {{
              const response = await fetch("/infer", {{
                method: "POST",
                body: new FormData(form),
              }});
              const text = await response.text();
              let payload;
              try {{
                payload = JSON.parse(text);
              }} catch {{
                payload = text;
              }}

              if (!response.ok) {{
                result.textContent = "ERROR " + response.status + "\\n" + JSON.stringify(payload, null, 2);
              }} else {{
                result.textContent = JSON.stringify(payload, null, 2);
              }}
            }} catch (error) {{
              result.textContent = "REQUEST FAILED\\n" + error;
            }} finally {{
              button.disabled = false;
            }}
          }});
        </script>
      </body>
    </html>
    """


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


@app.post("/infer")
async def infer(
    image_file: UploadFile = File(...),
    ocr_json: Optional[str] = Form(None),
    endpoint: str = Form(DOCUMENT_AI_ENDPOINT),
    re_threshold: float = Form(0.5),
):
    content = await image_file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded image is empty")

    image_path = save_upload(content, image_file.filename)
    with Image.open(io.BytesIO(content)) as image:
        image_size = image.convert("RGB").size

    if ocr_json:
        try:
            google_response = json.loads(ocr_json)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail=f"Invalid ocr_json: {exc}") from exc
    else:
        google_response = request_document_ai(
            content,
            mime_type=image_file.content_type or "image/png",
            endpoint=endpoint,
        )

    ocr_payload = normalize_ocr_payload(google_response, image_size)
    prediction = run_inference(image_path, ocr_payload, re_threshold)
    return {
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


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8006)
