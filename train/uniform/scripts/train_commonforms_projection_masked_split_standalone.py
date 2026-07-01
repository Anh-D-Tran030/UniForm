from __future__ import annotations

import csv
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pytesseract
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps
from torch.utils.data import BatchSampler, DataLoader, Dataset
from transformers import (
    AutoProcessor,
    LayoutLMv3ImageProcessor,
    LayoutLMv3Model,
    LayoutLMv3Processor,
    LayoutLMv3TokenizerFast,
    get_linear_schedule_with_warmup,
)


MODEL_NAME = "microsoft/layoutlmv3-base"
SPLIT_OCR_ROOT = Path(r"A:\RealForm\processed\synthetic_fill_images_10k_folders_degraded_ocr_split_9400_300_300_augmented")
DEGRADED_ROOT = Path(r"A:\RealForm\processed\synthetic_fill_images_10k_folders_degraded")
ORIGINAL_ROOT = Path(r"A:\RealForm\processed\CommonFormsEnglish\images")
ORIGINAL_OCR_ROOT = Path(r"A:\RealForm\processed\CommonFormsEnglish_original_ocr_cache_projection")
OUTPUT_DIR = Path(r"A:\RealForm\outputs\layoutlmv3_projection_commonforms_masked_split_standalone")
TESSERACT_CMD = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")

EPOCHS = 10
BATCH_SIZE = 8
EVAL_BATCH_SIZE = 8
MAX_LENGTH = 512
POOLING = "cls"
PROJECTION_DIM = 128
CLASSES_PER_BATCH = 4
SAMPLES_PER_CLASS = 2
TRAIN_BACKBONE_LR = 2e-5
HEAD_LR = 1e-4
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
SUPCON_TEMPERATURE = 0.07
ARCFACE_MARGIN = 0.2
ARCFACE_SCALE = 30.0
ARCFACE_WEIGHT = 0.2
SUPCON_WEIGHT = 0.8
UNFREEZE_LAST_N = 4
SEEN_VAL_FRACTION = 0.1
TRAIN_RETRIEVAL_SAMPLE_COUNT = 285
GRADIENT_CLIP_NORM = 1.0
PSM = 11
OEM = 3
MIN_CONFIDENCE = 30.0
SEED = 42
IMAGE_GLOB = "fill_*.png"


@dataclass(frozen=True)
class DocumentSample:
    subset: str
    stem: str
    class_label: str
    image_path: Path
    record_id: int
    words: list[str]
    boxes: list[list[int]]
    is_original: bool = False


@dataclass(frozen=True)
class SplitSelection:
    stem: str
    query_dir: Path
    original_image_path: Path


@dataclass(frozen=True)
class TemplateExample:
    document: DocumentSample
    class_label: str
    train_label_id: int


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict[str, Any] | list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def write_history_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clamp(value: float, lower: int = 0, upper: int = 1000) -> int:
    return int(max(lower, min(upper, round(value))))


def normalize_box(box: list[int], width: int, height: int) -> list[int]:
    x1, y1, x2, y2 = box
    return [
        clamp(1000 * x1 / max(width, 1)),
        clamp(1000 * y1 / max(height, 1)),
        clamp(1000 * x2 / max(width, 1)),
        clamp(1000 * y2 / max(height, 1)),
    ]


def configure_tesseract() -> None:
    if not TESSERACT_CMD.exists():
        raise FileNotFoundError(f"Tesseract was not found at {TESSERACT_CMD}")
    pytesseract.pytesseract.tesseract_cmd = str(TESSERACT_CMD)


def run_tesseract(image_path: Path) -> dict[str, Any]:
    with Image.open(image_path) as image:
        rgb = image.convert("RGB")
        width, height = rgb.size
        scale_factor = 2.0
        prepared = ImageOps.autocontrast(rgb.convert("L")).resize(
            (int(width * scale_factor), int(height * scale_factor)),
            resample=Image.Resampling.BICUBIC,
        )
        data = pytesseract.image_to_data(
            prepared,
            output_type=pytesseract.Output.DICT,
            config=f"--psm {PSM} --oem {OEM}",
            lang="eng",
        )

    words: list[str] = []
    boxes: list[list[int]] = []
    for i in range(len(data.get("text", []))):
        text = str(data["text"][i]).strip()
        if not text:
            continue
        try:
            confidence = float(data["conf"][i])
        except (TypeError, ValueError):
            confidence = -1.0
        if confidence < MIN_CONFIDENCE:
            continue
        left = int(round(int(data["left"][i]) / scale_factor))
        top = int(round(int(data["top"][i]) / scale_factor))
        box_width = max(1, int(round(int(data["width"][i]) / scale_factor)))
        box_height = max(1, int(round(int(data["height"][i]) / scale_factor)))
        words.append(text)
        boxes.append([left, top, left + box_width, top + box_height])

    if not words:
        words = ["[EMPTY]"]
        boxes = [[0, 0, 1, 1]]

    return {
        "image_size": {"width": width, "height": height},
        "words": words,
        "boxes": boxes,
    }


def original_cache_path(stem: str) -> Path:
    return ORIGINAL_OCR_ROOT / f"{stem}.ocr.json"


def build_original_image_index() -> dict[str, Path]:
    index: dict[str, Path] = {}
    for image_path in sorted([path for path in ORIGINAL_ROOT.rglob("*") if path.is_file()], key=lambda p: str(p).lower()):
        index.setdefault(image_path.stem, image_path)
    return index


def count_matching_files(path: Path, pattern: str) -> int:
    return len(list(path.glob(pattern))) if path.exists() else 0


def discover_split_selections(split_name: str, original_index: dict[str, Path]) -> list[SplitSelection]:
    selections: list[SplitSelection] = []
    split_dir = SPLIT_OCR_ROOT / split_name
    if not split_dir.exists():
        return selections
    for template_dir in sorted([path for path in split_dir.iterdir() if path.is_dir()], key=lambda p: p.name.lower()):
        if count_matching_files(template_dir, "*.ocr.json") != 10:
            continue
        query_dir = DEGRADED_ROOT / template_dir.name
        if count_matching_files(query_dir, IMAGE_GLOB) != 10:
            continue
        original_image_path = original_index.get(template_dir.name)
        if original_image_path is None:
            continue
        selections.append(SplitSelection(template_dir.name, query_dir, original_image_path))
    return selections


def ensure_original_ocr_cache(selections: list[SplitSelection]) -> None:
    configure_tesseract()
    ORIGINAL_OCR_ROOT.mkdir(parents=True, exist_ok=True)
    for selection in selections:
        cache_path = original_cache_path(selection.stem)
        if cache_path.exists():
            continue
        payload = run_tesseract(selection.original_image_path)
        payload["template_stem"] = selection.stem
        write_json(cache_path, payload)


def build_documents_for_split(selections: list[SplitSelection], split_name: str) -> tuple[list[DocumentSample], list[DocumentSample]]:
    query_documents: list[DocumentSample] = []
    original_documents: list[DocumentSample] = []
    for selection in selections:
        original_payload = read_json(original_cache_path(selection.stem))
        original_width = int(original_payload["image_size"]["width"])
        original_height = int(original_payload["image_size"]["height"])
        original_documents.append(
            DocumentSample(
                subset="commonforms_original",
                stem=selection.stem,
                class_label=selection.stem,
                image_path=selection.original_image_path.resolve(),
                record_id=0,
                words=list(original_payload["words"]),
                boxes=[normalize_box(box, original_width, original_height) for box in original_payload["boxes"]],
                is_original=True,
            )
        )

        query_cache_dir = SPLIT_OCR_ROOT / split_name / selection.stem
        for record_id, image_path in enumerate(sorted(selection.query_dir.glob(IMAGE_GLOB), key=lambda p: p.name.lower()), start=1):
            payload = read_json(query_cache_dir / f"{image_path.stem}.ocr.json")
            width = int(payload["image_size"]["width"])
            height = int(payload["image_size"]["height"])
            query_documents.append(
                DocumentSample(
                    subset=f"commonforms_degraded_{split_name}",
                    stem=selection.stem,
                    class_label=selection.stem,
                    image_path=image_path.resolve(),
                    record_id=record_id,
                    words=list(payload["words"]),
                    boxes=[normalize_box(box, width, height) for box in payload["boxes"]],
                    is_original=False,
                )
            )
    return query_documents, original_documents


def sample_train_queries(documents: list[DocumentSample]) -> list[DocumentSample]:
    if len(documents) <= TRAIN_RETRIEVAL_SAMPLE_COUNT:
        return list(documents)
    rng = random.Random(SEED)
    chosen_indices = sorted(rng.sample(range(len(documents)), k=TRAIN_RETRIEVAL_SAMPLE_COUNT))
    return [documents[index] for index in chosen_indices]


class TemplateDataset(Dataset):
    def __init__(self, examples: list[TemplateExample]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> TemplateExample:
        return self.examples[index]


class BalancedBatchSampler(BatchSampler):
    def __init__(self, examples: list[TemplateExample]):
        self.seed = SEED
        self.class_to_indices: dict[int, list[int]] = {}
        for index, example in enumerate(examples):
            self.class_to_indices.setdefault(example.train_label_id, []).append(index)
        self.class_ids = sorted(self.class_to_indices.keys())
        self.batch_count = max(1, math.ceil(len(examples) / (CLASSES_PER_BATCH * SAMPLES_PER_CLASS)))

    def __len__(self) -> int:
        return self.batch_count

    def __iter__(self):
        rng = random.Random(self.seed + int(time.time()))
        for _ in range(self.batch_count):
            batch_indices: list[int] = []
            chosen_classes = rng.sample(self.class_ids, k=min(CLASSES_PER_BATCH, len(self.class_ids)))
            for class_id in chosen_classes:
                candidates = self.class_to_indices[class_id]
                if len(candidates) >= SAMPLES_PER_CLASS:
                    batch_indices.extend(rng.sample(candidates, k=SAMPLES_PER_CLASS))
                else:
                    batch_indices.extend(rng.choices(candidates, k=SAMPLES_PER_CLASS))
            yield batch_indices


def load_processor():
    try:
        return AutoProcessor.from_pretrained(MODEL_NAME, apply_ocr=False)
    except OSError:
        return LayoutLMv3Processor(
            image_processor=LayoutLMv3ImageProcessor(apply_ocr=False),
            tokenizer=LayoutLMv3TokenizerFast.from_pretrained(MODEL_NAME),
        )


class ProcessorCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, examples: list[TemplateExample]) -> dict[str, object]:
        images = [Image.open(example.document.image_path).convert("RGB") for example in examples]
        try:
            encoding = self.processor(
                images=images,
                text=[example.document.words for example in examples],
                boxes=[example.document.boxes for example in examples],
                truncation=True,
                padding="max_length",
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )
        finally:
            for image in images:
                image.close()
        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "bbox": encoding["bbox"],
            "pixel_values": encoding["pixel_values"],
            "labels": torch.tensor([example.train_label_id for example in examples], dtype=torch.long),
            "documents": [example.document for example in examples],
        }


def move_batch_to_device(batch: dict[str, object], device: torch.device) -> dict[str, object]:
    moved = dict(batch)
    for key in ("input_ids", "attention_mask", "bbox", "pixel_values", "labels"):
        moved[key] = moved[key].to(device)
    return moved


def build_training_examples(documents: list[DocumentSample], class_to_id: dict[str, int]) -> list[TemplateExample]:
    return [TemplateExample(document, document.class_label, class_to_id[document.class_label]) for document in documents]


def build_inference_loader(documents: list[DocumentSample], collator: ProcessorCollator, batch_size: int) -> DataLoader:
    examples = [TemplateExample(document, document.class_label, -1) for document in documents]
    return DataLoader(TemplateDataset(examples), batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collator)


def split_train_seen_val(examples: list[TemplateExample]) -> tuple[list[TemplateExample], list[TemplateExample]]:
    grouped: dict[str, list[TemplateExample]] = {}
    for example in examples:
        grouped.setdefault(example.class_label, []).append(example)
    rng = random.Random(SEED)
    train_examples: list[TemplateExample] = []
    seen_val_examples: list[TemplateExample] = []
    for class_examples in grouped.values():
        shuffled = list(class_examples)
        rng.shuffle(shuffled)
        val_count = min(max(1, int(round(len(shuffled) * SEEN_VAL_FRACTION))), max(1, len(shuffled) - 1))
        seen_val_examples.extend(shuffled[:val_count])
        train_examples.extend(shuffled[val_count:])
    return train_examples, seen_val_examples


class ProjectionHead(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, PROJECTION_DIM),
        )

    def forward(self, pooled_features: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.network(pooled_features), p=2, dim=-1)


class ArcFaceHead(nn.Module):
    def __init__(self, embedding_dim: int, num_classes: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
        self.cos_margin = math.cos(ARCFACE_MARGIN)
        self.sin_margin = math.sin(ARCFACE_MARGIN)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        normalized_embeddings = F.normalize(embeddings, p=2, dim=-1)
        normalized_weights = F.normalize(self.weight, p=2, dim=-1)
        cosine = F.linear(normalized_embeddings, normalized_weights).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        sine = torch.sqrt(torch.clamp(1.0 - cosine * cosine, min=1e-7))
        phi = cosine * self.cos_margin - sine * self.sin_margin
        one_hot = F.one_hot(labels, num_classes=self.weight.shape[0]).to(dtype=cosine.dtype)
        return (one_hot * phi + (1.0 - one_hot) * cosine) * ARCFACE_SCALE


class ProjectionModel(nn.Module):
    def __init__(self, num_train_classes: int):
        super().__init__()
        self.backbone = LayoutLMv3Model.from_pretrained(MODEL_NAME)
        self.projection_head = ProjectionHead(self.backbone.config.hidden_size)
        self.arcface_head = ArcFaceHead(PROJECTION_DIM, num_train_classes)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, bbox: torch.Tensor, pixel_values: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            bbox=bbox,
            pixel_values=pixel_values,
            return_dict=True,
        )
        if POOLING == "cls":
            pooled = outputs.last_hidden_state[:, 0, :]
        else:
            pooled = (
                (outputs.last_hidden_state * attention_mask.unsqueeze(-1)).sum(dim=1)
                / attention_mask.unsqueeze(-1).sum(dim=1).clamp(min=1)
            )
        return {"embeddings": self.projection_head(pooled)}


def freeze_backbone_except_top_layers(model: ProjectionModel) -> None:
    for parameter in model.backbone.parameters():
        parameter.requires_grad = False
    for layer in list(model.backbone.encoder.layer)[-max(1, UNFREEZE_LAST_N) :]:
        for parameter in layer.parameters():
            parameter.requires_grad = True
    for parameter in model.projection_head.parameters():
        parameter.requires_grad = True
    for parameter in model.arcface_head.parameters():
        parameter.requires_grad = True


def create_optimizer(model: ProjectionModel) -> torch.optim.Optimizer:
    backbone_parameters = [parameter for parameter in model.backbone.parameters() if parameter.requires_grad]
    head_parameters = [
        parameter
        for module in (model.projection_head, model.arcface_head)
        for parameter in module.parameters()
        if parameter.requires_grad
    ]
    return torch.optim.AdamW(
        [
            {"params": backbone_parameters, "lr": TRAIN_BACKBONE_LR, "weight_decay": WEIGHT_DECAY},
            {"params": head_parameters, "lr": HEAD_LR, "weight_decay": WEIGHT_DECAY},
        ]
    )


def supervised_contrastive_loss(embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    normalized = F.normalize(embeddings, p=2, dim=-1)
    logits = normalized @ normalized.T / SUPCON_TEMPERATURE
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    self_mask = torch.eye(labels.shape[0], device=labels.device, dtype=torch.bool)
    positive_mask = labels.unsqueeze(0).eq(labels.unsqueeze(1)) & ~self_mask
    exp_logits = torch.exp(logits) * (~self_mask)
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp(min=1e-12))
    positive_counts = positive_mask.sum(dim=1)
    mean_log_prob = (positive_mask * log_prob).sum(dim=1) / positive_counts.clamp(min=1)
    valid = positive_counts > 0
    return embeddings.new_tensor(0.0) if not torch.any(valid) else -mean_log_prob[valid].mean()
