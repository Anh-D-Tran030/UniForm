from __future__ import annotations

import gc
import json
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoProcessor, LayoutLMv3Model


MODEL_NAME = "microsoft/layoutlmv3-base"
SPLIT_OCR_ROOT = Path(r"A:\RealForm\processed\synthetic_fill_images_10k_folders_degraded_ocr_split_9400_300_300_augmented")
DEGRADED_ROOT = Path(r"A:\RealForm\processed\synthetic_fill_images_10k_folders_degraded")
ORIGINAL_ROOT = Path(r"A:\RealForm\processed\CommonFormsEnglish\images")
ORIGINAL_OCR_ROOT = Path(r"A:\RealForm\processed\CommonFormsEnglish_original_ocr_cache_projection")
OUTPUT_DIR = Path(r"A:\RealForm\outputs\layoutlmv3_base_cls_retrieval_baseline")

BATCH_SIZE = 24
MAX_LENGTH = 512
SCORE_CHUNK_SIZE = 512
EMBEDDING_SHARD_BATCHES = 100
SPLITS = ["test", "train"]


@dataclass(frozen=True)
class Document:
    stem: str
    image_path: Path
    words: list[str]
    boxes: list[list[int]]
    is_original: bool


class DocumentDataset(Dataset):
    def __init__(self, documents: list[Document]):
        self.documents = documents

    def __len__(self) -> int:
        return len(self.documents)

    def __getitem__(self, index: int) -> Document:
        return self.documents[index]


class Collator:
    def __init__(self, processor, max_length: int):
        self.processor = processor
        self.max_length = max_length

    def __call__(self, documents: list[Document]):
        images = [Image.open(document.image_path).convert("RGB") for document in documents]
        try:
            encoding = self.processor(
                images=images,
                text=[document.words for document in documents],
                boxes=[document.boxes for document in documents],
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
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
            "documents": documents,
        }


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def clear_gpu_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def clamp_box(box: list[int]) -> list[int]:
    if len(box) != 4:
        return [0, 0, 1, 1]
    x1, y1, x2, y2 = [int(round(float(value))) for value in box]
    x1 = max(0, min(1000, x1))
    y1 = max(0, min(1000, y1))
    x2 = max(0, min(1000, x2))
    y2 = max(0, min(1000, y2))
    if x2 <= x1:
        x2 = min(1000, x1 + 1)
    if y2 <= y1:
        y2 = min(1000, y1 + 1)
    return [x1, y1, x2, y2]


def load_document(payload_path: Path, image_path: Path, stem: str, is_original: bool) -> Document:
    payload = read_json(payload_path)
    words = [str(word) for word in payload.get("words", [])]
    boxes = [clamp_box(box) for box in payload.get("boxes", [])]

    if not words or len(words) != len(boxes):
        words = ["[EMPTY]"]
        boxes = [[0, 0, 1, 1]]

    return Document(
        stem=stem,
        image_path=image_path.resolve(),
        words=words,
        boxes=boxes,
        is_original=is_original,
    )


def build_original_image_index() -> dict[str, Path]:
    index: dict[str, Path] = {}
    for image_path in sorted(ORIGINAL_ROOT.rglob("*"), key=lambda path: str(path).lower()):
        if image_path.is_file():
            index.setdefault(image_path.stem, image_path)
    return index


def load_split(split_name: str) -> tuple[list[Document], list[Document]]:
    original_index = build_original_image_index()
    query_documents: list[Document] = []
    original_documents: list[Document] = []

    split_dir = SPLIT_OCR_ROOT / split_name
    for template_dir in sorted([path for path in split_dir.iterdir() if path.is_dir()], key=lambda path: path.name.lower()):
        stem = template_dir.name
        original_image = original_index.get(stem)
        original_ocr = ORIGINAL_OCR_ROOT / f"{stem}.ocr.json"
        degraded_dir = DEGRADED_ROOT / stem

        if original_image is None or not original_ocr.exists() or not degraded_dir.exists():
            continue

        original_documents.append(load_document(original_ocr, original_image, stem, is_original=True))

        for image_path in sorted(degraded_dir.glob("fill_*.png"), key=lambda path: path.name.lower()):
            ocr_path = template_dir / f"{image_path.stem}.ocr.json"
            if ocr_path.exists():
                query_documents.append(load_document(ocr_path, image_path, stem, is_original=False))

    return query_documents, original_documents


def collect_base_cls_embeddings(model, processor, documents: list[Document], device: torch.device, cache_path: Path) -> np.ndarray:
    if cache_path.exists():
        return np.load(cache_path)

    shard_dir = cache_path.with_suffix("")
    shard_dir.mkdir(parents=True, exist_ok=True)

    existing_shards = sorted(shard_dir.glob("shard_*.npy"), key=lambda path: path.name.lower())
    completed_docs = sum(np.load(path, mmap_mode="r").shape[0] for path in existing_shards)
    completed_docs = min(completed_docs, len(documents))
    completed_batches = completed_docs // BATCH_SIZE
    if completed_docs:
        print(f"resuming {cache_path.stem} after {completed_docs}/{len(documents)} documents", flush=True)

    remaining_documents = documents[completed_docs:]
    total_batches = (len(documents) + BATCH_SIZE - 1) // BATCH_SIZE
    loader = DataLoader(
        DocumentDataset(remaining_documents),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=Collator(processor, MAX_LENGTH),
    )

    shard_embeddings = []
    model.eval()

    with torch.no_grad():
        for local_step, batch in enumerate(loader, start=1):
            step = completed_batches + local_step
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                bbox=batch["bbox"].to(device),
                pixel_values=batch["pixel_values"].to(device),
                return_dict=True,
            )
            cls_embeddings = outputs.last_hidden_state[:, 0, :]
            shard_embeddings.append(cls_embeddings.detach().cpu().numpy().astype("float32"))

            if step % EMBEDDING_SHARD_BATCHES == 0 or step == total_batches:
                shard_index = (step - 1) // EMBEDDING_SHARD_BATCHES
                shard_path = shard_dir / f"shard_{shard_index:05d}.npy"
                np.save(shard_path, np.concatenate(shard_embeddings, axis=0))
                shard_embeddings = []

            if step == 1 or step % 100 == 0 or step == total_batches:
                print(f"embedded {step}/{total_batches} batches for {cache_path.stem}", flush=True)

    shard_paths = sorted(shard_dir.glob("shard_*.npy"), key=lambda path: path.name.lower())
    embeddings = np.concatenate([np.load(path) for path in shard_paths], axis=0)[: len(documents)]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, embeddings)
    return embeddings


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.maximum(norms, 1e-12)


def compute_retrieval(query_documents: list[Document], query_embeddings: np.ndarray, original_documents: list[Document], original_embeddings: np.ndarray):
    query_embeddings = normalize_embeddings(query_embeddings)
    original_embeddings = normalize_embeddings(original_embeddings)
    original_labels = np.array([document.stem for document in original_documents])

    top1 = 0
    top5 = 0
    top10 = 0
    reciprocal_ranks = []

    for start in range(0, len(query_documents), SCORE_CHUNK_SIZE):
        end = min(len(query_documents), start + SCORE_CHUNK_SIZE)
        scores = query_embeddings[start:end] @ original_embeddings.T
        orders = np.argsort(-scores, axis=1)

        for local_index, order in enumerate(orders):
            query_doc = query_documents[start + local_index]
            ranked_labels = original_labels[order]
            matches = np.where(ranked_labels == query_doc.stem)[0]
            if len(matches) == 0:
                rank = len(original_documents) + 1
            else:
                rank = int(matches[0]) + 1
            top1 += int(rank == 1)
            top5 += int(rank <= 5)
            top10 += int(rank <= 10)
            reciprocal_ranks.append(1.0 / rank)

        print(f"scored {end}/{len(query_documents)} queries", flush=True)

    query_count = len(query_documents)
    return {
        "query_count": query_count,
        "gallery_count": len(original_documents),
        "retrieval_at_1": top1 / max(1, query_count),
        "retrieval_at_5": top5 / max(1, query_count),
        "retrieval_at_10": top10 / max(1, query_count),
        "mrr": float(sum(reciprocal_ranks) / max(1, query_count)),
        "correct_top1": top1,
    }


def benchmark_split(split_name: str, model, processor, device: torch.device):
    print(f"loading {split_name} split", flush=True)
    query_documents, original_documents = load_split(split_name)
    print(f"{split_name}: queries={len(query_documents)} originals={len(original_documents)}", flush=True)

    split_dir = OUTPUT_DIR / split_name
    write_json(
        split_dir / "documents_summary.json",
        {
            "split": split_name,
            "query_count": len(query_documents),
            "gallery_count": len(original_documents),
            "query_stems": [document.stem for document in query_documents],
            "original_stems": [document.stem for document in original_documents],
        },
    )

    query_embeddings = collect_base_cls_embeddings(model, processor, query_documents, device, split_dir / "query_base_cls_embeddings.npy")
    original_embeddings = collect_base_cls_embeddings(model, processor, original_documents, device, split_dir / "original_base_cls_embeddings.npy")
    metrics = compute_retrieval(query_documents, query_embeddings, original_documents, original_embeddings)
    write_json(split_dir / "base_cls_retrieval_metrics.json", metrics)
    print(f"{split_name} metrics: {metrics}", flush=True)
    return metrics


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    clear_gpu_cache()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)
    print("loading processor/model", flush=True)
    processor = AutoProcessor.from_pretrained(MODEL_NAME, apply_ocr=False, local_files_only=True)
    model = LayoutLMv3Model.from_pretrained(MODEL_NAME, local_files_only=True).to(device)

    metrics_by_split = {}
    for split_name in SPLITS:
        metrics_by_split[split_name] = benchmark_split(split_name, model, processor, device)
        clear_gpu_cache()

    summary = {
        "method": "layoutlmv3_base_raw_cls",
        "model_name": MODEL_NAME,
        "pooling": "last_hidden_state[:, 0, :]",
        "batch_size": BATCH_SIZE,
        "max_length": MAX_LENGTH,
        **metrics_by_split,
    }
    write_json(OUTPUT_DIR / "base_cls_retrieval_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
