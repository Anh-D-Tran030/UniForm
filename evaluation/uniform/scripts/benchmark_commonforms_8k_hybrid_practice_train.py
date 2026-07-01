from __future__ import annotations

import csv
import gc
import json
import math
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as func
from PIL import Image
from scipy import sparse
from torch.utils.data import DataLoader, Dataset

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")

from transformers import AutoProcessor, LayoutLMv3Model


CHECKPOINT_PATH = Path(r"A:\RealForm\outputs\practice_train\last_projection_model.pt")
EXTRA8K_DEGRADED_ROOT = Path(r"A:\RealForm\processed\synthetic_fill_images_8k_extra_3fills_degraded")
EXTRA8K_OCR_ROOT = Path(r"A:\RealForm\processed\synthetic_fill_images_8k_extra_3fills_degraded_ocr_clean\ocr_json")
ORIGINAL_IMAGE_ROOT = Path(r"A:\RealForm\processed\CommonFormsEnglish\images")
ORIGINAL_OCR_ROOT = Path(r"A:\RealForm\processed\CommonFormsEnglish_ocr_cache")
OUTPUT_ROOT = Path(r"A:\RealForm\processed\commonforms_8k_hybrid_practice_train_benchmark")

MAX_LENGTH = 512
BATCH_SIZE = 4
BM25_K1 = 1.5
BM25_B = 0.75
SHORTLIST_SIZES = (25, 50, 100)
TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


@dataclass(frozen=True)
class DocSample:
    subset: str
    stem: str
    class_label: str
    image_path: Path
    record_id: int
    words: list[str]
    boxes: list[list[int]]
    is_original: bool = False


@dataclass(frozen=True)
class TemplateExample:
    document: DocSample
    class_label: str
    train_label_id: int


@dataclass
class Bm25Index:
    template_names: list[str]
    template_to_id: dict[str, int]
    vocab: dict[str, int]
    matrix: sparse.csr_matrix


def log(message: str) -> None:
    print(message, flush=True)


def clear_gpu_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


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


def load_ocr_doc(payload: dict[str, Any], image_path: Path, stem: str, subset: str, record_id: int, is_original: bool = False) -> DocSample:
    width = int(payload["image_size"]["width"])
    height = int(payload["image_size"]["height"])
    words = [str(word) for word in payload.get("words", [])]
    boxes = [normalize_box(box, width, height) for box in payload.get("boxes", [])]
    if not words:
        words = ["[EMPTY]"]
        boxes = [[0, 0, 1, 1]]
    return DocSample(
        subset=subset,
        stem=stem,
        class_label=stem,
        image_path=image_path.resolve(),
        record_id=record_id,
        words=words,
        boxes=boxes,
        is_original=is_original,
    )


def build_original_image_index(original_root: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for image_path in sorted(original_root.rglob("*"), key=lambda p: str(p).lower()):
        if image_path.is_file():
            index.setdefault(image_path.stem, image_path.resolve())
    return index


def load_extra8k_documents() -> tuple[list[DocSample], list[DocSample]]:
    original_index = build_original_image_index(ORIGINAL_IMAGE_ROOT)
    query_docs: list[DocSample] = []
    original_docs: list[DocSample] = []

    template_dirs = sorted([path for path in EXTRA8K_OCR_ROOT.iterdir() if path.is_dir()], key=lambda p: p.name.lower())
    for template_dir in template_dirs:
        stem = template_dir.name
        original_image = original_index.get(stem)
        original_ocr = ORIGINAL_OCR_ROOT / f"{stem}.ocr.json"
        degraded_dir = EXTRA8K_DEGRADED_ROOT / stem
        if original_image is None or not original_ocr.exists() or not degraded_dir.exists():
            continue

        original_docs.append(
            load_ocr_doc(
                payload=read_json(original_ocr),
                image_path=original_image,
                stem=stem,
                subset="extra8k_original",
                record_id=0,
                is_original=True,
            )
        )

        for record_id, image_path in enumerate(sorted(degraded_dir.glob("fill_*.png"), key=lambda p: p.name.lower()), start=1):
            ocr_path = template_dir / f"{image_path.stem}.ocr.json"
            if not ocr_path.exists():
                continue
            query_docs.append(
                load_ocr_doc(
                    payload=read_json(ocr_path),
                    image_path=image_path,
                    stem=stem,
                    subset="extra8k_query",
                    record_id=record_id,
                    is_original=False,
                )
            )

    return query_docs, original_docs


class DenseRetriveHead(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 128),
        )

    def forward(self, pool: torch.Tensor) -> torch.Tensor:
        return func.normalize(self.net(pool), p=2, dim=-1)


class AccelerateArcFace(nn.Module):
    def __init__(self, emb_dim: int, num_class: int):
        super().__init__()
        self.param = nn.Parameter(torch.empty(num_class, emb_dim))
        nn.init.xavier_uniform_(self.param)
        self.cos_margin = math.cos(0.2)
        self.sin_margin = math.sin(0.2)

    def forward(self, embed: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        norm_emb = func.normalize(embed, p=2, dim=-1)
        norm_weight = func.normalize(self.param, p=2, dim=-1)
        cos = func.linear(norm_emb, norm_weight).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        sin = torch.sqrt(torch.clamp(1.0 - cos * cos, min=1e-7))
        phi = cos * self.cos_margin - sin * self.sin_margin
        one_hot = func.one_hot(labels, num_classes=self.param.shape[0]).to(dtype=cos.dtype)
        return (one_hot * phi + (1.0 - one_hot) * cos) * 30.0


class ODCModel(nn.Module):
    def __init__(self, num_class: int):
        super().__init__()
        self.backbone = LayoutLMv3Model.from_pretrained("microsoft/layoutlmv3-base")
        self.proj = DenseRetriveHead(self.backbone.config.hidden_size)
        self.accelerate = AccelerateArcFace(128, num_class)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, bbox: torch.Tensor, pixel_values: torch.Tensor) -> torch.Tensor:
        embeds = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            bbox=bbox,
            pixel_values=pixel_values,
            return_dict=True,
        )
        pooled = embeds.last_hidden_state[:, 0, :]
        return self.proj(pooled)


class TemplateDocumentDataset(Dataset):
    def __init__(self, examples: list[TemplateExample]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> TemplateExample:
        return self.examples[index]


class ProcessorCollator:
    def __init__(self, processor: Any, max_length: int = 512):
        self.processor = processor
        self.max_length = max_length

    def __call__(self, examples: list[TemplateExample]) -> dict[str, Any]:
        images = [Image.open(example.document.image_path).convert("RGB") for example in examples]
        try:
            encoding = self.processor(
                images=images,
                text=[example.document.words for example in examples],
                boxes=[example.document.boxes for example in examples],
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
            "labels": torch.tensor([example.train_label_id for example in examples], dtype=torch.long),
            "documents": [example.document for example in examples],
        }


def build_examples(documents: list[DocSample], class_to_id: dict[str, int]) -> list[TemplateExample]:
    return [
        TemplateExample(
            document=document,
            class_label=document.class_label,
            train_label_id=class_to_id[document.class_label],
        )
        for document in documents
    ]


def build_inference_examples(documents: list[DocSample]) -> list[TemplateExample]:
    labels = sorted({document.class_label for document in documents})
    class_to_id = {label: idx for idx, label in enumerate(labels)}
    return build_examples(documents, class_to_id)


def build_inference_loader(collator: ProcessorCollator, query_docs: list[DocSample], original_docs: list[DocSample], batch_size: int) -> DataLoader:
    docs = query_docs + original_docs
    examples = build_inference_examples(docs)
    return DataLoader(
        TemplateDocumentDataset(examples),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collator,
    )


def collect_embeddings(model: ODCModel, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, list[DocSample]]:
    model.eval()
    all_embs: list[np.ndarray] = []
    all_docs: list[DocSample] = []
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader, start=1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            bbox = batch["bbox"].to(device)
            pixel_values = batch["pixel_values"].to(device)
            embs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                bbox=bbox,
                pixel_values=pixel_values,
            )
            all_embs.append(embs.detach().cpu().numpy())
            all_docs.extend(batch["documents"])
            if batch_index == 1 or batch_index % 500 == 0:
                log(f"Embedding batches processed: {batch_index}")
    return np.concatenate(all_embs, axis=0), all_docs


def tokenize_words(words: list[str]) -> list[str]:
    tokens: list[str] = []
    for word in words:
        tokens.extend(match.group(0).lower() for match in TOKEN_RE.finditer(str(word)))
    return tokens or ["empty"]


def build_bm25_index(original_docs: list[DocSample]) -> Bm25Index:
    token_counts: list[Counter[str]] = []
    doc_lengths: list[int] = []
    template_names: list[str] = []
    vocab: dict[str, int] = {}

    for document in original_docs:
        tokens = tokenize_words(document.words)
        template_names.append(document.stem)
        doc_lengths.append(len(tokens))
        counts = Counter(tokens)
        token_counts.append(counts)
        for token in counts:
            if token not in vocab:
                vocab[token] = len(vocab)

    total_docs = len(template_names)
    average_doc_length = sum(doc_lengths) / max(total_docs, 1)
    doc_frequencies: Counter[str] = Counter()
    for counts in token_counts:
        doc_frequencies.update(counts.keys())

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for doc_id, counts in enumerate(token_counts):
        length_factor = BM25_K1 * (1.0 - BM25_B + BM25_B * doc_lengths[doc_id] / max(average_doc_length, 1e-12))
        for token, frequency in counts.items():
            document_frequency = doc_frequencies[token]
            idf = math.log(1.0 + (total_docs - document_frequency + 0.5) / (document_frequency + 0.5))
            weight = idf * (frequency * (BM25_K1 + 1.0)) / (frequency + length_factor)
            rows.append(doc_id)
            cols.append(vocab[token])
            data.append(weight)

    matrix = sparse.csr_matrix((data, (rows, cols)), shape=(total_docs, len(vocab)), dtype=np.float32)
    return Bm25Index(
        template_names=template_names,
        template_to_id={template_name: index for index, template_name in enumerate(template_names)},
        vocab=vocab,
        matrix=matrix,
    )


def build_query_matrix(index: Bm25Index, query_token_rows: list[list[str]]) -> sparse.csr_matrix:
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for row_id, tokens in enumerate(query_token_rows):
        for token, count in Counter(tokens).items():
            token_id = index.vocab.get(token)
            if token_id is None:
                continue
            rows.append(row_id)
            cols.append(token_id)
            data.append(float(count))
    return sparse.csr_matrix((data, (rows, cols)), shape=(len(query_token_rows), len(index.vocab)), dtype=np.float32)


def compute_bm25_shortlists(query_docs: list[DocSample], index: Bm25Index, shortlist_size: int, batch_size: int = 256) -> np.ndarray:
    shortlists = np.empty((len(query_docs), shortlist_size), dtype=np.int32)
    for batch_start in range(0, len(query_docs), batch_size):
        batch_docs = query_docs[batch_start : batch_start + batch_size]
        batch_tokens = [tokenize_words(document.words) for document in batch_docs]
        query_matrix = build_query_matrix(index, batch_tokens)
        score_matrix = (query_matrix @ index.matrix.T).toarray()
        for offset, scores in enumerate(score_matrix):
            if shortlist_size >= scores.shape[0]:
                candidate_ids = np.argsort(-scores)
            else:
                candidate_ids = np.argpartition(-scores, shortlist_size - 1)[:shortlist_size]
                candidate_ids = candidate_ids[np.argsort(-scores[candidate_ids])]
            shortlists[batch_start + offset] = candidate_ids[:shortlist_size]
        processed = min(batch_start + len(batch_docs), len(query_docs))
        if processed == len(batch_docs) or processed % 5000 < batch_size or processed == len(query_docs):
            log(f"BM25 shortlist progress: {processed}/{len(query_docs)} queries")
    return shortlists


def evaluate_hybrid(
    query_docs: list[DocSample],
    query_embeddings: np.ndarray,
    original_docs: list[DocSample],
    original_embeddings: np.ndarray,
    bm25_index: Bm25Index,
    bm25_shortlists: np.ndarray,
    shortlist_size: int,
) -> dict[str, Any]:
    query_embeddings = query_embeddings / np.maximum(np.linalg.norm(query_embeddings, axis=1, keepdims=True), 1e-12)
    original_embeddings = original_embeddings / np.maximum(np.linalg.norm(original_embeddings, axis=1, keepdims=True), 1e-12)

    top1 = top5 = top10 = top25 = top50 = top100 = 0
    reciprocal_ranks: list[float] = []
    rows: list[dict[str, Any]] = []
    gallery_count = len(original_docs)

    for index, query_doc in enumerate(query_docs):
        candidate_ids = bm25_shortlists[index, :shortlist_size]
        candidate_embs = original_embeddings[candidate_ids]
        scores = candidate_embs @ query_embeddings[index]
        order = np.argsort(-scores)
        ranked_candidate_ids = candidate_ids[order]
        ranked_labels = [original_docs[candidate_id].stem for candidate_id in ranked_candidate_ids]
        true_label = query_doc.stem
        try:
            rank = ranked_labels.index(true_label) + 1
        except ValueError:
            rank = gallery_count + 1

        top1 += int(rank == 1)
        top5 += int(rank <= 5)
        top10 += int(rank <= 10)
        top25 += int(rank <= 25)
        top50 += int(rank <= 50)
        top100 += int(rank <= 100)
        reciprocal_ranks.append(0.0 if rank > gallery_count else 1.0 / rank)

        best_candidate_id = int(ranked_candidate_ids[0])
        rows.append(
            {
                "template_name": query_doc.stem,
                "query_image": str(query_doc.image_path),
                "shortlist_size": shortlist_size,
                "rank": rank,
                "top1": int(rank == 1),
                "top5": int(rank <= 5),
                "top10": int(rank <= 10),
                "top25": int(rank <= 25),
                "top50": int(rank <= 50),
                "top100": int(rank <= 100),
                "best_match": original_docs[best_candidate_id].stem,
                "best_score": float(scores[order[0]]),
            }
        )

    n = len(query_docs)
    summary = {
        "method": "bm25_shortlist_plus_practice_train_layoutlmv3_rerank",
        "checkpoint_path": str(CHECKPOINT_PATH),
        "gallery_templates": len(original_docs),
        "query_images": n,
        "shortlist_size": shortlist_size,
        "retrieval_at_1": top1 / n,
        "retrieval_at_5": top5 / n,
        "retrieval_at_10": top10 / n,
        "retrieval_at_25": top25 / n,
        "retrieval_at_50": top50 / n,
        "retrieval_at_100": top100 / n,
        "mrr": float(sum(reciprocal_ranks) / n),
        "correct_top1_queries": top1,
    }
    return {"summary": summary, "rows": rows}


def load_model(device: torch.device) -> tuple[ODCModel, ProcessorCollator]:
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    num_class = int(checkpoint["model_state_dict"]["accelerate.param"].shape[0])
    model = ODCModel(num_class=num_class).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    processor = AutoProcessor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
    collator = ProcessorCollator(processor, max_length=MAX_LENGTH)
    return model, collator


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clear_gpu_cache()
    log(f"Loading 8k documents from {EXTRA8K_OCR_ROOT}")
    query_docs, original_docs = load_extra8k_documents()
    log(f"Loaded {len(original_docs)} originals and {len(query_docs)} query images")
    if not query_docs or not original_docs:
        raise RuntimeError("No 8k documents were loaded.")

    log("Building BM25 index")
    bm25_index = build_bm25_index(original_docs)
    max_shortlist = max(SHORTLIST_SIZES)
    bm25_shortlists = compute_bm25_shortlists(query_docs, bm25_index, shortlist_size=max_shortlist)
    write_json(
        OUTPUT_ROOT / "bm25_shortlist_summary.json",
        {
            "gallery_templates": len(original_docs),
            "query_images": len(query_docs),
            "max_shortlist": max_shortlist,
            "shortlist_sizes": list(SHORTLIST_SIZES),
        },
    )

    log(f"Loading practice_train model checkpoint on {device}")
    model, collator = load_model(device)

    log("Embedding originals and queries with practice_train model")
    loader = build_inference_loader(collator, query_docs, original_docs, batch_size=BATCH_SIZE)
    embeddings, _ordered_docs = collect_embeddings(model, loader, device)
    query_count = len(query_docs)
    gallery_count = len(original_docs)
    query_embeddings = embeddings[:query_count]
    original_embeddings = embeddings[query_count : query_count + gallery_count]

    model.cpu()
    del model
    clear_gpu_cache()
    log("Model moved off GPU; reranking runs on CPU with cached embeddings")

    all_summaries: dict[str, Any] = {}
    for shortlist_size in SHORTLIST_SIZES:
        log(f"Evaluating hybrid rerank with BM25 top-{shortlist_size}")
        result = evaluate_hybrid(
            query_docs=query_docs,
            query_embeddings=query_embeddings,
            original_docs=original_docs,
            original_embeddings=original_embeddings,
            bm25_index=bm25_index,
            bm25_shortlists=bm25_shortlists,
            shortlist_size=shortlist_size,
        )
        summary = result["summary"]
        all_summaries[f"top_{shortlist_size}"] = summary
        output_dir = OUTPUT_ROOT / f"top_{shortlist_size}"
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json(output_dir / "summary.json", summary)
        with (output_dir / "per_query_rerank.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(result["rows"][0].keys()))
            writer.writeheader()
            writer.writerows(result["rows"])
        clear_gpu_cache()

    write_json(OUTPUT_ROOT / "comparison_summary.json", all_summaries)
    log(json.dumps(all_summaries, indent=2))


if __name__ == "__main__":
    main()
