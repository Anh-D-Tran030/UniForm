from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

import torch
from PIL import Image

from colpali_engine.models import ColFlor, ColFlorProcessor


ROOT = Path("A:/RealForm")
MODEL_NAME = "ahmed-masry/ColFlor"

COMMONFORMS_IMAGES = ROOT / "processed/CommonFormsEnglish/images"
SPLIT_ROOT = ROOT / "processed/synthetic_fill_images_10k_folders_degraded_ocr_split_9400_300_300_augmented"
DEGRADED_10K = ROOT / "processed/synthetic_fill_images_10k_folders_degraded"
DEGRADED_8K = ROOT / "processed/synthetic_fill_images_8k_extra_3fills_degraded"
SELECTED_8K = ROOT / "processed/commonforms_8k_extra_3fills_retrieval_benchmark/selected_templates.json"
OUT_ROOT = ROOT / "processed/colflor_document_similarity_benchmark"


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text: str) -> int:
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


@dataclass(frozen=True)
class GalleryItem:
    stem: str
    image_path: Path


@dataclass(frozen=True)
class QueryItem:
    template_stem: str
    query_id: str
    image_path: Path


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def find_original_images() -> dict[str, Path]:
    originals: dict[str, Path] = {}
    for path in COMMONFORMS_IMAGES.rglob("*"):
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}:
            originals[path.stem] = path
    return originals


def load_test_split_items() -> tuple[list[GalleryItem], list[QueryItem]]:
    originals = find_original_images()
    stems = sorted(p.name for p in (SPLIT_ROOT / "test").iterdir() if p.is_dir())
    gallery: list[GalleryItem] = []
    queries: list[QueryItem] = []
    missing_originals: list[str] = []
    missing_queries: list[str] = []

    for stem in stems:
        original = originals.get(stem)
        if original is None:
            missing_originals.append(stem)
            continue
        query_dir = DEGRADED_10K / stem
        fills = sorted(query_dir.glob("fill_*.png"))
        if not fills:
            missing_queries.append(stem)
            continue
        gallery.append(GalleryItem(stem=stem, image_path=original))
        for fill_path in fills:
            queries.append(QueryItem(stem, fill_path.stem, fill_path))

    print(f"Loaded test split: {len(gallery)} gallery templates, {len(queries)} filled queries")
    if missing_originals:
        print(f"Missing originals: {len(missing_originals)}")
    if missing_queries:
        print(f"Missing query image folders: {len(missing_queries)}")
    return gallery, queries


def load_extra8k_items() -> tuple[list[GalleryItem], list[QueryItem]]:
    selected = read_json(SELECTED_8K)
    by_stem: dict[str, Path] = {}
    for row in selected:
        stem = row["stem"] if isinstance(row, dict) else str(row)
        original_path = Path(row["original_image_path"]) if isinstance(row, dict) and row.get("original_image_path") else None
        if original_path and original_path.exists():
            by_stem[stem] = original_path

    originals = find_original_images()
    gallery: list[GalleryItem] = []
    queries: list[QueryItem] = []
    missing_originals: list[str] = []
    missing_queries: list[str] = []

    stems = sorted(p.name for p in DEGRADED_8K.iterdir() if p.is_dir())
    for stem in stems:
        original = by_stem.get(stem) or originals.get(stem)
        if original is None:
            missing_originals.append(stem)
            continue
        fills = sorted((DEGRADED_8K / stem).glob("fill_*.png"))
        if not fills:
            missing_queries.append(stem)
            continue
        gallery.append(GalleryItem(stem=stem, image_path=original))
        for fill_path in fills:
            queries.append(QueryItem(stem, fill_path.stem, fill_path))

    print(f"Loaded extra8k: {len(gallery)} gallery templates, {len(queries)} filled queries")
    if missing_originals:
        print(f"Missing originals: {len(missing_originals)}")
    if missing_queries:
        print(f"Missing query image folders: {len(missing_queries)}")
    return gallery, queries


def open_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def safe_cache_name(path: Path) -> str:
    return path.stem.replace(":", "_").replace("\\", "_").replace("/", "_") + ".pt"


def load_model():
    device = torch.device("cpu")
    print(f"Loading {MODEL_NAME} on CPU only")
    model = ColFlor.from_pretrained(
        MODEL_NAME,
        device_map="cpu",
        attn_implementation="eager",
        local_files_only=True,
    ).eval()
    processor = ColFlorProcessor.from_pretrained(MODEL_NAME, local_files_only=True)
    return model, processor, device


def to_device(batch, device: torch.device):
    return {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}


def encode_paths(
    model,
    processor,
    device: torch.device,
    paths: list[Path],
    cache_dir: Path,
    batch_size: int,
) -> list[torch.Tensor]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    embeddings: list[torch.Tensor | None] = []
    todo: list[tuple[int, Path, Path]] = []

    for idx, path in enumerate(paths):
        cache_path = cache_dir / safe_cache_name(path)
        if cache_path.exists():
            emb = torch.load(cache_path, map_location="cpu", weights_only=False)
            if emb.dtype != torch.float32:
                emb = emb.to(torch.float32)
                torch.save(emb, cache_path)
            embeddings.append(emb)
        else:
            embeddings.append(None)
            todo.append((idx, path, cache_path))

    start = time.time()
    for offset in range(0, len(todo), batch_size):
        chunk = todo[offset : offset + batch_size]
        images = [open_rgb(path) for _, path, _ in chunk]
        batch = to_device(processor.process_images(images), device)
        with torch.no_grad():
            out = model(**batch)
        for row, (idx, _, cache_path) in zip(out, chunk):
            emb = row.detach().cpu().to(torch.float32)
            torch.save(emb, cache_path)
            embeddings[idx] = emb
        done = min(offset + len(chunk), len(todo))
        if done == len(todo) or done % max(batch_size * 10, 10) == 0:
            elapsed = max(time.time() - start, 1e-6)
            rate = done / elapsed
            remain = (len(todo) - done) / rate if rate > 0 else 0
            print(f"Encoded {done}/{len(todo)} missing images ({rate:.2f}/s, eta {remain/60:.1f} min)")
        del batch, out, images
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return [emb for emb in embeddings if emb is not None]


def chunks(items: list, size: int) -> Iterable[list]:
    for offset in range(0, len(items), size):
        yield items[offset : offset + size]


def evaluate(
    processor,
    device: torch.device,
    gallery: list[GalleryItem],
    gallery_embs: list[torch.Tensor],
    queries: list[QueryItem],
    query_embs: list[torch.Tensor],
    out_dir: Path,
    query_chunk_size: int,
    gallery_chunk_size: int,
    score_batch_size: int,
) -> dict:
    gallery_stems = [item.stem for item in gallery]
    per_query_path = out_dir / "per_query_results.csv"
    correct_at_1 = 0
    correct_at_5 = 0
    correct_at_10 = 0
    mrr_total = 0.0
    rows = []
    started = time.time()

    with per_query_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "query_path",
                "template_stem",
                "rank",
                "top1_stem",
                "top1_score",
                "top5_stems",
                "top5_scores",
            ],
        )
        writer.writeheader()

        for q_offset in range(0, len(queries), query_chunk_size):
            q_items = queries[q_offset : q_offset + query_chunk_size]
            q_vecs = query_embs[q_offset : q_offset + query_chunk_size]
            score_parts: list[torch.Tensor] = []

            for g_offset in range(0, len(gallery), gallery_chunk_size):
                g_vecs = gallery_embs[g_offset : g_offset + gallery_chunk_size]
                scores = processor.score(
                    q_vecs,
                    g_vecs,
                    batch_size=score_batch_size,
                    device=device,
                ).cpu()
                score_parts.append(scores)
                del scores
                gc.collect()

            full_scores = torch.cat(score_parts, dim=1)
            order = torch.argsort(full_scores, dim=1, descending=True)

            for row_idx, query in enumerate(q_items):
                ranked_indices = order[row_idx].tolist()
                ranked_stems = [gallery_stems[i] for i in ranked_indices]
                rank = ranked_stems.index(query.template_stem) + 1
                scores = full_scores[row_idx]
                top5_indices = ranked_indices[:5]
                top5_stems = [gallery_stems[i] for i in top5_indices]
                top5_scores = [float(scores[i]) for i in top5_indices]
                top1_stem = top5_stems[0]
                top1_score = top5_scores[0]

                correct_at_1 += int(rank <= 1)
                correct_at_5 += int(rank <= 5)
                correct_at_10 += int(rank <= 10)
                mrr_total += 1.0 / rank

                result = {
                    "query_path": str(query.image_path),
                    "template_stem": query.template_stem,
                    "rank": rank,
                    "top1_stem": top1_stem,
                    "top1_score": top1_score,
                    "top5_stems": "|".join(top5_stems),
                    "top5_scores": "|".join(f"{s:.6f}" for s in top5_scores),
                }
                writer.writerow(result)
                rows.append(result)

            done = min(q_offset + len(q_items), len(queries))
            elapsed = max(time.time() - started, 1e-6)
            rate = done / elapsed
            remain = (len(queries) - done) / rate if rate > 0 else 0
            print(f"Scored {done}/{len(queries)} queries ({rate:.2f}/s, eta {remain/60:.1f} min)")

            del full_scores, order, score_parts
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    n = max(len(queries), 1)
    summary = {
        "query_count": len(queries),
        "gallery_count": len(gallery),
        "retrieval_at_1": correct_at_1 / n,
        "retrieval_at_5": correct_at_5 / n,
        "retrieval_at_10": correct_at_10 / n,
        "mrr": mrr_total / n,
        "correct_at_1": correct_at_1,
        "correct_at_5": correct_at_5,
        "correct_at_10": correct_at_10,
        "elapsed_seconds": time.time() - started,
        "per_query_csv": str(per_query_path),
    }
    write_json(out_dir / "colflor_retrieval_summary.json", summary)
    return summary


def run_benchmark(args) -> dict:
    if args.dataset == "test":
        gallery, queries = load_test_split_items()
    elif args.dataset == "extra8k":
        gallery, queries = load_extra8k_items()
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    if args.limit_templates:
        keep = {item.stem for item in gallery[: args.limit_templates]}
        gallery = [item for item in gallery if item.stem in keep]
        queries = [item for item in queries if item.template_stem in keep]
        print(f"Limited to {len(gallery)} templates and {len(queries)} queries")
    if args.limit_queries:
        queries = queries[: args.limit_queries]
        print(f"Limited to {len(queries)} queries")

    out_dir = OUT_ROOT / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        out_dir / "benchmark_manifest.json",
        {
            "dataset": args.dataset,
            "model_name": MODEL_NAME,
            "gallery_count": len(gallery),
            "query_count": len(queries),
            "gallery": [{"stem": item.stem, "image_path": str(item.image_path)} for item in gallery],
            "queries": [
                {
                    "template_stem": item.template_stem,
                    "query_id": item.query_id,
                    "image_path": str(item.image_path),
                }
                for item in queries
            ],
        },
    )

    model, processor, device = load_model()
    gallery_embs = encode_paths(
        model,
        processor,
        device,
        [item.image_path for item in gallery],
        out_dir / "embedding_cache/gallery",
        args.encode_batch_size,
    )
    query_embs = encode_paths(
        model,
        processor,
        device,
        [item.image_path for item in queries],
        out_dir / "embedding_cache/queries",
        args.encode_batch_size,
    )
    summary = evaluate(
        processor,
        device,
        gallery,
        gallery_embs,
        queries,
        query_embs,
        out_dir,
        args.query_chunk_size,
        args.gallery_chunk_size,
        args.score_batch_size,
    )
    print(json.dumps(summary, indent=2))
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark ColFlor OCR-free document image similarity.")
    parser.add_argument("--dataset", choices=["test", "extra8k"], required=True)
    parser.add_argument("--limit-templates", type=int, default=0)
    parser.add_argument("--limit-queries", type=int, default=0)
    parser.add_argument("--encode-batch-size", type=int, default=4)
    parser.add_argument("--query-chunk-size", type=int, default=8)
    parser.add_argument("--gallery-chunk-size", type=int, default=512)
    parser.add_argument("--score-batch-size", type=int, default=32)
    parser.add_argument("--log-file", type=Path, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    log_path = parsed.log_file or (OUT_ROOT / parsed.dataset / "colflor_benchmark.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = Tee(sys.__stdout__, log)
        sys.stderr = Tee(sys.__stderr__, log)
        try:
            print(f"\n===== ColFlor benchmark started {time.strftime('%Y-%m-%d %H:%M:%S')} =====")
            print(f"Command: {' '.join(sys.argv)}")
            print(f"Log file: {log_path}")
            run_benchmark(parsed)
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
