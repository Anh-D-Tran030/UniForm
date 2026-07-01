from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")

import benchmark_commonforms_8k_hybrid_practice_train as base


OUTPUT_ROOT = Path(r"A:\RealForm\processed\commonforms_8k_score_fusion_practice_train_benchmark")
ALPHAS = (0.9, 0.7, 0.5, 0.3)
BM25_BATCH_SIZE = 256


def minmax_rows(scores: np.ndarray) -> np.ndarray:
    row_min = scores.min(axis=1, keepdims=True)
    row_max = scores.max(axis=1, keepdims=True)
    denom = np.maximum(row_max - row_min, 1e-12)
    return (scores - row_min) / denom


def evaluate_score_fusion(
    query_docs: list[base.DocSample],
    query_embeddings: np.ndarray,
    original_docs: list[base.DocSample],
    original_embeddings: np.ndarray,
    bm25_index: base.Bm25Index,
    alpha: float,
    batch_size: int = BM25_BATCH_SIZE,
) -> dict[str, Any]:
    query_embeddings = query_embeddings / np.maximum(np.linalg.norm(query_embeddings, axis=1, keepdims=True), 1e-12)
    original_embeddings = original_embeddings / np.maximum(np.linalg.norm(original_embeddings, axis=1, keepdims=True), 1e-12)

    top1 = top5 = top10 = top25 = top50 = top100 = 0
    reciprocal_ranks: list[float] = []
    rows: list[dict[str, Any]] = []
    gallery_count = len(original_docs)

    for batch_start in range(0, len(query_docs), batch_size):
        batch_docs = query_docs[batch_start : batch_start + batch_size]
        batch_tokens = [base.tokenize_words(document.words) for document in batch_docs]
        query_matrix = base.build_query_matrix(bm25_index, batch_tokens)
        bm25_scores = (query_matrix @ bm25_index.matrix.T).toarray()
        dense_scores = query_embeddings[batch_start : batch_start + len(batch_docs)] @ original_embeddings.T

        fused_scores = alpha * minmax_rows(bm25_scores) + (1.0 - alpha) * minmax_rows(dense_scores)

        for offset, query_doc in enumerate(batch_docs):
            scores = fused_scores[offset]
            order = np.argsort(-scores)
            ranked_labels = [original_docs[candidate_id].stem for candidate_id in order]
            true_label = query_doc.stem
            rank = ranked_labels.index(true_label) + 1

            top1 += int(rank == 1)
            top5 += int(rank <= 5)
            top10 += int(rank <= 10)
            top25 += int(rank <= 25)
            top50 += int(rank <= 50)
            top100 += int(rank <= 100)
            reciprocal_ranks.append(1.0 / rank)

            best_candidate_id = int(order[0])
            rows.append(
                {
                    "template_name": query_doc.stem,
                    "query_image": str(query_doc.image_path),
                    "alpha": alpha,
                    "rank": rank,
                    "top1": int(rank == 1),
                    "top5": int(rank <= 5),
                    "top10": int(rank <= 10),
                    "top25": int(rank <= 25),
                    "top50": int(rank <= 50),
                    "top100": int(rank <= 100),
                    "best_match": original_docs[best_candidate_id].stem,
                    "fused_score": float(scores[best_candidate_id]),
                    "bm25_score": float(bm25_scores[offset, best_candidate_id]),
                    "dense_score": float(dense_scores[offset, best_candidate_id]),
                }
            )

        processed = min(batch_start + len(batch_docs), len(query_docs))
        if processed == len(batch_docs) or processed % 5000 < batch_size or processed == len(query_docs):
            base.log(f"Fusion alpha={alpha:.1f} progress: {processed}/{len(query_docs)} queries")

    n = len(query_docs)
    summary = {
        "method": "score_fusion_bm25_plus_practice_train_layoutlmv3",
        "checkpoint_path": str(base.CHECKPOINT_PATH),
        "gallery_templates": len(original_docs),
        "query_images": n,
        "alpha": alpha,
        "bm25_weight": alpha,
        "dense_weight": 1.0 - alpha,
        "score_normalization": "per_query_minmax",
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


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    device = base.torch.device("cuda" if base.torch.cuda.is_available() else "cpu")
    base.clear_gpu_cache()
    base.log(f"Loading 8k documents from {base.EXTRA8K_OCR_ROOT}")
    query_docs, original_docs = base.load_extra8k_documents()
    base.log(f"Loaded {len(original_docs)} originals and {len(query_docs)} query images")
    if not query_docs or not original_docs:
        raise RuntimeError("No 8k documents were loaded.")

    base.log("Building BM25 index")
    bm25_index = base.build_bm25_index(original_docs)
    base.write_json(
        OUTPUT_ROOT / "benchmark_scope.json",
        {
            "gallery_templates": len(original_docs),
            "query_images": len(query_docs),
            "alphas": list(ALPHAS),
            "score_normalization": "per_query_minmax",
        },
    )

    base.log(f"Loading practice_train model checkpoint on {device}")
    model, collator = base.load_model(device)

    base.log("Embedding originals and queries with practice_train model")
    loader = base.build_inference_loader(collator, query_docs, original_docs, batch_size=base.BATCH_SIZE)
    embeddings, _ordered_docs = base.collect_embeddings(model, loader, device)
    query_count = len(query_docs)
    gallery_count = len(original_docs)
    query_embeddings = embeddings[:query_count]
    original_embeddings = embeddings[query_count : query_count + gallery_count]

    model.cpu()
    del model
    base.clear_gpu_cache()
    base.log("Model moved off GPU; score fusion runs on CPU with cached embeddings")

    all_summaries: dict[str, Any] = {}
    for alpha in ALPHAS:
        base.log(f"Evaluating score fusion with alpha={alpha:.1f}")
        result = evaluate_score_fusion(
            query_docs=query_docs,
            query_embeddings=query_embeddings,
            original_docs=original_docs,
            original_embeddings=original_embeddings,
            bm25_index=bm25_index,
            alpha=alpha,
        )
        summary = result["summary"]
        key = f"alpha_{alpha:.1f}".replace(".", "_")
        all_summaries[key] = summary
        output_dir = OUTPUT_ROOT / key
        output_dir.mkdir(parents=True, exist_ok=True)
        base.write_json(output_dir / "summary.json", summary)
        with (output_dir / "per_query_fusion.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(result["rows"][0].keys()))
            writer.writeheader()
            writer.writerows(result["rows"])
        base.clear_gpu_cache()

    base.write_json(OUTPUT_ROOT / "comparison_summary.json", all_summaries)
    base.log(json.dumps(all_summaries, indent=2))


if __name__ == "__main__":
    main()
