from __future__ import annotations

import csv
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup


EXPERIMENT_ROOT = Path(r"C:\Users\thanh\OneDrive - UTS\Personal_code\LayoutLMv3 Experiment")
SCRIPT_ROOT = Path(__file__).resolve().parent
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

import benchmark_commonforms_projection_retrieval as bench
import benchmark_photograph_projection as photo_bench
import train_layoutlmv3_template_projection as proj
import visualize_layoutlmv3_template_embeddings as viz


PHOTO_ROOT = Path(r"A:\FDT_TO _PROCESS\Photograph")
PHOTO_IMAGE_ROOT = Path(r"A:\RealForm\processed\Photograph_image_data")
PHOTO_OCR_ROOT = Path(r"A:\RealForm\processed\Photograph_ocr_cache")
CONVERTED_PDF_ROOT = Path(r"A:\FDT_TO _PROCESS\converted_pdf")
CONVERTED_PDF_OCR_ROOT = Path(r"A:\RealForm\processed\converted_pdf_ocr_cache")
OUTPUT_DIR = Path(r"A:\RealForm\outputs\photograph_projection_head_finetune_8_2_degrade")
DEGRADED_AUG_ROOT = OUTPUT_DIR / "train_degraded_aug"
TESSERACT_CMD = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")

SEED = 123
EPOCHS = 10
BATCH_SIZE = 8
EVAL_BATCH_SIZE = 8
CLASSES_PER_BATCH = 4
SAMPLES_PER_CLASS = 2
HEAD_LR = 2e-4
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.08
GRADIENT_CLIP_NORM = 1.0
ARCFACE_WEIGHT = 1.0
SUPCON_WEIGHT = 0.2
SUPCON_TEMPERATURE = 0.1
PSM = 11
OEM = 3
MIN_CONFIDENCE = 30.0


@dataclass(frozen=True)
class ConvertedPage:
    template_name: str
    role: str
    record_id: int
    image_path: Path


def write_json(path: Path, payload: dict[str, Any] | list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def latest_checkpoint_dir() -> Path:
    candidates = sorted(
        Path(r"A:\RealForm\outputs").glob("**/best_projection_model.pt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError("No best_projection_model.pt found under A:\\RealForm\\outputs.")
    return candidates[0].parent.resolve()


def load_new_head_model(checkpoint_dir: Path, num_train_classes: int, device: torch.device) -> tuple[proj.LayoutLMv3TemplateProjectionModel, Any, dict[str, Any]]:
    training_config = read_json(checkpoint_dir / "training_config.json")
    checkpoint = torch.load(checkpoint_dir / "best_projection_model.pt", map_location=device, weights_only=False)
    state_dict = checkpoint["model_state_dict"]
    filtered_state_dict = {key: value for key, value in state_dict.items() if not key.startswith("arcface_head.")}
    model = proj.LayoutLMv3TemplateProjectionModel(
        model_name=training_config["model_name"],
        projection_dim=int(training_config["projection_dim"]),
        pooling=training_config["pooling"],
        num_train_classes=num_train_classes,
        arcface_margin=float(training_config["arcface_margin"]),
        arcface_scale=float(training_config["arcface_scale"]),
    )
    missing, unexpected = model.load_state_dict(filtered_state_dict, strict=False)
    model.to(device)
    processor = viz.load_processor(training_config["model_name"])
    training_config["finetune_missing_keys"] = missing
    training_config["finetune_unexpected_keys"] = unexpected
    return model, processor, training_config


def freeze_everything_except_projection_and_arcface(model: proj.LayoutLMv3TemplateProjectionModel) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.projection_head.parameters():
        parameter.requires_grad = True
    for parameter in model.arcface_head.parameters():
        parameter.requires_grad = True


def load_photograph_documents() -> tuple[list[viz.DocumentSample], list[viz.DocumentSample]]:
    original_pages, query_pages = photo_bench.prepare_photograph_image_data(PHOTO_ROOT, PHOTO_IMAGE_ROOT)
    photo_bench.build_ocr_cache(
        pages=original_pages + query_pages,
        ocr_cache_root=PHOTO_OCR_ROOT,
        tesseract_cmd=TESSERACT_CMD,
        psm=PSM,
        oem=OEM,
        min_confidence=MIN_CONFIDENCE,
    )
    return (
        photo_bench.build_documents(original_pages, PHOTO_OCR_ROOT),
        photo_bench.build_documents(query_pages, PHOTO_OCR_ROOT),
    )


def split_template_names(template_names: list[str]) -> tuple[set[str], set[str]]:
    names = sorted(template_names)
    rng = random.Random(SEED)
    rng.shuffle(names)
    train_count = max(1, int(round(len(names) * 0.8)))
    train_names = set(names[:train_count])
    test_names = set(names[train_count:])
    return train_names, test_names


def degrade_image_once(source_path: Path, output_path: Path, rng: random.Random) -> None:
    if output_path.exists():
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source_path) as image:
        result = ImageOps.exif_transpose(image).convert("RGB")
        result = ImageEnhance.Contrast(result).enhance(rng.uniform(0.82, 1.10))
        result = ImageEnhance.Brightness(result).enhance(rng.uniform(0.86, 1.06))
        if rng.random() < 0.75:
            result = result.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.2, 0.8)))

        array = np.asarray(result).astype(np.int16)
        noise = rng.uniform(3.0, 9.0)
        noise_array = np.random.default_rng(rng.randint(0, 2**31 - 1)).normal(0, noise, array.shape)
        array = np.clip(array + noise_array, 0, 255).astype(np.uint8)
        result = Image.fromarray(array, mode="RGB")
        result.save(output_path, quality=rng.randint(60, 88))
        result.close()


def create_degraded_augmented_documents(train_documents: list[viz.DocumentSample]) -> list[viz.DocumentSample]:
    augmented_documents: list[viz.DocumentSample] = []
    rng = random.Random(SEED + 999)
    manifest: list[dict[str, Any]] = []
    for document in train_documents:
        output_path = DEGRADED_AUG_ROOT / document.stem / f"{Path(document.image_path).stem}_degraded.jpg"
        degrade_image_once(Path(document.image_path), output_path, rng)
        augmented_documents.append(
            viz.DocumentSample(
                subset="photograph_train_degraded_aug",
                stem=document.stem,
                class_label=document.class_label,
                image_path=output_path.resolve(),
                record_id=int(document.record_id) + 100000,
                words=list(document.words),
                boxes=[list(box) for box in document.boxes],
                is_original=document.is_original,
            )
        )
        manifest.append(
            {
                "template_name": document.stem,
                "source_image": str(document.image_path),
                "degraded_image": str(output_path.resolve()),
            }
        )
    write_json(OUTPUT_DIR / "degradation_manifest.json", manifest)
    return augmented_documents


def build_classifier_examples(documents: list[viz.DocumentSample], class_to_id: dict[str, int]) -> list[proj.TemplateExample]:
    return [
        proj.TemplateExample(document=document, class_label=document.class_label, train_label_id=class_to_id[document.class_label])
        for document in documents
        if document.class_label in class_to_id
    ]


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if torch.is_tensor(value) else value
    return moved


def supervised_contrastive_loss(embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    normalized = F.normalize(embeddings, p=2, dim=-1)
    logits = normalized @ normalized.T / SUPCON_TEMPERATURE
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    self_mask = torch.eye(labels.shape[0], device=labels.device, dtype=torch.bool)
    positive_mask = labels.unsqueeze(0).eq(labels.unsqueeze(1)) & ~self_mask
    exp_logits = torch.exp(logits) * (~self_mask)
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp(min=1e-12))
    positive_counts = positive_mask.sum(dim=1)
    valid = positive_counts > 0
    if not torch.any(valid):
        return embeddings.new_tensor(0.0)
    mean_log_prob = (positive_mask * log_prob).sum(dim=1) / positive_counts.clamp(min=1)
    return -mean_log_prob[valid].mean()


def collect_embeddings(
    model: proj.LayoutLMv3TemplateProjectionModel,
    processor: Any,
    documents: list[viz.DocumentSample],
    max_length: int,
    device: torch.device,
) -> np.ndarray:
    class_to_id = {label: index for index, label in enumerate(sorted({document.class_label for document in documents}))}
    examples = build_classifier_examples(documents, class_to_id)
    collator = proj.ProcessorCollator(processor, max_length=max_length)
    loader = DataLoader(
        proj.TemplateDocumentDataset(examples),
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=collator,
    )
    embeddings, ordered_documents = proj.collect_embeddings(model, loader, device)
    if [str(document.image_path) for document in ordered_documents] != [str(document.image_path) for document in documents]:
        raise RuntimeError("Embedding document order changed unexpectedly.")
    return embeddings


def retrieval_metrics(
    query_documents: list[viz.DocumentSample],
    query_embeddings: np.ndarray,
    original_documents: list[viz.DocumentSample],
    original_embeddings: np.ndarray,
    output_dir: Path,
    prefix: str,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    q = query_embeddings / np.maximum(np.linalg.norm(query_embeddings, axis=1, keepdims=True), 1e-12)
    g = original_embeddings / np.maximum(np.linalg.norm(original_embeddings, axis=1, keepdims=True), 1e-12)
    scores = q @ g.T
    gallery_labels = [document.stem for document in original_documents]
    top1 = top5 = top10 = 0
    reciprocal_ranks: list[float] = []
    rows: list[dict[str, Any]] = []
    for query_document, similarities in zip(query_documents, scores, strict=False):
        order = np.argsort(-similarities)
        ranked_labels = [gallery_labels[index] for index in order]
        true_label = query_document.stem
        top1 += int(ranked_labels[0] == true_label)
        top5 += int(true_label in ranked_labels[:5])
        top10 += int(true_label in ranked_labels[:10])
        reciprocal_rank = 0.0
        for rank, label in enumerate(ranked_labels, start=1):
            if label == true_label:
                reciprocal_rank = 1.0 / rank
                break
        reciprocal_ranks.append(reciprocal_rank)
        row: dict[str, Any] = {
            "query_image": str(query_document.image_path),
            "true_template": true_label,
            "predicted_template_top1": ranked_labels[0],
            "top1_score": float(similarities[order[0]]),
            "hit_at_1": ranked_labels[0] == true_label,
        }
        for slot in range(5):
            if slot < len(order):
                row[f"rank_{slot + 1}_template"] = ranked_labels[slot]
                row[f"rank_{slot + 1}_score"] = float(similarities[order[slot]])
        rows.append(row)

    if rows:
        with (output_dir / f"{prefix}_per_query_rankings.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    return {
        "gallery_templates": len(original_documents),
        "query_images": len(query_documents),
        "retrieval_at_1": top1 / max(1, len(query_documents)),
        "retrieval_at_5": top5 / max(1, len(query_documents)),
        "retrieval_at_10": top10 / max(1, len(query_documents)),
        "mrr": float(sum(reciprocal_ranks) / max(1, len(query_documents))),
    }


def evaluate_retrieval(
    model: proj.LayoutLMv3TemplateProjectionModel,
    processor: Any,
    original_documents: list[viz.DocumentSample],
    query_documents: list[viz.DocumentSample],
    max_length: int,
    device: torch.device,
    output_dir: Path,
    prefix: str,
) -> dict[str, Any]:
    original_embeddings = collect_embeddings(model, processor, original_documents, max_length, device)
    query_embeddings = collect_embeddings(model, processor, query_documents, max_length, device)
    np.save(output_dir / f"{prefix}_original_embeddings.npy", original_embeddings)
    np.save(output_dir / f"{prefix}_query_embeddings.npy", query_embeddings)
    return retrieval_metrics(query_documents, query_embeddings, original_documents, original_embeddings, output_dir, prefix)


def discover_converted_pdf_documents() -> tuple[list[viz.DocumentSample], list[viz.DocumentSample]]:
    pages: list[ConvertedPage] = []
    for template_dir in sorted([path for path in CONVERTED_PDF_ROOT.iterdir() if path.is_dir()], key=lambda path: path.name.lower()):
        original_path = template_dir / "original_template.png"
        if not original_path.exists():
            continue
        pages.append(ConvertedPage(template_dir.name, "original", 1, original_path.resolve()))
        for index, query_path in enumerate(sorted(template_dir.glob("page_*.png"), key=lambda path: path.name.lower()), start=2):
            pages.append(ConvertedPage(template_dir.name, "query", index, query_path.resolve()))

    bench.configure_tesseract(TESSERACT_CMD)
    for index, page in enumerate(pages, start=1):
        cache_path = CONVERTED_PDF_OCR_ROOT / page.template_name / f"{page.image_path.stem}.ocr.json"
        if photo_bench.valid_ocr_cache(cache_path):
            continue
        payload = photo_bench.run_photo_tesseract(page.image_path, psm=PSM, oem=OEM, min_confidence=MIN_CONFIDENCE)
        payload.update({"template_name": page.template_name, "role": page.role, "record_id": page.record_id})
        write_json(cache_path, payload)
        if index == 1 or index % 25 == 0 or index == len(pages):
            print(f"converted_pdf OCR cached {index}/{len(pages)}", flush=True)

    originals: list[viz.DocumentSample] = []
    queries: list[viz.DocumentSample] = []
    for page in pages:
        payload = read_json(CONVERTED_PDF_OCR_ROOT / page.template_name / f"{page.image_path.stem}.ocr.json")
        width = int(payload["image_size"]["width"])
        height = int(payload["image_size"]["height"])
        document = viz.DocumentSample(
            subset="converted_pdf",
            stem=page.template_name,
            class_label=page.template_name,
            image_path=page.image_path,
            record_id=page.record_id,
            words=list(payload["words"]),
            boxes=[bench.normalize_box(box, width, height) for box in payload["boxes"]],
            is_original=page.role == "original",
        )
        if page.role == "original":
            originals.append(document)
        else:
            queries.append(document)
    return originals, queries


def train_projection_head(
    model: proj.LayoutLMv3TemplateProjectionModel,
    processor: Any,
    train_documents: list[viz.DocumentSample],
    train_class_to_id: dict[str, int],
    max_length: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    train_examples = build_classifier_examples(train_documents, train_class_to_id)
    collator = proj.ProcessorCollator(processor, max_length=max_length)
    sampler = proj.ClassBalancedBatchSampler(
        train_examples,
        classes_per_batch=CLASSES_PER_BATCH,
        samples_per_class=SAMPLES_PER_CLASS,
        batches_per_epoch=None,
        seed=SEED,
    )
    loader = DataLoader(
        proj.TemplateDocumentDataset(train_examples),
        batch_sampler=sampler,
        num_workers=0,
        collate_fn=collator,
    )

    optimizer = torch.optim.AdamW(
        [
            {"params": model.projection_head.parameters(), "lr": HEAD_LR, "weight_decay": WEIGHT_DECAY},
            {"params": model.arcface_head.parameters(), "lr": HEAD_LR, "weight_decay": WEIGHT_DECAY},
        ]
    )
    total_steps = len(loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * WARMUP_RATIO),
        num_training_steps=total_steps,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history: list[dict[str, Any]] = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        total_arcface = 0.0
        total_supcon = 0.0
        started = time.time()
        for step, batch in enumerate(loader, start=1):
            batch = move_batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    bbox=batch["bbox"],
                    pixel_values=batch["pixel_values"],
                )
                arcface_logits = model.arcface_head(outputs["embeddings"], batch["labels"])
                arcface_loss = F.cross_entropy(arcface_logits, batch["labels"])
                supcon_loss = supervised_contrastive_loss(outputs["embeddings"], batch["labels"])
                loss = ARCFACE_WEIGHT * arcface_loss + SUPCON_WEIGHT * supcon_loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total_loss += float(loss.item())
            total_arcface += float(arcface_loss.item())
            total_supcon += float(supcon_loss.item())
            if step == 1 or step % 20 == 0 or step == len(loader):
                print(
                    f"epoch {epoch}/{EPOCHS} step {step}/{len(loader)} loss={loss.item():.4f} arcface={arcface_loss.item():.4f} supcon={supcon_loss.item():.4f}",
                    flush=True,
                )
        row = {
            "epoch": epoch,
            "loss": total_loss / max(1, len(loader)),
            "arcface_loss": total_arcface / max(1, len(loader)),
            "supcon_loss": total_supcon / max(1, len(loader)),
            "seconds": time.time() - started,
        }
        history.append(row)
        write_json(OUTPUT_DIR / "training_history.json", history)
        print(f"epoch {epoch} complete {row}", flush=True)
    return history


def main() -> None:
    set_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = latest_checkpoint_dir()
    print(f"Using checkpoint: {checkpoint_dir}", flush=True)

    photo_originals, photo_queries = load_photograph_documents()
    template_names = sorted({document.stem for document in photo_originals})
    train_names, test_names = split_template_names(template_names)
    train_documents_base = [document for document in photo_originals + photo_queries if document.stem in train_names]
    test_originals = [document for document in photo_originals if document.stem in test_names]
    test_queries = [document for document in photo_queries if document.stem in test_names]
    train_augmented = create_degraded_augmented_documents(train_documents_base)
    train_documents = train_documents_base + train_augmented
    train_class_to_id = {name: index for index, name in enumerate(sorted(train_names))}

    converted_originals, converted_queries = discover_converted_pdf_documents()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, processor, training_config = load_new_head_model(checkpoint_dir, len(train_class_to_id), device)
    max_length = int(training_config["max_length"])
    write_json(
        OUTPUT_DIR / "split_summary.json",
        {
            "train_templates": sorted(train_names),
            "test_templates": sorted(test_names),
            "train_documents_before_aug": len(train_documents_base),
            "train_documents_after_aug": len(train_documents),
            "photo_test_originals": len(test_originals),
            "photo_test_queries": len(test_queries),
            "converted_pdf_originals": len(converted_originals),
            "converted_pdf_queries": len(converted_queries),
            "checkpoint_dir": str(checkpoint_dir),
            "trainable": "projection_head and arcface_head only; LayoutLMv3 backbone frozen",
        },
    )

    print("Evaluating baseline before finetune...", flush=True)
    baseline_photo = evaluate_retrieval(model, processor, test_originals, test_queries, max_length, device, OUTPUT_DIR, "baseline_photo_test")
    baseline_converted = evaluate_retrieval(model, processor, converted_originals, converted_queries, max_length, device, OUTPUT_DIR, "baseline_converted_pdf")

    freeze_everything_except_projection_and_arcface(model)
    history = train_projection_head(model, processor, train_documents, train_class_to_id, max_length, device)

    print("Evaluating after finetune...", flush=True)
    finetuned_photo = evaluate_retrieval(model, processor, test_originals, test_queries, max_length, device, OUTPUT_DIR, "finetuned_photo_test")
    finetuned_converted = evaluate_retrieval(model, processor, converted_originals, converted_queries, max_length, device, OUTPUT_DIR, "finetuned_converted_pdf")

    summary = {
        "baseline_photo_test": baseline_photo,
        "finetuned_photo_test": finetuned_photo,
        "delta_photo_test": {
            key: finetuned_photo[key] - baseline_photo[key]
            for key in ("retrieval_at_1", "retrieval_at_5", "retrieval_at_10", "mrr")
        },
        "baseline_converted_pdf": baseline_converted,
        "finetuned_converted_pdf": finetuned_converted,
        "delta_converted_pdf": {
            key: finetuned_converted[key] - baseline_converted[key]
            for key in ("retrieval_at_1", "retrieval_at_5", "retrieval_at_10", "mrr")
        },
        "drastic_drop_threshold": -0.05,
        "converted_pdf_drastic_down": (finetuned_converted["retrieval_at_1"] - baseline_converted["retrieval_at_1"]) <= -0.05,
        "history": history,
    }
    write_json(OUTPUT_DIR / "final_comparison_summary.json", summary)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "training_config": training_config,
            "split_summary": read_json(OUTPUT_DIR / "split_summary.json"),
            "final_comparison_summary": summary,
        },
        OUTPUT_DIR / "projection_head_finetuned_model.pt",
    )
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Saved outputs to {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
