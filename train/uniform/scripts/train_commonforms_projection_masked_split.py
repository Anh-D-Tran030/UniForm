from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup


SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import train_commonforms_template_projection as commontrain
import train_layoutlmv3_template_projection as baseproj
import visualize_layoutlmv3_template_embeddings as viz


DEFAULT_MODEL_NAME = "microsoft/layoutlmv3-base"
DEFAULT_SPLIT_OCR_ROOT = Path(r"A:\RealForm\processed\synthetic_fill_images_10k_folders_degraded_ocr_split_9400_300_300_augmented")
DEFAULT_DEGRADED_ROOT = Path(r"A:\RealForm\processed\synthetic_fill_images_10k_folders_degraded")
DEFAULT_ORIGINAL_ROOT = Path(r"A:\RealForm\processed\CommonFormsEnglish\images")
DEFAULT_ORIGINAL_OCR_ROOT = Path(r"A:\RealForm\processed\CommonFormsEnglish_original_ocr_cache_projection")
DEFAULT_OUTPUT_DIR = Path(r"A:\RealForm\outputs\layoutlmv3_projection_commonforms_masked_split_10ep")
DEFAULT_TESSERACT_CMD = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")


@dataclass(frozen=True)
class SplitSelection:
    stem: str
    query_dir: Path
    original_image_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a LayoutLMv3 projection model on degraded CommonForms images using the prepared "
            "masked/corrupted OCR train/val/test split."
        )
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--split-ocr-root", type=Path, default=DEFAULT_SPLIT_OCR_ROOT)
    parser.add_argument("--degraded-root", type=Path, default=DEFAULT_DEGRADED_ROOT)
    parser.add_argument("--original-root", type=Path, default=DEFAULT_ORIGINAL_ROOT)
    parser.add_argument("--original-ocr-root", type=Path, default=DEFAULT_ORIGINAL_OCR_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tesseract-cmd", type=Path, default=DEFAULT_TESSERACT_CMD)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--pooling", choices=["cls", "mean"], default="cls")
    parser.add_argument("--projection-dim", type=int, default=128)
    parser.add_argument("--classes-per-batch", type=int, default=4)
    parser.add_argument("--samples-per-class", type=int, default=2)
    parser.add_argument("--batches-per-epoch", type=int)
    parser.add_argument("--train-backbone-lr", type=float, default=2e-5)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--supcon-temperature", type=float, default=0.07)
    parser.add_argument("--arcface-margin", type=float, default=0.2)
    parser.add_argument("--arcface-scale", type=float, default=30.0)
    parser.add_argument("--arcface-weight", type=float, default=0.2)
    parser.add_argument("--supcon-weight", type=float, default=0.8)
    parser.add_argument("--unfreeze-last-n", type=int, default=4)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--seen-val-fraction", type=float, default=0.1)
    parser.add_argument("--train-retrieval-sample-count", type=int, default=285)
    parser.add_argument("--psm", type=int, default=11)
    parser.add_argument("--oem", type=int, default=3)
    parser.add_argument("--min-confidence", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--resume-from",
        type=Path,
        help="Optional checkpoint path to resume from, such as last_projection_model.pt or best_projection_model.pt.",
    )
    return parser.parse_args()


def discover_split_selections(
    split_name: str,
    split_ocr_root: Path,
    degraded_root: Path,
    original_image_index: dict[str, Path],
) -> list[SplitSelection]:
    split_dir = split_ocr_root / split_name
    if not split_dir.exists():
        return []
    selections: list[SplitSelection] = []
    for template_dir in sorted([path for path in split_dir.iterdir() if path.is_dir()], key=lambda path: path.name.lower()):
        if commontrain.count_matching_files(template_dir, "*.ocr.json") != 10:
            continue
        query_dir = degraded_root / template_dir.name
        if commontrain.count_matching_files(query_dir, commontrain.IMAGE_GLOB) != 10:
            continue
        original_image_path = original_image_index.get(template_dir.name)
        if original_image_path is None:
            continue
        selections.append(
            SplitSelection(
                stem=template_dir.name,
                query_dir=query_dir,
                original_image_path=original_image_path,
            )
        )
    return selections


def ensure_original_cache(
    selections: list[SplitSelection],
    original_ocr_root: Path,
    tesseract_cmd: Path,
    psm: int,
    oem: int,
    min_confidence: float,
    logger: Any,
) -> None:
    proxy = [commontrain.TemplateSelection(item.stem, item.query_dir, item.original_image_path) for item in selections]
    commontrain.ensure_original_ocr_cache(proxy, original_ocr_root, tesseract_cmd, psm, oem, min_confidence, logger)


def build_documents_for_split(
    selections: list[SplitSelection],
    split_ocr_root: Path,
    split_name: str,
    original_ocr_root: Path,
) -> tuple[list[viz.DocumentSample], list[viz.DocumentSample]]:
    query_documents: list[viz.DocumentSample] = []
    original_documents: list[viz.DocumentSample] = []
    for selection in selections:
        original_payload = commontrain.read_json(commontrain.make_original_cache_path(original_ocr_root, selection.stem))
        original_width = int(original_payload["image_size"]["width"])
        original_height = int(original_payload["image_size"]["height"])
        original_documents.append(
            viz.DocumentSample(
                subset="commonforms_original",
                stem=selection.stem,
                class_label=selection.stem,
                image_path=selection.original_image_path.resolve(),
                record_id=0,
                words=list(original_payload["words"]),
                boxes=[commontrain.normalize_box(box, original_width, original_height) for box in original_payload["boxes"]],
                is_original=True,
            )
        )

        query_cache_dir = split_ocr_root / split_name / selection.stem
        for record_id, image_path in enumerate(sorted(selection.query_dir.glob(commontrain.IMAGE_GLOB), key=lambda path: path.name.lower()), start=1):
            payload = commontrain.read_json(query_cache_dir / f"{image_path.stem}.ocr.json")
            width = int(payload["image_size"]["width"])
            height = int(payload["image_size"]["height"])
            query_documents.append(
                viz.DocumentSample(
                    subset=f"commonforms_degraded_{split_name}",
                    stem=selection.stem,
                    class_label=selection.stem,
                    image_path=image_path.resolve(),
                    record_id=record_id,
                    words=list(payload["words"]),
                    boxes=[commontrain.normalize_box(box, width, height) for box in payload["boxes"]],
                    is_original=False,
                )
            )
    return query_documents, original_documents


def sample_train_query_documents(
    documents: list[viz.DocumentSample],
    sample_count: int,
    seed: int,
) -> list[viz.DocumentSample]:
    if sample_count <= 0 or len(documents) <= sample_count:
        return list(documents)
    rng = commontrain.random.Random(seed)
    chosen_indices = sorted(rng.sample(range(len(documents)), k=sample_count))
    return [documents[index] for index in chosen_indices]


def load_history_rows(history_path: Path) -> list[dict[str, Any]]:
    if not history_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with history_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw_row in reader:
            parsed_row: dict[str, Any] = {}
            for key, value in raw_row.items():
                if value is None:
                    parsed_row[key] = value
                    continue
                try:
                    if key == "epoch":
                        parsed_row[key] = int(value)
                    else:
                        parsed_row[key] = float(value)
                except ValueError:
                    parsed_row[key] = value
            rows.append(parsed_row)
    return rows


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = baseproj.build_logger(output_dir)
    baseproj.set_seed(args.seed)

    split_ocr_root = args.split_ocr_root.resolve()
    degraded_root = args.degraded_root.resolve()
    original_root = args.original_root.resolve()
    original_image_index = commontrain.build_original_image_index(original_root)

    train_selections = discover_split_selections("train", split_ocr_root, degraded_root, original_image_index)
    val_selections = discover_split_selections("val", split_ocr_root, degraded_root, original_image_index)
    test_selections = discover_split_selections("test", split_ocr_root, degraded_root, original_image_index)
    all_selections = train_selections + val_selections + test_selections
    if not train_selections or not val_selections or not test_selections:
        raise RuntimeError(
            f"Need non-empty train/val/test splits. Found train={len(train_selections)} val={len(val_selections)} test={len(test_selections)}."
        )

    commontrain.write_json(
        output_dir / "split_summary.json",
        {
            "train_templates": [selection.stem for selection in train_selections],
            "val_templates": [selection.stem for selection in val_selections],
            "test_templates": [selection.stem for selection in test_selections],
            "split_ocr_root": str(split_ocr_root),
        },
    )
    logger.info(
        "Using processed split templates: train=%s val=%s test=%s",
        len(train_selections),
        len(val_selections),
        len(test_selections),
    )

    ensure_original_cache(
        selections=all_selections,
        original_ocr_root=args.original_ocr_root.resolve(),
        tesseract_cmd=args.tesseract_cmd.resolve(),
        psm=args.psm,
        oem=args.oem,
        min_confidence=args.min_confidence,
        logger=logger,
    )

    logger.info("Loading OCR-backed documents for processed train/val/test splits.")
    train_query_documents, train_original_documents = build_documents_for_split(train_selections, split_ocr_root, "train", args.original_ocr_root.resolve())
    val_query_documents, val_original_documents = build_documents_for_split(val_selections, split_ocr_root, "val", args.original_ocr_root.resolve())
    test_query_documents, test_original_documents = build_documents_for_split(test_selections, split_ocr_root, "test", args.original_ocr_root.resolve())
    train_query_documents_for_retrieval = sample_train_query_documents(
        train_query_documents,
        sample_count=args.train_retrieval_sample_count,
        seed=args.seed,
    )
    logger.info(
        "Loaded documents: train_query=%s train_query_retrieval=%s train_original=%s val_query=%s val_original=%s test_query=%s test_original=%s",
        len(train_query_documents),
        len(train_query_documents_for_retrieval),
        len(train_original_documents),
        len(val_query_documents),
        len(val_original_documents),
        len(test_query_documents),
        len(test_original_documents),
    )

    train_class_labels = sorted({document.class_label for document in train_query_documents})
    train_class_to_id = {class_label: index for index, class_label in enumerate(train_class_labels)}
    all_train_examples = commontrain.build_classifier_examples(train_query_documents, train_class_to_id)
    train_examples, seen_val_examples = baseproj.split_train_val_examples(all_train_examples, args.seen_val_fraction, args.seed)

    processor = viz.load_processor(args.model_name)
    collator = baseproj.ProcessorCollator(processor, args.max_length)

    train_dataset = baseproj.TemplateDocumentDataset(train_examples)
    seen_val_dataset = baseproj.TemplateDocumentDataset(seen_val_examples)
    train_sampler = baseproj.ClassBalancedBatchSampler(
        train_examples,
        classes_per_batch=args.classes_per_batch,
        samples_per_class=args.samples_per_class,
        batches_per_epoch=args.batches_per_epoch,
        seed=args.seed,
    )
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=0, collate_fn=collator)
    train_accuracy_loader = DataLoader(train_dataset, batch_size=args.eval_batch_size, shuffle=False, num_workers=0, collate_fn=collator)
    seen_val_loader = DataLoader(seen_val_dataset, batch_size=args.eval_batch_size, shuffle=False, num_workers=0, collate_fn=collator)
    train_retrieval_loader = commontrain.build_inference_loader(
        train_query_documents_for_retrieval + train_original_documents,
        collator,
        args.eval_batch_size,
    )
    val_retrieval_loader = commontrain.build_inference_loader(val_query_documents + val_original_documents, collator, args.eval_batch_size)
    test_retrieval_loader = commontrain.build_inference_loader(test_query_documents + test_original_documents, collator, args.eval_batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)
    model = baseproj.LayoutLMv3TemplateProjectionModel(
        model_name=args.model_name,
        projection_dim=args.projection_dim,
        pooling=args.pooling,
        num_train_classes=len(train_class_labels),
        arcface_margin=args.arcface_margin,
        arcface_scale=args.arcface_scale,
    )
    baseproj.freeze_backbone_except_top_layers(model, args.unfreeze_last_n)
    model.to(device)

    parameter_summary = baseproj.summarize_trainable_parameters(model)
    commontrain.write_json(output_dir / "model_parameter_summary.json", parameter_summary)
    optimizer = baseproj.create_optimizer(model, args)
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_metric = float("-inf")
    best_val_metrics: dict[str, Any] | None = None
    history_rows: list[dict[str, Any]] = []
    start_epoch = 1
    resume_from_path: Path | None = None

    if args.resume_from:
        resume_from_path = args.resume_from.resolve()
        if not resume_from_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_from_path}")
        checkpoint = torch.load(resume_from_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        best_metric = float(checkpoint.get("best_metric", float("-inf")))
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        history_rows = load_history_rows(output_dir / "training_history.csv")
        logger.info(
            "Resumed from checkpoint %s at epoch %s. Next epoch=%s best_metric=%.4f history_rows=%s",
            resume_from_path,
            checkpoint.get("epoch"),
            start_epoch,
            best_metric,
            len(history_rows),
        )

    training_config = vars(args).copy()
    training_config["device"] = str(device)
    training_config["total_steps"] = total_steps
    training_config["warmup_steps"] = warmup_steps
    training_config["effective_train_templates"] = len(train_selections)
    training_config["effective_val_templates"] = len(val_selections)
    training_config["effective_test_templates"] = len(test_selections)
    training_config["effective_train_query_documents"] = len(train_query_documents)
    training_config["effective_train_query_documents_for_retrieval"] = len(train_query_documents_for_retrieval)
    training_config["resume_from"] = str(resume_from_path) if resume_from_path else None
    training_config["resume_start_epoch"] = start_epoch
    commontrain.write_json(output_dir / "training_config.json", training_config)
    commontrain.write_json(
        output_dir / "train_retrieval_sample_rows.json",
        [
            {
                "template_stem": document.stem,
                "record_id": document.record_id,
                "image_path": str(document.image_path),
            }
            for document in train_query_documents_for_retrieval
        ],
    )

    for epoch_index in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()
        model.train()
        total_loss = 0.0
        total_arcface_loss = 0.0
        total_supcon_loss = 0.0
        step_count = 0

        for batch_index, batch in enumerate(train_loader, start=1):
            batch = baseproj.move_batch_to_device(batch, device)
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
                supcon_loss = baseproj.supervised_contrastive_loss(outputs["embeddings"], batch["labels"], temperature=args.supcon_temperature)
                total_batch_loss = args.arcface_weight * arcface_loss + args.supcon_weight * supcon_loss

            scaler.scale(total_batch_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += float(total_batch_loss.item())
            total_arcface_loss += float(arcface_loss.item())
            total_supcon_loss += float(supcon_loss.item())
            step_count += 1
            if batch_index == 1 or batch_index % 20 == 0 or batch_index == len(train_loader):
                logger.info(
                    "Epoch %s step %s/%s loss=%.4f arcface=%.4f supcon=%.4f",
                    epoch_index,
                    batch_index,
                    len(train_loader),
                    total_batch_loss.item(),
                    arcface_loss.item(),
                    supcon_loss.item(),
                )

        train_classifier_accuracy = baseproj.evaluate_classifier_accuracy(model, train_accuracy_loader, device)
        seen_val_classifier_accuracy = baseproj.evaluate_classifier_accuracy(model, seen_val_loader, device)

        train_embeddings_all, _ = baseproj.collect_embeddings(model, train_retrieval_loader, device)
        train_query_embeddings = train_embeddings_all[: len(train_query_documents_for_retrieval)]
        train_original_embeddings = train_embeddings_all[len(train_query_documents_for_retrieval) :]
        train_metrics = commontrain.compute_retrieval_vs_original(
            train_query_documents_for_retrieval, train_query_embeddings, train_original_documents, train_original_embeddings
        )

        val_embeddings_all, _ = baseproj.collect_embeddings(model, val_retrieval_loader, device)
        val_query_embeddings = val_embeddings_all[: len(val_query_documents)]
        val_original_embeddings = val_embeddings_all[len(val_query_documents) :]
        val_metrics = commontrain.compute_retrieval_vs_original(
            val_query_documents, val_query_embeddings, val_original_documents, val_original_embeddings
        )

        history_rows.append(
            {
                "epoch": epoch_index,
                "train_loss": total_loss / max(1, step_count),
                "train_arcface_loss": total_arcface_loss / max(1, step_count),
                "train_supcon_loss": total_supcon_loss / max(1, step_count),
                "train_classifier_accuracy": train_classifier_accuracy,
                "seen_val_classifier_accuracy": seen_val_classifier_accuracy,
                "train_retrieval_at_1": train_metrics["retrieval_at_1"],
                "train_retrieval_at_5": train_metrics["retrieval_at_5"],
                "val_retrieval_at_1": val_metrics["retrieval_at_1"],
                "val_retrieval_at_5": val_metrics["retrieval_at_5"],
                "val_retrieval_at_10": val_metrics["retrieval_at_10"],
                "epoch_minutes": (time.time() - epoch_start) / 60.0,
            }
        )
        commontrain.save_history_csv(output_dir, history_rows)
        commontrain.plot_history(output_dir, history_rows)
        logger.info(
            "Epoch %s complete: train_loss=%.4f train_cls_acc=%.4f seen_val_cls_acc=%.4f train_r@1=%.4f val_r@1=%.4f val_r@5=%.4f",
            epoch_index,
            history_rows[-1]["train_loss"],
            train_classifier_accuracy,
            seen_val_classifier_accuracy,
            train_metrics["retrieval_at_1"],
            val_metrics["retrieval_at_1"],
            val_metrics["retrieval_at_5"],
        )

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "epoch": epoch_index,
                "best_metric": best_metric,
                "args": training_config,
            },
            output_dir / "last_projection_model.pt",
        )

        if val_metrics["retrieval_at_1"] > best_metric:
            best_metric = float(val_metrics["retrieval_at_1"])
            best_val_metrics = val_metrics
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "epoch": epoch_index,
                    "best_metric": best_metric,
                    "args": training_config,
                },
                output_dir / "best_projection_model.pt",
            )
            logger.info("Saved new best checkpoint at epoch %s with val_retrieval_at_1=%.4f.", epoch_index, best_metric)

    best_checkpoint_path = output_dir / "best_projection_model.pt"
    if not best_checkpoint_path.exists():
        if resume_from_path is None:
            raise FileNotFoundError(f"Best checkpoint was not found at {best_checkpoint_path}")
        best_checkpoint_path = resume_from_path
    checkpoint = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info("Loaded best checkpoint from epoch %s.", checkpoint["epoch"])

    train_embeddings_all, _ = baseproj.collect_embeddings(model, train_retrieval_loader, device)
    train_query_embeddings = train_embeddings_all[: len(train_query_documents_for_retrieval)]
    train_original_embeddings = train_embeddings_all[len(train_query_documents_for_retrieval) :]
    final_train_metrics = commontrain.compute_retrieval_vs_original(
        train_query_documents_for_retrieval, train_query_embeddings, train_original_documents, train_original_embeddings
    )
    commontrain.save_retrieval_artifacts(
        output_dir,
        "train_retrieval",
        final_train_metrics,
        train_query_documents_for_retrieval,
        train_query_embeddings,
        train_original_documents,
        train_original_embeddings,
    )

    val_embeddings_all, _ = baseproj.collect_embeddings(model, val_retrieval_loader, device)
    val_query_embeddings = val_embeddings_all[: len(val_query_documents)]
    val_original_embeddings = val_embeddings_all[len(val_query_documents) :]
    final_val_metrics = commontrain.compute_retrieval_vs_original(
        val_query_documents, val_query_embeddings, val_original_documents, val_original_embeddings
    )
    commontrain.save_retrieval_artifacts(
        output_dir, "val_retrieval", final_val_metrics, val_query_documents, val_query_embeddings, val_original_documents, val_original_embeddings
    )

    test_embeddings_all, _ = baseproj.collect_embeddings(model, test_retrieval_loader, device)
    test_query_embeddings = test_embeddings_all[: len(test_query_documents)]
    test_original_embeddings = test_embeddings_all[len(test_query_documents) :]
    final_test_metrics = commontrain.compute_retrieval_vs_original(
        test_query_documents, test_query_embeddings, test_original_documents, test_original_embeddings
    )
    commontrain.save_retrieval_artifacts(
        output_dir, "test_retrieval", final_test_metrics, test_query_documents, test_query_embeddings, test_original_documents, test_original_embeddings
    )

    final_summary = {
        "best_epoch": checkpoint["epoch"],
        "best_val_retrieval_at_1": best_metric,
        "best_val_metrics": best_val_metrics,
        "final_train_metrics": final_train_metrics,
        "final_val_metrics": final_val_metrics,
        "final_test_metrics": final_test_metrics,
        "effective_train_templates": len(train_selections),
        "effective_val_templates": len(val_selections),
        "effective_test_templates": len(test_selections),
    }
    commontrain.write_json(output_dir / "final_metrics_summary.json", final_summary)
    logger.info("Final train retrieval@1=%.4f", final_train_metrics["retrieval_at_1"])
    logger.info("Final val retrieval@1=%.4f", final_val_metrics["retrieval_at_1"])
    logger.info("Final test retrieval@1=%.4f", final_test_metrics["retrieval_at_1"])
    logger.info("Saved final metrics summary to %s", output_dir / "final_metrics_summary.json")


if __name__ == "__main__":
    main()
