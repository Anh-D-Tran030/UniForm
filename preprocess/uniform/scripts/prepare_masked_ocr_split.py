from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import time
from pathlib import Path
from typing import Any


DEFAULT_DEGRADED_ROOT = Path(r"A:\RealForm\processed\synthetic_fill_images_10k_folders_degraded")
DEFAULT_SOURCE_OCR_ROOT = Path(r"A:\RealForm\processed\synthetic_fill_images_10k_folders_degraded_ocr_cache\ocr_json")
DEFAULT_OUTPUT_ROOT = Path(r"A:\RealForm\processed\synthetic_fill_images_10k_folders_degraded_ocr_split_9400_300_300_augmented")
DEFAULT_TRAIN_TEMPLATES = 9400
DEFAULT_VAL_TEMPLATES = 300
DEFAULT_TEST_TEMPLATES = 300
DEFAULT_SEED = 42
IMAGE_GLOB = "fill_*.png"
OCR_GLOB = "*.ocr.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split degraded CommonForms templates into train/val/test and build a processed OCR cache "
            "where train templates receive masking/corruption augmentations."
        )
    )
    parser.add_argument("--degraded-root", type=Path, default=DEFAULT_DEGRADED_ROOT)
    parser.add_argument("--source-ocr-root", type=Path, default=DEFAULT_SOURCE_OCR_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--train-templates", type=int, default=DEFAULT_TRAIN_TEMPLATES)
    parser.add_argument("--val-templates", type=int, default=DEFAULT_VAL_TEMPLATES)
    parser.add_argument("--test-templates", type=int, default=DEFAULT_TEST_TEMPLATES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--poll-seconds", type=int, default=180)
    parser.add_argument("--idle-polls-before-exit", type=int, default=3)
    parser.add_argument("--watch", action="store_true")
    return parser.parse_args()


def stable_seed(text: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{text}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def log(message: str, log_path: Path) -> None:
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    line = f"[{timestamp}] {message}"
    print(line, flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict[str, Any] | list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def count_matching_files(path: Path, pattern: str) -> int:
    if not path.exists():
        return 0
    return len(list(path.glob(pattern)))


def discover_template_names(degraded_root: Path) -> list[str]:
    templates = [
        path.name
        for path in sorted(degraded_root.iterdir(), key=lambda item: item.name.lower())
        if path.is_dir() and count_matching_files(path, IMAGE_GLOB) == 10
    ]
    if not templates:
        raise FileNotFoundError(f"No complete degraded template folders found under {degraded_root}.")
    return templates


def build_split_lists(template_names: list[str], train_count: int, val_count: int, test_count: int, seed: int) -> dict[str, list[str]]:
    total = train_count + val_count + test_count
    if len(template_names) < total:
        raise RuntimeError(f"Need {total} templates but only found {len(template_names)}.")
    shuffled = list(template_names)
    random.Random(seed).shuffle(shuffled)
    return {
        "train": shuffled[:train_count],
        "val": shuffled[train_count : train_count + val_count],
        "test": shuffled[train_count + val_count : total],
    }


def ensure_split_manifest(output_root: Path, split_lists: dict[str, list[str]], seed: int) -> None:
    manifest_path = output_root / "split_summary.json"
    if manifest_path.exists():
        return
    write_json(
        manifest_path,
        {
            "seed": seed,
            "train_count": len(split_lists["train"]),
            "val_count": len(split_lists["val"]),
            "test_count": len(split_lists["test"]),
            "train_templates": split_lists["train"],
            "val_templates": split_lists["val"],
            "test_templates": split_lists["test"],
        },
    )
    for split_name, template_names in split_lists.items():
        write_json(output_root / f"{split_name}_templates.json", template_names)


def choose_indices(template_name: str, seed: int) -> tuple[set[int], set[int]]:
    rng = random.Random(stable_seed(template_name, seed))
    indices = list(range(10))
    rng.shuffle(indices)
    mask25 = set(indices[:5])
    mask20_corrupt10 = set(indices[5:7])
    return mask25, mask20_corrupt10


def choose_token_indices(word_count: int, fraction: float, rng: random.Random, excluded: set[int] | None = None) -> list[int]:
    valid_indices = [index for index in range(word_count) if excluded is None or index not in excluded]
    if not valid_indices:
        return []
    target = max(1, round(word_count * fraction))
    target = min(target, len(valid_indices))
    return sorted(rng.sample(valid_indices, k=target))


def corrupt_word(word: str, rng: random.Random) -> str:
    cleaned = word.strip()
    if not cleaned or cleaned == "[MASK]":
        return cleaned
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    operation = rng.choice(["replace", "delete", "insert", "swap"])
    characters = list(cleaned)
    if operation == "replace" and characters:
        position = rng.randrange(len(characters))
        characters[position] = rng.choice(alphabet).upper() if characters[position].isupper() else rng.choice(alphabet)
        candidate = "".join(characters)
        return candidate if candidate != cleaned else candidate + rng.choice(alphabet)
    if operation == "delete" and len(characters) > 1:
        position = rng.randrange(len(characters))
        del characters[position]
        return "".join(characters)
    if operation == "insert":
        position = rng.randrange(len(characters) + 1)
        characters.insert(position, rng.choice(alphabet))
        return "".join(characters)
    if operation == "swap" and len(characters) > 1:
        position = rng.randrange(len(characters) - 1)
        characters[position], characters[position + 1] = characters[position + 1], characters[position]
        return "".join(characters)
    return cleaned + rng.choice(alphabet)


def transform_payload(payload: dict[str, Any], mode: str, rng: random.Random) -> tuple[dict[str, Any], dict[str, Any]]:
    transformed = json.loads(json.dumps(payload))
    words = list(transformed.get("words", []))
    word_count = len(words)
    masked_indices: list[int] = []
    corrupted_indices: list[int] = []

    if words == ["[EMPTY]"] or word_count == 0:
        transformed["augmentation"] = {
            "mode": mode,
            "masked_indices": masked_indices,
            "corrupted_indices": corrupted_indices,
            "source_word_count": word_count,
        }
        return transformed, transformed["augmentation"]

    if mode == "mask25":
        masked_indices = choose_token_indices(word_count, 0.25, rng)
        for index in masked_indices:
            words[index] = "[MASK]"
    elif mode == "mask20_corrupt10":
        masked_indices = choose_token_indices(word_count, 0.20, rng)
        for index in masked_indices:
            words[index] = "[MASK]"
        corrupted_indices = choose_token_indices(word_count, 0.10, rng, excluded=set(masked_indices))
        for index in corrupted_indices:
            words[index] = corrupt_word(words[index], rng)

    transformed["words"] = words
    transformed["word_count"] = len(words)
    transformed["augmentation"] = {
        "mode": mode,
        "masked_indices": masked_indices,
        "corrupted_indices": corrupted_indices,
        "source_word_count": word_count,
    }
    return transformed, transformed["augmentation"]


def process_template(
    template_name: str,
    split_name: str,
    source_ocr_root: Path,
    output_root: Path,
    seed: int,
) -> dict[str, Any] | None:
    source_dir = source_ocr_root / template_name
    if count_matching_files(source_dir, OCR_GLOB) != 10:
        return None

    target_dir = output_root / split_name / template_name
    manifest_path = target_dir / "template_manifest.json"
    if manifest_path.exists() and count_matching_files(target_dir, OCR_GLOB) == 10:
        return {"template_name": template_name, "split": split_name, "status": "already_processed"}

    target_dir.mkdir(parents=True, exist_ok=True)
    image_paths = sorted(source_dir.glob("fill_*.ocr.json"), key=lambda path: path.name.lower())
    mask25_indices, mask20_corrupt10_indices = choose_indices(template_name, seed)
    image_manifests: list[dict[str, Any]] = []
    template_rng = random.Random(stable_seed(template_name + ":transform", seed))

    for image_index, source_path in enumerate(image_paths):
        payload = read_json(source_path)
        if split_name == "train":
            if image_index in mask25_indices:
                mode = "mask25"
            elif image_index in mask20_corrupt10_indices:
                mode = "mask20_corrupt10"
            else:
                mode = "clean"
        else:
            mode = "clean"

        if mode == "clean":
            transformed = json.loads(json.dumps(payload))
            transformed["augmentation"] = {
                "mode": "clean",
                "masked_indices": [],
                "corrupted_indices": [],
                "source_word_count": int(payload.get("word_count", len(payload.get("words", [])))),
            }
            augmentation = transformed["augmentation"]
        else:
            transformed, augmentation = transform_payload(payload, mode, template_rng)

        transformed["dataset_split"] = split_name
        transformed["template_name"] = template_name
        transformed["source_ocr_path"] = str(source_path)
        write_json(target_dir / source_path.name, transformed)
        image_manifests.append(
            {
                "image_name": source_path.name,
                "mode": augmentation["mode"],
                "masked_count": len(augmentation["masked_indices"]),
                "corrupted_count": len(augmentation["corrupted_indices"]),
                "word_count": int(transformed.get("word_count", len(transformed.get("words", [])))),
            }
        )

    write_json(
        manifest_path,
        {
            "template_name": template_name,
            "split": split_name,
            "mask25_images": sorted(mask25_indices) if split_name == "train" else [],
            "mask20_corrupt10_images": sorted(mask20_corrupt10_indices) if split_name == "train" else [],
            "images": image_manifests,
        },
    )
    return {"template_name": template_name, "split": split_name, "status": "processed"}


def save_state(output_root: Path, split_lists: dict[str, list[str]], stats: dict[str, Any]) -> None:
    payload = {
        "split_counts": {split_name: len(template_names) for split_name, template_names in split_lists.items()},
        "stats": stats,
    }
    write_json(output_root / "processing_state.json", payload)


def main() -> None:
    args = parse_args()
    degraded_root = args.degraded_root.resolve()
    source_ocr_root = args.source_ocr_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    log_path = output_root / "prepare_masked_ocr_split.log"

    template_names = discover_template_names(degraded_root)
    split_lists = build_split_lists(
        template_names,
        train_count=args.train_templates,
        val_count=args.val_templates,
        test_count=args.test_templates,
        seed=args.seed,
    )
    ensure_split_manifest(output_root, split_lists, args.seed)

    idle_polls = 0
    while True:
        processed_this_pass = 0
        stats: dict[str, Any] = {"ready_source_templates": 0, "processed_templates": 0, "pending_templates": 0}
        for split_name, names in split_lists.items():
            split_processed = 0
            split_pending = 0
            split_ready = 0
            for template_name in names:
                source_dir = source_ocr_root / template_name
                if count_matching_files(source_dir, OCR_GLOB) == 10:
                    split_ready += 1
                result = process_template(template_name, split_name, source_ocr_root, output_root, args.seed)
                if result is None:
                    split_pending += 1
                    continue
                if result["status"] == "processed":
                    processed_this_pass += 1
                split_processed += 1

            stats[f"{split_name}_ready_source_templates"] = split_ready
            stats[f"{split_name}_processed_templates"] = split_processed
            stats[f"{split_name}_pending_templates"] = len(names) - split_processed
            stats["ready_source_templates"] += split_ready
            stats["processed_templates"] += split_processed
            stats["pending_templates"] += len(names) - split_processed

        save_state(output_root, split_lists, stats)
        log(
            (
                "Pass complete: processed_now=%s processed_total=%s/%s "
                "ready_source=%s pending=%s"
            )
            % (
                processed_this_pass,
                stats["processed_templates"],
                len(split_lists["train"]) + len(split_lists["val"]) + len(split_lists["test"]),
                stats["ready_source_templates"],
                stats["pending_templates"],
            ),
            log_path,
        )

        if not args.watch:
            break
        if stats["pending_templates"] == 0:
            log("All templates processed. Exiting.", log_path)
            break
        if processed_this_pass == 0:
            idle_polls += 1
            log(f"No new templates processed in this pass. idle_polls={idle_polls}/{args.idle_polls_before_exit}", log_path)
            if idle_polls >= args.idle_polls_before_exit:
                log("Reached idle poll limit. Exiting watch mode.", log_path)
                break
        else:
            idle_polls = 0
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
