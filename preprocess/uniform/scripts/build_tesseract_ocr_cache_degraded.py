from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytesseract
from PIL import Image, ImageDraw, ImageOps


DEFAULT_DEGRADED_ROOT = Path(r"A:\RealForm\processed\synthetic_fill_images_10k_folders_degraded")
DEFAULT_CACHE_ROOT = Path(r"A:\RealForm\processed\synthetic_fill_images_10k_folders_degraded_ocr_cache")
DEFAULT_TESSERACT_CMD = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
IMAGE_GLOB = "fill_*.png"


@dataclass(frozen=True)
class OcrTarget:
    template_stem: str
    image_path: Path


def log(message: str) -> None:
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    print(f"[{timestamp}] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a Tesseract OCR cache for degraded synthetic fill images."
    )
    parser.add_argument("--degraded-root", type=Path, default=DEFAULT_DEGRADED_ROOT)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--tesseract-cmd", type=Path, default=DEFAULT_TESSERACT_CMD)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--limit-templates", type=int)
    parser.add_argument("--limit-images", type=int)
    parser.add_argument("--overlay-count", type=int, default=8)
    parser.add_argument("--psm", type=int, default=11, help="Tesseract page segmentation mode.")
    parser.add_argument("--oem", type=int, default=3, help="Tesseract OCR engine mode.")
    parser.add_argument("--min-confidence", type=float, default=30.0)
    parser.add_argument("--watch", action="store_true", help="Keep scanning for new degraded images.")
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--idle-polls", type=int, default=3)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict[str, Any] | list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def configure_tesseract(executable_path: Path) -> None:
    if not executable_path.exists():
        raise FileNotFoundError(f"Tesseract executable was not found at {executable_path}.")
    pytesseract.pytesseract.tesseract_cmd = str(executable_path)


def discover_targets(
    degraded_root: Path,
    limit_templates: int | None,
    limit_images: int | None,
) -> list[OcrTarget]:
    targets: list[OcrTarget] = []
    template_dirs = sorted(
        [path for path in degraded_root.iterdir() if path.is_dir()],
        key=lambda path: path.name.lower(),
    )
    if limit_templates is not None:
        template_dirs = template_dirs[:limit_templates]

    for template_dir in template_dirs:
        image_paths = sorted(template_dir.glob(IMAGE_GLOB), key=lambda path: path.name.lower())
        if limit_images is not None:
            image_paths = image_paths[:limit_images]
        for image_path in image_paths:
            targets.append(OcrTarget(template_stem=template_dir.name, image_path=image_path))
    return targets


def make_cache_path(cache_root: Path, target: OcrTarget) -> Path:
    return cache_root / "ocr_json" / target.template_stem / f"{target.image_path.stem}.ocr.json"


def make_overlay_path(cache_root: Path, target: OcrTarget) -> Path:
    return cache_root / "proof_overlays" / target.template_stem / f"{target.image_path.stem}.png"


def run_tesseract(image_path: Path, psm: int, oem: int, min_confidence: float) -> dict[str, Any]:
    with Image.open(image_path) as image:
        rgb_image = image.convert("RGB")
        width, height = rgb_image.size
        scale_factor = 2.0
        preprocessed_image = ImageOps.autocontrast(rgb_image.convert("L")).resize(
            (int(width * scale_factor), int(height * scale_factor)),
            resample=Image.Resampling.BICUBIC,
        )
        config = f"--psm {psm} --oem {oem}"
        data = pytesseract.image_to_data(
            preprocessed_image,
            output_type=pytesseract.Output.DICT,
            config=config,
            lang="eng",
        )

    words: list[str] = []
    boxes: list[list[int]] = []
    confidences: list[float] = []

    for index in range(len(data.get("text", []))):
        text = str(data["text"][index]).strip()
        if not text:
            continue
        try:
            confidence = float(data["conf"][index])
        except (TypeError, ValueError):
            confidence = -1.0
        if confidence < min_confidence:
            continue

        left = int(round(int(data["left"][index]) / scale_factor))
        top = int(round(int(data["top"][index]) / scale_factor))
        box_width = max(1, int(round(int(data["width"][index]) / scale_factor)))
        box_height = max(1, int(round(int(data["height"][index]) / scale_factor)))
        words.append(text)
        boxes.append([left, top, left + box_width, top + box_height])
        confidences.append(confidence)

    if not words:
        words = ["[EMPTY]"]
        boxes = [[0, 0, 1, 1]]
        confidences = [0.0]

    return {
        "image_path": str(image_path.resolve()),
        "image_name": image_path.name,
        "image_size": {"width": width, "height": height},
        "word_count": len(words),
        "words": words,
        "boxes": boxes,
        "confidences": confidences,
        "engine": "tesseract",
        "psm": psm,
        "oem": oem,
    }


def save_overlay(image_path: Path, ocr_payload: dict[str, Any], overlay_path: Path) -> None:
    with Image.open(image_path) as image:
        overlay_image = image.convert("RGB")
    draw = ImageDraw.Draw(overlay_image)
    for word, box in zip(ocr_payload["words"], ocr_payload["boxes"]):
        x1, y1, x2, y2 = box
        draw.rectangle((x1, y1, x2, y2), outline=(255, 64, 64), width=2)
        draw.text((x1, max(0, y1 - 12)), word[:32], fill=(0, 128, 255))
    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    overlay_image.save(overlay_path)


def process_targets(
    targets: list[OcrTarget],
    cache_root: Path,
    args: argparse.Namespace,
    overlays_remaining: int,
) -> dict[str, Any]:
    processed = 0
    reused = 0
    total_words = 0
    examples: list[dict[str, Any]] = []

    for index, target in enumerate(targets, start=1):
        cache_path = make_cache_path(cache_root, target)
        if args.skip_existing and cache_path.exists():
            payload = read_json(cache_path)
            reused += 1
        else:
            payload = run_tesseract(
                image_path=target.image_path,
                psm=args.psm,
                oem=args.oem,
                min_confidence=args.min_confidence,
            )
            payload.update(
                {
                    "template_stem": target.template_stem,
                    "relative_image_path": str(target.image_path.relative_to(args.degraded_root.resolve())),
                }
            )
            write_json(cache_path, payload)
            processed += 1

        total_words += int(payload["word_count"])
        if len(examples) < 10:
            examples.append(
                {
                    "template_stem": target.template_stem,
                    "image_name": target.image_path.name,
                    "word_count": payload["word_count"],
                    "first_words": payload["words"][:12],
                }
            )
        if overlays_remaining > 0:
            save_overlay(target.image_path, payload, make_overlay_path(cache_root, target))
            overlays_remaining -= 1

        if index == 1 or index % 50 == 0 or index == len(targets):
            log(f"OCR processed {index}/{len(targets)} image(s) in this pass.")

    return {
        "processed": processed,
        "reused": reused,
        "total_words": total_words,
        "examples": examples,
        "overlays_remaining": overlays_remaining,
        "targets_total": len(targets),
    }


def write_summary(
    cache_root: Path,
    degraded_root: Path,
    tesseract_cmd: Path,
    tesseract_version: str,
    args: argparse.Namespace,
    aggregate: dict[str, Any],
) -> None:
    summary = {
        "degraded_root": str(degraded_root),
        "cache_root": str(cache_root),
        "tesseract_cmd": str(tesseract_cmd),
        "tesseract_version": tesseract_version,
        "psm": args.psm,
        "oem": args.oem,
        "min_confidence": args.min_confidence,
        "watch": args.watch,
        "images_seen": aggregate["images_seen"],
        "images_processed": aggregate["images_processed"],
        "images_reused": aggregate["images_reused"],
        "overlays_saved": aggregate["overlays_saved"],
        "total_words": aggregate["total_words"],
        "average_words_per_image": aggregate["total_words"] / max(aggregate["images_seen"], 1),
        "examples": aggregate["examples"],
    }
    write_json(cache_root / "ocr_cache_summary.json", summary)


def build_cache(args: argparse.Namespace) -> dict[str, Any]:
    degraded_root = args.degraded_root.resolve()
    cache_root = args.cache_root.resolve()
    cache_root.mkdir(parents=True, exist_ok=True)

    configure_tesseract(args.tesseract_cmd.resolve())
    tesseract_version = str(pytesseract.get_tesseract_version())
    log(
        f"Using Tesseract {tesseract_version} with psm={args.psm}, oem={args.oem} "
        f"against {degraded_root}."
    )

    aggregate = {
        "images_seen": 0,
        "images_processed": 0,
        "images_reused": 0,
        "overlays_saved": 0,
        "total_words": 0,
        "examples": [],
    }
    seen_paths: set[str] = set()
    overlays_remaining = args.overlay_count
    idle_polls = 0

    while True:
        targets = discover_targets(
            degraded_root=degraded_root,
            limit_templates=args.limit_templates,
            limit_images=args.limit_images,
        )
        pending_targets = [target for target in targets if str(target.image_path.resolve()) not in seen_paths]

        if not pending_targets:
            if not args.watch:
                break
            idle_polls += 1
            log(f"No new degraded images found. Idle poll {idle_polls}/{args.idle_polls}.")
            if idle_polls >= args.idle_polls:
                break
            time.sleep(args.poll_seconds)
            continue

        idle_polls = 0
        for target in pending_targets:
            seen_paths.add(str(target.image_path.resolve()))

        result = process_targets(
            targets=pending_targets,
            cache_root=cache_root,
            args=args,
            overlays_remaining=overlays_remaining,
        )
        overlays_remaining = int(result["overlays_remaining"])

        aggregate["images_seen"] += int(result["targets_total"])
        aggregate["images_processed"] += int(result["processed"])
        aggregate["images_reused"] += int(result["reused"])
        aggregate["total_words"] += int(result["total_words"])
        aggregate["overlays_saved"] = args.overlay_count - overlays_remaining
        if len(aggregate["examples"]) < 10:
            slots_left = 10 - len(aggregate["examples"])
            aggregate["examples"].extend(result["examples"][:slots_left])

        write_summary(
            cache_root=cache_root,
            degraded_root=degraded_root,
            tesseract_cmd=args.tesseract_cmd.resolve(),
            tesseract_version=tesseract_version,
            args=args,
            aggregate=aggregate,
        )

        if not args.watch:
            break

    return {
        **aggregate,
        "tesseract_version": tesseract_version,
    }


def main() -> None:
    args = parse_args()
    summary = build_cache(args)
    log(
        f"Finished OCR cache: {summary['images_seen']} image(s), "
        f"{summary['total_words'] / max(summary['images_seen'], 1):.2f} words/image on average."
    )


if __name__ == "__main__":
    main()
