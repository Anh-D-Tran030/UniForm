from __future__ import annotations

import json
import random
from pathlib import Path

import pytesseract
from PIL import Image, ImageOps


INPUT_DIR = Path(r"A:\RealForm\processed\synthetic_fill_images_10k_folders_degraded")
OUTPUT_DIR = Path(r"A:\RealForm\processed\simple_ocr_mask_corrupt")
TESSERACT_EXE = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
WORD_BANK = "name address date form account city state country phone email number office street signature total amount reference".split()


def ocr_image(image_path: Path) -> dict:
    with Image.open(image_path) as image:
        rgb = image.convert("RGB")
        width, height = rgb.size
        scale = 2.0
        prepared = ImageOps.autocontrast(rgb.convert("L")).resize(
            (int(width * scale), int(height * scale)),
            resample=Image.Resampling.BICUBIC,
        )
        data = pytesseract.image_to_data(
            prepared,
            output_type=pytesseract.Output.DICT,
            config="--psm 11 --oem 3",
            lang="eng",
        )

    words, boxes, confidences = [], [], []
    for i, text in enumerate(data.get("text", [])):
        text = str(text).strip()
        if not text:
            continue
        try:
            conf = float(data["conf"][i])
        except ValueError:
            conf = -1.0
        if conf < 30:
            continue

        left = int(round(int(data["left"][i]) / scale))
        top = int(round(int(data["top"][i]) / scale))
        w = max(1, int(round(int(data["width"][i]) / scale)))
        h = max(1, int(round(int(data["height"][i]) / scale)))
        words.append(text)
        boxes.append([left, top, left + w, top + h])
        confidences.append(conf)

    if not words:
        words, boxes, confidences = ["[EMPTY]"], [[0, 0, 1, 1]], [0.0]

    return {
        "image_path": str(image_path),
        "image_size": {"width": width, "height": height},
        "words": words,
        "boxes": boxes,
        "confidences": confidences,
        "word_count": len(words),
    }


def mask_and_corrupt(payload: dict, mask_rate: float, corrupt_rate: float, seed: str) -> dict:
    rng = random.Random(seed)
    words = list(payload["words"])
    indices = list(range(len(words)))
    rng.shuffle(indices)

    mask_count = round(len(words) * mask_rate)
    corrupt_count = round(len(words) * corrupt_rate)
    for i in indices[:mask_count]:
        words[i] = "[MASK]"
    for i in indices[mask_count : mask_count + corrupt_count]:
        words[i] = rng.choice(WORD_BANK)

    updated = dict(payload)
    updated["words"] = words
    updated["mask_rate"] = mask_rate
    updated["corrupt_rate"] = corrupt_rate
    return updated


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    pytesseract.pytesseract.tesseract_cmd = str(TESSERACT_EXE)
    for image_path in sorted(INPUT_DIR.rglob("*")):
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        relative = image_path.relative_to(INPUT_DIR).with_suffix("")
        clean_path = OUTPUT_DIR / "clean" / f"{relative}.ocr.json"
        mask25_path = OUTPUT_DIR / "mask25" / f"{relative}.ocr.json"
        mask20_corrupt10_path = OUTPUT_DIR / "mask20_corrupt10" / f"{relative}.ocr.json"
        if clean_path.exists() and mask25_path.exists() and mask20_corrupt10_path.exists():
            continue

        clean = ocr_image(image_path)
        save_json(clean_path, clean)
        save_json(mask25_path, mask_and_corrupt(clean, 0.25, 0.0, str(image_path)))
        save_json(mask20_corrupt10_path, mask_and_corrupt(clean, 0.20, 0.10, str(image_path)))


if __name__ == "__main__":
    main()
