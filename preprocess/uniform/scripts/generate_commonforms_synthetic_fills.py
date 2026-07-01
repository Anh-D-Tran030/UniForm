from __future__ import annotations

import argparse
import json
import random
import re
import time
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageFont


SOURCE_DIR = Path(r"A:\RealForm\processed\CommonFormsEnglish")
DATASET_DIR = Path(r"A:\RealForm\data\CommonForms\data")
FONTS_DIR = Path(r"A:\RealForm\Fonts\usable_fonts")
WINDOWS_FONTS_DIR = Path(r"C:\Windows\Fonts")
OUTPUT_ROOT = Path(r"A:\RealForm\processed\synthetic_fill_images")
STATE_PATH = OUTPUT_ROOT / "synthetic_fill_state.json"
LOG_PATH = OUTPUT_ROOT / "synthetic_fill.log"
DICTIONARY_PATH = OUTPUT_ROOT / "dictionary_500_words.json"
OCR_MANIFEST_PATH = SOURCE_DIR / "ocr_manifest.jsonl"
EXCLUDED_SELECTION_MANIFEST = Path(r"A:\RealForm\processed\CommonFormsEnglishSelected600\selection_manifest.json")
WORDS_REQUIRED = 500
DIGITAL_FONT_CANDIDATES = [
    "arial.ttf",
    "ARIALN.TTF",
    "calibri.ttf",
    "segoeui.ttf",
    "tahoma.ttf",
    "verdana.ttf",
    "times.ttf",
    "cour.ttf",
]


def utc_now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def log(message: str) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    line = f"[{utc_now()}] {message}"
    print(line, flush=True)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create synthetic filled-form images for CommonForms English templates using "
            "random handwriting fonts and a 500-word dictionary."
        )
    )
    parser.add_argument(
        "--templates-per-run",
        type=int,
        default=0,
        help="Optional cap on templates to process in this invocation. Zero means no cap.",
    )
    parser.add_argument(
        "--fills-per-template",
        type=int,
        default=10,
        help="Number of synthetic filled images to create for each template.",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Keep polling for new English templates while extraction is still running.",
    )
    parser.add_argument(
        "--poll-interval-seconds",
        type=int,
        default=30,
        help="Seconds between watch polls when no new work is ready.",
    )
    parser.add_argument(
        "--idle-rounds-before-exit",
        type=int,
        default=3,
        help="How many idle polls before exiting in watch mode.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Base random seed for reproducible synthetic fills.",
    )
    return parser.parse_args()


def load_json(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def list_template_metadata() -> list[Path]:
    return sorted((SOURCE_DIR / "metadata").glob("*/*.json"))


def template_image_path(metadata_path: Path) -> Path | None:
    split = metadata_path.parent.name
    candidates = list((SOURCE_DIR / "images" / split).glob(f"{metadata_path.stem}.*"))
    return candidates[0] if candidates else None


def available_fonts() -> list[Path]:
    fonts = sorted(path for path in FONTS_DIR.glob("*") if path.suffix.lower() in {".ttf", ".otf"})
    if not fonts:
        raise RuntimeError(f"No fonts found in {FONTS_DIR}")
    return fonts


def available_digital_fonts() -> list[Path]:
    fonts = [WINDOWS_FONTS_DIR / name for name in DIGITAL_FONT_CANDIDATES if (WINDOWS_FONTS_DIR / name).exists()]
    if not fonts:
        raise RuntimeError(f"No digital fonts found in {WINDOWS_FONTS_DIR}")
    return fonts


def excluded_template_names() -> set[str]:
    if not EXCLUDED_SELECTION_MANIFEST.exists():
        return set()
    payload = json.loads(EXCLUDED_SELECTION_MANIFEST.read_text(encoding="utf-8"))
    records = list(payload.get("records") or [])
    return {Path(str(record.get("file_name") or "")).stem for record in records if record.get("file_name")}


def normalize_word(word: str) -> str:
    cleaned = re.sub(r"[^A-Za-z']", "", word).strip("'").lower()
    return cleaned


def build_dictionary_from_ocr() -> list[str]:
    if not OCR_MANIFEST_PATH.exists():
        raise FileNotFoundError(f"OCR manifest not found: {OCR_MANIFEST_PATH}")

    frequencies: dict[str, int] = {}
    with OCR_MANIFEST_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            text = str(record.get("ocr_preview") or "")
            for raw_word in re.findall(r"[A-Za-z']+", text):
                word = normalize_word(raw_word)
                if len(word) < 2 or len(word) > 12:
                    continue
                frequencies[word] = frequencies.get(word, 0) + 1

    ranked = sorted(frequencies.items(), key=lambda item: (-item[1], item[0]))
    dictionary = [word for word, _count in ranked[:WORDS_REQUIRED]]
    if len(dictionary) < WORDS_REQUIRED:
        raise RuntimeError(
            f"Could only derive {len(dictionary)} words from {OCR_MANIFEST_PATH}; needed {WORDS_REQUIRED}."
        )
    save_json(DICTIONARY_PATH, {"word_count": len(dictionary), "words": dictionary})
    return dictionary


def load_or_build_dictionary() -> list[str]:
    if DICTIONARY_PATH.exists():
        payload = json.loads(DICTIONARY_PATH.read_text(encoding="utf-8"))
        words = list(payload.get("words") or [])
        if len(words) >= WORDS_REQUIRED:
            return words[:WORDS_REQUIRED]
    return build_dictionary_from_ocr()


@lru_cache(maxsize=8)
def parquet_rows_by_id(parquet_path_str: str) -> dict[int, dict[str, Any]]:
    parquet_path = Path(parquet_path_str)
    table = pq.read_table(parquet_path, columns=["id", "objects"])
    rows = {}
    for row in table.to_pylist():
        rows[int(row["id"])] = row["objects"]
    return rows


def bbox_lookup_for_metadata(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    parquet_path = DATASET_DIR / metadata["source_parquet"]
    objects = parquet_rows_by_id(str(parquet_path)).get(int(metadata["row_id"]))
    if objects is None:
        raise KeyError(f"row_id={metadata['row_id']} not found in {parquet_path}")

    object_ids = list(objects.get("id") or [])
    boxes = list(objects.get("bbox") or [])
    box_map = {
        int(object_ids[index]): [float(value) for value in boxes[index]]
        for index in range(min(len(object_ids), len(boxes)))
    }

    results: list[dict[str, Any]] = []
    for item in metadata.get("qualifying_objects", []):
        object_id = int(item["object_id"])
        bbox = box_map.get(object_id)
        if bbox is None:
            continue
        results.append(
            {
                "object_id": object_id,
                "area": float(item["area"]),
                "bbox": bbox,
            }
        )
    return results


def load_font(font_path: str, font_size: int) -> ImageFont.FreeTypeFont | None:
    try:
        return ImageFont.truetype(font_path, font_size)
    except OSError:
        return None


def text_size(font: ImageFont.FreeTypeFont, text: str) -> tuple[int, int] | None:
    try:
        left, top, right, bottom = font.getbbox(text)
    except OSError:
        return None
    return max(1, right - left), max(1, bottom - top)


def choose_font_group(
    rng: random.Random,
    handwriting_fonts: list[Path],
    digital_fonts: list[Path],
    box_width: int,
    box_height: int,
) -> tuple[str, list[Path]]:
    wide_line = box_width >= max(260, box_height * 8)
    digital_probability = 0.45 if wide_line else 0.25
    if digital_fonts and rng.random() < digital_probability:
        return "digital", digital_fonts
    return "handwritten", handwriting_fonts


def word_budget(box_width: int, box_height: int) -> int:
    if box_width < 110:
        return 1
    if box_width < 180:
        return 2
    if box_width < 280:
        return 3
    if box_width < 420:
        return 4
    if box_width < 650:
        return 5
    if box_width < 900:
        return 6
    return 7


def fit_text_to_box(
    rng: random.Random,
    words: list[str],
    handwriting_fonts: list[Path],
    digital_fonts: list[Path],
    box_width: int,
    box_height: int,
) -> dict[str, Any]:
    font_style, font_paths = choose_font_group(rng, handwriting_fonts, digital_fonts, box_width, box_height)
    min_size = max(16, int(box_height * (0.62 if font_style == "digital" else 0.68)))
    max_size = max(min_size, min(84, int(box_height * (0.96 if font_style == "digital" else 1.08))))
    width_limit = max(12, int(box_width * 0.90))
    height_limit = max(10, int(box_height * 0.92))
    max_words = word_budget(box_width, box_height)

    for _font_attempt in range(20):
        font_path = rng.choice(font_paths)
        for font_size in range(max_size, min_size - 1, -1):
            font = load_font(str(font_path), font_size)
            if font is None:
                continue

            chosen: list[str] = []
            for _word_attempt in range(max_words):
                next_word = rng.choice(words)
                candidate = " ".join(chosen + [next_word]) if chosen else next_word
                size = text_size(font, candidate)
                if size is None:
                    chosen = []
                    break
                text_w, text_h = size
                if text_w <= width_limit and text_h <= height_limit:
                    chosen.append(next_word)
                else:
                    break

            if chosen:
                text = " ".join(chosen)
                size = text_size(font, text)
                if size is None:
                    continue
                text_w, text_h = size
                return {
                    "font_path": str(font_path),
                    "font_style": font_style,
                    "font_size": font_size,
                    "text": text,
                    "word_count": len(chosen),
                    "text_width": text_w,
                    "text_height": text_h,
                }

    fallback_word = rng.choice(words)
    fallback_font_path = None
    fallback_font = None
    fallback_style = font_style
    for candidate_font in font_paths:
        fallback_font = load_font(str(candidate_font), min_size)
        if fallback_font is None:
            continue
        size = text_size(fallback_font, fallback_word)
        if size is None:
            continue
        text_w, text_h = size
        fallback_font_path = candidate_font
        break
    else:
        fallback_font = ImageFont.load_default()
        text_w, text_h = text_size(fallback_font, fallback_word) or (len(fallback_word) * 8, 12)
        fallback_font_path = Path("PIL_default_font")

    return {
        "font_path": str(fallback_font_path),
        "font_style": fallback_style,
        "font_size": min_size,
        "text": fallback_word,
        "word_count": 1,
        "text_width": text_w,
        "text_height": text_h,
    }


def render_field(
    canvas_image: Image.Image,
    field: dict[str, Any],
    fill_spec: dict[str, Any],
    rng: random.Random,
) -> dict[str, Any]:
    x, y, width, height = [int(round(value)) for value in field["bbox"]]
    font = load_font(fill_spec["font_path"], fill_spec["font_size"])
    if font is None:
        font = ImageFont.load_default()
    text = fill_spec["text"]
    text_w, text_h = text_size(font, text) or (max(1, len(text) * 8), 12)

    is_digital = fill_spec.get("font_style") == "digital"
    x_jitter = int(rng.uniform(max(-2, -width * 0.02), max(2, width * 0.02)))
    y_jitter = int(rng.uniform(max(-1, -height * 0.05), max(1, height * 0.05)))
    angle = rng.uniform(-0.9, 0.9) if is_digital else rng.uniform(-3.2, 3.2)
    ink_value = rng.randint(10, 55)
    ink = (ink_value, ink_value, ink_value, rng.randint(205, 245))

    text_layer = Image.new("RGBA", (text_w + 24, text_h + 24), (255, 255, 255, 0))
    draw = ImageDraw.Draw(text_layer)
    draw.text((12, 12), text, font=font, fill=ink)
    rotated = text_layer.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True)

    paste_x = max(0, x + max(0, int((width - rotated.width) / 2)) + x_jitter)
    paste_y = max(0, y + max(0, int((height - rotated.height) / 2)) + y_jitter)
    canvas_image.alpha_composite(rotated, (paste_x, paste_y))

    return {
        "object_id": field["object_id"],
        "bbox": field["bbox"],
        "area": field["area"],
        "text": text,
        "word_count": fill_spec["word_count"],
        "font_style": fill_spec.get("font_style", "handwritten"),
        "font_path": fill_spec["font_path"],
        "font_size": fill_spec["font_size"],
        "rotation_degrees": round(angle, 3),
        "ink_rgba": list(ink),
        "paste_xy": [paste_x, paste_y],
    }


def template_output_dir(metadata_path: Path) -> Path:
    return OUTPUT_ROOT / metadata_path.stem


def existing_fill_indices(template_dir: Path) -> set[int]:
    indices: set[int] = set()
    for path in template_dir.glob("fill_*.json"):
        match = re.match(r"fill_(\d+)\.json$", path.name)
        if match:
            indices.add(int(match.group(1)))
    return indices


def process_template(
    metadata_path: Path,
    image_path: Path,
    words: list[str],
    handwriting_fonts: list[Path],
    digital_fonts: list[Path],
    fills_per_template: int,
    seed: int,
    state: dict[str, Any],
) -> tuple[int, bool]:
    template_dir = template_output_dir(metadata_path)
    template_dir.mkdir(parents=True, exist_ok=True)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    fields = bbox_lookup_for_metadata(metadata)
    if not fields:
        log(f"Skipping {metadata_path.stem}: no qualifying boxes.")
        return 0, False

    existing = existing_fill_indices(template_dir)
    created = 0

    for fill_index in range(fills_per_template):
        if fill_index in existing:
            continue

        rng = random.Random(f"{seed}:{metadata_path.stem}:{fill_index}")
        with Image.open(image_path).convert("RGBA") as template_image:
            rendered_fields: list[dict[str, Any]] = []

            for field in fields:
                box_width = max(1, int(round(field["bbox"][2])))
                box_height = max(1, int(round(field["bbox"][3])))
                fill_spec = fit_text_to_box(rng, words, handwriting_fonts, digital_fonts, box_width, box_height)
                rendered_fields.append(render_field(template_image, field, fill_spec, rng))

            image_output_path = template_dir / f"fill_{fill_index:02d}.png"
            json_output_path = template_dir / f"fill_{fill_index:02d}.json"
            template_image.convert("RGB").save(image_output_path)

        payload = {
            "template_name": metadata_path.stem,
            "template_image": str(image_path),
            "source_metadata": str(metadata_path),
            "source_parquet": metadata["source_parquet"],
            "row_id": metadata["row_id"],
            "fill_index": fill_index,
            "fields": rendered_fields,
        }
        save_json(json_output_path, payload)
        created += 1
        log(f"Created {image_output_path.name} for {metadata_path.stem} with {len(rendered_fields)} fields.")

    state.setdefault("templates", {})[metadata_path.stem] = {
        "fills_per_template": fills_per_template,
        "completed_fills": sorted(existing_fill_indices(template_dir)),
        "last_updated": utc_now(),
        "template_dir": str(template_dir),
    }
    save_json(STATE_PATH, state)
    return created, True


def pending_templates(state: dict[str, Any], fills_per_template: int) -> list[tuple[Path, Path]]:
    ready: list[tuple[Path, Path]] = []
    excluded_templates = excluded_template_names()
    for metadata_path in list_template_metadata():
        if metadata_path.stem in excluded_templates:
            continue
        image_path = template_image_path(metadata_path)
        if image_path is None:
            continue
        template_dir = template_output_dir(metadata_path)
        completed = existing_fill_indices(template_dir)
        if len(completed) >= fills_per_template:
            continue
        ready.append((metadata_path, image_path))
    return ready


def main() -> None:
    args = parse_args()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    handwriting_fonts = available_fonts()
    digital_fonts = available_digital_fonts()
    words = load_or_build_dictionary()
    state = load_json(
        STATE_PATH,
        {
            "created_at": utc_now(),
            "fills_per_template": args.fills_per_template,
            "templates": {},
        },
    )

    log(
        "Starting synthetic fill generator with fonts={}, dictionary_words={}, fills_per_template={}, watch={}, excluded_templates={}.".format(
            len(handwriting_fonts) + len(digital_fonts),
            len(words),
            args.fills_per_template,
            args.watch,
            len(excluded_template_names()),
        )
    )

    idle_round = 0
    processed_templates = 0

    while True:
        pending = pending_templates(state, args.fills_per_template)
        if args.templates_per_run > 0:
            remaining = args.templates_per_run - processed_templates
            if remaining <= 0:
                break
            pending = pending[:remaining]

        if pending:
            idle_round = 0
            for metadata_path, image_path in pending:
                created, attempted = process_template(
                    metadata_path=metadata_path,
                    image_path=image_path,
                    words=words,
                    handwriting_fonts=handwriting_fonts,
                    digital_fonts=digital_fonts,
                    fills_per_template=args.fills_per_template,
                    seed=args.seed,
                    state=state,
                )
                if attempted:
                    processed_templates += 1
                    log(
                        "Template {} processed. Newly created fills={} total_completed={}.".format(
                            metadata_path.stem,
                            created,
                            len(existing_fill_indices(template_output_dir(metadata_path))),
                        )
                    )
        else:
            if not args.watch:
                break
            idle_round += 1
            log(
                "No pending English templates. idle_wait {}/{} sleeping {}s.".format(
                    idle_round,
                    args.idle_rounds_before_exit,
                    args.poll_interval_seconds,
                )
            )
            if idle_round >= args.idle_rounds_before_exit:
                break
            time.sleep(args.poll_interval_seconds)

    save_json(STATE_PATH, state)
    log("Synthetic fill generation stopped.")


if __name__ == "__main__":
    main()
