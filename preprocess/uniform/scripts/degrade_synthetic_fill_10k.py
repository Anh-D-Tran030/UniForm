from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

try:
    from augraphy import (
        BadPhotoCopy,
        Folding,
        Geometric,
        LinesDegradation,
        LowInkPeriodicLines,
        LowInkRandomLines,
        NoisyLines,
        Scribbles,
    )
except ImportError:  # pragma: no cover
    BadPhotoCopy = None
    Folding = None
    Geometric = None
    LinesDegradation = None
    LowInkPeriodicLines = None
    LowInkRandomLines = None
    NoisyLines = None
    Scribbles = None


DEFAULT_SOURCE_ROOT = Path(r"A:\RealForm\processed\synthetic_fill_images_10k_folders")
DEFAULT_OUTPUT_ROOT = Path(r"A:\RealForm\processed\synthetic_fill_images_10k_folders_degraded")
IMAGE_GLOB = "fill_*.png"
UNSTABLE_EFFECTS_LOGGED: set[str] = set()


@dataclass
class TemplateFolder:
    stem: str
    path: Path
    image_count: int


def log(message: str) -> None:
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    print(f"[{timestamp}] {message}", flush=True)


def discover_template_folders(source_root: Path, max_templates: int | None) -> list[TemplateFolder]:
    folders: list[TemplateFolder] = []
    for template_dir in sorted(source_root.iterdir(), key=lambda path: path.name.lower()):
        if not template_dir.is_dir():
            continue
        image_count = len(list(template_dir.glob(IMAGE_GLOB)))
        if image_count == 0:
            continue
        folders.append(TemplateFolder(stem=template_dir.name, path=template_dir, image_count=image_count))
    if max_templates is not None:
        folders = folders[:max_templates]
    return folders


def deterministic_rng(key: str) -> np.random.Generator:
    seed = abs(hash(key)) % (2**32)
    return np.random.default_rng(seed)


def fit_to_canvas(image: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    canvas = np.full((target_h, target_w, 3), 255, dtype=np.uint8)
    copy_h = min(target_h, image.shape[0])
    copy_w = min(target_w, image.shape[1])
    src_y = max(0, (image.shape[0] - copy_h) // 2)
    src_x = max(0, (image.shape[1] - copy_w) // 2)
    dst_y = max(0, (target_h - copy_h) // 2)
    dst_x = max(0, (target_w - copy_w) // 2)
    canvas[dst_y : dst_y + copy_h, dst_x : dst_x + copy_w] = image[src_y : src_y + copy_h, src_x : src_x + copy_w]
    return canvas


def apply_stretch_effect(image: np.ndarray, rng: np.random.Generator, severity: float) -> np.ndarray:
    height, width = image.shape[:2]
    # Use a mild non-uniform resize so the page looks scanned/warped instead of perfectly flat.
    stretch_x = float(rng.uniform(0.985, min(1.045, 1.01 + 0.025 * severity)))
    stretch_y = float(rng.uniform(0.985, min(1.035, 1.005 + 0.018 * severity)))
    stretched = cv2.resize(
        image,
        dsize=None,
        fx=stretch_x,
        fy=stretch_y,
        interpolation=cv2.INTER_LINEAR,
    )
    return fit_to_canvas(stretched, height, width)


def pick_effects(
    rng: np.random.Generator,
    severity: float,
    include_folding: bool,
) -> list[object]:
    if BadPhotoCopy is None:
        raise RuntimeError("Augraphy is not installed. Install it with `python -m pip install augraphy`.")

    rotation_limit = max(2, int(round((2.0 + 1.6 * severity) * 1.25)))

    geometric = Geometric(
        scale=(1, 1),
        translation=(0, 0),
        fliplr=0,
        flipud=0,
        crop=(),
        rotate_range=(-rotation_limit, rotation_limit),
        padding=[0, 0, 0, 0],
        padding_type="fill",
        padding_value=(255, 255, 255),
        randomize=0,
        p=1,
    )

    folding = Folding(
        fold_angle_range=(0, 1),
        fold_count=int(rng.integers(2, 4)),
        fold_noise=0.0,
        gradient_width=(0.1, 0.2),
        gradient_height=(0.003, 0.005),
        p=1,
    )

    lines_degradation = LinesDegradation(
        line_roi=(0.0, 0.0, 1.0, 1.0),
        line_gradient_range=(20, 220),
        line_gradient_direction=(0, 2),
        line_split_probability=(0.45, min(0.85, (0.55 + 0.12 * severity) * 1.25)),
        line_replacement_value=(215, 255),
        line_min_length=(5, max(25, int(round((35 + 15 * severity) * 1.25)))),
        line_long_to_short_ratio=(2, 5),
        line_replacement_probability=(0.35, min(0.8, (0.45 + 0.15 * severity) * 1.25)),
        line_replacement_thickness=(1, 3),
        p=1,
    )

    low_ink_periodic = LowInkPeriodicLines(
        count_range=(2, max(5, int(round(3 + 3 * severity)))),
        period_range=(10, max(30, int(round(24 + 16 * severity)))),
        use_consistent_lines=True,
        noise_probability=min(0.16, 0.08 + 0.05 * severity),
        p=1,
    )

    optional_effects: list[object] = [
        BadPhotoCopy(
            noise_type=int(rng.choice([1, 2, 3, 5])),
            noise_side=str(rng.choice(["random", "left", "right", "top", "bottom", "none"])),
            noise_iteration=(1, max(1, int(round(1 + severity)))),
            noise_size=(1, max(2, int(round(2 + severity)))),
            noise_value=(int(max(52, 88 - 16 * severity)), int(max(118, 152 - 10 * severity))),
            noise_sparsity=(0.03, min(0.24, 0.08 + 0.08 * severity)),
            noise_concentration=(0.03, min(0.22, 0.08 + 0.07 * severity)),
            blur_noise=1,
            blur_noise_kernel=(3, 3),
            wave_pattern=0,
            edge_effect=0,
            numba_jit=0,
            p=1,
        ),
        NoisyLines(
            noisy_lines_direction="random",
            noisy_lines_location="random",
            noisy_lines_number_range=(1, 3),
            noisy_lines_color=(0, 0, 0),
            noisy_lines_thickness_range=(1, 1),
            noisy_lines_random_noise_intensity_range=(0.008, min(0.03, 0.02 + 0.01 * severity)),
            noisy_lines_length_interval_range=(8, max(30, int(round(30 + 20 * severity)))),
            noisy_lines_gaussian_kernel_value_range=(3, 3),
            noisy_lines_overlay_method="ink_to_paper",
            p=1,
        ),
        LowInkRandomLines(
            count_range=(3, max(5, int(round(4 + 2 * severity)))),
            use_consistent_lines=True,
            noise_probability=min(0.15, 0.08 + 0.04 * severity),
            p=1,
        ),
        Scribbles(
            scribbles_type="lines",
            scribbles_ink="pencil",
            scribbles_location="random",
            scribbles_size_range=(80, 180),
            scribbles_count_range=(1, 1),
            scribbles_thickness_range=(1, 1),
            scribbles_brightness_change=[96, 128],
            scribbles_lines_stroke_count_range=(1, 2),
            scribbles_skeletonize=0,
            scribbles_color=(40, 40, 40),
            p=1,
        )
    ]

    optional_count = min(len(optional_effects), int(rng.integers(2, min(5, len(optional_effects)) + 1)))
    chosen_indices = rng.choice(len(optional_effects), size=optional_count, replace=False)
    sampled_optional = [optional_effects[int(index)] for index in np.atleast_1d(chosen_indices)]

    sequence = [geometric]
    if include_folding and rng.random() < 0.45:
        sequence.append(folding)

    sequence.extend([lines_degradation, low_ink_periodic])
    sequence.extend(sampled_optional)
    return sequence


def apply_effect_sequence(image: np.ndarray, effects: list[object]) -> np.ndarray:
    current = image.copy()
    for effect in effects:
        effect_name = effect.__class__.__name__
        try:
            result = effect(current)
            if result is not None:
                current = result
        except Exception as exc:
            if effect_name not in UNSTABLE_EFFECTS_LOGGED:
                log(f"Skipping unstable Augraphy effect {effect_name}. First error: {exc}")
                UNSTABLE_EFFECTS_LOGGED.add(effect_name)
            continue
    return current


def degrade_image(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    severity = float(rng.uniform(0.6, 1.05))
    stretched = apply_stretch_effect(image=image, rng=rng, severity=severity)
    effects = pick_effects(
        rng=rng,
        severity=severity,
        include_folding=True,
    )
    degraded = apply_effect_sequence(image=stretched, effects=effects)
    return np.clip(degraded, 0, 255).astype(np.uint8)


def process_template_folder(folder: TemplateFolder, output_root: Path, overwrite: bool) -> None:
    target_dir = output_root / folder.stem
    target_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(folder.path.glob(IMAGE_GLOB), key=lambda path: path.name)
    total_images = len(image_paths)
    progress_step = max(1, min(10, total_images // 10 if total_images > 10 else total_images))

    for index, image_path in enumerate(image_paths, start=1):
        target_path = target_dir / image_path.name
        if target_path.exists() and not overwrite:
            if index == 1 or index == total_images or index % progress_step == 0:
                log(f"{folder.stem}: {index}/{total_images} degraded images already present.")
            continue

        source = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if source is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")

        rng = deterministic_rng(str(image_path.relative_to(folder.path.parent.parent)))
        degraded = degrade_image(source, rng)
        cv2.imwrite(str(target_path), degraded)

        if index == 1 or index == total_images or index % progress_step == 0:
            log(f"{folder.stem}: saved {index}/{total_images} degraded images.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Degrade the 10k synthetic fill template folders using Augraphy and heavier geometric/line effects."
    )
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT, help="Root folder of the clean 10k filled templates.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Where to write the degraded template folders.")
    parser.add_argument("--max-templates", type=int, help="Optional cap on how many template folders to process.")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate degraded images even if they already exist.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    folders = discover_template_folders(source_root, args.max_templates)
    if not folders:
        raise FileNotFoundError(f"No filled template folders found under {source_root}.")

    log(f"Processing {len(folders)} template folder(s) into {output_root}.")
    for folder in folders:
        log(f"Degrading template {folder.stem} with {folder.image_count} source image(s).")
        process_template_folder(folder=folder, output_root=output_root, overwrite=args.overwrite)

    summary = {
        "source_root": str(source_root),
        "output_root": str(output_root),
        "max_templates": args.max_templates,
        "overwrite": args.overwrite,
    }
    with (output_root / "degradation_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    log("Finished generating degraded synthetic images.")


if __name__ == "__main__":
    main()
