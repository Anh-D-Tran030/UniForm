import shutil
from pathlib import Path
import cv2
from augraphy import AugraphyPipeline, BadPhotoCopy, LowInkRandomLines,LinesDegradation, NoisyLines, Folding, Geometric, Scribbles, LowInkPeriodicLines
import random
SYTHETIC_PATH = "./synthetic_002.png" # FOR TESTING
OUTPUT_PATH = "./degraded_synthetic_002.png"

rng = random.Random(42)

def randomize_effect(num_effects = 3, severity = 2):
    effects = [
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
            LowInkRandomLines(count_range=(3, 5), noise_probability = 0.1 , p = 1),
            LowInkPeriodicLines(
                count_range=(2, max(5, int(round(3 + 3 * severity)))),
                period_range=(10, max(30, int(round(24 + 16 * severity)))),
                use_consistent_lines=True,
                noise_probability=min(0.16, 0.08 + 0.05 * severity),
                p=1
            ),
            LinesDegradation(
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
            Folding(
                fold_angle_range=(0, 1),
                fold_count=int(rng.choice([2,3, 4])),
                fold_noise=0.0,
                gradient_width=(0.1, 0.2),
                gradient_height=(0.003, 0.005),
                p=1,
            ),
            Geometric(
                scale=(1, 1),
                translation=(0, 0),
                fliplr=0,
                flipud=0,
                crop=(),
                rotate_range=(-2, 2),  
                padding=[0, 0, 0, 0],
                randomize=0,
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

    return rng.sample(effects, num_effects)

def initalize_folders(original_root, degraded_root):
    sub_folder_dataset = ["train", "test"]
    original_root = Path(original_root)
    degraded_root = Path(degraded_root)

    if not original_root.exists():
        raise FileNotFoundError(f"Original root folder '{original_root}' does not exist.")
    
    if degraded_root.exists():
        shutil.rmtree(degraded_root)
    degraded_root.mkdir(parents=True, exist_ok=True)
    for sub_folder in sub_folder_dataset:
        (degraded_root / sub_folder).mkdir(parents=True, exist_ok=True)
        for template_folder in (original_root / sub_folder).iterdir():
            if template_folder.is_dir():
                (degraded_root / sub_folder / template_folder.name).mkdir(parents=True, exist_ok=True)

def apply_effect(image_path, output_path):
    image = cv2.imread(image_path)
    pipeline = AugraphyPipeline(
        randomize_effect(num_effects=rng.choice([1,4,5,6]), severity=0.7)
    )
    degraded_image = pipeline(image)
    cv2.imwrite(output_path, degraded_image)   

def apply_effect_to_folder(original_folder, degraded_folder):
    for image_path in Path(original_folder).rglob("*.png"):
        relative_path = image_path.relative_to(original_folder)
        output_image_path = Path(degraded_folder) / relative_path
        output_image_path.parent.mkdir(parents=True, exist_ok=True)
        apply_effect(image_path, output_image_path)

if __name__ == "__main__":
    initalize_folders("./synthetic_data_common", "./degraded")
    apply_effect_to_folder("./synthetic_data_common", "./degraded")