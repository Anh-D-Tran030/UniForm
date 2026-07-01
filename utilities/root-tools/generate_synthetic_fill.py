from __future__ import annotations
import argparse
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont, ImageOps
# from trdg import handwritten_text_generator as handwritten_text_generator
import gc


DEFAULT_SUBSET = "eccv"
DEFAULT_RECORD_COUNT = 100
DEFAULT_INPUT_ROOT = Path("forms-data")
DEFAULT_OUTPUT_ROOT = Path("synthetic_data")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
TRDG_ROOT = Path("TextRecognitionDataGenerator")
FONTS_DIR = Path(r"A:\RealForm\Fonts\usable_fonts")
WINDOWS_FONTS_DIR = Path(r"C:\Windows\Fonts")
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
AVAILABLE_DIGITAL_FONT_PATHS = []
FIRST_NAMES = [
    "Avery", "Jordan", "Taylor", "Morgan", "Riley", "Casey", "Parker", "Quinn",
    "Hayden", "Cameron", "Skyler", "Reese", "Dakota", "Emerson", "Finley", "Rowan",
]
LAST_NAMES = [
    "Carter", "Lee", "Morgan", "Brooks", "Parker", "Bennett", "Collins", "Ward",
    "Mitchell", "Hayes", "Nguyen", "Patel", "Reed", "Foster", "Campbell", "Price",
]
COMPANY_NAMES = [
    "North Ridge Services", "Blue Harbor Group", "Clearview Holdings", "Summit Valley LLC",
    "Red Cedar Partners", "Evergreen Transit", "Lakefront Supply Co", "Atlas Field Services",
    "Silver Pine Logistics", "Westbrook Manufacturing", "Crescent Oak Ltd", "Granite Hill Corp",
]
STREET_ADDRESSES = [
    "18 Harbor Ave", "407 Willow St", "92 Cedar Lane", "155 Oak Ridge Dr", "271 Maple Road",
    "64 Pine Street", "889 Birch Blvd", "12 River Park Way", "730 Elm Terrace", "415 Meadow Court",
]
CITY_STATE_LINES = [
    "Portland, OR 97205", "Denver, CO 80212", "Seattle, WA 98118", "Austin, TX 78704",
    "Madison, WI 53703", "Phoenix, AZ 85016", "Boise, ID 83702", "Tampa, FL 33602",
    "Salem, OR 97301", "Reno, NV 89501",
]
VOCABULARY_WORDS = [
    "north", "harbor", "cedar", "field", "silver", "meadow", "atlas", "river", "brook", "pine",
    "summit", "delta", "stone", "lighthouse", "willow", "forest", "prime", "clear", "amber", "ridge",
    "valley", "coastal", "urban", "signal", "orchard", "trail", "vista", "market", "public", "safety",
    "transit", "review", "service", "supply", "report", "network", "green", "central", "west", "east",
]
EMAIL_DOMAINS = ["example.com", "mail.com", "sample.org", "demo.net"]
SHORT_FALLBACK_VALUES = [
    "Approved", "Pending", "Completed", "Verified", "Primary", "Secondary", "Current", "Standard", "None",
]
PRINTED_FONT_CANDIDATES = ["arial.ttf", "calibri.ttf", "tahoma.ttf", "times.ttf", "consola.ttf"]
HANDWRITING_FONT_CANDIDATES = ["segoesc.ttf", "comic.ttf", "comicbd.ttf", "ariali.ttf", "calibrii.ttf"]


FULL_VOCABULARY = set(FIRST_NAMES + LAST_NAMES + COMPANY_NAMES + STREET_ADDRESSES + CITY_STATE_LINES + VOCABULARY_WORDS)



def get_dataset_subfolders(original_root):
    original_root = Path(original_root)
    dataset_subfolders = []
    for sub_folder in sorted(path.name for path in original_root.iterdir() if path.is_dir()):
        image_root = original_root / sub_folder / "data" / "data" / "imgs"
        json_root = original_root / sub_folder / "data" / "data" / "jsons"
        if image_root.exists() and json_root.exists():
            dataset_subfolders.append(sub_folder)
    return dataset_subfolders


def initalize_folders(original_root, synthetic_root):
    original_root = Path(original_root)
    synthetic_root = Path(synthetic_root)

    if not original_root.exists():
        raise FileNotFoundError(f"Original root folder '{original_root}' does not exist.")

    synthetic_root.mkdir(parents=True, exist_ok=True)

    for sub_folder in get_dataset_subfolders(original_root):
        original_subset_root = original_root / sub_folder / "data" / "data" / "imgs"
        synthetic_subset_root = synthetic_root / sub_folder
        synthetic_subset_root.mkdir(parents=True, exist_ok=True)

        for template_path in sorted(original_subset_root.iterdir()):
            if template_path.is_file() and template_path.suffix.lower() in IMAGE_EXTENSIONS:
                (synthetic_subset_root / template_path.stem).mkdir(parents=True, exist_ok=True)


def load_font_with_candidates(font_candidates, font_size):
    font_size = max(1, int(round(font_size)))
    for font_name in font_candidates:
        try:
            return ImageFont.truetype(font_name, font_size)
        except OSError:
            continue
    return ImageFont.load_default()

def calculate_number_of_characters(bbox, char_height):
    x_min = min(bbox[0][0], bbox[1][0], bbox[2][0], bbox[3][0])
    x_max = max(bbox[0][0], bbox[1][0], bbox[2][0], bbox[3][0])
    y_min = min(bbox[0][1], bbox[1][1], bbox[2][1], bbox[3][1])
    y_max = max(bbox[0][1], bbox[1][1], bbox[2][1], bbox[3][1])

    width = x_max - x_min
    height = y_max - y_min

    avg_char_width = max(10, int(char_height * 0.62))
    avg_char_height = int(char_height * 1.20)


    num_chars_width = width // avg_char_width
    num_lines = height // avg_char_height
    print(f"Calculated num_chars_width: {num_chars_width}, num_lines: {num_lines} for bbox: {bbox}")
    return int(num_chars_width),int(num_lines)

def generate_random_text(bbox, char_height):
    num_chars_width,num_lines = calculate_number_of_characters(bbox, char_height)
    if num_chars_width <= 0 or num_lines <= 0:
        return None
    words = []

    for _ in range(num_lines):
        # Assuming each word is 8 characters long on average
        # iteratively add word until we fill the width of the bbox
        line = ""
        while len(line) < num_chars_width:
            word = random.choice(list(FULL_VOCABULARY))
            if len(line) + len(word) + 1 > num_chars_width:
                break
            line += word + " "
        words.append(line)
    return "\n".join(words)

def load_bboxes(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    widgets = data.get("Widget", [])

    bboxes = []
    
    for item in widgets:
        x_top_left = item.get("x")
        y_top_left = item.get("y")
        x_top_right = item.get("x") + item.get("w")
        y_top_right = item.get("y")
        x_bottom_left = item.get("x")
        y_bottom_left = item.get("y") + item.get("h")

        x_bottom_right = item.get("x") + item.get("w")
        y_bottom_right = item.get("y") + item.get("h")

        bboxes.append(((x_top_left, y_top_left), (x_top_right, y_top_right), (x_bottom_left, y_bottom_left), (x_bottom_right, y_bottom_right)))
    return bboxes

def load_image(image_path):
    return Image.open(image_path).convert("RGB")

def calculate_good_height(bboxes):
    heights = []
    for bbox in bboxes:
        y_min = min(bbox[0][1], bbox[1][1], bbox[2][1], bbox[3][1])
        y_max = max(bbox[0][1], bbox[1][1], bbox[2][1], bbox[3][1])
        heights.append(y_max - y_min)
    if not heights:
        return 28
    
    avg_height = sum(heights) // len(heights)
    print(f"Calculated average height: {avg_height}")
    return max(16, min(avg_height, 48)) * 0.8

def fill_images(image_path, json_path, output_path, font=None, char_height=None, text_mode="handwritten"):
    image = load_image(image_path)
    bboxes = load_bboxes(json_path)
    text_generator = TrdgHandwrittenTextGenerator()
    if char_height is None:
        char_height = calculate_good_height(bboxes)

    for bbox in bboxes:
        text = generate_random_text(bbox, char_height)
        if text is None:
            continue
        x_min = int(round(min(point[0] for point in bbox)))
        x_max = int(round(max(point[0] for point in bbox)))
        y_min = int(round(min(point[1] for point in bbox)))
        y_max = int(round(max(point[1] for point in bbox)))
        box_width = max(1, x_max - x_min)
        box_height = max(1, y_max - y_min)

        if text_mode == "digital":
            rendered = text_generator.render_digital_multiline_text(
                text,
                box_width,
                box_height,
                char_height,
                font=font,
            )
        else:
            rendered = text_generator.render_multiline_text(text, box_width, box_height, char_height)
        if rendered is None:
            continue

        image.paste("white", [x_min, y_min, x_max, y_max])
        alpha = rendered.getchannel("A")
        image.paste(rendered.convert("RGB"), (x_min, y_min), alpha)

    image.save(output_path)

class TrdgHandwrittenTextGenerator:
    def __init__(self, *args, **kwargs):
        TRDG_ROOT = Path("TextRecognitionDataGenerator")
        trdg_root = Path(__file__).parent / TRDG_ROOT
        self._trdg_warning_emitted = False
        self.generator = None
        try:
            sys.path.append(str(trdg_root))
            from trdg import handwritten_text_generator as handwritten_text_generator
            self.generator = handwritten_text_generator
        except Exception as error:
            self.emit_trdg_warning_once(error)

    def apply_text(self, image, text_patch, bbox):
        x_min = int(round(min(point[0] for point in bbox)))
        x_max = int(round(max(point[0] for point in bbox)))
        y_min = int(round(min(point[1] for point in bbox)))
        y_max = int(round(max(point[1] for point in bbox)))

        alpha = text_patch.getchannel("A")
        image.paste(text_patch.convert("RGB"), (x_min, y_min), alpha)
    def crop_transparent_background(self, image):
        a_box = image.getchannel("A").getbbox()
        if a_box:
            return image.crop(a_box)
        else:
            return image
    def join_images(self, images, spacing = 8):
        if not images:
            return Image.new("RGBA", (1, 1), (255, 255, 255, 0))
        if len(images) == 1:
            return self.crop_transparent_background(images[0])

        total_width = sum(patch.width for patch in images) + spacing * (len(images) - 1)
        max_height = max(patch.height for patch in images)
        canvas = Image.new("RGBA", (max(1, total_width), max(1, max_height)), (255, 255, 255, 0))

        cursor_x = 0
        for patch in images:
            offset_y = max(0, max_height - patch.height)
            canvas.alpha_composite(patch, dest=(cursor_x, offset_y))
            cursor_x += patch.width + spacing
        return self.crop_transparent_background(canvas)

    def stack_lines(self, images, spacing=4):
        if not images:
            return Image.new("RGBA", (1, 1), (255, 255, 255, 0))
        if len(images) == 1:
            return self.crop_transparent_background(images[0])

        max_width = max(patch.width for patch in images)
        total_height = sum(patch.height for patch in images) + spacing * (len(images) - 1)
        canvas = Image.new("RGBA", (max(1, max_width), max(1, total_height)), (255, 255, 255, 0))

        cursor_y = 0
        for patch in images:
            canvas.alpha_composite(patch, dest=(0, cursor_y))
            cursor_y += patch.height + spacing
        return self.crop_transparent_background(canvas)

    def fit_patch_to_box(self, patch, max_width, max_height):
        if patch.width <= 0 or patch.height <= 0:
            return patch

        scale = min(max_width / patch.width, max_height / patch.height, 1.0)
        if scale <= 0:
            return Image.new("RGBA", (1, 1), (255, 255, 255, 0))
        if scale < 1.0:
            new_width = max(1, int(round(patch.width * scale)))
            new_height = max(1, int(round(patch.height * scale)))
            patch = patch.resize((new_width, new_height), resample=Image.LANCZOS)
        return self.crop_transparent_background(patch)

    def resize_handwriting_patch(self, patch, target_height, allow_upscale=False):
        patch = self.crop_transparent_background(patch)
        if patch.width <= 0 or patch.height <= 0:
            return patch

        target_height = max(1, int(round(target_height)))
        if not allow_upscale:
            target_height = min(target_height, patch.height)
        if target_height == patch.height:
            return patch

        new_width = max(1, int(round(patch.width * (target_height / patch.height))))
        resample = Image.LANCZOS if target_height < patch.height else Image.BILINEAR
        return self.crop_transparent_background(
            patch.resize((new_width, target_height), resample=resample)
        )

    def render_multiline_text(self, text, max_width, max_height, char_height):
        line_patches = []
        for line in text.splitlines():
            clean_line = line.strip()
            if not clean_line:
                continue
            line_patches.append(self.render_text(clean_line, char_height))
        if not line_patches:
            return None

        joined = self.stack_lines(line_patches, spacing=max(2, int(round(char_height * 0.18))))
        return self.fit_patch_to_box(joined, max_width, max_height)

    def render_digital_multiline_text(self, text, max_width, max_height, char_height, font=None):
        line_patches = []
        for line in text.splitlines():
            clean_line = line.strip()
            if not clean_line:
                continue
            line_patches.append(self.render_digital_text(clean_line, char_height, font=font))
        if not line_patches:
            return None

        joined = self.stack_lines(line_patches, spacing=max(2, int(round(char_height * 0.18))))
        return self.fit_patch_to_box(joined, max_width, max_height)

    def emit_trdg_warning_once(self, error):
        if not self._trdg_warning_emitted:
            print(f"TRDG handwriting generator unavailable, falling back to PIL font rendering: {error}")
            self._trdg_warning_emitted = True

    def render_text(self, text, char_height):
        if self.generator is None:
            return self.render_handWritten_text(
                text,
                canvas_height=max(int(char_height * 1.8), 32),
                canvas_width=max(int(char_height * max(6, len(text) * 0.8)), 64),
                text_height=char_height,
                font=None,
            )
        try:
            target_height = max(1, int(round(char_height)))
            working_height = max(target_height, target_height * 3)
            image, _ = self.generator.generate(text, "#111111,#333333")
            image = self.crop_transparent_background(image.convert("RGBA"))
            image = self.resize_handwriting_patch(image, working_height, allow_upscale=True)
            image = self.thicken_image(image)
            image = self.resize_handwriting_patch(image, target_height, allow_upscale=False)
        except Exception as error:
            self.emit_trdg_warning_once(error)
            fallback_width = max(int(char_height * max(6, len(text) * 0.8)), 64)
            fallback_height = max(int(char_height * 1.8), 32)
            return self.render_handWritten_text(
                text,
                canvas_height=fallback_height,
                canvas_width=fallback_width,
                text_height=char_height,
                font=None,
            )
        return self.thicken_image(image)

    def render_digital_text(self, text, char_height, font=None):
        if font is None:
            font = load_font_with_candidates(PRINTED_FONT_CANDIDATES, char_height)
        elif isinstance(font, (str, Path)):
            font = ImageFont.truetype(str(font), max(1, int(round(char_height))))

        left, top, right, bottom = font.getbbox(text)
        text_width = max(1, right - left)
        actual_text_height = max(1, bottom - top)
        patch = Image.new("RGBA", (text_width + 24, actual_text_height + 16), (255, 255, 255, 0))
        draw = ImageDraw.Draw(patch)
        draw.text((12 - left, 8 - top), text, fill=(17, 17, 17, 255), font=font)
        return self.crop_transparent_background(patch)

    def thicken_image(self, image):
        if image.width <= 0 or image.height <= 0:
            return image

        alpha_channel = image.getchannel("A")
        luminance = ImageOps.grayscale(image.convert("RGB"))
        ink_from_luminance = ImageOps.invert(luminance)
        combined_ink = ImageChops.multiply(alpha_channel, ink_from_luminance)
        darker_alpha = combined_ink.point(
            lambda value: 0 if value < 32 else min(255, int((value - 16) * 2.2))
        )
        thick_patch = Image.new("RGBA", image.size, (12, 12, 12, 0))
        thick_patch.putalpha(darker_alpha)
        return self.crop_transparent_background(thick_patch)
    
    def render_handWritten_text(self, text, canvas_height, canvas_width, text_height, font):
        if font is None:
            font = load_font_with_candidates(HANDWRITING_FONT_CANDIDATES + PRINTED_FONT_CANDIDATES, text_height)
        elif isinstance(font, (str, Path)):
            font = ImageFont.truetype(str(font), max(1, int(round(text_height))))

        left, top, right, bottom = font.getbbox(text)
        text_width = max(1, right - left)
        actual_text_height = max(1, bottom - top)
        canvas_width = max(int(canvas_width), text_width + 24)
        canvas_height = max(int(canvas_height), actual_text_height + 16, int(text_height) + 8)

        patch = Image.new("RGBA", (canvas_width, canvas_height), (255, 255, 255, 0))
        draw = ImageDraw.Draw(patch)
        x = 10 + random.randint(-2, 3)
        y = max(4, (canvas_height - actual_text_height) // 2 + random.randint(-3, 3))
        for _ in range(3):
            offset_x = random.randint(-1, 1)
            offset_y = random.randint(-1, 1)
            alpha = random.randint(200, 240)
            draw.text((x + offset_x, y + offset_y), text, fill=(17, 17, 17, alpha), font=font)
        return self.thicken_image(patch)
    def __iter__(self):
        return self

    def __next__(self):
        raise StopIteration


class Filler:
    def __init__(
        self,
        original_root=DEFAULT_INPUT_ROOT,
        output_root=DEFAULT_OUTPUT_ROOT,
        records_per_template=DEFAULT_RECORD_COUNT,
    ):
        self.original_root = Path(original_root)
        self.output_root = Path(output_root)
        self.records_per_template = records_per_template
        self.subsets = get_dataset_subfolders(self.original_root)

    def iter_templates(self):
        for subset in self.subsets:
            image_root = self.original_root / subset / "data" / "data" / "imgs"
            json_root = self.original_root / subset / "data" / "data" / "jsons"

            for image_path in sorted(image_root.iterdir()):
                if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue

                json_path = json_root / f"{image_path.stem}.json"
                if not json_path.exists():
                    continue

                output_dir = self.output_root / subset / image_path.stem
                yield image_path, json_path, output_dir

    def iter_pending_outputs(self, output_dir):
        for sample_index in range(1, self.records_per_template + 1):
            output_path = output_dir / f"fill_{sample_index:03d}.png"
            if output_path.exists():
                continue
            yield output_path

    def fill_image(self):
        initalize_folders(self.original_root, self.output_root)

        for image_path, json_path, output_dir in self.iter_templates():
            gc.collect()
            output_dir.mkdir(parents=True, exist_ok=True)
            for output_path in self.iter_pending_outputs(output_dir):
                text_mode = random.choice(["handwritten", "digital"])
                fill_images(image_path, json_path, output_path, char_height=None, text_mode=text_mode)

if __name__ == "__main__":

    # IMAGE_PATH = "50-135_1.2.png"
    # JSON_PATH = "50-135_1.2.json"
    # OUTPUT_PATH = "filled_50-135_1.2.png"
    # fill_images(IMAGE_PATH, JSON_PATH, OUTPUT_PATH, char_height=None)
    Filler().fill_image()
