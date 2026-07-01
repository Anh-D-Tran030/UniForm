import json
import random
from pathlib import Path
# from trdg import handwritten_text_generator as handwritten_text_generator
from PIL import Image, ImageDraw, ImageFont
import gc


DEFAULT_RECORD_COUNT = 10
DEFAULT_INPUT_ROOT = Path("CommonFormsEnglish")
DEFAULT_OUTPUT_ROOT = Path("synthetic_data_common")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
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
AVAILABLE_DIGITAL_FONT_PATHS = []
def initalize_font(paths = AVAILABLE_DIGITAL_FONT_PATHS):
    seen = set()
    for name in DIGITAL_FONT_CANDIDATES:
        canidate = Path.joinpath(WINDOWS_FONTS_DIR,name)
        key = str(canidate).lower()
        if canidate.exists() and canidate.suffix.lower() in {".ttf", ".otf"} and key not in seen:
                paths.append(canidate)
                seen.add(key)
    for canidate in FONTS_DIR.glob("*"):
        key = str(canidate).lower()
        if canidate.is_file() and canidate.suffix.lower() in {".ttf", ".otf"}:
            paths.append(canidate)
            seen.add(key)

def build_digital_font_pool_for_size(font_size: float) -> list[ImageFont.ImageFont]:
    size = max(1, int(round(font_size)))
    pool: list[ImageFont.ImageFont] = []
    for font_path in AVAILABLE_DIGITAL_FONT_PATHS:
        try:
            f = ImageFont.truetype(str(font_path), size)
            f.getbbox("Test 123")
            pool.append(ImageFont.truetype(str(font_path), size))

        except OSError:
            continue
    if not pool:
        pool.append(ImageFont.load_default())
    return pool



def initalize_folders(original_root, synthetic_root):
    original_root = Path(original_root)
    synthetic_root = Path(synthetic_root)

    if not original_root.exists():
        raise FileNotFoundError(f"Original root folder '{original_root}' does not exist.")

    synthetic_root.mkdir(parents=True, exist_ok=True)
    sub_folders = ["train", "test"]
    for sub_folder in sub_folders:
        original_subset_root = original_root / "images" / sub_folder
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
    widgets = data.get("regions", [])

    bboxes = []
    
    for item in widgets:
        bbox = item.get("bbox",[])
        x_top_left = bbox[0]
        y_top_left = bbox[1]
        x_top_right = bbox[0] + bbox[2]
        y_top_right = bbox[1]
        x_bottom_left = bbox[0]
        y_bottom_left = bbox[1] + bbox[3]

        x_bottom_right = bbox[0] + bbox[3]
        y_bottom_right = bbox[1] + bbox[3]

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

def crop_transparent_background(image):
    a_box = image.getchannel("A").getbbox()
    if a_box:
        return image.crop(a_box)
    return image


def stack_lines(images, spacing=4):
    if not images:
        return Image.new("RGBA", (1, 1), (255, 255, 255, 0))
    if len(images) == 1:
        return crop_transparent_background(images[0])

    max_width = max(patch.width for patch in images)
    total_height = sum(patch.height for patch in images) + spacing * (len(images) - 1)
    canvas = Image.new("RGBA", (max(1, max_width), max(1, total_height)), (255, 255, 255, 0))

    cursor_y = 0
    for patch in images:
        canvas.alpha_composite(patch, dest=(0, cursor_y))
        cursor_y += patch.height + spacing
    return crop_transparent_background(canvas)


def fit_patch_to_box(patch, max_width, max_height):
    if patch.width <= 0 or patch.height <= 0:
        return patch

    scale = min(max_width / patch.width, max_height / patch.height, 1.0)
    if scale <= 0:
        return Image.new("RGBA", (1, 1), (255, 255, 255, 0))
    if scale < 1.0:
        new_width = max(1, int(round(patch.width * scale)))
        new_height = max(1, int(round(patch.height * scale)))
        patch = patch.resize((new_width, new_height), resample=Image.LANCZOS)
    return crop_transparent_background(patch)


def render_digital_text(text, char_height, font=None):
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
    return crop_transparent_background(patch)


def render_digital_multiline_text(text, max_width, max_height, char_height, font=None):
    line_patches = []
    for line in text.splitlines():
        clean_line = line.strip()
        if not clean_line:
            continue
        line_patches.append(render_digital_text(clean_line, char_height, font=font))
    if not line_patches:
        return None

    joined = stack_lines(line_patches, spacing=max(2, int(round(char_height * 0.18))))
    return fit_patch_to_box(joined, max_width, max_height)


def fill_images(image_path, json_path, output_path, font=None, char_height=None):
    image = load_image(image_path)
    bboxes = load_bboxes(json_path)
    if char_height is None:
        char_height = calculate_good_height(bboxes)
    font_pool = build_digital_font_pool_for_size(char_height)
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

        rendered = render_digital_multiline_text(
            text,
            box_width,
            box_height,
            char_height,
            font=random.choice(font_pool),
        )
        if rendered is None:
            continue

        image.paste("white", [x_min, y_min, x_max, y_max])
        alpha = rendered.getchannel("A")
        image.paste(rendered.convert("RGB"), (x_min, y_min), alpha)

    image.save(output_path)


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
        self.subsets = ["train","test"]

    def iter_templates(self):
        for subset in self.subsets:
            image_root = self.original_root /"images"/ subset
            json_root = self.original_root / "regions"/ subset

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
            output_path = output_dir / f"synthetic_{sample_index:03d}.png"
            if output_path.exists():
                continue
            yield output_path

    def fill_image(self):
        initalize_folders(self.original_root, self.output_root)

        for image_path, json_path, output_dir in self.iter_templates():
            gc.collect()
            output_dir.mkdir(parents=True, exist_ok=True)
            for output_path in self.iter_pending_outputs(output_dir):
                try:
                    fill_images(image_path, json_path, output_path, char_height=None)
                except OSError:
                    gc.collect()
                    fill_images(image_path, json_path, output_path, char_height=None)


if __name__ == "__main__":

    # IMAGE_PATH = "50-135_1.2.png"
    # JSON_PATH = "50-135_1.2.json"
    # OUTPUT_PATH = "filled_50-135_1.2.png"
    # fill_images(IMAGE_PATH, JSON_PATH, OUTPUT_PATH, char_height=None)
    initalize_font()
    Filler().fill_image()
