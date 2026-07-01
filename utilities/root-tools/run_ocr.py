import argparse
import json
import time
from dataclasses import dataclass
import random
from pathlib import Path
import pytesseract
from PIL import Image, ImageDraw, ImageOps


################################
# OVERALL PLAN:
# 1. Find degrade Image
# 2. Group into templates
# 3. Run tesseract
# 4. Keep ocr words, bboxes in x1x2y1y2
# 5. Mask and corrupt random
# 6. Iteratively split 92 train done move to 3 val to 3 test
################################

DEFAULT_DEGRADED_ROOT = Path(r" ./synthetic_data_common_degraded_cache")
DEFAULT_CACHE_ROOT = Path(r"./synthetic_data_common_degraded/train")
DEFAULT_TESSERACT_CMD = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
IMAGE_GLOB = "fill_*.png"
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

# Copy from fill for corruptions
FULL_VOCABULARY = set(FIRST_NAMES + LAST_NAMES + COMPANY_NAMES + STREET_ADDRESSES + CITY_STATE_LINES + VOCABULARY_WORDS)


def ocr_image(image_path, psm, oem, min_confidence):
    with Image.open(image_path) as img:
        # 1. Load images , convert and get width and height
        image = img.convert("RGB")
        w,h = image.size
        scale = 2
        processed = ImageOps.autocontrast(
            image.convert("L").resize((int(w * scale), int(h * scale)), resample=Image.Resampling.BICUBIC)
        )
        # 2. Run tesseract 
        config = f"--psm {psm} --oem {oem}"
        data = pytesseract.image_to_data(processed, config=config, output_type=pytesseract.Output.DICT, lang="eng")
        words = []
        bboxes = []
        confidences = []
        
        # 3. extract bboxes, if no words then empty
        for i in range(len(data.get("text", []))):
            try:
                confidence = float(data["conf"][i])
            except (TypeError, ValueError):
                confidence = -1.0
            if confidence < min_confidence:
                continue
            text = data["text"][i].strip()
            conf = int(data["conf"][i])
            if text and conf >= min_confidence:
                words.append(text)
                # 4. convert to x1x2y1y2
                x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
                bboxes.append((x, y, x + w, y + h))
                confidences.append(conf)
        if not words:
            words = ["[EMPTY]"]
            bboxes = [(0, 0, 1, 1)]
            confidences = [0]
        # 5. return json
        return{
            "image_name": image_path.name,
            "image_size": {"width": w, "height": h},
            "word_count": len(words),
            "words": words,
            "bboxes": bboxes,
            "confidences": confidences,
            "engine": "tesseract",
            "psm": psm,
            "oem": oem
        }
def corrupt_and_mask_ocr_results(payload, mask_rate, corrupt_rate, seed):
    # 1. shuffle the words list indicies
    rng = random.Random(36)
    words = list(payload["words"])
    indices = list(range(len(words)))
    rng.shuffle(indices)
    mask_count = round(len(words)*mask_rate)
    corrupt_count = round(len(words)*corrupt_rate)
    # 2. take first mask rate to mask and the following corrupt rate to corrupt
    for i in indices[:mask_count]:
        words[i] = "[MASK]"
    for i in indices[mask_count : mask_count + corrupt_count]:
        words[i] = rng.choice(FULL_VOCABULARY)
    updated = dict(payload)
    updated["words"] = words
    updated["mask_rate"] = mask_rate
    updated["corrupt_rate"] = corrupt_rate
    return updated
def save_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encosding="utf-8")

if __name__ == "__main__":
    pytesseract.pytesseract.tesseract_cmd = str(DEFAULT_TESSERACT_CMD)

    ##NOTE: Change 10k -> 8k for stress test  propportion does not matter. file manipulation required


    TOTAL_TEMPLATES = 10000 # Tottal templates to ocr
    PROPORTION_TRAIN = 94 # OCR 94 train then Move on
    PROPORTION_VAL = 3
    PROPORTION_TEST = 3
    OVERWRITE_EXISTING = True # For re-running
    FULL_VOCABULARY = list(FULL_VOCABULARY)

    input_root = Path("./synthetic_data_common_degraded")

    output_root = Path(str(DEFAULT_CACHE_ROOT).strip())
    if output_root.name in {"train", "val", "test"}:
        output_root = output_root.parent

    image_paths = sorted(input_root.rglob(IMAGE_GLOB)) # sort so it does not complains

    split_order = ["train", "val", "test"]
    split_proportions = {
        "train": PROPORTION_TRAIN,
        "val": PROPORTION_VAL,
        "test": PROPORTION_TEST,
    }

    source_splits = {"train", "test"} # NOTE: this only applies to olds one new one brought everything out
    templates = {}
    for image_path in image_paths: # Get all of the paths
        relative = image_path.relative_to(input_root) # get the relative path
        if len(relative.parts) < 3:
            continue #skiping anything outside, we only want the templae
        if relative.parts[0] not in source_splits:
            continue #the the next root should be either train or test
        template_rel = Path(relative.parts[1])
        templates.setdefault(template_rel, []).append(image_path)

    template_items = sorted(templates.items(), key=lambda kv: str(kv[0]))

    template_items = template_items[: min(TOTAL_TEMPLATES, len(template_items))] # ensure the it does not go beyond number of template inside the gallery and extract only 10k templates

    selected_template_count = len(template_items)
    total_weight = sum(split_proportions.values())
    split_template_targets = {"train": 0, "val": 0, "test": 0}
    remainders = []
    assigned = 0
    for split_name in split_order: # caculate exact number of samples in each split
        weight = split_proportions[split_name]
        if weight <= 0:
            continue
        exact = (selected_template_count * weight) / total_weight
        base = int(exact)
        split_template_targets[split_name] = base
        assigned += base
        remainders.append((exact - base, split_name))
    remaining = selected_template_count - assigned # This probably be 0 but just keep in case
    for _, split_name in sorted(remainders, key=lambda x: (-x[0], split_order.index(x[1]))): ## put in the largest first (determine by the the decimal place) this equivilance to put everything in train set
        if remaining <= 0:
            break
        split_template_targets[split_name] += 1
        remaining -= 1

    template_split_sequence = []
    for split_name in split_order:
        template_split_sequence.extend([split_name] * split_template_targets[split_name])#Applying the mask of the order

    split_counts = {"train": 0, "val": 0, "test": 0}
    mode_counts = {"clean": 0, "mask": 0, "corrupt": 0}
    processed = 0
    processed_templates = 0
    random.Random(42).shuffle(template_items)

    prepared_templates = []
    for idx, (template_rel, template_images) in enumerate(template_items):
        assigned_split = template_split_sequence[idx] #split of it belongs
        template_images = sorted(template_images) #take out the filled forms
        n = len(template_images)
        if n == 0:
            continue

        n_clean = int(n * 0.50) # 50% going to be clean images
        n_mask = int(n * 0.30) # 30% going to mask only
        n_corrupt = n - n_clean - n_mask # remain 20% will apply corruption extras

        assignment_order = list(range(n))
        rng = random.Random(f"{template_rel}")
        rng.shuffle(assignment_order)
        # map the the template to the fill mode 
        mode_by_index = ["clean"] * n
        for idx in assignment_order[n_clean:n_clean + n_mask]:
            mode_by_index[idx] = "mask"
        for idx in assignment_order[n_clean + n_mask:n_clean + n_mask + n_corrupt]:
            mode_by_index[idx] = "corrupt"

        template_samples = []
        for i, image_path in enumerate(template_images):
            template_samples.append((image_path, mode_by_index[i]))
        prepared_templates.append((assigned_split, template_rel, template_samples))
        
    #DONE PREPARATION MOVE ON TO OCR EXECUTION
    for split_name, template_rel, template_samples in prepared_templates:
        wrote_any = False
        for image_path, mode in template_samples:
            output_path = output_root / split_name / template_rel / f"{image_path.stem}.ocr.json"
            if output_path.exists() and not OVERWRITE_EXISTING:
                continue

            clean = ocr_image(image_path, psm=6, oem=3, min_confidence=0)
            if mode == "clean":
                payload = dict(clean)
                payload["mask_rate"] = 0.0
                payload["corrupt_rate"] = 0.0
            elif mode == "mask":
                payload = corrupt_and_mask_ocr_results(clean, 0.25, 0.0, str(image_path))
            else:
                payload = corrupt_and_mask_ocr_results(clean, 0.20, 0.10, str(image_path))

            save_json(output_path, payload)
            wrote_any = True
            split_counts[split_name] += 1
            mode_counts[mode] += 1
            processed += 1
        if wrote_any:
            processed_templates += 1