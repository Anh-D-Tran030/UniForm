from __future__ import annotations

import argparse
import base64
import csv
import json
import math
import mimetypes
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import requests


ROOT = Path("A:/RealForm")
ENDPOINT = (
    "https://us-documentai.googleapis.com/v1/"
    "projects/971280404964/locations/us/processors/257ccb8b78c21b0:process"
)
COMMONFORMS_IMAGES = ROOT / "processed/CommonFormsEnglish/images"
SPLIT_ROOT = ROOT / "processed/synthetic_fill_images_10k_folders_degraded_ocr_split_9400_300_300_augmented"
DEGRADED_10K = ROOT / "processed/synthetic_fill_images_10k_folders_degraded"
OUT_DIR = ROOT / "processed/documentai_classifier_retrieval_benchmark_us_257ccb8b78c21b0/test"
LOG_PATH = ROOT / "logs/documentai-classifier-us-257ccb8b78c21b0-test.log"
GCLOUD_CONFIG = ROOT / ".gcloud-codex"
GCLOUD_CMD = Path(r"C:\Users\thanh\AppData\Local\Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd")


@dataclass(frozen=True)
class GalleryItem:
    stem: str
    image_path: Path


@dataclass(frozen=True)
class QueryItem:
    template_stem: str
    query_id: str
    image_path: Path


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text: str) -> int:
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_original_images() -> dict[str, Path]:
    originals: dict[str, Path] = {}
    for path in COMMONFORMS_IMAGES.rglob("*"):
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}:
            originals[path.stem] = path
    return originals


def load_test_split() -> tuple[list[GalleryItem], list[QueryItem]]:
    originals = find_original_images()
    stems = sorted(p.name for p in (SPLIT_ROOT / "test").iterdir() if p.is_dir())
    gallery: list[GalleryItem] = []
    queries: list[QueryItem] = []

    for stem in stems:
        original = originals.get(stem)
        query_dir = DEGRADED_10K / stem
        fills = sorted(query_dir.glob("fill_*.png"))
        if not original or not fills:
            continue
        gallery.append(GalleryItem(stem=stem, image_path=original))
        for fill_path in fills:
            queries.append(QueryItem(stem, fill_path.stem, fill_path))

    print(f"Loaded test split: {len(gallery)} gallery templates, {len(queries)} query images")
    return gallery, queries


def mime_type(path: Path) -> str:
    guessed, _ = mimetypes.guess_type(str(path))
    if guessed:
        return guessed
    if path.suffix.lower() in {".jpg", ".jpeg"}:
        return "image/jpeg"
    return "image/png"


def get_access_token() -> str:
    env = os.environ.copy()
    if GCLOUD_CONFIG.exists():
        env["CLOUDSDK_CONFIG"] = str(GCLOUD_CONFIG)
    result = subprocess.run(
        [str(GCLOUD_CMD), "auth", "print-access-token"],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    token = result.stdout.strip()
    if not token:
        raise RuntimeError("gcloud returned an empty access token")
    print("Using gcloud access token from active account")
    return token


def process_document(path: Path, token: str, cache_path: Path, timeout_seconds: int) -> dict:
    if cache_path.exists():
        return read_json(cache_path)

    raw = base64.b64encode(path.read_bytes()).decode("ascii")
    payload = {
        "rawDocument": {
            "content": raw,
            "mimeType": mime_type(path),
        }
    }
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    response = requests.post(ENDPOINT, headers=headers, json=payload, timeout=timeout_seconds)
    if response.status_code >= 400:
        detail_path = cache_path.with_suffix(".error.json")
        write_json(
            detail_path,
            {
                "status_code": response.status_code,
                "text": response.text[:4000],
                "image_path": str(path),
            },
        )
        raise RuntimeError(f"Document AI failed for {path}: {response.status_code}; see {detail_path}")

    data = response.json()
    write_json(cache_path, data)
    return data


def entity_vector(response: dict) -> dict[str, float]:
    document = response.get("document", response)
    vector: dict[str, float] = {}

    for entity in document.get("entities", []) or []:
        label = (
            entity.get("type")
            or entity.get("mentionText")
            or entity.get("normalizedValue", {}).get("text")
            or "UNKNOWN"
        )
        conf = float(entity.get("confidence", 0.0) or 0.0)
        vector[label] = max(vector.get(label, 0.0), conf)

    # Some classifier responses put class predictions under pages/entities-like fields.
    for page in document.get("pages", []) or []:
        for detected in page.get("detectedLanguages", []) or []:
            label = "lang:" + str(detected.get("languageCode", "unknown"))
            conf = float(detected.get("confidence", 0.0) or 0.0)
            vector[label] = max(vector.get(label, 0.0), conf)

    return vector


def cosine(a: dict[str, float], b: dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(v * b.get(k, 0.0) for k, v in a.items())
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0


def top_labels(vector: dict[str, float], n: int = 5) -> list[dict[str, float | str]]:
    return [
        {"label": label, "confidence": score}
        for label, score in sorted(vector.items(), key=lambda kv: kv[1], reverse=True)[:n]
    ]


def run(args) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    gallery, queries = load_test_split()
    if args.limit_templates:
        keep = {item.stem for item in gallery[: args.limit_templates]}
        gallery = [item for item in gallery if item.stem in keep]
        queries = [item for item in queries if item.template_stem in keep]
    if args.limit_queries:
        queries = queries[: args.limit_queries]
    print(f"Benchmark scope: {len(gallery)} gallery, {len(queries)} queries")

    token = get_access_token()
    gallery_vectors: dict[str, dict[str, float]] = {}
    gallery_rows = []

    for idx, item in enumerate(gallery, start=1):
        response = process_document(
            item.image_path,
            token,
            OUT_DIR / "cache/gallery" / f"{item.stem}.json",
            args.timeout_seconds,
        )
        vector = entity_vector(response)
        gallery_vectors[item.stem] = vector
        gallery_rows.append({"stem": item.stem, "image_path": str(item.image_path), "labels": top_labels(vector)})
        if idx == 1:
            write_json(OUT_DIR / "sample_gallery_response.json", response)
        if idx % 25 == 0 or idx == len(gallery):
            print(f"Processed gallery {idx}/{len(gallery)}")

    correct_1 = correct_5 = correct_10 = 0
    mrr = 0.0
    per_query_path = OUT_DIR / "per_query_results.csv"
    with per_query_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "query_path",
                "template_stem",
                "rank",
                "top1_stem",
                "top1_score",
                "top10_stems",
                "top10_scores",
                "query_labels",
            ],
        )
        writer.writeheader()

        for idx, query in enumerate(queries, start=1):
            response = process_document(
                query.image_path,
                token,
                OUT_DIR / "cache/queries" / query.template_stem / f"{query.query_id}.json",
                args.timeout_seconds,
            )
            if idx == 1:
                write_json(OUT_DIR / "sample_query_response.json", response)
            qvec = entity_vector(response)
            scored = [
                (stem, cosine(qvec, gvec))
                for stem, gvec in gallery_vectors.items()
            ]
            scored.sort(key=lambda row: row[1], reverse=True)
            ranked_stems = [stem for stem, _ in scored]
            rank = ranked_stems.index(query.template_stem) + 1 if query.template_stem in ranked_stems else len(scored) + 1
            top10 = scored[:10]

            correct_1 += int(rank <= 1)
            correct_5 += int(rank <= 5)
            correct_10 += int(rank <= 10)
            mrr += 1.0 / rank

            writer.writerow(
                {
                    "query_path": str(query.image_path),
                    "template_stem": query.template_stem,
                    "rank": rank,
                    "top1_stem": top10[0][0] if top10 else "",
                    "top1_score": top10[0][1] if top10 else 0.0,
                    "top10_stems": "|".join(stem for stem, _ in top10),
                    "top10_scores": "|".join(f"{score:.6f}" for _, score in top10),
                    "query_labels": json.dumps(top_labels(qvec), ensure_ascii=False),
                }
            )
            if idx % 50 == 0 or idx == len(queries):
                print(f"Processed queries {idx}/{len(queries)}")

    n = max(len(queries), 1)
    summary = {
        "endpoint": ENDPOINT,
        "processor_type": "custom_classifier",
        "gallery_count": len(gallery),
        "query_count": len(queries),
        "retrieval_at_1": correct_1 / n,
        "retrieval_at_5": correct_5 / n,
        "retrieval_at_10": correct_10 / n,
        "mrr": mrr / n,
        "correct_at_1": correct_1,
        "correct_at_5": correct_5,
        "correct_at_10": correct_10,
        "per_query_csv": str(per_query_path),
        "sample_gallery_response": str(OUT_DIR / "sample_gallery_response.json"),
        "sample_query_response": str(OUT_DIR / "sample_query_response.json"),
    }
    write_json(OUT_DIR / "documentai_classifier_retrieval_summary.json", summary)
    write_json(OUT_DIR / "gallery_labels.json", gallery_rows)
    print(json.dumps(summary, indent=2))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit-templates", type=int, default=0)
    parser.add_argument("--limit-queries", type=int, default=0)
    parser.add_argument("--timeout-seconds", type=int, default=180)
    return parser.parse_args()


if __name__ == "__main__":
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    with LOG_PATH.open("a", encoding="utf-8") as log:
        sys.stdout = Tee(sys.__stdout__, log)
        sys.stderr = Tee(sys.__stderr__, log)
        try:
            print(f"\n===== Document AI classifier benchmark {time.strftime('%Y-%m-%d %H:%M:%S')} =====")
            print(f"Endpoint: {ENDPOINT}")
            print(f"Command: {' '.join(sys.argv)}")
            run(parse_args())
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
