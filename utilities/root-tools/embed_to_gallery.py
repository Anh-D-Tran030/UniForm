import time
from pathlib import Path

import requests


API_URL = "http://localhost:8005"
SOURCE_DIR = Path("to_gallery")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
DELAY_SECONDS = 2


def main() -> int:
    images = sorted(
        path for path in SOURCE_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not images:
        print("No images found in to_gallery.", flush=True)
        return 1

    print(f"Found {len(images)} images.", flush=True)

    failures = []
    for index, image_path in enumerate(images, start=1):
        template_id = f"t_{index}"
        display_name = f"Template {index}"
        print(f"[{index}/{len(images)}] Embedding {image_path.name} as {template_id} ({display_name})", flush=True)

        try:
            with image_path.open("rb") as image_file:
                response = requests.post(
                    f"{API_URL}/embed",
                    data={"template_id": template_id, "display_name": display_name},
                    files={"image": (image_path.name, image_file, "application/octet-stream")},
                    timeout=600,
                )
        except requests.RequestException as exc:
            failures.append((template_id, image_path.name, str(exc)))
            print(f"  FAILED: {exc}", flush=True)
        else:
            if response.ok:
                print("  OK", flush=True)
            else:
                try:
                    detail = response.json()
                except ValueError:
                    detail = response.text
                failures.append((template_id, image_path.name, str(detail)))
                print(f"  FAILED: HTTP {response.status_code}: {detail}", flush=True)

        if index < len(images):
            time.sleep(DELAY_SECONDS)

    print("", flush=True)
    if failures:
        print(f"Finished with {len(failures)} failure(s):", flush=True)
        for template_id, filename, error in failures:
            print(f"- {template_id} ({filename}): {error}", flush=True)
        return 1

    print("Finished successfully. All images embedded.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
