#!/usr/bin/env python3
"""Remove all demo-seed objects from MinIO.

Deletes every object in the ``document-extractions`` bucket whose name contains
the ``demo-seed`` prefix/tag (both bronze images and silver JSON). This is the
inverse of seed_demo_data.py and does NOT touch any of your real forms or the
dashboard metrics file.

Run locally (NOT in the sandbox):

    python scripts/remove_demo_data.py            # delete, with a confirm prompt
    python scripts/remove_demo_data.py --yes      # skip the prompt
    python scripts/remove_demo_data.py --dry-run  # just list what would be deleted
"""

from __future__ import annotations

import argparse
import os

try:
    from minio import Minio
except ImportError:  # pragma: no cover
    raise SystemExit("The 'minio' package is required. Install it with:\n    pip install minio")

MINIO_ENDPOINT = os.environ.get("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY = os.environ.get("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.environ.get("MINIO_SECRET_KEY", "minioadmin123")
MINIO_SECURE = os.environ.get("MINIO_SECURE", "false").lower() == "true"
BUCKET = os.environ.get("MINIO_BUCKET", "document-extractions")

DEMO_TAG = "demo-seed"


def main() -> None:
    ap = argparse.ArgumentParser(description="Remove demo-seed objects from MinIO.")
    ap.add_argument("--yes", action="store_true", help="Skip the confirmation prompt.")
    ap.add_argument("--dry-run", action="store_true", help="List matches without deleting.")
    args = ap.parse_args()

    print(f"Connecting to MinIO at {MINIO_ENDPOINT} ...")
    client = Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=MINIO_SECURE,
    )

    if not client.bucket_exists(BUCKET):
        print(f"Bucket '{BUCKET}' does not exist. Nothing to do.")
        return

    matches = [
        obj.object_name
        for obj in client.list_objects(BUCKET, recursive=True)
        if DEMO_TAG in obj.object_name
    ]

    if not matches:
        print("No demo-seed objects found. Nothing to remove.")
        return

    print(f"Found {len(matches)} demo-seed object(s):")
    for name in matches:
        print(f"  - {name}")

    if args.dry_run:
        print("\nDry run: nothing deleted.")
        return

    if not args.yes:
        reply = input(f"\nDelete these {len(matches)} object(s)? [y/N] ").strip().lower()
        if reply not in {"y", "yes"}:
            print("Aborted.")
            return

    removed = 0
    for name in matches:
        client.remove_object(BUCKET, name)
        removed += 1

    print(f"\nDeleted {removed} demo-seed object(s) from '{BUCKET}'.")
    print("Refresh the Uploaded Forms page to confirm.")


if __name__ == "__main__":
    main()
