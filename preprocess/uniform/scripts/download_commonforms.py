from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from huggingface_hub import snapshot_download


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the jbarrow/CommonForms dataset into RealForm on A: with resumable Hugging Face cache."
    )
    parser.add_argument(
        "--repo-id",
        default="jbarrow/CommonForms",
        help="Hugging Face dataset repo id.",
    )
    parser.add_argument(
        "--local-dir",
        default=str(Path("A:/RealForm/data/CommonForms")),
        help="Directory where the dataset snapshot should be materialized.",
    )
    parser.add_argument(
        "--cache-root",
        default=str(Path("A:/RealForm/hf-cache")),
        help="Root cache directory kept on A: so interrupted downloads can resume.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Parallel download workers.",
    )
    parser.add_argument(
        "--post-complete-waits",
        type=int,
        default=3,
        help="Number of extra waits before exit after the download completes.",
    )
    parser.add_argument(
        "--post-complete-sleep-seconds",
        type=int,
        default=20,
        help="Seconds to sleep for each post-completion wait.",
    )
    return parser.parse_args()


def ensure_env(cache_root: Path) -> None:
    hub_cache = cache_root / "hub"
    datasets_cache = cache_root / "datasets"
    assets_cache = cache_root / "assets"
    tmp_dir = cache_root / "tmp"

    for path in (cache_root, hub_cache, datasets_cache, assets_cache, tmp_dir):
        path.mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = str(cache_root)
    os.environ["HF_HUB_CACHE"] = str(hub_cache)
    os.environ["HF_DATASETS_CACHE"] = str(datasets_cache)
    os.environ["HF_ASSETS_CACHE"] = str(assets_cache)
    os.environ["XDG_CACHE_HOME"] = str(cache_root)
    os.environ["TMP"] = str(tmp_dir)
    os.environ["TEMP"] = str(tmp_dir)


def main() -> None:
    args = parse_args()
    local_dir = Path(args.local_dir)
    cache_root = Path(args.cache_root)

    local_dir.mkdir(parents=True, exist_ok=True)
    ensure_env(cache_root)

    print(f"repo_id={args.repo_id}", flush=True)
    print(f"local_dir={local_dir}", flush=True)
    print(f"cache_root={cache_root}", flush=True)
    print("If the process is terminated, rerun this script to resume from the cache on A:.", flush=True)

    resolved_path = snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        cache_dir=cache_root / "hub",
        local_dir=local_dir,
        force_download=False,
        max_workers=args.max_workers,
    )

    print(f"download_complete={resolved_path}", flush=True)

    for wait_index in range(args.post_complete_waits):
        print(
            f"post_complete_wait {wait_index + 1}/{args.post_complete_waits} "
            f"sleeping {args.post_complete_sleep_seconds}s before exit",
            flush=True,
        )
        time.sleep(args.post_complete_sleep_seconds)


if __name__ == "__main__":
    main()
