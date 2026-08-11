"""
Download the 4 competition data files straight from the organizer's Yandex
Cloud bucket into data/ -- no Kaggle account/credentials needed (unlike the
kaggle-dataset route documented in README.md). Stdlib only.

Usage:
    python scripts/download_data.py                              # all 4 files
    python scripts/download_data.py matches.parquet items_human.parquet
    python scripts/download_data.py --overwrite items.parquet    # force redownload

Notes:
    - Skips a file if it already exists in data/ (pass --overwrite to force).
    - Downloads to a "<file>.parquet.part" temp name first, then atomically
      renames on success -- an interrupted download never leaves a corrupt
      *.parquet file in place, but isn't resumable (delete the .part and
      rerun if interrupted, no partial-range support here).
"""

import argparse
import os
import sys
import urllib.request

BASE_URL = "https://storage.yandexcloud.net/ozon-ecup-2026/Matching"
FILES = ["matches.parquet", "matches_llm.parquet", "items.parquet", "items_human.parquet"]
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def _format_bytes(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def download_file(filename, dest_dir=DATA_DIR, overwrite=False):
    url = f"{BASE_URL}/{filename}"
    dest_path = os.path.join(dest_dir, filename)

    if os.path.exists(dest_path) and not overwrite:
        print(f"[SKIP] {filename} already exists at {dest_path} (--overwrite to redownload)")
        return dest_path

    os.makedirs(dest_dir, exist_ok=True)
    tmp_path = dest_path + ".part"

    print(f"[DOWNLOAD] {url}")

    def _report(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100.0, downloaded * 100.0 / total_size)
            print(f"\r  {filename}: {_format_bytes(downloaded)}/{_format_bytes(total_size)} ({pct:.1f}%)", end="", flush=True)
        else:
            print(f"\r  {filename}: {_format_bytes(downloaded)}", end="", flush=True)

    try:
        urllib.request.urlretrieve(url, tmp_path, reporthook=_report)
        print()
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise

    os.replace(tmp_path, dest_path)
    print(f"[DONE] {filename} -> {dest_path} ({_format_bytes(os.path.getsize(dest_path))})")
    return dest_path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="*", help=f"specific files to download (default: all). Choices: {', '.join(FILES)}")
    parser.add_argument("--overwrite", action="store_true", help="redownload even if the file already exists")
    args = parser.parse_args()

    targets = args.files or FILES
    unknown = [f for f in targets if f not in FILES]
    if unknown:
        parser.error(f"unknown file(s): {unknown}. Choices: {FILES}")

    for filename in targets:
        download_file(filename, overwrite=args.overwrite)


if __name__ == "__main__":
    sys.exit(main())
