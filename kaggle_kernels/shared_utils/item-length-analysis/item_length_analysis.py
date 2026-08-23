"""
How much information do we lose by truncating item text fields to
MAX_NAME_CHARS=128 / MAX_CATEGORY_CHARS=64 / MAX_ATTR_CHARS=384 /
MAX_ITEM_CHARS=640 (see CLAUDE.md "Text preprocessing")?

Computes the UNTRUNCATED character-length distribution (deciles) for each
raw field across the full items.parquet catalog (13M+ rows), plus what
fraction of rows would actually get truncated at each current limit.

CPU-only, no GPU needed (enable_gpu: false in kernel-metadata.json) --
designed to run standalone in parallel with the (GPU) training experiment in
kaggle_kernels/lora-llm-continue-train/ without competing for GPU quota.
"""
import os
import json

import numpy as np
import polars as pl

ITEMS_PATH = "/kaggle/input/datasets/maxamsecret/ozon-ecup-2026-matching/items.parquet"

MAX_NAME_CHARS = 128
MAX_CATEGORY_CHARS = 64
MAX_ATTR_CHARS = 384
PREFIX_CHARS = len("Name: ") + len(" Category: ") + len(" Attributes: ")  # 30
MAX_ITEM_CHARS = MAX_NAME_CHARS + MAX_CATEGORY_CHARS + MAX_ATTR_CHARS + 64  # 640, matches CLAUDE.md's "+64" headroom

WORKING_DIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."

DECILES = list(range(0, 101, 10))


def main():
    if os.path.isdir("/kaggle/input"):
        for _root, _dirs, _files in os.walk("/kaggle/input"):
            depth = _root[len("/kaggle/input"):].count(os.sep)
            if depth <= 3:
                print(f"[DEBUG] {_root} -> dirs={_dirs} files={_files}")

    print("[DATA] Scanning items.parquet for field lengths (lazy, streaming, no full-text materialization) ...")
    lengths = (
        pl.scan_parquet(ITEMS_PATH)
        .select([
            pl.col("name").fill_null("").cast(pl.Utf8, strict=False).str.len_chars().alias("name_len"),
            pl.col("category").fill_null("").cast(pl.Utf8, strict=False).str.len_chars().alias("category_len"),
            pl.col("attributes").fill_null("").cast(pl.Utf8, strict=False).str.len_chars().alias("attributes_len"),
        ])
        .with_columns(
            (pl.col("name_len") + pl.col("category_len") + pl.col("attributes_len") + PREFIX_CHARS)
            .alias("combined_len")
        )
        .collect(engine="streaming")
    )
    n = lengths.height
    print(f"[DATA] items scanned: {n:,}")

    report = {"n_items": n, "fields": {}}

    for col, limit in [
        ("name_len", MAX_NAME_CHARS),
        ("category_len", MAX_CATEGORY_CHARS),
        ("attributes_len", MAX_ATTR_CHARS),
        ("combined_len", MAX_ITEM_CHARS),
    ]:
        arr = lengths.get_column(col).to_numpy()
        pct = {f"p{d}": float(np.percentile(arr, d)) for d in DECILES}
        truncated_frac = float((arr > limit).mean())

        report["fields"][col] = {
            "current_limit_chars": limit,
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "max": float(arr.max()),
            "percentiles": pct,
            "fraction_truncated": truncated_frac,
        }

        print("\n" + "=" * 78)
        print(f"[{col}]  current_limit={limit} chars  |  "
              f"mean={arr.mean():.1f}  std={arr.std():.1f}  max={arr.max():.0f}  |  "
              f"fraction_truncated={truncated_frac:.2%}")
        print("=" * 78)
        for d in DECILES:
            marker = "  <-- current limit" if d > 0 and pct[f"p{d - 10}"] <= limit < pct[f"p{d}"] else ""
            print(f"  p{d:3d} = {pct[f'p{d}']:7.1f} chars{marker}")

    out_path = os.path.join(WORKING_DIR, "item_length_report.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[DONE] report saved to {out_path}")


if __name__ == "__main__":
    main()
