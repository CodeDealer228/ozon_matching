"""
Generate the canonical train/val split of matches.parquet.

This is the single source of truth for which pairs are "val" -- it must be
byte-for-byte reproducible from matches.parquet using SEED/stratify alone,
because notebooks/cross-encoder-sft-pipline (3).ipynb, the Kaggle validation
kernel (kaggle_kernels/validate-baseline-vs-lora), and any future script all
need to land on the exact same val set or scores leak/aren't comparable.

Run once locally:
    python scripts/make_canonical_split.py

Writes data/splits/matches_split_seed42.parquet with columns:
    id1, id2, label, split ("train" | "val")
and prints its sha256 so it can be pinned in experiment logs / CLAUDE.md.
"""

import hashlib
import os

import numpy as np
import polars as pl
from sklearn.model_selection import train_test_split

SEED = 42
TEST_SIZE = 0.20

MATCHES_PATH = os.path.join("data", "matches.parquet")
OUT_PATH = os.path.join("data", "splits", "matches_split_seed42.parquet")


def main():
    matches_pl = pl.read_parquet(MATCHES_PATH)
    matches_pl = (
        matches_pl
        .filter(pl.col("id1").is_not_null() & pl.col("id2").is_not_null())
        .with_columns([
            pl.col("target").fill_null(0).cast(pl.Float32),
            (pl.col("target") >= 0.5).cast(pl.Int8).alias("label"),
        ])
        .select(["id1", "id2", "label"])
    )
    matches_pd = matches_pl.to_pandas()

    labels_all = matches_pd["label"].to_numpy().astype(np.int64)
    indices = np.arange(len(labels_all), dtype=np.int64)

    train_idx, val_idx = train_test_split(
        indices, test_size=TEST_SIZE, random_state=SEED, stratify=labels_all
    )

    split_col = np.full(len(matches_pd), "train", dtype=object)
    split_col[val_idx] = "val"

    out = matches_pd.copy()
    out["split"] = split_col

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    out_pl = pl.from_pandas(out)
    out_pl.write_parquet(OUT_PATH)

    with open(OUT_PATH, "rb") as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()

    print(f"[SPLIT] rows total={len(out):,} | train={len(train_idx):,} | val={len(val_idx):,}")
    print(f"[SPLIT] written to {OUT_PATH}")
    print(f"[SPLIT] sha256={file_hash}")


if __name__ == "__main__":
    main()
