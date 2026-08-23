"""
Samples a small, human-readable set of labeled pairs per category, for
manual negative-pattern mining (the goal: write down, per category, the
logical rules that make a pair a non-match in matches.parquet's human
labels, to later drive synthetic negative generation).

For each of the 20 categories: 15 negative (label=0) + 5 positive (label=1)
pairs, sampled uniformly at random (seed=42, reproducible), restricted to
pairs where BOTH items have "adequate" text length -- len(name) +
len(attributes) <= MAX_ITEM_TEXT_CHARS (raw string lengths, matching how
these fields are actually stored -- attributes is a raw JSON string, not
parsed, same convention as the production run.py per CLAUDE.md). This keeps
the sample readable for manual review (no walls of truncated JSON) while
still reflecting real labeled data, not synthetic examples.

CPU-only (no GPU/TPU needed -- this is a pandas/polars join + sample, not a
model run), matching this repo's convention of running non-GPU-bound work as
a Kaggle CPU kernel rather than locally (items.parquet, the full 13M+ row/
4.1GB catalog, is only practical to read in the Kaggle mount; items_human.parquet
locally only covers matches.parquet's own item scope, which happens to be
enough here too, but this keeps the kernel reusable if the sampling scope
ever needs to grow beyond matches.parquet's items).
"""
import json
import os

import numpy as np
import pandas as pd
import polars as pl

SEED = 42
N_NEGATIVE_PER_CATEGORY = 15
N_POSITIVE_PER_CATEGORY = 5
MAX_ITEM_TEXT_CHARS = 1000  # len(name) + len(attributes), raw strings, per item

ITEMS_PATH = "/kaggle/input/datasets/maxamsecret/ozon-ecup-2026-matching/items.parquet"
MATCHES_PATH = "/kaggle/input/datasets/maxamsecret/ozon-ecup-2026-matching/matches.parquet"
OUTPUT_PATH = "/kaggle/working/negative_pattern_samples.json"


def main():
    print("[DATA] loading matches.parquet ...")
    matches_pl = (
        pl.read_parquet(MATCHES_PATH)
        .filter(pl.col("id1").is_not_null() & pl.col("id2").is_not_null())
        .with_columns([
            pl.col("target").fill_null(0).cast(pl.Float32),
            (pl.col("target") >= 0.5).cast(pl.Int8).alias("label"),
        ])
        .select(["id1", "id2", "label"])
    )
    matches_pd = matches_pl.to_pandas()
    print(f"[DATA] matches: {len(matches_pd):,} pairs "
          f"(positive rate: {matches_pd['label'].mean():.4f})")

    needed_ids = pd.unique(pd.concat([matches_pd["id1"], matches_pd["id2"]]))
    needed_ids_lazy = pl.DataFrame({"id": pd.Series(needed_ids, dtype="int64")}).lazy()
    print(f"[DATA] unique item ids needed: {len(needed_ids):,}")

    print("[DATA] loading items.parquet (semi-join, streaming) for name/category/attributes ...")
    items = (
        pl.scan_parquet(ITEMS_PATH)
        .join(needed_ids_lazy, on="id", how="semi")
        .select(["id", "name", "category", "attributes"])
        .with_columns([
            pl.col("name").fill_null(""),
            pl.col("category").fill_null("UNKNOWN"),
            pl.col("attributes").fill_null(""),
        ])
        .collect(engine="streaming")
    )
    items_pd = items.to_pandas()
    items_pd["text_len"] = items_pd["name"].str.len() + items_pd["attributes"].str.len()
    print(f"[DATA] items loaded: {len(items_pd):,}")

    id_to_row = {int(v): i for i, v in enumerate(items_pd["id"].to_numpy())}
    names = items_pd["name"].to_numpy()
    categories = items_pd["category"].to_numpy()
    attrs = items_pd["attributes"].to_numpy()
    text_lens = items_pd["text_len"].to_numpy()

    row1 = matches_pd["id1"].map(id_to_row)
    row2 = matches_pd["id2"].map(id_to_row)
    missing = row1.isna() | row2.isna()
    if missing.any():
        print(f"[DATA] dropping {int(missing.sum())} pairs missing an item id in items.parquet")
    matches_pd = matches_pd[~missing].copy()
    row1 = row1[~missing].to_numpy().astype(np.int64)
    row2 = row2[~missing].to_numpy().astype(np.int64)

    matches_pd["category"] = categories[row1]  # both items share category, per CLAUDE.md (verified 100%)
    matches_pd["len1"] = text_lens[row1]
    matches_pd["len2"] = text_lens[row2]
    matches_pd["row1"] = row1
    matches_pd["row2"] = row2

    adequate = matches_pd[(matches_pd["len1"] <= MAX_ITEM_TEXT_CHARS) & (matches_pd["len2"] <= MAX_ITEM_TEXT_CHARS)]
    print(f"[FILTER] pairs with both items <= {MAX_ITEM_TEXT_CHARS} chars: "
          f"{len(adequate):,} / {len(matches_pd):,} ({100 * len(adequate) / len(matches_pd):.1f}%)")

    rng = np.random.RandomState(SEED)
    all_categories = sorted(adequate["category"].unique().tolist())
    print(f"[DATA] categories found: {len(all_categories)}")

    records = []
    shortfalls = []
    for cat in all_categories:
        cat_df = adequate[adequate["category"] == cat]
        neg = cat_df[cat_df["label"] == 0]
        pos = cat_df[cat_df["label"] == 1]

        n_neg = min(N_NEGATIVE_PER_CATEGORY, len(neg))
        n_pos = min(N_POSITIVE_PER_CATEGORY, len(pos))
        if n_neg < N_NEGATIVE_PER_CATEGORY or n_pos < N_POSITIVE_PER_CATEGORY:
            shortfalls.append((cat, n_neg, n_pos, len(neg), len(pos)))

        neg_sample = neg.sample(n=n_neg, random_state=rng) if n_neg > 0 else neg.iloc[:0]
        pos_sample = pos.sample(n=n_pos, random_state=rng) if n_pos > 0 else pos.iloc[:0]

        for _, r in pd.concat([neg_sample, pos_sample]).iterrows():
            r1, r2 = int(r["row1"]), int(r["row2"])
            records.append({
                "category": cat,
                "label": int(r["label"]),
                "id1": int(r["id1"]),
                "name1": names[r1],
                "attributes1": attrs[r1],
                "id2": int(r["id2"]),
                "name2": names[r2],
                "attributes2": attrs[r2],
            })

    print(f"[SAMPLE] total records: {len(records)} (expected {len(all_categories)} x "
          f"{N_NEGATIVE_PER_CATEGORY + N_POSITIVE_PER_CATEGORY} = "
          f"{len(all_categories) * (N_NEGATIVE_PER_CATEGORY + N_POSITIVE_PER_CATEGORY)})")
    if shortfalls:
        print("[SAMPLE] WARNING -- categories short of the requested quota after the length filter:")
        for cat, n_neg, n_pos, avail_neg, avail_pos in shortfalls:
            print(f"    {cat!r}: got neg={n_neg}/{N_NEGATIVE_PER_CATEGORY} (available={avail_neg}), "
                  f"pos={n_pos}/{N_POSITIVE_PER_CATEGORY} (available={avail_pos})")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"[DONE] wrote {len(records)} records to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
