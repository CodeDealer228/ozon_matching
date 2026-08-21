"""
Builds an ITEM-DISJOINT, stratified 70/30 train/val split over
matches.parquet.

SCOPE -- this split is for FROM-SCRATCH training ONLY (the full-SFT FRIDA
kernel). It must NOT be used for the 20 per-category LoRA kernel: that one
builds on an already-trained adapter which was itself trained and validated
against the OLD canonical pair-level split
(data/splits/matches_split_seed42.parquet, 80/20, seed 42). Re-splitting
under it would move items the existing adapter already trained on into the
new val half -- a leak that would silently inflate every per-category
number. Anything continuing an existing checkpoint keeps that checkpoint's
original split; only a fresh initialization is free to adopt this one.

WHY item-disjoint, and why it's cheap here
-------------------------------------------
The canonical seed-42 split (scripts/make_canonical_split.py) splits at the
PAIR level, so the same item can appear in both train and val via two
different pairs. That leaks: the model can memorize an item's
representation on one pair and be graded on it in another. This split
guarantees no item id is shared.

Naively that's a graph-partitioning problem (pairs are edges; every
connected component must land wholly in train or wholly in val, or an item
would straddle the boundary). Measured on the real data it's nearly free:

    365,654 pairs over 711,304 items  ->  345,654 connected components
    95.7% of components are a single lone pair
    largest component: 30 items / 31 pairs  = 0.008% of all pairs

There is no giant component, so component-level assignment barely
constrains the stratification at all.

Stratification, in the priority order requested
------------------------------------------------
  1. category            (the metric is a plain mean over 20 categories)
  2. positive rate within each category
  3. total description length

Implemented as one composite stratum key -- (category, label, length
tercile) -- rather than three sequential passes. Splitting 70/30 inside
every such bucket preserves all three marginals simultaneously, which
sequential stratification does not guarantee. Buckets stay large enough to
be stable: ~18K pairs per category / (2 labels x 3 terciles) = ~3K pairs
per bucket.

Note on which weighting actually matters (measured, not assumed): category
SIZES are nearly uniform (17,286-23,357 pairs, only a 1.4x spread), so any
"weight inversely proportional to category size" scheme is close to a
no-op. What really varies is the POSITIVE RATE: 0.073 to 0.562, a 7.7x
spread. Per-category pos_weight is the lever with real dynamic range.

Output: data/splits/matches_item_disjoint_70_30_seed42.parquet
        (id1, id2, target, label, category, split, component)
plus a JSON sidecar with the verification report and a val fingerprint
(sha256 over sorted "id1-id2"), so any downstream run can prove it used
this exact split -- same convention as the canonical split.
"""
import hashlib
import json
import os
import time

import numpy as np
import pandas as pd

SEED = 42
VAL_FRACTION = 0.30
N_LENGTH_BUCKETS = 3

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MATCHES_PATH = os.path.join(ROOT, "data", "matches.parquet")
ITEMS_PATH = os.path.join(ROOT, "data", "items_human.parquet")
OUT_DIR = os.path.join(ROOT, "data", "splits")
OUT_PARQUET = os.path.join(OUT_DIR, "matches_item_disjoint_70_30_seed42.parquet")
OUT_JSON = os.path.join(OUT_DIR, "matches_item_disjoint_70_30_seed42.json")

t0 = time.time()


def _log(stage, msg):
    print(f"[{time.time() - t0:6.1f}s] {stage}: {msg}", flush=True)


def connected_components(ids_a, ids_b):
    """Union-find over the pair graph. Returns (component_id_per_pair,
    component_id_per_unique_item, unique_items) with component ids
    densified to 0..n_components-1."""
    uniq = pd.unique(np.concatenate([ids_a, ids_b]))
    index_of = {v: i for i, v in enumerate(uniq)}
    parent = np.arange(len(uniq), dtype=np.int64)

    def find(x):
        # path halving -- keeps this near-linear without recursion depth risk
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    a = np.fromiter((index_of[v] for v in ids_a), dtype=np.int64, count=len(ids_a))
    b = np.fromiter((index_of[v] for v in ids_b), dtype=np.int64, count=len(ids_b))
    for x, y in zip(a, b):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    item_root = np.array([find(i) for i in range(len(uniq))], dtype=np.int64)
    # densify root ids -> 0..k-1
    _, item_comp = np.unique(item_root, return_inverse=True)
    pair_comp = item_comp[a]
    return pair_comp, item_comp, uniq


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # ---------------- load + canonical label convention ----------------
    m = pd.read_parquet(MATCHES_PATH)
    n_raw = len(m)
    m = m[m["id1"].notna() & m["id2"].notna()].copy()
    m["label"] = (m["target"] >= 0.5).astype(np.int8)
    _log("DATA", f"matches: {n_raw:,} rows -> {len(m):,} after null-id filter "
                 f"(pos_rate={m['label'].mean():.4f})")

    items = pd.read_parquet(ITEMS_PATH, columns=["id", "category", "name", "attributes"])
    items["name"] = items["name"].fillna("")
    items["attributes"] = items["attributes"].fillna("")
    items["category"] = items["category"].fillna("UNKNOWN")
    # char length as the length proxy -- no tokenizer needed, and the two
    # correlate tightly enough for stratification (a tokenizer here would
    # tie this split to one specific model's vocabulary, which it should
    # not be: both kernels share this split).
    items["text_len"] = (items["name"].str.len() + items["attributes"].str.len()).astype(np.int32)
    cat_of = dict(zip(items["id"].to_numpy(), items["category"].to_numpy()))
    len_of = dict(zip(items["id"].to_numpy(), items["text_len"].to_numpy()))
    _log("DATA", f"items: {len(items):,}  categories: {items['category'].nunique()}")

    id1 = m["id1"].to_numpy()
    id2 = m["id2"].to_numpy()
    cat1 = pd.Series(id1).map(cat_of).to_numpy()
    cat2 = pd.Series(id2).map(cat_of).to_numpy()

    # Verify the "both items of a pair share a category" assumption that
    # the FRIDA inference code already relies on (it locates and excises a
    # single "Category:" span per pair). Cheap to check here, and CLAUDE.md
    # flags it as an unverified train-time assumption.
    same_cat = (cat1 == cat2)
    n_mismatch = int((~same_cat).sum())
    _log("CHECK", f"pairs whose two items share a category: {int(same_cat.sum()):,}/{len(m):,} "
                  f"({100 * same_cat.mean():.4f}%) -- mismatches: {n_mismatch:,}")

    m["category"] = cat1  # id1's category is the pair's category (see check above)
    m["pair_len"] = (pd.Series(id1).map(len_of).to_numpy()
                     + pd.Series(id2).map(len_of).to_numpy())

    # ---------------- connected components ----------------
    pair_comp, item_comp, uniq_items = connected_components(id1, id2)
    m["component"] = pair_comp
    n_comp = int(item_comp.max()) + 1
    comp_pair_counts = np.bincount(pair_comp, minlength=n_comp)
    _log("GRAPH", f"unique items={len(uniq_items):,}  components={n_comp:,}  "
                  f"largest component={comp_pair_counts.max()} pairs "
                  f"({100 * comp_pair_counts.max() / len(m):.4f}% of data)")

    # ---------------- per-component stratum key ----------------
    # A component is the atomic unit of assignment, so its stratum has to be
    # a single value. 95.7% of components are one pair, so for the vast
    # majority this is exact; multi-pair components take the mode of the
    # category and the rounded mean of the label.
    comp_df = (
        m.groupby("component")
        .agg(n_pairs=("label", "size"),
             pos_mean=("label", "mean"),
             category=("category", lambda s: s.mode().iat[0]),
             pair_len=("pair_len", "mean"))
        .reset_index()
    )
    comp_df["label_bucket"] = (comp_df["pos_mean"] >= 0.5).astype(np.int8)

    # Length terciles computed WITHIN each category, so the length bucket
    # means the same thing (relative to that category's own distribution)
    # everywhere -- categories differ a lot in typical description length.
    comp_df["len_bucket"] = (
        comp_df.groupby("category")["pair_len"]
        .transform(lambda s: pd.qcut(s.rank(method="first"), N_LENGTH_BUCKETS,
                                      labels=False, duplicates="drop"))
        .fillna(0).astype(np.int8)
    )
    comp_df["stratum"] = (comp_df["category"].astype(str) + "|"
                          + comp_df["label_bucket"].astype(str) + "|"
                          + comp_df["len_bucket"].astype(str))
    _log("STRATA", f"components={len(comp_df):,}  strata={comp_df['stratum'].nunique()}  "
                   f"min stratum size={comp_df.groupby('stratum').size().min()} components")

    # ---------------- assign 70/30 inside each stratum ----------------
    # Assignment is by PAIR count, not component count: components carry
    # different numbers of pairs, and 70/30 is meant to hold over pairs.
    rng = np.random.RandomState(SEED)
    split_of_comp = {}
    for stratum, grp in comp_df.groupby("stratum", sort=True):
        order = rng.permutation(len(grp))
        g = grp.iloc[order]
        target_val = VAL_FRACTION * g["n_pairs"].sum()
        cum = g["n_pairs"].cumsum()
        # everything up to the point where cumulative pairs first reaches the
        # val target goes to val; the rest to train
        is_val = cum <= target_val
        if not is_val.any():          # tiny stratum: give its first component to val
            is_val.iloc[0] = True
        for comp_id, v in zip(g["component"].to_numpy(), is_val.to_numpy()):
            split_of_comp[int(comp_id)] = "val" if v else "train"

    m["split"] = m["component"].map(split_of_comp)
    assert m["split"].notna().all(), "some component was never assigned"

    # ---------------- verification ----------------
    train, val = m[m["split"] == "train"], m[m["split"] == "val"]
    train_items = set(np.concatenate([train["id1"].to_numpy(), train["id2"].to_numpy()]).tolist())
    val_items = set(np.concatenate([val["id1"].to_numpy(), val["id2"].to_numpy()]).tolist())
    shared = train_items & val_items

    _log("SPLIT", f"train={len(train):,} ({100*len(train)/len(m):.2f}%)  "
                  f"val={len(val):,} ({100*len(val)/len(m):.2f}%)")
    _log("VERIFY", f"train items={len(train_items):,}  val items={len(val_items):,}  "
                   f"SHARED={len(shared):,}")
    assert not shared, f"item-disjointness violated: {len(shared)} shared ids"

    # per-category comparison of the two marginals that matter
    overall = m.groupby("category").agg(pairs=("label", "size"), pos_rate=("label", "mean"))
    tr = train.groupby("category").agg(pairs=("label", "size"), pos_rate=("label", "mean"))
    va = val.groupby("category").agg(pairs=("label", "size"), pos_rate=("label", "mean"))
    cmp = overall.join(tr, rsuffix="_tr").join(va, rsuffix="_va")
    cmp["val_share"] = cmp["pairs_va"] / cmp["pairs"]
    cmp["pos_rate_drift"] = (cmp["pos_rate_va"] - cmp["pos_rate"]).abs()

    print("\n  category                                 pairs   val%   pos_rate(all/tr/val)   drift")
    for cat, r in cmp.sort_values("pairs", ascending=False).iterrows():
        print(f"  {str(cat)[:38]:<38} {int(r['pairs']):>6,}  {100*r['val_share']:5.2f}%  "
              f"{r['pos_rate']:.3f}/{r['pos_rate_tr']:.3f}/{r['pos_rate_va']:.3f}   "
              f"{r['pos_rate_drift']:.4f}")

    _log("VERIFY", f"max per-category val-share deviation from {VAL_FRACTION:.0%}: "
                   f"{(cmp['val_share'] - VAL_FRACTION).abs().max():.4f}")
    _log("VERIFY", f"max per-category pos_rate drift: {cmp['pos_rate_drift'].max():.4f}")
    _log("VERIFY", f"mean pair_len  all={m['pair_len'].mean():.0f}  "
                   f"train={train['pair_len'].mean():.0f}  val={val['pair_len'].mean():.0f}")
    assert cmp["pos_rate_va"].notna().all(), "a category is missing from val"
    assert (val.groupby("category")["label"].nunique() == 2).all(), \
        "a category has only one class in val -- PR-AUC would be undefined there"

    # ---------------- persist ----------------
    out = m[["id1", "id2", "target", "label", "category", "component", "split"]].copy()
    out.to_parquet(OUT_PARQUET, index=False)

    val_fingerprint = hashlib.sha256(
        "\n".join(sorted(f"{a}-{b}" for a, b in zip(val["id1"], val["id2"]))).encode()
    ).hexdigest()

    report = {
        "seed": SEED,
        "val_fraction_target": VAL_FRACTION,
        "n_pairs": int(len(m)),
        "n_train": int(len(train)),
        "n_val": int(len(val)),
        "val_fraction_actual": float(len(val) / len(m)),
        "n_components": int(n_comp),
        "largest_component_pairs": int(comp_pair_counts.max()),
        "n_train_items": len(train_items),
        "n_val_items": len(val_items),
        "shared_items": len(shared),
        "pairs_with_category_mismatch": n_mismatch,
        "max_val_share_deviation": float((cmp["val_share"] - VAL_FRACTION).abs().max()),
        "max_pos_rate_drift": float(cmp["pos_rate_drift"].max()),
        "val_fingerprint_sha256": val_fingerprint,
        "per_category": {
            str(cat): {
                "pairs": int(r["pairs"]),
                "pos_rate_all": float(r["pos_rate"]),
                "pos_rate_train": float(r["pos_rate_tr"]),
                "pos_rate_val": float(r["pos_rate_va"]),
                "val_share": float(r["val_share"]),
                # neg/pos on the TRAIN half -- this is the number the
                # per-category pos_weight should be derived from
                "train_neg_per_pos": float((1 - r["pos_rate_tr"]) / max(r["pos_rate_tr"], 1e-9)),
            }
            for cat, r in cmp.iterrows()
        },
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    _log("DONE", f"-> {os.path.relpath(OUT_PARQUET, ROOT)}")
    _log("DONE", f"-> {os.path.relpath(OUT_JSON, ROOT)}  (val fingerprint {val_fingerprint[:16]}...)")


if __name__ == "__main__":
    main()
