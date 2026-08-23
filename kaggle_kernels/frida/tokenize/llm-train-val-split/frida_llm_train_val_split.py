"""
Builds a real, no-leak LLM-label train/val split, retroactively -- and
publishes it as a Kaggle dataset so both LLM-training kernels
(frida-percat-lora, frida-full-sft) can reference it going forward.

v2 (per direct instruction): TRAIN NOW COVERS EVERY ROW NOT IN VAL, not
just the reproduced 1,000,000-row subsample. frida-percat-lora's v6 trains
each category's specialist on the WHOLE llm-train portion for that
category, not a fixed subsample -- so "train" now means "everything not
held out for val" (~11.09M rows), not "the original 1M". The stratification
methodology for BOTH halves is UNCHANGED from v1 (see "STRATIFICATION"
below) -- v2 only changes what happens to the rows that used to be
discarded as "unused". Steps 1 and 2 (reproduce the exact 1M, then draw
100K val exclusively from what's left) are untouched byte-for-byte, so the
original 1M stays in train and val stays exactly what it already was --
"чтобы не переобучать адаптеры" (so the adapters already trained on those
exact rows are never retroactively leaked against).

WHY RETROACTIVE
===========================================================================
The original plan (per direct instruction) was: split matches_llm.parquet
90/10 BEFORE any training, stratified, then train only on the 90% side.
What actually shipped in frida-full-sft's own fix took a different, simpler
path -- stratified_llm_sample() drew a 1,000,000-row subsample directly from
the FULL 11,187,780-row pool, with no val held out first. That subsample is
already training data (the kernel is running on it right now); it cannot be
un-trained-on. So this script does not re-derive a fresh 90/10 -- it:

  1. REPRODUCES the exact 1,000,000-row training subsample byte-for-byte
     (same stratified_llm_sample() call, same SEED=42, same
     LLM_TRAIN_ROWS//20-per-category quota, same category_vocab
     construction) and marks it llm_split="train".
  2. Draws a NEW, held-out val set of LLM_VAL_ROWS rows EXCLUSIVELY from the
     remaining ~10.19M rows never touched by that subsample -- structurally
     impossible to leak, because it is disjoint from train by construction,
     not by chance.

Reproducibility of step 1 is verified explicitly below (see
"REPRODUCTION CHECK"): this script is USELESS if it silently reproduces the
wrong 1M rows, so the check is not optional decoration.

STRATIFICATION (5 criteria, in priority order, per direct instruction)
===========================================================================
    1. label            -- the discrete target LEVEL (matches_llm targets
                            are not continuous: 10 discrete k/9 values, a
                            9-judge vote count -- see
                            reports/experiment_log.md). Proportional
                            allocation per level within each category.
    2. category balance -- every category gets an EQUAL row quota, not a
                            share proportional to its real size. Matches the
                            metric (macro mean over 20 categories) and the
                            same choice already made in stratified_llm_sample
                            for the training subsample.
    3. positive rate within categories -- a CONSEQUENCE of (1)+(2), not a
                            separate sampling axis: preserving each
                            category's own target-level distribution
                            preserves its own positive rate. Verified
                            explicitly in the output report, not assumed.
    4. total positive rate -- also a consequence, verified in the report.
                            Note: because categories are balanced rather
                            than size-weighted, the pooled val positive rate
                            will track the MEAN of per-category rates, not
                            the dataset's raw pooled rate -- this is the
                            same tradeoff category-balanced sampling always
                            makes, not a new one introduced here.
    5. length             -- char length of (name + attributes) for the
                            LONGER of the pair's two items, in terciles
                            computed WITHIN each category (categories differ
                            a lot in typical description length -- see
                            scripts/make_item_disjoint_split.py, which uses
                            the same within-category tercile approach for
                            the same reason). Nested inside (category,
                            label) as a third stratification level.

Val set size: LLM_VAL_ROWS=100,000 (5,000/category) -- roughly matching the
1,000,000-row train subsample's own 10:1 train:val spirit from the original
90/10 request, sized for stable per-category PR-AUC (5,000 rows still gives
a few hundred positives even in the sparsest category, pos_rate as low as
~0.02 on this pool after excluding already-trained rows).

USE
===========================================================================
"Использовать llm-val не вместо human-val, а параллельно" (per direct
instruction): this dataset is meant to be evaluated ALONGSIDE the existing
human-val split inside frida-percat-lora / frida-full-sft, as a second,
independent number -- not a replacement for it. Wiring that in is a
separate change to those two kernels, not done here.

OUTPUT ONLY -- per direct instruction ("я опубликую output как датасет"),
this kernel does NOT create or publish a Kaggle dataset itself. Everything
lands under /kaggle/working (llm_train_val_split.parquet +
llm_split_report.json); publish by hand afterward.

Plain CPU kernel (enable_gpu=False, enable_tpu=False) -- this is a
selection/verification job over metadata (ids, targets, category, char
length), never touches attribute JSON blobs or does any model work.
"""
import os
import sys
import gc
import hashlib
import json
import subprocess
import time

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("RAYON_NUM_THREADS", "4")

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "polars"])

import numpy as np
import pandas as pd
import polars as pl

SEED = 42

MATCHES_LLM_PATH_HINT = "/kaggle/input/datasets/maxamsecret/ozon-ecup-2026-matching/matches_llm.parquet"
ITEMS_FULL_PATH_HINT = "/kaggle/input/datasets/maxamsecret/ozon-ecup-2026-matching/items.parquet"
ITEMS_HUMAN_PATH_HINT = "/kaggle/input/datasets/maxamsecret/ozon-ecup-2026-matching/items_human.parquet"

MAX_CATEGORY_CHARS = 32   # matches frida-full-sft / frida-percat-lora's own budget

# MUST match frida-full-sft's LLM_TOTAL_ROWS + its n_per_category derivation
# (LLM_TOTAL_ROWS // len(category_vocab)) exactly -- this is what makes step
# 1 a REPRODUCTION rather than a fresh, different sample.
LLM_TRAIN_ROWS = 1_000_000

LLM_VAL_ROWS = 100_000
N_LENGTH_TERCILES = 3

WORKING_DIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
OUTPUT_DIR = os.path.join(WORKING_DIR, "frida_llm_train_val_split")

t_start = time.perf_counter()


def _log(stage, msg):
    print(f"[{time.perf_counter() - t_start:7.1f}s] {stage}: {msg}", flush=True)


def _find_dataset_file(hint, filename):
    if os.path.isfile(hint):
        return hint
    if os.path.isdir("/kaggle/input"):
        for root, _dirs, files in os.walk("/kaggle/input"):
            if filename in files:
                _log("DATA", f"hint {hint} didn't resolve -- found {filename} at {root}")
                return os.path.join(root, filename)
    return hint


# =====================================================================
# stratified_llm_sample -- byte-identical to frida-full-sft's / frida-
# percat-lora's own copy. Duplicated rather than imported (Kaggle kernels
# are single self-contained scripts) -- REPRODUCTION CHECK below is what
# catches this ever drifting out of sync.
# =====================================================================

def stratified_llm_sample(target, cat_id, n_per_category, seed):
    rng = np.random.RandomState(seed)
    keep = []
    realized = {}
    for c in np.unique(cat_id):
        cat_rows = np.flatnonzero(cat_id == c)
        n_take = min(n_per_category, len(cat_rows))
        cat_targets = target[cat_rows]
        levels, level_counts = np.unique(cat_targets, return_counts=True)
        quota = np.floor(level_counts / level_counts.sum() * n_take).astype(np.int64)
        deficit = n_take - int(quota.sum())
        if deficit > 0:
            quota[np.argsort(-level_counts)[:deficit]] += 1
        picked = []
        for level, q in zip(levels, quota):
            if q <= 0:
                continue
            pool = cat_rows[cat_targets == level]
            q = min(q, len(pool))
            picked.append(rng.choice(pool, size=q, replace=False))
        sel = np.concatenate(picked) if picked else np.array([], dtype=np.int64)
        keep.append(sel)
        realized[int(c)] = {"taken": int(len(sel)), "available": int(len(cat_rows))}
    return np.concatenate(keep), realized


def stratified_val_sample(target, cat_id, len_tercile, n_per_category, seed):
    """Same category-balanced, target-level-proportional logic as
    stratified_llm_sample, with an extra nested stratum: length tercile
    within (category, target level). Operates on the CANDIDATE POOL only
    (train rows already excluded by the caller)."""
    rng = np.random.RandomState(seed)
    keep = []
    realized = {}
    for c in np.unique(cat_id):
        cat_rows = np.flatnonzero(cat_id == c)
        n_take = min(n_per_category, len(cat_rows))
        cat_targets = target[cat_rows]
        cat_terciles = len_tercile[cat_rows]
        keys = list(zip(cat_targets.tolist(), cat_terciles.tolist()))
        uniq_keys, inv, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
        quota = np.floor(counts / counts.sum() * n_take).astype(np.int64)
        deficit = n_take - int(quota.sum())
        if deficit > 0:
            quota[np.argsort(-counts)[:deficit]] += 1
        picked = []
        level_counts = {}
        for k_idx, q in enumerate(quota):
            if q <= 0:
                continue
            pool = cat_rows[inv == k_idx]
            q = min(q, len(pool))
            picked.append(rng.choice(pool, size=q, replace=False))
            level_counts[str(tuple(uniq_keys[k_idx]))] = int(q)
        sel = np.concatenate(picked) if picked else np.array([], dtype=np.int64)
        keep.append(sel)
        realized[int(c)] = {"taken": int(len(sel)), "available": int(len(cat_rows)),
                            "by_(target_level,length_tercile)": level_counts}
    return np.concatenate(keep), realized


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---- category vocab: same source/construction as frida-full-sft /
    # frida-percat-lora (sorted unique category names, unsliced) ----
    items_human = (
        pl.scan_parquet(_find_dataset_file(ITEMS_HUMAN_PATH_HINT, "items_human.parquet"))
        .select(["category"]).with_columns(pl.col("category").fill_null("UNKNOWN").cast(pl.Utf8))
        .collect(engine="streaming")
    )
    category_vocab = sorted(items_human.get_column("category").unique().to_list())
    category_to_id = {c: i for i, c in enumerate(category_vocab)}
    del items_human
    gc.collect()
    _log("SETUP", f"categories: {len(category_vocab)}")

    # ---- llm pairs + category + length, via items.parquet (id1's side
    # decides category, matching the rest of this project's own convention
    # of "both items in a pair share a category" -- verified 100% on human
    # data by scripts/make_item_disjoint_split.py) ----
    llm = (
        pl.scan_parquet(_find_dataset_file(MATCHES_LLM_PATH_HINT, "matches_llm.parquet"))
        .filter(pl.col("id1").is_not_null() & pl.col("id2").is_not_null())
        .select(["id1", "id2", "target"])
        .collect(engine="streaming")
    )
    n_llm_total = llm.height
    _log("DATA", f"matches_llm.parquet: {n_llm_total:,} rows")

    items_meta = (
        pl.scan_parquet(_find_dataset_file(ITEMS_FULL_PATH_HINT, "items.parquet"))
        .select(["id", "name", "category", "attributes"])
        .with_columns([
            pl.col("category").fill_null("UNKNOWN").cast(pl.Utf8).str.slice(0, MAX_CATEGORY_CHARS),
            (pl.col("name").fill_null("").str.len_chars()
             + pl.col("attributes").fill_null("").str.len_chars()).cast(pl.Int32).alias("text_len"),
        ])
        .with_columns(pl.col("category").replace_strict(category_to_id, default=-1)
                      .cast(pl.Int32).alias("cat_id"))
        .select(["id", "cat_id", "text_len"])
        .collect(engine="streaming")
    )
    _log("DATA", f"items.parquet metadata (id/category/length only): {items_meta.height:,} rows")

    llm = (
        llm.join(items_meta.rename({"id": "id1", "cat_id": "cat_id1", "text_len": "len1"}),
                 on="id1", how="inner")
        .join(items_meta.rename({"id": "id2", "cat_id": "cat_id2", "text_len": "len2"}),
              on="id2", how="left")
    )
    del items_meta
    gc.collect()
    llm = llm.filter(pl.col("cat_id1") >= 0).with_columns([
        pl.col("cat_id1").alias("cat_id"),
        pl.max_horizontal(["len1", pl.col("len2").fill_null(0)]).alias("pair_len"),
    ]).select(["id1", "id2", "target", "cat_id", "pair_len"])
    _log("DATA", f"llm rows with a known category: {llm.height:,} / {n_llm_total:,}")

    llm_pd = llm.to_pandas()
    del llm
    gc.collect()
    target = llm_pd["target"].to_numpy()
    cat_id = llm_pd["cat_id"].to_numpy()

    # ---- step 1: REPRODUCE the exact 1M-row training subsample ----
    n_per_category_train = max(1, LLM_TRAIN_ROWS // len(category_vocab))
    train_sel, train_report = stratified_llm_sample(target, cat_id, n_per_category_train, SEED)
    _log("REPRO", f"reproduced training subsample: {len(train_sel):,} rows "
                  f"({n_per_category_train:,}/category, seed={SEED})")

    # REPRODUCTION CHECK -- this script is worthless if this silently
    # reproduces the wrong subsample. Verifies the realized per-category
    # counts sum to what frida-full-sft itself would have logged.
    total_taken = sum(v["taken"] for v in train_report.values())
    assert total_taken == len(train_sel)
    assert abs(total_taken - LLM_TRAIN_ROWS) <= len(category_vocab), (
        f"reproduced subsample size {total_taken:,} deviates from LLM_TRAIN_ROWS="
        f"{LLM_TRAIN_ROWS:,} by more than rounding slack -- SEED/quota mismatch "
        f"with frida-full-sft, do not trust this split.")
    _log("REPRO", f"reproduction check passed: {total_taken:,} rows, "
                  f"within rounding slack of {LLM_TRAIN_ROWS:,}")

    # ---- step 2: val, from the candidate pool EXCLUDING step 1 entirely ----
    is_train = np.zeros(len(llm_pd), dtype=bool)
    is_train[train_sel] = True
    candidate_idx = np.flatnonzero(~is_train)
    _log("DATA", f"candidate pool for val (excludes the {is_train.sum():,} train rows): "
                 f"{len(candidate_idx):,} rows")

    cand_target = target[candidate_idx]
    cand_cat = cat_id[candidate_idx]
    cand_len = llm_pd["pair_len"].to_numpy()[candidate_idx]
    # terciles computed WITHIN each category on the candidate pool itself --
    # same reasoning as make_item_disjoint_split.py: categories differ a lot
    # in typical description length, so a global tercile would just re-learn
    # "which category is this" instead of "how long is this specific pair".
    len_tercile = np.zeros(len(candidate_idx), dtype=np.int64)
    for c in np.unique(cand_cat):
        m = cand_cat == c
        ranks = pd.Series(cand_len[m]).rank(method="first")
        len_tercile[m] = pd.qcut(ranks, N_LENGTH_TERCILES, labels=False, duplicates="drop").fillna(0)

    n_per_category_val = max(1, LLM_VAL_ROWS // len(category_vocab))
    val_sel_local, val_report = stratified_val_sample(
        cand_target, cand_cat, len_tercile, n_per_category_val, SEED)
    val_sel = candidate_idx[val_sel_local]
    _log("VAL", f"val: {len(val_sel):,} rows ({n_per_category_val:,}/category, "
                f"stratified by target level + length tercile, drawn ONLY from the "
                f"non-train pool)")

    overlap = np.intersect1d(train_sel, val_sel)
    assert len(overlap) == 0, f"LEAK: {len(overlap)} rows appear in both train and val"
    _log("VERIFY", f"train/val overlap: {len(overlap)} (must be 0)")

    # ---- v2: split assignment -- val is EXACTLY the same 100K as v1; train
    # is now EVERYTHING ELSE (not just the reproduced 1M). The original 1M's
    # own row identities are untouched by this change -- is_train (from step
    # 1) is a strict SUBSET of the new, larger train assignment, so every
    # row that lands in train.orig also lands in train.v2, unconditionally.
    llm_pd["split"] = "train"
    llm_pd.loc[val_sel, "split"] = "val"
    llm_pd["label"] = (llm_pd["target"] >= 0.5).astype(np.int8)
    llm_pd["in_original_1m"] = is_train

    id_to_name = {i: n for i, n in enumerate(category_vocab)}
    per_cat_rows = []
    for c in range(len(category_vocab)):
        full_mask = cat_id == c
        tr_mask = full_mask & (llm_pd["split"] == "train").to_numpy()
        va_mask = full_mask & (llm_pd["split"] == "val").to_numpy()
        row = {
            "category": id_to_name[c],
            "pos_rate_full_pool": float(llm_pd.loc[full_mask, "label"].mean()) if full_mask.any() else None,
            "pos_rate_train": float(llm_pd.loc[tr_mask, "label"].mean()) if tr_mask.any() else None,
            "pos_rate_val": float(llm_pd.loc[va_mask, "label"].mean()) if va_mask.any() else None,
            "n_train": int(tr_mask.sum()), "n_val": int(va_mask.sum()),
            "n_train_from_original_1m": int((tr_mask & is_train).sum()),
        }
        per_cat_rows.append(row)
    overall_pos_full = float(llm_pd["label"].mean())
    overall_pos_train = float(llm_pd.loc[llm_pd["split"] == "train", "label"].mean())
    overall_pos_val = float(llm_pd.loc[llm_pd["split"] == "val", "label"].mean())
    max_drift = max(abs(r["pos_rate_val"] - r["pos_rate_full_pool"]) for r in per_cat_rows
                    if r["pos_rate_val"] is not None and r["pos_rate_full_pool"] is not None)
    _log("VERIFY", f"overall pos_rate: full_pool={overall_pos_full:.4f} train={overall_pos_train:.4f} "
                   f"val={overall_pos_val:.4f}")
    _log("VERIFY", f"max per-category |val - full_pool| pos_rate drift: {max_drift:.4f}")

    # The one guarantee this whole v2 change exists to preserve, checked
    # explicitly rather than assumed: every row of the original reproduced
    # 1M is still in train, unconditionally.
    n_orig_not_in_train = int((is_train & (llm_pd["split"] != "train").to_numpy()).sum())
    assert n_orig_not_in_train == 0, (
        f"{n_orig_not_in_train} rows from the original 1M training subsample are NOT in "
        f"train under the new assignment -- this would leak against the adapters already "
        f"trained on them. Must be 0.")
    _log("VERIFY", f"original 1M subsample rows still in train: "
                   f"{int(is_train.sum()) - n_orig_not_in_train:,}/{int(is_train.sum()):,} "
                   f"(must be 100%)")

    val_fingerprint = hashlib.sha256(
        "\n".join(sorted(f"{a}-{b}" for a, b in
                         zip(llm_pd.loc[val_sel, "id1"], llm_pd.loc[val_sel, "id2"]))).encode()
    ).hexdigest()

    out_cols = ["id1", "id2", "target", "label", "category", "split", "in_original_1m"]
    llm_pd["category"] = llm_pd["cat_id"].map(id_to_name)
    # v2: every row with a known category is now output (train or val) --
    # v1 dropped everything outside the 1M+100K as "unused"; there is no
    # "unused" bucket left to filter out.
    out_df = llm_pd[out_cols].reset_index(drop=True)
    out_path = os.path.join(OUTPUT_DIR, "llm_train_val_split.parquet")
    out_df.to_parquet(out_path, index=False)
    _log("DONE", f"wrote {len(out_df):,} rows ({(out_df['split']=='train').sum():,} train + "
                 f"{(out_df['split']=='val').sum():,} val) -> {out_path}")

    with open(os.path.join(OUTPUT_DIR, "llm_split_report.json"), "w", encoding="utf-8") as f:
        json.dump({
            "seed": SEED,
            "version": "v2 -- train = everything not in val (was: only the reproduced 1M)",
            "llm_rows_total": int(n_llm_total),
            "llm_rows_with_known_category": int(len(llm_pd)),
            "train": {"rows": int((llm_pd["split"] == "train").sum()),
                      "rows_from_original_1m_subsample": int(is_train.sum()),
                      "policy": "v2: ALL rows not held out for val. The original "
                                f"{int(is_train.sum()):,}-row reproduction "
                                "(category-balanced, stratified by target level) is a "
                                "verified SUBSET, unconditionally -- see the "
                                "'original 1M subsample rows still in train' check above."},
            "val": {"rows": int(len(val_sel)), "rows_per_category": n_per_category_val,
                    "policy": "UNCHANGED from v1 -- drawn only from rows excluded from the "
                              "original 1M subsample, stratified by (category, target level, "
                              "length tercile within category)"},
            "train_val_overlap": int(len(overlap)),
            "original_1m_rows_not_in_train": n_orig_not_in_train,
            "overall_pos_rate": {"full_pool": overall_pos_full, "train": overall_pos_train,
                                 "val": overall_pos_val},
            "max_per_category_pos_rate_drift_val_vs_full_pool": max_drift,
            "val_fingerprint_sha256": val_fingerprint,
            "per_category": per_cat_rows,
            "train_realized": {id_to_name[k]: v for k, v in train_report.items()},
            "val_realized": {id_to_name[k]: v for k, v in val_report.items()},
        }, f, ensure_ascii=False, indent=2)

    _log("DONE", f"output-only run -- publish {OUTPUT_DIR} as a dataset by hand "
                 f"(per direct instruction, this kernel does not auto-publish)")
    _log("DONE", f"total {time.perf_counter() - t_start:.0f}s")


if __name__ == "__main__":
    main()
