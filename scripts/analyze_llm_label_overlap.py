"""
Does matches_llm.parquet (11.19M LLM soft-labeled pairs) overlap at all with
matches.parquet (365K human-labeled pairs)? If it does, the overlapping
pairs let us score LLM-label quality directly against human ground truth
(correlation, AUC/AP treating the soft score as a classifier, calibration,
confusion at 0.5) -- overall and per category.

Checks three levels of overlap, from strictest to loosest:
  1. Exact-order pairs: identical (id1, id2) in both files.
  2. Unordered pairs: same pair regardless of which id is id1/id2 -- the
     right notion of "same pair" for this comparison.
  3. Item ids: does the *catalog* overlap at all, even without any shared
     pair (informational -- doesn't by itself enable a label comparison).

Runs entirely locally: both matches.parquet and matches_llm.parquet are
already present in full under data/ (confirmed: matches_llm.parquet has all
11,187,780 rows locally, not just a subset -- see CLAUDE.md). Only needs
items_human.parquet for category lookup (covers every id matches.parquet
references, which is all that's needed since any overlap can only involve
matches.parquet's own ids).
"""
import os
import json

import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import (
    average_precision_score, roc_auc_score, precision_score, recall_score,
    f1_score, accuracy_score,
)
from scipy.stats import pearsonr, spearmanr

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MATCHES_PATH = os.path.join(REPO, "data", "matches.parquet")
MATCHES_LLM_PATH = os.path.join(REPO, "data", "matches_llm.parquet")
ITEMS_HUMAN_PATH = os.path.join(REPO, "data", "items_human.parquet")
REPORT_DIR = os.path.join(REPO, "reports")


def compute_metrics_at_row_level(y_true, y_prob, categories=None):
    """Same macro-averaging convention as the rest of this project (see
    CLAUDE.md 'Metric') -- per-category AP, then plain mean, skipping
    categories with only one class present."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    if categories is None or len(np.unique(y_true)) < 2:
        return None
    categories = np.asarray(categories)
    per_cat = []
    for cat in np.unique(categories):
        mask = categories == cat
        yt, yp = y_true[mask], y_prob[mask]
        if len(np.unique(yt)) < 2:
            continue
        per_cat.append(average_precision_score(yt, yp))
    return float(np.mean(per_cat)) if per_cat else None


def main():
    print("=" * 90)
    print("[DATA] Loading matches.parquet (human) ...")
    human = pl.read_parquet(MATCHES_PATH).select(["id1", "id2", "target"])
    human = human.filter(pl.col("id1").is_not_null() & pl.col("id2").is_not_null())
    human = human.with_columns((pl.col("target") >= 0.5).cast(pl.Int8).alias("human_label"))
    print(f"[DATA] human rows: {human.height:,}")

    print("[DATA] Loading matches_llm.parquet (LLM) ...")
    llm = pl.read_parquet(MATCHES_LLM_PATH).select(["id1", "id2", "target"])
    llm = llm.filter(pl.col("id1").is_not_null() & pl.col("id2").is_not_null())
    llm = llm.rename({"target": "llm_score"})
    print(f"[DATA] llm rows: {llm.height:,}")

    # -------------------------------------------------------------
    # Level 1: exact-order pair overlap
    # -------------------------------------------------------------
    exact_overlap = llm.join(human.select(["id1", "id2"]).unique(), on=["id1", "id2"], how="inner")
    print(f"\n[OVERLAP] exact-order (id1,id2) pairs in both: {exact_overlap.height:,}")

    # -------------------------------------------------------------
    # Level 2: unordered pair overlap (the one that actually matters for
    # comparing labels -- "same pair" regardless of id1/id2 assignment)
    # -------------------------------------------------------------
    def add_norm_key(df):
        return df.with_columns([
            pl.min_horizontal("id1", "id2").alias("_lo"),
            pl.max_horizontal("id1", "id2").alias("_hi"),
        ])

    human_norm = add_norm_key(human)
    llm_norm = add_norm_key(llm)

    unordered_overlap = llm_norm.join(
        human_norm.select(["_lo", "_hi", "human_label"]).unique(subset=["_lo", "_hi"]),
        on=["_lo", "_hi"], how="inner",
    )
    n_unordered = unordered_overlap.height
    print(f"[OVERLAP] unordered pairs (same 2 items, either order) in both: {n_unordered:,}")

    # -------------------------------------------------------------
    # Level 3: item-id catalog overlap (informational)
    # -------------------------------------------------------------
    human_ids = pl.concat([human.select(pl.col("id1").alias("id")), human.select(pl.col("id2").alias("id"))]).unique()
    llm_ids = pl.concat([llm.select(pl.col("id1").alias("id")), llm.select(pl.col("id2").alias("id"))]).unique()
    shared_ids = human_ids.join(llm_ids, on="id", how="inner")
    print(f"[OVERLAP] unique item ids -- human: {human_ids.height:,} | llm: {llm_ids.height:,} | "
          f"shared: {shared_ids.height:,}")

    summary = {
        "human_rows": human.height,
        "llm_rows": llm.height,
        "exact_order_pair_overlap": exact_overlap.height,
        "unordered_pair_overlap": n_unordered,
        "human_unique_item_ids": human_ids.height,
        "llm_unique_item_ids": llm_ids.height,
        "shared_item_ids": shared_ids.height,
    }
    print("\n" + "=" * 90)
    print("[SUMMARY]")
    print(json.dumps(summary, indent=2))
    print("=" * 90)

    os.makedirs(REPORT_DIR, exist_ok=True)
    with open(os.path.join(REPORT_DIR, "llm_label_overlap_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    if n_unordered == 0:
        print("\n[RESULT] Zero overlap at every level checked -- matches_llm.parquet and "
              "matches.parquet are genuinely disjoint pair/item pools (confirmed, not just "
              "assumed). No pairs exist where both a human label and an LLM score are "
              "available for the same items, so LLM-label quality cannot be scored directly "
              "against human ground truth from this data alone. Per the task instructions, "
              "skipping the conditional quality-metrics section.")
        return

    # -------------------------------------------------------------
    # Overlap > 0: score LLM-label quality against human ground truth
    # -------------------------------------------------------------
    print(f"\n[METRICS] Computing LLM-label quality metrics on {n_unordered:,} overlapping pairs ...")
    overlap_pd = unordered_overlap.select(["_lo", "_hi", "llm_score", "human_label"]).to_pandas()

    print("[DATA] Loading items_human.parquet for category lookup ...")
    items = pl.read_parquet(ITEMS_HUMAN_PATH).select(["id", "category"])
    id_to_cat = dict(zip(items.get_column("id").to_list(), items.get_column("category").to_list()))
    overlap_pd["category"] = overlap_pd["_lo"].map(id_to_cat).fillna("UNKNOWN")

    y_true = overlap_pd["human_label"].to_numpy().astype(int)
    y_score = overlap_pd["llm_score"].to_numpy().astype(float)
    categories = overlap_pd["category"].to_numpy()

    pear_r, pear_p = pearsonr(y_score, y_true)
    spear_r, spear_p = spearmanr(y_score, y_true)
    y_pred_bin = (y_score >= 0.5).astype(int)

    overall = {
        "n_pairs": int(len(y_true)),
        "human_positive_rate": float(y_true.mean()),
        "llm_score_mean": float(y_score.mean()),
        "pearson_r": float(pear_r), "pearson_p": float(pear_p),
        "spearman_r": float(spear_r), "spearman_p": float(spear_p),
        "roc_auc": float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else None,
        "average_precision": float(average_precision_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else None,
        "accuracy_at_0.5": float(accuracy_score(y_true, y_pred_bin)),
        "precision_at_0.5": float(precision_score(y_true, y_pred_bin, zero_division=0)),
        "recall_at_0.5": float(recall_score(y_true, y_pred_bin, zero_division=0)),
        "f1_at_0.5": float(f1_score(y_true, y_pred_bin, zero_division=0)),
        "mean_llm_score_given_human_1": float(y_score[y_true == 1].mean()) if (y_true == 1).any() else None,
        "mean_llm_score_given_human_0": float(y_score[y_true == 0].mean()) if (y_true == 0).any() else None,
        "mae_llm_vs_human": float(np.mean(np.abs(y_score - y_true))),
        "macro_ap_per_category": compute_metrics_at_row_level(y_true, y_score, categories),
        "n_categories_present": int(len(np.unique(categories))),
    }

    per_cat_rows = []
    for cat in sorted(np.unique(categories)):
        mask = categories == cat
        yt, ys = y_true[mask], y_score[mask]
        row = {"category": cat, "n_pairs": int(mask.sum()), "human_positive_rate": float(yt.mean())}
        if len(np.unique(yt)) > 1:
            row["roc_auc"] = float(roc_auc_score(yt, ys))
            row["average_precision"] = float(average_precision_score(yt, ys))
        else:
            row["roc_auc"] = None
            row["average_precision"] = None
        row["mae_llm_vs_human"] = float(np.mean(np.abs(ys - yt)))
        per_cat_rows.append(row)
    per_cat_df = pd.DataFrame(per_cat_rows)

    print("\n[OVERALL METRICS]")
    print(json.dumps(overall, indent=2))
    print("\n[PER-CATEGORY METRICS]")
    print(per_cat_df.to_string(index=False))

    with open(os.path.join(REPORT_DIR, "llm_label_quality_overall.json"), "w") as f:
        json.dump(overall, f, indent=2)
    per_cat_df.to_csv(os.path.join(REPORT_DIR, "llm_label_quality_per_category.csv"), index=False)
    print(f"\n[DONE] Reports written to {REPORT_DIR}")


if __name__ == "__main__":
    main()
