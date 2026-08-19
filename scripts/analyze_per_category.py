"""
Offline post-analysis of the per-category val scores produced by
kaggle_kernels/val-per-category-metrics (val_scores.parquet).

Everything here works off the saved per-pair probabilities, so it costs no
forward pass and can be re-run/extended freely:

  * the per-category metric table + macro decomposition;
  * headroom: how much macro PR-AUC each category could contribute if it were
    lifted to the median / best "lift over prevalence" of the other categories,
    i.e. where the 20-way mean actually loses its points;
  * a free-ensemble probe (rank-blend of the model score with the naive name
    Jaccard) -- per category, to see whether lexical signal the encoder misses
    is worth wiring in;
  * concrete error samples (worst false positives / false negatives) for the
    weakest categories, resolved back to real item names via items_human.parquet.

Usage:
    python scripts/analyze_per_category.py \
        --scores kaggle_kernels/val-per-category-metrics/output/val_scores.parquet \
        --out-dir kaggle_kernels/val-per-category-metrics/output
"""

import argparse
import json
import os

import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score, roc_auc_score

ITEMS_HUMAN = os.path.join("data", "items_human.parquet")
N_ERROR_SAMPLES = 8
N_WEAK_CATEGORIES = 3


def rankdata(x):
    """Average-rank transform (ties get the mean rank), so two differently
    scaled scores can be blended without either dominating."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    ranks[order] = np.arange(1, len(x) + 1)
    # average ties
    sx = x[order]
    i = 0
    while i < len(sx):
        j = i
        while j + 1 < len(sx) and sx[j + 1] == sx[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
        i = j + 1
    return ranks / len(x)


def main():
    ap_arg = argparse.ArgumentParser()
    ap_arg.add_argument("--scores", required=True)
    ap_arg.add_argument("--out-dir", required=True)
    ap_arg.add_argument("--blend-weight", type=float, default=0.15,
                        help="weight of the naive Jaccard rank in the ensemble probe")
    args = ap_arg.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    scores = pl.read_parquet(args.scores)
    print(f"[IN] {args.scores}: {scores.height:,} rows, "
          f"{scores.get_column('category').n_unique()} categories")

    cats = scores.get_column("category").to_numpy()
    y = scores.get_column("label").to_numpy().astype(int)
    p = scores.get_column("prob").to_numpy().astype(float)
    j = scores.get_column("naive_jaccard").to_numpy().astype(float)

    rows = []
    for cat in sorted(np.unique(cats)):
        m = cats == cat
        y_c, p_c, j_c = y[m], p[m], j[m]
        pos_rate = float(y_c.mean())
        ap = float(average_precision_score(y_c, p_c))
        blend = (1 - args.blend_weight) * rankdata(p_c) + args.blend_weight * rankdata(j_c)
        rows.append({
            "category": cat,
            "n_pairs": int(m.sum()),
            "n_pos": int(y_c.sum()),
            "pos_rate": pos_rate,
            "ap": ap,
            # Prevalence-corrected skill score: 0 = no better than a constant
            # scorer (whose AP is exactly the prevalence), 1 = perfect ranking.
            # Raw AP is NOT comparable across categories here -- prevalence
            # ranges ~0.07..0.58 -- so this is the column to rank categories by.
            "ap_skill": (ap - pos_rate) / (1.0 - pos_rate),
            "lift": ap / pos_rate,
            "roc_auc": float(roc_auc_score(y_c, p_c)),
            "ap_naive": float(average_precision_score(y_c, j_c)),
            "ap_blend": float(average_precision_score(y_c, blend)),
        })

    per_cat = pl.DataFrame(rows).sort("ap_skill")
    macro = float(per_cat.get_column("ap").mean())
    macro_blend = float(per_cat.get_column("ap_blend").mean())

    # ---- headroom: the macro is a plain 20-way mean, so every category owns
    # exactly 1/20 of it. Counterfactuals are expressed on the skill score
    # (prevalence-corrected) and then mapped back to AP, since "raise this
    # category to the median AP" would be meaningless across prevalences.
    aps = per_cat.get_column("ap").to_numpy()
    pos_rates = per_cat.get_column("pos_rate").to_numpy()
    skills = per_cat.get_column("ap_skill").to_numpy()
    median_skill = float(np.median(skills))
    top_q_skill = float(np.percentile(skills, 75))

    ap_at_median = pos_rates + median_skill * (1.0 - pos_rates)
    ap_at_top_q = pos_rates + top_q_skill * (1.0 - pos_rates)

    per_cat = per_cat.with_columns([
        pl.Series("macro_share", aps / len(aps)),
        pl.Series("ap_if_median_skill", ap_at_median),
        pl.Series("gain_if_median_skill", np.maximum(0.0, ap_at_median - aps) / len(aps)),
        pl.Series("gain_if_top_quartile_skill", np.maximum(0.0, ap_at_top_q - aps) / len(aps)),
    ])

    # ---- how much (if any) lexical signal the encoder is leaving on the table
    blend_sweep = {}
    for w in [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]:
        vals = []
        for cat in sorted(np.unique(cats)):
            m = cats == cat
            b = (1 - w) * rankdata(p[m]) + w * rankdata(j[m])
            vals.append(average_precision_score(y[m], b))
        blend_sweep[f"w={w:.2f}"] = float(np.mean(vals))

    summary = {
        "macro_pr_auc": macro,
        "macro_pr_auc_rank_blend_with_jaccard": macro_blend,
        "blend_weight_jaccard": args.blend_weight,
        "blend_delta": macro_blend - macro,
        "blend_sweep_macro_pr_auc": blend_sweep,
        "macro_pr_auc_naive_jaccard_only": float(per_cat.get_column("ap_naive").mean()),
        "prevalence_floor": float(per_cat.get_column("pos_rate").mean()),
        "macro_ap_skill": float(np.mean(skills)),
        "median_ap_skill": median_skill,
        "top_quartile_ap_skill": top_q_skill,
        "macro_if_all_at_median_skill": float(np.maximum(aps, ap_at_median).mean()),
        "macro_if_all_at_top_quartile_skill": float(np.maximum(aps, ap_at_top_q).mean()),
        "corr_ap_vs_pos_rate": float(np.corrcoef(aps, pos_rates)[0, 1]),
        "corr_ap_skill_vs_pos_rate": float(np.corrcoef(skills, pos_rates)[0, 1]),
        "corr_ap_vs_ap_naive": float(np.corrcoef(aps, per_cat.get_column("ap_naive").to_numpy())[0, 1]),
    }

    with pl.Config(tbl_rows=30, fmt_str_lengths=40, tbl_width_chars=200):
        print(per_cat)
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    per_cat.write_csv(os.path.join(args.out_dir, "per_category_analysis.csv"))
    with open(os.path.join(args.out_dir, "analysis_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # ---- error samples for the weakest categories
    if not os.path.exists(ITEMS_HUMAN):
        print(f"[SKIP] {ITEMS_HUMAN} not found -- skipping error samples")
        return

    weak = per_cat.get_column("category").to_list()[:N_WEAK_CATEGORIES]
    print(f"\n[ERRORS] resolving item text for weakest categories: {weak}")

    sel = scores.filter(pl.col("category").is_in(weak))
    needed = pl.concat([
        sel.select(pl.col("id1").alias("id")),
        sel.select(pl.col("id2").alias("id")),
    ]).unique()
    items = (
        pl.scan_parquet(ITEMS_HUMAN)
        .select(["id", "name"])
        .join(needed.lazy(), on="id", how="semi")
        .collect()
    )

    sel = (
        sel.join(items.rename({"id": "id1", "name": "name1"}), on="id1", how="left")
           .join(items.rename({"id": "id2", "name": "name2"}), on="id2", how="left")
    )

    lines = []
    for cat in weak:
        c = sel.filter(pl.col("category") == cat)
        fp = c.filter(pl.col("label") == 0).sort("prob", descending=True).head(N_ERROR_SAMPLES)
        fn = c.filter(pl.col("label") == 1).sort("prob").head(N_ERROR_SAMPLES)
        lines.append(f"\n{'=' * 100}\n{cat} -- top false positives (label=0, highest score)\n{'=' * 100}")
        for r in fp.iter_rows(named=True):
            lines.append(f"  p={r['prob']:.4f} jac={r['naive_jaccard']:.3f}\n"
                         f"    A: {r['name1']}\n    B: {r['name2']}")
        lines.append(f"\n{'-' * 100}\n{cat} -- worst false negatives (label=1, lowest score)\n{'-' * 100}")
        for r in fn.iter_rows(named=True):
            lines.append(f"  p={r['prob']:.4f} jac={r['naive_jaccard']:.3f}\n"
                         f"    A: {r['name1']}\n    B: {r['name2']}")

    text = "\n".join(lines)
    print(text)
    with open(os.path.join(args.out_dir, "error_samples.txt"), "w", encoding="utf-8") as f:
        f.write(text)
    print(f"\n[OUT] wrote per_category_analysis.csv, analysis_summary.json, error_samples.txt to {args.out_dir}")


if __name__ == "__main__":
    main()
