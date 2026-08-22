from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import (
    ROOT, load_config, read_jsonl, wilson_interval, markdown_table
)


MODEL_ERROR_CLASSES = {
    "truncation_related",
    "critical_attribute_mismatch",
    "product_type_confusion",
    "semantic_or_alias_failure",
    "variant_policy_failure",
    "other_model_error",
}


def collect() -> pd.DataFrame:
    rows = []
    for d in sorted(p for p in (ROOT / "audit_batches").glob("batch_*") if p.is_dir()):
        inp = {r["audit_unit_id"]: r for r in read_jsonl(d / "input.jsonl")}
        out = {r["audit_unit_id"]: r for r in read_jsonl(d / "output.jsonl")}
        for uid, source in inp.items():
            rec = dict(source)
            rec.update({f"audit_{k}": v for k, v in out[uid].items() if k != "audit_unit_id"})
            rows.append(rec)
    return pd.DataFrame(rows)


def rate_summary(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    rows = []
    for vals, g in df.groupby(keys, sort=True, dropna=False):
        if not isinstance(vals, tuple):
            vals = (vals,)
        row = dict(zip(keys, vals))
        n = len(g)
        li = int((g["audit_primary_class"] == "likely_label_issue").sum())
        amb = int((g["audit_primary_class"] == "ambiguous_business_rule").sum())
        mf = int(g["audit_primary_class"].isin(MODEL_ERROR_CLASSES).sum())
        tr = int((g["audit_primary_class"] == "truncation_related").sum())
        ca = int((g["audit_primary_class"] == "critical_attribute_mismatch").sum())

        li_lo, li_hi = wilson_interval(li, n)
        mf_lo, mf_hi = wilson_interval(mf, n)

        row.update({
            "n": n,
            "label_issue_n": li,
            "label_issue_rate": li / n,
            "label_issue_ci95_low": li_lo,
            "label_issue_ci95_high": li_hi,
            "ambiguous_n": amb,
            "ambiguous_rate": amb / n,
            "model_failure_n": mf,
            "model_failure_rate": mf / n,
            "model_failure_ci95_low": mf_lo,
            "model_failure_ci95_high": mf_hi,
            "truncation_n": tr,
            "truncation_rate": tr / n,
            "critical_attribute_n": ca,
            "critical_attribute_rate": ca / n,
            "mean_confidence": g["audit_confidence"].astype(float).mean(),
            "other_model_correct_rate": g["other_model_correct_at_threshold"].astype(float).mean(),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def esc(x):
    return html.escape("" if x is None else str(x))


def pre(x):
    return (
        "<pre style='white-space:pre-wrap;word-break:break-word;"
        "border:1px solid #555;border-radius:8px;padding:8px'>"
        + esc(x) + "</pre>"
    )


def make_review_html(df: pd.DataFrame, path: Path):
    parts = [
        "<html><head><meta charset='utf-8'><style>"
        "body{font-family:Arial,sans-serif;max-width:1500px;margin:20px auto;padding:0 16px}"
        "details{border:1px solid #555;border-radius:8px;padding:8px;margin:10px 0}"
        "summary{font-weight:700;cursor:pointer}</style></head><body>",
        "<h1>Manual review queue</h1>",
    ]
    for _, r in df.iterrows():
        title = (
            f"{r['audit_unit_id']} | {r['audit_primary_class']} | "
            f"conf={float(r['audit_confidence']):.2f}"
        )
        parts.append(f"<details><summary>{esc(title)}</summary>")
        parts.append(
            f"<p><b>focal:</b> {esc(r['focal_model'])} {esc(r['error_type'])} | "
            f"<b>band:</b> {esc(r['severity_band'])} | "
            f"<b>target:</b> {esc(r['target'])} | "
            f"<b>score:</b> {float(r['focal_score']):.6f}</p>"
        )
        parts.append("<h3>Classification</h3>")
        parts.append(pre(json.dumps({
            "primary_class": r["audit_primary_class"],
            "secondary_subtype": r["audit_secondary_subtype"],
            "label_assessment": r["audit_label_assessment"],
            "model_failure_assessment": r["audit_model_failure_assessment"],
            "model_saw_critical_difference": r["audit_model_saw_critical_difference"],
            "truncation_relevance": r["audit_truncation_relevance"],
            "critical_difference": r["audit_critical_difference"],
            "reason_ru": r["audit_reason_ru"],
            "confidence": r["audit_confidence"],
        }, ensure_ascii=False, indent=2)))
        parts.append("<h3>Item 1 FULL</h3>")
        parts.append(pre(r["item1_name_full"]))
        parts.append(pre(r["item1_attributes_full"]))
        parts.append("<h3>Item 2 FULL</h3>")
        parts.append(pre(r["item2_name_full"]))
        parts.append(pre(r["item2_attributes_full"]))
        parts.append("<h3>Focal model visible input — item 1</h3>")
        parts.append(pre(r["focal_item1_visible_input"]))
        parts.append("<h3>Focal model visible input — item 2</h3>")
        parts.append(pre(r["focal_item2_visible_input"]))
        parts.append("</details>")
    parts.append("</body></html>")
    path.write_text("\n".join(parts), encoding="utf-8")


def main():
    cfg = load_config()
    out_dir = ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = collect()
    if len(df) == 0:
        raise SystemExit("No classified outputs found.")

    # Save merged result. CSV is always available; parquet is optional.
    try:
        df.to_parquet(out_dir / "classified_audit.parquet", index=False)
    except ImportError as e:
        print("WARNING: classified_audit.parquet skipped:", e)
    df.to_csv(out_dir / "classified_audit.csv", index=False)

    by_class = (
        df.groupby(["focal_model", "error_type", "audit_primary_class"], sort=True)
        .size().reset_index(name="n")
    )
    by_class["share_within_model_error"] = (
        by_class["n"] /
        by_class.groupby(["focal_model", "error_type"])["n"].transform("sum")
    )
    by_class.to_csv(out_dir / "summary_by_primary_class.csv", index=False)

    by_sev = rate_summary(
        df, ["focal_model", "error_type", "severity_band_index", "severity_band"]
    ).sort_values(["focal_model", "error_type", "severity_band_index"])
    by_sev.to_csv(out_dir / "summary_by_severity.csv", index=False)

    by_cat = rate_summary(
        df, ["focal_model", "error_type", "category"]
    ).sort_values(["focal_model", "error_type", "model_failure_rate"], ascending=[True, True, False])
    by_cat.to_csv(out_dir / "summary_by_category.csv", index=False)

    by_subtype = (
        df.groupby(["focal_model", "error_type", "audit_secondary_subtype"], sort=True)
        .size().reset_index(name="n")
        .sort_values(["focal_model", "error_type", "n"], ascending=[True, True, False])
    )
    by_subtype.to_csv(out_dir / "summary_by_subtype.csv", index=False)

    # Hypothesis table: does label noise decline as rank gets less extreme?
    hyp_rows = []
    for (model, et), g in by_sev.groupby(["focal_model", "error_type"], sort=True):
        g = g.sort_values("severity_band_index")
        rates = g["label_issue_rate"].tolist()
        monotone_nonincreasing = all(
            rates[i] >= rates[i + 1] for i in range(len(rates) - 1)
        )
        first = rates[0] if rates else np.nan
        last = rates[-1] if rates else np.nan
        hyp_rows.append({
            "focal_model": model,
            "error_type": et,
            "first_band_label_issue_rate": first,
            "last_band_label_issue_rate": last,
            "delta_last_minus_first": last - first,
            "monotone_nonincreasing": monotone_nonincreasing,
            "interpretation": (
                "supports_user_hypothesis"
                if pd.notna(first) and pd.notna(last) and last < first
                else "does_not_support_or_inconclusive"
            ),
        })
    hyp = pd.DataFrame(hyp_rows)
    hyp.to_csv(out_dir / "hypothesis_label_noise_by_depth.csv", index=False)

    # Manual review queue.
    conf_thr = float(cfg["classification_confidence_review_threshold"])
    review = df[
        (df["audit_confidence"].astype(float) < conf_thr)
        | (df["audit_label_assessment"] == "ambiguous")
        | (
            (df["audit_primary_class"] == "likely_label_issue")
            & (df["audit_confidence"].astype(float) < 0.90)
        )
        | (
            (df["audit_primary_class"] == "truncation_related")
            & (df["audit_confidence"].astype(float) < 0.85)
        )
    ].copy().sort_values(["audit_confidence", "severity_rank"])

    review.to_json(
        out_dir / "manual_review_queue.jsonl",
        orient="records",
        lines=True,
        force_ascii=False,
    )
    make_review_html(review, out_dir / "manual_review_queue.html")

    # Final report draft.
    report = []
    report.append("# BERTA / FRIDA hard FP-FN audit\n")
    report.append(
        f"Classified audit assignments: **{len(df)}**; "
        f"unique source pairs: **{df['source_row_id'].nunique()}**.\n"
    )

    report.append("## Primary-class mix\n")
    report.append(markdown_table(by_class.sort_values(
        ["focal_model", "error_type", "share_within_model_error"],
        ascending=[True, True, False]
    ), max_rows=60))
    report.append("\n")

    report.append("## Severity-band trends\n")
    report.append(markdown_table(by_sev[[
        "focal_model", "error_type", "severity_band",
        "n", "label_issue_rate", "ambiguous_rate",
        "model_failure_rate", "truncation_rate",
        "critical_attribute_rate", "other_model_correct_rate",
    ]], max_rows=40))
    report.append("\n")

    report.append("## User hypothesis: label issues vs depth\n")
    report.append(markdown_table(hyp, max_rows=20))
    report.append("\n")

    report.append("## Most frequent secondary subtypes\n")
    report.append(markdown_table(by_subtype, max_rows=60))
    report.append("\n")

    report.append("## Categories with highest audited model-failure rate\n")
    report.append(markdown_table(by_cat[[
        "focal_model", "error_type", "category", "n",
        "label_issue_rate", "model_failure_rate",
        "truncation_rate", "critical_attribute_rate",
    ]].sort_values(
        ["focal_model", "error_type", "model_failure_rate"],
        ascending=[True, True, False]
    ), max_rows=80))
    report.append("\n")

    report.append("## Manual review\n")
    report.append(
        f"Cases queued for manual review: **{len(review)}**. "
        "See `manual_review_queue.html`.\n"
    )
    report.append(
        "\n## Main-agent conclusions\n\n"
        "_The main Codex agent must append evidence-grounded conclusions here after the "
        "required manual review from AGENTS.md._\n"
    )

    (out_dir / "FINAL_REPORT.md").write_text(
        "\n".join(report), encoding="utf-8"
    )

    print("Saved outputs to:", out_dir)
    for p in sorted(out_dir.iterdir()):
        print(" ", p.name)


if __name__ == "__main__":
    main()
