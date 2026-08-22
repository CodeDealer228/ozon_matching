from __future__ import annotations

import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd

from common import (
    ROOT, find_source_artifact, load_config, read_table, stable_seed,
    write_jsonl, sha256_file, canonical_text
)


REQUIRED_BASE = {
    "row_id", "id1", "id2", "label", "category",
    "berta_score", "frida_score",
    "item1_name_full", "item2_name_full",
    "item1_attributes_full", "item2_attributes_full",
    "berta_item1_visible_decoded", "berta_item2_visible_decoded",
    "frida_item1_visible_decoded", "frida_item2_visible_decoded",
}

OPTIONAL_COPY = [
    "berta_item1_pre_token_text", "berta_item2_pre_token_text",
    "frida_item1_pre_token_text", "frida_item2_pre_token_text",
    "berta_final_pair_decoded", "frida_final_pair_decoded",
    "berta_pair_truncated", "berta_pair_tokens_lost",
    "frida_pair_truncated", "frida_pair_tokens_lost",
    "berta_item1_pair_tokens_lost", "berta_item2_pair_tokens_lost",
    "frida_item1_pair_tokens_lost", "frida_item2_pair_tokens_lost",
    "berta_item1_attr_chars_lost", "berta_item2_attr_chars_lost",
    "frida_item1_attr_chars_lost", "frida_item2_attr_chars_lost",
    "berta_item1_hit_item_cap", "berta_item2_hit_item_cap",
    "frida_item1_hit_item_cap", "frida_item2_hit_item_cap",
]


def ensure_error_columns(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    out = df.copy()
    for model in ["berta", "frida"]:
        score = out[f"{model}_score"].astype(float)
        pred = (score >= threshold).astype(np.int8)
        y = out["label"].astype(int)

        err = np.full(len(out), "correct", dtype=object)
        err[(y == 0) & (pred == 1)] = "FP"
        err[(y == 1) & (pred == 0)] = "FN"
        out[f"{model}_error_type"] = err
        out[f"{model}_wrongness"] = np.where(y == 0, score, 1.0 - score)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=None)
    ap.add_argument("--sample-per-band", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config()
    source = args.input or find_source_artifact()
    df = read_table(source)

    missing = REQUIRED_BASE - set(df.columns)
    if missing:
        raise SystemExit(
            "Source artifact is missing required columns:\n  - "
            + "\n  - ".join(sorted(missing))
            + "\nUse the exact-input explorer output, not top_errors_*."
        )

    df = ensure_error_columns(df, float(cfg["threshold"]))
    df["label"] = df["label"].astype(int)

    out_dir = ROOT / "artifacts" / "prepared"
    out_dir.mkdir(parents=True, exist_ok=True)

    sample_per_band = (
        int(args.sample_per_band)
        if args.sample_per_band is not None
        else int(cfg["sample_per_band"])
    )

    selected_records = []
    coverage = []

    for group in cfg["groups"]:
        model = group["model"]
        error_type = group["error_type"]
        other = "frida" if model == "berta" else "berta"

        g = df[df[f"{model}_error_type"] == error_type].copy()
        g = g.sort_values(
            [f"{model}_wrongness", "row_id"],
            ascending=[False, True],
            kind="mergesort",
        ).reset_index(drop=True)

        g["severity_rank"] = np.arange(1, len(g) + 1, dtype=np.int64)
        g["total_errors_in_group"] = len(g)
        if len(g) > 1:
            g["hardness_percentile"] = 1.0 - (g["severity_rank"] - 1) / (len(g) - 1)
        else:
            g["hardness_percentile"] = 1.0

        for band_idx, band in enumerate(cfg["severity_bands"], start=1):
            lo, hi = int(band["start_rank"]), int(band["end_rank"])
            b = g[(g["severity_rank"] >= lo) & (g["severity_rank"] <= hi)].copy()

            n_take = min(sample_per_band, len(b))
            seed = stable_seed(
                model, error_type, band["name"],
                base_seed=int(cfg["random_seed"])
            )

            if n_take > 0:
                chosen = b.sample(n=n_take, random_state=seed, replace=False)
                chosen = chosen.sort_values("severity_rank")
            else:
                chosen = b

            coverage.append({
                "focal_model": model,
                "error_type": error_type,
                "severity_band": band["name"],
                "band_index": band_idx,
                "rank_start": lo,
                "rank_end": hi,
                "available": len(b),
                "sampled": n_take,
                "requested": sample_per_band,
            })

            for _, r in chosen.iterrows():
                target = int(r["label"])
                focal_score = float(r[f"{model}_score"])
                other_score = float(r[f"{other}_score"])
                other_pred = int(other_score >= float(cfg["threshold"]))
                other_correct = int(other_pred == target)

                rec = {
                    "audit_unit_id": (
                        f"{model}_{error_type}_rank{int(r['severity_rank']):04d}"
                        f"_row{int(r['row_id'])}"
                    ),
                    "source_row_id": int(r["row_id"]),
                    "focal_model": model,
                    "other_model": other,
                    "error_type": error_type,
                    "severity_rank": int(r["severity_rank"]),
                    "severity_band": band["name"],
                    "severity_band_index": band_idx,
                    "hardness_percentile": float(r["hardness_percentile"]),
                    "total_errors_in_group": int(r["total_errors_in_group"]),
                    "target": target,
                    "category": canonical_text(r["category"]),
                    "id1": canonical_text(r["id1"]),
                    "id2": canonical_text(r["id2"]),
                    "focal_score": focal_score,
                    "other_model_score": other_score,
                    "other_model_correct_at_threshold": other_correct,
                    "berta_score": float(r["berta_score"]),
                    "frida_score": float(r["frida_score"]),
                    "item1_name_full": canonical_text(r["item1_name_full"]),
                    "item2_name_full": canonical_text(r["item2_name_full"]),
                    "item1_attributes_full": canonical_text(r["item1_attributes_full"]),
                    "item2_attributes_full": canonical_text(r["item2_attributes_full"]),
                    "focal_item1_visible_input": canonical_text(
                        r[f"{model}_item1_visible_decoded"]
                    ),
                    "focal_item2_visible_input": canonical_text(
                        r[f"{model}_item2_visible_decoded"]
                    ),
                    "other_item1_visible_input": canonical_text(
                        r[f"{other}_item1_visible_decoded"]
                    ),
                    "other_item2_visible_input": canonical_text(
                        r[f"{other}_item2_visible_decoded"]
                    ),
                }

                for c in OPTIONAL_COPY:
                    if c in r.index:
                        v = r[c]
                        if pd.isna(v):
                            rec[c] = None
                        elif hasattr(v, "item"):
                            rec[c] = v.item()
                        else:
                            rec[c] = v

                selected_records.append(rec)

    sample = pd.DataFrame(selected_records)
    if sample["audit_unit_id"].duplicated().any():
        dups = sample.loc[sample["audit_unit_id"].duplicated(), "audit_unit_id"].tolist()
        raise SystemExit(f"Duplicate audit_unit_id values: {dups[:10]}")

    coverage_df = pd.DataFrame(coverage)

    jsonl_path = out_dir / "audit_sample.jsonl"
    parquet_path = out_dir / "audit_sample.parquet"
    coverage_path = out_dir / "coverage.csv"

    write_jsonl(sample, jsonl_path)

    parquet_written = False
    try:
        sample.to_parquet(parquet_path, index=False)
        parquet_written = True
    except ImportError as e:
        print("WARNING: parquet output skipped because no parquet engine is installed:", e)

    coverage_df.to_csv(coverage_path, index=False)

    manifest = {
        "source_path": str(source),
        "source_sha256": sha256_file(source),
        "threshold": cfg["threshold"],
        "random_seed": cfg["random_seed"],
        "sample_per_band": sample_per_band,
        "n_audit_units": int(len(sample)),
        "n_unique_source_rows": int(sample["source_row_id"].nunique()),
        "groups": cfg["groups"],
        "severity_bands": cfg["severity_bands"],
        "audit_sample_jsonl_sha256": sha256_file(jsonl_path),
        "audit_sample_parquet_sha256": sha256_file(parquet_path) if parquet_written else None,
        "audit_sample_parquet_written": parquet_written,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(coverage_df.to_string(index=False))
    print("\nAudit units:", len(sample))
    print("Unique source rows:", sample["source_row_id"].nunique())
    print("Saved:", jsonl_path)
    if parquet_written:
        print("Saved:", parquet_path)
    print("Saved:", coverage_path)
    print("Saved:", out_dir / "manifest.json")


if __name__ == "__main__":
    main()
