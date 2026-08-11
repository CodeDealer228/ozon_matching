"""
End-to-end smoke test for a submission zip's run.py, without needing the
full 4.1GB items.parquet or GPU access.

Builds a small items.parquet + matches.parquet sample from data/items_human.parquet
+ data/matches.parquet -- items_human.parquet is the organizer-provided subset
"for local emulation of the run procedure and debugging performance" and fully
covers every id referenced in matches.parquet (verified: 0 missing).

Invokes run.py through the exact CLI contract the harness uses
(`python -u run.py --items_path --matches_path --output_path`), then checks:
  - process exits 0
  - output CSV has exactly columns id1,id2,predict
  - row count == input row count (no silently dropped pairs)
  - id1/id2 in the output match the input pairs, in order
  - predict is finite and in [0, 1] for every row
  - a rerun on the same sample produces near-identical scores (determinism)

Reports wall-clock time and pairs/sec so the full-scale runtime can be
extrapolated (rough only -- see CLAUDE.md for the caveat that this box is not
the judging hardware).

Usage:
    python scripts/test_submission.py --submission-dir submits/lora --n-pairs 2000
    python scripts/test_submission.py --submission-dir submits/baseline/matching-baseline-lightweight (unzipped) ...
"""

import argparse
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import polars as pl

DATA_ITEMS_HUMAN = os.path.join("data", "items_human.parquet")
DATA_MATCHES = os.path.join("data", "matches.parquet")


def build_sample(n_pairs, sample_dir, seed=42):
    os.makedirs(sample_dir, exist_ok=True)
    items_path = os.path.join(sample_dir, "items.parquet")
    matches_path = os.path.join(sample_dir, "matches.parquet")

    matches = pl.read_parquet(DATA_MATCHES)
    if n_pairs < matches.height:
        matches = matches.sample(n=n_pairs, seed=seed)

    needed_ids = pl.concat([matches["id1"], matches["id2"]]).unique().to_list()

    items = (
        pl.scan_parquet(DATA_ITEMS_HUMAN)
        .filter(pl.col("id").is_in(needed_ids))
        .collect()
    )

    # matches_path at eval time has no "target" column -- drop it here too so
    # the test exercises the same code path as the real harness.
    matches.drop("target").write_parquet(matches_path)
    items.write_parquet(items_path)

    print(f"[SAMPLE] {matches.height:,} pairs, {items.height:,} items -> {sample_dir}")
    return items_path, matches_path, matches


def run_once(submission_dir, items_path, matches_path, output_path):
    items_path = os.path.abspath(items_path)
    matches_path = os.path.abspath(matches_path)
    output_path = os.path.abspath(output_path)

    cmd = [
        sys.executable, "-u", "run.py",
        "--items_path", items_path,
        "--matches_path", matches_path,
        "--output_path", output_path,
    ]
    print(f"[RUN] {' '.join(cmd)} (cwd={submission_dir})")

    t0 = time.time()
    proc = subprocess.run(cmd, cwd=submission_dir, capture_output=True, text=True)
    elapsed = time.time() - t0

    print(proc.stdout[-4000:])
    if proc.returncode != 0:
        print("[STDERR]", proc.stderr[-4000:])
        raise RuntimeError(f"run.py exited with code {proc.returncode}")

    return output_path, elapsed


def validate_output(output_path, input_matches):
    out = pd.read_csv(output_path)

    assert list(out.columns) == ["id1", "id2", "predict"], f"unexpected columns: {list(out.columns)}"
    assert len(out) == len(input_matches), f"row count {len(out)} != input {len(input_matches)}"

    exp_id1 = input_matches["id1"].to_numpy()
    exp_id2 = input_matches["id2"].to_numpy()
    assert np.array_equal(out["id1"].to_numpy(), exp_id1), "id1 order/values mismatch vs input"
    assert np.array_equal(out["id2"].to_numpy(), exp_id2), "id2 order/values mismatch vs input"

    predict = out["predict"].to_numpy(dtype=float)
    assert np.isfinite(predict).all(), "non-finite predict values present"
    assert (predict >= 0).all() and (predict <= 1).all(), \
        f"predict out of [0,1] range: min={predict.min()}, max={predict.max()}"

    print(f"[CHECK] columns OK, {len(out):,} rows OK, id1/id2 order OK, "
          f"predict range [{predict.min():.4f}, {predict.max():.4f}] OK")
    return predict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission-dir", required=True)
    parser.add_argument("--n-pairs", type=int, default=2000)
    parser.add_argument("--sample-dir", default=os.path.join("artifacts", "test_sample"))
    parser.add_argument("--skip-determinism-check", action="store_true")
    args = parser.parse_args()

    items_path, matches_path, matches_df = build_sample(args.n_pairs, args.sample_dir)

    out1_path = os.path.join(os.path.abspath(args.sample_dir), "predictions_1.csv")
    out_path, elapsed = run_once(args.submission_dir, items_path, matches_path, out1_path)
    predict1 = validate_output(out_path, matches_df)

    pairs_per_sec = len(matches_df) / max(elapsed, 1e-9)
    print(f"[TIME] {elapsed:.1f}s total for {len(matches_df):,} pairs "
          f"({pairs_per_sec:.1f} pairs/sec on THIS machine, not judge hardware)")

    if not args.skip_determinism_check:
        out2_path = os.path.join(os.path.abspath(args.sample_dir), "predictions_2.csv")
        _, elapsed2 = run_once(args.submission_dir, items_path, matches_path, out2_path)
        predict2 = validate_output(out2_path, matches_df)

        max_diff = float(np.max(np.abs(predict1 - predict2)))
        print(f"[CHECK] max |predict1 - predict2| = {max_diff:.6f} (determinism across reruns)")
        assert max_diff < 1e-3, "predictions differ meaningfully across identical reruns"

    print("[PASS] submission smoke test passed")


if __name__ == "__main__":
    main()
