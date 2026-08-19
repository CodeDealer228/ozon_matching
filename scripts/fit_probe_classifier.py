"""
Fit the StandardScaler+LogisticRegression classifier.joblib for one of the
cross-encoder-size-probe submissions (submits/<name>_probe/), using the
*same* extract_cls_embeddings/load_encoder code the submission's own run.py
uses at inference time (imported straight from its src/utils.py) so the
classifier is never fit on a formula that then drifts from what predict_pipeline
actually does at judging time.

Trains on a sample of the canonical train split (data/splits/matches_split_seed42.parquet,
split=="train") against data/items_human.parquet (organizer-provided subset
covering every id in matches.parquet). Sample size is intentionally modest --
this box has no usable CUDA (local GPU is a 4GB laptop card), so the fit runs
on CPU; the classifier only needs to be non-degenerate (both classes present,
finite predict_proba), not accurate -- these submissions exist to probe
wall-clock feasibility on the real test stand, not leaderboard score.

Usage:
    python scripts/fit_probe_classifier.py --submission-dir submits/berta_probe --model-dir models/BERTA --n-samples 2000
"""

import argparse
import os
import sys
import tempfile

import pandas as pd

ITEMS_HUMAN_PATH = os.path.join("data", "items_human.parquet")
SPLIT_PATH = os.path.join("data", "splits", "matches_split_seed42.parquet")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission-dir", required=True)
    parser.add_argument("--model-dir", required=True, help="relative to submission-dir, e.g. models/BERTA")
    parser.add_argument("--n-samples", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=64, help="small default: CPU-only local fit, not the H100 test stand")
    parser.add_argument("--output", default="classifier.joblib", help="relative to submission-dir")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    sub_dir_abs = os.path.abspath(args.submission_dir)
    sys.path.insert(0, sub_dir_abs)
    from src.utils import train_pipeline  # noqa: E402

    split_df = pd.read_parquet(SPLIT_PATH)
    train_df = split_df[split_df["split"] == "train"][["id1", "id2", "label"]]
    sample = train_df.sample(n=min(args.n_samples, len(train_df)), random_state=args.seed).reset_index(drop=True)
    sample = sample.rename(columns={"label": "target"})
    sample["target"] = sample["target"].astype(float)
    print(f"[FIT] sampled {len(sample):,} train pairs "
          f"(pos={int(sample['target'].sum())}, neg={int((sample['target'] == 0).sum())})")

    with tempfile.TemporaryDirectory() as tmp:
        matches_tmp = os.path.join(tmp, "matches_sample.parquet")
        sample.to_parquet(matches_tmp)

        model_path = os.path.join(sub_dir_abs, args.model_dir)
        output_path = os.path.join(sub_dir_abs, args.output)

        train_pipeline(
            items_path=os.path.abspath(ITEMS_HUMAN_PATH),
            matches_path=matches_tmp,
            model_path=model_path,
            output_classifier_path=output_path,
            batch_size=args.batch_size,
            n_samples=None,  # already sampled above
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
