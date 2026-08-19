"""
Placeholder classifier.joblib for submits/giga3b_probe/.

The real classifier should be fit on actual Giga-Embeddings-instruct pooled
embeddings (2048-dim, via the model's own latent-attention head,
return_embeddings=True), the same way scripts/fit_probe_classifier.py does
for the BERTA/matryoshka/FRIDA probes. That isn't possible on this dev box:
bitsandbytes int8 requires an actual CUDA GPU even just to load the model
(verified locally -- loading raises ImportError/backend errors with no GPU
present), and this box's only GPU is a 4GB laptop card with no usable CUDA
for this workload.

This script instead fits a LogisticRegression on synthetic separable 2048-dim
Gaussian clusters, purely so classifier.joblib exists, is structurally valid
(same StandardScaler+LogisticRegression Pipeline shape predict_pipeline
expects), and produces varied predict_proba values in [0,1] -- consistent
with the whole submits/giga3b_probe deliverable being a WALL-CLOCK TIMING
PROBE, not a quality submission (same as the batch/max_seq_length tuning:
none of it has been verified on real hardware either). Predictions from this
submission carry no real signal until refit on real embeddings from an
actual GPU run (e.g. a Kaggle kernel, the same way kaggle_kernels/
giga3b-quantize-benchmark produced the int8 weights in the first place).

Usage:
    python scripts/fit_giga_stub_classifier.py
"""

import os

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DIM = 2048
OUT_PATH = os.path.join("submits", "giga3b_probe", "classifier.joblib")
SEED = 42


def main():
    rng = np.random.default_rng(SEED)
    n = 200
    x_pos = rng.normal(loc=0.3, scale=1.0, size=(n, DIM))
    x_neg = rng.normal(loc=-0.3, scale=1.0, size=(n, DIM))
    x = np.vstack([x_pos, x_neg]).astype(np.float32)
    y = np.array([1] * n + [0] * n)

    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=1234)),
    ])
    clf.fit(x, y)

    joblib.dump(clf, OUT_PATH)
    print(f"[STUB] wrote placeholder classifier -> {OUT_PATH} (dim={DIM}, "
          f"NOT fit on real Giga embeddings -- see this script's docstring)")


if __name__ == "__main__":
    main()
