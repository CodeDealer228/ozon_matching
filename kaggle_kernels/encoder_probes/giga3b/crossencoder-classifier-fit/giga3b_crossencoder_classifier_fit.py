"""
Fits submits/giga3b_probe/classifier.joblib on REAL Giga-Embeddings-instruct
(int8) pooled pair embeddings -- the local dev box has no usable CUDA
(bitsandbytes int8 requires an actual CUDA GPU even just to load the model),
so the classifier shipped in the zip so far was a synthetic-data stub
(scripts/fit_giga_stub_classifier.py). This kernel runs the ACTUAL submission
code (not a reimplementation -- copied verbatim into
maxamsecret/ozon-giga3b-crossencoder-code) on a sample of the canonical train
split, and as a free byproduct, measures real int8 cross-encoder throughput
(batch=1024, seq=192, instruction-prefixed single-string pair text, native
latent-attention pooling via return_embeddings=True) -- the first real
measurement of this specific (corrected, cross-encoder) pipeline on any GPU.
"""

import os
import subprocess
import sys

print("[SETUP] installing pinned deps (internet enabled on this kernel, "
      "unlike the offline judged run) ...", flush=True)
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "einops", "bitsandbytes", "accelerate", "transformers==4.51.0",
])

import shutil
import time

import numpy as np
import pandas as pd
import torch

for _root, _dirs, _files in os.walk("/kaggle/input"):
    depth = _root[len("/kaggle/input"):].count(os.sep)
    if depth <= 3:
        print(f"[DEBUG] {_root} -> dirs={_dirs} files={_files}", flush=True)

CODE_DIR = "/kaggle/input/datasets/maxamsecret/ozon-giga3b-crossencoder-code"
WEIGHTS_SRC = "/kaggle/input/datasets/maxamsecret/giga3b-quantize/giga_int8"
DATA_DIR = "/kaggle/input/datasets/maxamsecret/ozon-ecup-2026-matching"

SUB_DIR = "/kaggle/working/submission_fit"
MODEL_DST = os.path.join(SUB_DIR, "models", "giga_int8")

print("[SETUP] reconstructing the real submission's src/ + weights ...", flush=True)
os.makedirs(os.path.join(SUB_DIR, "src"), exist_ok=True)
shutil.copy(os.path.join(CODE_DIR, "model.py"), os.path.join(SUB_DIR, "src", "model.py"))
shutil.copy(os.path.join(CODE_DIR, "utils.py"), os.path.join(SUB_DIR, "src", "utils.py"))
open(os.path.join(SUB_DIR, "src", "__init__.py"), "w").close()
shutil.copytree(WEIGHTS_SRC, MODEL_DST)

print("[SETUP] layout:", flush=True)
for root, dirs, files in os.walk(SUB_DIR):
    if "models" in root and root != MODEL_DST:
        continue
    for f in files:
        p = os.path.join(root, f)
        print(f"  {p} ({os.path.getsize(p):,} bytes)", flush=True)
print(f"  {MODEL_DST}/ (weights, not listed individually)", flush=True)

sys.path.insert(0, SUB_DIR)
from src.model import GigaCrossEncoder  # noqa: E402
from src.utils import train_pipeline, build_pair_texts, load_needed_item_texts, extract_pair_embeddings  # noqa: E402

gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
print(f"[INFO] torch={torch.__version__} GPU={gpu_name}", flush=True)

# =====================================================================
# Sample the canonical train split (seed=42 stratified 80/20, see CLAUDE.md)
# =====================================================================
SEED = 42
N_SAMPLE = 4000

print("[DATA] loading matches.parquet ...", flush=True)
matches_path = os.path.join(DATA_DIR, "matches.parquet")
items_path = os.path.join(DATA_DIR, "items_human.parquet")

from sklearn.model_selection import train_test_split

m = pd.read_parquet(matches_path)
m = m[m["id1"].notna() & m["id2"].notna()].reset_index(drop=True)
m["target"] = m["target"].fillna(0.0).astype(float)
label = (m["target"] >= 0.5).astype(int).to_numpy()
idx = np.arange(len(m))
train_idx, _val_idx = train_test_split(idx, test_size=0.20, random_state=SEED, stratify=label)
train_df = m.iloc[train_idx].reset_index(drop=True)

sample = train_df.sample(n=min(N_SAMPLE, len(train_df)), random_state=SEED).reset_index(drop=True)
print(f"[DATA] sampled {len(sample):,} train pairs "
      f"(pos={int((sample['target'] >= 0.5).sum())}, neg={int((sample['target'] < 0.5).sum())})", flush=True)

sample_path = "/kaggle/working/train_sample.parquet"
sample[["id1", "id2", "target"]].to_parquet(sample_path)

# =====================================================================
# Load the encoder (real int8 load -- first real test of this cross-encoder
# rewrite on any GPU: attn-backend fallback, return_embeddings=True pooling)
# =====================================================================
print("[LOAD] loading GigaCrossEncoder (int8) ...", flush=True)
t0 = time.time()
encoder = GigaCrossEncoder(MODEL_DST, dtype="int8")
print(f"[LOAD] loaded in {time.time() - t0:.1f}s", flush=True)

# =====================================================================
# Real throughput probe -- small batch-size sweep so we have an actual
# measurement instead of the hand VRAM estimate in submits/giga3b_probe/run.py
# =====================================================================
print("=" * 80, flush=True)
print("[PROBE] batch-size / throughput sweep", flush=True)
id_to_text = load_needed_item_texts(items_path, pd.unique(pd.concat([sample["id1"], sample["id2"]])).tolist())
probe_texts = build_pair_texts(sample.head(256), id_to_text)

for bs in (64, 256, 1024):
    if bs > len(probe_texts):
        continue
    try:
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        _ = encoder.encode_pairs(probe_texts[:bs], batch_size=bs)
        torch.cuda.synchronize()
        dt = time.time() - t0
        peak_gb = torch.cuda.max_memory_allocated() / 1024**3
        print(f"[PROBE] batch_size={bs}: {bs / dt:.1f} pairs/s, peak VRAM {peak_gb:.1f}GB", flush=True)
    except RuntimeError as e:
        print(f"[PROBE] batch_size={bs} FAILED: {type(e).__name__}: {e}", flush=True)
        torch.cuda.empty_cache()

# =====================================================================
# Fit the real classifier
# =====================================================================
print("=" * 80, flush=True)
print("[FIT] fitting classifier on real embeddings ...", flush=True)
t0 = time.time()
OUT_PATH = "/kaggle/working/classifier.joblib"
train_pipeline(
    items_path=items_path,
    matches_path=sample_path,
    encoder=encoder,
    output_classifier_path=OUT_PATH,
    batch_size=1024,
)
elapsed = time.time() - t0
print(f"[FIT] done in {elapsed:.1f}s ({len(sample) / elapsed:.2f} pairs/s including tokenize+forward+fit)", flush=True)
print(f"[DONE] classifier saved to {OUT_PATH}", flush=True)
