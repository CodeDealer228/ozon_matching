"""
First real GPU test of the rewritten submits/giga3b_probe: the pre-quantized
iMiW/Giga-Embeddings-instruct-4bit-nf4 checkpoint (swapped in for this
project's own int8 quantize-and-save pipeline), the new startup
probe_batch_size() (grows to a real OOM then binary-searches, instead of the
old hand-derived "1024, unverified on H100" guess), the torch.compile canary,
and the tokenize/forward prefetch in extract_pair_embeddings.

Answers, in one run:
  1. Does the new checkpoint + rewritten src/model.py actually load on a
     real GPU (not just diffed-against-old-code confidence)?
  2. What does probe_batch_size() actually recommend on this GPU, and does
     the growth/binary-search behave sanely?
  3. Real throughput (pairs/sec) and peak VRAM at that recommended batch
     size, for an H100 projection via the same peak-FLOPS-ratio methodology
     already established in this project (kaggle_kernels/
     giga3b-quantize-benchmark).
  4. Quality: refit classifier.joblib on this checkpoint's embeddings and
     score val macro_pr_auc_by_category, to compare against the int8
     (0.3973) and raw bf16 (0.4004) numbers already on record -- 4-bit
     quantization noise could plausibly land anywhere near those, must be
     measured, not assumed.
"""

import os
import subprocess
import sys

print("[SETUP] installing pinned deps (internet enabled on this kernel, "
      "unlike the offline judged run) ...", flush=True)
try:
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q",
        "einops", "bitsandbytes", "accelerate", "transformers==4.51.0",
    ])
    TRANSFORMERS_PIN = "4.51.0"
except subprocess.CalledProcessError as e:
    print(f"[SETUP] transformers==4.51.0 install failed ({e}); "
          f"falling back to latest transformers", flush=True)
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q", "-U",
        "einops", "bitsandbytes", "accelerate", "transformers",
    ])
    TRANSFORMERS_PIN = "latest"

import gc
import json
import shutil
import time

import numpy as np
import pandas as pd
import torch

for _root, _dirs, _files in os.walk("/kaggle/input"):
    depth = _root[len("/kaggle/input"):].count(os.sep)
    if depth <= 4:
        print(f"[DEBUG] {_root} -> dirs={_dirs} files={_files}", flush=True)


def _find_dir(hint, marker_file):
    """os.walk-based discovery instead of trusting a hardcoded mount path --
    this project hit real, inconsistent Kaggle mount-path variants this same
    week (see reports/experiment_log.md's FRIDA gotchas), so don't assume."""
    if os.path.isfile(os.path.join(hint, marker_file)) or os.path.isdir(os.path.join(hint, marker_file)):
        return hint
    for root, dirs, files in os.walk("/kaggle/input"):
        if marker_file in files or marker_file in dirs:
            return root
    raise RuntimeError(f"could not find {marker_file} anywhere under /kaggle/input (hint was {hint})")


CODE_DIR = _find_dir("/kaggle/input/datasets/maxamsecret/ozon-giga3b-crossencoder-code", "model.py")
WEIGHTS_SRC = _find_dir("/kaggle/input/datasets/maxamsecret/giga3b-4bit-nf4-weights", "giga_4bit_nf4")
DATA_DIR = _find_dir("/kaggle/input/datasets/maxamsecret/ozon-ecup-2026-matching", "matches.parquet")
print(f"[SETUP] CODE_DIR={CODE_DIR}", flush=True)
print(f"[SETUP] WEIGHTS_SRC={WEIGHTS_SRC}", flush=True)
print(f"[SETUP] DATA_DIR={DATA_DIR}", flush=True)

SUB_DIR = "/kaggle/working/submission_validate"
MODEL_DST = os.path.join(SUB_DIR, "models", "giga_4bit_nf4")
RESULTS_PATH = "/kaggle/working/giga3b_4bit_results.json"

print("[SETUP] reconstructing the real submission's src/ + weights ...", flush=True)
os.makedirs(os.path.join(SUB_DIR, "src"), exist_ok=True)
shutil.copy(os.path.join(CODE_DIR, "model.py"), os.path.join(SUB_DIR, "src", "model.py"))
shutil.copy(os.path.join(CODE_DIR, "utils.py"), os.path.join(SUB_DIR, "src", "utils.py"))
open(os.path.join(SUB_DIR, "src", "__init__.py"), "w").close()
weights_src_dir = os.path.join(WEIGHTS_SRC, "giga_4bit_nf4") if os.path.isdir(os.path.join(WEIGHTS_SRC, "giga_4bit_nf4")) else WEIGHTS_SRC
shutil.copytree(weights_src_dir, MODEL_DST)

sys.path.insert(0, SUB_DIR)
from src.model import GigaCrossEncoder  # noqa: E402
from src.utils import (train_pipeline, build_pair_texts, load_needed_item_texts,  # noqa: E402
                        extract_pair_embeddings, INSTRUCTION)

results = {"transformers_pin": TRANSFORMERS_PIN}


def save_results():
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)


gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
total_vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.is_available() else 0
print(f"[INFO] torch={torch.__version__} GPU={gpu_name} total_vram={total_vram_gb:.1f}GB", flush=True)
results["env"] = {"gpu": gpu_name, "total_vram_gb": total_vram_gb, "torch": torch.__version__}
save_results()

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
# Load the encoder -- real first test of the rewritten model.py: checkpoint
# load, attention-backend cascade, torch.compile canary.
# =====================================================================
print("=" * 80, flush=True)
print("[LOAD] loading GigaCrossEncoder (4-bit NF4) ...", flush=True)
t0 = time.time()
encoder = GigaCrossEncoder(MODEL_DST)
load_secs = time.time() - t0
print(f"[LOAD] loaded in {load_secs:.1f}s (compiled={encoder._compiled})", flush=True)
results["load"] = {"secs": load_secs, "compiled": encoder._compiled}
if torch.cuda.is_available():
    results["load"]["baseline_vram_gb"] = torch.cuda.memory_allocated() / 1024**3
    print(f"[LOAD] baseline VRAM (after load, before any batch): "
          f"{results['load']['baseline_vram_gb']:.2f}GB", flush=True)
save_results()

# =====================================================================
# Real batch-size auto-probe -- the main point of this run: does it behave
# sanely, and what does it actually recommend on this GPU?
# =====================================================================
print("=" * 80, flush=True)
print("[PROBE] running probe_batch_size() for real ...", flush=True)
t0 = time.time()
recommended_bs = encoder.probe_batch_size(start=64)
probe_secs = time.time() - t0
print(f"[PROBE] recommended batch_size={recommended_bs} (probe took {probe_secs:.1f}s)", flush=True)
results["probe"] = {"recommended_batch_size": recommended_bs, "probe_secs": probe_secs}
save_results()

# =====================================================================
# Compile decision -- ONLY now, at the single final batch size, and only
# kept if it measurably wins (see model.py's benchmark_and_maybe_compile
# docstring: a prior version compiled at load time, before the probe, and
# that polluted every probe step with a fresh recompile).
# =====================================================================
print("=" * 80, flush=True)
print("[COMPILE] benchmarking eager vs compiled at the recommended batch size ...", flush=True)
encoder.benchmark_and_maybe_compile(recommended_bs)
results["compile"] = {"used": encoder._compiled}
save_results()

# =====================================================================
# Throughput measurement at the recommended batch size, through the REAL
# pipeline entry point (extract_pair_embeddings: tokenize prefetch +
# forward), not a raw forward-pass timing loop.
# =====================================================================
print("=" * 80, flush=True)
print("[THROUGHPUT] measuring real pairs/sec at the recommended batch size", flush=True)
id_to_text = load_needed_item_texts(items_path, pd.unique(pd.concat([sample["id1"], sample["id2"]])).tolist())
probe_texts = build_pair_texts(sample.head(min(max(2048, recommended_bs * 4), len(sample))), id_to_text)

try:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    _ = extract_pair_embeddings(encoder, probe_texts, batch_size=recommended_bs, show_progress=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dt = time.time() - t0
    peak_gb = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0
    ips = len(probe_texts) / dt
    print(f"[THROUGHPUT] batch_size={recommended_bs}: {ips:.1f} pairs/s over {len(probe_texts):,} pairs, "
          f"peak VRAM {peak_gb:.1f}GB", flush=True)
    results["throughput"] = {"batch_size": recommended_bs, "pairs_per_sec": ips, "peak_vram_gb": peak_gb,
                              "n_pairs_measured": len(probe_texts)}
except RuntimeError as e:
    print(f"[THROUGHPUT] FAILED: {type(e).__name__}: {e}", flush=True)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
gc.collect()
save_results()

# H100 projection via the same peak-FLOPS-ratio methodology as
# kaggle_kernels/giga3b-quantize-benchmark, so it's directly comparable to
# the numbers already on record for the int8/bf16 passes.
GPU_PEAK_TFLOPS = {"Tesla T4": 65.0, "Tesla P100": 19.0, "A100": 312.0, "H100": 990.0, "L4": 121.0}
H100_PEAK = 990.0
peak_key = next((k for k in GPU_PEAK_TFLOPS if k in gpu_name), None)
ratio = (H100_PEAK / GPU_PEAK_TFLOPS[peak_key]) if peak_key else None
if ratio and "throughput" in results:
    ips = results["throughput"]["pairs_per_sec"]
    h100_ips = ips * ratio
    results["h100_projection"] = {"ratio": ratio, "this_gpu_ips": ips, "h100_ips_est": h100_ips}
    for n, label_ in ((115_000, "Public"), (275_000, "Private")):
        mins = n / h100_ips / 60
        print(f"[PROJ] {label_} ({n:,} pairs): H100 est {mins:.1f}min (x{ratio:.1f} vs {gpu_name}, indicative)", flush=True)
    save_results()

# =====================================================================
# Fit the real classifier + score quality (macro_pr_auc_by_category)
# =====================================================================
print("=" * 80, flush=True)
print("[FIT] fitting classifier on real 4-bit embeddings ...", flush=True)
t0 = time.time()
OUT_PATH = "/kaggle/working/classifier.joblib"
train_pipeline(
    items_path=items_path,
    matches_path=sample_path,
    encoder=encoder,
    output_classifier_path=OUT_PATH,
    batch_size=recommended_bs,
)
elapsed = time.time() - t0
print(f"[FIT] done in {elapsed:.1f}s ({len(sample) / elapsed:.2f} pairs/s including tokenize+forward+fit)", flush=True)
print(f"[DONE] classifier saved to {OUT_PATH}", flush=True)
print("       references: int8 raw macro_pr_auc=0.3973 | bf16 raw macro_pr_auc=0.4004 | "
      "baseline(trained)=0.3710 | LoRA cross-encoder(trained)=0.5282", flush=True)

print("=" * 80 + "\n[SUMMARY]\n" + json.dumps(results, indent=2, default=str), flush=True)
