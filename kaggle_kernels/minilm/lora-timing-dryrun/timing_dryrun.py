"""
End-to-end timing dry-run of the LoRA submission, on Kaggle GPU, fully
offline (enable_internet: false in kernel-metadata.json) -- this both
sanity-checks wall-clock time against the 20-minute budget AND proves the
submission works with zero network access, matching the real inference
constraint (no internet at judging time).

Reconstructs the exact submission directory layout (run.py, src/utils.py,
classifier.pt, models/lora-merged-ms-marco-MiniLM-L12-v2/) from the attached
datasets, then invokes run.py through the identical CLI contract the judging
harness uses -- not an inlined reimplementation.
"""

import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import shutil
import subprocess
import sys
import time

for _root, _dirs, _files in os.walk("/kaggle/input"):
    depth = _root[len("/kaggle/input"):].count(os.sep)
    if depth <= 4:
        print(f"[DEBUG] {_root} -> dirs={_dirs} files={_files}")

CODE_DIR = "/kaggle/input/datasets/maxamsecret/ozon-lora-submission-code"
MERGED_DIR = "/kaggle/input/datasets/maxamsecret/ozon-lora-merged-minilm"
DATA_DIR = "/kaggle/input/datasets/maxamsecret/ozon-ecup-2026-matching"

SUB_DIR = "/kaggle/working/submission_test"
MODEL_DST = os.path.join(SUB_DIR, "models", "lora-merged-ms-marco-MiniLM-L12-v2")

print("[SETUP] reconstructing submission directory layout ...")
os.makedirs(os.path.join(SUB_DIR, "src"), exist_ok=True)
os.makedirs(MODEL_DST, exist_ok=True)

shutil.copy(os.path.join(CODE_DIR, "run.py"), os.path.join(SUB_DIR, "run.py"))
shutil.copy(os.path.join(CODE_DIR, "metadata.json"), os.path.join(SUB_DIR, "metadata.json"))
shutil.copy(os.path.join(CODE_DIR, "classifier.pt"), os.path.join(SUB_DIR, "classifier.pt"))
shutil.copy(os.path.join(CODE_DIR, "utils.py"), os.path.join(SUB_DIR, "src", "utils.py"))
open(os.path.join(SUB_DIR, "src", "__init__.py"), "w").close()

for fname in os.listdir(MERGED_DIR):
    shutil.copy(os.path.join(MERGED_DIR, fname), os.path.join(MODEL_DST, fname))

print("[SETUP] layout:")
for root, dirs, files in os.walk(SUB_DIR):
    for f in files:
        p = os.path.join(root, f)
        print(f"  {p} ({os.path.getsize(p):,} bytes)")

items_path = os.path.join(DATA_DIR, "items.parquet")
matches_path = os.path.join(DATA_DIR, "matches.parquet")
output_path = "/kaggle/working/predictions.csv"

cmd = [
    sys.executable, "-u", "run.py",
    "--items_path", items_path,
    "--matches_path", matches_path,
    "--output_path", output_path,
]
print(f"[RUN] {' '.join(cmd)} (cwd={SUB_DIR})")

t0 = time.time()
proc = subprocess.run(cmd, cwd=SUB_DIR)
elapsed = time.time() - t0

if proc.returncode != 0:
    raise RuntimeError(f"run.py exited with code {proc.returncode}")

print(f"[TIME] run.py completed in {elapsed:.1f}s ({elapsed / 60:.2f} min)")

import pandas as pd

out = pd.read_csv(output_path)
matches_df = pd.read_parquet(matches_path)

print(f"[CHECK] output rows={len(out):,} vs input matches rows={len(matches_df):,}")
assert len(out) == len(matches_df), "row count mismatch"
assert list(out.columns) == ["id1", "id2", "predict"], f"unexpected columns: {list(out.columns)}"
assert out["predict"].between(0, 1).all(), "predict out of [0,1] range"
assert out["predict"].notna().all(), "NaN predictions present"

print(f"[CHECK] predict range [{out['predict'].min():.4f}, {out['predict'].max():.4f}]")
print(f"[DONE] {len(matches_df):,} pairs in {elapsed:.1f}s = {len(matches_df) / elapsed:.1f} pairs/sec on this GPU")
print("[DONE] fully offline run succeeded (HF_HUB_OFFLINE=1, TRANSFORMERS_OFFLINE=1, enable_internet=false)")
