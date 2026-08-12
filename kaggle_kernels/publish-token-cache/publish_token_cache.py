"""
One-shot, CPU-only kernel: publishes the tokenized memmap cache produced by
maxamsecret/lora-llm-continue-train's full 11.48M-row run (cancelled by
Kaggle's 12h session cap at ~69% of epoch 1, but its /kaggle/working output
-- including the ~9GB token_cache_llm_continue/ memmap files -- was still
captured as kernel output) as its own standalone dataset.

Runs entirely on Kaggle's infrastructure: the source kernel's output is
mounted read-only via kernel_sources (no local machine involved on either
leg -- earlier attempts to route this ~9GB through a local network hit
repeated SSL failures on the largest file). Needs Kaggle Secrets
(KAGGLE_USERNAME/KAGGLE_KEY) configured for this kernel to call the Datasets
API from inside itself.

Once published, kaggle_kernels/lora-llm-continue-train/lora_llm_continue_train.py's
PRETOKENIZED_CACHE_DIR can point at this dataset's mount path to skip
matches_llm/items loading and the ~9.5h tokenization pass entirely.
"""
import os
import json
import shutil
import subprocess

WORKING_DIR = "/kaggle/working"
STAGE_DIR = os.path.join(WORKING_DIR, "cache_dataset_staging")

DATASET_ID = "maxamsecret/ozon-lora-llm-token-cache"
DATASET_TITLE = "Ozon LoRA continue-train token cache (11.48M rows)"

# From the source run's own logs (CLAUDE.md "Full 11.48M-row run: hit
# Kaggle's 12h session cap"): matches.parquet's 292,523-row human train
# split + all 11,187,780 matches_llm rows (0 dropped by the val anti-join).
TRAIN_LEN = 11_480_303
VAL_LEN = 73_131
N_HUMAN = 292_523
MAX_LENGTH = 128

REQUIRED_FILES = [
    "train_input_ids.bin", "train_attention_mask.bin", "train_token_type_ids.bin",
    "train_labels.bin", "train_category.bin",
    "val_input_ids.bin", "val_attention_mask.bin", "val_token_type_ids.bin",
    "val_labels.bin", "val_category.bin",
]


def _bootstrap_kaggle_secrets():
    from kaggle_secrets import UserSecretsClient
    us = UserSecretsClient()
    os.environ["KAGGLE_USERNAME"] = us.get_secret("KAGGLE_USERNAME")
    os.environ["KAGGLE_KEY"] = us.get_secret("KAGGLE_KEY")
    print("[SECRETS] Kaggle API credentials loaded from kernel secrets.")


def _find_cache_dir():
    # Walk rather than hardcode the kernel_sources mount path -- this
    # project's own CLAUDE.md flags a real, previously-hit gotcha where
    # dataset mount paths have an extra path segment; kernel_sources mounts
    # aren't yet exercised anywhere else in this repo, so don't assume.
    for root, _dirs, files in os.walk("/kaggle/input"):
        if "train_input_ids.bin" in files:
            print(f"[FIND] cache dir located at {root}")
            return root
    raise RuntimeError(
        "Could not locate train_input_ids.bin under /kaggle/input -- "
        "check the [DEBUG] tree above for the actual kernel_sources mount path."
    )


def main():
    print("=" * 90)
    print("[DEBUG] /kaggle/input tree:")
    for _root, _dirs, _files in os.walk("/kaggle/input"):
        depth = _root[len("/kaggle/input"):].count(os.sep)
        if depth <= 4:
            print(f"[DEBUG] {_root} -> dirs={_dirs} files={_files}")
    print("=" * 90)

    _bootstrap_kaggle_secrets()

    cache_dir = _find_cache_dir()

    missing = [f for f in REQUIRED_FILES if not os.path.exists(os.path.join(cache_dir, f))]
    if missing:
        raise RuntimeError(f"Missing expected cache files in {cache_dir}: {missing}")

    os.makedirs(STAGE_DIR, exist_ok=True)
    total_bytes = 0
    for fname in REQUIRED_FILES:
        src = os.path.join(cache_dir, fname)
        dst = os.path.join(STAGE_DIR, fname)
        size = os.path.getsize(src)
        total_bytes += size
        print(f"[COPY] {fname}: {size / 1e9:.3f}GB")
        shutil.copy2(src, dst)
    print(f"[COPY] total: {total_bytes / 1e9:.3f}GB")

    cache_meta = {
        "train_len": TRAIN_LEN, "val_len": VAL_LEN, "n_human": N_HUMAN, "max_length": MAX_LENGTH,
        "source_kernel": "maxamsecret/lora-llm-continue-train",
        "note": "Human 80% train split (292,523 rows) + full matches_llm.parquet (11,187,780 rows), "
                "0 dropped by val anti-join. See CLAUDE.md 'Full 11.48M-row run'.",
    }
    with open(os.path.join(STAGE_DIR, "cache_meta.json"), "w") as f:
        json.dump(cache_meta, f, indent=2)
    print(f"[META] {cache_meta}")

    with open(os.path.join(STAGE_DIR, "dataset-metadata.json"), "w") as f:
        json.dump({"id": DATASET_ID, "title": DATASET_TITLE}, f)

    print(f"[PUBLISH] Creating dataset {DATASET_ID} ...")
    subprocess.check_call(["kaggle", "datasets", "create", "-p", STAGE_DIR, "-t"])
    print(f"[DONE] Published at https://www.kaggle.com/datasets/{DATASET_ID}")


if __name__ == "__main__":
    main()
