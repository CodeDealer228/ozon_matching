"""
Pre-tokenize every unique item referenced by matches.parquet (human-labeled
pairs, ~711K items) using ai-forever/FRIDA's tokenizer with the
"paraphrase: " prefix -- FRIDA's own model card names deduplication
explicitly as a paraphrase-prefix use case ("paraphrase prefix is for
symmetric paraphrasing related tasks (STS, paraphrase mining,
deduplication)"), matching this task exactly.

Human/LLM scopes run as two SEPARATE kernels (this one + the sibling
kaggle_kernels/frida-tokenize-items-llm/) rather than one combined script,
so they can run in parallel, isolate failures, and use different hardware:
this scope is small (~711K items) and runs comfortably on a plain CPU
session -- no need to queue for the TPU session the much larger LLM scope
(kaggle_kernels/frida-tokenize-items-llm/) needs for RAM headroom.

Why per-ITEM, not per-pair: FRIDA is a bi-encoder (confirmed via its model
card -- T5-encoder-based, CLS pooling, no text_pair/cross-attention), so
unlike a cross-encoder there's no joint pair-truncation step to replicate.
Each item is tokenized independently, once, to a fixed max_length.

max_length=256 chosen empirically, not FRIDA's 512 hard cap: a 20K-item
local sample tokenized with this exact text formula + prefix came back
mean=126.6, p99=202.0, p99.9=223.0, true max=272 tokens -- 256 covers
~99.99% of items with zero truncation at half the storage of the 512 cap.

Publishes maxamsecret/frida-item-tokens-human so training never needs to
re-tokenize raw item text again -- just look up (input_ids, attention_mask)
by item id.
"""
import os
import sys
import subprocess

# polars is NOT preinstalled on Kaggle's TPU image (confirmed the hard way:
# ModuleNotFoundError on a real run). Must install BEFORE `import polars`
# below -- that import runs at module-load time, before main() is ever
# called, so installing inside main() would be too late.
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "sentencepiece", "polars"])
# Sibling LLM-scope kernel found the `kaggle` CLI binary missing from PATH
# on the TPU image (this CPU image has it preinstalled -- confirmed working
# below -- but installing it explicitly here too removes the dependency on
# that happening to be true).
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "kaggle"])
# Sibling LLM-scope kernel hit a TPU-image-specific transformers/tokenizers
# version mismatch loading FRIDA's RobertaTokenizer (TypeError:
# RobertaProcessing.__new__() got an unexpected keyword argument 'cls').
# `pip install -U` (no pin) is not reliable here -- it resolved fine on this
# CPU image (tokenizers==0.22.2) but resolved a broken pre-release
# (tokenizers==0.23.0rc0) on the TPU image. Pin the exact pair this CPU
# image's own run confirmed working, rather than trusting the resolver.
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "transformers==5.15.0", "tokenizers==0.22.2"])

import gc
import ctypes
import json
import shutil

import numpy as np
import polars as pl

from tqdm.auto import tqdm
from transformers import AutoTokenizer

ITEMS_HUMAN_PATH = "/kaggle/input/datasets/maxamsecret/ozon-ecup-2026-matching/items_human.parquet"

FRIDA_MODEL_NAME = "ai-forever/FRIDA"
FRIDA_PREFIX = "paraphrase: "
MAX_LENGTH = 256

# Rescaled for MAX_LENGTH=256 -- the 128/64/384 CLAUDE.md formula was sized
# for MiniLM's 128-token regime; carried over unscaled here it wastes budget
# on category (real max 8 tokens) while truncating attributes below its real
# p90/p99 before the tokenizer ever sees it. See
# kaggle_kernels/frida-char-token-ratio for the source ratios and
# kaggle_kernels/frida-tokenize-items-v2 for the full derivation.
MAX_NAME_CHARS = 156
MAX_CATEGORY_CHARS = 28
MAX_ATTR_CHARS = 564
MAX_ITEM_CHARS = MAX_NAME_CHARS + MAX_CATEGORY_CHARS + MAX_ATTR_CHARS + 64  # 812

CHUNK_SIZE_START = 20_000
CHUNK_SIZE_MIN = 2_000
CHUNK_SIZE_MAX = 200_000
RAM_TARGET_GB = 16.0
RAM_HARD_GB = 24.0
RAM_GROWTH_FACTOR = 1.3
RAM_SHRINK_FACTOR = 0.5

WORKING_DIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
OUT_DIR = os.path.join(WORKING_DIR, "frida_item_tokens_human")

DATASET_ID = "maxamsecret/frida-tokenized-items-human"
DATASET_TITLE = "FRIDA tokenized items -- human-labeled scope"


def _malloc_trim():
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def _current_rss_gb():
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / 1e9
    except ImportError:
        pass
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1e6
    except Exception:
        pass
    return None


def _log_memory(stage):
    rss_gb = _current_rss_gb()
    if rss_gb is None:
        print(f"[MEM] {stage}: unavailable")
        return
    try:
        import psutil
        avail = psutil.virtual_memory().available / 1e9
        print(f"[MEM] {stage}: rss={rss_gb:.2f}GB available={avail:.2f}GB")
    except ImportError:
        print(f"[MEM] {stage}: rss={rss_gb:.2f}GB")


def _require_disk_space(path, required_bytes, label, margin=1.35):
    os.makedirs(path, exist_ok=True)
    free = shutil.disk_usage(path).free
    need = int(required_bytes * margin)
    print(f"[DISK] {label}: free={free / 1e9:.2f}GB required(+35% margin)={need / 1e9:.2f}GB")
    if free < need:
        raise RuntimeError(f"[DISK] insufficient space for {label}: need {need / 1e9:.2f}GB, have {free / 1e9:.2f}GB")


def build_item_texts(items_lazy):
    return (
        items_lazy.select(["id", "name", "category", "attributes"])
        .with_columns([
            pl.col("name").fill_null("").cast(pl.Utf8, strict=False).str.slice(0, MAX_NAME_CHARS),
            pl.col("category").fill_null("").cast(pl.Utf8, strict=False).str.slice(0, MAX_CATEGORY_CHARS),
            pl.col("attributes").fill_null("").cast(pl.Utf8, strict=False).str.slice(0, MAX_ATTR_CHARS),
        ])
        .with_columns(
            (pl.lit("Name: ") + pl.col("name") + pl.lit(" Category: ") + pl.col("category") +
             pl.lit(" Attributes: ") + pl.col("attributes")).str.slice(0, MAX_ITEM_CHARS).alias("text")
        )
        .select(["id", "text"])
        .collect()
    )


def tokenize_items_to_memmap(items_pl, tokenizer, out_dir, max_length=MAX_LENGTH):
    n = items_pl.height
    ids_path = os.path.join(out_dir, "ids.bin")
    input_ids_path = os.path.join(out_dir, "input_ids.bin")
    attn_path = os.path.join(out_dir, "attention_mask.bin")

    bytes_per_row = 8 + max_length * 4 + max_length * 1
    # bytes_per_row is exact (fixed dtypes/shape, no estimation involved), so
    # the default 35% margin is unnecessarily conservative -- confirmed the
    # hard way on the sibling llm-scope script: needed 21.53GB (15.95GB data
    # +35%) against a real 20.94GB /kaggle/working ceiling, RuntimeError'd
    # despite the actual data fitting with ~5GB to spare. 15% still leaves
    # real headroom for the small JSON sidecar files.
    _require_disk_space(out_dir, n * bytes_per_row, "item token cache", margin=1.15)

    ids_mm = np.memmap(ids_path, dtype=np.int64, mode="w+", shape=(n,))
    input_ids_mm = np.memmap(input_ids_path, dtype=np.int32, mode="w+", shape=(n, max_length))
    attn_mm = np.memmap(attn_path, dtype=np.int8, mode="w+", shape=(n, max_length))

    all_ids = items_pl.get_column("id").to_numpy()
    all_texts = items_pl.get_column("text").to_list()

    chunk_size = CHUNK_SIZE_START
    written = 0
    pbar = tqdm(total=n, desc="Tokenize items", dynamic_ncols=True)

    while written < n:
        cur = min(chunk_size, n - written)
        chunk_texts = [FRIDA_PREFIX + t for t in all_texts[written:written + cur]]

        enc = tokenizer(
            chunk_texts, padding="max_length", truncation=True,
            max_length=max_length, return_tensors="np",
        )

        ids_mm[written:written + cur] = all_ids[written:written + cur]
        input_ids_mm[written:written + cur] = enc["input_ids"].astype(np.int32)
        attn_mm[written:written + cur] = enc["attention_mask"].astype(np.int8)

        written += cur
        pbar.update(cur)

        gc.collect()
        _malloc_trim()
        rss_gb = _current_rss_gb()
        if rss_gb is not None:
            if rss_gb < RAM_TARGET_GB:
                chunk_size = min(CHUNK_SIZE_MAX, int(chunk_size * RAM_GROWTH_FACTOR))
            elif rss_gb > RAM_HARD_GB:
                chunk_size = max(CHUNK_SIZE_MIN, int(chunk_size * RAM_SHRINK_FACTOR))
            pbar.set_postfix(rss_gb=round(rss_gb, 2), next_chunk=chunk_size)

    pbar.close()
    for mm in (ids_mm, input_ids_mm, attn_mm):
        mm.flush()
    del ids_mm, input_ids_mm, attn_mm
    gc.collect()
    return {"ids": ids_path, "input_ids": input_ids_path, "attention_mask": attn_path}


def main():
    if os.path.isdir("/kaggle/input"):
        for _root, _dirs, _files in os.walk("/kaggle/input"):
            depth = _root[len("/kaggle/input"):].count(os.sep)
            if depth <= 3:
                print(f"[DEBUG] {_root} -> dirs={_dirs} files={_files}")

    _log_memory("start")

    import transformers
    import tokenizers
    print(f"[TOKENIZER] transformers={transformers.__version__} tokenizers={tokenizers.__version__}")
    print(f"[TOKENIZER] Loading {FRIDA_MODEL_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(FRIDA_MODEL_NAME)
    print(f"[TOKENIZER] vocab_size={tokenizer.vocab_size} prefix={FRIDA_PREFIX!r} max_length={MAX_LENGTH}")

    print("[DATA] items_human.parquet is already exactly matches.parquet's item scope -- no filter needed")
    items = build_item_texts(pl.scan_parquet(ITEMS_HUMAN_PATH))
    n_items = items.height
    print(f"[DATA] human scope items: {n_items:,}")
    _log_memory("after items load")

    os.makedirs(OUT_DIR, exist_ok=True)
    tokenize_items_to_memmap(items, tokenizer, OUT_DIR)
    del items
    gc.collect()
    _malloc_trim()
    _log_memory("after tokenize")

    with open(os.path.join(OUT_DIR, "cache_meta.json"), "w") as f:
        json.dump({
            "n_items": n_items, "max_length": MAX_LENGTH, "prefix": FRIDA_PREFIX,
            "model": FRIDA_MODEL_NAME, "source": "items_human.parquet (== matches.parquet item scope)",
        }, f, indent=2)
    with open(os.path.join(OUT_DIR, "dataset-metadata.json"), "w") as f:
        json.dump({"id": DATASET_ID, "title": DATASET_TITLE, "licenses": [{"name": "CC0-1.0"}]}, f)

    print(f"[PUBLISH] Creating dataset {DATASET_ID} ...")
    subprocess.check_call(["kaggle", "datasets", "create", "-p", OUT_DIR, "-t"])
    print(f"[DONE] Published at https://www.kaggle.com/datasets/{DATASET_ID}")


if __name__ == "__main__":
    main()
