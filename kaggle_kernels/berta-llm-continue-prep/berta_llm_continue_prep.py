"""
TPU prep kernel for Experiment A (LLM continue-train of BERTA LoRA).

Samples a fixed-size subset of matches_llm.parquet (11,187,780 LLM
soft-labeled pairs, item universe fully disjoint from items_human.parquet --
see CLAUDE.md), sized so a downstream GPU training kernel with the same
recipe as berta-lora-sft-human-only-20-epochs (TOTAL_BATCH_SIZE=256,
PAIR_MAX_LENGTH=128, measured ~61 pairs/sec on 2xT4 from the real
berta-lora-sft-continue-5-epochs run: 292,352 train rows/epoch in ~4,800s of
pure train time, backed out from mlflow's own metric timestamps) finishes
ONE pass + one validation inside a 2-3h GPU budget.

Run on a TPU-shaped Kaggle instance for the host RAM (not the TPU chips --
nothing here touches them, pure CPU tokenization), same rationale as
../rubert-mini-frida-tokenize: the full items.parquet join (13M+ rows,
4.1GB) plus holding ~1M+ tokenized items in memory is exactly the kind of
step a T4 instance's much smaller RAM has hit OOM on before in this project
(berta-lora-pair-cache v3, SIGKILL after ~1.8h). Real available RAM is
logged at the top of main() rather than assumed -- do not trust "TPU has
~40GB" without checking.

Why re-tokenize instead of reusing the already-published
maxamsecret/berta-tokenized-items-llm cache (12,384,610 items, max_length=
256): verified via a dedicated probe kernel (berta-llm-cache-probe) that
its cache_meta.json's prefix="" is real, not a stale default -- 0/10 sampled
decoded items contain "paraphrase" anywhere in the text. The checkpoint
this experiment resumes from was trained exclusively on items with
PREFIX="paraphrase: " baked in BEFORE tokenization (see
berta-lora-sft-human-only-20-epochs). Splicing prefix token ids onto the
front of already-tokenized sequences would risk a BPE/WordPiece boundary
mismatch at the splice point vs. tokenizing the prefixed string in one
call -- re-tokenizing the (small, sampled-only) subset needed here from raw
text is cheap and avoids the risk entirely.

Output (published by hand afterward as a Kaggle dataset, same convention as
../rubert-mini-frida-tokenize): ids.bin / input_ids.bin / attention_mask.bin
/ cache_meta.json (item-level cache, ITEM_MAX_LENGTH=128 -- matches the
human-scope cache's own storage convention, see berta-lora-sft-human-only-
20-epochs's ITEM_MAX_LENGTH comment for why 128 is not lossy at
PAIR_MAX_LENGTH=128) + sampled_pairs.parquet (id1, id2, target float32,
category string -- the actual training rows for Experiment A's GPU kernel).
"""
import os
import sys
import gc
import ctypes
import json
import subprocess
import time

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "polars"])

import numpy as np
import pandas as pd
import polars as pl
from transformers import AutoTokenizer

MODEL_NAME = "sergeyzh/BERTA"

MATCHES_LLM_PATH_HINT = "/kaggle/input/datasets/maxamsecret/ozon-ecup-2026-matching/matches_llm.parquet"
ITEMS_FULL_PATH_HINT = "/kaggle/input/datasets/maxamsecret/ozon-ecup-2026-matching/items.parquet"

SEED = 42

# Sized so combined_train (this sample + the full 292,523-row human split,
# mixed together -- see the GPU kernel) totals ~512K rows, i.e. ~8,400s of
# pure train time at the ~61 pairs/sec calibration above, leaving headroom
# inside a 2.5h (9,000s) budget for the item-cache load, VRAM probe, and one
# full 73,131-row validation pass. See berta_llm_continue_train.py's module
# docstring for the exact arithmetic.
N_LLM_PAIRS = 220_000

PREFIX = "paraphrase: "
MAX_NAME_CHARS, MAX_CATEGORY_CHARS, MAX_ATTR_CHARS = 128, 64, 384
MAX_ITEM_CHARS = MAX_NAME_CHARS + MAX_CATEGORY_CHARS + MAX_ATTR_CHARS + 64  # 640, byte-matches CLAUDE.md
ITEM_MAX_LENGTH = 128   # matches the human-scope cache's storage cap exactly

TOKENIZE_CHUNK_SIZE_START = 20_000
TOKENIZE_CHUNK_SIZE_MIN = 2_000
TOKENIZE_CHUNK_SIZE_MAX = 200_000
RAM_TARGET_GB = 80.0    # generous on purpose -- real TPU RAM is checked and
RAM_HARD_GB = 150.0     # logged below, not assumed; see module docstring.
RAM_GROWTH_FACTOR = 1.3
RAM_SHRINK_FACTOR = 0.5

WORKING_DIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
OUTPUT_DIR = os.path.join(WORKING_DIR, "berta_llm_continue_prep")

t_script_start = time.perf_counter()


def _elapsed():
    return time.perf_counter() - t_script_start


def _log(stage, msg):
    print(f"[{_elapsed():7.1f}s] {stage}: {msg}", flush=True)


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
        return None


def _log_real_ram():
    try:
        import psutil
        vm = psutil.virtual_memory()
        _log("RAM", f"REAL host RAM: total={vm.total / 1e9:.1f}GB available={vm.available / 1e9:.1f}GB "
                     f"(do not trust the ~40GB guess -- this is the measured number)")
    except ImportError:
        _log("RAM", "psutil unavailable -- installing")
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "psutil"])
        _log_real_ram()


def _find_dataset_file(hint, filename):
    if os.path.isfile(hint):
        return hint
    if os.path.isdir("/kaggle/input"):
        for root, _dirs, files in os.walk("/kaggle/input"):
            if filename in files:
                _log("DATA", f"hint path {hint} didn't resolve -- found {filename} via search at {root}")
                return os.path.join(root, filename)
    return hint


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
        .select(["id", "category", "text"])
        .collect(engine="streaming")
    )


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    _log_real_ram()

    matches_llm_path = _find_dataset_file(MATCHES_LLM_PATH_HINT, "matches_llm.parquet")
    items_path = _find_dataset_file(ITEMS_FULL_PATH_HINT, "items.parquet")
    _log("CONFIG", f"matches_llm={matches_llm_path} items={items_path} N_LLM_PAIRS={N_LLM_PAIRS:,} "
                   f"prefix={PREFIX!r} item_max_length={ITEM_MAX_LENGTH}")

    # -------------------------------------------------------------
    # Sample N_LLM_PAIRS rows from matches_llm.parquet (target is a
    # continuous 0..1 LLM soft label -- kept as float32, NOT thresholded,
    # per CLAUDE.md).
    # -------------------------------------------------------------
    matches_llm = (
        pl.scan_parquet(matches_llm_path)
        .filter(pl.col("id1").is_not_null() & pl.col("id2").is_not_null())
        .with_columns(pl.col("target").fill_null(0).cast(pl.Float32))
        .select(["id1", "id2", "target"])
        .collect(engine="streaming")
    )
    n_total = matches_llm.height
    _log("DATA", f"matches_llm.parquet real row count: {n_total:,}")

    rng = np.random.RandomState(SEED)
    sample_idx = rng.choice(n_total, size=min(N_LLM_PAIRS, n_total), replace=False)
    sampled = matches_llm[sample_idx.tolist()]
    del matches_llm
    gc.collect()
    _log("DATA", f"sampled {sampled.height:,} LLM pairs (seed={SEED})")

    needed_ids = pd.unique(np.concatenate([
        sampled.get_column("id1").to_numpy(), sampled.get_column("id2").to_numpy()
    ]))
    _log("DATA", f"unique item ids needed: {len(needed_ids):,} (from {sampled.height:,} pairs, "
                 f"reuse factor {2 * sampled.height / len(needed_ids):.2f}x)")

    # -------------------------------------------------------------
    # Semi-join the FULL items.parquet catalog (13M+ rows, 4.1GB) down to
    # just the needed ids -- this is the RAM-heavy step this kernel exists
    # to run on a TPU instance for.
    # -------------------------------------------------------------
    needed_ids_lazy = pl.DataFrame({"id": pd.Series(needed_ids, dtype="int64")}).lazy()
    items = build_item_texts(
        pl.scan_parquet(items_path).join(needed_ids_lazy, on="id", how="semi")
    )
    n_items = items.height
    _log("DATA", f"items resolved from catalog: {n_items:,} / {len(needed_ids):,} needed "
                 f"({'ALL FOUND' if n_items == len(needed_ids) else 'SOME MISSING -- see below'})")
    if n_items < len(needed_ids):
        found_ids = set(items.get_column("id").to_list())
        missing = [i for i in needed_ids.tolist() if i not in found_ids][:10]
        _log("WARN", f"{len(needed_ids) - n_items:,} ids not found in items.parquet, e.g. {missing}")
    del needed_ids, needed_ids_lazy
    gc.collect()
    _malloc_trim()
    _log_real_ram()

    # Drop any sampled pairs referencing an item that wasn't found (should be
    # ~0 given items.parquet is the full catalog, but don't assume).
    valid_ids = set(items.get_column("id").to_list())
    before = sampled.height
    sampled = sampled.filter(pl.col("id1").is_in(valid_ids) & pl.col("id2").is_in(valid_ids))
    if sampled.height < before:
        _log("WARN", f"dropped {before - sampled.height:,} pairs referencing a missing item id")

    # -------------------------------------------------------------
    # Tokenize (item-level, prefix baked in, add_special_tokens=False --
    # [CLS]/[SEP]/token_type_ids assembled at collate time downstream,
    # identical convention to build_item_cache() in the human-scope script).
    # -------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    _log("SETUP", f"tokenizer={type(tokenizer).__name__} vocab_size={tokenizer.vocab_size}")

    all_ids = items.get_column("id").to_numpy().astype(np.int64)
    all_categories = items.get_column("category").fill_null("UNKNOWN").to_list()
    all_texts = items.get_column("text").to_list()
    del items
    gc.collect()

    id_to_category = dict(zip(all_ids.tolist(), all_categories))
    sampled_pd = sampled.to_pandas()
    sampled_pd["category1"] = sampled_pd["id1"].map(id_to_category).fillna("UNKNOWN")
    del sampled, id_to_category
    gc.collect()

    n = len(all_ids)
    ids_mm = np.memmap(os.path.join(OUTPUT_DIR, "ids.bin"), dtype=np.int64, mode="w+", shape=(n,))
    input_ids_mm = np.memmap(os.path.join(OUTPUT_DIR, "input_ids.bin"), dtype=np.int32, mode="w+",
                              shape=(n, ITEM_MAX_LENGTH))
    attn_mm = np.memmap(os.path.join(OUTPUT_DIR, "attention_mask.bin"), dtype=np.int8, mode="w+",
                         shape=(n, ITEM_MAX_LENGTH))
    ids_mm[:] = all_ids

    chunk_size = TOKENIZE_CHUNK_SIZE_START
    written = 0
    next_log_at = 0
    while written < n:
        cur = min(chunk_size, n - written)
        chunk_texts = [PREFIX + t for t in all_texts[written:written + cur]]

        enc = tokenizer(
            chunk_texts, add_special_tokens=False, padding="max_length",
            truncation=True, max_length=ITEM_MAX_LENGTH, return_tensors="np",
        )
        input_ids_mm[written:written + cur] = enc["input_ids"].astype(np.int32)
        attn_mm[written:written + cur] = enc["attention_mask"].astype(np.int8)

        written += cur
        gc.collect()
        _malloc_trim()
        rss_gb = _current_rss_gb()
        if rss_gb is not None:
            if rss_gb < RAM_TARGET_GB:
                chunk_size = min(TOKENIZE_CHUNK_SIZE_MAX, int(chunk_size * RAM_GROWTH_FACTOR))
            elif rss_gb > RAM_HARD_GB:
                chunk_size = max(TOKENIZE_CHUNK_SIZE_MIN, int(chunk_size * RAM_SHRINK_FACTOR))

        if written >= next_log_at:
            rss_part = f"rss={rss_gb:.2f}GB, " if rss_gb is not None else ""
            _log("TOKENIZE", f"{written:,}/{n:,} ({rss_part}next_chunk={chunk_size:,})")
            next_log_at += 100_000

    for mm in (ids_mm, input_ids_mm, attn_mm):
        mm.flush()

    lengths = np.asarray(attn_mm).sum(axis=1)
    _log("TOKENIZE", f"real length: mean={lengths.mean():.1f} p50={np.percentile(lengths, 50):.0f} "
                      f"p90={np.percentile(lengths, 90):.0f} p99={np.percentile(lengths, 99):.0f} "
                      f"max={lengths.max()} at_cap({ITEM_MAX_LENGTH})={(lengths == ITEM_MAX_LENGTH).sum():,} "
                      f"({100 * (lengths == ITEM_MAX_LENGTH).mean():.2f}%)")
    del ids_mm, input_ids_mm, attn_mm
    gc.collect()
    _malloc_trim()

    with open(os.path.join(OUTPUT_DIR, "cache_meta.json"), "w", encoding="utf-8") as f:
        json.dump({
            "n_items": n, "max_length": ITEM_MAX_LENGTH, "model": MODEL_NAME,
            "prefix": PREFIX, "add_special_tokens": False,
            "text_formula": "Name/Category/Attributes, CLAUDE.md char budgets (128/64/384/640), "
                             "prefix baked in BEFORE tokenization",
            "source": "items.parquet (full catalog) semi-joined to a random N_LLM_PAIRS-pair sample "
                       "of matches_llm.parquet",
            "seed": SEED, "n_sampled_pairs": int(sampled_pd.shape[0]),
        }, f, indent=2)

    sampled_pd = sampled_pd.rename(columns={"target": "target"})[["id1", "id2", "target", "category1"]]
    sampled_pd.to_parquet(os.path.join(OUTPUT_DIR, "sampled_pairs.parquet"), index=False)
    _log("DONE", f"{n:,} items tokenized, {sampled_pd.shape[0]:,} sampled pairs saved -> {OUTPUT_DIR}, "
                 f"total {_elapsed():.0f}s")


if __name__ == "__main__":
    main()
