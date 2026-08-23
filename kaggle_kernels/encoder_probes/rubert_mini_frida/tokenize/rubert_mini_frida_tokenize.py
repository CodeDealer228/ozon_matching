"""
Tokenizes all 711,304 items_human.parquet items for
sergeyzh/rubert-mini-frida -- a SEPARATE kernel from embedding computation
(see ../rubert-mini-frida-item-embeddings), deliberately run on a
TPU-shaped Kaggle instance. Nothing here touches the TPU chips themselves
-- this is plain CPU tokenization, no torch, no model forward pass at all
-- the point is the much larger HOST RAM that comes with a TPU instance on
Kaggle, removing the OOM risk this exact task already hit once
(kaggle_kernels/berta-lora-pair-cache v3: SIGKILL on both ranks after
~1.8h, tokenizing the same-sized item set unchunked on a T4 instance's much
smaller RAM budget) rather than just mitigating it with tighter chunking
on the same cramped machine. Chunking is kept anyway, for progress
visibility and as a second line of defense, not because it's the only
thing standing between this and OOM this time.

Output: ids.bin / input_ids.bin / attention_mask.bin / cache_meta.json --
the same memmap-cache convention already used by
maxamsecret/frida-item-tokens-human elsewhere in this project. Publish as
a Kaggle dataset BY HAND afterward (not from inside this kernel, matching
this project's established convention) so
../rubert-mini-frida-item-embeddings can mount it and skip tokenization
entirely -- pure GPU-bound embedding computation there, no CPU/RAM-bound
step competing with it.

Text formula: the canonical CLAUDE.md one (Name/Category/Attributes, same
128/64/384 char budgets used everywhere else in this project), prefixed
with "paraphrase: " -- rubert-mini-frida is in the same FRIDA-distillation
lineage as sergeyzh/BERTA (already used elsewhere in this project), whose
own README calls out "paraphrase: " by name for dedup/matching tasks; kept
consistent rather than guessing a different prefix for a sibling model.
Category is INCLUDED here (unlike the per-category probes elsewhere in
this project that deliberately excise it) -- this cache spans the full
catalog across all 20 categories, where category is genuinely not
constant, unlike a single-category-scoped script.
"""

import os
import gc
import ctypes
import json
import time

import numpy as np
import polars as pl
from transformers import AutoTokenizer

MODEL_NAME = "sergeyzh/rubert-mini-frida"

ITEMS_HUMAN_PATH_HINT = "/kaggle/input/datasets/maxamsecret/ozon-ecup-2026-matching/items_human.parquet"

MAX_NAME_CHARS, MAX_CATEGORY_CHARS, MAX_ATTR_CHARS = 128, 64, 384
MAX_ITEM_CHARS = MAX_NAME_CHARS + MAX_CATEGORY_CHARS + MAX_ATTR_CHARS + 64  # 640, byte-matches CLAUDE.md

PREFIX = "paraphrase: "
ITEM_MAX_LENGTH = 256

# TPU instance RAM is much larger than a T4 instance's -- these targets are
# deliberately generous (vs. berta-lora-pair-cache's 16/24GB) rather than
# tight, since headroom is the whole point of running this here. Chunking
# still adapts within these bounds as a safety net, not because it's
# expected to need to shrink much.
TOKENIZE_CHUNK_SIZE_START = 20_000
TOKENIZE_CHUNK_SIZE_MIN = 2_000
TOKENIZE_CHUNK_SIZE_MAX = 200_000
RAM_TARGET_GB = 80.0
RAM_HARD_GB = 150.0
RAM_GROWTH_FACTOR = 1.3
RAM_SHRINK_FACTOR = 0.5

WORKING_DIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
OUTPUT_DIR = os.path.join(WORKING_DIR, "rubert_mini_frida_item_tokens")

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


def _find_dataset_file(hint, filename):
    if os.path.isfile(hint):
        return hint
    if os.path.isdir("/kaggle/input"):
        for root, _dirs, files in os.walk("/kaggle/input"):
            if filename in files:
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
        .select(["id", "text"])
        .collect()
    )


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    items_path = _find_dataset_file(ITEMS_HUMAN_PATH_HINT, "items_human.parquet")
    _log("CONFIG", f"model={MODEL_NAME!r} item_max_length={ITEM_MAX_LENGTH} items_path={items_path}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    _log("SETUP", f"tokenizer loaded, vocab_size={tokenizer.vocab_size}")

    items = build_item_texts(pl.scan_parquet(items_path))
    n = items.height
    all_ids = items.get_column("id").to_numpy().astype(np.int64)
    all_texts = items.get_column("text").to_list()
    del items
    gc.collect()
    _log("DATA", f"{n:,} items loaded")

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
            chunk_texts, add_special_tokens=True, padding="max_length",
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
    del ids_mm, input_ids_mm, attn_mm
    gc.collect()
    _malloc_trim()

    with open(os.path.join(OUTPUT_DIR, "cache_meta.json"), "w", encoding="utf-8") as f:
        json.dump({
            "n_items": n, "max_length": ITEM_MAX_LENGTH, "model": MODEL_NAME,
            "prefix": PREFIX, "add_special_tokens": True,
            "text_formula": "Name/Category/Attributes, CLAUDE.md char budgets (128/64/384/640)",
        }, f, indent=2)

    _log("DONE", f"{n:,} items tokenized -> {OUTPUT_DIR}, total {_elapsed():.0f}s")


if __name__ == "__main__":
    main()
