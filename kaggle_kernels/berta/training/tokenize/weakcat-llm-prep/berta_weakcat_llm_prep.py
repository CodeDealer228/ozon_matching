"""
TPU prep kernel for the weak-category LLM continue-train experiment.

Filters matches_llm.parquet (11,187,780 LLM soft-labeled pairs) down to
pairs whose id1 belongs to one of the 10 weakest categories by val
macro PR-AUC, then samples 1,000,000 such pairs. The 10 weakest categories
were derived from the ../berta-hardneg-uniform experiment's per-category val
breakdown (that run started from the same source checkpoint this experiment
also resumes from, and regressed only slightly (0.7246->0.7215) after just
1 epoch, so its per-category profile is a reasonable proxy for the source
checkpoint's own weaknesses without needing a separate eval-only pass):
Обувь (0.455), Одежда (0.475), Ювелирские изделия (0.488), Мебель (0.616),
Автотовары (0.636), Галантерея и аксессуары (0.647), Спорт и отдых (0.690),
Электроника (0.711), Строительство и ремонт (0.717), Канцелярские товары
(0.734) -- consistent with CLAUDE.md's own flagged headroom categories for
the sibling MiniLM model (Ювелирные изделия, Одежда, Обувь, Мебель).

Same rationale as ../berta-llm-continue-prep for running this on TPU
(RAM-heavy items.parquet join + item-level tokenization) and for
re-tokenizing with the correct "paraphrase: " prefix from raw text rather
than reusing the already-published maxamsecret/berta-tokenized-items-llm
cache (confirmed prefix="" there via a dedicated probe kernel). Real
available RAM is logged, not assumed.

Output: item token cache (ids.bin/input_ids.bin/attention_mask.bin/
cache_meta.json, ITEM_MAX_LENGTH=128, prefix baked in) for the union of
items referenced by the 1M sampled pairs, + sampled_pairs.parquet
(id1, id2, target float32, category1).
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

WEAK_CATEGORIES = [
    "Обувь", "Одежда", "Ювелирные изделия", "Мебель", "Автотовары",
    "Галантерея и аксессуары", "Спорт и отдых", "Электроника",
    "Строительство и ремонт", "Канцелярские товары",
]

N_LLM_PAIRS = 1_000_000   # per explicit instruction

PREFIX = "paraphrase: "
MAX_NAME_CHARS, MAX_CATEGORY_CHARS, MAX_ATTR_CHARS = 128, 64, 384
MAX_ITEM_CHARS = MAX_NAME_CHARS + MAX_CATEGORY_CHARS + MAX_ATTR_CHARS + 64  # 640, byte-matches CLAUDE.md
ITEM_MAX_LENGTH = 128

TOKENIZE_CHUNK_SIZE_START = 20_000
TOKENIZE_CHUNK_SIZE_MIN = 2_000
TOKENIZE_CHUNK_SIZE_MAX = 200_000
RAM_TARGET_GB = 80.0
RAM_HARD_GB = 150.0
RAM_GROWTH_FACTOR = 1.3
RAM_SHRINK_FACTOR = 0.5

WORKING_DIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
OUTPUT_DIR = os.path.join(WORKING_DIR, "berta_weakcat_llm_prep")

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
        _log("RAM", f"REAL host RAM: total={vm.total / 1e9:.1f}GB available={vm.available / 1e9:.1f}GB")
    except ImportError:
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
                   f"weak_categories={WEAK_CATEGORIES}")

    # -------------------------------------------------------------
    # ids belonging to a weak category (full catalog scan, id + category
    # only -- cheap columns).
    # -------------------------------------------------------------
    weak_ids = (
        pl.scan_parquet(items_path)
        .select(["id", "category"])
        .filter(pl.col("category").is_in(WEAK_CATEGORIES))
        .select("id")
        .collect(engine="streaming")
    )
    _log("DATA", f"weak-category item ids in full catalog: {weak_ids.height:,}")

    # -------------------------------------------------------------
    # Filter matches_llm to pairs whose id1 is a weak-category item, then
    # sample N_LLM_PAIRS.
    # -------------------------------------------------------------
    matches_llm_weak = (
        pl.scan_parquet(matches_llm_path)
        .filter(pl.col("id1").is_not_null() & pl.col("id2").is_not_null())
        .with_columns(pl.col("target").fill_null(0).cast(pl.Float32))
        .join(weak_ids.lazy(), left_on="id1", right_on="id", how="semi")
        .select(["id1", "id2", "target"])
        .collect(engine="streaming")
    )
    n_pool = matches_llm_weak.height
    _log("DATA", f"matches_llm pairs with id1 in a weak category: {n_pool:,}")

    rng = np.random.RandomState(SEED)
    n_sample = min(N_LLM_PAIRS, n_pool)
    if n_sample < N_LLM_PAIRS:
        _log("WARN", f"wanted {N_LLM_PAIRS:,} pairs, pool only has {n_pool:,} -- using all of it")
    sample_idx = rng.choice(n_pool, size=n_sample, replace=False)
    sampled = matches_llm_weak[sample_idx.tolist()]
    del matches_llm_weak, weak_ids
    gc.collect()
    _log("DATA", f"sampled {sampled.height:,} pairs (seed={SEED})")

    needed_ids = pd.unique(np.concatenate([
        sampled.get_column("id1").to_numpy(), sampled.get_column("id2").to_numpy()
    ]))
    _log("DATA", f"unique item ids needed: {len(needed_ids):,} (reuse factor "
                 f"{2 * sampled.height / len(needed_ids):.2f}x)")

    # -------------------------------------------------------------
    # Resolve item text/category for the needed ids from the full catalog.
    # -------------------------------------------------------------
    needed_ids_lazy = pl.DataFrame({"id": pd.Series(needed_ids, dtype="int64")}).lazy()
    items = build_item_texts(
        pl.scan_parquet(items_path).join(needed_ids_lazy, on="id", how="semi")
    )
    n_items = items.height
    _log("DATA", f"items resolved from catalog: {n_items:,} / {len(needed_ids):,} needed")
    if n_items < len(needed_ids):
        found_ids = set(items.get_column("id").to_list())
        missing = [i for i in needed_ids.tolist() if i not in found_ids][:10]
        _log("WARN", f"{len(needed_ids) - n_items:,} ids not found in items.parquet, e.g. {missing}")
    del needed_ids, needed_ids_lazy
    gc.collect()
    _malloc_trim()
    _log_real_ram()

    valid_ids = set(items.get_column("id").to_list())
    before = sampled.height
    sampled = sampled.filter(pl.col("id1").is_in(valid_ids) & pl.col("id2").is_in(valid_ids))
    if sampled.height < before:
        _log("WARN", f"dropped {before - sampled.height:,} pairs referencing a missing item id")

    # -------------------------------------------------------------
    # Tokenize (prefix baked in, add_special_tokens=False -- collate-time
    # assembly downstream, same convention as the human-scope cache).
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
            next_log_at += 200_000

    for mm in (ids_mm, input_ids_mm, attn_mm):
        mm.flush()

    lengths = np.asarray(attn_mm).sum(axis=1)
    _log("TOKENIZE", f"real length: mean={lengths.mean():.1f} p50={np.percentile(lengths, 50):.0f} "
                      f"p99={np.percentile(lengths, 99):.0f} max={lengths.max()} "
                      f"at_cap({ITEM_MAX_LENGTH})={(lengths == ITEM_MAX_LENGTH).sum():,} "
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
            "source": "items.parquet (full catalog) semi-joined to a weak-category-filtered "
                      f"{N_LLM_PAIRS:,}-pair sample of matches_llm.parquet",
            "weak_categories": WEAK_CATEGORIES, "seed": SEED, "n_sampled_pairs": int(sampled_pd.shape[0]),
        }, f, indent=2)

    sampled_pd = sampled_pd[["id1", "id2", "target", "category1"]]
    sampled_pd.to_parquet(os.path.join(OUTPUT_DIR, "sampled_pairs.parquet"), index=False)
    _log("DONE", f"{n:,} items tokenized, {sampled_pd.shape[0]:,} sampled pairs saved -> {OUTPUT_DIR}, "
                 f"total {_elapsed():.0f}s")


if __name__ == "__main__":
    main()
