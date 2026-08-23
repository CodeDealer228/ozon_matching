"""
TPU prep kernel: full-scope, correctly-prefixed, 256-length item token cache
for matches_llm.parquet (11,187,780 LLM soft-labeled pairs), superseding
maxamsecret/berta-tokenized-items-llm (same 12,384,610-item scope, but
confirmed via a dedicated probe kernel to have prefix="" baked in -- 0/10
sampled decoded items contained "paraphrase" anywhere in the text, despite
cache_meta.json's own field claiming a real value; every other item cache
in this project, human or LLM-scope, has PREFIX="paraphrase: " baked in
BEFORE tokenization, and the base sergeyzh/BERTA model's own card calls out
this prefix by name for dedup/matching tasks -- worth keeping consistent
even for a fresh LoRA init, since the FROZEN backbone's own pretraining
still expects it).

max_length=256 this time (was 128 everywhere else in this project) -- the
project's own documented headroom analysis (CLAUDE.md Open TODO 3b) flags
128 as the single biggest remaining bottleneck: at PAIR_MAX_LENGTH=128,
98.76% of real pairs get truncated, mean 174.6 tokens lost. Raising to 256
roughly doubles the per-side content budget available to
allocate_pair_budget downstream (content_budget = PAIR_MAX_LENGTH-3).

No sampling this time -- every id referenced anywhere in matches_llm.parquet
gets tokenized (matching maxamsecret/berta-tokenized-items-llm's own full
scope), since the point of moving this SFT run to TPU v5e-8 is precisely to
afford training at a scale a 2xT4 GPU session's ~9.5h-per-run rejoin
bottleneck never could (CLAUDE.md's own paused-continue-SFT history).

Same TPU-for-RAM rationale as every other *-prep kernel in this project
(13M+-row items.parquet join + tokenizing ~12.4M items) -- real available
RAM logged, not assumed.
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
ITEMS_HUMAN_PATH_HINT = "/kaggle/input/datasets/maxamsecret/ozon-ecup-2026-matching/items_human.parquet"

PREFIX = "paraphrase: "
# Char budget derived from a real, full-dataset rule -- see
# reports/experiment_log.md "Char-budget rule v2" entry. Exact
# percentiles over ALL 711,304 items_human.parquet rows + chars/token
# measured directly against THIS tokenizer (BERTA BertTokenizer,
# WordPiece, notably less efficient than FRIDA's byte-BPE on
# attributes): name=2.783, category=2.612, attributes=2.500
# chars/token, skeleton overhead=15 tokens.
#   NAME_CHAR_TARGET=140     -- p98=137 char + margin -> ~98.1% of real
#                               names fit UNTRUNCATED (target: 95-99%)
#   CATEGORY_CHAR_TARGET=32  -- true max=23 chars across all 20 known
#                               categories + margin -> 100% fit
#   MAX_ATTR_CHARS=442       -- remainder of ITEM_MAX_LENGTH=256 tokens
#                               after overhead+name+category token cost
MAX_NAME_CHARS, MAX_CATEGORY_CHARS, MAX_ATTR_CHARS = 140, 32, 442
# 30 = exact len("Name: ")+len(" Category: ")+len(" Attributes: "), not
# a rounded guess.
MAX_ITEM_CHARS = MAX_NAME_CHARS + MAX_CATEGORY_CHARS + MAX_ATTR_CHARS + 30  # 644
ITEM_MAX_LENGTH = 256   # raised from 128 -- see module docstring

TOKENIZE_CHUNK_SIZE_START = 20_000
TOKENIZE_CHUNK_SIZE_MIN = 2_000
TOKENIZE_CHUNK_SIZE_MAX = 200_000
RAM_TARGET_GB = 100.0   # a real TPU v5e-8 host was measured at 405.7GB total /
RAM_HARD_GB = 200.0     # 389GB available (berta-llm-continue-prep's own log) --
                          # generous targets, not the conservative T4-instance
                          # ones used elsewhere, since 12.4M items x 256 tokens
                          # is a bigger working set than any prior *-prep run.

WORKING_DIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
OUTPUT_DIR = os.path.join(WORKING_DIR, "berta_tokenize_llm_256")

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


def tokenize_scope(items, tokenizer, out_dir, extra_meta):
    """items: a polars DataFrame with columns id/category/text (already
    built via build_item_texts). Tokenizes into out_dir's memmap cache +
    writes cache_meta.json/category_vocab.json/categories.npy there."""
    os.makedirs(out_dir, exist_ok=True)
    n_items = items.height

    all_ids = items.get_column("id").to_numpy().astype(np.int64)
    all_categories = items.get_column("category").fill_null("UNKNOWN").to_list()
    all_texts = items.get_column("text").to_list()
    del items
    gc.collect()

    n = len(all_ids)
    import shutil
    require_bytes = n * (8 + ITEM_MAX_LENGTH * 4 + ITEM_MAX_LENGTH * 1)
    free_bytes = shutil.disk_usage(out_dir).free
    _log("DISK", f"{out_dir}: free={free_bytes / 1e9:.1f}GB required(+15%)={require_bytes * 1.15 / 1e9:.1f}GB")
    if free_bytes < require_bytes * 1.15:
        raise RuntimeError(f"insufficient disk space for {out_dir}: need {require_bytes * 1.15 / 1e9:.1f}GB, "
                            f"have {free_bytes / 1e9:.1f}GB")

    ids_mm = np.memmap(os.path.join(out_dir, "ids.bin"), dtype=np.int64, mode="w+", shape=(n,))
    input_ids_mm = np.memmap(os.path.join(out_dir, "input_ids.bin"), dtype=np.int32, mode="w+",
                              shape=(n, ITEM_MAX_LENGTH))
    attn_mm = np.memmap(os.path.join(out_dir, "attention_mask.bin"), dtype=np.int8, mode="w+",
                         shape=(n, ITEM_MAX_LENGTH))
    ids_mm[:] = all_ids

    category_vocab = sorted(set(all_categories) | {"UNKNOWN"})
    category_to_id = {c: i for i, c in enumerate(category_vocab)}
    categories_arr = np.array([category_to_id[c] for c in all_categories], dtype=np.int32)
    np.save(os.path.join(out_dir, "categories.npy"), categories_arr)
    with open(os.path.join(out_dir, "category_vocab.json"), "w", encoding="utf-8") as f:
        json.dump(category_vocab, f, ensure_ascii=False, indent=2)
    _log("DATA", f"{out_dir}: category vocab size: {len(category_vocab)}")

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
                chunk_size = min(TOKENIZE_CHUNK_SIZE_MAX, int(chunk_size * 1.3))
            elif rss_gb > RAM_HARD_GB:
                chunk_size = max(TOKENIZE_CHUNK_SIZE_MIN, int(chunk_size * 0.5))

        if written >= next_log_at:
            rss_part = f"rss={rss_gb:.2f}GB, " if rss_gb is not None else ""
            _log("TOKENIZE", f"{out_dir}: {written:,}/{n:,} ({rss_part}next_chunk={chunk_size:,})")
            next_log_at += 500_000

    for mm in (ids_mm, input_ids_mm, attn_mm):
        mm.flush()

    lengths = np.asarray(attn_mm).sum(axis=1)
    _log("TOKENIZE", f"{out_dir}: real length: mean={lengths.mean():.1f} p50={np.percentile(lengths, 50):.0f} "
                      f"p90={np.percentile(lengths, 90):.0f} p99={np.percentile(lengths, 99):.0f} "
                      f"max={lengths.max()} at_cap({ITEM_MAX_LENGTH})={(lengths == ITEM_MAX_LENGTH).sum():,} "
                      f"({100 * (lengths == ITEM_MAX_LENGTH).mean():.2f}%)")
    del ids_mm, input_ids_mm, attn_mm
    gc.collect()
    _malloc_trim()

    meta = {
        "n_items": n, "max_length": ITEM_MAX_LENGTH, "model": MODEL_NAME,
        "prefix": PREFIX, "add_special_tokens": False,
        "text_formula": "Name/Category/Attributes, CLAUDE.md char budgets (128/64/384/640), "
                         "prefix baked in BEFORE tokenization",
        "category_vocab_file": "category_vocab.json", "categories_file": "categories.npy",
    }
    meta.update(extra_meta)
    with open(os.path.join(out_dir, "cache_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    _log("DONE", f"{out_dir}: {n:,} items tokenized")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    _log_real_ram()

    matches_llm_path = _find_dataset_file(MATCHES_LLM_PATH_HINT, "matches_llm.parquet")
    items_path = _find_dataset_file(ITEMS_FULL_PATH_HINT, "items.parquet")
    items_human_path = _find_dataset_file(ITEMS_HUMAN_PATH_HINT, "items_human.parquet")
    _log("CONFIG", f"matches_llm={matches_llm_path} items={items_path} items_human={items_human_path} "
                   f"prefix={PREFIX!r} item_max_length={ITEM_MAX_LENGTH} (no sampling, full scope)")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    _log("SETUP", f"tokenizer={type(tokenizer).__name__} vocab_size={tokenizer.vocab_size}")

    # -------------------------------------------------------------
    # Phase 1: LLM scope -- every id referenced anywhere in
    # matches_llm.parquet (11,187,780 pairs, no sampling).
    # -------------------------------------------------------------
    matches_llm = (
        pl.scan_parquet(matches_llm_path)
        .filter(pl.col("id1").is_not_null() & pl.col("id2").is_not_null())
        .select(["id1", "id2"])
        .collect(engine="streaming")
    )
    n_pairs = matches_llm.height
    _log("DATA", f"matches_llm.parquet real row count: {n_pairs:,}")

    needed_ids = pd.unique(np.concatenate([
        matches_llm.get_column("id1").to_numpy(), matches_llm.get_column("id2").to_numpy()
    ]))
    del matches_llm
    gc.collect()
    _log("DATA", f"unique LLM-scope item ids needed: {len(needed_ids):,}")

    needed_ids_lazy = pl.DataFrame({"id": pd.Series(needed_ids, dtype="int64")}).lazy()
    llm_items = build_item_texts(
        pl.scan_parquet(items_path).join(needed_ids_lazy, on="id", how="semi")
    )
    _log("DATA", f"LLM-scope items resolved from catalog: {llm_items.height:,} / {len(needed_ids):,} needed")
    if llm_items.height < len(needed_ids):
        found_ids = set(llm_items.get_column("id").to_list())
        missing = [i for i in needed_ids.tolist() if i not in found_ids][:10]
        _log("WARN", f"{len(needed_ids) - llm_items.height:,} ids not found in items.parquet, e.g. {missing}")
    del needed_ids, needed_ids_lazy
    gc.collect()
    _malloc_trim()
    _log_real_ram()

    tokenize_scope(llm_items, tokenizer, os.path.join(OUTPUT_DIR, "llm_scope"), {
        "source": "items.parquet (full catalog) semi-joined to EVERY id referenced by "
                  "matches_llm.parquet (no sampling)",
        "n_pairs_source": int(n_pairs),
    })
    del llm_items
    gc.collect()
    _malloc_trim()
    _log_real_ram()

    # -------------------------------------------------------------
    # Phase 2: human scope -- items_human.parquet (711,304 items), same
    # 256-length + prefix convention, so the training kernel can mix human
    # + LLM data at the new PAIR_MAX_LENGTH=256 without re-tokenizing
    # anything itself (fresh LoRA on TPU should not repeat the ~9.5h/12h
    # per-chunk-rejoin bottleneck this project already hit once).
    # -------------------------------------------------------------
    human_items = build_item_texts(pl.scan_parquet(items_human_path))
    _log("DATA", f"human-scope items: {human_items.height:,}")
    tokenize_scope(human_items, tokenizer, os.path.join(OUTPUT_DIR, "human_scope"), {
        "source": "items_human.parquet, full scope",
    })
    del human_items
    gc.collect()
    _malloc_trim()

    _log("DONE", f"all scopes tokenized -> {OUTPUT_DIR}, total {_elapsed():.0f}s")


if __name__ == "__main__":
    main()
