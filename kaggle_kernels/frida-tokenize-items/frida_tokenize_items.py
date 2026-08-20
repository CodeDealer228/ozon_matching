"""
Pre-tokenize every unique item referenced by matches.parquet (human-labeled
pairs) and, separately, every unique item referenced by matches_llm.parquet
(LLM-labeled pairs), using ai-forever/FRIDA's tokenizer with the
"paraphrase: " prefix -- FRIDA's own model card names deduplication
explicitly as a paraphrase-prefix use case ("paraphrase prefix is for
symmetric paraphrasing related tasks (STS, paraphrase mining,
deduplication)"), which matches this task exactly (symmetric: does item A
duplicate item B, order doesn't matter).

Publishes two separate Kaggle datasets (per the human/LLM split requested)
so a from-scratch FRIDA bi-encoder run never needs to re-tokenize raw item
text again -- just look up (input_ids, attention_mask) by item id.

Why per-ITEM, not per-pair (unlike the MiniLM cross-encoder's
tokenize_to_memmap in kaggle_kernels/lora-llm-continue-train/): FRIDA is a
bi-encoder (confirmed via its model card -- T5-encoder-based, CLS pooling,
no text_pair/cross-attention mechanism), so unlike a cross-encoder there is
no joint pair-truncation step to replicate at all. Each item is tokenized
independently, once, to a fixed max_length -- full stop. A pair's model
input at train/inference time becomes a pure lookup-and-concatenate (or
lookup-embed-and-combine) operation with no tokenizer call and no per-chunk
rejoin against the items table -- the exact inefficiency that made the
MiniLM pipeline's tokenization take 9.5h of a 12h Kaggle session (see
CLAUDE.md "Full 11.48M-row run").

max_length=256 chosen empirically, not FRIDA's 512 hard cap: a 20K-item
local sample tokenized with this exact text formula + prefix came back
mean=126.6, p99=202.0, p99.9=223.0, true max=272 tokens over the sample --
256 covers ~99.99% of items with zero truncation while using half the
storage/RAM of the 512 cap.

Runs on a Kaggle TPU session (enable_tpu, not enable_gpu -- no TPU compute
is actually used, this is pure CPU tokenization) specifically for the
larger RAM allocation TPU sessions get relative to GPU sessions, and
because TPU quota (20h/week) is entirely separate from GPU quota
(30h/week), so this doesn't eat into the GPU budget the training runs need.
"""
import os
import sys
import gc
import ctypes
import json
import subprocess

import numpy as np
import pandas as pd
import polars as pl

from tqdm.auto import tqdm
from transformers import AutoTokenizer

ITEMS_PATH = "/kaggle/input/datasets/maxamsecret/ozon-ecup-2026-matching/items.parquet"
ITEMS_HUMAN_PATH = "/kaggle/input/datasets/maxamsecret/ozon-ecup-2026-matching/items_human.parquet"
MATCHES_PATH = "/kaggle/input/datasets/maxamsecret/ozon-ecup-2026-matching/matches.parquet"
MATCHES_LLM_PATH = "/kaggle/input/datasets/maxamsecret/ozon-ecup-2026-matching/matches_llm.parquet"

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

SCOPES = {
    "human": {
        "dataset_id": "maxamsecret/frida-item-tokens-human",
        "dataset_title": "FRIDA item tokens -- human-labeled scope",
    },
    "llm": {
        "dataset_id": "maxamsecret/frida-item-tokens-llm",
        "dataset_title": "FRIDA item tokens -- LLM-labeled scope",
    },
}


# =====================================================================
# Utilities (ported from kaggle_kernels/lora-llm-continue-train/ -- proven
# on real multi-hour, multi-million-row Kaggle runs; see CLAUDE.md for the
# incidents that shaped each of these)
# =====================================================================

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


def _log_cgroup_limit():
    for path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            with open(path) as f:
                raw = f.read().strip()
            if raw.isdigit():
                print(f"[MEM] cgroup limit ({path}): {int(raw) / 1e9:.2f}GB")
            else:
                print(f"[MEM] cgroup limit ({path}): {raw}")
            return
        except Exception:
            continue
    print("[MEM] cgroup limit: unavailable")


def _require_disk_space(path, required_bytes, label, margin=1.35):
    os.makedirs(path, exist_ok=True)
    import shutil
    free = shutil.disk_usage(path).free
    need = int(required_bytes * margin)
    print(f"[DISK] {label}: free={free / 1e9:.2f}GB required(+35% margin)={need / 1e9:.2f}GB")
    if free < need:
        raise RuntimeError(f"[DISK] insufficient space for {label}: need {need / 1e9:.2f}GB, have {free / 1e9:.2f}GB")


# =====================================================================
# Per-item tokenization -> memmap
# =====================================================================

def tokenize_items_to_memmap(items_pl, tokenizer, out_dir, max_length=MAX_LENGTH):
    """items_pl: polars DataFrame with columns [id, text] (text WITHOUT the
    FRIDA prefix yet -- added here). Writes ids.bin (int64), input_ids.bin
    (int32, n x max_length), attention_mask.bin (int8, n x max_length)."""
    n = items_pl.height
    ids_path = os.path.join(out_dir, "ids.bin")
    input_ids_path = os.path.join(out_dir, "input_ids.bin")
    attn_path = os.path.join(out_dir, "attention_mask.bin")

    bytes_per_row = 8 + max_length * 4 + max_length * 1
    _require_disk_space(out_dir, n * bytes_per_row, "item token cache")

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


def build_item_texts(items_lazy_or_df, needed_ids_lazy=None, streaming=False):
    """Shared truncation/formatting formula, byte-matching CLAUDE.md's
    'Text preprocessing'. If needed_ids_lazy is given, semi-joins first
    (for the LLM scope's ~13M-row full catalog scan); items_human.parquet
    is already exactly the needed set for the human scope, so no filter
    needed there."""
    lf = items_lazy_or_df
    if needed_ids_lazy is not None:
        lf = lf.join(needed_ids_lazy, on="id", how="semi")
    lf = (
        lf.select(["id", "name", "category", "attributes"])
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
    )
    return lf.collect(engine="streaming") if streaming else lf.collect()


def publish_dataset(stage_dir, dataset_id, dataset_title, extra_meta):
    with open(os.path.join(stage_dir, "cache_meta.json"), "w") as f:
        json.dump(extra_meta, f, indent=2)
    with open(os.path.join(stage_dir, "dataset-metadata.json"), "w") as f:
        json.dump({"id": dataset_id, "title": dataset_title}, f)
    print(f"[PUBLISH] Creating dataset {dataset_id} from {stage_dir} ...")
    subprocess.check_call(["kaggle", "datasets", "create", "-p", stage_dir, "-t"])
    print(f"[DONE] Published at https://www.kaggle.com/datasets/{dataset_id}")


def main():
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "sentencepiece"])

    if os.path.isdir("/kaggle/input"):
        for _root, _dirs, _files in os.walk("/kaggle/input"):
            depth = _root[len("/kaggle/input"):].count(os.sep)
            if depth <= 3:
                print(f"[DEBUG] {_root} -> dirs={_dirs} files={_files}")

    _log_cgroup_limit()
    _log_memory("start")

    print(f"[TOKENIZER] Loading {FRIDA_MODEL_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(FRIDA_MODEL_NAME)
    print(f"[TOKENIZER] vocab_size={tokenizer.vocab_size} prefix={FRIDA_PREFIX!r} max_length={MAX_LENGTH}")

    # -------------------------------------------------------------
    # Human scope: items_human.parquet is ALREADY exactly the item set
    # matches.parquet references (confirmed in CLAUDE.md) -- no filter
    # needed, just load, format, tokenize.
    # -------------------------------------------------------------
    print("\n" + "=" * 90)
    print("[SCOPE] human")
    print("=" * 90)
    human_items = build_item_texts(pl.scan_parquet(ITEMS_HUMAN_PATH))
    print(f"[DATA] human scope items: {human_items.height:,}")
    _log_memory("after human items load")

    human_out = os.path.join(WORKING_DIR, "frida_item_tokens_human")
    os.makedirs(human_out, exist_ok=True)
    human_paths = tokenize_items_to_memmap(human_items, tokenizer, human_out)
    n_human_items = human_items.height
    del human_items
    gc.collect()
    _malloc_trim()
    _log_memory("after human tokenize")

    publish_dataset(
        human_out, SCOPES["human"]["dataset_id"], SCOPES["human"]["dataset_title"],
        extra_meta={
            "n_items": n_human_items, "max_length": MAX_LENGTH, "prefix": FRIDA_PREFIX,
            "model": FRIDA_MODEL_NAME, "source": "items_human.parquet (== matches.parquet item scope)",
        },
    )

    # -------------------------------------------------------------
    # LLM scope: full items.parquet (~13.4M rows), semi-joined to the ids
    # matches_llm.parquet actually references (~12.38M unique ids).
    # -------------------------------------------------------------
    print("\n" + "=" * 90)
    print("[SCOPE] llm")
    print("=" * 90)
    print("[DATA] Loading matches_llm.parquet for needed ids ...")
    llm_pairs = pl.read_parquet(MATCHES_LLM_PATH).select(["id1", "id2"])
    needed_ids = pd.unique(
        pd.concat([llm_pairs["id1"].to_pandas(), llm_pairs["id2"].to_pandas()])
    )
    del llm_pairs
    gc.collect()
    needed_ids_lazy = pl.DataFrame({"id": pd.Series(needed_ids, dtype="int64")}).lazy()
    print(f"[DATA] llm scope unique item ids: {len(needed_ids):,}")

    llm_items = build_item_texts(pl.scan_parquet(ITEMS_PATH), needed_ids_lazy=needed_ids_lazy, streaming=True)
    print(f"[DATA] llm scope items loaded: {llm_items.height:,}")
    del needed_ids, needed_ids_lazy
    gc.collect()
    _malloc_trim()
    _log_memory("after llm items load")

    llm_out = os.path.join(WORKING_DIR, "frida_item_tokens_llm")
    os.makedirs(llm_out, exist_ok=True)
    llm_paths = tokenize_items_to_memmap(llm_items, tokenizer, llm_out)
    n_llm_items = llm_items.height
    del llm_items
    gc.collect()
    _malloc_trim()
    _log_memory("after llm tokenize")

    publish_dataset(
        llm_out, SCOPES["llm"]["dataset_id"], SCOPES["llm"]["dataset_title"],
        extra_meta={
            "n_items": n_llm_items, "max_length": MAX_LENGTH, "prefix": FRIDA_PREFIX,
            "model": FRIDA_MODEL_NAME, "source": "items.parquet semi-joined to matches_llm.parquet ids",
        },
    )

    print("\n[DONE] Both scopes tokenized and published.")


if __name__ == "__main__":
    main()
