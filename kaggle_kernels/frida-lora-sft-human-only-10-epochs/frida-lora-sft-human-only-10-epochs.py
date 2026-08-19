"""
LoRA SFT of ai-forever/FRIDA (T5EncoderModel, 823M) as a cross-encoder, on
the human-labeled train split ONLY (matches.parquet, 292,523 rows) -- see
CLAUDE.md / the giga3b-cross-encoder-probe memory for why: FRIDA's forward
pass is ~25x MiniLM's (823M vs 33M params), so the two-stage recipe that
worked for MiniLM (human SFT, then continue on 11.19M matches_llm rows) does
not fit a 30h/week Kaggle quota for a model this size -- even MiniLM's own
continue-train run didn't finish one epoch of 11.48M rows in a 12h session.
Human-only first: cheaper, reliable labels, and tells us whether LoRA-tuning
FRIDA is worth the quota before ever touching matches_llm.

Architecture: item-level pre-tokenization + at-train-time concatenation, NOT
the pair-level tokenize_to_memmap() approach kaggle_kernels/
lora-llm-continue-train used for MiniLM. That approach's per-chunk re-join of
the pairs list against the items table (rebuilt from scratch every chunk) is
the documented ~9.5h-of-12h bottleneck in this repo's own history. This
dataset has ~1.007 items/pair (CLAUDE.md's bi-encoder finding: 145,238 unique
items / ~146,262 references across 73,131 val pairs) -- so item-level
caching buys almost nothing from item REUSE, but it still sidesteps the
expensive repeated pair<->items join.

The item-level cache itself is NOT built by this script -- it reads a
pre-built one, `maxamsecret/frida-item-tokens-human`: `"paraphrase: " +
item_text` tokenized WITH special tokens (bos ... eos), padded/truncated to
256, for all 711,304 unique items referenced anywhere in matches.parquet
(train+val combined). Verified directly against the real .bin files (not
assumed from a README): every row starts with token id 1 and its last
non-padded token is id 2, confirming bos/eos are already baked in per item.
So a pair's input_ids is just `item1_real_tokens ++ item2_real_tokens`
(each already a complete bos...eos sequence) -- no separate bos/eos handling
needed at concatenation time, unlike an earlier draft of this script that
assumed a bos/eos-free per-item cache. Structurally up to 2x256=512 tokens
(FRIDA's own native context limit) would fit; PAIR_MAX_LENGTH is set to 128
instead -- a deliberate cost/quota-over-completeness choice, NOT the
data-driven near-lossless budget (that's 384, measured against all real
matches.parquet pairs -- see PAIR_MAX_LENGTH's own comment and this kernel's
README "Token-budget sizing" for the full trade-off, including why Category
gets excised from every item regardless of budget).

Prefix: "paraphrase: " on EVERY item (not once on the pair) -- FRIDA's own
README calls this prefix out by name for "deduplication" tasks, and its
usage pattern prefixes each independently-encoded text, which this keeps
even though the two are then concatenated into one cross-encoder sequence.

LoRA target_modules verified against the REAL loaded module tree (not
guessed): T5Attention uses q/k/v/o (not BERT's query/key/value), and FRIDA's
gated-gelu DenseReluDense FFN uses wi_0/wi_1/wo. "For all weights" (this
script's brief) means all seven of those per encoder block, not just
attention. Confirmed locally: get_peft_model reports 11,206,656 trainable
params (1.34% of 834,607,872), and a real forward+backward pass on CPU
produced nonzero gradients on 336 LoRA parameter tensors -- not a silent
no-op.

Optimization schedule, per instruction: peak LR=3e-4 (not the 2e-4 that
worked for MiniLM's continue-train -- that number is specific to RESUMING an
already-good checkpoint, where a previous test at higher LR (4e-4) measurably
regressed it; this is a FRESH LoRA init with nothing to wreck, so that
history doesn't directly transfer either way), 10% linear warmup, then a
cosine decay back toward 0 over the remaining steps -- gradual by
construction, not a step schedule.

bf16, NOT fp16 -- reversed from an earlier version of this script. A real
run at fp16+LR=3e-4 produced non-finite logits within the first ~7 steps
(GradScaler's own "scheduler.step() before optimizer.step()" warning fired
that early -- a tell that it was already silently skipping overflowed
updates before warmup even ramped up), and NaN'd out at the first checkpoint
(step 229/11430). This repo's own earlier finding was "prefer fp16 over bf16
on T4" (CLAUDE.md / giga3b probe: torch.cuda.is_bf16_supported() lies on
Turing, real bf16 tensor cores don't exist there, so bf16 runs emulated and
~5x slower) -- still true, and accepted here anyway: a slow run that doesn't
NaN beats a fast one that does. bf16's wider exponent range (matches fp32,
unlike fp16's narrow one) removes the specific failure mode observed, and
needs no GradScaler (no loss-scaling required when the exponent range
already covers fp32's). NaN/Inf are still checked for explicitly every step
regardless (see the training loop) -- bf16 makes overflow far less likely,
not impossible.

Structured as importable functions + main() guarded by __name__ == "__main__",
so the collate/concat mechanics and model wiring can be (and were) smoke-
tested locally against real data/tokenizer without spending Kaggle GPU quota
-- see this kernel's README for exactly what was checked.
"""

import os
import sys
import gc
import json
import math
import shutil
import hashlib
import subprocess

import numpy as np
import pandas as pd
import polars as pl

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from tqdm.auto import tqdm

from transformers import AutoTokenizer, T5EncoderModel, get_scheduler, set_seed
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score

try:
    import peft.import_utils
    import peft.tuners.lora.torchao

    peft.import_utils.is_torchao_available = lambda: False
    peft.tuners.lora.torchao.is_torchao_available = lambda: False
    print("[INFO] torchao disabled for PEFT")
except Exception as e:
    print(f"[WARN] torchao patch failed (likely fine): {e}")

from peft import LoraConfig, get_peft_model


# =====================================================================
# CONFIG
# =====================================================================

SEED = 42
MODEL_NAME = "ai-forever/FRIDA"

ITEMS_PATH = "/kaggle/input/datasets/maxamsecret/ozon-ecup-2026-matching/items.parquet"
MATCHES_PATH = "/kaggle/input/datasets/maxamsecret/ozon-ecup-2026-matching/matches.parquet"
# Pre-built item-level token cache (see module docstring for exactly how it
# was verified). Mount path searched at runtime, not hardcoded -- this
# repo's own history (lora-llm-continue-train's _find_pretokenized_cache_dir)
# shows Kaggle dataset mount paths aren't 100% predictable.
ITEM_CACHE_HINT = "/kaggle/input/datasets/maxamsecret/frida-item-tokens-human"
ITEM_CACHE_N_ITEMS = 711_304    # from the published cache_meta.json, cross-
                                 # checked against file sizes below at runtime
ITEM_MAX_LENGTH = 256           # from the published cache_meta.json

PAIR_MAX_LENGTH = 128                   # Deliberate choice for cost/quota over
                                         # completeness -- NOT the data-driven
                                         # near-lossless budget (that's 384:
                                         # p99 of real combined pair length,
                                         # only 0.97% truncated -- see
                                         # kaggle_kernels/frida-lora-sft README
                                         # "Token-budget sizing"). At 128,
                                         # ~94% of pairs get truncated, ~3x
                                         # cheaper per step than 384. Category
                                         # excision (see allocate_pair_budget /
                                         # build_compact_ids) matters far more
                                         # here than at 384: without it 0.44%
                                         # of items never reach ANY Attributes
                                         # content under this budget; with it,
                                         # 0.09% -- still not zero, but close.

MAX_CATEGORY_CHARS = 64   # only category is read from items.parquet now (the
                           # cache already carries name/attributes as tokens)

PREFIX = "paraphrase: "   # matches the cache's own cache_meta.json["prefix"]
                           # -- documented here for readability, not re-applied

EPOCHS = 1   # was 10 -- a real run at fp16/LR=3e-4 NaN'd at the very first
             # checkpoint (step 229/11430, before warmup even finished). This
             # run validates the bf16 + NaN-guard fix (see LR/autocast
             # comments below) on a single epoch before committing 10x the
             # quota to it again.

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.10
# Real module names verified via named_modules() on the loaded checkpoint
# (see this kernel's README) -- T5Attention's q/k/v/o, NOT BERT's
# query/key/value; FRIDA's gated-gelu FFN's wi_0/wi_1/wo. "All weights":
# every Linear in both the attention and FFN sublayers of every encoder block.
LORA_TARGET_MODULES = ["q", "k", "v", "o", "wi_0", "wi_1", "wo"]

LR = 3e-4                          # peak LR, per instruction -- see module
                                    # docstring for why MiniLM's validated
                                    # 2e-4 doesn't directly transfer here
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.10                 # linear warmup to peak, then cosine decay
N_CHECKPOINTS_PER_EPOCH = 5         # finer-grained than continue-train's
                                     # implicit ~3/epoch -- explicit ask for
                                     # good Kaggle-log visibility into train/
                                     # val loss+metric as the run progresses

MICRO_BATCH_SIZE = 16               # starting point for the VRAM probe, NOT
                                     # the final value -- this repo has zero
                                     # real training-memory data for FRIDA at
                                     # seq=512 (the giga3b_probe OOMs were
                                     # inference-only bnb numbers for a
                                     # different model, not informative for a
                                     # plain fp16 backward pass here).
EVAL_MICRO_BATCH_SIZE = 32
TOTAL_BATCH_SIZE = 256              # effective batch after grad accumulation

TARGET_VRAM_GB = 14.0
SAFETY_MARGIN_GB = 1.5
PROBE_CAP = 512
PROBE_ROUND_TO = 4

ENABLE_DATASET_PUSH = False
LORA_DATASET_ID = "maxamsecret/ozon-lora-frida-finetuned"
LORA_DATASET_NESTED_PATH = "lora_frida_human_sft/best"

WORKING_DIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
OUTPUT_DIR = os.path.join(WORKING_DIR, "lora_frida_sft")
MLFLOW_TRACKING_DIR = os.path.join(WORKING_DIR, "experiments", "mlruns")


# =====================================================================
# SMALL UTILITIES
# =====================================================================

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
        print(f"[MEM] {stage}: rss={rss_gb:.2f}GB (via /proc, psutil unavailable)")


def compute_metrics(y_true, y_prob, categories=None):
    """Competition metric: average_precision_score per category, plain mean
    (macro) across categories -- see CLAUDE.md, not sklearn's `auc`."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    if len(np.unique(y_true)) < 2:
        return float("nan"), float("nan"), 0

    pos_pr_auc = float(average_precision_score(y_true, y_prob))

    if categories is None:
        return pos_pr_auc, pos_pr_auc, 0

    categories = np.asarray(categories)
    per_category_aps = []
    for cat in np.unique(categories):
        mask = categories == cat
        y_t, y_p = y_true[mask], y_prob[mask]
        if len(np.unique(y_t)) < 2:
            continue
        per_category_aps.append(average_precision_score(y_t, y_p))

    if not per_category_aps:
        return float("nan"), pos_pr_auc, 0

    return float(np.mean(per_category_aps)), pos_pr_auc, len(per_category_aps)


# =====================================================================
# ITEM-LEVEL TOKEN CACHE -- reads the pre-built maxamsecret/frida-item-
# tokens-human dataset, does not tokenize anything itself.
# =====================================================================

# Every cached item is "paraphrase: Name: {name} Category: {category}
# Attributes: {attributes}". Category is proven zero-signal for this
# pairwise task (CLAUDE.md: identical between both items of every pair,
# verified 100%) and item-length-analysis shows its 64-char budget was never
# once needed (real max ever seen: 23 chars) -- so it's excised at collate
# time regardless of PAIR_MAX_LENGTH, freeing that space for Attributes
# (item-length-analysis: 47% of items lose attribute content at the old
# 384-char budget -- this is where the real signal lives, CLAUDE.md's own
# per-category analysis).
#
# The " Category" and " Attributes" BPE tokens are stable regardless of
# surrounding content (verified empirically: tokenizing " Category: y"/
# " Attributes: {" after several different preceding names all produced the
# same token ids), so a fixed-pattern search locates both labels in every
# item reliably -- confirmed 711,304/711,304 (100%) on the real cache.
CATEGORY_LABEL_TOKEN = 55917    # " Category" (+ colon=30 immediately after)
ATTRIBUTES_LABEL_TOKEN = 91837  # " Attributes" (+ colon=30 immediately after)
LABEL_COLON_TOKEN = 30


def _find_token_pattern_starts(arr, tok_a, tok_b):
    """First index i per row where arr[row,i]==tok_a and arr[row,i+1]==tok_b,
    or -1 if not found. Vectorized (no Python loop over rows) -- took 1.5s
    over the full 711,304-item cache when this was verified locally."""
    mask = (arr[:, :-1] == tok_a) & (arr[:, 1:] == tok_b)
    rows, cols = np.where(mask)
    uniq_rows, first_idx = np.unique(rows, return_index=True)
    starts = np.full(arr.shape[0], -1, dtype=np.int32)
    starts[uniq_rows] = cols[first_idx]
    return starts


def _find_item_cache_dir(hint):
    if os.path.isdir(hint) and os.path.exists(os.path.join(hint, "ids.bin")):
        return hint
    if os.path.isdir("/kaggle/input"):
        for root, _dirs, files in os.walk("/kaggle/input"):
            if "ids.bin" in files:
                print(f"[CACHE] hint path {hint} didn't have the cache directly -- "
                      f"found it via search at {root} instead")
                return root
    return hint


def load_item_cache(cache_dir):
    """Returns (item_input_ids memmap, item_lengths, category_label_start,
    attr_label_start, compact_lengths, id_to_row). compact_lengths is
    item_lengths with the " Category: {value}" span excised (see
    allocate_pair_budget) -- the length actually used for truncation
    decisions. Validates real file sizes against cache_meta.json instead of
    trusting the hardcoded ITEM_CACHE_N_ITEMS blindly (that constant is
    cross-checked here, not assumed)."""
    meta_path = os.path.join(cache_dir, "cache_meta.json")
    with open(meta_path) as f:
        meta = json.load(f)
    n_items = meta["n_items"]
    max_length = meta["max_length"]
    print(f"[CACHE] cache_meta.json: {meta}")
    if max_length != ITEM_MAX_LENGTH:
        raise RuntimeError(f"cache max_length={max_length} != this script's ITEM_MAX_LENGTH={ITEM_MAX_LENGTH}")

    ids_path = os.path.join(cache_dir, "ids.bin")
    input_ids_path = os.path.join(cache_dir, "input_ids.bin")
    attn_path = os.path.join(cache_dir, "attention_mask.bin")

    expected_ids_bytes = n_items * np.dtype(np.int64).itemsize
    expected_input_ids_bytes = n_items * max_length * np.dtype(np.int32).itemsize
    expected_attn_bytes = n_items * max_length * np.dtype(np.int8).itemsize
    actual = {
        "ids.bin": (os.path.getsize(ids_path), expected_ids_bytes),
        "input_ids.bin": (os.path.getsize(input_ids_path), expected_input_ids_bytes),
        "attention_mask.bin": (os.path.getsize(attn_path), expected_attn_bytes),
    }
    for fname, (got, want) in actual.items():
        if got != want:
            raise RuntimeError(f"[CACHE] {fname} size mismatch: got {got:,} bytes, expected {want:,} "
                                f"(n_items={n_items:,}, max_length={max_length})")
    print(f"[CACHE] file sizes verified against cache_meta.json for all 3 .bin files")

    ids = np.memmap(ids_path, dtype=np.int64, mode="r", shape=(n_items,))
    item_input_ids = np.memmap(input_ids_path, dtype=np.int32, mode="r", shape=(n_items, max_length))
    item_attention_mask = np.memmap(attn_path, dtype=np.int8, mode="r", shape=(n_items, max_length))

    print("[CACHE] building id -> row index map ...")
    id_to_row = {int(item_id): i for i, item_id in enumerate(ids)}
    assert len(id_to_row) == n_items, f"duplicate item ids in cache: {n_items} rows but {len(id_to_row)} unique ids"

    print("[CACHE] precomputing per-item real lengths (one-time vectorized sum, avoids "
          "re-summing attention_mask on every collate_fn call across 10 epochs) ...")
    item_lengths = np.asarray(item_attention_mask).sum(axis=1).astype(np.int32)
    print(f"[CACHE] loaded {n_items:,} items, real length mean={item_lengths.mean():.1f} "
          f"p50={np.percentile(item_lengths, 50):.0f} p90={np.percentile(item_lengths, 90):.0f} "
          f"max={item_lengths.max()} (cap={max_length})")

    # Sanity spot-check (cheap, catches a wrong-dtype/shape mount early rather
    # than deep into training): every real item cache row must start with
    # bos (1) and its last non-padded token must be eos (2) -- verified
    # against the real dataset by hand before writing this script; re-checked
    # here on every load in case the dataset ever gets rebuilt differently.
    sample_rows = np.random.RandomState(SEED).choice(n_items, size=min(50, n_items), replace=False)
    for r in sample_rows:
        length = int(item_lengths[r])
        assert item_input_ids[r, 0] == 1, f"row {r}: expected bos=1 at position 0, got {item_input_ids[r, 0]}"
        assert item_input_ids[r, length - 1] == 2, f"row {r}: expected eos=2 at position {length - 1}, got {item_input_ids[r, length - 1]}"
    print(f"[CACHE] bos/eos sanity check passed on {len(sample_rows)} random rows")

    print("[CACHE] locating Category:/Attributes: label positions (for category excision -- "
          "see module docstring / allocate_pair_budget) ...")
    arr = np.asarray(item_input_ids)
    category_label_start = _find_token_pattern_starts(arr, CATEGORY_LABEL_TOKEN, LABEL_COLON_TOKEN)
    attr_label_start = _find_token_pattern_starts(arr, ATTRIBUTES_LABEL_TOKEN, LABEL_COLON_TOKEN)
    both_found = (category_label_start >= 0) & (attr_label_start >= 0)
    n_missing = n_items - int(both_found.sum())
    if n_missing > 0:
        print(f"[CACHE] WARNING: {n_missing:,}/{n_items:,} items missing a locatable Category:/"
              f"Attributes: label (unexpected -- was 0/711,304 when this was verified). "
              f"Falling back to no-excision (full item kept) for those.")
    category_span = np.where(both_found, attr_label_start - category_label_start, 0).astype(np.int32)
    compact_lengths = item_lengths - category_span
    print(f"[CACHE] category span excised: mean={category_span[both_found].mean():.1f} tokens/item "
          f"(items-length-analysis: real category content never exceeds 23 chars, ~0 discriminative "
          f"value for this pairwise task -- see CLAUDE.md) | compact length: mean={compact_lengths.mean():.1f} "
          f"p50={np.percentile(compact_lengths, 50):.0f} (was {item_lengths.mean():.1f}/{np.percentile(item_lengths, 50):.0f})")

    return item_input_ids, item_lengths, category_label_start, attr_label_start, compact_lengths, id_to_row


class PairIndexDataset(Dataset):
    """Rows are (row1, row2, label, category) -- already resolved from item
    id -> memmap row index, so __getitem__ and the collate_fn never touch a
    Python dict per-sample (only numpy indexing)."""

    def __init__(self, row1, row2, labels, categories):
        self.row1 = row1
        self.row2 = row2
        self.labels = labels
        self.categories = categories

    def __len__(self):
        return len(self.row1)

    def __getitem__(self, idx):
        return self.row1[idx], self.row2[idx], self.labels[idx], self.categories[idx]


def allocate_pair_budget(len1, len2, budget):
    """Fair-share truncation policy over COMPACT lengths (category already
    excised -- see build_compact_ids). Real measurements against all
    365,654 matches.parquet pairs: at budget=384 only 0.97% of pairs need
    ANY truncation at all (losing an average of 16.7 tokens); at budget=128
    (a much tighter, "экономично" option) 94% do, but with category excised
    only 0.09% of items ever fail to reach ANY Attributes content (was 0.44%
    without excision) -- see this kernel's README "Token-budget sizing" for
    the full breakdown at both budgets.

    Principle: never cut a side that already fits its equal half-share --
    only the excess of a side that's OVER half gets trimmed, and any slack
    left by a genuinely short side is handed to the other one. Symmetric (no
    built-in bias toward id1 or id2, appropriate for a task with no query/
    document asymmetry) and never wastes budget: a short side is never cut
    below what it actually needs just to enforce an arbitrary fixed 50/50
    split.
    """
    half = budget // 2
    if len1 <= half and len2 <= half:
        return len1, len2
    if len1 <= half:
        return len1, budget - len1
    if len2 <= half:
        return budget - len2, len2
    return half, budget - half


def build_compact_ids(item_input_ids_mm, row, item_length, category_start, attr_start):
    """Excise " Category: {value}" from one item's real (unpadded) tokens --
    proven zero-signal for this pairwise task (see module-level comment
    above CATEGORY_LABEL_TOKEN). item_length is the item's REAL length (bos
    through eos, before padding) -- NOT the compact length; the tail slice
    needs to know where the real content actually ends. Falls back to the
    untouched slice if this item's labels weren't both found (see
    load_item_cache's WARNING path; 0/711,304 in the verified cache, kept as
    a defensive fallback only). Always returns exactly compact_lengths[row]
    tokens (by construction: category_start + (item_length - attr_start),
    which is exactly how compact_lengths was computed)."""
    if category_start < 0 or attr_start < 0:
        return item_input_ids_mm[row, :item_length]
    head = item_input_ids_mm[row, :category_start]
    tail = item_input_ids_mm[row, attr_start:item_length]
    return np.concatenate((head, tail))


def make_pair_collate_fn(item_input_ids_mm, item_lengths, compact_lengths, category_label_start,
                          attr_label_start, pad_id, pair_max_length):
    """The "конкатенировать во время обучения" step: array slicing +
    concatenation of two already-tokenized (bos...eos-complete) items, no
    tokenizer call, no items-table join -- this is what replaces the
    expensive per-chunk pair<->items rejoin that dominated continue-train's
    tokenization time. Each item's Category span is excised first (see
    build_compact_ids), THEN the two compact sequences are fair-share-
    truncated (allocate_pair_budget) and concatenated."""

    def collate_fn(batch):
        row1, row2, labels, categories = zip(*batch)
        bsz = len(batch)
        input_ids = np.full((bsz, pair_max_length), pad_id, dtype=np.int64)
        attention_mask = np.zeros((bsz, pair_max_length), dtype=np.int64)

        for i in range(bsz):
            r1, r2 = row1[i], row2[i]

            compact1 = build_compact_ids(item_input_ids_mm, r1, int(item_lengths[r1]),
                                          int(category_label_start[r1]), int(attr_label_start[r1]))
            compact2 = build_compact_ids(item_input_ids_mm, r2, int(item_lengths[r2]),
                                          int(category_label_start[r2]), int(attr_label_start[r2]))
            cl1, cl2 = len(compact1), len(compact2)

            if cl1 + cl2 <= pair_max_length:
                keep1, keep2 = cl1, cl2
            else:
                keep1, keep2 = allocate_pair_budget(cl1, cl2, pair_max_length)

            # Note: when keep1 < cl1 (or keep2 < cl2), the kept slice loses
            # its trailing eos (each item's real content ends in eos) --
            # acceptable for the minority of pairs this applies to (0.97% at
            # budget=384, ~94% at budget=128 -- see allocate_pair_budget);
            # item2's own bos still marks the boundary either way.
            combined = np.concatenate((compact1[:keep1], compact2[:keep2]))

            n = len(combined)
            input_ids[i, :n] = combined
            attention_mask[i, :n] = 1

        return (
            torch.from_numpy(input_ids),
            torch.from_numpy(attention_mask),
            torch.tensor(labels, dtype=torch.float32),
            np.asarray(categories, dtype=np.int64),
        )

    return collate_fn


def make_pair_loader(pairs_pd, id_to_row, category_to_id, item_input_ids_mm, item_lengths, compact_lengths,
                      category_label_start, attr_label_start, pad_id, pair_max_length, batch_size, shuffle):
    row1_raw = pairs_pd["id1"].map(id_to_row)
    row2_raw = pairs_pd["id2"].map(id_to_row)
    missing = row1_raw.isna() | row2_raw.isna()
    if missing.any():
        raise RuntimeError(f"{int(missing.sum())} pairs reference an item id missing from the token cache")
    row1 = row1_raw.to_numpy().astype(np.int64)
    row2 = row2_raw.to_numpy().astype(np.int64)
    labels = pairs_pd["label"].to_numpy().astype(np.float32)
    categories = pairs_pd["category1"].map(category_to_id).fillna(category_to_id.get("UNKNOWN", -1)).to_numpy().astype(np.int64)

    dataset = PairIndexDataset(row1, row2, labels, categories)
    collate_fn = make_pair_collate_fn(item_input_ids_mm, item_lengths, compact_lengths,
                                       category_label_start, attr_label_start, pad_id, pair_max_length)
    drop_last = shuffle and (len(dataset) >= batch_size)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0,
        collate_fn=collate_fn, pin_memory=torch.cuda.is_available(), drop_last=drop_last,
    )
    return loader


# =====================================================================
# MODEL
# =====================================================================

class FridaCrossEncoderClassifier(nn.Module):
    """CLS (position-0) pooling -- FRIDA's own README recommendation
    ("It is recommended to use CLS pooling"), same convention this repo
    already uses for FRIDA in submits/frida_probe. No token_type_ids: T5 has
    no segment-embedding concept.

    autocast lives INSIDE forward(), not wrapped around calls to this module
    from the outside -- root-caused via a dedicated GPU diagnostic
    (kaggle_kernels/frida-nan-diagnose) after both fp16 and bf16 produced
    IDENTICAL (bit-for-bit) NaN patterns under nn.DataParallel, while a
    single-GPU bf16 run with no DataParallel was completely clean (same real
    batch, same weights, loss=0.6925 matching plain fp32 exactly). That
    identical-pattern tell is the signature of PyTorch's documented
    DataParallel+autocast pitfall: DataParallel runs each replica's forward()
    in its own thread (torch.nn.parallel.parallel_apply), and an autocast
    context entered in the calling thread does NOT propagate into those
    worker threads -- PyTorch's own docs prescribe entering autocast inside
    the model's forward() for exactly this reason. Wrapping it outside (the
    previous code, at all three call sites: train loop, evaluate(), and the
    VRAM probe) left replica threads silently running with stale/leaked
    autocast state instead of the requested dtype."""

    def __init__(self, encoder, head):
        super().__init__()
        self.encoder = encoder
        self.classifier = head

    def forward(self, input_ids, attention_mask):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            cls = out.last_hidden_state[:, 0, :]
            return self.classifier(cls).view(-1).float()


def build_fresh_model(device):
    print(f"[MODEL] Loading base encoder {MODEL_NAME} (T5EncoderModel explicitly -- "
          f"AutoModel would default to the full encoder-decoder T5Model and "
          f"randomly-init a decoder this checkpoint never shipped) ...")
    # attn_implementation="eager" -- forced, not left to transformers' default
    # (which auto-picks "sdpa" when available). Root-caused a 100%-of-steps
    # NaN-from-step-0 run this way: a local CPU repro of the exact same real
    # batch + real model was completely clean in fp32 AND under
    # torch.autocast(cpu, bfloat16) (finite logits, ~0.02 magnitude, sane BCE
    # loss) -- ruling out the data/collate and the bf16 dtype itself. The one
    # thing CPU can't reproduce is which SDPA backend CUDA dispatches to:
    # sdpa_attention_forward -> F.scaled_dot_product_attention picks a fused
    # flash/mem-efficient CUDA kernel at runtime, and those fused kernels have
    # documented bf16/fp16 NaN edge cases with padding masks -- more so on
    # older architectures like T4 (Turing, sm_75) than on Ampere+. This repo
    # already hit an SDPA fragility issue with FRIDA once before (see
    # submits/frida_probe's load-time TypeError/ValueError fallback). eager
    # uses the plain matmul+softmax reference path (what CPU's "math" backend
    # already does, hence why CPU never reproduced this) -- slower, but
    # numerically boring on purpose. Same TypeError/ValueError defensive
    # fallback as that earlier fix, in case this transformers version
    # doesn't accept the kwarg for T5EncoderModel.
    try:
        base_model = T5EncoderModel.from_pretrained(MODEL_NAME, attn_implementation="eager")
    except (TypeError, ValueError):
        base_model = T5EncoderModel.from_pretrained(MODEL_NAME)
    hidden_dim = base_model.config.d_model

    lora_config = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES, bias="none",
    )
    peft_model = get_peft_model(base_model, lora_config)
    peft_model.print_trainable_parameters()

    n_trainable_lora = sum(p.numel() for n, p in peft_model.named_parameters() if "lora_" in n and p.requires_grad)
    assert n_trainable_lora > 0, "LoRA adapter has zero trainable params -- target_modules likely matched nothing."
    print(f"[MODEL] trainable LoRA params: {n_trainable_lora:,}")

    classifier_head = nn.Linear(hidden_dim, 1)
    model = FridaCrossEncoderClassifier(peft_model, classifier_head).to(device)
    return model


# =====================================================================
# VRAM AUTO-PROBE (same philosophy as lora-llm-continue-train.py, no
# token_type_ids since T5 doesn't use them). See also frida_batch_probe.py,
# a standalone version of just this step for a cheap pre-flight check.
# =====================================================================

def _probe_peak_gb(batch_size, n_gpu, pair_max_length, probe_model, probe_loss_fn, trainable_params, device):
    dummy_input_ids = torch.randint(0, 1000, (batch_size, pair_max_length), dtype=torch.long, device=device)
    dummy_attention_mask = torch.ones((batch_size, pair_max_length), dtype=torch.long, device=device)
    dummy_labels = torch.randint(0, 2, (batch_size,), dtype=torch.float32, device=device)

    for p in trainable_params:
        p.grad = None
    for i in range(max(n_gpu, 1)):
        torch.cuda.reset_peak_memory_stats(i)

    # bf16 autocast happens INSIDE FridaCrossEncoderClassifier.forward() now
    # (see that class for why -- DataParallel+autocast pitfall), not wrapped
    # here.
    logits = probe_model(input_ids=dummy_input_ids, attention_mask=dummy_attention_mask)
    loss = probe_loss_fn(logits, dummy_labels)
    loss.backward()

    peak_bytes = max(torch.cuda.max_memory_allocated(i) for i in range(max(n_gpu, 1)))

    for p in trainable_params:
        p.grad = None
    del dummy_input_ids, dummy_attention_mask, dummy_labels, logits, loss
    torch.cuda.empty_cache()
    return peak_bytes / (1024 ** 3)


def _is_oom(exc):
    return isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()


def run_vram_probe(model, trainable_params, device, n_gpu, pair_max_length, start_batch):
    if not torch.cuda.is_available():
        print("[PROBE] No GPU available, skipping batch-size probe.")
        return start_batch

    print(f"[PROBE] Searching for max micro-batch under {TARGET_VRAM_GB:.1f}GB/GPU "
          f"(safety margin {SAFETY_MARGIN_GB:.1f}GB) ...")
    probe_loss_fn = torch.nn.BCEWithLogitsLoss()
    probe_threshold = TARGET_VRAM_GB - SAFETY_MARGIN_GB

    candidate = start_batch
    last_good, last_good_gb = None, None

    while candidate <= PROBE_CAP:
        try:
            peak_gb = _probe_peak_gb(candidate, n_gpu, pair_max_length, model, probe_loss_fn, trainable_params, device)
            print(f"[PROBE] batch={candidate:5d} -> peak={peak_gb:.2f}GB/GPU")
        except RuntimeError as e:
            if not _is_oom(e):
                raise
            torch.cuda.empty_cache()
            print(f"[PROBE] batch={candidate:5d} -> OOM")
            break

        last_good, last_good_gb = candidate, peak_gb
        if peak_gb > probe_threshold:
            break
        candidate *= 2

    if last_good is not None and last_good_gb is not None and last_good_gb <= probe_threshold and candidate <= PROBE_CAP:
        lo, hi = last_good, candidate
        while hi - lo > PROBE_ROUND_TO:
            mid = ((lo + hi) // 2 // PROBE_ROUND_TO) * PROBE_ROUND_TO
            if mid <= lo:
                break
            try:
                peak_gb = _probe_peak_gb(mid, n_gpu, pair_max_length, model, probe_loss_fn, trainable_params, device)
                print(f"[PROBE] batch={mid:5d} -> peak={peak_gb:.2f}GB/GPU")
                if peak_gb > probe_threshold:
                    hi = mid
                else:
                    lo, last_good, last_good_gb = mid, mid, peak_gb
            except RuntimeError as e:
                if not _is_oom(e):
                    raise
                torch.cuda.empty_cache()
                print(f"[PROBE] batch={mid:5d} -> OOM")
                hi = mid

    if last_good is None:
        print(f"[PROBE] First candidate ({start_batch}) already OOM'd -- keeping config value "
              f"and relying on GRAD_ACCUM; consider lowering PAIR_MAX_LENGTH by hand.")
        return start_batch

    discovered = max(PROBE_ROUND_TO, (last_good // PROBE_ROUND_TO) * PROBE_ROUND_TO)
    print(f"[PROBE] Selected MICRO_BATCH_SIZE={discovered} (was {start_batch}, measured peak={last_good_gb:.2f}GB/GPU)")
    return discovered


# =====================================================================
# EVAL
# =====================================================================

def evaluate(eval_model, loader, loss_fn, device):
    eval_model.eval()

    all_logits, all_labels, all_categories = [], [], []
    total_loss, n_batches = 0.0, 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Eval", dynamic_ncols=True, leave=False):
            input_ids, attention_mask, batch_labels, batch_categories = batch
            input_ids = input_ids.to(device, non_blocking=True)
            attention_mask = attention_mask.to(device, non_blocking=True)
            batch_labels_dev = batch_labels.to(device, non_blocking=True)

            # bf16 autocast happens INSIDE FridaCrossEncoderClassifier.forward()
            # now, not wrapped here (see that class for why).
            logits = eval_model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(logits, batch_labels_dev)
            total_loss += loss.item()
            n_batches += 1

            all_logits.append(logits.float().cpu())
            all_labels.append(batch_labels.cpu())
            all_categories.append(batch_categories)

    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    categories = np.concatenate(all_categories, axis=0)

    val_loss = total_loss / max(1, n_batches)
    probs = torch.sigmoid(logits).cpu().numpy()
    y_true = (labels.numpy() >= 0.5).astype(int)

    macro_pr_auc, pos_pr_auc, n_categories = compute_metrics(y_true, probs, categories)

    del logits, labels, probs, y_true, all_logits, all_labels, all_categories
    gc.collect()

    return {
        "val_loss": float(val_loss), "macro_pr_auc": float(macro_pr_auc),
        "pos_pr_auc": float(pos_pr_auc), "n_categories_evaluated": int(n_categories),
    }


# =====================================================================
# CRASH-RESILIENT CHECKPOINT PUSH (same pattern as lora-llm-continue-train.py)
# =====================================================================

def _bootstrap_kaggle_secrets():
    try:
        from kaggle_secrets import UserSecretsClient
        us = UserSecretsClient()
        os.environ["KAGGLE_USERNAME"] = us.get_secret("KAGGLE_USERNAME")
        os.environ["KAGGLE_KEY"] = us.get_secret("KAGGLE_KEY")
        print("[SECRETS] Kaggle API credentials loaded from kernel secrets.")
        return True
    except Exception as e:
        print(f"[WARN] Kaggle secrets unavailable, dataset-version push disabled this run: {e}")
        return False


def save_checkpoint(model, tokenizer, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    raw = model.module if isinstance(model, nn.DataParallel) else model
    raw.encoder.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    torch.save(raw.classifier.state_dict(), os.path.join(save_dir, "classifier.pt"))


# =====================================================================
# MAIN
# =====================================================================

def main():
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "peft", "mlflow"])
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    import mlflow

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1")
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    os.environ["OMP_NUM_THREADS"] = str(os.cpu_count() or 8)
    os.environ["MKL_NUM_THREADS"] = str(os.cpu_count() or 8)

    set_seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("=" * 90)
    print("CONFIG")
    print("=" * 90)
    print(f"model:              {MODEL_NAME}")
    print(f"item_max_length:    {ITEM_MAX_LENGTH}  (prefix={PREFIX!r}, from pre-built cache)")
    print(f"pair_max_length:    {PAIR_MAX_LENGTH}")
    print(f"epochs:             {EPOCHS}")
    print(f"lr:                 {LR}  (peak, 10% warmup then cosine decay)")
    print(f"lora r/alpha/drop:  {LORA_R}/{LORA_ALPHA}/{LORA_DROPOUT}")
    print(f"lora target_modules:{LORA_TARGET_MODULES}")
    print(f"total_batch_size:   {TOTAL_BATCH_SIZE}")
    print(f"micro_batch_size:   {MICRO_BATCH_SIZE} (probe start)")
    print(f"output_dir:         {OUTPUT_DIR}")
    print("=" * 90)

    if os.path.isdir("/kaggle/input"):
        for _root, _dirs, _files in os.walk("/kaggle/input"):
            depth = _root[len("/kaggle/input"):].count(os.sep)
            if depth <= 3:
                print(f"[DEBUG] {_root} -> dirs={_dirs} files={_files}")

    mlflow.set_tracking_uri(f"file:{MLFLOW_TRACKING_DIR}")
    mlflow.set_experiment("ozon-matching")
    has_kaggle_creds = _bootstrap_kaggle_secrets() if ENABLE_DATASET_PUSH else False

    _log_memory("start")

    print("[TOKENIZER] Loading tokenizer (for pad_token_id only -- item content is already tokenized) ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    pad_id = tokenizer.pad_token_id
    print(f"[TOKENIZER] pad={pad_id} (bos/eos already baked into the item cache, not re-added here)")

    # -------------------------------------------------------------
    # Canonical human-only split (byte-identical to the rest of this repo:
    # filter nulls -> label = target>=0.5 -> seed-42 stratified 80/20).
    # -------------------------------------------------------------
    print("[DATA] Loading matches.parquet for canonical split ...")
    matches_pl = pl.read_parquet(MATCHES_PATH)
    matches_pl = (
        matches_pl
        .filter(pl.col("id1").is_not_null() & pl.col("id2").is_not_null())
        .with_columns([
            pl.col("target").fill_null(0).cast(pl.Float32),
            (pl.col("target") >= 0.5).cast(pl.Int8).alias("label"),
        ])
        .select(["id1", "id2", "label"])
    )
    matches_pd = matches_pl.to_pandas()
    del matches_pl
    gc.collect()

    labels_all = matches_pd["label"].to_numpy().astype(np.int64)
    indices = np.arange(len(labels_all), dtype=np.int64)
    train_idx, val_idx = train_test_split(indices, test_size=0.20, random_state=SEED, stratify=labels_all)
    train_pd = matches_pd.iloc[train_idx][["id1", "id2", "label"]].reset_index(drop=True).copy()
    val_pd = matches_pd.iloc[val_idx][["id1", "id2", "label"]].reset_index(drop=True).copy()
    print(f"[DATA] train={len(train_pd):,} val={len(val_pd):,} (expect 292,523 / 73,131)")

    val_fingerprint = hashlib.sha256(
        ",".join(f"{a}-{b}" for a, b in sorted(zip(val_pd["id1"], val_pd["id2"]))).encode("utf-8")
    ).hexdigest()[:16]
    print(f"[DATA] val_fingerprint={val_fingerprint}")

    del matches_pd, labels_all, indices, train_idx, val_idx
    gc.collect()

    # -------------------------------------------------------------
    # Load the pre-built item token cache (no tokenization happens in this
    # script at all -- see module docstring).
    # -------------------------------------------------------------
    cache_dir = _find_item_cache_dir(ITEM_CACHE_HINT)
    print(f"[CACHE] loading item token cache from {cache_dir} ...")
    (item_input_ids_mm, item_lengths, category_label_start, attr_label_start,
     compact_lengths, id_to_row) = load_item_cache(cache_dir)
    _log_memory("after item cache load")

    # -------------------------------------------------------------
    # items.parquet -- ONLY for category (name/attributes are already tokens
    # in the cache, no text processing needed here at all).
    # -------------------------------------------------------------
    needed_ids = pd.unique(pd.concat([train_pd["id1"], train_pd["id2"], val_pd["id1"], val_pd["id2"]]))
    needed_ids_lazy = pl.DataFrame({"id": pd.Series(needed_ids, dtype="int64")}).lazy()
    print(f"[DATA] unique item ids needed: {len(needed_ids):,}")

    print("[DATA] Loading items.parquet for category only (semi-join, streaming) ...")
    items_cat = (
        pl.scan_parquet(ITEMS_PATH)
        .join(needed_ids_lazy, on="id", how="semi")
        .select(["id", "category"])
        .with_columns(pl.col("category").fill_null("UNKNOWN").cast(pl.Utf8).str.slice(0, MAX_CATEGORY_CHARS))
        .collect(engine="streaming")
    )
    print(f"[DATA] categories loaded for {items_cat.height:,} items")
    del needed_ids, needed_ids_lazy
    gc.collect()

    category_vocab = sorted(items_cat.get_column("category").unique().to_list()) + ["UNKNOWN"]
    category_to_id = {c: i for i, c in enumerate(category_vocab)}
    print(f"[DATA] category vocab size: {len(category_vocab)}")

    id_to_category = dict(zip(items_cat.get_column("id").to_list(), items_cat.get_column("category").to_list()))
    train_pd["category1"] = train_pd["id1"].map(id_to_category).fillna("UNKNOWN")
    val_pd["category1"] = val_pd["id1"].map(id_to_category).fillna("UNKNOWN")
    del items_cat, id_to_category
    gc.collect()
    _log_memory("after category load")

    micro_batch_size = MICRO_BATCH_SIZE
    eval_batch_size = EVAL_MICRO_BATCH_SIZE
    total_batch_size = TOTAL_BATCH_SIZE

    train_loader = make_pair_loader(
        train_pd, id_to_row, category_to_id, item_input_ids_mm, item_lengths, compact_lengths,
        category_label_start, attr_label_start, pad_id, PAIR_MAX_LENGTH, micro_batch_size, shuffle=True,
    )
    val_loader = make_pair_loader(
        val_pd, id_to_row, category_to_id, item_input_ids_mm, item_lengths, compact_lengths,
        category_label_start, attr_label_start, pad_id, PAIR_MAX_LENGTH, eval_batch_size, shuffle=False,
    )

    pos_rate = float(train_pd["label"].mean())
    print(f"[DATA] train positive rate: {pos_rate:.4f}")

    # -------------------------------------------------------------
    # Model: fresh LoRA init (first SFT run for FRIDA, nothing to resume)
    # -------------------------------------------------------------
    model = build_fresh_model(device)

    n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if n_gpu > 1:
        print(f"[GPU] Using DataParallel on {n_gpu} GPUs")
        model = nn.DataParallel(model, device_ids=list(range(n_gpu)))
    else:
        print("[GPU] Using single device")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params_count = sum(p.numel() for p in trainable_params)
    print(f"[MODEL] Total params: {total_params:,} | Trainable: {trainable_params_count:,} "
          f"({100.0 * trainable_params_count / max(1, total_params):.4f}%)")
    _log_memory("after model load")

    discovered = run_vram_probe(model, trainable_params, device, n_gpu, PAIR_MAX_LENGTH, micro_batch_size)
    grad_accum = max(1, total_batch_size // micro_batch_size)

    if discovered != micro_batch_size:
        micro_batch_size = discovered
        eval_batch_size = micro_batch_size * 2
        if micro_batch_size >= total_batch_size:
            total_batch_size = micro_batch_size
            grad_accum = 1
        else:
            grad_accum = max(1, round(total_batch_size / micro_batch_size))
            total_batch_size = micro_batch_size * grad_accum

        print("[PROBE] Rebuilding data loaders with the new batch size ...")
        train_loader = make_pair_loader(
            train_pd, id_to_row, category_to_id, item_input_ids_mm, item_lengths, compact_lengths,
            category_label_start, attr_label_start, pad_id, PAIR_MAX_LENGTH, micro_batch_size, shuffle=True,
        )
        val_loader = make_pair_loader(
            val_pd, id_to_row, category_to_id, item_input_ids_mm, item_lengths, compact_lengths,
            category_label_start, attr_label_start, pad_id, PAIR_MAX_LENGTH, eval_batch_size, shuffle=False,
        )
        print(f"[PROBE] New config: MICRO_BATCH_SIZE={micro_batch_size} | EVAL_BATCH_SIZE={eval_batch_size} | "
              f"GRAD_ACCUM={grad_accum} | effective TOTAL_BATCH_SIZE={total_batch_size}")
    _log_memory("after probe")

    optimizer = torch.optim.AdamW(trainable_params, lr=LR, weight_decay=WEIGHT_DECAY)

    num_update_steps_per_epoch = math.ceil(len(train_loader) / grad_accum)
    total_training_steps = num_update_steps_per_epoch * EPOCHS
    warmup_steps = max(1, int(total_training_steps * WARMUP_RATIO))

    scheduler = get_scheduler(
        name="cosine", optimizer=optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_training_steps
    )
    print(f"[TRAIN] steps/epoch: {num_update_steps_per_epoch} | total steps: {total_training_steps} "
          f"| warmup: {warmup_steps} ({WARMUP_RATIO:.0%}) -> cosine decay to 0")

    if pos_rate <= 0.0 or pos_rate >= 1.0:
        pos_weight_value = 1.0
    else:
        pos_weight_value = float(np.clip((1.0 - pos_rate) / pos_rate, 0.1, 20.0))
    print(f"[LOSS] BCE pos_weight: {pos_weight_value:.4f}")
    pos_weight = torch.tensor([pos_weight_value], device=device, dtype=torch.float32)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    checkpoint_interval = max(1, math.ceil(num_update_steps_per_epoch / N_CHECKPOINTS_PER_EPOCH))
    print(f"[TRAIN] checkpoint every {checkpoint_interval} optimizer steps (~{N_CHECKPOINTS_PER_EPOCH}/epoch) "
          f"-- each checkpoint prints train_loss/train_macro_pr_auc AND val_loss/val_macro_pr_auc together, "
          f"the main signal to watch in Kaggle logs.")

    # No GradScaler: bf16 doesn't need loss-scaling (its exponent range
    # already matches fp32, unlike fp16's) -- see module docstring for why
    # this is bf16 now, not fp16. GradScaler's skip-on-overflow behavior is
    # replaced by the explicit isfinite() checks below.

    mlflow.start_run(run_name="lora_frida_sft_human_only")
    mlflow.log_params({
        "variant": "lora_frida_sft_human_only", "encoder": MODEL_NAME,
        "item_max_length": ITEM_MAX_LENGTH, "pair_max_length": PAIR_MAX_LENGTH, "prefix": PREFIX,
        "lora_r": LORA_R, "lora_alpha": LORA_ALPHA, "lora_dropout": LORA_DROPOUT,
        "lora_target_modules": ",".join(LORA_TARGET_MODULES),
        "micro_batch_size": micro_batch_size, "eval_batch_size": eval_batch_size,
        "gradient_accumulation_steps": grad_accum, "effective_total_batch_size": total_batch_size,
        "lr": LR, "scheduler": "cosine", "warmup_ratio": WARMUP_RATIO, "weight_decay": WEIGHT_DECAY,
        "epochs": EPOCHS, "split_seed": SEED, "val_fingerprint": val_fingerprint,
        "train_rows": len(train_pd), "val_rows": len(val_pd),
        "n_checkpoints_per_epoch": N_CHECKPOINTS_PER_EPOCH,
        "item_cache": "maxamsecret/frida-item-tokens-human",
    })

    log_rows = []
    best_macro = -1.0
    global_step = 0
    window_logits, window_labels, window_categories = [], [], []
    window_loss_sum, window_loss_count = 0.0, 0
    n_skipped_microbatches = 0   # non-finite loss -> excluded from backward/window
    n_skipped_optimizer_steps = 0  # non-finite gradient after accumulation -> step skipped

    def run_checkpoint():
        nonlocal window_logits, window_labels, window_categories
        nonlocal window_loss_sum, window_loss_count, best_macro

        train_logits = torch.cat(window_logits, dim=0).numpy()
        train_labels_np = torch.cat(window_labels, dim=0).numpy()
        train_cats = np.concatenate(window_categories, axis=0)
        train_probs = 1.0 / (1.0 + np.exp(-train_logits))
        train_macro, train_pos_ap, _ = compute_metrics((train_labels_np >= 0.5).astype(int), train_probs, train_cats)
        train_window_loss = window_loss_sum / max(1, window_loss_count)

        window_logits, window_labels, window_categories = [], [], []
        window_loss_sum, window_loss_count = 0.0, 0

        eval_model = model.module if isinstance(model, nn.DataParallel) else model
        val_metrics = evaluate(eval_model, val_loader, loss_fn, device)
        model.train()

        progress = global_step / total_training_steps
        row = {
            "global_step": global_step, "progress": round(progress, 4),
            "train_loss": train_window_loss, "train_macro_pr_auc": train_macro, "train_pos_pr_auc": train_pos_ap,
            "val_loss": val_metrics["val_loss"], "val_macro_pr_auc": val_metrics["macro_pr_auc"],
            "val_pos_pr_auc": val_metrics["pos_pr_auc"], "lr": scheduler.get_last_lr()[0],
            "n_skipped_microbatches": n_skipped_microbatches, "n_skipped_optimizer_steps": n_skipped_optimizer_steps,
        }
        log_rows.append(row)
        pl.DataFrame(log_rows).write_csv(os.path.join(OUTPUT_DIR, "train_log_steps.csv"))
        mlflow.log_metrics({k: v for k, v in row.items() if k != "progress"}, step=global_step)

        # Single-line, grep-friendly summary -- this is the main thing to
        # watch for in Kaggle's log viewer (explicit ask: train metric, val
        # metric, train/val loss, all together, every checkpoint). Skip
        # counts included so a NaN/Inf problem (like the one that killed the
        # fp16 run at step 229) shows up here immediately instead of only at
        # a hard crash.
        print("=" * 100)
        print(f"[METRICS step={global_step}/{total_training_steps} epoch_progress={progress:.1%} lr={row['lr']:.2e}] "
              f"TRAIN loss={train_window_loss:.5f} macro_pr_auc={train_macro:.5f} | "
              f"VAL loss={row['val_loss']:.5f} macro_pr_auc={row['val_macro_pr_auc']:.5f} | "
              f"skipped: {n_skipped_microbatches} microbatches, {n_skipped_optimizer_steps} optimizer steps (NaN/Inf guard)")
        print("=" * 100)
        _log_memory(f"checkpoint step={global_step}")

        if not np.isnan(row["val_macro_pr_auc"]) and row["val_macro_pr_auc"] > best_macro:
            best_macro = row["val_macro_pr_auc"]
            save_dir = os.path.join(OUTPUT_DIR, "best")
            save_checkpoint(model, tokenizer, save_dir)
            print(f"[SAVE] new best val_macro_pr_auc={best_macro:.5f} -> {save_dir}")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}", dynamic_ncols=True)
        print(f"[EPOCH] starting epoch {epoch + 1}/{EPOCHS}", flush=True)

        for step, batch in enumerate(pbar):
            input_ids, attention_mask, batch_labels, batch_categories = batch
            input_ids = input_ids.to(device, non_blocking=True)
            attention_mask = attention_mask.to(device, non_blocking=True)
            batch_labels_dev = batch_labels.to(device, non_blocking=True)

            # bf16 autocast happens INSIDE FridaCrossEncoderClassifier.forward()
            # now, not wrapped here (see that class for why).
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(logits, batch_labels_dev)

            # NaN/Inf guard #1 (loss level): the failure mode that killed the
            # earlier fp16 run at step 229 -- catch it here, per-microbatch,
            # instead of only downstream at the next metric computation. A
            # bad microbatch is excluded entirely (no backward, no window
            # contribution) rather than corrupting the accumulated gradient
            # or the train-metric window with a NaN.
            if not torch.isfinite(loss):
                n_skipped_microbatches += 1
                print(f"[WARN] step={step} non-finite loss ({loss.item()!r}) -- skipping this microbatch "
                      f"(total skipped so far: {n_skipped_microbatches})", flush=True)
                # Critical: no .backward() runs on this branch, so the forward
                # activation graph (all 24 T5 layers) is NOT freed by autograd
                # like it normally would be. Without this del, the graph stays
                # alive -- pinned by the `logits`/`loss` names -- until the NEXT
                # iteration's `logits = model(...)` reassigns them, which means
                # that new forward pass has to allocate its own full graph
                # WHILE the stale one is still resident. A real run OOM'd
                # exactly this way: step 0 NaN'd and was skipped, step 1's
                # forward then died mid-layer with the old graph still parked
                # in memory (measured single-graph peak was already ~12GB on a
                # 14.56GB T4 -- no room for two at once).
                del logits, loss
                torch.cuda.empty_cache()
            else:
                loss_item = loss.item()
                (loss / grad_accum).backward()

                window_logits.append(logits.detach().float().cpu())
                window_labels.append(batch_labels.detach().cpu())
                window_categories.append(batch_categories)
                window_loss_sum += loss_item
                window_loss_count += 1
                pbar.set_postfix(loss=round(loss_item, 5), lr=f"{scheduler.get_last_lr()[0]:.2e}", step=global_step)

            need_step = ((step + 1) % grad_accum == 0) or ((step + 1) == len(train_loader))
            if need_step:
                # NaN/Inf guard #2 (gradient level): even if every
                # contributing microbatch had a finite loss, accumulated
                # gradients can still overflow -- this is what GradScaler
                # used to silently guard against (its "scheduler.step()
                # before optimizer.step()" warning was exactly this
                # happening, early and often, in the failed fp16 run).
                # Explicit here since bf16 training has no GradScaler.
                grad_ok = all(p.grad is None or torch.isfinite(p.grad).all() for p in trainable_params)
                if grad_ok:
                    torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                    optimizer.step()
                    scheduler.step()
                    global_step += 1
                else:
                    n_skipped_optimizer_steps += 1
                    print(f"[WARN] optimizer step skipped: non-finite gradient after accumulation "
                          f"(total skipped so far: {n_skipped_optimizer_steps})", flush=True)
                optimizer.zero_grad()

                is_epoch_end = (step + 1) == len(train_loader)
                if grad_ok and ((global_step % checkpoint_interval == 0) or is_epoch_end):
                    run_checkpoint()

        print(f"[EPOCH] finished epoch {epoch + 1}/{EPOCHS} | best val_macro_pr_auc so far={best_macro:.5f}", flush=True)

    last_dir = os.path.join(OUTPUT_DIR, "last")
    save_checkpoint(model, tokenizer, last_dir)

    print(f"[DONE] Best macro_pr_auc={best_macro:.5f}")
    print(f"[DONE] Step log: {os.path.join(OUTPUT_DIR, 'train_log_steps.csv')}")
    print(f"[DONE] Best model dir: {os.path.join(OUTPUT_DIR, 'best')}")
    print(f"[DONE] Last model dir: {last_dir}")

    mlflow.log_metric("best_macro_pr_auc", best_macro)
    mlflow.log_artifact(os.path.join(OUTPUT_DIR, "train_log_steps.csv"))
    mlflow.end_run()
    print(f"[DONE] mlflow run logged under {MLFLOW_TRACKING_DIR}")


if __name__ == "__main__":
    main()
