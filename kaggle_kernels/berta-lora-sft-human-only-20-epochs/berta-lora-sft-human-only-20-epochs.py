"""
LoRA SFT sergeyzh/BERTA (BertModel, 128.3M) as a cross-encoder, on the
human-labeled train split ONLY (matches.parquet, 292,523 train / 73,131 val,
canonical seed-42 split) -- BERTA sibling of kaggle_kernels/frida-lora-sft,
same overall recipe, adapted for a real BERT architecture instead of T5.

ONE script, not two: tokenization happens as an upfront phase inside main(),
not a separate published Kaggle dataset + separate kernel (frida-lora-sft's
own item-cache WAS a separate kernel+dataset -- deliberately simplified here
per explicit instruction). BERTA is 128.3M params, ~6.5x smaller than
FRIDA's 823M, and its WordPiece tokenizer is fast -- re-tokenizing all
~711,304 human-scope items from scratch on every run (including a resumed
run) costs a few minutes, not hours, so there is no real re-join/re-tokenize
bottleneck here to engineer around the way there was for the pair-level
lora-llm-continue-train.py (~9.5h/12h session, the reason FRIDA's cache is
item-level and pre-published in the first place). Item-level (not pair-level)
tokenization is still used -- each item is tokenized exactly once regardless
-- just done in-process instead of via a separate cache dataset.

PAIR_MAX_LENGTH=128 (fixed, per explicit instruction -- matches
frida-lora-sft's own choice, NOT the data-driven near-lossless budget: a real
measurement over all 365,654 matches.parquet pairs with BERTA's tokenizer
found only 0.00% truncated at 512, but 98.76% truncated at 128 -- see this
kernel's README "Token-budget sizing" for the full table). EPOCHS=20 (fixed,
per explicit instruction).

Architecture differences from FRIDA (T5, bi-encoder-shaped even when used as
a cross-encoder) that matter here:
  - BERTA is architecturally BertModel (12 layers, hidden=768, WordPiece,
    vocab=55,083) -- a REAL sentence-pair-capable encoder, unlike T5 (no
    segment concept at all). Verified via config.json + named_modules() on
    the actually-loaded checkpoint, not assumed.
  - Canonical BERT pair format is used: `[CLS] item1 [SEP] item2 [SEP]` with
    token_type_ids (0 for CLS+item1+first SEP, 1 for item2+final SEP) -- NOT
    "two self-contained sequences concatenated" (that was a T5-specific
    workaround; T5 has no [CLS]/[SEP]/token_type_ids at all). This means the
    item-level cache stores RAW content only (add_special_tokens=False) --
    special tokens and token_type_ids are assembled at collate time once the
    actual pair is known, mirroring frida-lora-sft's "concatenate at
    train time" principle but for the correct BERT structure.
  - Pooling: MEAN pooling (mask-weighted average over last_hidden_state), per
    BERTA's own model card -- it explicitly replaces FRIDA's CLS pooling
    with mean pooling ("Основной режим использования FRIDA - CLS pooling
    заменен на mean pooling"), the one deliberate behavior change the
    distillation made. frida-lora-sft used CLS/position-0; this does not.
  - LoRA target_modules=["query","key","value","dense"] -- BERT's real
    Linear leaf names (verified via named_modules() on the loaded
    checkpoint: attention.self.{query,key,value}, attention.output.dense,
    intermediate.dense, output.dense -- "dense" matches all three of the
    latter by leaf name, so "all weights" here means all 6 Linear layers per
    block, the BERT analogue of FRIDA's 7-per-block T5 count).

Lessons carried over from frida-lora-sft's own debugging saga, applied from
the START instead of rediscovered the hard way again:
  - autocast lives INSIDE BertaCrossEncoderClassifier.forward(), never
    wrapped around a call to an nn.DataParallel-wrapped model from the
    outside. Root-caused for FRIDA via a dedicated GPU diagnostic
    (kaggle_kernels/frida-nan-diagnose): DataParallel runs each replica's
    forward() in its own thread, and an autocast context entered by the
    caller does not propagate into those threads -- PyTorch's own docs
    prescribe entering autocast inside forward() for exactly this reason.
    Wrapping it outside produced NaN identical (bit-for-bit) to a
    known-broken fp16 run, even under bf16.
  - Explicit NaN/Inf guards in the training loop include `del logits, loss`
    in the skip branch -- frida-lora-sft's first version of this guard
    caused a real CUDA OOM (the skipped step's forward graph stayed pinned
    in memory, coexisting with the next step's fresh graph, on a card where
    one graph already used ~12GB of 14.56GB) because it only skipped
    .backward() without freeing the tensors.
  - `is_trainable=True` when resuming a saved PEFT adapter via
    PeftModel.from_pretrained() -- this repo's own documented "PEFT
    is_trainable trap" (CLAUDE.md / lora-llm-continue-train history):
    defaults to False, which would silently freeze every LoRA weight with no
    error and no visible symptom other than a training loss that never
    moves.

Checkpoint + resume ("если что-то сломалось то в output все осталось"): 20
epochs on 292,523 pairs is very likely more than one 12h Kaggle session, even
at BERTA's much smaller size, so this is NOT optional this time the way it
was for FRIDA's 1-epoch validation run. Every checkpoint (N_CHECKPOINTS_PER_
EPOCH per epoch, matching frida-lora-sft's cadence) saves the LoRA adapter +
classifier head + optimizer + scheduler + exact step/epoch counters to
/kaggle/working -- recoverable via `kaggle kernels output` even after a hard
cancellation (this repo's own MiniLM continue-train precedent). If Kaggle API
credentials are available via kernel secrets, the SAME checkpoint also gets
published/versioned as a Kaggle dataset (mirrors the proven
ozon-lora-minilm-finetuned pattern) so a later kernel push can point at it
and auto-resume cleanly -- see this kernel's README "Resuming across
sessions" for the exact steps.

NOT launched by this commit -- explicit instruction: write and locally
validate now, run on Kaggle later.
"""
import os
import sys
import gc
import ctypes
import json
import math
import shutil
import hashlib
import subprocess

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "polars"])

import numpy as np
import pandas as pd
import polars as pl

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from tqdm.auto import tqdm

from transformers import AutoModel, AutoTokenizer, get_scheduler, set_seed
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

from peft import LoraConfig, PeftModel, get_peft_model


# =====================================================================
# CONFIG
# =====================================================================

SEED = 42
MODEL_NAME = "sergeyzh/BERTA"

ITEMS_HUMAN_PATH_HINT = "/kaggle/input/datasets/maxamsecret/ozon-ecup-2026-matching/items_human.parquet"
MATCHES_PATH_HINT = "/kaggle/input/datasets/maxamsecret/ozon-ecup-2026-matching/matches.parquet"

PREFIX = "paraphrase: "          # inherited from FRIDA (BERTA distills FRIDA's
                                   # full prefix vocabulary, see this kernel's
                                   # README) -- kept consistent with
                                   # frida-lora-sft's own choice for the same
                                   # paraphrase/deduplication-shaped task.
ITEM_MAX_LENGTH = 128             # per-item storage cap -- deliberately equal
                                   # to PAIR_MAX_LENGTH, not the item's own
                                   # real-length distribution (mean=148.6,
                                   # p99=228, true max=347 over the full
                                   # 711,304-item population -- most items
                                   # DO exceed 128 on their own). This is not
                                   # lossy in practice: allocate_pair_budget's
                                   # content_budget is PAIR_MAX_LENGTH-3=125,
                                   # so no single side of a pair can ever use
                                   # more than 125 tokens regardless of how
                                   # much the cache holds -- 128 already
                                   # covers that ceiling. Verified directly
                                   # (150K-item sample, add_special_tokens=
                                   # False, max_length=128): 100.00% still
                                   # have a locatable Category:/Attributes:
                                   # label (Name -> Category -> Attributes
                                   # ordering puts the label well before
                                   # position 128 for virtually every item),
                                   # mean 80.8 real Attributes tokens still
                                   # remain after excision, before any
                                   # pair-level truncation even applies.

MAX_NAME_CHARS, MAX_CATEGORY_CHARS, MAX_ATTR_CHARS = 128, 64, 384
MAX_ITEM_CHARS = MAX_NAME_CHARS + MAX_CATEGORY_CHARS + MAX_ATTR_CHARS + 64  # 640, byte-matches CLAUDE.md

PAIR_MAX_LENGTH = 128   # Fixed per explicit instruction (matches
                         # frida-lora-sft). Real measurement over all 365,654
                         # matches.parquet pairs with BERTA's own tokenizer
                         # (CLS+SEP+SEP included, no truncation): mean=300.3,
                         # p50=310, p90=393, p99=443, true max=536 --
                         # 98.76% of real pairs get truncated at 128 (mean
                         # 174.6 tokens lost when truncated). A deliberate
                         # cost/quota-over-completeness choice, not a
                         # data-driven one -- see this kernel's README for
                         # the full budget table (512 would be 0.00%
                         # truncated).

EPOCHS = 20   # Fixed per explicit instruction.

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.10
LORA_TARGET_MODULES = ["query", "key", "value", "dense"]   # BERT's real Linear
                                                              # leaf names,
                                                              # verified via
                                                              # named_modules()
                                                              # on the loaded
                                                              # checkpoint --
                                                              # "dense" matches
                                                              # attention.
                                                              # output.dense +
                                                              # intermediate.
                                                              # dense + output.
                                                              # dense (3 of the
                                                              # 6 Linears/block).

LR = 3e-4                # peak LR, warmup then cosine decay -- same schedule
                           # shape as frida-lora-sft (no LR was specified for
                           # BERTA specifically; this is a reasonable general
                           # LoRA fine-tuning rate, kept consistent with the
                           # sibling script rather than newly guessed).
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.10

N_CHECKPOINTS_PER_EPOCH = 5   # 100 checkpoints across 20 epochs -- frequent
                                # enough that a crash never loses much more
                                # than a fraction of an epoch's progress.

MICRO_BATCH_SIZE = 16          # VRAM-probe starting point, not the final value.
EVAL_MICRO_BATCH_SIZE = 32
TOTAL_BATCH_SIZE = 256         # effective (post-grad-accum) batch size, fixed
                                 # regardless of what the probe picks for
                                 # MICRO_BATCH_SIZE -- keeps the LR schedule's
                                 # total-step count (and therefore any resumed
                                 # scheduler state) independent of which GPU
                                 # this happens to run on.

TARGET_VRAM_GB = 14.0
SAFETY_MARGIN_GB = 3.0    # was 1.5 -- a real run OOM'd on step 0 at
                            # MICRO_BATCH_SIZE=320, which the probe measured
                            # in isolation at exactly 12.50GB (== the 1.5GB
                            # margin's target, zero real cushion). Real
                            # training hit ~14GB and OOM'd trying to allocate
                            # 240MB more. The gap (~1.5GB) between an
                            # isolated single forward+backward and a real
                            # step's actual footprint isn't captured by the
                            # probe's dummy all-ones-mask tensor -- doubling
                            # the margin leaves real headroom for it instead
                            # of trusting the isolated measurement exactly.
PROBE_CAP = 512
PROBE_ROUND_TO = 4

# Category/Attributes label token patterns -- verified against BERTA's real
# tokenizer (see this kernel's README): "Category"+":" is a clean 2-token
# WordPiece match ([14948, 30]); "Attributes"+":" splits into 4 subword
# tokens ('Att','##ribute','##s',':' = [4816, 37519, 521, 30]) since
# "Attributes" isn't a single vocab entry -- unlike FRIDA's byte-BPE tokenizer
# (2-token match there), so the pattern-finder below is written for an
# ARBITRARY-length pattern, not hardcoded to 2. Same excision rationale as
# frida-lora-sft: `category` is 100% identical between both items of every
# pair (CLAUDE.md) -- zero discriminative signal, pure budget waste -- and at
# PAIR_MAX_LENGTH=128 (98.76% truncated) freeing it for Attributes matters
# more here than it did even for FRIDA.
CATEGORY_LABEL_PATTERN = [14948, 30]
ATTRIBUTES_LABEL_PATTERN = [4816, 37519, 521, 30]

WORKING_DIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
OUTPUT_DIR = os.path.join(WORKING_DIR, "lora_berta_sft")
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoint")
MLFLOW_TRACKING_DIR = os.path.join(WORKING_DIR, "experiments", "mlruns")

CHECKPOINT_DATASET_ID = "rays0n/berta-lora-sft-checkpoint"   # separate Kaggle
                                                                # account from
                                                                # frida-lora-sft
                                                                # (maxamsecret)
                                                                # -- see
                                                                # CLAUDE.md
                                                                # "Kaggle
                                                                # accounts" for
                                                                # why.
CHECKPOINT_DATASET_TITLE = "BERTA LoRA SFT -- resumable checkpoint"
ENABLE_DATASET_PUSH = True   # unlike frida-lora-sft (off by default, a
                               # 1-epoch validation run didn't need it): 20
                               # epochs very likely spans multiple 12h
                               # sessions, so cross-session resume is a real
                               # requirement here, not a nice-to-have. Falls
                               # back to local-only checkpointing (still
                               # recoverable via `kaggle kernels output`) if
                               # kernel secrets aren't configured.

TOKENIZE_CHUNK_SIZE_START = 20_000
TOKENIZE_CHUNK_SIZE_MIN = 2_000
TOKENIZE_CHUNK_SIZE_MAX = 200_000
RAM_TARGET_GB = 16.0
RAM_HARD_GB = 24.0
RAM_GROWTH_FACTOR = 1.3
RAM_SHRINK_FACTOR = 0.5


# =====================================================================
# MEMORY / DISK HELPERS
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


def _require_disk_space(path, required_bytes, label, margin=1.15):
    os.makedirs(path, exist_ok=True)
    free = shutil.disk_usage(path).free
    need = int(required_bytes * margin)
    print(f"[DISK] {label}: free={free / 1e9:.2f}GB required(+margin)={need / 1e9:.2f}GB")
    if free < need:
        raise RuntimeError(f"[DISK] insufficient space for {label}: need {need / 1e9:.2f}GB, have {free / 1e9:.2f}GB")


def _find_dataset_file(hint, filename):
    if os.path.isfile(hint):
        return hint
    if os.path.isdir("/kaggle/input"):
        for root, _dirs, files in os.walk("/kaggle/input"):
            if filename in files:
                print(f"[DATA] hint path {hint} didn't resolve -- found {filename} via search at {root}")
                return os.path.join(root, filename)
    return hint


# =====================================================================
# PHASE 1: TOKENIZE (item-level, in-process -- see module docstring for why
# this isn't a separate cache dataset the way FRIDA's is)
# =====================================================================

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


def tokenize_items_to_memmap(items_pl, tokenizer, out_dir, max_length=ITEM_MAX_LENGTH):
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

    chunk_size = TOKENIZE_CHUNK_SIZE_START
    written = 0
    pbar = tqdm(total=n, desc="Tokenize items", dynamic_ncols=True)

    while written < n:
        cur = min(chunk_size, n - written)
        chunk_texts = [PREFIX + t for t in all_texts[written:written + cur]]

        # add_special_tokens=False -- [CLS]/[SEP]/token_type_ids are
        # assembled at collate time once the actual pair is known (see
        # make_pair_collate_fn), not baked into each item independently.
        enc = tokenizer(
            chunk_texts, add_special_tokens=False,
            padding="max_length", truncation=True,
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
                chunk_size = min(TOKENIZE_CHUNK_SIZE_MAX, int(chunk_size * RAM_GROWTH_FACTOR))
            elif rss_gb > RAM_HARD_GB:
                chunk_size = max(TOKENIZE_CHUNK_SIZE_MIN, int(chunk_size * RAM_SHRINK_FACTOR))
            pbar.set_postfix(rss_gb=round(rss_gb, 2), next_chunk=chunk_size)

    pbar.close()
    for mm in (ids_mm, input_ids_mm, attn_mm):
        mm.flush()
    del ids_mm, input_ids_mm, attn_mm
    gc.collect()


def _find_token_pattern_starts(arr, pattern):
    """First index i per row where arr[row, i:i+len(pattern)] == pattern, or
    -1 if not found. Vectorized (no Python loop over rows). Generalized over
    an arbitrary-length pattern (FRIDA's own version only needed length-2 --
    BERTA's "Attributes:" is a 4-token WordPiece split, see module docstring)."""
    pattern = np.asarray(pattern)
    plen = len(pattern)
    n_positions = arr.shape[1] - plen + 1
    match = np.ones((arr.shape[0], n_positions), dtype=bool)
    for offset, tok in enumerate(pattern):
        match &= (arr[:, offset:offset + n_positions] == tok)
    rows, cols = np.where(match)
    uniq_rows, first_idx = np.unique(rows, return_index=True)
    starts = np.full(arr.shape[0], -1, dtype=np.int32)
    starts[uniq_rows] = cols[first_idx]
    return starts


def build_item_cache(tokenizer):
    """Returns (item_input_ids, item_lengths, category_label_start,
    attr_label_start, compact_lengths, id_to_row). Tokenizes all human-scope
    items in-process (see module docstring) instead of reading a pre-built
    dataset."""
    items_path = _find_dataset_file(ITEMS_HUMAN_PATH_HINT, "items_human.parquet")
    print(f"[CACHE] tokenizing items from {items_path} ...")
    items = build_item_texts(pl.scan_parquet(items_path))
    n_items = items.height
    print(f"[CACHE] human scope items: {n_items:,}")
    _log_memory("after items load")

    cache_dir = os.path.join(WORKING_DIR, "berta_item_tokens")
    os.makedirs(cache_dir, exist_ok=True)
    tokenize_items_to_memmap(items, tokenizer, cache_dir)
    all_ids = items.get_column("id").to_numpy().copy()
    del items
    gc.collect()
    _malloc_trim()
    _log_memory("after tokenize")

    item_input_ids = np.memmap(os.path.join(cache_dir, "input_ids.bin"), dtype=np.int32, mode="r",
                                shape=(n_items, ITEM_MAX_LENGTH))
    item_attention_mask = np.memmap(os.path.join(cache_dir, "attention_mask.bin"), dtype=np.int8, mode="r",
                                     shape=(n_items, ITEM_MAX_LENGTH))

    print("[CACHE] building id -> row index map ...")
    id_to_row = {int(item_id): i for i, item_id in enumerate(all_ids)}
    assert len(id_to_row) == n_items, f"duplicate item ids: {n_items} rows but {len(id_to_row)} unique ids"

    item_lengths = np.asarray(item_attention_mask).sum(axis=1).astype(np.int32)
    print(f"[CACHE] real length mean={item_lengths.mean():.1f} p50={np.percentile(item_lengths, 50):.0f} "
          f"p90={np.percentile(item_lengths, 90):.0f} max={item_lengths.max()} (cap={ITEM_MAX_LENGTH})")

    print("[CACHE] locating Category:/Attributes: label positions (for excision) ...")
    arr = np.asarray(item_input_ids)
    category_label_start = _find_token_pattern_starts(arr, CATEGORY_LABEL_PATTERN)
    attr_label_start = _find_token_pattern_starts(arr, ATTRIBUTES_LABEL_PATTERN)
    both_found = (category_label_start >= 0) & (attr_label_start >= 0)
    n_missing = n_items - int(both_found.sum())
    if n_missing > 0:
        print(f"[CACHE] WARNING: {n_missing:,}/{n_items:,} items missing a locatable Category:/Attributes: "
              f"label. Falling back to no-excision (full item kept) for those.")
    category_span = np.where(both_found, attr_label_start - category_label_start, 0).astype(np.int32)
    compact_lengths = item_lengths - category_span
    print(f"[CACHE] category span excised: mean={category_span[both_found].mean():.1f} tokens/item | "
          f"compact length: mean={compact_lengths.mean():.1f} p50={np.percentile(compact_lengths, 50):.0f}")

    return item_input_ids, item_lengths, category_label_start, attr_label_start, compact_lengths, id_to_row


# =====================================================================
# PAIR ASSEMBLY (collate-time concatenation, fair-share truncation +
# Category excision -- same principle as frida-lora-sft, adapted for BERT's
# [CLS] A [SEP] B [SEP] + token_type_ids structure)
# =====================================================================

def allocate_pair_budget(len1, len2, budget):
    """Fair-share truncation over COMPACT lengths (category already
    excised). Principle: never cut a side that already fits its equal
    half-share -- only the excess of a side that's OVER half gets trimmed,
    and any slack left by a genuinely short side is handed to the other one.
    Symmetric, never wastes budget. Identical logic to frida-lora-sft's own
    (proven, unit-tested there); `budget` here is PAIR_MAX_LENGTH - 3 (CLS +
    SEP + SEP reserved separately, not part of this budget)."""
    half = budget // 2
    if len1 <= half and len2 <= half:
        return len1, len2
    if len1 <= half:
        return len1, budget - len1
    if len2 <= half:
        return budget - len2, len2
    return half, budget - half


def build_compact_ids(item_input_ids_mm, row, item_length, category_start, attr_start):
    """Excise the Category: span from one item's real (unpadded) tokens.
    Falls back to the untouched slice if labels weren't both found."""
    if category_start < 0 or attr_start < 0:
        return item_input_ids_mm[row, :item_length]
    head = item_input_ids_mm[row, :category_start]
    tail = item_input_ids_mm[row, attr_start:item_length]
    return np.concatenate((head, tail))


class PairIndexDataset(Dataset):
    def __init__(self, row1, row2, labels, categories):
        self.row1 = row1
        self.row2 = row2
        self.labels = labels
        self.categories = categories

    def __len__(self):
        return len(self.row1)

    def __getitem__(self, idx):
        return self.row1[idx], self.row2[idx], self.labels[idx], self.categories[idx]


def make_pair_collate_fn(item_input_ids_mm, item_lengths, compact_lengths, category_label_start,
                          attr_label_start, cls_id, sep_id, pad_id, pair_max_length):
    """Builds `[CLS] item1[:keep1] [SEP] item2[:keep2] [SEP]` + token_type_ids
    (0 for CLS+item1+first SEP, 1 for item2+final SEP) + attention_mask, pure
    array slicing/concatenation -- no tokenizer call at train time."""
    content_budget = pair_max_length - 3   # reserve CLS + SEP + SEP

    def collate_fn(batch):
        row1, row2, labels, categories = zip(*batch)
        bsz = len(batch)
        input_ids = np.full((bsz, pair_max_length), pad_id, dtype=np.int64)
        token_type_ids = np.zeros((bsz, pair_max_length), dtype=np.int64)
        attention_mask = np.zeros((bsz, pair_max_length), dtype=np.int64)

        for i in range(bsz):
            r1, r2 = row1[i], row2[i]

            compact1 = build_compact_ids(item_input_ids_mm, r1, int(item_lengths[r1]),
                                          int(category_label_start[r1]), int(attr_label_start[r1]))
            compact2 = build_compact_ids(item_input_ids_mm, r2, int(item_lengths[r2]),
                                          int(category_label_start[r2]), int(attr_label_start[r2]))
            cl1, cl2 = len(compact1), len(compact2)

            if cl1 + cl2 <= content_budget:
                keep1, keep2 = cl1, cl2
            else:
                keep1, keep2 = allocate_pair_budget(cl1, cl2, content_budget)

            seg1 = compact1[:keep1]
            seg2 = compact2[:keep2]

            n1 = 1 + keep1 + 1          # CLS + item1 + SEP
            n2 = keep2 + 1              # item2 + SEP
            n = n1 + n2

            input_ids[i, 0] = cls_id
            input_ids[i, 1:1 + keep1] = seg1
            input_ids[i, 1 + keep1] = sep_id
            input_ids[i, n1:n1 + keep2] = seg2
            input_ids[i, n1 + keep2] = sep_id

            token_type_ids[i, n1:n] = 1
            attention_mask[i, :n] = 1

        return (
            torch.from_numpy(input_ids),
            torch.from_numpy(attention_mask),
            torch.from_numpy(token_type_ids),
            torch.tensor(labels, dtype=torch.float32),
            np.asarray(categories, dtype=np.int64),
        )

    return collate_fn


def make_pair_loader(pairs_pd, id_to_row, category_to_id, item_input_ids_mm, item_lengths, compact_lengths,
                      category_label_start, attr_label_start, cls_id, sep_id, pad_id, pair_max_length,
                      batch_size, shuffle):
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
                                       category_label_start, attr_label_start,
                                       cls_id, sep_id, pad_id, pair_max_length)
    drop_last = shuffle and (len(dataset) >= batch_size)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0,
        collate_fn=collate_fn, pin_memory=torch.cuda.is_available(), drop_last=drop_last,
    )
    return loader


# =====================================================================
# MODEL
# =====================================================================

class BertaCrossEncoderClassifier(nn.Module):
    """Mean pooling (mask-weighted average over last_hidden_state) -- BERTA's
    own model card explicitly replaces FRIDA's CLS pooling with mean pooling
    ("Основной режим использования FRIDA - CLS pooling заменен на mean
    pooling"); this is the one deliberate behavior change the distillation
    made, so it's followed here rather than reusing frida-lora-sft's CLS
    convention. token_type_ids used (real BERT segment embeddings, unlike T5).

    autocast lives INSIDE forward(), never wrapped around calls to this
    module from the outside -- see module docstring for why (root-caused via
    frida-nan-diagnose: DataParallel replica threads don't see an
    outer-entered autocast context)."""

    def __init__(self, encoder, head):
        super().__init__()
        self.encoder = encoder
        self.classifier = head

    def forward(self, input_ids, attention_mask, token_type_ids):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask,
                                token_type_ids=token_type_ids)
            hidden = out.last_hidden_state
            mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
            summed = (hidden * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-9)
            pooled = summed / counts
            return self.classifier(pooled).view(-1).float()


def build_fresh_model(device):
    print(f"[MODEL] Loading base encoder {MODEL_NAME} (AutoModel resolves to BertModel per config.json) ...")
    base_model = AutoModel.from_pretrained(MODEL_NAME)
    hidden_dim = base_model.config.hidden_size

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
    model = BertaCrossEncoderClassifier(peft_model, classifier_head).to(device)
    return model


def load_model_from_checkpoint(device, adapter_dir, classifier_path):
    """Same as build_fresh_model, but loads a previously-saved adapter
    instead of a fresh LoRA init. is_trainable=True is NOT the default --
    this repo's own documented gotcha (CLAUDE.md / lora-llm-continue-train
    history): PeftModel.from_pretrained defaults is_trainable=False, which
    silently freezes every LoRA weight (loss would never move, no error)."""
    print(f"[RESUME] Loading base encoder {MODEL_NAME} + resuming adapter from {adapter_dir} ...")
    base_model = AutoModel.from_pretrained(MODEL_NAME)
    hidden_dim = base_model.config.hidden_size
    peft_model = PeftModel.from_pretrained(base_model, adapter_dir, is_trainable=True)
    peft_model.print_trainable_parameters()

    n_trainable_lora = sum(p.numel() for n, p in peft_model.named_parameters() if "lora_" in n and p.requires_grad)
    assert n_trainable_lora > 0, "Resumed LoRA adapter has zero trainable params -- is_trainable didn't take."

    classifier_head = nn.Linear(hidden_dim, 1)
    classifier_head.load_state_dict(torch.load(classifier_path, map_location="cpu"))
    model = BertaCrossEncoderClassifier(peft_model, classifier_head).to(device)
    return model


# =====================================================================
# VRAM AUTO-PROBE (autocast now happens inside forward(), so the probe
# doesn't need its own autocast wrap -- see BertaCrossEncoderClassifier)
# =====================================================================

def _probe_peak_gb(batch_size, n_gpu, pair_max_length, probe_model, probe_loss_fn, trainable_params, device,
                    train_mode=True):
    """train_mode=True: forward+backward (real training footprint -- gradients
    + activations retained for backward). train_mode=False: forward only
    under no_grad (real eval footprint -- evaluate() never calls backward,
    so probing WITH one would be a pessimistic, wrong estimate for it; this
    is why eval got its own probe rather than reusing the training one's
    result directly)."""
    dummy_input_ids = torch.randint(0, 1000, (batch_size, pair_max_length), dtype=torch.long, device=device)
    dummy_attention_mask = torch.ones((batch_size, pair_max_length), dtype=torch.long, device=device)
    dummy_token_type_ids = torch.zeros((batch_size, pair_max_length), dtype=torch.long, device=device)
    dummy_labels = torch.randint(0, 2, (batch_size,), dtype=torch.float32, device=device)

    if train_mode:
        for p in trainable_params:
            p.grad = None
    for i in range(max(n_gpu, 1)):
        torch.cuda.reset_peak_memory_stats(i)

    if train_mode:
        logits = probe_model(input_ids=dummy_input_ids, attention_mask=dummy_attention_mask,
                              token_type_ids=dummy_token_type_ids)
        loss = probe_loss_fn(logits, dummy_labels)
        loss.backward()
    else:
        with torch.no_grad():
            logits = probe_model(input_ids=dummy_input_ids, attention_mask=dummy_attention_mask,
                                  token_type_ids=dummy_token_type_ids)
            loss = probe_loss_fn(logits, dummy_labels)

    peak_bytes = max(torch.cuda.max_memory_allocated(i) for i in range(max(n_gpu, 1)))

    if train_mode:
        for p in trainable_params:
            p.grad = None
    del dummy_input_ids, dummy_attention_mask, dummy_token_type_ids, dummy_labels, logits, loss
    torch.cuda.empty_cache()
    return peak_bytes / (1024 ** 3)


def _is_oom(exc):
    return isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()


def run_vram_probe(model, trainable_params, device, n_gpu, pair_max_length, start_batch, probe_cap,
                    train_mode=True, label="MICRO_BATCH_SIZE"):
    if not torch.cuda.is_available():
        print(f"[PROBE] No GPU -- skipping VRAM probe, keeping the configured {label}.")
        return start_batch

    probe_loss_fn = torch.nn.BCEWithLogitsLoss()
    print(f"[PROBE] Searching for max {label} under {TARGET_VRAM_GB}GB/GPU "
          f"(safety margin {SAFETY_MARGIN_GB}GB), capped at {probe_cap} "
          f"({'train: forward+backward' if train_mode else 'eval: forward only, no_grad'}) ...")

    candidate = start_batch
    last_good = None
    while candidate <= probe_cap:
        try:
            peak_gb = _probe_peak_gb(candidate, n_gpu, pair_max_length, model, probe_loss_fn, trainable_params,
                                      device, train_mode=train_mode)
            print(f"[PROBE] batch={candidate:5d} -> peak={peak_gb:.2f}GB/GPU")
            if peak_gb >= TARGET_VRAM_GB - SAFETY_MARGIN_GB:
                break
            last_good = candidate
            candidate *= 2
        except RuntimeError as e:
            if not _is_oom(e):
                raise
            torch.cuda.empty_cache()
            print(f"[PROBE] batch={candidate:5d} -> OOM")
            break
        gc.collect()

    if last_good is None:
        last_good = max(PROBE_ROUND_TO, start_batch // 2)

    # min(candidate, probe_cap): the doubling loop above can exit with
    # candidate already doubled PAST probe_cap (loop condition checked before
    # the *=2, not after) -- without this the binary search would still
    # explore batches above the cap, defeating the cap just applied.
    lo, hi = last_good, min(candidate, probe_cap)
    while hi - lo > PROBE_ROUND_TO:
        mid = ((lo + hi) // 2 // PROBE_ROUND_TO) * PROBE_ROUND_TO
        if mid <= lo:
            break
        try:
            peak_gb = _probe_peak_gb(mid, n_gpu, pair_max_length, model, probe_loss_fn, trainable_params, device,
                                      train_mode=train_mode)
            print(f"[PROBE] batch={mid:5d} -> peak={peak_gb:.2f}GB/GPU")
            if peak_gb >= TARGET_VRAM_GB - SAFETY_MARGIN_GB:
                hi = mid
            else:
                lo = mid
        except RuntimeError as e:
            if not _is_oom(e):
                raise
            torch.cuda.empty_cache()
            print(f"[PROBE] batch={mid:5d} -> OOM")
            hi = mid
        gc.collect()

    print(f"[PROBE] Selected {label}={lo}")
    return lo


# =====================================================================
# METRICS + EVAL
# =====================================================================

def compute_metrics(probs, labels, categories, category_names):
    """Macro-averaged PR-AUC, per category (CLAUDE.md metric definition):
    average_precision_score per category, plain mean across categories that
    have both classes present. Never sklearn.metrics.auc."""
    per_cat = {}
    for cat_id, cat_name in enumerate(category_names):
        mask = categories == cat_id
        if mask.sum() < 2:
            continue
        y_true = labels[mask]
        if len(np.unique(y_true)) < 2:
            continue
        y_prob = probs[mask]
        per_cat[cat_name] = average_precision_score(y_true, y_prob)
    macro_pr_auc = float(np.mean(list(per_cat.values()))) if per_cat else float("nan")
    return macro_pr_auc, per_cat


def evaluate(eval_model, loader, loss_fn, device, category_names):
    eval_model.eval()

    all_logits, all_labels, all_categories = [], [], []
    total_loss, n_batches = 0.0, 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Eval", dynamic_ncols=True, leave=False):
            input_ids, attention_mask, token_type_ids, batch_labels, batch_categories = batch
            input_ids = input_ids.to(device, non_blocking=True)
            attention_mask = attention_mask.to(device, non_blocking=True)
            token_type_ids = token_type_ids.to(device, non_blocking=True)
            batch_labels_dev = batch_labels.to(device, non_blocking=True)

            logits = eval_model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
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
    macro_pr_auc, per_cat = compute_metrics(probs, labels.numpy(), categories, category_names)

    return {"loss": val_loss, "macro_pr_auc": macro_pr_auc, "per_category": per_cat}


# =====================================================================
# CHECKPOINT / RESUME
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
        print(f"[WARN] Kaggle secrets unavailable, checkpoint dataset publish disabled this run: {e}")
        return False


def _find_checkpoint_dir():
    """Searches /kaggle/input for a directory containing train_state.json --
    a previously-published checkpoint dataset, if this kernel's
    dataset_sources has been updated to include CHECKPOINT_DATASET_ID (see
    this kernel's README "Resuming across sessions"). Returns None if not
    found -- a fresh start."""
    if os.path.isdir("/kaggle/input"):
        for root, _dirs, files in os.walk("/kaggle/input"):
            if "train_state.json" in files:
                print(f"[RESUME] found checkpoint at {root}")
                return root
    return None


def save_checkpoint(model, optimizer, scheduler, epoch, global_step, best_val_macro_pr_auc,
                     n_skipped_microbatches, n_skipped_optimizer_steps, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    raw = model.module if isinstance(model, nn.DataParallel) else model
    raw.encoder.save_pretrained(os.path.join(save_dir, "adapter"))
    torch.save(raw.classifier.state_dict(), os.path.join(save_dir, "classifier.pt"))
    torch.save(optimizer.state_dict(), os.path.join(save_dir, "optimizer.pt"))
    torch.save(scheduler.state_dict(), os.path.join(save_dir, "scheduler.pt"))
    with open(os.path.join(save_dir, "train_state.json"), "w") as f:
        json.dump({
            "completed_epochs": epoch, "global_step": global_step,
            "best_val_macro_pr_auc": best_val_macro_pr_auc,
            "n_skipped_microbatches": n_skipped_microbatches,
            "n_skipped_optimizer_steps": n_skipped_optimizer_steps,
        }, f, indent=2)


_dataset_created = [False]


def publish_checkpoint_dataset(save_dir, message, has_kaggle_creds):
    if not (ENABLE_DATASET_PUSH and has_kaggle_creds):
        return
    with open(os.path.join(save_dir, "dataset-metadata.json"), "w") as f:
        json.dump({"id": CHECKPOINT_DATASET_ID, "title": CHECKPOINT_DATASET_TITLE,
                    "licenses": [{"name": "CC0-1.0"}]}, f)
    try:
        if not _dataset_created[0]:
            subprocess.run(["kaggle", "datasets", "create", "-p", save_dir, "-q"],
                            check=True, capture_output=True, text=True)
            _dataset_created[0] = True
            print(f"[PUBLISH] created {CHECKPOINT_DATASET_ID}")
        else:
            subprocess.run(["kaggle", "datasets", "version", "-p", save_dir, "-m", message, "-q", "-r", "tar"],
                            check=True, capture_output=True, text=True)
            print(f"[PUBLISH] versioned {CHECKPOINT_DATASET_ID}: {message}")
    except subprocess.CalledProcessError as e:
        # Non-fatal by design: the LOCAL checkpoint in /kaggle/working is the
        # real safety net (recoverable via `kaggle kernels output` even after
        # a hard cancellation) -- a failed dataset publish shouldn't crash
        # training over a resumability nicety.
        print(f"[WARN] checkpoint dataset publish failed (local checkpoint is still safe): "
              f"{e.stderr or e.stdout or e}")


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
    print(f"item_max_length:    {ITEM_MAX_LENGTH}  (prefix={PREFIX!r})")
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

    print("[TOKENIZER] Loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    cls_id, sep_id, pad_id = tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id
    print(f"[TOKENIZER] class={type(tokenizer).__name__} vocab_size={tokenizer.vocab_size} "
          f"cls={cls_id} sep={sep_id} pad={pad_id}")

    # -------------------------------------------------------------
    # Canonical human-only split (byte-identical to the rest of this repo:
    # filter nulls -> label = target>=0.5 -> seed-42 stratified 80/20).
    # -------------------------------------------------------------
    matches_path = _find_dataset_file(MATCHES_PATH_HINT, "matches.parquet")
    print(f"[DATA] Loading {matches_path} for canonical split ...")
    matches_pl = pl.read_parquet(matches_path)
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
    # Phase 1: tokenize (in-process, see module docstring)
    # -------------------------------------------------------------
    (item_input_ids_mm, item_lengths, category_label_start,
     attr_label_start, compact_lengths, id_to_row) = build_item_cache(tokenizer)
    _log_memory("after item cache build")

    # -------------------------------------------------------------
    # items.parquet -- ONLY for category
    # -------------------------------------------------------------
    items_path = _find_dataset_file(ITEMS_HUMAN_PATH_HINT, "items_human.parquet")
    needed_ids = pd.unique(pd.concat([train_pd["id1"], train_pd["id2"], val_pd["id1"], val_pd["id2"]]))
    needed_ids_lazy = pl.DataFrame({"id": pd.Series(needed_ids, dtype="int64")}).lazy()
    print(f"[DATA] unique item ids needed: {len(needed_ids):,}")

    items_cat = (
        pl.scan_parquet(items_path)
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
        category_label_start, attr_label_start, cls_id, sep_id, pad_id, PAIR_MAX_LENGTH,
        micro_batch_size, shuffle=True,
    )
    val_loader = make_pair_loader(
        val_pd, id_to_row, category_to_id, item_input_ids_mm, item_lengths, compact_lengths,
        category_label_start, attr_label_start, cls_id, sep_id, pad_id, PAIR_MAX_LENGTH,
        eval_batch_size, shuffle=False,
    )

    pos_rate = float(train_pd["label"].mean())
    print(f"[DATA] train positive rate: {pos_rate:.4f}")

    # -------------------------------------------------------------
    # Model: fresh LoRA init, UNLESS a previous checkpoint is found (resume)
    # -------------------------------------------------------------
    checkpoint_dir = _find_checkpoint_dir()
    resumed = checkpoint_dir is not None
    if resumed:
        model = load_model_from_checkpoint(device, os.path.join(checkpoint_dir, "adapter"),
                                            os.path.join(checkpoint_dir, "classifier.pt"))
        with open(os.path.join(checkpoint_dir, "train_state.json")) as f:
            resume_state = json.load(f)
        print(f"[RESUME] resuming from epoch={resume_state['completed_epochs']} "
              f"global_step={resume_state['global_step']} "
              f"best_val_macro_pr_auc={resume_state['best_val_macro_pr_auc']}")
    else:
        model = build_fresh_model(device)
        resume_state = {"completed_epochs": 0, "global_step": 0, "best_val_macro_pr_auc": -1.0,
                          "n_skipped_microbatches": 0, "n_skipped_optimizer_steps": 0}
        print("[RESUME] no checkpoint found -- starting fresh")

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

    # Never probe past TOTAL_BATCH_SIZE for TRAINING: grad_accum = max(1,
    # total//micro) floors at 1, so a micro-batch bigger than TOTAL_BATCH_SIZE
    # buys nothing (no fewer optimizer steps than micro==TOTAL_BATCH_SIZE
    # already gives) while pushing needlessly close to the VRAM ceiling --
    # exactly how a real run picked MICRO_BATCH_SIZE=320 against
    # TOTAL_BATCH_SIZE=256 and OOM'd on step 0 for no schedule benefit at all.
    micro_batch_size = run_vram_probe(model, trainable_params, device, n_gpu, PAIR_MAX_LENGTH, micro_batch_size,
                                       probe_cap=min(PROBE_CAP, TOTAL_BATCH_SIZE), train_mode=True,
                                       label="MICRO_BATCH_SIZE")
    grad_accum = max(1, total_batch_size // micro_batch_size)

    # Separate eval probe (no cap tied to TOTAL_BATCH_SIZE -- that constraint
    # is about grad-accum/schedule semantics, irrelevant to eval, which never
    # calls backward()). A real run measured training-with-gradients safely
    # fitting batch=256 at 10.14GB; eval's true footprint (forward-only,
    # no_grad) is meaningfully smaller per example, so reusing the training
    # probe's number would leave real eval throughput on the table -- a real
    # run's eval pass took ~14 minutes per checkpoint at the old fixed
    # EVAL_MICRO_BATCH_SIZE=32 (~100 checkpoints planned across 20 epochs =>
    # eval alone would have dominated total wall-clock time).
    model.eval()
    eval_batch_size = run_vram_probe(model, trainable_params, device, n_gpu, PAIR_MAX_LENGTH, eval_batch_size,
                                      probe_cap=PROBE_CAP, train_mode=False, label="EVAL_MICRO_BATCH_SIZE")
    model.train()

    print(f"[PROBE] Rebuilding data loaders with the new batch size ...")
    train_loader = make_pair_loader(
        train_pd, id_to_row, category_to_id, item_input_ids_mm, item_lengths, compact_lengths,
        category_label_start, attr_label_start, cls_id, sep_id, pad_id, PAIR_MAX_LENGTH,
        micro_batch_size, shuffle=True,
    )
    val_loader = make_pair_loader(
        val_pd, id_to_row, category_to_id, item_input_ids_mm, item_lengths, compact_lengths,
        category_label_start, attr_label_start, cls_id, sep_id, pad_id, PAIR_MAX_LENGTH,
        eval_batch_size, shuffle=False,
    )
    print(f"[PROBE] New config: MICRO_BATCH_SIZE={micro_batch_size} | EVAL_BATCH_SIZE={eval_batch_size} | "
          f"GRAD_ACCUM={grad_accum} | effective TOTAL_BATCH_SIZE={micro_batch_size * grad_accum}")

    steps_per_epoch = math.ceil(len(train_loader) / grad_accum)
    total_steps = steps_per_epoch * EPOCHS
    warmup_steps = max(1, int(total_steps * WARMUP_RATIO))
    checkpoint_interval = max(1, steps_per_epoch // N_CHECKPOINTS_PER_EPOCH)
    print(f"[TRAIN] steps/epoch: {steps_per_epoch} | total steps: {total_steps} | "
          f"warmup: {warmup_steps} ({WARMUP_RATIO:.0%}) -> cosine decay to 0")
    print(f"[TRAIN] checkpoint every {checkpoint_interval} optimizer steps "
          f"(~{N_CHECKPOINTS_PER_EPOCH}/epoch), publish_to_dataset={ENABLE_DATASET_PUSH and has_kaggle_creds}")

    pos_weight = torch.tensor((1.0 - pos_rate) / max(pos_rate, 1e-9), device=device)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    print(f"[LOSS] BCE pos_weight: {pos_weight.item():.4f}")

    optimizer = torch.optim.AdamW(trainable_params, lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = get_scheduler("cosine", optimizer=optimizer, num_warmup_steps=warmup_steps,
                               num_training_steps=total_steps)

    if resumed:
        optimizer.load_state_dict(torch.load(os.path.join(checkpoint_dir, "optimizer.pt"), map_location=device))
        scheduler.load_state_dict(torch.load(os.path.join(checkpoint_dir, "scheduler.pt")))
        print("[RESUME] optimizer + scheduler state restored")

    global_step = resume_state["global_step"]
    best_val_macro_pr_auc = resume_state["best_val_macro_pr_auc"]
    n_skipped_microbatches = resume_state["n_skipped_microbatches"]
    n_skipped_optimizer_steps = resume_state["n_skipped_optimizer_steps"]
    start_epoch = resume_state["completed_epochs"]

    category_names = category_vocab

    def run_checkpoint(epoch_display, is_epoch_end=False):
        nonlocal best_val_macro_pr_auc
        train_metrics = {}
        if window_loss_count[0] > 0:
            train_probs = torch.sigmoid(torch.cat(window_logits)).numpy()
            train_labels_np = torch.cat(window_labels).numpy()
            train_cats_np = np.concatenate(window_categories)
            train_macro_pr_auc, _ = compute_metrics(train_probs, train_labels_np, train_cats_np, category_names)
            train_metrics = {"loss": window_loss_sum[0] / window_loss_count[0], "macro_pr_auc": train_macro_pr_auc}
            window_logits.clear(); window_labels.clear(); window_categories.clear()
            window_loss_sum[0] = 0.0; window_loss_count[0] = 0

        eval_model = model.module if isinstance(model, nn.DataParallel) else model
        val_metrics = evaluate(eval_model, val_loader, loss_fn, device, category_names)
        model.train()

        improved = val_metrics["macro_pr_auc"] > best_val_macro_pr_auc
        if improved:
            best_val_macro_pr_auc = val_metrics["macro_pr_auc"]

        print(f"[METRICS] epoch={epoch_display} step={global_step} "
              f"train_loss={train_metrics.get('loss', float('nan')):.4f} "
              f"train_macro_pr_auc={train_metrics.get('macro_pr_auc', float('nan')):.4f} "
              f"val_loss={val_metrics['loss']:.4f} val_macro_pr_auc={val_metrics['macro_pr_auc']:.4f} "
              f"best={best_val_macro_pr_auc:.4f} skipped_micro={n_skipped_microbatches} "
              f"skipped_opt={n_skipped_optimizer_steps}", flush=True)

        mlflow.log_metrics({
            "train_loss": train_metrics.get("loss", float("nan")),
            "train_macro_pr_auc": train_metrics.get("macro_pr_auc", float("nan")),
            "val_loss": val_metrics["loss"], "val_macro_pr_auc": val_metrics["macro_pr_auc"],
            "n_skipped_microbatches": n_skipped_microbatches,
            "n_skipped_optimizer_steps": n_skipped_optimizer_steps,
        }, step=global_step)

        completed_epochs_for_save = epoch_display if is_epoch_end else start_epoch
        save_checkpoint(model, optimizer, scheduler, completed_epochs_for_save, global_step,
                         best_val_macro_pr_auc, n_skipped_microbatches, n_skipped_optimizer_steps,
                         CHECKPOINT_DIR)
        publish_checkpoint_dataset(CHECKPOINT_DIR, f"epoch={epoch_display} step={global_step} "
                                    f"val_macro_pr_auc={val_metrics['macro_pr_auc']:.4f}", has_kaggle_creds)
        if improved:
            print(f"[SAVE] new best val_macro_pr_auc={best_val_macro_pr_auc:.4f}", flush=True)

    with mlflow.start_run():
        mlflow.log_params({
            "model": MODEL_NAME, "pair_max_length": PAIR_MAX_LENGTH, "epochs": EPOCHS, "lr": LR,
            "lora_r": LORA_R, "lora_alpha": LORA_ALPHA, "lora_dropout": LORA_DROPOUT,
            "lora_target_modules": ",".join(LORA_TARGET_MODULES), "total_batch_size": total_batch_size,
            "micro_batch_size": micro_batch_size, "val_fingerprint": val_fingerprint, "resumed": resumed,
        })

        window_logits, window_labels, window_categories = [], [], []
        window_loss_sum, window_loss_count = [0.0], [0]

        for epoch in range(start_epoch, EPOCHS):
            model.train()
            optimizer.zero_grad()
            pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}", dynamic_ncols=True)
            print(f"[EPOCH] starting epoch {epoch + 1}/{EPOCHS}", flush=True)

            for step, batch in enumerate(pbar):
                input_ids, attention_mask, token_type_ids, batch_labels, batch_categories = batch
                input_ids = input_ids.to(device, non_blocking=True)
                attention_mask = attention_mask.to(device, non_blocking=True)
                token_type_ids = token_type_ids.to(device, non_blocking=True)
                batch_labels_dev = batch_labels.to(device, non_blocking=True)

                # autocast happens INSIDE BertaCrossEncoderClassifier.forward()
                # now, not wrapped here (see that class for why).
                logits = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
                loss = loss_fn(logits, batch_labels_dev)

                if not torch.isfinite(loss):
                    n_skipped_microbatches += 1
                    print(f"[WARN] step={step} non-finite loss ({loss.item()!r}) -- skipping this microbatch "
                          f"(total skipped so far: {n_skipped_microbatches})", flush=True)
                    # Critical: no .backward() runs on this branch, so the
                    # forward activation graph is NOT freed by autograd like
                    # it normally would be -- del it explicitly (see
                    # frida-lora-sft's own history for exactly what happens
                    # if this is skipped: a real CUDA OOM one step later).
                    del logits, loss
                    torch.cuda.empty_cache()
                else:
                    loss_item = loss.item()
                    (loss / grad_accum).backward()

                    window_logits.append(logits.detach().float().cpu())
                    window_labels.append(batch_labels.detach().cpu())
                    window_categories.append(batch_categories)
                    window_loss_sum[0] += loss_item
                    window_loss_count[0] += 1
                    pbar.set_postfix(loss=round(loss_item, 5), lr=f"{scheduler.get_last_lr()[0]:.2e}", step=global_step)

                need_step = ((step + 1) % grad_accum == 0) or ((step + 1) == len(train_loader))
                if need_step:
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
                        run_checkpoint(epoch + 1 if is_epoch_end else epoch, is_epoch_end=is_epoch_end)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(f"[DONE] training complete. best_val_macro_pr_auc={best_val_macro_pr_auc:.4f}")


if __name__ == "__main__":
    main()
