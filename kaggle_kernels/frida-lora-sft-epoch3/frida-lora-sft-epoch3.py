"""
LoRA SFT of ai-forever/FRIDA (T5EncoderModel, 823M) as a cross-encoder --
THIRD session in this lineage, continuing frida-lora-sft-epoch2's
checkpoint (which itself continued frida-lora-sft-human-only-10-epochs's).
Same architecture/data/text-budget decisions as those two scripts -- see
their own module docstrings and kaggle_kernels/frida-lora-sft-human-only-
10-epochs/README.md for the full rationale (item-level pre-tokenized
cache, PAIR_MAX_LENGTH=128, Category excision, bf16 + eager attention +
autocast-inside-forward() to avoid the DataParallel NaN trap diagnosed in
kaggle_kernels/frida-nan-diagnose). NOT launched by this commit -- prepared
for review, GPU sessions need explicit go-ahead first (two Kaggle accounts,
each with its own weekly quota -- see CLAUDE.md).

PREREQUISITE this script assumes but does not create: epoch 2's checkpoint
(val_macro_pr_auc=0.67397 @ step 458/1143, WITH optimizer/scheduler state)
published as a new version of the maxamsecret/frida-lora dataset (the same
ref epoch 2 itself resumed from -- this repo's own convention of one
canonical "latest FRIDA LoRA checkpoint" dataset ref, versioned forward,
mirroring ozon-lora-minilm-finetuned). If it ends up published under a
different ref, update DATASET_SOURCES in this kernel's kernel-metadata.json
before pushing -- the resume logic below just walks /kaggle/input for
whatever is actually mounted, so the ref name itself doesn't matter to the
code, only to kernel-metadata.json's dataset_sources.

===========================================================================
EVAL BOTTLENECK -- why neither of the first two sessions finished 1 epoch
===========================================================================
Both frida-lora-sft-human-only-10-epochs and frida-lora-sft-epoch2 hit the
hard 12h Kaggle session limit well before completing even one pass over the
292,523-row train split (60% and ~70% of one epoch respectively -- see
their own READMEs for the exact per-checkpoint numbers). Root cause,
confirmed independently two ways (this repo's own log-timestamp analysis,
and the team's separate real-time observation while it ran): EVAL, not
TRAINING, dominated session wall-clock.

Measured from frida-lora-sft-epoch2's actual Kaggle log: one full eval pass
over all 73,131 val pairs (EVAL_MICRO_BATCH_SIZE=64) took ~6.46s/iteration
x 1143 iterations =~ 2.05h. The gap between consecutive checkpoints (229
train steps + 1 eval pass) was ~4.19h wall-clock -- i.e. eval alone was
~49% of every checkpoint's wall-clock, roughly AS EXPENSIVE as the 229
training steps between checkpoints, despite eval being forward-only (no
backward pass) on a smaller batch. At N_CHECKPOINTS_PER_EPOCH=5 (eval every
20% of an epoch), one full epoch's worth of eval overhead alone is close to
5 x 2.05h =~ 10.25h -- almost the entire 12h budget, before any training
gets credited. This is NOT a training-throughput problem; it is an
eval-frequency/eval-size problem.

Fix applied in this script (see EVAL_SUBSAMPLE_FRAC and
N_CHECKPOINTS_PER_EPOCH below): evaluate interim (mid-run) checkpoints on a
FIXED random ~25% subsample of val instead of the full 73,131, and reduce
checkpoint frequency from 5/epoch to 3/epoch. Combined, this cuts eval
overhead from ~10.25h/epoch to roughly 3 x (2.05h x 0.25) =~ 1.5h/epoch --
freeing most of a 12h session back for actual training. A single FULL-val
evaluation still runs once, unconditionally, right after the training loop
ends (see the end of main()) -- so if the loop actually completes within
the session, the final reported number is the real, full-val macro
PR-AUC, directly comparable to epoch 1/2's 0.652/0.674 (both full-val
numbers), not a subsample estimate. If the session gets CANCEL_ACKNOWLEDGED
again before the loop finishes, that final full-val step simply never runs
(process is killed) -- exactly as with any other post-loop code -- but the
interim subsample-based "best" checkpoints saved along the way are still
real, recoverable checkpoints, same guarantee as before.

The subsample is drawn ONCE per run (fixed SEED) and reused for every
interim checkpoint within that run -- not re-sampled each time -- so
"new best" comparisons between checkpoints stay apples-to-apples within a
single session (a different random subsample each time would make
best-tracking noisy). It does NOT need to match across sessions (epoch 2's
own run used the full val throughout, no subsample) -- only internal
consistency within this run matters for its own best-tracking; the final
full-val number is what actually gets compared against the other epochs.
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

from peft import LoraConfig, PeftModel, get_peft_model


# =====================================================================
# CONFIG
# =====================================================================

SEED = 42
MODEL_NAME = "ai-forever/FRIDA"

ITEMS_PATH = "/kaggle/input/datasets/maxamsecret/ozon-ecup-2026-matching/items.parquet"
MATCHES_PATH = "/kaggle/input/datasets/maxamsecret/ozon-ecup-2026-matching/matches.parquet"
# Pre-built item-level token cache. Mount path searched at runtime, not
# hardcoded -- Kaggle dataset mount paths aren't 100% predictable.
ITEM_CACHE_HINT = "/kaggle/input/datasets/maxamsecret/frida-tokenized-items-human"
ITEM_CACHE_N_ITEMS = 711_304    # from the published cache_meta.json, cross-
                                 # checked against file sizes below at runtime
ITEM_MAX_LENGTH = 256           # from the published cache_meta.json

PAIR_MAX_LENGTH = 128           # Same deliberate cost/quota choice as
                                 # epoch 1/2 -- see their READMEs for the
                                 # full token-budget-sizing numbers.

MAX_CATEGORY_CHARS = 64   # only category is read from items.parquet now (the
                           # cache already carries name/attributes as tokens)

PREFIX = "paraphrase: "   # matches the cache's own cache_meta.json["prefix"]
                           # -- documented here for readability, not re-applied

EPOCHS = 1   # Must stay 1 -- a resumed optimizer/scheduler's internal state
             # (see the resume block in main()) was built against
             # total_training_steps computed from EPOCHS=1 in epoch 2's own
             # run. Changing EPOCHS here would recompute a DIFFERENT
             # total_training_steps and desync the resumed cosine schedule
             # from where it actually is. If epoch 2's checkpoint being
             # resumed does NOT carry optimizer/scheduler state (e.g. the
             # published dataset ended up being epoch 1's instead), this
             # constraint doesn't apply -- but check train_state.json in
             # whatever gets mounted before assuming either way.

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.10
LORA_TARGET_MODULES = ["q", "k", "v", "o", "wi_0", "wi_1", "wo"]

LR = 3e-4                          # peak LR -- unchanged from epoch 1/2.
                                    # Irrelevant to a real optimizer/
                                    # scheduler resume anyway (LR comes from
                                    # the resumed scheduler's own state, not
                                    # rebuilt from this constant) -- kept
                                    # only as the fallback for a weights-only
                                    # resume (no optimizer.pt/scheduler.pt
                                    # found).
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.10                 # linear warmup to peak, then cosine decay
N_CHECKPOINTS_PER_EPOCH = 3         # was 5 in epoch 1/2 -- see the module
                                     # docstring's "EVAL BOTTLENECK" section.
                                     # Fewer checkpoints AND each one cheaper
                                     # (see EVAL_SUBSAMPLE_FRAC) is the fix.

EVAL_SUBSAMPLE_FRAC = 0.25          # Interim (mid-run) checkpoints evaluate
                                     # on a FIXED random ~25% subsample of
                                     # val (~18,283 of 73,131 pairs), drawn
                                     # once with SEED and reused for every
                                     # interim checkpoint in this run --
                                     # keeps "new best" comparisons
                                     # consistent within the run. A full-val
                                     # (73,131) evaluation still runs once,
                                     # unconditionally, right after the
                                     # training loop -- see module docstring.

MICRO_BATCH_SIZE = 32                # starting point for the VRAM probe --
                                      # epoch 2's own probe landed on 32 for
                                      # this exact config, used as the
                                      # starting guess here too (skips
                                      # re-discovering the same number via
                                      # the doubling search, though the probe
                                      # still runs and will correct it if the
                                      # actual GPU/session differs).
EVAL_MICRO_BATCH_SIZE = 64
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
# tokens-human dataset (or its frida-tokenized-items-human twin), does not
# tokenize anything itself.
# =====================================================================

CATEGORY_LABEL_TOKEN = 55917    # " Category" (+ colon=30 immediately after)
ATTRIBUTES_LABEL_TOKEN = 91837  # " Attributes" (+ colon=30 immediately after)
LABEL_COLON_TOKEN = 30


def _find_token_pattern_starts(arr, tok_a, tok_b):
    """First index i per row where arr[row,i]==tok_a and arr[row,i+1]==tok_b,
    or -1 if not found. Vectorized (no Python loop over rows)."""
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
    attr_label_start, compact_lengths, id_to_row)."""
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

    print("[CACHE] precomputing per-item real lengths ...")
    item_lengths = np.asarray(item_attention_mask).sum(axis=1).astype(np.int32)
    print(f"[CACHE] loaded {n_items:,} items, real length mean={item_lengths.mean():.1f} "
          f"p50={np.percentile(item_lengths, 50):.0f} p90={np.percentile(item_lengths, 90):.0f} "
          f"max={item_lengths.max()} (cap={max_length})")

    sample_rows = np.random.RandomState(SEED).choice(n_items, size=min(50, n_items), replace=False)
    for r in sample_rows:
        length = int(item_lengths[r])
        assert item_input_ids[r, 0] == 1, f"row {r}: expected bos=1 at position 0, got {item_input_ids[r, 0]}"
        assert item_input_ids[r, length - 1] == 2, f"row {r}: expected eos=2 at position {length - 1}, got {item_input_ids[r, length - 1]}"
    print(f"[CACHE] bos/eos sanity check passed on {len(sample_rows)} random rows")

    print("[CACHE] locating Category:/Attributes: label positions ...")
    arr = np.asarray(item_input_ids)
    category_label_start = _find_token_pattern_starts(arr, CATEGORY_LABEL_TOKEN, LABEL_COLON_TOKEN)
    attr_label_start = _find_token_pattern_starts(arr, ATTRIBUTES_LABEL_TOKEN, LABEL_COLON_TOKEN)
    both_found = (category_label_start >= 0) & (attr_label_start >= 0)
    n_missing = n_items - int(both_found.sum())
    if n_missing > 0:
        print(f"[CACHE] WARNING: {n_missing:,}/{n_items:,} items missing a locatable Category:/"
              f"Attributes: label. Falling back to no-excision (full item kept) for those.")
    category_span = np.where(both_found, attr_label_start - category_label_start, 0).astype(np.int32)
    compact_lengths = item_lengths - category_span
    print(f"[CACHE] category span excised: mean={category_span[both_found].mean():.1f} tokens/item "
          f"| compact length: mean={compact_lengths.mean():.1f} p50={np.percentile(compact_lengths, 50):.0f} "
          f"(was {item_lengths.mean():.1f}/{np.percentile(item_lengths, 50):.0f})")

    return item_input_ids, item_lengths, category_label_start, attr_label_start, compact_lengths, id_to_row


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


def allocate_pair_budget(len1, len2, budget):
    """Fair-share truncation over COMPACT (category-excised) lengths."""
    half = budget // 2
    if len1 <= half and len2 <= half:
        return len1, len2
    if len1 <= half:
        return len1, budget - len1
    if len2 <= half:
        return budget - len2, len2
    return half, budget - half


def build_compact_ids(item_input_ids_mm, row, item_length, category_start, attr_start):
    if category_start < 0 or attr_start < 0:
        return item_input_ids_mm[row, :item_length]
    head = item_input_ids_mm[row, :category_start]
    tail = item_input_ids_mm[row, attr_start:item_length]
    return np.concatenate((head, tail))


def make_pair_collate_fn(item_input_ids_mm, item_lengths, compact_lengths, category_label_start,
                          attr_label_start, pad_id, pair_max_length):
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
    """CLS (position-0) pooling. autocast lives INSIDE forward(), not
    wrapped around calls to this module from the outside -- see
    kaggle_kernels/frida-nan-diagnose for why (DataParallel replica threads
    don't see an outer-entered autocast context)."""

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
    print(f"[MODEL] Loading base encoder {MODEL_NAME} (T5EncoderModel explicitly) ...")
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


def _find_resume_adapter_dir():
    """Searches /kaggle/input for a directory containing adapter_config.json
    -- expected to be epoch 2's published checkpoint this time (see module
    docstring). Returns None if not found -- a fresh LoRA init, which would
    be a mistake for this kernel (it exists specifically to continue, not
    restart) -- treat a None here as a sign dataset_sources isn't set up
    correctly rather than proceeding silently."""
    if os.path.isdir("/kaggle/input"):
        for root, _dirs, files in os.walk("/kaggle/input"):
            if "adapter_config.json" in files:
                print(f"[RESUME] found adapter checkpoint at {root}")
                return root
    return None


def build_model_from_checkpoint(device, adapter_dir):
    print(f"[MODEL] Loading base encoder {MODEL_NAME} + resuming adapter from {adapter_dir} ...")
    try:
        base_model = T5EncoderModel.from_pretrained(MODEL_NAME, attn_implementation="eager")
    except (TypeError, ValueError):
        base_model = T5EncoderModel.from_pretrained(MODEL_NAME)
    hidden_dim = base_model.config.d_model

    peft_model = PeftModel.from_pretrained(base_model, adapter_dir, is_trainable=True)
    peft_model.print_trainable_parameters()

    n_trainable_lora = sum(p.numel() for n, p in peft_model.named_parameters() if "lora_" in n and p.requires_grad)
    assert n_trainable_lora > 0, "Resumed LoRA adapter has zero trainable params -- is_trainable didn't take."
    print(f"[MODEL] trainable LoRA params: {n_trainable_lora:,}")

    classifier_head = nn.Linear(hidden_dim, 1)
    classifier_path = os.path.join(adapter_dir, "classifier.pt")
    classifier_head.load_state_dict(torch.load(classifier_path, map_location="cpu"))

    model = FridaCrossEncoderClassifier(peft_model, classifier_head).to(device)
    return model


# =====================================================================
# VRAM AUTO-PROBE
# =====================================================================

def _probe_peak_gb(batch_size, n_gpu, pair_max_length, probe_model, probe_loss_fn, trainable_params, device):
    dummy_input_ids = torch.randint(0, 1000, (batch_size, pair_max_length), dtype=torch.long, device=device)
    dummy_attention_mask = torch.ones((batch_size, pair_max_length), dtype=torch.long, device=device)
    dummy_labels = torch.randint(0, 2, (batch_size,), dtype=torch.float32, device=device)

    for p in trainable_params:
        p.grad = None
    for i in range(max(n_gpu, 1)):
        torch.cuda.reset_peak_memory_stats(i)

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
# CRASH-RESILIENT CHECKPOINT PUSH
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


def save_checkpoint(model, tokenizer, save_dir, optimizer=None, scheduler=None, global_step=None,
                     n_skipped_microbatches=None, n_skipped_optimizer_steps=None, best_macro=None):
    os.makedirs(save_dir, exist_ok=True)
    raw = model.module if isinstance(model, nn.DataParallel) else model
    raw.encoder.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    torch.save(raw.classifier.state_dict(), os.path.join(save_dir, "classifier.pt"))
    if optimizer is not None:
        torch.save(optimizer.state_dict(), os.path.join(save_dir, "optimizer.pt"))
    if scheduler is not None:
        torch.save(scheduler.state_dict(), os.path.join(save_dir, "scheduler.pt"))
    if global_step is not None:
        with open(os.path.join(save_dir, "train_state.json"), "w") as f:
            json.dump({
                "global_step": global_step,
                "n_skipped_microbatches": n_skipped_microbatches,
                "n_skipped_optimizer_steps": n_skipped_optimizer_steps,
                "best_macro_pr_auc": best_macro,
            }, f, indent=2)


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
    print(f"model:                {MODEL_NAME}")
    print(f"item_max_length:      {ITEM_MAX_LENGTH}  (prefix={PREFIX!r}, from pre-built cache)")
    print(f"pair_max_length:      {PAIR_MAX_LENGTH}")
    print(f"epochs:               {EPOCHS}")
    print(f"lr:                   {LR}  (peak, ignored if optimizer/scheduler state resumes)")
    print(f"lora r/alpha/drop:    {LORA_R}/{LORA_ALPHA}/{LORA_DROPOUT}")
    print(f"lora target_modules:  {LORA_TARGET_MODULES}")
    print(f"total_batch_size:     {TOTAL_BATCH_SIZE}")
    print(f"micro_batch_size:     {MICRO_BATCH_SIZE} (probe start)")
    print(f"n_checkpoints/epoch:  {N_CHECKPOINTS_PER_EPOCH}  (was 5 in epoch 1/2 -- see module docstring)")
    print(f"eval_subsample_frac:  {EVAL_SUBSAMPLE_FRAC:.0%}  (interim checkpoints only; full val once at the end)")
    print(f"output_dir:           {OUTPUT_DIR}")
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

    print("[TOKENIZER] Loading tokenizer (for pad_token_id only) ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    pad_id = tokenizer.pad_token_id
    print(f"[TOKENIZER] pad={pad_id} (bos/eos already baked into the item cache, not re-added here)")

    # -------------------------------------------------------------
    # Canonical human-only split (byte-identical to the rest of this repo).
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

    # Fixed subsample for interim (mid-run) checkpoint evals -- see module
    # docstring "EVAL BOTTLENECK". Drawn once, reused for every interim
    # checkpoint in this run.
    val_pd_subsample = val_pd.sample(frac=EVAL_SUBSAMPLE_FRAC, random_state=SEED).reset_index(drop=True)
    print(f"[DATA] eval subsample for interim checkpoints: {len(val_pd_subsample):,}/{len(val_pd):,} "
          f"({EVAL_SUBSAMPLE_FRAC:.0%}) -- the full {len(val_pd):,} is only evaluated once, after training ends")

    del matches_pd, labels_all, indices, train_idx, val_idx
    gc.collect()

    # -------------------------------------------------------------
    # Load the pre-built item token cache.
    # -------------------------------------------------------------
    cache_dir = _find_item_cache_dir(ITEM_CACHE_HINT)
    print(f"[CACHE] loading item token cache from {cache_dir} ...")
    (item_input_ids_mm, item_lengths, category_label_start, attr_label_start,
     compact_lengths, id_to_row) = load_item_cache(cache_dir)
    _log_memory("after item cache load")

    # -------------------------------------------------------------
    # items.parquet -- ONLY for category.
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
    val_pd_subsample["category1"] = val_pd_subsample["id1"].map(id_to_category).fillna("UNKNOWN")
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
    val_loader_subsample = make_pair_loader(
        val_pd_subsample, id_to_row, category_to_id, item_input_ids_mm, item_lengths, compact_lengths,
        category_label_start, attr_label_start, pad_id, PAIR_MAX_LENGTH, eval_batch_size, shuffle=False,
    )

    pos_rate = float(train_pd["label"].mean())
    print(f"[DATA] train positive rate: {pos_rate:.4f}")

    # -------------------------------------------------------------
    # Model: MUST resume from a checkpoint -- this kernel exists to
    # continue, not to start fresh. See _find_resume_adapter_dir's own
    # docstring for what a None here means.
    # -------------------------------------------------------------
    resume_adapter_dir = _find_resume_adapter_dir()
    if resume_adapter_dir:
        model = build_model_from_checkpoint(device, resume_adapter_dir)
    else:
        print("[RESUME] WARNING: no adapter checkpoint found in /kaggle/input -- "
              "this kernel expects one (see kernel-metadata.json dataset_sources). "
              "Falling back to a fresh LoRA init, which is almost certainly NOT what "
              "was intended for an 'epoch 3' continuation run.")
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
        val_loader_subsample = make_pair_loader(
            val_pd_subsample, id_to_row, category_to_id, item_input_ids_mm, item_lengths, compact_lengths,
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
    print(f"[TRAIN] interim checkpoint every {checkpoint_interval} optimizer steps (~{N_CHECKPOINTS_PER_EPOCH}/epoch), "
          f"each evaluated on the {EVAL_SUBSAMPLE_FRAC:.0%} val subsample -- full val runs once at the very end.")

    mlflow.start_run(run_name="lora_frida_sft_epoch3")
    mlflow.log_params({
        "variant": "lora_frida_sft_epoch3", "encoder": MODEL_NAME,
        "item_max_length": ITEM_MAX_LENGTH, "pair_max_length": PAIR_MAX_LENGTH, "prefix": PREFIX,
        "lora_r": LORA_R, "lora_alpha": LORA_ALPHA, "lora_dropout": LORA_DROPOUT,
        "lora_target_modules": ",".join(LORA_TARGET_MODULES),
        "micro_batch_size": micro_batch_size, "eval_batch_size": eval_batch_size,
        "gradient_accumulation_steps": grad_accum, "effective_total_batch_size": total_batch_size,
        "lr": LR, "scheduler": "cosine", "warmup_ratio": WARMUP_RATIO, "weight_decay": WEIGHT_DECAY,
        "epochs": EPOCHS, "split_seed": SEED, "val_fingerprint": val_fingerprint,
        "train_rows": len(train_pd), "val_rows": len(val_pd),
        "n_checkpoints_per_epoch": N_CHECKPOINTS_PER_EPOCH,
        "eval_subsample_frac": EVAL_SUBSAMPLE_FRAC,
        "item_cache": "maxamsecret/frida-tokenized-items-human",
        "resumed_from": resume_adapter_dir or "",
    })

    log_rows = []
    best_macro = -1.0
    global_step = 0
    window_logits, window_labels, window_categories = [], [], []
    window_loss_sum, window_loss_count = 0.0, 0
    n_skipped_microbatches = 0
    n_skipped_optimizer_steps = 0

    if resume_adapter_dir:
        optimizer_state_path = os.path.join(resume_adapter_dir, "optimizer.pt")
        scheduler_state_path = os.path.join(resume_adapter_dir, "scheduler.pt")
        train_state_path = os.path.join(resume_adapter_dir, "train_state.json")
        if os.path.isfile(optimizer_state_path) and os.path.isfile(scheduler_state_path):
            optimizer.load_state_dict(torch.load(optimizer_state_path, map_location=device))
            scheduler.load_state_dict(torch.load(scheduler_state_path))
            print(f"[RESUME] optimizer + scheduler state restored from {resume_adapter_dir}")
            if os.path.isfile(train_state_path):
                with open(train_state_path) as f:
                    resumed_state = json.load(f)
                global_step = resumed_state.get("global_step", 0)
                n_skipped_microbatches = resumed_state.get("n_skipped_microbatches", 0) or 0
                n_skipped_optimizer_steps = resumed_state.get("n_skipped_optimizer_steps", 0) or 0
                best_macro = resumed_state.get("best_macro_pr_auc", -1.0)
                if best_macro is None:
                    best_macro = -1.0
                print(f"[RESUME] step/skip counters + best_macro_pr_auc={best_macro:.5f} restored "
                      f"from train_state.json (global_step={global_step}). NOTE: that best_macro was "
                      f"measured on FULL val by epoch 2's run -- this run's interim best-tracking uses "
                      f"the {EVAL_SUBSAMPLE_FRAC:.0%} subsample instead, so it is not perfectly "
                      f"apples-to-apples until the final full-val eval at the end of this run.")
        else:
            print(f"[RESUME] {resume_adapter_dir} has weights only (no optimizer.pt/scheduler.pt) -- "
                  f"warm-started from those weights, optimizer/scheduler/step counters start fresh. "
                  f"(Unexpected for this kernel -- epoch 2's checkpoint should have carried them; "
                  f"double check which checkpoint actually got mounted.)")

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
        # Interim checkpoint -- subsample, not full val. See module docstring.
        val_metrics = evaluate(eval_model, val_loader_subsample, loss_fn, device)
        model.train()

        progress = global_step / total_training_steps
        row = {
            "global_step": global_step, "progress": round(progress, 4),
            "train_loss": train_window_loss, "train_macro_pr_auc": train_macro, "train_pos_pr_auc": train_pos_ap,
            "val_loss": val_metrics["val_loss"], "val_macro_pr_auc": val_metrics["macro_pr_auc"],
            "val_pos_pr_auc": val_metrics["pos_pr_auc"], "lr": scheduler.get_last_lr()[0],
            "n_skipped_microbatches": n_skipped_microbatches, "n_skipped_optimizer_steps": n_skipped_optimizer_steps,
            "val_is_subsample": True,
        }
        log_rows.append(row)
        pl.DataFrame(log_rows).write_csv(os.path.join(OUTPUT_DIR, "train_log_steps.csv"))
        mlflow.log_metrics({k: v for k, v in row.items() if k not in ("progress", "val_is_subsample")}, step=global_step)

        print("=" * 100)
        print(f"[METRICS step={global_step}/{total_training_steps} epoch_progress={progress:.1%} lr={row['lr']:.2e}] "
              f"TRAIN loss={train_window_loss:.5f} macro_pr_auc={train_macro:.5f} | "
              f"VAL(subsample={EVAL_SUBSAMPLE_FRAC:.0%}) loss={row['val_loss']:.5f} macro_pr_auc={row['val_macro_pr_auc']:.5f} | "
              f"skipped: {n_skipped_microbatches} microbatches, {n_skipped_optimizer_steps} optimizer steps (NaN/Inf guard)")
        print("=" * 100)
        _log_memory(f"checkpoint step={global_step}")

        if not np.isnan(row["val_macro_pr_auc"]) and row["val_macro_pr_auc"] > best_macro:
            best_macro = row["val_macro_pr_auc"]
            save_dir = os.path.join(OUTPUT_DIR, "best")
            save_checkpoint(model, tokenizer, save_dir, optimizer=optimizer, scheduler=scheduler,
                             global_step=global_step, n_skipped_microbatches=n_skipped_microbatches,
                             n_skipped_optimizer_steps=n_skipped_optimizer_steps, best_macro=best_macro)
            print(f"[SAVE] new best val_macro_pr_auc={best_macro:.5f} (subsample) -> {save_dir}")

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

            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(logits, batch_labels_dev)

            if not torch.isfinite(loss):
                n_skipped_microbatches += 1
                print(f"[WARN] step={step} non-finite loss ({loss.item()!r}) -- skipping this microbatch "
                      f"(total skipped so far: {n_skipped_microbatches})", flush=True)
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

        print(f"[EPOCH] finished epoch {epoch + 1}/{EPOCHS} | best val_macro_pr_auc so far={best_macro:.5f} "
              f"(subsample-based until the final full-val eval below)", flush=True)

    # -------------------------------------------------------------
    # Final FULL-val evaluation -- unconditional, runs once, only if the
    # training loop actually completes within this session (if the session
    # gets CANCEL_ACKNOWLEDGED first, this code simply never runs -- exactly
    # like any other post-loop step -- but the interim subsample-based
    # "best" checkpoints saved above are still real). This is what makes the
    # final reported number directly comparable to epoch 1/2's full-val
    # 0.652/0.674, not a subsample estimate. See module docstring.
    # -------------------------------------------------------------
    print(f"[EVAL] running final FULL-val evaluation (all {len(val_pd):,} pairs, "
          f"not the {EVAL_SUBSAMPLE_FRAC:.0%} subsample used for interim checkpoints) ...", flush=True)
    eval_model = model.module if isinstance(model, nn.DataParallel) else model
    final_full_val_metrics = evaluate(eval_model, val_loader, loss_fn, device)
    print(f"[EVAL] FINAL full-val macro_pr_auc={final_full_val_metrics['macro_pr_auc']:.5f} "
          f"(subsample-tracked best during the run: {best_macro:.5f})")
    mlflow.log_metric("final_full_val_macro_pr_auc", final_full_val_metrics["macro_pr_auc"])

    if not np.isnan(final_full_val_metrics["macro_pr_auc"]) and final_full_val_metrics["macro_pr_auc"] > best_macro:
        best_macro = final_full_val_metrics["macro_pr_auc"]
        save_dir = os.path.join(OUTPUT_DIR, "best")
        save_checkpoint(model, tokenizer, save_dir, optimizer=optimizer, scheduler=scheduler,
                         global_step=global_step, n_skipped_microbatches=n_skipped_microbatches,
                         n_skipped_optimizer_steps=n_skipped_optimizer_steps, best_macro=best_macro)
        print(f"[SAVE] final full-val is a new best ({best_macro:.5f}) -> {save_dir}")

    last_dir = os.path.join(OUTPUT_DIR, "last")
    save_checkpoint(model, tokenizer, last_dir, optimizer=optimizer, scheduler=scheduler,
                     global_step=global_step, n_skipped_microbatches=n_skipped_microbatches,
                     n_skipped_optimizer_steps=n_skipped_optimizer_steps, best_macro=best_macro)

    print(f"[DONE] Best macro_pr_auc={best_macro:.5f}")
    print(f"[DONE] Final full-val macro_pr_auc={final_full_val_metrics['macro_pr_auc']:.5f}")
    print(f"[DONE] Step log: {os.path.join(OUTPUT_DIR, 'train_log_steps.csv')}")
    print(f"[DONE] Best model dir: {os.path.join(OUTPUT_DIR, 'best')}")
    print(f"[DONE] Last model dir: {last_dir}")

    mlflow.log_metric("best_macro_pr_auc", best_macro)
    mlflow.log_artifact(os.path.join(OUTPUT_DIR, "train_log_steps.csv"))
    mlflow.end_run()
    print(f"[DONE] mlflow run logged under {MLFLOW_TRACKING_DIR}")


if __name__ == "__main__":
    main()
