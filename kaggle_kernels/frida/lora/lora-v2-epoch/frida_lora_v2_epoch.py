"""
LoRA SFT of ai-forever/FRIDA (T5EncoderModel, 823M) as a cross-encoder --
v2 of the frida-lora-continue-2ep lineage. Resumes from whatever the best
available FRIDA checkpoint is at push time (adapter_config.json search
under /kaggle/input, unchanged convention) and trains a FRESH single
epoch's worth of steps from there (EPOCHS=1 per invocation -- meant to be
launched twice, once per "train an epoch" round, per explicit instruction;
each launch resumes from the PREVIOUS launch's own output). Runs on
maximmarkovbb's separate, clean Kaggle GPU quota.

Six fixes over frida-lora-continue-2ep, all from real observations of that
script's own live run:

1) OUTPUT ALWAYS CARRIES 4 NUMBERS: train_loss, train_macro_pr_auc,
   val_loss, val_macro_pr_auc. Every checkpoint (quiet or full) overwrites
   latest_metrics.json with these (val_* is null for a quiet save -- see
   fix 2) instead of only accumulating inside train_log_steps.csv, so the
   current state is readable from kernel output without parsing a growing
   CSV.

2) TIME-AWARE QUIET CHECKPOINTING. Kaggle's session cap is a hard 12h kill
   with no graceful shutdown hook -- anything not already on disk when it
   fires is gone. Once wall-clock since script start crosses
   QUIET_CHECKPOINT_AFTER_SECONDS (11h), every remaining checkpoint trigger
   in this run skips evaluate() entirely (a full val pass -- exactly the
   thing that's already OOM'd unpredictably once, see fix 5) and saves
   weights+optimizer only. A periodic quiet save (independent of the
   epoch-end trigger from fix 4) also runs every QUIET_SAVE_EVERY_SECONDS
   throughout the WHOLE run, not just past the 11h mark -- an epoch here
   can run several hours, and losing all of it to an unrelated mid-epoch
   crash (fix 3 catches what it can, but "maximally" isn't "certainly")
   would be a bigger loss than a cheap, frequent weights-only save.

3) STEP-LEVEL RESILIENCE. The single-step forward+backward+optimizer.step
   block is now wrapped in try/except for CUDA OOM specifically -- skip the
   microbatch, free what can be freed, keep training. This existed already
   for the checkpoint/eval path (a real incident: a mid-run eval OOM killed
   the whole kernel after real unsaved training progress); it did not yet
   exist for the step loop itself, which runs orders of magnitude more
   often and is exactly where the "quiet save didn't happen in time"
   failure mode would originate.

4) VALIDATION ONLY AT EPOCH END. Was every 0.5 epoch. Combined with fix 2's
   periodic quiet saves, this is a deliberate split: cheap safety
   (weights+optimizer, no eval) stays frequent; expensive, failure-prone
   observability (full val) happens once, at the one point in the run
   ("did this epoch's worth of training help") where it's actually decision
   -relevant.

5) THE REAL "eval OOM -- retrying at chunk_size=512" ROOT CAUSE. The old
   _forward_in_chunks always restarted from the FULL probed eval batch size
   on every new eval batch, so it re-discovered "512 actually works" via a
   fresh halving retry on EVERY batch of EVERY eval pass -- wasted retries,
   not a bug in the halving logic itself. Fixed by making the working
   chunk_size STICKY across an evaluate() call: it starts at the probed
   size once, and after the first OOM-driven reduction, every subsequent
   batch in that same eval pass starts from the size that last actually
   worked (still allowed to shrink further, never grows back within a
   single evaluate() call -- growing back would just reintroduce the same
   OOM it already paid to discover).

6) THE NEW ATTRIBUTE-ORDERING POLICY (see kaggle_kernels/frida-tokenize-
   items-v2's own module docstring for the exact rules and the measured
   oem-key-vs-value deviation) applied at effectively ZERO train-time cost:
   both possible item texts (base-ordered and the 5%-probability shuffled
   variant) are pre-tokenized ONCE by that kernel; here, the collate
   function only draws a coin per item-slot and indexes into one memmap
   array or the other -- no json parsing, no re-tokenization, ever, at
   train time.

Architecture/data mechanics unchanged from the rest of this FRIDA lineage:
item-level pre-tokenized cache (no tokenizer calls at train time),
PAIR_MAX_LENGTH=128, Category:/Attributes: excision, bf16 + eager attention
+ autocast INSIDE forward(), is_trainable=True on resume, best-only
checkpoint saving with optimizer state. See kaggle_kernels/frida-lora-sft-
human-only-10-epochs/README.md for the original rationale.
"""

import os
import sys
import gc
import json
import math
import shutil
import hashlib
import subprocess
import time

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
ITEM_CACHE_HINT = "/kaggle/input/datasets/maximmarkovbb/frida-tokenize-items-v2"
ITEM_MAX_LENGTH = 256

PAIR_MAX_LENGTH = 128
MAX_CATEGORY_CHARS = 64
PREFIX = "paraphrase: "

ATTRIBUTE_SHUFFLE_PROB = 0.05   # per-item-slot draw at collate time -- see
                                  # fix 6 in the module docstring.

EPOCHS = 1                       # per invocation -- see module docstring.

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.10
LORA_TARGET_MODULES = ["q", "k", "v", "o", "wi_0", "wi_1", "wo"]

LR = 3e-4
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.10

MICRO_BATCH_SIZE = 32
EVAL_MICRO_BATCH_SIZE = 64
TOTAL_BATCH_SIZE = 256

TARGET_VRAM_GB = 14.0
SAFETY_MARGIN_GB = 1.5
PROBE_CAP = 512
EVAL_PROBE_CAP = 4096
PROBE_ROUND_TO = 4

# --- Fix 2: time-aware quiet checkpointing ---
QUIET_CHECKPOINT_AFTER_SECONDS = 11 * 3600   # past this, no more full (eval-
                                                # including) checkpoints --
                                                # only weights-only quiet saves.
QUIET_SAVE_EVERY_SECONDS = 1800               # periodic safety save cadence,
                                                # active for the WHOLE run,
                                                # not just past the 11h mark.

ENABLE_DATASET_PUSH = False

WORKING_DIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
OUTPUT_DIR = os.path.join(WORKING_DIR, "lora_frida_sft")
MLFLOW_TRACKING_DIR = os.path.join(WORKING_DIR, "experiments", "mlruns")

t_script_start = time.perf_counter()


def _elapsed():
    return time.perf_counter() - t_script_start


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


def _write_latest_metrics(path, global_step, total_steps, train_loss, train_score, val_loss, val_score,
                           best_score, quiet, elapsed_s):
    """Fix 1: the 4 numbers (train/val x loss/score), always in one small
    file that's overwritten every checkpoint -- readable from kernel output
    without parsing train_log_steps.csv."""
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "global_step": global_step, "total_steps": total_steps,
                "train_loss": train_loss, "train_macro_pr_auc": train_score,
                "val_loss": val_loss, "val_macro_pr_auc": val_score,
                "best_val_macro_pr_auc": best_score, "quiet_save": quiet,
                "elapsed_seconds": elapsed_s,
            }, f, indent=2)
    except Exception as e:
        print(f"[WARN] failed to write {path}: {e}", flush=True)


# =====================================================================
# ITEM-LEVEL TOKEN CACHE -- TWO variants (base + shuffled), see
# kaggle_kernels/frida-tokenize-items-v2 for how they were built.
# =====================================================================

CATEGORY_LABEL_TOKEN = 55917
ATTRIBUTES_LABEL_TOKEN = 91837
LABEL_COLON_TOKEN = 30


def _find_token_pattern_starts(arr, tok_a, tok_b):
    mask = (arr[:, :-1] == tok_a) & (arr[:, 1:] == tok_b)
    rows, cols = np.where(mask)
    uniq_rows, first_idx = np.unique(rows, return_index=True)
    starts = np.full(arr.shape[0], -1, dtype=np.int32)
    starts[uniq_rows] = cols[first_idx]
    return starts


def _find_item_cache_root(hint):
    if os.path.isdir(hint) and os.path.exists(os.path.join(hint, "cache_meta.json")):
        return hint
    if os.path.isdir("/kaggle/input"):
        for root, _dirs, files in os.walk("/kaggle/input"):
            if "cache_meta.json" in files and "variant_base" in _dirs:
                print(f"[CACHE] hint path {hint} didn't have the v2 cache directly -- found it at {root}")
                return root
    return hint


def _load_one_variant(variant_dir, n_items, max_length):
    ids_path = os.path.join(variant_dir, "ids.bin")
    input_ids_path = os.path.join(variant_dir, "input_ids.bin")
    attn_path = os.path.join(variant_dir, "attention_mask.bin")

    expected = {
        "ids.bin": (os.path.getsize(ids_path), n_items * 8),
        "input_ids.bin": (os.path.getsize(input_ids_path), n_items * max_length * 4),
        "attention_mask.bin": (os.path.getsize(attn_path), n_items * max_length),
    }
    for fname, (got, want) in expected.items():
        if got != want:
            raise RuntimeError(f"[CACHE] {variant_dir}/{fname} size mismatch: got {got:,}, expected {want:,}")

    ids = np.memmap(ids_path, dtype=np.int64, mode="r", shape=(n_items,))
    input_ids = np.memmap(input_ids_path, dtype=np.int32, mode="r", shape=(n_items, max_length))
    attn = np.memmap(attn_path, dtype=np.int8, mode="r", shape=(n_items, max_length))

    lengths = np.asarray(attn).sum(axis=1).astype(np.int32)

    sample_rows = np.random.RandomState(SEED).choice(n_items, size=min(50, n_items), replace=False)
    for r in sample_rows:
        L = int(lengths[r])
        assert input_ids[r, 0] == 1, f"{variant_dir} row {r}: expected bos=1, got {input_ids[r, 0]}"
        assert input_ids[r, L - 1] == 2, f"{variant_dir} row {r}: expected eos=2 at {L - 1}, got {input_ids[r, L - 1]}"

    arr = np.asarray(input_ids)
    category_label_start = _find_token_pattern_starts(arr, CATEGORY_LABEL_TOKEN, LABEL_COLON_TOKEN)
    attr_label_start = _find_token_pattern_starts(arr, ATTRIBUTES_LABEL_TOKEN, LABEL_COLON_TOKEN)
    both_found = (category_label_start >= 0) & (attr_label_start >= 0)
    category_span = np.where(both_found, attr_label_start - category_label_start, 0).astype(np.int32)
    compact_lengths = lengths - category_span

    return ids, input_ids, lengths, category_label_start, attr_label_start, compact_lengths


def load_item_cache_v2(cache_root):
    with open(os.path.join(cache_root, "cache_meta.json")) as f:
        meta = json.load(f)
    n_items = meta["n_items"]
    max_length = meta["max_length"]
    print(f"[CACHE] cache_meta.json: n_items={n_items:,} max_length={max_length} "
          f"prefix={meta.get('prefix')!r} attribute_policy={meta.get('attribute_policy')}")
    if max_length != ITEM_MAX_LENGTH:
        raise RuntimeError(f"cache max_length={max_length} != this script's ITEM_MAX_LENGTH={ITEM_MAX_LENGTH}")

    variants = {}
    for name in ("variant_base", "variant_shuffled"):
        t0 = time.perf_counter()
        variants[name] = _load_one_variant(os.path.join(cache_root, name), n_items, max_length)
        print(f"[CACHE] {name} loaded + bos/eos + label-position check OK ({time.perf_counter() - t0:.1f}s)")

    ids_base = variants["variant_base"][0]
    ids_shuf = variants["variant_shuffled"][0]
    if not np.array_equal(np.asarray(ids_base), np.asarray(ids_shuf)):
        raise RuntimeError("[CACHE] variant_base and variant_shuffled have different id orderings -- "
                            "the two caches must share row indices for the collate-time coin-flip to be valid.")
    print("[CACHE] variant_base and variant_shuffled confirmed to share row indices")

    id_to_row = {int(item_id): i for i, item_id in enumerate(np.asarray(ids_base))}
    assert len(id_to_row) == n_items, f"duplicate item ids in cache: {n_items} rows but {len(id_to_row)} unique ids"

    return variants, id_to_row


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


def make_pair_collate_fn(variants, pad_id, pair_max_length, shuffle_prob, augment):
    """augment=False (used for the val loader) always draws variant_base --
    val must stay stable/comparable across checkpoints, no randomness there.
    augment=True (train loader) draws variant_shuffled with probability
    shuffle_prob, INDEPENDENTLY per item-slot -- a pointer choice into an
    already-tokenized array, zero extra compute (see module docstring fix 6)."""
    base = variants["variant_base"]
    shuf = variants["variant_shuffled"]
    rng = np.random.RandomState(SEED + 1)

    def pick(row):
        arrs = shuf if (augment and rng.random_sample() < shuffle_prob) else base
        item_input_ids, item_lengths, category_label_start, attr_label_start = arrs[1], arrs[2], arrs[3], arrs[4]
        return build_compact_ids(item_input_ids, row, int(item_lengths[row]),
                                  int(category_label_start[row]), int(attr_label_start[row]))

    def collate_fn(batch):
        row1, row2, labels, categories = zip(*batch)
        bsz = len(batch)
        input_ids = np.full((bsz, pair_max_length), pad_id, dtype=np.int64)
        attention_mask = np.zeros((bsz, pair_max_length), dtype=np.int64)

        for i in range(bsz):
            compact1 = pick(row1[i])
            compact2 = pick(row2[i])
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


def make_pair_loader(pairs_pd, id_to_row, category_to_id, variants, pad_id, pair_max_length,
                      batch_size, shuffle, augment):
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
    collate_fn = make_pair_collate_fn(variants, pad_id, pair_max_length, ATTRIBUTE_SHUFFLE_PROB, augment)
    drop_last = shuffle and (len(dataset) >= batch_size)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0,
        collate_fn=collate_fn, pin_memory=torch.cuda.is_available(), drop_last=drop_last,
    )
    return loader


# =====================================================================
# MODEL (unchanged)
# =====================================================================

class FridaCrossEncoderClassifier(nn.Module):
    def __init__(self, encoder, head):
        super().__init__()
        self.encoder = encoder
        self.classifier = head

    def forward(self, input_ids, attention_mask):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            cls = out.last_hidden_state[:, 0, :]
            return self.classifier(cls).view(-1).float()


def _find_resume_adapter_dir():
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
# VRAM AUTO-PROBE (unchanged)
# =====================================================================

def _probe_peak_gb_train(batch_size, n_gpu, pair_max_length, probe_model, probe_loss_fn, trainable_params, device):
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


def _probe_peak_gb_eval(batch_size, n_gpu, pair_max_length, probe_model, device):
    dummy_input_ids = torch.randint(0, 1000, (batch_size, pair_max_length), dtype=torch.long, device=device)
    dummy_attention_mask = torch.ones((batch_size, pair_max_length), dtype=torch.long, device=device)

    for i in range(max(n_gpu, 1)):
        torch.cuda.reset_peak_memory_stats(i)

    with torch.no_grad():
        logits = probe_model(input_ids=dummy_input_ids, attention_mask=dummy_attention_mask)

    peak_bytes = max(torch.cuda.max_memory_allocated(i) for i in range(max(n_gpu, 1)))

    del dummy_input_ids, dummy_attention_mask, logits
    torch.cuda.empty_cache()
    return peak_bytes / (1024 ** 3)


def _is_oom(exc):
    return isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()


def run_vram_probe(probe_fn, start_batch, probe_cap, label):
    print(f"[PROBE] {label}: searching for max batch under {TARGET_VRAM_GB:.1f}GB/GPU "
          f"(safety margin {SAFETY_MARGIN_GB:.1f}GB), capped at {probe_cap} ...")
    probe_threshold = TARGET_VRAM_GB - SAFETY_MARGIN_GB

    candidate = start_batch
    last_good, last_good_gb = None, None

    while candidate <= probe_cap:
        try:
            peak_gb = probe_fn(candidate)
            print(f"[PROBE] {label}: batch={candidate:5d} -> peak={peak_gb:.2f}GB/GPU")
        except RuntimeError as e:
            if not _is_oom(e):
                raise
            torch.cuda.empty_cache()
            print(f"[PROBE] {label}: batch={candidate:5d} -> OOM")
            break

        last_good, last_good_gb = candidate, peak_gb
        if peak_gb > probe_threshold:
            break
        candidate *= 2

    if last_good is not None and last_good_gb is not None and last_good_gb <= probe_threshold and candidate <= probe_cap:
        lo, hi = last_good, candidate
        while hi - lo > PROBE_ROUND_TO:
            mid = ((lo + hi) // 2 // PROBE_ROUND_TO) * PROBE_ROUND_TO
            if mid <= lo:
                break
            try:
                peak_gb = probe_fn(mid)
                print(f"[PROBE] {label}: batch={mid:5d} -> peak={peak_gb:.2f}GB/GPU")
                if peak_gb > probe_threshold:
                    hi = mid
                else:
                    lo, last_good, last_good_gb = mid, mid, peak_gb
            except RuntimeError as e:
                if not _is_oom(e):
                    raise
                torch.cuda.empty_cache()
                print(f"[PROBE] {label}: batch={mid:5d} -> OOM")
                hi = mid

    if last_good is None:
        print(f"[PROBE] {label}: first candidate ({start_batch}) already OOM'd -- keeping config value.")
        return start_batch

    discovered = max(PROBE_ROUND_TO, (last_good // PROBE_ROUND_TO) * PROBE_ROUND_TO)
    print(f"[PROBE] {label}: selected batch={discovered} (was {start_batch}, measured peak={last_good_gb:.2f}GB/GPU)")
    return discovered


# =====================================================================
# EVAL -- fix 5: chunk_size is STICKY within one evaluate() call.
# =====================================================================

class _ChunkSizeState:
    """Carries the last-known-working chunk size ACROSS batches within one
    evaluate() call -- the actual fix for the repeated-OOM-retry bug (see
    module docstring fix 5). Re-created fresh at the start of each
    evaluate() call: a size that stopped working at the END of the previous
    epoch's eval isn't assumed to still be right after another full epoch
    of training has changed the allocator's state."""

    def __init__(self, start_size):
        self.size = start_size


def _forward_in_chunks(eval_model, input_ids, attention_mask, state):
    chunk_size = min(state.size, input_ids.shape[0])
    while True:
        try:
            outs = []
            for start in range(0, input_ids.shape[0], chunk_size):
                outs.append(eval_model(input_ids=input_ids[start:start + chunk_size],
                                        attention_mask=attention_mask[start:start + chunk_size]))
            result = torch.cat(outs, dim=0) if len(outs) > 1 else outs[0]
            state.size = chunk_size   # sticky: remember what actually worked
            return result
        except torch.cuda.OutOfMemoryError:
            if chunk_size <= 1:
                raise
            chunk_size = max(1, chunk_size // 2)
            gc.collect()
            torch.cuda.empty_cache()
            print(f"[WARN] eval OOM -- retrying this batch at chunk_size={chunk_size} "
                  f"(now sticky for the rest of this eval pass)", flush=True)


def evaluate(eval_model, loader, loss_fn, device, start_chunk_size):
    eval_model.eval()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    state = _ChunkSizeState(start_chunk_size)
    all_logits, all_labels, all_categories = [], [], []
    total_loss, n_batches = 0.0, 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Eval", dynamic_ncols=True, leave=False):
            input_ids, attention_mask, batch_labels, batch_categories = batch
            input_ids = input_ids.to(device, non_blocking=True)
            attention_mask = attention_mask.to(device, non_blocking=True)
            batch_labels_dev = batch_labels.to(device, non_blocking=True)

            logits = _forward_in_chunks(eval_model, input_ids, attention_mask, state)
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
        "final_chunk_size": state.size,
    }


# =====================================================================
# CHECKPOINT
# =====================================================================

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
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "polars", "peft", "mlflow"])
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    import mlflow

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1")
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    os.environ["OMP_NUM_THREADS"] = str(os.cpu_count() or 8)
    os.environ["MKL_NUM_THREADS"] = str(os.cpu_count() or 8)

    set_seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    latest_metrics_path = os.path.join(OUTPUT_DIR, "latest_metrics.json")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("=" * 90)
    print("CONFIG -- FRIDA LoRA v2 (new attribute policy, time-aware quiet checkpointing)")
    print("=" * 90)
    print(f"model:                  {MODEL_NAME}")
    print(f"item_max_length:        {ITEM_MAX_LENGTH}  (prefix={PREFIX!r}, 2 pre-tokenized variants)")
    print(f"attribute_shuffle_prob: {ATTRIBUTE_SHUFFLE_PROB:.0%}")
    print(f"pair_max_length:        {PAIR_MAX_LENGTH}")
    print(f"epochs (this run):      {EPOCHS}")
    print(f"lr (fresh schedule):    {LR}")
    print(f"quiet_checkpoint_after: {QUIET_CHECKPOINT_AFTER_SECONDS / 3600:.1f}h")
    print(f"quiet_save_every:       {QUIET_SAVE_EVERY_SECONDS / 60:.0f}min")
    print(f"output_dir:             {OUTPUT_DIR}")
    print("=" * 90)

    if os.path.isdir("/kaggle/input"):
        for _root, _dirs, _files in os.walk("/kaggle/input"):
            depth = _root[len("/kaggle/input"):].count(os.sep)
            if depth <= 3:
                print(f"[DEBUG] {_root} -> dirs={_dirs} files={_files}")

    mlflow.set_tracking_uri(f"file:{MLFLOW_TRACKING_DIR}")
    mlflow.set_experiment("ozon-matching")

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

    del matches_pd, labels_all, indices, train_idx, val_idx
    gc.collect()

    # -------------------------------------------------------------
    # Load the v2 item token cache (2 variants).
    # -------------------------------------------------------------
    cache_root = _find_item_cache_root(ITEM_CACHE_HINT)
    print(f"[CACHE] loading v2 item token cache from {cache_root} ...")
    variants, id_to_row = load_item_cache_v2(cache_root)
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
    del items_cat, id_to_category
    gc.collect()
    _log_memory("after category load")

    micro_batch_size = MICRO_BATCH_SIZE
    eval_batch_size = EVAL_MICRO_BATCH_SIZE
    total_batch_size = TOTAL_BATCH_SIZE

    train_loader = make_pair_loader(
        train_pd, id_to_row, category_to_id, variants, pad_id, PAIR_MAX_LENGTH, micro_batch_size,
        shuffle=True, augment=True,
    )
    val_loader = make_pair_loader(
        val_pd, id_to_row, category_to_id, variants, pad_id, PAIR_MAX_LENGTH, eval_batch_size,
        shuffle=False, augment=False,
    )

    pos_rate = float(train_pd["label"].mean())
    print(f"[DATA] train positive rate: {pos_rate:.4f}")

    # -------------------------------------------------------------
    # Model: MUST resume -- this kernel exists to continue, not to start fresh.
    # -------------------------------------------------------------
    resume_adapter_dir = _find_resume_adapter_dir()
    if resume_adapter_dir is None:
        raise RuntimeError("No adapter checkpoint found under /kaggle/input -- this kernel requires "
                            "resuming from the best available FRIDA checkpoint (dataset_sources).")
    model = build_model_from_checkpoint(device, resume_adapter_dir)

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

    probe_loss_fn = torch.nn.BCEWithLogitsLoss()
    discovered_train = run_vram_probe(
        lambda b: _probe_peak_gb_train(b, n_gpu, PAIR_MAX_LENGTH, model, probe_loss_fn, trainable_params, device),
        micro_batch_size, min(PROBE_CAP, TOTAL_BATCH_SIZE), "TRAIN",
    )
    micro_batch_size = discovered_train
    grad_accum = max(1, total_batch_size // micro_batch_size)

    model.eval()
    discovered_eval = run_vram_probe(
        lambda b: _probe_peak_gb_eval(b, n_gpu, PAIR_MAX_LENGTH, model, device),
        eval_batch_size, EVAL_PROBE_CAP, "EVAL",
    )
    # Still halve the probe's own number up front (real gap between a clean
    # pre-training probe and mid-training reality, unchanged reasoning from
    # v1) -- fix 5 additionally makes _forward_in_chunks remember whatever
    # it discovers from there instead of re-paying the discovery cost every
    # single batch.
    eval_batch_size = max(PROBE_ROUND_TO, discovered_eval // 2)
    print(f"[PROBE] EVAL: using {eval_batch_size} (half of probed {discovered_eval}); "
          f"_forward_in_chunks will remember whatever it actually settles on (fix 5)")
    model.train()

    print("[PROBE] Rebuilding data loaders with the probed batch sizes ...")
    train_loader = make_pair_loader(
        train_pd, id_to_row, category_to_id, variants, pad_id, PAIR_MAX_LENGTH, micro_batch_size,
        shuffle=True, augment=True,
    )
    val_loader = make_pair_loader(
        val_pd, id_to_row, category_to_id, variants, pad_id, PAIR_MAX_LENGTH, eval_batch_size,
        shuffle=False, augment=False,
    )
    print(f"[PROBE] New config: MICRO_BATCH_SIZE={micro_batch_size} | EVAL_BATCH_SIZE={eval_batch_size} | "
          f"GRAD_ACCUM={grad_accum} | effective TOTAL_BATCH_SIZE={micro_batch_size * grad_accum}")
    _log_memory("after probe")

    optimizer = torch.optim.AdamW(trainable_params, lr=LR, weight_decay=WEIGHT_DECAY)

    num_update_steps_per_epoch = math.ceil(len(train_loader) / grad_accum)
    total_training_steps = num_update_steps_per_epoch * EPOCHS
    warmup_steps = max(1, int(total_training_steps * WARMUP_RATIO))

    scheduler = get_scheduler(
        name="cosine", optimizer=optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_training_steps
    )
    print(f"[TRAIN] steps/epoch: {num_update_steps_per_epoch} | total steps ({EPOCHS} epoch): "
          f"{total_training_steps} | warmup: {warmup_steps} ({WARMUP_RATIO:.0%}) -> cosine decay to 0")

    optimizer_state_path = os.path.join(resume_adapter_dir, "optimizer.pt")
    if os.path.isfile(optimizer_state_path):
        optimizer.load_state_dict(torch.load(optimizer_state_path, map_location=device))
        print(f"[RESUME] optimizer state (Adam moments) restored from {resume_adapter_dir}; "
              f"scheduler is fresh for this run's own {EPOCHS}-epoch target")
    else:
        print(f"[RESUME] WARNING: no optimizer.pt found at {resume_adapter_dir} -- optimizer starts fresh.")

    resumed_best_macro = -1.0
    train_state_path = os.path.join(resume_adapter_dir, "train_state.json")
    if os.path.isfile(train_state_path):
        with open(train_state_path) as f:
            resumed_state = json.load(f)
        resumed_best_macro = resumed_state.get("best_macro_pr_auc", -1.0) or -1.0
        print(f"[RESUME] source checkpoint's own best_macro_pr_auc={resumed_best_macro:.5f}")

    if pos_rate <= 0.0 or pos_rate >= 1.0:
        pos_weight_value = 1.0
    else:
        pos_weight_value = float(np.clip((1.0 - pos_rate) / pos_rate, 0.1, 20.0))
    print(f"[LOSS] BCE pos_weight: {pos_weight_value:.4f}")
    pos_weight = torch.tensor([pos_weight_value], device=device, dtype=torch.float32)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    print(f"[TRAIN] full-val checkpoint ONLY at epoch end (fix 4). Quiet (weights-only) safety saves every "
          f"{QUIET_SAVE_EVERY_SECONDS / 60:.0f}min throughout, and exclusively (no more full evals) once past "
          f"{QUIET_CHECKPOINT_AFTER_SECONDS / 3600:.1f}h wall-clock (fix 2).")

    mlflow.start_run(run_name="frida_lora_v2_epoch")
    mlflow.log_params({
        "variant": "frida_lora_v2_epoch", "encoder": MODEL_NAME,
        "item_max_length": ITEM_MAX_LENGTH, "pair_max_length": PAIR_MAX_LENGTH, "prefix": PREFIX,
        "attribute_shuffle_prob": ATTRIBUTE_SHUFFLE_PROB,
        "lora_r": LORA_R, "lora_alpha": LORA_ALPHA, "lora_dropout": LORA_DROPOUT,
        "lora_target_modules": ",".join(LORA_TARGET_MODULES),
        "micro_batch_size": micro_batch_size, "eval_batch_size": eval_batch_size,
        "gradient_accumulation_steps": grad_accum, "effective_total_batch_size": micro_batch_size * grad_accum,
        "lr": LR, "scheduler": "cosine_fresh", "warmup_ratio": WARMUP_RATIO, "weight_decay": WEIGHT_DECAY,
        "epochs_this_run": EPOCHS, "split_seed": SEED, "val_fingerprint": val_fingerprint,
        "train_rows": len(train_pd), "val_rows": len(val_pd),
        "quiet_checkpoint_after_seconds": QUIET_CHECKPOINT_AFTER_SECONDS,
        "quiet_save_every_seconds": QUIET_SAVE_EVERY_SECONDS,
        "item_cache": "maximmarkovbb/frida-tokenize-items-v2",
        "resumed_from": resume_adapter_dir or "",
        "resumed_best_macro_pr_auc": resumed_best_macro,
    })

    log_rows = []
    best_macro = resumed_best_macro
    global_step = 0
    window_logits, window_labels, window_categories = [], [], []
    window_loss_sum, window_loss_count = 0.0, 0
    n_skipped_microbatches = 0
    n_skipped_optimizer_steps = 0
    last_quiet_save_at = 0.0
    quiet_mode_entered = False   # once True, no more full (eval) checkpoints this run.

    def do_quiet_save(reason):
        nonlocal last_quiet_save_at
        try:
            save_dir = os.path.join(OUTPUT_DIR, "last")
            save_checkpoint(model, tokenizer, save_dir, optimizer=optimizer, scheduler=scheduler,
                             global_step=global_step, n_skipped_microbatches=n_skipped_microbatches,
                             n_skipped_optimizer_steps=n_skipped_optimizer_steps, best_macro=best_macro)
            _write_latest_metrics(latest_metrics_path, global_step, total_training_steps,
                                   None, None, None, None, best_macro, True, _elapsed())
            last_quiet_save_at = _elapsed()
            print(f"[SAVE] quiet checkpoint ({reason}) at step={global_step}, elapsed={_elapsed():.0f}s "
                  f"-> {save_dir}", flush=True)
        except Exception as e:
            print(f"[ERROR] quiet checkpoint FAILED ({reason}): {type(e).__name__}: {e}", flush=True)
            import traceback
            traceback.print_exc()

    def run_full_checkpoint():
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
        val_metrics = evaluate(eval_model, val_loader, loss_fn, device, eval_batch_size)
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
        try:
            pl.DataFrame(log_rows).write_csv(os.path.join(OUTPUT_DIR, "train_log_steps.csv"))
        except Exception as e:
            print(f"[WARN] failed to write train_log_steps.csv: {e}", flush=True)
        try:
            mlflow.log_metrics({k: v for k, v in row.items() if k != "progress"}, step=global_step)
        except Exception as e:
            print(f"[WARN] mlflow logging failed: {e}", flush=True)

        print("=" * 100)
        print(f"[METRICS step={global_step}/{total_training_steps} progress={progress:.1%} lr={row['lr']:.2e}] "
              f"TRAIN loss={train_window_loss:.5f} macro_pr_auc={train_macro:.5f} | "
              f"VAL(full) loss={row['val_loss']:.5f} macro_pr_auc={row['val_macro_pr_auc']:.5f} | "
              f"best_so_far={best_macro:.5f} | eval_chunk_size_settled={val_metrics['final_chunk_size']} | "
              f"skipped: {n_skipped_microbatches} microbatches, {n_skipped_optimizer_steps} optimizer steps")
        print("=" * 100)
        _log_memory(f"checkpoint step={global_step}")

        _write_latest_metrics(latest_metrics_path, global_step, total_training_steps, train_window_loss,
                               train_macro, row["val_loss"], row["val_macro_pr_auc"], best_macro, False, _elapsed())

        if not np.isnan(row["val_macro_pr_auc"]) and row["val_macro_pr_auc"] > best_macro:
            best_macro = row["val_macro_pr_auc"]
            save_dir = os.path.join(OUTPUT_DIR, "best")
            save_checkpoint(model, tokenizer, save_dir, optimizer=optimizer, scheduler=scheduler,
                             global_step=global_step, n_skipped_microbatches=n_skipped_microbatches,
                             n_skipped_optimizer_steps=n_skipped_optimizer_steps, best_macro=best_macro)
            print(f"[SAVE] NEW BEST val_macro_pr_auc={best_macro:.5f} (full val) -> {save_dir}")

        # "last" always reflects the most recent state too, full or quiet.
        save_checkpoint(model, tokenizer, os.path.join(OUTPUT_DIR, "last"), optimizer=optimizer,
                         scheduler=scheduler, global_step=global_step,
                         n_skipped_microbatches=n_skipped_microbatches,
                         n_skipped_optimizer_steps=n_skipped_optimizer_steps, best_macro=best_macro)

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

            # --- fix 3: step-level OOM resilience ---
            try:
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_fn(logits, batch_labels_dev)

                if not torch.isfinite(loss):
                    n_skipped_microbatches += 1
                    print(f"[WARN] step={step} non-finite loss ({loss.item()!r}) -- skipping microbatch "
                          f"(total skipped: {n_skipped_microbatches})", flush=True)
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
                    pbar.set_postfix(loss=round(loss_item, 5), lr=f"{scheduler.get_last_lr()[0]:.2e}",
                                      step=global_step)
            except torch.cuda.OutOfMemoryError as e:
                n_skipped_microbatches += 1
                print(f"[WARN] step={step} CUDA OOM on the forward/backward itself -- skipping this "
                      f"microbatch entirely (total skipped: {n_skipped_microbatches}). {e}", flush=True)
                optimizer.zero_grad(set_to_none=True)
                gc.collect()
                torch.cuda.empty_cache()
                continue

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
                          f"(total skipped: {n_skipped_optimizer_steps})", flush=True)
                optimizer.zero_grad()

                is_epoch_end = (step + 1) == len(train_loader)   # fix 4: only trigger here.
                elapsed_now = _elapsed()

                if elapsed_now >= QUIET_CHECKPOINT_AFTER_SECONDS and not quiet_mode_entered:
                    quiet_mode_entered = True
                    print(f"[TIME] {elapsed_now / 3600:.2f}h elapsed >= "
                          f"{QUIET_CHECKPOINT_AFTER_SECONDS / 3600:.1f}h -- entering quiet-only mode "
                          f"(no more full/eval checkpoints this run, see fix 2)", flush=True)
                    do_quiet_save("entering quiet mode")

                if grad_ok and is_epoch_end and not quiet_mode_entered:
                    try:
                        run_full_checkpoint()
                    except Exception as e:
                        print(f"[ERROR] full checkpoint/eval at step={global_step} FAILED and was skipped "
                              f"-- training continues if there are more steps. {type(e).__name__}: {e}",
                              flush=True)
                        import traceback
                        traceback.print_exc()
                        model.train()
                        optimizer.zero_grad(set_to_none=True)
                        window_logits.clear(); window_labels.clear(); window_categories.clear()
                        window_loss_sum, window_loss_count = 0.0, 0
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        do_quiet_save("full checkpoint failed, falling back to quiet save")
                elif is_epoch_end and quiet_mode_entered:
                    do_quiet_save("epoch end reached but already in quiet mode")
                elif (elapsed_now - last_quiet_save_at) >= QUIET_SAVE_EVERY_SECONDS:
                    do_quiet_save("periodic safety save")

        print(f"[EPOCH] finished epoch {epoch + 1}/{EPOCHS} | best val_macro_pr_auc so far={best_macro:.5f}",
              flush=True)

    print(f"[DONE] Best macro_pr_auc={best_macro:.5f} (started from {resumed_best_macro:.5f})")
    print(f"[DONE] Step log: {os.path.join(OUTPUT_DIR, 'train_log_steps.csv')}")
    print(f"[DONE] Latest metrics: {latest_metrics_path}")
    print(f"[DONE] Best model dir: {os.path.join(OUTPUT_DIR, 'best')}")
    print(f"[DONE] Last model dir: {os.path.join(OUTPUT_DIR, 'last')}")

    try:
        mlflow.log_metric("best_macro_pr_auc", best_macro)
        mlflow.log_artifact(os.path.join(OUTPUT_DIR, "train_log_steps.csv"))
        mlflow.end_run()
    except Exception as e:
        print(f"[WARN] final mlflow logging failed: {e}", flush=True)
    print(f"[DONE] mlflow run logged under {MLFLOW_TRACKING_DIR}")


if __name__ == "__main__":
    main()
