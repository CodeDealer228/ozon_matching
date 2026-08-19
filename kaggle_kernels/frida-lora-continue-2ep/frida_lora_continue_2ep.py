"""
LoRA SFT of ai-forever/FRIDA (T5EncoderModel, 823M) as a cross-encoder --
continues the frida-lora-sft-epoch2 checkpoint (maxamsecret/frida-lora-sft-
epoch2-40, val_macro_pr_auc=0.67397 @ step 458/1143, WITH optimizer state),
targeting a fresh 2-EPOCH run from this resume point (per explicit
instruction). NOT the same design as the prepared-but-never-launched
frida-lora-sft-epoch3 (which is still sitting there for reference, EPOCHS=1
only) -- this is a genuinely different script, not an edit of it, kept in
its own kaggle_kernels/ directory for the same reason epoch3 itself gave for
not editing epoch2 in place: epoch2's real 12h-session result must stay
reproducible/inspectable as its own artifact.

===========================================================================
Why NOT epoch3's own fix (EVAL_SUBSAMPLE_FRAC=0.25), and why EPOCHS=2 here
when epoch3's own docstring says "EPOCHS must stay 1"
===========================================================================
epoch3 correctly diagnosed the real bottleneck neither epoch1 nor epoch2
got past: a full eval pass over 73,131 val pairs at a NEVER-AUTO-PROBED
EVAL_MICRO_BATCH_SIZE=64 (just 2x the train micro-batch, a heuristic, not a
measurement) took ~6.46s/iteration x 1143 iterations =~ 2.05h -- at
N_CHECKPOINTS_PER_EPOCH=5, eval alone ate ~10.25h of a 12h session before
one full training epoch got credited. epoch3's fix was to SUBSAMPLE eval to
25% for interim checkpoints, running one full-val pass only at the very
end. This script fixes the SAME root cause differently: an actual
eval-mode VRAM auto-probe (forward-only, no_grad -- mirroring the
train-mode probe both scripts already had, just never applied to eval),
raised to a much higher cap than epoch1/2/3's shared PROBE_CAP=512. This
project's own BERTA scripts hit an identical bug (EVAL_MICRO_BATCH_SIZE
capped at 512 when a real run only used 1.69GB of an 11GB budget there --
8x real headroom sitting unused) -- the fix (a separate, much larger
EVAL_PROBE_CAP) is proven on a real Kaggle GPU session already, just never
applied to this FRIDA lineage. If the real probed eval batch is large
enough, FULL val at every checkpoint becomes cheap enough that subsampling
is unnecessary -- no statistical noise in "new best" tracking, and the
number reported at every checkpoint is already the real, final metric, not
an estimate needing a later full-val reconciliation pass.

EPOCHS=2 (not 1): epoch3's own EPOCHS=1 constraint existed because IT
resumes epoch2's own optimizer AND scheduler state, whose internal position
was computed against epoch2's own total_training_steps (built for
EPOCHS=1) -- changing EPOCHS there would desync the resumed cosine schedule
from where it actually sits. This script sidesteps that constraint
entirely rather than fighting it: the OPTIMIZER (Adam moment estimates)
IS resumed from epoch2-40's checkpoint -- legitimate, real curvature
information for these exact LoRA parameters, not schedule-position-
dependent -- but the SCHEDULER is deliberately NOT resumed; a fresh cosine
warmup+decay is built here, sized for THIS run's own 2-epoch target from
this resume point forward. This is the same "resume weights+optimizer,
fresh schedule" pattern already used successfully in this project's BERTA
LLM-continue-train experiment, for the identical reason: a stale resumed
scheduler built for a different total-step count is a correctness risk
(cosine's progress = (step-warmup)/(total-warmup) exceeding 1 is an
undefined/oscillating regime, not a clamped one), not a stability feature
worth preserving. "2 full epochs" here counts from THIS resume point
forward, not from the start of the whole frida-lora-sft-human-only-10-
epochs/epoch2/epoch3 lineage -- that lineage's own per-kernel-fresh-
schedule history (each of epoch1/epoch2 built its OWN EPOCHS=1 schedule,
resuming weights but not a globally-coherent step count) makes an exact
cumulative epoch-fraction count across kernels ill-defined already.

Everything else (item-level pre-tokenized cache -- no tokenizer calls at
train time, PAIR_MAX_LENGTH=128, Category:/Attributes: excision, bf16 +
eager attention + autocast INSIDE forward() to avoid the DataParallel NaN
trap diagnosed in kaggle_kernels/frida-nan-diagnose, is_trainable=True on
resume per this repo's own documented PEFT trap, best-only checkpoint
saving with optimizer state -- already correct in epoch2/epoch3, unchanged
here) is identical to the rest of this FRIDA lineage -- see
kaggle_kernels/frida-lora-sft-human-only-10-epochs/README.md for the full
original rationale, not repeated here.

PREPARED, NOT LAUNCHED -- explicit instruction. Needs dataset_sources
confirmed (maxamsecret/frida-lora-sft-epoch2-40) and an account decision
(maxamsecret GPU credits, per explicit instruction) before pushing.
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
ITEM_CACHE_HINT = "/kaggle/input/datasets/maxamsecret/frida-tokenized-items-human"
ITEM_MAX_LENGTH = 256           # from the published cache_meta.json

PAIR_MAX_LENGTH = 128           # unchanged from epoch1/2/3 -- see their
                                  # READMEs for the token-budget-sizing numbers.

MAX_CATEGORY_CHARS = 64

PREFIX = "paraphrase: "         # matches the cache's own cache_meta.json

EPOCHS = 2                       # fresh target from THIS resume point -- see
                                   # module docstring for why this diverges
                                   # from epoch3's "must stay 1" constraint.

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.10
LORA_TARGET_MODULES = ["q", "k", "v", "o", "wi_0", "wi_1", "wo"]

LR = 3e-4                        # peak LR for the FRESH schedule this run builds.
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.10

VAL_EVERY_FRACTION = 0.50        # every 0.5 epoch, per explicit instruction
                                   # (was N_CHECKPOINTS_PER_EPOCH=3/5 in
                                   # epoch2/3 -- this is a plain fraction,
                                   # not a per-epoch count, so it stays
                                   # meaningful across a 2-epoch run too).

MICRO_BATCH_SIZE = 32             # train probe starting point -- epoch2's
                                    # own probe landed here for this config.
EVAL_MICRO_BATCH_SIZE = 64        # eval probe starting point.
TOTAL_BATCH_SIZE = 256

TARGET_VRAM_GB = 14.0
SAFETY_MARGIN_GB = 1.5
PROBE_CAP = 512                   # train probe cap -- tied to TOTAL_BATCH_SIZE
                                    # semantics same as every other script in
                                    # this project (see BERTA scripts' own
                                    # comment on this).
EVAL_PROBE_CAP = 4096             # eval probe cap -- NOT shared with train's,
                                    # see module docstring "why not epoch3's
                                    # own fix". No grad-accum coupling for
                                    # eval, so no reason to share the cap.
PROBE_ROUND_TO = 4

ENABLE_DATASET_PUSH = False

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
# ITEM-LEVEL TOKEN CACHE (unchanged from epoch1/2/3 -- reads the pre-built
# cache, no tokenizer calls at train time)
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
# MODEL (unchanged from epoch1/2/3)
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
# VRAM AUTO-PROBE -- train (unchanged) + a REAL eval probe this time
# (forward-only, no_grad, own cap -- see module docstring)
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
    """probe_fn(batch_size) -> peak_gb, or raises RuntimeError on OOM."""
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
# EVAL (unchanged logic; now always called on the FULL val set -- no
# subsample, see module docstring)
# =====================================================================

def _forward_in_chunks(eval_model, input_ids, attention_mask, chunk_size):
    """Runs the forward pass in sub-chunks, halving on CUDA OOM and
    retrying, down to a single row. A REAL incident this exists to prevent:
    an eval batch size chosen by the pre-training VRAM probe (clean model,
    nothing else resident) OOM'd for real once eval actually ran MID-
    training -- at that point gradients, optimizer state, and allocator
    fragmentation are all live, so materially less HBM is free than the
    probe ever saw ("Tried to allocate 3.00 GiB... 414.81 MiB is free").
    Rather than crash a multi-hour run over a transient allocation, back
    off to whatever chunk actually fits, right now, on this GPU."""
    while True:
        try:
            outs = []
            for start in range(0, input_ids.shape[0], chunk_size):
                outs.append(eval_model(input_ids=input_ids[start:start + chunk_size],
                                        attention_mask=attention_mask[start:start + chunk_size]))
            return torch.cat(outs, dim=0) if len(outs) > 1 else outs[0]
        except torch.cuda.OutOfMemoryError:
            if chunk_size <= 1:
                raise
            chunk_size = max(1, chunk_size // 2)
            gc.collect()
            torch.cuda.empty_cache()
            print(f"[WARN] eval OOM -- retrying this batch at chunk_size={chunk_size}", flush=True)


def evaluate(eval_model, loader, loss_fn, device):
    eval_model.eval()

    # Give eval the cleanest allocator state we can before it starts -- it
    # runs mid-training, after backward()/optimizer.step() have left both
    # real allocations and fragmentation behind.
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    all_logits, all_labels, all_categories = [], [], []
    total_loss, n_batches = 0.0, 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Eval", dynamic_ncols=True, leave=False):
            input_ids, attention_mask, batch_labels, batch_categories = batch
            input_ids = input_ids.to(device, non_blocking=True)
            attention_mask = attention_mask.to(device, non_blocking=True)
            batch_labels_dev = batch_labels.to(device, non_blocking=True)

            logits = _forward_in_chunks(eval_model, input_ids, attention_mask, input_ids.shape[0])
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
# CHECKPOINT (best-only saving -- already correct in epoch2/3, unchanged)
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
    print("CONFIG -- FRIDA LoRA continue-train, 2-epoch target from this resume point")
    print("=" * 90)
    print(f"model:                {MODEL_NAME}")
    print(f"item_max_length:      {ITEM_MAX_LENGTH}  (prefix={PREFIX!r}, from pre-built cache)")
    print(f"pair_max_length:      {PAIR_MAX_LENGTH}")
    print(f"epochs (this run):    {EPOCHS}")
    print(f"lr (fresh schedule):  {LR}")
    print(f"lora r/alpha/drop:    {LORA_R}/{LORA_ALPHA}/{LORA_DROPOUT}")
    print(f"lora target_modules:  {LORA_TARGET_MODULES}")
    print(f"total_batch_size:     {TOTAL_BATCH_SIZE}")
    print(f"val_every_fraction:   {VAL_EVERY_FRACTION:.0%} (full val, no subsample -- auto-probed eval batch)")
    print(f"output_dir:           {OUTPUT_DIR}")
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
    # Model: MUST resume -- this kernel exists to continue frida-lora-sft-
    # epoch2-40, not to start fresh.
    # -------------------------------------------------------------
    resume_adapter_dir = _find_resume_adapter_dir()
    if resume_adapter_dir is None:
        raise RuntimeError("No adapter checkpoint found under /kaggle/input -- expected "
                            "maxamsecret/frida-lora-sft-epoch2-40 mounted (dataset_sources). "
                            "This kernel requires resuming from the best FRIDA checkpoint.")
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
    # Halve what the probe found. The probe runs on a CLEAN model before
    # training starts; the real eval runs mid-training with gradients,
    # optimizer state and allocator fragmentation all resident -- a real
    # run OOM'd at exactly this gap (probe saw room, mid-training eval had
    # only 414MB free). _forward_in_chunks self-heals if this is still too
    # big, but backing off up front avoids paying repeated OOM-retry
    # cycles on every single eval batch.
    eval_batch_size = max(PROBE_ROUND_TO, discovered_eval // 2)
    print(f"[PROBE] EVAL: using {eval_batch_size} (half of the probed {discovered_eval} -- "
          f"mid-training headroom is materially smaller than the probe's own)")
    model.train()

    print("[PROBE] Rebuilding data loaders with the probed batch sizes ...")
    train_loader = make_pair_loader(
        train_pd, id_to_row, category_to_id, item_input_ids_mm, item_lengths, compact_lengths,
        category_label_start, attr_label_start, pad_id, PAIR_MAX_LENGTH, micro_batch_size, shuffle=True,
    )
    val_loader = make_pair_loader(
        val_pd, id_to_row, category_to_id, item_input_ids_mm, item_lengths, compact_lengths,
        category_label_start, attr_label_start, pad_id, PAIR_MAX_LENGTH, eval_batch_size, shuffle=False,
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
    print(f"[TRAIN] steps/epoch: {num_update_steps_per_epoch} | total steps ({EPOCHS} epochs): "
          f"{total_training_steps} | warmup: {warmup_steps} ({WARMUP_RATIO:.0%}) -> cosine decay to 0")

    # Optimizer state IS resumed (Adam moments -- real curvature info for
    # these exact LoRA params). Scheduler is deliberately fresh -- see
    # module docstring "why EPOCHS=2 here".
    optimizer_state_path = os.path.join(resume_adapter_dir, "optimizer.pt")
    if os.path.isfile(optimizer_state_path):
        optimizer.load_state_dict(torch.load(optimizer_state_path, map_location=device))
        print(f"[RESUME] optimizer state (Adam moments) restored from {resume_adapter_dir}; "
              f"scheduler is fresh for this run's own {EPOCHS}-epoch target")
    else:
        print(f"[RESUME] WARNING: no optimizer.pt found at {resume_adapter_dir} -- optimizer starts fresh "
              f"(weights-only resume). Unexpected for frida-lora-sft-epoch2-40 specifically.")

    resumed_best_macro = -1.0
    train_state_path = os.path.join(resume_adapter_dir, "train_state.json")
    if os.path.isfile(train_state_path):
        with open(train_state_path) as f:
            resumed_state = json.load(f)
        resumed_best_macro = resumed_state.get("best_macro_pr_auc", -1.0) or -1.0
        print(f"[RESUME] source checkpoint's own best_macro_pr_auc={resumed_best_macro:.5f} "
              f"(full-val, from its own run) -- this run's own best-tracking starts from here")

    if pos_rate <= 0.0 or pos_rate >= 1.0:
        pos_weight_value = 1.0
    else:
        pos_weight_value = float(np.clip((1.0 - pos_rate) / pos_rate, 0.1, 20.0))
    print(f"[LOSS] BCE pos_weight: {pos_weight_value:.4f}")
    pos_weight = torch.tensor([pos_weight_value], device=device, dtype=torch.float32)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    checkpoint_interval = max(1, round(num_update_steps_per_epoch * VAL_EVERY_FRACTION))
    print(f"[TRAIN] full-val checkpoint every {checkpoint_interval} optimizer steps "
          f"(~every {VAL_EVERY_FRACTION:.0%} of an epoch) -- no subsample, auto-probed eval batch "
          f"makes full val cheap enough to run every time (see module docstring).")

    mlflow.start_run(run_name="frida_lora_continue_2ep")
    mlflow.log_params({
        "variant": "frida_lora_continue_2ep", "encoder": MODEL_NAME,
        "item_max_length": ITEM_MAX_LENGTH, "pair_max_length": PAIR_MAX_LENGTH, "prefix": PREFIX,
        "lora_r": LORA_R, "lora_alpha": LORA_ALPHA, "lora_dropout": LORA_DROPOUT,
        "lora_target_modules": ",".join(LORA_TARGET_MODULES),
        "micro_batch_size": micro_batch_size, "eval_batch_size": eval_batch_size,
        "gradient_accumulation_steps": grad_accum, "effective_total_batch_size": micro_batch_size * grad_accum,
        "lr": LR, "scheduler": "cosine_fresh", "warmup_ratio": WARMUP_RATIO, "weight_decay": WEIGHT_DECAY,
        "epochs_this_run": EPOCHS, "split_seed": SEED, "val_fingerprint": val_fingerprint,
        "train_rows": len(train_pd), "val_rows": len(val_pd),
        "val_every_fraction": VAL_EVERY_FRACTION,
        "item_cache": "maxamsecret/frida-tokenized-items-human",
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
        val_metrics = evaluate(eval_model, val_loader, loss_fn, device)   # FULL val, every time
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

        print("=" * 100)
        print(f"[METRICS step={global_step}/{total_training_steps} progress={progress:.1%} lr={row['lr']:.2e}] "
              f"TRAIN loss={train_window_loss:.5f} macro_pr_auc={train_macro:.5f} | "
              f"VAL(full) loss={row['val_loss']:.5f} macro_pr_auc={row['val_macro_pr_auc']:.5f} | "
              f"best_so_far={best_macro:.5f} | skipped: {n_skipped_microbatches} microbatches, "
              f"{n_skipped_optimizer_steps} optimizer steps (NaN/Inf guard)")
        print("=" * 100)
        _log_memory(f"checkpoint step={global_step}")

        # Best-only saving -- this is what "чекпоины при лучшем качестве"
        # means here: the checkpoint on disk always reflects the best real
        # val_macro_pr_auc seen so far, not just the latest step.
        if not np.isnan(row["val_macro_pr_auc"]) and row["val_macro_pr_auc"] > best_macro:
            best_macro = row["val_macro_pr_auc"]
            save_dir = os.path.join(OUTPUT_DIR, "best")
            save_checkpoint(model, tokenizer, save_dir, optimizer=optimizer, scheduler=scheduler,
                             global_step=global_step, n_skipped_microbatches=n_skipped_microbatches,
                             n_skipped_optimizer_steps=n_skipped_optimizer_steps, best_macro=best_macro)
            print(f"[SAVE] NEW BEST val_macro_pr_auc={best_macro:.5f} (full val) -> {save_dir}")

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
                    # NOTHING inside the checkpoint/metrics path is allowed
                    # to kill a multi-hour training run. A real incident:
                    # a mid-training eval OOM'd and took the whole kernel
                    # down after real training progress had already been
                    # made but not yet saved. Metrics and checkpoints are
                    # observability, not the training itself -- on failure,
                    # log loudly, free what can be freed, and keep training.
                    # (The eval path additionally self-heals via
                    # _forward_in_chunks; this is the outer backstop for
                    # anything else -- metric computation, disk writes,
                    # mlflow, a torch.save hiccup.)
                    try:
                        run_checkpoint()
                    except Exception as e:
                        print(f"[ERROR] checkpoint/eval at step={global_step} FAILED and was skipped "
                              f"-- training continues. {type(e).__name__}: {e}", flush=True)
                        import traceback
                        traceback.print_exc()
                        model.train()
                        optimizer.zero_grad(set_to_none=True)
                        window_logits.clear(); window_labels.clear(); window_categories.clear()
                        window_loss_sum, window_loss_count = 0.0, 0
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

        print(f"[EPOCH] finished epoch {epoch + 1}/{EPOCHS} | best val_macro_pr_auc so far={best_macro:.5f}",
              flush=True)

    last_dir = os.path.join(OUTPUT_DIR, "last")
    try:
        save_checkpoint(model, tokenizer, last_dir, optimizer=optimizer, scheduler=scheduler,
                         global_step=global_step, n_skipped_microbatches=n_skipped_microbatches,
                         n_skipped_optimizer_steps=n_skipped_optimizer_steps, best_macro=best_macro)
    except Exception as e:
        # The "best" checkpoint saved during the run is the one that
        # actually matters; losing this final "last" snapshot to a disk
        # hiccup must not turn a completed run into a failed one.
        print(f"[ERROR] final 'last' checkpoint save FAILED: {type(e).__name__}: {e}", flush=True)
        import traceback
        traceback.print_exc()

    print(f"[DONE] Best macro_pr_auc={best_macro:.5f} (started from {resumed_best_macro:.5f})")
    print(f"[DONE] Step log: {os.path.join(OUTPUT_DIR, 'train_log_steps.csv')}")
    print(f"[DONE] Best model dir: {os.path.join(OUTPUT_DIR, 'best')}")
    print(f"[DONE] Last model dir: {last_dir}")

    mlflow.log_metric("best_macro_pr_auc", best_macro)
    mlflow.log_artifact(os.path.join(OUTPUT_DIR, "train_log_steps.csv"))
    mlflow.end_run()
    print(f"[DONE] mlflow run logged under {MLFLOW_TRACKING_DIR}")


if __name__ == "__main__":
    main()
