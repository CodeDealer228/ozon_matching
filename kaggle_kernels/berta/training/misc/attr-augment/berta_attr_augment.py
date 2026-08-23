"""
Experiment: continue-train the current best BERTA LoRA cross-encoder
checkpoint (berta-lora-sft-continue-5-epochs, epoch 9, global_step=10726,
val_macro_pr_auc=0.7246) with a text augmentation on the `attributes` field:
each item, each time it's drawn into a batch, independently gets one of
three variants of its attributes JSON key ORDER -- shuffled (p=0.3), sorted
alphabetically (p=0.3), or the original untouched order (p=0.4). Trained 2
epochs on the full human train split (292,523 pairs), canonical val
(73,131) unchanged.

Motivation: `attributes` is a collapsed JSON object string (CLAUDE.md);
key order is an accidental serialization artifact, not signal -- a model
that's sensitive to it is overfitting to something meaningless. Augmenting
with reordered variants should make the model invariant to this, which
should matter more where attributes carry the deciding signal (the
project's own documented headroom analysis: variant-level detail like
size/colour/article lives in attributes, and truncation already discards
most of it at PAIR_MAX_LENGTH=128 -- an order-robust representation of
what's left should be worth more per token).

Three variants are pre-tokenized ONCE (not re-tokenized per batch, which
would defeat the whole point of a token cache and reintroduce the
per-chunk-retokenize bottleneck this project has hit before) -- see
build_augmented_item_caches(). Reordering a JSON object's keys changes
nothing about total character count (same characters, permuted order), and
the Name/Category/"Attributes: " prefix text is byte-identical across all
three variants (only what comes AFTER "Attributes:" changes) -- so
category_label_start/attr_label_start (found once, on the original variant)
are reused for all three rather than re-run per variant; item_lengths are
still measured per-variant directly from each variant's own attention_mask
(cheap, no reason to assume rather than check). At collate time, each of
the two item slots in a pair independently rolls which variant to draw from
-- re-rolled every time an item is sampled, not fixed once per item, so the
same item can appear in different augmented forms across steps/epochs.

Resume design (weights: yes / optimizer moments: yes / scheduler: fresh),
the ClassifierHead dropout-wrapper fix, and the MAX_TRAIN_SECONDS wall-clock
cutoff are identical to, and use the same reasoning as, ../berta-llm-
continue-train and ../berta-hardneg-balanced -- see those scripts' module
docstrings, not repeated here. EVAL_PROBE_CAP is raised to 4096 (was 512,
shared with the train probe cap) -- a completed real run measured eval at
batch=512 using only 1.69GB of an 11GB budget, meaning the probe cap itself,
not VRAM, was the limiting factor; eval has no TOTAL_BATCH_SIZE-tied
grad-accum constraint the way train does, so there's no reason to share the
same cap.
"""
import os
import sys
import gc
import ctypes
import json
import math
import random
import shutil
import hashlib
import subprocess
import time

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

PREFIX = "paraphrase: "
ITEM_MAX_LENGTH = 128
MAX_NAME_CHARS, MAX_CATEGORY_CHARS, MAX_ATTR_CHARS = 128, 64, 384
MAX_ITEM_CHARS = MAX_NAME_CHARS + MAX_CATEGORY_CHARS + MAX_ATTR_CHARS + 64  # 640, byte-matches CLAUDE.md
PAIR_MAX_LENGTH = 128

AUG_VARIANTS = ["shuffled", "sorted", "original"]
AUG_PROBS = [0.3, 0.3, 0.4]

EPOCHS = 2   # per explicit instruction

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.15   # documentation only for a resume -- PeftModel.from_pretrained()
                        # reconstructs the real value from the checkpoint's own
                        # adapter_config.json regardless of this constant.
CLASSIFIER_DROPOUT = 0.10   # the checkpoint's classifier head is a ClassifierHead
                              # (Dropout -> Linear) wrapper, not a bare nn.Linear --
                              # see ClassifierHead below. Dropout has no learned state,
                              # so this rate must be set explicitly.
LORA_TARGET_MODULES = ["query", "key", "value", "dense"]

LR = 2e-4
WEIGHT_DECAY = 0.04   # documentation only for a resume -- optimizer.load_state_dict()
                        # overwrites param_groups (where weight_decay lives) with
                        # whatever the checkpoint's own optimizer.pt saved (0.04) --
                        # this constant is a no-op for the resumed AdamW instance.
WARMUP_RATIO = 0.10

MICRO_BATCH_SIZE = 16
EVAL_MICRO_BATCH_SIZE = 32
TOTAL_BATCH_SIZE = 256

TARGET_VRAM_GB = 14.0
SAFETY_MARGIN_GB = 3.0
PROBE_CAP = 512          # train: kept <= TOTAL_BATCH_SIZE-driven considerations.
EVAL_PROBE_CAP = 4096    # eval: no grad-accum coupling, real headroom measured
                           # at only 1.69/11GB used at the old cap of 512 -- see
                           # module docstring.
PROBE_ROUND_TO = 4

CATEGORY_LABEL_PATTERN = [14948, 30]
ATTRIBUTES_LABEL_PATTERN = [4816, 37519, 521, 30]

MAX_TRAIN_SECONDS = 10200   # 2h50m -- 2 epochs over 292,523 rows at the ~61
                              # pairs/sec calibration is ~9,591s (2.66h) of pure
                              # train; this leaves headroom for item-cache
                              # tokenization (3x, ~9min) + VRAM probe + 1 eval
                              # pass inside a ~3h ceiling.

WORKING_DIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
OUTPUT_DIR = os.path.join(WORKING_DIR, "lora_berta_attr_augment")
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoint")
MLFLOW_TRACKING_DIR = os.path.join(WORKING_DIR, "experiments", "mlruns")

CHECKPOINT_DATASET_ID = "rays0n/berta-attr-augment-checkpoint"
CHECKPOINT_DATASET_TITLE = "BERTA attribute augmentation -- resumable checkpoint"
ENABLE_DATASET_PUSH = True

TOKENIZE_CHUNK_SIZE_START = 20_000
TOKENIZE_CHUNK_SIZE_MIN = 2_000
TOKENIZE_CHUNK_SIZE_MAX = 200_000
RAM_TARGET_GB = 16.0
RAM_HARD_GB = 24.0
RAM_GROWTH_FACTOR = 1.3
RAM_SHRINK_FACTOR = 0.5

t_script_start = time.perf_counter()


def _elapsed():
    return time.perf_counter() - t_script_start


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
# PHASE 1: attribute-augmented item caches (3 variants, human scope)
# =====================================================================

def _reorder_attributes(attr_str, mode, rng):
    """mode in {"shuffled", "sorted"}. Returns the re-serialized JSON string
    with keys reordered, or the untouched input on any parse failure
    (malformed/empty/non-dict attributes -- same fallback the rest of this
    project uses for anything not cleanly parseable)."""
    if not attr_str:
        return attr_str
    try:
        d = json.loads(attr_str)
    except Exception:
        return attr_str
    if not isinstance(d, dict) or not d:
        return attr_str
    keys = list(d.keys())
    if mode == "shuffled":
        rng.shuffle(keys)
    else:
        keys = sorted(keys)
    return json.dumps({k: d[k] for k in keys}, ensure_ascii=False)


def build_item_texts_variant(items_pd, mode, rng):
    """items_pd has columns id/name/category/attributes (UNTRUNCATED
    attributes -- reordering must happen before the MAX_ATTR_CHARS slice,
    or a pre-truncated string is very likely invalid JSON to parse)."""
    names = items_pd["name"].fillna("").astype(str).str.slice(0, MAX_NAME_CHARS)
    categories = items_pd["category"].fillna("").astype(str).str.slice(0, MAX_CATEGORY_CHARS)
    if mode == "original":
        attrs_full = items_pd["attributes"].fillna("").astype(str)
    else:
        attrs_full = items_pd["attributes"].fillna("").astype(str).map(
            lambda s: _reorder_attributes(s, mode, rng)
        )
    attrs = attrs_full.str.slice(0, MAX_ATTR_CHARS)
    text = ("Name: " + names + " Category: " + categories + " Attributes: " + attrs).str.slice(0, MAX_ITEM_CHARS)
    return text.tolist()


def tokenize_texts_to_memmap(all_ids, all_texts, tokenizer, out_dir, max_length=ITEM_MAX_LENGTH):
    n = len(all_ids)
    ids_path = os.path.join(out_dir, "ids.bin")
    input_ids_path = os.path.join(out_dir, "input_ids.bin")
    attn_path = os.path.join(out_dir, "attention_mask.bin")

    bytes_per_row = 8 + max_length * 4 + max_length * 1
    _require_disk_space(out_dir, n * bytes_per_row, f"item token cache ({out_dir})")

    ids_mm = np.memmap(ids_path, dtype=np.int64, mode="w+", shape=(n,))
    input_ids_mm = np.memmap(input_ids_path, dtype=np.int32, mode="w+", shape=(n, max_length))
    attn_mm = np.memmap(attn_path, dtype=np.int8, mode="w+", shape=(n, max_length))
    ids_mm[:] = all_ids

    chunk_size = TOKENIZE_CHUNK_SIZE_START
    written = 0
    pbar = tqdm(total=n, desc=f"Tokenize {os.path.basename(out_dir)}", dynamic_ncols=True)

    while written < n:
        cur = min(chunk_size, n - written)
        chunk_texts = [PREFIX + t for t in all_texts[written:written + cur]]

        enc = tokenizer(
            chunk_texts, add_special_tokens=False,
            padding="max_length", truncation=True,
            max_length=max_length, return_tensors="np",
        )
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

    pbar.close()
    for mm in (ids_mm, input_ids_mm, attn_mm):
        mm.flush()
    del ids_mm, input_ids_mm, attn_mm
    gc.collect()

    input_ids = np.asarray(np.memmap(input_ids_path, dtype=np.int32, mode="r", shape=(n, max_length))).copy()
    attn = np.asarray(np.memmap(attn_path, dtype=np.int8, mode="r", shape=(n, max_length))).copy()
    return input_ids, attn


def _find_token_pattern_starts(arr, pattern):
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


def build_augmented_item_caches(tokenizer):
    """Returns (variant_arrays, id_to_row) where variant_arrays is a dict
    mode -> (input_ids, item_lengths, category_label_start, attr_label_start).
    category_label_start/attr_label_start are computed ONCE (on "original")
    and reused for "shuffled"/"sorted" -- see module docstring for why
    that's provably correct here, not an approximation."""
    items_path = _find_dataset_file(ITEMS_HUMAN_PATH_HINT, "items_human.parquet")
    print(f"[CACHE] loading items (untruncated) from {items_path} ...")
    items_pl = pl.scan_parquet(items_path).select(["id", "name", "category", "attributes"]).collect()
    items_pd = items_pl.to_pandas()
    del items_pl
    gc.collect()
    n_items = len(items_pd)
    all_ids = items_pd["id"].to_numpy().astype(np.int64)
    print(f"[CACHE] human scope items: {n_items:,}")
    _log_memory("after items load")

    rng = random.Random(SEED)
    variant_arrays = {}
    category_label_start = attr_label_start = None

    # "original" MUST be processed first -- category_label_start/
    # attr_label_start are only computed for it and then reused for the
    # other variants (see module docstring for why that's correct, not an
    # approximation). Storing a variant before "original" has run would
    # store None for these -- a real crash this experiment hit once
    # ("TypeError: 'NoneType' object is not subscriptable" in collate_fn)
    # when AUG_VARIANTS listed "shuffled"/"sorted" ahead of "original".
    ordered_modes = ["original"] + [m for m in AUG_VARIANTS if m != "original"]
    for mode in ordered_modes:
        t0 = time.perf_counter()
        texts = build_item_texts_variant(items_pd, mode, rng)
        cache_dir = os.path.join(WORKING_DIR, f"berta_item_tokens_{mode}")
        os.makedirs(cache_dir, exist_ok=True)
        input_ids, attn = tokenize_texts_to_memmap(all_ids, texts, tokenizer, cache_dir)
        item_lengths = attn.sum(axis=1).astype(np.int32)

        if mode == "original":
            print("[CACHE] locating Category:/Attributes: label positions (original variant only) ...")
            category_label_start = _find_token_pattern_starts(input_ids, CATEGORY_LABEL_PATTERN)
            attr_label_start = _find_token_pattern_starts(input_ids, ATTRIBUTES_LABEL_PATTERN)
            both_found = (category_label_start >= 0) & (attr_label_start >= 0)
            n_missing = n_items - int(both_found.sum())
            if n_missing > 0:
                print(f"[CACHE] WARNING: {n_missing:,}/{n_items:,} items missing a locatable label "
                      f"(no-excision fallback for those, all variants).")

        variant_arrays[mode] = (input_ids, item_lengths, category_label_start, attr_label_start)
        del texts
        gc.collect()
        _malloc_trim()
        print(f"[CACHE] variant={mode} done in {time.perf_counter() - t0:.1f}s | "
              f"real length mean={item_lengths.mean():.1f} p50={np.percentile(item_lengths, 50):.0f}")
        _log_memory(f"after variant={mode}")

    id_to_row = {int(item_id): i for i, item_id in enumerate(all_ids)}
    assert len(id_to_row) == n_items, f"duplicate item ids: {n_items} rows but {len(id_to_row)} unique ids"

    del items_pd
    gc.collect()
    return variant_arrays, id_to_row


# =====================================================================
# PAIR ASSEMBLY -- collate independently rolls an augmentation variant per
# item slot on every draw (see module docstring)
# =====================================================================

def allocate_pair_budget(len1, len2, budget):
    half = budget // 2
    if len1 <= half and len2 <= half:
        return len1, len2
    if len1 <= half:
        return len1, budget - len1
    if len2 <= half:
        return budget - len2, len2
    return half, budget - half


def build_compact_ids(item_input_ids, row, item_length, category_start, attr_start):
    if category_start < 0 or attr_start < 0:
        return item_input_ids[row, :item_length]
    head = item_input_ids[row, :category_start]
    tail = item_input_ids[row, attr_start:item_length]
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


def make_augmented_pair_collate_fn(variant_arrays, cls_id, sep_id, pad_id, pair_max_length,
                                     aug_variants=AUG_VARIANTS, aug_probs=AUG_PROBS, augment=True):
    """augment=False (used for the val loader) always uses the "original"
    variant -- val must stay byte-identical to every other BERTA eval run
    in this project for comparability, no augmentation there."""
    content_budget = pair_max_length - 3
    rng = np.random.RandomState(SEED + 1)

    def pick_variant():
        if not augment:
            return variant_arrays["original"]
        mode = aug_variants[rng.choice(len(aug_variants), p=aug_probs)]
        return variant_arrays[mode]

    def collate_fn(batch):
        row1, row2, labels, categories = zip(*batch)
        bsz = len(batch)
        input_ids = np.full((bsz, pair_max_length), pad_id, dtype=np.int64)
        token_type_ids = np.zeros((bsz, pair_max_length), dtype=np.int64)
        attention_mask = np.zeros((bsz, pair_max_length), dtype=np.int64)

        for i in range(bsz):
            r1, r2 = row1[i], row2[i]
            ii1, len1, cs1, as1 = pick_variant()
            ii2, len2, cs2, as2 = pick_variant()

            compact1 = build_compact_ids(ii1, r1, int(len1[r1]), int(cs1[r1]), int(as1[r1]))
            compact2 = build_compact_ids(ii2, r2, int(len2[r2]), int(cs2[r2]), int(as2[r2]))
            cl1, cl2 = len(compact1), len(compact2)

            if cl1 + cl2 <= content_budget:
                keep1, keep2 = cl1, cl2
            else:
                keep1, keep2 = allocate_pair_budget(cl1, cl2, content_budget)

            seg1 = compact1[:keep1]
            seg2 = compact2[:keep2]

            n1 = 1 + keep1 + 1
            n2 = keep2 + 1
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


def make_pair_loader(pairs_pd, id_to_row, category_to_id, variant_arrays, cls_id, sep_id, pad_id,
                      pair_max_length, batch_size, shuffle, augment):
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
    collate_fn = make_augmented_pair_collate_fn(variant_arrays, cls_id, sep_id, pad_id, pair_max_length,
                                                  augment=augment)
    drop_last = shuffle and (len(dataset) >= batch_size)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0,
        collate_fn=collate_fn, pin_memory=torch.cuda.is_available(), drop_last=drop_last,
    )
    return loader


# =====================================================================
# MODEL
# =====================================================================

class ClassifierHead(nn.Module):
    """Dropout before the final Linear -- matches berta-lora-sft-continue-5-
    epochs (the run that produced the checkpoint this experiment resumes
    from). State dict keys are "linear.weight"/"linear.bias", NOT bare
    "weight"/"bias" -- confirmed directly against a real crash."""

    def __init__(self, hidden_dim, dropout_p):
        super().__init__()
        self.dropout = nn.Dropout(dropout_p)
        self.linear = nn.Linear(hidden_dim, 1)

    def forward(self, pooled):
        return self.linear(self.dropout(pooled))


class BertaCrossEncoderClassifier(nn.Module):
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


def load_model_from_checkpoint(device, adapter_dir, classifier_path):
    print(f"[RESUME] Loading base encoder {MODEL_NAME} + resuming adapter from {adapter_dir} ...")
    base_model = AutoModel.from_pretrained(MODEL_NAME)
    hidden_dim = base_model.config.hidden_size
    peft_model = PeftModel.from_pretrained(base_model, adapter_dir, is_trainable=True)
    peft_model.print_trainable_parameters()

    n_trainable_lora = sum(p.numel() for n, p in peft_model.named_parameters() if "lora_" in n and p.requires_grad)
    assert n_trainable_lora > 0, "Resumed LoRA adapter has zero trainable params -- is_trainable didn't take."

    classifier_head = ClassifierHead(hidden_dim, CLASSIFIER_DROPOUT)
    raw_state = torch.load(classifier_path, map_location="cpu")
    if "linear.weight" in raw_state:
        classifier_head.load_state_dict(raw_state)
        print("[RESUME] classifier state loaded directly (checkpoint already has the dropout wrapper)")
    else:
        classifier_head.linear.load_state_dict(raw_state)
        print("[RESUME] classifier state loaded into .linear (pre-dropout checkpoint format)")
    model = BertaCrossEncoderClassifier(peft_model, classifier_head).to(device)
    return model


# =====================================================================
# VRAM AUTO-PROBE
# =====================================================================

def _probe_peak_gb(batch_size, n_gpu, pair_max_length, probe_model, probe_loss_fn, trainable_params, device,
                    train_mode=True):
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


def _find_resume_checkpoint_dir():
    if os.path.isdir("/kaggle/input"):
        for root, _dirs, files in os.walk("/kaggle/input"):
            if "train_state.json" in files and "adapter" in _dirs:
                print(f"[RESUME] found resume checkpoint at {root}")
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
    print("CONFIG -- Experiment: BERTA attribute-order augmentation")
    print("=" * 90)
    print(f"model:              {MODEL_NAME}")
    print(f"pair_max_length:    {PAIR_MAX_LENGTH}")
    print(f"epochs:             {EPOCHS}")
    print(f"aug_variants/probs: {AUG_VARIANTS} / {AUG_PROBS}")
    print(f"lr (fresh schedule):{LR}")
    print(f"total_batch_size:   {TOTAL_BATCH_SIZE}")
    print(f"max_train_seconds:  {MAX_TRAIN_SECONDS} ({MAX_TRAIN_SECONDS / 3600:.2f}h)")
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
    # Canonical human split (byte-identical split logic; unchanged).
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
    train_pd["label"] = train_pd["label"].astype(np.float32)
    val_pd["label"] = val_pd["label"].astype(np.float32)
    print(f"[DATA] train={len(train_pd):,} val={len(val_pd):,} (expect 292,523 / 73,131)")

    val_fingerprint = hashlib.sha256(
        ",".join(f"{a}-{b}" for a, b in sorted(zip(val_pd["id1"], val_pd["id2"]))).encode("utf-8")
    ).hexdigest()[:16]
    print(f"[DATA] val_fingerprint={val_fingerprint}")

    del matches_pd, labels_all, indices, train_idx, val_idx
    gc.collect()

    # -------------------------------------------------------------
    # Attribute-augmented item caches (3 variants).
    # -------------------------------------------------------------
    variant_arrays, id_to_row = build_augmented_item_caches(tokenizer)
    _log_memory("after augmented item caches")

    items_path = _find_dataset_file(ITEMS_HUMAN_PATH_HINT, "items_human.parquet")
    needed_ids = pd.unique(pd.concat([train_pd["id1"], train_pd["id2"], val_pd["id1"], val_pd["id2"]]))
    needed_ids_lazy = pl.DataFrame({"id": pd.Series(needed_ids, dtype="int64")}).lazy()
    items_cat = (
        pl.scan_parquet(items_path)
        .join(needed_ids_lazy, on="id", how="semi")
        .select(["id", "category"])
        .with_columns(pl.col("category").fill_null("UNKNOWN").cast(pl.Utf8).str.slice(0, MAX_CATEGORY_CHARS))
        .collect(engine="streaming")
    )
    category_vocab = sorted(items_cat.get_column("category").unique().to_list()) + ["UNKNOWN"]
    category_vocab = sorted(set(category_vocab))
    category_to_id = {c: i for i, c in enumerate(category_vocab)}
    print(f"[DATA] category vocab size: {len(category_vocab)}")

    id_to_category = dict(zip(items_cat.get_column("id").to_list(), items_cat.get_column("category").to_list()))
    train_pd["category1"] = train_pd["id1"].map(id_to_category).fillna("UNKNOWN")
    val_pd["category1"] = val_pd["id1"].map(id_to_category).fillna("UNKNOWN")
    del items_cat, needed_ids, needed_ids_lazy, id_to_category
    gc.collect()

    micro_batch_size = MICRO_BATCH_SIZE
    eval_batch_size = EVAL_MICRO_BATCH_SIZE
    total_batch_size = TOTAL_BATCH_SIZE

    train_loader = make_pair_loader(
        train_pd, id_to_row, category_to_id, variant_arrays, cls_id, sep_id, pad_id, PAIR_MAX_LENGTH,
        micro_batch_size, shuffle=True, augment=True,
    )
    val_loader = make_pair_loader(
        val_pd, id_to_row, category_to_id, variant_arrays, cls_id, sep_id, pad_id, PAIR_MAX_LENGTH,
        eval_batch_size, shuffle=False, augment=False,
    )

    pos_rate = float(train_pd["label"].mean())
    print(f"[DATA] train positive rate: {pos_rate:.4f}")

    # -------------------------------------------------------------
    # Model: MUST resume from the published best checkpoint.
    # -------------------------------------------------------------
    checkpoint_dir = _find_resume_checkpoint_dir()
    if checkpoint_dir is None:
        raise RuntimeError("Resume checkpoint not found under /kaggle/input -- expected "
                            "maxamsecret/berta-lora-sft-continue-5-epochs mounted (kernel_sources).")
    model = load_model_from_checkpoint(device, os.path.join(checkpoint_dir, "adapter"),
                                        os.path.join(checkpoint_dir, "classifier.pt"))
    with open(os.path.join(checkpoint_dir, "train_state.json")) as f:
        source_state = json.load(f)
    print(f"[RESUME] source checkpoint: completed_epochs={source_state['completed_epochs']} "
          f"global_step={source_state['global_step']} "
          f"best_val_macro_pr_auc={source_state['best_val_macro_pr_auc']}")

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

    micro_batch_size = run_vram_probe(model, trainable_params, device, n_gpu, PAIR_MAX_LENGTH, micro_batch_size,
                                       probe_cap=min(PROBE_CAP, TOTAL_BATCH_SIZE), train_mode=True,
                                       label="MICRO_BATCH_SIZE")
    grad_accum = max(1, total_batch_size // micro_batch_size)

    model.eval()
    eval_batch_size = run_vram_probe(model, trainable_params, device, n_gpu, PAIR_MAX_LENGTH, eval_batch_size,
                                      probe_cap=EVAL_PROBE_CAP, train_mode=False, label="EVAL_MICRO_BATCH_SIZE")
    model.train()

    print(f"[PROBE] Rebuilding data loaders with the new batch size ...")
    train_loader = make_pair_loader(
        train_pd, id_to_row, category_to_id, variant_arrays, cls_id, sep_id, pad_id, PAIR_MAX_LENGTH,
        micro_batch_size, shuffle=True, augment=True,
    )
    val_loader = make_pair_loader(
        val_pd, id_to_row, category_to_id, variant_arrays, cls_id, sep_id, pad_id, PAIR_MAX_LENGTH,
        eval_batch_size, shuffle=False, augment=False,
    )
    print(f"[PROBE] New config: MICRO_BATCH_SIZE={micro_batch_size} | EVAL_BATCH_SIZE={eval_batch_size} | "
          f"GRAD_ACCUM={grad_accum} | effective TOTAL_BATCH_SIZE={micro_batch_size * grad_accum}")

    steps_per_epoch = math.ceil(len(train_loader) / grad_accum)
    total_steps = steps_per_epoch * EPOCHS
    warmup_steps = max(1, int(total_steps * WARMUP_RATIO))
    print(f"[TRAIN] steps/epoch: {steps_per_epoch} | total steps: {total_steps} | "
          f"warmup: {warmup_steps} ({WARMUP_RATIO:.0%}) -> cosine decay to 0")

    pos_weight = torch.tensor((1.0 - pos_rate) / max(pos_rate, 1e-9), device=device)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    print(f"[LOSS] BCE pos_weight: {pos_weight.item():.4f}")

    optimizer = torch.optim.AdamW(trainable_params, lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = get_scheduler("cosine", optimizer=optimizer, num_warmup_steps=warmup_steps,
                               num_training_steps=total_steps)

    optimizer.load_state_dict(torch.load(os.path.join(checkpoint_dir, "optimizer.pt"), map_location=device))
    print("[RESUME] optimizer state (Adam moments) restored from checkpoint; scheduler is fresh for this run")

    global_step = 0
    best_val_macro_pr_auc = source_state["best_val_macro_pr_auc"]
    n_skipped_microbatches = 0
    n_skipped_optimizer_steps = 0
    category_names = category_vocab

    def run_checkpoint(epoch_display, window_state, train_loop_start):
        nonlocal best_val_macro_pr_auc
        window_logits, window_labels, window_categories, window_loss_sum, window_loss_count = window_state

        val_metrics = evaluate(model, val_loader, loss_fn, device, category_names)
        improved = val_metrics["macro_pr_auc"] > best_val_macro_pr_auc
        if improved:
            best_val_macro_pr_auc = val_metrics["macro_pr_auc"]

        train_metrics = {}
        if window_loss_count[0] > 0:
            try:
                train_probs = torch.sigmoid(torch.cat(window_logits)).numpy()
                train_labels_np = torch.cat(window_labels).numpy()
                train_cats_np = np.concatenate(window_categories)
                train_macro_pr_auc, _ = compute_metrics(train_probs, train_labels_np, train_cats_np, category_names)
                train_metrics = {"loss": window_loss_sum[0] / window_loss_count[0], "macro_pr_auc": train_macro_pr_auc}
            except Exception as e:
                print(f"[WARN] train-window macro_pr_auc computation failed (non-fatal): {e!r}", flush=True)
                train_metrics = {"loss": window_loss_sum[0] / window_loss_count[0], "macro_pr_auc": float("nan")}
            window_logits.clear(); window_labels.clear(); window_categories.clear()
            window_loss_sum[0] = 0.0; window_loss_count[0] = 0

        print(f"[METRICS] epoch={epoch_display} step={global_step} "
              f"train_loss={train_metrics.get('loss', float('nan')):.4f} "
              f"train_macro_pr_auc={train_metrics.get('macro_pr_auc', float('nan')):.4f} "
              f"val_loss={val_metrics['loss']:.4f} val_macro_pr_auc={val_metrics['macro_pr_auc']:.4f} "
              f"best={best_val_macro_pr_auc:.4f} (source was {source_state['best_val_macro_pr_auc']:.4f}) "
              f"skipped_micro={n_skipped_microbatches} skipped_opt={n_skipped_optimizer_steps}", flush=True)
        print(f"[METRICS] per_category val: {json.dumps(val_metrics['per_category'], ensure_ascii=False, indent=2)}")

        mlflow.log_metrics({
            "train_loss": train_metrics.get("loss", float("nan")),
            "train_macro_pr_auc": train_metrics.get("macro_pr_auc", float("nan")),
            "val_loss": val_metrics["loss"], "val_macro_pr_auc": val_metrics["macro_pr_auc"],
            "n_skipped_microbatches": n_skipped_microbatches, "n_skipped_optimizer_steps": n_skipped_optimizer_steps,
        }, step=global_step)

        save_checkpoint(model, optimizer, scheduler, epoch_display, global_step, best_val_macro_pr_auc,
                         n_skipped_microbatches, n_skipped_optimizer_steps, CHECKPOINT_DIR)
        publish_checkpoint_dataset(CHECKPOINT_DIR, f"epoch={epoch_display} step={global_step} "
                                    f"val_macro_pr_auc={val_metrics['macro_pr_auc']:.4f}", has_kaggle_creds)
        if improved:
            print(f"[SAVE] NEW BEST val_macro_pr_auc={best_val_macro_pr_auc:.4f}", flush=True)
        model.train()
        return val_metrics

    with mlflow.start_run():
        mlflow.log_params({
            "experiment": "attr_augment", "aug_variants": ",".join(AUG_VARIANTS),
            "aug_probs": ",".join(str(p) for p in AUG_PROBS), "model": MODEL_NAME,
            "pair_max_length": PAIR_MAX_LENGTH, "epochs": EPOCHS, "lr": LR,
            "lora_r": LORA_R, "lora_alpha": LORA_ALPHA, "lora_dropout": LORA_DROPOUT,
            "classifier_dropout": CLASSIFIER_DROPOUT,
            "lora_target_modules": ",".join(LORA_TARGET_MODULES), "total_batch_size": total_batch_size,
            "micro_batch_size": micro_batch_size, "val_fingerprint": val_fingerprint,
            "resumed_from_global_step": source_state["global_step"],
            "resumed_from_best_val_macro_pr_auc": source_state["best_val_macro_pr_auc"],
            "max_train_seconds": MAX_TRAIN_SECONDS,
        })

        window_logits, window_labels, window_categories = [], [], []
        window_loss_sum, window_loss_count = [0.0], [0]
        window_state = (window_logits, window_labels, window_categories, window_loss_sum, window_loss_count)
        train_loop_start = time.perf_counter()
        stopped_early = False
        val_metrics = None

        for epoch in range(EPOCHS):
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

                logits = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
                loss = loss_fn(logits, batch_labels_dev)

                if not torch.isfinite(loss):
                    n_skipped_microbatches += 1
                    print(f"[WARN] step={step} non-finite loss -- skipping microbatch "
                          f"(total skipped: {n_skipped_microbatches})", flush=True)
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
                        print(f"[WARN] optimizer step skipped: non-finite gradient "
                              f"(total skipped: {n_skipped_optimizer_steps})", flush=True)
                    optimizer.zero_grad()

                    elapsed_train = time.perf_counter() - train_loop_start
                    if elapsed_train > MAX_TRAIN_SECONDS:
                        print(f"[TRAIN] wall-clock cutoff reached ({elapsed_train:.0f}s > "
                              f"{MAX_TRAIN_SECONDS}s) -- stopping early, proceeding to eval+checkpoint.", flush=True)
                        stopped_early = True
                        break

            pbar.close()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            val_metrics = run_checkpoint(epoch + 1, window_state, train_loop_start)
            if stopped_early:
                break

    print(f"[DONE] Experiment attr_augment complete. val_macro_pr_auc={val_metrics['macro_pr_auc']:.4f} "
          f"(source was {source_state['best_val_macro_pr_auc']:.4f}). total wall-clock: {_elapsed():.0f}s")


if __name__ == "__main__":
    main()
