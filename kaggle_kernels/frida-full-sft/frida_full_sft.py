"""
FRIDA (ai-forever/FRIDA, T5EncoderModel, 823M) full fine-tune from the
PRETRAINED base -- no LoRA, no resumed adapter. A second, independent
training track alongside kaggle_kernels/frida-percat-lora's 20 specialists:
where that kernel specializes an already-strong general model per category,
this one asks whether a genuinely fresh full-parameter fit, with its own
split/loss/optimizer choices, reaches a better general model in the first
place.

WHY FULL FINE-TUNE, AND WHY THAT FORCES ADAFACTOR
===========================================================================
Every FRIDA checkpoint in this project to date is LoRA. Full-FT unlocks
capacity LoRA structurally cannot touch (LayerNorm, embeddings, biases), but
Adam's own memory cost is the reason none of this project's TPU kernels have
done it: fp32 weights + fp32 grad + Adam (m, v) = 4x the 3.29GB parameter
tensor = 13.2GB, before a single activation, against a 16GB v5e-8 core.

Adafactor factorizes the second moment into a per-row and per-column vector
instead of a full matrix, so its own state is a few MB per weight tensor
rather than 1x its size. Weights + grad + (factored) state ~= 6.6GB, leaving
~9GB for activations -- and it is genuinely the native choice here, not a
memory hack bolted on: it is the T5 family's own paper optimizer, and FRIDA
IS a T5 encoder. Combined with gradient checkpointing (recomputing forward
activations during backward instead of storing all 24 layers' worth), the
per-sample activation cost drops to O(1) layers instead of O(24), which is
what the batch size below is sized against.

PER_CORE_TRAIN_BATCH is a documented ARITHMETIC ESTIMATE, not a measured
point -- consistent with this project's own TPU rule (kaggle-experiment-
safety: never auto-probe batch size on TPU, every probed candidate compiles
its own XLA program and stays resident in HBM, self-inflicting the OOM it
was meant to prevent). If a real run OOMs, halve it and relaunch; that is a
cheap, fast failure at the first training step, not an expensive one.

SPLIT: THE NEW ITEM-DISJOINT ONE, DELIBERATELY
===========================================================================
This is a FROM-SCRATCH run, so it is free to adopt
data/splits/matches_item_disjoint_70_30_seed42.parquet (union-find over the
pair graph, then stratified 70/30 by category x label x length-tercile; 0
shared item ids between train and val, verified). The OLD canonical
pair-level 80/20 split stays reserved for anything that continues an
existing checkpoint (see kaggle_kernels/frida-percat-lora, which must not
move items an already-trained adapter has seen into a "new" val half).

LLM DATA: A SUBSAMPLE, NOT THE FULL 11.19M ROWS -- SAY SO EVERYWHERE
===========================================================================
Measured throughput math (this project's own 120 rows/s at
PAIR_MAX_LENGTH=256 on frida-tpu-human-sft, scaled to seq=192 and to a
full-FT+Adafactor+grad-checkpointing forward/backward -- ~128 rows/s):
against the full 11,187,780-row LLM set, 3 epochs would need ~74h, well past
even a full week's 20h TPU quota. LLM_TOTAL_ROWS=1,000,000 (not per
category, unlike frida-percat-lora -- this run pools all categories into one
model) keeps 3 epochs inside a single ~8.2h session:
3 x (256,006 human-train + 1,000,000 llm) / 128 rows/s ~= 8.18h.

Subsample policy is the SAME stratified_llm_sample as frida-percat-lora
(duplicated here -- kernels are standalone scripts): category-balanced
quota, stratified by the 10 discrete target levels within each category, no
confidence filter (see that kernel's docstring for the full reasoning --
short version: matches_llm targets are a 9-judge vote count, already 81.25%
"confident" by nature, and PR-AUC is decided at the boundary that filtering
would discard). README.md carries the same prominent warning: any reported
number from this run describes training on "human + an LLM subsample", not
"human + LLM".

THE HEAD: [1024, 512, 128, 1], OPPOSITE-DIRECTION SCHEDULE FROM THE BACKBONE
===========================================================================
A random head sitting on top of a pretrained backbone is the textbook case
for two learning rates: early on, large gradients from the untrained head
would otherwise backpropagate noise into the encoder. The two groups get
literally opposite-shaped warmup:
  - backbone: 0 -> LR linearly over the warmup window (ordinary warmup),
    then cosine decay to a floor -- exactly frida-percat-lora's schedule.
  - head: HEAD_LR_PEAK -> LR linearly over the SAME warmup window (starts
    high, "cools" down to the backbone's own peak), then joins the identical
    cosine decay -- so by the time warmup ends, both groups are moving
    together at the same rate for the remainder of training.
This lets the head get useful fast while it still cannot dominate the
encoder's own gradient signal early, and removes the two-schedule split
entirely once both are past the point where head-vs-backbone imbalance
would matter.

LOSS: CATEGORY-WEIGHTED FOCAL LOSS, PLUS 2x ON HUMAN ROWS
===========================================================================
This is genuinely different from frida-percat-lora's per-category
pos_weight, and for a real structural reason: that kernel trains 20
independent models, one per category, so each one only ever sees its own
category's positive rate -- category SIZE never enters the picture because
there is no pooling across categories to begin with. This kernel pools every
category into ONE model and ONE loss, where a large category's gradient
would otherwise simply outvote a small one during standard mean-reduction.
Measured over the real data (data/splits/matches_item_disjoint_70_30_seed42.
json): category sizes span only 17,286-23,357 pairs (1.4x) in the human set,
but the human+LLM-subsample TRAINING mix skews further once LLM rows are
folded in -- weighting by that mix's own realized category counts (computed
at setup time, not assumed) is what actually corrects it here, unlike
frida-percat-lora where it would have been close to a no-op.

Focal modulation (Lin et al., gamma=2.0, the standard literature default --
no other value was specified) downweights already-easy examples so the loss
budget concentrates on the ones still being gotten wrong, which matters more
here than in frida-percat-lora precisely because pooling means an easy
majority category could otherwise dominate total loss mass even after
per-category reweighting. HUMAN_WEIGHT=2.0 (same convention as
frida-percat-lora) doubles human-labeled rows against LLM soft-labeled ones
in the same weighted-mean reduction.

CHECKPOINT CONTRACT (per direct instruction, different from every other
kernel in this project)
===========================================================================
  - Every CHECKPOINT_EVERY_SECONDS (5h): weights + optimizer state ONLY, NO
    eval, val score written as an explicit `null` (this project's own
    established convention for "skipped, not stale" -- see
    kaggle-experiment-safety's checkpoint-hygiene section). Adafactor's own
    state is small enough that this is cheap; running eval every 5h on top
    of an 8.2h budget would itself eat a real fraction of the session.
  - Real evaluation runs ONLY at an epoch boundary.
  - A cancelled/hard-capped session still leaves its last quiet save in
    /kaggle/working -- Kaggle persists that regardless of how the session
    ends, same guarantee every other checkpoint in this project's history
    has leaned on.

TPU LESSONS CARRIED FORWARD (all real, previously hit -- see
frida-percat-lora / frida-llm-tpu-sft / berta-llm-tpu-sft docstrings for the
original incidents)
===========================================================================
  1. os.environ.pop("TPU_PROCESS_ADDRESSES"/"CLOUD_TPU_TASK_ID") +
     start_method="fork".
  2. STATIC SHAPES: drop_last=True on every loader, fixed per-core batch.
  3. FIXED batch size, no auto-probing on TPU.
  4. xm.optimizer_step(optimizer), not optimizer.step().
  5. autocast opened INSIDE forward().
  6. ALL CPU-only setup ONCE in main() before torch_xla.launch().
  7. Eval wrapped in try/except that cannot kill the run.
  8. Checkpoint hygiene: own directory per save, explicit null val score,
     wall-clock (not epoch) checkpoint cadence.
"""
import os
import sys
import gc
import json
import math
import time
import hashlib
import subprocess

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "polars"])

import numpy as np
import pandas as pd
import polars as pl

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from transformers import T5EncoderModel, set_seed
from transformers.optimization import Adafactor
from sklearn.metrics import average_precision_score

import torch_xla
import torch_xla.core.xla_model as xm
from torch_xla import runtime as xr
import torch_xla.distributed.parallel_loader as pl_xla


# =====================================================================
# CONFIG
# =====================================================================

SEED = 42
MODEL_NAME = "ai-forever/FRIDA"

MATCHES_LLM_PATH_HINT = "/kaggle/input/datasets/maxamsecret/ozon-ecup-2026-matching/matches_llm.parquet"
ITEMS_FULL_PATH_HINT = "/kaggle/input/datasets/maxamsecret/ozon-ecup-2026-matching/items.parquet"
ITEM_SPLIT_PATH_HINT = "/kaggle/input/datasets/maxamsecret/ozon-ecup-2026-matching/matches_item_disjoint_70_30_seed42.parquet"

# v2 caches (frida-tokenize-items-cpu): correct char budget (140/32/529, see
# reports/experiment_log.md "Char-budget rule v2") + the light v2 attribute
# cleanup. Human scope alone would be items_human.parquet-derived; this run
# needs BOTH scopes (human for train+val, LLM for the subsample), so both
# halves of the one published dataset are used.
HUMAN_CACHE_HINT = "/kaggle/input/datasets/maxamsecret/frida-tokenize-items-v3/human_scope"
LLM_CACHE_HINT = "/kaggle/input/datasets/maxamsecret/frida-tokenize-items-v3/llm_scope"

ITEM_MAX_LENGTH = 256
# Same reasoning as frida-percat-lora: 192 is what the real judging stand can
# actually SERVE inside its time budget (176 passed at 0.4087; 192 measured
# at 1.65x over Public's budget). Training at the serving length removes the
# train/serve mismatch every FRIDA submission so far has paid for.
PAIR_MAX_LENGTH = 192

MAX_CATEGORY_CHARS = 32   # matches the v2 cache's own NAME/CATEGORY/ATTR budget

CATEGORY_LABEL_TOKEN = 55917    # " Category" (+ colon=30 immediately after)
ATTRIBUTES_LABEL_TOKEN = 91837  # " Attributes"
LABEL_COLON_TOKEN = 30

# Head: [1024, 512, 128, 1] per direct instruction -- three hidden blocks
# (Linear+LayerNorm+GELU+Dropout) narrowing from the encoder's 1536-dim
# pooled output down to a single logit.
HEAD_HIDDEN = (1024, 512, 128)
HEAD_DROPOUT = 0.2

# Backbone warms 0 -> LR; head starts at LR * HEAD_LR_PEAK_MULTIPLIER and
# cools DOWN to LR over the same warmup window (see docstring). Both then
# share one cosine decay to LR_MIN_RATIO * LR.
LR = 1e-4                       # backbone peak -- lower than LoRA's 2e-4 on
                                 # purpose: full-FT touches every parameter,
                                 # not a small low-rank delta, so a LoRA-sized
                                 # LR here would be a much larger relative step
HEAD_LR_PEAK_MULTIPLIER = 10.0  # head starts at LR * this (1e-3), typical
                                 # for a freshly-initialised small MLP head
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.10
LR_MIN_RATIO = 0.05
HUMAN_WEIGHT = 2.0               # human rows count double vs LLM soft labels
FOCAL_GAMMA = 2.0                # Lin et al. RetinaNet default; not tuned

# ARITHMETIC ESTIMATE, not measured -- see docstring. Halve on a real OOM.
PER_CORE_TRAIN_BATCH = 64
PER_CORE_EVAL_BATCH = 96

# NOT the full 11,187,780-row LLM set -- see docstring's "LLM DATA" section.
LLM_TOTAL_ROWS = 1_000_000
EPOCHS = 3

MAX_TRAIN_SECONDS = 30600       # 8.5h -- Kaggle's TPU session cap is 9h
CHECKPOINT_EVERY_SECONDS = 18000  # 5h, weights+optimizer only, no eval (see
                                   # docstring's CHECKPOINT CONTRACT)
LOG_EVERY_STEPS = 50

# 128 measured-and-scaled (see docstring) at seq=192, full-FT+grad-checkpoint
# -- used ONLY to size the LR schedule's horizon, not as a promise.
PROJECTED_ROWS_PER_SEC = 128

WORKING_DIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
OUTPUT_DIR = os.path.join(WORKING_DIR, "frida_full_sft")

t_script_start = time.perf_counter()


def _elapsed():
    return time.perf_counter() - t_script_start


def _log(rank, stage, msg, force=False):
    if rank == 0 or force:
        print(f"[{_elapsed():8.1f}s][rank{rank}] {stage}: {msg}", flush=True)


def _find_dataset_file(hint, filename):
    if os.path.isfile(hint):
        return hint
    if os.path.isdir("/kaggle/input"):
        for root, _dirs, files in os.walk("/kaggle/input"):
            if filename in files:
                return os.path.join(root, filename)
    return hint


# =====================================================================
# ITEM CACHES -- v2 layout (same flat ids/input_ids/attention_mask shape as
# v1; only the char-budget/attribute-policy that built them differs, which
# is invisible at this layer). Verbatim from frida-percat-lora/
# frida-llm-tpu-sft -- proven on real multi-hour TPU runs, not re-derived.
# =====================================================================

def _pattern_starts_chunked(input_ids, n_items, tok_a, tok_b, chunk=500_000):
    starts = np.full(n_items, -1, dtype=np.int32)
    for lo in range(0, n_items, chunk):
        hi = min(lo + chunk, n_items)
        block = np.asarray(input_ids[lo:hi])
        mask = (block[:, :-1] == tok_a) & (block[:, 1:] == tok_b)
        rows, cols = np.where(mask)
        if len(rows):
            uniq_rows, first_idx = np.unique(rows, return_index=True)
            starts[lo + uniq_rows] = cols[first_idx]
        del block, mask, rows, cols
    return starts


def _lengths_chunked(attn, n_items, chunk=500_000):
    lengths = np.empty(n_items, dtype=np.int32)
    for lo in range(0, n_items, chunk):
        hi = min(lo + chunk, n_items)
        lengths[lo:hi] = np.asarray(attn[lo:hi]).sum(axis=1)
    return lengths


EXPECTED_N_ITEMS = {"human": 711_304, "llm": 12_384_610}


def _find_v1_cache_root(hint, expect_n_items):
    for candidate in (hint,):
        meta_path = os.path.join(candidate, "cache_meta.json")
        if os.path.isdir(candidate) and os.path.exists(meta_path):
            with open(meta_path) as f:
                if json.load(f).get("n_items") == expect_n_items:
                    return candidate
    if os.path.isdir("/kaggle/input"):
        for root, _dirs, files in os.walk("/kaggle/input"):
            if "cache_meta.json" in files and "input_ids.bin" in files:
                with open(os.path.join(root, "cache_meta.json")) as f:
                    meta = json.load(f)
                if meta.get("n_items") == expect_n_items:
                    print(f"[CACHE] hint {hint} missed -- matched n_items={expect_n_items:,} "
                          f"at {root}", flush=True)
                    return root
    raise RuntimeError(f"[CACHE] no item cache with n_items={expect_n_items:,} found "
                       f"(hint was {hint})")


def load_v1_cache(hint, name):
    root = _find_v1_cache_root(hint, EXPECTED_N_ITEMS[name])
    with open(os.path.join(root, "cache_meta.json")) as f:
        meta = json.load(f)
    n_items, max_length = meta["n_items"], meta["max_length"]
    print(f"[CACHE:{name}] {root}: n_items={n_items:,} max_length={max_length} "
          f"prefix={meta.get('prefix')!r}", flush=True)
    if max_length != ITEM_MAX_LENGTH:
        raise RuntimeError(f"[CACHE:{name}] max_length={max_length} != {ITEM_MAX_LENGTH}")

    ids_path = os.path.join(root, "ids.bin")
    input_ids_path = os.path.join(root, "input_ids.bin")
    attn_path = os.path.join(root, "attention_mask.bin")
    for fname, want in ((ids_path, n_items * 8),
                        (input_ids_path, n_items * max_length * 4),
                        (attn_path, n_items * max_length)):
        got = os.path.getsize(fname)
        if got != want:
            raise RuntimeError(f"[CACHE:{name}] {fname} size mismatch: got {got:,}, want {want:,}")

    ids = np.asarray(np.memmap(ids_path, dtype=np.int64, mode="r", shape=(n_items,)))
    input_ids = np.memmap(input_ids_path, dtype=np.int32, mode="r", shape=(n_items, max_length))
    attn = np.memmap(attn_path, dtype=np.int8, mode="r", shape=(n_items, max_length))

    t0 = time.perf_counter()
    lengths = _lengths_chunked(attn, n_items)
    del attn
    cat_start = _pattern_starts_chunked(input_ids, n_items, CATEGORY_LABEL_TOKEN, LABEL_COLON_TOKEN)
    attr_start = _pattern_starts_chunked(input_ids, n_items, ATTRIBUTES_LABEL_TOKEN, LABEL_COLON_TOKEN)
    print(f"[CACHE:{name}] lengths + label positions scanned ({time.perf_counter() - t0:.1f}s)",
          flush=True)

    sample_rows = np.random.RandomState(SEED).choice(n_items, size=min(50, n_items), replace=False)
    for r in sample_rows:
        L = int(lengths[r])
        assert input_ids[r, 0] == 1, f"[{name}] row {r}: expected bos=1, got {input_ids[r, 0]}"
        assert input_ids[r, L - 1] == 2, f"[{name}] row {r}: expected eos=2, got {input_ids[r, L - 1]}"

    order = np.argsort(ids, kind="stable")
    sorted_ids = ids[order]

    pad_id = _derive_pad_id(input_ids, lengths, max_length, name)
    return dict(name=name, n_items=n_items, input_ids=input_ids, lengths=lengths,
                cat_start=cat_start, attr_start=attr_start, order=order,
                sorted_ids=sorted_ids, pad_id=pad_id)


def _derive_pad_id(input_ids, lengths, max_length, name):
    candidates = np.where(lengths < max_length)[0]
    if len(candidates) == 0:
        raise RuntimeError(f"[CACHE:{name}] every row is at max_length -- cannot derive pad_id")
    sample = candidates[:: max(1, len(candidates) // 500)][:500]
    pads = {int(input_ids[r, int(lengths[r])]) for r in sample}
    if len(pads) != 1:
        raise RuntimeError(f"[CACHE:{name}] inconsistent pad token: {sorted(pads)}")
    pad = pads.pop()
    print(f"[CACHE:{name}] pad_id={pad} (derived from the cache's own padding)", flush=True)
    return pad


def lookup_rows(cache, item_ids):
    pos = np.searchsorted(cache["sorted_ids"], item_ids)
    pos_clipped = np.clip(pos, 0, len(cache["sorted_ids"]) - 1)
    hit = cache["sorted_ids"][pos_clipped] == item_ids
    rows = np.where(hit, cache["order"][pos_clipped], -1)
    return rows.astype(np.int64)


# =====================================================================
# PAIR ASSEMBLY (byte-identical to every other FRIDA kernel)
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


def build_compact_ids(cache, row):
    item_length = int(cache["lengths"][row])
    category_start = int(cache["cat_start"][row])
    attr_start = int(cache["attr_start"][row])
    ids = cache["input_ids"]
    if category_start < 0 or attr_start < 0:
        return np.asarray(ids[row, :item_length])
    head = np.asarray(ids[row, :category_start])
    tail = np.asarray(ids[row, attr_start:item_length])
    return np.concatenate((head, tail))


class PairIndexDataset(Dataset):
    def __init__(self, scope, row1, row2, labels, categories):
        self.scope = scope
        self.row1 = row1
        self.row2 = row2
        self.labels = labels
        self.categories = categories

    def __len__(self):
        return len(self.row1)

    def __getitem__(self, i):
        return self.scope[i], self.row1[i], self.row2[i], self.labels[i], self.categories[i]


def make_pair_collate_fn(caches, pad_id, pair_max_length):
    def collate_fn(batch):
        scope, row1, row2, labels, categories = zip(*batch)
        bsz = len(batch)
        input_ids = np.full((bsz, pair_max_length), pad_id, dtype=np.int64)
        attention_mask = np.zeros((bsz, pair_max_length), dtype=np.int64)
        for i in range(bsz):
            cache = caches[int(scope[i])]
            c1 = build_compact_ids(cache, int(row1[i]))
            c2 = build_compact_ids(cache, int(row2[i]))
            l1, l2 = len(c1), len(c2)
            if l1 + l2 <= pair_max_length:
                k1, k2 = l1, l2
            else:
                k1, k2 = allocate_pair_budget(l1, l2, pair_max_length)
            combined = np.concatenate((c1[:k1], c2[:k2]))
            n = len(combined)
            input_ids[i, :n] = combined
            attention_mask[i, :n] = 1
        return (torch.from_numpy(input_ids), torch.from_numpy(attention_mask),
                torch.tensor(labels, dtype=torch.float32),
                np.asarray(categories, dtype=np.int64),
                torch.tensor([int(s) for s in scope], dtype=torch.int64))
    return collate_fn


def make_train_loader(frame, caches, pad_id, per_core_batch, num_replicas, rank, seed, epoch):
    dataset = PairIndexDataset(frame["scope"], frame["row1"], frame["row2"],
                               frame["label"], frame["category"])
    sampler = DistributedSampler(dataset, num_replicas=num_replicas, rank=rank, shuffle=True,
                                 seed=seed, drop_last=True)
    sampler.set_epoch(epoch)
    loader = DataLoader(dataset, batch_size=per_core_batch, sampler=sampler, drop_last=True,
                        collate_fn=make_pair_collate_fn(caches, pad_id, PAIR_MAX_LENGTH),
                        num_workers=2, persistent_workers=False)
    return loader, sampler


def make_eval_loader(frame, caches, pad_id, eval_batch):
    dataset = PairIndexDataset(frame["scope"], frame["row1"], frame["row2"],
                               frame["label"], frame["category"])
    collate_fn = make_pair_collate_fn(caches, pad_id, PAIR_MAX_LENGTH)
    n = len(dataset)
    n_full = (n // eval_batch) * eval_batch
    full_loader = DataLoader(torch.utils.data.Subset(dataset, list(range(n_full))),
                             batch_size=eval_batch, shuffle=False, drop_last=True,
                             collate_fn=collate_fn, num_workers=2)
    remainder_loader = None
    if n_full < n:
        remainder_loader = DataLoader(torch.utils.data.Subset(dataset, list(range(n_full, n))),
                                      batch_size=n - n_full, shuffle=False, drop_last=False,
                                      collate_fn=collate_fn, num_workers=0)
    return full_loader, remainder_loader


# =====================================================================
# MODEL -- genuine full fine-tune, no PeftModel anywhere
# =====================================================================

class FullSftHead(nn.Module):
    """[1024, 512, 128, 1] per direct instruction -- three Linear+LayerNorm+
    GELU+Dropout blocks narrowing the encoder's pooled 1536-dim [CLS] output
    down to one logit."""

    def __init__(self, hidden_dim, layer_sizes=HEAD_HIDDEN, dropout=HEAD_DROPOUT):
        super().__init__()
        blocks = []
        d = hidden_dim
        for h in layer_sizes:
            blocks.append(nn.Sequential(
                nn.Linear(d, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(dropout)))
            d = h
        self.blocks = nn.ModuleList(blocks)
        self.out = nn.Linear(d, 1)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.out(x).view(-1)


class FridaFullSftClassifier(nn.Module):
    def __init__(self, encoder, head):
        super().__init__()
        self.encoder = encoder
        self.classifier = head

    def forward(self, input_ids, attention_mask):
        with torch.autocast("xla", dtype=torch.bfloat16):
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :].float()
        return self.classifier(cls)


def build_fresh_model(device, rank):
    """No PeftModel, no adapter, no bf16-frozen base -- every parameter is
    trainable and kept in fp32 (autocast handles the bf16 forward pass; see
    forward() above), which is what "full fine-tune" means here. Gradient
    checkpointing recomputes each layer's activations during backward
    instead of storing all 24 layers' worth -- the memory trade this
    kernel's batch size is sized against (see module docstring)."""
    model = T5EncoderModel.from_pretrained(MODEL_NAME, dtype=torch.float32,
                                           attn_implementation="eager")
    model.gradient_checkpointing_enable()
    hidden_dim = model.config.d_model
    for p in model.parameters():
        p.requires_grad = True
    n_params = sum(p.numel() for p in model.parameters())
    _log(rank, "MODEL", f"full fine-tune: {n_params:,} trainable encoder params "
                        f"(gradient checkpointing ON)", force=True)

    head = FullSftHead(hidden_dim)
    return FridaFullSftClassifier(model, head).to(device), hidden_dim


def make_param_groups(model):
    """Two groups so the head and backbone can run opposite-direction warmup
    (see docstring's THE HEAD section)."""
    return [{"params": list(model.encoder.parameters()), "lr": LR, "name": "backbone"},
            {"params": list(model.classifier.parameters()),
             "lr": LR * HEAD_LR_PEAK_MULTIPLIER, "name": "head"}]


def _backbone_lr_lambda(warmup_steps, total_steps, min_ratio):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = min((step - warmup_steps) / max(1, total_steps - warmup_steps), 1.0)
        return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr_lambda


def _head_lr_lambda(warmup_steps, total_steps, min_ratio):
    """Starts at 1.0 (i.e. the head's own peak, LR*HEAD_LR_PEAK_MULTIPLIER)
    and cools LINEARLY down to 1/HEAD_LR_PEAK_MULTIPLIER (i.e. matching the
    backbone's peak LR in absolute terms) over the warmup window -- the
    mirror image of the backbone's 0->1 ramp. After warmup both groups share
    the identical cosine trajectory, so the "opposite direction" framing only
    describes the warmup phase, not the whole run."""
    floor = 1.0 / HEAD_LR_PEAK_MULTIPLIER
    def lr_lambda(step):
        if step < warmup_steps:
            t = step / max(1, warmup_steps)
            return 1.0 + (floor - 1.0) * t
        progress = min((step - warmup_steps) / max(1, total_steps - warmup_steps), 1.0)
        return floor * (min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress)))
    return lr_lambda


# =====================================================================
# LOSS -- category-weighted focal, +2x on human rows
# =====================================================================

def focal_bce_per_sample(logits, targets, gamma):
    """Binary focal loss, per-sample (no reduction) -- the caller applies
    category/human weighting and does its own weighted mean. targets may be
    continuous (LLM soft labels): pt is defined the same way regardless
    (Lin et al.'s pt = p if y=1 else 1-p generalises directly to soft y as
    the probability mass agreeing with the label)."""
    p = torch.sigmoid(logits)
    ce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    pt = targets * p + (1 - targets) * (1 - p)
    return ((1 - pt).clamp(min=1e-6) ** gamma) * ce


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
        per_cat[cat_name] = average_precision_score(y_true, probs[mask])
    macro = float(np.mean(list(per_cat.values()))) if per_cat else float("nan")
    return macro, per_cat


def evaluate(model, full_loader, remainder_loader, device, category_names):
    model.eval()
    all_logits, all_labels, all_categories = [], [], []
    with torch.no_grad():
        for loader in (full_loader, remainder_loader):
            if loader is None:
                continue
            for input_ids, attention_mask, batch_labels, batch_categories, _scope in loader:
                logits = model(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device))
                all_logits.append(logits.cpu())
                all_labels.append(batch_labels)
                all_categories.append(batch_categories)
    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    categories = np.concatenate(all_categories, axis=0)
    macro, per_cat = compute_metrics(torch.sigmoid(logits).numpy(), labels.numpy(),
                                     categories, category_names)
    model.train()
    return {"macro_pr_auc": macro, "per_category": per_cat, "n": len(labels)}


# =====================================================================
# CHECKPOINT -- own directory per save; val score explicit null on the
# 5-hourly weights-only saves (see docstring's CHECKPOINT CONTRACT)
# =====================================================================

def save_checkpoint(model, optimizer, global_step, epoch, val_macro_pr_auc, label, rank):
    if rank != 0:
        return
    save_dir = os.path.join(OUTPUT_DIR, label)
    xm.mark_step()
    os.makedirs(save_dir, exist_ok=True)

    torch.save({k: v.detach().cpu() for k, v in model.encoder.state_dict().items()},
              os.path.join(save_dir, "encoder.pt"))
    torch.save({k: v.detach().cpu() for k, v in model.classifier.state_dict().items()},
              os.path.join(save_dir, "head.pt"))
    xm.save(optimizer.state_dict(), os.path.join(save_dir, "optimizer.pt"))

    with open(os.path.join(save_dir, "train_state.json"), "w") as f:
        json.dump({
            "epoch": epoch, "global_step": global_step,
            "val_macro_pr_auc": val_macro_pr_auc,   # explicit null if unvalidated
            "head": {"type": "FullSftHead", "layer_sizes": list(HEAD_HIDDEN),
                     "dropout": HEAD_DROPOUT},
            "pair_max_length": PAIR_MAX_LENGTH,
            "split": "item-disjoint 70/30 seed42",
            "llm_is_subsample": True, "llm_total_rows": LLM_TOTAL_ROWS,
        }, f, indent=2)
    print(f"[SAVE] {label}: epoch={epoch} step={global_step} "
          f"val_macro_pr_auc={val_macro_pr_auc}", flush=True)


# =====================================================================
# SHARED CPU-ONLY SETUP -- ONCE, pre-fork
# =====================================================================

_SHARED = {}


def _build_frame(id1, id2, label, category, scope, caches):
    cache = caches[scope]
    row1 = lookup_rows(cache, np.asarray(id1, dtype=np.int64))
    row2 = lookup_rows(cache, np.asarray(id2, dtype=np.int64))
    missing = int((row1 < 0).sum() + (row2 < 0).sum())
    if missing:
        raise RuntimeError(f"[SETUP] {missing:,} endpoints missing from the "
                           f"{cache['name']} item cache -- wrong cache scope?")
    return dict(scope=np.full(len(row1), scope, dtype=np.int8), row1=row1, row2=row2,
                label=np.asarray(label, dtype=np.float32),
                category=np.asarray(category, dtype=np.int64))


def _concat_frames(a, b):
    return {k: np.concatenate([a[k], b[k]]) for k in a}


def stratified_llm_sample(target, cat_id, n_per_category, seed):
    """Verbatim policy from frida-percat-lora -- see that kernel's own
    docstring for the full derivation (10 discrete target levels, confidence
    filtering considered and rejected). Duplicated rather than imported:
    kernels are standalone scripts."""
    rng = np.random.RandomState(seed)
    keep = []
    realized = {}
    for c in np.unique(cat_id):
        cat_rows = np.flatnonzero(cat_id == c)
        n_take = min(n_per_category, len(cat_rows))
        cat_targets = target[cat_rows]
        levels, level_counts = np.unique(cat_targets, return_counts=True)
        quota = np.floor(level_counts / level_counts.sum() * n_take).astype(np.int64)
        deficit = n_take - int(quota.sum())
        if deficit > 0:
            quota[np.argsort(-level_counts)[:deficit]] += 1
        picked = []
        for level, q in zip(levels, quota):
            if q <= 0:
                continue
            pool = cat_rows[cat_targets == level]
            q = min(q, len(pool))
            picked.append(rng.choice(pool, size=q, replace=False))
        sel = np.concatenate(picked) if picked else np.array([], dtype=np.int64)
        keep.append(sel)
        realized[int(c)] = {"taken": int(len(sel)), "available": int(len(cat_rows))}
    return np.concatenate(keep), realized


def _load_shared_data():
    print(f"[{_elapsed():8.1f}s][SETUP] loading all CPU-only data once, pre-fork ...", flush=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    human_cache = load_v1_cache(HUMAN_CACHE_HINT, "human")
    llm_cache = load_v1_cache(LLM_CACHE_HINT, "llm")
    if human_cache["pad_id"] != llm_cache["pad_id"]:
        raise RuntimeError(f"[SETUP] pad_id differs between caches: "
                           f"{human_cache['pad_id']} vs {llm_cache['pad_id']}")
    caches = [human_cache, llm_cache]
    pad_id = human_cache["pad_id"]

    # ---- item-disjoint 70/30 split (from-scratch training only -- see the
    # split script's own scope note) ----
    split_path = _find_dataset_file(ITEM_SPLIT_PATH_HINT, "matches_item_disjoint_70_30_seed42.parquet")
    split_df = pl.read_parquet(split_path).to_pandas()
    category_vocab = sorted(split_df["category"].unique().tolist())
    category_to_id = {c: i for i, c in enumerate(category_vocab)}
    split_df["cat_id"] = split_df["category"].map(category_to_id).astype(np.int64)

    human_train = split_df[split_df["split"] == "train"].reset_index(drop=True)
    human_val = split_df[split_df["split"] == "val"].reset_index(drop=True)
    val_fingerprint = hashlib.sha256(
        ",".join(f"{a}-{b}" for a, b in sorted(zip(human_val["id1"], human_val["id2"]))).encode("utf-8")
    ).hexdigest()[:16]
    print(f"[{_elapsed():8.1f}s][SETUP] human(item-disjoint) train={len(human_train):,} "
          f"val={len(human_val):,} val_fingerprint={val_fingerprint} categories={len(category_vocab)}",
          flush=True)

    human_train_frame = _build_frame(human_train["id1"], human_train["id2"], human_train["label"],
                                     human_train["cat_id"], 0, caches)
    val_frame = _build_frame(human_val["id1"], human_val["id2"], human_val["label"],
                             human_val["cat_id"], 0, caches)
    del split_df, human_val
    gc.collect()

    # ---- LLM subsample: category-balanced quota, stratified by target level
    # -- NOT the full 11.19M-row set. See docstring's "LLM DATA" section. ----
    llm = (
        pl.scan_parquet(_find_dataset_file(MATCHES_LLM_PATH_HINT, "matches_llm.parquet"))
        .filter(pl.col("id1").is_not_null() & pl.col("id2").is_not_null())
        .select(["id1", "id2", "target"])
        .collect(engine="streaming")
    )
    n_llm_total = llm.height
    llm_item_cat = (
        pl.scan_parquet(_find_dataset_file(ITEMS_FULL_PATH_HINT, "items.parquet"))
        .select(["id", "category"])
        .with_columns(pl.col("category").fill_null("UNKNOWN").cast(pl.Utf8)
                      .str.slice(0, MAX_CATEGORY_CHARS))
        .with_columns(pl.col("category").replace_strict(category_to_id, default=-1)
                      .cast(pl.Int32).alias("cat_id"))
        .select(["id", "cat_id"])
        .collect(engine="streaming")
    )
    llm = llm.join(llm_item_cat.rename({"id": "id1"}), on="id1", how="inner")
    del llm_item_cat
    gc.collect()
    llm = llm.filter(pl.col("cat_id") >= 0)
    print(f"[{_elapsed():8.1f}s][SETUP] llm rows with a known category: "
          f"{llm.height:,} / {n_llm_total:,}", flush=True)

    llm_pd = llm.to_pandas()
    del llm
    gc.collect()
    n_per_category = max(1, LLM_TOTAL_ROWS // len(category_vocab))
    sel, sample_report = stratified_llm_sample(
        llm_pd["target"].to_numpy(), llm_pd["cat_id"].to_numpy(), n_per_category, SEED)
    llm_pd = llm_pd.iloc[sel].reset_index(drop=True)
    print(f"[{_elapsed():8.1f}s][SETUP] LLM SUBSAMPLE: {len(llm_pd):,} rows "
          f"({n_per_category:,}/category, stratified by target level, seed={SEED}) "
          f"-- NOT the full {n_llm_total:,}-row set", flush=True)

    llm_frame = _build_frame(llm_pd["id1"], llm_pd["id2"], llm_pd["target"],
                             llm_pd["cat_id"].to_numpy().astype(np.int64), 1, caches)
    del llm_pd
    gc.collect()

    train_frame = _concat_frames(human_train_frame, llm_frame)
    print(f"[{_elapsed():8.1f}s][SETUP] mixed train={len(train_frame['label']):,} "
          f"(human {len(human_train_frame['label']):,} + llm_subsample {len(llm_frame['label']):,})",
          flush=True)
    del human_train_frame, llm_frame
    gc.collect()

    # ---- realized per-category counts on the ACTUAL training mix, for the
    # category weight in the loss (see docstring's LOSS section on why this
    # matters here and did not for frida-percat-lora) ----
    cat_counts = np.bincount(train_frame["category"], minlength=len(category_vocab)).astype(np.float64)
    cat_counts = np.maximum(cat_counts, 1.0)
    category_weight = (cat_counts.sum() / len(category_vocab)) / cat_counts
    print(f"[{_elapsed():8.1f}s][SETUP] category weights (train-mix inverse-frequency): "
          f"min={category_weight.min():.3f} max={category_weight.max():.3f} "
          f"({category_vocab[int(np.argmin(cat_counts))]!r} largest, "
          f"{category_vocab[int(np.argmax(cat_counts))]!r} smallest)", flush=True)

    with open(os.path.join(OUTPUT_DIR, "llm_subsample_report.json"), "w", encoding="utf-8") as f:
        json.dump({
            "policy": "stratified by discrete target level, category-balanced, "
                       "NO confidence filter (see stratified_llm_sample docstring)",
            "rows_per_category": n_per_category, "seed": SEED,
            "llm_rows_total_available": int(n_llm_total), "llm_rows_used": int(len(sel)),
            "per_category": {category_vocab[k]: v for k, v in sample_report.items()
                             if k < len(category_vocab)},
        }, f, ensure_ascii=False, indent=2)

    _SHARED.update(dict(
        caches=caches, pad_id=pad_id, train_frame=train_frame, val_frame=val_frame,
        category_vocab=category_vocab, category_weight=category_weight,
        val_fingerprint=val_fingerprint, n_llm_total=int(n_llm_total), n_llm_used=int(len(sel)),
    ))
    print(f"[{_elapsed():8.1f}s][SETUP] done -- ready for fork", flush=True)


# =====================================================================
# PER-REPLICA ENTRYPOINT
# =====================================================================

def _mp_fn(index):
    rank = xr.global_ordinal()
    world_size = xr.world_size()
    device = torch_xla.device()
    set_seed(SEED + rank)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    caches = _SHARED["caches"]
    pad_id = _SHARED["pad_id"]
    train_frame, val_frame = _SHARED["train_frame"], _SHARED["val_frame"]
    category_vocab = _SHARED["category_vocab"]
    category_weight = torch.tensor(_SHARED["category_weight"], dtype=torch.float32, device=device)

    _log(rank, "CONFIG", f"world_size={world_size} pair_max_length={PAIR_MAX_LENGTH} "
                         f"lr={LR} head_lr_peak={LR * HEAD_LR_PEAK_MULTIPLIER} "
                         f"warmup={WARMUP_RATIO} human_weight={HUMAN_WEIGHT} "
                         f"focal_gamma={FOCAL_GAMMA} epochs={EPOCHS} "
                         f"llm_subsample_total={_SHARED['n_llm_used']:,} "
                         f"(of {_SHARED['n_llm_total']:,})", force=True)

    model, hidden_dim = build_fresh_model(device, rank)
    total_batch = PER_CORE_TRAIN_BATCH * world_size
    _log(rank, "BATCH", f"per_core_train={PER_CORE_TRAIN_BATCH} per_core_eval={PER_CORE_EVAL_BATCH} "
                        f"total_train_batch={total_batch} (arithmetic estimate, not measured -- "
                        f"see module docstring)", force=True)

    optimizer = Adafactor(make_param_groups(model), scale_parameter=False,
                          relative_step=False, warmup_init=False, weight_decay=WEIGHT_DECAY)

    n_train = len(train_frame["label"])
    steps_per_epoch = max(1, n_train // total_batch)
    epoch_steps = steps_per_epoch * EPOCHS
    projected_steps = int(MAX_TRAIN_SECONDS * PROJECTED_ROWS_PER_SEC / total_batch)
    total_steps = max(1, min(epoch_steps, projected_steps))
    warmup_steps = max(1, int(total_steps * WARMUP_RATIO))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, [
        _backbone_lr_lambda(warmup_steps, total_steps, LR_MIN_RATIO),
        _head_lr_lambda(warmup_steps, total_steps, LR_MIN_RATIO),
    ])
    _log(rank, "TRAIN", f"n_train={n_train:,} steps/epoch={steps_per_epoch} "
                        f"epoch_steps={epoch_steps} projected_reachable={projected_steps} "
                        f"-> schedule_steps={total_steps} "
                        f"({100 * total_steps / max(epoch_steps, 1):.1f}% of {EPOCHS} epochs) "
                        f"warmup={warmup_steps}", force=True)

    val_full_loader, val_remainder_loader = make_eval_loader(val_frame, caches, pad_id,
                                                              PER_CORE_EVAL_BATCH)

    global_step = 0
    best_val = -1.0
    train_start = time.perf_counter()
    last_checkpoint_at = 0.0
    stopped = False

    def do_quiet_save(reason, epoch_num):
        try:
            save_checkpoint(model, optimizer, global_step, epoch_num, None, "last", rank)
            _log(rank, "SAVE", f"quiet save ({reason}) epoch={epoch_num} step={global_step} "
                              f"(weights+optimizer only, no eval -- see docstring's CHECKPOINT "
                              f"CONTRACT)", force=True)
        except Exception as e:
            _log(rank, "ERROR", f"quiet save FAILED ({reason}): {type(e).__name__}: {e}", force=True)

    def do_full_checkpoint(epoch_num, label):
        nonlocal best_val
        try:
            m = evaluate(model, val_full_loader, val_remainder_loader, device, category_vocab)
            improved = m["macro_pr_auc"] > best_val
            if improved:
                best_val = m["macro_pr_auc"]
            _log(rank, "METRICS", f"{label}: epoch={epoch_num} step={global_step} "
                                  f"val_macro_pr_auc={m['macro_pr_auc']:.4f} best={best_val:.4f} "
                                  f"n_val={m['n']}", force=True)
            if rank == 0:
                print(f"[METRICS] per_category: {json.dumps(m['per_category'], ensure_ascii=False)}",
                      flush=True)
            save_checkpoint(model, optimizer, global_step, epoch_num, m["macro_pr_auc"], label, rank)
            if improved:
                save_checkpoint(model, optimizer, global_step, epoch_num, m["macro_pr_auc"], "best", rank)
            return True
        except Exception as e:
            _log(rank, "ERROR", f"eval/checkpoint FAILED at epoch={epoch_num} step={global_step}: "
                                f"{type(e).__name__}: {e}", force=True)
            if rank == 0:
                import traceback
                traceback.print_exc()
            model.train()
            optimizer.zero_grad(set_to_none=True)
            gc.collect()
            do_quiet_save("eval failed, falling back", epoch_num)
            return False

    for epoch in range(1, EPOCHS + 1):
        if stopped:
            break
        train_loader, sampler = make_train_loader(train_frame, caches, pad_id,
                                                  PER_CORE_TRAIN_BATCH, world_size, rank, SEED,
                                                  epoch=epoch)
        mp_loader = pl_xla.MpDeviceLoader(train_loader, device)
        _log(rank, "EPOCH", f"starting epoch {epoch}/{EPOCHS}", force=True)

        for step, batch in enumerate(mp_loader):
            input_ids, attention_mask, batch_labels, batch_cats, batch_scope = batch
            batch_labels = batch_labels.to(device)
            batch_cats_t = torch.as_tensor(batch_cats, device=device)
            batch_scope = batch_scope.to(device)
            try:
                optimizer.zero_grad()
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
                per_sample = focal_bce_per_sample(logits, batch_labels, FOCAL_GAMMA)
                w_cat = category_weight[batch_cats_t]
                w_human = torch.where(batch_scope == 0,
                                      torch.full_like(per_sample, HUMAN_WEIGHT),
                                      torch.ones_like(per_sample))
                w = w_cat * w_human
                loss = (per_sample * w).sum() / w.sum()
                loss.backward()
                xm.optimizer_step(optimizer)
                scheduler.step()
                global_step += 1
            except Exception as e:
                _log(rank, "WARN", f"step failed and was skipped: {type(e).__name__}: {e}", force=True)
                try:
                    del logits, per_sample, loss
                except NameError:
                    pass
                optimizer.zero_grad(set_to_none=True)
                gc.collect()
                continue

            if global_step % LOG_EVERY_STEPS == 0:
                xm.mark_step()
                if rank == 0:
                    et = time.perf_counter() - train_start
                    print(f"[{_elapsed():8.1f}s] epoch={epoch} step={global_step}/{total_steps} "
                          f"loss={loss.item():.4f} lr_backbone={scheduler.get_last_lr()[0]:.2e} "
                          f"lr_head={scheduler.get_last_lr()[1]:.2e} elapsed_train={et:.0f}s", flush=True)

            elapsed_train = time.perf_counter() - train_start
            if (elapsed_train - last_checkpoint_at) >= CHECKPOINT_EVERY_SECONDS:
                do_quiet_save(f"5h wall-clock cadence", epoch)
                last_checkpoint_at = elapsed_train
                model.train()

            if elapsed_train > MAX_TRAIN_SECONDS:
                _log(rank, "TRAIN", f"wall-clock cutoff ({elapsed_train:.0f}s > {MAX_TRAIN_SECONDS}s) "
                                    f"-- stopping mid-epoch.", force=True)
                do_quiet_save("wall-clock cutoff", epoch)
                stopped = True
                break

        del mp_loader, train_loader, sampler
        gc.collect()
        if stopped:
            break

        do_full_checkpoint(epoch, f"checkpoint_epoch{epoch:02d}")
        last_checkpoint_at = time.perf_counter() - train_start
        model.train()

    _log(rank, "DONE", f"total wall-clock {_elapsed():.0f}s best_val_macro_pr_auc={best_val:.4f}",
         force=True)


def main():
    os.environ.pop("TPU_PROCESS_ADDRESSES", None)
    os.environ.pop("CLOUD_TPU_TASK_ID", None)

    _load_shared_data()

    torch_xla.launch(_mp_fn, args=(), start_method="fork")


if __name__ == "__main__":
    main()
