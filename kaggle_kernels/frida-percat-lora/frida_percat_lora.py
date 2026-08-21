"""
FRIDA per-category specialists: merge the general adapter into the base
weights, then train 20 INDEPENDENT (LoRA + linear classifier) pairs, one per
category.

WHY per-category at all
===========================================================================
The competition metric is a plain mean of per-category PR-AUC over 20
categories -- it is literally 20 independent problems averaged. A single
model has to spend capacity reconciling categories whose positive rates
differ by 7.7x (measured over the real data: 0.073 for the sparsest to 0.562
for the densest). Specialists do not.

The general model is NOT thrown away: the already-validated adapter (val
0.7660) is MERGED into the base weights first, so every specialist starts
from the strong general model and only has to learn its own category's
deviation from it.

THE MERGED WEIGHTS ARE SAVED BEFORE TRAINING STARTS
===========================================================================
merge_and_save_base() runs pre-fork, on CPU, and writes the merged encoder
to OUTPUT_DIR immediately -- before a single training step. Two reasons:
(1) they are a deliverable in their own right (inference needs merged base +
one adapter per category), and (2) if training later dies, the merge
survives in /kaggle/working, which Kaggle persists even for a cancelled or
hard-capped session. Same "persist the expensive thing early" logic the rest
of this project's checkpointing follows. Cost: ~1.65GB of the 20GB output
budget, plus ~20 x 25MB of per-category adapters = well inside it.

SEQUENTIAL CATEGORIES, NOT ONE CATEGORY PER CORE
===========================================================================
The tempting layout is 8 categories on 8 TPU cores at once, in waves of
(8, 8, 4). It is the wrong shape here, for a reason that would have been
silent rather than loud:

    xm.optimizer_step() ALL-REDUCES gradients across cores.

With a different LoRA per core that averages 8 different categories'
gradients into each other on every step, quietly destroying the very
specialization the design exists to create. Avoiding it means dropping to a
bare optimizer.step(), which is exactly the call this project's
kaggle-experiment-safety skill lists as a silent run-killer on TPU.

So: categories run ONE AT A TIME, each using all 8 cores in ordinary data
parallelism. Same total compute (the per-category row counts sum to the full
dataset), standard proven code path, and measurably LESS wall-clock: the
wave scheme costs max-of-wave x 3 = ~59.1K pair-units against 45.7K for
sequential-with-DP, because seven cores idle waiting on each wave's largest
category.

One LoRA module is created ONCE and its weights/optimizer are RESET between
categories, rather than building a fresh PeftModel 20 times -- XLA compiles
per shape, and the shape never changes, so this keeps the compile count at
one instead of twenty.

TIME BUDGET (arithmetic, from this project's own measured 120 rows/s at
PAIR_MAX_LENGTH=256 -- see frida-llm-tpu-sft's docstring)
===========================================================================
PAIR_MAX_LENGTH=192 scales the attention term by (192/256)^2 and the rest
linearly, giving ~170 rows/s. Per category: all its human train rows
(~14.6K) + LLM_ROWS_PER_CATEGORY (100K) = ~114.6K rows; x20 categories =
~2.36M rows = ~3.9h. Comfortably inside one 9h TPU session with the 8.5h
stop, with quota left for a second pass.

SPLIT: THE OLD CANONICAL ONE, DELIBERATELY
===========================================================================
This kernel uses the 80/20 pair-level seed-42 split, NOT the newer
item-disjoint 70/30 one (data/splits/matches_item_disjoint_70_30_seed42.*).
That is not an oversight. The adapter being merged here was trained and
validated against the old split; re-splitting under it would move items it
already trained on into the new val half and silently inflate every
per-category number. A checkpoint keeps the split it was born with.

WHAT IS WEIGHTED, AND WHY IT IS NOT CATEGORY SIZE
===========================================================================
Category SIZES are nearly uniform in this dataset -- 17,286 to 23,357 pairs,
a 1.4x spread -- so weighting by inverse category size would be close to a
no-op. Positive RATE is what varies (7.7x). Each specialist therefore gets
its OWN pos_weight, computed from its own training mix, plus a 2x weight on
human-labeled rows relative to LLM soft-labeled ones.

HEAD SCHEDULE
===========================================================================
The classifier is randomly initialised while the backbone is already strong,
so the two need different schedules. Two param groups: the head starts at
HEAD_LR (5x the adapter's) and holds it flat through the first 10% of steps
while the adapter linearly warms 0 -> LR, after which both cosine-decay
together to a floor. Without this the random head would backpropagate noise
into a good backbone for the first few hundred steps.

INHERITED LESSONS (kept verbatim from frida-llm-tpu-sft, do not re-derive)
===========================================================================
  1. autocast INSIDE the model's own forward(), never around the outer call.
  2. PeftModel.from_pretrained(..., is_trainable=True).
  3. Fixed batch sizes, NEVER a TPU probe -- each probed candidate compiles
     its own XLA program and those stay resident in HBM.
  4. xm.optimizer_step(optimizer), not optimizer.step().
  5. drop_last=True on every loader so XLA only ever sees two shapes.
  6. del logits, loss before continue on a skipped step.
  7. ALL CPU-only setup ONCE in main() before torch_xla.launch().
  8. Results mirrored to JSON in /kaggle/working -- a cancelled session can
     come back with a 0-byte log.
"""
import os
import sys
import gc
import json
import math
import time
import hashlib
import subprocess

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "polars", "peft"])

import numpy as np
import pandas as pd
import polars as pl

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from transformers import T5EncoderModel, set_seed
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score

try:
    import peft.import_utils
    import peft.tuners.lora.torchao
    peft.import_utils.is_torchao_available = lambda: False
    peft.tuners.lora.torchao.is_torchao_available = lambda: False
except Exception as e:
    print(f"[WARN] torchao patch failed (likely fine): {e}")

from peft import (LoraConfig, PeftModel, TaskType, get_peft_model,
                   get_peft_model_state_dict)
from safetensors.torch import save_file as safetensors_save_file

import torch_xla
import torch_xla.core.xla_model as xm
from torch_xla import runtime as xr
import torch_xla.distributed.parallel_loader as pl_xla


# =====================================================================
# CONFIG
# =====================================================================

SEED = 42
MODEL_NAME = "ai-forever/FRIDA"

MATCHES_PATH_HINT = "/kaggle/input/datasets/maxamsecret/ozon-ecup-2026-matching/matches.parquet"
MATCHES_LLM_PATH_HINT = "/kaggle/input/datasets/maxamsecret/ozon-ecup-2026-matching/matches_llm.parquet"
ITEMS_HUMAN_PATH_HINT = "/kaggle/input/datasets/maxamsecret/ozon-ecup-2026-matching/items_human.parquet"

# v2 caches (frida-tokenize-items-cpu): correct char budget (140/32/529,
# see reports/experiment_log.md "Char-budget rule v2") + the light v2
# attribute cleanup. NOT the v1 caches, which carry the under-scaled
# 128/64/384 budget sized for MiniLM's old 128-token regime.
#
# Published as ONE dataset covering both scopes (frida-tokenize-items-cpu's
# own two publish_dataset() calls failed at the very last step -- the
# tokenization itself completed correctly, confirmed via cache_meta.json:
# 711,304 human items, v2 policy applied to 711,232 of them -- so the user
# published /kaggle/working by hand as a single dataset instead). These
# hints therefore won't resolve to an exact path; that's fine,
# _find_v1_cache_root() falls back to a content-matching walk of
# /kaggle/input keyed on n_items, which is what actually finds them.
HUMAN_CACHE_HINT = "/kaggle/input/datasets/maxamsecret/frida-tokenize-items-v3/human_scope"
LLM_CACHE_HINT = "/kaggle/input/datasets/maxamsecret/frida-tokenize-items-v3/llm_scope"

ITEMS_FULL_PATH_HINT = "/kaggle/input/datasets/maxamsecret/ozon-ecup-2026-matching/items.parquet"

ITEM_MAX_LENGTH = 256
# 192, not 256: this is the length the real judging stand can actually
# SERVE inside its time budget (176 passed at 0.4087, 192 measured at
# 1.65x over Public's budget and so is the current ceiling to aim at),
# and training at the serving length removes the train/serve mismatch
# that every FRIDA submission so far has paid for.
PAIR_MAX_LENGTH = 192

MAX_CATEGORY_CHARS = 64

CATEGORY_LABEL_TOKEN = 55917
ATTRIBUTES_LABEL_TOKEN = 91837
LABEL_COLON_TOKEN = 30

# A single nn.Linear per category, per the design: 20 small specialists,
# not 20 copies of the multi-layer head. The merged backbone already
# carries the general model's capacity; the head only has to place one
# category's decision boundary.
HEAD_DROPOUT = 0.1

# Deliberately SMALLER and more regularized than the project default
# (r=16/alpha=32/dropout=0.10): each specialist sees ~115K rows, not
# 11M, so a larger adapter would mostly memorize.
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.15
LORA_TARGET_MODULES = ["q", "k", "v", "o", "wi_0", "wi_1", "wo"]

# BERTA recipe's own numbers, and the MiniLM continue-SFT fix's numbers.
LR = 2e-4                  # adapter peak
HEAD_LR_MULTIPLIER = 5.0   # head peak = LR * this, held flat through warmup
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.10
HUMAN_WEIGHT = 2.0         # human rows count double against LLM soft labels

# Scaled from a PROVEN point, not from a probe (probing is forbidden on
# TPU -- see docstring lesson 3). frida-tpu-human-sft runs 16/32 at
# PAIR_MAX_LENGTH=256 without OOM. Per-sample activation memory at 192 is
# 0.62x of that at 256 (attention scales with seq^2, the rest linearly),
# which would permit ~26; 24 keeps the same margin the proven point had.
PER_CORE_TRAIN_BATCH = 24
PER_CORE_EVAL_BATCH = 48

# Per category, not in total: 20 x (its human train rows + this) rows.
# 100K x 20 = 2M LLM rows + ~292K human = ~2.36M, ~3.9h at ~170 rows/s.
LLM_ROWS_PER_CATEGORY = 100_000
EPOCHS = 1                     # per category
MAX_TRAIN_SECONDS = 30600      # 8.5h. Kaggle's TPU session cap is 9h (the 12h
                                # cap is GPU) -- raise this only if a real TPU
                                # run here is observed surviving past 9h.
CHECKPOINT_EVERY_SECONDS = 3600
LOG_EVERY_STEPS = 50

# Used ONLY to size the LR schedule (see docstring). Measured on
# frida-tpu-human-sft at this exact model/sequence/batch. If a future run
# reports a different rate, update this rather than the schedule shape.
PROJECTED_ROWS_PER_SEC = 170   # 120 measured at seq=256, scaled to seq=192
LR_MIN_RATIO = 0.05            # LR floor as a fraction of peak

WORKING_DIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
OUTPUT_DIR = os.path.join(WORKING_DIR, "frida_percat_lora")
MERGED_BASE_DIR = os.path.join(OUTPUT_DIR, "merged_base")
RESULTS_JSON = os.path.join(OUTPUT_DIR, "percat_results.json")

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


def _find_resume_adapter_dir():
    if os.path.isdir("/kaggle/input"):
        for root, _dirs, files in os.walk("/kaggle/input"):
            if "adapter_config.json" in files:
                return root
    return None


def _pattern_starts_chunked(input_ids, n_items, tok_a, tok_b, chunk=500_000):
    """Chunked so the boolean mask never materialises at full 12.38M x 255
    scale at once -- that alone would be several GB of transient RAM on top
    of the 12.7GB memmap, on the host that also has to survive an 8-way
    fork."""
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


# Two v1 caches are mounted at once and both look identical to a filename
# scan, so the fallback search must disambiguate by CONTENT (n_items), not by
# whichever directory os.walk happens to reach first. Silently training the
# human split against the LLM cache would produce garbage that no assertion
# downstream would catch.
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
    raise RuntimeError(f"[CACHE] no v1 item cache with n_items={expect_n_items:,} found "
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

    # searchsorted index instead of a Python dict: a 12.38M-entry dict costs
    # well over a GB of host RAM that the 8-way fork then has to keep alive.
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
    """Vectorised id -> cache row. Returns -1 where an id is absent, so the
    caller can fail loudly at setup instead of silently embedding whatever
    sits at row 0."""
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
        # scope rides along (0=human, 1=llm) so the training loop can apply
        # HUMAN_WEIGHT without a second lookup.
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
# MODEL
# =====================================================================

class LinearHead(nn.Module):
    """One Linear per category. The merged backbone already carries the
    general model's capacity, so a specialist head only has to place its own
    category's decision boundary -- and with ~115K rows per category, a
    deeper head would mostly memorize."""

    def __init__(self, hidden_dim, dropout=HEAD_DROPOUT):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        return self.out(self.dropout(x)).view(-1)


class FridaCrossEncoderClassifier(nn.Module):
    def __init__(self, encoder, head):
        super().__init__()
        self.encoder = encoder
        self.classifier = head

    def forward(self, input_ids, attention_mask):
        with torch.autocast("xla", dtype=torch.bfloat16):
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :].float()
        return self.classifier(cls).view(-1).float()


def merge_and_save_base(resume_dir):
    """Merges the validated general adapter into the base weights and writes
    the result to MERGED_BASE_DIR *before any training happens*.

    Runs pre-fork on CPU: 8 replicas each re-merging the same thing would be
    pure waste, and /kaggle/working is persisted by Kaggle even when a
    session is cancelled or hard-capped, so a crash three hours into
    training still leaves these weights recoverable.

    Idempotent -- a re-run that finds a finished merge reuses it."""
    marker = os.path.join(MERGED_BASE_DIR, "config.json")
    if os.path.isfile(marker):
        print(f"[{_elapsed():8.1f}s][MERGE] reusing existing merge at {MERGED_BASE_DIR}", flush=True)
        return MERGED_BASE_DIR

    print(f"[{_elapsed():8.1f}s][MERGE] merging {resume_dir} into base weights ...", flush=True)
    try:
        base = T5EncoderModel.from_pretrained(MODEL_NAME, dtype=torch.bfloat16,
                                               attn_implementation="eager")
    except (TypeError, ValueError):
        base = T5EncoderModel.from_pretrained(MODEL_NAME, dtype=torch.bfloat16)

    # Snapshot one real weight so the merge can be PROVEN to have changed the
    # model. A silent no-op merge (wrong dir, adapter that never loaded) would
    # otherwise train 20 specialists on top of the plain base model and look
    # completely normal in the logs -- exactly the class of silent failure
    # this project keeps paying for.
    probe_name = "encoder.block.0.layer.0.SelfAttention.q.weight"
    before = dict(base.named_parameters())[probe_name].detach().clone().float()

    peft_model = PeftModel.from_pretrained(base, resume_dir, is_trainable=False)
    merged = peft_model.merge_and_unload()

    after = dict(merged.named_parameters())[probe_name].detach().float()
    delta = (after - before).abs().max().item()
    if delta == 0.0:
        raise RuntimeError(
            f"[MERGE] merge was a NO-OP: {probe_name} is byte-identical before and "
            f"after merge_and_unload(). The adapter at {resume_dir} did not apply.")
    print(f"[{_elapsed():8.1f}s][MERGE] verified: {probe_name} max|delta|={delta:.6f}", flush=True)

    os.makedirs(MERGED_BASE_DIR, exist_ok=True)
    merged.save_pretrained(MERGED_BASE_DIR, safe_serialization=True)
    with open(os.path.join(MERGED_BASE_DIR, "MERGE_PROVENANCE.json"), "w") as f:
        json.dump({"base_model": MODEL_NAME, "merged_adapter_dir": resume_dir,
                   "probe_param": probe_name, "probe_max_abs_delta": delta,
                   "dtype": "bfloat16"}, f, indent=2)
    del peft_model, merged, base
    gc.collect()
    print(f"[{_elapsed():8.1f}s][MERGE] saved -> {MERGED_BASE_DIR}", flush=True)
    return MERGED_BASE_DIR


def build_specialist_model(device, rank, merged_dir):
    """One PeftModel built ONCE and reused across all 20 categories (weights
    reset between them by reset_specialist_). Rebuilding it per category would
    make XLA compile a fresh program each time; the shape never changes, so
    one compile is all this should ever cost."""
    try:
        base = T5EncoderModel.from_pretrained(merged_dir, dtype=torch.bfloat16,
                                               attn_implementation="eager")
    except (TypeError, ValueError):
        base = T5EncoderModel.from_pretrained(merged_dir, dtype=torch.bfloat16)
    hidden_dim = base.config.d_model

    lora_cfg = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
                          target_modules=LORA_TARGET_MODULES, bias="none",
                          task_type=TaskType.FEATURE_EXTRACTION)
    peft_model = get_peft_model(base, lora_cfg)
    for name, p in peft_model.named_parameters():
        if "lora_" in name:
            p.data = p.data.float()
            p.requires_grad = True
    n_trainable = sum(p.numel() for n, p in peft_model.named_parameters()
                      if "lora_" in n and p.requires_grad)
    assert n_trainable > 0, "fresh LoRA has zero trainable params"

    model = FridaCrossEncoderClassifier(peft_model, LinearHead(hidden_dim)).to(device)
    _log(rank, "MODEL", f"fresh LoRA r={LORA_R} alpha={LORA_ALPHA} "
                        f"dropout={LORA_DROPOUT} trainable={n_trainable:,}", force=True)
    return model, hidden_dim


def reset_specialist_(model, seed):
    """Returns the adapter + head to a fresh initialisation, in place.

    peft's own convention is lora_A ~ kaiming_uniform, lora_B = 0, so the
    adapter starts as an exact identity and the first steps see the merged
    model unchanged -- which is what makes starting from a strong backbone
    safe. Reproduced here rather than rebuilding the module (see
    build_specialist_model on why the module is reused)."""
    torch.manual_seed(seed)
    for name, p in model.encoder.named_parameters():
        if "lora_A" in name:
            nn.init.kaiming_uniform_(p, a=math.sqrt(5))
        elif "lora_B" in name:
            nn.init.zeros_(p)
    for m in model.classifier.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            nn.init.zeros_(m.bias)


def make_param_groups(model):
    """Two groups so the randomly-initialised head and the warm backbone can
    follow different schedules -- see the docstring's HEAD SCHEDULE section."""
    lora_params = [p for n, p in model.encoder.named_parameters()
                   if "lora_" in n and p.requires_grad]
    head_params = [p for p in model.classifier.parameters() if p.requires_grad]
    return [{"params": lora_params, "lr": LR, "name": "lora"},
            {"params": head_params, "lr": LR * HEAD_LR_MULTIPLIER, "name": "head"}]


def _make_head_lr_lambda(warmup_steps, total_steps, min_ratio):
    """Head holds its (higher) peak FLAT through warmup while the adapter is
    still ramping from zero, then joins the same cosine decay. The point is
    to let a random head become useful before it can push noise back into a
    good backbone."""
    def lr_lambda(step):
        if step < warmup_steps:
            return 1.0
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(progress, 1.0)
        return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr_lambda


def _make_lr_lambda(warmup_steps, total_steps, min_ratio):
    """Linear warmup then cosine decay to a FLOOR of min_ratio * peak, never
    to zero. Written out with LambdaLR (core PyTorch) rather than
    transformers' cosine_with_min_lr, whose availability depends on the
    transformers version the image happens to ship -- and TPU-image version
    drift has already killed one kernel in this project."""
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(progress, 1.0)
        return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr_lambda


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
                logits = model(input_ids=input_ids.to(device),
                               attention_mask=attention_mask.to(device))
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
# CHECKPOINT -- own directory per save, explicit null when unvalidated
# =====================================================================

def save_checkpoint(model, optimizer, global_step, epoch, val_macro_pr_auc, label, rank,
                     extra=None):
    if rank != 0:
        return
    save_dir = os.path.join(OUTPUT_DIR, label)
    xm.mark_step()
    os.makedirs(save_dir, exist_ok=True)

    adapter_dir = os.path.join(save_dir, "adapter")
    os.makedirs(adapter_dir, exist_ok=True)
    lora_sd = get_peft_model_state_dict(model.encoder)
    safetensors_save_file({k: v.detach().cpu() for k, v in lora_sd.items()},
                          os.path.join(adapter_dir, "adapter_model.safetensors"))
    model.encoder.peft_config["default"].save_pretrained(adapter_dir)

    torch.save({k: v.detach().cpu() for k, v in model.classifier.state_dict().items()},
               os.path.join(save_dir, "head.pt"))
    xm.save(optimizer.state_dict(), os.path.join(save_dir, "optimizer.pt"))

    with open(os.path.join(save_dir, "train_state.json"), "w") as f:
        json.dump({
            "epoch": epoch, "global_step": global_step,
            "val_pr_auc": val_macro_pr_auc,   # explicit null if unvalidated
            "category": extra.get("category") if extra else None,
            "head": {"type": "LinearHead", "dropout": HEAD_DROPOUT},
            "lora": {"r": LORA_R, "alpha": LORA_ALPHA, "dropout": LORA_DROPOUT,
                     "target_modules": LORA_TARGET_MODULES},
            "pair_max_length": PAIR_MAX_LENGTH,
            "merged_base": "merged_base/",
            **(extra or {}),
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


def _index_frame(frame, mask):
    return {k: v[mask] for k, v in frame.items()}


def stratified_llm_sample(target, cat_id, n_per_category, seed):
    """Picks up to n_per_category LLM rows for EVERY category, preserving that
    category's own target-value distribution exactly.

    Why this and not a confidence filter (the alternative that was
    considered and rejected): matches_llm targets are not continuous, they
    are 10 discrete levels k/9 -- a 9-judge vote count. Measured over all
    11,187,780 rows:

        0.000  64.22%   |  0.556   2.41%
        0.111   0.01%   |  0.667   1.52%
        0.222   5.65%   |  0.778   2.47%
        0.333   4.44%   |  0.889   6.08%
        0.444   2.26%   |  1.000  10.94%

    So "confident" rows (<=0.2 or >=0.8) are ALREADY 81.25% of the data by
    nature. Filtering on confidence would push that to 100% while discarding
    the 18.75% of rows sitting nearest the decision boundary -- and PR-AUC is
    a ranking metric, decided precisely there. It would also be the worst
    possible moment to do it: the backbone merged into this run is
    frida-tpu-human-sft, trained human-ONLY, so these specialists are the
    model's first exposure to LLM-labelled data at all.

    Category balance is enforced (every category gets the same quota) rather
    than sampled proportionally, because the metric is a plain mean over
    categories -- each one is equally worth getting right.
    """
    rng = np.random.RandomState(seed)
    keep = []
    realized = {}
    for c in np.unique(cat_id):
        cat_rows = np.flatnonzero(cat_id == c)
        n_take = min(n_per_category, len(cat_rows))
        cat_targets = target[cat_rows]
        levels, level_counts = np.unique(cat_targets, return_counts=True)
        # proportional allocation, then fix the rounding drift on the largest
        # level so the total lands exactly on n_take
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
        realized[int(c)] = {"taken": int(len(sel)), "available": int(len(cat_rows)),
                            "levels": {float(l): int(q) for l, q in zip(levels, quota) if q > 0}}
    return np.concatenate(keep), realized


def _load_shared_data():
    print(f"[{_elapsed():8.1f}s][SETUP] loading all CPU-only data once, pre-fork ...", flush=True)

    # ---- fail cheap: everything the run cannot proceed without, first ----
    resume_dir = _find_resume_adapter_dir()
    if resume_dir is None:
        raise RuntimeError("[SETUP] no adapter_config.json under /kaggle/input -- mount the "
                           "validated general FRIDA adapter via dataset_sources.")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---- the merge, saved BEFORE any training (see docstring) ----
    merged_dir = merge_and_save_base(resume_dir)

    human_cache = load_v1_cache(HUMAN_CACHE_HINT, "human")
    llm_cache = load_v1_cache(LLM_CACHE_HINT, "llm")
    if human_cache["pad_id"] != llm_cache["pad_id"]:
        raise RuntimeError(f"[SETUP] pad_id differs between caches: "
                           f"{human_cache['pad_id']} vs {llm_cache['pad_id']}")
    caches = [human_cache, llm_cache]
    pad_id = human_cache["pad_id"]

    # ---- human: the OLD canonical 80/20 split, deliberately (see docstring) ----
    matches_pd = (
        pl.read_parquet(_find_dataset_file(MATCHES_PATH_HINT, "matches.parquet"))
        .filter(pl.col("id1").is_not_null() & pl.col("id2").is_not_null())
        .with_columns([pl.col("target").fill_null(0).cast(pl.Float32),
                       (pl.col("target") >= 0.5).cast(pl.Int8).alias("label")])
        .select(["id1", "id2", "label"])
        .to_pandas()
    )
    labels_all = matches_pd["label"].to_numpy().astype(np.int64)
    indices = np.arange(len(labels_all), dtype=np.int64)
    train_idx, val_idx = train_test_split(indices, test_size=0.20, random_state=SEED,
                                          stratify=labels_all)
    human_train = matches_pd.iloc[train_idx].reset_index(drop=True)
    human_val = matches_pd.iloc[val_idx].reset_index(drop=True)
    val_fingerprint = hashlib.sha256(
        ",".join(f"{a}-{b}" for a, b in sorted(zip(human_val["id1"], human_val["id2"]))).encode("utf-8")
    ).hexdigest()[:16]
    print(f"[{_elapsed():8.1f}s][SETUP] human train={len(human_train):,} val={len(human_val):,} "
          f"val_fingerprint={val_fingerprint} (OLD canonical 80/20 split -- see docstring)",
          flush=True)

    items_cat = (
        pl.scan_parquet(_find_dataset_file(ITEMS_HUMAN_PATH_HINT, "items_human.parquet"))
        .select(["id", "category"])
        .with_columns(pl.col("category").fill_null("UNKNOWN").cast(pl.Utf8)
                      .str.slice(0, MAX_CATEGORY_CHARS))
        .collect(engine="streaming")
    )
    category_vocab = sorted(items_cat.get_column("category").unique().to_list())
    category_to_id = {c: i for i, c in enumerate(category_vocab)}
    id_to_category = dict(zip(items_cat.get_column("id").to_list(),
                              items_cat.get_column("category").to_list()))
    print(f"[{_elapsed():8.1f}s][SETUP] categories: {len(category_vocab)}", flush=True)

    def cat_ids(frame):
        # No UNKNOWN fallback on purpose. items_human.parquet is built to cover
        # every id matches.parquet references, and a local check over all
        # 365,654 pairs found both endpoints sharing a real category in 100.0%
        # of them. If that ever stops holding, an unmapped id silently becoming
        # "UNKNOWN" would quietly move rows into a 21st pseudo-category and
        # corrupt the per-category split -- so fail loudly instead.
        mapped = frame["id1"].map(id_to_category).map(category_to_id)
        n_missing = int(mapped.isna().sum())
        if n_missing:
            raise RuntimeError(f"[SETUP] {n_missing:,} pairs have an id1 with no category in "
                               f"items_human.parquet -- the per-category split cannot be built.")
        return mapped.to_numpy().astype(np.int64)

    human_train_frame = _build_frame(human_train["id1"], human_train["id2"],
                                     human_train["label"], cat_ids(human_train), 0, caches)
    human_val_frame = _build_frame(human_val["id1"], human_val["id2"],
                                   human_val["label"], cat_ids(human_val), 0, caches)
    del matches_pd, items_cat, id_to_category, labels_all, indices, train_idx, val_idx
    gc.collect()

    # ---- LLM: bucket by category, then a stratified per-category subsample ----
    # NOTE: this run does NOT use the full 11.19M-row LLM set. It uses
    # LLM_ROWS_PER_CATEGORY per category. Anything reporting this run's
    # results must say "LLM subsample", not "LLM".
    #
    # Categories for LLM items come from the FULL items.parquet, but ONLY the
    # id and category columns are ever decoded -- the attributes column is
    # what makes a naive read of that file OOM, and it is never touched here.
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
    sel, sample_report = stratified_llm_sample(
        llm_pd["target"].to_numpy(), llm_pd["cat_id"].to_numpy(),
        LLM_ROWS_PER_CATEGORY, SEED)
    llm_pd = llm_pd.iloc[sel].reset_index(drop=True)
    print(f"[{_elapsed():8.1f}s][SETUP] LLM SUBSAMPLE: {len(llm_pd):,} rows "
          f"({LLM_ROWS_PER_CATEGORY:,}/category, stratified by target level, seed={SEED}) "
          f"-- NOT the full {n_llm_total:,}-row set", flush=True)

    llm_frame = _build_frame(llm_pd["id1"], llm_pd["id2"], llm_pd["target"],
                             llm_pd["cat_id"].to_numpy().astype(np.int64), 1, caches)
    del llm_pd
    gc.collect()

    # ---- split everything into per-category frames ----
    train_by_cat, val_by_cat = {}, {}
    all_train = _concat_frames(human_train_frame, llm_frame)
    del human_train_frame, llm_frame
    gc.collect()
    for cat_id in range(len(category_vocab)):
        train_by_cat[cat_id] = _index_frame(all_train, all_train["category"] == cat_id)
        val_by_cat[cat_id] = _index_frame(human_val_frame, human_val_frame["category"] == cat_id)
    del all_train, human_val_frame
    gc.collect()

    for cat_id, name in enumerate(category_vocab):
        tr, va = train_by_cat[cat_id], val_by_cat[cat_id]
        n_human = int((tr["scope"] == 0).sum())
        print(f"[{_elapsed():8.1f}s][SETUP]   [{cat_id:02d}] {name[:34]:<34} "
              f"train={len(tr['label']):>7,} (human {n_human:>6,} + llm {len(tr['label']) - n_human:>7,})"
              f"  val={len(va['label']):>6,}", flush=True)

    with open(os.path.join(OUTPUT_DIR, "llm_subsample_report.json"), "w", encoding="utf-8") as f:
        json.dump({
            "policy": "stratified by discrete target level, category-balanced, "
                       "NO confidence filter (see stratified_llm_sample docstring)",
            "rows_per_category": LLM_ROWS_PER_CATEGORY,
            "seed": SEED,
            "llm_rows_total_available": int(n_llm_total),
            "llm_rows_used": int(len(sel)),
            "per_category": {category_vocab[k]: v for k, v in sample_report.items()
                             if k < len(category_vocab)},
        }, f, ensure_ascii=False, indent=2)

    _SHARED.update(dict(
        caches=caches, pad_id=pad_id, train_by_cat=train_by_cat, val_by_cat=val_by_cat,
        category_vocab=category_vocab, val_fingerprint=val_fingerprint,
        merged_dir=merged_dir, n_llm_total=int(n_llm_total), n_llm_used=int(len(sel)),
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
    train_by_cat, val_by_cat = _SHARED["train_by_cat"], _SHARED["val_by_cat"]
    category_vocab = _SHARED["category_vocab"]

    _log(rank, "CONFIG", f"world_size={world_size} pair_max_length={PAIR_MAX_LENGTH} "
                         f"lora_r={LORA_R} lr={LR} head_lr={LR * HEAD_LR_MULTIPLIER} "
                         f"warmup={WARMUP_RATIO} human_weight={HUMAN_WEIGHT} "
                         f"llm_subsample={LLM_ROWS_PER_CATEGORY:,}/category "
                         f"(of {_SHARED['n_llm_total']:,} total)", force=True)

    # Built ONCE; reset between categories so XLA compiles a single program.
    model, hidden_dim = build_specialist_model(device, rank, _SHARED["merged_dir"])
    total_batch = PER_CORE_TRAIN_BATCH * world_size
    _log(rank, "BATCH", f"per_core_train={PER_CORE_TRAIN_BATCH} "
                        f"per_core_eval={PER_CORE_EVAL_BATCH} "
                        f"total_train_batch={total_batch} (fixed, no probing)", force=True)

    run_start = time.perf_counter()
    results = {}
    stopped = False

    for cat_id, cat_name in enumerate(category_vocab):
        if stopped:
            break
        label_prefix = f"cat{cat_id:02d}"
        done_marker = os.path.join(OUTPUT_DIR, f"{label_prefix}_best", "train_state.json")
        if os.path.isfile(done_marker):
            _log(rank, "SKIP", f"[{cat_id:02d}] {cat_name}: already finished in an earlier "
                               f"session -- skipping", force=True)
            continue

        train_frame, val_frame = train_by_cat[cat_id], val_by_cat[cat_id]
        n_train = len(train_frame["label"])
        if n_train < total_batch:
            _log(rank, "SKIP", f"[{cat_id:02d}] {cat_name}: only {n_train} train rows, "
                               f"< one global batch", force=True)
            continue

        reset_specialist_(model, SEED + cat_id)
        model.train()

        # pos_weight from THIS category's own mix -- the lever with real
        # dynamic range (positive rate varies 7.7x across categories, while
        # category SIZE varies only 1.4x and is therefore not weighted).
        labels = train_frame["label"]
        pos_mass = float(labels.sum())
        pos_weight = torch.tensor(
            (len(labels) - pos_mass) / max(pos_mass, 1e-9), device=device)
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")

        optimizer = torch.optim.AdamW(make_param_groups(model), weight_decay=WEIGHT_DECAY)
        steps_per_epoch = max(1, n_train // total_batch)
        total_steps = steps_per_epoch * EPOCHS
        warmup_steps = max(1, int(total_steps * WARMUP_RATIO))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, [
            _make_lr_lambda(warmup_steps, total_steps, LR_MIN_RATIO),        # adapter
            _make_head_lr_lambda(warmup_steps, total_steps, LR_MIN_RATIO),   # head
        ])

        _log(rank, "CAT", f"[{cat_id:02d}] {cat_name}: train={n_train:,} "
                          f"val={len(val_frame['label']):,} steps={total_steps} "
                          f"warmup={warmup_steps} pos_weight={pos_weight.item():.3f}", force=True)

        val_full_loader, val_remainder_loader = make_eval_loader(
            val_frame, caches, pad_id, PER_CORE_EVAL_BATCH)
        global_step = 0

        for epoch in range(1, EPOCHS + 1):
            train_loader, sampler = make_train_loader(train_frame, caches, pad_id,
                                                       PER_CORE_TRAIN_BATCH, world_size,
                                                       rank, SEED, epoch=epoch)
            mp_loader = pl_xla.MpDeviceLoader(train_loader, device)
            for step, batch in enumerate(mp_loader):
                input_ids, attention_mask, batch_labels, _cats, batch_scope = batch
                batch_labels = batch_labels.to(device)
                batch_scope = batch_scope.to(device)
                try:
                    optimizer.zero_grad()
                    logits = model(input_ids=input_ids, attention_mask=attention_mask)
                    per_sample = loss_fn(logits, batch_labels)
                    # human rows count HUMAN_WEIGHT-fold against LLM soft labels
                    w = torch.where(batch_scope == 0,
                                    torch.full_like(per_sample, HUMAN_WEIGHT),
                                    torch.ones_like(per_sample))
                    loss = (per_sample * w).sum() / w.sum()
                    loss.backward()
                    xm.optimizer_step(optimizer)
                    scheduler.step()
                    global_step += 1
                except Exception as e:
                    _log(rank, "WARN", f"step failed and was skipped: {type(e).__name__}: {e}",
                         force=True)
                    # del before continue: a live reference to the failed step's
                    # graph keeps its buffers alive into the next step.
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
                        print(f"[{_elapsed():8.1f}s] cat={cat_id:02d} step={global_step}/"
                              f"{total_steps} loss={loss.item():.4f} "
                              f"lr={scheduler.get_last_lr()[0]:.2e}/"
                              f"{scheduler.get_last_lr()[1]:.2e}", flush=True)

                if (time.perf_counter() - run_start) > MAX_TRAIN_SECONDS:
                    _log(rank, "TRAIN", f"wall-clock cutoff during category {cat_id:02d} "
                                        f"-- stopping. Finished categories keep their "
                                        f"checkpoints; a re-run resumes from here.", force=True)
                    stopped = True
                    break
            del mp_loader, train_loader, sampler
            gc.collect()
            if stopped:
                break

        try:
            m = evaluate(model, val_full_loader, val_remainder_loader, device, category_vocab)
            pr_auc = m["per_category"].get(cat_name, float("nan"))
            _log(rank, "METRICS", f"[{cat_id:02d}] {cat_name}: val_pr_auc={pr_auc:.4f} "
                                  f"n_val={m['n']}", force=True)
            save_checkpoint(model, optimizer, global_step, EPOCHS, float(pr_auc),
                            f"{label_prefix}_best", rank,
                            extra={"category": cat_name, "category_id": cat_id,
                                   "n_train": int(n_train),
                                   "llm_rows_per_category": LLM_ROWS_PER_CATEGORY,
                                   "llm_is_subsample": True})
            results[cat_name] = float(pr_auc)
        except Exception as e:
            _log(rank, "ERROR", f"[{cat_id:02d}] eval failed: {type(e).__name__}: {e}", force=True)
            save_checkpoint(model, optimizer, global_step, EPOCHS, None,
                            f"{label_prefix}_last", rank,
                            extra={"category": cat_name, "category_id": cat_id,
                                   "llm_is_subsample": True})
            results[cat_name] = None
        model.train()

        # Mirror results to disk after EVERY category: a cancelled session can
        # come back with a 0-byte log, and this is then the only record.
        if rank == 0:
            done = [v for v in results.values() if v is not None]
            with open(RESULTS_JSON, "w", encoding="utf-8") as f:
                json.dump({"per_category_val_pr_auc": results,
                           "macro_over_finished": float(np.mean(done)) if done else None,
                           "n_finished": len(done),
                           "n_categories": len(category_vocab),
                           "llm_is_subsample": True,
                           "llm_rows_per_category": LLM_ROWS_PER_CATEGORY,
                           "llm_rows_total_available": _SHARED["n_llm_total"],
                           "llm_rows_used": _SHARED["n_llm_used"],
                           "pair_max_length": PAIR_MAX_LENGTH,
                           "val_fingerprint": _SHARED["val_fingerprint"],
                           "split": "OLD canonical pair-level 80/20 seed42"}, f,
                          ensure_ascii=False, indent=2)

    done = [v for v in results.values() if v is not None]
    _log(rank, "DONE", f"total wall-clock {_elapsed():.0f}s  finished={len(done)}/"
                       f"{len(category_vocab)} categories  "
                       f"macro_over_finished={np.mean(done) if done else float('nan'):.4f}",
         force=True)


def main():
    os.environ.pop("TPU_PROCESS_ADDRESSES", None)
    os.environ.pop("CLOUD_TPU_TASK_ID", None)

    _load_shared_data()

    torch_xla.launch(_mp_fn, args=(), start_method="fork")


if __name__ == "__main__":
    main()
