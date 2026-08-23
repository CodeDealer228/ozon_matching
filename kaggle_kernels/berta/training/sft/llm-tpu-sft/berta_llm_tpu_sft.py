"""
Fresh LoRA SFT of BERTA (merged base -- see ../berta-merge-lora, which
folded the current best adapter, rays0n/berta-llm-continue-train,
val_macro_pr_auc=0.7255, into the base sergeyzh/BERTA weights via
merge_and_unload()) on the FULL matches_llm.parquet (11,187,780 LLM soft-
labeled pairs, no sampling this time -- see ../berta-tokenize-llm-256 for
why TPU affords skipping the sampling every prior LLM-continue-train
experiment needed), mixed with the full human train split (292,523 pairs),
at PAIR_MAX_LENGTH=256 (raised from 128 everywhere else in this project --
CLAUDE.md's own Open TODO 3b: 128 truncates 98.76% of real pairs, mean
174.6 tokens lost).

Runs on TPU v5e-8 (8 chips) via PyTorch/XLA, NOT GPU -- a genuinely
different execution model from every other training script in this
project, with its own real gotchas (verified against current PyTorch/XLA
docs before writing this, not assumed from GPU-CUDA habits):

  1. STATIC SHAPES ARE MANDATORY. XLA compiles a fresh graph for every
     distinct input shape it sees; a training loop that lets shapes drift
     (e.g. a ragged final batch) pays a real compile (seconds to minutes)
     on every new shape, not just once. This project's collate function
     already produces a fixed (batch, PAIR_MAX_LENGTH) tensor per batch
     regardless of real per-pair content length (fair-share truncation +
     padding, unchanged from every GPU script) -- the only remaining shape
     variable is the batch dimension itself, fixed here via a CONSTANT
     per-core batch size (chosen once, see the probe phase below) and
     drop_last=True on every DataLoader (train AND eval) so the final
     partial batch of an epoch/pass never introduces a second shape.
  2. BATCH SIZES ARE FIXED CONSTANTS, NOT AUTO-PROBED. This started as a
     port of the GPU-style probe every other training script in this
     project uses, and three real failed TPU runs proved that approach
     doesn't transfer -- it actively causes the OOM it's meant to avoid.
     Every probed candidate compiles its own XLA program and those
     programs STAY RESIDENT in HBM, so probing 5 train candidates left
     ~27GB of compiled programs occupying a 15.75GB device before eval
     probing even started; meanwhile xm.get_memory_info() reported a flat
     0.27GB for every candidate (it cannot see program memory), making it
     useless as a probe signal. Full detail in PER_CORE_TRAIN_BATCH's own
     config comment. Exactly two graph shapes are compiled per run now:
     one train, one eval. A genuine OOM is then a fast, loud, honest
     failure that says "lower the constant" -- not a self-inflicted one.
  3. xm.optimizer_step(optimizer), not optimizer.step() -- handles the
     cross-core gradient all-reduce AND the lazy-graph sync in one call;
     current PyTorch/XLA docs (docs.pytorch.org/xla, pytorch/xla
     API_GUIDE.md, both fetched directly before writing this) confirm this
     is still the correct call, launched via torch_xla.launch(_mp_fn, args=())
     (the current replacement for the older xmp.spawn entrypoint) with
     torch_xla.device() for device acquisition and
     torch_xla.distributed.parallel_loader.MpDeviceLoader wrapping a
     DistributedSampler-sharded DataLoader for the train split.
  4. VALIDATION runs the FULL val set (73,131 pairs, unchanged, unsharded)
     REDUNDANTLY on every one of the 8 replicas rather than sharding val
     + gathering results back to rank 0 -- average_precision_score isn't a
     simple sum-reducible statistic (needs every (score, label) pair
     sorted together per category), so a correct sharded-then-gathered
     eval would need an explicit xm.all_gather of logits/labels/categories
     across replicas; running the (comparatively cheap, no backward pass)
     full eval redundantly on all 8 cores is simpler and safer for a
     first-ever TPU script in this project, at the cost of some idle
     compute during eval -- a deliberate stability-over-throughput choice
     given "оптимизировать и стабилизировать" explicitly asked for both,
     and getting a multi-replica gather subtly wrong would corrupt the one
     number (val_macro_pr_auc) every decision in this project is made
     from. Only rank 0 logs/checkpoints (all ranks compute the identical
     redundant result, so there is nothing to reconcile).
  5. CHECKPOINTING is rank-0-only, with an EXPLICIT .cpu() move of every
     tensor before torch.save -- xm.save() exists specifically to do this
     correctly (handles the XLA-tensor-to-CPU materialization + guards
     against concurrent writes across replicas), used here for the
     classifier head and optimizer state; the LoRA adapter goes through
     peft's own save_pretrained() AFTER an explicit full-model .cpu() copy
     (not attempted in-place on the live XLA-resident training model),
     avoiding any assumption about whether HF's own save path correctly
     materializes lazy XLA tensors on its own.
  6. Precision: the FROZEN base is loaded bf16 (TPU-native fast dtype,
     matches how the base was already saved by ../berta-merge-lora) --
     LoRA A/B and the classifier head are kept float32 (peft's LoraLayer
     casts across the base/adapter dtype boundary internally; the
     trainable low-rank deltas and Adam's own moment estimates are the
     part that actually needs fp32's wider mantissa for stable small
     updates, not the frozen backbone doing forward-only matmuls).

Classifier head is WARM-STARTED from the merged checkpoint's own
classifier.pt (see ../berta-merge-lora), not randomly re-initialized --
merge_and_unload() only touches the LoRA-wrapped encoder layers, so the
task head's already-learned pooled-vector->logit mapping is untouched by
the merge and still directly useful; LoRA itself starts genuinely fresh
(B-matrices zero-initialized, so the merged model's behavior is UNCHANGED
at step 0 -- a clean continuation point, not a discontinuity) since the
old adapter's own parameters no longer exist post-merge and there is no
valid mapping from its optimizer state to a fresh LoRA layer's tensors
either (different objects entirely, same shape only by coincidence) -- the
optimizer starts fresh for the same reason.

POST-MORTEM of the crash this version fixes (real run, real log): training
reached global_step=22400 (a full epoch's worth -- LR had decayed to
~5.87e-10, exactly matching total_steps=steps_per_epoch_global) at
elapsed_train=8769s (~2.44h, nowhere near MAX_TRAIN_SECONDS or Kaggle's 9h
TPU cap), then died with a bare `concurrent.futures.process.BrokenProcessPool:
A process in the process pool was terminated abruptly` bubbling out of
torch_xla.launch() itself -- not a Python exception inside _mp_fn (those are
already caught and logged), so one of the 8 replica processes was killed
from OUTSIDE Python, at the exact moment the `while not stopped` outer loop
looped into a SECOND epoch it was never supposed to run (the cosine
schedule's total_steps is sized for exactly 1 epoch; every further step
trains at ~0 LR, pure wasted TPU time). Two real, compounding bugs, both
fixed below:

  A. HOST RAM x8. Every one of the 8 forked replica processes independently
     re-ran the ENTIRE CPU-side setup: re-read+re-streamed matches.parquet
     AND the full 11.19M-row matches_llm.parquet from disk, and -- worse --
     independently called np.concatenate() on the human+llm item caches
     (12.4M+711K items x 256 tokens), which MATERIALIZES a real ~13GB dense
     array (the surrounding comment about memmaps staying "lazy" was true
     right up until that concatenate call, which is exactly the point it
     stops being true). x8 replicas x ~13GB+ for that one array alone, on
     top of x8 redundant copies of the ~11M-row training DataFrame, is a
     very plausible host-RAM exhaustion -- and it would show up as a silent
     OOM-kill of one worker process, i.e. precisely a BrokenProcessPool,
     with no Python traceback because the kernel never got a chance to
     raise one. Fixed by moving every byte of this CPU-only, TPU-independent
     setup into _load_shared_data(), called ONCE in main() BEFORE
     torch_xla.launch() forks the 8 replicas. torch_xla.launch(...,
     start_method="fork") means every replica is a literal fork() of this
     already-fully-loaded parent process -- copy-on-write means all 8
     replicas share the SAME physical RAM for this data for free, as long
     as nothing mutates it afterward (nothing does; every array here is
     read-only past setup). Net effect: ~8x less host RAM and ~8x less
     redundant CPU/IO work for the exact same training behavior.
  B. WASTED SECOND EPOCH. The `while not stopped` outer loop kept starting
     new epochs (rebuilding a fresh DataLoader + DistributedSampler each
     time) as long as MAX_TRAIN_SECONDS allowed, but the cosine schedule
     was only ever sized for ONE epoch (`total_steps = steps_per_epoch_global`)
     -- so epoch 2 onward trains at an LR pinned near zero, contributing
     essentially nothing, while paying real TPU-hours and (per bug A) a
     real crash-risk moment: tearing down one DataLoader's workers and
     immediately spinning up a fresh one's right at the epoch boundary.
     Fixed by stopping after exactly one epoch, matching what the schedule
     was actually sized for -- consistent with this project's established
     "resume with a fresh schedule sized for the realistic run" convention
     (see kaggle_kernels/frida-lora-v2-epoch). MAX_TRAIN_SECONDS remains as
     an in-epoch safety cutoff (now effectively a no-op at current
     throughput, kept as a guard against a slower future config).
  Also: persistent_workers=True on the train DataLoader was pure downside
  here -- it exists to keep worker subprocesses alive ACROSS epochs of the
  SAME loader object, but this script discards and rebuilds the loader
  every epoch anyway, so the flag only risked a transient overlap between
  the old epoch's workers tearing down and the new epoch's spinning up.
  Now False.
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
import time

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "polars", "peft"])

import numpy as np
import pandas as pd
import polars as pl

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from transformers import AutoModel, AutoTokenizer, get_scheduler, set_seed
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score

try:
    import peft.import_utils
    import peft.tuners.lora.torchao
    peft.import_utils.is_torchao_available = lambda: False
    peft.tuners.lora.torchao.is_torchao_available = lambda: False
except Exception as e:
    print(f"[WARN] torchao patch failed (likely fine): {e}")

from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
from safetensors.torch import save_file as safetensors_save_file

import torch_xla
import torch_xla.core.xla_model as xm
from torch_xla import runtime as xr
import torch_xla.distributed.parallel_loader as pl_xla


# =====================================================================
# CONFIG
# =====================================================================

SEED = 42
MODEL_NAME = "sergeyzh/BERTA"

ITEMS_HUMAN_PATH_HINT = "/kaggle/input/datasets/maxamsecret/ozon-ecup-2026-matching/items_human.parquet"
MATCHES_PATH_HINT = "/kaggle/input/datasets/maxamsecret/ozon-ecup-2026-matching/matches.parquet"
MATCHES_LLM_PATH_HINT = "/kaggle/input/datasets/maxamsecret/ozon-ecup-2026-matching/matches_llm.parquet"

PREFIX = "paraphrase: "
ITEM_MAX_LENGTH = 256   # raised from 128, matches ../berta-tokenize-llm-256's cache
MAX_NAME_CHARS, MAX_CATEGORY_CHARS, MAX_ATTR_CHARS = 128, 64, 384
MAX_ITEM_CHARS = MAX_NAME_CHARS + MAX_CATEGORY_CHARS + MAX_ATTR_CHARS + 64  # 640
PAIR_MAX_LENGTH = 256   # raised from 128 -- see module docstring

CLASSIFIER_DROPOUT = 0.10
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.15
LORA_TARGET_MODULES = ["query", "key", "value", "dense"]

LR = 2e-4
WEIGHT_DECAY = 0.04
WARMUP_RATIO = 0.10

CATEGORY_LABEL_PATTERN = [14948, 30]
ATTRIBUTES_LABEL_PATTERN = [4816, 37519, 521, 30]

# FIXED per-core batch sizes -- NO auto-probing. Three real failed runs
# established that batch-size probing is actively HARMFUL on TPU, in a way
# it simply isn't on GPU (where this project probes successfully in every
# other script):
#
#   1. Every probed candidate compiles its OWN XLA program, and those
#      compiled programs STAY RESIDENT in HBM. Probing 5 train candidates
#      then starting the eval probe hit "Used 27.12G of 15.75G hbm ...
#      program 27.12G" -- the accumulated compiled programs from probing
#      were themselves the thing that exhausted HBM, at an eval batch of
#      only 32. The probe's own cost, not any candidate's real footprint.
#   2. xm.get_memory_info() cannot see that memory at all: it reported a
#      flat 0.27GB/16.91GB (1.6%) for EVERY train candidate from 16 to
#      256 -- including 256, which independently OOM'd at compile time in
#      an earlier run. It reports persistent buffer allocations only, so
#      it is useless as a probe signal here (unlike CUDA's
#      max_memory_allocated, which this project's GPU probes rely on).
#   3. XLA compiles per SHAPE, so a "search" is inherently many
#      compilations -- minutes of wall-clock and HBM each -- where the GPU
#      equivalent is a cheap, instantly-freed allocation attempt.
#
# So: pick fixed sizes, compile exactly ONE train graph and ONE eval graph,
# and let a real OOM (if any) be a loud, fast failure that tells us to
# lower these constants -- rather than a self-inflicted one from probing.
# Sized conservatively against the one hard datapoint available: a single
# compiled train graph at per-core batch 128 fit, while 256 did not.
PER_CORE_TRAIN_BATCH = 64     # x8 cores = 512 effective train batch
PER_CORE_EVAL_BATCH = 128     # forward-only, no gradients/optimizer state

MAX_TRAIN_SECONDS = 28800   # 8h -- Kaggle's TPU session cap is 9h; this
                              # leaves ~1h margin for setup (item cache
                              # mmap open, tokenizer load, batch-size probe)
                              # and the final checkpoint+eval to complete
                              # safely inside the session regardless of
                              # real measured throughput.
VAL_EVERY_FRACTION = 0.50   # validate at every 50% of an epoch, per
                              # explicit instruction.
LOG_EVERY_STEPS = 50

WORKING_DIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
OUTPUT_DIR = os.path.join(WORKING_DIR, "lora_berta_tpu_sft")
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoint")

t_script_start = time.perf_counter()


def _elapsed():
    return time.perf_counter() - t_script_start


def _log(rank, stage, msg, force=False):
    if rank == 0 or force:
        print(f"[{_elapsed():8.1f}s][rank{rank}] {stage}: {msg}", flush=True)


def _malloc_trim():
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def _find_dataset_file(hint, filename):
    if os.path.isfile(hint):
        return hint
    if os.path.isdir("/kaggle/input"):
        for root, _dirs, files in os.walk("/kaggle/input"):
            if filename in files:
                return os.path.join(root, filename)
    return hint


def _find_dir_with_file(filename, near=None):
    if os.path.isdir("/kaggle/input"):
        for root, _dirs, files in os.walk("/kaggle/input"):
            if filename in files and (near is None or near in root):
                return root
    return None


# =====================================================================
# ITEM CACHE LOADING (pre-tokenized by ../berta-tokenize-llm-256, both
# scopes -- no tokenizer calls at train time on TPU at all)
# =====================================================================

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


def load_scope_cache(scope_dir, rank):
    with open(os.path.join(scope_dir, "cache_meta.json")) as f:
        meta = json.load(f)
    n_items = meta["n_items"]
    max_length = meta["max_length"]
    assert max_length == ITEM_MAX_LENGTH, f"{scope_dir}: max_length={max_length} != {ITEM_MAX_LENGTH}"
    assert meta["prefix"] == PREFIX, f"{scope_dir}: prefix={meta['prefix']!r} != {PREFIX!r}"

    # Deliberately kept as lazy memmap views (NOT .copy()'d) here -- with 8
    # replica processes each loading this same cache independently
    # (torch_xla.launch spawns one process per TPU core, no shared memory
    # between them), an eager .copy() would materialize a full private
    # RAM copy of a ~16GB array (12.4M items x 256 tokens, LLM scope) in
    # EACH of the 8 processes before the unified concatenation downstream
    # even happens -- doubling real peak RAM for no benefit, since
    # np.concatenate has to read through these arrays fully anyway and IS
    # the one genuinely necessary materialization point (see build of
    # item_input_ids in _mp_fn). Read-only memmaps of the SAME file path
    # across independent processes also let the OS page cache serve
    # already-read pages to sibling replicas instead of each one paying
    # its own disk I/O -- an eager copy forfeits that too.
    ids = np.memmap(os.path.join(scope_dir, "ids.bin"), dtype=np.int64, mode="r", shape=(n_items,))
    input_ids = np.memmap(os.path.join(scope_dir, "input_ids.bin"), dtype=np.int32, mode="r",
                           shape=(n_items, max_length))
    attn = np.memmap(os.path.join(scope_dir, "attention_mask.bin"), dtype=np.int8, mode="r",
                      shape=(n_items, max_length))
    item_lengths = np.asarray(attn).sum(axis=1).astype(np.int32)
    category_label_start = _find_token_pattern_starts(input_ids, CATEGORY_LABEL_PATTERN)
    attr_label_start = _find_token_pattern_starts(input_ids, ATTRIBUTES_LABEL_PATTERN)

    with open(os.path.join(scope_dir, "category_vocab.json")) as f:
        category_vocab = json.load(f)
    categories_arr = np.load(os.path.join(scope_dir, "categories.npy"))

    _log(rank, "CACHE", f"{scope_dir}: {n_items:,} items loaded")
    return ids, input_ids, item_lengths, category_label_start, attr_label_start, category_vocab, categories_arr


# =====================================================================
# PAIR ASSEMBLY (unchanged fair-share-truncation principle, PAIR_MAX_LENGTH
# now 256 -- content_budget = 256-3 = 253)
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


def make_pair_collate_fn(item_input_ids, item_lengths, category_label_start, attr_label_start,
                          cls_id, sep_id, pad_id, pair_max_length):
    content_budget = pair_max_length - 3

    def collate_fn(batch):
        row1, row2, labels, categories = zip(*batch)
        bsz = len(batch)
        input_ids = np.full((bsz, pair_max_length), pad_id, dtype=np.int64)
        token_type_ids = np.zeros((bsz, pair_max_length), dtype=np.int64)
        attention_mask = np.zeros((bsz, pair_max_length), dtype=np.int64)

        for i in range(bsz):
            r1, r2 = row1[i], row2[i]
            compact1 = build_compact_ids(item_input_ids, r1, int(item_lengths[r1]),
                                          int(category_label_start[r1]), int(attr_label_start[r1]))
            compact2 = build_compact_ids(item_input_ids, r2, int(item_lengths[r2]),
                                          int(category_label_start[r2]), int(attr_label_start[r2]))
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


def make_train_loader(pairs_pd, id_to_row, category_to_id, item_input_ids, item_lengths,
                       category_label_start, attr_label_start, cls_id, sep_id, pad_id,
                       pair_max_length, per_core_batch, num_replicas, rank, seed, epoch):
    row1 = pairs_pd["id1"].map(id_to_row).to_numpy().astype(np.int64)
    row2 = pairs_pd["id2"].map(id_to_row).to_numpy().astype(np.int64)
    labels = pairs_pd["label"].to_numpy().astype(np.float32)
    categories = pairs_pd["category1"].map(category_to_id).fillna(category_to_id.get("UNKNOWN", -1)).to_numpy().astype(np.int64)

    dataset = PairIndexDataset(row1, row2, labels, categories)
    sampler = DistributedSampler(dataset, num_replicas=num_replicas, rank=rank, shuffle=True, seed=seed,
                                  drop_last=True)
    sampler.set_epoch(epoch)
    collate_fn = make_pair_collate_fn(item_input_ids, item_lengths, category_label_start, attr_label_start,
                                       cls_id, sep_id, pad_id, pair_max_length)
    loader = DataLoader(dataset, batch_size=per_core_batch, sampler=sampler, drop_last=True,
                         collate_fn=collate_fn, num_workers=2, persistent_workers=False)
    # persistent_workers=False deliberately: this loader is rebuilt fresh
    # every epoch (new DistributedSampler each time), so there is no
    # "next batch of the same loader" for persistent workers to serve --
    # keeping it True only risked a transient overlap between the outgoing
    # epoch's workers tearing down and the incoming epoch's spinning up
    # (a real contributor to the RAM-exhaustion crash this script now
    # avoids in the first place by never reaching a second epoch -- see
    # module docstring post-mortem point B -- but harmless to fix either way).
    return loader, sampler


def make_eval_loader(pairs_pd, id_to_row, category_to_id, item_input_ids, item_lengths,
                      category_label_start, attr_label_start, cls_id, sep_id, pad_id,
                      pair_max_length, eval_batch):
    """UNSHARDED -- every replica evaluates the FULL val set redundantly,
    see module docstring point 4. drop_last=True keeps the shape fixed;
    the tiny remainder (< eval_batch rows) is covered by a second,
    separately-shaped pass -- two fixed shapes total (one compile each),
    not a ragged stream of unique shapes."""
    row1 = pairs_pd["id1"].map(id_to_row).to_numpy().astype(np.int64)
    row2 = pairs_pd["id2"].map(id_to_row).to_numpy().astype(np.int64)
    labels = pairs_pd["label"].to_numpy().astype(np.float32)
    categories = pairs_pd["category1"].map(category_to_id).fillna(category_to_id.get("UNKNOWN", -1)).to_numpy().astype(np.int64)

    dataset = PairIndexDataset(row1, row2, labels, categories)
    collate_fn = make_pair_collate_fn(item_input_ids, item_lengths, category_label_start, attr_label_start,
                                       cls_id, sep_id, pad_id, pair_max_length)
    n = len(dataset)
    n_full = (n // eval_batch) * eval_batch
    full_loader = DataLoader(torch.utils.data.Subset(dataset, list(range(n_full))), batch_size=eval_batch,
                              shuffle=False, drop_last=True, collate_fn=collate_fn, num_workers=2)
    remainder_loader = None
    if n_full < n:
        remainder_loader = DataLoader(torch.utils.data.Subset(dataset, list(range(n_full, n))),
                                       batch_size=n - n_full, shuffle=False, drop_last=False,
                                       collate_fn=collate_fn, num_workers=0)
    return full_loader, remainder_loader


# =====================================================================
# MODEL
# =====================================================================

class ClassifierHead(nn.Module):
    def __init__(self, hidden_dim, dropout_p):
        super().__init__()
        self.dropout = nn.Dropout(dropout_p)
        self.linear = nn.Linear(hidden_dim, 1)

    def forward(self, pooled):
        return self.linear(self.dropout(pooled))


class BertaCrossEncoderClassifier(nn.Module):
    """No autocast here (unlike the GPU scripts) -- the frozen base is
    already loaded bf16 directly (see module docstring point 6), and XLA's
    own lazy execution + the compiler's own layout choices handle mixed
    fp32 (LoRA/classifier) + bf16 (frozen base) composition without needing
    torch.autocast's CUDA-oriented context manager at all."""

    def __init__(self, encoder, head):
        super().__init__()
        self.encoder = encoder
        self.classifier = head

    def forward(self, input_ids, attention_mask, token_type_ids):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        hidden = out.last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        summed = (hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        pooled = (summed / counts).float()
        return self.classifier(pooled).view(-1).float()


def _find_merged_base_dir():
    if os.path.isdir("/kaggle/input"):
        for root, _dirs, files in os.walk("/kaggle/input"):
            if "merge_meta.json" in files:
                return root
    raise RuntimeError("Merged BERTA base not found under /kaggle/input -- expected "
                        "maxamsecret/berta-merge-lora mounted (kernel_sources).")


def build_fresh_model(device, rank):
    merged_dir = _find_merged_base_dir()
    with open(os.path.join(merged_dir, "merge_meta.json")) as f:
        merge_meta = json.load(f)
    _log(rank, "MODEL", f"merged base meta: {merge_meta}")

    encoder_dir = os.path.join(merged_dir, "merged_encoder")
    base_model = AutoModel.from_pretrained(encoder_dir, dtype=torch.bfloat16)
    hidden_dim = base_model.config.hidden_size

    lora_config = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
                              target_modules=LORA_TARGET_MODULES, bias="none")
    peft_model = get_peft_model(base_model, lora_config)
    if rank == 0:
        peft_model.print_trainable_parameters()

    # LoRA A/B kept fp32 for optimizer stability (see module docstring
    # point 6) -- get_peft_model() matches the base's bf16 dtype by
    # default, so this is an explicit upcast, not the default.
    for name, p in peft_model.named_parameters():
        if "lora_" in name:
            p.data = p.data.float()

    classifier_head = ClassifierHead(hidden_dim, CLASSIFIER_DROPOUT)
    classifier_path = os.path.join(merged_dir, "classifier.pt")
    if os.path.isfile(classifier_path):
        raw_state = torch.load(classifier_path, map_location="cpu")
        if "linear.weight" in raw_state:
            classifier_head.load_state_dict(raw_state)
        else:
            classifier_head.linear.load_state_dict(raw_state)
        _log(rank, "MODEL", "classifier head WARM-STARTED from the merged checkpoint's own classifier.pt")
    else:
        _log(rank, "MODEL", "no classifier.pt found -- classifier head randomly initialized")

    model = BertaCrossEncoderClassifier(peft_model, classifier_head).to(device)
    return model, hidden_dim


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


def evaluate(model, full_loader, remainder_loader, device, category_names, rank):
    model.eval()
    all_logits, all_labels, all_categories = [], [], []
    with torch.no_grad():
        for loader in (full_loader, remainder_loader):
            if loader is None:
                continue
            for batch in loader:
                input_ids, attention_mask, token_type_ids, batch_labels, batch_categories = batch
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                token_type_ids = token_type_ids.to(device)
                logits = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
                # .cpu() itself forces real materialization of logits (reads
                # its data). Deliberately NOT a bare xm.mark_step() with
                # nothing yet consuming the tensor -- that exact ordering
                # produced a real "Check failed: tensor_data"
                # (xla_graph_executor.cpp:691) crash in an earlier version
                # of this script's now-removed eval probe. No separate
                # mark_step() call needed here; .cpu() already triggers it.
                all_logits.append(logits.cpu())
                all_labels.append(batch_labels)
                all_categories.append(batch_categories)
    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    categories = np.concatenate(all_categories, axis=0)
    probs = torch.sigmoid(logits).numpy()
    macro, per_cat = compute_metrics(probs, labels.numpy(), categories, category_names)
    model.train()
    return {"macro_pr_auc": macro, "per_category": per_cat, "n": len(labels)}


# =====================================================================
# CHECKPOINT (rank 0 only, explicit CPU moves -- see module docstring point 5)
# =====================================================================

def save_checkpoint(model, optimizer, global_step, best_val_macro_pr_auc, save_dir, rank):
    """Rank-0-only. Deliberately does NOT call model.to('cpu') -- that would
    mutate the LIVE training model's parameter tensors in place (nn.Module.to
    moves .data, not a copy), leaving it stranded on CPU for the next
    training step unless carefully moved back, a real footgun for a run
    that's supposed to keep training for hours after this save. Every read
    below is a non-mutating .detach().cpu() on an individual tensor instead.
    The (nested, param-id-keyed) optimizer state dict is the one exception --
    handed to xm.save(), which is specifically documented to recursively
    materialize any XLA tensor it finds, rather than hand-rolling that
    recursion here."""
    if rank != 0:
        return
    xm.mark_step()
    os.makedirs(save_dir, exist_ok=True)

    adapter_dir = os.path.join(save_dir, "adapter")
    os.makedirs(adapter_dir, exist_ok=True)
    lora_sd = get_peft_model_state_dict(model.encoder)
    lora_sd_cpu = {k: v.detach().cpu() for k, v in lora_sd.items()}
    safetensors_save_file(lora_sd_cpu, os.path.join(adapter_dir, "adapter_model.safetensors"))
    model.encoder.peft_config["default"].save_pretrained(adapter_dir)

    classifier_sd_cpu = {k: v.detach().cpu() for k, v in model.classifier.state_dict().items()}
    torch.save(classifier_sd_cpu, os.path.join(save_dir, "classifier.pt"))

    xm.save(optimizer.state_dict(), os.path.join(save_dir, "optimizer.pt"))

    with open(os.path.join(save_dir, "train_state.json"), "w") as f:
        json.dump({"global_step": global_step, "best_val_macro_pr_auc": best_val_macro_pr_auc}, f, indent=2)
    print(f"[SAVE] checkpoint written to {save_dir} (step={global_step}, "
          f"best_val_macro_pr_auc={best_val_macro_pr_auc:.4f})", flush=True)


# =====================================================================
# SHARED CPU-ONLY SETUP -- loaded EXACTLY ONCE, in the parent process,
# BEFORE torch_xla.launch() forks the 8 replicas (see module docstring
# post-mortem point A). Every replica then inherits this via copy-on-write
# for free instead of re-reading+re-processing it 8 times independently.
# Stored in this module-level dict rather than passed through
# torch_xla.launch()'s args= (which would risk pickling ~13GB+ of arrays)
# -- fork() makes any object already bound at fork time available to every
# child process without any explicit passing at all.
# =====================================================================

_SHARED = {}


def _load_shared_data():
    print(f"[{_elapsed():8.1f}s][SETUP] loading all CPU-only data once, pre-fork ...", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    cls_id, sep_id, pad_id = tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id

    # -------------------------------------------------------------
    # Canonical human split (byte-identical split logic to every other
    # BERTA script in this project).
    # -------------------------------------------------------------
    matches_path = _find_dataset_file(MATCHES_PATH_HINT, "matches.parquet")
    matches_pl = pl.read_parquet(matches_path)
    matches_pl = (
        matches_pl.filter(pl.col("id1").is_not_null() & pl.col("id2").is_not_null())
        .with_columns([pl.col("target").fill_null(0).cast(pl.Float32),
                       (pl.col("target") >= 0.5).cast(pl.Int8).alias("label")])
        .select(["id1", "id2", "label"])
    )
    matches_pd = matches_pl.to_pandas()
    del matches_pl
    labels_all = matches_pd["label"].to_numpy().astype(np.int64)
    indices = np.arange(len(labels_all), dtype=np.int64)
    train_idx, val_idx = train_test_split(indices, test_size=0.20, random_state=SEED, stratify=labels_all)
    human_train_pd = matches_pd.iloc[train_idx][["id1", "id2", "label"]].reset_index(drop=True).copy()
    val_pd = matches_pd.iloc[val_idx][["id1", "id2", "label"]].reset_index(drop=True).copy()
    human_train_pd["label"] = human_train_pd["label"].astype(np.float32)
    val_pd["label"] = val_pd["label"].astype(np.float32)
    val_fingerprint = hashlib.sha256(
        ",".join(f"{a}-{b}" for a, b in sorted(zip(val_pd["id1"], val_pd["id2"]))).encode("utf-8")
    ).hexdigest()[:16]
    print(f"[{_elapsed():8.1f}s][SETUP] human train={len(human_train_pd):,} val={len(val_pd):,} "
          f"val_fingerprint={val_fingerprint}", flush=True)
    del matches_pd, labels_all, indices, train_idx, val_idx
    gc.collect()

    # -------------------------------------------------------------
    # Full matches_llm.parquet -- no sampling (see module docstring).
    # -------------------------------------------------------------
    matches_llm_path = _find_dataset_file(MATCHES_LLM_PATH_HINT, "matches_llm.parquet")
    llm_train_pl = (
        pl.scan_parquet(matches_llm_path)
        .filter(pl.col("id1").is_not_null() & pl.col("id2").is_not_null())
        .with_columns(pl.col("target").fill_null(0).cast(pl.Float32).alias("label"))
        .select(["id1", "id2", "label"])
        .collect(engine="streaming")
    )
    llm_train_pd = llm_train_pl.to_pandas()
    del llm_train_pl
    gc.collect()
    print(f"[{_elapsed():8.1f}s][SETUP] matches_llm (full, no sampling): {len(llm_train_pd):,}", flush=True)

    # -------------------------------------------------------------
    # Item caches (both scopes, pre-tokenized at 256 -- ../berta-tokenize-llm-256).
    # -------------------------------------------------------------
    llm_scope_dir = _find_dir_with_file("cache_meta.json", near="llm_scope")
    human_scope_dir = _find_dir_with_file("cache_meta.json", near="human_scope")
    if llm_scope_dir is None or human_scope_dir is None:
        raise RuntimeError("Tokenized item caches not found under /kaggle/input -- expected "
                            "maxamsecret/berta-tokenize-llm-256 mounted (kernel_sources) with "
                            "llm_scope/ and human_scope/ subdirs.")

    (human_ids, human_input_ids, human_lengths, human_cat_start, human_attr_start,
     human_cat_vocab, _human_cats_arr) = load_scope_cache(human_scope_dir, 0)
    (llm_ids, llm_input_ids, llm_lengths, llm_cat_start, llm_attr_start,
     llm_cat_vocab, _llm_cats_arr) = load_scope_cache(llm_scope_dir, 0)

    n_human = len(human_ids)
    # This np.concatenate() is the ~13GB materialization flagged in the
    # module docstring post-mortem -- now paid exactly ONCE (here, pre-fork)
    # instead of once per replica, and then shared read-only across all 8
    # forked processes via copy-on-write.
    item_input_ids = np.concatenate([human_input_ids, llm_input_ids], axis=0)
    item_lengths = np.concatenate([human_lengths, llm_lengths])
    category_label_start = np.concatenate([human_cat_start, llm_cat_start])
    attr_label_start = np.concatenate([human_attr_start, llm_attr_start])
    del human_input_ids, llm_input_ids, human_lengths, llm_lengths
    del human_cat_start, llm_cat_start, human_attr_start, llm_attr_start
    gc.collect()

    id_to_row = {int(i): row for row, i in enumerate(human_ids.tolist())}
    for row, i in enumerate(llm_ids.tolist()):
        if int(i) not in id_to_row:
            id_to_row[int(i)] = n_human + row
    print(f"[{_elapsed():8.1f}s][SETUP] unified item cache: {item_input_ids.shape[0]:,} rows "
          f"(human={n_human:,} + llm={len(llm_ids):,})", flush=True)

    category_vocab = sorted(set(human_cat_vocab) | set(llm_cat_vocab))
    category_to_id = {c: i for i, c in enumerate(category_vocab)}

    items_human_path = _find_dataset_file(ITEMS_HUMAN_PATH_HINT, "items_human.parquet")
    needed_human_ids = pd.unique(pd.concat([human_train_pd["id1"], val_pd["id1"]]))
    needed_ids_lazy = pl.DataFrame({"id": pd.Series(needed_human_ids, dtype="int64")}).lazy()
    items_cat = (
        pl.scan_parquet(items_human_path).join(needed_ids_lazy, on="id", how="semi")
        .select(["id", "category"])
        .with_columns(pl.col("category").fill_null("UNKNOWN").cast(pl.Utf8).str.slice(0, MAX_CATEGORY_CHARS))
        .collect(engine="streaming")
    )
    id_to_category = dict(zip(items_cat.get_column("id").to_list(), items_cat.get_column("category").to_list()))
    human_train_pd["category1"] = human_train_pd["id1"].map(id_to_category).fillna("UNKNOWN")
    val_pd["category1"] = val_pd["id1"].map(id_to_category).fillna("UNKNOWN")
    del items_cat, needed_human_ids, needed_ids_lazy, id_to_category
    gc.collect()

    llm_id_to_cat = {int(i): llm_cat_vocab[int(c)] for i, c in zip(llm_ids.tolist(), _llm_cats_arr.tolist())}
    llm_train_pd["category1"] = llm_train_pd["id1"].map(llm_id_to_cat).fillna("UNKNOWN")
    del llm_id_to_cat, _llm_cats_arr, _human_cats_arr
    gc.collect()

    train_pd = pd.concat([human_train_pd, llm_train_pd], ignore_index=True)
    print(f"[{_elapsed():8.1f}s][SETUP] combined train: human={len(human_train_pd):,} + "
          f"llm={len(llm_train_pd):,} = {len(train_pd):,}", flush=True)
    del human_train_pd, llm_train_pd
    gc.collect()

    _SHARED.update(dict(
        cls_id=cls_id, sep_id=sep_id, pad_id=pad_id,
        train_pd=train_pd, val_pd=val_pd, val_fingerprint=val_fingerprint,
        item_input_ids=item_input_ids, item_lengths=item_lengths,
        category_label_start=category_label_start, attr_label_start=attr_label_start,
        id_to_row=id_to_row, category_vocab=category_vocab, category_to_id=category_to_id,
    ))
    print(f"[{_elapsed():8.1f}s][SETUP] done -- {len(_SHARED)} shared objects ready for fork", flush=True)


# =====================================================================
# MAIN (per-replica entrypoint)
# =====================================================================

def _mp_fn(index):
    rank = xr.global_ordinal()
    world_size = xr.world_size()
    device = torch_xla.device()
    set_seed(SEED + rank)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    _log(rank, "CONFIG", f"world_size={world_size} device={device} pair_max_length={PAIR_MAX_LENGTH}", force=True)
    if rank == 0 and os.path.isdir("/kaggle/input"):
        for _root, _dirs, _files in os.walk("/kaggle/input"):
            depth = _root[len("/kaggle/input"):].count(os.sep)
            if depth <= 3:
                print(f"[DEBUG] {_root} -> dirs={_dirs} files={_files}")

    # Everything below was loaded ONCE in the parent process by
    # _load_shared_data() before torch_xla.launch() forked this replica --
    # inherited via copy-on-write, not re-read/re-processed here. See
    # module docstring post-mortem point A.
    cls_id, sep_id, pad_id = _SHARED["cls_id"], _SHARED["sep_id"], _SHARED["pad_id"]
    train_pd, val_pd, val_fingerprint = _SHARED["train_pd"], _SHARED["val_pd"], _SHARED["val_fingerprint"]
    item_input_ids, item_lengths = _SHARED["item_input_ids"], _SHARED["item_lengths"]
    category_label_start, attr_label_start = _SHARED["category_label_start"], _SHARED["attr_label_start"]
    id_to_row = _SHARED["id_to_row"]
    category_vocab, category_to_id = _SHARED["category_vocab"], _SHARED["category_to_id"]
    _log(rank, "DATA", f"inherited from parent (fork/COW): train={len(train_pd):,} val={len(val_pd):,} "
                        f"val_fingerprint={val_fingerprint} item_cache_rows={item_input_ids.shape[0]:,}",
         force=True)

    # -------------------------------------------------------------
    # Model + optimizer (fresh LoRA on the merged base, warm-started
    # classifier -- see module docstring).
    # -------------------------------------------------------------
    model, hidden_dim = build_fresh_model(device, rank)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    total_params = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in trainable_params)
    _log(rank, "MODEL", f"total={total_params:,} trainable={trainable_count:,} "
                        f"({100 * trainable_count / total_params:.4f}%)", force=True)

    # Fixed batch sizes, NO probing -- see PER_CORE_TRAIN_BATCH's own
    # comment for the three real failed runs that established why probing
    # is harmful specifically on TPU. Exactly two graph shapes get compiled
    # this whole run: one train, one eval.
    per_core_train_batch = PER_CORE_TRAIN_BATCH
    per_core_eval_batch = PER_CORE_EVAL_BATCH
    total_batch = per_core_train_batch * world_size
    _log(rank, "BATCH", f"FIXED per_core_train_batch={per_core_train_batch} "
                        f"per_core_eval_batch={per_core_eval_batch} world_size={world_size} "
                        f"total_train_batch={total_batch} (no probing -- see config comment)", force=True)

    pos_rate = float(train_pd["label"].mean())
    pos_weight = torch.tensor((1.0 - pos_rate) / max(pos_rate, 1e-9), device=device)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(trainable_params, lr=LR, weight_decay=WEIGHT_DECAY)
    steps_per_epoch_global = len(train_pd) // total_batch
    total_steps = steps_per_epoch_global  # 1 epoch's worth of steps drives the cosine schedule --
                                            # and this run now STOPS after exactly that one epoch
                                            # (see the training loop below), rather than continuing
                                            # into a second epoch that would train at the schedule's
                                            # near-zero tail LR for free. Was previously allowed to
                                            # keep looping epochs until MAX_TRAIN_SECONDS -- see
                                            # module docstring post-mortem point B for why that was
                                            # both wasted compute AND a real crash contributor.
    warmup_steps = max(1, int(total_steps * WARMUP_RATIO))
    scheduler = get_scheduler("cosine", optimizer=optimizer, num_warmup_steps=warmup_steps,
                               num_training_steps=total_steps)
    _log(rank, "TRAIN", f"steps/epoch (global)={steps_per_epoch_global} warmup={warmup_steps} "
                        f"max_train_seconds={MAX_TRAIN_SECONDS}", force=True)

    val_full_loader, val_remainder_loader = make_eval_loader(
        val_pd, id_to_row, category_to_id, item_input_ids, item_lengths,
        category_label_start, attr_label_start, cls_id, sep_id, pad_id, PAIR_MAX_LENGTH, per_core_eval_batch,
    )

    global_step = 0
    best_val_macro_pr_auc = -1.0
    checkpoint_interval = max(1, int(steps_per_epoch_global * VAL_EVERY_FRACTION))
    train_start = time.perf_counter()

    # Exactly ONE epoch, matching total_steps above -- no outer `while`
    # looping into further epochs. See module docstring post-mortem point B:
    # the schedule was always sized for 1 epoch, so a second epoch would
    # have trained at ~0 LR for free while paying real TPU-hours and (via
    # the old per-replica data reload this version also removes) a real
    # crash-risk moment at the epoch boundary. MAX_TRAIN_SECONDS stays as
    # an in-epoch safety cutoff below.
    train_loader, sampler = make_train_loader(
        train_pd, id_to_row, category_to_id, item_input_ids, item_lengths,
        category_label_start, attr_label_start, cls_id, sep_id, pad_id, PAIR_MAX_LENGTH,
        per_core_train_batch, world_size, rank, SEED, epoch=0,
    )
    mp_loader = pl_xla.MpDeviceLoader(train_loader, device)
    _log(rank, "EPOCH", "starting the (only) epoch", force=True)

    window_loss_sum, window_loss_n = 0.0, 0
    for step, batch in enumerate(mp_loader):
        input_ids, attention_mask, token_type_ids, batch_labels, _batch_categories = batch
        batch_labels = batch_labels.to(device)

        optimizer.zero_grad()
        logits = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        loss = loss_fn(logits, batch_labels)
        loss.backward()
        xm.optimizer_step(optimizer)
        scheduler.step()
        global_step += 1

        if global_step % LOG_EVERY_STEPS == 0:
            xm.mark_step()
            loss_val = loss.item()
            window_loss_sum += loss_val
            window_loss_n += 1
            if rank == 0:
                elapsed_train = time.perf_counter() - train_start
                print(f"[{_elapsed():8.1f}s] step={global_step} loss={loss_val:.4f} "
                      f"lr={scheduler.get_last_lr()[0]:.2e} elapsed_train={elapsed_train:.0f}s", flush=True)

        if global_step % checkpoint_interval == 0:
            # NOTHING in the checkpoint/metrics path may kill a
            # multi-hour run -- these are observability, not training.
            # On failure: log loudly, restore train mode, keep going.
            try:
                val_metrics = evaluate(model, val_full_loader, val_remainder_loader, device,
                                        category_vocab, rank)
                improved = val_metrics["macro_pr_auc"] > best_val_macro_pr_auc
                if improved:
                    best_val_macro_pr_auc = val_metrics["macro_pr_auc"]
                _log(rank, "METRICS", f"step={global_step} val_macro_pr_auc={val_metrics['macro_pr_auc']:.4f} "
                                      f"best={best_val_macro_pr_auc:.4f} n_val={val_metrics['n']}", force=True)
                if rank == 0:
                    print(f"[METRICS] per_category: "
                          f"{json.dumps(val_metrics['per_category'], ensure_ascii=False)}", flush=True)
                save_checkpoint(model, optimizer, global_step, best_val_macro_pr_auc, CHECKPOINT_DIR, rank)
            except Exception as e:
                _log(rank, "ERROR", f"checkpoint/eval at step={global_step} FAILED and was skipped -- "
                                    f"training continues. {type(e).__name__}: {e}", force=True)
                if rank == 0:
                    import traceback
                    traceback.print_exc()
                model.train()
                optimizer.zero_grad(set_to_none=True)
                gc.collect()

        elapsed_train = time.perf_counter() - train_start
        if elapsed_train > MAX_TRAIN_SECONDS:
            _log(rank, "TRAIN", f"wall-clock cutoff reached ({elapsed_train:.0f}s > {MAX_TRAIN_SECONDS}s) "
                                f"-- stopping.", force=True)
            break

    del mp_loader, train_loader, sampler
    gc.collect()

    # Final eval + checkpoint. The eval is wrapped (same reasoning as the
    # in-loop one), but the final save is attempted REGARDLESS of whether
    # eval succeeded -- hours of real training must not be lost just
    # because the metric that describes it couldn't be computed.
    try:
        val_metrics = evaluate(model, val_full_loader, val_remainder_loader, device, category_vocab, rank)
        if val_metrics["macro_pr_auc"] > best_val_macro_pr_auc:
            best_val_macro_pr_auc = val_metrics["macro_pr_auc"]
        _log(rank, "FINAL", f"val_macro_pr_auc={val_metrics['macro_pr_auc']:.4f} "
                            f"best={best_val_macro_pr_auc:.4f}", force=True)
    except Exception as e:
        _log(rank, "ERROR", f"final eval FAILED ({type(e).__name__}: {e}) -- saving the checkpoint anyway",
             force=True)
        if rank == 0:
            import traceback
            traceback.print_exc()

    try:
        save_checkpoint(model, optimizer, global_step, best_val_macro_pr_auc, CHECKPOINT_DIR, rank)
    except Exception as e:
        _log(rank, "ERROR", f"final checkpoint save FAILED: {type(e).__name__}: {e}", force=True)
        if rank == 0:
            import traceback
            traceback.print_exc()
    _log(rank, "DONE", f"total wall-clock: {_elapsed():.0f}s", force=True)


def main():
    # Kaggle-specific fix, confirmed against the official pytorch/xla Kaggle
    # example notebook (contrib/kaggle/distributed-pytorch-xla-basics-with-
    # pjrt.ipynb) -- a real crash without this: "TPU initialization failed:
    # Invalid --..._slice_builder_worker_addresses specified. Expected 8
    # worker addresses, got 1." Kaggle's TPU kernel environment pre-sets
    # TPU_PROCESS_ADDRESSES/CLOUD_TPU_TASK_ID (for some other framework's
    # multi-host assumptions), which conflicts with xmp.spawn's own
    # mechanism for establishing its 8-worker (one per v5e-8 chip) PJRT
    # topology -- the notebook calls this a "temporary hack: remove some
    # TPU environment variables to support multiprocessing. These will be
    # set later by xmp.spawn." Also per that same notebook: Kaggle kernels
    # (even "script" type ones -- confirmed from this project's own kernel
    # logs, which show Kaggle converting every kernel to a notebook
    # internally via nbconvert regardless of declared kernel_type) run as
    # an "interactive notebook" from torch_xla's perspective, which
    # REQUIRES start_method="fork" explicitly (the default differs outside
    # notebooks).
    os.environ.pop("TPU_PROCESS_ADDRESSES", None)
    os.environ.pop("CLOUD_TPU_TASK_ID", None)

    # Load all CPU-only data ONCE, here, before forking -- see _load_shared_data
    # and the module docstring post-mortem (point A) for why this is the fix
    # for the real host-RAM-exhaustion crash this version addresses. Must
    # happen strictly before torch_xla.launch() so the fork()'d replicas
    # inherit it via copy-on-write instead of each re-loading it themselves.
    _load_shared_data()

    torch_xla.launch(_mp_fn, args=(), start_method="fork")


if __name__ == "__main__":
    main()
