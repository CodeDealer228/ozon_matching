"""
Caches the per-PAIR pooled representation (768-dim, mean-pooled
last_hidden_state) that the trained BERTA LoRA adapter
(maxamsecret/berta-lora-sft-5epoch, val_macro_pr_auc=0.7111) produces for
every one of the 365,654 human-labeled pairs (292,523 train + 73,131 val,
canonical seed-42 split) -- forward-only, no gradients, no training. NOT a
per-item embedding: BERTA is used here as a cross-encoder ([CLS] item1
[SEP] item2 [SEP], jointly attended), so the cached vector is already a
function of the specific pair, not something you can mix-and-match across
different pairs the way a bi-encoder embedding would be.

Purpose: decouple head-architecture experiments (MLP vs Linear, activation,
depth, dropout, ...) from the encoder forward pass. Once this cache exists,
training a new head is pure array math on a fixed 365,654x768 feature
matrix -- no GPU-encoder pass needed per experiment. Measured (not
estimated) from berta-lora-sft-human-only-20-epochs's own real training
log: one full 73,131-pair forward-only eval pass took 14:43 (883s,
consistent across 3 separate checkpoints in that run, batch=512 under
DataParallel across 2 GPUs). All 365,654 pairs is exactly 5x that -- this
kernel's own 2-independent-process split should land in the same ballpark
total GPU-time, ideally faster given no DataParallel scatter/gather
overhead and no classifier head in the forward pass.

OUTPUT ONLY -- this kernel does not create or publish a Kaggle dataset
itself (explicit instruction). Everything lands under /kaggle/working;
publish by hand afterward, same as berta-lora-sft-5epoch was.

===========================================================================
WHY 2 INDEPENDENT PROCESSES, NOT DataParallel, NOT DistributedDataParallel
===========================================================================
This is pure inference: no backward() call anywhere, hence nothing for
DDP's gradient-sync machinery to actually do -- wrapping in
DistributedDataParallel would add setup overhead (gradient bucket
allocation, hook registration) for zero benefit. DataParallel would work
(BERTA's own training script already proves DataParallel + autocast-
inside-forward is safe, see BertaCrossEncoderClassifier there and
kaggle_kernels/frida-nan-diagnose for why the inside-forward placement
matters), but its per-batch scatter/replicate/gather round trip through
the main process is pure overhead for an embarrassingly-parallel workload
where the two GPUs never need to exchange anything until the very end.

Instead: `torch.distributed.run --standalone --nproc_per_node=N` launches N
fully independent worker processes, each pinned to one GPU
(torch.cuda.set_device(local_rank)), each loading its OWN full copy of the
model and processing its OWN disjoint half of the data, writing its own
shard to disk. dist.init_process_group is used only for a cheap barrier
before rank 0 merges the two shards -- not for gradient sync. A useful
side effect: each rank is a genuinely separate OS process with its own
Python main thread, so the specific DataParallel+autocast pitfall
documented in frida-nan-diagnose (autocast entered in a calling thread not
propagating into DataParallel's per-replica WORKER threads within one
process) structurally cannot occur here at all -- there are no worker
threads, only worker processes, each running its own autocast normally.

This is the SAME launch mechanism already proven on this exact Kaggle
2xT4 environment by kaggle_kernels/qwen3-bi-encoder-lora-sft (bi-encoder
branch) -- deliberately reused rather than torch.multiprocessing.spawn,
which was tried first here and crashed: rank 0 died with no Python
traceback at all (an OOM-kill signature -- see CLAUDE.md's own documented
"a real Linux OOM-kill isn't a catchable MemoryError") partway through
loading the checkpoint, taking down its TCPStore server and rank 1 with
a "connection closed" error. Root cause wasn't fully isolated before
switching mechanisms; torch.distributed.run is the known-safe choice, not
a guess that it'll avoid the same issue.

Sequential model loading across ranks (rank 0 first, then rank 1, with a
barrier between) -- same lesson already applied in this project's other
multi-GPU kernels (avoids two workers materializing the same checkpoint
simultaneously).

Data split (contiguous halves per rank, not interleaved): each pair's
global row index is deterministic, so writing per-rank shards to
CONTIGUOUS index ranges keeps them trivially mergeable in the right order
afterward as a single concatenation, with no risk of two processes ever
touching overlapping bytes of the same file (each writes its own separate
shard file).
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
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset

from tqdm.auto import tqdm

from transformers import AutoModel, AutoTokenizer, set_seed
from sklearn.model_selection import train_test_split

# NOTE: `peft` is intentionally NOT imported at module level. This module
# gets re-imported fresh in each of mp.spawn's child processes (the "spawn"
# start method does not fork memory -- each child re-executes module-level
# code). Installing/importing peft at import time would mean the pip
# install subprocess runs redundantly in the parent AND in both children,
# with a real risk of two children's installs racing each other on the
# same site-packages concurrently. Instead: main() runs the pip install
# exactly once, synchronously, in the parent process, BEFORE calling
# mp.spawn -- so by the time any child imports peft (done lazily, inside
# _import_peft() below, called from worker()), it's already on disk and
# importing is just a normal, safe, already-installed import.


def _import_peft():
    try:
        import peft.import_utils
        import peft.tuners.lora.torchao
        peft.import_utils.is_torchao_available = lambda: False
        peft.tuners.lora.torchao.is_torchao_available = lambda: False
    except Exception as e:
        print(f"[WARN] torchao patch failed (likely fine): {e}")
    from peft import PeftModel
    return PeftModel


# =====================================================================
# CONFIG
# =====================================================================

SEED = 42
MODEL_NAME = "sergeyzh/BERTA"
HIDDEN_DIM = 768

ITEMS_HUMAN_PATH_HINT = "/kaggle/input/datasets/maxamsecret/ozon-ecup-2026-matching/items_human.parquet"
MATCHES_PATH_HINT = "/kaggle/input/datasets/maxamsecret/ozon-ecup-2026-matching/matches.parquet"

MAX_NAME_CHARS, MAX_CATEGORY_CHARS, MAX_ATTR_CHARS = 128, 64, 384
MAX_ITEM_CHARS = MAX_NAME_CHARS + MAX_CATEGORY_CHARS + MAX_ATTR_CHARS + 64  # 640

PREFIX = "paraphrase: "
ITEM_MAX_LENGTH = 128       # byte-matches berta-lora-sft-human-only-20-epochs
PAIR_MAX_LENGTH = 128       # byte-matches training -- MUST NOT change independently

CATEGORY_LABEL_PATTERN = [14948, 30]
ATTRIBUTES_LABEL_PATTERN = [4816, 37519, 521, 30]

BATCH_SIZE_START = 256      # conservative starting point for the forward-only
                             # probe -- BERTA training's own auto-probe found
                             # 512 TOTAL under DataParallel across 2 GPUs
                             # (~256/GPU actually processed per replica after
                             # DataParallel's internal split); this script has
                             # no DataParallel splitting and no classifier
                             # head in the forward pass, so 256/GPU here is
                             # already a same-or-easier footprint than what
                             # training safely ran for hours -- the probe
                             # below is free to grow past it if there's room.
BATCH_SIZE_CAP = 1024
TARGET_VRAM_GB = 14.0
SAFETY_MARGIN_GB = 3.0      # matches training script's own (widened after a
                             # real OOM there -- see its module docstring)
PROBE_ROUND_TO = 8

WORKING_DIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
SHARD_DIR = os.path.join(WORKING_DIR, "shards")
CACHE_DIR = os.path.join(WORKING_DIR, "berta_lora_pair_cache")


# =====================================================================
# SMALL UTILITIES
# =====================================================================

def _current_rss_gb():
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / 1e9
    except ImportError:
        return None


def _log(rank, stage, msg):
    print(f"[rank {rank}] {stage}: {msg}", flush=True)


def _log_memory(rank, stage):
    rss_gb = _current_rss_gb()
    if rss_gb is not None:
        _log(rank, "MEM", f"{stage}: rss={rss_gb:.2f}GB")


def _find_dataset_file(hint, filename):
    if os.path.isfile(hint):
        return hint
    if os.path.isdir("/kaggle/input"):
        for root, _dirs, files in os.walk("/kaggle/input"):
            if filename in files:
                return os.path.join(root, filename)
    return hint


def _find_adapter_dir():
    """Searches /kaggle/input for a directory containing adapter_config.json
    -- expected to be the published berta-lora-sft-5epoch checkpoint's
    adapter/ subfolder. Raises if not found: this kernel has nothing useful
    to do without it (unlike a training kernel, there's no sensible
    "fresh init" fallback for a caching job)."""
    if os.path.isdir("/kaggle/input"):
        for root, _dirs, files in os.walk("/kaggle/input"):
            if "adapter_config.json" in files:
                return root
    raise RuntimeError(
        "No adapter_config.json found under /kaggle/input -- expected the "
        "berta-lora-sft-5epoch checkpoint to be mounted (dataset_sources)."
    )


# =====================================================================
# ITEM TOKENIZATION -- byte-identical to berta-lora-sft-human-only-20-epochs
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


def build_item_cache(tokenizer, rank):
    items_path = _find_dataset_file(ITEMS_HUMAN_PATH_HINT, "items_human.parquet")
    _log(rank, "CACHE", f"tokenizing items from {items_path} ...")
    items = build_item_texts(pl.scan_parquet(items_path))
    n_items = items.height
    all_ids = items.get_column("id").to_numpy().copy()
    all_texts = [PREFIX + t for t in items.get_column("text").to_list()]
    del items
    gc.collect()

    enc = tokenizer(
        all_texts, add_special_tokens=False, padding="max_length",
        truncation=True, max_length=ITEM_MAX_LENGTH, return_tensors="np",
    )
    item_input_ids = enc["input_ids"].astype(np.int32)
    item_attention_mask = enc["attention_mask"].astype(np.int8)
    del all_texts, enc
    gc.collect()

    id_to_row = {int(item_id): i for i, item_id in enumerate(all_ids)}
    assert len(id_to_row) == n_items, f"duplicate item ids: {n_items} rows but {len(id_to_row)} unique ids"

    item_lengths = item_attention_mask.sum(axis=1).astype(np.int32)

    category_label_start = _find_token_pattern_starts(item_input_ids, CATEGORY_LABEL_PATTERN)
    attr_label_start = _find_token_pattern_starts(item_input_ids, ATTRIBUTES_LABEL_PATTERN)

    _log(rank, "CACHE", f"{n_items:,} items tokenized, mean length={item_lengths.mean():.1f}")
    return item_input_ids, item_lengths, category_label_start, attr_label_start, id_to_row


# =====================================================================
# PAIR ASSEMBLY -- byte-identical to berta-lora-sft-human-only-20-epochs
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


def build_compact_ids(item_input_ids_mm, row, item_length, category_start, attr_start):
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


def make_pair_collate_fn(item_input_ids_mm, item_lengths, category_label_start,
                          attr_label_start, cls_id, sep_id, pad_id, pair_max_length):
    content_budget = pair_max_length - 3

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
            torch.from_numpy(input_ids), torch.from_numpy(attention_mask),
            torch.from_numpy(token_type_ids), torch.tensor(labels, dtype=torch.float32),
            np.asarray(categories, dtype=np.int64),
        )

    return collate_fn


def make_pair_loader(pairs_pd, id_to_row, category_to_id, item_input_ids_mm, item_lengths,
                      category_label_start, attr_label_start, cls_id, sep_id, pad_id,
                      pair_max_length, batch_size):
    row1 = pairs_pd["id1"].map(id_to_row).to_numpy().astype(np.int64)
    row2 = pairs_pd["id2"].map(id_to_row).to_numpy().astype(np.int64)
    labels = pairs_pd["label"].to_numpy().astype(np.float32)
    categories = pairs_pd["category1"].map(category_to_id).fillna(category_to_id.get("UNKNOWN", -1)).to_numpy().astype(np.int64)

    dataset = PairIndexDataset(row1, row2, labels, categories)
    collate_fn = make_pair_collate_fn(item_input_ids_mm, item_lengths, category_label_start,
                                       attr_label_start, cls_id, sep_id, pad_id, pair_max_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0,
                       collate_fn=collate_fn, pin_memory=torch.cuda.is_available())


# =====================================================================
# MODEL -- encoder + mean pooling ONLY, no classifier (caching pre-head
# representations, that's the whole point)
# =====================================================================

class BertaPoolingEncoder(nn.Module):
    """Mean pooling (mask-weighted average over last_hidden_state) --
    matches BertaCrossEncoderClassifier's own pooling exactly (BERTA's
    model card: CLS pooling replaced with mean pooling). autocast INSIDE
    forward() -- carried over from the training script even though this
    script never uses DataParallel (no threaded replicas to break here),
    kept for exact numerical parity with how the checkpoint was trained
    and evaluated."""

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, input_ids, attention_mask, token_type_ids):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask,
                                token_type_ids=token_type_ids)
            hidden = out.last_hidden_state
            mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
            summed = (hidden * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-9)
            return (summed / counts).float()


# =====================================================================
# FORWARD-ONLY VRAM PROBE (adapted from berta-lora-sft-human-only-20-epochs's
# eval-mode probe -- no backward here at all, so always train_mode=False)
# =====================================================================

def _probe_peak_gb(batch_size, pair_max_length, probe_model, device):
    dummy_input_ids = torch.randint(0, 1000, (batch_size, pair_max_length), dtype=torch.long, device=device)
    dummy_attention_mask = torch.ones((batch_size, pair_max_length), dtype=torch.long, device=device)
    dummy_token_type_ids = torch.zeros((batch_size, pair_max_length), dtype=torch.long, device=device)

    torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        probe_model(input_ids=dummy_input_ids, attention_mask=dummy_attention_mask,
                    token_type_ids=dummy_token_type_ids)
    peak_bytes = torch.cuda.max_memory_allocated(device)
    del dummy_input_ids, dummy_attention_mask, dummy_token_type_ids
    torch.cuda.empty_cache()
    return peak_bytes / (1024 ** 3)


def _is_oom(exc):
    return isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()


def run_vram_probe(model, device, rank):
    if not torch.cuda.is_available():
        return BATCH_SIZE_START

    threshold = TARGET_VRAM_GB - SAFETY_MARGIN_GB
    candidate = BATCH_SIZE_START
    last_good = None

    while candidate <= BATCH_SIZE_CAP:
        try:
            peak_gb = _probe_peak_gb(candidate, PAIR_MAX_LENGTH, model, device)
            _log(rank, "PROBE", f"batch={candidate} -> peak={peak_gb:.2f}GB")
            if peak_gb >= threshold:
                break
            last_good = candidate
            candidate *= 2
        except RuntimeError as e:
            if not _is_oom(e):
                raise
            torch.cuda.empty_cache()
            _log(rank, "PROBE", f"batch={candidate} -> OOM")
            break

    if last_good is None:
        return max(PROBE_ROUND_TO, BATCH_SIZE_START // 2)

    lo, hi = last_good, min(candidate, BATCH_SIZE_CAP)
    while hi - lo > PROBE_ROUND_TO:
        mid = ((lo + hi) // 2 // PROBE_ROUND_TO) * PROBE_ROUND_TO
        if mid <= lo:
            break
        try:
            peak_gb = _probe_peak_gb(mid, PAIR_MAX_LENGTH, model, device)
            _log(rank, "PROBE", f"batch={mid} -> peak={peak_gb:.2f}GB")
            if peak_gb >= threshold:
                hi = mid
            else:
                lo = mid
        except RuntimeError as e:
            if not _is_oom(e):
                raise
            torch.cuda.empty_cache()
            hi = mid

    _log(rank, "PROBE", f"selected batch_size={lo}")
    return lo


# =====================================================================
# PER-SPLIT CACHING (one rank's shard)
# =====================================================================

def cache_split(split_name, split_pd, rank, world_size, model, batch_size, device,
                 id_to_row, category_to_id, item_input_ids_mm, item_lengths,
                 category_label_start, attr_label_start, cls_id, sep_id, pad_id):
    n = len(split_pd)
    chunk = math.ceil(n / world_size)
    lo = rank * chunk
    hi = min(n, lo + chunk)
    shard_pd = split_pd.iloc[lo:hi].reset_index(drop=True)
    _log(rank, split_name, f"shard rows [{lo}:{hi}) of {n:,} total")

    if len(shard_pd) == 0:
        embeddings = np.empty((0, HIDDEN_DIM), dtype=np.float16)
    else:
        loader = make_pair_loader(shard_pd, id_to_row, category_to_id, item_input_ids_mm,
                                   item_lengths, category_label_start, attr_label_start,
                                   cls_id, sep_id, pad_id, PAIR_MAX_LENGTH, batch_size)
        embeddings = np.empty((len(shard_pd), HIDDEN_DIM), dtype=np.float16)
        offset = 0
        model.eval()
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"[rank {rank}] {split_name}", dynamic_ncols=True):
                input_ids, attention_mask, token_type_ids, _labels, _cats = batch
                input_ids = input_ids.to(device, non_blocking=True)
                attention_mask = attention_mask.to(device, non_blocking=True)
                token_type_ids = token_type_ids.to(device, non_blocking=True)

                pooled = model(input_ids=input_ids, attention_mask=attention_mask,
                                token_type_ids=token_type_ids)
                pooled_np = pooled.cpu().numpy().astype(np.float16)
                bsz = pooled_np.shape[0]
                embeddings[offset:offset + bsz] = pooled_np
                offset += bsz

    split_shard_dir = os.path.join(SHARD_DIR, split_name)
    os.makedirs(split_shard_dir, exist_ok=True)
    np.save(os.path.join(split_shard_dir, f"embeddings_rank{rank}.npy"), embeddings)
    np.save(os.path.join(split_shard_dir, f"id1_rank{rank}.npy"), shard_pd["id1"].to_numpy())
    np.save(os.path.join(split_shard_dir, f"id2_rank{rank}.npy"), shard_pd["id2"].to_numpy())
    np.save(os.path.join(split_shard_dir, f"label_rank{rank}.npy"), shard_pd["label"].to_numpy().astype(np.float32))
    np.save(os.path.join(split_shard_dir, f"category_id_rank{rank}.npy"),
            shard_pd["category1"].map(category_to_id).to_numpy().astype(np.int16))
    _log(rank, split_name, f"shard saved: {embeddings.shape}")


def merge_split(split_name, world_size, rank):
    """rank-0-only: concatenate per-rank shards (in rank order, which is
    also global row order since shards were contiguous by rank) into one
    final set of arrays, then delete the shard files."""
    split_shard_dir = os.path.join(SHARD_DIR, split_name)
    parts = {"embeddings": [], "id1": [], "id2": [], "label": [], "category_id": []}
    for r in range(world_size):
        for key in parts:
            parts[key].append(np.load(os.path.join(split_shard_dir, f"{key}_rank{r}.npy")))

    out_dir = os.path.join(CACHE_DIR, split_name)
    os.makedirs(out_dir, exist_ok=True)
    merged = {key: np.concatenate(arrs, axis=0) for key, arrs in parts.items()}
    np.save(os.path.join(out_dir, "embeddings.npy"), merged["embeddings"])
    np.save(os.path.join(out_dir, "id1.npy"), merged["id1"])
    np.save(os.path.join(out_dir, "id2.npy"), merged["id2"])
    np.save(os.path.join(out_dir, "label.npy"), merged["label"])
    np.save(os.path.join(out_dir, "category_id.npy"), merged["category_id"])
    _log(0, split_name, f"merged {merged['embeddings'].shape[0]:,} rows -> {out_dir}")

    shutil.rmtree(split_shard_dir, ignore_errors=True)
    return merged["embeddings"].shape[0]


# =====================================================================
# WORKER
# =====================================================================

def worker():
    """Entry point when launched BY `torch.distributed.run` (RANK/WORLD_SIZE/
    LOCAL_RANK already set in the environment -- see main_launcher() and the
    module docstring's launch-mechanism note)."""
    PeftModel = _import_peft()

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    set_seed(SEED)

    if rank == 0:
        print("=" * 90, flush=True)
        print("CONFIG", flush=True)
        print(f"model: {MODEL_NAME} | pair_max_length: {PAIR_MAX_LENGTH} | world_size: {world_size}", flush=True)
        print("=" * 90, flush=True)
        if os.path.isdir("/kaggle/input"):
            for _root, _dirs, _files in os.walk("/kaggle/input"):
                depth = _root[len("/kaggle/input"):].count(os.sep)
                if depth <= 3:
                    print(f"[DEBUG] {_root} -> dirs={_dirs} files={_files}", flush=True)

    # Sequential load across ranks -- avoids double-materializing the same
    # checkpoint simultaneously (see module docstring).
    tokenizer = None
    model = None
    for turn in range(world_size):
        if rank == turn:
            _log(rank, "MODEL", f"loading {MODEL_NAME} + adapter ...")
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            base_model = AutoModel.from_pretrained(MODEL_NAME).to(device)
            adapter_dir = _find_adapter_dir()
            _log(rank, "MODEL", f"adapter dir: {adapter_dir}")
            peft_model = PeftModel.from_pretrained(base_model, adapter_dir, is_trainable=False)
            model = BertaPoolingEncoder(peft_model).to(device)
            model.eval()
            _log(rank, "MODEL", "loaded")
        dist.barrier()

    cls_id, sep_id, pad_id = tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id

    # Each rank independently recomputes the SAME canonical split (cheap,
    # deterministic -- same convention used everywhere else in this repo).
    matches_path = _find_dataset_file(MATCHES_PATH_HINT, "matches.parquet")
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

    if rank == 0:
        val_fingerprint = hashlib.sha256(
            ",".join(f"{a}-{b}" for a, b in sorted(zip(val_pd["id1"], val_pd["id2"]))).encode("utf-8")
        ).hexdigest()[:16]
        print(f"[DATA] train={len(train_pd):,} val={len(val_pd):,} val_fingerprint={val_fingerprint} "
              f"(expect 292,523 / 73,131)", flush=True)

    del matches_pd, labels_all, indices, train_idx, val_idx
    gc.collect()

    # Item cache -- every rank tokenizes the FULL human-scope item set (not
    # just its own shard's items) so category_vocab below is identical and
    # deterministic across ranks regardless of which pairs end up in which
    # shard.
    (item_input_ids, item_lengths, category_label_start, attr_label_start,
     id_to_row) = build_item_cache(tokenizer, rank)
    _log_memory(rank, "after item cache build")

    needed_ids = pd.unique(pd.concat([train_pd["id1"], train_pd["id2"], val_pd["id1"], val_pd["id2"]]))
    needed_ids_lazy = pl.DataFrame({"id": pd.Series(needed_ids, dtype="int64")}).lazy()
    items_path = _find_dataset_file(ITEMS_HUMAN_PATH_HINT, "items_human.parquet")
    items_cat = (
        pl.scan_parquet(items_path)
        .join(needed_ids_lazy, on="id", how="semi")
        .select(["id", "category"])
        .with_columns(pl.col("category").fill_null("UNKNOWN").cast(pl.Utf8).str.slice(0, MAX_CATEGORY_CHARS))
        .collect(engine="streaming")
    )
    category_vocab = sorted(items_cat.get_column("category").unique().to_list()) + ["UNKNOWN"]
    category_to_id = {c: i for i, c in enumerate(category_vocab)}
    id_to_category = dict(zip(items_cat.get_column("id").to_list(), items_cat.get_column("category").to_list()))
    train_pd["category1"] = train_pd["id1"].map(id_to_category).fillna("UNKNOWN")
    val_pd["category1"] = val_pd["id1"].map(id_to_category).fillna("UNKNOWN")
    del needed_ids, needed_ids_lazy, items_cat, id_to_category
    gc.collect()

    if rank == 0:
        print(f"[DATA] category vocab size: {len(category_vocab)}", flush=True)

    batch_size = run_vram_probe(model, device, rank)
    dist.barrier()

    n_train = n_val = 0
    for split_name, split_pd in [("train", train_pd), ("val", val_pd)]:
        _log_memory(rank, f"before {split_name}")
        cache_split(split_name, split_pd, rank, world_size, model, batch_size, device,
                    id_to_row, category_to_id, item_input_ids, item_lengths,
                    category_label_start, attr_label_start, cls_id, sep_id, pad_id)
        dist.barrier()
        if rank == 0:
            n = merge_split(split_name, world_size, rank)
            if split_name == "train":
                n_train = n
            else:
                n_val = n
        dist.barrier()

    if rank == 0:
        meta = {
            "source_checkpoint": "maxamsecret/berta-lora-sft-5epoch (lora_berta_sft/checkpoint/adapter)",
            "source_val_macro_pr_auc": 0.7111,
            "model": MODEL_NAME,
            "hidden_dim": HIDDEN_DIM,
            "pair_max_length": PAIR_MAX_LENGTH,
            "embedding_dtype": "float16",
            "pooling": "mean (mask-weighted), bf16 autocast",
            "split_seed": SEED,
            "n_train": int(n_train),
            "n_val": int(n_val),
            "category_vocab": category_vocab,
            "batch_size_used": batch_size,
            "world_size": world_size,
        }
        with open(os.path.join(CACHE_DIR, "cache_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        print(f"[DONE] cache_meta.json written to {CACHE_DIR}", flush=True)
        print(f"[DONE] train: {n_train:,} rows | val: {n_val:,} rows", flush=True)

    dist.barrier()
    dist.destroy_process_group()


def main_launcher():
    """Top-level kernel entry point (RANK not set in the environment yet).
    Re-invokes THIS SAME FILE under `torch.distributed.run`, which sets
    RANK/WORLD_SIZE/LOCAL_RANK and spawns one process per GPU -- each of
    those re-imports this module and, finding RANK set, calls worker()
    instead. This is the exact launch mechanism already proven on this
    Kaggle 2xT4 environment by kaggle_kernels/qwen3-bi-encoder-lora-sft (on
    the bi-encoder branch) -- deliberately reused rather than
    torch.multiprocessing.spawn, which crashed here: rank 0 died with no
    Python traceback (an OOM-kill signature -- see CLAUDE.md's own
    documented "a real Linux OOM-kill isn't a catchable MemoryError")
    somewhere inside PeftModel.from_pretrained, killing its TCPStore server
    and taking rank 1 down with a "connection closed" error. Root cause
    wasn't fully isolated before switching mechanisms -- torch.distributed.run
    is the known-safe choice already validated in this exact environment,
    not a guess."""
    # v2 crashed differently: a clean traceback this time (elastic launcher
    # surfaces it properly, unlike mp.spawn) --
    # "ImportError: cannot import name 'EmbeddingParallel' from
    # transformers.integrations.tensor_parallel", raised from inside
    # PeftModel.from_pretrained -> peft's own set_peft_model_state_dict ->
    # _maybe_shard_state_dict_for_tp. Unpinned `pip install peft` pulled
    # PyPI's latest, which has newer tensor-parallel-aware state-dict
    # loading code expecting a transformers class this Kaggle image's
    # pre-installed transformers doesn't have -- plain version skew, not a
    # launch-mechanism issue (the mp.spawn crash was almost certainly this
    # exact same ImportError, just not surfaced cleanly). Fix: pin to
    # exactly the transformers/peft/huggingface-hub triple
    # qwen3-bi-encoder-lora-sft (bi-encoder branch) already proved
    # compatible on this same Kaggle image, --no-deps --force-reinstall for
    # the same reason that notebook uses it (pip's normal resolver would
    # otherwise happily "fix" half the environment trying to satisfy these
    # three exactly).
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q", "--no-deps", "--force-reinstall",
        "huggingface-hub==0.36.0", "transformers==4.57.6", "peft==0.17.1",
    ])

    if not torch.cuda.is_available():
        raise RuntimeError("No GPU available -- this kernel needs at least 1 GPU (2 expected).")
    world_size = torch.cuda.device_count()
    print(f"[MAIN] {world_size} GPU(s) visible: "
          f"{[torch.cuda.get_device_name(i) for i in range(world_size)]}", flush=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(SHARD_DIR, exist_ok=True)

    cmd = [
        sys.executable, "-m", "torch.distributed.run",
        "--standalone", f"--nproc_per_node={world_size}",
        os.path.abspath(__file__),
    ]
    print(f"[MAIN] launching: {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"torch.distributed.run exited with code {result.returncode}")
    print("[MAIN] all ranks finished.", flush=True)


if __name__ == "__main__":
    if "RANK" in os.environ:
        worker()
    else:
        main_launcher()
