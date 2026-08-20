"""
FRIDA pair-embedding cache over the FULL human labelled set (all 365,654
pairs of matches.parquet -- train AND val), on TPU v5e-8.

WHAT THIS PRODUCES, AND WHY IT IS THE SHAPE IT IS
===========================================================================
For every human pair, one row of:
  * pooled pair representation  -- last_hidden_state[:, 0, :] (CLS position),
    the exact tensor the LoRA checkpoint's own classifier head consumes,
    BEFORE that head is applied. float32, d_model wide (read from the model
    config, not hardcoded).
  * classifier_logit            -- the checkpoint's own head applied to that
    representation, so the existing head stays reproducible from the cache
    alone and any new head can be compared against it without a second GPU
    pass.
  * label, category, split      -- so downstream head training needs nothing
    but this dataset.

`split` INHERITS THE LoRA TRAINING SPLIT (per direct instruction). It is
recomputed here with the project's canonical recipe -- filter null ids, cast
label = (target >= 0.5), then train_test_split(test_size=0.20,
random_state=42, stratify=labels) -- which is byte-identical to what every
training kernel in this project does, and meta.json records the val
fingerprint (sha256 of sorted id1-id2 pairs) so any consumer can PROVE it is
looking at the same val set the encoder was validated on. Rows are stored in
the original matches.parquet order, never shuffled, so row i means the same
pair to every consumer.

That matters because the head trained on this cache must not be trained on
pairs the encoder already fit. Everything with split==0 is contaminated for
that purpose; split==1 (73,131 pairs) is the leak-free slice. Both are cached
because the contaminated slice is still useful for measuring how much of a
gap contamination creates -- the same leak-vs-noleak A/B methodology
kaggle_kernels/berta-catboost-leak vs. -noleak already established.

WHY TPU, WHEN THE BERTA EQUIVALENT WAS GPU
===========================================================================
kaggle_kernels/berta-embed-cache-all-human does this on 1-2xT4 for a 128M
model. FRIDA is 823M and this pass is ~5x the pair count of the val-only
extraction, so the 8 TPU cores are worth the different execution model --
and TPU quota is a separate 20h/week pool that does not compete with the
training runs this cache exists to feed.

Forward-only, no backward pass, no optimizer: the memory profile is far
below any training run at the same batch size, which is why
PER_CORE_BATCH below sits well above frida-tpu-human-sft's train batch.

TPU LESSONS CARRIED FORWARD (all real, previously-hit in this project --
see kaggle_kernels/berta-llm-tpu-sft and frida-tpu-human-sft docstrings)
===========================================================================
  1. os.environ.pop("TPU_PROCESS_ADDRESSES"/"CLOUD_TPU_TASK_ID") +
     start_method="fork".
  2. STATIC SHAPES. Exactly ONE graph shape compiles for the whole run.
     Inference cannot use drop_last=True the way training does -- dropping
     the ragged tail would silently lose pairs, and this cache must cover
     every row. Instead the final batch of each replica's slice is PADDED to
     full width by repeating its last pair, and only the valid prefix is
     written back. Same static shape, zero rows lost.
  3. NO AUTO-PROBING of batch size. Every probed candidate compiles its own
     XLA program and those stay resident in HBM, which self-inflicts OOM;
     xm.get_memory_info() cannot see program memory, so it cannot even warn.
     PER_CORE_BATCH is a fixed constant with its arithmetic written out
     below. A wrong constant fails at compile time on batch 1 -- cheap.
  4. ALL CPU-only setup (parquet, item cache mmap, category join, output
     memmap creation) happens ONCE in main() BEFORE torch_xla.launch(), so
     the 8 forked replicas share it copy-on-write. Doing it after the fork
     is what caused berta-llm-tpu-sft's BrokenProcessPool.
  5. autocast opened INSIDE forward(), never around the outer call.
  6. Frozen base bf16; the classifier head stays fp32.

CRASH SAFETY
===========================================================================
Output memmaps are created up front at final size and written in place, batch
by batch, with a periodic flush plus a per-replica progress_rank{N}.json
recording rows_written / n_rows / complete. /kaggle/working survives a
hard-cap cancellation (kaggle kernels output recovers it), so a session that
dies at 80% leaves 80% of a usable cache plus an exact record of where it
stopped -- not a corrupt file of unknown completeness.

BATCH SIZE (fixed, not probed)
===========================================================================
Real data point from this project's own TPU history: FRIDA at
PAIR_MAX_LENGTH=256 trains at PER_CORE_TRAIN_BATCH=16 on a 15.75GB/core
budget. Training holds, per core: activations for backward, gradients for
every trainable parameter, and two AdamW moment tensors per trainable
parameter. This pass has none of those -- forward only, under no_grad, so
activations are freed as the forward proceeds and nothing but the frozen
weights is resident between batches. 4x the train batch is a deliberately
conservative read of that headroom (an 8x or 16x argument is defensible;
this is not the run to find the ceiling on). PER_CORE_BATCH=64, world 8 ->
512 pairs per step.
"""
import os
import sys
import gc
import json
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

from transformers import T5EncoderModel, AutoConfig, set_seed
from sklearn.model_selection import train_test_split

try:
    import peft.import_utils
    import peft.tuners.lora.torchao
    peft.import_utils.is_torchao_available = lambda: False
    peft.tuners.lora.torchao.is_torchao_available = lambda: False
except Exception as e:
    print(f"[WARN] torchao patch failed (likely fine): {e}")

from peft import PeftModel

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
ITEMS_HUMAN_PATH_HINT = "/kaggle/input/datasets/maxamsecret/ozon-ecup-2026-matching/items_human.parquet"

ITEM_CACHE_HINT = "/kaggle/input/datasets/maximmarkovbb/frida-tokenize-items-v2-cache"

# MUST match the PAIR_MAX_LENGTH the cached checkpoint was TRAINED at, or the
# embeddings describe a distribution the checkpoint never saw. Set this to the
# checkpoint's own value before pushing -- 256 for frida-tpu-human-sft's
# output, 128 for frida-lora-checkpoint-v1-best / frida-lora-v2-epoch-last.
ITEM_MAX_LENGTH = 256
PAIR_MAX_LENGTH = 256

MAX_CATEGORY_CHARS = 64

CATEGORY_LABEL_TOKEN = 55917
ATTRIBUTES_LABEL_TOKEN = 91837
LABEL_COLON_TOKEN = 30

# FIXED, NOT probed -- see module docstring's batch-size section.
PER_CORE_BATCH = 64

FLUSH_EVERY_BATCHES = 200
LOG_EVERY_BATCHES = 100

# Rank 0 waits for the other replicas' shards before writing the manifest.
SHARD_WAIT_TIMEOUT_S = 3600
SHARD_POLL_S = 10

DATASET_ID = "maximmarkovbb/frida-pair-embed-cache-human"
DATASET_TITLE = "FRIDA pair embedding cache (human, all 365654 pairs)"
ENABLE_DATASET_PUSH = True

WORKING_DIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
OUTPUT_DIR = os.path.join(WORKING_DIR, "frida_pair_embed_cache_human")

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


def _find_adapter_dir():
    if os.path.isdir("/kaggle/input"):
        for root, _dirs, files in os.walk("/kaggle/input"):
            if "adapter_config.json" in files:
                return root
    return None


# =====================================================================
# ITEM CACHE (identical contract to frida-tpu-human-sft -- the pair text a
# cached embedding describes must byte-match what the checkpoint trained on)
# =====================================================================

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
                print(f"[CACHE] hint {hint} missed -- found the v2 cache at {root}", flush=True)
                return root
    raise RuntimeError(f"[CACHE] no v2 item cache found (hint was {hint})")


def _load_base_variant(cache_root, n_items, max_length):
    variant_dir = os.path.join(cache_root, "variant_base")
    ids_path = os.path.join(variant_dir, "ids.bin")
    input_ids_path = os.path.join(variant_dir, "input_ids.bin")
    attn_path = os.path.join(variant_dir, "attention_mask.bin")

    for fname, want in (
        (ids_path, n_items * 8),
        (input_ids_path, n_items * max_length * 4),
        (attn_path, n_items * max_length),
    ):
        got = os.path.getsize(fname)
        if got != want:
            raise RuntimeError(f"[CACHE] {fname} size mismatch: got {got:,}, expected {want:,}")

    ids = np.memmap(ids_path, dtype=np.int64, mode="r", shape=(n_items,))
    input_ids = np.memmap(input_ids_path, dtype=np.int32, mode="r", shape=(n_items, max_length))
    attn = np.memmap(attn_path, dtype=np.int8, mode="r", shape=(n_items, max_length))
    lengths = np.asarray(attn).sum(axis=1).astype(np.int32)

    sample_rows = np.random.RandomState(SEED).choice(n_items, size=min(50, n_items), replace=False)
    for r in sample_rows:
        L = int(lengths[r])
        assert input_ids[r, 0] == 1, f"row {r}: expected bos=1, got {input_ids[r, 0]}"
        assert input_ids[r, L - 1] == 2, f"row {r}: expected eos=2 at {L - 1}, got {input_ids[r, L - 1]}"

    arr = np.asarray(input_ids)
    category_label_start = _find_token_pattern_starts(arr, CATEGORY_LABEL_TOKEN, LABEL_COLON_TOKEN)
    attr_label_start = _find_token_pattern_starts(arr, ATTRIBUTES_LABEL_TOKEN, LABEL_COLON_TOKEN)
    return ids, input_ids, lengths, category_label_start, attr_label_start


def derive_pad_id(input_ids, lengths, max_length):
    candidates = np.where(lengths < max_length)[0]
    if len(candidates) == 0:
        raise RuntimeError("[CACHE] every row is at max_length -- cannot derive pad_id")
    sample = candidates[:: max(1, len(candidates) // 500)][:500]
    pads = {int(input_ids[r, int(lengths[r])]) for r in sample}
    if len(pads) != 1:
        raise RuntimeError(f"[CACHE] inconsistent pad token across rows: {sorted(pads)}")
    return pads.pop()


def load_item_cache():
    """Only variant_base is loaded. The shuffled attribute-order variant is a
    TRAIN-time augmentation; a cache of representations must be deterministic,
    or the same pair would embed differently on a re-run."""
    cache_root = _find_item_cache_root(ITEM_CACHE_HINT)
    with open(os.path.join(cache_root, "cache_meta.json")) as f:
        meta = json.load(f)
    n_items, max_length = meta["n_items"], meta["max_length"]
    print(f"[CACHE] n_items={n_items:,} max_length={max_length} prefix={meta.get('prefix')!r}", flush=True)
    if max_length != ITEM_MAX_LENGTH:
        raise RuntimeError(f"cache max_length={max_length} != ITEM_MAX_LENGTH={ITEM_MAX_LENGTH}")

    t0 = time.perf_counter()
    base = _load_base_variant(cache_root, n_items, max_length)
    print(f"[CACHE] variant_base loaded + bos/eos checked ({time.perf_counter() - t0:.1f}s)", flush=True)

    pad_id = derive_pad_id(base[1], base[2], max_length)
    print(f"[CACHE] pad_id={pad_id} (derived from the cache's own padding)", flush=True)

    id_to_row = {int(item_id): i for i, item_id in enumerate(np.asarray(base[0]))}
    assert len(id_to_row) == n_items, f"duplicate item ids: {n_items} rows, {len(id_to_row)} unique"
    return base, id_to_row, pad_id


# =====================================================================
# PAIR ASSEMBLY (byte-identical to the training kernels)
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
    """Yields the GLOBAL row index alongside the pair, so a replica can write
    each result straight back to its own slot in the output memmap without any
    ordering assumption about how batches were formed."""

    def __init__(self, global_idx, row1, row2):
        self.global_idx = global_idx
        self.row1 = row1
        self.row2 = row2

    def __len__(self):
        return len(self.global_idx)

    def __getitem__(self, i):
        return self.global_idx[i], self.row1[i], self.row2[i]


def make_collate_fn(base, pad_id, pair_max_length):
    def pick(row):
        return build_compact_ids(base[1], row, int(base[2][row]), int(base[3][row]), int(base[4][row]))

    def collate_fn(batch):
        gidx, row1, row2 = zip(*batch)
        bsz = len(batch)
        input_ids = np.full((bsz, pair_max_length), pad_id, dtype=np.int64)
        attention_mask = np.zeros((bsz, pair_max_length), dtype=np.int64)
        for i in range(bsz):
            c1, c2 = pick(row1[i]), pick(row2[i])
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
                np.asarray(gidx, dtype=np.int64))
    return collate_fn


# =====================================================================
# MODEL -- encoder + the checkpoint's own head, both needed
# =====================================================================

class FridaPairEncoder(nn.Module):
    def __init__(self, encoder, head):
        super().__init__()
        self.encoder = encoder
        self.classifier = head

    def forward(self, input_ids, attention_mask):
        with torch.autocast("xla", dtype=torch.bfloat16):
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.last_hidden_state[:, 0, :].float()
        logit = self.classifier(pooled).view(-1).float()
        return pooled, logit


def build_model(device, rank, adapter_dir, hidden_dim):
    try:
        base_model = T5EncoderModel.from_pretrained(MODEL_NAME, dtype=torch.bfloat16,
                                                     attn_implementation="eager")
    except (TypeError, ValueError):
        base_model = T5EncoderModel.from_pretrained(MODEL_NAME, dtype=torch.bfloat16)

    # is_trainable stays False on purpose: this is a pure inference pass, and
    # the frozen-adapter default is exactly what we want here. (The opposite
    # trap -- silently frozen LoRA during TRAINING -- is why every training
    # kernel in this project passes is_trainable=True explicitly.)
    #
    # Deliberately NOT merge_and_unload(): merging folds fp32 LoRA deltas into
    # bf16 base weights, which perturbs the output slightly. Training kept the
    # adapter separate with its A/B matrices in fp32, so keeping it separate
    # here is what makes the cached representation the SAME tensor the model
    # produced during training -- and makes classifier_logit reproduce the
    # checkpoint's own val score exactly rather than approximately.
    peft_model = PeftModel.from_pretrained(base_model, adapter_dir)
    for name, p in peft_model.named_parameters():
        if "lora_" in name:
            p.data = p.data.float()

    head = nn.Linear(hidden_dim, 1)
    head.load_state_dict(torch.load(os.path.join(adapter_dir, "classifier.pt"), map_location="cpu"))
    _log(rank, "MODEL", f"adapter merged + classifier loaded from {adapter_dir}", force=True)

    model = FridaPairEncoder(peft_model, head).to(device)
    model.eval()
    return model


# =====================================================================
# SHARED CPU-ONLY SETUP -- runs ONCE, pre-fork
# =====================================================================

_SHARED = {}


def _load_shared_data():
    print(f"[{_elapsed():8.1f}s][SETUP] loading all CPU-only data once, pre-fork ...", flush=True)

    matches_path = _find_dataset_file(MATCHES_PATH_HINT, "matches.parquet")
    matches_pd = (
        pl.read_parquet(matches_path)
        .filter(pl.col("id1").is_not_null() & pl.col("id2").is_not_null())
        .with_columns([pl.col("target").fill_null(0).cast(pl.Float32),
                       (pl.col("target") >= 0.5).cast(pl.Int8).alias("label")])
        .select(["id1", "id2", "label"])
        .to_pandas()
    )
    n_rows = len(matches_pd)

    # Canonical seed-42 split, recomputed exactly as every training kernel
    # does it -- this is what "inherit the LoRA train/eval split" means.
    labels_all = matches_pd["label"].to_numpy().astype(np.int64)
    indices = np.arange(n_rows, dtype=np.int64)
    train_idx, val_idx = train_test_split(indices, test_size=0.20, random_state=SEED,
                                           stratify=labels_all)
    split = np.zeros(n_rows, dtype=np.int8)
    split[val_idx] = 1
    matches_pd["split"] = split

    val_pairs = matches_pd.iloc[np.sort(val_idx)]
    val_fingerprint = hashlib.sha256(
        ",".join(f"{a}-{b}" for a, b in sorted(zip(val_pairs["id1"], val_pairs["id2"]))).encode("utf-8")
    ).hexdigest()[:16]
    print(f"[{_elapsed():8.1f}s][SETUP] pairs={n_rows:,} train={int((split == 0).sum()):,} "
          f"val={int((split == 1).sum()):,} val_fingerprint={val_fingerprint}", flush=True)
    del labels_all, indices, train_idx, val_idx
    gc.collect()

    base, id_to_row, pad_id = load_item_cache()

    items_human_path = _find_dataset_file(ITEMS_HUMAN_PATH_HINT, "items_human.parquet")
    needed_ids = pd.unique(matches_pd["id1"])
    items_cat = (
        pl.scan_parquet(items_human_path)
        .join(pl.DataFrame({"id": pd.Series(needed_ids, dtype="int64")}).lazy(), on="id", how="semi")
        .select(["id", "category"])
        .with_columns(pl.col("category").fill_null("UNKNOWN").cast(pl.Utf8).str.slice(0, MAX_CATEGORY_CHARS))
        .collect(engine="streaming")
    )
    category_vocab = sorted(items_cat.get_column("category").unique().to_list()) + ["UNKNOWN"]
    category_to_id = {c: i for i, c in enumerate(category_vocab)}
    id_to_category = dict(zip(items_cat.get_column("id").to_list(),
                              items_cat.get_column("category").to_list()))
    matches_pd["category1"] = matches_pd["id1"].map(id_to_category).fillna("UNKNOWN")
    del items_cat, needed_ids, id_to_category
    gc.collect()

    # Every pair must resolve to a cached item row, or the cache would silently
    # carry garbage embeddings. Fail here, at second ~60, not at 80% of the run.
    row1 = matches_pd["id1"].map(id_to_row)
    row2 = matches_pd["id2"].map(id_to_row)
    missing = int(row1.isna().sum() + row2.isna().sum())
    if missing:
        raise RuntimeError(f"[SETUP] {missing:,} pair endpoints are absent from the item cache -- "
                           f"wrong cache scope for matches.parquet?")

    hidden_dim = int(AutoConfig.from_pretrained(MODEL_NAME).d_model)
    print(f"[{_elapsed():8.1f}s][SETUP] hidden_dim={hidden_dim} (from model config)", flush=True)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    emb_path = os.path.join(OUTPUT_DIR, "pair_embedding.bin")
    logit_path = os.path.join(OUTPUT_DIR, "classifier_logit.bin")
    written_path = os.path.join(OUTPUT_DIR, "written.bin")

    need_gb = (n_rows * hidden_dim * 4 + n_rows * 4 + n_rows) / 1e9
    free_gb = os.statvfs(WORKING_DIR).f_bavail * os.statvfs(WORKING_DIR).f_frsize / 1e9
    print(f"[{_elapsed():8.1f}s][SETUP] output needs {need_gb:.2f}GB, free {free_gb:.2f}GB", flush=True)
    if need_gb * 1.15 > free_gb:
        raise RuntimeError(f"[SETUP] not enough disk: need {need_gb * 1.15:.2f}GB with margin, "
                           f"have {free_gb:.2f}GB")

    # Created at final size pre-fork; replicas reopen r+ and write disjoint
    # slices. `written` is a per-row flag so completeness is verifiable after
    # a crash rather than inferred from a counter.
    np.memmap(emb_path, dtype=np.float32, mode="w+", shape=(n_rows, hidden_dim)).flush()
    np.memmap(logit_path, dtype=np.float32, mode="w+", shape=(n_rows,)).flush()
    np.memmap(written_path, dtype=np.int8, mode="w+", shape=(n_rows,)).flush()

    matches_pd.assign(row_index=np.arange(n_rows, dtype=np.int64)).to_parquet(
        os.path.join(OUTPUT_DIR, "pairs.parquet"), index=False)

    _SHARED.update(dict(
        n_rows=n_rows, hidden_dim=hidden_dim, pad_id=pad_id, base=base,
        row1=row1.to_numpy().astype(np.int64), row2=row2.to_numpy().astype(np.int64),
        val_fingerprint=val_fingerprint, category_vocab=category_vocab,
        category_to_id=category_to_id,
        paths=dict(emb=emb_path, logit=logit_path, written=written_path),
    ))
    print(f"[{_elapsed():8.1f}s][SETUP] done -- output memmaps allocated, ready for fork", flush=True)


# =====================================================================
# PER-REPLICA ENTRYPOINT
# =====================================================================

def _mp_fn(index):
    rank = xr.global_ordinal()
    world_size = xr.world_size()
    device = torch_xla.device()
    set_seed(SEED + rank)

    n_rows = _SHARED["n_rows"]
    hidden_dim = _SHARED["hidden_dim"]
    paths = _SHARED["paths"]

    # Contiguous slice per replica: disjoint memmap regions, no coordination
    # needed while writing, and a crash leaves an interpretable partial file.
    per_rank = (n_rows + world_size - 1) // world_size
    start = rank * per_rank
    end = min(start + per_rank, n_rows)
    _log(rank, "SHARD", f"rows [{start:,}, {end:,}) of {n_rows:,}", force=True)

    if start >= end:
        _write_progress(rank, 0, 0, True)
        return

    adapter_dir = _find_adapter_dir()
    if adapter_dir is None:
        raise RuntimeError("No adapter found under /kaggle/input -- mount a FRIDA LoRA "
                           "checkpoint via dataset_sources.")
    model = build_model(device, rank, adapter_dir, hidden_dim)

    gidx = np.arange(start, end, dtype=np.int64)
    dataset = PairIndexDataset(gidx, _SHARED["row1"][start:end], _SHARED["row2"][start:end])
    collate_fn = make_collate_fn(_SHARED["base"], _SHARED["pad_id"], PAIR_MAX_LENGTH)

    n_shard = end - start
    n_full = (n_shard // PER_CORE_BATCH) * PER_CORE_BATCH
    full_loader = DataLoader(torch.utils.data.Subset(dataset, list(range(n_full))),
                             batch_size=PER_CORE_BATCH, shuffle=False, drop_last=True,
                             collate_fn=collate_fn, num_workers=2)
    tail = list(range(n_full, n_shard))
    _log(rank, "LOADER", f"{n_full // PER_CORE_BATCH} full batches + tail of {len(tail)}", force=True)

    emb_mm = np.memmap(paths["emb"], dtype=np.float32, mode="r+", shape=(n_rows, hidden_dim))
    logit_mm = np.memmap(paths["logit"], dtype=np.float32, mode="r+", shape=(n_rows,))
    written_mm = np.memmap(paths["written"], dtype=np.int8, mode="r+", shape=(n_rows,))

    rows_done = 0
    t0 = time.perf_counter()

    def run_batch(input_ids, attention_mask, batch_gidx, n_valid):
        nonlocal rows_done
        with torch.no_grad():
            pooled, logit = model(input_ids=input_ids, attention_mask=attention_mask)
        pooled = pooled.cpu().numpy()[:n_valid]
        logit = logit.cpu().numpy()[:n_valid]
        target_rows = batch_gidx[:n_valid]
        emb_mm[target_rows] = pooled
        logit_mm[target_rows] = logit
        written_mm[target_rows] = 1
        rows_done += n_valid

    mp_loader = pl_xla.MpDeviceLoader(full_loader, device)
    for bi, (input_ids, attention_mask, batch_gidx) in enumerate(mp_loader, start=1):
        run_batch(input_ids, attention_mask, batch_gidx, PER_CORE_BATCH)
        if bi % FLUSH_EVERY_BATCHES == 0:
            emb_mm.flush(); logit_mm.flush(); written_mm.flush()
            _write_progress(rank, rows_done, n_shard, False)
        if bi % LOG_EVERY_BATCHES == 0:
            rate = rows_done / max(time.perf_counter() - t0, 1e-9)
            _log(rank, "PROGRESS", f"{rows_done:,}/{n_shard:,} rows ({rate:.0f} rows/s)", force=True)

    # Ragged tail: padded up to the SAME static shape by repeating the last
    # pair, then only the valid prefix is written back. Keeps exactly one
    # compiled graph while losing zero rows -- see docstring lesson 2.
    if tail:
        pad_to = PER_CORE_BATCH
        padded = tail + [tail[-1]] * (pad_to - len(tail))
        batch = collate_fn([dataset[i] for i in padded])
        input_ids, attention_mask, batch_gidx = batch
        run_batch(input_ids.to(device), attention_mask.to(device), batch_gidx, len(tail))
        _log(rank, "TAIL", f"{len(tail)} rows written from a batch padded to {pad_to}", force=True)

    emb_mm.flush(); logit_mm.flush(); written_mm.flush()
    del emb_mm, logit_mm, written_mm
    gc.collect()

    _write_progress(rank, rows_done, n_shard, rows_done == n_shard)
    _log(rank, "SHARD", f"done: {rows_done:,}/{n_shard:,} rows in "
                        f"{time.perf_counter() - t0:.0f}s", force=True)

    if rank == 0:
        _finalize(world_size, n_rows, hidden_dim)


def _write_progress(rank, rows_written, n_shard, complete):
    with open(os.path.join(OUTPUT_DIR, f"progress_rank{rank}.json"), "w") as f:
        json.dump({"rank": rank, "rows_written": int(rows_written), "n_shard": int(n_shard),
                   "complete": bool(complete)}, f, indent=2)


def _finalize(world_size, n_rows, hidden_dim):
    """Rank 0 polls the other replicas' progress files rather than using a
    collective barrier: a stalled shard then shows up as an observable,
    debuggable timeout instead of hanging the XLA runtime, and the partial
    output stays recoverable either way."""
    deadline = time.perf_counter() + SHARD_WAIT_TIMEOUT_S
    while time.perf_counter() < deadline:
        states = []
        for r in range(world_size):
            p = os.path.join(OUTPUT_DIR, f"progress_rank{r}.json")
            if not os.path.exists(p):
                states.append(None)
                continue
            with open(p) as f:
                states.append(json.load(f))
        if all(s is not None and s["complete"] for s in states):
            break
        done = sum(s["rows_written"] for s in states if s)
        print(f"[{_elapsed():8.1f}s][FINALIZE] waiting: {done:,}/{n_rows:,} rows across "
              f"{sum(1 for s in states if s)}/{world_size} shards", flush=True)
        time.sleep(SHARD_POLL_S)

    written = np.memmap(os.path.join(OUTPUT_DIR, "written.bin"), dtype=np.int8, mode="r",
                        shape=(n_rows,))
    n_written = int(np.asarray(written).sum())
    complete = n_written == n_rows
    print(f"[{_elapsed():8.1f}s][FINALIZE] {n_written:,}/{n_rows:,} rows written "
          f"(complete={complete})", flush=True)

    meta = {
        "model": MODEL_NAME,
        "adapter_dir": _find_adapter_dir(),
        "n_rows": int(n_rows),
        "n_rows_written": n_written,
        "complete": bool(complete),
        "hidden_dim": int(hidden_dim),
        "item_max_length": ITEM_MAX_LENGTH,
        "pair_max_length": PAIR_MAX_LENGTH,
        "item_cache": ITEM_CACHE_HINT,
        "variant": "variant_base (deterministic, no attribute shuffle)",
        "split": "canonical seed-42, test_size=0.20, stratified -- 0=train, 1=val",
        "val_fingerprint": _SHARED["val_fingerprint"],
        "category_vocab": _SHARED["category_vocab"],
        "files": {
            "pair_embedding.bin": f"float32 ({n_rows}, {hidden_dim}) -- CLS pooled, pre-head",
            "classifier_logit.bin": f"float32 ({n_rows},) -- checkpoint's own head, raw logit",
            "written.bin": f"int8 ({n_rows},) -- 1 = row computed",
            "pairs.parquet": "id1, id2, label, split, category1, row_index",
        },
    }
    with open(os.path.join(OUTPUT_DIR, "cache_meta.json"), "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"[{_elapsed():8.1f}s][FINALIZE] cache_meta.json written", flush=True)

    if ENABLE_DATASET_PUSH and complete:
        _publish_dataset()
    elif ENABLE_DATASET_PUSH:
        print("[FINALIZE] incomplete cache -- NOT publishing. Recover with "
              "`kaggle kernels output` and inspect written.bin.", flush=True)


def _publish_dataset():
    # `licenses` is required by the CLI and its absence fails AFTER all the
    # work is done -- a real incident in this project (see the kaggle-
    # experiment-safety skill). `kaggle` is not on PATH on the TPU image.
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "kaggle"])
        with open(os.path.join(OUTPUT_DIR, "dataset-metadata.json"), "w") as f:
            json.dump({"id": DATASET_ID, "title": DATASET_TITLE,
                       "licenses": [{"name": "CC0-1.0"}]}, f, indent=2)
        subprocess.check_call([sys.executable, "-m", "kaggle", "datasets", "create",
                               "-p", OUTPUT_DIR, "-r", "skip"])
        print("[PUBLISH] dataset created", flush=True)
    except Exception as e:
        print(f"[PUBLISH] FAILED ({type(e).__name__}: {e}) -- output is still in "
              f"/kaggle/working, recover with `kaggle kernels output`", flush=True)


def main():
    os.environ.pop("TPU_PROCESS_ADDRESSES", None)
    os.environ.pop("CLOUD_TPU_TASK_ID", None)

    _load_shared_data()

    torch_xla.launch(_mp_fn, args=(), start_method="fork")


if __name__ == "__main__":
    main()
