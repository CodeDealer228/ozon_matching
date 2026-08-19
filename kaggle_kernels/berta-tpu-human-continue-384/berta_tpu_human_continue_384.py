"""
Continue-train the TOP BERTA LoRA cross-encoder checkpoint (the one that
actually shipped as submits/matching-berta_tpu_lora-submit -- test 0.47,
val_macro_pr_auc=0.7087279672868713 at 50% of one epoch on matches_llm.parquet
+human, see reports/berta_tpu_lora_provenance/) further on the HUMAN-ONLY
canonical split (matches.parquet, no matches_llm.parquet this time), at a
raised PAIR_MAX_LENGTH=384, for 10 real epochs, with validation+checkpoint
after EVERY epoch.

Resume checkpoint: maxamsecret/berta-tpu-lora-top-checkpoint (published this
session specifically for this kernel -- adapter_config.json/
adapter_model.safetensors/classifier.pt, sha256-verified identical to what's
actually inside the shipped zip; NO optimizer.pt was recoverable from the
original crashed run, so the optimizer starts fresh here -- see "Resume"
section below).

Base encoder: same merged base as the original run (maxamsecret/
berta-merge-lora's merged_encoder/, bf16) -- unchanged, only the LoRA
adapter/classifier are new starting points.

Item cache: reuses maxamsecret/berta-tokenize-llm-256's human_scope
(711,304 items, ITEM_MAX_LENGTH=256) -- the only human-scope cache that
exists. IMPORTANT CAVEAT, stated up front rather than buried: raising
PAIR_MAX_LENGTH to 384 while the item cache itself is still capped at 256
tokens/item only helps pairs where BOTH items are short enough to combine
past 256 total -- no single item can ever contribute more than 256 tokens
regardless of how much pair budget is available, since that's the ceiling
of what was ever tokenized for it. Re-tokenizing the item cache at a higher
ITEM_MAX_LENGTH was not requested and is out of scope here; this is a
real, known limitation of this specific run, not a silent gap.

Six lessons carried forward from this project's TPU history (all directly
inherited from kaggle_kernels/berta-llm-tpu-sft, not rediscovered):
  1. Kaggle env-var pop + start_method="fork" (TPU_PROCESS_ADDRESSES/
     CLOUD_TPU_TASK_ID conflict with xmp.spawn's own topology setup).
  2. FIXED per-core batch sizes, no auto-probing -- probing itself pins
     compiled XLA programs in HBM and self-inflicts the OOM it's meant to
     avoid; xm.get_memory_info() can't even see that memory to warn about it.
  3. autocast OPENED INSIDE forward(), never around a DataParallel-style
     outer call (irrelevant here -- TPU replicas are separate processes, not
     threads -- but the frozen-bf16-base + fp32-LoRA/classifier precision
     split itself still applies).
  4. Validation redundant-but-simple across all 8 replicas (no all_gather),
     only rank 0 logs/checkpoints.
  5. All CPU-only setup (parquet loading, item cache load, category join)
     happens ONCE in main() BEFORE torch_xla.launch() forks the 8 replicas
     -- the real fix (this session, see kaggle_kernels/README.md) for a
     genuine BrokenProcessPool crash that turned out to be 8x redundant
     per-replica RAM usage, not a training bug.
  6. xm.save()/explicit .cpu() moves for checkpointing, never model.to('cpu')
     in place on the live training model.

Two things this script does DIFFERENTLY from berta-llm-tpu-sft, per direct
instruction (2026-08-20):
  A. Every epoch gets its OWN checkpoint directory
     (checkpoint_epoch{01..10}/), never overwritten -- berta-llm-tpu-sft's
     save_checkpoint() unconditionally overwrote the SAME directory every
     interval with whatever the CURRENT live weights were, while
     train_state.json kept reporting the historical best METRIC (achieved
     at a possibly-different, now-overwritten step). Concretely caught this
     session: the checkpoint at 100% of that run's epoch had REAL per-rank
     eval around 0.696-0.699, while its train_state.json still claimed the
     50%-checkpoint's stale 0.7087 -- see reports/berta_tpu_lora_provenance/.
     Separate directories make this class of confusion structurally
     impossible: every saved checkpoint's train_state.json describes
     exactly the weights sitting next to it, nothing else.
  B. Any checkpoint saved WITHOUT running eval (the wall-clock safety
     fallback below) writes `"val_macro_pr_auc": null` explicitly, rather
     than omitting the field or carrying forward a stale number -- so
     "not measured" is never confused with "measured and this is the value".
"""
import os
import sys
import gc
import json
import math
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

from peft import PeftModel, get_peft_model_state_dict
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

MATCHES_PATH_HINT = "/kaggle/input/datasets/maxamsecret/ozon-ecup-2026-matching/matches.parquet"
ITEMS_HUMAN_PATH_HINT = "/kaggle/input/datasets/maxamsecret/ozon-ecup-2026-matching/items_human.parquet"

PREFIX = "paraphrase: "
ITEM_MAX_LENGTH = 256   # human_scope cache's own length -- see module docstring caveat
MAX_CATEGORY_CHARS = 64
PAIR_MAX_LENGTH = 384   # raised from 256, per instruction (2026-08-20)

CLASSIFIER_DROPOUT = 0.10

LR = 2e-4                # unchanged from the original recipe this resumes
WEIGHT_DECAY = 0.04
WARMUP_RATIO = 0.10

CATEGORY_LABEL_PATTERN = [14948, 30]
ATTRIBUTES_LABEL_PATTERN = [4816, 37519, 521, 30]

# FIXED per-core batch sizes -- NO auto-probing (see module docstring
# lesson 2). batch=64 @ PAIR_MAX_LENGTH=256 was proven safe by the run this
# resumes; 384 is 1.5x longer, kept at the same batch as a conservative
# starting point (BERT-scale attention over 384 tokens is still cheap in
# absolute HBM terms for a 128M model) -- a real OOM here is a fast, loud
# signal to lower this constant, not a silently-corrupted probe result.
PER_CORE_TRAIN_BATCH = 64
PER_CORE_EVAL_BATCH = 96   # slightly lower than the 128 used at 256 -- eval
                             # is forward-only (no gradients/optimizer state)
                             # but the longer sequence still costs more
                             # activation memory per example.

EPOCHS = 10
MAX_TRAIN_SECONDS = 30600   # 8.5h safety cap -- TPU session hard limit is 9h
                              # (CLAUDE.md); human-only data is ~40x smaller
                              # than the LLM-mixed run this resumes (292,523
                              # vs 11,480,303 rows), so all 10 epochs are
                              # expected to fit comfortably inside one
                              # session -- this cap is a safety net, not the
                              # expected stopping condition.
QUIET_SAVE_EVERY_SECONDS = 1800   # periodic weights-only safety save,
                                     # independent of the epoch-end trigger
                                     # (see module docstring point B).
LOG_EVERY_STEPS = 50

WORKING_DIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
OUTPUT_DIR = os.path.join(WORKING_DIR, "lora_berta_human_continue_384")

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


def _find_dir_with_file(filename, near=None):
    if os.path.isdir("/kaggle/input"):
        for root, _dirs, files in os.walk("/kaggle/input"):
            if filename in files and (near is None or near in root):
                return root
    return None


def _find_resume_adapter_dir():
    """Searches by CONTENT (adapter_config.json presence), not by dataset
    name -- matches this project's established convention so a future
    checkpoint swap only needs kernel-metadata.json's dataset_sources
    updated, not this function."""
    if os.path.isdir("/kaggle/input"):
        for root, _dirs, files in os.walk("/kaggle/input"):
            if "adapter_config.json" in files:
                return root
    return None


def _find_merged_base_dir():
    if os.path.isdir("/kaggle/input"):
        for root, _dirs, files in os.walk("/kaggle/input"):
            if "merge_meta.json" in files:
                return root
    raise RuntimeError("Merged BERTA base not found under /kaggle/input -- expected "
                        "maxamsecret/berta-merge-lora mounted (kernel_sources).")


# =====================================================================
# ITEM CACHE (human_scope only -- see module docstring)
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


def load_scope_cache(scope_dir):
    with open(os.path.join(scope_dir, "cache_meta.json")) as f:
        meta = json.load(f)
    n_items = meta["n_items"]
    max_length = meta["max_length"]
    assert max_length == ITEM_MAX_LENGTH, f"{scope_dir}: max_length={max_length} != {ITEM_MAX_LENGTH}"
    assert meta["prefix"] == PREFIX, f"{scope_dir}: prefix={meta['prefix']!r} != {PREFIX!r}"

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

    print(f"[{_elapsed():8.1f}s][SETUP] {scope_dir}: {n_items:,} items loaded", flush=True)
    return ids, input_ids, item_lengths, category_label_start, attr_label_start, category_vocab


# =====================================================================
# PAIR ASSEMBLY
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
    # persistent_workers=False deliberately -- this loader is rebuilt fresh
    # every epoch (new DistributedSampler each time), so persistent workers
    # provide no benefit and only risk a transient double-worker overlap at
    # the epoch boundary (see kaggle_kernels/berta-llm-tpu-sft's own fix).
    loader = DataLoader(dataset, batch_size=per_core_batch, sampler=sampler, drop_last=True,
                         collate_fn=collate_fn, num_workers=2, persistent_workers=False)
    return loader, sampler


def make_eval_loader(pairs_pd, id_to_row, category_to_id, item_input_ids, item_lengths,
                      category_label_start, attr_label_start, cls_id, sep_id, pad_id,
                      pair_max_length, eval_batch):
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


def build_resumed_model(device, rank, merged_dir, resume_dir):
    encoder_dir = os.path.join(merged_dir, "merged_encoder")
    base_model = AutoModel.from_pretrained(encoder_dir, dtype=torch.bfloat16)
    hidden_dim = base_model.config.hidden_size

    peft_model = PeftModel.from_pretrained(base_model, resume_dir, is_trainable=True)
    if rank == 0:
        peft_model.print_trainable_parameters()

    n_trainable_lora = sum(p.numel() for n, p in peft_model.named_parameters() if "lora_" in n and p.requires_grad)
    assert n_trainable_lora > 0, "Resumed LoRA adapter has zero trainable params -- is_trainable didn't take."

    # LoRA A/B kept fp32 for optimizer stability, matching every other
    # script in this project's LoRA lineage.
    for name, p in peft_model.named_parameters():
        if "lora_" in name:
            p.data = p.data.float()

    classifier_head = ClassifierHead(hidden_dim, CLASSIFIER_DROPOUT)
    classifier_path = os.path.join(resume_dir, "classifier.pt")
    raw_state = torch.load(classifier_path, map_location="cpu")
    if "linear.weight" in raw_state:
        classifier_head.load_state_dict(raw_state)
    else:
        classifier_head.linear.load_state_dict(raw_state)
    _log(rank, "MODEL", f"classifier WARM-STARTED (continued) from {resume_dir}/classifier.pt", force=True)

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
# CHECKPOINT -- per module docstring point A: every checkpoint gets its
# OWN directory, never overwritten. Point B: val_macro_pr_auc is ALWAYS an
# explicit field -- null when this save skipped eval, a real float when it
# didn't. No other script in this project's TPU lineage does this; it is
# the direct fix for the stale-metric confusion found in berta_tpu_lora's
# provenance trace (reports/berta_tpu_lora_provenance/).
# =====================================================================

def save_checkpoint(model, optimizer, epoch, global_step, val_macro_pr_auc, label, rank):
    if rank != 0:
        return
    save_dir = os.path.join(OUTPUT_DIR, label)
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
        json.dump({
            "epoch": epoch,
            "global_step": global_step,
            # Explicit null (valid JSON, unambiguous "not measured") rather
            # than a stale carried-over number or an omitted field -- see
            # module docstring point B.
            "val_macro_pr_auc": val_macro_pr_auc if val_macro_pr_auc is not None else None,
        }, f, indent=2)
    print(f"[SAVE] checkpoint written to {save_dir} (epoch={epoch}, global_step={global_step}, "
          f"val_macro_pr_auc={val_macro_pr_auc})", flush=True)


# =====================================================================
# SHARED CPU-ONLY SETUP -- loaded ONCE, pre-fork (see module docstring
# lesson 5 / kaggle_kernels/berta-llm-tpu-sft's own RAM-exhaustion fix).
# =====================================================================

_SHARED = {}


def _load_shared_data():
    print(f"[{_elapsed():8.1f}s][SETUP] loading all CPU-only data once, pre-fork ...", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    cls_id, sep_id, pad_id = tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id

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
    train_pd = matches_pd.iloc[train_idx][["id1", "id2", "label"]].reset_index(drop=True).copy()
    val_pd = matches_pd.iloc[val_idx][["id1", "id2", "label"]].reset_index(drop=True).copy()
    train_pd["label"] = train_pd["label"].astype(np.float32)
    val_pd["label"] = val_pd["label"].astype(np.float32)
    val_fingerprint = hashlib.sha256(
        ",".join(f"{a}-{b}" for a, b in sorted(zip(val_pd["id1"], val_pd["id2"]))).encode("utf-8")
    ).hexdigest()[:16]
    print(f"[{_elapsed():8.1f}s][SETUP] human train={len(train_pd):,} val={len(val_pd):,} "
          f"val_fingerprint={val_fingerprint}", flush=True)
    del matches_pd, labels_all, indices, train_idx, val_idx
    gc.collect()

    human_scope_dir = _find_dir_with_file("cache_meta.json", near="human_scope")
    if human_scope_dir is None:
        raise RuntimeError("human_scope item cache not found under /kaggle/input -- expected "
                            "maxamsecret/berta-tokenize-llm-256 mounted (kernel_sources).")
    (ids, item_input_ids, item_lengths, category_label_start, attr_label_start,
     category_vocab) = load_scope_cache(human_scope_dir)
    id_to_row = {int(i): row for row, i in enumerate(ids.tolist())}
    category_to_id = {c: i for i, c in enumerate(category_vocab)}

    items_human_path = _find_dataset_file(ITEMS_HUMAN_PATH_HINT, "items_human.parquet")
    needed_ids = pd.unique(pd.concat([train_pd["id1"], val_pd["id1"]]))
    needed_ids_lazy = pl.DataFrame({"id": pd.Series(needed_ids, dtype="int64")}).lazy()
    items_cat = (
        pl.scan_parquet(items_human_path).join(needed_ids_lazy, on="id", how="semi")
        .select(["id", "category"])
        .with_columns(pl.col("category").fill_null("UNKNOWN").cast(pl.Utf8).str.slice(0, MAX_CATEGORY_CHARS))
        .collect(engine="streaming")
    )
    id_to_category = dict(zip(items_cat.get_column("id").to_list(), items_cat.get_column("category").to_list()))
    train_pd["category1"] = train_pd["id1"].map(id_to_category).fillna("UNKNOWN")
    val_pd["category1"] = val_pd["id1"].map(id_to_category).fillna("UNKNOWN")
    del items_cat, needed_ids, needed_ids_lazy, id_to_category
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

    _log(rank, "CONFIG", f"world_size={world_size} device={device} pair_max_length={PAIR_MAX_LENGTH} "
                        f"epochs={EPOCHS}", force=True)
    if rank == 0 and os.path.isdir("/kaggle/input"):
        for _root, _dirs, _files in os.walk("/kaggle/input"):
            depth = _root[len("/kaggle/input"):].count(os.sep)
            if depth <= 3:
                print(f"[DEBUG] {_root} -> dirs={_dirs} files={_files}")

    cls_id, sep_id, pad_id = _SHARED["cls_id"], _SHARED["sep_id"], _SHARED["pad_id"]
    train_pd, val_pd, val_fingerprint = _SHARED["train_pd"], _SHARED["val_pd"], _SHARED["val_fingerprint"]
    item_input_ids, item_lengths = _SHARED["item_input_ids"], _SHARED["item_lengths"]
    category_label_start, attr_label_start = _SHARED["category_label_start"], _SHARED["attr_label_start"]
    id_to_row = _SHARED["id_to_row"]
    category_vocab, category_to_id = _SHARED["category_vocab"], _SHARED["category_to_id"]
    _log(rank, "DATA", f"inherited from parent (fork/COW): train={len(train_pd):,} val={len(val_pd):,} "
                        f"val_fingerprint={val_fingerprint}", force=True)

    merged_dir = _find_merged_base_dir()
    resume_dir = _find_resume_adapter_dir()
    if resume_dir is None:
        raise RuntimeError("No resume adapter found under /kaggle/input -- expected "
                            "maxamsecret/berta-tpu-lora-top-checkpoint mounted (dataset_sources).")
    _log(rank, "MODEL", f"merged_base={merged_dir} resume_adapter={resume_dir}", force=True)

    model, hidden_dim = build_resumed_model(device, rank, merged_dir, resume_dir)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    total_params = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in trainable_params)
    _log(rank, "MODEL", f"total={total_params:,} trainable={trainable_count:,} "
                        f"({100 * trainable_count / total_params:.4f}%)", force=True)

    per_core_train_batch = PER_CORE_TRAIN_BATCH
    per_core_eval_batch = PER_CORE_EVAL_BATCH
    total_batch = per_core_train_batch * world_size
    _log(rank, "BATCH", f"FIXED per_core_train_batch={per_core_train_batch} "
                        f"per_core_eval_batch={per_core_eval_batch} world_size={world_size} "
                        f"total_train_batch={total_batch} (no probing -- see module docstring)", force=True)

    pos_rate = float(train_pd["label"].mean())
    pos_weight = torch.tensor((1.0 - pos_rate) / max(pos_rate, 1e-9), device=device)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(trainable_params, lr=LR, weight_decay=WEIGHT_DECAY)

    # Optimizer resume: try, but the checkpoint this run resumes from
    # (maxamsecret/berta-tpu-lora-top-checkpoint) has NO optimizer.pt -- it
    # was recovered from a crashed session that never got that far (see
    # reports/berta_tpu_lora_provenance/) -- so this falls through to a
    # fresh AdamW, logged clearly either way.
    optimizer_state_path = os.path.join(resume_dir, "optimizer.pt")
    if os.path.isfile(optimizer_state_path):
        optimizer.load_state_dict(torch.load(optimizer_state_path, map_location="cpu"))
        _log(rank, "RESUME", f"optimizer state restored from {resume_dir}", force=True)
    else:
        _log(rank, "RESUME", f"no optimizer.pt at {resume_dir} -- optimizer starts fresh "
                            f"(expected: the source checkpoint never saved one)", force=True)

    steps_per_epoch_global = len(train_pd) // total_batch
    total_steps = steps_per_epoch_global * EPOCHS   # fresh schedule sized for
                                                       # THIS run's own 10-epoch
                                                       # target, per this
                                                       # project's established
                                                       # "resume weights, fresh
                                                       # schedule" convention.
    warmup_steps = max(1, int(total_steps * WARMUP_RATIO))
    scheduler = get_scheduler("cosine", optimizer=optimizer, num_warmup_steps=warmup_steps,
                               num_training_steps=total_steps)
    _log(rank, "TRAIN", f"steps/epoch={steps_per_epoch_global} total_steps({EPOCHS}ep)={total_steps} "
                        f"warmup={warmup_steps} max_train_seconds={MAX_TRAIN_SECONDS}", force=True)

    val_full_loader, val_remainder_loader = make_eval_loader(
        val_pd, id_to_row, category_to_id, item_input_ids, item_lengths,
        category_label_start, attr_label_start, cls_id, sep_id, pad_id, PAIR_MAX_LENGTH, per_core_eval_batch,
    )

    global_step = 0
    best_val_macro_pr_auc = -1.0
    train_start = time.perf_counter()
    last_quiet_save_at = 0.0

    def do_quiet_save(reason):
        nonlocal last_quiet_save_at
        try:
            save_checkpoint(model, optimizer, epoch_num, global_step, None, "last", rank)
            last_quiet_save_at = time.perf_counter() - train_start
            _log(rank, "SAVE", f"quiet save ({reason}) at epoch={epoch_num} step={global_step}", force=True)
        except Exception as e:
            _log(rank, "ERROR", f"quiet save FAILED ({reason}): {type(e).__name__}: {e}", force=True)

    stopped = False
    for epoch_num in range(1, EPOCHS + 1):
        if stopped:
            break
        train_loader, sampler = make_train_loader(
            train_pd, id_to_row, category_to_id, item_input_ids, item_lengths,
            category_label_start, attr_label_start, cls_id, sep_id, pad_id, PAIR_MAX_LENGTH,
            per_core_train_batch, world_size, rank, SEED, epoch=epoch_num,
        )
        mp_loader = pl_xla.MpDeviceLoader(train_loader, device)
        _log(rank, "EPOCH", f"starting epoch {epoch_num}/{EPOCHS}", force=True)

        for step, batch in enumerate(mp_loader):
            input_ids, attention_mask, token_type_ids, batch_labels, _batch_categories = batch
            batch_labels = batch_labels.to(device)

            # Step-level resilience: an OOM or transient failure on one
            # microbatch should not kill hours of progress.
            try:
                optimizer.zero_grad()
                logits = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
                loss = loss_fn(logits, batch_labels)
                loss.backward()
                xm.optimizer_step(optimizer)
                scheduler.step()
                global_step += 1
            except Exception as e:
                _log(rank, "WARN", f"step failed and was skipped: {type(e).__name__}: {e}", force=True)
                optimizer.zero_grad(set_to_none=True)
                gc.collect()
                continue

            if global_step % LOG_EVERY_STEPS == 0:
                xm.mark_step()
                loss_val = loss.item()
                if rank == 0:
                    elapsed_train = time.perf_counter() - train_start
                    print(f"[{_elapsed():8.1f}s] epoch={epoch_num} step={global_step} loss={loss_val:.4f} "
                          f"lr={scheduler.get_last_lr()[0]:.2e} elapsed_train={elapsed_train:.0f}s", flush=True)

            elapsed_train = time.perf_counter() - train_start
            if (elapsed_train - last_quiet_save_at) >= QUIET_SAVE_EVERY_SECONDS:
                do_quiet_save("periodic safety save")

            if elapsed_train > MAX_TRAIN_SECONDS:
                _log(rank, "TRAIN", f"wall-clock cutoff reached ({elapsed_train:.0f}s > {MAX_TRAIN_SECONDS}s) "
                                    f"-- stopping mid-epoch.", force=True)
                do_quiet_save("wall-clock cutoff")
                stopped = True
                break

        del mp_loader, train_loader, sampler
        gc.collect()
        if stopped:
            break

        # Validation + checkpoint at the end of EVERY epoch -- per direct
        # instruction (2026-08-20), not the 50%-of-epoch cadence used
        # upstream. NOTHING here may kill hours of training: wrapped.
        try:
            val_metrics = evaluate(model, val_full_loader, val_remainder_loader, device, category_vocab, rank)
            improved = val_metrics["macro_pr_auc"] > best_val_macro_pr_auc
            if improved:
                best_val_macro_pr_auc = val_metrics["macro_pr_auc"]
            _log(rank, "METRICS", f"epoch={epoch_num} val_macro_pr_auc={val_metrics['macro_pr_auc']:.4f} "
                                  f"best={best_val_macro_pr_auc:.4f} n_val={val_metrics['n']}", force=True)
            if rank == 0:
                print(f"[METRICS] per_category: {json.dumps(val_metrics['per_category'], ensure_ascii=False)}",
                      flush=True)
            # Own directory per epoch -- see module docstring point A.
            save_checkpoint(model, optimizer, epoch_num, global_step, val_metrics["macro_pr_auc"],
                             f"checkpoint_epoch{epoch_num:02d}", rank)
            if improved:
                save_checkpoint(model, optimizer, epoch_num, global_step, val_metrics["macro_pr_auc"],
                                 "best", rank)
            last_quiet_save_at = time.perf_counter() - train_start
        except Exception as e:
            _log(rank, "ERROR", f"epoch-end checkpoint/eval FAILED at epoch={epoch_num}: "
                                f"{type(e).__name__}: {e}", force=True)
            if rank == 0:
                import traceback
                traceback.print_exc()
            model.train()
            optimizer.zero_grad(set_to_none=True)
            gc.collect()
            do_quiet_save(f"epoch {epoch_num} full checkpoint failed, falling back")

    _log(rank, "DONE", f"total wall-clock: {_elapsed():.0f}s best_val_macro_pr_auc={best_val_macro_pr_auc:.4f}",
         force=True)


def main():
    os.environ.pop("TPU_PROCESS_ADDRESSES", None)
    os.environ.pop("CLOUD_TPU_TASK_ID", None)

    _load_shared_data()

    torch_xla.launch(_mp_fn, args=(), start_method="fork")


if __name__ == "__main__":
    main()
