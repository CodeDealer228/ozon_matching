"""
FRIDA (ai-forever/FRIDA, T5EncoderModel, 823M) LoRA SFT on TPU v5e-8, on the
canonical HUMAN-ONLY split (matches.parquet, 292,523 train / 73,131 val --
no matches_llm.parquet this time), PAIR_MAX_LENGTH=256, target 4 epochs.
Resumes maximmarkovbb/frida-lora-checkpoint-v1-best (val_macro_pr_auc=
0.69377, WITH optimizer state). frida-lora-v2-epoch has since finished
(cancel-acknowledged at 10.5h, 98.6% of its epoch) but its only checkpoint
is a wall-clock QUIET save with no eval, its kernel log came back empty
(0 bytes, lost on cancellation), and its train_state.json carries the
INHERITED 0.69377 running-best rather than a metric for its own weights --
so its real quality is unknown. Published as
maximmarkovbb/frida-lora-v2-epoch-last and queued for
maximmarkovbb/frida-validate-checkpoint; swap dataset_sources to it only
once that measurement comes back better than 0.69377. Resuming a validated
checkpoint is the deliberate choice here: a 9h TPU session is too expensive
to spend on weights of unknown quality.

THE ONE THING THAT ACTUALLY CHANGES vs. frida-lora-v2-epoch: PAIR_MAX_LENGTH
128 -> 256. That is the real bottleneck for FRIDA (per direct instruction),
not the per-item budget: frida-tokenize-items-v2's own docstring measured
that a 128-token PAIR budget truncates ~94% of real pairs, so the encoder
never sees most of the attribute signal the pair actually carries.
ITEM_MAX_LENGTH stays at 256 -- raising it would be pointless here, since
fair-share pair truncation caps each item at ~128 tokens of a 256-token
pair anyway (and at most ~250 in the asymmetric case where the partner item
is tiny, which 256 already covers).

Because ITEM_MAX_LENGTH stays 256, this kernel does NOT tokenize items
in-script: it reuses the already-published item cache
maximmarkovbb/frida-tokenize-items-v2-cache (256 tokens, human scope,
711,304 items, both variant_base and variant_shuffled) -- the exact same
cache frida-lora-v2-epoch trains against. Three concrete wins over
tokenizing here: (a) zero tokenization time inside a hard-capped TPU
session, (b) no AutoTokenizer call at all, which sidesteps the TPU image's
transformers/tokenizers mismatch that killed this kernel's own first run
(TypeError: RobertaProcessing.__new__() got an unexpected keyword argument
'cls' -- the same failure kaggle_kernels/frida-tokenize-items-llm hit and
had to pin exact versions to work around), and (c) the item text this run
trains on byte-matches what the resumed checkpoint already saw, so the only
moving part is the pair budget. pad_id is derived FROM the cache itself
(the token sitting past each row's eos), which needs no tokenizer and
self-verifies across rows.

The cache's v2 attribute-ordering policy (long/https/oem values demoted to
the end) comes along for free, including its 5%-probability shuffled
variant on the train loader only -- a pointer choice between two
already-tokenized arrays, zero extra compute.

===========================================================================
TPU LESSONS CARRIED FORWARD (all real, previously-hit failures in this
project -- not rediscovered here, see kaggle_kernels/berta-llm-tpu-sft's
own module docstring for the original incidents)
===========================================================================
  1. os.environ.pop("TPU_PROCESS_ADDRESSES"/"CLOUD_TPU_TASK_ID") +
     start_method="fork" -- Kaggle's TPU kernel env pre-sets variables that
     conflict with xmp.spawn's own 8-worker topology setup otherwise
     ("Invalid ..._slice_builder_worker_addresses ... Expected 8 worker
     addresses, got 1").
  2. STATIC SHAPES: every DataLoader (train AND eval) uses drop_last=True
     and a FIXED per-core batch size -- exactly 2 XLA graph shapes get
     compiled the whole run (one train, one eval), never a third from a
     ragged final batch.
  3. FIXED per-core batch sizes, NO auto-probing. Three separate real
     incidents in this project's TPU history (documented at length in
     berta-llm-tpu-sft's PER_CORE_TRAIN_BATCH comment) established that
     probing on TPU is actively harmful, not just unhelpful: every probed
     candidate compiles its own XLA program and those programs STAY
     RESIDENT in HBM (unlike a CUDA allocation, which frees cleanly on
     failure), so a probe sweep self-inflicts the exact OOM it's meant to
     avoid -- and xm.get_memory_info() can't even see that program memory
     to warn about it (reported a flat, useless 0.27GB for every candidate
     in that incident). PER_CORE_TRAIN_BATCH/PER_CORE_EVAL_BATCH below are
     therefore a reasoned ESTIMATE (see their own comment for the exact
     arithmetic), not a measured fact -- a real OOM here is a fast, loud,
     CHEAP failure (dies at compile time, on the very first batch, before
     any real training progress exists to lose) that says "lower the
     constant", not a silently-corrupted probe result.
  4. xm.optimizer_step(optimizer), not optimizer.step() -- cross-core
     gradient all-reduce + lazy-graph sync in one call.
  5. autocast OPENED INSIDE forward() (FridaCrossEncoderClassifier below),
     matching every other model class in this project's FRIDA lineage.
  6. Precision: frozen T5 base loaded bf16, LoRA A/B + classifier kept
     fp32 (Adam's own moment estimates and small low-rank updates need
     fp32's wider mantissa; the frozen backbone doing forward-only matmuls
     does not).
  7. ALL CPU-only setup (parquet loading, item-cache mmap + label-position
     scan, category join) happens ONCE in main(), BEFORE torch_xla.launch()
     forks the 8 replicas -- kaggle_kernels/berta-llm-tpu-sft's real,
     root-caused BrokenProcessPool crash was 8x redundant per-replica RAM
     usage from doing this AFTER the fork, not a training bug.
  8. Validation eval wrapped in try/except that CANNOT kill the run --
     "аккуратно считать метрики чтобы не уронить обучение" (careful metric
     computation that can't bring training down), per direct instruction.
     A failed eval falls back to a QUIET (weights-only) save rather than
     losing the checkpoint interval entirely or crashing the session.

===========================================================================
CHECKPOINT CADENCE -- HOURLY, per direct instruction ("нужно делать
чекпоинты каждый час"), not epoch-boundary-based like berta-tpu-human-
continue-384. Session time is the binding constraint (Kaggle TPU cap is
9h; MAX_TRAIN_SECONDS below stops training at 8.5h regardless of epoch
progress), so safety saves are anchored to WALL-CLOCK, not to how much of
an epoch happened to complete in that time. Same checkpoint-hygiene
convention as every kernel in this project since the berta_tpu_lora
provenance trace: every checkpoint gets its OWN directory (never
overwritten), val_macro_pr_auc is EXPLICITLY null when a save skipped
eval (wall-clock-driven quiet save, or a failed eval falling back) rather
than a stale or omitted value. "best" tracks whichever VALIDATED
checkpoint had the top score so far.
===========================================================================

BATCH SIZE ESTIMATE (no TPU probing allowed -- see lesson 3 above):
FRIDA (823M) is ~6.4x more parameters than BERTA (128.3M), with
proportionally more layers/hidden dim (T5-large-scale: ~24 layers,
d_model~1024, vs BERTA's ~12 layers/768) -- both layer COUNT and per-token
activation width roughly double relative to BERTA, so per-token activation
memory is estimated ~2.5-3x BERTA's at equal batch/sequence length. Real
BERTA-TPU data point: batch=64 fit at PAIR_MAX_LENGTH=256 (used well under
the 15.75GB/core budget); batch=64 did NOT fit at 384 (needed 19.11G,
exceeded by 3.36G -- see kaggle_kernels/berta-tpu-human-continue-384's own
module docstring), i.e. roughly halving batch absorbed a 1.5x sequence
increase. Applying the ~2.5-3x per-token cost estimate for FRIDA's size
directly to BERTA's own batch=64-safe-at-256 data point (same sequence
length here, so no separate scaling needed for that axis) gives a rough
safe batch in the 64/2.5..64/3 = ~21-26 range; PER_CORE_TRAIN_BATCH=16
below is deliberately BELOW that estimate (extra safety margin given this
is FRIDA's first-ever TPU run in this project, no prior data point at all
to sanity-check the estimate against) -- easy to raise on a future push
once a real run confirms real headroom, per lesson 3's own philosophy.
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

from transformers import T5EncoderModel, get_scheduler, set_seed
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score

try:
    import peft.import_utils
    import peft.tuners.lora.torchao
    peft.import_utils.is_torchao_available = lambda: False
    peft.tuners.lora.torchao.is_torchao_available = lambda: False
except Exception as e:
    print(f"[WARN] torchao patch failed (likely fine): {e}")

from peft import LoraConfig, PeftModel, get_peft_model, get_peft_model_state_dict
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
ITEMS_HUMAN_PATH_HINT = "/kaggle/input/datasets/maxamsecret/ozon-ecup-2026-matching/items_human.parquet"

PREFIX = "paraphrase: "

# Must equal the cache's own max_length (asserted against cache_meta.json at
# load time, not just assumed). 256 is deliberate, NOT a leftover: fair-share
# pair truncation gives each item ~PAIR_MAX_LENGTH/2 tokens, so a bigger item
# budget buys nothing here -- the pair budget below is the real constraint.
ITEM_MAX_LENGTH = 256
PAIR_MAX_LENGTH = 256   # per direct instruction: 128 -> 256 is THE change

# Only used for the category join below (bounding the category string), NOT
# for item text -- item text comes pre-tokenized from the v2 cache, whose own
# character budget (128/64/384 -> 640) is baked in there.
MAX_CATEGORY_CHARS = 64

# Train loader only: probability of drawing the shuffled attribute-order
# variant per item slot. Matches frida-lora-v2-epoch.
ATTRIBUTE_SHUFFLE_PROB = 0.05

ITEM_CACHE_HINT = "/kaggle/input/datasets/maximmarkovbb/frida-tokenize-items-v2-cache"

CATEGORY_LABEL_TOKEN = 55917
ATTRIBUTES_LABEL_TOKEN = 91837
LABEL_COLON_TOKEN = 30

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.10
LORA_TARGET_MODULES = ["q", "k", "v", "o", "wi_0", "wi_1", "wo"]   # T5's real module names

LR = 3e-4
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.10

# FIXED, NOT probed -- see module docstring's batch-size-estimate section.
PER_CORE_TRAIN_BATCH = 16
PER_CORE_EVAL_BATCH = 32

EPOCHS = 4
MAX_TRAIN_SECONDS = 30600      # 8.5h safety cutoff, TPU session hard cap is 9h
CHECKPOINT_EVERY_SECONDS = 3600  # hourly, per direct instruction
LOG_EVERY_STEPS = 50

WORKING_DIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
OUTPUT_DIR = os.path.join(WORKING_DIR, "lora_frida_tpu_human_sft")

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


# =====================================================================
# PRE-BUILT ITEM CACHE (frida-tokenize-items-v2-cache, 256 tokens, both
# variants). Loaded ONCE, pre-fork -- see module docstring lesson 7.
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


def _load_one_variant(variant_dir, n_items, max_length):
    ids_path = os.path.join(variant_dir, "ids.bin")
    input_ids_path = os.path.join(variant_dir, "input_ids.bin")
    attn_path = os.path.join(variant_dir, "attention_mask.bin")

    expected = {
        ids_path: (os.path.getsize(ids_path), n_items * 8),
        input_ids_path: (os.path.getsize(input_ids_path), n_items * max_length * 4),
        attn_path: (os.path.getsize(attn_path), n_items * max_length),
    }
    for fname, (got, want) in expected.items():
        if got != want:
            raise RuntimeError(f"[CACHE] {fname} size mismatch: got {got:,}, expected {want:,}")

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
    return ids, input_ids, lengths, category_label_start, attr_label_start


def derive_pad_id(input_ids, lengths, max_length):
    """The pad token is whatever sits immediately past each row's eos -- read
    it straight off the cache instead of loading a tokenizer just for this
    one integer (the tokenizer load is exactly what the TPU image can't do,
    see module docstring). Cross-checked over many rows so a wrong guess
    can't slip through silently."""
    candidates = np.where(lengths < max_length)[0]
    if len(candidates) == 0:
        raise RuntimeError("[CACHE] every row is at max_length -- cannot derive pad_id from padding")
    sample = candidates[:: max(1, len(candidates) // 500)][:500]
    pads = {int(input_ids[r, int(lengths[r])]) for r in sample}
    if len(pads) != 1:
        raise RuntimeError(f"[CACHE] inconsistent pad token across rows: {sorted(pads)}")
    return pads.pop()


def load_item_cache_v2():
    cache_root = _find_item_cache_root(ITEM_CACHE_HINT)
    with open(os.path.join(cache_root, "cache_meta.json")) as f:
        meta = json.load(f)
    n_items = meta["n_items"]
    max_length = meta["max_length"]
    print(f"[CACHE] n_items={n_items:,} max_length={max_length} prefix={meta.get('prefix')!r}", flush=True)
    if max_length != ITEM_MAX_LENGTH:
        raise RuntimeError(f"cache max_length={max_length} != ITEM_MAX_LENGTH={ITEM_MAX_LENGTH}")

    variants = {}
    for name in ("variant_base", "variant_shuffled"):
        t0 = time.perf_counter()
        variants[name] = _load_one_variant(os.path.join(cache_root, name), n_items, max_length)
        print(f"[CACHE] {name} loaded + bos/eos checked ({time.perf_counter() - t0:.1f}s)", flush=True)

    ids_base = np.asarray(variants["variant_base"][0])
    if not np.array_equal(ids_base, np.asarray(variants["variant_shuffled"][0])):
        raise RuntimeError("[CACHE] the two variants have different id orderings -- the per-slot "
                            "coin-flip between them would silently mix up items.")

    pad_id = derive_pad_id(variants["variant_base"][1], variants["variant_base"][2], max_length)
    print(f"[CACHE] pad_id={pad_id} (derived from the cache's own padding, no tokenizer)", flush=True)

    id_to_row = {int(item_id): i for i, item_id in enumerate(ids_base)}
    assert len(id_to_row) == n_items, f"duplicate item ids: {n_items} rows, {len(id_to_row)} unique"
    return variants, id_to_row, pad_id


# =====================================================================
# PAIR ASSEMBLY (unchanged fair-share-truncation, matches frida-lora-v2-epoch)
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


def make_pair_collate_fn(variants, pad_id, pair_max_length, augment):
    """augment=False (val loader) always draws variant_base -- val must stay
    stable and comparable across checkpoints. augment=True (train loader)
    draws variant_shuffled with ATTRIBUTE_SHUFFLE_PROB per item slot: a
    pointer choice into an already-tokenized array, zero extra compute."""
    base = variants["variant_base"]
    shuf = variants["variant_shuffled"]
    rng = np.random.RandomState(SEED + 1)

    def pick(row):
        a = shuf if (augment and rng.random_sample() < ATTRIBUTE_SHUFFLE_PROB) else base
        return build_compact_ids(a[1], row, int(a[2][row]), int(a[3][row]), int(a[4][row]))

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
            torch.from_numpy(input_ids), torch.from_numpy(attention_mask),
            torch.tensor(labels, dtype=torch.float32), np.asarray(categories, dtype=np.int64),
        )
    return collate_fn


def make_train_loader(pairs_pd, id_to_row, category_to_id, variants, pad_id, pair_max_length,
                       per_core_batch, num_replicas, rank, seed, epoch):
    row1 = pairs_pd["id1"].map(id_to_row).to_numpy().astype(np.int64)
    row2 = pairs_pd["id2"].map(id_to_row).to_numpy().astype(np.int64)
    labels = pairs_pd["label"].to_numpy().astype(np.float32)
    categories = pairs_pd["category1"].map(category_to_id).fillna(category_to_id.get("UNKNOWN", -1)).to_numpy().astype(np.int64)

    dataset = PairIndexDataset(row1, row2, labels, categories)
    sampler = DistributedSampler(dataset, num_replicas=num_replicas, rank=rank, shuffle=True, seed=seed,
                                  drop_last=True)
    sampler.set_epoch(epoch)
    collate_fn = make_pair_collate_fn(variants, pad_id, pair_max_length, augment=True)
    # persistent_workers=False -- this loader is rebuilt fresh every epoch
    # (new DistributedSampler each time); see kaggle_kernels/berta-llm-tpu-sft's
    # own fix for why True here only risks a transient double-worker overlap.
    loader = DataLoader(dataset, batch_size=per_core_batch, sampler=sampler, drop_last=True,
                         collate_fn=collate_fn, num_workers=2, persistent_workers=False)
    return loader, sampler


def make_eval_loader(pairs_pd, id_to_row, category_to_id, variants, pad_id, pair_max_length, eval_batch):
    row1 = pairs_pd["id1"].map(id_to_row).to_numpy().astype(np.int64)
    row2 = pairs_pd["id2"].map(id_to_row).to_numpy().astype(np.int64)
    labels = pairs_pd["label"].to_numpy().astype(np.float32)
    categories = pairs_pd["category1"].map(category_to_id).fillna(category_to_id.get("UNKNOWN", -1)).to_numpy().astype(np.int64)

    dataset = PairIndexDataset(row1, row2, labels, categories)
    collate_fn = make_pair_collate_fn(variants, pad_id, pair_max_length, augment=False)
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

class FridaCrossEncoderClassifier(nn.Module):
    def __init__(self, encoder, head):
        super().__init__()
        self.encoder = encoder
        self.classifier = head

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :].float()
        return self.classifier(cls).view(-1).float()


def build_resumed_model(device, rank, resume_dir):
    try:
        base_model = T5EncoderModel.from_pretrained(MODEL_NAME, dtype=torch.bfloat16, attn_implementation="eager")
    except (TypeError, ValueError):
        base_model = T5EncoderModel.from_pretrained(MODEL_NAME, dtype=torch.bfloat16)
    hidden_dim = base_model.config.d_model

    peft_model = PeftModel.from_pretrained(base_model, resume_dir, is_trainable=True)
    if rank == 0:
        peft_model.print_trainable_parameters()
    n_trainable_lora = sum(p.numel() for n, p in peft_model.named_parameters() if "lora_" in n and p.requires_grad)
    assert n_trainable_lora > 0, "Resumed LoRA adapter has zero trainable params -- is_trainable didn't take."

    for name, p in peft_model.named_parameters():
        if "lora_" in name:
            p.data = p.data.float()

    classifier_head = nn.Linear(hidden_dim, 1)
    classifier_path = os.path.join(resume_dir, "classifier.pt")
    classifier_head.load_state_dict(torch.load(classifier_path, map_location="cpu"))
    _log(rank, "MODEL", f"classifier warm-started from {resume_dir}/classifier.pt", force=True)

    model = FridaCrossEncoderClassifier(peft_model, classifier_head).to(device)
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
                input_ids, attention_mask, batch_labels, batch_categories = batch
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
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
# CHECKPOINT -- own directory per save, explicit null val when unvalidated.
# =====================================================================

def save_checkpoint(model, optimizer, global_step, epoch, val_macro_pr_auc, label, rank):
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
            "epoch": epoch, "global_step": global_step,
            "val_macro_pr_auc": val_macro_pr_auc,   # explicit null if unvalidated
        }, f, indent=2)
    print(f"[SAVE] {label}: epoch={epoch} step={global_step} val_macro_pr_auc={val_macro_pr_auc}", flush=True)


# =====================================================================
# SHARED CPU-ONLY SETUP -- loaded ONCE, pre-fork.
# =====================================================================

_SHARED = {}


def _load_shared_data():
    print(f"[{_elapsed():8.1f}s][SETUP] loading all CPU-only data once, pre-fork ...", flush=True)

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

    variants, id_to_row, pad_id = load_item_cache_v2()

    items_human_path = _find_dataset_file(ITEMS_HUMAN_PATH_HINT, "items_human.parquet")
    needed_ids = pd.unique(pd.concat([train_pd["id1"], val_pd["id1"]]))
    needed_ids_lazy = pl.DataFrame({"id": pd.Series(needed_ids, dtype="int64")}).lazy()
    items_cat = (
        pl.scan_parquet(items_human_path).join(needed_ids_lazy, on="id", how="semi")
        .select(["id", "category"])
        .with_columns(pl.col("category").fill_null("UNKNOWN").cast(pl.Utf8).str.slice(0, MAX_CATEGORY_CHARS))
        .collect(engine="streaming")
    )
    category_vocab = sorted(items_cat.get_column("category").unique().to_list()) + ["UNKNOWN"]
    category_to_id = {c: i for i, c in enumerate(category_vocab)}
    id_to_category = dict(zip(items_cat.get_column("id").to_list(), items_cat.get_column("category").to_list()))
    train_pd["category1"] = train_pd["id1"].map(id_to_category).fillna("UNKNOWN")
    val_pd["category1"] = val_pd["id1"].map(id_to_category).fillna("UNKNOWN")
    del items_cat, needed_ids, needed_ids_lazy, id_to_category
    gc.collect()

    _SHARED.update(dict(
        pad_id=pad_id, train_pd=train_pd, val_pd=val_pd, val_fingerprint=val_fingerprint,
        variants=variants, id_to_row=id_to_row,
        category_vocab=category_vocab, category_to_id=category_to_id,
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
                        f"item_max_length={ITEM_MAX_LENGTH} epochs={EPOCHS}", force=True)
    if rank == 0 and os.path.isdir("/kaggle/input"):
        for _root, _dirs, _files in os.walk("/kaggle/input"):
            depth = _root[len("/kaggle/input"):].count(os.sep)
            if depth <= 3:
                print(f"[DEBUG] {_root} -> dirs={_dirs} files={_files}")

    pad_id = _SHARED["pad_id"]
    train_pd, val_pd, val_fingerprint = _SHARED["train_pd"], _SHARED["val_pd"], _SHARED["val_fingerprint"]
    variants = _SHARED["variants"]
    id_to_row = _SHARED["id_to_row"]
    category_vocab, category_to_id = _SHARED["category_vocab"], _SHARED["category_to_id"]
    _log(rank, "DATA", f"inherited from parent (fork/COW): train={len(train_pd):,} val={len(val_pd):,} "
                        f"val_fingerprint={val_fingerprint}", force=True)

    resume_dir = _find_resume_adapter_dir()
    if resume_dir is None:
        raise RuntimeError("No resume adapter found under /kaggle/input -- expected "
                            "maximmarkovbb/frida-lora-checkpoint-v1-best mounted (dataset_sources).")
    _log(rank, "MODEL", f"resume_adapter={resume_dir}", force=True)
    model, hidden_dim = build_resumed_model(device, rank, resume_dir)
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
    optimizer_state_path = os.path.join(resume_dir, "optimizer.pt")
    if os.path.isfile(optimizer_state_path):
        optimizer.load_state_dict(torch.load(optimizer_state_path, map_location="cpu"))
        _log(rank, "RESUME", f"optimizer state restored from {resume_dir}", force=True)
    else:
        _log(rank, "RESUME", f"no optimizer.pt at {resume_dir} -- optimizer starts fresh", force=True)

    steps_per_epoch_global = len(train_pd) // total_batch
    total_steps = steps_per_epoch_global * EPOCHS   # fresh schedule sized for THIS run's own target
    warmup_steps = max(1, int(total_steps * WARMUP_RATIO))
    scheduler = get_scheduler("cosine", optimizer=optimizer, num_warmup_steps=warmup_steps,
                               num_training_steps=total_steps)
    _log(rank, "TRAIN", f"steps/epoch={steps_per_epoch_global} total_steps({EPOCHS}ep)={total_steps} "
                        f"warmup={warmup_steps} max_train_seconds={MAX_TRAIN_SECONDS} "
                        f"checkpoint_every_seconds={CHECKPOINT_EVERY_SECONDS}", force=True)

    val_full_loader, val_remainder_loader = make_eval_loader(
        val_pd, id_to_row, category_to_id, variants, pad_id, PAIR_MAX_LENGTH, per_core_eval_batch,
    )

    global_step = 0
    best_val_macro_pr_auc = -1.0
    train_start = time.perf_counter()
    last_checkpoint_at = 0.0
    stopped = False

    def do_quiet_save(reason, epoch_num):
        try:
            save_checkpoint(model, optimizer, global_step, epoch_num, None, "last", rank)
            _log(rank, "SAVE", f"quiet save ({reason}) at epoch={epoch_num} step={global_step}", force=True)
        except Exception as e:
            _log(rank, "ERROR", f"quiet save FAILED ({reason}): {type(e).__name__}: {e}", force=True)

    def do_full_checkpoint(epoch_num, label):
        nonlocal best_val_macro_pr_auc
        try:
            val_metrics = evaluate(model, val_full_loader, val_remainder_loader, device, category_vocab, rank)
            improved = val_metrics["macro_pr_auc"] > best_val_macro_pr_auc
            if improved:
                best_val_macro_pr_auc = val_metrics["macro_pr_auc"]
            _log(rank, "METRICS", f"{label}: epoch={epoch_num} step={global_step} "
                                  f"val_macro_pr_auc={val_metrics['macro_pr_auc']:.4f} "
                                  f"best={best_val_macro_pr_auc:.4f} n_val={val_metrics['n']}", force=True)
            if rank == 0:
                print(f"[METRICS] per_category: {json.dumps(val_metrics['per_category'], ensure_ascii=False)}",
                      flush=True)
            save_checkpoint(model, optimizer, global_step, epoch_num, val_metrics["macro_pr_auc"], label, rank)
            if improved:
                save_checkpoint(model, optimizer, global_step, epoch_num, val_metrics["macro_pr_auc"], "best", rank)
            return True
        except Exception as e:
            _log(rank, "ERROR", f"full checkpoint/eval FAILED at epoch={epoch_num} step={global_step}: "
                                f"{type(e).__name__}: {e}", force=True)
            if rank == 0:
                import traceback
                traceback.print_exc()
            model.train()
            optimizer.zero_grad(set_to_none=True)
            gc.collect()
            do_quiet_save("full checkpoint failed, falling back", epoch_num)
            return False

    for epoch in range(1, EPOCHS + 1):
        if stopped:
            break
        train_loader, sampler = make_train_loader(
            train_pd, id_to_row, category_to_id, variants, pad_id, PAIR_MAX_LENGTH,
            per_core_train_batch, world_size, rank, SEED, epoch=epoch,
        )
        mp_loader = pl_xla.MpDeviceLoader(train_loader, device)
        _log(rank, "EPOCH", f"starting epoch {epoch}/{EPOCHS}", force=True)

        for step, batch in enumerate(mp_loader):
            input_ids, attention_mask, batch_labels, _batch_categories = batch
            batch_labels = batch_labels.to(device)

            try:
                optimizer.zero_grad()
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
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
                    print(f"[{_elapsed():8.1f}s] epoch={epoch} step={global_step} loss={loss_val:.4f} "
                          f"lr={scheduler.get_last_lr()[0]:.2e} elapsed_train={elapsed_train:.0f}s", flush=True)

            elapsed_train = time.perf_counter() - train_start
            if (elapsed_train - last_checkpoint_at) >= CHECKPOINT_EVERY_SECONDS:
                do_full_checkpoint(epoch, f"checkpoint_hour{int(elapsed_train // CHECKPOINT_EVERY_SECONDS):02d}")
                last_checkpoint_at = elapsed_train
                model.train()

            if elapsed_train > MAX_TRAIN_SECONDS:
                _log(rank, "TRAIN", f"wall-clock cutoff reached ({elapsed_train:.0f}s > {MAX_TRAIN_SECONDS}s) "
                                    f"-- stopping mid-epoch.", force=True)
                do_quiet_save("wall-clock cutoff", epoch)
                stopped = True
                break

        del mp_loader, train_loader, sampler
        gc.collect()
        if stopped:
            break

        # Epoch boundary ALSO gets a full checkpoint (on top of the hourly
        # cadence) -- cheap insurance, and gives a clean "epoch N done" data
        # point even if it lands close to an hourly one.
        do_full_checkpoint(epoch, f"checkpoint_epoch{epoch:02d}")
        last_checkpoint_at = time.perf_counter() - train_start
        model.train()

    _log(rank, "DONE", f"total wall-clock: {_elapsed():.0f}s best_val_macro_pr_auc={best_val_macro_pr_auc:.4f}",
         force=True)


def main():
    os.environ.pop("TPU_PROCESS_ADDRESSES", None)
    os.environ.pop("CLOUD_TPU_TASK_ID", None)

    _load_shared_data()

    torch_xla.launch(_mp_fn, args=(), start_method="fork")


if __name__ == "__main__":
    main()
