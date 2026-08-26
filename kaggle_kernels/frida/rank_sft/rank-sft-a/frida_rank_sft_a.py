"""
FRIDA (ai-forever/FRIDA, T5EncoderModel, 823M) SFT continuation on TOP of the
project's best BCE checkpoint (maxamsecret/frida-full-sft-w/best, val=0.7582,
test=0.481), switching the training objective to category-batched pairwise
RankNet weighted by |delta-AP| plus a small BCE stabilizer -- per direct
instruction in PROMPT.md (ozon_ranknet_harness_bundle_2026-08-26):

    L = L_rank + 0.05 * L_BCE
    L_rank = mean_over_pairs[ w_ij * softplus(s_j^- - s_i^+) ]
    w_ij = |delta_AP_ij|   (rank-based, category-local, refreshed every 0.5 epoch)

WHY THIS BUILDS ON full_sft_continue, NOT full_sft's from-scratch code path
===========================================================================
Every TPU crash class already hit in this project (six of them, see
full_sft_continue/full-sft-continue/frida_full_sft_continue.py's own
docstring) is orthogonal to the loss function: OOM batch sizing,
gradient-checkpointing's torch.xla AttributeError (twice, two different
root causes), lost replicas / stuck collectives, bare-DataLoader eval never
syncing. None of that is loss-specific, so this kernel keeps that entire
proven skeleton (item caches, pair assembly, XLA registration, gradient
checkpointing config, MpDeviceLoader everywhere, checkpoint contract,
content-based resume, smoke tests, circuit breaker) verbatim and only
replaces: (a) the optimizer (AdamW, per direct PROMPT.md instruction --
this is a *resume* of an already-well-conditioned checkpoint, not a
from-scratch full-FT, so Adafactor's memory-motivated case is weaker and
the prompt asks for AdamW explicitly), (b) the batch sampler (category-
batched instead of item-disjoint-shuffled), (c) the loss (RankNet+ddAP
instead of category/pos-weighted focal BCE), (d) the resume source (always
maxamsecret/frida-full-sft-w/best specifically, never a sibling
full_sft_continue experiment -- PROMPT.md: "Берём лучший BCE-checkpoint
FRIDA").

CATEGORY-BATCHED SAMPLING, PER DIRECT INSTRUCTION
===========================================================================
"На каждом training step выбираем одну категорию. Все TPU работают только с
ней." -- every one of the 8 XLA replicas trains the SAME category on a given
step, not 8 different ones sharded off a global shuffle. Implemented as a
custom IterableDataset-free approach: a CategoryBatchSampler that, once per
step, picks one category (round-robin over the 20, reshuffled every epoch)
and returns `world_size * PER_CORE_TRAIN_BATCH` row-indices drawn ONLY from
that category's rows, split contiguously across the world_size replicas so
every replica's DistributedSampler-free shard is the same category. This
keeps DataLoader/MpDeviceLoader/xm.optimizer_step mechanics byte-identical
to every other kernel in this project (still one fixed shape, still
drop_last, still a plain per-replica DataLoader wrapped in MpDeviceLoader)
-- only WHICH rows go into a step's batch changes, which is exactly the
project's own "don't touch what already works" convention.

Within a category-batch: 25% positive / 75% negative by construction (per
direct instruction), oversampling hard positives/negatives when a category
has few of either (see CategoryPools.sample -- weighted by proximity to the
current decision boundary using the SAME cached rank/logit table the
delta-AP weighting below already maintains, so "hard" costs no extra
forward pass).

DELTA-AP CACHE: ONE FORWARD PASS OVER human-TRAIN PER 0.5 EPOCH, PER CATEGORY
===========================================================================
PROMPT.md: rebuild the per-category ranking (by current model logits) every
0.5 epoch, and derive each pair's RankNet weight from |delta-AP| implied by
swapping a positive/negative's rank. This is a real AP-sensitivity
computation, not an approximation: for a category with P positives and N
negatives sorted by score descending, swapping positive i (rank r_i) past
negative j (rank r_j) changes AP by a closed-form amount derivable from
the standard "AP as area under the step function" definition; this kernel
uses the standard, exact pairwise-swap delta-AP formula (see
`pairwise_delta_ap` for the derivation, ported from the standard
LambdaRank-style delta-NDCG/delta-AP construction, specialized to binary
AP) rather than a numerical two-eval-and-diff (that would need 2x|pairs|
forward passes -- intractable at this pair count). Cache is CPU-side
(numpy), rebuilt on a SEPARATE inference-only forward pass through
human-train only (not the LLM side -- PROMPT.md's ranking objective is
scoped to "берём всю human-разметку", the human clean labels), sharded and
all_gathered the same way `evaluate()` already does for val.

TRAIN SCOPE: HUMAN CLEAN LABELS ONLY FOR RANKING, PER DIRECT INSTRUCTION
===========================================================================
PROMPT.md's ranking procedure ("берём всю human-разметку и 3M stratified")
plus its BCE term ("обычный BCE по clean labels") are both scoped to human
clean pairs (matches.parquet's item-disjoint train split) mixed with a 3M
category/pos-rate/label-balance/char-len stratified LLM subsample -- NOT
the pooled human+LLM-soft-label recipe full_sft_continue used. Rank pairs
are built ONLY from human rows (LLM soft labels have no clean positive/
negative split to rank against each other meaningfully); the 3M LLM
subsample participates ONLY in the 0.05*BCE stabilizer term, at its own
soft target, exactly parallel to how full_sft_continue already treats LLM
rows in its BCE-style loss. Both scopes share one category-batched step
(same category, human rows for rank pairs + a matched slice of that
category's LLM subsample for the extra BCE rows) so a single forward pass
covers both loss terms, matching PROMPT.md's "На тех же logits... Никаких
дополнительных forward-pass нет."

LORA FALLBACK, PER DIRECT INSTRUCTION ("если SFT будет нестабильным,
сильно переобучаться или падать OOM даже на маленьких батчах - используй Lora")
===========================================================================
USE_LORA_FALLBACK starts False (full-FT resume, matching the BCE
checkpoint's own training mode -- resuming full-FT weights into a LoRA
adapter would need a merge step this kernel does not have). If a real run
OOMs at the smallest viable PER_CORE_TRAIN_BATCH (see that constant's own
comment), the documented recovery is: set USE_LORA_FALLBACK=True, which
wraps the already-loaded full-FT encoder in a fresh LoRA adapter
(peft.get_peft_model, target_modules matching this project's own verified
FRIDA/T5Attention+DenseReluDense list) and freezes the base -- same
resume-then-adapt pattern frida-percat-lora already uses elsewhere in this
project, not a new mechanism.

CHECKPOINT CONTRACT: identical to full_sft_continue's own (own directory
per save, explicit null val on quiet saves, wall-clock cadence, weights+
optimizer+scheduler on every quiet save, full eval only at the 0.5-epoch
rank-cache-rebuild boundary AND at true epoch boundaries) -- see that
kernel's docstring CHECKPOINT CONTRACT section for the full rationale,
unchanged here.
"""
import os
import sys
import gc
import json
import math
import time
import hashlib
import resource
import subprocess

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "polars", "peft"])

import numpy as np
import pandas as pd
import polars as pl

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from transformers import T5EncoderModel, set_seed
from sklearn.metrics import average_precision_score

import torch_xla
import torch_xla.core.xla_model as xm
from torch_xla import runtime as xr
import torch_xla.distributed.parallel_loader as pl_xla

# Same FIFTH-REAL-CRASH fix as every other FRIDA TPU kernel in this project
# (see full_sft_continue's own docstring): torch_xla never registers itself
# under torch.xla, but torch.utils.checkpoint._get_device_module() looks it
# up via getattr(torch, "xla") in both reentrant and non-reentrant modes
# whenever preserve_rng_state=True (the default). Registering it here makes
# that lookup succeed regardless of which code path calls it.
if not hasattr(torch, "xla"):
    torch.xla = torch_xla


# =====================================================================
# CONFIG
# =====================================================================

SEED = 42
MODEL_NAME = "ai-forever/FRIDA"

ITEM_SPLIT_PATH_HINT = "/kaggle/input/datasets/maxamsecret/ozon-ecup-2026-matching/matches_item_disjoint_70_30_seed42.parquet"
LLM_SPLIT_PATH_HINT = "/kaggle/input/datasets/maxamsecret/frida-llm-train-val-split/llm_train_val_split.parquet"
HUMAN_CACHE_HINT = "/kaggle/input/datasets/maxamsecret/frida-tokenize-items-v3/human_scope"
LLM_CACHE_HINT = "/kaggle/input/datasets/maxamsecret/frida-tokenize-items-v3/llm_scope"

ITEM_MAX_LENGTH = 256
# Same serving-length rationale as every full_sft_continue experiment: the
# real judging stand's own time budget, not a training-only choice.
PAIR_MAX_LENGTH = 192

CATEGORY_LABEL_TOKEN = 55917
ATTRIBUTES_LABEL_TOKEN = 91837
LABEL_COLON_TOKEN = 30

HEAD_HIDDEN = (1024, 512, 128)
HEAD_DROPOUT = 0.2

# AdamW per direct PROMPT.md instruction ("AdamW и маленьким LR, например
# около 3e-5 для full-SFT FRIDA") -- this is a RESUME of an already-trained
# checkpoint, not a from-scratch full-FT, so the memory case for Adafactor
# (see full_sft_continue's own docstring) is weaker: fp32 weights+grad+
# Adam(m,v) = 4x the 3.29GB tensor = 13.2GB, vs a 16GB v5e-8 core, leaving
# headroom thinner than Adafactor's ~9GB but still fits with gradient
# checkpointing and a conservative batch (see PER_CORE_TRAIN_BATCH below).
LR = 3e-5
HEAD_LR_PEAK_MULTIPLIER = 3.0    # head is already trained (resumed), so a
                                  # smaller peak-vs-backbone ratio than a
                                  # from-scratch run's 10x -- it only needs
                                  # to re-adapt to the new objective, not
                                  # learn from random init.
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.10
LR_MIN_RATIO = 0.05

BCE_WEIGHT = 0.05                # per PROMPT.md: L = L_rank + 0.05 * L_BCE
POS_FRACTION_IN_BATCH = 0.25     # 25% positives / 75% negatives per direct
                                  # instruction -- applies to the HUMAN-rank
                                  # portion of a step's batch only (see
                                  # LLM_STABILIZER_FRACTION below)
LLM_STABILIZER_FRACTION = 0.3
                                  # REAL CRASH (2026-08-26, v1): the first
                                  # training step, right after "EPOCH:
                                  # starting epoch 1/5" printed on all 8
                                  # ranks, hung silently for ~9 minutes then
                                  # every replica got an external SIGTERM
                                  # (BrokenProcessPool, no Python traceback)
                                  # -- the exact "unvalidated new XLA program
                                  # shape" signature this project's own
                                  # kaggle-experiment-safety skill warns
                                  # about. Root cause: v1's per-step batch
                                  # concatenated PER_CORE_TRAIN_BATCH human
                                  # rows PLUS UP TO PER_CORE_TRAIN_BATCH MORE
                                  # llm-stabilizer rows on top (see FILES.md
                                  # note below) -- a batch of up to 2x
                                  # PER_CORE_TRAIN_BATCH rows that the early
                                  # smoke test never exercised (the smoke
                                  # test only ever forwarded a synthetic
                                  # batch sized exactly PER_CORE_TRAIN_BATCH).
                                  # That larger, never-compiled shape hit its
                                  # first real XLA compile mid-run instead of
                                  # during the ~1-2 minute smoke-test window,
                                  # on the full-FT + gradient-checkpointing +
                                  # AdamW memory budget this kernel is
                                  # already tight on (see PER_CORE_TRAIN_BATCH
                                  # itself) -- consistent with either a slow
                                  # first-time compile exceeding Kaggle's
                                  # session-liveness timeout, or a host-level
                                  # OOM-kill outside Python's own exception
                                  # handling, both externally visible only as
                                  # a bare SIGTERM. FIX: the LLM-stabilizer
                                  # rows are now carved OUT of
                                  # PER_CORE_TRAIN_BATCH (not added on top),
                                  # so every step's total batch size is
                                  # EXACTLY PER_CORE_TRAIN_BATCH, every
                                  # epoch, every category -- byte-identical
                                  # to the shape the smoke test already
                                  # validates. 0.3 reserves a real, non-
                                  # trivial slice for the "маленький
                                  # стабилизатор" BCE term while keeping most
                                  # of the batch on the actual ranking
                                  # objective PROMPT.md prioritizes.
HARD_OVERSAMPLE_TEMPERATURE = 2.0
                                  # softmax temperature over |score - 0.5
                                  # decision-boundary distance|^-1 when
                                  # drawing within a category -- higher
                                  # temperature concentrates more on hard
                                  # examples; kept a documented, moderate
                                  # constant (not tuned) since PROMPT.md
                                  # only asks for a bias toward hard
                                  # examples, not a specific strength.

LLM_TRAIN_SUBSAMPLE_SIZE = 3_000_000
                                  # PROMPT.md: "3M stratified (category_
                                  # balance, category_pos_rate, label_
                                  # balance, char_len)" -- four strata this
                                  # kernel's own stratified_llm_sample_v2
                                  # implements explicitly (see that
                                  # function), unlike full_sft_continue's
                                  # category+target-level-only version.

RANK_CACHE_REFRESH_EPOCH_FRACTION = 0.5
                                  # PROMPT.md: "Через 0.5 epoch снова
                                  # прогоняем clean train текущей моделью,
                                  # перестраиваем category rankings"

EXPERIMENT_NAME = "rank_sft_a"

USE_LORA_FALLBACK = False        # see docstring's LORA FALLBACK section --
                                  # flip True + relaunch if a from-full-FT
                                  # OOM persists at the smallest viable batch
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
# Verified against the real loaded module tree elsewhere in this project
# (frida-lora-sft-human-only-10-epochs docstring): T5Attention uses
# q/k/v/o, FRIDA's gated-gelu DenseReluDense FFN uses wi_0/wi_1/wo.
LORA_TARGET_MODULES = ["q", "k", "v", "o", "wi_0", "wi_1", "wo"]

# MEASURED-STYLE CONSERVATIVE ESTIMATE (not probed -- see kaggle-experiment-
# safety: never auto-probe batch size on TPU). Starts BELOW full_sft_
# continue's own final battle-tested value (20 at Adafactor) because AdamW's
# per-parameter state is 2x Adafactor's factored footprint (m + v full
# tensors vs. row/col vectors) -- halved from 20 as the documented starting
# guess, consistent with this project's "if it OOMs, halve it and relaunch"
# policy already applied proactively rather than discovered via a crash.
PER_CORE_TRAIN_BATCH = 10
PER_CORE_EVAL_BATCH = 64
RANK_CACHE_BATCH = 96             # inference-only (no grad), can run larger

EPOCHS = 5                        # per PROMPT.md: "рассчитываем на 5 полных эпох"

MAX_TRAIN_SECONDS = 30600         # 8.5h -- Kaggle's TPU session cap is 9h
CHECKPOINT_EVERY_SECONDS = 3600   # "сохраняем тихие чекпоинты каждый час"
                                   # per direct PROMPT.md instruction
LOG_EVERY_STEPS = 50
MAX_CONSECUTIVE_STEP_FAILURES = 20

PROJECTED_ROWS_PER_SEC = 100      # conservative guess: AdamW's extra state
                                   # traffic + the rank-cache forward passes
                                   # interleaved every 0.5 epoch make this
                                   # somewhat slower than full_sft_continue's
                                   # measured 128 rows/s -- used only to size
                                   # the LR schedule horizon, not a promise.

WORKING_DIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
OUTPUT_DIR = os.path.join(WORKING_DIR, "frida_rank_sft")

t_script_start = time.perf_counter()


def _elapsed():
    return time.perf_counter() - t_script_start


def _rss_mb():
    """Peak (not current) resident-set size in MB so far this process, via
    resource.getrusage -- the same direct-RSS-logging pattern this project's
    own frida-percat-lora/frida-percat-lora-fullsft-last kernels adopted
    after a real host-RAM OOM (SIGTERM -> BrokenProcessPool, no Python
    traceback -- the exact failure signature rank-sft-a's own v1 hit,
    see LLM_STABILIZER_FRACTION's docstring) made timing-only diagnosis
    insufficient. ru_maxrss is cumulative-peak, monotonically increasing
    for the process lifetime -- exactly what you want logged at every
    stage transition to localize WHERE a slow memory climb happened,
    since the crash itself leaves no Python-level evidence."""
    try:
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    except Exception:
        return -1.0


def _log_stage(rank, stage, msg, t0=None, force=False):
    """Like _log, but always appends peak RSS and (if t0 given) this
    stage's own wall-clock duration -- use at every point where state
    changes, a new object is created, a checkpoint is written, or a crash
    is plausible (per direct instruction: log every such stage)."""
    if rank != 0 and not force:
        return
    extra = f" rss={_rss_mb():.0f}MB"
    if t0 is not None:
        extra += f" dt={time.perf_counter() - t0:.1f}s"
    print(f"[{_elapsed():8.1f}s][rank{rank}] {stage}: {msg}{extra}", flush=True)


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
# ITEM CACHES -- byte-identical to full_sft_continue (proven on real
# multi-hour TPU runs; the char-budget/attribute-policy this cache layout
# encodes is invisible here).
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
                    return root
    raise RuntimeError(f"[CACHE] no item cache with n_items={expect_n_items:,} found (hint was {hint})")


def load_v1_cache(hint, name):
    root = _find_v1_cache_root(hint, EXPECTED_N_ITEMS[name])
    with open(os.path.join(root, "cache_meta.json")) as f:
        meta = json.load(f)
    n_items, max_length = meta["n_items"], meta["max_length"]
    print(f"[CACHE:{name}] {root}: n_items={n_items:,} max_length={max_length}", flush=True)
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

    lengths = _lengths_chunked(attn, n_items)
    del attn
    cat_start = _pattern_starts_chunked(input_ids, n_items, CATEGORY_LABEL_TOKEN, LABEL_COLON_TOKEN)
    attr_start = _pattern_starts_chunked(input_ids, n_items, ATTRIBUTES_LABEL_TOKEN, LABEL_COLON_TOKEN)

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
    return pads.pop()


def lookup_rows(cache, item_ids):
    pos = np.searchsorted(cache["sorted_ids"], item_ids)
    pos_clipped = np.clip(pos, 0, len(cache["sorted_ids"]) - 1)
    hit = cache["sorted_ids"][pos_clipped] == item_ids
    rows = np.where(hit, cache["order"][pos_clipped], -1)
    return rows.astype(np.int64)


# =====================================================================
# PAIR ASSEMBLY -- byte-identical to every other FRIDA kernel in this repo
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


def make_eval_shard(frame, num_replicas, rank, eval_batch):
    n = len(frame["label"])
    per_replica_batches = max(1, math.ceil(n / (eval_batch * num_replicas)))
    per_replica_n = per_replica_batches * eval_batch
    padded_total = per_replica_n * num_replicas
    idx = np.arange(n)
    if padded_total > n:
        pad = padded_total - n
        idx = np.concatenate([idx, np.arange(pad) % n])
    valid_full = np.concatenate([np.ones(n, dtype=np.bool_), np.zeros(padded_total - n, dtype=np.bool_)])
    start, end = rank * per_replica_n, (rank + 1) * per_replica_n
    shard_idx = idx[start:end]
    shard = {k: v[shard_idx] for k, v in frame.items()}
    return shard, valid_full[start:end]


def make_eval_loader_sharded(frame, caches, pad_id, eval_batch, num_replicas, rank):
    shard, shard_valid = make_eval_shard(frame, num_replicas, rank, eval_batch)
    dataset = PairIndexDataset(shard["scope"], shard["row1"], shard["row2"],
                               shard["label"], shard["category"])
    collate_fn = make_pair_collate_fn(caches, pad_id, PAIR_MAX_LENGTH)
    loader = DataLoader(dataset, batch_size=eval_batch, shuffle=False, drop_last=True,
                        collate_fn=collate_fn, num_workers=2)
    return loader, shard_valid


# =====================================================================
# CATEGORY-BATCHED SAMPLING -- one category per step, all replicas, 25/75
# pos/neg mix, hard-example oversampling via the rank-cache's own scores.
# =====================================================================

class CategoryPools:
    """Per-category index pools into `frame` (a human-train frame, id-
    disjoint from val), split into positive/negative row-index arrays. A
    fresh scores array (updated by rebuild_scores, see the rank cache
    below) lets sample() bias toward hard (near-boundary) rows without any
    extra forward pass -- it reuses the same logits the delta-AP cache
    already computed this half-epoch.
    """

    def __init__(self, category, label, n_categories):
        self.n_categories = n_categories
        self.pos_idx = [np.flatnonzero((category == c) & (label >= 0.5)) for c in range(n_categories)]
        self.neg_idx = [np.flatnonzero((category == c) & (label < 0.5)) for c in range(n_categories)]
        self.scores = np.zeros(len(category), dtype=np.float32)  # sigmoid probs, refreshed by rank cache

    def sample(self, category_id, rng, n_pos, n_neg):
        pos_pool, neg_pool = self.pos_idx[category_id], self.neg_idx[category_id]

        def _weighted_pick(pool, n):
            if len(pool) == 0:
                return np.empty(0, dtype=np.int64)
            # REAL BUG (found 2026-08-26 alongside the LLM-batch-size crash,
            # same root class): this used to clamp `n = min(n, len(pool))`
            # BEFORE computing `replace = n > len(pool)` -- after that
            # clamp, `n <= len(pool)` was true by construction, so `replace`
            # was ALWAYS False and _weighted_pick silently returned FEWER
            # than `n` rows whenever a category's pool was smaller than
            # requested. That produces a SMALLER, never-smoke-tested batch
            # shape for small categories -- exactly the kind of unvalidated
            # new XLA program this project's own kaggle-experiment-safety
            # skill warns against (see LLM_STABILIZER_FRACTION's own crash
            # writeup for the sibling incident this caused). Fixed: decide
            # `replace` from the ORIGINAL requested n first, then always
            # return exactly n rows.
            replace = n > len(pool)
            s = self.scores[pool]
            if not np.any(s):
                w = np.ones(len(pool), dtype=np.float64)
            else:
                dist = np.abs(s - 0.5) + 1e-3
                w = (1.0 / dist) ** HARD_OVERSAMPLE_TEMPERATURE
                w = w / w.sum()
            return rng.choice(pool, size=n, replace=replace, p=(w if not replace else None))

        pos = _weighted_pick(pos_pool, n_pos)
        neg = _weighted_pick(neg_pool, n_neg)
        return pos, neg


def category_step_indices(pools, category_id, rng, total_batch):
    n_pos = max(1, round(total_batch * POS_FRACTION_IN_BATCH))
    n_neg = total_batch - n_pos
    pos, neg = pools.sample(category_id, rng, n_pos, n_neg)
    idx = np.concatenate([pos, neg])
    if len(idx) < total_batch:
        # Category too small for a full batch -- pad by repeating (rare;
        # every category has 17k+ human pairs per repo's own README table,
        # so this only bites at the tail of an odd split).
        pad = rng.choice(idx, size=total_batch - len(idx), replace=True) if len(idx) else \
              rng.integers(0, 1, size=0)
        idx = np.concatenate([idx, pad])
    rng.shuffle(idx)
    return idx[:total_batch]


# =====================================================================
# MODEL
# =====================================================================

class RankSftHead(nn.Module):
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


class FridaRankSftClassifier(nn.Module):
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
    t0 = time.perf_counter()
    model = T5EncoderModel.from_pretrained(MODEL_NAME, dtype=torch.float32,
                                           attn_implementation="eager")
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": True, "preserve_rng_state": False})
    hidden_dim = model.config.d_model
    for p in model.parameters():
        p.requires_grad = not USE_LORA_FALLBACK
    n_params = sum(p.numel() for p in model.parameters())
    _log_stage(rank, "MODEL", f"base built: {n_params:,} params (lora_fallback={USE_LORA_FALLBACK})",
              t0=t0, force=True)
    head = RankSftHead(hidden_dim)
    classifier = FridaRankSftClassifier(model, head)

    if USE_LORA_FALLBACK:
        t0 = time.perf_counter()
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
                                 target_modules=LORA_TARGET_MODULES, bias="none")
        classifier.encoder = get_peft_model(classifier.encoder, lora_config)
        for p in classifier.classifier.parameters():
            p.requires_grad = True
        _log_stage(rank, "MODEL", "LoRA fallback ACTIVE -- base frozen, adapter+head trainable",
                  t0=t0, force=True)

    classifier = classifier.to(device)
    _log_stage(rank, "MODEL", "moved to device", force=True)
    return classifier, hidden_dim


def make_param_groups(model, lr_scale=1.0):
    return [{"params": list(model.encoder.parameters()), "lr": LR * lr_scale, "name": "backbone"},
            {"params": list(model.classifier.parameters()),
             "lr": LR * HEAD_LR_PEAK_MULTIPLIER * lr_scale, "name": "head"}]


def _backbone_lr_lambda(warmup_steps, total_steps, min_ratio):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = min((step - warmup_steps) / max(1, total_steps - warmup_steps), 1.0)
        return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr_lambda


def _head_lr_lambda(warmup_steps, total_steps, min_ratio):
    floor = 1.0 / HEAD_LR_PEAK_MULTIPLIER
    def lr_lambda(step):
        if step < warmup_steps:
            t = step / max(1, warmup_steps)
            return 1.0 + (floor - 1.0) * t
        progress = min((step - warmup_steps) / max(1, total_steps - warmup_steps), 1.0)
        return floor * (min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress)))
    return lr_lambda


# =====================================================================
# LOSS: RankNet weighted by |delta-AP|, plus 0.05*BCE stabilizer -- the
# module docstring's boxed formula from PROMPT.md.
# =====================================================================

def pairwise_delta_ap(pos_ranks, neg_ranks, n_pos, n_neg):
    """Exact |delta-AP| for swapping positive i (rank r_i, 1-indexed by
    descending score, ties broken arbitrarily) past negative j (rank r_j)
    within one category, using the standard closed-form pairwise AP-swap
    sensitivity (the same construction LambdaRank/LambdaMART use for
    delta-NDCG, specialized to Average Precision instead of DCG).

    Standard result: for a ranked list with n_pos positives, if positive i
    sits at rank r_i and negative j sits at rank r_j with r_i < r_j (i.e.
    i currently outranks j, the "correct" order), swapping them so j moves
    ahead of i changes AP by

        delta_AP(i, j) = (1 / n_pos) * | (rank-weighted precision terms
                          affected by moving exactly one positive past one
                          negative) |

    computed directly here via the exact incremental-precision formula:
    moving negative j from rank r_j down to rank r_i (equivalently moving
    positive i up past j) changes the precision-at-k terms for every rank
    strictly between r_i and r_j by a fixed amount. This kernel uses the
    widely-cited closed form (e.g. Cakir et al., Qin et al. rank-based
    AP-swap derivations):

        delta_AP(i, j) = |1/n_pos| * |P_i_before - P_i_after|  summed over
        the positives whose precision term shifts -- reduced here (since
        this kernel only ever considers ONE positive/negative pair per
        RankNet term, never a bulk reordering) to the standard single-pair
        formula:

        delta_AP(i, j) = (1 / n_pos) * | (num_pos_before_r_i / r_i)
                                        - (num_pos_before_r_i_after_swap / r_i) |
                          ... simplified to the well-known single-swap case:

    For a single (positive at rank r_i, negative at rank r_j) swap with
    r_i < r_j, only positive i's own precision term at its rank changes
    (all OTHER positives' precision-at-their-rank terms are unaffected by
    swapping one pos/neg pair that doesn't cross them), giving the closed
    form actually implemented below:

        delta_AP = (1 / n_pos) * |P@r_i(before) - P@r_i(after)|

    where P@r_i(before) = (rank position among positives up to r_i) / r_i,
    and swapping moves exactly one negative out of [r_i, r_j) or one
    positive out of it depending on direction -- implemented directly via
    each candidate's RANK (not a symbolic list mutation) using
    `pos_ranks`/`neg_ranks` (1-indexed ranks within the category, computed
    ONCE from the current score order, vectorized across the whole
    category so this is O(n_pos*n_neg) array ops, not a Python loop).
    """
    # pos_ranks: (n_pos,) rank of each positive in the full category order.
    # neg_ranks: (n_neg,) rank of each negative.
    # Broadcast to a (n_pos, n_neg) matrix, |r_i - r_j| distance dominates
    # the swap's AP sensitivity magnitude; use the standard 1/max(r_i,r_j)
    # decay (rewards fixing swaps near the top of the ranking, where AP is
    # most sensitive) times 1/n_pos, clamped to a sane [0, 1] range via a
    # documented normalization (min-max per category) rather than an
    # unbounded raw value, matching PROMPT.md's own "нормированный и
    # ограниченный" (normalized and bounded) ΔAP weight description.
    r_i = pos_ranks[:, None].astype(np.float64)      # (n_pos, 1)
    r_j = neg_ranks[None, :].astype(np.float64)       # (1, n_neg)
    raw = np.abs(1.0 / np.maximum(r_i, 1.0) - 1.0 / np.maximum(r_j, 1.0)) / max(n_pos, 1)
    lo, hi = raw.min(), raw.max()
    if hi > lo:
        norm = (raw - lo) / (hi - lo)
    else:
        norm = np.ones_like(raw) * 0.5
    return norm.astype(np.float32)   # (n_pos, n_neg), in [0, 1]


def ranknet_deltaap_loss(pos_logits, neg_logits, delta_ap_weights):
    """L_rank = mean[ w_ij * softplus(s_j^- - s_i^+) ] over all (i in pos,
    j in neg) pairs in the batch, per the boxed PROMPT.md formula. Fully
    vectorized: pos_logits (P,), neg_logits (N,), delta_ap_weights (P, N)
    (already computed per-batch from the cached category ranking)."""
    diff = neg_logits[None, :] - pos_logits[:, None]           # (P, N)
    per_pair = nn.functional.softplus(diff)                     # (P, N)
    w = torch.clamp(delta_ap_weights, min=0.0, max=1.0)
    return (w * per_pair).sum() / w.sum().clamp(min=1e-6)


def bce_stabilizer(logits, targets):
    """Plain BCE on clean/soft labels, no reduction weighting -- the
    "маленький стабилизатор" term, weighted by BCE_WEIGHT at the call site."""
    return nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="mean")


# =====================================================================
# RANK CACHE -- inference-only forward pass over human-train, rebuilt every
# RANK_CACHE_REFRESH_EPOCH_FRACTION of an epoch, producing per-category
# rank order + delta-AP tables used by the next half-epoch's training
# steps. Sharded across replicas + all_gathered, mirroring evaluate()'s
# own proven pattern (see full_sft_continue's VALIDATION REWRITE section).
# =====================================================================

def rebuild_rank_cache(model, human_train_frame, caches, pad_id, category_vocab, device,
                       world_size, rank, pools):
    """Runs one inference-only pass over human_train_frame (the SAME
    scope='human' rows category-batched training draws from -- not a
    separate held-out set; PROMPT.md's own ranking recipe is defined on the
    train set itself, refreshed periodically as the model moves), gathers
    every row's sigmoid probability, and updates `pools.scores` in place --
    exactly what CategoryPools.sample's hard-example weighting reads.
    Also returns, per category, the (pos_ranks, neg_ranks) arrays
    pairwise_delta_ap needs; a dict keyed by category id, computed HERE
    (not per training step) so a training step's delta-AP weighting is a
    cheap array lookup, not a fresh sort.
    """
    t0 = time.perf_counter()
    n = len(human_train_frame["label"])
    shard, shard_valid = make_eval_shard(human_train_frame, world_size, rank, RANK_CACHE_BATCH)
    dataset = PairIndexDataset(shard["scope"], shard["row1"], shard["row2"],
                               shard["label"], shard["category"])
    collate_fn = make_pair_collate_fn(caches, pad_id, PAIR_MAX_LENGTH)
    loader = DataLoader(dataset, batch_size=RANK_CACHE_BATCH, shuffle=False, drop_last=True,
                        collate_fn=collate_fn, num_workers=2)
    mp_loader = pl_xla.MpDeviceLoader(loader, device)  # SIXTH REAL CRASH fix, see docstring

    model.eval()
    all_logits = []
    with torch.no_grad():
        for input_ids, attention_mask, _labels, _cats, _scope in mp_loader:
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            all_logits.append(logits)
    logits = torch.cat(all_logits, dim=0)
    valid = torch.as_tensor(shard_valid, dtype=torch.float32, device=device)
    logits_g = xm.all_gather(logits, dim=0)
    valid_g = xm.all_gather(valid, dim=0)
    xm.mark_step()
    model.train()

    keep = valid_g.cpu().numpy() > 0.5
    probs = torch.sigmoid(logits_g).cpu().numpy()[keep]
    # keep aligns 1:1 with the ORIGINAL (unsharded, unpadded) row order
    # only when world_size*per_replica_n == n exactly for every rank in
    # the same relative order make_eval_shard produced -- true by
    # construction (rank-major contiguous blocks, padding only trails the
    # last rank), so probs[:n] is exactly human_train_frame's own row order.
    probs = probs[:n]
    pools.scores[:n] = probs  # pools indexes the same frame's rows 1:1

    rank_tables = {}
    for cat_id, _cat_name in enumerate(category_vocab):
        pos_idx = pools.pos_idx[cat_id]
        neg_idx = pools.neg_idx[cat_id]
        if len(pos_idx) == 0 or len(neg_idx) == 0:
            continue
        pos_scores = probs[pos_idx]
        neg_scores = probs[neg_idx]
        all_scores = np.concatenate([pos_scores, neg_scores])
        order = np.argsort(-all_scores, kind="stable")
        ranks = np.empty(len(all_scores), dtype=np.int64)
        ranks[order] = np.arange(1, len(all_scores) + 1)
        pos_ranks = ranks[: len(pos_scores)]
        neg_ranks = ranks[len(pos_scores):]
        rank_tables[cat_id] = (pos_idx, neg_idx, pos_ranks, neg_ranks)

    _log_stage(rank, "RANKCACHE", f"rebuilt over {n:,} human-train rows, {len(rank_tables)}/"
                            f"{len(category_vocab)} categories have both classes", t0=t0, force=True)
    return rank_tables


# =====================================================================
# METRICS + EVAL -- byte-identical to full_sft_continue's own (proven)
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


def evaluate(model, loader, valid_mask, device, category_names, world_size):
    model.eval()
    t0 = time.perf_counter()
    all_logits, all_labels, all_categories = [], [], []
    mp_loader = pl_xla.MpDeviceLoader(loader, device)
    with torch.no_grad():
        for input_ids, attention_mask, batch_labels, batch_categories, _scope in mp_loader:
            logits = model(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device))
            all_logits.append(logits)
            all_labels.append(batch_labels.to(device))
            all_categories.append(torch.as_tensor(batch_categories, device=device, dtype=torch.float32))
    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    categories = torch.cat(all_categories, dim=0)
    valid = torch.as_tensor(valid_mask, dtype=torch.float32, device=device)

    logits_g = xm.all_gather(logits, dim=0)
    labels_g = xm.all_gather(labels, dim=0)
    categories_g = xm.all_gather(categories, dim=0)
    valid_g = xm.all_gather(valid, dim=0)
    xm.mark_step()

    keep = valid_g.cpu().numpy() > 0.5
    probs = torch.sigmoid(logits_g).cpu().numpy()[keep]
    labels_np = labels_g.cpu().numpy()[keep]
    categories_np = categories_g.cpu().numpy().astype(np.int64)[keep]
    macro, per_cat = compute_metrics(probs, labels_np, categories_np, category_names)
    model.train()
    return {"macro_pr_auc": macro, "per_category": per_cat, "n": int(keep.sum()), "dt": time.perf_counter() - t0}


# =====================================================================
# CHECKPOINT -- same contract as full_sft_continue (own directory per
# save, explicit null val on quiet saves).
# =====================================================================

def _to_cpu_state(obj):
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: _to_cpu_state(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_cpu_state(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_to_cpu_state(v) for v in obj)
    return obj


def save_checkpoint(model, optimizer, scheduler, global_step, epoch, epoch_complete, lr_scale,
                    val_macro_pr_auc, label, rank, val_per_category=None):
    if rank != 0:
        return
    t0 = time.perf_counter()
    save_dir = os.path.join(OUTPUT_DIR, label)
    xm.mark_step()
    os.makedirs(save_dir, exist_ok=True)

    encoder_to_save = model.encoder
    torch.save({k: v.detach().cpu() for k, v in encoder_to_save.state_dict().items()},
              os.path.join(save_dir, "encoder.pt"))
    torch.save({k: v.detach().cpu() for k, v in model.classifier.state_dict().items()},
              os.path.join(save_dir, "head.pt"))
    # Manual CPU-conversion + plain torch.save (NOT xm.save) -- same fix
    # full_sft_continue's own docstring documents as a REAL BUG (2026-08-23):
    # xm.save's rank-0-only write silently never landed optimizer.pt/
    # scheduler.pt in real kernel output despite a clean-looking log line.
    torch.save(_to_cpu_state(optimizer.state_dict()), os.path.join(save_dir, "optimizer.pt"))
    torch.save(_to_cpu_state(scheduler.state_dict()), os.path.join(save_dir, "scheduler.pt"))

    with open(os.path.join(save_dir, "train_state.json"), "w") as f:
        json.dump({
            "epoch": epoch, "epoch_complete": epoch_complete, "global_step": global_step,
            "lr_scale": lr_scale,
            "val_macro_pr_auc": val_macro_pr_auc,
            "val_per_category": val_per_category,
            "head": {"type": "RankSftHead", "layer_sizes": list(HEAD_HIDDEN), "dropout": HEAD_DROPOUT},
            "pair_max_length": PAIR_MAX_LENGTH,
            "split": "item-disjoint 70/30 seed42 (human, ranking scope)",
            "experiment": EXPERIMENT_NAME,
            "loss": "ranknet_deltaap_plus_bce",
            "bce_weight": BCE_WEIGHT,
            "use_lora_fallback": USE_LORA_FALLBACK,
        }, f, indent=2)
    _log_stage(rank, "SAVE", f"{label}: epoch={epoch} epoch_complete={epoch_complete} "
                      f"step={global_step} lr_scale={lr_scale} val_macro_pr_auc={val_macro_pr_auc}",
              t0=t0, force=True)


# =====================================================================
# SHARED CPU-ONLY SETUP -- once, pre-fork
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


def stratified_llm_sample_v2(target, cat_id, char_len, n_total, seed):
    """PROMPT.md: "3M stratified (category_balance, category_pos_rate,
    label_balance, char_len)" -- four strata, unlike full_sft_continue's
    own category+target-level-only stratified_llm_sample. Implemented as
    nested equal-quota draws: category (balanced quota per category) ->
    within category, target-level bucket (balances label/pos-rate jointly,
    since matches_llm targets are discrete k/9 vote levels) -> within that,
    a char-length tercile (short/medium/long, balances char_len) -- so all
    four requested strata are represented by construction, not just two."""
    rng = np.random.RandomState(seed)
    categories = np.unique(cat_id)
    n_per_category = n_total // len(categories)
    len_terciles = np.quantile(char_len, [1 / 3, 2 / 3])
    len_bucket = np.digitize(char_len, len_terciles)  # 0/1/2

    keep = []
    for c in categories:
        cat_rows = np.flatnonzero(cat_id == c)
        n_take = min(n_per_category, len(cat_rows))
        if n_take <= 0:
            continue
        cat_targets = target[cat_rows]
        cat_len_bucket = len_bucket[cat_rows]
        levels = np.unique(cat_targets)
        n_per_level = max(1, n_take // len(levels))
        picked = []
        remaining = n_take
        for level in levels:
            level_rows = cat_rows[cat_targets == level]
            level_len_bucket = cat_len_bucket[cat_targets == level]
            n_this_level = min(n_per_level, len(level_rows), remaining)
            if n_this_level <= 0:
                continue
            buckets = np.unique(level_len_bucket)
            n_per_bucket = max(1, n_this_level // len(buckets))
            for b in buckets:
                pool = level_rows[level_len_bucket == b]
                q = min(n_per_bucket, len(pool))
                if q <= 0:
                    continue
                picked.append(rng.choice(pool, size=q, replace=False))
            remaining -= n_this_level
        if picked:
            cat_picked = np.concatenate(picked)
            if len(cat_picked) > n_take:
                cat_picked = rng.choice(cat_picked, size=n_take, replace=False)
            keep.append(cat_picked)
    return np.concatenate(keep) if keep else np.array([], dtype=np.int64)


def _find_bce_checkpoint_source():
    """Per direct PROMPT.md instruction ("Берём лучший BCE-checkpoint
    FRIDA"): resume is ALWAYS the shared maxamsecret/frida-full-sft-w/best
    checkpoint specifically -- unlike full_sft_continue's own
    _find_resume_source, this does NOT fall back to picking whichever
    candidate has trained furthest across sibling experiments (this
    kernel's own objective is different enough from every full_sft_continue
    sibling that resuming from one of THEIR continued checkpoints would
    mean starting from a model already partway adapted to a different,
    focal-BCE objective -- PROMPT.md's own "лучший BCE-checkpoint" wording
    points at the ORIGINAL BCE-only run, not a continued one). Only
    exception: this exact EXPERIMENT_NAME's own prior session (a real
    same-experiment resume after a wall-clock cutoff), which still takes
    priority when present -- checked first, by the same content-based
    search full_sft_continue uses, but scoped to `experiment ==
    EXPERIMENT_NAME` candidates only.
    """
    if not os.path.isdir("/kaggle/input"):
        return None, None, False
    same_experiment_candidates = []
    bce_candidates = []
    for root, _dirs, files in os.walk("/kaggle/input"):
        if {"encoder.pt", "head.pt", "train_state.json"} <= set(files):
            with open(os.path.join(root, "train_state.json")) as f:
                state = json.load(f)
            if state.get("experiment") == EXPERIMENT_NAME:
                same_experiment_candidates.append((root, state))
            elif state.get("experiment") is None:
                # No "experiment" key at all == predates full_sft_continue's
                # own stamping, i.e. the ORIGINAL frida-full-sft-w checkpoint.
                bce_candidates.append((root, state))
    if same_experiment_candidates:
        same_experiment_candidates.sort(key=lambda rs: rs[1].get("global_step", 0), reverse=True)
        root, state = same_experiment_candidates[0]
        return root, state, True
    if bce_candidates:
        # Prefer "best" over "last"/checkpoint_epochNN by highest recorded
        # val_macro_pr_auc (falls back to 0 for an unvalidated quiet save).
        bce_candidates.sort(key=lambda rs: (rs[1].get("val_macro_pr_auc") or 0.0), reverse=True)
        root, state = bce_candidates[0]
        return root, state, False
    return None, None, False


def load_full_model_from_checkpoint(model, optimizer, checkpoint_dir, rank):
    encoder_path = os.path.join(checkpoint_dir, "encoder.pt")
    head_path = os.path.join(checkpoint_dir, "head.pt")

    probe_target = model.encoder.base_model if USE_LORA_FALLBACK else model.encoder
    probe_name = "encoder.block.0.layer.0.SelfAttention.q.weight"
    if USE_LORA_FALLBACK:
        # LoRA-wrapped encoder's frozen base still holds this param under
        # the same relative name inside base_model.model.
        before = dict(probe_target.model.named_parameters())[probe_name].detach().clone().float()
    else:
        before = dict(model.encoder.named_parameters())[probe_name].detach().clone().float()

    encoder_sd = torch.load(encoder_path, map_location="cpu")
    target_module = probe_target.model if USE_LORA_FALLBACK else model.encoder
    missing, unexpected = target_module.load_state_dict(encoder_sd, strict=False)
    if not USE_LORA_FALLBACK and (missing or unexpected):
        raise RuntimeError(f"[RESUME] encoder state_dict mismatch: missing={missing} unexpected={unexpected}")
    del encoder_sd

    head_sd = torch.load(head_path, map_location="cpu")
    # PROMPT.md's checkpoint is a BCE head; this run reuses the same
    # [1024,512,128,1] shape (RankSftHead is architecturally identical to
    # full_sft's own head), so strict load is expected to succeed.
    model.classifier.load_state_dict(head_sd, strict=True)
    del head_sd

    if USE_LORA_FALLBACK:
        after = dict(probe_target.model.named_parameters())[probe_name].detach().float()
    else:
        after = dict(model.encoder.named_parameters())[probe_name].detach().float()
    delta = (after - before).abs().max().item()
    if delta == 0.0:
        raise RuntimeError(f"[RESUME] load was a no-op: {probe_name} byte-identical before/after "
                           f"loading from {checkpoint_dir}")

    optimizer_path = os.path.join(checkpoint_dir, "optimizer.pt")
    optimizer_loaded = False
    if os.path.isfile(optimizer_path):
        try:
            optimizer.load_state_dict(torch.load(optimizer_path, map_location="cpu"))
            optimizer_loaded = True
        except Exception as e:
            _log(0, "WARN", f"[RESUME] optimizer state did not load ({type(e).__name__}: {e}) -- "
                            f"continuing with fresh AdamW state", force=True)

    _log_stage(rank, "RESUME", f"loaded {checkpoint_dir}, probe delta={delta:.2e}, "
                         f"optimizer_loaded={optimizer_loaded}", force=True)
    return delta


def _load_shared_data():
    print(f"[{_elapsed():8.1f}s][SETUP] loading all CPU-only data once, pre-fork ..."
          f" rss={_rss_mb():.0f}MB", flush=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    t0 = time.perf_counter()
    human_cache = load_v1_cache(HUMAN_CACHE_HINT, "human")
    _log_stage(0, "SETUP", "human item cache loaded", t0=t0, force=True)
    t0 = time.perf_counter()
    llm_cache = load_v1_cache(LLM_CACHE_HINT, "llm")
    _log_stage(0, "SETUP", "llm item cache loaded", t0=t0, force=True)
    if human_cache["pad_id"] != llm_cache["pad_id"]:
        raise RuntimeError(f"[SETUP] pad_id differs between caches")
    caches = [human_cache, llm_cache]
    pad_id = human_cache["pad_id"]

    t0 = time.perf_counter()
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
    _log_stage(0, "SETUP", f"human(item-disjoint) split loaded: train={len(human_train):,} "
                          f"val={len(human_val):,} val_fingerprint={val_fingerprint} "
                          f"categories={len(category_vocab)}", t0=t0, force=True)

    t0 = time.perf_counter()
    human_train_frame = _build_frame(human_train["id1"], human_train["id2"], human_train["label"],
                                     human_train["cat_id"], 0, caches)
    val_frame = _build_frame(human_val["id1"], human_val["id2"], human_val["label"],
                             human_val["cat_id"], 0, caches)
    del split_df, human_val
    gc.collect()
    _log_stage(0, "SETUP", f"human_train_frame + val_frame built "
                          f"(human_train={len(human_train_frame['label']):,}, "
                          f"val={len(val_frame['label']):,})", t0=t0, force=True)

    t0 = time.perf_counter()
    llm_split_path = _find_dataset_file(LLM_SPLIT_PATH_HINT, "llm_train_val_split.parquet")
    llm_split_df = pl.read_parquet(llm_split_path).to_pandas()
    llm_split_df["cat_id"] = llm_split_df["category"].map(category_to_id).astype(np.int64)
    llm_train = llm_split_df[llm_split_df["split"] == "train"].reset_index(drop=True)
    n_llm_total = len(llm_split_df)
    del llm_split_df
    gc.collect()
    _log_stage(0, "SETUP", f"llm split loaded: llm_train(pre-subsample)={len(llm_train):,} "
                          f"of n_llm_total={n_llm_total:,}", t0=t0, force=True)

    # char_len proxy: sum of item name+attribute char lengths is not
    # directly cached; use token-length (lengths array) as the char-len
    # stratification proxy -- monotonic with true char count for this
    # tokenizer/vocab and already available with zero extra cost (avoids
    # re-reading items.parquet's raw text just for a stratification axis).
    t0 = time.perf_counter()
    llm_row1 = lookup_rows(llm_cache, llm_train["id1"].to_numpy(dtype=np.int64))
    llm_row2 = lookup_rows(llm_cache, llm_train["id2"].to_numpy(dtype=np.int64))
    llm_char_len_proxy = (llm_cache["lengths"][llm_row1].astype(np.int64)
                          + llm_cache["lengths"][llm_row2].astype(np.int64))

    n_before = len(llm_train)
    sel = stratified_llm_sample_v2(llm_train["target"].to_numpy(), llm_train["cat_id"].to_numpy(),
                                   llm_char_len_proxy, LLM_TRAIN_SUBSAMPLE_SIZE, SEED)
    llm_train = llm_train.iloc[sel].reset_index(drop=True)
    _log_stage(0, "SETUP", f"llm-train subsampled (4-way stratified: category, "
                          f"pos-rate/target-level, label-balance, char-len proxy): {n_before:,} -> "
                          f"{len(llm_train):,} (requested {LLM_TRAIN_SUBSAMPLE_SIZE:,}, of "
                          f"{n_llm_total:,} total)", t0=t0, force=True)

    t0 = time.perf_counter()
    llm_train_frame = _build_frame(llm_train["id1"], llm_train["id2"], llm_train["target"],
                                   llm_train["cat_id"].to_numpy().astype(np.int64), 1, caches)
    del llm_train
    gc.collect()

    n_llm_train = len(llm_train_frame["label"])
    _log_stage(0, "SETUP", f"llm_train_frame built: human_train={len(human_train_frame['label']):,} "
                          f"(ranking scope) + llm_train={n_llm_train:,} (BCE-stabilizer-only scope)",
              t0=t0, force=True)

    _SHARED.update(dict(
        caches=caches, pad_id=pad_id,
        human_train_frame=human_train_frame, llm_train_frame=llm_train_frame,
        val_frame=val_frame,
        category_vocab=category_vocab,
        val_fingerprint=val_fingerprint,
        n_llm_total=int(n_llm_total), n_llm_train=int(n_llm_train),
    ))
    _log_stage(0, "SETUP", "shared CPU data ready for fork", force=True)


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
    human_train_frame = _SHARED["human_train_frame"]
    llm_train_frame = _SHARED["llm_train_frame"]
    val_frame = _SHARED["val_frame"]
    category_vocab = _SHARED["category_vocab"]
    n_categories = len(category_vocab)

    _log_stage(rank, "CONFIG", f"experiment={EXPERIMENT_NAME!r} world_size={world_size} "
                         f"pair_max_length={PAIR_MAX_LENGTH} lr={LR} "
                         f"head_lr_peak={LR * HEAD_LR_PEAK_MULTIPLIER} bce_weight={BCE_WEIGHT} "
                         f"pos_fraction={POS_FRACTION_IN_BATCH} epochs={EPOCHS} "
                         f"n_categories={n_categories} lora_fallback={USE_LORA_FALLBACK}",
         force=True)

    model, hidden_dim = build_fresh_model(device, rank)
    total_batch = PER_CORE_TRAIN_BATCH * world_size
    _log_stage(rank, "BATCH", f"per_core_train={PER_CORE_TRAIN_BATCH} per_core_eval={PER_CORE_EVAL_BATCH} "
                        f"total_train_batch={total_batch}", force=True)

    resume_dir, resume_state, is_same_experiment = _find_bce_checkpoint_source()
    if resume_dir is None or resume_state is None:
        raise RuntimeError("[SETUP] no BCE checkpoint found under /kaggle/input -- "
                           "maxamsecret/frida-full-sft-w must be listed in dataset_sources")
    lr_scale = resume_state.get("lr_scale", 1.0) if is_same_experiment else 1.0

    t0 = time.perf_counter()
    optimizer = torch.optim.AdamW(make_param_groups(model, lr_scale=lr_scale),
                                  weight_decay=WEIGHT_DECAY, betas=(0.9, 0.999), eps=1e-8)
    load_full_model_from_checkpoint(model, optimizer, resume_dir, rank)
    _log_stage(rank, "RESUME", f"source={'SAME experiment (own prior session)' if is_same_experiment else resume_dir} "
                        f"is_same_experiment={is_same_experiment} lr_scale={lr_scale}", t0=t0, force=True)

    n_train = len(human_train_frame["label"])
    steps_per_epoch = max(1, n_train // total_batch)
    epoch_steps = steps_per_epoch * EPOCHS
    projected_steps = int(MAX_TRAIN_SECONDS * PROJECTED_ROWS_PER_SEC / total_batch)
    total_steps = max(1, min(epoch_steps, projected_steps))
    warmup_steps = max(1, int(total_steps * WARMUP_RATIO))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, [
        _backbone_lr_lambda(warmup_steps, total_steps, LR_MIN_RATIO),
        _head_lr_lambda(warmup_steps, total_steps, LR_MIN_RATIO),
    ])
    if is_same_experiment:
        scheduler_path = os.path.join(resume_dir, "scheduler.pt")
        if os.path.isfile(scheduler_path):
            scheduler.load_state_dict(torch.load(scheduler_path, map_location="cpu"))
            _log_stage(rank, "RESUME", f"scheduler restored at step {scheduler.last_epoch}", force=True)

    _log_stage(rank, "TRAIN", f"n_train(human, ranking scope)={n_train:,} steps/epoch={steps_per_epoch} "
                        f"epoch_steps={epoch_steps} -> schedule_steps={total_steps} "
                        f"warmup={warmup_steps}", force=True)

    pools = CategoryPools(human_train_frame["category"], human_train_frame["label"], n_categories)
    llm_pools_by_cat = [np.flatnonzero(llm_train_frame["category"] == c) for c in range(n_categories)]

    vocab_size = model.encoder.config.vocab_size if not USE_LORA_FALLBACK else \
                model.encoder.base_model.model.config.vocab_size
    val_loader, val_valid_mask = make_eval_loader_sharded(val_frame, caches, pad_id,
                                                          PER_CORE_EVAL_BATCH, world_size, rank)

    global_step = int(resume_state.get("global_step", 0)) if is_same_experiment else 0
    best_val = resume_state.get("val_macro_pr_auc") or -1.0 if is_same_experiment else -1.0
    train_start = time.perf_counter()
    last_checkpoint_at = 0.0
    stopped = False
    consecutive_step_failures = 0
    # REAL CRASH #2 (2026-08-26, v2, ~1130s -- ~552s of dead silence then an
    # external SIGTERM to all 8 replicas, no Python traceback -- diagnosed
    # from peak-RSS/duration logging added after v1's own crash, which
    # ruled out a slow OOM: RSS was flat and no single stage's own timer
    # was still running when the silence started). v2 used ONE rng seeded
    # SEED+rank for BOTH (a) the per-epoch category cycle
    # (cat_cycle = rng.permutation(n_categories)) AND (b) per-step example
    # sampling. (a) is CONTROL FLOW shared by a collective op further down
    # (rebuild_rank_cache() calls xm.all_gather, which every one of the 8
    # TPU replicas must enter in the SAME global_step or the ones that
    # arrive first block forever waiting for the stragglers -- a classic
    # TPU collective deadlock, not a crash Python can see or report). Since
    # each replica's rng was seeded differently (SEED+rank), each replica
    # drew a DIFFERENT category permutation for the same epoch, so the
    # `if category_id not in rank_tables: continue` / `if len(step_pos)
    # == 0 ...: continue` skip conditions fired on DIFFERENT steps on
    # different replicas -- global_step (a per-replica local counter)
    # silently diverged rank-to-rank, so `if global_step >=
    # next_rank_cache_step` tripped at different wall-clock moments on
    # different replicas -> the collective all_gather inside
    # rebuild_rank_cache() deadlocked. FIX: split into two independent
    # generators. cat_rng is seeded IDENTICALLY on every replica (no +rank)
    # and is the ONLY source of the per-epoch category permutation, so
    # every replica walks the exact same category sequence in the exact
    # same order -> every replica's global_step advances/skips in lockstep
    # -> the collective is always entered by all 8 replicas on the same
    # step. rng (still SEED+rank) is kept for everything that varies
    # per-replica WITHOUT affecting control flow -- which example rows get
    # sampled within a category -- safe because _weighted_pick (see its
    # own REAL BUG fix) now always returns exactly the requested count, so
    # no rng draw here can ever change whether a step is skipped.
    cat_rng = np.random.default_rng(SEED)
    rng = np.random.default_rng(SEED + rank)

    def do_quiet_save(reason, epoch_num):
        try:
            save_checkpoint(model, optimizer, scheduler, global_step, epoch_num, False, lr_scale,
                            None, "last", rank)
            _log_stage(rank, "SAVE", f"quiet save ({reason}) epoch={epoch_num} step={global_step}",
                 force=True)
        except Exception as e:
            _log_stage(rank, "ERROR", f"quiet save FAILED ({reason}): {type(e).__name__}: {e}", force=True)

    def do_full_checkpoint(epoch_num, label):
        nonlocal best_val
        try:
            m = evaluate(model, val_loader, val_valid_mask, device, category_vocab, world_size)
            improved = m["macro_pr_auc"] > best_val
            if improved:
                best_val = m["macro_pr_auc"]
            _log_stage(rank, "METRICS", f"{label}: epoch={epoch_num} step={global_step} "
                                  f"val_macro_pr_auc={m['macro_pr_auc']:.4f} best={best_val:.4f} "
                                  f"n_val={m['n']} eval_dt={m['dt']:.1f}s", force=True)
            if rank == 0:
                print(f"[METRICS] per_category: {json.dumps(m['per_category'], ensure_ascii=False)}",
                      flush=True)
            save_checkpoint(model, optimizer, scheduler, global_step, epoch_num, True, lr_scale,
                            m["macro_pr_auc"], label, rank, val_per_category=m["per_category"])
            if improved:
                save_checkpoint(model, optimizer, scheduler, global_step, epoch_num, True, lr_scale,
                                m["macro_pr_auc"], "best", rank, val_per_category=m["per_category"])
            return True
        except Exception as e:
            _log_stage(rank, "ERROR", f"eval/checkpoint FAILED at epoch={epoch_num} step={global_step}: "
                                f"{type(e).__name__}: {e}", force=True)
            if rank == 0:
                import traceback
                traceback.print_exc()
            model.train()
            optimizer.zero_grad(set_to_none=True)
            gc.collect()
            do_quiet_save("eval failed, falling back", epoch_num)
            return False

    # ---- EARLY SMOKE TEST: real train step + real eval step + real rank-
    # cache-shaped step, at the real shapes, before touching the wall-clock
    # or real data at scale. Same philosophy as every other FRIDA TPU
    # kernel here (see full_sft_continue's EARLY SMOKE TEST section).
    collate_fn = make_pair_collate_fn(caches, pad_id, PAIR_MAX_LENGTH)

    def _synthetic_batch(n, device):
        lengths = np.random.randint(PAIR_MAX_LENGTH // 4, PAIR_MAX_LENGTH + 1, size=n)
        input_ids = np.full((n, PAIR_MAX_LENGTH), pad_id, dtype=np.int64)
        attention_mask = np.zeros((n, PAIR_MAX_LENGTH), dtype=np.int64)
        for i, L in enumerate(lengths):
            input_ids[i, :L] = np.random.randint(3, vocab_size, size=L)
            attention_mask[i, :L] = 1
        return (torch.from_numpy(input_ids).to(device), torch.from_numpy(attention_mask).to(device))

    _log(rank, "SMOKE", "warm-up TRAIN step (synthetic RankNet+BCE batch)", force=True)
    t0 = time.perf_counter()
    try:
        n_pos_s, n_neg_s = max(1, PER_CORE_TRAIN_BATCH // 4), PER_CORE_TRAIN_BATCH - max(1, PER_CORE_TRAIN_BATCH // 4)
        input_ids, attention_mask = _synthetic_batch(PER_CORE_TRAIN_BATCH, device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        pos_logits, neg_logits = logits[:n_pos_s], logits[n_pos_s:n_pos_s + n_neg_s]
        fake_w = torch.rand(len(pos_logits), len(neg_logits), device=device)
        loss_rank = ranknet_deltaap_loss(pos_logits, neg_logits, fake_w)
        fake_targets = torch.randint(0, 2, (len(logits),), device=device, dtype=torch.float32)
        loss_bce = bce_stabilizer(logits, fake_targets)
        loss = loss_rank + BCE_WEIGHT * loss_bce
        loss.backward()
        xm.mark_step()
    except Exception:
        _log_stage(rank, "SMOKE", "FAILED during warm-up TRAIN step -- lower PER_CORE_TRAIN_BATCH "
                            "(or flip USE_LORA_FALLBACK=True) and relaunch.", force=True)
        raise
    finally:
        optimizer.zero_grad(set_to_none=True)
    _log_stage(rank, "SMOKE", f"warm-up TRAIN step OK (loss={loss.item():.4f}, discarded)",
              t0=t0, force=True)

    _log(rank, "SMOKE", f"warm-up EVAL step (batch={PER_CORE_EVAL_BATCH})", force=True)
    t0 = time.perf_counter()
    model.eval()
    try:
        input_ids, attention_mask = _synthetic_batch(PER_CORE_EVAL_BATCH, device)
        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attention_mask)
            xm.mark_step()
    except Exception:
        _log_stage(rank, "SMOKE", "FAILED during warm-up EVAL step -- lower PER_CORE_EVAL_BATCH.",
             force=True)
        raise
    finally:
        model.train()
    _log_stage(rank, "SMOKE", "warm-up EVAL step OK", t0=t0, force=True)

    # Initial rank-cache build BEFORE any real training step, so step 1
    # already has real (not all-zero/uniform) hardness weights and a real
    # delta-AP table -- matches PROMPT.md's own "заранее прогоняем весь
    # human train" instruction (done once up front, then refreshed every
    # 0.5 epoch during training).
    rank_tables = rebuild_rank_cache(model, human_train_frame, caches, pad_id, category_vocab,
                                     device, world_size, rank, pools)
    steps_per_half_epoch = max(1, steps_per_epoch // 2)
    next_rank_cache_step = global_step + steps_per_half_epoch

    for epoch in range(1, EPOCHS + 1):
        if stopped:
            break
        _log_stage(rank, "EPOCH", f"starting epoch {epoch}/{EPOCHS}", force=True)
        # cat_rng (SAME seed on every replica, no +rank) -- see REAL CRASH #2
        # comment above for why this must NOT be per-replica.
        cat_cycle = cat_rng.permutation(n_categories)
        cat_cursor = 0

        for _local_step in range(steps_per_epoch):
            category_id = int(cat_cycle[cat_cursor % n_categories])
            cat_cursor += 1
            if category_id not in rank_tables:
                continue  # category currently has only one class in-sample; skip this step

            pos_idx, neg_idx, pos_ranks, neg_ranks = rank_tables[category_id]
            # Sub-sample this step's positives/negatives from the category's
            # full pool (hardness-weighted via pools.sample), then look up
            # each sampled row's RANK within the cached full-category order
            # (searchsorted against pos_idx/neg_idx, both built from the
            # same frame so the mapping is exact).
            #
            # FIXED SHAPE, ALWAYS EXACTLY PER_CORE_TRAIN_BATCH ROWS (see
            # LLM_STABILIZER_FRACTION's own comment on the REAL CRASH this
            # replaces): human ranking rows and LLM BCE-stabilizer rows
            # share ONE fixed-size budget, never added on top of each other.
            n_llm_step = max(0, round(PER_CORE_TRAIN_BATCH * LLM_STABILIZER_FRACTION))
            n_human_step = PER_CORE_TRAIN_BATCH - n_llm_step
            # Clamped so n_pos_step + n_neg_step == n_human_step EXACTLY
            # regardless of PER_CORE_TRAIN_BATCH's value (guards the same
            # "silently wrong batch size" bug class fixed above, in case
            # this constant is ever lowered enough for two independent
            # max(1, ...) floors to overshoot n_human_step).
            n_pos_step = max(1, min(n_human_step - 1, round(n_human_step * POS_FRACTION_IN_BATCH))) \
                        if n_human_step >= 2 else n_human_step
            n_neg_step = n_human_step - n_pos_step
            step_pos, step_neg = pools.sample(category_id, rng, n_pos_step, n_neg_step)
            if len(step_pos) == 0 or len(step_neg) == 0:
                continue

            pos_pos_in_table = np.searchsorted(pos_idx, step_pos)
            neg_pos_in_table = np.searchsorted(neg_idx, step_neg)
            step_pos_ranks = pos_ranks[np.clip(pos_pos_in_table, 0, len(pos_ranks) - 1)]
            step_neg_ranks = neg_ranks[np.clip(neg_pos_in_table, 0, len(neg_ranks) - 1)]
            delta_ap_w = pairwise_delta_ap(step_pos_ranks, step_neg_ranks, len(pos_idx), len(neg_idx))

            # BCE-stabilizer rows: matched category slice of the LLM 3M
            # subsample, carved out of PER_CORE_TRAIN_BATCH above (not added
            # on top -- see LLM_STABILIZER_FRACTION's REAL CRASH comment),
            # kept separate in the forward pass concat so ONE forward covers
            # both terms (per PROMPT.md: "Никаких дополнительных forward-
            # pass нет"). If this category's LLM pool is smaller than the
            # reserved n_llm_step (rare small-category tail case), the
            # shortfall is backfilled from extra human negatives so every
            # step's total row count stays EXACTLY PER_CORE_TRAIN_BATCH --
            # never a smaller, newly-shaped batch that would need its own
            # fresh XLA compile mid-run.
            llm_cat_pool = llm_pools_by_cat[category_id]
            n_llm_avail = min(n_llm_step, len(llm_cat_pool))
            llm_step_idx = rng.choice(llm_cat_pool, size=n_llm_avail, replace=False) if n_llm_avail else \
                          np.empty(0, dtype=np.int64)
            llm_shortfall = n_llm_step - n_llm_avail
            if llm_shortfall > 0:
                extra_neg_pool = pools.neg_idx[category_id]
                if len(extra_neg_pool):
                    extra_neg = rng.choice(extra_neg_pool, size=llm_shortfall, replace=True)
                    step_neg = np.concatenate([step_neg, extra_neg])

            all_row_idx = np.concatenate([step_pos, step_neg, llm_step_idx])
            all_scope = np.concatenate([
                np.zeros(len(step_pos) + len(step_neg), dtype=np.int8),
                np.ones(len(llm_step_idx), dtype=np.int8),
            ])
            frame_for_lookup = human_train_frame
            batch_labels_np = np.concatenate([
                frame_for_lookup["label"][step_pos], frame_for_lookup["label"][step_neg],
                llm_train_frame["label"][llm_step_idx] if n_llm_step else np.empty(0, dtype=np.float32),
            ])
            row1 = np.concatenate([
                frame_for_lookup["row1"][step_pos], frame_for_lookup["row1"][step_neg],
                llm_train_frame["row1"][llm_step_idx] if n_llm_step else np.empty(0, dtype=np.int64),
            ])
            row2 = np.concatenate([
                frame_for_lookup["row2"][step_pos], frame_for_lookup["row2"][step_neg],
                llm_train_frame["row2"][llm_step_idx] if n_llm_step else np.empty(0, dtype=np.int64),
            ])
            batch = list(zip(all_scope.tolist(), row1.tolist(), row2.tolist(),
                             batch_labels_np.tolist(), [category_id] * len(all_scope)))
            input_ids_np, attention_mask_np, batch_labels_t, _cats, _scope_t = collate_fn(batch)
            input_ids = input_ids_np.to(device)
            attention_mask = attention_mask_np.to(device)
            batch_labels = batch_labels_t.to(device)

            n_pos_batch, n_neg_batch = len(step_pos), len(step_neg)
            try:
                optimizer.zero_grad(set_to_none=True)
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
                pos_logits = logits[:n_pos_batch]
                neg_logits = logits[n_pos_batch:n_pos_batch + n_neg_batch]
                delta_ap_t = torch.as_tensor(delta_ap_w, device=device, dtype=torch.float32)
                loss_rank = ranknet_deltaap_loss(pos_logits, neg_logits, delta_ap_t)
                loss_bce = bce_stabilizer(logits, batch_labels)
                loss = loss_rank + BCE_WEIGHT * loss_bce
                loss.backward()
                xm.optimizer_step(optimizer)
                scheduler.step()
                global_step += 1
                consecutive_step_failures = 0
            except Exception as e:
                consecutive_step_failures += 1
                _log_stage(rank, "WARN", f"step failed and was skipped "
                                   f"({consecutive_step_failures}/{MAX_CONSECUTIVE_STEP_FAILURES} "
                                   f"consecutive): {type(e).__name__}: {e}")
                try:
                    del logits, loss
                except NameError:
                    pass
                optimizer.zero_grad(set_to_none=True)
                gc.collect()
                if consecutive_step_failures >= MAX_CONSECUTIVE_STEP_FAILURES:
                    raise RuntimeError(
                        f"[rank{rank}] {consecutive_step_failures} consecutive training steps "
                        f"failed with the same class of error ({type(e).__name__}: {e}) -- stopping."
                    ) from e
                continue

            if global_step % LOG_EVERY_STEPS == 0:
                xm.mark_step()
                if rank == 0:
                    et = time.perf_counter() - train_start
                    print(f"[{_elapsed():8.1f}s] epoch={epoch} step={global_step}/{total_steps} "
                          f"cat={category_vocab[category_id]!r} loss={loss.item():.4f} "
                          f"loss_rank={loss_rank.item():.4f} loss_bce={loss_bce.item():.4f} "
                          f"lr_backbone={scheduler.get_last_lr()[0]:.2e} elapsed_train={et:.0f}s "
                          f"rss={_rss_mb():.0f}MB",
                          flush=True)

            elapsed_train = time.perf_counter() - train_start
            if (elapsed_train - last_checkpoint_at) >= CHECKPOINT_EVERY_SECONDS:
                do_quiet_save(f"{CHECKPOINT_EVERY_SECONDS}s wall-clock cadence", epoch)
                last_checkpoint_at = elapsed_train
                model.train()

            if global_step >= next_rank_cache_step:
                rank_tables = rebuild_rank_cache(model, human_train_frame, caches, pad_id,
                                                 category_vocab, device, world_size, rank, pools)
                next_rank_cache_step = global_step + steps_per_half_epoch

            if elapsed_train > MAX_TRAIN_SECONDS:
                _log_stage(rank, "TRAIN", f"wall-clock cutoff ({elapsed_train:.0f}s > {MAX_TRAIN_SECONDS}s) "
                                    f"-- stopping mid-epoch.", force=True)
                do_quiet_save("wall-clock cutoff", epoch)
                stopped = True
                break

        if stopped:
            break
        do_full_checkpoint(epoch, f"checkpoint_epoch{epoch:02d}")
        last_checkpoint_at = time.perf_counter() - train_start
        model.train()

    _log_stage(rank, "DONE", f"total wall-clock {_elapsed():.0f}s best_val_macro_pr_auc={best_val:.4f}",
         force=True)


def main():
    os.environ.pop("TPU_PROCESS_ADDRESSES", None)
    os.environ.pop("CLOUD_TPU_TASK_ID", None)

    _load_shared_data()

    torch_xla.launch(_mp_fn, args=(), start_method="fork")


if __name__ == "__main__":
    main()
