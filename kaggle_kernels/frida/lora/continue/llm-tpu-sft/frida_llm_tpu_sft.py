"""
FRIDA (ai-forever/FRIDA, T5EncoderModel, 823M) LoRA SFT on TPU v5e-8 using
the recipe that produced this project's best real test score -- a port of
kaggle_kernels/berta-llm-tpu-sft to FRIDA, with the multi-layer head from
kaggle_kernels/frida-head-train in place of the single nn.Linear.

WHY THIS RECIPE
===========================================================================
Test-stand scores, not val (see README.md):

  BERTA, matches_llm+human mixed, PAIR_MAX_LENGTH=256   val 0.7087 -> test 0.47
  FRIDA, human-only,              PAIR_MAX_LENGTH=128   val ~0.69  -> test 0.386
  BERTA, human-only continuation, PAIR_MAX_LENGTH=384   val 0.7551 -> test 0.43
                                                        (and 0.38 / 0.369 at
                                                         epochs 2 / 4)

Two readings that survive those numbers. First, more human-only epochs make
things worse on the test stand, monotonically, even while val climbs -- the
384 continuation run is the clean demonstration. Second, the single
configuration that ever reached 0.47 mixed LLM soft labels into human data
AND doubled the pair budget. FRIDA has never been trained that way: every
FRIDA checkpoint to date is human-only at 128. This run changes both axes at
once, deliberately, because they are the two differences between the FRIDA
lineage and the one BERTA configuration that generalised.

That is an A/B/n confound and worth naming: if this improves on 0.386 we
will not know from this run alone how the credit splits between the LLM mix,
the pair budget, and the new head. Given each measurement costs a submission
slot out of 5/day and ~8.5h of TPU, buying the whole delta first and
decomposing later is the right order.

THE HEAD
===========================================================================
MultiLayerHead (duplicated verbatim from frida-head-train -- kernels are
standalone scripts with no shared import, so a mismatch shows up as a
state_dict load error, not silent misbehaviour) is warm-started from
maximmarkovbb/frida-multilayer-head. The point of the warm start is that the
head arrives already knowing how to read FRIDA's pooled representation, so
this run's first hour goes into the encoder rather than into re-teaching a
random head what the old linear one already knew.

REQUIRE_PRETRAINED_HEAD defaults to True and fails in the first seconds if
the head dataset is not mounted. Starting an 8.5h TPU session on a randomly
initialised head is exactly the kind of expensive-then-worthless run the
project's failure history is full of; a loud failure at second 30 costs
nothing.

OPTIMIZER STATE IS DELIBERATELY NOT RESUMED
===========================================================================
Every other continue-run in this project restores optimizer.pt. Here the
parameter set genuinely changed -- the single Linear became a multi-layer
head -- so the saved Adam moments no longer correspond to the parameters
being optimised. Loading them would either raise or, worse, silently pair
moment estimates with the wrong tensors. Fresh optimizer + a fresh cosine
schedule sized for this run's own reachable horizon (see the LR section
below), with WARMUP_RATIO=0.10 doing the work of easing back in. That warmup
ratio is not arbitrary: the MiniLM
continue-SFT history showed an over-aggressive restart (LR=4e-4, 3% warmup)
regressing below its own starting point and never recovering, and LR=2e-4
with 10% warmup plus human data mixed into every batch is precisely what
fixed it. Same numbers here.

ITEM CACHES: v1 POLICY ON BOTH SCOPES, ON PURPOSE
===========================================================================
This run needs item tokens for BOTH scopes -- matches.parquet's 711K human
items and matches_llm.parquet's 12.38M. Only the v1 caches
(maxamsecret/frida-tokenized-items-human / -llm) cover both; the v2
attribute-ordering cache (maximmarkovbb/frida-tokenize-items-v2-cache) is
human-scope only, built from items_human.parquet.

So both scopes use v1. A consistent policy across a single training run
matters more than which policy it is: mixing v2-ordered human items with
v1-ordered LLM items would make attribute order correlate with label source,
which is a feature the model would happily learn and which does not exist at
inference. The alternative -- re-tokenizing the LLM scope under the v2 policy
(~83 min on a TPU CPU session, precedent: frida-tokenize-items-llm) -- is the
cleaner long-term fix if the v2 ordering proves to be worth something.

KNOWN, ACCEPTED MISMATCH as of the checkpoint this run resumes: the reasoning
above was written when this kernel resumed frida-lora-checkpoint-v1-best,
itself trained on v1-ordered data -- so v1-both-scopes was fully consistent
end to end. It now resumes maximmarkovbb/frida-tpu-human-256-epoch4
(val 0.7660), which was trained on the v2 cache
(maximmarkovbb/frida-tokenize-items-v2-cache). This run's own human+LLM data
is still internally consistent with ITSELF (v1 both scopes, as argued
above), but no longer with what the resumed checkpoint already learned --
the same class of train/serve mismatch that this project has real evidence
costs real score (README.md's FRIDA-0.386 analysis: a v1/v2 mismatch between
a submission's preprocessing and its training cache was a leading candidate
cause). Accepted here rather than blocking the launch on an ~83-minute
v2-LLM-cache build the user did not ask for -- if this run's val trajectory
looks off relative to frida-tpu-human-sft's own (0.7437 -> 0.7660 over 4
epochs, human-only), this mismatch is the first thing to suspect.

The two id universes are fully disjoint (0 shared ids of 12.4M vs 711K,
re-verified in scripts/analyze_llm_label_overlap.py), so every pair lives
entirely inside one scope and a per-pair scope tag is enough to route the
lookup -- no merged 13GB index needed.

SIZING: ONE EPOCH OVER EVERYTHING, HOWEVER FAR IT GETS
===========================================================================
Per direct instruction: no LLM sampling (LLM_SAMPLE_ROWS=None -> all
11,187,780 rows) and EPOCHS=1. Whatever the session reaches is what it
reaches. berta-llm-tpu-sft took exactly this posture, and the checkpoint it
produced at ~50% of its single epoch is still this project's best test score
(0.47) -- a cut-off epoch is a normal outcome here, not a failure.

Measured throughput (frida-tpu-human-sft: same model, same
PAIR_MAX_LENGTH=256, same per-core batch) is ~120 rows/s -- half a
292,523-row epoch in ~20 minutes. Against 11,480,303 mixed rows a full epoch
needs ~26.5 hours, so the wall-clock cutoff, not the epoch boundary, is
guaranteed to be what ends this run. Expect roughly 3.7M rows, ~32% of the
epoch.

THE LR SCHEDULE IS SIZED TO THE CLOCK, NOT TO THE EPOCH
===========================================================================
That guarantee has a consequence worth handling rather than ignoring. A
cosine sized for the full 89,690-step epoch would still sit at ~96% of peak
LR when the session dies at ~32%, leaving the final checkpoint at a hot,
un-annealed point -- measurably worse than an annealed one. berta-llm-tpu-sft
got away with a full-epoch schedule because it reached 50%, where cosine is
already at half peak; 32% is a different situation.

So num_training_steps is projected from the time budget instead:
MAX_TRAIN_SECONDS * PROJECTED_ROWS_PER_SEC / global_batch, capped at the real
epoch length. Both directions of a wrong projection are made graceful rather
than sharp:
  * faster than projected -> the schedule runs out early, and LR_MIN_RATIO
    floors the LR at 5% of peak instead of the 0 a plain cosine decays to.
    Training keeps making progress rather than sitting dead for hours.
  * slower than projected -> the cosine simply does not finish annealing,
    which is the ordinary mid-run state every recovered checkpoint in this
    project has been in anyway.

The schedule stays STEP-based, not wall-clock-based, on purpose. All 8
replicas execute the same step count by construction, so they compute
identical LRs. Driving LR from each replica's own clock would let their
weights drift apart, because xm.optimizer_step all-reduces gradients but each
replica then applies its own update with its own LR.

TPU LESSONS CARRIED FORWARD (all real, previously-hit -- originals in
kaggle_kernels/berta-llm-tpu-sft and frida-tpu-human-sft docstrings)
===========================================================================
  1. os.environ.pop("TPU_PROCESS_ADDRESSES"/"CLOUD_TPU_TASK_ID") +
     start_method="fork".
  2. STATIC SHAPES: drop_last=True on every loader, fixed per-core batch,
     exactly two compiled graph shapes for the whole run.
  3. FIXED batch sizes, NO auto-probing -- probed candidates leave compiled
     XLA programs resident in HBM and self-inflict OOM, and
     xm.get_memory_info() cannot see that memory to warn. 16/32 are the
     values frida-tpu-human-sft is currently running at PAIR_MAX_LENGTH=256
     without OOM, so unlike that kernel these are now a MEASURED data point
     rather than an estimate.
  4. xm.optimizer_step(optimizer), not optimizer.step().
  5. autocast opened INSIDE forward().
  6. Frozen base bf16; LoRA A/B and the head kept fp32.
  7. ALL CPU-only setup ONCE in main() before torch_xla.launch() -- the 8
     replicas share it copy-on-write. Doing it after the fork is what caused
     berta-llm-tpu-sft's BrokenProcessPool.
  8. Eval wrapped in try/except that cannot kill the run; a failed eval falls
     back to a quiet weights-only save.
  9. Checkpoint hygiene: own directory per save, val_macro_pr_auc explicitly
     null when a save skipped eval, hourly wall-clock cadence so the 9h cap
     can never cost more than the last hour.
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
MODEL_NAME = "ai-forever/FRIDA"

MATCHES_PATH_HINT = "/kaggle/input/datasets/maxamsecret/ozon-ecup-2026-matching/matches.parquet"
MATCHES_LLM_PATH_HINT = "/kaggle/input/datasets/maxamsecret/ozon-ecup-2026-matching/matches_llm.parquet"
ITEMS_HUMAN_PATH_HINT = "/kaggle/input/datasets/maxamsecret/ozon-ecup-2026-matching/items_human.parquet"

HUMAN_CACHE_HINT = "/kaggle/input/datasets/maxamsecret/frida-tokenized-items-human"
LLM_CACHE_HINT = "/kaggle/input/datasets/maxamsecret/frida-tokenized-items-llm"

ITEM_MAX_LENGTH = 256
PAIR_MAX_LENGTH = 256      # the BERTA-recipe change: 128 -> 256

MAX_CATEGORY_CHARS = 64

CATEGORY_LABEL_TOKEN = 55917
ATTRIBUTES_LABEL_TOKEN = 91837
LABEL_COLON_TOKEN = 30

# Multi-layer head -- MUST match frida-head-train's definition exactly.
# (1024, 256)/0.2 is the WINNER of a real 5-architecture comparison run
# (frida-head-train, all_architectures_result.json), not a guess: "wide"
# beat linear_plus/shallow_wide/deep/deep_narrow at macro_pr_auc=0.76696 on
# the clean split==1 holdout (baseline linear head there: 0.76600 -- all 5
# candidates landed within +0.0006 to +0.0010 of that baseline and of each
# other, a thin margin worth remembering as "roughly tied", not "wide is
# clearly best"). _find_head_file() below re-derives this pair from the
# dataset's own result JSON and asserts it matches these two constants, so a
# future edit to one without the other fails loudly here rather than as a
# cryptic state_dict shape mismatch inside build_resumed_model().
HEAD_HIDDEN = (1024, 256)
HEAD_DROPOUT = 0.2
HEAD_HINT = "/kaggle/input/datasets/maximmarkovbb/frida-head-trained"
REQUIRE_PRETRAINED_HEAD = True

# BERTA recipe's own numbers, and the MiniLM continue-SFT fix's numbers.
LR = 2e-4
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.10

# MEASURED (not estimated): frida-tpu-human-sft runs these at PAIR_MAX_LENGTH
# =256 on v5e-8 without OOM. See docstring lesson 3.
PER_CORE_TRAIN_BATCH = 16
PER_CORE_EVAL_BATCH = 32

LLM_SAMPLE_ROWS = None         # None = all 11,187,780 rows, no sampling
EPOCHS = 1
MAX_TRAIN_SECONDS = 30600      # 8.5h. Kaggle's TPU session cap is 9h (the 12h
                                # cap is GPU) -- raise this only if a real TPU
                                # run here is observed surviving past 9h.
CHECKPOINT_EVERY_SECONDS = 3600
LOG_EVERY_STEPS = 50

# Used ONLY to size the LR schedule (see docstring). Measured on
# frida-tpu-human-sft at this exact model/sequence/batch. If a future run
# reports a different rate, update this rather than the schedule shape.
PROJECTED_ROWS_PER_SEC = 120
LR_MIN_RATIO = 0.05            # LR floor as a fraction of peak

WORKING_DIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
OUTPUT_DIR = os.path.join(WORKING_DIR, "frida_llm_tpu_sft")

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


def _find_head_file():
    """The published frida-head-trained dataset holds SIX head.pt files, not
    one: a winner copy plus one per losing architecture under
    architectures/<name>/head.pt (kept for comparison). An os.walk for the
    first file literally named "head.pt" -- this function's original,
    simpler version -- is genuinely ambiguous there and could silently pick
    a loser. Resolve unambiguously instead: find the dataset's own
    all_architectures_result.json (there is exactly one), read which
    architecture it declares as winner, and build that architecture's own
    path from the name -- self-describing, no dependence on os.walk order
    or on the dataset's exact directory nesting (which has already differed
    from this kernel's own naming assumption once: DATASET_ID in
    frida-head-train says frida-multilayer-head, the actually-published
    dataset is frida-head-trained, with an extra frida_head_train/ nesting
    level neither name predicts).

    Also asserts the winner matches this module's own HEAD_HIDDEN/
    HEAD_DROPOUT -- both are set from this exact result file's winner as of
    this writing, but if either constant is ever hand-edited without the
    other being updated to match, this fails loudly here (clear message)
    rather than inside build_resumed_model() as an opaque state_dict shape
    mismatch."""
    if not os.path.isdir("/kaggle/input"):
        return None
    for root, _dirs, files in os.walk("/kaggle/input"):
        if "all_architectures_result.json" not in files:
            continue
        with open(os.path.join(root, "all_architectures_result.json")) as f:
            summary = json.load(f)
        winner = summary["winner"]
        winner_layers = tuple(winner["layer_sizes"])
        winner_dropout = winner["dropout"]
        if winner_layers != HEAD_HIDDEN or winner_dropout != HEAD_DROPOUT:
            raise RuntimeError(
                f"[HEAD] frida-head-trained's own winner is {winner['name']} "
                f"layers={winner_layers} dropout={winner_dropout}, but this module's "
                f"HEAD_HIDDEN={HEAD_HIDDEN} HEAD_DROPOUT={HEAD_DROPOUT} don't match -- "
                f"update the constants above to agree before running.")
        candidate = os.path.join(root, "architectures", winner["name"], "head.pt")
        if not os.path.isfile(candidate):
            raise RuntimeError(f"[HEAD] result JSON declares winner={winner['name']!r} "
                               f"but {candidate} does not exist")
        print(f"[HEAD] winner={winner['name']} layers={winner_layers} "
              f"dropout={winner_dropout} macro_pr_auc={winner['macro_pr_auc']:.5f} "
              f"(baseline linear head: {summary['baseline_linear_head_macro_pr_auc']:.5f})",
              flush=True)
        return candidate
    return None


# =====================================================================
# ITEM CACHES -- v1 layout (flat ids/input_ids/attention_mask, no variants)
# =====================================================================

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
        return (torch.from_numpy(input_ids), torch.from_numpy(attention_mask),
                torch.tensor(labels, dtype=torch.float32),
                np.asarray(categories, dtype=np.int64))
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

class MultiLayerHead(nn.Module):
    """Verbatim copy of frida-head-train's class -- keep in sync by hand."""

    def __init__(self, hidden_dim, layer_sizes=HEAD_HIDDEN, dropout=HEAD_DROPOUT):
        super().__init__()
        blocks = []
        d = hidden_dim
        for h in layer_sizes:
            blocks.append(nn.Sequential(
                nn.Linear(d, h),
                nn.LayerNorm(h),
                nn.GELU(),
                nn.Dropout(dropout),
            ))
            d = h
        self.blocks = nn.ModuleList(blocks)
        self.out = nn.Linear(d, 1)
        self.feature_dim = d

    def forward(self, x, return_features=False):
        for block in self.blocks:
            x = block(x)
        logit = self.out(x).view(-1)
        if return_features:
            return logit, x
        return logit


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


def build_resumed_model(device, rank, resume_dir, head_path):
    try:
        base_model = T5EncoderModel.from_pretrained(MODEL_NAME, dtype=torch.bfloat16,
                                                     attn_implementation="eager")
    except (TypeError, ValueError):
        base_model = T5EncoderModel.from_pretrained(MODEL_NAME, dtype=torch.bfloat16)
    hidden_dim = base_model.config.d_model

    peft_model = PeftModel.from_pretrained(base_model, resume_dir, is_trainable=True)
    if rank == 0:
        peft_model.print_trainable_parameters()
    n_trainable_lora = sum(p.numel() for n, p in peft_model.named_parameters()
                           if "lora_" in n and p.requires_grad)
    assert n_trainable_lora > 0, "Resumed LoRA adapter has zero trainable params -- is_trainable didn't take."
    for name, p in peft_model.named_parameters():
        if "lora_" in name:
            p.data = p.data.float()

    head = MultiLayerHead(hidden_dim)
    if head_path is not None:
        head.load_state_dict(torch.load(head_path, map_location="cpu"))
        _log(rank, "MODEL", f"multi-layer head warm-started from {head_path}", force=True)
    else:
        _log(rank, "MODEL", "multi-layer head RANDOMLY INITIALISED "
                            "(REQUIRE_PRETRAINED_HEAD=False)", force=True)

    # The resumed checkpoint's own classifier.pt is a single nn.Linear and is
    # deliberately ignored: the head it describes no longer exists in this
    # model. Noted rather than silently skipped, because "the head did not
    # load" is exactly the class of silent failure this project keeps hitting.
    if os.path.isfile(os.path.join(resume_dir, "classifier.pt")):
        _log(rank, "MODEL", f"ignoring {resume_dir}/classifier.pt (single Linear, "
                            f"superseded by the multi-layer head)", force=True)

    model = FridaCrossEncoderClassifier(peft_model, head).to(device)
    return model, hidden_dim


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
            for input_ids, attention_mask, batch_labels, batch_categories in loader:
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

def save_checkpoint(model, optimizer, global_step, epoch, val_macro_pr_auc, label, rank):
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
            "val_macro_pr_auc": val_macro_pr_auc,   # explicit null if unvalidated
            "head": {"type": "MultiLayerHead", "layer_sizes": list(HEAD_HIDDEN),
                     "dropout": HEAD_DROPOUT},
            "pair_max_length": PAIR_MAX_LENGTH,
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


def _load_shared_data():
    print(f"[{_elapsed():8.1f}s][SETUP] loading all CPU-only data once, pre-fork ...", flush=True)

    # Fail-cheap preflight: everything this run cannot proceed without gets
    # checked before a single expensive thing happens.
    head_path = _find_head_file()
    if head_path is None and REQUIRE_PRETRAINED_HEAD:
        raise RuntimeError("[SETUP] no head.pt found under /kaggle/input. Mount "
                           f"{HEAD_HINT} via dataset_sources, or set "
                           "REQUIRE_PRETRAINED_HEAD=False to start from a random head "
                           "(see module docstring for why that is discouraged).")
    if _find_resume_adapter_dir() is None:
        raise RuntimeError("[SETUP] no adapter_config.json under /kaggle/input -- mount a "
                           "validated FRIDA LoRA checkpoint via dataset_sources.")

    human_cache = load_v1_cache(HUMAN_CACHE_HINT, "human")
    llm_cache = load_v1_cache(LLM_CACHE_HINT, "llm")
    if human_cache["pad_id"] != llm_cache["pad_id"]:
        raise RuntimeError(f"[SETUP] pad_id differs between caches: "
                           f"{human_cache['pad_id']} vs {llm_cache['pad_id']}")
    caches = [human_cache, llm_cache]
    pad_id = human_cache["pad_id"]

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
          f"val_fingerprint={val_fingerprint}", flush=True)

    # Categories are needed for the metric, which is computed on the human val
    # split only -- so items_human.parquet is enough and the 13M-row
    # items.parquet is never touched.
    items_cat = (
        pl.scan_parquet(_find_dataset_file(ITEMS_HUMAN_PATH_HINT, "items_human.parquet"))
        .join(pl.DataFrame({"id": pd.Series(pd.unique(matches_pd["id1"]), dtype="int64")}).lazy(),
              on="id", how="semi")
        .select(["id", "category"])
        .with_columns(pl.col("category").fill_null("UNKNOWN").cast(pl.Utf8)
                      .str.slice(0, MAX_CATEGORY_CHARS))
        .collect(engine="streaming")
    )
    category_vocab = sorted(items_cat.get_column("category").unique().to_list()) + ["UNKNOWN"]
    category_to_id = {c: i for i, c in enumerate(category_vocab)}
    unknown_id = category_to_id["UNKNOWN"]
    id_to_category = dict(zip(items_cat.get_column("id").to_list(),
                              items_cat.get_column("category").to_list()))

    def cat_ids(frame):
        return frame["id1"].map(id_to_category).map(category_to_id).fillna(unknown_id) \
            .to_numpy().astype(np.int64)

    human_train_frame = _build_frame(human_train["id1"], human_train["id2"],
                                     human_train["label"], cat_ids(human_train), 0, caches)
    val_frame = _build_frame(human_val["id1"], human_val["id2"],
                             human_val["label"], cat_ids(human_val), 0, caches)
    del matches_pd, items_cat, id_to_category, labels_all, indices, train_idx, val_idx
    gc.collect()

    llm = pl.read_parquet(_find_dataset_file(MATCHES_LLM_PATH_HINT, "matches_llm.parquet"))
    llm = llm.filter(pl.col("id1").is_not_null() & pl.col("id2").is_not_null())
    n_llm_total = llm.height
    if LLM_SAMPLE_ROWS is not None and LLM_SAMPLE_ROWS < n_llm_total:
        llm = llm.sample(n=LLM_SAMPLE_ROWS, seed=SEED, shuffle=True)
    llm_pd = llm.select(["id1", "id2", "target"]).to_pandas()
    del llm
    gc.collect()
    llm_fingerprint = hashlib.sha256(
        f"{SEED}:{LLM_SAMPLE_ROWS}:{n_llm_total}:{len(llm_pd)}".encode("utf-8")
    ).hexdigest()[:16]
    print(f"[{_elapsed():8.1f}s][SETUP] llm sampled {len(llm_pd):,} of {n_llm_total:,} "
          f"(seed={SEED}, fingerprint={llm_fingerprint})", flush=True)

    # LLM targets stay CONTINUOUS (soft labels straight into BCEWithLogitsLoss,
    # never thresholded) -- the same treatment berta-llm-tpu-sft and the MiniLM
    # continue-train used. Categories are irrelevant here: metrics are computed
    # on the human val split only, so every LLM row gets the UNKNOWN id.
    llm_frame = _build_frame(llm_pd["id1"], llm_pd["id2"], llm_pd["target"],
                             np.full(len(llm_pd), unknown_id, dtype=np.int64), 1, caches)
    del llm_pd
    gc.collect()

    train_frame = _concat_frames(human_train_frame, llm_frame)
    print(f"[{_elapsed():8.1f}s][SETUP] mixed train={len(train_frame['label']):,} "
          f"(human {len(human_train_frame['label']):,} + llm {len(llm_frame['label']):,})",
          flush=True)
    del human_train_frame, llm_frame
    gc.collect()

    _SHARED.update(dict(
        caches=caches, pad_id=pad_id, train_frame=train_frame, val_frame=val_frame,
        category_vocab=category_vocab, val_fingerprint=val_fingerprint,
        llm_fingerprint=llm_fingerprint, head_path=head_path,
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

    _log(rank, "CONFIG", f"world_size={world_size} pair_max_length={PAIR_MAX_LENGTH} "
                         f"epochs={EPOCHS} lr={LR} warmup={WARMUP_RATIO} "
                         f"head={HEAD_HIDDEN} "
                         f"llm_sample={'ALL' if LLM_SAMPLE_ROWS is None else format(LLM_SAMPLE_ROWS, ',')}",
         force=True)
    _log(rank, "DATA", f"train={len(train_frame['label']):,} val={len(val_frame['label']):,} "
                       f"val_fingerprint={_SHARED['val_fingerprint']} "
                       f"llm_fingerprint={_SHARED['llm_fingerprint']}", force=True)

    resume_dir = _find_resume_adapter_dir()
    model, hidden_dim = build_resumed_model(device, rank, resume_dir, _SHARED["head_path"])
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    total_params = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in trainable_params)
    _log(rank, "MODEL", f"resume={resume_dir} total={total_params:,} "
                        f"trainable={trainable_count:,} "
                        f"({100 * trainable_count / total_params:.4f}%)", force=True)

    total_batch = PER_CORE_TRAIN_BATCH * world_size
    _log(rank, "BATCH", f"per_core_train={PER_CORE_TRAIN_BATCH} per_core_eval={PER_CORE_EVAL_BATCH} "
                        f"total_train_batch={total_batch} (fixed, no probing)", force=True)

    # pos_weight over the MIXED set, with soft targets counted as the mass they
    # actually carry rather than thresholded -- otherwise the weight describes
    # a label distribution that is not the one being trained on.
    labels = train_frame["label"]
    pos_mass = float(labels.sum())
    pos_weight = torch.tensor((len(labels) - pos_mass) / max(pos_mass, 1e-9), device=device)
    _log(rank, "LOSS", f"pos_mass={pos_mass:,.0f}/{len(labels):,} pos_weight={pos_weight.item():.4f}",
         force=True)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Fresh optimizer on purpose -- the parameter set changed with the new
    # head, so the checkpoint's Adam moments no longer match. See docstring.
    optimizer = torch.optim.AdamW(trainable_params, lr=LR, weight_decay=WEIGHT_DECAY)
    _log(rank, "RESUME", "optimizer starts FRESH (head architecture changed); "
                         "warmup_ratio absorbs the restart", force=True)

    steps_per_epoch = len(labels) // total_batch
    epoch_steps = steps_per_epoch * EPOCHS

    # Schedule horizon = what the clock can actually buy, not what the epoch
    # nominally contains. See the docstring's LR section for why, and for why
    # this stays step-based rather than reading each replica's own clock.
    projected_steps = int(MAX_TRAIN_SECONDS * PROJECTED_ROWS_PER_SEC / total_batch)
    total_steps = max(1, min(epoch_steps, projected_steps))
    warmup_steps = max(1, int(total_steps * WARMUP_RATIO))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, _make_lr_lambda(warmup_steps, total_steps, LR_MIN_RATIO))
    _log(rank, "TRAIN", f"steps/epoch={epoch_steps} projected_reachable={projected_steps} "
                        f"-> schedule_steps={total_steps} "
                        f"({100 * total_steps / max(epoch_steps, 1):.1f}% of the epoch) "
                        f"warmup={warmup_steps} lr_floor={LR_MIN_RATIO * LR:.2e} "
                        f"max_train_seconds={MAX_TRAIN_SECONDS} "
                        f"checkpoint_every={CHECKPOINT_EVERY_SECONDS}s", force=True)

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
            _log(rank, "SAVE", f"quiet save ({reason}) epoch={epoch_num} step={global_step}",
                 force=True)
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
                print(f"[METRICS] per_category: "
                      f"{json.dumps(m['per_category'], ensure_ascii=False)}", flush=True)
            save_checkpoint(model, optimizer, global_step, epoch_num, m["macro_pr_auc"], label, rank)
            if improved:
                save_checkpoint(model, optimizer, global_step, epoch_num, m["macro_pr_auc"],
                                "best", rank)
            return True
        except Exception as e:
            _log(rank, "ERROR", f"eval/checkpoint FAILED at epoch={epoch_num} "
                                f"step={global_step}: {type(e).__name__}: {e}", force=True)
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
                                                  PER_CORE_TRAIN_BATCH, world_size, rank,
                                                  SEED, epoch=epoch)
        mp_loader = pl_xla.MpDeviceLoader(train_loader, device)
        _log(rank, "EPOCH", f"starting epoch {epoch}/{EPOCHS}", force=True)

        for step, batch in enumerate(mp_loader):
            input_ids, attention_mask, batch_labels, _cats = batch
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
                _log(rank, "WARN", f"step failed and was skipped: {type(e).__name__}: {e}",
                     force=True)
                # del before continue: a live reference to the failed step's
                # graph keeps its buffers alive into the next step.
                try:
                    del logits, loss
                except NameError:
                    pass
                optimizer.zero_grad(set_to_none=True)
                gc.collect()
                continue

            if global_step % LOG_EVERY_STEPS == 0:
                xm.mark_step()
                loss_val = loss.item()
                if rank == 0:
                    et = time.perf_counter() - train_start
                    print(f"[{_elapsed():8.1f}s] epoch={epoch} step={global_step}/{total_steps} "
                          f"loss={loss_val:.4f} lr={scheduler.get_last_lr()[0]:.2e} "
                          f"elapsed_train={et:.0f}s", flush=True)

            elapsed_train = time.perf_counter() - train_start
            if (elapsed_train - last_checkpoint_at) >= CHECKPOINT_EVERY_SECONDS:
                do_full_checkpoint(epoch, f"checkpoint_hour{int(elapsed_train // 3600):02d}")
                last_checkpoint_at = elapsed_train
                model.train()

            if elapsed_train > MAX_TRAIN_SECONDS:
                _log(rank, "TRAIN", f"wall-clock cutoff ({elapsed_train:.0f}s > "
                                    f"{MAX_TRAIN_SECONDS}s) -- stopping mid-epoch.", force=True)
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
