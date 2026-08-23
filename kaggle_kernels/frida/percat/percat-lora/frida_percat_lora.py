# %% [code]
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

HEAD SCHEDULE, GRADIENT CLIPPING, FAIL-LOUD ON NaN
===========================================================================
The classifier is randomly initialised while the backbone is already strong,
so the two need different schedules. Two param groups: the head ramps to
HEAD_LR (3x the adapter's) over a SHORT warmup (HEAD_WARMUP_STEPS), holds
that peak flat until the adapter's own (longer) warmup ends, then both
cosine-decay together to a floor.

This is a real fix, not a first-principles design choice: an earlier
version jumped the head straight to peak at step 0 with no ramp and no
gradient clipping anywhere, and a real run (frida-percat-lora.log)
reproducibly hit loss=nan by the very first logged step on 15/20
categories, with 2 more drifting into NaN once a harder batch hit later --
17/20 specialists lost, only 3 (07/10/14) survived. Fixed with three
changes together, not any one alone: the short head ramp above,
HEAD_LR_MULTIPLIER backed off 5.0->3.0, and grad-norm clipping
(GRAD_CLIP_NORM) before every optimizer step. A NaN loss also now fails
LOUDLY and immediately (TrainingDivergedError, raised the moment
torch.isfinite(loss) is False) instead of silently continuing to "train"
on garbage for the rest of that category's steps -- the previous behavior
wasted each failing category's full step budget before the problem
surfaced at eval time.

V6: RESUME FROM v5, FULL LLM SPLIT, QUIET CHECKPOINTS
===========================================================================
Three changes, all per direct instruction, none of them touching the NaN
fix above:

  1. RESUME, NOT SKIP. v5 treated an already-good external checkpoint as
     "done, copy it forward" (07/10/14 in that real run). v6 treats ANY
     existing checkpoint -- a validated _best, or failing that a quiet
     mid-training _last -- as a starting point to CONTINUE training from,
     via load_specialist_from_checkpoint_() (probe-delta verified, same
     no-op-detection reasoning as merge_and_save_base's own check). A
     category with no checkpoint anywhere still falls back to
     reset_specialist_'s fresh init. Resumed categories train at
     LR * LR_RESUME_MULTIPLIER (0.5) -- a smaller step for continuing an
     adapter that is already at a good point, same reasoning this project's
     own LLM continue-SFT already used (LR halved from its from-scratch
     peak). The warmup/clip shape is UNCHANGED for resumed categories, only
     the peak magnitude scales down.
  2. FULL LLM TRAIN SPLIT, NOT A SUBSAMPLE. v5 capped LLM rows at
     LLM_ROWS_PER_CATEGORY (100K/category) via stratified_llm_sample. v6
     loads frida-llm-train-val-split-no-leak-retroactive v2's own output and
     uses every row marked split=="train" (~11.09M rows, category-balance
     and target-level stratification already applied by that kernel, not
     re-applied here) -- no per-category cap. That kernel's own "train"
     covers everything NOT held out for its fixed, unchanged 100K val; its
     original 1,000,000-row reproduction (the exact rows v5 already trained
     specialists on) is a verified subset, so nothing already trained on
     moves into a held-out set under this run.
  3. QUIET CHECKPOINTS, ACTUALLY WIRED UP. CHECKPOINT_EVERY_SECONDS existed
     in v5 but nothing ever checked it -- a killed session lost the WHOLE
     in-progress category, not just the time since its last checkpoint. v6
     checks wall-clock against it (7200s = 2h) inside the per-step loop and
     saves adapter+head under f"{label_prefix}_last" (val_pr_auc explicit
     None, the same unvalidated-checkpoint convention save_checkpoint
     already uses) when crossed -- no eval, so this is cheap enough to run
     mid-category without disrupting the training loop's own timing.

REAL CRASH: load_specialist_from_checkpoint_'s missing-keys check was itself
wrong on its first real run (all 8 ranks died identically at cat00, ~6.3min
in). set_peft_model_state_dict() does model.load_state_dict(adapter_sd,
strict=False) under the hood, and adapter_sd only ever holds the LoRA delta
weights -- every base-model weight (base_model.model.*, *.base_layer.weight,
relative_attention_bias, layer_norm, ...) is CORRECTLY reported "missing" by
strict=False on every single load. The real traceback showed exactly that:
~230 "missing" keys, every one a base weight, zero lora_ keys -- the adapter
itself loaded fine, the check was just wrong. Fixed by filtering missing_keys
to only lora_-containing entries before deciding it's an error; a missing
base weight is how a correct partial adapter load is supposed to look.

REAL CRASH #2: fixing #1 let the resume actually happen -- and the very next
run crashed the WHOLE process at cat00's first training step (loss=nan at
step 1/339), taking all 8 ranks and every other category down with it. Two
compounding problems, both fixed together:
  1. cat00_last (and every OTHER non-_best checkpoint in
     frida-with-merged-lora-and-8-10-14-cat-loras) is a casualty of the
     ORIGINAL pre-NaN-fix run this project already diagnosed once (see the
     NaN root-cause section above) -- that dataset's name ("...8-10-14...")
     literally IS the user's own recollection of which 3 categories survived
     it, and cat07_best/cat10_best/cat14_best are the only _best entries
     present. Its 17 _last checkpoints are pre-fix weights already pushed
     into NaN, not legitimately-cut-short good ones. load_specialist_from_
     checkpoint_ now checks torch.isfinite() on every loaded lora_ and head
     param right after loading and raises if any are non-finite, so a
     corrupted checkpoint is caught before a single training step touches it.
  2. Independently of #1: TrainingDivergedError was NEVER actually caught at
     "the category level" despite the class's own docstring and the step-
     loop's own comment both claiming it was -- the only try/except at that
     scope wrapped just the evaluate() call, well outside the epoch loop
     TrainingDivergedError is raised from, so it propagated straight out of
     _mp_fn and killed the whole run (confirmed by the real traceback: line
     numbers pointing at the raise inside _mp_fn itself, uncaught). Fixed by
     wrapping the per-category epoch loop in its own try/except
     TrainingDivergedError: on catch, save a quiet _last checkpoint (marked
     "diverged": true), record results[cat_name] = None, and move on to the
     next category -- exactly the behavior the old comment already claimed
     existed. The resume call site also now catches any
     load_specialist_from_checkpoint_ failure (not just non-finite weights)
     and falls back to reset_specialist_ rather than propagating -- a
     category that can't be resumed trains from scratch instead of taking
     the whole run down.

REAL CRASH #3 (host RAM OOM, 2026-08-22) -- fixing #2 exposed a resource
leak that used to be structurally impossible to hit
===========================================================================
The very next real run (with #2's fix applied) survived the whole 20
-category loop crashing outright, but died ~1500s / 10 categories in with
Kaggle's own "tried to allocate more memory than is available" -- host RAM,
not HBM. Root cause: `del mp_loader, train_loader, sampler` only ever sat
AFTER the step loop (a normal epoch end reaches it); a TrainingDivergedError
raised mid-step jumps straight past it to the except clause, which had no
cleanup of its own. val_full_loader/val_remainder_loader (also
num_workers=2 each) were never explicitly freed AT ALL, relying only on the
implicit refcount drop when those names get reassigned next category.
Before v6 this never mattered: divergence used to crash the WHOLE process
immediately (nothing had a chance to accumulate), and v5's own
"already-good, skip via continue" path returned before eval loaders were
even constructed. v6 made both of those the exception rather than the rule
-- resuming means every category (not just the ones needing real training)
builds eval loaders, and catching divergence means a category that fails in
2-5 steps immediately starts the next one. The real run that found this hit
8/10 diverging categories in ~1500s, i.e. many rapid DataLoader create
-then-abandon cycles across 8 ranks in parallel -- exactly the shape that
turns "technically a leak" into "exhausts host RAM in 25 minutes." Fixed by
adding the same del+gc.collect() to the except-TrainingDivergedError branch,
and an unconditional del of both val loaders once they're no longer needed
for the category (moved from "never" to "every category, right after use").

Two more additions, same session, aimed at the root cause this exposed
rather than another symptom:
  - SMOKE TEST (_smoke_test_train_step/_smoke_test_eval_step/
    _smoke_test_real_loader_batch, run once before the category loop):
    frida-percat-lora never had one, unlike frida-full-sft (which earned
    its own the hard way, five real crashes). Every bug in THIS kernel's
    history -- three separate checkpoint-path bugs, the over-strict
    missing-keys check, this OOM -- was found only after a real, paid TPU
    run. Doesn't fix any of those; means the next HBM-fit/shape bug costs
    1-2 minutes instead of hours.
  - MAX_CONSECUTIVE_CATEGORY_FAILURES=6: the SAME real run that found the
    OOM also showed 8 of its first 10 categories diverging to NaN from a
    FRESH init, using the exact HEAD_LR_MULTIPLIER/warmup/GRAD_CLIP_NORM
    config already described above as "the NaN fix" -- across multiple
    independent real runs now (this one and the original one that produced
    frida-with-merged-lora-and-8-10-14-cat-loras), that config has never
    actually gotten past ~3/20 categories surviving. That is a separate,
    deeper, still-unsolved problem (see frida-percat-lora-nan-diagnose) --
    but whatever its cause, if it is systemic (bad merged base, bad data
    mix, a config regression) rather than per-category, burning the full
    8.5h MAX_TRAIN_SECONDS to discover that one category at a time, 20
    times, is pure waste. Six consecutive category-level failures now stops
    the run loudly with a clear message instead.

REAL CRASH #3, ROUND 2: the leak fix above did NOT fully solve the OOM --
same crash, one category count earlier
===========================================================================
The very next real run (leak fix + smoke test + circuit breaker all
applied) hit the SAME "tried to allocate more memory than is available"
kill, this time starting category 04 at ~836s instead of surviving to
category 09 at ~1479s in the run that found the leak. The per-category
divergence steps were IDENTICAL to the previous run (00->step40, 01->step2,
02->step5, 03->step32 -- confirming the NaN divergence itself is
deterministic, not noise), but each category's own wall-clock grew 55-96s
past the leak-fix-free baseline (631s/686s/759s/834s vs 552s/605s/668s/738s)
-- a GROWING gap, not a fixed one-time offset the smoke test alone would
produce. Not fully root-caused, but a real, checkable hypothesis: the fix
added THREE gc.collect() calls per diverged category (step-level,
category-level, val-loader) where one used to run; CPython's cyclic
collector walks the whole object graph, and under fork+refcount semantics,
merely VISITING an object bumps its refcount field -- which lives on the
same memory page as the object -- triggering copy-on-write duplication of
that page in this rank's process even for otherwise-untouched, pre-fork
-shared data (the large item-token caches, val_by_cat/train_by_cat). More
collect() calls could mean more incremental COW duplication of supposedly
-shared memory, which would show up exactly as "gets slower and eventually
OOMs, worse than before the extra collect() calls" -- what was observed.
Two changes, aimed at making this checkable rather than guessed a third
time:
  1. Consolidated three gc.collect() calls per diverged category down to
     one (kept the del statements -- those alone already drop non-cyclic
     objects to refcount 0 immediately, no collect() needed for that part;
     collect() only adds value for reference cycles, and one pass per
     category catches those same as three would).
  2. Added direct RSS logging (resource.getrusage) to the per-category CAT
     log line -- the next real run will show the actual memory curve
     instead of this docstring inferring it from wall-clock gaps a third
     time.

The RSS logs from that run then SETTLED the question the gc.collect()
guess was trying to answer, and pointed somewhere else entirely: RSS did
NOT climb category-over-category (17.9-18.9GB at cat00, then a ONE-TIME
jump to 19.6-22.9GB at cat01, then FLAT through cat02/03/04) -- not a
leak, a high BASELINE. 8 ranks x ~21GB =~ 168GB, evidently close to this
TPU v5e-8 VM's actual host RAM ceiling regardless of category content.

V11: build_specialist_model_cpu + MpModelWrapper -- fixing the baseline
itself, not another symptom of hitting it
===========================================================================
PyTorch/XLA's own docs describe the exact anti-pattern this kernel had:
"instead of creating models on each multiprocessing process ..., the model
is created once at global scope, and then moved into each device inside
the spawn() target function" (xmp.MpModelWrapper). This kernel's own
_load_shared_data() already does this correctly for DATA (loaded once in
main(), pre-fork) -- but the MODEL was built inside _mp_fn, i.e. every one
of 8 ranks independently called T5EncoderModel.from_pretrained() (a real
823M-param disk read + fresh CPU allocation) and get_peft_model() AFTER
the fork, none of it shared. build_specialist_model_cpu now does exactly
that construction ONCE in main(), pre-fork; the result is wrapped in
xmp.MpModelWrapper and every rank's own _mp_fn just does
`_WRAPPED_MODEL.to(device)` -- a cheap transfer of already-loaded host
pages, not a fresh load. Per-category reset_specialist_/
load_specialist_from_checkpoint_ are unaffected: they mutate the
already-on-device, per-rank-independent tensors .to(device) produces, same
as before.

Pushed deliberately WITHOUT also changing init_lora_weights or batch size
(that combination went to a separate account/push instead, see
kernel-metadata.json history) -- two independent fixes for two independent
problems (host RAM baseline vs. NaN), run in parallel on separate quotas
specifically so which one actually helps is separable, not guessed a
fourth time from a run that changed everything at once.

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
import resource
import time
import hashlib
import shutil
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
                   get_peft_model_state_dict, set_peft_model_state_dict)
from safetensors.torch import save_file as safetensors_save_file
from safetensors.torch import load_file as safetensors_load_file

import torch_xla
import torch_xla.core.xla_model as xm
from torch_xla import runtime as xr
import torch_xla.distributed.parallel_loader as pl_xla
import torch_xla.distributed.xla_multiprocessing as xmp


# =====================================================================
# CONFIG
# =====================================================================

SEED = 42
MODEL_NAME = "ai-forever/FRIDA"

MATCHES_PATH_HINT = "/kaggle/input/datasets/maxamsecret/ozon-ecup-2026-matching/matches.parquet"
MATCHES_LLM_PATH_HINT = "/kaggle/input/datasets/maxamsecret/ozon-ecup-2026-matching/matches_llm.parquet"
# v2 of frida-llm-train-val-split-no-leak-retroactive: train = every row
# NOT held out for the fixed 100K val (~11.09M rows), not a subsample --
# see that kernel's own docstring. Publishes to the SAME dataset id
# frida-full-sft already references (maxamsecret/frida-llm-train-val-split),
# just a new version of it.
LLM_SPLIT_PATH_HINT = (
    "/kaggle/input/datasets/maxamsecret/"
    "frida-llm-train-val-split/llm_train_val_split.parquet"
)
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
# v6: categories resuming from an already-trained checkpoint (see
# load_specialist_from_checkpoint_) use LR * LR_RESUME_MULTIPLIER as their
# peak instead of the full fresh-init LR -- continuing an adapter that is
# already at a good point should take smaller steps than initializing one
# from a random start, same reasoning any continued-fine-tuning run in
# this project has already used (e.g. the original LLM continue-SFT's own
# LR=2e-4, half of a from-scratch run's peak). The warmup/clip machinery
# below is UNCHANGED for resumed categories -- only the peak magnitude
# scales down, not the shape of the ramp.
LR_RESUME_MULTIPLIER = 0.5

# 5.0 combined with an instant step-0 jump to peak (see
# _make_head_lr_lambda) is the confirmed root cause of a reproducible
# early-NaN failure: a real run (frida-percat-lora.log) showed loss=nan
# already by the FIRST logged step on 15/20 categories, with the other 2
# of the 17 failures drifting into NaN a bit later once a harder batch
# hit -- i.e. a freshly-initialised linear head jumping straight to a
# 5x-multiplied LR with no ramp and no gradient clipping anywhere in the
# step. Backed off to 3.0 and given its own short ramp, plus clipping
# added to the step itself (see the training loop) -- not relying on
# either fix alone.
HEAD_LR_MULTIPLIER = 3.0
HEAD_WARMUP_STEPS = 20     # short ramp 0 -> peak, then flat until warmup_steps
GRAD_CLIP_NORM = 1.0
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

# v5-only: capped LLM rows/category for the from-scratch run. v6 trains on
# the full frida-llm-train-val-split-no-leak-retroactive train split instead
# (see LLM_SPLIT_PATH_HINT below, no cap) -- this constant is kept only
# because stratified_llm_sample (still accurate documentation of the target-
# level distribution) takes it as a parameter and nothing currently calls
# that function from _load_shared_data any more.
LLM_ROWS_PER_CATEGORY = 100_000
EPOCHS = 1                     # per category
MAX_TRAIN_SECONDS = 30600      # 8.5h. Kaggle's TPU session cap is 9h (the 12h
                                # cap is GPU) -- raise this only if a real TPU
                                # run here is observed surviving past 9h.
# v6: 3600 -> 7200, per direct instruction. Quiet checkpoints now save the
# FULL state (adapter + head, see the per-step check below) so a session
# killed mid-category loses at most 2h of that category's progress instead
# of losing the whole category.
CHECKPOINT_EVERY_SECONDS = 7200
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


def _find_resume_adapter_dir(prefer_name="best"):
    """Collects EVERY directory under /kaggle/input containing an
    adapter_config.json, rather than trusting os.walk's first hit.

    Real bug #1, found the hard way: the previous version hardcoded a
    priority path with a typo -- "/kaggle/inputdatasets/..." instead of
    "/kaggle/input/datasets/..." (missing slash) -- which never exists,
    so it silently fell through every time to a bare os.walk("/kaggle/
    input") first-match. With TWO valid adapters now mounted (this SFT
    checkpoint's own "best", and the older epoch4 checkpoint), which one
    won was undefined -- the run this exact bug shipped in got lucky, a
    future one might not.

    Real bug #2, found the hard way when frida-with-merged-lora-and-8-10-
    14-cat-loras (20 per-category checkpoints) was mounted alongside the
    general SFT checkpoints: `candidates` holds the directory that
    DIRECTLY contains adapter_config.json. Under this project's own
    save_checkpoint (and frida-llm-tpu-sft's own checkpointing), that
    directory is ALWAYS literally named "adapter" -- the checkpoint's real
    label ("best", "checkpoint_hour05", "cat07_best", ...) is one level up,
    in the PARENT directory's name. Comparing prefer_name against the
    candidate's OWN basename can therefore never match anything saved this
    way -- a real run hit exactly this, 0/29 candidates matched "best",
    RuntimeError. Fixed by preferring the parent's name when the candidate
    itself is named "adapter", falling back to the candidate's own name
    for any adapter saved without that extra nesting."""
    if not os.path.isdir("/kaggle/input"):
        return None
    candidates = []
    for root, _dirs, files in os.walk("/kaggle/input"):
        if "adapter_config.json" in files:
            candidates.append(root)
    if not candidates:
        return None

    def _label(c):
        c = c.rstrip("/\\")
        base = os.path.basename(c)
        return os.path.basename(os.path.dirname(c)) if base == "adapter" else base

    preferred = [c for c in candidates if _label(c) == prefer_name]
    if len(candidates) > 1:
        print(f"[SETUP] {len(candidates)} candidate adapter dirs under /kaggle/input: "
              f"{candidates}", flush=True)
        if len(preferred) != 1:
            raise RuntimeError(
                f"multiple adapter checkpoints present but exactly one named "
                f"'{prefer_name}' was expected (found {len(preferred)}) -- refusing to "
                f"guess which to resume. Candidates: {candidates}")
        print(f"[SETUP] using the one named '{prefer_name}': {preferred[0]}", flush=True)
    return preferred[0] if preferred else candidates[0]


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


# The already-published dataset holding a verified merge + 3 trained
# specialists (categories 07/10/14 -- confirmed by directly listing the
# dataset's files, NOT by trusting the "8, 10, 14" recalled verbally;
# the actual successful categories are 07/10/14). Its own internal
# layout is frida_percat_lora/{merged_base, cat{NN}_best, cat{NN}_last}/,
# i.e. a direct publish of an earlier run's own OUTPUT_DIR.
EXTERNAL_CHECKPOINTS_HINT = (
    "/kaggle/input/datasets/maxamsecret/"
    "frida-with-merged-lora-and-8-10-14-cat-loras/frida_percat_lora"
)


def _find_external_checkpoints_root():
    if os.path.isdir(EXTERNAL_CHECKPOINTS_HINT):
        return EXTERNAL_CHECKPOINTS_HINT
    if os.path.isdir("/kaggle/input"):
        for root, dirs, _files in os.walk("/kaggle/input"):
            if "merged_base" in dirs and any(d.endswith("_best") for d in dirs):
                return root
    return None


def _find_external_merged_base_dir():
    root = _find_external_checkpoints_root()
    if root is None:
        return None
    candidate = os.path.join(root, "merged_base")
    return candidate if os.path.isfile(os.path.join(candidate, "config.json")) else None


def _find_external_category_checkpoint(cat_id):
    """Returns the external dataset's cat{cat_id:02d}_best/ dir IF it has a
    real (non-null) val_pr_auc in its own train_state.json, else None --
    the same explicit-null-means-unvalidated convention save_checkpoint
    itself uses, so a checkpoint that merely exists but never finished
    validating (or was itself a NaN casualty saved under _last) can never
    be mistaken for a good one. Used by v5-style 'reuse, don't retrain'
    logic and by v6's resume-preference (best over last)."""
    root = _find_external_checkpoints_root()
    if root is None:
        return None
    cat_dir = os.path.join(root, f"cat{cat_id:02d}_best")
    state_path = os.path.join(cat_dir, "train_state.json")
    if not os.path.isfile(state_path):
        return None
    with open(state_path) as f:
        state = json.load(f)
    if state.get("val_pr_auc") is None:
        return None
    return cat_dir, state["val_pr_auc"]


def _find_any_external_category_checkpoint(cat_id):
    """v6: resume source for EVERY category, not just the validated ones.
    Prefers cat{NN}_best (real val_pr_auc); falls back to cat{NN}_last
    (a quiet mid-training save, val_pr_auc=null) if that's the only thing
    v5's own session produced for this category before its wall-clock cutoff
    -- any already-trained state is a better starting point than a fresh
    random init, per direct instruction ('Инициализация существующими
    адаптерами+головами'). Returns (checkpoint_dir, source_label,
    val_pr_auc_or_None) or None if nothing exists for this category at all."""
    best = _find_external_category_checkpoint(cat_id)
    if best is not None:
        return best[0], "best", best[1]
    root = _find_external_checkpoints_root()
    if root is None:
        return None
    cat_dir = os.path.join(root, f"cat{cat_id:02d}_last")
    if not os.path.isdir(cat_dir):
        return None
    return cat_dir, "last", None


def merge_and_save_base(resume_dir):
    """Merges the validated general adapter into the base weights and writes
    the result to MERGED_BASE_DIR *before any training happens*.

    Runs pre-fork on CPU: 8 replicas each re-merging the same thing would be
    pure waste, and /kaggle/working is persisted by Kaggle even when a
    session is cancelled or hard-capped, so a crash three hours into
    training still leaves these weights recoverable.

    Idempotent -- a re-run that finds a finished merge reuses it. Also
    reuses the merge already published in the EXTERNAL 8/10/14-checkpoint
    dataset when present, per direct instruction ('Mergенные веса Frida...
    будут использоваться') -- the merge is deterministic given the same
    adapter, but copying ~15s of already-done work is still cheaper and
    removes any chance of a subtly different re-merge."""
    marker = os.path.join(MERGED_BASE_DIR, "config.json")
    if os.path.isfile(marker):
        print(f"[{_elapsed():8.1f}s][MERGE] reusing existing merge at {MERGED_BASE_DIR}", flush=True)
        return MERGED_BASE_DIR

    external_merge_dir = _find_external_merged_base_dir()
    if external_merge_dir is not None:
        print(f"[{_elapsed():8.1f}s][MERGE] reusing already-published merge from "
              f"{external_merge_dir} (not re-merging)", flush=True)
        shutil.copytree(external_merge_dir, MERGED_BASE_DIR)
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


def build_specialist_model_cpu(merged_dir):
    """Builds the merged-base + fresh-LoRA + head model ONCE, on CPU, BEFORE
    fork -- meant to be wrapped in xmp.MpModelWrapper by the caller so all 8
    ranks share this ONE set of host-memory pages via fork's copy-on-write,
    instead of each rank independently calling T5EncoderModel.from_pretrained
    (a real disk read + fresh CPU allocation) the way build_specialist_model
    used to.

    REAL CRASH (host RAM OOM, three real runs, 2026-08-22): PyTorch/XLA's own
    docs describe this exact anti-pattern -- "instead of creating models on
    each multiprocessing process ..., the model is created once at global
    scope, and then moved into each device inside the spawn() target
    function" -- and this kernel was doing precisely the thing being warned
    against: model construction lived inside _mp_fn (per-rank, post-fork),
    while _load_shared_data()'s own DATA loading already correctly ran
    pre-fork. Splitting this out is the fix: build here (pre-fork, wrapped),
    only `.to(device)` happens per-rank in _mp_fn now -- a cheap transfer of
    already-loaded pages, not a fresh load. Untested against the OOM
    directly (this is one of two independent fixes being tried in parallel,
    per direct instruction, specifically so the effect of each is
    separable) -- see frida_percat_lora's sibling push (maximmarkovbb,
    init_lora_weights="olora" + smaller batch) for the other."""
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

    model = FridaCrossEncoderClassifier(peft_model, LinearHead(hidden_dim))
    return model, hidden_dim, n_trainable


class TrainingDivergedError(RuntimeError):
    """Raised the moment a step's loss is non-finite. Deliberately a
    DIFFERENT exception type than the generic ones the step-level
    try/except already tolerates (a genuinely transient/recoverable
    single-step glitch) -- see the training loop for how this is let
    through instead of being silently caught and skipped like an ordinary
    failed step. Per direct instruction: a NaN loss must fail loudly with
    a clear error, not silently keep "training" on garbage for the rest
    of that category's ~600 steps only to discover the problem at eval
    time (average_precision_score raising on NaN predictions) -- which is
    exactly what the previous version did, and it also wasted real TPU
    wall-clock: every one of the 17 failing categories in the real run
    burned its full step budget before failing."""


def reset_specialist_(model, seed):
    """Returns the adapter + head to a fresh initialisation, in place.

    peft's own convention is lora_A ~ kaiming_uniform, lora_B = 0, so the
    adapter starts as an exact identity and the first steps see the merged
    model unchanged -- which is what makes starting from a strong backbone
    safe. Reproduced here rather than rebuilding the module (see
    build_specialist_model_cpu on why the module is reused)."""
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


def load_specialist_from_checkpoint_(model, checkpoint_dir):
    """Loads an existing category's adapter + head weights into the
    (already-built, reused-across-categories) model IN PLACE -- v6's
    replacement for reset_specialist_ when a real prior checkpoint exists
    for this category (see _find_any_external_category_checkpoint).

    Verifies the load actually changed something, same reasoning as
    merge_and_save_base's own probe-delta check: a silent no-op load
    (wrong dir, key mismatch swallowed by strict=False) would otherwise
    train on top of the FRESH init from build_specialist_model_cpu's own
    get_peft_model() call and look completely normal in the logs."""
    adapter_dir = os.path.join(checkpoint_dir, "adapter")
    head_path = os.path.join(checkpoint_dir, "head.pt")
    if not os.path.isfile(os.path.join(adapter_dir, "adapter_model.safetensors")):
        raise RuntimeError(f"[RESUME] no adapter_model.safetensors under {adapter_dir}")
    if not os.path.isfile(head_path):
        raise RuntimeError(f"[RESUME] no head.pt at {head_path}")

    probe_name = next(n for n, p in model.encoder.named_parameters() if "lora_A" in n)
    before = dict(model.encoder.named_parameters())[probe_name].detach().clone()

    adapter_sd = safetensors_load_file(os.path.join(adapter_dir, "adapter_model.safetensors"))
    load_result = set_peft_model_state_dict(model.encoder, adapter_sd)
    missing = getattr(load_result, "missing_keys", None) or []
    unexpected = getattr(load_result, "unexpected_keys", None)
    # set_peft_model_state_dict does model.load_state_dict(adapter_sd, strict=False)
    # under the hood, and adapter_sd only ever contains the LoRA delta weights (lora_A/
    # lora_B/...) -- every base-model weight (base_model.model.*, *.base_layer.weight,
    # relative_attention_bias, layer_norm, ...) is CORRECTLY reported "missing" by
    # strict=False every single time, real crash confirmed against a live run: every
    # one of ~230 "missing" keys was a base weight, zero were lora_ keys, the adapter
    # itself loaded fine. Only a missing LORA key is a real problem (rank/target-module
    # mismatch); missing base keys are how a partial adapter load is SUPPOSED to look.
    missing_lora = [k for k in missing if "lora_" in k]
    if missing_lora:
        raise RuntimeError(f"[RESUME] adapter load from {adapter_dir} missing LoRA keys: "
                           f"{missing_lora}")
    if unexpected:
        raise RuntimeError(f"[RESUME] adapter load from {adapter_dir} unexpected keys: {unexpected}")

    head_sd = torch.load(head_path, map_location="cpu")
    model.classifier.load_state_dict(head_sd, strict=True)

    # Real crash: cat00_last's own weights were already non-finite (this
    # dataset's non-_best categories are the ORIGINAL pre-NaN-fix run's
    # casualties -- see the NaN root-cause section above; only 07/10/14 ever
    # reached a validated _best there). Loading them produced loss=nan at
    # step 1 of the very next training step, which used to crash the WHOLE
    # multi-category run (see the REAL CRASH note above on
    # TrainingDivergedError never actually being caught at category level).
    # Catch it HERE instead, before a single training step runs on it, so
    # the caller can fall back to a fresh init instead of training on
    # garbage.
    bad = [n for n, p in model.encoder.named_parameters()
           if "lora_" in n and not torch.isfinite(p).all()]
    bad += [n for n, p in model.classifier.named_parameters()
            if not torch.isfinite(p).all()]
    if bad:
        raise RuntimeError(f"[RESUME] checkpoint at {checkpoint_dir} has non-finite "
                           f"values in {len(bad)} param(s) (e.g. {bad[0]}) -- refusing "
                           f"to resume from a corrupted checkpoint.")

    after = dict(model.encoder.named_parameters())[probe_name].detach()
    delta = (after.float() - before.float()).abs().max().item()
    if delta == 0.0:
        raise RuntimeError(
            f"[RESUME] load was a NO-OP: {probe_name} is byte-identical before and "
            f"after loading from {checkpoint_dir} -- the checkpoint did not apply.")
    for name, p in model.encoder.named_parameters():
        if "lora_" in name:
            p.data = p.data.float()
            p.requires_grad = True
    return delta


def make_param_groups(model, lr_scale=1.0):
    """Two groups so the randomly-initialised head and the warm backbone can
    follow different schedules -- see the docstring's HEAD SCHEDULE section."""
    lora_params = [p for n, p in model.encoder.named_parameters()
                   if "lora_" in n and p.requires_grad]
    head_params = [p for p in model.classifier.parameters() if p.requires_grad]
    return [{"params": lora_params, "lr": LR * lr_scale, "name": "lora"},
            {"params": head_params, "lr": LR * lr_scale * HEAD_LR_MULTIPLIER, "name": "head"}]


def _make_head_lr_lambda(warmup_steps, total_steps, min_ratio):
    """Head ramps to its (higher) peak over a SHORT warmup
    (HEAD_WARMUP_STEPS, much shorter than the adapter's own warmup_steps),
    then holds that peak flat until the adapter's own warmup ends, then
    joins the same cosine decay.

    NOT an instant jump to peak at step 0 -- that was the shipped version
    and produced loss=nan from the very first logged step on most
    categories, reproducibly (see HEAD_LR_MULTIPLIER's own comment for the
    real evidence). A step-0 jump to a multiplied LR on a freshly
    -initialised linear head has no gradient history to scale against; a
    short ramp (still much faster than the adapter's) plus gradient
    clipping in the training loop fixes it."""
    head_warmup = max(1, min(HEAD_WARMUP_STEPS, warmup_steps))

    def lr_lambda(step):
        if step < head_warmup:
            return step / head_warmup
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
# Set once in main(), pre-fork, via xmp.MpModelWrapper -- see
# build_specialist_model_cpu's own docstring for why. Global (not folded
# into _SHARED) to match torch_xla's own documented MpModelWrapper pattern
# exactly, so a future reader who's seen that pattern recognizes it here.
_WRAPPED_MODEL = None


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

    # ---- LLM: full train split of frida-llm-train-val-split-no-leak-
    # retroactive (v2), not a fixed subsample -- per direct instruction
    # ("Обучение не на llm-subsample, а на всей train-части llm-разметки").
    # That kernel's own "train" already means "every row not held out for
    # its fixed 100K val" (~11.09M rows); category balance/stratification
    # was already applied there (category -> positive_rate-within-category
    # -> length-within-category), not re-applied here. Its "category"
    # column is the string name (same category_vocab construction, per its
    # own docstring), so it maps onto this run's own category_to_id
    # directly -- no re-join against items.parquet needed at all.
    llm_split_pd = pd.read_parquet(
        _find_dataset_file(LLM_SPLIT_PATH_HINT, "llm_train_val_split.parquet"))
    n_llm_total = len(llm_split_pd)
    llm_pd = llm_split_pd[llm_split_pd["split"] == "train"].reset_index(drop=True)
    del llm_split_pd
    gc.collect()

    llm_pd["cat_id"] = llm_pd["category"].map(category_to_id)
    n_unmapped = int(llm_pd["cat_id"].isna().sum())
    if n_unmapped:
        raise RuntimeError(f"[SETUP] {n_unmapped:,} llm-train rows carry a category not in "
                           f"this run's own category_vocab -- split/vocab mismatch, do not "
                           f"trust this run.")
    llm_pd["cat_id"] = llm_pd["cat_id"].astype(np.int64)
    n_llm_used = len(llm_pd)
    print(f"[{_elapsed():8.1f}s][SETUP] LLM TRAIN (full split, no subsample): "
          f"{n_llm_used:,} rows (of {n_llm_total:,} total split rows, train+val) -- "
          f"frida-llm-train-val-split-no-leak-retroactive v2", flush=True)

    llm_frame = _build_frame(llm_pd["id1"], llm_pd["id2"], llm_pd["target"],
                             llm_pd["cat_id"].to_numpy(), 1, caches)
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

    with open(os.path.join(OUTPUT_DIR, "llm_split_usage_report.json"), "w", encoding="utf-8") as f:
        json.dump({
            "policy": "v6: full train split of frida-llm-train-val-split-no-leak-"
                       "retroactive v2, no per-category cap -- stratification "
                       "(category / positive_rate-within-category / length-within-"
                       "category) was already applied by that kernel, not here",
            "llm_split_path_hint": LLM_SPLIT_PATH_HINT,
            "llm_rows_total_in_split": int(n_llm_total),
            "llm_rows_used_train": int(n_llm_used),
        }, f, ensure_ascii=False, indent=2)

    _SHARED.update(dict(
        caches=caches, pad_id=pad_id, train_by_cat=train_by_cat, val_by_cat=val_by_cat,
        category_vocab=category_vocab, val_fingerprint=val_fingerprint,
        merged_dir=merged_dir, n_llm_total=int(n_llm_total), n_llm_used=int(n_llm_used),
    ))
    print(f"[{_elapsed():8.1f}s][SETUP] done -- ready for fork", flush=True)


# =====================================================================
# SMOKE TEST -- catch an HBM-fit / shape-compile problem in ~1-2 minutes,
# not category N. Adapted from frida-full-sft's own (that kernel earned
# this the hard way -- five real crashes, each now caught here instead of
# hours in). frida-percat-lora never had one: every real bug found in this
# kernel's history (three separate checkpoint-path bugs, the over-strict
# missing-keys check, the DataLoader cleanup leak) was discovered only
# after a real, paid TPU run -- this doesn't fix any of those, it just
# means the NEXT shape/memory-class bug costs 1-2 minutes instead of one.
# =====================================================================

def _make_synthetic_batch(batch_size, pair_max_length, pad_id, vocab_size, device):
    lengths = np.random.randint(pair_max_length // 4, pair_max_length + 1, size=batch_size)
    input_ids = np.full((batch_size, pair_max_length), pad_id, dtype=np.int64)
    attention_mask = np.zeros((batch_size, pair_max_length), dtype=np.int64)
    for i, L in enumerate(lengths):
        input_ids[i, :L] = np.random.randint(3, vocab_size, size=L)  # skip special ids 0/1/2
        attention_mask[i, :L] = 1
    labels = np.random.randint(0, 2, size=batch_size).astype(np.float32)
    return (torch.from_numpy(input_ids).to(device), torch.from_numpy(attention_mask).to(device),
            torch.from_numpy(labels).to(device))


def _smoke_test_train_step(model, device, pad_id, vocab_size, rank):
    _log(rank, "SMOKE", f"warm-up TRAIN step (batch={PER_CORE_TRAIN_BATCH} seq={PAIR_MAX_LENGTH}) "
                        f"-- forcing the real forward+backward compile now, before any real data "
                        f"or wall-clock budget is spent", force=True)
    t0 = time.perf_counter()
    input_ids, attention_mask, labels = _make_synthetic_batch(
        PER_CORE_TRAIN_BATCH, PAIR_MAX_LENGTH, pad_id, vocab_size, device)
    loss = None
    try:
        model.zero_grad(set_to_none=True)
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
        loss.backward()
        xm.mark_step()  # force compile+execution NOW, at the real memory cost
    except Exception:
        _log(rank, "SMOKE", "FAILED during warm-up TRAIN step -- this shape does not fit in HBM. "
                            "Lower PER_CORE_TRAIN_BATCH (or PAIR_MAX_LENGTH) and relaunch. Nothing "
                            "real was touched past this point.", force=True)
        raise
    finally:
        # backward() only populates .grad, no optimizer.step() was ever
        # called (there is no optimizer yet at this point in the run --
        # each category builds its own), so the real merged+LoRA weights
        # are provably bit-for-bit unchanged afterward.
        model.zero_grad(set_to_none=True)
    _log(rank, "SMOKE", f"warm-up TRAIN step OK ({time.perf_counter() - t0:.1f}s, "
                        f"loss={loss.item():.4f}, discarded -- model weights unchanged)", force=True)


def _smoke_test_eval_step(model, device, pad_id, vocab_size, rank):
    _log(rank, "SMOKE", f"warm-up EVAL step (batch={PER_CORE_EVAL_BATCH} seq={PAIR_MAX_LENGTH})",
         force=True)
    t0 = time.perf_counter()
    input_ids, attention_mask, _labels = _make_synthetic_batch(
        PER_CORE_EVAL_BATCH, PAIR_MAX_LENGTH, pad_id, vocab_size, device)
    model.eval()
    try:
        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attention_mask)
            xm.mark_step()
    except Exception:
        _log(rank, "SMOKE", "FAILED during warm-up EVAL step -- this shape does not fit in HBM. "
                            "Lower PER_CORE_EVAL_BATCH and relaunch.", force=True)
        raise
    finally:
        model.train()
    _log(rank, "SMOKE", f"warm-up EVAL step OK ({time.perf_counter() - t0:.1f}s)", force=True)


def _smoke_test_real_loader_batch(model, device, caches, pad_id, train_frame, world_size, rank):
    """Same gap frida-full-sft's FIFTH REAL CRASH documented: a synthetic
    batch can succeed while the exact same model call fails on a real
    MpDeviceLoader-sourced one. Goes through the real make_train_loader ->
    MpDeviceLoader -> next(iter(...)) path at the real shape. Uses a
    THROWAWAY loader (epoch=0, never reused, category 0's own train frame)
    so this cannot disturb any real category's own sampler state."""
    _log(rank, "SMOKE", "warm-up REAL-LOADER batch -- same make_train_loader -> MpDeviceLoader "
                        "path real training uses, not a synthetic batch", force=True)
    t0 = time.perf_counter()
    throwaway_loader, _throwaway_sampler = make_train_loader(
        train_frame, caches, pad_id, PER_CORE_TRAIN_BATCH, world_size, rank, SEED, epoch=0)
    mp_throwaway = pl_xla.MpDeviceLoader(throwaway_loader, device)
    loss = None
    try:
        input_ids, attention_mask, batch_labels, _cats, _scope = next(iter(mp_throwaway))
        batch_labels = batch_labels.to(device)
        model.zero_grad(set_to_none=True)
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, batch_labels)
        loss.backward()
        xm.mark_step()
    except Exception:
        _log(rank, "SMOKE", "FAILED during warm-up REAL-LOADER batch -- this is the exact "
                            "path/shape real training uses, so this failure WOULD have "
                            "recurred mid-run. Nothing beyond this point was touched.", force=True)
        raise
    finally:
        model.zero_grad(set_to_none=True)
        del throwaway_loader, mp_throwaway
        gc.collect()
    _log(rank, "SMOKE", f"warm-up REAL-LOADER batch OK ({time.perf_counter() - t0:.1f}s, "
                        f"loss={loss.item():.4f}, discarded -- model weights unchanged)", force=True)


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
                         f"lr_resume_multiplier={LR_RESUME_MULTIPLIER} "
                         f"llm_train_full_split={_SHARED['n_llm_used']:,} "
                         f"(of {_SHARED['n_llm_total']:,} split rows)", force=True)

    # Built ONCE pre-fork (see build_specialist_model_cpu + _WRAPPED_MODEL in
    # main()); this is a cheap per-rank device transfer of already-loaded
    # host memory, not a fresh T5EncoderModel.from_pretrained() disk read --
    # reset between categories so XLA also only ever compiles one program.
    model = _WRAPPED_MODEL.to(device)
    _log(rank, "MODEL", f"fresh LoRA r={LORA_R} alpha={LORA_ALPHA} "
                        f"dropout={LORA_DROPOUT} trainable={_SHARED['n_trainable']:,}", force=True)
    total_batch = PER_CORE_TRAIN_BATCH * world_size
    _log(rank, "BATCH", f"per_core_train={PER_CORE_TRAIN_BATCH} "
                        f"per_core_eval={PER_CORE_EVAL_BATCH} "
                        f"total_train_batch={total_batch} (fixed, no probing)", force=True)

    # Hardcoded rather than introspected off model.encoder.config: FRIDA's
    # real tokenizer vocab is 93,651 (RobertaTokenizer, byte-BPE -- verified
    # directly against the loaded tokenizer, see CLAUDE.md); the synthetic
    # batch only needs any valid in-range id, not the exact size, and
    # PeftModel's `.config` delegation to the base model isn't something
    # this file has relied on anywhere else.
    vocab_size = 90_000
    _smoke_test_train_step(model, device, pad_id, vocab_size, rank)
    _smoke_test_eval_step(model, device, pad_id, vocab_size, rank)
    _smoke_test_real_loader_batch(model, device, caches, pad_id, train_by_cat[0], world_size, rank)
    # No explicit reset needed after: every smoke step only calls
    # .backward() then zero_grad(set_to_none=True), never an optimizer
    # step, so model.encoder/.classifier's actual .data is provably
    # unchanged from build_specialist_model_cpu's own fresh init -- category
    # 0's own loop iteration (reset_specialist_ or
    # load_specialist_from_checkpoint_, whichever applies) starts from
    # exactly the state it would have without the smoke tests ever running.

    run_start = time.perf_counter()
    results = {}
    stopped = False
    # REAL AUDIT FINDING (2026-08-22): a real run resuming from
    # frida-with-merged-lora-and-8-10-14-cat-loras -- which turned out to be
    # the ORIGINAL pre-NaN-fix run's own casualties, not legitimately-cut-
    # short good checkpoints -- diverged on 8 of the first 10 categories
    # attempted, even from a fresh init with the already-"fixed"
    # HEAD_LR_MULTIPLIER/warmup/grad-clip config. That fix has never
    # actually solved NaN for most categories across MULTIPLE real runs; it
    # only ever kept 07/10/14 alive. If something is systemically wrong
    # this run (a bad merged base, a bad data mix, a config regression),
    # burning the full 8.5h MAX_TRAIN_SECONDS discovering that only at the
    # very end -- one diverged category at a time -- wastes the entire
    # session on a foregone conclusion. Stop loudly instead, once enough
    # CONSECUTIVE categories have diverged that "this one specialist's data
    # is just hard" stops being the more likely explanation.
    MAX_CONSECUTIVE_CATEGORY_FAILURES = 6
    consecutive_category_failures = 0

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
            # Full-split training (v6) should make this practically
            # unreachable, but keep v5's safety net: a category too small to
            # form one global batch this run still shouldn't lose an
            # already-validated external checkpoint -- copy it forward
            # instead of silently dropping it.
            fallback = _find_external_category_checkpoint(cat_id)
            if fallback is not None and rank == 0:
                ext_dir, ext_val = fallback
                dest_dir = os.path.join(OUTPUT_DIR, f"{label_prefix}_best")
                _log(rank, "REUSE", f"[{cat_id:02d}] {cat_name}: only {n_train} train rows "
                                    f"(< one global batch) -- copying forward the existing "
                                    f"validated checkpoint (val_pr_auc={ext_val:.4f}) instead "
                                    f"of training: {ext_dir} -> {dest_dir}", force=True)
                shutil.copytree(ext_dir, dest_dir)
                results[cat_name] = float(ext_val)
            elif fallback is None:
                _log(rank, "SKIP", f"[{cat_id:02d}] {cat_name}: only {n_train} train rows, "
                                   f"< one global batch, no existing checkpoint to fall back "
                                   f"to", force=True)
            continue

        # v6: resume from ANY existing checkpoint (a validated _best, or
        # failing that a quiet mid-training _last) and CONTINUE training on
        # the full llm-train split -- per direct instruction ("Инициализация
        # существующими адаптерами+головами"). v5's "already good -> copy
        # forward, skip entirely" behavior is gone: even a _best checkpoint
        # only ever saw LLM_ROWS_PER_CATEGORY (100K) rows, so there is real
        # value left in continuing on the full split regardless of how good
        # it already is. A category with no checkpoint at all (never reached
        # by the run being resumed) still falls back to reset_specialist_.
        external = _find_any_external_category_checkpoint(cat_id)
        if external is not None:
            ext_dir, ext_source, ext_val = external
            try:
                delta = load_specialist_from_checkpoint_(model, ext_dir)
                lr_scale = LR_RESUME_MULTIPLIER
                _log(rank, "RESUME", f"[{cat_id:02d}] {cat_name}: loaded {ext_source} "
                                     f"checkpoint (val_pr_auc={ext_val}) from {ext_dir} "
                                     f"(probe delta={delta:.2e}) -- continuing at "
                                     f"lr_scale={lr_scale}", force=True)
            except Exception as e:
                # Real crash: cat00_last (this dataset's non-_best categories
                # are the ORIGINAL pre-NaN-fix run's casualties) had already-
                # corrupted weights, caught by load_specialist_from_
                # checkpoint_'s own finiteness check -- but ANY resume
                # failure (a bad file, a shape mismatch from a differently-
                # configured checkpoint) deserves the same fallback: a
                # category that can't be resumed is not a category that
                # should crash the whole 20-category run, it's one that
                # trains from scratch instead.
                _log(rank, "WARN", f"[{cat_id:02d}] {cat_name}: checkpoint at {ext_dir} "
                                   f"unusable ({type(e).__name__}: {e}) -- falling back to "
                                   f"fresh init", force=True)
                reset_specialist_(model, SEED + cat_id)
                lr_scale = 1.0
                external = None
        else:
            reset_specialist_(model, SEED + cat_id)
            lr_scale = 1.0
            _log(rank, "INIT", f"[{cat_id:02d}] {cat_name}: no existing checkpoint found -- "
                               f"fresh init, lr_scale={lr_scale}", force=True)
        model.train()

        # pos_weight from THIS category's own mix -- the lever with real
        # dynamic range (positive rate varies 7.7x across categories, while
        # category SIZE varies only 1.4x and is therefore not weighted).
        labels = train_frame["label"]
        pos_mass = float(labels.sum())
        pos_weight = torch.tensor(
            (len(labels) - pos_mass) / max(pos_mass, 1e-9), device=device)
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")

        optimizer = torch.optim.AdamW(make_param_groups(model, lr_scale=lr_scale),
                                      weight_decay=WEIGHT_DECAY)
        steps_per_epoch = max(1, n_train // total_batch)
        total_steps = steps_per_epoch * EPOCHS
        warmup_steps = max(1, int(total_steps * WARMUP_RATIO))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, [
            _make_lr_lambda(warmup_steps, total_steps, LR_MIN_RATIO),        # adapter
            _make_head_lr_lambda(warmup_steps, total_steps, LR_MIN_RATIO),   # head
        ])

        # v9: direct RSS per category, not inferred from wall-clock gaps.
        # v9's own first real run showed inter-category timing growing
        # (55s -> 73s -> 75s more than the leak-fix-free baseline) alongside
        # the SAME host-RAM-OOM crash the leak fix was meant to prevent --
        # consistent with, but not proof of, the extra gc.collect() calls
        # themselves touching (and possibly COW-duplicating) pages in the
        # large pre-fork-shared heap under CPython's fork+refcount
        # semantics. This makes that hypothesis checkable directly instead
        # of inferred a third time from timing.
        rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        _log(rank, "CAT", f"[{cat_id:02d}] {cat_name}: train={n_train:,} "
                          f"val={len(val_frame['label']):,} steps={total_steps} "
                          f"warmup={warmup_steps} pos_weight={pos_weight.item():.3f} "
                          f"rss={rss_mb:.0f}MB", force=True)

        val_full_loader, val_remainder_loader = make_eval_loader(
            val_frame, caches, pad_id, PER_CORE_EVAL_BATCH)
        global_step = 0
        last_quiet_checkpoint = time.perf_counter()

        diverged = False
        try:
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
                        # FAIL LOUD, not silently: BCEWithLogitsLoss does not
                        # raise on NaN, so without this check a diverged run
                        # would keep computing garbage for the rest of the
                        # category's steps and only surface at eval time. This
                        # check is cheap (one XLA scalar sync via .item(), same
                        # cost class as the existing loss.item() logging below)
                        # and catches it the moment it happens instead.
                        if not torch.isfinite(loss).item():
                            raise TrainingDivergedError(
                                f"[{cat_id:02d}] {cat_name}: loss is {loss.item()} (non-finite) "
                                f"at step {global_step + 1}/{total_steps}")
                        loss.backward()
                        # Added after the same real NaN: clips across BOTH param
                        # groups together (one global norm), matching
                        # clip_grad_norm_'s own default behavior.
                        torch.nn.utils.clip_grad_norm_(
                            [p for g in optimizer.param_groups for p in g["params"]],
                            GRAD_CLIP_NORM)
                        xm.optimizer_step(optimizer)
                        scheduler.step()
                        global_step += 1
                    except TrainingDivergedError:
                        # Let this one through -- caught at the CATEGORY level
                        # below, not swallowed here as "just another skipped
                        # step". Cleans up the same way a skipped step would.
                        try:
                            del logits, per_sample, loss
                        except NameError:
                            pass
                        optimizer.zero_grad(set_to_none=True)
                        gc.collect()
                        raise
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

                    # v6: quiet checkpoints -- adapter + head only (no eval, no
                    # val_pr_auc), so a session killed mid-category loses at most
                    # CHECKPOINT_EVERY_SECONDS of progress instead of the whole
                    # category. Saved under the same "_last" label a NaN-failure
                    # already used, since both mean the same thing to a resumer:
                    # "unvalidated, but has real trained weights."
                    if (time.perf_counter() - last_quiet_checkpoint) > CHECKPOINT_EVERY_SECONDS:
                        xm.mark_step()
                        save_checkpoint(model, optimizer, global_step, epoch, None,
                                        f"{label_prefix}_last", rank,
                                        extra={"category": cat_name, "category_id": cat_id,
                                               "n_train": int(n_train), "quiet_checkpoint": True})
                        last_quiet_checkpoint = time.perf_counter()
                        _log(rank, "CKPT", f"[{cat_id:02d}] {cat_name}: quiet checkpoint "
                                           f"(every {CHECKPOINT_EVERY_SECONDS}s) at "
                                           f"step={global_step}/{total_steps}")

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
        except TrainingDivergedError as e:
            _log(rank, "ERROR", f"[{cat_id:02d}] {cat_name}: training diverged -- {e} -- "
                                f"skipping eval and moving to the next category instead of "
                                f"crashing the whole run.", force=True)
            diverged = True
            # REAL CRASH (host RAM OOM, v8): a raise from inside the step loop
            # jumps straight here, OVER the `del mp_loader, train_loader,
            # sampler` a normal epoch end reaches (a few lines up) -- before
            # v8 this never mattered (divergence used to crash the whole
            # process immediately, so nothing ever accumulated across
            # categories). Now that divergence is the common case (8/10
            # categories in the real run that found this), the abandoned
            # DataLoaders' num_workers=2 subprocesses piled up across rapid
            # category transitions (several categories diverged within 2-5
            # steps) until the host ran out of RAM -- confirmed by the
            # "Your notebook tried to allocate more memory than is
            # available" kill and by rank divergence timing spreading to 28s
            # (vs <1s normally) right before it. Same cleanup as the normal
            # path, just reachable from here too. Unconditional del is safe:
            # TrainingDivergedError can only be raised from inside the step
            # loop, which never runs before these three are assigned for the
            # current epoch. No gc.collect() here -- del alone already drops
            # these to refcount 0 (non-cyclic), and the ONE combined collect
            # below (after val loaders too) covers cycles for the whole
            # category in a single pass instead of three separate full-heap
            # scans stacking up per category (v9's own real run showed
            # per-category wall-clock growing 55-96s more than a run without
            # the extra collect() calls -- consistent with each full GC pass
            # over a large pre-fork-shared heap costing real, and possibly
            # COW-duplicating, time under CPython's fork+refcount semantics).
            del mp_loader, train_loader, sampler

        if diverged:
            xm.mark_step()
            save_checkpoint(model, optimizer, global_step, EPOCHS, None,
                            f"{label_prefix}_last", rank,
                            extra={"category": cat_name, "category_id": cat_id,
                                   "resumed_from": external[1] if external is not None else None,
                                   "lr_scale": lr_scale, "diverged": True})
            results[cat_name] = None
        else:
            try:
                m = evaluate(model, val_full_loader, val_remainder_loader, device, category_vocab)
                pr_auc = m["per_category"].get(cat_name, float("nan"))
                _log(rank, "METRICS", f"[{cat_id:02d}] {cat_name}: val_pr_auc={pr_auc:.4f} "
                                      f"n_val={m['n']}", force=True)
                save_checkpoint(model, optimizer, global_step, EPOCHS, float(pr_auc),
                                f"{label_prefix}_best", rank,
                                extra={"category": cat_name, "category_id": cat_id,
                                       "n_train": int(n_train),
                                       "resumed_from": external[1] if external is not None else None,
                                       "lr_scale": lr_scale,
                                       "llm_is_subsample": False})
                results[cat_name] = float(pr_auc)
            except Exception as e:
                _log(rank, "ERROR", f"[{cat_id:02d}] eval failed: {type(e).__name__}: {e}", force=True)
                save_checkpoint(model, optimizer, global_step, EPOCHS, None,
                                f"{label_prefix}_last", rank,
                                extra={"category": cat_name, "category_id": cat_id,
                                       "resumed_from": external[1] if external is not None else None,
                                       "lr_scale": lr_scale,
                                       "llm_is_subsample": False})
                results[cat_name] = None
        # REAL CRASH (host RAM OOM, v8): val_full_loader/val_remainder_loader
        # (num_workers=2 each) are built fresh every category -- ALL 20 now,
        # not just the ones that needed real training (v8 dropped v5's
        # skip-if-externally-good `continue`, which used to return before
        # these were even constructed) -- and were never explicitly freed,
        # relying only on the implicit refcount drop when the names get
        # reassigned at the top of the NEXT category's loop. Explicit here
        # for the same reason as the train-loader del above: force the
        # worker subprocesses to tear down now, not whenever gc gets to it.
        # ONE gc.collect() here covers both this del and the diverged-path
        # train-loader del above (which deliberately skips its own call) --
        # cycles from the whole category collected in a single pass.
        del val_full_loader, val_remainder_loader
        gc.collect()
        model.train()

        if diverged:
            consecutive_category_failures += 1
        else:
            consecutive_category_failures = 0
        if consecutive_category_failures >= MAX_CONSECUTIVE_CATEGORY_FAILURES:
            raise RuntimeError(
                f"[rank{rank}] {consecutive_category_failures} categories in a row diverged to "
                f"NaN ({cat_id - consecutive_category_failures + 1:02d}..{cat_id:02d}) -- this is "
                f"not '{cat_name}'s own data is hard', it is a systemic problem (merged base, "
                f"data mix, or config) common to all of them. Stopping now instead of burning "
                f"MAX_TRAIN_SECONDS to reach the same conclusion 14 categories later. Checkpoints "
                f"already saved for finished/attempted categories are untouched -- a re-run "
                f"resumes from here once the systemic cause is fixed.")

        # Mirror results to disk after EVERY category: a cancelled session can
        # come back with a 0-byte log, and this is then the only record.
        if rank == 0:
            done = [v for v in results.values() if v is not None]
            with open(RESULTS_JSON, "w", encoding="utf-8") as f:
                json.dump({"per_category_val_pr_auc": results,
                           "macro_over_finished": float(np.mean(done)) if done else None,
                           "n_finished": len(done),
                           "n_categories": len(category_vocab),
                           "llm_is_subsample": False,
                           "llm_split_path_hint": LLM_SPLIT_PATH_HINT,
                           "llm_rows_total_in_split": _SHARED["n_llm_total"],
                           "llm_rows_used_train": _SHARED["n_llm_used"],
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
    global _WRAPPED_MODEL
    os.environ.pop("TPU_PROCESS_ADDRESSES", None)
    os.environ.pop("CLOUD_TPU_TASK_ID", None)

    _load_shared_data()

    # v11: build ONCE here, pre-fork, and wrap -- see
    # build_specialist_model_cpu's docstring (REAL CRASH, host RAM OOM) for
    # why this moved out of _mp_fn. MpModelWrapper's whole point is that the
    # wrap must happen BEFORE the fork inside torch_xla.launch() below, so
    # every rank inherits these host pages via copy-on-write instead of
    # each one loading its own.
    model_cpu, hidden_dim, n_trainable = build_specialist_model_cpu(_SHARED["merged_dir"])
    _SHARED["n_trainable"] = n_trainable
    _WRAPPED_MODEL = xmp.MpModelWrapper(model_cpu)
    print(f"[{_elapsed():8.1f}s][MODEL] built once pre-fork, wrapped for "
          f"copy-on-write sharing across ranks (hidden_dim={hidden_dim}, "
          f"trainable={n_trainable:,})", flush=True)

    torch_xla.launch(_mp_fn, args=(), start_method="fork")


if __name__ == "__main__":
    main()
