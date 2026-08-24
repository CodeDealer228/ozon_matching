"""
FRIDA per-category specialists on the exact continue_c_llm3m `last` full-SFT
backbone: train 20 independent LoRA + deep-head pairs, one per category.

V13 CURRENT CONTRACT (later V3-V12 text is retained only as incident history)
===========================================================================
The attached submit archive is content-searched, SHA256-verified, and loaded
with its exact deep classifier head. Token inputs must be Tokenize Items v3;
v4 is rejected by metadata. All CPU setup and the 823M model build happen once
before fork on the 384GB/224-vCPU TPU host; eight ranks use 28 CPU threads each,
num_workers=0 and two fixed XLA shapes (train 16, eval 32, pair length 188).

AdamW is kept for optimization. Categories are sequential, so only one
optimizer exists in RAM. `best` stores finite weights + validation metadata;
`last` stores weights + AdamW + scheduler + exact batch cursor every 15 minutes
and before eval. A compatible attached previous output resumes all three states.
Forward-loss and post-all-reduce gradient checks are replica-synchronous; a
bad loss never enters backward and a bad gradient never advances AdamW or LR.
The 32-update preflight covers the real forward/backward/reduce/optimizer path
before category 00. The deterministic 100K/category LLM cap keeps the projected
run below Kaggle's 9-hour TPU session cap.

V12 FINAL REPAIR (the later V3-V11 sections are retained as failure history)
===========================================================================
The public Kaggle v11 log exposed two independent blockers.  First, 16
DataLoader worker subprocesses inherited the multi-gigabyte pre-fork cache
state and the host was OOM-killed.  All loaders now use num_workers=0, cache
metadata is hydrated only for sampled rows, and validation is sharded/padded
to one static shape.  Second, v11 checked only finite forward loss before an
uninspected gradient all-reduce/AdamW update.  V12 explicitly reduces first,
checks/clips the global gradient norm, and skips a bad update identically on
all ranks.  Fresh heads start at zero and get a 50-step head-only burn-in; a
96-real-batch preflight exercises the complete optimizer path before cat00.

The full 11.09M LLM train split cannot fit a 9h session (18.6h at the
notebook's own optimistic 170 rows/s).  V12 deterministically preserves the
ten soft-target levels while capping each of the 20 categories at 100K rows:
about 2.29M total rows, or 5.3h even at a conservative 120 rows/s.  The global
8h15 deadline is measured from process setup; a partial category is saved as
_last and can never be mislabeled _best.

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

HEAD SCHEDULE AND GLOBAL-GRADIENT GUARD
===========================================================================
The classifier is zero-initialised at the balanced logit implied by each
category's pos_weight.  For a fresh specialist, the first 50 steps detach the
encoder and fit only that head; LoRA then warms from zero while the stable head
continues at a 2x peak.  Gradients are averaged before their global norm is
checked/clipped, so every rank makes the same guarded update decision.

This is a real fix, not a first-principles design choice: an earlier
version jumped the head straight to peak at step 0 with no ramp and no
gradient clipping anywhere, and a real run (frida-percat-lora.log)
reproducibly hit loss=nan by the very first logged step on 15/20
categories, with 2 more drifting into NaN once a harder batch hit later --
17/20 specialists lost, only 3 (07/10/14) survived.  The later v11 run proved
that head warmup plus rank-local clipping did not solve it.  V12's complete
optimizer preflight and replica-synchronous finite-gradient guard replace
that earlier, incomplete fix.

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
import zipfile
import subprocess
import importlib.util
import importlib.metadata
import ctypes
import contextlib

_PROCESS_START = time.perf_counter()

# Kaggle TPU v5e host: 224 vCPU shared by 8 rank processes. Restrict each
# process to its fair 28-thread share before importing NumPy/PyTorch so BLAS
# pools cannot oversubscribe the host 8x.
for _thread_var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "RAYON_NUM_THREADS"):
    os.environ.setdefault(_thread_var, "28")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

# Kaggle's TPU image ships a mutually compatible torch/torch_xla/
# transformers/peft stack.  Reinstalling peft unconditionally can upgrade a
# dependency underneath torch_xla and has broken otherwise healthy kernels.
# Install only genuinely missing leaf packages, without dependency upgrades.
_MISSING_PACKAGES = []
try:
    _POLARS_INSTALLED = importlib.metadata.version("polars")
except importlib.metadata.PackageNotFoundError:
    _POLARS_INSTALLED = None
if _POLARS_INSTALLED != "1.32.3":
    _MISSING_PACKAGES.append("polars==1.32.3")
if importlib.util.find_spec("peft") is None:
    _MISSING_PACKAGES.append("peft==0.19.1")
if _MISSING_PACKAGES:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "--no-deps", "--upgrade",
         *_MISSING_PACKAGES]
    )

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

from peft import (LoraConfig, TaskType, get_peft_model,
                   get_peft_model_state_dict, set_peft_model_state_dict)
from safetensors.torch import save_file as safetensors_save_file
from safetensors.torch import load_file as safetensors_load_file

import torch_xla
import torch_xla.core.xla_model as xm
from torch_xla import runtime as xr
import torch_xla.distributed.parallel_loader as pl_xla
import torch_xla.distributed.xla_multiprocessing as xmp

print(f"[ENV] torch={torch.__version__} torch_xla={torch_xla.__version__} "
      f"transformers={__import__('transformers').__version__} "
      f"peft={__import__('peft').__version__} polars={pl.__version__}", flush=True)
_TORCH_MM = ".".join(torch.__version__.split("+")[0].split(".")[:2])
_XLA_MM = ".".join(torch_xla.__version__.split("+")[0].split(".")[:2])
if _TORCH_MM != _XLA_MM:
    raise RuntimeError(
        f"Incompatible Kaggle image: torch={torch.__version__}, "
        f"torch_xla={torch_xla.__version__}. Do not pip-upgrade either package."
    )


# =====================================================================
# CONFIG
# =====================================================================

SEED = 42
# Exact project-best backbone requested by the user.  The CPU submit-builder
# kernel is attached as a kernel_source; content search avoids relying on its
# unstable Kaggle mount path.  Hash is from the local archive named below.
BACKBONE_ZIP_NAME = "matching-frida_continue_c_last_188-submit.zip"
BACKBONE_ZIP_SHA256 = "6622022adf93bd159ebd9be2c938feed6cf5e606ce56f0e99c598a02dff4cb73"
BACKBONE_MODEL_SUBDIR = "models/frida-full-sft"

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
# Byte-match the requested backbone's real serving contract.
PAIR_MAX_LENGTH = 188

MAX_CATEGORY_CHARS = 64

CATEGORY_LABEL_TOKEN = 55917
ATTRIBUTES_LABEL_TOKEN = 91837
LABEL_COLON_TOKEN = 30

# Same deep head as the requested project-best backbone.  Every category is
# reset to this proven general head before specialist training.
HEAD_DROPOUT = 0.2
HEAD_HIDDEN = (1024, 512, 128)

# Deliberately SMALLER and more regularized than the project default
# (r=16/alpha=32/dropout=0.10): each specialist sees ~115K rows, not
# 11M, so a larger adapter would mostly memorize.
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.10
LORA_TARGET_MODULES = ["q", "k", "v", "o", "wi_0", "wi_1", "wo"]

# Conservative specialist profile.  The previous 2e-4/6e-4 LoRA/head
# peaks still diverged deterministically on fresh cat00/01/02.  Both sets
# of trainable weights stay fp32; AdamW also uses a larger epsilon below.
LR = 5e-5                  # conservative adapter peak on a strong full-FT base
# v6: categories resuming from an already-trained checkpoint (see
# load_specialist_from_checkpoint_) use LR * LR_RESUME_MULTIPLIER as their
# peak instead of the full fresh-init LR -- continuing an adapter that is
# already at a good point should take smaller steps than initializing one
# from a random start, same reasoning any continued-fine-tuning run in
# this project has already used (e.g. the original LLM continue-SFT's own
# LR=2e-4, half of a from-scratch run's peak). The warmup/clip machinery
# below is UNCHANGED for resumed categories -- only the peak magnitude
# scales down, not the shape of the ramp.
LR_RESUME_MULTIPLIER = 1.0

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
HEAD_LR_MULTIPLIER = 0.2   # 1e-5: preserve the already-good general head
HEAD_WARMUP_STEPS = 50
HEAD_ONLY_STEPS = 0
PREFLIGHT_STEPS = 32
MAX_BAD_STEP_FRACTION = 0.02
GRAD_CLIP_NORM = 1.0
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.10
HUMAN_WEIGHT = 2.0         # human rows count double against LLM soft labels
MAX_POS_WEIGHT = 20.0

# Use the project's already-proven 16/32 train/eval shapes.  Sequence length
# 192 is smaller than the 256-token run that established this fit, so this
# keeps HBM margin without compiling speculative probe shapes.
PER_CORE_TRAIN_BATCH = 16
PER_CORE_EVAL_BATCH = 32

# A full 11.09M-row LLM epoch needs at least 18.6h at the measured 170
# rows/s, before setup/eval.  A deterministic 100K/category cap yields about
# 2.29M total rows (5.3h even at a conservative 120 rows/s), which is the
# only single-session configuration with a defensible margin under Kaggle's
# 9h TPU limit.
LLM_ROWS_PER_CATEGORY = 100_000
EPOCHS = 1                     # per category
MAX_WALL_SECONDS = 29_700      # 8h15 from process start; 45m shutdown margin
LOG_EVERY_STEPS = 50
LAST_SAVE_SECONDS = 900        # full resume checkpoint every 15 wall-clock min

# Used ONLY to size the LR schedule (see docstring). Measured on
# frida-tpu-human-sft at this exact model/sequence/batch. If a future run
# reports a different rate, update this rather than the schedule shape.
PROJECTED_ROWS_PER_SEC = 170   # conservative estimate at seq=188
LR_MIN_RATIO = 0.05            # LR floor as a fraction of peak

WORKING_DIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
OUTPUT_DIR = os.path.join(WORKING_DIR, "frida_percat_lora")
BACKBONE_DIR = os.path.join(WORKING_DIR, "frida_continue_c_last_188")
RESULTS_JSON = os.path.join(OUTPUT_DIR, "percat_results.json")
RUN_CONTRACT_PATH = os.path.join(OUTPUT_DIR, "run_contract.json")

t_script_start = _PROCESS_START

EXPECTED_LLM_SPLIT_TOTAL = 11_187_780
EXPECTED_LLM_SPLIT_TRAIN = 11_087_780
EXPECTED_CATEGORIES = [
    "Автотовары", "Аптека", "Бытовая техника", "Бытовая химия",
    "Галантерея и аксессуары", "Детские товары", "Дом и сад",
    "Канцелярские товары", "Красота и гигиена", "Мебель",
    "Музыкальные инструменты", "Обувь", "Одежда", "Продукты питания",
    "Спорт и отдых", "Строительство и ремонт", "Товары для животных",
    "Хобби и творчество", "Электроника", "Ювелирные изделия",
]


def _elapsed():
    return time.perf_counter() - t_script_start


def _log(rank, stage, msg, force=False):
    if rank == 0 or force:
        print(f"[{_elapsed():8.1f}s][rank{rank}] {stage}: {msg}", flush=True)


def _find_dataset_file(hint, filename):
    if os.path.isfile(hint):
        return hint
    matches = []
    if os.path.isdir("/kaggle/input"):
        for root, _dirs, files in os.walk("/kaggle/input"):
            if filename in files:
                matches.append(os.path.join(root, filename))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        dataset_slug = os.path.basename(os.path.dirname(hint.rstrip("/\\")))
        preferred = [p for p in matches if dataset_slug in p.replace("\\", "/").split("/")]
        if len(preferred) == 1:
            return preferred[0]
        raise RuntimeError(
            f"[SETUP] ambiguous input {filename!r}: found {matches}; "
            f"expected the dataset identified by {hint!r}."
        )
    raise FileNotFoundError(
        f"[SETUP] required Kaggle input file {filename!r} was not found. "
        f"Attach the dataset containing {hint}."
    )


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
    if meta.get("prefix") != "paraphrase: " or "Category" not in meta.get("text_formula", ""):
        raise RuntimeError(
            f"[CACHE:{name}] cache contract is not Tokenize Items v3 "
            f"(prefix={meta.get('prefix')!r}, formula={meta.get('text_formula')!r})")

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

    # searchsorted index instead of a Python dict: a 12.38M-entry dict costs
    # well over a GB of host RAM that the 8-way fork then has to keep alive.
    order = np.argsort(ids, kind="stable")
    sorted_ids = ids[order]

    # Do not scan the full 12.7GB token matrix or 3.2GB attention mask here.
    # _prepare_cache_rows() runs after pair sampling and touches only rows
    # referenced by the actual 100K/category training set and human val.
    return dict(name=name, n_items=n_items, max_length=max_length,
                input_ids=input_ids, attention_mask_path=attn_path,
                lengths=None, cat_start=None, attr_start=None,
                order=order, sorted_ids=sorted_ids, pad_id=None)


def _prepare_cache_rows(cache, referenced_rows, chunk=100_000):
    """Hydrate token lengths/label offsets only for rows used this run.

    Dense int16 lookup arrays keep collate O(1) without reading every row of
    the 12.38M-item LLM cache into the host page cache before the 8-way fork.
    """
    rows = np.unique(np.asarray(referenced_rows, dtype=np.int64))
    if len(rows) == 0:
        raise RuntimeError(f"[CACHE:{cache['name']}] no referenced rows")
    if rows[0] < 0 or rows[-1] >= cache["n_items"]:
        raise RuntimeError(f"[CACHE:{cache['name']}] referenced row outside cache bounds")

    n_items = cache["n_items"]
    max_length = cache["max_length"]
    lengths = np.full(n_items, -1, dtype=np.int16)
    cat_start = np.full(n_items, -1, dtype=np.int16)
    attr_start = np.full(n_items, -1, dtype=np.int16)
    attention_mask = np.memmap(
        cache["attention_mask_path"], dtype=np.int8, mode="r",
        shape=(n_items, max_length),
    )

    t0 = time.perf_counter()
    for lo in range(0, len(rows), chunk):
        selected = rows[lo:lo + chunk]
        mask_block = np.asarray(attention_mask[selected])
        token_block = np.asarray(cache["input_ids"][selected])
        lengths[selected] = mask_block.sum(axis=1).astype(np.int16)

        for tok_a, destination in (
            (CATEGORY_LABEL_TOKEN, cat_start),
            (ATTRIBUTES_LABEL_TOKEN, attr_start),
        ):
            hits = ((token_block[:, :-1] == tok_a) &
                    (token_block[:, 1:] == LABEL_COLON_TOKEN))
            has_hit = hits.any(axis=1)
            if has_hit.any():
                destination[selected[has_hit]] = hits[has_hit].argmax(axis=1).astype(np.int16)
        del mask_block, token_block

    used_lengths = lengths[rows].astype(np.int32)
    bad_length = (used_lengths < 2) | (used_lengths > max_length)
    if bad_length.any():
        example = int(rows[np.flatnonzero(bad_length)[0]])
        raise RuntimeError(
            f"[CACHE:{cache['name']}] invalid token length for referenced row {example}: "
            f"{int(lengths[example])}"
        )

    sample_rows = np.random.RandomState(SEED).choice(
        rows, size=min(50, len(rows)), replace=False)
    for r in sample_rows:
        length = int(lengths[r])
        if int(cache["input_ids"][r, 0]) != 1:
            raise RuntimeError(f"[{cache['name']}] row {r}: expected bos=1")
        if int(cache["input_ids"][r, length - 1]) != 2:
            raise RuntimeError(f"[{cache['name']}] row {r}: expected eos=2")

    pad_id = _derive_pad_id(cache["input_ids"], lengths, max_length,
                            cache["name"], candidate_rows=rows)
    for array in (lengths, cat_start, attr_start):
        array.flags.writeable = False
    cache.update(lengths=lengths, cat_start=cat_start, attr_start=attr_start,
                 pad_id=pad_id, n_prepared_rows=int(len(rows)))
    del attention_mask, used_lengths
    print(f"[CACHE:{cache['name']}] prepared {len(rows):,}/{n_items:,} referenced rows "
          f"({time.perf_counter() - t0:.1f}s)", flush=True)


def _derive_pad_id(input_ids, lengths, max_length, name, candidate_rows=None):
    if candidate_rows is None:
        candidate_rows = np.arange(len(lengths), dtype=np.int64)
    candidate_rows = np.asarray(candidate_rows, dtype=np.int64)
    candidates = candidate_rows[(lengths[candidate_rows] >= 0) &
                                (lengths[candidate_rows] < max_length)]
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
    def __init__(self, scope, row1, row2, labels, categories, valid=None):
        self.scope = scope
        self.row1 = row1
        self.row2 = row2
        self.labels = labels
        self.categories = categories
        self.valid = valid

    def __len__(self):
        return len(self.row1)

    def __getitem__(self, i):
        item = (self.scope[i], self.row1[i], self.row2[i],
                self.labels[i], self.categories[i])
        if self.valid is not None:
            item += (self.valid[i],)
        return item


def make_pair_collate_fn(caches, pad_id, pair_max_length, include_valid=False):
    def collate_fn(batch):
        if include_valid:
            scope, row1, row2, labels, categories, valid = zip(*batch)
        else:
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
        result = (torch.from_numpy(input_ids), torch.from_numpy(attention_mask),
                  torch.tensor(labels, dtype=torch.float32),
                  np.asarray(categories, dtype=np.int64),
                  torch.tensor([int(s) for s in scope], dtype=torch.int64))
        if include_valid:
            result += (torch.tensor(valid, dtype=torch.bool),)
        return result
    return collate_fn


def make_train_loader(frame, caches, pad_id, per_core_batch, num_replicas, rank, seed, epoch):
    dataset = PairIndexDataset(frame["scope"], frame["row1"], frame["row2"],
                               frame["label"], frame["category"])
    sampler = DistributedSampler(dataset, num_replicas=num_replicas, rank=rank, shuffle=True,
                                 seed=seed, drop_last=True)
    sampler.set_epoch(epoch)
    loader = DataLoader(dataset, batch_size=per_core_batch, sampler=sampler, drop_last=True,
                        collate_fn=make_pair_collate_fn(caches, pad_id, PAIR_MAX_LENGTH),
                        num_workers=0, pin_memory=False, persistent_workers=False)
    return loader, sampler


def make_eval_loader(frame, caches, pad_id, eval_batch, num_replicas, rank):
    """Return one fixed-shape, disjoint validation shard per TPU rank.

    Every rank receives the same number of full batches.  Padding repeats a
    real row but carries valid=False, so it is removed after all_gather and
    never enters PR-AUC.  This replaces both 8x redundant evaluation and 20
    category-specific remainder shapes.
    """
    n = len(frame["label"])
    if n == 0:
        raise RuntimeError("validation frame is empty")
    slots_per_rank = max(1, math.ceil(n / (num_replicas * eval_batch))) * eval_batch
    global_slots = rank * slots_per_rank + np.arange(slots_per_rank, dtype=np.int64)
    valid = global_slots < n
    source_rows = np.where(valid, global_slots, 0)
    dataset = PairIndexDataset(
        frame["scope"][source_rows], frame["row1"][source_rows],
        frame["row2"][source_rows], frame["label"][source_rows],
        frame["category"][source_rows], valid=valid,
    )
    return DataLoader(
        dataset, batch_size=eval_batch, shuffle=False, drop_last=True,
        collate_fn=make_pair_collate_fn(caches, pad_id, PAIR_MAX_LENGTH,
                                        include_valid=True),
        num_workers=0, pin_memory=False, persistent_workers=False,
    )


# =====================================================================
# MODEL
# =====================================================================

class FullSftHead(nn.Module):
    """Exact deep-head contract stored in the requested c_last submit."""

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


class FridaCrossEncoderClassifier(nn.Module):
    def __init__(self, encoder, head):
        super().__init__()
        self.encoder = encoder
        self.classifier = head

    def forward(self, input_ids, attention_mask, detach_encoder=False):
        context = torch.no_grad() if detach_encoder else contextlib.nullcontext()
        with context:
            with torch.autocast("xla", dtype=torch.bfloat16):
                out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :].float()
        if detach_encoder:
            cls = cls.detach()
        return self.classifier(cls).view(-1).float()


def _find_external_checkpoints_root():
    """Only accept checkpoints staged under this run's verified contract."""
    return OUTPUT_DIR if os.path.isfile(RUN_CONTRACT_PATH) else None


def _run_contract():
    return {
        "schema_version": 1,
        "backbone_zip": BACKBONE_ZIP_NAME,
        "backbone_sha256": BACKBONE_ZIP_SHA256,
        "token_cache_dataset": "maxamsecret/frida-tokenize-items-v3",
        "pair_max_length": PAIR_MAX_LENGTH,
        "lora": {"r": LORA_R, "alpha": LORA_ALPHA,
                 "dropout": LORA_DROPOUT,
                 "target_modules": LORA_TARGET_MODULES},
        "head": {"type": "FullSftHead", "hidden": list(HEAD_HIDDEN),
                 "dropout": HEAD_DROPOUT},
        "optimizer": "AdamW",
    }


def stage_compatible_resume():
    """Copy an attached previous output only when its contract is exact.

    A first run has no resume source. For a continuation, attach the previous
    kernel output (or a dataset made from it); content discovery handles both
    Kaggle mount layouts. Incompatible old per-category runs are ignored.
    """
    if not os.path.isdir("/kaggle/input"):
        return
    expected = _run_contract()
    candidates = []
    for root, _dirs, files in os.walk("/kaggle/input"):
        if "run_contract.json" not in files:
            continue
        path = os.path.join(root, "run_contract.json")
        try:
            with open(path, encoding="utf-8") as handle:
                if json.load(handle) == expected:
                    candidates.append(root)
        except (OSError, json.JSONDecodeError):
            continue
    if len(candidates) > 1:
        raise RuntimeError(f"multiple compatible resume roots found: {candidates}")
    if not candidates:
        return
    source = candidates[0]
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    copied = []
    for name in os.listdir(source):
        if not (name.startswith("cat") or name in {
                "run_contract.json", "percat_results.json", "specialists_manifest.json"}):
            continue
        src = os.path.join(source, name)
        dst = os.path.join(OUTPUT_DIR, name)
        if os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
        elif os.path.isfile(src):
            shutil.copy2(src, dst)
        copied.append(name)
    print(f"[{_elapsed():8.1f}s][RESUME] staged {len(copied)} artifacts from {source}",
          flush=True)


def write_run_contract():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    temp = RUN_CONTRACT_PATH + ".tmp"
    with open(temp, "w", encoding="utf-8") as handle:
        json.dump(_run_contract(), handle, ensure_ascii=False, indent=2)
    os.replace(temp, RUN_CONTRACT_PATH)


def _find_backbone_zip():
    matches = []
    for root, _dirs, files in os.walk("/kaggle/input"):
        if BACKBONE_ZIP_NAME in files:
            matches.append(os.path.join(root, BACKBONE_ZIP_NAME))
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one {BACKBONE_ZIP_NAME!r} from the attached "
            f"submit-build kernel, found {matches}")
    return matches[0]


def prepare_backbone():
    """Verify and extract the exact requested project-best submit pre-fork."""
    marker = os.path.join(BACKBONE_DIR, BACKBONE_MODEL_SUBDIR, "config.json")
    if os.path.isfile(marker) and os.path.isfile(os.path.join(BACKBONE_DIR, "classifier.pt")):
        return BACKBONE_DIR

    archive = _find_backbone_zip()
    digest = hashlib.sha256()
    with open(archive, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != BACKBONE_ZIP_SHA256:
        raise RuntimeError(
            f"wrong backbone archive: sha256={actual}, expected={BACKBONE_ZIP_SHA256}")

    wanted = {
        "classifier.pt",
        f"{BACKBONE_MODEL_SUBDIR}/config.json",
        f"{BACKBONE_MODEL_SUBDIR}/model.safetensors",
    }
    temp_dir = BACKBONE_DIR + ".tmp"
    if os.path.isdir(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)
    with zipfile.ZipFile(archive) as zf:
        names = set(zf.namelist())
        missing = wanted - names
        if missing:
            raise RuntimeError(f"backbone archive missing required files: {sorted(missing)}")
        for name in names:
            if name == "classifier.pt" or name.startswith(BACKBONE_MODEL_SUBDIR + "/"):
                zf.extract(name, temp_dir)
    os.replace(temp_dir, BACKBONE_DIR)
    with open(os.path.join(BACKBONE_DIR, "BACKBONE_PROVENANCE.json"), "w", encoding="utf-8") as f:
        json.dump({"archive": BACKBONE_ZIP_NAME, "sha256": actual,
                   "pair_max_length": PAIR_MAX_LENGTH}, f, indent=2)
    print(f"[{_elapsed():8.1f}s][BACKBONE] verified + extracted {archive}", flush=True)
    return BACKBONE_DIR


def _find_any_external_category_checkpoint(cat_id):
    """v6: resume source for EVERY category, not just the validated ones.
    Prefers cat{NN}_best (real val_pr_auc); falls back to cat{NN}_last
    (a quiet mid-training save, val_pr_auc=null) if that's the only thing
    v5's own session produced for this category before its wall-clock cutoff
    -- any already-trained state is a better starting point than a fresh
    random init, per direct instruction ('Инициализация существующими
    адаптерами+головами'). Returns (checkpoint_dir, source_label,
    val_pr_auc_or_None) or None if nothing exists for this category at all."""
    root = _find_external_checkpoints_root()
    if root is None:
        return None
    cat_dir = os.path.join(root, f"cat{cat_id:02d}_last")
    state_path = os.path.join(cat_dir, "train_state.json")
    if not os.path.isfile(state_path):
        previous = cat_dir + ".previous"
        previous_state = os.path.join(previous, "train_state.json")
        if os.path.isfile(previous_state):
            cat_dir, state_path = previous, previous_state
    if not os.path.isfile(state_path):
        return None
    with open(state_path, encoding="utf-8") as handle:
        state = json.load(handle)
    if state.get("checkpoint_kind") != "last":
        raise RuntimeError(f"[RESUME] {cat_dir} is not a full last checkpoint")
    for filename in ("optimizer.pt", "scheduler.pt"):
        if not os.path.isfile(os.path.join(cat_dir, filename)):
            raise RuntimeError(f"[RESUME] {cat_dir} is missing {filename}")
    return cat_dir, "last", None


def build_specialist_model_cpu(backbone_dir):
    """Build the requested full-SFT base + fresh LoRA + proven head pre-fork.

    Builds the model ONCE, on CPU, BEFORE
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
        base = T5EncoderModel.from_pretrained(
            os.path.join(backbone_dir, BACKBONE_MODEL_SUBDIR), dtype=torch.bfloat16,
                                               attn_implementation="eager")
    except (TypeError, ValueError):
        base = T5EncoderModel.from_pretrained(
            os.path.join(backbone_dir, BACKBONE_MODEL_SUBDIR), dtype=torch.bfloat16)
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

    head = FullSftHead(hidden_dim)
    head_state = torch.load(os.path.join(backbone_dir, "classifier.pt"), map_location="cpu")
    head.load_state_dict(head_state, strict=True)
    if any(not torch.isfinite(v).all().item() for v in head.state_dict().values()):
        raise RuntimeError("requested backbone classifier.pt contains NaN/Inf")
    model = FridaCrossEncoderClassifier(peft_model, head)
    return model, hidden_dim, n_trainable


def reset_specialist_(model, seed):
    """Return LoRA to identity and head to the proven general checkpoint.

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
    model.classifier.load_state_dict(_BASE_HEAD_STATE, strict=True)


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

    adapter_sd = safetensors_load_file(os.path.join(adapter_dir, "adapter_model.safetensors"))
    head_sd = torch.load(head_path, map_location="cpu")
    bad_adapter = [name for name, tensor in adapter_sd.items()
                   if not torch.isfinite(tensor).all().item()]
    bad_head = [name for name, tensor in head_sd.items()
                if not torch.isfinite(tensor).all().item()]
    if bad_adapter or bad_head:
        example = (bad_adapter or bad_head)[0]
        raise RuntimeError(
            f"[RESUME] checkpoint at {checkpoint_dir} contains non-finite CPU "
            f"tensor {example!r}; refusing to copy it to XLA"
        )

    # B starts at exact zero and changes on the first valid LoRA update;
    # unlike A, it also proves a very-early checkpoint was actually applied.
    probe_name = next(n for n, p in model.encoder.named_parameters() if "lora_B" in n)
    before = dict(model.encoder.named_parameters())[probe_name].detach().clone()
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

    model.classifier.load_state_dict(head_sd, strict=True)

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
    """Separate LoRA/head groups for delayed adapter and fast head schedules."""
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


def _make_lr_lambda(warmup_steps, total_steps, min_ratio, delay_steps=0):
    """Linear warmup then cosine decay to a FLOOR of min_ratio * peak, never
    to zero. Written out with LambdaLR (core PyTorch) rather than
    transformers' cosine_with_min_lr, whose availability depends on the
    transformers version the image happens to ship -- and TPU-image version
    drift has already killed one kernel in this project."""
    def lr_lambda(step):
        if step < delay_steps:
            return 0.0
        joint_step = step - delay_steps
        if joint_step < warmup_steps:
            return joint_step / max(1, warmup_steps)
        progress = (joint_step - warmup_steps) / max(
            1, total_steps - delay_steps - warmup_steps)
        progress = min(progress, 1.0)
        return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr_lambda


def _safe_optimizer_update(model, optimizer, scheduler, loss_fn, batch, device,
                           world_size, head_only):
    """One replica-synchronous, non-finite-safe TPU optimizer update.

    The old loop checked only the forward loss, clipped rank-local gradients,
    and then let xm.optimizer_step all-reduce them.  A single rank's NaN/Inf
    backward therefore poisoned every replica and Adam moment.  Here the loss
    flag and gradients are reduced first, the *global* norm is clipped, and a
    bad update skips both AdamW and its scheduler identically on every rank.
    There is no rank-local collective branch.
    """
    input_ids, attention_mask, batch_labels, _cats, batch_scope = batch
    batch_labels = batch_labels.to(device)
    batch_scope = batch_scope.to(device)
    optimizer.zero_grad(set_to_none=True)

    logits = model(input_ids=input_ids, attention_mask=attention_mask,
                   detach_encoder=head_only)
    per_sample = loss_fn(logits, batch_labels)
    weights = torch.where(batch_scope == 0,
                          torch.full_like(per_sample, HUMAN_WEIGHT),
                          torch.ones_like(per_sample))
    loss = (per_sample * weights).sum() / weights.sum()

    local_loss_finite = torch.isfinite(loss.detach()).to(torch.float32)
    finite_replicas = xm.all_reduce(xm.REDUCE_SUM, local_loss_finite)
    global_loss_finite = bool((finite_replicas == float(world_size)).item())
    if not global_loss_finite:
        # torch.where() is not a safe NaN guard for backward: the rejected
        # branch can still produce 0*NaN. All ranks take this same branch
        # because finite_replicas was globally reduced.
        optimizer.zero_grad(set_to_none=True)
        xm.mark_step()
        bad_norm = torch.full((), float("inf"), device=device)
        return loss.detach(), bad_norm, True, False

    loss.backward()

    # This is the reduction half of xm.optimizer_step().  Splitting it out
    # makes the norm check happen on the same averaged gradients on all ranks.
    xm.reduce_gradients(optimizer)
    trainable = [p for group in optimizer.param_groups for p in group["params"]]
    grad_norm = torch.nn.utils.clip_grad_norm_(
        trainable, GRAD_CLIP_NORM, error_if_nonfinite=False, foreach=False)
    step_finite = torch.isfinite(grad_norm.detach())

    # grad_norm is computed after an all-reduce, so this scalar is identical
    # on every rank.  A host decision costs one sync per update (the previous
    # v11 loop already paid that cost for loss.isfinite().item()), but makes a
    # bad batch a TRUE no-op: no Adam momentum/weight-decay change, no moment
    # counter increment, and no scheduler advance.
    update_applied = bool(step_finite.item())
    if update_applied:
        optimizer.step()  # gradients were already averaged above
        scheduler.step()
    else:
        optimizer.zero_grad(set_to_none=True)
        xm.mark_step()
    return loss.detach(), grad_norm.detach(), (not update_applied), update_applied


def _trainable_parameters_finite(model, device, world_size):
    flags = [torch.isfinite(p.detach()).all().to(torch.float32)
             for p in model.parameters() if p.requires_grad]
    # XLA 2.8 left amin(dim=None) unreduced here (350 flags); an explicit
    # dimension guarantees the scalar required by the replica vote.
    local = torch.stack(flags).amin(dim=0) if flags else torch.ones((), device=device)
    if local.numel() != 1:
        raise RuntimeError(f"finite-parameter reduction is not scalar: {tuple(local.shape)}")
    replicas = xm.all_reduce(xm.REDUCE_SUM, local)
    return bool((replicas == float(world_size)).item())


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


def evaluate(model, eval_loader, device, category_names, expected_n):
    model.eval()
    all_logits, all_labels, all_categories = [], [], []
    with torch.no_grad():
        for (input_ids, attention_mask, batch_labels, batch_categories,
             _scope, batch_valid) in eval_loader:
            logits = model(input_ids=input_ids.to(device),
                           attention_mask=attention_mask.to(device))
            labels = batch_labels.to(device)
            categories = torch.from_numpy(batch_categories).to(device=device, dtype=torch.int64)
            valid = batch_valid.to(device=device, dtype=torch.int32)

            # One packed fixed-size collective per batch (not four separate
            # all_gather + host-sync calls). Category ids and valid flags are
            # exact in float32 at these small integer values.
            packed = torch.stack([
                logits.float(), labels.float(), categories.float(), valid.float()
            ], dim=1)
            gathered = xm.all_gather(packed, dim=0)
            xm.mark_step()
            gathered = gathered.cpu()
            gathered_valid = gathered[:, 3] > 0.5
            all_logits.append(gathered[gathered_valid, 0])
            all_labels.append(gathered[gathered_valid, 1])
            all_categories.append(
                gathered[gathered_valid, 2].to(torch.int64).numpy())
    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    categories = np.concatenate(all_categories, axis=0)
    if len(labels) != expected_n:
        raise RuntimeError(
            f"sharded eval recovered {len(labels):,} rows, expected {expected_n:,}"
        )
    if not torch.isfinite(logits).all().item():
        raise RuntimeError("validation logits contain NaN/Inf")
    macro, per_cat = compute_metrics(torch.sigmoid(logits).numpy(), labels.numpy(),
                                     categories, category_names)
    model.train()
    return {"macro_pr_auc": macro, "per_category": per_cat, "n": len(labels)}


# =====================================================================
# CHECKPOINT -- own directory per save, explicit null when unvalidated
# =====================================================================

def save_checkpoint(model, optimizer, scheduler, global_step, epoch,
                    val_macro_pr_auc, label, rank, kind, extra=None):
    """Save weights-only best or a full resumable last checkpoint."""
    if kind not in {"best", "last"}:
        raise ValueError(f"unknown checkpoint kind: {kind}")
    if rank != 0:
        return
    save_dir = os.path.join(OUTPUT_DIR, label)
    temp_dir = save_dir + ".tmp"
    xm.mark_step()
    if os.path.isdir(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=False)

    adapter_dir = os.path.join(temp_dir, "adapter")
    os.makedirs(adapter_dir, exist_ok=True)
    lora_sd = get_peft_model_state_dict(model.encoder)
    safetensors_save_file({k: v.detach().cpu() for k, v in lora_sd.items()},
                          os.path.join(adapter_dir, "adapter_model.safetensors"))
    model.encoder.peft_config["default"].save_pretrained(adapter_dir)

    torch.save({k: v.detach().cpu() for k, v in model.classifier.state_dict().items()},
               os.path.join(temp_dir, "head.pt"))
    if kind == "last":
        xm.save(optimizer.state_dict(), os.path.join(temp_dir, "optimizer.pt"))
        torch.save(scheduler.state_dict(), os.path.join(temp_dir, "scheduler.pt"))

    with open(os.path.join(temp_dir, "train_state.json"), "w", encoding="utf-8") as f:
        json.dump({
            "epoch": epoch, "global_step": global_step,
            "checkpoint_kind": kind,
            "val_pr_auc": val_macro_pr_auc,   # explicit null if unvalidated
            "category": extra.get("category") if extra else None,
            "head": {"type": "FullSftHead", "hidden": list(HEAD_HIDDEN),
                     "dropout": HEAD_DROPOUT},
            "lora": {"r": LORA_R, "alpha": LORA_ALPHA, "dropout": LORA_DROPOUT,
                     "target_modules": LORA_TARGET_MODULES},
            "pair_max_length": PAIR_MAX_LENGTH,
            "backbone_sha256": BACKBONE_ZIP_SHA256,
            "token_cache_dataset": "maxamsecret/frida-tokenize-items-v3",
            **(extra or {}),
        }, f, ensure_ascii=False, indent=2)
    previous_dir = save_dir + ".previous"
    if os.path.isdir(previous_dir):
        shutil.rmtree(previous_dir)
    if os.path.isdir(save_dir):
        os.replace(save_dir, previous_dir)
    os.replace(temp_dir, save_dir)
    if os.path.isdir(previous_dir):
        shutil.rmtree(previous_dir)
    print(f"[SAVE] {label} ({kind}): epoch={epoch} step={global_step} "
          f"val_macro_pr_auc={val_macro_pr_auc}", flush=True)


def _optimizer_state_finite(state):
    def tensors(value):
        if torch.is_tensor(value):
            yield value
        elif isinstance(value, dict):
            for child in value.values():
                yield from tensors(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                yield from tensors(child)
    return all(torch.isfinite(t).all().item() for t in tensors(state))


# =====================================================================
# SHARED CPU-ONLY SETUP -- ONCE, pre-fork
# =====================================================================

_SHARED = {}
# Set once in main(), pre-fork, via xmp.MpModelWrapper -- see
# build_specialist_model_cpu's own docstring for why. Global (not folded
# into _SHARED) to match torch_xla's own documented MpModelWrapper pattern
# exactly, so a future reader who's seen that pattern recognizes it here.
_WRAPPED_MODEL = None
_BASE_HEAD_STATE = None


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
    stage_compatible_resume()
    write_run_contract()
    backbone_dir = prepare_backbone()

    human_cache = load_v1_cache(HUMAN_CACHE_HINT, "human")
    llm_cache = load_v1_cache(LLM_CACHE_HINT, "llm")
    caches = [human_cache, llm_cache]

    # ---- human: the OLD canonical 80/20 split, deliberately (see docstring) ----
    matches_pd = (
        pl.read_parquet(_find_dataset_file(MATCHES_PATH_HINT, "matches.parquet"))
        .filter(pl.col("id1").is_not_null() & pl.col("id2").is_not_null())
        .with_columns([pl.col("target").fill_null(0).cast(pl.Float32),
                       (pl.col("target").fill_null(0) >= 0.5)
                       .cast(pl.Int8).alias("label")])
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
    if category_vocab != EXPECTED_CATEGORIES:
        raise RuntimeError(
            "[SETUP] category vocabulary/order differs from the required 20-category "
            f"checkpoint contract. Got {category_vocab!r}"
        )
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

    # ---- LLM: v2 no-leak train split, then deterministic 100K/category ----
    # Only the five required columns are materialised and pandas is avoided:
    # a pandas frame containing 11M repeated category strings was a large
    # pre-fork host-RAM baseline.  Sampling preserves each category's ten
    # soft-target levels via stratified_llm_sample().
    split_path = _find_dataset_file(LLM_SPLIT_PATH_HINT, "llm_train_val_split.parquet")
    split_scan = pl.scan_parquet(split_path)
    required_columns = {"id1", "id2", "target", "category", "split"}
    actual_columns = set(split_scan.collect_schema().names())
    missing_columns = sorted(required_columns - actual_columns)
    if missing_columns:
        raise RuntimeError(f"[SETUP] LLM split is missing columns: {missing_columns}")

    n_llm_total = int(
        split_scan.select(pl.len().alias("n")).collect(engine="streaming").item()
    )
    category_map = pl.DataFrame({
        "category": category_vocab,
        "cat_id": np.arange(len(category_vocab), dtype=np.int64),
    })
    llm_table = (
        split_scan
        .filter(pl.col("split") == "train")
        .select(["id1", "id2", "target", "category"])
        .join(category_map.lazy(), on="category", how="left")
        .select(["id1", "id2", "target", "cat_id"])
        .collect(engine="streaming")
    )
    n_llm_full_train = len(llm_table)
    if (n_llm_total != EXPECTED_LLM_SPLIT_TOTAL or
            n_llm_full_train != EXPECTED_LLM_SPLIT_TRAIN):
        raise RuntimeError(
            "[SETUP] wrong version of frida-llm-train-val-split: expected "
            f"total/train={EXPECTED_LLM_SPLIT_TOTAL:,}/{EXPECTED_LLM_SPLIT_TRAIN:,}, "
            f"got {n_llm_total:,}/{n_llm_full_train:,}. Attach public v2."
        )

    invalid_llm = llm_table.select([
        (pl.col("id1").is_null() | pl.col("id2").is_null()).sum().alias("null_ids"),
        pl.col("cat_id").is_null().sum().alias("unknown_categories"),
        (pl.col("target").is_null() | ~pl.col("target").is_finite() |
         (pl.col("target") < 0) | (pl.col("target") > 1)).sum().alias("bad_targets"),
    ]).row(0, named=True)
    if any(int(v) for v in invalid_llm.values()):
        raise RuntimeError(f"[SETUP] invalid rows in LLM train split: {invalid_llm}")

    llm_target = llm_table.get_column("target").to_numpy().astype(np.float32, copy=False)
    llm_cat_id = llm_table.get_column("cat_id").to_numpy().astype(np.int64, copy=False)
    keep_idx, sample_report = stratified_llm_sample(
        llm_target, llm_cat_id, LLM_ROWS_PER_CATEGORY, SEED)
    sampled_counts = np.bincount(llm_cat_id[keep_idx], minlength=len(category_vocab))
    if len(keep_idx) == 0 or np.any(sampled_counts <= 0):
        raise RuntimeError(f"[SETUP] empty category in LLM sample: {sampled_counts.tolist()}")
    n_llm_used = int(len(keep_idx))
    print(f"[{_elapsed():8.1f}s][SETUP] LLM TRAIN: sampled {n_llm_used:,} from "
          f"{n_llm_full_train:,} v2 train rows (cap={LLM_ROWS_PER_CATEGORY:,}/category)",
          flush=True)

    llm_frame = _build_frame(
        llm_table.get_column("id1").to_numpy()[keep_idx],
        llm_table.get_column("id2").to_numpy()[keep_idx],
        llm_target[keep_idx], llm_cat_id[keep_idx], 1, caches)

    # Hydrate only token-cache rows that survived sampling (plus all human
    # train/val endpoints).  pad_id can only be compared after this step.
    _prepare_cache_rows(human_cache, np.concatenate([
        human_train_frame["row1"], human_train_frame["row2"],
        human_val_frame["row1"], human_val_frame["row2"],
    ]))
    _prepare_cache_rows(llm_cache, np.concatenate([
        llm_frame["row1"], llm_frame["row2"],
    ]))
    if human_cache["pad_id"] != llm_cache["pad_id"]:
        raise RuntimeError(f"[SETUP] pad_id differs between caches: "
                           f"{human_cache['pad_id']} vs {llm_cache['pad_id']}")
    pad_id = human_cache["pad_id"]
    del llm_table, llm_target, llm_cat_id, keep_idx
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
            "policy": "v13: deterministic target-level-stratified sample from the "
                       "no-leak v2 train split, capped equally per category",
            "llm_split_path_hint": LLM_SPLIT_PATH_HINT,
            "llm_rows_total_in_split": int(n_llm_total),
            "llm_rows_full_train": int(n_llm_full_train),
            "llm_rows_used_train": int(n_llm_used),
            "llm_rows_per_category_cap": LLM_ROWS_PER_CATEGORY,
            "sampled_counts": sampled_counts.tolist(),
            "stratification": sample_report,
        }, f, ensure_ascii=False, indent=2)

    _SHARED.update(dict(
        caches=caches, pad_id=pad_id, train_by_cat=train_by_cat, val_by_cat=val_by_cat,
        category_vocab=category_vocab, val_fingerprint=val_fingerprint,
        backbone_dir=backbone_dir, n_llm_total=int(n_llm_total),
        n_llm_full_train=int(n_llm_full_train), n_llm_used=int(n_llm_used),
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
    input_ids, attention_mask, labels = _make_synthetic_batch(
        PER_CORE_EVAL_BATCH, PAIR_MAX_LENGTH, pad_id, vocab_size, device)
    model.eval()
    try:
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            packed = torch.stack([
                logits.float(), labels.float(),
                torch.zeros_like(logits), torch.ones_like(logits),
            ], dim=1)
            gathered = xm.all_gather(packed, dim=0)
            xm.mark_step()
            if gathered.shape[1] != 4:
                raise RuntimeError(f"unexpected packed eval shape: {tuple(gathered.shape)}")
    except Exception:
        _log(rank, "SMOKE", "FAILED during warm-up EVAL step -- this shape does not fit in HBM. "
                            "Lower PER_CORE_EVAL_BATCH and relaunch.", force=True)
        raise
    finally:
        model.train()
    _log(rank, "SMOKE", f"warm-up EVAL step OK ({time.perf_counter() - t0:.1f}s)", force=True)


def _preflight_optimizer_updates(model, device, caches, pad_id, train_frame,
                                 world_size, rank):
    """Exercise the exact failing path before spending hours on specialists.

    V11's smoke test stopped after backward while the real failure appeared
    after all-reduce + clipping + AdamW (as early as step 3).  This preflight
    runs 96 real cat00 batches, crosses the head-only/joint boundary, verifies
    finite parameters, and then resets the model byte-clean for real training.
    """
    _log(rank, "PREFLIGHT", f"{PREFLIGHT_STEPS} real optimizer updates "
                            f"({HEAD_ONLY_STEPS} head-only, then joint)", force=True)
    t0 = time.perf_counter()
    reset_specialist_(model, SEED)
    labels = train_frame["label"]
    pos_mass = float(labels.sum())
    pos_weight = min((len(labels) - pos_mass) / max(pos_mass, 1e-9), MAX_POS_WEIGHT)
    loss_fn = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight, device=device), reduction="none")
    optimizer = torch.optim.AdamW(
        make_param_groups(model), weight_decay=WEIGHT_DECAY, eps=1e-6)
    total_steps = max(PREFLIGHT_STEPS + 1,
                      len(labels) // (PER_CORE_TRAIN_BATCH * world_size))
    warmup_steps = max(HEAD_WARMUP_STEPS, int(total_steps * WARMUP_RATIO))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, [
        _make_lr_lambda(warmup_steps, total_steps, LR_MIN_RATIO,
                        delay_steps=HEAD_ONLY_STEPS),
        _make_head_lr_lambda(warmup_steps, total_steps, LR_MIN_RATIO),
    ])
    loader, sampler = make_train_loader(
        train_frame, caches, pad_id, PER_CORE_TRAIN_BATCH,
        world_size, rank, SEED, epoch=0)
    mp_loader = pl_xla.MpDeviceLoader(loader, device)
    attempts = 0
    successful_updates = 0
    n_bad = 0
    last_loss = torch.zeros((), device=device)
    last_norm = torch.zeros((), device=device)
    for step, batch in enumerate(mp_loader):
        if step >= PREFLIGHT_STEPS:
            break
        last_loss, last_norm, bad, applied = _safe_optimizer_update(
            model, optimizer, scheduler, loss_fn, batch, device, world_size,
            head_only=(successful_updates < HEAD_ONLY_STEPS))
        attempts += 1
        successful_updates += int(applied)
        n_bad += int(bad)
    xm.mark_step()
    if attempts != PREFLIGHT_STEPS:
        raise RuntimeError(
            f"[PREFLIGHT] loader produced only {attempts}/{PREFLIGHT_STEPS} batches"
        )
    finite_params = _trainable_parameters_finite(model, device, world_size)
    if not finite_params or n_bad != 0 or successful_updates != PREFLIGHT_STEPS:
        raise RuntimeError(
            f"[PREFLIGHT] numerical stability check failed: finite_params={finite_params}, "
            f"updates={successful_updates}/{PREFLIGHT_STEPS}, "
            f"bad_updates={n_bad}, last_loss={last_loss.item()}, "
            f"last_grad_norm={last_norm.item()}"
        )
    _log(rank, "PREFLIGHT", f"passed in {time.perf_counter() - t0:.1f}s: "
                            f"bad_updates={n_bad}/{PREFLIGHT_STEPS} "
                            f"loss={last_loss.item():.4f} grad_norm={last_norm.item():.4f}",
         force=True)

    model.zero_grad(set_to_none=True)
    reset_specialist_(model, SEED)
    xm.mark_step()
    del mp_loader, loader, sampler, optimizer, scheduler, loss_fn


def _deadline_reached(device):
    """Replica-synchronised wall-clock decision, called only at common steps."""
    local = torch.tensor(float(_elapsed() >= MAX_WALL_SECONDS), device=device)
    votes = xm.all_reduce(xm.REDUCE_SUM, local)
    return bool((votes > 0).item())


def _periodic_save_due(device, last_save_wall):
    """Replica-synchronised checkpoint decision at a common log step."""
    local = torch.tensor(
        float(_elapsed() - last_save_wall >= LAST_SAVE_SECONDS), device=device)
    votes = xm.all_reduce(xm.REDUCE_SUM, local)
    return bool((votes > 0).item())


def _write_results(results, category_vocab, stopped):
    done = [v for v in results.values() if v is not None]
    payload = {
        "per_category_val_pr_auc": results,
        "macro_over_finished": float(np.mean(done)) if done else None,
        "n_finished": len(done),
        "n_categories": len(category_vocab),
        "complete": len(done) == len(category_vocab),
        "stopped_at_wall_clock_limit": bool(stopped),
        "llm_is_subsample": True,
        "llm_rows_per_category_cap": LLM_ROWS_PER_CATEGORY,
        "llm_split_path_hint": LLM_SPLIT_PATH_HINT,
        "llm_rows_total_in_split": _SHARED["n_llm_total"],
        "llm_rows_full_train": _SHARED["n_llm_full_train"],
        "llm_rows_used_train": _SHARED["n_llm_used"],
        "pair_max_length": PAIR_MAX_LENGTH,
        "val_fingerprint": _SHARED["val_fingerprint"],
        "split": "OLD canonical pair-level 80/20 seed42",
    }
    temp_path = RESULTS_JSON + ".tmp"
    with open(temp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    os.replace(temp_path, RESULTS_JSON)


def _validate_final_checkpoints(category_vocab):
    """Validate all deliverables and publish an explicit category manifest."""
    entries = []
    for cat_id, cat_name in enumerate(category_vocab):
        label = f"cat{cat_id:02d}_best"
        root = os.path.join(OUTPUT_DIR, label)
        state_path = os.path.join(root, "train_state.json")
        adapter_config_path = os.path.join(root, "adapter", "adapter_config.json")
        adapter_path = os.path.join(root, "adapter", "adapter_model.safetensors")
        head_path = os.path.join(root, "head.pt")
        for required in (state_path, adapter_config_path, adapter_path, head_path):
            if not os.path.isfile(required):
                raise RuntimeError(f"[FINAL] missing specialist artifact: {required}")
        with open(state_path, encoding="utf-8") as handle:
            state = json.load(handle)
        score = state.get("val_pr_auc")
        if score is None or not math.isfinite(float(score)):
            raise RuntimeError(f"[FINAL] {label} has no finite validation PR-AUC")
        if int(state.get("attempted_steps", -1)) != int(state.get("planned_steps", -2)):
            raise RuntimeError(f"[FINAL] {label} is a partial training checkpoint")
        if int(state.get("global_step", -1)) != int(state.get("successful_updates", -2)):
            raise RuntimeError(f"[FINAL] {label} has inconsistent optimizer-step metadata")
        adapter_state = safetensors_load_file(adapter_path)
        head_state = torch.load(head_path, map_location="cpu")
        bad_adapter = [name for name, tensor in adapter_state.items()
                       if not torch.isfinite(tensor).all().item()]
        bad_head = [name for name, tensor in head_state.items()
                    if not torch.isfinite(tensor).all().item()]
        if bad_adapter or bad_head:
            raise RuntimeError(
                f"[FINAL] {label} contains non-finite tensors: "
                f"adapter={bad_adapter[:1]} head={bad_head[:1]}"
            )
        entries.append({
            "category_id": cat_id, "category": cat_name, "checkpoint": label,
            "val_pr_auc": float(score), "global_step": int(state["global_step"]),
        })
        del adapter_state, head_state
    manifest_path = os.path.join(OUTPUT_DIR, "specialists_manifest.json")
    temp_path = manifest_path + ".tmp"
    with open(temp_path, "w", encoding="utf-8") as handle:
        json.dump({"complete": True, "n_specialists": len(entries),
                   "specialists": entries}, handle, ensure_ascii=False, indent=2)
    os.replace(temp_path, manifest_path)
    return manifest_path


def _train_categories_v13(model, device, caches, pad_id, train_by_cat,
                          val_by_cat, category_vocab, world_size, rank):
    """Numerically safe, bounded, sequential 20-specialist training loop."""
    total_batch = PER_CORE_TRAIN_BATCH * world_size
    results = {}
    stopped = False

    for cat_id, cat_name in enumerate(category_vocab):
        if _deadline_reached(device):
            stopped = True
            _log(rank, "STOP", f"wall-clock limit reached before category {cat_id:02d}; "
                                "finished checkpoints are intact", force=True)
            break

        label_prefix = f"cat{cat_id:02d}"
        done_marker = os.path.join(OUTPUT_DIR, f"{label_prefix}_best", "train_state.json")
        done_votes = xm.all_reduce(
            xm.REDUCE_SUM,
            torch.tensor(float(os.path.isfile(done_marker)), device=device),
        )
        n_done_votes = int(done_votes.item())
        if n_done_votes not in (0, world_size):
            raise RuntimeError(f"[RESUME] {done_marker} is not visible on every TPU rank")
        if n_done_votes == world_size:
            with open(done_marker, encoding="utf-8") as handle:
                existing_state = json.load(handle)
            existing_score = existing_state.get("val_pr_auc")
            if existing_score is None or not math.isfinite(float(existing_score)):
                raise RuntimeError(f"[RESUME] invalid completed marker: {done_marker}")
            results[cat_name] = float(existing_score)
            _log(rank, "SKIP", f"[{cat_id:02d}] already complete in /kaggle/working", force=True)
            continue

        train_frame = train_by_cat[cat_id]
        val_frame = val_by_cat[cat_id]
        n_train = len(train_frame["label"])
        if n_train < total_batch:
            raise RuntimeError(
                f"[{cat_id:02d}] {cat_name}: only {n_train} rows; need at least {total_batch}"
            )

        external = _find_any_external_category_checkpoint(cat_id)
        has_external_votes = xm.all_reduce(
            xm.REDUCE_SUM,
            torch.tensor(float(external is not None), device=device),
        )
        n_has_external = int(has_external_votes.item())
        if n_has_external not in (0, world_size):
            _log(rank, "WARN", f"[{cat_id:02d}] external checkpoint visibility differs "
                               "between ranks; all ranks will use fresh init", force=True)
            external = None

        if n_has_external == world_size and external is not None:
            ext_dir, ext_source, ext_val = external
            load_error = None
            try:
                reset_specialist_(model, SEED + cat_id)
                delta = load_specialist_from_checkpoint_(model, ext_dir)
                local_load_ok = 1.0
            except Exception as error:
                load_error = error
                local_load_ok = 0.0
            load_votes = xm.all_reduce(
                xm.REDUCE_SUM, torch.tensor(local_load_ok, device=device))
            if int(load_votes.item()) == world_size:
                lr_scale = LR_RESUME_MULTIPLIER
                _log(rank, "RESUME", f"[{cat_id:02d}] {cat_name}: {ext_source} "
                                     f"val={ext_val}, probe_delta={delta:.2e}", force=True)
            else:
                detail = (f"{type(load_error).__name__}: {load_error}"
                          if load_error is not None else "failed on another rank")
                _log(rank, "WARN", f"[{cat_id:02d}] external checkpoint rejected "
                                   f"collectively ({detail}); fresh init", force=True)
                reset_specialist_(model, SEED + cat_id)
                external = None
                lr_scale = 1.0
        else:
            reset_specialist_(model, SEED + cat_id)
            lr_scale = 1.0
            _log(rank, "INIT", f"[{cat_id:02d}] {cat_name}: general head + identity LoRA",
                 force=True)
        model.train()

        labels = train_frame["label"]
        if not np.isfinite(labels).all() or labels.min() < 0 or labels.max() > 1:
            raise RuntimeError(f"[{cat_id:02d}] invalid training labels")
        pos_mass = float(labels.sum())
        raw_pos_weight = (len(labels) - pos_mass) / max(pos_mass, 1e-9)
        pos_weight_value = min(raw_pos_weight, MAX_POS_WEIGHT)
        pos_weight = torch.tensor(pos_weight_value, device=device)
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")

        optimizer = torch.optim.AdamW(
            make_param_groups(model, lr_scale=lr_scale),
            weight_decay=WEIGHT_DECAY, eps=1e-6)
        steps_per_epoch = n_train // total_batch
        total_steps = steps_per_epoch * EPOCHS
        warmup_steps = max(HEAD_WARMUP_STEPS, int(total_steps * WARMUP_RATIO))
        fresh_delay = HEAD_ONLY_STEPS if external is None else 0
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, [
            _make_lr_lambda(warmup_steps, total_steps, LR_MIN_RATIO,
                            delay_steps=fresh_delay),
            _make_head_lr_lambda(warmup_steps, total_steps, LR_MIN_RATIO),
        ])

        resume_state = None
        if external is not None:
            state_path = os.path.join(external[0], "train_state.json")
            with open(state_path, encoding="utf-8") as handle:
                resume_state = json.load(handle)
            if int(resume_state.get("category_id", -1)) != cat_id:
                raise RuntimeError(f"[{cat_id:02d}] resume category contract mismatch")
            if int(resume_state.get("planned_steps", -1)) != total_steps:
                raise RuntimeError(f"[{cat_id:02d}] resume schedule length mismatch")
            optimizer_state = torch.load(
                os.path.join(external[0], "optimizer.pt"), map_location="cpu")
            if not _optimizer_state_finite(optimizer_state):
                raise RuntimeError(f"[{cat_id:02d}] optimizer checkpoint contains NaN/Inf")
            optimizer.load_state_dict(optimizer_state)
            scheduler.load_state_dict(torch.load(
                os.path.join(external[0], "scheduler.pt"), map_location="cpu"))
            del optimizer_state
        rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        _log(rank, "CAT", f"[{cat_id:02d}] {cat_name}: train={n_train:,} "
                          f"val={len(val_frame['label']):,} steps={total_steps} "
                          f"head_only={fresh_delay} pos_weight={pos_weight_value:.3f} "
                          f"rss_highwater={rss_mb:.0f}MB", force=True)

        attempted_steps = int(resume_state.get("attempted_steps", 0)) if resume_state else 0
        successful_updates = int(resume_state.get("successful_updates", 0)) if resume_state else 0
        n_bad = int(resume_state.get("bad_updates", 0)) if resume_state else 0
        start_epoch = int(resume_state.get("next_epoch", 1)) if resume_state else 1
        resume_batch = int(resume_state.get("next_batch_in_epoch", 0)) if resume_state else 0
        if not (1 <= start_epoch <= EPOCHS + 1):
            raise RuntimeError(f"[{cat_id:02d}] invalid resume epoch {start_epoch}")
        if attempted_steps != successful_updates + n_bad:
            raise RuntimeError(f"[{cat_id:02d}] inconsistent resume counters")
        last_loss = torch.zeros((), device=device)
        last_norm = torch.zeros((), device=device)
        last_save_wall = _elapsed()
        current_epoch = min(start_epoch, EPOCHS)
        next_batch_in_epoch = resume_batch
        for epoch in range(start_epoch, EPOCHS + 1):
            current_epoch = epoch
            train_loader, sampler = make_train_loader(
                train_frame, caches, pad_id, PER_CORE_TRAIN_BATCH,
                world_size, rank, SEED, epoch=epoch)
            mp_loader = pl_xla.MpDeviceLoader(train_loader, device)
            for batch_index, batch in enumerate(mp_loader):
                if epoch == start_epoch and batch_index < resume_batch:
                    continue
                head_only = external is None and successful_updates < HEAD_ONLY_STEPS
                last_loss, last_norm, bad, applied = _safe_optimizer_update(
                    model, optimizer, scheduler, loss_fn, batch, device,
                    world_size, head_only=head_only)
                attempted_steps += 1
                successful_updates += int(applied)
                n_bad += int(bad)
                next_batch_in_epoch = batch_index + 1

                if attempted_steps % LOG_EVERY_STEPS == 0:
                    xm.mark_step()
                    if rank == 0:
                        print(f"[{_elapsed():8.1f}s] cat={cat_id:02d} "
                              f"batch={attempted_steps}/{total_steps} "
                              f"updates={successful_updates} "
                              f"loss={torch.nan_to_num(last_loss).item():.4f} "
                              f"grad_norm={torch.nan_to_num(last_norm).item():.4f} "
                              f"bad_updates={n_bad} "
                              f"lr={scheduler.get_last_lr()[0]:.2e}/"
                              f"{scheduler.get_last_lr()[1]:.2e}", flush=True)
                    if _periodic_save_due(device, last_save_wall):
                        periodic_extra = {
                            "category": cat_name, "category_id": cat_id,
                            "n_train": int(n_train), "planned_steps": int(total_steps),
                            "bad_updates": n_bad, "attempted_steps": attempted_steps,
                            "successful_updates": successful_updates,
                            "next_epoch": epoch,
                            "next_batch_in_epoch": next_batch_in_epoch,
                            "resumed_from": external[1] if external else None,
                            "lr_scale": lr_scale, "llm_is_subsample": True,
                            "llm_rows_per_category": LLM_ROWS_PER_CATEGORY,
                        }
                        xm.rendezvous(f"v13_{label_prefix}_periodic_before_{attempted_steps}")
                        save_checkpoint(
                            model, optimizer, scheduler, successful_updates, epoch, None,
                            f"{label_prefix}_last", rank, "last", periodic_extra)
                        xm.rendezvous(f"v13_{label_prefix}_periodic_after_{attempted_steps}")
                        last_save_wall = _elapsed()
                    if _deadline_reached(device):
                        stopped = True
                        break
            del mp_loader, train_loader, sampler
            if not stopped:
                next_batch_in_epoch = 0
                current_epoch = epoch + 1
            if stopped:
                break

        xm.mark_step()
        finite_params = _trainable_parameters_finite(model, device, world_size)
        if not finite_params:
            raise RuntimeError(f"[{cat_id:02d}] trainable parameters became non-finite")
        if n_bad > math.floor(max(1, attempted_steps) * MAX_BAD_STEP_FRACTION):
            raise RuntimeError(
                f"[{cat_id:02d}] excessive guarded updates: {n_bad}/{attempted_steps}"
            )

        checkpoint_extra = {
            "category": cat_name, "category_id": cat_id, "n_train": int(n_train),
            "planned_steps": int(total_steps), "bad_updates": n_bad,
            "attempted_steps": int(attempted_steps),
            "successful_updates": int(successful_updates),
            "next_epoch": int(current_epoch),
            "next_batch_in_epoch": int(next_batch_in_epoch),
            "resumed_from": external[1] if external is not None else None,
            "lr_scale": lr_scale, "llm_is_subsample": True,
            "llm_rows_per_category": LLM_ROWS_PER_CATEGORY,
        }
        if stopped:
            xm.rendezvous(f"v13_{label_prefix}_deadline_before")
            save_checkpoint(model, optimizer, scheduler, successful_updates,
                            current_epoch, None, f"{label_prefix}_last", rank, "last",
                            extra={**checkpoint_extra, "partial_wall_clock_stop": True})
            results[cat_name] = None
            if rank == 0:
                _write_results(results, category_vocab, stopped=True)
            xm.rendezvous(f"v13_{label_prefix}_partial_saved")
            break

        if attempted_steps != total_steps:
            raise RuntimeError(
                f"[{cat_id:02d}] consumed {attempted_steps} batches, expected {total_steps}"
            )

        # Preserve an exact continuation point before validation. A failure in
        # eval can no longer erase the completed category's optimizer state.
        checkpoint_extra["next_epoch"] = EPOCHS + 1
        checkpoint_extra["next_batch_in_epoch"] = 0
        xm.rendezvous(f"v13_{label_prefix}_trained_before")
        save_checkpoint(model, optimizer, scheduler, successful_updates, EPOCHS,
                        None, f"{label_prefix}_last", rank, "last",
                        extra=checkpoint_extra)
        xm.rendezvous(f"v13_{label_prefix}_trained_saved")

        eval_loader = make_eval_loader(
            val_frame, caches, pad_id, PER_CORE_EVAL_BATCH, world_size, rank)
        try:
            metrics = evaluate(model, eval_loader, device, category_vocab,
                               expected_n=len(val_frame["label"]))
            pr_auc = metrics["per_category"].get(cat_name)
            if pr_auc is None or not math.isfinite(float(pr_auc)):
                raise RuntimeError(f"validation PR-AUC is not finite: {pr_auc}")
        except Exception as error:
            # The full `last` was saved immediately before eval. Do not throw
            # away the remaining TPU session because one validation sidecar
            # failed; the category can be validated/resumed separately.
            _log(rank, "EVAL_ERROR", f"[{cat_id:02d}] {cat_name}: "
                 f"{type(error).__name__}: {error}; kept resumable last", force=True)
            results[cat_name] = None
            del eval_loader, optimizer, scheduler, loss_fn
            model.train()
            xm.rendezvous(f"v13_{label_prefix}_eval_failed")
            if rank == 0:
                _write_results(results, category_vocab, stopped=False)
            continue
        _log(rank, "METRICS", f"[{cat_id:02d}] {cat_name}: val_pr_auc={pr_auc:.4f} "
                              f"n_val={metrics['n']} bad_updates={n_bad}", force=True)
        save_checkpoint(model, optimizer, scheduler, successful_updates, EPOCHS,
                        float(pr_auc), f"{label_prefix}_best", rank, "best",
                        extra=checkpoint_extra)
        results[cat_name] = float(pr_auc)
        del eval_loader, optimizer, scheduler, loss_fn
        model.train()

        # Rank 0 may still be writing checkpoint files; align before any rank
        # resets the shared-shaped specialist for the next category.
        xm.rendezvous(f"v13_{label_prefix}_complete")
        if rank == 0:
            _write_results(results, category_vocab, stopped=False)

    done = [v for v in results.values() if v is not None]
    if rank == 0:
        _write_results(results, category_vocab, stopped=stopped)
        if len(done) == len(category_vocab):
            manifest_path = _validate_final_checkpoints(category_vocab)
            print(f"[FINAL] validated 20/20 finite specialists -> {manifest_path}", flush=True)
    _log(rank, "DONE", f"wall={_elapsed():.0f}s finished={len(done)}/"
                       f"{len(category_vocab)} macro="
                       f"{np.mean(done) if done else float('nan'):.4f}", force=True)


# =====================================================================
# PER-REPLICA ENTRYPOINT
# =====================================================================

def _mp_fn(index):
    rank = xr.global_ordinal()
    world_size = xr.world_size()
    device = torch_xla.device()
    if world_size != 8:
        raise RuntimeError(f"This notebook requires a TPU v5e-8 (8 ranks), got {world_size}")
    if "xla" not in str(device).lower():
        raise RuntimeError(f"Expected an XLA device, got {device}")
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
                         f"llm_train_sample={_SHARED['n_llm_used']:,} "
                         f"(of {_SHARED['n_llm_full_train']:,} v2 train rows)", force=True)

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
    _preflight_optimizer_updates(
        model, device, caches, pad_id, train_by_cat[0], world_size, rank)
    _train_categories_v13(
        model, device, caches, pad_id, train_by_cat, val_by_cat,
        category_vocab, world_size, rank)
    return


def main():
    global _WRAPPED_MODEL, _BASE_HEAD_STATE
    os.environ.pop("TPU_PROCESS_ADDRESSES", None)
    os.environ.pop("CLOUD_TPU_TASK_ID", None)

    _load_shared_data()

    # v11: build ONCE here, pre-fork, and wrap -- see
    # build_specialist_model_cpu's docstring (REAL CRASH, host RAM OOM) for
    # why this moved out of _mp_fn. MpModelWrapper's whole point is that the
    # wrap must happen BEFORE the fork inside torch_xla.launch() below, so
    # every rank inherits these host pages via copy-on-write instead of
    # each one loading its own.
    model_cpu, hidden_dim, n_trainable = build_specialist_model_cpu(_SHARED["backbone_dir"])
    _BASE_HEAD_STATE = {
        k: v.detach().clone() for k, v in model_cpu.classifier.state_dict().items()
    }
    _SHARED["n_trainable"] = n_trainable
    _WRAPPED_MODEL = xmp.MpModelWrapper(model_cpu)
    print(f"[{_elapsed():8.1f}s][MODEL] built once pre-fork, wrapped for "
          f"copy-on-write sharing across ranks (hidden_dim={hidden_dim}, "
          f"trainable={n_trainable:,})", flush=True)

    # Return transient parquet/model-loading arenas to libc, then freeze the
    # long-lived Python object graph so post-fork cyclic GC cannot dirty its
    # copy-on-write pages independently in eight ranks.
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception as error:
        print(f"[WARN] malloc_trim unavailable: {error}", flush=True)
    if hasattr(gc, "freeze"):
        gc.freeze()

    torch_xla.launch(_mp_fn, args=(), start_method="fork")


if __name__ == "__main__":
    main()
