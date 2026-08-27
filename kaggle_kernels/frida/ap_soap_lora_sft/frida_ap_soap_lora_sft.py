"""FRIDA LoRA fine-tune with category-homogeneous stochastic AP/SOAP on TPU v5e-8.

LoRA variant of frida_ap_soap_full_sft.py -- SOAP/StreamingAPLoss are byte-identical
copies (see ../ap_soap_full_sft/tests/), everything else here changes: the encoder is
peft-wrapped (base frozen, small adapters trained) instead of fully fine-tuned, and
every step between fork and the first TRAIN log is now individually timed and
memory-logged. Root cause for this variant: 3 consecutive full-SFT runs (kernel
versions 6/7/8) all got CANCEL_ACKNOWLEDGED at a consistent ~2h9-10m with an
identical 885-byte recovered run_summary.json (still "preflight_complete", never
reaching "training") -- that specific consistency points at Kaggle's undocumented
"TPU idle for 2h -> killed" mechanism (confirmed by Kaggle staff on the product-
feedback board), not random preemption or a code exception. If the actual TPU chip
sits mostly idle while the HOST CPU chews through first-time XLA compilation of
SOAP's ~170-parameter Python-unrolled per-parameter loop, that would look exactly
like this. LoRA cuts trainable parameters from ~823M to ~13M, shrinking that
compile surface by ~60x -- and SOAP_MAX_PRECOND_DIM is dropped from 1024 to 8 for
this run specifically (see below) so it degrades to plain decoupled Adam
everywhere, deliberately removing eigh/qr (an already-confirmed TPU/XLA fragility
via the argsort incident) as a variable while this hypothesis gets tested.

libauc EVALUATED, NOT ADOPTED (2026-08-27)
===========================================================================
Checked whether the pip-installable `libauc` package (libauc.losses.
AveragePrecisionLoss / libauc.losses.APLoss, libauc.optimizers.SOAP) could
replace the custom StreamingAPLoss/SOAP below. Read both source files
directly (pip install libauc==2.0.1, no --deps, then read losses/auc.py and
optimizers/soap.py). Conclusion: keep the custom implementations. Concretely:

  - `libauc.optimizers.SOAP` is NOT the same algorithm as this file's `SOAP`
    class despite the shared name -- it is plain per-parameter Adam/SGD with
    an optional epoch-decay regularizer (see its own `step()`: exp_avg/
    exp_avg_sq, bias correction, addcdiv_ -- textbook Adam, no eigenbasis
    anywhere). It optimizes libauc's own AveragePrecisionLoss specifically
    (its docstring literally says so) and has nothing to do with the SOAP
    paper (Vyas et al. 2409.11321, Shampoo-eigenbasis Adam) this file's own
    `SOAP` class ports. Naming collision only -- adopting libauc's SOAP here
    would silently drop the eigenbasis preconditioning entirely, the actual
    point of using SOAP over plain Adam.
  - `libauc.losses.AveragePrecisionLoss` (aliased `APLoss`) implements the
    SAME algorithm family as this file's `StreamingAPLoss` (same paper, same
    persistent per-row EMA state u_all/u_pos, same coefficient-times-
    surrogate-loss construction) -- but its distributed-training path is
    hardwired to `torch.distributed.nn.all_gather` (see its `forward`'s
    `if self.distributed: ... torch.distributed.nn.all_gather(...)`), which
    requires a `torch.distributed` process group. This project's TPU
    replicas run under torch_xla/PJRT (`xmp.spawn`/`torch_xla.launch`), which
    does NOT initialize `torch.distributed` -- there is no process group for
    `is_distributed()` to find, so libauc's own distributed branch would
    either silently take the single-device path (wrong: scores would never
    be gathered across the 8 replicas, breaking the "one global category-
    homogeneous batch" contract every other part of this kernel depends on)
    or raise, depending on libauc's exact internals at that call. This
    file's own `StreamingAPLoss` + `differentiable_all_gather` (see below)
    already solves exactly this by hand-rolling the gather through
    `xm.all_gather`/torch_xla's own autograd.Function pattern -- swapping in
    libauc's loss would mean re-deriving that same TPU-specific plumbing
    ANYWAY, while losing this file's own already-tested FP32 dtype
    contracts, category-homogeneous anchor-position handling
    (anchor_positions(), PER_CORE_POSITIVES), and reset_()/checkpoint
    integration.
  - Net: both libauc pieces are built for torch.distributed (GPU/NCCL)
    training, not torch_xla/PJRT (TPU) training. The custom StreamingAPLoss/
    SOAP pair below is a from-scratch TPU-native reimplementation of the
    SAME underlying algorithms (same papers, same math) that already
    accounts for this project's specific constraints (FP32 AP state despite
    a BF16 forward, torch_xla's collective API instead of torch.distributed,
    the argsort/eigh TPU fragility documented above). Migrating to libauc
    would trade a working, project-tested implementation for one that needs
    the same TPU-specific rework done again, for no algorithmic gain.
"""
import copy
import gc
import hashlib
import json
import math
import os
import resource
import shutil
import subprocess
import sys
import time
import warnings

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "4")
# Deliberately do not set XLA_USE_BF16/XLA_DOWNCAST_BF16: AP must stay FP32.

# peft is used by build_model_cpu() below but is NOT confirmed preinstalled
# on Kaggle's TPU image (unlike the GPU image lora-sft-human-only-10-epochs
# imports it from directly) -- this project's own frida-rank-sft-a (also
# TPU) installs it explicitly for exactly this reason. Cheap and idempotent
# if it's already present; avoids a bare ImportError deep inside
# build_model_cpu() on a cold TPU session.
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "peft"])

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset


SEED = 42
EXPERIMENT = "frida_ap_soap_lora_sft_v1"
PAIR_MAX_LENGTH = 188
ITEM_MAX_LENGTH = 256
WORLD_SIZE = 8
PER_CORE_TRAIN_BATCH = 16
PER_CORE_POSITIVES = 8
PER_CORE_EVAL_BATCH = 32
HEAD_WIDTHS = (1024, 512, 128)
HEAD_DROPOUT = 0.2

# Same LoRA recipe already validated on this exact base model architecture by
# lora-sft-human-only-10-epochs (11,206,656/834,607,872 = 1.34% trainable, confirmed
# nonzero gradients on all 336 LoRA tensors there) -- T5Attention's real module names
# (q/k/v/o), not BERT's query/key/value; FRIDA's gated-gelu FFN's wi_0/wi_1/wo.
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.10
LORA_TARGET_MODULES = ["q", "k", "v", "o", "wi_0", "wi_1", "wo"]

PEAK_LR = 2e-6          # head only -- classifier.out is already trained (loaded from
                        # head_path), so it gets the SAME conservative peak used by the
                        # full-SFT sibling kernel (frida_ap_soap_full_sft.py's own
                        # PEAK_LR=2e-6), which fine-tunes this exact head end-to-end.
LORA_LR = 3e-5          # LoRA adapters only. REAL BUG (2026-08-26/27, run #1+#2,
                        # annkrasnoshchekova): make_optimizer() originally put LoRA's A/B
                        # matrices in the SAME param groups as the head, both driven by
                        # PEAK_LR=2e-6 -- a value copied verbatim from full_sft's own
                        # full-finetune LR (where EVERY parameter, including the encoder
                        # body, is already pretrained and needs only gentle nudging).
                        # LoRA's B matrix is zero-initialized by construction (see
                        # LORA_PROBE_GRACE_STEPS's own comment): the adapter starts
                        # contributing exactly zero to the forward pass and must "catch
                        # up" through a rank-16 bottleneck from scratch, which standard
                        # LoRA practice (Hu et al. 2106.09685) handles with an LR one to
                        # two orders of magnitude above a full-finetune LR specifically
                        # because of this cold start -- not a smaller LR like a
                        # near-converged full-finetune parameter would want. At
                        # PEAK_LR=2e-6 with 5% warmup, step 2's actual LR was
                        # ~2e-6*0.28 =~ 5.6e-7; combined with SOAP's own first-step-is-
                        # init-only behavior and the B=0 gating, lora_A's weight showed
                        # literally 0.0 max|delta| across the entire 8-step grace window
                        # (run #2 log) -- not a timing-window problem (run #1's diagnosis)
                        # but the adapter moving far too slowly for ANY probe window to
                        # observe, at any width. FIX (run #2): give LoRA its own param
                        # groups at LORA_LR=3e-4 -- the SAME value already validated on
                        # this exact base architecture with this exact LoRA config
                        # (r=16, alpha=32, same target_modules) by
                        # kaggle_kernels/frida/lora/lora-sft-human-only-10-epochs
                        # (confirmed nonzero gradients on all 336 LoRA tensors there).
                        #
                        # REAL BUG (2026-08-27, run #3, cron-diagnosed): run #2 (fixed LR
                        # routing, v3) completed all 19,340 steps with no crash, no NaN,
                        # every logged health signal clean -- but validated_step019340's
                        # own run_summary.json shows real_val_mean_pr_auc collapsing from
                        # baseline_val_mean_pr_auc=0.76164265 (source_init, pre-training)
                        # to final_val_mean_pr_auc=0.68986397, and best_checkpoint stayed
                        # "source_init" for the ENTIRE run -- the model never once beat
                        # its own starting point. Crucially this wasn't a few weak
                        # categories dragging the mean down: ALL 20/20 categories degraded
                        # (e.g. Автотовары 0.690->0.604, Обувь 0.488->0.401, Ювелирные
                        # изделия 0.496->0.439, Хобби и творчество 0.927->0.850) -- a
                        # uniform, whole-model shift, not selective catastrophic
                        # forgetting of hard categories. That pattern is the signature of
                        # an update magnitude too large for how "cooked" this particular
                        # base already is, not of a LoRA that failed to learn.
                        # kaggle_kernels/frida/lora/lora-sft-human-only-10-epochs (where
                        # LORA_LR=3e-4 was validated) trains a FRESH LoRA on top of a
                        # merely decent base -- a completely different regime from THIS
                        # kernel, which loads the project's own best full-FT-continued
                        # FRIDA checkpoint (source_init baseline 0.7616 PR-AUC, in the
                        # same range as the project's actual best submitted checkpoint,
                        # ~0.7595 val / 0.48999 test -- see AGENTS.md's results table).
                        # PEAK_LR=2e-6 for the head was ALREADY chosen deliberately
                        # conservative for exactly this "near-converged, handle gently"
                        # reason (see PEAK_LR's own comment); LORA_LR=3e-4 is 150x that,
                        # a full 2 orders of magnitude of headroom appropriate for a cold
                        # LoRA start on an undertrained base, but far too large a step for
                        # nudging a checkpoint already sitting near this project's
                        # ceiling. No periodic mid-run validation existed in run #2/#3 to
                        # even show WHEN in training the degradation set in (only
                        # start-of-run and end-of-run evaluate() calls were ever made,
                        # both visible in the log) -- so a genuinely better intermediate
                        # checkpoint may have existed and gone uncaptured. FIX (two
                        # parts, see EVAL_CHECKPOINT_INTERVAL_SECONDS below for the
                        # second): cut LORA_LR by 10x to 3e-5 -- still ~15x PEAK_LR
                        # (keeps LoRA's standard higher-than-full-FT-LR headroom, so the
                        # run #1/#2 zero-drift failure class cannot recur) but an order
                        # of magnitude gentler on a base this close to the project
                        # ceiling. The head keeps its own separate, much lower PEAK_LR
                        # unchanged.
MIN_LR_RATIO = 0.20
WARMUP_RATIO = 0.05
WEIGHT_DECAY = 0.02
SOAP_BETAS = (0.9, 0.999)
SOAP_EPS = 1e-6
SOAP_CLIP_VALUE = 1.0
SOAP_SHAMPOO_BETA = 0.999
SOAP_PRECONDITION_FREQUENCY = 10   # paper default; refresh is a cheap QR, not a full eigh
SOAP_MAX_PRECOND_DIM = 8           # Deliberately below LoRA's own r=16 (and every FullSftHead
                                    # width): every LoRA A/B matrix has one side == r=16, so with
                                    # the full-SFT kernel's max_precond_dim=1024 cutoff, ALL 336
                                    # LoRA tensors (not just a handful of head tensors) would get
                                    # real eigh/qr treatment -- a large, untested increase in
                                    # linalg-op call count on a TPU/XLA build that already broke
                                    # once on a related op (argsort). This run is specifically
                                    # about isolating the "TPU idle -> Kaggle kills the session"
                                    # hypothesis (see module docstring), so SOAP is deliberately
                                    # forced to its own plain-decoupled-Adam fallback path
                                    # everywhere (self-tested and proven exact in
                                    # ../ap_soap_full_sft/tests/) -- one fewer variable while that
                                    # gets tested. Raise this back up once a full run completes.
AP_GAMMA = 0.9
AP_SIGMOID_SCALE = 2.0
AP_EPS = 1e-12

LORA_PROBE_GRACE_STEPS = 8
# REAL CRASH (2026-08-26, run #1, annkrasnoshchekova): the original
# "LoRA parameter never changed" contract check ran at EXACTLY
# start_step+2 and raised on a real, correctly-training run. Root cause,
# worked out step by step:
#   Step 1 (SOAP's own first-ever .step() call for EVERY parameter):
#     only initializes GG/Q, applies NO update to anything (see SOAP.step's
#     own "first-ever step only initializes GG/Q" comment) -- model
#     unchanged after this call, for every parameter, by design.
#   Step 2 (SOAP's second call): the FIRST real update lands for every
#     parameter, using each parameter's OWN gradient from step 2's OWN
#     backward pass -- which ran on step-1-frozen weights (step 1 updated
#     nothing). LoRA's B matrix is zero-initialized (standard LoRA
#     convention: the adapter starts as an exact identity), so dL/dB is
#     already nonzero at step 2 (depends only on A_init and the batch) --
#     B gets a real, nonzero update here. But dL/dA = scale * B^T @ dL/dy
#     @ x^T is EXACTLY zero for as long as B is exactly zero (chain rule,
#     not an approximation) -- and step 2's own forward pass still used
#     the PRE-step-2 B (=0, untouched by step 1). So A's step-2 gradient
#     is exactly 0 -> A's SOAP exp_avg/exp_avg_sq are built from an exact
#     zero -> A's step-2 "update" is exactly 0, not just small.
#   Step 3: NOW the forward pass finally sees the post-step-2 B (nonzero)
#     for the first time -> dL/dA is finally nonzero -> A shows its first
#     real, nonzero drift only after THIS step's optimizer.step() call.
# So start_step+2 is PROVABLY, exactly one step too early for the LoRA-A
# probe specifically (the head probe has no such zero-init gating and is
# correctly checked at start_step+2, left unchanged below). FIX: check
# across a grace WINDOW (start_step+2 .. start_step+1+LORA_PROBE_GRACE_
# STEPS) instead of one hard step -- log every step's drift status
# (diagnostics), confirm and stop checking as soon as ANY step in the
# window shows real drift, and only raise once the window closes with
# zero drift throughout. 8 steps gives 6 steps of margin beyond the
# mathematically-proven earliest possible step (step 3) for real-world
# noise (tiny warmup LR, bf16 forward rounding) while still catching the
# actual failure this check defends against (PEFT + gradient checkpointing
# silently dropping the whole gradient chain before it reaches ANY LoRA
# matrix, which would leave A frozen forever, not just for one step) --
# and stays cheap: a handful of small-tensor .cpu() syncs during the first
# few seconds of a multi-hour run, same cost class as the smoke test's own
# upfront checks.
LOG_EVERY_STEPS = 20
# Two real runs of this exact kernel got CANCEL_ACKNOWLEDGED (once at ~2h15m, once at
# ~55m) with no error, no traceback, and Kaggle's own log capture going stale after only
# ~7 minutes each time despite the process staying healthy for another 50min-2h+ after that
# (status polling confirmed RUNNING throughout) -- consistent with the TPU being preempted
# out from under an otherwise-fine job, not a bug in this script. Nothing survives a
# preemption that has no checkpoint on disk yet, so the fix is resilience, not a specific
# root-cause patch: checkpoint REPEATEDLY on a short wall-clock cadence (not once, latched),
# deleting the previous quiet save right after each new one lands so disk stays bounded to
# ~1 checkpoint's footprint (~6.6GB: encoder.pt 3.29GB fp32 + optimizer.pt ~3.29GB, mostly
# bf16 moments) instead of growing unbounded across many saves over a long run.
QUIET_CHECKPOINT_INTERVAL_SECONDS = 25 * 60
EVAL_CHECKPOINT_INTERVAL_SECONDS = 25 * 60
# REAL BUG (found 2026-08-27, v3 "complete" run): this file previously ran
# evaluate() exactly TWICE for the whole run -- once as a pre-training
# baseline, once after every step of the plan had executed -- with NO
# periodic evaluation or best-checkpoint selection in between, unlike this
# project's own rank_sft_a sibling kernel (do_full_checkpoint() every epoch
# via `improved = m["macro_pr_auc"] > best_val`). v3 "completed" (no crash)
# but degraded monotonically on EVERY one of the 20 categories: baseline
# val_mean_pr_auc=0.7616 (the resumed checkpoint, already well-trained full
# FT) -> final val_mean_pr_auc=0.6899 after the full 19,340-step plan (~9.7
# equivalent epochs over the 256,006-row human-train set at
# PER_CORE_TRAIN_BATCH=16 x WORLD_SIZE=8) -- best_checkpoint ended up
# "source_init", i.e. training never found ANY point better than not
# training at all, and the only checkpoint actually SAVED to disk was the
# final (worst-observed) step, because nothing periodic was ever evaluated
# to compare against. With LORA_LR=3e-4 (a cold-start LoRA magnitude) run
# for ~9.7 epochs on top of an already-converged checkpoint, degradation
# this severe is very plausibly overfitting/drift that set in partway
# through -- but there is no way to know from v3's own log, because no
# intermediate point was ever measured. This constant adds that missing
# periodic full validation (using the SAME evaluate() already called for
# baseline/final) so a mid-run best point can actually be found and saved,
# instead of only ever knowing the answer after the entire budget is spent.
# Set coarser than QUIET_CHECKPOINT_INTERVAL_SECONDS (full eval is far more
# expensive: 109,648 val rows vs. a cheap resume-state dump) but still
# frequent enough across a ~7.5h budget (STOP_TRAINING_AT_SECONDS) to catch
# a degradation trend well before the whole budget is spent chasing it.
FULL_EVAL_INTERVAL_SECONDS = 90 * 60  # this is the EVAL_CHECKPOINT_INTERVAL_SECONDS referenced above
# Kaggle TPU sessions stop at 9h. Setup, final human validation and final save
# are included in this clock and get at least 90 minutes of headroom.
STOP_TRAINING_AT_SECONDS = 27_000
OUTPUT_ROOT = os.path.join(
    "/kaggle/working" if os.path.isdir("/kaggle/working") else ".",
    "frida_ap_soap_lora_sft",
)
SUMMARY_PATH = os.path.join(OUTPUT_ROOT, "run_summary.json")
EXPECTED_HUMAN_ROWS = 365_654
EXPECTED_HUMAN_TRAIN = 256_006
EXPECTED_HUMAN_VAL = 109_648
EXPECTED_LLM_TRAIN = 1_000_000
EXPECTED_CATEGORIES = 20
EXPECTED_VAL_FINGERPRINT = "847d04d081432cc0"
EXPECTED_SOURCE_MODEL_BYTES = 1_646_829_336
EXPECTED_SOURCE_HEAD_SHA256 = "0fd9134e8cd4a9b945aee7fb3d12d991208c9ff93d293e958c0ec3b35eb94612"

_START = time.perf_counter()
_SHARED = {}
_WRAPPED_MODEL = None


def log(rank, stage, message, force=False):
    if rank == 0 or force:
        elapsed = time.perf_counter() - _START
        print(f"[{elapsed:8.1f}s][rank{rank}] {stage}: {message}", flush=True)


def mem_info(device=None):
    """Best-effort memory snapshot for this rank's own process (fork-local RSS)
    plus TPU HBM if a device is given. xm.get_memory_info() is known-unreliable
    for compiled-program memory specifically (incidents.md), but still worth
    logging -- any signal beats none while root-causing an unexplained stall."""
    host_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    parts = [f"host_rss={host_mb:.0f}MB"]
    if device is not None:
        try:
            parts.append(f"hbm={xm.get_memory_info(device)}")
        except Exception as error:
            parts.append(f"hbm_unavailable({type(error).__name__}: {error})")
    return " ".join(parts)


def fingerprint_arrays(*arrays):
    digest = hashlib.sha256()
    for array in arrays:
        value = np.ascontiguousarray(array)
        digest.update(str((value.shape, value.dtype)).encode())
        digest.update(memoryview(value).cast("B"))
    return digest.hexdigest()[:16]


def make_category_plan(labels, categories, world_size, per_core_batch, seed):
    """Return [step, world_size * batch] with one category and 1:1 labels per step."""
    labels = np.asarray(labels, dtype=np.uint8)
    categories = np.asarray(categories, dtype=np.int16)
    if per_core_batch % 2:
        raise ValueError("per_core_batch must be even")
    per_core_pos = per_core_batch // 2
    global_pos = world_size * per_core_pos
    category_ids = np.unique(categories)
    rng = np.random.RandomState(seed)
    pools = {}
    required = []
    for category in category_ids:
        pos = np.flatnonzero((categories == category) & (labels == 1)).astype(np.int32)
        neg = np.flatnonzero((categories == category) & (labels == 0)).astype(np.int32)
        if not len(pos) or not len(neg):
            raise RuntimeError(f"category {int(category)} lacks a positive or negative pool")
        rng.shuffle(pos)
        rng.shuffle(neg)
        pools[int(category)] = (pos, neg)
        required.append(max(math.ceil(len(pos) / global_pos), math.ceil(len(neg) / global_pos)))

    # Equal optimizer-step count per category aligns training with macro PR-AUC.
    steps_per_category = max(required)
    schedule = np.repeat(category_ids, steps_per_category)
    rng.shuffle(schedule)
    cursors = {int(category): [0, 0] for category in category_ids}
    plan = np.empty((len(schedule), world_size * per_core_batch), dtype=np.int32)
    offsets = np.arange(global_pos, dtype=np.int64)
    for step, category_value in enumerate(schedule):
        category = int(category_value)
        pos, neg = pools[category]
        cursor = cursors[category]
        selected_pos = pos[(offsets + cursor[0]) % len(pos)]
        selected_neg = neg[(offsets + cursor[1]) % len(neg)]
        cursor[0] += global_pos
        cursor[1] += global_pos
        for rank in range(world_size):
            out = rank * per_core_batch
            lo = rank * per_core_pos
            hi = lo + per_core_pos
            plan[step, out:out + per_core_pos] = selected_pos[lo:hi]
            plan[step, out + per_core_pos:out + per_core_batch] = selected_neg[lo:hi]

    if np.unique(plan).size != len(labels):
        raise RuntimeError("the category-balanced plan does not cover every requested train row")
    return plan, steps_per_category


class StreamingAPLoss(nn.Module):
    """LibAUC-style stochastic AP gradient surrogate with persistent FP32 EMAs."""
    def __init__(self, n_examples, gamma=AP_GAMMA, scale=AP_SIGMOID_SCALE, eps=AP_EPS):
        super().__init__()
        self.gamma = float(gamma)
        self.scale = float(scale)
        self.eps = float(eps)
        self.register_buffer("u_all", torch.zeros(n_examples, dtype=torch.float32))
        self.register_buffer("u_pos", torch.zeros(n_examples, dtype=torch.float32))

    def forward(self, scores, labels, row_indices, anchor_positions):
        scores = scores.float()
        labels = labels.float()
        if scores.dtype != torch.float32 or self.u_all.dtype != torch.float32:
            raise RuntimeError("AP scores and state must be FP32")
        anchor_scores = scores.index_select(0, anchor_positions)
        anchor_rows = row_indices.index_select(0, anchor_positions).long()
        pairwise = torch.sigmoid(
            self.scale * (scores.unsqueeze(0) - anchor_scores.unsqueeze(1))
        ).float()
        current_all = pairwise.mean(dim=1)
        current_pos = (pairwise * labels.unsqueeze(0)).mean(dim=1)
        updated_all = self.gamma * self.u_all[anchor_rows] + (1.0 - self.gamma) * current_all
        updated_pos = self.gamma * self.u_pos[anchor_rows] + (1.0 - self.gamma) * current_pos
        with torch.no_grad():
            self.u_all.index_copy_(0, anchor_rows, updated_all.detach())
            self.u_pos.index_copy_(0, anchor_rows, updated_pos.detach())
        denominator = updated_all.square().clamp_min(self.eps)
        coefficient = (
            updated_pos.unsqueeze(1) - updated_all.unsqueeze(1) * labels.unsqueeze(0)
        ) / denominator.unsqueeze(1)
        return (coefficient.detach() * pairwise).mean()

    @torch.no_grad()
    def reset_(self):
        self.u_all.zero_()
        self.u_pos.zero_()


class SOAP(Optimizer):
    """SOAP (Vyas et al., arxiv:2409.11321): Adam run in Shampoo's eigenbasis.

    Close port of the reference implementation (github.com/nikhilvyas/SOAP),
    trimmed of conv-only machinery this model has no use for (merge_dims,
    channels_last, normalize_grads, precondition_1d -- 1D params here are
    always small biases/LayerNorms, never worth preconditioning), plus this
    project's FP32-master memory contract: exp_avg/exp_avg_sq are stored
    BF16 for any parameter SOAP_MAX_PRECOND_DIM leaves fully unpreconditioned
    (the encoder body -- huge tensors, and project/project_back are an exact
    identity there, so BF16 costs nothing beyond the rounding a plain BF16
    Adam moment already implies), and FP32 for any parameter that gets real
    eigenbasis rotation (the head -- a few MB, so full precision is free and
    avoids compounding round-trip error in a rotating basis). Weight decay
    is decoupled (AdamW-style, applied to the parameter directly), matching
    the reference -- NOT coupled into the gradient like a from-scratch Adam.
    """

    def __init__(self, params, lr, betas=SOAP_BETAS, eps=SOAP_EPS, weight_decay=WEIGHT_DECAY,
                 shampoo_beta=SOAP_SHAMPOO_BETA, precondition_frequency=SOAP_PRECONDITION_FREQUENCY,
                 max_precond_dim=SOAP_MAX_PRECOND_DIM, clip=SOAP_CLIP_VALUE):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                         shampoo_beta=shampoo_beta, precondition_frequency=precondition_frequency,
                         max_precond_dim=max_precond_dim, clip=clip)
        super().__init__(params, defaults)

    @staticmethod
    def _covariance_side(size, max_precond_dim, device):
        if size > max_precond_dim:
            return []
        return torch.zeros(size, size, dtype=torch.float32, device=device)

    @classmethod
    def _init_state(cls, grad, state, group):
        state["step"] = 0
        if grad.dim() > 1:
            state["GG"] = [cls._covariance_side(side, group["max_precond_dim"], grad.device)
                           for side in grad.shape]
        else:
            state["GG"] = [[]]  # 1D params (biases, LayerNorm) are never preconditioned
        state["Q"] = None
        moment_dtype = torch.float32 if any(len(side) for side in state["GG"]) else torch.bfloat16
        state["moment_dtype"] = moment_dtype
        state["exp_avg"] = torch.zeros_like(grad, dtype=moment_dtype)
        state["exp_avg_sq"] = torch.zeros_like(grad, dtype=moment_dtype)

    @staticmethod
    def _orthogonal_basis(covariances):
        basis = []
        for covariance in covariances:
            if not len(covariance):
                basis.append([])
                continue
            eye = torch.eye(covariance.shape[0], dtype=torch.float32, device=covariance.device)
            _, vectors = torch.linalg.eigh(covariance + 1e-30 * eye)
            basis.append(torch.flip(vectors, [1]))
        return basis

    @staticmethod
    def _refresh_basis(covariances, basis):
        # The reference implementation also sorts columns by estimated eigenvalue before the
        # QR step (torch.argsort) and permutes exp_avg_sq to match, to keep "coordinate i"
        # meaning roughly the same thing across refreshes. Dropped here: a real TPU run hit a
        # fatal XLA-internal crash inside this exact preconditioner-refresh window (`Check
        # failed: state Expected an array shape. Got (f32[32], s64[32])` in shape.cc -- the
        # (values, indices) tuple shape torch.argsort/sort produce, at the first
        # precondition_frequency boundary of real training) -- consistent with argsort not
        # being reliably lowered by this torch_xla/TPU build. The power iteration + QR still
        # converges to a valid orthonormal basis for the SAME subspace without the sort; only
        # which physical column ends up "coordinate 0" becomes unordered across refreshes, so
        # exp_avg_sq's per-coordinate estimate is very occasionally attributed to a shuffled
        # coordinate right after a refresh -- a minor, EMA-smoothed cost, not a correctness bug.
        refreshed = []
        for covariance, vectors in zip(covariances, basis):
            if not len(covariance):
                refreshed.append([])
                continue
            power_iterate = covariance @ vectors
            new_vectors, _ = torch.linalg.qr(power_iterate)
            refreshed.append(new_vectors)
        return refreshed

    @staticmethod
    def _rotate(value, basis, transpose):
        """transpose=False projects raw -> eigenbasis; transpose=True is the inverse."""
        for vectors in basis:
            if len(vectors):
                value = torch.tensordot(value, vectors, dims=[[0], [1 if transpose else 0]])
            else:
                value = value.permute(*range(1, value.dim()), 0)
        return value

    @staticmethod
    def _update_covariance(grad, state, group):
        for index, side in enumerate(state["GG"]):
            if not len(side):
                continue
            contracted = [dimension for dimension in range(grad.dim()) if dimension != index]
            outer_product = torch.tensordot(grad, grad, dims=[contracted, contracted])
            side.lerp_(outer_product, 1.0 - group["shampoo_beta"])

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                grad = parameter.grad.detach().float().clamp(-group["clip"], group["clip"])
                state = self.state[parameter]

                if "step" not in state:
                    self._init_state(grad, state, group)
                    self._update_covariance(grad, state, group)
                    state["Q"] = self._orthogonal_basis(state["GG"])
                    continue  # first-ever step only initializes GG/Q, matching the reference --
                              # it must never precondition using the gradient that built Q.

                old_basis = state["Q"]
                grad_projected = self._rotate(grad, old_basis, transpose=False)
                exp_avg = state["exp_avg"].float().mul_(beta1).add_(grad_projected, alpha=1.0 - beta1)
                exp_avg_sq = state["exp_avg_sq"].float().mul_(beta2).addcmul_(
                    grad_projected, grad_projected, value=1.0 - beta2
                )
                state["step"] += 1
                bias1 = 1.0 - beta1 ** state["step"]
                bias2 = 1.0 - beta2 ** state["step"]
                step_size = group["lr"] * (bias2 ** 0.5) / bias1
                denominator = exp_avg_sq.sqrt().add_(group["eps"])
                update = self._rotate(exp_avg / denominator, old_basis, transpose=True)
                parameter.add_(update, alpha=-step_size)
                if group["weight_decay"] > 0.0:
                    parameter.add_(parameter, alpha=-group["lr"] * group["weight_decay"])

                # Preconditioner refresh, exactly like the reference's update_preconditioner:
                # un-rotate the just-updated running mean out of the OLD basis, fold this
                # step's raw gradient into GG, maybe refresh the basis (a cheap power-iteration
                # + QR, not a full eigh -- that only ever runs once, at init, above), then
                # re-rotate exp_avg for next step's use.
                raw_exp_avg = self._rotate(exp_avg, old_basis, transpose=True)
                self._update_covariance(grad, state, group)
                new_basis = old_basis
                if state["step"] % group["precondition_frequency"] == 0:
                    new_basis = self._refresh_basis(state["GG"], old_basis)
                state["Q"] = new_basis
                state["exp_avg"] = self._rotate(raw_exp_avg, new_basis, transpose=False).to(state["moment_dtype"])
                state["exp_avg_sq"] = exp_avg_sq.to(state["moment_dtype"])
        return loss

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        for state in self.state.values():
            moment_dtype = state.get("moment_dtype", torch.float32)
            state["exp_avg"] = state["exp_avg"].to(moment_dtype)
            state["exp_avg_sq"] = state["exp_avg_sq"].to(moment_dtype)

    @torch.no_grad()
    def reset_(self):
        self.state.clear()


class FullSftHead(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        blocks, width = [], hidden_dim
        for hidden in HEAD_WIDTHS:
            blocks.append(nn.Sequential(
                nn.Linear(width, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(HEAD_DROPOUT)
            ))
            width = hidden
        self.blocks = nn.ModuleList(blocks)
        self.out = nn.Linear(width, 1)

    def forward(self, value):
        for block in self.blocks:
            value = block(value)
        return self.out(value).view(-1)


class FridaRanker(nn.Module):
    def __init__(self, encoder, classifier):
        super().__init__()
        self.encoder = encoder
        self.classifier = classifier

    def forward(self, input_ids, attention_mask):
        with torch.autocast("xla", dtype=torch.bfloat16):
            encoded = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            logits = self.classifier(encoded.last_hidden_state[:, 0, :])
        return logits.float()


class IndexDataset(Dataset):
    def __init__(self, rows, valid=None):
        self.rows = np.asarray(rows, dtype=np.int32)
        self.valid = None if valid is None else np.asarray(valid, dtype=np.bool_)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        if self.valid is None:
            return int(self.rows[index])
        return int(self.rows[index]), bool(self.valid[index])


def allocate_pair_budget(length1, length2, budget):
    half = budget // 2
    if length1 <= half:
        return length1, min(length2, budget - length1)
    if length2 <= half:
        return min(length1, budget - length2), length2
    return half, budget - half


def make_collate(frame, caches, pad_id, require_category_batch):
    def collate(batch):
        if isinstance(batch[0], tuple):
            rows = np.asarray([item[0] for item in batch], dtype=np.int32)
            valid = np.asarray([item[1] for item in batch], dtype=np.bool_)
        else:
            rows = np.asarray(batch, dtype=np.int32)
            valid = np.ones(len(rows), dtype=np.bool_)
        scopes = frame["scope"][rows]
        rows1 = frame["row1"][rows]
        rows2 = frame["row2"][rows]
        labels = frame["label"][rows]
        categories = frame["category"][rows]
        if require_category_batch:
            if not np.all(categories == categories[0]):
                raise RuntimeError("category-homogeneous batch contract violated")
            half = len(rows) // 2
            if not (np.all(labels[:half] == 1) and np.all(labels[half:] == 0)):
                raise RuntimeError("fixed positive-first 1:1 batch contract violated")
        input_ids = np.full((len(rows), PAIR_MAX_LENGTH), pad_id, dtype=np.int64)
        attention_mask = np.zeros((len(rows), PAIR_MAX_LENGTH), dtype=np.int64)
        for index, (scope, row1, row2) in enumerate(zip(scopes, rows1, rows2)):
            cache = caches[int(scope)]
            length1 = int(cache["lengths"][row1])
            length2 = int(cache["lengths"][row2])
            keep1, keep2 = allocate_pair_budget(length1, length2, PAIR_MAX_LENGTH)
            input_ids[index, :keep1] = cache["input_ids"][row1, :keep1]
            input_ids[index, keep1:keep1 + keep2] = cache["input_ids"][row2, :keep2]
            attention_mask[index, :keep1 + keep2] = 1
        return (
            torch.from_numpy(input_ids), torch.from_numpy(attention_mask),
            torch.from_numpy(labels.astype(np.float32)),
            torch.from_numpy(categories.astype(np.int64)),
            torch.from_numpy(rows.astype(np.int64)), torch.from_numpy(valid),
        )
    return collate


def find_file(filename):
    found = []
    for root, _dirs, files in os.walk("/kaggle/input"):
        if filename in files:
            found.append(os.path.join(root, filename))
    if len(found) != 1:
        raise RuntimeError(f"expected one {filename!r}, got {found}")
    return found[0]


def find_source_model():
    # The published _188 dataset does NOT put classifier.pt alongside config.json/
    # model.safetensors -- classifier.pt sits at the dataset root while the model files live
    # one level down in models/frida-format-adapter-v4/ (confirmed via `kaggle datasets files`
    # against the real maxamsecret/frida-format-adapter-v4-188 dataset). Locate each
    # independently, the same "exactly one match under /kaggle/input" contract find_file uses.
    found = []
    for root, _dirs, files in os.walk("/kaggle/input"):
        if {"config.json", "model.safetensors"} <= set(files):
            found.append(root)
    if len(found) != 1:
        raise RuntimeError(f"expected one merged FRIDA source directory, got {found}")
    root = found[0]
    model_path = os.path.join(root, "model.safetensors")
    head_path = find_file("classifier.pt")
    if os.path.getsize(model_path) != EXPECTED_SOURCE_MODEL_BYTES:
        raise RuntimeError("source model.safetensors size differs from published _188 weights")
    digest = hashlib.sha256()
    with open(head_path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != EXPECTED_SOURCE_HEAD_SHA256:
        raise RuntimeError("source classifier.pt hash differs from published _188 head")
    return root, head_path


def load_cache(scope):
    found = []
    required = {"cache_meta.json", "ids.bin", "input_ids.bin", "attention_mask.bin"}
    for root, _dirs, files in os.walk("/kaggle/input"):
        if required <= set(files):
            with open(os.path.join(root, "cache_meta.json"), encoding="utf-8") as handle:
                meta = json.load(handle)
            if meta.get("format_version") == 4 and meta.get("scope") == scope:
                found.append((root, meta))
    if len(found) != 1:
        raise RuntimeError(f"expected one v4 {scope} cache, got {found}")
    root, meta = found[0]
    n_items, max_length = int(meta["n_items"]), int(meta["max_length"])
    if max_length != ITEM_MAX_LENGTH or meta.get("prefix") != "" or meta.get("category_in_text") is not False:
        raise RuntimeError(f"wrong v4 cache contract: {meta}")
    ids = np.memmap(os.path.join(root, "ids.bin"), np.int64, "r", shape=(n_items,))
    input_ids = np.memmap(
        os.path.join(root, "input_ids.bin"), np.int32, "r", shape=(n_items, max_length)
    )
    mask = np.memmap(
        os.path.join(root, "attention_mask.bin"), np.int8, "r", shape=(n_items, max_length)
    )
    lengths = np.empty(n_items, dtype=np.int16)
    for low in range(0, n_items, 500_000):
        high = min(low + 500_000, n_items)
        lengths[low:high] = np.asarray(mask[low:high]).sum(axis=1).astype(np.int16)
    del mask
    order = np.argsort(ids, kind="stable")
    sorted_ids = np.asarray(ids)[order]
    if np.any(sorted_ids[1:] == sorted_ids[:-1]):
        raise RuntimeError(f"duplicate ids in {scope} cache")
    candidates = np.flatnonzero(lengths < max_length)
    sample = candidates[::max(1, len(candidates) // 200)][:200]
    pad_ids = {int(input_ids[row, int(lengths[row])]) for row in sample}
    if len(pad_ids) != 1:
        raise RuntimeError(f"cannot derive one pad id for {scope}: {pad_ids}")
    return dict(
        root=root, input_ids=input_ids, lengths=lengths, order=order,
        sorted_ids=sorted_ids, pad_id=pad_ids.pop(),
    )


def lookup_rows(cache, item_ids):
    item_ids = np.asarray(item_ids, dtype=np.int64)
    positions = np.searchsorted(cache["sorted_ids"], item_ids)
    valid = positions < len(cache["sorted_ids"])
    valid[valid] &= cache["sorted_ids"][positions[valid]] == item_ids[valid]
    rows = np.full(len(item_ids), -1, dtype=np.int32)
    rows[valid] = cache["order"][positions[valid]].astype(np.int32)
    if not valid.all():
        raise RuntimeError(f"{int((~valid).sum()):,} pair endpoints are absent from a v4 cache")
    return rows


def build_frame(dataframe, scope, cache, category_to_id):
    categories = dataframe["category"].map(category_to_id)
    if categories.isna().any():
        raise RuntimeError("unknown category in pair data")
    return dict(
        scope=np.full(len(dataframe), scope, dtype=np.uint8),
        row1=lookup_rows(cache, dataframe["id1"].to_numpy()),
        row2=lookup_rows(cache, dataframe["id2"].to_numpy()),
        label=dataframe["label"].to_numpy(dtype=np.uint8),
        category=categories.to_numpy(dtype=np.uint8),
    )


def concatenate_frames(*frames):
    return {key: np.concatenate([frame[key] for frame in frames]) for key in frames[0]}


def find_resume(train_fingerprint, plan_fingerprint):
    required = {
        "encoder.pt", "head.pt", "optimizer.pt", "scheduler.pt", "ap_state.pt", "train_state.json"
    }
    candidates = []
    for root, _dirs, files in os.walk("/kaggle/input"):
        if required <= set(files):
            with open(os.path.join(root, "train_state.json"), encoding="utf-8") as handle:
                state = json.load(handle)
            contract = (
                state.get("experiment") == EXPERIMENT
                and state.get("train_fingerprint") == train_fingerprint
                and state.get("plan_fingerprint") == plan_fingerprint
                and int(state.get("world_size", -1)) == WORLD_SIZE
                and int(state.get("per_core_batch", -1)) == PER_CORE_TRAIN_BATCH
                and int(state.get("pair_max_length", -1)) == PAIR_MAX_LENGTH
            )
            if contract:
                candidates.append((root, state))
    if not candidates:
        return None
    candidates.sort(key=lambda item: int(item[1]["global_step"]), reverse=True)
    return candidates[0]


def build_model_cpu(source_dir, head_path, resume):
    from peft import LoraConfig, get_peft_model
    from transformers import T5EncoderModel

    t0 = time.perf_counter()
    encoder = T5EncoderModel.from_pretrained(
        source_dir, torch_dtype=torch.float32, attn_implementation="eager", local_files_only=True
    )
    d_model = encoder.config.d_model  # captured before peft wrapping -- see module docstring
    log(0, "MODEL", f"T5EncoderModel.from_pretrained done in {time.perf_counter()-t0:.1f}s; "
        f"{mem_info()}", force=True)

    lora_config = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES, bias="none",
    )
    encoder = get_peft_model(encoder, lora_config)
    n_trainable = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in encoder.parameters())
    log(0, "MODEL", f"LoRA applied: {n_trainable:,}/{n_total:,} trainable "
        f"({100 * n_trainable / n_total:.2f}%)", force=True)
    if n_trainable == 0:
        raise RuntimeError("LoRA adapter has zero trainable params -- target_modules matched nothing")

    classifier = FullSftHead(d_model)
    classifier.load_state_dict(torch.load(head_path, map_location="cpu", weights_only=True), strict=True)
    model = FridaRanker(encoder, classifier).float()
    if resume is not None:
        root, _state = resume
        model.encoder.load_state_dict(
            torch.load(os.path.join(root, "encoder.pt"), map_location="cpu", weights_only=True), strict=True
        )
        model.classifier.load_state_dict(
            torch.load(os.path.join(root, "head.pt"), map_location="cpu", weights_only=True), strict=True
        )
    # LoRA freezes the embedding layer along with the rest of the base model; without this,
    # gradient checkpointing's recomputed forward gets no requires_grad=True input anywhere in
    # the chain and silently drops every gradient before it reaches a single LoRA matrix (a
    # well-documented PEFT + gradient-checkpointing interaction, not specific to this codebase).
    model.encoder.enable_input_require_grads()
    model.encoder.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": True, "preserve_rng_state": False}
    )
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True  # head stays fully trainable; LoRA already set encoder grad flags
    return model


def load_shared_data():
    import pandas as pd

    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    log(0, "SETUP", f"loading v4 caches and exact pair splits pre-fork; {mem_info()}")
    t_phase = time.perf_counter()
    human_cache = load_cache("human")
    log(0, "SETUP", f"human item cache loaded in {time.perf_counter() - t_phase:.1f}s; {mem_info()}")
    t_phase = time.perf_counter()
    llm_cache = load_cache("llm")
    log(0, "SETUP", f"llm item cache loaded in {time.perf_counter() - t_phase:.1f}s; {mem_info()}")
    if human_cache["pad_id"] != llm_cache["pad_id"]:
        raise RuntimeError("human/llm v4 caches use different pad ids")
    caches = [human_cache, llm_cache]

    t_phase = time.perf_counter()
    human_path = find_file("matches_item_disjoint_70_30_seed42.parquet")
    human = pd.read_parquet(human_path)
    if len(human) != EXPECTED_HUMAN_ROWS:
        raise RuntimeError(f"human split has {len(human):,} rows")
    category_vocab = sorted(human["category"].unique().tolist())
    if len(category_vocab) != EXPECTED_CATEGORIES:
        raise RuntimeError(f"expected 20 categories, got {len(category_vocab)}")
    category_to_id = {name: index for index, name in enumerate(category_vocab)}
    human_train = human[human["split"].eq("train")].reset_index(drop=True)
    human_val = human[human["split"].eq("val")].reset_index(drop=True)
    if len(human_train) != EXPECTED_HUMAN_TRAIN or len(human_val) != EXPECTED_HUMAN_VAL:
        raise RuntimeError("human 70/30 item-disjoint row counts differ from the contract")
    # Must byte-match scripts/make_item_disjoint_split.py's own fingerprint exactly (newline
    # join, sort the FORMATTED STRINGS -- not the (id1,id2) tuples, which sorts numerically
    # and silently produces a different hash on identical data) -- this is what
    # EXPECTED_VAL_FINGERPRINT below was actually derived from.
    val_fingerprint_full = hashlib.sha256(
        "\n".join(
            sorted(f"{left}-{right}" for left, right in zip(human_val["id1"], human_val["id2"]))
        ).encode()
    ).hexdigest()
    val_fingerprint = val_fingerprint_full[:16]
    if val_fingerprint != EXPECTED_VAL_FINGERPRINT:
        raise RuntimeError(f"wrong human val fingerprint: {val_fingerprint} (full: {val_fingerprint_full})")
    log(0, "SETUP", f"human split loaded+verified in {time.perf_counter() - t_phase:.1f}s: "
        f"train={len(human_train):,} val={len(human_val):,} categories={len(category_vocab)}; "
        f"{mem_info()}")

    t_phase = time.perf_counter()
    llm_path = find_file("llm_train_val_split.parquet")
    llm = pd.read_parquet(
        llm_path,
        columns=["id1", "id2", "label", "target", "category"],
        filters=[("split", "==", "train"), ("in_original_1m", "==", True)],
    ).reset_index(drop=True)
    if len(llm) != EXPECTED_LLM_TRAIN:
        raise RuntimeError(f"exact original LLM-train subset has {len(llm):,} rows, not 1,000,000")
    llm_counts = llm.groupby("category", observed=True).size()
    if set(llm_counts.index) != set(category_vocab) or not np.all(llm_counts.to_numpy() == 50_000):
        raise RuntimeError(f"original 1M LLM subset is not 50k x 20 categories: {llm_counts.to_dict()}")
    if not np.array_equal(llm["label"].to_numpy(), (llm["target"].to_numpy() >= 0.5).astype(np.uint8)):
        raise RuntimeError("LLM binary label is inconsistent with target >= 0.5")
    log(0, "SETUP", f"llm split loaded+verified in {time.perf_counter() - t_phase:.1f}s: "
        f"llm_train={len(llm):,}; {mem_info()}")

    t_phase = time.perf_counter()
    human_train_frame = build_frame(human_train, 0, human_cache, category_to_id)
    val_frame = build_frame(human_val, 0, human_cache, category_to_id)
    llm_frame = build_frame(llm, 1, llm_cache, category_to_id)
    train_frame = concatenate_frames(human_train_frame, llm_frame)
    del human, human_train, human_val, llm, human_train_frame, llm_frame
    gc.collect()
    log(0, "SETUP", f"frames built+concatenated in {time.perf_counter() - t_phase:.1f}s: "
        f"train_frame={len(train_frame['label']):,} val_frame={len(val_frame['label']):,}; "
        f"{mem_info()}")

    t_phase = time.perf_counter()
    train_fingerprint = fingerprint_arrays(
        train_frame["scope"], train_frame["row1"], train_frame["row2"],
        train_frame["label"], train_frame["category"],
    )
    plan, steps_per_category = make_category_plan(
        train_frame["label"], train_frame["category"], WORLD_SIZE, PER_CORE_TRAIN_BATCH, SEED
    )
    plan_fingerprint = fingerprint_arrays(plan)
    resume = find_resume(train_fingerprint, plan_fingerprint)
    log(0, "SETUP", f"category plan built in {time.perf_counter() - t_phase:.1f}s: "
        f"steps={len(plan):,} steps_per_category={steps_per_category} "
        f"resume={'none' if resume is None else resume[0]}; {mem_info()}")

    t_phase = time.perf_counter()
    source_dir, head_path = find_source_model()
    model = build_model_cpu(source_dir, head_path, resume)
    log(0, "SETUP", f"source model + LoRA built on CPU in {time.perf_counter() - t_phase:.1f}s; "
        f"{mem_info()}")

    t_phase = time.perf_counter()
    resume_payload = None
    if resume is not None:
        root, state = resume
        resume_payload = dict(
            root=root, state=state,
            optimizer=torch.load(os.path.join(root, "optimizer.pt"), map_location="cpu", weights_only=True),
            scheduler=torch.load(os.path.join(root, "scheduler.pt"), map_location="cpu", weights_only=True),
            ap=torch.load(os.path.join(root, "ap_state.pt"), map_location="cpu", weights_only=True),
        )
        log(0, "SETUP", f"resume payload loaded in {time.perf_counter() - t_phase:.1f}s "
            f"from {root}; {mem_info()}")
    global _WRAPPED_MODEL
    _WRAPPED_MODEL = xmp.MpModelWrapper(model)
    log(0, "SETUP", f"model wrapped for fork (xmp.MpModelWrapper); {mem_info()}")
    _SHARED.update(dict(
        caches=caches, pad_id=human_cache["pad_id"], train_frame=train_frame,
        val_frame=val_frame, plan=plan, category_vocab=category_vocab,
        train_fingerprint=train_fingerprint, plan_fingerprint=plan_fingerprint,
        val_fingerprint=val_fingerprint, steps_per_category=steps_per_category,
        resume=resume_payload,
    ))
    preflight = dict(
        experiment=EXPERIMENT, status="preflight_complete", source="maxamsecret/frida-format-adapter-v4-188",
        human_train_rows=EXPECTED_HUMAN_TRAIN, human_val_rows=EXPECTED_HUMAN_VAL,
        llm_train_rows=EXPECTED_LLM_TRAIN, categories=EXPECTED_CATEGORIES,
        train_fingerprint=train_fingerprint, plan_fingerprint=plan_fingerprint,
        val_fingerprint=val_fingerprint, total_steps=len(plan), steps_per_category=steps_per_category,
        category_batch_contract="all 8 ranks use the same one category; each core is positive-first 8:8",
        dtypes=dict(model_compute="bfloat16", master_parameters="float32",
                    soap_moments="bfloat16 (encoder, unpreconditioned) / float32 (head, real Shampoo basis)",
                    soap_preconditioner_scope=f"max_precond_dim={SOAP_MAX_PRECOND_DIM} -> head only",
                    sigmoid="float32", pairwise="float32", u_all_u_pos="float32"),
        resume_from=None if resume is None else resume[0],
    )
    atomic_json(SUMMARY_PATH, preflight)
    log(0, "SETUP", f"train={len(train_frame['label']):,}, val={len(val_frame['label']):,}, "
        f"steps={len(plan):,}, plan={plan_fingerprint}, resume={preflight['resume_from']}; "
        f"{mem_info()} -- ready for fork")


def make_optimizer(model):
    # Separate LoRA-adapter parameters from head parameters so each family
    # can run its own LR: LoRA needs LORA_LR (cold-start, rank-16 bottleneck)
    # while the head needs the much lower PEAK_LR (already trained). See
    # LORA_LR's own comment for the full incident this fixes.
    lora_decay, lora_no_decay, head_decay, head_no_decay = [], [], [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue  # frozen LoRA base weights -- SOAP.step() would just skip these (no .grad)
                      # anyway, but excluding them keeps its per-parameter Python loop to the
                      # ~13M trainable params instead of iterating ~823M dead entries every step.
        is_lora = "lora_" in name
        bucket_decay = (lora_decay if is_lora else head_decay)
        bucket_no_decay = (lora_no_decay if is_lora else head_no_decay)
        (bucket_decay if parameter.ndim > 1 else bucket_no_decay).append(parameter)
    n_lora = sum(p.numel() for p in lora_decay + lora_no_decay)
    n_head = sum(p.numel() for p in head_decay + head_no_decay)
    if n_lora == 0:
        raise RuntimeError("make_optimizer found zero LoRA parameters -- 'lora_' name match failed")
    log(0, "OPTIMIZER", f"param groups: lora={n_lora:,} params @ lr_scale=1.0 (peak={LORA_LR:.1e}), "
        f"head={n_head:,} params @ lr_scale={PEAK_LR / LORA_LR:.4f} (peak={PEAK_LR:.1e})", force=True)
    return SOAP([
        {"params": lora_decay, "weight_decay": WEIGHT_DECAY, "lr_scale": 1.0, "name": "lora_decay"},
        {"params": lora_no_decay, "weight_decay": 0.0, "lr_scale": 1.0, "name": "lora_no_decay"},
        {"params": head_decay, "weight_decay": WEIGHT_DECAY, "lr_scale": PEAK_LR / LORA_LR, "name": "head_decay"},
        {"params": head_no_decay, "weight_decay": 0.0, "lr_scale": PEAK_LR / LORA_LR, "name": "head_no_decay"},
    ], lr=LORA_LR)


def scheduled_lr(step, total_steps, peak_lr=LORA_LR):
    warmup_steps = max(1, round(total_steps * WARMUP_RATIO))
    if step < warmup_steps:
        return peak_lr * (0.1 + 0.9 * (step + 1) / warmup_steps)
    progress = min((step - warmup_steps) / max(1, total_steps - warmup_steps), 1.0)
    ratio = MIN_LR_RATIO + (1.0 - MIN_LR_RATIO) * 0.5 * (1.0 + math.cos(math.pi * progress))
    return peak_lr * ratio


def make_train_loader(rank, start_step, one_step=False):
    plan = _SHARED["plan"]
    stop = min(start_step + 1, len(plan)) if one_step else len(plan)
    rows = plan[start_step:stop, rank * PER_CORE_TRAIN_BATCH:(rank + 1) * PER_CORE_TRAIN_BATCH].reshape(-1)
    return DataLoader(
        IndexDataset(rows), batch_size=PER_CORE_TRAIN_BATCH, shuffle=False, drop_last=True,
        num_workers=2, collate_fn=make_collate(
            _SHARED["train_frame"], _SHARED["caches"], _SHARED["pad_id"], True
        ),
    )


def make_eval_loader(rank):
    n_rows = len(_SHARED["val_frame"]["label"])
    per_rank = math.ceil(n_rows / (WORLD_SIZE * PER_CORE_EVAL_BATCH)) * PER_CORE_EVAL_BATCH
    absolute = np.arange(rank * per_rank, (rank + 1) * per_rank, dtype=np.int64)
    valid = absolute < n_rows
    rows = (absolute % n_rows).astype(np.int32)
    return DataLoader(
        IndexDataset(rows, valid), batch_size=PER_CORE_EVAL_BATCH, shuffle=False, drop_last=True,
        num_workers=2, collate_fn=make_collate(
            _SHARED["val_frame"], _SHARED["caches"], _SHARED["pad_id"], False
        ),
    )


def anchor_positions(device):
    positions = [
        rank * PER_CORE_TRAIN_BATCH + offset
        for rank in range(WORLD_SIZE) for offset in range(PER_CORE_POSITIVES)
    ]
    return torch.tensor(positions, dtype=torch.long, device=device)


class _DifferentiableAllGather(torch.autograd.Function):
    """All-gather that backprops -- xm.all_gather itself does not.

    Forward matches xm.all_gather exactly (each replica's dim-0 slice is
    concatenated in ordinal order); backward hands each replica back only
    its own slice of the incoming gradient, which is the correct adjoint of
    a concatenation. torch_xla ships this exact pattern as
    torch_xla.core.functions.all_gather, but that helper's forward calls the
    now-removed xm.get_ordinal()/xm.xrt_world_size() (deprecated-and-erroring
    since torch_xla 2.7); this project's own full_sft_continue kernel already
    confirms xr.global_ordinal()/xr.world_size() are the live replacements on
    the current Kaggle TPU image, so this reimplements the same autograd
    wrapper on top of those instead of importing the broken helper.
    """

    @staticmethod
    def forward(ctx, value, dim):
        ctx.dim = dim
        ctx.ordinal = xr.global_ordinal()
        ctx.world_size = xr.world_size()
        return xm.all_gather(value, dim=dim)

    @staticmethod
    def backward(ctx, grad_output):
        slice_size = grad_output.size(ctx.dim) // ctx.world_size
        gradient = torch.narrow(
            grad_output.clone(), ctx.dim, ctx.ordinal * slice_size, slice_size
        )
        return gradient, None


def differentiable_all_gather(value, dim=0):
    return _DifferentiableAllGather.apply(value, dim)


def ap_forward(model, ap_loss, batch, anchors):
    input_ids, attention_mask, labels, categories, rows, _valid = batch
    local_scores = torch.sigmoid(model(input_ids, attention_mask).float()).float()
    # Only local_scores carries a gradient back to this replica's own model
    # weights -- it MUST use the differentiable gather. labels/categories/
    # rows feed no backward pass, so the plain (cheaper) xm.all_gather is
    # correct for them, same as the eval-side gathers below.
    global_scores = differentiable_all_gather(local_scores, dim=0)
    global_labels = xm.all_gather(labels.float(), dim=0)
    global_categories = xm.all_gather(categories.long(), dim=0)
    global_rows = xm.all_gather(rows.long(), dim=0)
    loss = ap_loss(global_scores, global_labels, global_rows, anchors)
    return loss, global_scores, global_labels, global_categories, global_rows


def evaluate(model, device, rank):
    model.eval()
    predictions, labels, categories, indices, valid_flags = [], [], [], [], []
    loader = pl_xla.MpDeviceLoader(make_eval_loader(rank), device)
    t_start = time.perf_counter()
    n_batches = 0
    # ~428 batches/rank expected (109,648 val rows / 8 ranks / batch 32) -- log every 50 so a
    # stall inside eval (previously silent until the very end) shows up within a couple minutes.
    with torch.no_grad():
        for input_ids, attention_mask, local_labels, local_categories, local_indices, local_valid in loader:
            local_predictions = torch.sigmoid(model(input_ids, attention_mask).float()).float()
            gathered = [
                xm.all_gather(value, dim=0).cpu().numpy()
                for value in (local_predictions, local_labels.float(), local_categories.long(),
                              local_indices.long(), local_valid.long())
            ]
            predictions.append(gathered[0])
            labels.append(gathered[1])
            categories.append(gathered[2])
            indices.append(gathered[3])
            valid_flags.append(gathered[4])
            n_batches += 1
            if n_batches % 50 == 0:
                log(rank, "EVAL_PROGRESS", f"batch={n_batches} elapsed={time.perf_counter() - t_start:.1f}s "
                    f"{mem_info(device)}", force=True)
    log(rank, "EVAL_PROGRESS", f"loop done: batch={n_batches} elapsed={time.perf_counter() - t_start:.1f}s",
        force=True)
    predictions = np.concatenate(predictions)
    labels = np.concatenate(labels).astype(np.uint8)
    categories = np.concatenate(categories).astype(np.int16)
    indices = np.concatenate(indices).astype(np.int64)
    valid = np.concatenate(valid_flags).astype(bool)
    predictions, labels, categories, indices = (
        value[valid] for value in (predictions, labels, categories, indices)
    )
    order = np.argsort(indices)
    predictions, labels, categories, indices = (
        value[order] for value in (predictions, labels, categories, indices)
    )
    if len(indices) != EXPECTED_HUMAN_VAL or not np.array_equal(indices, np.arange(len(indices))):
        raise RuntimeError("distributed human validation did not recover each row exactly once")
    per_category = {}
    for category_id, category_name in enumerate(_SHARED["category_vocab"]):
        selected = categories == category_id
        per_category[category_name] = float(average_precision_score(labels[selected], predictions[selected]))
    macro = float(np.mean(list(per_category.values())))
    model.train()
    log(rank, "VAL", f"Mean PR-AUC={macro:.8f}", force=True)
    return macro, per_category


def to_cpu_state(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: to_cpu_state(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_cpu_state(item) for item in value]
    if isinstance(value, tuple):
        return tuple(to_cpu_state(item) for item in value)
    return value


def atomic_json(path, payload):
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    os.replace(temporary, path)


def save_checkpoint(model, optimizer, ap_loss, global_step, metric, per_category,
                    best_metric, best_checkpoint, label, rank, device):
    xm.rendezvous(f"before_save_{global_step}_{label}")
    destination = None
    save_error = None
    if rank == 0:
        try:
            destination = os.path.join(OUTPUT_ROOT, f"{label}_step{global_step:06d}")
            if os.path.exists(destination):
                raise RuntimeError(f"refusing to overwrite checkpoint {destination}")
            free = shutil.disk_usage(OUTPUT_ROOT).free
            if free < 6_000_000_000:
                raise RuntimeError(f"only {free / 1e9:.1f}GB free before checkpoint")
            os.makedirs(destination)
            torch.save(
                {key: value.detach().cpu() for key, value in model.encoder.state_dict().items()},
                os.path.join(destination, "encoder.pt"),
            )
            torch.save(
                {key: value.detach().cpu() for key, value in model.classifier.state_dict().items()},
                os.path.join(destination, "head.pt"),
            )
            torch.save(to_cpu_state(optimizer.state_dict()), os.path.join(destination, "optimizer.pt"))
            torch.save(
                {"global_step": global_step, "total_steps": len(_SHARED["plan"]),
                 "peak_lr_lora": LORA_LR, "peak_lr_head": PEAK_LR,
                 "warmup_ratio": WARMUP_RATIO, "min_lr_ratio": MIN_LR_RATIO},
                os.path.join(destination, "scheduler.pt"),
            )
            torch.save(to_cpu_state(ap_loss.state_dict()), os.path.join(destination, "ap_state.pt"))
            state = dict(
                experiment=EXPERIMENT, global_step=global_step, total_steps=len(_SHARED["plan"]),
                epoch_complete=global_step == len(_SHARED["plan"]),
                val_mean_pr_auc=metric, val_per_category=per_category,
                best_val_mean_pr_auc=best_metric, best_checkpoint=best_checkpoint,
                train_fingerprint=_SHARED["train_fingerprint"],
                plan_fingerprint=_SHARED["plan_fingerprint"],
                val_fingerprint=_SHARED["val_fingerprint"], world_size=WORLD_SIZE,
                per_core_batch=PER_CORE_TRAIN_BATCH, pair_max_length=PAIR_MAX_LENGTH,
                lr_lora=LORA_LR, lr_head=PEAK_LR,
                optimizer=f"SOAP (max_precond_dim={SOAP_MAX_PRECOND_DIM}, "
                          f"freq={SOAP_PRECONDITION_FREQUENCY}; FP32 master)",
                ap_state_dtype="float32", metric_is_quiet=metric is None,
            )
            atomic_json(os.path.join(destination, "train_state.json"), state)
            log(rank, "SAVE", f"{destination}; metric={metric}")
            del state
            gc.collect()
        except Exception as error:
            # REAL CRASH class (see smoke()'s own fix, same day, same
            # project): this used to raise straight out of the `if rank ==
            # 0:` block, which would skip the rendezvous below entirely --
            # ranks 1-7 (never entered this block, so never raise) would
            # reach `after_save` and block on it FOREVER since rank 0 died
            # without ever calling it. Checkpointing runs repeatedly
            # throughout a long training run (every LOG_EVERY_STEPS when
            # due), so this was actually a MORE likely deadlock trigger
            # than the one-shot smoke() check. FIX: catch here, still reach
            # the rendezvous below on every rank (so no deadlock), then
            # re-raise on every rank together once all have arrived.
            save_error = f"{type(error).__name__}: {error}"
            log(rank, "SAVE_ERROR", save_error, force=True)
    xm.rendezvous(f"after_save_{global_step}_{label}")
    if collective_flag(rank, save_error is not None, device):
        raise RuntimeError(f"checkpoint save failed on rank0: {save_error}"
                           if save_error is not None else
                           "checkpoint save failed on rank0 -- see rank0's own log for detail")
    return destination


def write_summary(rank, **updates):
    if rank != 0:
        return
    with open(SUMMARY_PATH, encoding="utf-8") as handle:
        summary = json.load(handle)
    summary.update(updates)
    atomic_json(SUMMARY_PATH, summary)


def collective_flag(rank, value, device):
    flag = torch.tensor([int(value) if rank == 0 else 0], dtype=torch.int32, device=device)
    flag = xm.all_reduce(xm.REDUCE_SUM, flag)
    xm.mark_step()
    return bool(flag.cpu().item())


def smoke(model, optimizer, ap_loss, device, rank, start_step, anchors):
    if start_step >= len(_SHARED["plan"]):
        return
    t_phase = time.perf_counter()
    loader = pl_xla.MpDeviceLoader(make_train_loader(rank, start_step, one_step=True), device)
    batch = next(iter(loader))
    log(rank, "SMOKE", f"batch fetched in {time.perf_counter() - t_phase:.1f}s; {mem_info(device)}", force=True)
    previous_lrs = [group["lr"] for group in optimizer.param_groups]
    previous_freqs = [group["precondition_frequency"] for group in optimizer.param_groups]
    for group in optimizer.param_groups:
        group["lr"] = 0.0
        group["precondition_frequency"] = 1  # forces the refresh CODE PATH on the next real
                                              # step; with SOAP_MAX_PRECOND_DIM=8 for this LoRA
                                              # run every GG entry is empty, so _refresh_basis
                                              # is still exercised but no real qr executes --
                                              # see the SOAP_MAX_PRECOND_DIM comment above.
    optimizer.zero_grad(set_to_none=True)
    t_phase = time.perf_counter()
    loss, scores, labels, categories, _rows = ap_forward(model, ap_loss, batch, anchors)
    if scores.dtype != torch.float32 or ap_loss.u_all.dtype != torch.float32:
        raise RuntimeError("FP32 AP contract failed in real-shape smoke")
    loss.backward()
    log(rank, "SMOKE", f"step#1 forward+backward traced in {time.perf_counter() - t_phase:.1f}s "
        "(lazy -- torch_xla hasn't compiled/executed anything yet)", force=True)
    t_phase = time.perf_counter()
    xm.optimizer_step(optimizer, barrier=False)
    xm.mark_step()  # this is where first-ever compilation + execution actually happens
    log(rank, "SMOKE", f"step#1 (init-only, no weight change expected) compiled+executed "
        f"in {time.perf_counter() - t_phase:.1f}s; {mem_info(device)}", force=True)
    if rank == 0:
        labels_cpu = labels.cpu().numpy()
        categories_cpu = categories.cpu().numpy()
        anchor_cpu = anchors.cpu().numpy()
        smoke_contract_ok = (len(np.unique(categories_cpu)) == 1
                            and bool(np.all(labels_cpu[anchor_cpu] == 1)))
    else:
        smoke_contract_ok = True
    # REAL CRASH (reference kernel rayson000/frida-ap-soap-full-sft, and
    # presumably this LoRA sibling too -- last log line at ~111.0s, then
    # ~7842.1s of dead silence with no traceback before Kaggle's own
    # watchdog killed the session): the two contract checks above used to
    # `raise RuntimeError(...)` directly inside `if rank == 0:`, with NO
    # collective between that raise and the next collective op
    # (evaluate()'s own xm.all_gather, called right after smoke() returns).
    # If either check ever fails, rank 0 alone raises and exits while the
    # other 7 replicas have no idea anything went wrong -- they proceed
    # straight into evaluate()'s all_gather and block there FOREVER, since
    # rank 0 (now dead) never arrives to complete the collective. That is
    # exactly a silent, traceback-free multi-hour stall ending only when an
    # external watchdog (Kaggle's own idle/session-length killer) SIGTERMs
    # the whole job -- this project's own frida-rank-sft-a kernel hit the
    # identical failure class from a different root cause (per-replica RNG
    # divergence) the same day; see that kernel's own REAL CRASH #2 comment
    # for the general pattern: ANY rank-0-only `raise` upstream of a later
    # collective is a deadlock waiting to happen, not a normal fail-fast.
    # FIX: broadcast rank 0's verdict to every replica via the SAME
    # collective_flag() helper already used for quiet-save/cutoff
    # decisions below, so every replica either continues together or
    # raises together -- never diverges.
    if not collective_flag(rank, not smoke_contract_ok, device):
        pass
    elif rank == 0:
        raise RuntimeError("global TPU batch contains more than one category, or fixed "
                           "AP anchor positions are not positive (see rank0 smoke log)")
    else:
        raise RuntimeError("rank0 smoke contract check failed -- see rank0's own log for detail")

    # SOAP's own first-ever step() call per parameter only initializes GG/Q (see SOAP.step);
    # a SECOND real step (still lr=0, so no visible weight drift) is needed to exercise its
    # actual update arithmetic before the expensive real run leans on it for the first time.
    optimizer.zero_grad(set_to_none=True)
    t_phase = time.perf_counter()
    loss2, _scores2, _labels2, _categories2, _rows2 = ap_forward(model, ap_loss, batch, anchors)
    loss2.backward()
    xm.optimizer_step(optimizer, barrier=False)
    xm.mark_step()
    log(rank, "SMOKE", f"step#2 (real update path) compiled+executed in "
        f"{time.perf_counter() - t_phase:.1f}s; {mem_info(device)}", force=True)

    optimizer.zero_grad(set_to_none=True)
    optimizer.reset_()
    ap_loss.reset_()
    for group, lr, freq in zip(optimizer.param_groups, previous_lrs, previous_freqs):
        group["lr"] = lr
        group["precondition_frequency"] = freq
    xm.mark_step()
    xm.rendezvous("real_shape_smoke_complete")
    log(rank, "SMOKE", f"real batch/AP/SOAP graph OK; loss={float(loss.cpu()):.6g}", force=True)


def mp_fn(_index):
    rank = xr.global_ordinal()
    world_size = xr.world_size()
    if world_size != WORLD_SIZE:
        raise RuntimeError(f"requires TPU v5e-8 world_size=8, got {world_size}")
    log(rank, "RANK_INIT", f"world_size={world_size}", force=True)
    t_phase = time.perf_counter()
    device = torch_xla.device()
    log(rank, "DEVICE", f"torch_xla.device() OK in {time.perf_counter() - t_phase:.1f}s; "
        f"{mem_info()}", force=True)
    torch.manual_seed(SEED + rank)
    t_phase = time.perf_counter()
    model = _WRAPPED_MODEL.to(device)
    log(rank, "MODEL_TO_DEVICE", f"{time.perf_counter() - t_phase:.1f}s; {mem_info(device)}", force=True)
    optimizer = make_optimizer(model)
    ap_loss = StreamingAPLoss(len(_SHARED["train_frame"]["label"])).to(device)
    anchors = anchor_positions(device)
    log(rank, "OPTIMIZER_READY", mem_info(device), force=True)

    resume = _SHARED["resume"]
    start_step = 0
    best_metric = -math.inf
    best_checkpoint = "source_init"
    if resume is not None:
        optimizer.load_state_dict(resume["optimizer"])
        ap_loss.load_state_dict(resume["ap"], strict=True)
        start_step = int(resume["state"]["global_step"])
        if int(resume["scheduler"]["global_step"]) != start_step:
            raise RuntimeError("scheduler/global checkpoint steps disagree")
        best_metric = float(resume["state"].get("best_val_mean_pr_auc", -math.inf))
        best_checkpoint = resume["state"].get("best_checkpoint", resume["root"])
        xm.mark_step()
        log(rank, "RESUME", f"step={start_step:,} from {resume['root']}", force=True)
    else:
        t_phase = time.perf_counter()
        smoke(model, optimizer, ap_loss, device, rank, start_step, anchors)
        log(rank, "SMOKE_DONE", f"total {time.perf_counter() - t_phase:.1f}s; {mem_info(device)}", force=True)

    t_phase = time.perf_counter()
    baseline_metric, baseline_per_category = None, None
    try:
        baseline_metric, baseline_per_category = evaluate(model, device, rank)
    except Exception as error:
        log(rank, "BASELINE_VAL_FAIL", f"{type(error).__name__}: {error}; "
            "training proceeds without a baseline number", force=True)
    log(rank, "BASELINE_EVAL_DONE", f"total {time.perf_counter() - t_phase:.1f}s; {mem_info(device)}", force=True)
    if baseline_metric is not None and baseline_metric > best_metric:
        best_metric, best_checkpoint = baseline_metric, (
            "source_init" if start_step == 0 else resume["root"]
        )
    write_summary(
        rank, status="training", baseline_step=start_step,
        baseline_val_mean_pr_auc=baseline_metric,
        baseline_val_per_category=baseline_per_category,
        best_val_mean_pr_auc=best_metric, best_checkpoint=best_checkpoint,
    )
    if start_step >= len(_SHARED["plan"]):
        write_summary(rank, status="already_complete")
        return

    loader = pl_xla.MpDeviceLoader(make_train_loader(rank, start_step), device)
    model.train()
    next_quiet_at = QUIET_CHECKPOINT_INTERVAL_SECONDS
    # See FULL_EVAL_INTERVAL_SECONDS's own comment: this is the missing
    # periodic-validation-with-best-checkpoint-selection mechanism (v3's
    # run_summary.json showed best_checkpoint stuck at "source_init" for an
    # entire 19,340-step run because nothing but baseline/final was ever
    # measured). Offset from resume time (not absolute wall-clock 0) so a
    # resumed run gets a full interval before its own first periodic eval,
    # same convention as next_quiet_at.
    next_full_eval_at = FULL_EVAL_INTERVAL_SECONDS
    previous_quiet_dir = None
    final_step = start_step
    first_probe = model.classifier.out.weight.detach().cpu().clone()
    lora_probe_name, lora_probe_param = next(
        (name, parameter) for name, parameter in model.encoder.named_parameters()
        if "lora_" in name and parameter.requires_grad
    )
    first_lora_probe = lora_probe_param.detach().cpu().clone()
    lora_probe_confirmed = False
    # See LORA_PROBE_GRACE_STEPS's own comment: A's gradient is PROVABLY
    # exactly zero until B receives its own first real update (B is
    # zero-init, dL/dA = scale*B^T@... = 0 while B=0), which itself only
    # lands at start_step+2 (SOAP's first real step for every parameter,
    # after step start_step+1's init-only step). So the window opens at
    # start_step+2 (earliest step worth even checking) and must close by
    # start_step+1+LORA_PROBE_GRACE_STEPS.
    lora_probe_window_end = start_step + 1 + LORA_PROBE_GRACE_STEPS
    losses = []
    for offset, batch in enumerate(loader):
        global_step = start_step + offset
        lora_lr = scheduled_lr(global_step, len(_SHARED["plan"]))
        for group in optimizer.param_groups:
            # Each group's own lr_scale (set in make_optimizer: 1.0 for LoRA,
            # PEAK_LR/LORA_LR for the head) rides the SAME warmup/cosine shape --
            # only the peak differs per family. See LORA_LR's own comment.
            group["lr"] = lora_lr * group["lr_scale"]
        optimizer.zero_grad(set_to_none=True)
        loss, _scores, _labels, _categories, _rows = ap_forward(
            model, ap_loss, batch, anchors
        )
        loss.backward()
        xm.optimizer_step(optimizer, barrier=False)
        xm.mark_step()
        final_step = global_step + 1
        losses.append(float(loss.detach().cpu()))

        if final_step == start_step + 2:
            # SOAP's own first-ever step() call per parameter only initializes GG/Q and
            # applies no weight update (it must not precondition with the gradient that
            # built the basis) -- so the earliest step that can show real drift is the
            # second one, not the first. head.out has no zero-init gating (unlike LoRA's
            # A matrix below), so it is correctly checked at this exact step.
            probe_after = model.classifier.out.weight.detach().cpu()
            changed = float((probe_after != first_probe).float().mean())
            max_delta = float((probe_after - first_probe).abs().max())
            if changed == 0.0 or max_delta == 0.0:
                raise RuntimeError("real SOAP step changed no FP32 master weights")
            log(rank, "DELTA", f"head.out changed={changed:.1%} max|delta|={max_delta:.3e}",
                force=True)

        # Head training alone doesn't prove LoRA is training -- PEFT + gradient
        # checkpointing can silently drop every gradient before it reaches the encoder
        # (see build_model_cpu's enable_input_require_grads note); catch that here
        # instead of after hours of a head-only "LoRA" run. Checked over a GRACE WINDOW,
        # not one hard step -- see LORA_PROBE_GRACE_STEPS's own comment for the exact
        # math proving LoRA-A's first possible nonzero drift is one step later than
        # head.out's (A's gradient is provably exactly zero until B, zero-init, receives
        # its own first real update). Stops probing as soon as real drift is confirmed;
        # only raises if the ENTIRE window closes with zero drift throughout.
        if not lora_probe_confirmed and start_step + 2 <= final_step <= lora_probe_window_end:
            lora_after = lora_probe_param.detach().cpu()
            lora_changed = float((lora_after != first_lora_probe).float().mean())
            lora_max_delta = float((lora_after - first_lora_probe).abs().max())
            if lora_changed > 0.0 and lora_max_delta > 0.0:
                lora_probe_confirmed = True
                log(rank, "DELTA", f"lora[{lora_probe_name}] changed={lora_changed:.1%} "
                    f"max|delta|={lora_max_delta:.3e} confirmed at step={final_step:,} "
                    f"({final_step - start_step}/{LORA_PROBE_GRACE_STEPS} steps into grace "
                    "window)", force=True)
            else:
                log(rank, "DELTA_PROBE", f"lora[{lora_probe_name}] still unchanged at "
                    f"step={final_step:,} ({final_step - start_step}/{LORA_PROBE_GRACE_STEPS} "
                    "steps into grace window)", force=True)
                if final_step == lora_probe_window_end:
                    raise RuntimeError(
                        f"LoRA parameter {lora_probe_name!r} never changed across the entire "
                        f"{LORA_PROBE_GRACE_STEPS}-step grace window (steps {start_step + 2}.."
                        f"{lora_probe_window_end}) -- LoRA adapter is silently frozen"
                    )

        if final_step % LOG_EVERY_STEPS == 0:
            mean_loss = float(np.mean(losses[-LOG_EVERY_STEPS:]))
            head_lr = lora_lr * (PEAK_LR / LORA_LR)
            log(rank, "TRAIN", f"step={final_step:,}/{len(_SHARED['plan']):,} "
                f"lr_lora={lora_lr:.3e} lr_head={head_lr:.3e} AP-surrogate={mean_loss:.6g} "
                f"{mem_info(device)}", force=True)
            quiet_due = time.perf_counter() - _START >= next_quiet_at
            if collective_flag(rank, quiet_due, device):
                new_quiet_dir = save_checkpoint(
                    model, optimizer, ap_loss, final_step, None, None,
                    best_metric, best_checkpoint, "quiet", rank, device,
                )
                if rank == 0 and previous_quiet_dir is not None:
                    # Bounds disk to ~1 checkpoint's footprint across a long run's many
                    # repeats; delete only AFTER the new save lands so a crash mid-write
                    # leaves the last known-good checkpoint intact rather than nothing.
                    shutil.rmtree(previous_quiet_dir, ignore_errors=True)
                previous_quiet_dir = new_quiet_dir
                next_quiet_at += QUIET_CHECKPOINT_INTERVAL_SECONDS
                write_summary(rank, quiet_checkpoint_step=final_step)
            # REAL BUG fix (see FULL_EVAL_INTERVAL_SECONDS's own comment): the
            # ONLY two evaluate() calls in the entire previous version of this
            # function were baseline (pre-training) and final (post-training)
            # -- v3's own run_summary.json proved that gap let a full
            # 19,340-step run complete with best_checkpoint stuck at
            # "source_init", because no intermediate point was ever measured
            # or compared. This mirrors rank_sft_a's own do_full_checkpoint()
            # (evaluate + `improved = m > best_val` once per epoch) adapted to
            # this file's flat step-plan (no epoch boundaries to hook into).
            full_eval_due = time.perf_counter() - _START >= next_full_eval_at
            if collective_flag(rank, full_eval_due, device):
                periodic_metric, periodic_per_category = None, None
                try:
                    periodic_metric, periodic_per_category = evaluate(model, device, rank)
                except Exception as error:
                    log(rank, "PERIODIC_VAL_FAIL", f"{type(error).__name__}: {error}; "
                        "training continues, this point is simply unmeasured", force=True)
                model.train()
                if periodic_metric is not None:
                    improved = periodic_metric > best_metric
                    log(rank, "METRICS", f"periodic: step={final_step:,} "
                        f"val_mean_pr_auc={periodic_metric:.4f} best={max(best_metric, periodic_metric):.4f} "
                        f"improved={improved}", force=True)
                    if improved:
                        best_metric = periodic_metric
                        best_checkpoint = f"validated_step{final_step:06d}"
                        # A genuine improvement is exactly the point future resumes and
                        # final submission-building should be able to find on disk --
                        # unlike the quiet checkpoint above (resume-only, no metric, no
                        # best-tracking), this one is a real, metric-backed validated
                        # save, using the SAME save_checkpoint() the final save uses.
                        save_checkpoint(
                            model, optimizer, ap_loss, final_step, periodic_metric, periodic_per_category,
                            best_metric, best_checkpoint, best_checkpoint, rank, device,
                        )
                    write_summary(rank, best_val_mean_pr_auc=best_metric, best_checkpoint=best_checkpoint)
                next_full_eval_at += FULL_EVAL_INTERVAL_SECONDS
            stop = time.perf_counter() - _START >= STOP_TRAINING_AT_SECONDS
            if collective_flag(rank, stop, device):
                log(rank, "CUTOFF", "wall-clock reserve reached; validating and saving exact resume", force=True)
                break

    final_metric = None
    final_per_category = None
    try:
        final_metric, final_per_category = evaluate(model, device, rank)
        if final_metric > best_metric:
            best_metric = final_metric
            best_checkpoint = f"validated_step{final_step:06d}"
    except Exception as error:
        log(rank, "VAL_FAIL", f"{type(error).__name__}: {error}; final checkpoint is quiet", force=True)
    label = "validated" if final_metric is not None else "quiet_final"
    save_checkpoint(
        model, optimizer, ap_loss, final_step, final_metric, final_per_category,
        best_metric, best_checkpoint, label, rank, device,
    )
    if rank == 0 and previous_quiet_dir is not None:
        # Superseded by the checkpoint just above; drop it so only one copy lingers.
        shutil.rmtree(previous_quiet_dir, ignore_errors=True)
    write_summary(
        rank, status="complete" if final_step == len(_SHARED["plan"]) else "cutoff_resume_ready",
        final_step=final_step, total_steps=len(_SHARED["plan"]),
        final_val_mean_pr_auc=final_metric, final_val_per_category=final_per_category,
        best_val_mean_pr_auc=best_metric, best_checkpoint=best_checkpoint,
    )


def self_test():
    labels = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=np.uint8)
    categories = np.repeat(np.arange(2), 6)
    plan, steps_per_category = make_category_plan(labels, categories, 2, 4, 42)
    # 3 positives/negatives per category need ceil(3/global_pos=4)=1 step (with wraparound
    # repeats filling the last slot) to cover the category once -- verified against the
    # function's own actual output, not assumed.
    assert steps_per_category == 1
    assert np.unique(plan).size == len(labels)
    for batch in plan:
        for rank in range(2):
            local = batch[rank * 4:(rank + 1) * 4]
            assert len(np.unique(categories[local])) == 1
            assert labels[local].tolist() == [1, 1, 0, 0]
        assert len(np.unique(categories[batch])) == 1
    scores = torch.linspace(0.1, 0.9, 8, dtype=torch.float32, requires_grad=True)
    gathered_labels = torch.tensor([1, 1, 0, 0, 1, 1, 0, 0], dtype=torch.float32)
    rows = torch.arange(8)
    anchors = torch.tensor([0, 1, 4, 5])
    ap = StreamingAPLoss(8)
    loss = ap(scores, gathered_labels, rows, anchors)
    loss.backward()
    assert loss.dtype == torch.float32 and torch.isfinite(loss)
    assert scores.grad is not None and torch.isfinite(scores.grad).all()
    assert ap.u_all.dtype == ap.u_pos.dtype == torch.float32
    print("self-test OK: category plan + FP32 stochastic AP")

    _test_soap_unpreconditioned_matches_adam()
    _test_soap_preconditioned_and_resume()
    _test_soap_1d_never_preconditioned()


def _manual_decoupled_adam_step(param, grad, exp_avg, exp_avg_sq, step, betas, eps, lr, weight_decay,
                                 clip=SOAP_CLIP_VALUE):
    beta1, beta2 = betas
    grad = grad.clamp(-clip, clip)
    exp_avg = beta1 * exp_avg + (1 - beta1) * grad
    exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * grad * grad
    bias1 = 1.0 - beta1 ** step
    bias2 = 1.0 - beta2 ** step
    step_size = lr * (bias2 ** 0.5) / bias1
    denom = exp_avg_sq.sqrt() + eps
    param = param - step_size * (exp_avg / denom)
    param = param - lr * weight_decay * param
    # SOAP stores exp_avg/exp_avg_sq BF16 for unpreconditioned (encoder-scale) params -- round
    # -trip the reference the same way so this checks the update RULE, not storage precision.
    exp_avg = exp_avg.to(torch.bfloat16).float()
    exp_avg_sq = exp_avg_sq.to(torch.bfloat16).float()
    return param, exp_avg, exp_avg_sq


def _test_soap_unpreconditioned_matches_adam():
    """A 2000x2000 param sits above SOAP_MAX_PRECOND_DIM on both sides, so
    project/project_back must be an exact identity and the whole class must
    degrade to plain decoupled Adam -- byte for byte, not just "close"."""
    torch.manual_seed(0)
    big = nn.Parameter(torch.randn(2000, 2000))
    ref_param = big.detach().clone()
    ref_avg = torch.zeros_like(ref_param)
    ref_avg_sq = torch.zeros_like(ref_param)
    betas, eps, lr, wd = (0.9, 0.999), 1e-6, 1e-3, 0.02
    opt = SOAP([{"params": [big], "weight_decay": wd}], lr=lr, betas=betas, eps=eps,
               max_precond_dim=SOAP_MAX_PRECOND_DIM, precondition_frequency=5)
    for step in range(1, 4):
        grad = torch.randn_like(big)
        big.grad = grad.clone()
        opt.step()
        if step == 1:
            assert torch.equal(big.detach(), ref_param), "SOAP's first-ever step must not touch weights"
            continue
        ref_param, ref_avg, ref_avg_sq = _manual_decoupled_adam_step(
            ref_param, grad, ref_avg, ref_avg_sq, step - 1, betas, eps, lr, wd
        )
        assert torch.allclose(big.detach(), ref_param, atol=1e-5, rtol=1e-5), \
            f"unpreconditioned SOAP step {step} diverged from the plain-Adam reference"
    state = opt.state[big]
    assert all(len(side) == 0 for side in state["GG"]), "2000x2000 must be fully excluded by max_precond_dim"
    assert state["moment_dtype"] == torch.bfloat16
    print("self-test OK: unpreconditioned SOAP == decoupled Adam reference")


def _test_soap_preconditioned_and_resume():
    """A 6x4 param sits fully inside max_precond_dim on both sides, so it
    gets real eigenbasis rotation. Checkpointing mid-run (right after a
    basis refresh, precondition_frequency=3) and replaying the remaining
    gradients from the saved state_dict must land exactly where an
    uninterrupted run does -- this is the same save/resume contract the
    real training kernel relies on for its own quiet checkpoints."""
    torch.manual_seed(1)
    small = nn.Parameter(torch.randn(6, 4))
    opt = SOAP([{"params": [small], "weight_decay": 0.0}], lr=1e-2,
               max_precond_dim=SOAP_MAX_PRECOND_DIM, precondition_frequency=3)
    init = small.detach().clone()
    grads = [torch.randn(6, 4) for _ in range(8)]
    snapshot = None
    for step, grad in enumerate(grads):
        small.grad = grad.clone()
        opt.step()
        if step == 3:
            # deepcopy, not .cpu(): on CPU tensors .cpu() is a no-op that would alias into
            # GG's later in-place lerp_ mutations -- on the real TPU run this is instead
            # forced by a genuine device transfer, see to_cpu_state's own docstring note.
            snapshot = dict(opt=copy.deepcopy(opt.state_dict()), param=small.detach().clone())
    state = opt.state[small]
    assert all(len(side) > 0 for side in state["GG"]), "6x4 must be fully covered by max_precond_dim"
    assert state["moment_dtype"] == torch.float32
    assert torch.isfinite(small.detach()).all()
    assert not torch.equal(small.detach(), init), "preconditioned SOAP made no progress"
    direct_final = small.detach().clone()

    resumed_param = nn.Parameter(snapshot["param"].clone())
    resumed_opt = SOAP([{"params": [resumed_param], "weight_decay": 0.0}], lr=1e-2,
                        max_precond_dim=SOAP_MAX_PRECOND_DIM, precondition_frequency=3)
    resumed_opt.load_state_dict(snapshot["opt"])
    for grad in grads[4:]:
        resumed_param.grad = grad.clone()
        resumed_opt.step()
    assert torch.allclose(resumed_param.detach(), direct_final, atol=1e-6), \
        "resuming from a mid-run state_dict must reproduce the uninterrupted run exactly"
    print("self-test OK: preconditioned SOAP progresses and resumes exactly")


def _test_soap_1d_never_preconditioned():
    bias = nn.Parameter(torch.zeros(32))
    opt = SOAP([{"params": [bias], "weight_decay": 0.0}], lr=1e-2, max_precond_dim=SOAP_MAX_PRECOND_DIM)
    for _ in range(3):
        bias.grad = torch.randn(32)
        opt.step()
    state = opt.state[bias]
    assert state["GG"] == [[]]
    assert state["moment_dtype"] == torch.bfloat16
    assert torch.isfinite(bias.detach()).all() and not torch.equal(bias.detach(), torch.zeros(32))
    print("self-test OK: 1D params stay unpreconditioned")


def main():
    global torch_xla, xm, pl_xla, xmp, xr, average_precision_score
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl_xla
    import torch_xla.distributed.xla_multiprocessing as xmp
    from torch_xla import runtime as xr
    from sklearn.metrics import average_precision_score

    # Kaggle TPU kernels pre-set TPU_PROCESS_ADDRESSES/CLOUD_TPU_TASK_ID to values that
    # describe some OTHER (larger/multi-host) topology; libtpu then expects that many workers
    # to coordinate and fails with "Expected N worker addresses, got 1" the moment any
    # process touches the TPU. Already hit and fixed this exact way in this project's other
    # TPU kernels (frida-percat-lora / frida-llm-tpu-sft / berta-llm-tpu-sft / full_sft_continue)
    # -- must run before any TPU/torch_xla device access, hence first thing in main().
    os.environ.pop("TPU_PROCESS_ADDRESSES", None)
    os.environ.pop("CLOUD_TPU_TASK_ID", None)

    # xm.mark_step() itself raises this on every single call (many per step, x8 forked
    # ranks); the working full_sft_continue kernel tolerates the noise, but this kernel's
    # own Kaggle log capture went stale after only ~7 minutes on two separate real runs
    # while the process kept training healthily for 50min-2h+ after that (confirmed via
    # status polling) -- cheap, safe to filter regardless of whether it's the actual cause,
    # since xm.mark_step() itself is staying (torch_xla.sync() was tried and reverted
    # elsewhere in this project after a worse crash, see full_sft_continue's own docstring).
    # Set before xmp.spawn/fork so every forked rank inherits the same filter state.
    warnings.filterwarnings("ignore", message="Use torch_xla.sync instead")

    # PyTorch checkpointing probes torch.xla even when preserve_rng_state=False.
    torch.xla = torch_xla
    load_shared_data()
    # Current torch_xla (PJRT runtime) only accepts nprocs=1 or nprocs=None (auto-detect all
    # local devices) -- an explicit device count is a hard ValueError now, unlike the older
    # XRT runtime this call was presumably written against. mp_fn's own
    # `if world_size != WORLD_SIZE: raise` still guards that auto-detection actually finds 8.
    xmp.spawn(mp_fn, args=(), nprocs=None, start_method="fork")


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        self_test()
    else:
        main()
