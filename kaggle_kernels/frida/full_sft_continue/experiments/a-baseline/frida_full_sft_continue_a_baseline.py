"""
FRIDA (ai-forever/FRIDA, T5EncoderModel, 823M) full fine-tune from the
PRETRAINED base -- no LoRA, no resumed adapter. A second, independent
training track alongside kaggle_kernels/frida-percat-lora's 20 specialists:
where that kernel specializes an already-strong general model per category,
this one asks whether a genuinely fresh full-parameter fit, with its own
split/loss/optimizer choices, reaches a better general model in the first
place.

WHY FULL FINE-TUNE, AND WHY THAT FORCES ADAFACTOR
===========================================================================
Every FRIDA checkpoint in this project to date is LoRA. Full-FT unlocks
capacity LoRA structurally cannot touch (LayerNorm, embeddings, biases), but
Adam's own memory cost is the reason none of this project's TPU kernels have
done it: fp32 weights + fp32 grad + Adam (m, v) = 4x the 3.29GB parameter
tensor = 13.2GB, before a single activation, against a 16GB v5e-8 core.

Adafactor factorizes the second moment into a per-row and per-column vector
instead of a full matrix, so its own state is a few MB per weight tensor
rather than 1x its size. Weights + grad + (factored) state ~= 6.6GB, leaving
~9GB for activations -- and it is genuinely the native choice here, not a
memory hack bolted on: it is the T5 family's own paper optimizer, and FRIDA
IS a T5 encoder. Combined with gradient checkpointing (recomputing forward
activations during backward instead of storing all 24 layers' worth), the
per-sample activation cost drops to O(1) layers instead of O(24), which is
what the batch size below is sized against.

PER_CORE_TRAIN_BATCH is a documented ARITHMETIC ESTIMATE, not a measured
point -- consistent with this project's own TPU rule (kaggle-experiment-
safety: never auto-probe batch size on TPU, every probed candidate compiles
its own XLA program and stays resident in HBM, self-inflicting the OOM it
was meant to prevent). If a real run OOMs, halve it and relaunch; that is a
cheap, fast failure at the first training step, not an expensive one.

SPLIT: THE NEW ITEM-DISJOINT ONE, DELIBERATELY
===========================================================================
This is a FROM-SCRATCH run, so it is free to adopt
data/splits/matches_item_disjoint_70_30_seed42.parquet (union-find over the
pair graph, then stratified 70/30 by category x label x length-tercile; 0
shared item ids between train and val, verified). The OLD canonical
pair-level 80/20 split stays reserved for anything that continues an
existing checkpoint (see kaggle_kernels/frida-percat-lora, which must not
move items an already-trained adapter has seen into a "new" val half).

LLM DATA: A SUBSAMPLE, NOT THE FULL 11.19M ROWS -- SAY SO EVERYWHERE
===========================================================================
Measured throughput math (this project's own 120 rows/s at
PAIR_MAX_LENGTH=256 on frida-tpu-human-sft, scaled to seq=192 and to a
full-FT+Adafactor+grad-checkpointing forward/backward -- ~128 rows/s):
against the full 11,187,780-row LLM set, 3 epochs would need ~74h, well past
even a full week's 20h TPU quota. 1,000,000 LLM training rows (not per
category, unlike frida-percat-lora -- this run pools all categories into one
model) keeps 3 epochs inside a single ~8.2h session:
3 x (256,006 human-train + 1,000,000 llm) / 128 rows/s ~= 8.18h.

TRAIN/VAL: THE PRE-BUILT, NO-LEAK SPLIT -- NOT DERIVED INLINE ANYMORE
===========================================================================
An earlier version of this kernel drew its 1M-row LLM training subsample
directly from the full 11.19M-row pool via an inline stratified_llm_sample
call, with no val held out first -- so there was never an LLM-label val set
that hadn't already been trained on. kaggle_kernels/frida-llm-train-val-
split fixes this RETROACTIVELY: it reproduces that exact 1M-row subsample
byte-for-byte (same seed=42, same category-balanced quota stratified by the
10 discrete target levels -- matches_llm targets are a 9-judge vote count,
already 81.25% "confident" by nature, so a confidence filter was considered
and rejected: PR-AUC is decided at the boundary such a filter would
discard) and marks it "train", then draws a NEW 100K-row val set
EXCLUSIVELY from the untouched ~10.19M remainder, stratified by target
level + length tercile within category -- structurally leak-free by
construction. This kernel now reads both sides straight from that published
parquet (LLM_SPLIT_PATH_HINT) instead of re-deriving anything inline.

"Использовать llm-val не вместо human-val, а параллельно" (per direct
instruction): llm_val_frame is evaluated as a SECOND, independent metric at
every checkpoint, alongside (never instead of) the human item-disjoint val
above -- see do_full_checkpoint's own two eval calls and
train_state.json's val_macro_pr_auc vs llm_val_macro_pr_auc fields.

README.md carries the same prominent warning: any reported number from this
run describes training on "human + an LLM subsample", not "human + LLM".

THE HEAD: [1024, 512, 128, 1], OPPOSITE-DIRECTION SCHEDULE FROM THE BACKBONE
===========================================================================
A random head sitting on top of a pretrained backbone is the textbook case
for two learning rates: early on, large gradients from the untrained head
would otherwise backpropagate noise into the encoder. The two groups get
literally opposite-shaped warmup:
  - backbone: 0 -> LR linearly over the warmup window (ordinary warmup),
    then cosine decay to a floor -- exactly frida-percat-lora's schedule.
  - head: HEAD_LR_PEAK -> LR linearly over the SAME warmup window (starts
    high, "cools" down to the backbone's own peak), then joins the identical
    cosine decay -- so by the time warmup ends, both groups are moving
    together at the same rate for the remainder of training.
This lets the head get useful fast while it still cannot dominate the
encoder's own gradient signal early, and removes the two-schedule split
entirely once both are past the point where head-vs-backbone imbalance
would matter.

LOSS: CATEGORY-WEIGHTED FOCAL LOSS, PLUS 2x ON HUMAN ROWS
===========================================================================
This is genuinely different from frida-percat-lora's per-category
pos_weight, and for a real structural reason: that kernel trains 20
independent models, one per category, so each one only ever sees its own
category's positive rate -- category SIZE never enters the picture because
there is no pooling across categories to begin with. This kernel pools every
category into ONE model and ONE loss, where a large category's gradient
would otherwise simply outvote a small one during standard mean-reduction.
Measured over the real data (data/splits/matches_item_disjoint_70_30_seed42.
json): category sizes span only 17,286-23,357 pairs (1.4x) in the human set,
but the human+LLM-subsample TRAINING mix skews further once LLM rows are
folded in -- weighting by that mix's own realized category counts (computed
at setup time, not assumed) is what actually corrects it here, unlike
frida-percat-lora where it would have been close to a no-op.

Focal modulation (Lin et al., gamma=2.0, the standard literature default --
no other value was specified) downweights already-easy examples so the loss
budget concentrates on the ones still being gotten wrong, which matters more
here than in frida-percat-lora precisely because pooling means an easy
majority category could otherwise dominate total loss mass even after
per-category reweighting. HUMAN_WEIGHT=2.0 (same convention as
frida-percat-lora) doubles human-labeled rows against LLM soft-labeled ones
in the same weighted-mean reduction.

CHECKPOINT CONTRACT (per direct instruction, different from every other
kernel in this project)
===========================================================================
  - Every CHECKPOINT_EVERY_SECONDS (1h, was 5h -- per direct instruction):
    weights + optimizer state ONLY, NO eval, val score written as an
    explicit `null` (this project's own established convention for
    "skipped, not stale" -- see kaggle-experiment-safety's
    checkpoint-hygiene section). Adafactor's own state is small enough
    that this is cheap even hourly.
  - do_quiet_save always writes to the SAME "last" label -- every hourly
    save OVERWRITES the previous one in place, it does not accumulate a
    new checkpoint_hourNN directory per save (unlike frida-llm-tpu-sft's
    own hourly checkpoints, a different kernel with a different contract).
    One quiet checkpoint's disk cost (~1.5-2GB: full fp32 encoder +
    Adafactor factored state) is paid ONCE per session, not once per hour.
  - Real evaluation runs ONLY at an epoch boundary.
  - A cancelled/hard-capped session still leaves its last quiet save in
    /kaggle/working -- Kaggle persists that regardless of how the session
    ends, same guarantee every other checkpoint in this project's history
    has leaned on.

REAL CRASH (2026-08-21/22), AND WHAT ACTUALLY CAUSED IT
===========================================================================
A real run at PER_CORE_TRAIN_BATCH=48 died with "XLA:TPU compile permanent
error. Ran out of memory in memory space hbm. Used 16.62G of 15.75G hbm.
Exceeded hbm capacity by 888.03M." The traceback (see
_mp_fn -> `for step, batch in enumerate(mp_loader)` -> torch_xla's OWN internal
`torch_xla.sync()` call inside parallel_loader.py's next()) proves this was
the FIRST TRAINING step's own forward+backward compile --
NOT eval. Eval was never reached (do_full_checkpoint only runs at an epoch
boundary, tens of minutes away). The "Largest program allocations" in the
error were 20+ separate 216MB f32[1152,192,192] buffers, all XLA label
`remat3`/`remat4` fusions -- the backward-pass gradient-checkpoint RECOMPUTE
buffers, one cluster per encoder layer, which is exactly what
`gradient_checkpointing_enable()` trades for. An earlier hypothesis (now
corrected) assumed adding the second llm_val eval pass caused this
regression; it did not -- the crash is upstream of any eval code, on plain
training alone, at this batch/seq_len/attn_implementation combination.

`attn_implementation="eager"` was deliberately kept (not switched to
`sdpa`) despite being the obvious memory lever here: frida-nan-diagnose
already tested "eager" as a NaN-bug hypothesis on this exact model family
and it did NOT fix or cause the NaN bug either way (the real NaN fix was
autocast-inside-forward, unrelated) -- so `eager` is inert leftover
convention, not a validated-safe choice, and every other FRIDA training
kernel in this project uses it too. Switching it for a memory win here
would be the FIRST time this project runs T5EncoderModel training under
sdpa on TPU -- an unvalidated change on a real quota-burning run, against
this project's own "don't auto-probe / don't guess on TPU" rule. Left
untouched; PER_CORE_TRAIN_BATCH lowered instead (a small, well-precedented,
purely quantitative change -- see the constant's own comment for the exact
scaling math from the real 888MB overage).

SECOND REAL CRASH (v6, same day): batch=40 STILL OOM'd -- the smoke test
caught it in ~6.5 minutes, exactly as designed, instead of hours into a
real run. Different PJRT error shape this time ("RESOURCE_EXHAUSTED: Error
loading program: Attempting to reserve 9.83G... There are 9.05G free"),
surfacing at the smoke test's own `loss.item()` call rather than at
`mark_step()` itself -- torch_xla's mark_step() enqueues device execution
asynchronously and returns immediately; a device-side failure can only
surface at the next call that blocks for a real value, which `.item()` is
and `mark_step()` alone is not. The implied ~6.7G already resident at that
point is roughly double the first crash's ~3.34G "fixed" estimate --
consistent with full fine-tuning's per-parameter gradient buffers (one
fp32 grad per fp32 weight, another ~3.3G) still being resident right after
backward, a cost the first crash's proportional-scaling estimate did not
isolate separately. Since two proportional estimates in a row (64->48,
48->40) both still underestimated the real cost, PER_CORE_TRAIN_BATCH was
halved outright this time (40->20) rather than re-deriving a third
proportional guess from noisier data -- see the constant's own comment.

THIRD REAL CRASH (v7, same day): NOT an OOM -- 2 of 8 replicas vanished,
survivors got stuck permanently retrying a broken collective state
===========================================================================
batch=20 got well past the smoke test and into real training (the failure
surfaced ~2500s in) -- but `kaggle kernels output`'s retained log for this
run starts ONLY at that ~2500s mark: every earlier line (setup, smoke test,
CONFIG, the first ~40 minutes of real training) is gone. The cause is the
crash itself: from ~2500s onward, EVERY subsequent training step on every
surviving replica failed identically with
`AttributeError: module 'torch' has no attribute 'xla'` (raised inside
`model()`'s own `torch.autocast("xla", ...)` call) -- and the existing
per-step except-handler printed a "step failed and was skipped" line, on
EVERY rank, for EVERY one of ~6,700 consecutive failed steps over the next
~14 minutes, with no circuit breaker. That flood is almost certainly what
evicted the earlier, actually-diagnostic log content from Kaggle's
retention -- a real, self-inflicted loss of evidence, not just noise.

What the surviving evidence DOES show: ranks 1 and 6 (of 8) never appear
ANYWHERE in the log, not even once -- they were already gone by the time
this failure mode started. `xm.optimizer_step` is a cross-core all-reduce;
losing 2 of 8 participants would stall it on the remaining 6 until some
internal PJRT/XLA recovery kicked in, which is a plausible mechanism for
the survivors ending up in a broken-but-not-fully-dead state where
`torch`'s `xla` binding (normally installed by `import torch_xla`'s own
side effects) is no longer present. WHY ranks 1 and 6 died is NOT
recoverable from this log -- whatever they printed, if anything, was in the
portion the flood evicted. This is a real, currently-unresolved gap.

Two changes, given what IS actionable:
  1. The per-step failure log moved to rank==0-only (was `force=True`,
     printing from every rank) and now reports a running
     "N/MAX_CONSECUTIVE_STEP_FAILURES consecutive" count instead of a bare
     message -- an 8x cut to the flood on its own, whatever the cause.
  2. MAX_CONSECUTIVE_STEP_FAILURES=20: a step failure resets the counter to
     0 on any LATER success, but N consecutive failures now raises loudly
     instead of retrying for the rest of the session. A failure mode that
     repeats identically 20 times in a row is not transient -- it is not
     going to resolve itself on attempt 21, 200, or 6,700, and the cost of
     being wrong (stopping a genuinely-recoverable run early) is a full,
     clean traceback plus a relaunch, versus the cost of being right this
     time: 14 minutes of pure waste and the evidence needed to ever
     diagnose it destroyed in the process.
`torch_xla.sync()` (this session's earlier replacement for the deprecated
`xm.mark_step()`, done purely to silence a cosmetic DeprecationWarning) was
reverted back to `xm.mark_step()` everywhere in this file as a precaution:
it is provably NOT the raise site for this specific crash (every call site
sits outside the try/except that caught it, and the ones inside functions
that DID run first -- the smoke test, `evaluate()` -- either succeeded or
were never reached), but it is also new, unproven code with zero track
record in this project, while `xm.mark_step()` has an extensive one across
many real successful TPU runs here. Removing an unproven variable during a
hard-to-diagnose infra-level failure is cheap insurance, not a confirmed
fix -- framed honestly as that, not as "found and fixed the root cause."

FOURTH REAL CRASH (v8, same day): the circuit breaker worked -- and gave a
full, clean, undamaged log for the first time, with the real answer in it
===========================================================================
v8's circuit breaker + rank-0-only logging did exactly their job: instead
of ~6,700 duplicate lines evicting the evidence, the log stayed intact
end-to-end, the run stopped itself after exactly 20 consecutive failures
(~13s of wasted wall-clock, not 14 minutes), and the real traceback was
sitting right there:

    torch/utils/checkpoint.py:1499 _checkpoint_without_reentrant_generator
      -> torch/utils/checkpoint.py:93 _get_device_module(device_type)
      -> device_module = getattr(torch, device)   # device="xla"
      -> AttributeError: module 'torch' has no attribute 'xla'

This is PyTorch's own NON-reentrant checkpoint implementation (HF's
`gradient_checkpointing_enable()` default) trying to fetch a per-device RNG
helper via `torch.<device_type>` -- a pattern PyTorch core supports for its
own backends (`torch.cuda`, `torch.mps`, ...) but that `torch_xla` never
registers itself under (`torch.xla` genuinely does not exist on this
image's stack). A real, previously-unknown incompatibility between this
project's first-ever full-fine-tune-with-gradient-checkpointing attempt and
this exact torch/torch_xla/transformers version triplet -- every other
FRIDA kernel in this project trains via LoRA, where the base model is
frozen and gradient checkpointing was never exercised the same way.

Puzzling wrinkle, not fully explained: the smoke test's own train step (an
IDENTICAL `model(...)` + `.backward()` call, same code, same device, run
moments earlier) printed a real, distinct loss for all 8 ranks --
`_get_device_module("xla")` did NOT raise there, only on the first REAL
`mp_loader`-sourced batch immediately after. Whatever the exact trigger
(some difference between a hand-built synthetic batch and an
`MpDeviceLoader`-sourced one that the smoke test does not reproduce), the
underlying bug and its standard fix do not depend on understanding it: pass
`gradient_checkpointing_kwargs={"use_reentrant": True}` to
`gradient_checkpointing_enable()`, which selects PyTorch's OLDER reentrant
checkpoint implementation. That one manages RNG state via
`torch.random.fork_rng`-style APIs instead of a per-backend device module
lookup, so it never touches this code path at all. The one real constraint
reentrant-mode checkpointing has -- at least one input to the checkpointed
function must have `requires_grad=True`, or backward silently no-ops
through it -- is already satisfied here: hidden_states flowing between
layers inherit `requires_grad=True` from the (fully trainable, per this
kernel's own full-FT design) embedding weights.

FIFTH REAL CRASH (2026-08-22): the FOURTH crash's fix, and its own
reasoning, were both wrong -- the IDENTICAL AttributeError recurred
===========================================================================
A real run on the "fixed" (use_reentrant=True) code crashed again with the
exact same `AttributeError: module 'torch' has no attribute 'xla'`. This
time the traceback proves reentrant mode WAS genuinely active when it
happened -- it runs through `torch/utils/checkpoint.py:488 checkpoint()`
-> `CheckpointFunction.apply` -> `CheckpointFunction.forward` (line 241)
-> `_get_device_module(ctx.device_type)` -> `getattr(torch, "xla")`. The
FOURTH crash's docstring claimed "[reentrant] manages RNG state without
that device-module lookup, sidestepping the bug entirely" -- that claim
was incorrect for this torch version: `_get_device_module` is called by
BOTH the reentrant and non-reentrant checkpoint implementations whenever
`preserve_rng_state=True`, which is the default in both, and
`use_reentrant=True` alone never overrode it because it was never the
active default in the first place -- the kwarg only selects WHICH
implementation runs the RNG save/restore, not WHETHER it runs.

The corrected fix, in two independent parts (either alone would likely
have been sufficient; both are applied together as defense in depth
rather than re-litigating which one "really" fixed it on a third
quota-burning attempt):
  1. `preserve_rng_state=False` added alongside `use_reentrant=True` --
     removes the actual need for the device-module lookup in either mode,
     since its only purpose is saving/restoring accelerator RNG state
     across the checkpoint boundary, which this kernel does not depend on
     (checkpointing's recompute-in-backward already tolerates dropout
     drawing different values than the original forward pass -- that
     tolerance is inherent to what checkpointing trades away for memory,
     everywhere it is used at this project's scale).
  2. `torch.xla = torch_xla` registered once at import time (see the
     import block), so `getattr(torch, "xla")` succeeds regardless of
     which code path calls it -- covers any OTHER place in this dependency
     stack that might make the same assumption PyTorch's own
     `torch.cuda`/`torch.mps` support, which this crash's root cause shows
     `torch_xla` does not provide for itself.

VALIDATION REWRITE: SHARDED ACROSS ALL 8 CORES, ONE FIXED SHAPE, all_gather
===========================================================================
The old `make_eval_loader`/`evaluate` (byte-identical copy from
frida-llm-tpu-sft, see README.md's "Eval на TPU без шардинга" section) ran
the FULL val set independently on all 8 cores (no DistributedSampler) and
compiled a SECOND XLA shape for the ragged last ("remainder") batch. Fixed
by `make_eval_shard`/`evaluate`:
  - Each rank gets its own disjoint ~1/8 slice of val (contiguous block
    partition), padded with wrapped-around duplicate rows so every rank
    runs the exact SAME number of eval_batch-sized batches -- one shape,
    one compile, for the whole run (no remainder loader at all).
  - Padding rows are tracked in a `valid` mask and dropped again before
    metrics are computed -- they never influence macro_pr_auc.
  - `xm.all_gather` concatenates every rank's (logits, labels, categories,
    valid) into an identical full-val tensor on every replica (this val
    set is ~100K rows, a few MB even gathered 8x over -- not a real memory
    concern), so `compute_metrics` runs ONCE, on the true full val set, not
    approximated per-shard. Deterministic: every rank computes the same
    macro_pr_auc independently; only rank 0's `save_checkpoint` persists.
  - This also directly speeds up eval ~8x (each core now does 1/8 the
    forward passes instead of all of them) and removes one whole cold
    XLA compile (the old ragged remainder shape) from the first checkpoint.

SIXTH REAL CRASH (2026-08-22): the rewrite above reasoned correctly about
GATHERED data size ("a few MB even gathered 8x over") and missed a
completely different cost -- the UNGATHERED, undeferred graph size before
the first sync point
===========================================================================
A run that survived all five training-side crashes above died at the very
first real eval (epoch 1 boundary, step 7850/23550) -- not a Python
exception, an external SIGTERM killing all 8 ranks ~200s after the last
training log line, surfacing at the top level as
`concurrent.futures.process.BrokenProcessPool`. `make_eval_loader_sharded`
returns a bare `torch.utils.data.DataLoader` -- unlike every TRAINING
loader in this file, which is wrapped in `pl_xla.MpDeviceLoader` before
being iterated. `evaluate()`'s per-batch loop moved tensors to device by
hand (`input_ids.to(device)`, etc.) but never called anything that
forces XLA to materialize the graph -- `torch.no_grad()` skips gradient
tracking, it does not sync. Training never needed a fix here because
`xm.optimizer_step()` is a real per-step sync point (confirmed by REAL
CRASH #1's own traceback: "torch_xla's OWN internal torch_xla.sync() call
inside parallel_loader.py's next()" -- MpDeviceLoader's next() syncs every
batch on its own, which is WHY training never hit this). Eval has no
analogous per-batch collective, so all ~191 forward passes in one rank's
shard (human val=109,648 rows / 8 replicas / eval_batch=72) piled up as
ONE undeferred XLA program, not materialized until the explicit
`xm.mark_step()` after `xm.all_gather` -- a graph on the order of 191x any
single program this kernel had built before, either hanging past whatever
liveness window Kaggle's infra allows or genuinely exhausting memory
during that one giant compile+execute. Fixed in `evaluate()` by wrapping
`loader` in `pl_xla.MpDeviceLoader` too, so each batch syncs (and is
device-placed) as it's produced -- see that function's own docstring for
the exact reasoning, kept there since that's where a future reader will
actually be looking when touching eval code.

EARLY SMOKE TEST: catch an HBM-fit problem in ~1-2 minutes, not mid-run
===========================================================================
Right after the model/optimizer/scheduler are built (before touching any
real data), `_smoke_test_train_step`/`_smoke_test_eval_step` run ONE
synthetic batch through the EXACT same forward+backward (train) and
forward-only (eval) path, at the real PER_CORE_TRAIN_BATCH/
PER_CORE_EVAL_BATCH/PAIR_MAX_LENGTH shapes, then force `xm.mark_step()` to
compile+execute immediately instead of lazily. The train smoke step calls
`.backward()` then discards gradients via `zero_grad(set_to_none=True)`
WITHOUT ever calling `xm.optimizer_step` -- backward() only populates
`.grad` tensors, never mutates `.data`, so the model's real pretrained
weights are provably bit-for-bit unchanged afterward; Adafactor's own state
is never initialized by this (state only exists after a real .step() call).
If either smoke step OOMs, the kernel dies loudly within ~1-2 minutes of
launch, with a clear log line naming which constant to lower -- instead of
hours into a real run (exactly what happened here).

TPU LESSONS CARRIED FORWARD (all real, previously hit -- see
frida-percat-lora / frida-llm-tpu-sft / berta-llm-tpu-sft docstrings for the
original incidents)
===========================================================================
  1. os.environ.pop("TPU_PROCESS_ADDRESSES"/"CLOUD_TPU_TASK_ID") +
     start_method="fork".
  2. STATIC SHAPES: drop_last=True on every loader, fixed per-core batch.
  3. FIXED batch size, no auto-probing on TPU.
  4. xm.optimizer_step(optimizer), not optimizer.step().
  5. autocast opened INSIDE forward().
  6. ALL CPU-only setup ONCE in main() before torch_xla.launch().
  7. Eval wrapped in try/except that cannot kill the run.
  8. Checkpoint hygiene: own directory per save, explicit null val score,
     wall-clock (not epoch) checkpoint cadence.
  9. Eval sharded across all replicas + all_gather'd, one fixed padded
     shape -- never re-run the full val set redundantly per core.
  10. A cheap synthetic smoke test at real batch/seq shapes, right after
      model construction, before any real data -- fails an HBM-fit problem
      in minutes, not hours.

CONTINUE LINEAGE (forked from frida-full-sft after its 0.7582 val / 0.481
test result -- see kaggle_kernels/frida-full-sft-continue/README.md for the
full experiment matrix and time estimates)
===========================================================================
Two additions on top of the original recipe, both requested after seeing
the finished 0.7582/0.481 run and reasoning about WHY it likely worked:

  1. RESUME (see _find_resume_source/load_full_model_from_checkpoint), TWO
     DIFFERENT MODES depending on WHOSE checkpoint is found:
       - CROSS-experiment (the shared frida-full-sft-w/best, or a sibling
         experiment's own output): fresh warmup+cosine LR schedule at
         LR_RESUME_MULTIPLIER x the from-scratch peak -- same convention
         frida-percat-lora's own resume path already uses (continuing an
         already-good checkpoint should take smaller steps than
         random-init). This is the mode every sibling's FIRST session uses.
       - SAME-experiment (this exact EXPERIMENT_NAME's own prior session,
         after a wall-clock cutoff): EXACT continuation -- optimizer
         param_groups/state, scheduler position (scheduler.pt,
         LambdaLR.state_dict()), global_step, and epoch/mid-epoch batch
         position are all restored, not rebuilt. A resumed run that landed
         mid-epoch (the common case for c/d/e, whose own epochs run
         5-17.5h -- far longer than one 9h session) re-requests that SAME
         epoch number from make_train_loader (its DistributedSampler
         shuffle is seeded by `epoch` alone, so the row order is
         byte-for-byte reproducible) and skips exactly the N batches
         global_step already implies before touching a single real step --
         see the epoch loop's own comment. This mode is essential for
         c/d/e to be resumable at all across the many sessions they need;
         it is UNTESTED against a real multi-session run (no prior run of
         this exact mechanism exists yet in this project) -- flagged
         honestly, not silently assumed correct.
  2. IN-CATEGORY POSITIVE-RATE WEIGHTING (see _positive_rate_weight):
     category_weight (original recipe) balances category SIZE only;
     positive RATE within a category was left unbalanced, unlike frida-
     percat-lora's own per-category pos_weight (that kernel trains one
     model per category, so this was automatic there). Added here as a
     third per-sample multiplier, all three combined by multiplication,
     for every experiment in this lineage including the from-scratch-
     recipe-repeat one -- not isolated as its own separate A/B, per direct
     instruction.

LLM_TRAIN_SUBSAMPLE_SIZE (see stratified_llm_sample) lets an experiment
scale the LLM side up from the original 1M without touching the human
side: None uses the full published train split (currently 11,087,780 rows,
NOT rebalanced by category/target-level -- see that split kernel's own v2
docstring), an int draws a fresh category-balanced, target-level-
stratified subsample of that size from it.
"""
import os
import sys
import gc
import json
import math
import time
import hashlib
import subprocess

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "polars"])

import numpy as np
import pandas as pd
import polars as pl

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from transformers import T5EncoderModel, set_seed
from transformers.optimization import Adafactor
from sklearn.metrics import average_precision_score

import torch_xla
import torch_xla.core.xla_model as xm
from torch_xla import runtime as xr
import torch_xla.distributed.parallel_loader as pl_xla

# FIFTH REAL CRASH fix, part 1 of 2 -- see the module docstring's own
# section for the full story. torch_xla never registers itself as a
# `torch.xla` submodule (unlike torch.cuda/torch.mps), but
# torch.utils.checkpoint._get_device_module() calls exactly
# `getattr(torch, "xla")` to resolve it for RNG-state handling --
# in BOTH the reentrant and non-reentrant checkpoint implementations
# whenever preserve_rng_state=True (the default), contrary to what the
# FOURTH REAL CRASH's fix assumed. Registering it here makes that lookup
# succeed everywhere in the process, regardless of which code path calls
# it -- a strict superset of the reentrant-only workaround.
if not hasattr(torch, "xla"):
    torch.xla = torch_xla


# =====================================================================
# CONFIG
# =====================================================================

SEED = 42
MODEL_NAME = "ai-forever/FRIDA"

ITEM_SPLIT_PATH_HINT = "/kaggle/input/datasets/maxamsecret/ozon-ecup-2026-matching/matches_item_disjoint_70_30_seed42.parquet"
# Pre-built, no-leak LLM train/val split (kaggle_kernels/frida-llm-train-val-
# split). Supersedes this kernel's OWN earlier inline stratified_llm_sample
# call against the raw matches_llm.parquet/items.parquet, which drew its 1M
# training rows straight from the full 11.19M-row pool with no val held out
# first -- a real gap the split kernel exists to close retroactively (its
# "train" side reproduces that exact 1M byte-for-byte; "val" is new, drawn
# only from the untouched ~10.19M remainder).
LLM_SPLIT_PATH_HINT = "/kaggle/input/datasets/maxamsecret/frida-llm-train-val-split/llm_train_val_split.parquet"

# v2 caches (frida-tokenize-items-cpu): correct char budget (140/32/529, see
# reports/experiment_log.md "Char-budget rule v2") + the light v2 attribute
# cleanup. Human scope alone would be items_human.parquet-derived; this run
# needs BOTH scopes (human for train+val, LLM for the subsample), so both
# halves of the one published dataset are used.
HUMAN_CACHE_HINT = "/kaggle/input/datasets/maxamsecret/frida-tokenize-items-v3/human_scope"
LLM_CACHE_HINT = "/kaggle/input/datasets/maxamsecret/frida-tokenize-items-v3/llm_scope"

ITEM_MAX_LENGTH = 256
# Same reasoning as frida-percat-lora: 192 is what the real judging stand can
# actually SERVE inside its time budget (176 passed at 0.4087; 192 measured
# at 1.65x over Public's budget). Training at the serving length removes the
# train/serve mismatch every FRIDA submission so far has paid for.
PAIR_MAX_LENGTH = 192

CATEGORY_LABEL_TOKEN = 55917    # " Category" (+ colon=30 immediately after)
ATTRIBUTES_LABEL_TOKEN = 91837  # " Attributes"
LABEL_COLON_TOKEN = 30

# Head: [1024, 512, 128, 1] per direct instruction -- three hidden blocks
# (Linear+LayerNorm+GELU+Dropout) narrowing from the encoder's 1536-dim
# pooled output down to a single logit.
HEAD_HIDDEN = (1024, 512, 128)
HEAD_DROPOUT = 0.2

# Backbone warms 0 -> LR; head starts at LR * HEAD_LR_PEAK_MULTIPLIER and
# cools DOWN to LR over the same warmup window (see docstring). Both then
# share one cosine decay to LR_MIN_RATIO * LR.
LR = 1e-4                       # backbone peak -- lower than LoRA's 2e-4 on
                                 # purpose: full-FT touches every parameter,
                                 # not a small low-rank delta, so a LoRA-sized
                                 # LR here would be a much larger relative step
HEAD_LR_PEAK_MULTIPLIER = 10.0  # head starts at LR * this (1e-3), typical
                                 # for a freshly-initialised small MLP head
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.10
LR_MIN_RATIO = 0.05
HUMAN_WEIGHT = 2.0
FOCAL_GAMMA = 2.0

# ---- CONTINUED-TRAINING EXPERIMENT KNOBS (added post-0.7582/0.481) -------
# This file is the shared base for kaggle_kernels/frida-full-sft-continue-*/
# -- five sibling kernels, each a copy of this file with only the five
# constants below edited, all resuming from the ORIGINAL frida-full-sft-w/
# best checkpoint (val=0.7582, test=0.481). See that directory's README for
# the full rationale and per-experiment values.
EXPERIMENT_NAME = 'continue_a_baseline_3ep'
                                     # every checkpoint's train_state.json so
                                     # _find_resume_source can tell "my own
                                     # prior session" from "someone else's".
LLM_TRAIN_SUBSAMPLE_SIZE = 1_000_000
                                     # published (stratified_llm_sample below
                                     # is only invoked when this is an int).
LR_RESUME_MULTIPLIER = 0.5          # peak LR scale for a WARM-STARTED run
                                     # (resumed from an existing checkpoint)
                                     # vs. a from-scratch one -- same halving
                                     # convention frida-percat-lora's own
                                     # LR_RESUME_MULTIPLIER already uses, for
                                     # the same reason: continuing an
                                     # already-good checkpoint should take
                                     # smaller steps than random-init.
# In-category positive-rate weighting (added alongside the resume knobs,
# for ALL experiments including a from-scratch run): category_weight above
# balances category SIZE; this balances 0/1 WITHIN each category, the same
# axis frida-percat-lora already corrects per-category via its own
# per-category pos_weight -- this kernel pools categories into one loss, so
# it needs its own version, computed on the realized training mix (see
# _load_shared_data). Applied as a per-sample multiplier that recovers the
# classic BCEWithLogitsLoss(pos_weight=...) behavior at label=1/0 and
# interpolates smoothly for LLM soft labels in between -- same smooth-
# generalization philosophy focal_bce_per_sample's own pt already uses.

# MEASURED, not the original arithmetic estimate, THREE times now -- each
# via a real compile-time OOM, the last one caught by the early smoke test
# in ~6.5 minutes instead of hours in:
#   64/96: "Used 19.05G of 15.75G hbm. Exceeded by 3.30G" (1.21x over) ->
#     scaled down to 48/72.
#   48/72: OOM'd AGAIN, still at compile time, still on the very first real
#     training step: "Used 16.62G of 15.75G hbm. Exceeded by 888.03M" (only
#     1.056x over -- eval was never reached). Proportional scaling from
#     this number gave 44 (exact) / 40 (with margin) -> tried 40.
#   40/72: OOM'd a THIRD time, caught by the new smoke test at the
#     `loss.item()` sync (mark_step()'s own device execution is async; the
#     failure only surfaces at the next blocking read) -- a DIFFERENT PJRT
#     error shape: "RESOURCE_EXHAUSTED: Error loading program: Attempting
#     to reserve 9.83G... There are 9.05G free". The ~6.7G already resident
#     at that point (15.75-9.05) is roughly double the previous crash's
#     ~3.34G "fixed" estimate -- consistent with full-FT's per-parameter
#     gradient buffers (another ~3.3G, same size as the params themselves)
#     still being resident post-backward, which the earlier proportional
#     estimate did not separately account for. Two proportional estimates
#     in a row still underestimated the real cost, so this time: a flat
#     halving (per this project's own "if it OOMs, halve it and relaunch"
#     TPU policy, stated below) instead of another fragile proportional
#     guess -- 40 -> 20. If 20 is still not enough, the smoke test will say
#     so within minutes, at effectively zero quota cost, and this comment
#     should record that fourth number too rather than being trusted as
#     final. PER_CORE_EVAL_BATCH is unchanged: eval is forward-only (no
#     backward, no grad-checkpoint recompute, no persistent gradient
#     buffers), so it was never the actual cost driver in any of the three
#     crashes above -- the smoke test's own eval-step check would catch it
#     directly and separately if that assumption is ever wrong.
PER_CORE_TRAIN_BATCH = 20
PER_CORE_EVAL_BATCH = 72

EPOCHS = 3

MAX_TRAIN_SECONDS = 30600       # 8.5h -- Kaggle's TPU session cap is 9h
# v2 of the CHECKPOINT CONTRACT (was 18000/5h): hourly per direct
# instruction. Still weights+optimizer only, no eval, and still a SINGLE
# reused "last" label (see do_quiet_save) -- overwritten in place every
# interval, not a new checkpoint_hourNN directory, so this does NOT
# accumulate ~1.5GB/interval of output the way the (unrelated)
# frida-llm-tpu-sft kernel's own hourly checkpoints do.
# CONTINUE lineage: 3600 -> 13000 (~3.6h), per direct instruction -- this
# lineage's own quiet save now ALSO carries scheduler.pt + epoch_complete +
# lr_scale (see save_checkpoint), specifically so it is a full, exact
# resume point (not just weights+optimizer) across the multi-session
# experiments (c/d/e) whose own epochs run 5-17.5h, far longer than one 9h
# TPU session -- every quiet save must be independently sufficient to
# reconstruct training state exactly, since a session can be killed between
# any two of them.
CHECKPOINT_EVERY_SECONDS = 13000   # weights+optimizer+scheduler, no eval
                                    # (see docstring's CHECKPOINT CONTRACT)
LOG_EVERY_STEPS = 50
# See docstring's "THIRD REAL CRASH": a real run hit an unrecoverable,
# permanently-repeating step failure (2 of 8 replicas silently vanished,
# the survivors' collective ops got stuck in a broken state) and spun
# uselessly for 14 minutes, logging ~6,700 identical lines from every
# surviving rank -- enough that Kaggle's own log retention evicted the
# actually-useful setup/smoke-test/early-training output entirely. A run
# that fails the SAME way this many times in a row is not going to recover;
# stop it loudly instead of burning the rest of the session's wall-clock
# budget on it.
MAX_CONSECUTIVE_STEP_FAILURES = 20

# 128 measured-and-scaled (see docstring) at seq=192, full-FT+grad-checkpoint
# -- used ONLY to size the LR schedule's horizon, not as a promise.
PROJECTED_ROWS_PER_SEC = 128

WORKING_DIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
OUTPUT_DIR = os.path.join(WORKING_DIR, "frida_full_sft")

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


# =====================================================================
# ITEM CACHES -- v2 layout (same flat ids/input_ids/attention_mask shape as
# v1; only the char-budget/attribute-policy that built them differs, which
# is invisible at this layer). Verbatim from frida-percat-lora/
# frida-llm-tpu-sft -- proven on real multi-hour TPU runs, not re-derived.
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
                    print(f"[CACHE] hint {hint} missed -- matched n_items={expect_n_items:,} "
                          f"at {root}", flush=True)
                    return root
    raise RuntimeError(f"[CACHE] no item cache with n_items={expect_n_items:,} found "
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


def make_eval_shard(frame, num_replicas, rank, eval_batch):
    """This rank's disjoint ~1/num_replicas slice of `frame` (contiguous
    block partition), padded with wrapped-around duplicate rows so every
    replica runs the exact SAME number of eval_batch-sized batches -- one
    fixed shape for the whole run, no ragged "remainder" batch, no
    redundant full-val-set-per-core work (see docstring's "VALIDATION
    REWRITE"). `valid` marks which rows in THIS rank's padded shard are
    real (True) vs. wraparound padding (False); padding is dropped again
    after all_gather, before any metric is computed."""
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
# MODEL -- genuine full fine-tune, no PeftModel anywhere
# =====================================================================

class FullSftHead(nn.Module):
    """[1024, 512, 128, 1] per direct instruction -- three Linear+LayerNorm+
    GELU+Dropout blocks narrowing the encoder's pooled 1536-dim [CLS] output
    down to one logit."""

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


class FridaFullSftClassifier(nn.Module):
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
    """No PeftModel, no adapter, no bf16-frozen base -- every parameter is
    trainable and kept in fp32 (autocast handles the bf16 forward pass; see
    forward() above), which is what "full fine-tune" means here. Gradient
    checkpointing recomputes each layer's activations during backward
    instead of storing all 24 layers' worth -- the memory trade this
    kernel's batch size is sized against (see module docstring)."""
    model = T5EncoderModel.from_pretrained(MODEL_NAME, dtype=torch.float32,
                                           attn_implementation="eager")
    # See docstring's "FIFTH REAL CRASH" -- the FOURTH crash's fix
    # (use_reentrant=True alone) turned out NOT to sidestep
    # _get_device_module("xla"): a real run crashed again with the
    # IDENTICAL AttributeError, its traceback going through
    # CheckpointFunction.apply (proving reentrant mode WAS active).
    # preserve_rng_state=False removes the actual need for that lookup in
    # either mode -- it exists purely to save/restore accelerator RNG state
    # across the checkpoint boundary, which this kernel does not need
    # (gradient checkpointing's recompute-in-backward already tolerates
    # dropout drawing different values than the original forward; that is
    # what checkpointing means everywhere it's used at this project's
    # scale). Kept use_reentrant=True too and added the torch.xla
    # registration at import time (see the import block) as defense in
    # depth -- three independent reasons this exact AttributeError cannot
    # recur, not reliance on any single one.
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": True, "preserve_rng_state": False})
    hidden_dim = model.config.d_model
    for p in model.parameters():
        p.requires_grad = True
    n_params = sum(p.numel() for p in model.parameters())
    _log(rank, "MODEL", f"full fine-tune: {n_params:,} trainable encoder params "
                        f"(gradient checkpointing ON)", force=True)

    head = FullSftHead(hidden_dim)
    return FridaFullSftClassifier(model, head).to(device), hidden_dim


def make_param_groups(model, lr_scale=1.0):
    """Two groups so the head and backbone can run opposite-direction warmup
    (see docstring's THE HEAD section). lr_scale < 1.0 for a warm-started
    (resumed) run -- see LR_RESUME_MULTIPLIER's own comment."""
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
    """Starts at 1.0 (i.e. the head's own peak, LR*HEAD_LR_PEAK_MULTIPLIER)
    and cools LINEARLY down to 1/HEAD_LR_PEAK_MULTIPLIER (i.e. matching the
    backbone's peak LR in absolute terms) over the warmup window -- the
    mirror image of the backbone's 0->1 ramp. After warmup both groups share
    the identical cosine trajectory, so the "opposite direction" framing only
    describes the warmup phase, not the whole run."""
    floor = 1.0 / HEAD_LR_PEAK_MULTIPLIER
    def lr_lambda(step):
        if step < warmup_steps:
            t = step / max(1, warmup_steps)
            return 1.0 + (floor - 1.0) * t
        progress = min((step - warmup_steps) / max(1, total_steps - warmup_steps), 1.0)
        return floor * (min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress)))
    return lr_lambda


# =====================================================================
# LOSS -- category-weighted focal, +2x on human rows
# =====================================================================

def focal_bce_per_sample(logits, targets, gamma):
    """Binary focal loss, per-sample (no reduction) -- the caller applies
    category/human weighting and does its own weighted mean. targets may be
    continuous (LLM soft labels): pt is defined the same way regardless
    (Lin et al.'s pt = p if y=1 else 1-p generalises directly to soft y as
    the probability mass agreeing with the label)."""
    p = torch.sigmoid(logits)
    ce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    pt = targets * p + (1 - targets) * (1 - p)
    return ((1 - pt).clamp(min=1e-6) ** gamma) * ce


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


def evaluate(model, loader, valid_mask, device, category_names, world_size):
    """Every replica runs its own disjoint shard (see make_eval_shard), then
    xm.all_gather assembles the full padded val set identically on every
    replica -- this is a COLLECTIVE op, so it must be called uniformly by
    all ranks (it already is: do_full_checkpoint calls this unconditionally
    on every rank, same as before). Metrics are computed once the padding
    is dropped, giving the true macro_pr_auc over the whole val set, not an
    average of 8 partial per-shard APs (which would be a different, wrong
    number for a per-category metric with uneven category counts).

    REAL CRASH (SIXTH, 2026-08-22): `loader` used to be iterated bare (a
    plain DataLoader, not wrapped in pl_xla.MpDeviceLoader like every
    training loader in this file is), with input_ids/attention_mask moved
    to device by hand inside the loop. A run that got past all 5 training
    crashes above died SILENTLY at the very first real eval (epoch 1
    boundary, step 7850) -- no Python traceback, just ~200s of total
    silence right after this function's own xm.mark_step() calls, then an
    external SIGTERM killing all 8 ranks (BrokenProcessPool at the top
    level). Root cause: a bare DataLoader has no per-batch sync point, so
    ALL ~191 forward passes in this rank's shard (human val=109,648 rows /
    8 replicas / eval_batch=72) got deferred into ONE uncompiled XLA graph,
    not materialized until the explicit xm.mark_step() after all_gather --
    a graph roughly 191x the size of any single compiled program this
    kernel had ever built, unlike training which calls xm.optimizer_step
    (a real sync point) every step. REAL CRASH #1's own docstring already
    named the mechanism without connecting it here: "torch_xla's OWN
    internal torch_xla.sync() call inside parallel_loader.py's next()" --
    that per-batch sync is exactly what MpDeviceLoader provides and a bare
    DataLoader does not. Fixed by wrapping `loader` in MpDeviceLoader here,
    matching every training loader in this file byte-for-byte in spirit --
    each batch is now synced (and its tensors device-placed) as it's
    produced, bounding the graph to one batch at a time instead of the
    whole shard. The `.to(device)` calls below are now redundant (a
    same-device .to() is a no-op) but left in place rather than removed --
    matches this file's own established style (the training loop keeps an
    explicit batch_labels.to(device) even under MpDeviceLoader too) and
    keeps this diff to the one line that matters."""
    model.eval()
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
    return {"macro_pr_auc": macro, "per_category": per_cat, "n": int(keep.sum())}


# =====================================================================
# EARLY SMOKE TEST -- one synthetic batch through the real train and eval
# paths, at the real shapes, right after the model is built and before any
# real data is touched (see docstring's "EARLY SMOKE TEST" section). Turns
# an HBM-fit problem into a ~1-2 minute failure instead of an hours-into-
# the-run one.
# =====================================================================

def _make_synthetic_batch(batch_size, pair_max_length, pad_id, vocab_size, n_categories, device):
    lengths = np.random.randint(pair_max_length // 4, pair_max_length + 1, size=batch_size)
    input_ids = np.full((batch_size, pair_max_length), pad_id, dtype=np.int64)
    attention_mask = np.zeros((batch_size, pair_max_length), dtype=np.int64)
    for i, L in enumerate(lengths):
        input_ids[i, :L] = np.random.randint(3, vocab_size, size=L)  # skip special ids 0/1/2
        attention_mask[i, :L] = 1
    labels = np.random.randint(0, 2, size=batch_size).astype(np.float32)
    categories = np.random.randint(0, n_categories, size=batch_size).astype(np.int64)
    scope = np.random.randint(0, 2, size=batch_size).astype(np.int64)
    return (torch.from_numpy(input_ids).to(device), torch.from_numpy(attention_mask).to(device),
            torch.from_numpy(labels).to(device), torch.from_numpy(categories).to(device),
            torch.from_numpy(scope).to(device))


def _positive_rate_weight(category_pos_weight, categories, labels):
    """1.0 + (pos_weight_c - 1.0) * label -- recovers the classic
    BCEWithLogitsLoss(pos_weight=w) multiplier at label=1 (full pos_weight)
    and 1.0 at label=0, interpolating smoothly for LLM soft labels in
    between (same smooth-generalization approach focal_bce_per_sample's own
    pt already uses for continuous targets)."""
    cat_pos_w = category_pos_weight[categories]
    return 1.0 + (cat_pos_w - 1.0) * labels


def _smoke_test_train_step(model, optimizer, category_weight, category_pos_weight, device, pad_id,
                           vocab_size, n_categories, rank):
    _log(rank, "SMOKE", f"warm-up TRAIN step (batch={PER_CORE_TRAIN_BATCH} seq={PAIR_MAX_LENGTH}) "
                        f"-- forcing the real forward+backward compile now, before any real data "
                        f"or wall-clock budget is spent", force=True)
    t0 = time.perf_counter()
    input_ids, attention_mask, labels, categories, scope = _make_synthetic_batch(
        PER_CORE_TRAIN_BATCH, PAIR_MAX_LENGTH, pad_id, vocab_size, n_categories, device)
    loss = None
    try:
        optimizer.zero_grad(set_to_none=True)
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        per_sample = focal_bce_per_sample(logits, labels, FOCAL_GAMMA)
        w_cat = category_weight[categories]
        w_human = torch.where(scope == 0, torch.full_like(per_sample, HUMAN_WEIGHT),
                              torch.ones_like(per_sample))
        w_pos = _positive_rate_weight(category_pos_weight, categories, labels)
        w = w_cat * w_human * w_pos
        loss = (per_sample * w).sum() / w.sum()
        loss.backward()
        xm.mark_step()  # force compile+execution NOW, at the real memory cost
    except Exception:
        _log(rank, "SMOKE", "FAILED during warm-up TRAIN step -- this shape does not fit in HBM. "
                            "Lower PER_CORE_TRAIN_BATCH (or PAIR_MAX_LENGTH) and relaunch. Nothing "
                            "real was touched past this point.", force=True)
        raise
    finally:
        # backward() only populates .grad, never mutates a parameter's .data;
        # xm.optimizer_step was never called, so the model's real pretrained
        # weights are bit-for-bit unchanged, and Adafactor's own per-tensor
        # state (only created on its first real .step()) was never touched.
        optimizer.zero_grad(set_to_none=True)
    _log(rank, "SMOKE", f"warm-up TRAIN step OK ({time.perf_counter() - t0:.1f}s, "
                        f"loss={loss.item():.4f}, discarded -- model weights unchanged)", force=True)


def _smoke_test_real_loader_batch(model, optimizer, category_weight, category_pos_weight, device,
                                  caches, pad_id, train_frame, world_size, rank):
    """Closes a real, documented gap in the two synthetic smoke tests above:
    the FIFTH REAL CRASH's own docstring section notes the synthetic train
    smoke test did NOT reproduce the actual failure -- it succeeded on a
    hand-built batch, then the identical-looking model call crashed on the
    very first REAL MpDeviceLoader-sourced batch immediately after,
    "some difference between a hand-built synthetic batch and an
    MpDeviceLoader-sourced one" never fully explained. Whatever that
    difference is, THIS check goes through the exact same
    make_train_loader -> MpDeviceLoader -> next(iter(...)) path real
    training uses, at the real batch/seq shape, so it would have caught
    that specific crash before a single real training step -- instead of
    ~40 minutes in, as actually happened once already.

    Uses a THROWAWAY loader (epoch=0, never reused) so this cannot disturb
    the real epoch-1 loader's own sampler state -- constructing a second,
    independent loader is cheap; getting sampler/epoch bookkeeping subtly
    wrong to avoid it is not worth the risk."""
    _log(rank, "SMOKE", "warm-up REAL-LOADER batch -- same make_train_loader -> "
                        "MpDeviceLoader path real training uses, not a synthetic batch "
                        "(see docstring's FIFTH REAL CRASH: the synthetic smoke test alone "
                        "did not catch that one)", force=True)
    t0 = time.perf_counter()
    throwaway_loader, _throwaway_sampler = make_train_loader(
        train_frame, caches, pad_id, PER_CORE_TRAIN_BATCH, world_size, rank, SEED, epoch=0)
    mp_throwaway = pl_xla.MpDeviceLoader(throwaway_loader, device)
    loss = None
    try:
        input_ids, attention_mask, batch_labels, batch_cats, batch_scope = next(iter(mp_throwaway))
        batch_labels = batch_labels.to(device)
        batch_cats_t = torch.as_tensor(batch_cats, device=device)
        batch_scope = batch_scope.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        per_sample = focal_bce_per_sample(logits, batch_labels, FOCAL_GAMMA)
        w_cat = category_weight[batch_cats_t]
        w_human = torch.where(batch_scope == 0, torch.full_like(per_sample, HUMAN_WEIGHT),
                              torch.ones_like(per_sample))
        w_pos = _positive_rate_weight(category_pos_weight, batch_cats_t, batch_labels)
        w = w_cat * w_human * w_pos
        loss = (per_sample * w).sum() / w.sum()
        loss.backward()
        xm.mark_step()
    except Exception:
        _log(rank, "SMOKE", "FAILED during warm-up REAL-LOADER batch -- this is the exact "
                            "path/shape real training uses, so this failure WOULD have "
                            "recurred mid-run. Nothing beyond this point was touched.",
             force=True)
        raise
    finally:
        # Same guarantee as the synthetic smoke test: backward() only
        # populates .grad, xm.optimizer_step was never called, so real
        # pretrained weights are bit-for-bit unchanged.
        optimizer.zero_grad(set_to_none=True)
        del throwaway_loader, mp_throwaway
        gc.collect()
    _log(rank, "SMOKE", f"warm-up REAL-LOADER batch OK ({time.perf_counter() - t0:.1f}s, "
                        f"loss={loss.item():.4f}, discarded -- model weights unchanged)",
         force=True)


def _smoke_test_eval_step(model, device, pad_id, vocab_size, n_categories, eval_batch, rank):
    _log(rank, "SMOKE", f"warm-up EVAL step (batch={eval_batch} seq={PAIR_MAX_LENGTH})", force=True)
    t0 = time.perf_counter()
    input_ids, attention_mask, labels, categories, scope = _make_synthetic_batch(
        eval_batch, PAIR_MAX_LENGTH, pad_id, vocab_size, n_categories, device)
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


# =====================================================================
# CHECKPOINT -- own directory per save; val score explicit null on the
# quiet weights-only saves (see docstring's CHECKPOINT CONTRACT; cadence is
# CHECKPOINT_EVERY_SECONDS, not literally hourly in this CONTINUE lineage)
# =====================================================================

def _to_cpu_state(obj):
    """Recursively detaches+CPU-moves every torch.Tensor inside a nested
    state_dict-like structure, leaving everything else (ints, floats,
    strings, None) untouched. Needed for optimizer/scheduler state:
    Adafactor's own state_dict() nests real tensors inside
    {"state": {idx: {...}}, "param_groups": [...]} -- not a flat dict of
    tensors like a plain nn.Module.state_dict(), so the simple dict
    comprehension encoder.pt/head.pt use below doesn't reach them."""
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
                    val_macro_pr_auc, label, rank, llm_val_macro_pr_auc=None,
                    val_per_category=None, llm_val_per_category=None):
    """val_per_category/llm_val_per_category: the SAME dict already printed
    to the log as "[METRICS] per_category: {...}" -- added 2026-08-23 per
    direct instruction, because that dict previously only ever existed in
    the log (train_state.json carried just the macro mean), so recovering
    a real per-category breakdown for an already-shipped checkpoint meant
    re-parsing raw kernel logs by hand. None on a quiet save (no eval ran)
    and on the (rare) llm_val-only-failed branch -- explicit None, same
    "unvalidated" convention every other optional field here already uses.

    epoch_complete distinguishes a QUIET save (False -- mid-epoch safety
    net, no eval) from a FULL/validated save (True -- real eval ran, this
    epoch's own training is actually done) -- _find_resume_source's caller
    uses this to decide whether a resumed run should move to the NEXT epoch
    or restart (and skip forward within) the SAME one. scheduler.pt +
    lr_scale together make every save a fully self-sufficient resume point:
    the exact LR-schedule position is recoverable without re-deriving it
    from global_step and guessing which lr_scale produced it.

    REAL BUG (2026-08-23, ovosh789/continue-d, confirmed against a live
    output): optimizer.pt and scheduler.pt used to be written via
    xm.save(...) -- but save_checkpoint's own `if rank != 0: return` guard
    above means ONLY rank 0 ever reaches that call; xm.save is designed for
    exactly this rank-0-only-writes pattern for XLA-tensor-containing data,
    but empirically the files it "wrote" never actually existed in the
    kernel's real /kaggle/working output (encoder.pt/head.pt, saved the
    line before via plain torch.save on a manually .detach().cpu()'d dict,
    DID exist and downloaded fine) -- confirmed by directly downloading
    ovosh789's own real output with `kaggle kernels output --file-pattern`:
    head.pt downloads, optimizer.pt/scheduler.pt do not, despite the log
    printing a clean "[SAVE] last: ..." line implying success every time.
    Root cause not fully isolated (xm.save's own rank-0/collective
    semantics are opaque from here), but the fix doesn't need to be: switch
    to the SAME pattern that already, verifiably works for encoder.pt/
    head.pt -- manual CPU conversion (_to_cpu_state, extended to handle the
    NESTED tensor structure an optimizer/scheduler state_dict has, unlike a
    flat nn.Module one) + plain torch.save. This silently broke EVERY
    same-experiment resume across all 5 continue experiments until now --
    see the resume-flow's own BUG note in _mp_fn for the compounding
    silent-wrong-LR consequence this had."""
    if rank != 0:
        return
    save_dir = os.path.join(OUTPUT_DIR, label)
    xm.mark_step()
    os.makedirs(save_dir, exist_ok=True)

    torch.save({k: v.detach().cpu() for k, v in model.encoder.state_dict().items()},
              os.path.join(save_dir, "encoder.pt"))
    torch.save({k: v.detach().cpu() for k, v in model.classifier.state_dict().items()},
              os.path.join(save_dir, "head.pt"))
    torch.save(_to_cpu_state(optimizer.state_dict()), os.path.join(save_dir, "optimizer.pt"))
    torch.save(_to_cpu_state(scheduler.state_dict()), os.path.join(save_dir, "scheduler.pt"))

    with open(os.path.join(save_dir, "train_state.json"), "w") as f:
        json.dump({
            "epoch": epoch, "epoch_complete": epoch_complete, "global_step": global_step,
            "lr_scale": lr_scale,
            "val_macro_pr_auc": val_macro_pr_auc,           # explicit null if unvalidated
            "val_per_category": val_per_category,           # same eval's per-category breakdown --
                                                             # explicit null if unvalidated (no eval ran)
            "llm_val_macro_pr_auc": llm_val_macro_pr_auc,   # explicit null if unvalidated/failed --
                                                             # a SECOND, independent metric, never a
                                                             # substitute for val_macro_pr_auc above
            "llm_val_per_category": llm_val_per_category,   # per-category breakdown for the llm_val
                                                             # metric above -- explicit null if that
                                                             # metric itself is null
            "head": {"type": "FullSftHead", "layer_sizes": list(HEAD_HIDDEN),
                     "dropout": HEAD_DROPOUT},
            "pair_max_length": PAIR_MAX_LENGTH,
            "split": "item-disjoint 70/30 seed42 (human) + no-leak retroactive split (llm)",
            # Stamped so _find_resume_source (and a human) can tell "my own
            # prior session" from "a different experiment's checkpoint" --
            # absent on the ORIGINAL frida-full-sft-w/best (predates this),
            # which is exactly what makes every new experiment correctly
            # treat it as a fresh (not same-experiment) resume source.
            "experiment": EXPERIMENT_NAME,
            "llm_train_subsample_size": LLM_TRAIN_SUBSAMPLE_SIZE,
            "human_weight": HUMAN_WEIGHT,
            "focal_gamma": FOCAL_GAMMA,
        }, f, indent=2)
    print(f"[SAVE] {label}: epoch={epoch} epoch_complete={epoch_complete} step={global_step} "
          f"lr_scale={lr_scale} val_macro_pr_auc={val_macro_pr_auc} "
          f"llm_val_macro_pr_auc={llm_val_macro_pr_auc}", flush=True)


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


def stratified_llm_sample(target, cat_id, n_total, seed):
    """Draws n_total rows from the llm-train pool, category-balanced (every
    category gets the same quota) and target-level-stratified within each
    category (matches_llm targets are 10 discrete k/9 vote-count levels, not
    continuous in practice) -- ported byte-for-byte from
    kaggle_kernels/frida-percat-lora/frida_percat_lora.py's own function of
    the same name (that project's own already-vetted category-balance
    rationale: the metric is a plain mean over categories, so each one is
    equally worth representing, not proportionally to its natural size).
    Only invoked when LLM_TRAIN_SUBSAMPLE_SIZE is an int -- the full-split
    path (None) skips this and uses every row split=='train' already marks,
    which is NOT internally rebalanced this way (see that split kernel's own
    v2 docstring: "train covers all non-val rows", not a stratified draw)."""
    rng = np.random.RandomState(seed)
    categories = np.unique(cat_id)
    n_per_category = n_total // len(categories)
    keep = []
    for c in categories:
        cat_rows = np.flatnonzero(cat_id == c)
        n_take = min(n_per_category, len(cat_rows))
        cat_targets = target[cat_rows]
        levels, level_counts = np.unique(cat_targets, return_counts=True)
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
        keep.append(np.concatenate(picked) if picked else np.array([], dtype=np.int64))
    return np.concatenate(keep)


def _find_resume_source():
    """Content-based search (kaggle-experiment-safety's own resume-path
    rule: by content, not dataset name) across /kaggle/input for any prior
    full-SFT checkpoint -- either THIS experiment's own earlier session
    (re-mounted after a wall-clock cutoff) or the shared starting point,
    maxamsecret/frida-full-sft-w's best/ (val=0.7582, test=0.481). Picks
    whichever has trained the FURTHEST (highest global_step): a same-
    experiment continuation and a fresh cross-experiment start from the
    shared checkpoint are both just "the most-trained candidate available",
    so one rule covers both cases without needing to special-case them."""
    if not os.path.isdir("/kaggle/input"):
        return None, None
    candidates = []
    for root, _dirs, files in os.walk("/kaggle/input"):
        if {"encoder.pt", "head.pt", "train_state.json"} <= set(files):
            with open(os.path.join(root, "train_state.json")) as f:
                state = json.load(f)
            candidates.append((root, state))
    if not candidates:
        return None, None
    candidates.sort(key=lambda rs: rs[1].get("global_step", 0), reverse=True)
    return candidates[0]


def load_full_model_from_checkpoint(model, optimizer, checkpoint_dir, state, rank):
    """Loads encoder+head weights (and, best-effort, optimizer state) from
    an external full-SFT checkpoint IN PLACE. Same paranoia as frida-percat-
    lora's load_specialist_from_checkpoint_ -- a silent no-op or corrupted
    load would otherwise train a fresh-init model under a "resumed" label
    and look completely normal in the logs:
      1. probe one real weight before/after, refuse a byte-identical (no-op)
         load.
      2. strict key-set check on the encoder state_dict (a full model has no
         legitimate "missing key" case the way a partial LoRA adapter does).
      3. finiteness check on every loaded param -- refuse a corrupted/NaN
         source before a single training step touches it.
    Optimizer-state load is best-effort and non-fatal: this run builds a
    FRESH LR schedule regardless (see LR_RESUME_MULTIPLIER's own comment),
    so a missing/incompatible optimizer.pt only costs Adafactor's warm
    per-tensor statistics, never correctness."""
    encoder_path = os.path.join(checkpoint_dir, "encoder.pt")
    head_path = os.path.join(checkpoint_dir, "head.pt")

    # model.encoder IS a T5EncoderModel instance (see build_fresh_model /
    # FridaFullSftClassifier), whose OWN internal submodule is itself named
    # "encoder" (a T5Stack) -- so its param names already carry an
    # "encoder." prefix, same as save_checkpoint's own model.encoder.
    # state_dict() call saves them.
    probe_name = "encoder.block.0.layer.0.SelfAttention.q.weight"
    before = dict(model.encoder.named_parameters())[probe_name].detach().clone().float()

    encoder_sd = torch.load(encoder_path, map_location="cpu")
    missing, unexpected = model.encoder.load_state_dict(encoder_sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"[RESUME] encoder state_dict mismatch loading {encoder_path} -- "
                           f"missing={missing} unexpected={unexpected} (a full model has no "
                           f"legitimate partial-load case)")
    del encoder_sd

    head_sd = torch.load(head_path, map_location="cpu")
    model.classifier.load_state_dict(head_sd, strict=True)
    del head_sd

    after = dict(model.encoder.named_parameters())[probe_name].detach().float()
    delta = (after - before).abs().max().item()
    if delta == 0.0:
        raise RuntimeError(f"[RESUME] load was a no-op: {probe_name} byte-identical before/after "
                           f"loading from {checkpoint_dir}")

    bad = [n for n, p in model.encoder.named_parameters() if not torch.isfinite(p).all()]
    bad += [n for n, p in model.classifier.named_parameters() if not torch.isfinite(p).all()]
    if bad:
        raise RuntimeError(f"[RESUME] checkpoint at {checkpoint_dir} has non-finite values in "
                           f"{len(bad)} param(s) (e.g. {bad[0]}) -- refusing to resume from a "
                           f"corrupted checkpoint")

    optimizer_path = os.path.join(checkpoint_dir, "optimizer.pt")
    optimizer_loaded = False
    if os.path.isfile(optimizer_path):
        try:
            optimizer.load_state_dict(torch.load(optimizer_path, map_location="cpu"))
            optimizer_loaded = True
        except Exception as e:
            _log(0, "WARN", f"[RESUME] optimizer state at {optimizer_path} did not load "
                            f"({type(e).__name__}: {e}) -- continuing with fresh Adafactor "
                            f"state (does not affect model weights, already loaded and "
                            f"verified above)", force=True)

    same_experiment = state.get("experiment") == EXPERIMENT_NAME
    _log(rank, "RESUME", f"loaded {checkpoint_dir} (experiment={state.get('experiment')!r}, "
                         f"{'SAME experiment -- own prior session' if same_experiment else 'DIFFERENT experiment -- fresh start from its weights'}, "
                         f"val_macro_pr_auc={state.get('val_macro_pr_auc')}, "
                         f"global_step={state.get('global_step')}, probe delta={delta:.2e}, "
                         f"optimizer_loaded={optimizer_loaded})", force=True)
    return delta


def _load_shared_data():
    print(f"[{_elapsed():8.1f}s][SETUP] loading all CPU-only data once, pre-fork ...", flush=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    human_cache = load_v1_cache(HUMAN_CACHE_HINT, "human")
    llm_cache = load_v1_cache(LLM_CACHE_HINT, "llm")
    if human_cache["pad_id"] != llm_cache["pad_id"]:
        raise RuntimeError(f"[SETUP] pad_id differs between caches: "
                           f"{human_cache['pad_id']} vs {llm_cache['pad_id']}")
    caches = [human_cache, llm_cache]
    pad_id = human_cache["pad_id"]

    # ---- item-disjoint 70/30 split (from-scratch training only -- see the
    # split script's own scope note) ----
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
    print(f"[{_elapsed():8.1f}s][SETUP] human(item-disjoint) train={len(human_train):,} "
          f"val={len(human_val):,} val_fingerprint={val_fingerprint} categories={len(category_vocab)}",
          flush=True)

    human_train_frame = _build_frame(human_train["id1"], human_train["id2"], human_train["label"],
                                     human_train["cat_id"], 0, caches)
    val_frame = _build_frame(human_val["id1"], human_val["id2"], human_val["label"],
                             human_val["cat_id"], 0, caches)
    del split_df, human_val
    gc.collect()

    # ---- LLM train/val: pre-built, no-leak split (frida-llm-train-val-split
    # kernel) -- NOT the full 11.19M-row set, and NOT re-derived inline
    # anymore. See docstring's "LLM DATA" section + LLM_SPLIT_PATH_HINT's
    # own comment for why this replaced an earlier inline version. ----
    llm_split_path = _find_dataset_file(LLM_SPLIT_PATH_HINT, "llm_train_val_split.parquet")
    llm_split_df = pl.read_parquet(llm_split_path).to_pandas()
    llm_split_df["cat_id"] = llm_split_df["category"].map(category_to_id).astype(np.int64)
    llm_train = llm_split_df[llm_split_df["split"] == "train"].reset_index(drop=True)
    llm_val = llm_split_df[llm_split_df["split"] == "val"].reset_index(drop=True)
    n_llm_total = len(llm_split_df)
    llm_val_fingerprint = hashlib.sha256(
        ",".join(f"{a}-{b}" for a, b in sorted(zip(llm_val["id1"], llm_val["id2"]))).encode("utf-8")
    ).hexdigest()[:16]
    print(f"[{_elapsed():8.1f}s][SETUP] llm(no-leak split) train={len(llm_train):,} "
          f"val={len(llm_val):,} val_fingerprint={llm_val_fingerprint} "
          f"-- NOT the full 11,187,780-row matches_llm.parquet", flush=True)
    del llm_split_df
    gc.collect()

    if LLM_TRAIN_SUBSAMPLE_SIZE is not None:
        n_before = len(llm_train)
        sel = stratified_llm_sample(llm_train["target"].to_numpy(),
                                    llm_train["cat_id"].to_numpy(), LLM_TRAIN_SUBSAMPLE_SIZE, SEED)
        llm_train = llm_train.iloc[sel].reset_index(drop=True)
        print(f"[{_elapsed():8.1f}s][SETUP] llm-train subsampled (category-balanced, "
              f"target-level-stratified, seed={SEED}): {n_before:,} -> {len(llm_train):,} "
              f"(requested {LLM_TRAIN_SUBSAMPLE_SIZE:,})", flush=True)

    # LLM train keeps the CONTINUOUS soft target (never thresholded) -- same
    # treatment this project has always given LLM rows in the loss. LLM val
    # uses the binarized label: average_precision_score needs a real 0/1
    # ground truth, same as the human val frame below.
    llm_train_frame = _build_frame(llm_train["id1"], llm_train["id2"], llm_train["target"],
                                   llm_train["cat_id"].to_numpy().astype(np.int64), 1, caches)
    llm_val_frame = _build_frame(llm_val["id1"], llm_val["id2"], llm_val["label"],
                                 llm_val["cat_id"].to_numpy().astype(np.int64), 1, caches)
    del llm_train, llm_val
    gc.collect()

    n_llm_train = len(llm_train_frame["label"])
    train_frame = _concat_frames(human_train_frame, llm_train_frame)
    print(f"[{_elapsed():8.1f}s][SETUP] mixed train={len(train_frame['label']):,} "
          f"(human {len(human_train_frame['label']):,} + llm_train {n_llm_train:,})",
          flush=True)
    del human_train_frame, llm_train_frame
    gc.collect()

    # ---- realized per-category counts on the ACTUAL training mix, for the
    # category weight in the loss (see docstring's LOSS section on why this
    # matters here and did not for frida-percat-lora) ----
    cat_counts = np.bincount(train_frame["category"], minlength=len(category_vocab)).astype(np.float64)
    cat_counts = np.maximum(cat_counts, 1.0)
    category_weight = (cat_counts.sum() / len(category_vocab)) / cat_counts
    print(f"[{_elapsed():8.1f}s][SETUP] category weights (train-mix inverse-frequency): "
          f"min={category_weight.min():.3f} max={category_weight.max():.3f} "
          f"({category_vocab[int(np.argmin(cat_counts))]!r} largest, "
          f"{category_vocab[int(np.argmax(cat_counts))]!r} smallest)", flush=True)

    # In-category positive-rate weighting (see the config block's own
    # comment for why this is separate from category_weight above): the
    # realized positive MASS per category on the actual training mix --
    # train_frame["label"] already mixes hard human 0/1 with continuous LLM
    # soft targets, and pos_mass sums both the same way pos_weight
    # computations elsewhere in this project already do (e.g. frida-percat-
    # lora's own per-category pos_weight).
    pos_mass = np.bincount(train_frame["category"], weights=train_frame["label"].astype(np.float64),
                           minlength=len(category_vocab))
    category_pos_weight = (cat_counts - pos_mass) / np.maximum(pos_mass, 1e-6)
    print(f"[{_elapsed():8.1f}s][SETUP] category pos-weights (in-category 0/1 balance): "
          f"min={category_pos_weight.min():.3f} max={category_pos_weight.max():.3f} "
          f"({category_vocab[int(np.argmin(category_pos_weight))]!r} highest positive rate, "
          f"{category_vocab[int(np.argmax(category_pos_weight))]!r} lowest)", flush=True)

    _SHARED.update(dict(
        caches=caches, pad_id=pad_id, train_frame=train_frame,
        val_frame=val_frame, llm_val_frame=llm_val_frame,
        category_vocab=category_vocab, category_weight=category_weight,
        category_pos_weight=category_pos_weight,
        val_fingerprint=val_fingerprint, llm_val_fingerprint=llm_val_fingerprint,
        n_llm_total=int(n_llm_total), n_llm_train=int(n_llm_train),
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
    llm_val_frame = _SHARED["llm_val_frame"]
    category_vocab = _SHARED["category_vocab"]
    category_weight = torch.tensor(_SHARED["category_weight"], dtype=torch.float32, device=device)
    category_pos_weight = torch.tensor(_SHARED["category_pos_weight"], dtype=torch.float32, device=device)

    _log(rank, "CONFIG", f"experiment={EXPERIMENT_NAME!r} world_size={world_size} "
                         f"pair_max_length={PAIR_MAX_LENGTH} lr={LR} "
                         f"head_lr_peak={LR * HEAD_LR_PEAK_MULTIPLIER} "
                         f"warmup={WARMUP_RATIO} human_weight={HUMAN_WEIGHT} "
                         f"focal_gamma={FOCAL_GAMMA} epochs={EPOCHS} "
                         f"llm_train_subsample_size={LLM_TRAIN_SUBSAMPLE_SIZE} "
                         f"llm_train={_SHARED['n_llm_train']:,} llm_val={len(llm_val_frame['label']):,} "
                         f"(of {_SHARED['n_llm_total']:,} in the no-leak split)", force=True)

    model, hidden_dim = build_fresh_model(device, rank)
    total_batch = PER_CORE_TRAIN_BATCH * world_size
    _log(rank, "BATCH", f"per_core_train={PER_CORE_TRAIN_BATCH} per_core_eval={PER_CORE_EVAL_BATCH} "
                        f"total_train_batch={total_batch} (scaled from a real compile-time OOM at "
                        f"batch=48, see module docstring's PER_CORE_TRAIN_BATCH comment -- verified "
                        f"below by the smoke test before any real data is touched)", force=True)

    # RESUME: content-based search (see _find_resume_source's own docstring)
    # -- covers both "warm-start THIS experiment from the shared
    # frida-full-sft-w/best checkpoint (or a DIFFERENT sibling experiment's
    # output)" and "continue THIS experiment's OWN prior session after a
    # wall-clock cutoff", but the two cases are handled differently now:
    # cross-experiment gets a fresh schedule at LR_RESUME_MULTIPLIER (as
    # before); same-experiment gets an EXACT resume -- same lr_scale, same
    # optimizer param_groups/state, same scheduler position, same
    # global_step -- because c/d/e's own epochs (5-17.5h) vastly outlast one
    # 9h session, so "continue THIS run" is the common case for them, not an
    # edge case.
    resume_dir, resume_state = _find_resume_source()
    same_experiment = resume_dir is not None and resume_state.get("experiment") == EXPERIMENT_NAME
    if resume_dir is not None and same_experiment:
        lr_scale = resume_state.get("lr_scale", LR_RESUME_MULTIPLIER)
        start_step = int(resume_state.get("global_step", 0))
        start_epoch = int(resume_state.get("epoch", 1))
        epoch_already_complete = bool(resume_state.get("epoch_complete", False))
    elif resume_dir is not None:
        lr_scale = LR_RESUME_MULTIPLIER
        start_step, start_epoch, epoch_already_complete = 0, 1, False
    else:
        lr_scale = 1.0
        start_step, start_epoch, epoch_already_complete = 0, 1, False

    # BUG FOUND 2026-08-23 (real run, ovosh789/continue-d): optimizer is now
    # ALWAYS built with the correct lr_scale BEFORE any load attempt below --
    # NOT rebuilt conditionally afterward, which is what this file used to
    # do. The old order (build unscaled -> maybe load optimizer.pt -> maybe
    # rebuild scaled) silently trained at the WRONG lr_scale whenever
    # optimizer.pt was missing on a same-experiment resume: same_experiment
    # skipped the rebuild-with-lr_scale branch entirely (it only ran for
    # cross-experiment), trusting optimizer.state_dict() to already carry
    # the right param_groups -- but a real run's own optimizer.pt turned
    # out to never actually exist (see save_checkpoint's own BUG note on
    # xm.save), so that trust was misplaced: training would have silently
    # continued at lr_scale=1.0 (the constructor default) instead of the
    # resumed run's own 0.5, with no error anywhere. Building at the
    # correct scale FIRST makes a missing/failed optimizer-state load fall
    # back to something still correct, not silently wrong; a real
    # optimizer.pt, when present, still fully overwrites these param_groups
    # via load_state_dict() below, so nothing is lost when resume works as
    # designed.
    optimizer = Adafactor(make_param_groups(model, lr_scale=lr_scale), scale_parameter=False,
                          relative_step=False, warmup_init=False, weight_decay=WEIGHT_DECAY)
    if resume_dir is not None:
        load_full_model_from_checkpoint(model, optimizer, resume_dir, resume_state, rank)
        if resume_state.get("val_macro_pr_auc") is None:
            _log(rank, "WARN", f"[RESUME] source checkpoint has val_macro_pr_auc=null (unvalidated) "
                               f"-- starting quality unknown, but safe to continue training from "
                               f"(see kaggle-experiment-safety on unvalidated checkpoints: this "
                               f"caveat is about treating it as a RESULT, not about training "
                               f"further from your own safety-net save)", force=True)
    else:
        _log(rank, "INIT", "no resume checkpoint found under /kaggle/input -- training from the "
                           "pretrained FRIDA base (fresh full-FT, matches the ORIGINAL "
                           "frida-full-sft run's own starting point)", force=True)
    _log(rank, "LR", f"lr_scale={lr_scale} same_experiment_resume={same_experiment} "
                    f"start_step={start_step} start_epoch={start_epoch} "
                    f"epoch_already_complete={epoch_already_complete} -> "
                    f"backbone_peak={LR * lr_scale:.2e} "
                    f"head_peak={LR * HEAD_LR_PEAK_MULTIPLIER * lr_scale:.2e}", force=True)

    n_train = len(train_frame["label"])
    steps_per_epoch = max(1, n_train // total_batch)
    epoch_steps = steps_per_epoch * EPOCHS
    projected_steps = int(MAX_TRAIN_SECONDS * PROJECTED_ROWS_PER_SEC / total_batch)
    total_steps = max(1, min(epoch_steps, projected_steps))
    warmup_steps = max(1, int(total_steps * WARMUP_RATIO))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, [
        _backbone_lr_lambda(warmup_steps, total_steps, LR_MIN_RATIO),
        _head_lr_lambda(warmup_steps, total_steps, LR_MIN_RATIO),
    ])
    if same_experiment:
        scheduler_path = os.path.join(resume_dir, "scheduler.pt")
        if os.path.isfile(scheduler_path):
            scheduler.load_state_dict(torch.load(scheduler_path, map_location="cpu"))
            _log(rank, "RESUME", f"scheduler restored at internal step {scheduler.last_epoch} "
                                 f"(global_step={start_step})", force=True)
        else:
            _log(rank, "WARN", f"[RESUME] same-experiment checkpoint at {resume_dir} has no "
                               f"scheduler.pt -- LR schedule restarting from step 0 despite "
                               f"resumed weights/optimizer (older checkpoint format?)", force=True)
    _log(rank, "TRAIN", f"n_train={n_train:,} steps/epoch={steps_per_epoch} "
                        f"epoch_steps={epoch_steps} projected_reachable={projected_steps} "
                        f"-> schedule_steps={total_steps} "
                        f"({100 * total_steps / max(epoch_steps, 1):.1f}% of {EPOCHS} epochs) "
                        f"warmup={warmup_steps}", force=True)

    # Early smoke test (see docstring's "EARLY SMOKE TEST") -- one real
    # forward+backward and one real forward-only pass at the exact real
    # shapes, forced to compile+execute NOW via xm.mark_step(), before any
    # real data is loaded. Turns an HBM-fit problem into a ~1-2 minute
    # failure instead of the hours-into-the-run one this kernel actually hit.
    vocab_size = model.encoder.config.vocab_size
    n_categories = len(category_vocab)
    _smoke_test_train_step(model, optimizer, category_weight, category_pos_weight, device, pad_id,
                           vocab_size, n_categories, rank)
    _smoke_test_eval_step(model, device, pad_id, vocab_size, n_categories,
                         PER_CORE_EVAL_BATCH, rank)
    # See docstring's FIFTH REAL CRASH -- the two synthetic checks above did
    # not catch that one; this one goes through the real data-loading path
    # and would have.
    _smoke_test_real_loader_batch(model, optimizer, category_weight, category_pos_weight, device,
                                  caches, pad_id, train_frame, world_size, rank)

    # Sharded across all `world_size` cores + all_gather'd inside evaluate()
    # -- see docstring's "VALIDATION REWRITE". One fixed padded shape per
    # frame, no ragged remainder loader.
    val_loader, val_valid_mask = make_eval_loader_sharded(val_frame, caches, pad_id,
                                                          PER_CORE_EVAL_BATCH, world_size, rank)
    # "Использовать llm-val не вместо human-val, а параллельно" -- a second,
    # independent loader/eval, never a replacement for the human one above.
    llm_val_loader, llm_val_valid_mask = make_eval_loader_sharded(
        llm_val_frame, caches, pad_id, PER_CORE_EVAL_BATCH, world_size, rank)

    global_step = start_step
    best_val = -1.0
    train_start = time.perf_counter()
    last_checkpoint_at = 0.0
    stopped = False
    consecutive_step_failures = 0

    def do_quiet_save(reason, epoch_num):
        try:
            save_checkpoint(model, optimizer, scheduler, global_step, epoch_num, False, lr_scale,
                            None, "last", rank)
            _log(rank, "SAVE", f"quiet save ({reason}) epoch={epoch_num} step={global_step} "
                              f"(weights+optimizer+scheduler, no eval -- see docstring's "
                              f"CHECKPOINT CONTRACT -- own directory, self-overwriting 'last')",
                 force=True)
        except Exception as e:
            _log(rank, "ERROR", f"quiet save FAILED ({reason}): {type(e).__name__}: {e}", force=True)

    def do_full_checkpoint(epoch_num, label):
        nonlocal best_val
        try:
            # evaluate() calls xm.all_gather -- a COLLECTIVE, so every rank
            # must reach it. This is safe under the SAME conditions the
            # early smoke test already validated (identical model, identical
            # shapes, identical synthetic-proven memory budget on every
            # rank) -- a genuine OOM here would hit all 8 ranks in lock-step,
            # not just one, so there is no new partial-failure/hang mode
            # beyond what the smoke test already ruled out.
            m = evaluate(model, val_loader, val_valid_mask, device, category_vocab, world_size)
            improved = m["macro_pr_auc"] > best_val
            if improved:
                best_val = m["macro_pr_auc"]
            _log(rank, "METRICS", f"{label}: epoch={epoch_num} step={global_step} "
                                  f"val_macro_pr_auc={m['macro_pr_auc']:.4f} best={best_val:.4f} "
                                  f"n_val={m['n']}", force=True)
            if rank == 0:
                print(f"[METRICS] per_category: {json.dumps(m['per_category'], ensure_ascii=False)}",
                      flush=True)

            # Second, independent eval -- parallel to human val, never a
            # substitute for it (per direct instruction). Its own failure
            # must not take down the human-val checkpoint above, so it gets
            # its own try/except and a None fallback rather than sharing the
            # outer one. (Same collective caveat as above applies here too.)
            llm_m = None
            try:
                llm_m = evaluate(model, llm_val_loader, llm_val_valid_mask, device,
                                 category_vocab, world_size)
                _log(rank, "METRICS", f"{label}: llm_val_macro_pr_auc={llm_m['macro_pr_auc']:.4f} "
                                      f"n_llm_val={llm_m['n']}", force=True)
                if rank == 0:
                    print(f"[METRICS] llm_val_per_category: "
                          f"{json.dumps(llm_m['per_category'], ensure_ascii=False)}", flush=True)
            except Exception as e:
                _log(rank, "ERROR", f"llm_val eval FAILED at epoch={epoch_num} step={global_step} "
                                    f"(human val above is unaffected): {type(e).__name__}: {e}",
                     force=True)
                model.train()

            # epoch_complete=True -- a real eval ran, this epoch's training
            # is genuinely done. Own directory per label (checkpoint_epochNN
            # and, separately, best), never overwritten -- unlike "last"
            # above, both persist across every future save this run makes.
            save_checkpoint(model, optimizer, scheduler, global_step, epoch_num, True, lr_scale,
                            m["macro_pr_auc"], label, rank,
                            llm_val_macro_pr_auc=llm_m["macro_pr_auc"] if llm_m else None,
                            val_per_category=m["per_category"],
                            llm_val_per_category=llm_m["per_category"] if llm_m else None)
            if improved:
                save_checkpoint(model, optimizer, scheduler, global_step, epoch_num, True, lr_scale,
                                m["macro_pr_auc"], "best", rank,
                                llm_val_macro_pr_auc=llm_m["macro_pr_auc"] if llm_m else None,
                                val_per_category=m["per_category"],
                                llm_val_per_category=llm_m["per_category"] if llm_m else None)
            return True
        except Exception as e:
            _log(rank, "ERROR", f"eval/checkpoint FAILED at epoch={epoch_num} step={global_step}: "
                                f"{type(e).__name__}: {e}", force=True)
            if rank == 0:
                import traceback
                traceback.print_exc()
            model.train()
            optimizer.zero_grad(set_to_none=True)
            gc.collect()
            do_quiet_save("eval failed, falling back", epoch_num)
            return False

    # Same-experiment resume into a COMPLETED epoch moves to the next one;
    # into an INCOMPLETE one (a quiet save landed mid-epoch) restarts that
    # SAME epoch number -- its DistributedSampler shuffle is seeded by
    # `epoch` alone (see make_train_loader), so re-requesting the same
    # epoch number reproduces the IDENTICAL row order byte-for-byte, which
    # is what makes the batch-skip below exact rather than approximate.
    resume_start_epoch = start_epoch + 1 if epoch_already_complete else start_epoch

    for epoch in range(resume_start_epoch, EPOCHS + 1):
        if stopped:
            break
        train_loader, sampler = make_train_loader(train_frame, caches, pad_id,
                                                  PER_CORE_TRAIN_BATCH, world_size, rank, SEED,
                                                  epoch=epoch)
        mp_loader = pl_xla.MpDeviceLoader(train_loader, device)

        # Resuming PARTWAY through THIS exact epoch: global_step already
        # counts every step from prior, fully-finished epochs plus some N
        # steps into this one -- skip exactly those N batches from the
        # freshly-rebuilt (but seed-identical) loader before touching a
        # single real training step, so no row is silently skipped or
        # double-trained. Zero for a from-scratch run, a cross-experiment
        # resume, or a same-experiment resume that landed exactly on an
        # epoch boundary.
        batch_iter = iter(mp_loader)
        skip_batches = 0
        if epoch == resume_start_epoch and same_experiment and not epoch_already_complete:
            skip_batches = max(0, global_step - (epoch - 1) * steps_per_epoch)
        _log(rank, "EPOCH", f"starting epoch {epoch}/{EPOCHS}"
                            + (f" (resuming mid-epoch: skipping {skip_batches:,} already-trained "
                               f"batches to reach step {global_step})" if skip_batches else ""),
             force=True)
        if skip_batches:
            skip_t0 = time.perf_counter()
            for _ in range(skip_batches):
                next(batch_iter)
            _log(rank, "EPOCH", f"batch-skip done ({time.perf_counter() - skip_t0:.1f}s)", force=True)

        for batch in batch_iter:
            input_ids, attention_mask, batch_labels, batch_cats, batch_scope = batch
            batch_labels = batch_labels.to(device)
            batch_cats_t = torch.as_tensor(batch_cats, device=device)
            batch_scope = batch_scope.to(device)
            try:
                optimizer.zero_grad()
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
                per_sample = focal_bce_per_sample(logits, batch_labels, FOCAL_GAMMA)
                w_cat = category_weight[batch_cats_t]
                w_human = torch.where(batch_scope == 0,
                                      torch.full_like(per_sample, HUMAN_WEIGHT),
                                      torch.ones_like(per_sample))
                w_pos = _positive_rate_weight(category_pos_weight, batch_cats_t, batch_labels)
                w = w_cat * w_human * w_pos
                loss = (per_sample * w).sum() / w.sum()
                loss.backward()
                xm.optimizer_step(optimizer)
                scheduler.step()
                global_step += 1
                consecutive_step_failures = 0
            except Exception as e:
                consecutive_step_failures += 1
                # rank==0 only (not force=True): a permanently-repeating
                # failure used to log identically from every surviving rank,
                # on every single step -- see MAX_CONSECUTIVE_STEP_FAILURES'
                # own comment for the real incident this caused (thousands
                # of duplicate lines evicting the actually-useful log
                # content). The circuit breaker below is what actually
                # bounds this; rank-0-only just keeps a RECOVERABLE failure
                # from being needlessly 8x-amplified in the meantime.
                _log(rank, "WARN", f"step failed and was skipped "
                                   f"({consecutive_step_failures}/{MAX_CONSECUTIVE_STEP_FAILURES} "
                                   f"consecutive): {type(e).__name__}: {e}")
                try:
                    del logits, per_sample, loss
                except NameError:
                    pass
                optimizer.zero_grad(set_to_none=True)
                gc.collect()
                if consecutive_step_failures >= MAX_CONSECUTIVE_STEP_FAILURES:
                    raise RuntimeError(
                        f"[rank{rank}] {consecutive_step_failures} consecutive training steps "
                        f"failed with the same class of error ({type(e).__name__}: {e}) -- this is "
                        f"not a transient/recoverable failure, stopping instead of spinning for the "
                        f"rest of the session's wall-clock budget. See docstring's THIRD REAL CRASH."
                    ) from e
                continue

            if global_step % LOG_EVERY_STEPS == 0:
                xm.mark_step()
                if rank == 0:
                    et = time.perf_counter() - train_start
                    print(f"[{_elapsed():8.1f}s] epoch={epoch} step={global_step}/{total_steps} "
                          f"loss={loss.item():.4f} lr_backbone={scheduler.get_last_lr()[0]:.2e} "
                          f"lr_head={scheduler.get_last_lr()[1]:.2e} elapsed_train={et:.0f}s", flush=True)

            elapsed_train = time.perf_counter() - train_start
            if (elapsed_train - last_checkpoint_at) >= CHECKPOINT_EVERY_SECONDS:
                do_quiet_save(f"{CHECKPOINT_EVERY_SECONDS}s wall-clock cadence", epoch)
                last_checkpoint_at = elapsed_train
                model.train()

            if elapsed_train > MAX_TRAIN_SECONDS:
                _log(rank, "TRAIN", f"wall-clock cutoff ({elapsed_train:.0f}s > {MAX_TRAIN_SECONDS}s) "
                                    f"-- stopping mid-epoch.", force=True)
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
