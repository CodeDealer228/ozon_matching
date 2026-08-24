---
name: kaggle-experiment-safety
description: "Rules for spending Kaggle GPU/TPU quota safely and efficiently in this project: session-cap budgeting, checkpoint hygiene, calls that silently destroy a training run, and a pre-push checklist for kernels. Use this BEFORE writing, editing, pushing, or relaunching any Kaggle kernel that trains, tokenizes, embeds, or otherwise burns quota-hours — and also when planning epochs/batch size/checkpoint cadence, diagnosing a kernel that crashed, was cancelled, or hit the session cap, deciding whether a checkpoint's reported metric can be trusted, or choosing which of the three project accounts to run on. Every rule here is derived from a real incident in this repo, so consult it even when the change looks like a small edit to an already-working kernel."
allowed-tools: Bash Read Grep Glob Write Edit
---

# Kaggle experiment safety

The scarce resource in this project is not compute — it is **quota-hours you
cannot get back**. GPU is 30h/week with a **hard 12h session cap**; TPU is a
separate 20h/week pool with a **9h cap**. A kernel that dies at minute 3 costs
almost nothing. A kernel that dies at hour 10, or finishes but produces weights
whose quality nobody can measure, costs a week of budget and an experiment
you then cannot compare against anything.

So the question to ask before every push is not "will this work?" but **"if
this goes wrong at hour 9, what do I still have?"** Everything below exists to
make that answer be "a checkpoint I can resume and a number I can trust."

Read `references/incidents.md` for the full catalogue of what actually went
wrong here and what each failure cost — useful when you need to justify a
rule, recognize a repeat of an old failure, or check whether something you're
about to do has already been tried. Read `references/environment.md` for
platform/image/CLI quirks (paths, preinstalled packages, dataset publishing,
accounts).

## Budget backwards from the session cap

Sessions do not extend. A run is cancelled mid-step at the cap, so treat the
cap as a hard deadline and design the schedule to land *inside* it:

- **Estimate total wall-clock before pushing**, and include the non-training
  phases. The single worst incident in this project lost a 12h GPU session to
  *tokenization*, not training: 9.54h of it (~80%) went into a per-chunk rejoin
  against the items table, while training itself would have needed ~3.5h. If a
  phase is not the experiment, it should not be in the session.
- **Precompute expensive setup once, as its own artifact.** Item tokenization
  now lives in dedicated cache kernels publishing `ids.bin`/`input_ids.bin`
  memmaps as datasets; training does a pure id lookup. Every training run since
  then starts at second zero of real work.
- **Push CPU-only work onto a CPU kernel** — it does not consume the GPU pool
  at all. Push RAM-hungry CPU work onto TPU when it needs the headroom: the TPU
  pool is separate, so it does not compete with training.
- **Set an explicit wall-clock stop** below the cap (e.g. `MAX_TRAIN_SECONDS`
  at 8.5h for a 9h TPU cap) so the run ends on *your* terms, flushing a final
  checkpoint, rather than being killed between a forward and a backward pass.
- **Plan an epoch count you can actually reach.** A 20-epoch plan that reaches
  22% is not a 20-epoch experiment; it is an unlabelled fraction of one. Either
  size the plan to the cap, or make every checkpoint independently usable so a
  partial run is still a result.
- **Anchor safety saves to wall-clock, not epoch progress.** Epoch boundaries
  can be hours apart, and the cap does not wait for one.

## The checkpoint contract

Kaggle persists `/kaggle/working` as kernel output **even when a session is
hard-capped or cancelled** — `kaggle kernels output <owner>/<slug>` recovers it.
That is the property everything below leans on, and it is why a cancelled run
can still be a new project best (one was: 0.53229, recovered from a session
killed at ~69% of its epoch).

The **kernel log does not enjoy the same guarantee.** One cancelled run came
back with a 0-byte log, so 10.5h of training produced weights with no record of
what happened. Two consequences:

- Write every result to a **JSON file in `/kaggle/working`** as each phase
  completes. Never let stdout be the only copy of a number you care about.
  Kaggle also truncates long logs to the *tail*, so early `[LOAD]`/`[DATA]`
  lines get evicted by thousands of late warnings even on a clean finish.
- Anything the log would have told you (config, phase timings, metric history)
  belongs in that JSON too.

Checkpoint hygiene, as adopted after a provenance trace found a checkpoint
whose reported score belonged to *different weights*:

- **Every checkpoint gets its own directory.** Never overwrite. Disk is cheap;
  a lost intermediate is not.
- **The metric written next to weights must describe those weights.** A running
  "best so far" copied into a new checkpoint's `train_state.json` is a
  fabrication — it happened twice here, and the second time it made 10.5h of
  training unusable until a separate validation kernel could re-measure it.
- **When a save skips eval, write the metric as an explicit `null`**, not a
  stale value and not an omitted key. A `null` is honest and machine-checkable;
  a missing key gets silently read as "unknown, probably fine."
- **Save optimizer and scheduler state**, or the checkpoint is a dead end
  rather than a resume point.
- **Resume by searching `/kaggle/input` for content** (`adapter_config.json`,
  `train_state.json`) rather than by dataset name, so relocating a checkpoint
  only requires editing `dataset_sources`.

Corollary for decisions: **an unvalidated checkpoint is not a result.** Don't
resume from one, don't submit it, don't compare it — run the cheap validation
kernel first. A 9h session is far too expensive to spend on weights of unknown
quality.

## Calls that silently destroy a run

These share a signature: no exception, no crash, just hours of quota spent on
something that was never going to work. That is what makes them worth a
checklist — a loud failure would have caught itself.

- **`PeftModel.from_pretrained(base, dir)` without `is_trainable=True`.**
  Adapters save with `inference_mode: true`, so every LoRA matrix loads with
  `requires_grad=False`. Training runs happily and updates only the head.
- **`torch.autocast` opened around a `DataParallel` call.** The context does not
  propagate into the per-replica `forward()` threads (documented PyTorch
  behavior). Result: NaN loss from a specific layer onward. Open autocast
  *inside* the model's own `forward()`.
- **`torch.cuda.is_bf16_supported()` as a dtype guard.** It returns `True` on a
  T4, which has no bf16 tensor cores — the run then executes *emulated* bf16 at
  ~5x the cost. Gate on `torch.cuda.get_device_capability()[0] >= 8`.
- **Auto-probing batch size on TPU.** Every probed candidate compiles its own
  XLA program and those programs **stay resident in HBM** (unlike a CUDA
  allocation, which frees on failure), so the probe self-inflicts the OOM it
  was meant to prevent — and `xm.get_memory_info()` cannot see program memory,
  so it reports a flat, useless number throughout. Three separate incidents
  established this. On TPU, use a **fixed constant derived by arithmetic** from
  a known-good data point at another size/model, documented as an estimate, and
  deliberately conservative. A genuine OOM then fails at compile time on the
  first batch — fast, loud, and cheap. GPU probing is fine and still used.
- **Ragged final batches on TPU.** XLA compiles a fresh graph per distinct input
  shape. Use `drop_last=True` on *every* loader, train and eval, so exactly two
  shapes are ever compiled.
- **`optimizer.step()` on TPU** instead of `xm.optimizer_step(optimizer)`, which
  does the cross-core all-reduce and the lazy-graph sync together.
- **Doing CPU setup after `torch_xla.launch()`.** All 8 forked replicas then
  load the same multi-GB data independently and exhaust host RAM — a real
  `BrokenProcessPool` crash here. Load once in `main()` before the fork; the
  replicas share it copy-on-write.
- **Eval that can crash training.** Wrap metric computation in `try/except` and
  fall back to a weights-only quiet save. Losing one measurement is a nuisance;
  losing the session is not.
- **A polars join with `maintain_order` against a multi-million-row table.**
  One real run went from 9.7GB to 31.1GB RSS on a single 20K-row chunk and was
  OOM-killed. Use a plain hash join plus an explicit `.sort(row_index)` — which
  is a stronger correctness guarantee anyway.
- **Assuming a Linux OOM-kill is catchable.** It is not a `MemoryError` under
  default overcommit (unlike CUDA OOM), so adaptive chunk sizing only reacts
  *after* a chunk survives. The starting chunk size still has to be safe on its
  own.
- **Decoding a full column before filtering.** `items.parquet` at 13M rows OOMs
  a naive `.select().with_columns()` even with `engine="streaming"`, because
  truncation happens after `attributes` is decoded. Semi-join down to the ids
  you need *first*.
- **Raising a token budget without raising the character budget.** A larger
  `max_length` buys nothing if the upstream character slice still caps the raw
  text — the extra capacity becomes pure padding. Scale `MAX_*_CHARS` by the
  same ratio, and confirm any preexisting token cache actually matches the new
  budget.
- **CatBoost `init_model` continuation on `task_type="GPU"`.** Unsupported;
  crashes after the first stage. Fit once to the full iteration count, then
  `model.copy().shrink(ntree_end=N)` for intermediate checkpoints.

## Fail cheap: put the fragile step first

A recurring, entirely avoidable pattern here is **doing hours of work and then
dying on the last line**: an 83-minute tokenization that failed at
`kaggle datasets create` because the CLI binary was not on PATH; another that
failed because `dataset-metadata.json` lacked a `licenses` key; a third that
finished loading 12.38M items and failed a disk-space precheck.

So: **validate the teardown path before doing the work.** At kernel start,
assert the output binaries exist, the metadata is well-formed, the disk math
holds, and the model/tokenizer actually loads. A kernel that dies in the first
minute has cost nothing.

The same logic favours **preflight kernels** over guessing inside a long run —
this project has used a minutes-long batch-size probe kernel and a cheap CPU
kernel that checked whether a cached prefix was really what the metadata
claimed (it was not). Both paid for themselves immediately.

And before relaunching work to (re)produce anything: **check whether it already
exists.** One kernel was queued to re-tokenize 12.38M items into a dataset a
human had already published in parallel; it was caught while still `QUEUED`, so
it cost nothing, but only by luck. This matters most when someone else is
acting on the same resources at the same time.

## Pre-push checklist

Run through this before `kaggle kernels push`. It is ordered by what the
mistake costs.

1. **Account** — which of the three quotas is this spending? Confirm rather
   than infer, and remember non-default accounts need env-var overrides, and
   any kernel needing Kaggle creds at runtime needs per-kernel secrets set in
   the web UI (not configurable from `kernel-metadata.json`).
2. **`kernel-metadata.json`** — `id` owner matches that account;
   `enable_gpu`/`enable_tpu` set correctly; **`machine_shape` present for GPU
   kernels** (omitting it once got a GPU whose kernels the preinstalled torch
   build had never been compiled for, killing the run at the first CUDA op);
   every needed input listed in `dataset_sources`, pointing at the dataset that
   actually holds the checkpoint you intend to resume.
3. **Time budget** — estimated wall-clock for *every* phase, summed, under the
   cap, with an explicit in-script stop below it.
4. **Checkpoints** — cadence anchored to wall-clock; own directory per save;
   optimizer/scheduler state included; metric explicitly `null` when eval was
   skipped; results mirrored to JSON in `/kaggle/working`.
5. **Resume path** — the checkpoint you resume from has a *measured* score, and
   the code finds it by content, not by dataset name.
6. **The silent killers** — walk the list above for the ones this kernel can
   hit (`is_trainable`, autocast placement, TPU probing/`drop_last`, pre-fork
   data loading, join memory, char-vs-token budget).
7. **Image assumptions** — the packages you import are actually present on
   *this* image (GPU and TPU images differ), pinned to exact versions, with
   `pip install` placed before the module-level import that needs it.
8. **Byte-match** — if the run trains anything that will later be submitted, the
   text/tokenization contract matches what inference will do, and matches what
   the resumed checkpoint already saw.

## Mandatory post-push monitoring

Creating or pushing a Kaggle kernel/image includes monitoring it; a successful
CLI push is not completion of the task.

- During the first hour, poll every **5–10 minutes** until the kernel has
  remained `RUNNING` for at least **10 continuous minutes**. Report and inspect
  any `ERROR` immediately; do not wait for the next scheduled checkpoint.
- Once startup is stable, poll at approximately **20%, 40%, 60%, 80%, and
  100%** of the expected total wall-clock. Calculate these checkpoints from
  the pre-push runtime estimate and the observed start time.
- A terminal state always ends the schedule early. On `COMPLETE`, verify the
  expected outputs; on `ERROR`/`CANCEL`, collect the log and diagnose it.
- Do not silently abandon an active monitor when the interactive turn ends;
  use the available recurring monitor/automation mechanism when the product
  provides one, otherwise state explicitly who or what continues polling.

This monitoring cadence is the project contract for every account and every
accelerator type, including CPU-only builders. It supersedes a single
post-push status check.
