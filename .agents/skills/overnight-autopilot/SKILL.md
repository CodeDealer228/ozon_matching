---
name: overnight-autopilot
description: Fully autonomous overnight execution mode for long-running engineering, ML, Kaggle, research, debugging, training, data-processing, and deployment tasks. Use only when the user explicitly invokes this skill or says the task should run autonomously overnight without waiting for them.
---

# Overnight Autopilot

## Purpose

This skill means the user is unavailable and has explicitly delegated routine operational decisions for the task. The agent must drive the task to a useful terminal state without putting the user into the execution loop.

The operating principle is:

> Do not stop at a decision point that can be resolved safely by analysis, experimentation, rollback, retry, an alternative implementation, or additional use of already-available compute/quota.

A partial attempt is not completion. Continue through diagnosis, repair, rerun, validation, and artifact preservation until the requested outcome is achieved or a genuine external blocker remains that cannot be bypassed.

## User authorization in this mode

Within the scope of the task, treat the following as pre-authorized and do not ask the user to confirm them:

- choosing reasonable hyperparameters, thresholds, seeds, batch sizes, learning rates, schedulers, regularization, optimizer settings, sampling rates, evaluation cadence, and other experiment details;
- running ablations or small pilot experiments to resolve uncertainty;
- creating, updating, versioning, or publishing Kaggle datasets required by the task;
- creating, saving, submitting, restarting, or replacing Kaggle notebooks/kernels required by the task;
- consuming additional already-available Kaggle/Colab/cloud compute quota when it materially increases the chance of finishing;
- retrying failed jobs;
- changing implementation details when the original route is broken;
- installing ordinary dependencies in the task environment;
- creating temporary files, caches, checkpoints, logs, diagnostics, intermediate datasets, and backup copies;
- using a slower or more expensive-in-compute fallback implementation when the preferred implementation fails;
- selecting among technically reasonable alternatives without asking the user;
- fixing obvious bugs discovered while executing the task;
- adding diagnostics or instrumentation needed to debug failures;
- resuming from checkpoints or restarting from an earlier stable state;
- making reversible edits to project files needed to complete the task.

This authorization covers consumption of resources already available to the user/project. It does **not** authorize purchasing new paid services, increasing billing limits, entering payment information, disclosing secrets, weakening security controls, or publishing credentials/private data.

## Core autonomy rules

1. **Never wait for routine confirmation.**
   Do not ask questions such as:
   - “Should I publish the dataset?”
   - “Which learning rate should I choose?”
   - “Should I retry?”
   - “Should I use another implementation?”
   - “Should I restart the kernel?”
   - “Should I run an ablation first?”

   Decide and proceed.

2. **Resolve ambiguity experimentally.**
   If several choices are plausible, prefer the fastest experiment that distinguishes them. Use pilot runs, small subsets, dry runs, reduced epochs, synthetic checks, unit tests, or local reproductions before committing large compute.

3. **Prefer progress over elegance.**
   A robust workaround is better than an elegant dead end. If the clean route is blocked, use another library, implementation, machine type, serialization format, launcher, dataset version, checkpoint strategy, or execution path.

4. **Failures trigger a debugging loop, not a stop.**
   On failure:
   - capture the exact error and relevant logs;
   - identify the most likely root causes;
   - add diagnostics when evidence is insufficient;
   - implement the highest-probability fix;
   - rerun the smallest useful reproduction;
   - if fixed, resume the full task;
   - if not fixed, try the next hypothesis;
   - continue until success or a genuine non-bypassable external blocker is proven.

5. **Do not repeatedly ask the same system to do the same failing thing.**
   After repeated failure, change the approach. Examples: different package version, different optimizer implementation, different accelerator strategy, smaller batch, gradient accumulation, alternative data format, different checkpoint, different Kaggle dataset version, local reproduction, CPU/GPU fallback, or a reduced validation experiment.

6. **Optimize for waking up to a finished result.**
   Spend available compute/quota when the expected value is positive. Avoid conserving quota at the cost of leaving an easily solvable task unfinished.

## Long-running jobs

A long-running job includes training, Kaggle kernels, large preprocessing, indexing, downloads/uploads, hyperparameter sweeps, long builds, large evaluations, and similar work that may take tens of minutes or hours.

### Monitoring cadence

When a long-running job starts or restarts:

1. Perform an immediate sanity check after launch:
   - job accepted/running;
   - expected accelerator is visible;
   - input paths and datasets are mounted;
   - logs begin normally;
   - no immediate OOM, dependency, permission, shape, NaN, deadlock, or configuration error.

2. During the initial stabilization phase, check approximately every **10 minutes**.

3. Remain on the 10-minute cadence until either:
   - at least **3 consecutive checks** show healthy forward progress; or
   - the job has run healthily for roughly **30 minutes** and there is no evidence of instability.

4. After stabilization, if the job is expected to continue for a long time, check approximately **once per hour**.

5. Return immediately to the 10-minute cadence after any of the following:
   - restart;
   - code/config change;
   - checkpoint recovery;
   - warning suggesting instability;
   - stalled progress;
   - loss/metric divergence;
   - resource-pressure change;
   - worker/process failure.

6. When the job finishes, inspect final logs and outputs immediately. Do not treat “process exited successfully” as sufficient validation.

### What each health check should verify

Check whichever signals are available:

- job state: queued/running/completed/failed/cancelled;
- step/epoch/global-step advancement;
- wall-clock progress;
- training and validation loss;
- target metric and auxiliary metrics;
- NaN/Inf occurrence;
- learning rate and scheduler state;
- GPU/TPU/CPU utilization where observable;
- memory/HBM/RAM/disk usage;
- dataloader throughput;
- checkpoint creation;
- worker restarts or collective-communication errors;
- suspiciously frozen logs;
- output artifacts and their sizes/timestamps.

If a metric temporarily worsens but the run is technically healthy, do not restart reflexively. Use trend, validation evidence, and prior baseline to decide.

## Hyperparameter decisions

Do not ask the user to choose ordinary hyperparameters.

Use this hierarchy:

1. Start from the strongest known baseline in the project.
2. Preserve pretrained/finetuned weights when the task is incremental.
3. Change as few variables as necessary.
4. For uncertain high-impact settings, run a short pilot/ablation.
5. Prefer conservative settings for already-good models unless evidence supports aggressive changes.
6. Record every nontrivial choice and its rationale in the final report.

For ML experiments, always define before the main run:

- primary metric;
- comparison baseline;
- checkpoint-selection rule;
- early-failure criteria;
- random seed(s) when meaningful;
- maximum resource envelope available in the existing environment;
- resume/retry strategy.

## Kaggle-specific behavior

When Kaggle is involved, the agent should autonomously handle routine Kaggle operations needed for completion.

### Dataset/version handling

- Create or publish a dataset/version when the kernel requires it.
- Do not wait for user confirmation to publish a task-specific dataset that contains only material already intended for the task.
- Verify the resulting dataset/version is accessible from the kernel before launching expensive compute.
- If a version is broken, create a corrected version and repoint the kernel.
- Keep names/version notes sufficiently descriptive to identify what was used.

Never publish secrets, API keys, credentials, private unrelated files, or data that is not clearly part of the delegated task.

### Kernel/notebook handling

Before an expensive run, validate as much as possible with a cheap smoke test:

- imports;
- data paths;
- accelerator initialization;
- model/checkpoint loading;
- one batch forward pass;
- loss computation;
- backward/optimizer step when training;
- checkpoint save when relevant.

For TPU/distributed runs, additionally verify:

- expected world size/device count;
- all ranks enter training;
- no rank-dependent shape/count mismatch;
- sampler behavior is intentional;
- collective operations do not deadlock;
- checkpointing is rank-safe;
- metric aggregation is correct.

If the kernel fails, inspect its logs, patch the cause, create whatever dataset/kernel version is needed, and relaunch without waiting for the user.

## Resource policy

Use resources deliberately but do not be timid.

Allowed without asking:

- extra retries;
- extra Kaggle accelerator sessions within available quota;
- additional temporary storage;
- extra diagnostic/pilot runs;
- redundant checkpoint copies when useful;
- a second implementation used to validate or bypass the first.

Prefer cheaper validation before expensive full runs, but once the route is validated, execute the full task rather than leaving only a recommendation.

Do not intentionally burn quota with no expected diagnostic or task value.

## Decision policy under uncertainty

When a decision is needed, classify it:

### A. Reversible + task-local

Examples: hyperparameters, retry, code patch, dataset version, kernel restart, temporary file, library choice.

**Action:** decide immediately and proceed.

### B. Expensive in existing quota but reversible

Examples: another multi-hour training run, another TPU session, an ablation sweep.

**Action:** estimate expected information/value. If it materially increases completion probability or final quality, proceed without asking.

### C. Potentially destructive or externally consequential

Examples: deleting irreplaceable data, force-pushing shared history, overwriting the only checkpoint, changing production data, making something public that may contain private information.

**Action:** first create a reversible backup or choose a non-destructive alternative. If the requested task explicitly requires the action and data-safety/privacy checks pass, proceed. Otherwise avoid unnecessary irreversible changes.

## Anti-stall rules

The following are not acceptable terminal states when a viable next action exists:

- “I found the problem.”
- “Here is what you should try tomorrow.”
- “The kernel failed; please rerun it.”
- “I need you to choose between A and B.”
- “Please publish this dataset.”
- “Please confirm I can restart.”
- “The run may take several hours, so I stopped here.”

Instead, execute the next action yourself.

## Iterative execution loop

Use this loop until the terminal condition is met:

1. Inspect current state.
2. Define the next measurable sub-goal.
3. Execute it.
4. Validate the result.
5. If successful, advance.
6. If unsuccessful, collect evidence and diagnose.
7. Patch, bypass, or change approach.
8. Retry.
9. Preserve useful artifacts/checkpoints/logs.
10. Repeat.

Never confuse activity with progress: each loop should either advance the task or reduce uncertainty.

## Validation requirements

Do not declare success merely because code ran.

Before declaring completion, verify task-specific acceptance criteria. Examples:

- training: expected checkpoint exists, is loadable, metric is computed, no silent NaNs, and the intended data/model configuration was actually used;
- data pipeline: expected row/file counts, schema, null rates, samples, and output integrity are checked;
- Kaggle: kernel status is successful and the required output/dataset version exists and is accessible;
- code: tests/smoke tests pass and the modified path is exercised;
- deployment: health/readiness endpoint or equivalent functional check succeeds;
- research: claims are supported by sources and actionable conclusion is produced.

If the requested goal includes improving a metric, compare against the stated or recovered baseline rather than reporting the new metric in isolation.

## Checkpointing and recoverability

For long or fragile tasks:

- save intermediate outputs frequently enough to avoid losing hours of work;
- preserve the last known-good checkpoint before risky changes;
- include configuration with checkpoints when possible;
- retain logs needed to diagnose failure;
- prefer resumable workflows;
- after every successful major stage, make the next stage able to start without repeating all previous work.

## Final report

When the task reaches a terminal state, leave a concise but complete report containing:

1. **Outcome** — what was completed and whether the original goal was achieved.
2. **Best result** — primary metric/output and comparison with baseline when applicable.
3. **What ran** — important experiments/jobs and which one produced the final result.
4. **Autonomous decisions** — significant hyperparameters, workarounds, retries, or architecture changes chosen without user input.
5. **Artifacts** — exact paths, dataset versions, kernel/notebook versions, checkpoints, logs, submissions, or generated files.
6. **Validation** — evidence that the result is actually usable.
7. **Failures encountered** — only material failures and how they were fixed.
8. **Remaining blocker** — include this only if completion was genuinely impossible; state exactly why and the minimal user action needed.

Do not end with routine questions such as “What would you like me to do next?” when operating under this skill.

## Terminal condition

Stop only when one of these is true:

### Success

The requested task is completed, validated, and the useful outputs are preserved.

### Proven external blocker

Completion requires an unavailable external capability or user-only action that cannot be bypassed legitimately after reasonable attempts. In this case:

- preserve all current progress;
- leave the environment in the most resumable state possible;
- document the blocker precisely;
- specify one minimal next action for the user.

Anything else means: continue working.
