# Incident catalogue — what actually went wrong, and what it cost

Every rule in `SKILL.md` traces back to something in here. Consult this file
when you need the exact numbers behind a rule, when a new failure looks like it
might be a repeat, or before proposing something that has already been tried
and ruled out.

Deeper narrative for the older entries lives in `reports/experiment_log.md`;
per-kernel detail lives in each kernel's own module docstring, which in this
project is where root causes are recorded.

## Contents

- [Sessions lost to the time cap](#sessions-lost-to-the-time-cap)
- [Work lost at the last step](#work-lost-at-the-last-step)
- [Results that could not be trusted](#results-that-could-not-be-trusted)
- [Silent failures — no error, wrong training](#silent-failures--no-error-wrong-training)
- [Memory: host RAM, VRAM, HBM](#memory-host-ram-vram-hbm)
- [Library and API traps](#library-and-api-traps)

---

## Sessions lost to the time cap

**Tokenization ate 80% of a 12h GPU session.**
`lora-llm-continue-train`, full 11.48M-row run. Phase timings from its own logs:
items semi-join 28.5s; val tokenize (73,131 rows) 173s; **train tokenize
(11,480,303 rows) 34,336s ≈ 9.54 hours**; model load + probe ~21s; training
~2.1h for 60% of the epoch (~3.5h projected for the full epoch). Root cause:
`tokenize_to_memmap` rejoined the 13M-row items table on *every* chunk, paying
the hash/probe cost hundreds of times instead of once. The session ended
`CANCEL_ACKNOWLEDGED` at ~69% of a single epoch. Training alone would have fit
comfortably.

Why it still was not a total loss: `/kaggle/working` persisted through the
hard-cap cancellation, and `kaggle kernels output` recovered a checkpoint at
val 0.53229 — a new project best at the time. That recoverability is the whole
reason the checkpoint contract is written the way it is.

**Multi-epoch plans that reached a fraction of themselves.**
`berta-lora-sft-human-only-20-epochs` planned 20 epochs and hit the wall at
global_step 5016/22840 (**22%**), at val 0.7111 and still rising.
`frida-lora-sft-human-only-10-epochs` reached 60% of epoch 1;
`frida-lora-sft-epoch2` reached 40% of its epoch. None of these are failures —
each produced a usable checkpoint — but the *plan* in each case was fiction
from the moment it was pushed. Size plans to the cap, or make partial runs
independently meaningful.

**A cancelled run at 98.6% whose result is unknown.**
`frida-lora-v2-epoch` was cancelled at 10.5h, step 1127/1143. Its only
checkpoint was a wall-clock "quiet" save with no eval, its kernel log came back
**0 bytes** (lost on cancellation), and its `train_state.json` carried the
*inherited* 0.69377 running-best rather than a metric for its own weights. Net
result: 10.5 GPU-hours produced weights of unknown quality, requiring a
separate validation kernel (`frida-validate-checkpoint`) just to find out what
was bought. Published honestly as `...-v2-epoch-last` with a `PROVENANCE.md`
carrying this warning.

---

## Work lost at the last step

All three of these ran the expensive phase to completion and then died on a
line that could have been checked in the first second.

- **`llm` v4** tokenized all 12,384,610 items (~83 min) and failed at
  `subprocess.check_call(["kaggle", "datasets", "create", ...])` with
  `FileNotFoundError: 'kaggle'` — the CLI binary is not on PATH on the TPU
  image (it is on the CPU/GPU image). Fix: `pip install kaggle` in the preamble.
- **`human` v1** tokenized all 711,304 items (~4 min) and failed because
  `dataset-metadata.json` lacked a `licenses` key — `kaggle datasets create`
  calls `get_or_fail(meta_data, "licenses")`. Fix:
  `"licenses": [{"name": "CC0-1.0"}]`.
- **`llm` v3** loaded all 12.38M items (RSS 20.75GB, fine) and failed a
  disk-space precheck: `need 21.53GB, have 20.94GB`. The real write was only
  15.95GB from an *exactly known* byte count (fixed dtypes and shape); a
  blanket 35% safety margin alone pushed the requirement past Kaggle's ~20GB
  `/kaggle/working` ceiling. Fix: `margin=1.15` at that specific call site.

**Near-miss worth remembering:** `llm` v6 was queued to re-tokenize 12.38M
items into a dataset the user had *already* published manually in parallel; it
would have burned ~83 min of TPU and then failed at `datasets create` on an
existing slug. Caught while still `QUEUED`, so zero TPU time was actually lost
— by luck, not by process. Check the state of a resource before relaunching
work to produce it, especially when a human is acting on it concurrently.

**A crash 47 seconds in, which is the good outcome.** `frida-tpu-human-sft` v1
died at 47s on the tokenizer version mismatch below. Cost: nothing. This is
what "fail cheap" looks like in practice, and why the fragile step belongs
first.

---

## Results that could not be trusted

**The stale-metric trap, twice.** `berta_llm_tpu_sft`'s `save_checkpoint()`
wrote `best_val_macro_pr_auc` — a *running* best — into every checkpoint,
including saves that had run no eval. A provenance trace
(`reports/berta_tpu_lora_provenance/`) had to reconstruct by hash which weights
a submission actually contained. The same trap then reappeared in
`frida-lora-v2-epoch-last`. Hence: the metric next to weights must describe
those weights, and an unvalidated save writes an explicit `null`.

**Kaggle truncates logs to the tail.** In `giga3b-quantize-benchmark`,
thousands of bitsandbytes `MatMul8bitLt` warnings evicted every early
`[LOAD]`/`[DATA]` line. Combined with the 0-byte-log incident above: stdout is
not storage. Write each phase's results to JSON in `/kaggle/working`.

**Val does not predict the judging stand.** Known pairs, same seed-42 val split
where noted: MiniLM baseline 0.3710 val → 0.1333 LB; MiniLM LoRA 0.5282 →
0.265; FRIDA epoch 1 0.652 → 0.3459. But `berta_catboost_top_final` scored val
**0.7467** (better than the classifier it was built on) and test **0.457**
(worse than that classifier's 0.47); `berta_tpu_human_continue_384` epoch 1
scored the highest val in the project, 0.7551, and **0.43** on test. A val gain
is a hypothesis, not a result — which is an argument for spending a submission
slot (5/day cap) rather than more quota, once val stops discriminating.

---

## Silent failures — no error, wrong training

**LoRA that was never training.** `PeftModel.from_pretrained(base, dir)`
defaults to the adapter's on-disk `inference_mode: true`, loading every LoRA
matrix with `requires_grad=False`. No error is raised; the run trains only the
classifier head and burns full quota doing it. Always pass `is_trainable=True`
when resuming to continue training.

**NaN from a specific layer onward under fp16 (and bf16 under
`nn.DataParallel`).** Root-caused in `frida-nan-diagnose`: `torch.autocast`
opened in the calling thread does not propagate into `DataParallel`'s
per-replica `forward()` threads — documented PyTorch behavior, not a bug in
this code. Fix: open autocast inside the model's own `forward()`.

**Emulated bf16 at 5x the cost.** `torch.cuda.is_bf16_supported()` returns
`True` on a T4 (sm_75, no bf16 tensor cores) under torch 2.10, so the dtype
guard never fired and the "full precision" pass ran emulated: 1.97 items/s
versus int8's 10.10. Gate on `torch.cuda.get_device_capability()[0] >= 8`.

**A larger token budget that carried no extra content.** Raising
`ITEM_MAX_LENGTH` 256→320 while leaving `MAX_ATTR_CHARS=384` means the
tokenizer never sees enough raw text to fill the new budget — the added
capacity becomes padding. Caught during review before the kernel was pushed;
fixed by scaling `MAX_ATTR_CHARS` 384→480 by the same 1.25x ratio. Related:
`PAIR_MAX_LENGTH=128` was measured to truncate ~94% of real pairs for FRIDA,
which is why raising it was the single change worth making in a new run.

**A cache whose metadata lied.** `berta-llm-cache-probe` was a cheap CPU kernel
written purely to check whether the prefix recorded in a token cache's metadata
was really the prefix baked into the tokens. It was not, and the discrepancy
forced a re-tokenization. Verify claims about artifacts against the artifacts.

---

## Memory: host RAM, VRAM, HBM

**TPU probing causes the OOM it prevents.** Porting the GPU-style batch-size
probe to TPU failed three separate times. Each probed candidate compiles its
own XLA program, and those programs **stay resident in HBM** — unlike a CUDA
allocation, which frees cleanly on failure. Probing 5 train candidates left
~27GB of compiled programs on a 15.75GB/core device before eval probing even
began. Worse, `xm.get_memory_info()` cannot see program memory: it reported a
flat 0.27GB for every candidate, so the probe's own signal was meaningless.
TPU batch sizes are now fixed constants derived by arithmetic from a known-good
data point, documented as estimates, chosen conservatively.

**Real TPU sizing data points** (v5e-8, 15.75GB/core), useful for deriving new
estimates: BERTA (128.3M) fit `batch=64` at `PAIR_MAX_LENGTH=256`; at 384 the
same batch needed **19.11G of 15.75G** and OOM'd in epoch 2 after epoch 1 had
completed cleanly — halving to 32 absorbed the 1.5x sequence increase. FRIDA
(823M, ~6.4x the parameters) was estimated at ~21–26 by scaling that point and
deliberately set to **16** for its first-ever TPU run.

**8 replicas each loading the same 13GB.** `berta-llm-tpu-sft` crashed with
`BrokenProcessPool`: all 8 fork processes independently reloaded and
concatenated the item cache and recomputed `matches_llm.parquet`, exhausting
host RAM. Fix: move all CPU-only setup into `main()` *before*
`torch_xla.launch()`, so replicas share it copy-on-write.

**A join that tripled RSS in one chunk.** A polars join with
`maintain_order="left"` against a 3.45M-row items table took a single 20K-row
chunk from 9.7GB to 31.1GB/32.21GB and got OOM-killed. Plain hash join plus an
explicit `.sort(row_index)` fixed it — and is a stronger correctness guarantee,
since it does not depend on undocumented internal join ordering.

**Linux OOM-kills are not catchable.** Unlike CUDA OOM, a real OOM-kill under
default overcommit is not a `MemoryError`. RAM-adaptive chunk sizing can only
react after a chunk *completes*, so the starting size must be independently
safe.

**Full-column decode before filtering.** `items.parquet` at 13M rows OOMs a
naive `.select().with_columns()` even with `engine="streaming"`, because the
truncation happens after the whole `attributes` column is decoded. Semi-join
to the needed ids first.

---

## Library and API traps

**CatBoost cannot continue training on GPU.** `init_model` under
`task_type="GPU"` raises
`catboost/cuda/train_lib/train.cpp:323: Training continuation for GPU is not
yet supported` — after the first stage has already succeeded. Workaround: one
full `.fit()` to the total iteration count, then `model.copy()` +
`.shrink(ntree_end=N)` to produce checkpoints at intermediate tree counts.

**GBDT on raw embeddings overfits when rows are scarce.** `berta-catboost-top-final`
trained on 768 raw embedding dims + 3 features over 58,504 rows;
`get_feature_importance()` put **96.11%** on the embedding dims (flat, near
noise, max single dim 0.43%) while `classifier_score` — independently worth
0.47 on test by itself — got 1.9%. Val 0.7467, test 0.457. PCA(32) plus
stronger regularization raised `classifier_score` to 19.29% and test to 0.469,
closing most but not all of the gap. Feature importance is a cheap, effective
overfitting diagnostic here.

**`kaggle datasets version -r <mode>` fails with `Incompatible Dataset Type`**
against some existing datasets, in every mode tried (skip/zip/tar) — and
uploads real bandwidth each time before failing without committing. Create a
new dataset instead, and update `dataset_sources` to point at it. Because
resume code searches `/kaggle/input` by content rather than name, that is a
one-line change.

**Windows zip round-trips corrupt Cyrillic filenames.** Per-category artifacts
named with Cyrillic survived a local directory test and then `FileNotFoundError`'d
after the real zip round-trip. `str.isalnum()` is Unicode-aware, so a
"sanitizer" built on it passes Cyrillic through unchanged. Use ASCII index
names (`tfidf_cat00.joblib`). More generally: test the *extracted built
archive*, not the source directory — packaging bugs only appear in the archive
(the same test also caught a wrapping folder that would have put `run.py` one
level below the archive root).
