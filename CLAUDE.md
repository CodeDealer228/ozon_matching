# Ozon E-CUP 2026 — Product Matching (Entity Resolution)

`d:/Projects/kaggle/ozon_matching` is the project root.

## Conventions
- Think in English, but respond to me in Russian.
- Write README.md in Russian.

## Task

Binary classification of pre-retrieved product-pair candidates: given two Ozon
product cards (`id1`, `id2`), predict whether they're duplicates. Retrieval is
already done by organizers — this repo is only the pairwise re-ranking stage.
Container competition: submit a Docker image + code archive, judged offline
on hidden test data.

## Data

Kaggle dataset **`maxamsecret/ozon-ecup-2026-matching`** (private). Local
copies in `data/` (gitignored — see README.md for download commands).

| File | Size | Columns |
|---|---|---|
| `items.parquet` | 13M+ rows, 4.1GB | `id`, `name`, `category`, `attributes` (JSON string) |
| `items_human.parquet` | 214MB — subset covering every id in `matches.parquet` | same schema; use for local dev/testing |
| `matches.parquet` | 365,654 rows, 4MB | `id1`, `id2`, `target` (0/1, human-labeled) |
| `matches_llm.parquet` | 11,187,780 rows, 104MB | `id1`, `id2`, `target` (float 0..1, LLM soft labels). Fully disjoint item universe from `matches.parquet` (0 shared ids of 12.4M vs 711K) — used for a second-stage continued SFT, see below |

`category` has **20 distinct values** — the grouping key for the metric.

Kaggle kernel input mount has an extra path segment:
`/kaggle/input/datasets/<owner>/<slug>/...`, not `/kaggle/input/<slug>/...`.

## Metric

**Macro-averaged PR-AUC, per category.** For each of the 20 `category`
values, `sklearn.metrics.average_precision_score(y_true, y_prob)`, then plain
mean across categories. Never `sklearn.metrics.auc` (interpolated, inflates
the score). Also not the same as macro-averaging over match/non-match
*classes* (AP for positive + AP for flipped-negative, averaged) — that's a
different, non-comparable number. `compute_metrics(..., categories=...)` in
the training notebook does it correctly; any new eval code must group by
`category`, skip categories with only one class present, average the rest.

## Text preprocessing (must byte-match between training and inference)

```python
MAX_NAME_CHARS, MAX_CATEGORY_CHARS, MAX_ATTR_CHARS = 128, 64, 384
MAX_ITEM_CHARS = 128 + 64 + 384 + 64  # = 640

text = f"Name: {name[:128]} Category: {category[:64]} Attributes: {attributes[:384]}"[:640]
```

`attributes` is a **raw string slice**, not `json.loads`'d (parsing 13M+ JSON
blobs OOMs). Tokenization: `max_length=128`, `padding="max_length"`,
`truncation=True`, `text=text1, text_pair=text2`.
`submits/baseline` uses full `json.loads` instead — intentional mismatch for
the baseline comparison, but our `run.py` must use the formula above (what
the LoRA weights were trained on).

## Canonical train/val split

```python
SEED = 42
train_idx, val_idx = train_test_split(indices, test_size=0.20, random_state=SEED, stratify=labels_all)
```
Applied to `matches.parquet` after filtering null ids and casting
`label = (target >= 0.5)`, in that order. → 292,523 train / 73,131 val, all 20
categories present in val with both classes. Materialized copy:
`data/splits/matches_split_seed42.parquet` (regenerate via
`scripts/make_canonical_split.py`). Every eval run also logs a
`val_fingerprint` (sha256 of sorted `id1-id2` pairs) to mlflow so any two runs
can be proven to share the same val set.

## Architecture

- **Baseline**: frozen `cross-encoder/ms-marco-MiniLM-L-12-v2` → `[CLS]` →
  `StandardScaler` → `LogisticRegression(class_weight="balanced")`.
- **Ours**: same encoder, LoRA SFT (`r=16, alpha=32, dropout=0.10`,
  `target_modules=["query","key","value"]`) + `nn.Linear(hidden,1)` head on
  `[CLS]`, end-to-end `BCEWithLogitsLoss(pos_weight=...)`. Submission uses
  **merged** weights (`merge_and_unload()`) — no `peft` at inference.
- Kaggle datasets: `maxamsecret/ozon-lora-merged-minilm` (merged, ready for
  inference) and `maxamsecret/ozon-lora-minilm-finetuned` (PEFT adapter,
  pre-merge — needed to resume/continue training).

### Results

| Variant | val macro_pr_auc | leaderboard Mean PR-AUC |
|---|---|---|
| Baseline | 0.3710 | 0.1333 |
| **LoRA SFT** | **0.5282** | **0.265** |

Leaderboard score < val score for both (hidden-test items differ from
`matches.parquet`'s), but LoRA's relative edge over baseline held up: ~2x on
leaderboard vs. ~1.4x on val.

## In progress: continued LoRA SFT on `matches_llm.parquet`

Second SFT stage on top of the epoch-7 checkpoint, one epoch over the 11.19M
soft-labeled pairs (fed to `BCEWithLogitsLoss` as continuous targets, no
threshold), `LR=4e-4` cosine (vs. original `2e-4` linear — original run's own
`train_log.csv` hadn't clearly plateaued, just decayed LR). Goal: close the
val→leaderboard generalization gap via wider catalog coverage.

Code: `kaggle_kernels/lora-llm-continue-train/` (script kernel, run via
`kaggle kernels push -p kaggle_kernels/lora-llm-continue-train --accelerator
NvidiaTeslaT4`, monitor via `kaggle kernels status/logs
maxamsecret/lora-llm-continue-train`).

Key gotchas:
- Resuming a saved PEFT adapter needs `PeftModel.from_pretrained(base, dir,
  is_trainable=True)` — adapters default to `inference_mode: true` on disk;
  omit the flag and every LoRA matrix silently loads `requires_grad=False`
  (no error, just wastes GPU quota training only the classifier head).
- `items.parquet` at full 13M-row scale OOMs a naive `.select().with_columns()`
  polars scan even with `engine="streaming"` — the truncation happens *after*
  the full `attributes` column is decoded. Fix: semi-join down to the actual
  needed ids *before* the string/truncation ops. Keep tokenizer chunk size
  small (~8K, not 50K+) — a real Linux OOM-kill isn't a catchable
  `MemoryError`, unlike CUDA OOM.

Status as of this writing: 4 Kaggle smoke iterations on 2xT4, real mount
paths, real adapter (logs in `kaggle_kernels/lora-llm-continue-train/output/`).
Memory is resolved: v1/v2 got Linux-OOM-killed (full 13M-row `items.parquet`
load, then 50K-row tokenizer batches); fixed via a semi-join to actually-
needed ids + `ctypes.CDLL("libc.so.6").malloc_trim(0)` after each tokenize
chunk (glibc wasn't releasing freed pages back to the OS). v4
(`MAX_TRAIN_ROWS=500_000`, a real full train pass, not just a mechanics
check) stayed flat at RSS≈6.5GB/32GB for the entire run — confirmed safe to
scale up.

**v4's training result is a concern, not a green light.** Over that 500K-row
subsample — a full, self-contained cosine cycle (warmup → peak `4e-4` →
anneal to ~0 within its own 198 steps; NOT a preview of "the first 500K rows
of the real 11.19M-row run," where LR would still be near-peak at that
point, not annealed) — `val_macro_pr_auc` dropped from the starting 0.5282
to ~0.503 by the first checkpoint (10%) and stayed in the 0.503–0.509 range
for the rest of the run (best 0.50857 at 20%, final 0.50642 at 100%),
**never recovering toward 0.5282**. Still well above the 0.371 baseline, but
a real regression vs. the checkpoint this continues from, on real data with
a real (if compressed) schedule. Before spending hours of GPU quota on the
real full run: worth lowering `LR` (`4e-4` may be too high for this data) or
otherwise re-examining the setup — don't launch the real run on the current
config without addressing this signal first.

`MAX_TRAIN_ROWS` and `ENABLE_DATASET_PUSH=False` are still set for
smoke-testing in the shipped script.

**Update:** the real full run happened. Best checkpoint **0.53229** at 60%
of the single epoch over all 11.48M rows — a new project best, above the
0.5282 starting point — before hitting the hard 12h Kaggle session limit
(`CANCEL_ACKNOWLEDGED` at ~69%). Not yet published as a Kaggle dataset; see
Open TODOs.

## In progress: FRIDA / BERTA encoder exploration

Replacing the MiniLM (33M) cross-encoder with something bigger. Two
parallel LoRA SFT tracks, human-only (`matches.parquet`) so far, both still
mid-training (neither has finished a full pass, let alone converged) — see
`kaggle_kernels/README.md` "Разведка новых энкодеров" for the full
per-checkpoint numbers.

- **FRIDA** (`ai-forever/FRIDA`, T5EncoderModel, 823M) — LoRA
  `target_modules=["q","k","v","o","wi_0","wi_1","wo"]` (T5's real module
  names, not BERT's), CLS pooling, item-level pre-tokenized cache
  (`maxamsecret/frida-item-tokens-human`, sidesteps the per-chunk
  pair↔items rejoin that cost continue-train ~9.5h/12h). `PAIR_MAX_LENGTH=128`
  (cost/quota choice, not the 384 near-lossless budget). Two sessions so
  far, each hit the 12h Kaggle wall before finishing one epoch: epoch 1
  (`kaggle_kernels/frida-lora-sft-human-only-10-epochs`, account
  `maxamsecret`) reached val macro PR-AUC **0.652** @ 60% of 1 epoch;
  epoch 2 (`kaggle_kernels/frida-lora-sft-epoch2`, account `rays0n`,
  resumes epoch 1's checkpoint) reached **0.674** @ 40% of its epoch. Both
  already beat the shipped MiniLM checkpoint (0.5282).
- **BERTA** (`sergeyzh/BERTA`, BertModel, 128.3M, a FRIDA distillation) —
  real BERT pair format (`[CLS] item1 [SEP] item2 [SEP]` +
  `token_type_ids`), **mean pooling** (not CLS — BERTA's own model card
  calls this out as the one deliberate change from FRIDA),
  `target_modules=["query","key","value","dense"]`.
  `kaggle_kernels/berta-lora-sft-human-only-20-epochs` (account `rays0n`)
  planned 20 epochs; hit the 12h wall at just 22% of that (global_step
  5016/22840), already at val macro PR-AUC **0.7111** — the best number in
  this project to date, still trending up, nowhere near converged.

Debugging note carried by both: fp16 (and bf16 under `nn.DataParallel`)
produced NaN from a specific layer onward — root-caused in
`kaggle_kernels/frida-nan-diagnose` to `torch.autocast` opened in the
calling thread not propagating into `DataParallel`'s per-replica `forward()`
threads (documented PyTorch behavior, not a bug in this code). Fix: open
autocast **inside** the model's own `forward()`, not around the outer call.
Both scripts also apply the already-known "PEFT `is_trainable=True`" and
"`del logits, loss` before `continue` on a skipped NaN microbatch" lessons
from the MiniLM continue-train history instead of rediscovering them.

Neither checkpoint is fully trained or published as a durable Kaggle
dataset: FRIDA epoch 2's best (0.674) and BERTA's best (0.7111, with full
optimizer/scheduler state) currently exist only as `kaggle kernels
output`-recovered files under `kaggle_kernels/*/output/` (gitignored,
local-only, pulled during this session).

**A working FRIDA submission already passed the contest's external test
bench**: `matching-frida_lora_adapter-submit.zip` (downloaded locally to
`submits/matching-frida_lora_adapter-submit/`, gitignored — 1.3GB zip + a
1.65GB bf16 base model). Verified byte-for-byte (sha256) against the
adapter/classifier of `frida-lora-sft-human-only-10-epochs`'s **epoch-1**
checkpoint (val 0.652) — *not* epoch 2's 0.674, which hasn't been submitted
anywhere yet. Its `src/utils.py` is a real, battle-tested reference for
FRIDA inference specifically:
- Does **not** merge the LoRA adapter (unlike `submits/lora`, which merges
  precisely to drop the `peft` runtime dependency) — ships `frida-base/`
  (bf16, to fit the 5GB zip budget) and `frida-lora-adapter/` separately,
  plus a `wheels/` dir of ~26 offline pip wheels for `peft` + its transitive
  deps (confirmed absent from `odsai/ecup26-matching-baseline:1.0` on the
  real test stand). Installs via `pip install --no-index --target
  {tempfile.gettempdir()}` at runtime, appended to `sys.path` — a real
  stand run hit `OSError: Read-only file system` on both system
  site-packages and the `/root/.local` pip fallback; only `/tmp` was
  writable in the judging container.
- **Its pairwise text-preprocessing contract is FRIDA-specific, not the
  formula in this file's "Text preprocessing" section above** (that one is
  `submits/lora`/MiniLM's contract only). FRIDA's is: item-level
  tokenization with special tokens, `Category:` span excision (zero
  discriminative signal, proven), fair-share truncation to 128 — byte-match
  to what `frida-lora-sft-*` kernels train against, detailed in
  `kaggle_kernels/frida-lora-sft-human-only-10-epochs/README.md`.

No tracked `submits/frida_lora_adapter/` exists yet (the working code above
is only in the local, gitignored zip) — worth promoting `run.py`/
`src/utils.py`/`metadata.json` into git, mirroring `submits/lora`'s
pattern, once it's decided which checkpoint (epoch 1, epoch 2, or a
resumed/finished FRIDA run) should be the one tracked.

## Repo layout

```
notebooks/                                       training notebook (Kaggle, 2xT4, LoRA SFT)
kaggle_kernels/validate-baseline-vs-lora/         baseline vs LoRA val comparison (script, 1xT4)
kaggle_kernels/merge-lora/                        LoRA adapter -> merged weights
kaggle_kernels/lora-timing-dryrun/                full-scale offline GPU timing dry-run
kaggle_kernels/lora-llm-continue-train/           continued SFT on matches_llm.parquet
scripts/make_canonical_split.py                   regenerates data/splits/*.parquet
scripts/merge_lora.py, build_submission_zip.py, test_submission.py   submission build/test pipeline
data/, data/splits/                               local data + canonical split
submits/baseline/                                 organizer-provided reference (structural template)
submits/lora/                                     our submission
```

## Submission constraints

- **Offline** at inference — everything must be in the Docker image + archive.
- Image `odsai/ecup26-matching-baseline:1.0` (declared in `metadata.json`,
  `{"image": ..., "entry_point": "python -u run.py"}`) **lacks `polars`** —
  `submits/lora/src/utils.py` only imports what `submits/baseline/src/utils.py`
  does: `numpy`, `pandas`+`pyarrow`, `torch`, `transformers`. Custom images can
  be pushed to Docker Hub if a package is truly needed, but that's unverified/
  higher-effort — last resort.
- Hardware: 1x H100 80GB, 20 CPU cores, 200GB RAM.
- **3 separate time limits** (not summed): Check 1 min/1000 pairs (not
  scored), Public 6 min/~115K pairs, Private 13 min/~275K pairs. Fails any one
  stage → not scored, no partial credit. Measured on Kaggle T4 (slower than
  H100) at full local scale (365,654 pairs, offline): 950.6s, 384.7 pairs/sec
  — Public/Private both fit with margin even on the slower GPU.
- Output CSV: exactly `id1,id2,predict`, one row per input pair, **no drops**.
- Size: zip ≤5GB, image ≤15GB. **5 submissions/day hard cap** — validate
  locally + on a Kaggle GPU dry-run before spending one.
- `run.py` CLI: `--items_path --matches_path --output_path` (underscore —
  rules prose has a hyphen typo, working baseline code uses underscore).
- Hardware-aware tuning in `submits/lora/src/utils.py` (mirrors baseline):
  `TRITON_CACHE_DIR`/`TORCHINDUCTOR_CACHE_DIR` → `/dev/shm`,
  `OMP_NUM_THREADS`/`MKL_NUM_THREADS`/`RAYON_NUM_THREADS=32`,
  `TOKENIZERS_PARALLELISM=true`, **bf16** autocast (not fp16 — native H100
  speed), `attn_implementation="sdpa"`, inference batch size 1024.

Build/test:
```bash
python scripts/merge_lora.py                # LoRA adapter -> merged weights
python scripts/build_submission_zip.py       # -> submits/lora/matching-lora-submit.zip (zip root, no wrapping folder)
python scripts/test_submission.py --submission-dir submits/lora --n-pairs 1000
```
Always also extract the actual built zip to a clean dir and test *that*
(packaging bugs only show up in the real archive).

## Environment notes

- `.venv/` has only `kaggle`/`jupyter` tooling by default — install
  `polars pandas pyarrow scikit-learn mlflow torch transformers peft
  safetensors` on demand.
- Kaggle creds: `~/.kaggle/kaggle.json` (not in repo). Two accounts, each
  with its own weekly GPU quota: `maxamsecret` (main — owns most datasets
  and the shipped-architecture kernels) and `rays0n` (second account, used
  for the FRIDA/BERTA exploration, public kernels). Creds for both live
  outside the repo (`maximus/kaggle creds/kaggle_maxamsecret.json` and
  `kaggle_rays0n.json`) — **ask which account before launching a new GPU
  session.**
- mlflow≥3 needs `os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"` before
  `set_tracking_uri("file:...")` or it raises (file store is in "maintenance
  mode"). Local tracking, no server: `mlflow ui --backend-store-uri
  experiments/mlruns`.
- Kaggle GPU quota: 30h/week.

## Open TODOs

1. ~~Continued SFT on `matches_llm.parquet`~~ — done: after lowering `LR` to
   `2e-4` + warmup + human-mix (validated on a 2M-row subsample first, steady
   growth to 0.52447), the real full 11.48M-row run reached a new best
   **0.53229** (60% of one epoch) before hitting the 12h session limit.
   Remaining: publish that checkpoint as a Kaggle dataset (currently only in
   the kernel's own output) and decide whether it replaces
   `ozon-lora-minilm-finetuned` as the shipped adapter.
2. First real training run since the batch-size auto-probe (VRAM-target
   search before training starts) and mlflow logging were added to
   `notebooks/cross-encoder-sft-pipline (3).ipynb` — watch it once end-to-end.
3. FRIDA/BERTA exploration (see above) — both checkpoints (FRIDA 0.674,
   BERTA 0.7111) only exist as locally-pulled kernel output, not published
   as Kaggle datasets; publish both. BERTA specifically needs Kaggle Secrets
   configured on the `rays0n` kernel so its own auto-publish-checkpoint step
   works, then resume for the remaining ~16 of 20 planned epochs. FRIDA's
   sessions are bottlenecked by eval (~2h per full 73,131-row val pass, most
   of a 12h session) more than by training itself — worth shrinking/
   subsampling val between checkpoints before the next run. Both already
   beat the shipped MiniLM checkpoint mid-training and unconverged, which
   raises the question of whether either should become the new base
   architecture instead of an addition to it.
