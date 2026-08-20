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
| `matches_llm.parquet` | 11,187,780 rows, 104MB | `id1`, `id2`, `target` (float 0..1, LLM soft labels). Fully disjoint item universe from `matches.parquet` (0 shared ids of 12.4M vs 711K, 0 shared pairs — re-verified via `scripts/analyze_llm_label_overlap.py`, see `reports/llm_label_overlap_summary.json`) — so LLM-label quality can't be scored directly against human ground truth, no overlapping pairs exist. Used for a second-stage continued SFT, see below |

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
  pre-merge — needed to resume/continue training; currently holds the
  0.53229 checkpoint from the LLM continue-SFT run, see below).

### Results

| Variant | val macro_pr_auc | leaderboard Mean PR-AUC |
|---|---|---|
| Baseline | 0.3710 | 0.1333 |
| **LoRA SFT** | **0.5282** | **0.265** |

Leaderboard score < val score for both (hidden-test items differ from
`matches.parquet`'s), but LoRA's relative edge over baseline held up: ~2x on
leaderboard vs. ~1.4x on val.

**Per-category, 4 categories hold 77% of the remaining headroom**: Ювелирные
изделия (prevalence-corrected skill 0.150), Одежда (0.189), Обувь (0.206),
Мебель (0.213) — vs. a 0.384 median across all 20. Root cause: the 128-token
window truncates 100% of val pairs, and English WordPiece shreds Russian to
1.33 chars/token, so the median item gets 0–14 attribute tokens after the
`Name:/Category:/Attributes:` prefix eats ~56 tokens — the deciding
variant-level signal (size/colour/article) lives in attributes that never
make it into the window. A naive lexical (name-Jaccard) blend was tried and
ruled out (monotonically hurts). Full derivation and raw numbers:
`reports/experiment_log.md`, `reports/val_per_category_lora.md`/`.csv`.

## LoRA SFT continued on `matches_llm.parquet` — paused for FRIDA pivot

Second-stage SFT on the epoch-7 checkpoint, mixing the human 80% train split
with `matches_llm.parquet` (11.19M LLM soft-labeled pairs, continuous
targets via `BCEWithLogitsLoss`, no threshold). Code:
`kaggle_kernels/lora-llm-continue-train/`.

- `LR=4e-4` LLM-only regressed val_macro_pr_auc below the 0.5282 starting
  point and never recovered. Fix: `LR=2e-4` (the original run's own peak),
  `WARMUP_RATIO=0.10`, human data mixed into every batch — validated at 2M
  rows (→0.52447), then at the full 11.48M-row scale.
- The full run climbed cleanly to **0.53229** (new best) before hitting
  Kaggle's **hard 12h GPU session cap** at ~69% of one epoch. Recovered via
  `kaggle kernels output` (persists `/kaggle/working` even on a hard-cap
  cancellation) and **published** as the current `ozon-lora-minilm-finetuned`
  best.
- Root cause of not finishing: tokenization alone took **9.54h** (~80% of
  the session) — `tokenize_to_memmap` rejoins the full items table on
  *every chunk* instead of once. Not yet fixed; blocks resuming this path.
- Paused in favor of the FRIDA bi-encoder pivot (below): the freed-up time
  budget made a bigger/different encoder affordable, and per-item
  pre-tokenization caching (the idea that motivated the pivot) can't work
  cleanly for a cross-encoder anyway — HF's joint `longest_first`
  pair-truncation would need exact replication outside the tokenizer, and
  `truncate_sequences` isn't reachable as an instance method to do that.

Full gotcha list (PEFT `is_trainable` trap, items.parquet OOM, `maintain_order`
join cost, non-catchable Linux OOM-kills) and exact phase timings:
`reports/experiment_log.md`.

## Bi-encoder probes

**Giga-Embeddings-instruct (3.45B, parked)**: raw zero-shot cosine similarity
already beats the *trained* baseline (0.4004 vs 0.3710 val macro PR-AUC), and
int8 quantization is nearly free (0.3973, 3.47GB on disk). But the candidate
pairs have almost no item reuse (1.94 items/pair), so a bi-encoder pays ~2x
the cross-encoder's token compute per pair instead of the usual "encode
once, compare many" win — FLOPs projection misses even the H100's time
limits (Public ~7min vs 6min limit, Private 17-22min vs 13min limit), before
counting IO/tokenization/weight loading. Full table, FLOPs math, and
T4-specific gotchas: `reports/experiment_log.md`.

**FRIDA (`ai-forever/FRIDA`, 0.8B)**: chosen over continuing MiniLM
continue-SFT because it's a bi-encoder — no joint pair-truncation to
replicate, so item text can be pre-tokenized and cached per-item instead of
re-tokenized per-pair. Confirmed directly against the loaded model (the card
is partly wrong): FRED-T5-encoder + CLS pooling + L2 norm, tokenizer is
actually `RobertaTokenizer` (byte-level BPE, vocab 93,651). Prefix
**`paraphrase: `** (card names "deduplication" explicitly under this
prefix). `max_length=256` (not the 512 cap) — empirically covers ~99.99% of
items (20K-item sample: mean=126.6, p99=202.0, true max=272).

Tokenization ran as two Kaggle kernels — `frida-tokenize-items-human`
(CPU) and `frida-tokenize-items-llm` (TPU, for RAM headroom + a separate
20h/week quota pool). **Done as of 2026-08-13**: both
`maxamsecret/frida-tokenized-items-human` (711,304 items) and
`maxamsecret/frida-tokenized-items-llm` (12,384,610 items) are published —
each an `ids.bin`/`input_ids.bin`/`attention_mask.bin` memmap dataset, so
training does a pure id lookup with no tokenizer call and no per-chunk
rejoin against the items table. Getting there took 5 rounds of environment
gotchas on the TPU image specifically (missing `polars`, a `transformers`/
`tokenizers` version mismatch, an over-conservative disk-space margin, a
missing `kaggle` CLI binary, and a dataset-naming mix-up with a manual
recovery the user did in parallel) — full detail in
`reports/experiment_log.md`. Training architecture on top of the cache
(classifier head over `[emb1, emb2, |emb1-emb2|]` vs. cosine similarity) not
yet decided.

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

Both checkpoints are now published as Kaggle datasets (by the team, not
from a kernel's own auto-publish step): `maxamsecret/frida-lora-sft-epoch2-40`
(FRIDA epoch 2, 0.674) and `maxamsecret/berta-lora-sft-5epoch` (BERTA,
0.7111, with full optimizer/scheduler state). Neither ended up as a new
*version* of the older canonical dataset it continues (`maxamsecret/frida-lora`
for FRIDA, `rays0n/berta-lora-sft-checkpoint` for BERTA, which was never
actually created) -- `kaggle datasets version` returned `Incompatible
Dataset Type` for every `-r` mode tried (skip/zip/tar) against
`frida-lora`, uploading real bandwidth each time without ever committing;
the older datasets are unchanged. `kaggle_kernels/frida-lora-sft-epoch3`
and `kaggle_kernels/berta-lora-sft-human-only-20-epochs`'s
`kernel-metadata.json` both point `dataset_sources` at the actual new
dataset names now (not the old ones) -- if a checkpoint ever needs to move
again, only `dataset_sources` needs updating, since the resume code
searches `/kaggle/input` by file content (`adapter_config.json` /
`train_state.json`), not by dataset name. The bi-encoder checkpoint is
also published, as `maxamsecret/qwen-0-6-lora-sft-1epoch` (no prior
dataset existed for it -- nothing to version).

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

### Known val ↔ leaderboard pairs

macro PR-AUC on our val split vs. the contest leaderboard's Mean PR-AUC:

| Variant | val macro PR-AUC | leaderboard |
|---|---|---|
| MiniLM baseline | 0.3710 | 0.1333 |
| MiniLM LoRA (shipped) | 0.5282 | 0.265 |
| FRIDA epoch 1 (submitted, passed test bench) | 0.652 | **0.3459** |
| FRIDA epoch 2 (not submitted) | 0.675 | ? |
| BERTA (not submitted) | 0.711 | ? |
| Qwen3 bi-encoder + LoRA + MLP head (teammate `grigoriyyy`, `bi-encoder` branch, not submitted, **different val split** — see caveat below) | 0.726 | ? |

A least-squares fit on the three *confirmed* points (all cross-encoders on
the same seed-42 val split) — `leaderboard ≈ 0.763 × val − 0.146` (R²≈0.996
on just those 3 points) — would put BERTA around **~0.40** and FRIDA epoch 2
around **~0.37** leaderboard. Treat as a rough sanity check, not a
prediction: it's a 3-point fit extrapolated past its own range, and it says
nothing about a *bi-encoder*'s val→test generalization (different
architecture family than every point used to fit it; its 0.726 is also on
a private item-disjoint 5-fold split, not the seed-42 one the other rows
share — see `kaggle_kernels/qwen3-bi-encoder-lora-sft/README.md` on the
`bi-encoder` branch). A real second data point from a non-cross-encoder
submission would be worth far more than refining this fit further.

## Repo layout

```
notebooks/                                       training notebook (Kaggle, 2xT4, LoRA SFT)
kaggle_kernels/validate-baseline-vs-lora/         baseline vs LoRA val comparison (script, 1xT4)
kaggle_kernels/merge-lora/                        LoRA adapter -> merged weights
kaggle_kernels/lora-timing-dryrun/                full-scale offline GPU timing dry-run
kaggle_kernels/lora-llm-continue-train/           continued SFT on matches_llm.parquet
kaggle_kernels/item-length-analysis/              untruncated field-length deciles over the 13M catalog
kaggle_kernels/val-per-category-metrics/          per-category val PR-AUC of the merged LoRA (CPU)
kaggle_kernels/giga3b-quantize-benchmark/         Giga-3B bi-encoder probe: raw quality + int8 save + throughput
kaggle_kernels/frida-tokenize-items-human/        FRIDA per-item tokenization cache, human scope (~711K items, CPU)
kaggle_kernels/frida-tokenize-items-llm/          FRIDA per-item tokenization cache, LLM scope (~12.38M items, TPU)
scripts/make_canonical_split.py                   regenerates data/splits/*.parquet
scripts/analyze_per_category.py                   offline post-analysis of saved val scores
scripts/merge_lora.py, build_submission_zip.py, test_submission.py   submission build/test pipeline
data/, data/splits/                               local data + canonical split
reports/                                          written findings (tracked; kernel outputs are not) — experiment_log.md has full narrative detail
submits/baseline/                                 organizer-provided reference (structural template)
submits/lora/                                     our submission
submits/giga3b_probe/                             bi-encoder pipeline (dependency-injected encoder); parked, see above
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
- **Three separate Kaggle accounts, three separate GPU/TPU quotas** — do not
  mix up which kernel/dataset was pushed under which, and **ask which
  account before launching a new GPU session** if it isn't obvious from
  context:
  - `maxamsecret` — default account, creds at `~/.kaggle/kaggle.json` (not in
    repo). Owns `ozon-ecup-2026-matching` (the private source dataset) and
    all FRIDA-related kernels/datasets (`frida-*`).
  - `rays0n` — second account, added 2026-08-13 specifically to run BERTA
    (`kaggle_kernels/berta-lora-sft`) on its own separate quota rather than
    competing with FRIDA's runs. Creds at `~/.kaggle/kaggle_rays0n.json` (not
    in repo, not the default file — never overwrite `kaggle.json` with
    this). Has been granted read access to `maxamsecret/ozon-ecup-2026-matching`
    (confirmed via `kaggle datasets files`). All BERTA kernels/datasets are
    namespaced `rays0n/berta-*`, not `maxamsecret/...`.
  - `maximmarkovbb` — third account, added for the FRIDA v2 continue-train
    (`kaggle_kernels/frida-lora-v2-epoch`) with a clean, unused GPU quota.
    Creds at `~/.kaggle/kaggle_maxim_markov.json`. Also has read access to
    `maxamsecret/ozon-ecup-2026-matching`. Kernels/datasets under this
    account are namespaced `maximmarkovbb/...` and, per its own convention,
    always public.
  - To run a CLI command as a non-default account, override via env vars
    rather than touching the default file: `KAGGLE_USERNAME=rays0n
    KAGGLE_KEY=$(python -c "import json;print(json.load(open('$HOME/.kaggle/kaggle_rays0n.json'))['key'])")
    kaggle ...` (swap in `kaggle_maxim_markov.json`/`maximmarkovbb` as
    needed). Any kernel pushed under a non-default account must also use
    that account's creds as kernel secrets (Kaggle website → kernel →
    Settings → Secrets) if it needs `KAGGLE_USERNAME`/`KAGGLE_KEY` at
    runtime (e.g. berta-lora-sft's checkpoint-dataset auto-publish) — this
    is a per-kernel web UI setting, not something `kernel-metadata.json` or
    the CLI push can configure.
- mlflow≥3 needs `os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"` before
  `set_tracking_uri("file:...")` or it raises (file store is in "maintenance
  mode"). Local tracking, no server: `mlflow ui --backend-store-uri
  experiments/mlruns`.
- Kaggle GPU quota: 30h/week, **12h max per session** (hard cap, confirmed
  the hard way — see `reports/experiment_log.md`). TPU quota: separate
  20h/week pool, 9h max per session.
- Before writing/pushing/relaunching any quota-burning kernel, use the
  **`kaggle-experiment-safety`** skill
  (`.claude/skills/kaggle-experiment-safety/`): session-cap budgeting, the
  checkpoint contract, the list of calls that silently ruin a run, and a
  pre-push checklist — all distilled from real incidents in this repo, with
  the evidence in its `references/incidents.md`.

## Open TODOs

1. **Paused for the FRIDA pivot** — resume `matches_llm.parquet` continue-SFT
   only after fixing `tokenize_to_memmap`'s per-chunk items rejoin (~9.5h of
   the last 12h run); the 0.53229 checkpoint is already published as current
   best regardless.
2. First real training run since the batch-size auto-probe (VRAM-target
   search before training starts) and mlflow logging were added to
   `notebooks/cross-encoder-sft-pipline (3).ipynb` — watch it once end-to-end.
3. Per-category fixes, cost/benefit order, all targeting the fashion/jewellery
   77%-of-headroom categories: (a) drop the category value + English
   `Name:/Category:/Attributes:` literals (frees ~29% of the token budget,
   zero signal loss); (b) raise `max_length` 128 → 256; (c) swap in a
   Russian-capable encoder (`rubert-tiny2`, multilingual MiniLM) — ~3x more
   content in the same window given the current 1.33 chars/token. Any
   text-formula/`max_length` change must land in both training code and
   `submits/lora/src/utils.py` (byte-match requirement).
4. Consider per-category loss weighting: the metric is a plain mean over 20
   categories, but the training loss is proportional to the pooled row counts.
5. **Next up**: FRIDA bi-encoder (see above) — per-item tokenization caches
   are done and published (2026-08-13). At ~1.94 items/pair a bi-encoder
   costs ~2x the cross-encoder's token compute at equal size, so it has to
   *win on quality*, not speed, to be worth the switch — not yet measured
   for FRIDA. Next: decide the training architecture (classifier head over
   `[emb1, emb2, |emb1-emb2|]` vs. cosine similarity), then train and get a
   real val macro PR-AUC to compare
   against 0.5282/0.53229.
6. Teammate `grigoriyyy`'s parallel **bi-encoder** track (Qwen3-Embedding-
   0.6B + LoRA + MLP head) already produced real scored submissions, tracked
   under `submits/ecup_qwen_stage2_submit` (test 0.41, current best of any
   submission in the repo — a checkpoint that *regressed* ~1% on val but
   *gained* on the real test stand, see `[[submission-test-scores]]`-style
   note in project memory) and `submits/ecup_qwen_biencoder_submit_128_length`
   (0.35/0.37 variants). Worth understanding *why* the train-overfit
   checkpoint generalizes better here before assuming it's a fluke — it's
   already informed how FRIDA/BERTA checkpoint selection for submission is
   being approached (building both "last" and "best" val-checkpoint
   submissions rather than assuming "best" always wins).
7. **Error analysis on the current best checkpoints**: look at *which*
   validation mistakes actually move macro PR-AUC the most (not just the
   aggregate number) and look for real patterns — category clusters,
   attribute-length effects, near-duplicate-vs-true-duplicate confusion,
   etc. Nothing systematic done yet beyond the per-category breakdown in
   `reports/val_per_category_lora.md` (MiniLM-only, stale relative to
   BERTA/FRIDA/Qwen3).
8. **Synthetic data generation** — design how to generate synthetic
   pairs/items for this task. Not started; `negative_patterns/` (rule-based
   hard-negative pairs per category, see `kaggle_kernels/negative-pattern-
   sampling`) is adjacent but is pattern-mined from real data, not
   synthetic generation.
9. **Synthetic data usage** — once (8) exists, decide how it actually feeds
   training (mixed into SFT like `matches_llm.parquet` already is, a
   separate pretraining/augmentation stage, targeted at the weak categories
   from (7), etc.) — depends on what (7)'s error patterns actually show.
10. **`submits/matching-qwen-berta-catboost-submit` fits the timing budget
    but scores badly** — root cause: the CatBoost head was trained on
    LLM-labeled data (`matches_llm.parquet`), but BERTA's own eval
    distribution and Qwen3's own eval distribution differ from each other
    and (presumably) from the LLM-label distribution the CatBoost head
    actually learned on — a mismatch, not a bug in the pipeline itself
    (the qwen+berta+catboost combination genuinely runs in time). Need a
    fix that does **not** require re-running the underlying qwen/BERTA
    training (~50 GPU-hours already sunk into those checkpoints) — only
    re-fit/re-derive the CatBoost head on non-leaking features. The
    project already has one relevant precedent to build on:
    `kaggle_kernels/berta-catboost-leak` vs. `berta-catboost-noleak`
    (explicit train/val leakage A/B for a BERTA-only CatBoost head, see
    `kaggle_kernels/README.md`) — extend that same leak-vs-noleak
    methodology to the combined qwen+berta feature set rather than
    inventing a new one.
