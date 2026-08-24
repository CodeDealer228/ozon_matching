# Ozon E-CUP 2026 — Product Matching (Entity Resolution)

`d:/Projects/kaggle/ozon_matching` is the project root.

## Conventions
- Think in English, but respond to me in Russian.
- Write README.md in Russian.
- Commit all important step

## Response style

Respond in Russian unless I ask otherwise.

Default to concise, decision-oriented answers.
Start with the answer or recommended action — no preamble, restatement,
or generic explanation.

For questions and investigations:
- Give the conclusion first.
- State only facts needed to support it.
- Use at most 5 bullets unless I ask for depth.
- Do not narrate your reasoning, search process, or obvious steps.
- Do not repeat information already present in the conversation or command output.
- If uncertainty materially affects the answer, state it in one sentence.

For coding tasks:
- Make the smallest correct change that satisfies the request.
- Do not over-engineer, add abstractions, dependencies, refactors, tests,
  documentation, compatibility layers, or error handling unless necessary
  for correctness or explicitly requested.
- Before editing, inspect only the files relevant to the task.
- Do not print full files, diffs, logs, generated code, or command output
  unless asked or necessary to diagnose a failure.
- After implementation, report only:
  1. What changed
  2. Validation performed and its result
  3. Remaining caveat or next action, only if needed

For explanations:
- Prefer a direct answer and one minimal example over a tutorial.
- Ask a clarifying question only when proceeding would likely produce
  the wrong result; otherwise state the assumption briefly and proceed.

Use precise technical language. Avoid filler, praise, summaries,
and phrases such as “Sure”, “Let’s”, “I’ll”, “Here’s what I found”.

## Task

Binary classification of pre-retrieved product-pair candidates: given two Ozon
product cards (`id1`, `id2`), predict whether they're duplicates. Retrieval is
already done by organizers — this repo is only the pairwise re-ranking stage.
Container competition: submit a Docker image + code archive, judged offline
on hidden test data.

**Разметка**: любое "переключение" параметра (размер, цвет и т. п.) внутри
одной родительской карточки Ozon по умолчанию считается организаторами
РАЗНЫМ товаром (non-match), даже при идентичных названии и бренде — прямая
позиция организаторов (Q&A с Alex Natekin, 2026-08-23), не эвристика.
Полная цитата и разбор риска (продавцы иногда прячут вообще другой товар за
переключением цвета/размера): `reports/organizer_qa_variant_switching.md`.

## Data

Kaggle dataset **`maxamsecret/ozon-ecup-2026-matching`** (private). Local
copies in `data/` (gitignored — see README.md for download commands).

| File | Size | Columns |
|---|---|---|
| `items.parquet` | 13M+ rows, 4.1GB | `id`, `name`, `category`, `attributes` (JSON string) |
| `items_human.parquet` | 214MB — subset covering every id in `matches.parquet` | same schema; use for local dev/testing |
| `matches.parquet` | 365,654 rows, 4MB | `id1`, `id2`, `target` (0/1, human-labeled) |
| `matches_llm.parquet` | 11,187,780 rows, 104MB | `id1`, `id2`, `target` (float 0..1, LLM soft labels). Fully disjoint item universe from `matches.parquet` (0 shared ids, 0 shared pairs — see `reports/llm_label_overlap_summary.json`), so it's used for a second-stage continued SFT, not scored against human ground truth |

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

## Architecture & experiment history

- **FRIDA** (`ai-forever/FRIDA`, T5EncoderModel, 823M) and **BERTA**
  (`sergeyzh/BERTA`, BertModel, 128.3M, a FRIDA distillation) replaced MiniLM
  as the main line — bigger encoders, LoRA then full fine-tune. FRIDA's
  pairwise text-preprocessing contract is its own (item-level tokenization
  with special tokens, `Category:` span excised, fair-share truncation) —
  detailed in `kaggle_kernels/frida/lora/lora-sft-human-only-10-epochs/README.md`.
  A parallel **bi-encoder track** (Qwen3-Embedding-0.6B + LoRA + MLP head,
  teammate `grigoriyyy`, `bi-encoder` branch) also produced real submissions.

**Known val ↔ leaderboard/test pairs** (macro PR-AUC on our val split vs. the
contest's real judging score — "test" always means a submission scored on
the real H100 judging stand, never a Kaggle run):

| Variant | val macro PR-AUC | test/leaderboard |
|---|---|---|
| MiniLM baseline | 0.3710 | 0.1333 |
| MiniLM LoRA (shipped) | 0.5282 | 0.265 |
| FRIDA LoRA epoch 1 | 0.652 | 0.3459 |
| FRIDA LoRA epoch 2 (not submitted) | 0.675 | ? |
| BERTA LoRA (not submitted) | 0.711 | ? |
| Qwen3 bi-encoder + LoRA + MLP (not submitted, different val split) | 0.726 | ? |
| FRIDA full fine-tune (`full_sft`, item-disjoint split, `PAIR_MAX_LENGTH=180`) | 0.7582 | 0.481 |
| FRIDA `continue_a_baseline_3ep` (full-FT continued, 3/3 epochs, `PAIR_MAX_LENGTH=180`) | 0.7555 | 0.478 |
| FRIDA `continue_c_llm3m_2ep` `best`/`checkpoint_epoch01` (`PAIR_MAX_LENGTH=188`) | 0.7595 | 0.487 |
| **FRIDA `continue_c_llm3m_2ep` `last` checkpoint (earlier, unvalidated quiet-save, `PAIR_MAX_LENGTH=188`)** | ? (GPU val run in progress) | **0.48999 — project best** |

**Current project-best submission is the `last` checkpoint, not `best`**, of
the same `continue_c_llm3m_2ep` run — a step-earlier, never-validated save
that out-tested its own later/validated sibling. `PAIR_MAX_LENGTH=188` is now
the default serving length for every `continue`-lineage submission. Full
narrative for every row above (LoRA gotchas, the NaN-under-`DataParallel`
bug, Kaggle dataset-versioning saga, the `last`-beats-`best` mechanism, the
val→test least-squares fit and its caveats, MiniLM `matches_llm.parquet`
continue-SFT, and the parked Giga-Embeddings-instruct bi-encoder probe):
`reports/experiment_log.md`.

## Repo layout

```
notebooks/                                       training notebook (Kaggle, 2xT4, LoRA SFT)
kaggle_kernels/minilm/                                   MiniLM baseline/LoRA/continue-train kernels
kaggle_kernels/frida/                                    FRIDA LoRA + full-FT kernels (tokenize/lora/full_sft/sft_continue/percat/probes)
kaggle_kernels/berta/                                    BERTA LoRA + full-FT kernels (training/catboost)
kaggle_kernels/encoder_probes/                           Giga-3B, Qwen3 bi-encoder probes
kaggle_kernels/shared_utils/                             item-length-analysis, val-per-category-metrics, publish-token-cache
scripts/make_canonical_split.py                   regenerates data/splits/*.parquet
scripts/analyze_per_category.py                   offline post-analysis of saved val scores
scripts/merge_lora.py, build_submission_zip.py, test_submission.py   submission build/test pipeline (MiniLM)
data/, data/splits/                               local data + canonical + item-disjoint splits
reports/                                          written findings (tracked; kernel outputs are not) — experiment_log.md has full narrative detail
submits/                                          per-family/variant submission builds (minilm/, frida/, berta/, qwen/, probes/)
```
Full per-kernel index with per-checkpoint numbers: `kaggle_kernels/README.md`.
Full per-submission scores: README.md's results table.

## Submission constraints

- **Offline** at inference — everything must be in the Docker image + archive.
- Image `odsai/ecup26-matching-baseline:1.0` (declared in `metadata.json`,
  `{"image": ..., "entry_point": "python -u run.py"}`) **lacks `polars`** —
  submission code should only import what `submits/minilm/baseline/src/utils.py`
  does: `numpy`, `pandas`+`pyarrow`, `torch`, `transformers` (FRIDA submissions
  additionally ship `peft` as offline wheels — see `reports/experiment_log.md`).
- Hardware: 1x H100 80GB, 20 CPU cores, 200GB RAM.
- **3 separate time limits** (not summed): Check 1 min/1000 pairs (not
  scored), Public 6 min/~115K pairs, Private 13 min/~275K pairs. Fails any one
  stage → not scored, no partial credit.
- Output CSV: exactly `id1,id2,predict`, one row per input pair, **no drops**.
- Size: zip ≤5GB, image ≤15GB. **5 submissions/day hard cap** — validate
  locally + on a Kaggle GPU dry-run before spending one.
- `run.py` CLI: `--items_path --matches_path --output_path` (underscore —
  rules prose has a hyphen typo, working baseline code uses underscore).
- Hardware-aware tuning (mirrors baseline): `TRITON_CACHE_DIR`/
  `TORCHINDUCTOR_CACHE_DIR` → `/dev/shm`, `OMP_NUM_THREADS`/`MKL_NUM_THREADS`/
  `RAYON_NUM_THREADS=32`, `TOKENIZERS_PARALLELISM=true`, **bf16** autocast
  (not fp16 — native H100 speed), `attn_implementation="sdpa"`.

Build/test (MiniLM pipeline; FRIDA/BERTA submissions build via their own
`kaggle_kernels/*/submit-build` kernels):
```bash
python scripts/merge_lora.py                # LoRA adapter -> merged weights
python scripts/build_submission_zip.py       # -> submits/minilm/lora/matching-lora-submit.zip (zip root, no wrapping folder)
python scripts/test_submission.py --submission-dir submits/minilm/lora --n-pairs 1000
```
Always also extract the actual built zip to a clean dir and test *that*
(packaging bugs only show up in the real archive).

## Environment notes

- `.venv/` has only `kaggle`/`jupyter` tooling by default — install
  `polars pandas pyarrow scikit-learn mlflow torch transformers peft
  safetensors` on demand.
- **Each Kaggle account has its own separate GPU/TPU quota** — do not
  mix up which kernel/dataset was pushed under which, and **ask which
  account before launching a new GPU session** if it isn't obvious from
  context:
- mlflow≥3 needs `os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"` before
  `set_tracking_uri("file:...")` or it raises (file store is in "maintenance
  mode"). Local tracking, no server: `mlflow ui --backend-store-uri
  experiments/mlruns`.
- Kaggle GPU quota: 30h/week, **12h max per session** (hard cap). TPU quota:
  separate 20h/week pool, 9h max per session.
- Before writing/pushing/relaunching any quota-burning kernel, use the
  **`kaggle-experiment-safety`** skill
  (`.claude/skills/kaggle-experiment-safety/`): session-cap budgeting, the
  checkpoint contract, the list of calls that silently ruin a run, and a
  pre-push checklist — all distilled from real incidents in this repo, with
  the evidence in its `references/incidents.md`.

## Open TODOs

1. **Error analysis on current best checkpoints**: which validation mistakes
   actually move macro PR-AUC most (category clusters, attribute-length
   effects, near-duplicate-vs-true-duplicate confusion). Nothing systematic
   done yet beyond the stale MiniLM-only per-category breakdown.
2. **Synthetic data generation + usage** — design synthetic pairs/items for
   this task, then decide how it feeds training (mixed into SFT, separate
   pretraining stage, targeted at weak categories from #6). Not started;
   `negative_patterns/` (rule-mined hard negatives from real data, see
   `kaggle_kernels/retrieval_probes/negative-pattern-sampling`) is adjacent
   but not synthetic generation.
