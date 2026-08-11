# Ozon E-CUP 2026 — Product Matching

Product-pair duplicate classification for the Ozon Tech E-CUP 2026 hackathon
(container competition — retrieval is already done by organizers, this repo
is the pairwise re-ranking / classification stage). Approach: LoRA-tuned
`cross-encoder/ms-marco-MiniLM-L-12-v2`.

For the deep-dive (exact preprocessing formula, metric definition, hardware
quirks, submission history/lessons learned) see **[CLAUDE.md](CLAUDE.md)** —
this README is the quick-start; CLAUDE.md is the full reference.

## Current result

| Variant | macro PR-AUC (our val split) | Mean PR-AUC (real leaderboard) |
|---|---|---|
| Baseline (frozen encoder + LogReg) | 0.3710 | 0.1333 |
| **LoRA SFT (ours)** | **0.5282** | **0.265** |

Leaderboard score is from the first submission that cleared all judging
stages (see CLAUDE.md → "Submission history"). It's lower than our val score
for both variants — hidden-test items are different products than anything
in `matches.parquet`, so some generalization gap is expected — but LoRA's
edge over baseline held up: ~2x on the leaderboard vs. ~1.4x on val.

## Setup

```bash
git clone <this-repo>
cd ozon_matching
python -m venv .venv
.venv\Scripts\activate           # Windows; source .venv/bin/activate on macOS/Linux
pip install polars pandas pyarrow scikit-learn mlflow kaggle torch transformers peft safetensors
```

Kaggle credentials (needed for any `kaggle datasets ...` / `kaggle kernels ...`
command below): put your `kaggle.json` at `~/.kaggle/kaggle.json`
(`C:\Users\<you>\.kaggle\kaggle.json` on Windows) — **never commit it**.
Get one from https://www.kaggle.com/settings → API Tokens → Create Legacy API Key.

## Data — download from Kaggle (not in git, too heavy)

All competition data: **[maxamsecret/ozon-ecup-2026-matching](https://www.kaggle.com/datasets/maxamsecret/ozon-ecup-2026-matching)** (private — ask to be added as a collaborator if you get a 403).

| File | Size | What it's for |
|---|---|---|
| `matches.parquet` | 4MB | 365K human-labeled pairs — training/val |
| `items_human.parquet` | 214MB | Every item referenced by `matches.parquet` — **use this for local dev/testing**, not the full `items.parquet` |
| `matches_llm.parquet` | 104MB | 11.19M LLM-labeled pairs (weak/soft labels, continuous 0..1) — being used for a second-stage continued LoRA SFT, see below |
| `items.parquet` | 4.1GB | Full item catalog (13M+ rows) — only needed for a full-scale run |

```bash
mkdir -p data
# Just what you need for local dev (matches.parquet + items_human.parquet):
kaggle datasets download maxamsecret/ozon-ecup-2026-matching -f matches.parquet -p data
kaggle datasets download maxamsecret/ozon-ecup-2026-matching -f items_human.parquet -p data
# Everything, including the 4.1GB items.parquet:
kaggle datasets download -d maxamsecret/ozon-ecup-2026-matching -p data --unzip
```

Model weights (needed to run/rebuild the submission, not in git):

- **[maxamsecret/ozon-lora-merged-minilm](https://www.kaggle.com/datasets/maxamsecret/ozon-lora-merged-minilm)** — merged LoRA weights ready for inference (no `peft` needed). This is what `submits/lora/` uses.
- **[maxamsecret/ozon-lora-minilm-finetuned](https://www.kaggle.com/datasets/maxamsecret/ozon-lora-minilm-finetuned)** — the original PEFT adapter checkpoint (pre-merge), if you need to re-merge or inspect the adapter itself.

```bash
kaggle datasets download -d maxamsecret/ozon-lora-merged-minilm -p artifacts/lora_merged_dataset --unzip
```

## Repo layout

```
notebooks/                  training notebook (run on Kaggle, 2xT4, LoRA SFT)
kaggle_kernels/              standalone Kaggle kernels (validation, LoRA merge, timing dry-runs,
                              continued SFT on LLM-labeled data)
scripts/                     local dev scripts (split generation, zip build, smoke tests)
data/                        local copies of the parquet files above (gitignored)
data/splits/                 canonical train/val split (small, tracked in git — see CLAUDE.md)
submits/baseline/            organizer-provided reference solution
submits/lora/                our submission (run.py / src/utils.py / metadata.json + model weights)
```

## Rebuilding and testing the submission

```bash
python scripts/merge_lora.py                # LoRA adapter -> merged weights in artifacts/lora_merged/
# copy the merged files into submits/lora/models/lora-merged-ms-marco-MiniLM-L12-v2/ and classifier.pt
python scripts/build_submission_zip.py       # -> submits/lora/matching-lora-submit.zip
python scripts/test_submission.py --submission-dir submits/lora --n-pairs 1000
```

`test_submission.py` builds a small real sample from `data/items_human.parquet`
+ `data/matches.parquet` and runs `run.py` through the exact CLI contract the
judging harness uses, checking output format, row-count parity, and
determinism. **Always also extract the actual built zip to a clean dir and
test that** (not just the loose `submits/lora/` folder) — see CLAUDE.md's
"Submission history" section for two real bugs (wrapping folder in the zip,
a package not present in the judging Docker image) that only showed up that
way.

## In-progress: continued LoRA SFT on `matches_llm.parquet`

Second SFT stage on top of the 0.5282 checkpoint above, using the 11.19M
LLM-labeled (soft, continuous 0..1) pairs that were previously unused —
1 epoch, `LR=4e-4` cosine schedule (2x the original run's peak LR, chosen
from analyzing the original run's real per-epoch loss curve — see CLAUDE.md
for the full reasoning), validated on the *exact same* 73,131-row canonical
val split. Confirmed empirically that `matches_llm.parquet` is fully disjoint
from `matches.parquet` in both pairs and item-id universe (0 shared ids out
of 12.4M vs 711K), so there's no leakage risk and this genuinely expands
catalog coverage rather than re-labeling the same pairs.

Kernel: [maxamsecret/lora-llm-continue-train](https://www.kaggle.com/code/maxamsecret/lora-llm-continue-train) (private) —
`kaggle_kernels/lora-llm-continue-train/`. Validates + checkpoints every ~10%
of steps within the epoch (not just per-epoch) so the best adapter is always
close to disk, plus throttled dataset-version pushes for crash resilience
beyond a Kaggle session. See CLAUDE.md → "Continued LoRA SFT on
matches_llm.parquet" for the full design, the RAM-debugging story (items.parquet
at full 13M-row scale OOM's without care), and current status.

## Experiment tracking

Local mlflow file-store, no server: `mlflow ui --backend-store-uri experiments/mlruns`
to browse. See CLAUDE.md → "Experiment tracking & caching" for the setup
details (there's a required `MLFLOW_ALLOW_FILE_STORE=true` env var on
mlflow≥3).

## Useful Kaggle kernels (private)

- [Baseline vs LoRA validation](https://www.kaggle.com/code/maxamsecret/ozon-baseline-vs-lora-val-macro-pr-auc) — the comparison behind the result table above
- [LoRA merge](https://www.kaggle.com/code/maxamsecret/ozon-merge-lora-adapter-into-base-weights) — produces the `ozon-lora-merged-minilm` dataset
- [Submission GPU timing dry-run](https://www.kaggle.com/code/maxamsecret/ozon-lora-submission-gpu-timing-dry-run-offline) — full-scale, offline, timing sanity check against the judging time limits
- [LoRA continue-train on LLM-labeled pairs](https://www.kaggle.com/code/maxamsecret/lora-llm-continue-train) — second SFT stage on `matches_llm.parquet`, see above

## Competition constraints, in short

Read CLAUDE.md for the full breakdown — the sharpest edges:

- Judged **offline**, in a fixed Docker image (`odsai/ecup26-matching-baseline:1.0`
  by default) — code may only import packages that image actually has
  (`numpy`, `pandas`+`pyarrow`, `torch`, `transformers`; **no** `polars`,
  learned this from a failed submission).
- 3 separate time limits, not one shared budget: Check 1 min (1000 pairs),
  Public 6 min (~115K pairs), Private 13 min (~275K pairs).
- **5 submissions/day, hard cap.** Test locally + on a Kaggle GPU dry-run
  before spending one.
