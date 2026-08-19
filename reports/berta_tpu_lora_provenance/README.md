# Provenance: `matching-berta_tpu_lora-submit.zip` (test score 0.47)

Reconstructed and **hash-verified**, not recalled from memory. The zip at
`submits/matching-berta_tpu_lora-submit.zip` was extracted and its adapter/
classifier sha256'd against a locally-preserved copy of the training run's
recovered checkpoint (`tpu_e9/` — downloaded via `kaggle kernels output`
during this project's original BERTA-TPU work, before the crash-fix
described below existed):

| File (inside the zip) | sha256 | Matches |
|---|---|---|
| `models/berta-lora-adapter/adapter_model.safetensors` | `26b294123b2445fa7bb46e0c36c2ee5ad259c218ea80e357661f49729800cd34` | ✅ byte-identical to `tpu_e9/lora_berta_tpu_sft/checkpoint/adapter/adapter_model.safetensors` |
| `classifier.pt` | `70ac231c64ad2f70ac73592afbce76c3ee9f0cb804ecb16e8b0f4258325d43fa` | ✅ byte-identical to `tpu_e9/lora_berta_tpu_sft/checkpoint/classifier.pt` |

This confirms with certainty which training run produced the shipped
weights — not just "probably this one."

## Exact kernel sequence

1. **`rays0n/berta-llm-continue-train`** (pre-existing checkpoint, part of
   the earlier BERTA lineage — see `kaggle_kernels/README.md`) → PEFT
   adapter, `global_step=2002`, `val_macro_pr_auc=0.7254629058736792`.
2. **`kaggle_kernels/berta-merge-lora`** (`maxamsecret/berta-merge-lora`) —
   loads base `sergeyzh/BERTA` + the adapter from (1), `merge_and_unload()`.
   Output: `merged_encoder/` (plain `BertModel`, no `peft` needed) +
   `classifier.pt` (the OLD classifier head from (1), carried through
   unchanged at this step — not retrained here) + `merge_meta.json`.
   Confirmed directly from the training run's own printed log:
   `{'base_model': 'sergeyzh/BERTA', 'hidden_dim': 768, 'source_checkpoint':
   'rays0n/berta-llm-continue-train', 'source_global_step': 2002,
   'source_val_macro_pr_auc': 0.7254629058736792}`.
3. **`kaggle_kernels/berta-tokenize-llm-256`**
   (`maxamsecret/berta-tokenize-llm-256`) — TPU item-token cache,
   `human_scope` (711,304 items) + `llm_scope` (12,384,610 items),
   `ITEM_MAX_LENGTH=256`, `PREFIX="paraphrase: "`. (This kernel was
   restarted once earlier in the project to fix a prefix bug — the run
   consumed here is confirmed to be the corrected one via the mounted path
   in the training log.)
4. **`kaggle_kernels/berta-llm-tpu-sft`** (`maxamsecret/berta-llm-tpu-sft`)
   — TPU v5e-8, `torch_xla`. Builds a **fresh** LoRA (`get_peft_model`, not
   a resume) on top of (2)'s merged base; classifier **warm-started** from
   (2)'s `classifier.pt` (then further trained, not frozen). Trained on the
   **full** `matches_llm.parquet` (11,187,780 pairs, no sampling) mixed
   with the full human train split (292,523) = 11,480,303 combined rows.
   - `PAIR_MAX_LENGTH=256`, LoRA `r=16, lora_alpha=32, lora_dropout=0.15`,
     `target_modules=["key","query","dense","value"]` (from the shipped
     `adapter_config.json` directly).
   - `per_core_train_batch=64 × world_size=8 = 512` effective batch,
     `steps_per_epoch_global=22422`, `warmup=2242` (10%),
     `max_train_seconds=28800` (8h cap).
   - **This run crashed** (`concurrent.futures.process.BrokenProcessPool`).
     Root-caused and fixed in a later session (see
     `kaggle_kernels/README.md` → BERTA section): all 8 forked TPU replicas
     were independently reloading/concatenating the full item cache and
     `matches_llm.parquet`, exhausting host RAM right at an unnecessary,
     schedule-mismatched second-epoch boundary. Not relevant to *this*
     checkpoint's validity — see the note below.

## The shipped checkpoint is a 50%-of-one-epoch checkpoint

The crashed run had exactly **one** successful checkpoint/eval before
dying: at **`global_step=11211`** — exactly 50% of the 22,422-step epoch
(`VAL_EVERY_FRACTION=0.5` → `checkpoint_interval=11211`) —
**`val_macro_pr_auc=0.7087279672868713`**. Per-step training logs continue
past this point to ~step 22400 (LR fully decayed, i.e. training *did* reach
~100% of the epoch), but the epoch-end (100%) checkpoint never finished
saving before the crash. So the 50%-mark checkpoint is the *only* one that
ever reached disk, hence the only one recoverable via `kaggle kernels
output` on a cancelled session — and it is exactly what's shipped.

## Build steps (this repo, not Kaggle)

5. Recovered checkpoint copied into `submits/berta_tpu_lora/`:
   `models/berta-lora-adapter/` (from (4)'s `adapter/`, unchanged),
   `classifier.pt` (from (4), unchanged, kept fp32 — the training script
   never casts the classifier to bf16, only the frozen base), `models/
   berta-base/` (bf16 conversion of (2)'s `merged_encoder/`), plus
   `wheels/` (offline `peft` + transitive deps, borrowed from
   `submits/berta_catboost/wheels/`), `src/utils.py` / `run.py` /
   `metadata.json` (BERTA cross-encoder inference contract,
   `PAIR_MAX_LENGTH=256` matching training).
6. `python scripts/build_submission_zip.py --submission-dir
   submits/berta_tpu_lora` → `submits/matching-berta_tpu_lora-submit.zip`
   — **the exact file that scored 0.47.**

## One finding worth flagging before the val/test discussion

`train_state.json`'s `best_val_macro_pr_auc` is a **running-best metric**,
not a guarantee that the *saved weights* achieved it: `save_checkpoint()`
in `berta_llm_tpu_sft.py` is called unconditionally at every checkpoint
interval with whatever the *current* live weights are, alongside the
*historical* best metric value — it does not re-save the weights from the
step that actually earned that best score. For **this** specific shipped
checkpoint that's not an issue (it's the run's only checkpoint event, so
weights and metric genuinely correspond to the same step) — but it's a
real trap for any *other* checkpoint pulled from a run with more than one
checkpoint event (e.g. a completed run that checkpoints at both 50% and
100%), and directly relevant to reasoning about `berta_tpu_lora_after_llm`.
