# Experiment log — full detail behind CLAUDE.md's summaries

This file is the detailed backing for the condensed summaries in
`CLAUDE.md` (per-category breakdown, LLM continue-SFT, bi-encoder probes).
Nothing here overrides `CLAUDE.md` — it exists so the always-loaded file can
stay short without losing any of the numbers, root causes, or ruled-out
ideas below. Organized in the same order `CLAUDE.md` references it.

## Per-category breakdown of the 0.5282 LoRA cross-encoder val score

Full write-up: `reports/val_per_category_lora.md`, numbers in
`reports/val_per_category_lora.csv`. Produced by
`kaggle_kernels/val-per-category-metrics/` (CPU, ~85 min, 14.4 pairs/sec) —
it re-derives the seed-42 val split and **hard-asserts** its fingerprint
`efddb3def997f87b`, and dumps per-pair scores to `val_scores.parquet` so any
further slicing costs no forward pass. Reproduced macro exactly: 0.52826.

- **Raw AP is not comparable across categories** — a constant scorer's AP is
  exactly the prevalence, and prevalence runs 0.069 (Аптека) → 0.576 (Детские
  товары); `corr(AP, prevalence) = 0.834`. Rank categories by the
  prevalence-corrected skill score `(AP - π)/(1 - π)` instead. Two traps this
  exposes: Аптека looks weak (AP 0.434) but is the **2nd best ranker of all 20**
  (ROC-AUC 0.909, AP capped by prevalence — nothing to win there); Канцелярские
  товары looks mid (AP 0.500) but has the **worst ROC-AUC of all 20** (0.677).
- **Weakest by skill**: Ювелирные изделия 0.150, Одежда 0.189, Обувь 0.206,
  Мебель 0.213, Канцелярские товары 0.252 (median across categories 0.384).
  Lifting everything below median up to the median → macro 0.5725 (+0.044);
  **the four fashion/jewellery categories alone are +0.0341 = 77% of that
  headroom.** The other 16 categories barely move the metric.
- **Why they fail**: labels there are variant-level. False positives are the
  same product differing only in size/colour/article (label 0, scored 0.84–0.95);
  false negatives are the same product under unrelated seller-written names
  (label 1, scored 0.05). The deciding signal lives in the attributes.
- **The 128-token window is the binding constraint**: 100% of val pairs are
  truncated (median 445–780 tokens per pair), `truncation="longest_first"` gives
  each item ~62 tokens, and the `Name: … Category: … Attributes: ` prefix alone
  takes a median of **56.5** of them — so the median item passes **0–14
  attribute tokens** (median 0 in 6 categories) out of 146–340 available. Root
  cause is the English WordPiece vocab shredding Russian per character
  (`кольцо` → `к ##о ##л ##ь ##ц ##о`, **1.33 chars/token**). Separately ~29% of
  the budget goes to the `Name:/Category:/Attributes:` literals plus the
  category value, which is identical for both items of every pair (verified
  100%) and so carries zero discriminative signal — free to reclaim, though
  honestly `corr(that overhead, skill) = +0.44`, so it does *not* explain the
  weak categories by itself.
- **Negative result**: the encoder has already absorbed the lexical signal.
  `corr(AP, AP of a naive name-Jaccard scorer) = 0.946`, and rank-blending the
  model with that Jaccard monotonically *hurts* (0.5283 → 0.5253 at w=0.05 →
  0.4704 at w=0.5). No free lexical win.

## Continued LoRA SFT on `matches_llm.parquet` (paused — see CLAUDE.md TODO #1)

Second SFT stage on top of the epoch-7 checkpoint, mixing the human 80%
train split with a sample of the 11.19M soft-labeled `matches_llm.parquet`
pairs (fed to `BCEWithLogitsLoss` as continuous targets, no threshold). Goal:
close the val→leaderboard generalization gap via wider catalog coverage,
without regressing the human-tuned signal.

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
  needed ids *before* the string/truncation ops.
- A join with `maintain_order="left"` against a multi-million-row items table
  is *much* more expensive than a plain hash join — a real run got
  OOM-killed processing a single 20K-row chunk against a 3.45M-row items
  table (RSS 9.7GB → 31.1GB/32.21GB in one chunk) purely from this. Fixed by
  dropping `maintain_order` and relying on an explicit `.sort(row_index)`
  after the join instead, which is actually a *stronger* correctness
  guarantee anyway (unambiguous regardless of the join's internal
  algorithm/order, not just an observed-but-undocumented polars behavior).
- A real Linux OOM-kill is **not** a catchable `MemoryError` under default
  overcommit (unlike CUDA OOM) — tokenization chunk size is now
  RAM-adaptive (grows/shrinks off measured RSS, see `CHUNK_SIZE_START`/
  `RAM_TARGET_GB`/`RAM_HARD_GB`) rather than a single guessed constant, but
  the *starting* size still has to be conservative on its own since the
  adaptive loop only reacts *after* a chunk completes.

### Result: LR=4e-4 LLM-only regressed; LR=2e-4 + gradual warmup + human-mix recovered it

First real (not just mechanics-smoke) test at `LR=4e-4`, LLM-only training,
500K-row subsample: `val_macro_pr_auc` dropped from the starting 0.5282 to
~0.503 by the first checkpoint and never recovered, even after a full cosine
anneal to 0 (best 0.50857, final 0.50642). Root cause suspected: too-aggressive
peak LR (2x the original run's own successful `2e-4`) reached almost
immediately (3% warmup) on noisy weak labels, with no anchor to the original
human-tuned signal.

Fix, validated on a 2M-row run (292,523 human train rows + 1,707,477 sampled
`matches_llm` rows, mixed together): `LR=2e-4` (back to the original run's own
peak), `WARMUP_RATIO=0.10` (more gradual ramp), `INCLUDE_HUMAN_TRAIN=True`
(human split mixed into every batch, not LLM-only). Result: `val_macro_pr_auc`
climbed steadily through the whole run — 0.51784 (10%) → 0.51939 (30%) →
0.52100 (50%) → 0.52352 (81%) → **0.52447 (100%, final = best)** — a clean
upward trend (unlike the 4e-4 run's flat/oscillating trajectory), landing
just under the 0.5282 starting point rather than regressing well below it,
and still trending up at the final checkpoint. Memory: RSS flat at ~12GB/32GB
for the entire run (items table this time: 3,449,444 rows, vs 1.05M at the
500K-row scale) — confirms the `maintain_order` fix scales. Cost: ~2.7h of
GPU quota (6.95h/30h used as of this run, refreshes 2026-08-15).

### Full 11.48M-row run: hit Kaggle's 12h session cap, but recovered a new-best checkpoint anyway

Launched with `TARGET_TOTAL_TRAIN_ROWS=None` (all 11,187,780 `matches_llm`
rows + all 292,523 human train rows = 11,480,303 combined). Kaggle GPU
sessions have a **hard 12-hour cap** (not checked before launching — should
have been) — the kernel status ended as `CANCEL_ACKNOWLEDGED` after
consuming ~12h of quota, partway through the single planned epoch.

**The result so far is genuinely good, not a failure**: `val_macro_pr_auc`
climbed cleanly through 6 logged checkpoints before the cutoff — 0.50172
(10%) → 0.51609 (20%) → 0.51726 (30%) → 0.52527 (40%) → **0.52839 (50%,
already above the 0.5282 starting point)** → **0.53229 (60%, best)** — still
rising when the session was cut off around step 3117/4541 (~69%, between the
60% and 70% checkpoints). This checkpoint is a real improvement over the
current baseline (0.5282 val / 0.265 leaderboard) and was successfully
recovered via `kaggle kernels output` (Kaggle does persist `/kaggle/working`
as kernel output even on a hard-cap cancellation, not just on a clean
finish) — sitting at
`kaggle_kernels/lora-llm-continue-train/output/run7_output/lora_llm_continue_train/best/`
locally, and has since been **published** as the new
`maxamsecret/ozon-lora-minilm-finetuned` best (confirmed via the dataset's
own "updated" timestamp + the presence of `train_log_steps.csv`, an artifact
this continue-train pipeline produces that the original one-off publish
didn't have).

**Root cause of not finishing: tokenization, not training, ate the session.**
Exact phase timings from the run's own logs:
- `items.parquet` semi-join + collect (13,095,914 rows needed): **28.5s** — fine.
- Val tokenize (73,131 rows): **173s** (~423 rows/sec).
- **Train tokenize (11,480,303 rows): 34,336s ≈ 9.54 HOURS** (~334 rows/sec)
  — ~80% of the entire 12h session.
- Model load + VRAM probe: ~21s.
- Training itself: ~2.1h for 60% of the epoch → **~3.5h projected for the
  full epoch** — this alone would fit in a 12h session with room to spare.

Val and train tokenize throughput (rows/sec) are similar, so the problem
isn't the tokenizer call itself scaling badly — it's `tokenize_to_memmap`'s
per-chunk join: `item_texts_id1`/`item_texts_id2` (views over the
13,095,914-row items table) get joined against fresh on **every chunk**
(hundreds of chunks across an 11.48M-row pass), so the cost of hashing/
probing a 13M-row table is likely being paid hundreds of times over instead
of once. Fix not yet implemented: do the id→text lookup **once** (e.g. one
big join up front, or precompute a plain dict, before the chunk loop) and
keep the chunk loop purely for bounding tokenizer-call memory, not for
re-joining. This should turn ~9.5h into something much smaller and make a
full epoch comfortably fit in one 12h session — needs validating on a
smaller scale before trusting it on another multi-hour attempt.

## Bi-encoder probe: `ai-sage/Giga-Embeddings-instruct` (3.45B) — good quality, too slow

Ran `kaggle_kernels/giga3b-quantize-benchmark/` (1xT4, ~2.8h) to gate the
"bi-encoder lets us afford a much bigger model" idea before investing in
hard-negative mining. Scored raw (un-fine-tuned) cosine similarity of both
items encoded as documents, on a stratified (category x label) 8,000-pair
subsample of the canonical val split (15,990 unique items, all 20 categories
with both classes, avg 121.3 tokens/item).

| | macro PR-AUC | items/s (T4) | peak VRAM | on disk |
|---|---|---|---|---|
| random-vector floor | 0.2649 | — | — | — |
| baseline (trained) | 0.3710 | — | — | — |
| **Giga-3B raw bf16** | **0.4004** | 1.97 | 12.96GB | 13.8GB (F32) |
| **Giga-3B raw int8** | **0.3973** | 10.10 | 10.97GB | **3.47GB** |
| LoRA cross-encoder (trained) | 0.5282 | 384.7 pairs/s | — | 127MB |

**Quality: promising.** With zero fine-tuning it beats the *trained* baseline
(0.4004 vs 0.3710) and covers ~half of the LoRA cross-encoder's lift over the
floor (+0.135 vs +0.267). int8 quantization is essentially free (-0.0031), and
the int8 save works end-to-end: 3.47GB, fits the 5GB zip, self-contained
(`save()` copies the `trust_remote_code` .py files into the output dir).

**Speed: does not fit, and the premise itself doesn't hold.** The pairs have
almost **no item reuse** — 145,238 unique items for 73,131 val pairs (1.986
items/pair; the full `matches.parquet` is 711K ids / 365,654 pairs = 1.94).
So a bi-encoder does ~2 forward passes per pair against the cross-encoder's 1,
over ~2x the tokens (121 tok/item x2 vs 128 tok/pair). The "encode once,
compare many" advantage that makes bi-encoders cheap in retrieval is absent
here, so the whole "bi-encoder is faster, therefore we can afford a bigger
model" rationale collapses for this candidate set.

Direct FLOPs projection for the stand's H100 (2 * 3.45e9 params * tokens,
more trustworthy than scaling a T4 measurement, since both T4 passes fell back
to eager attention):
- Private: ~275K pairs → ~546K items → 66.3M tok → 4.6e17 FLOPs →
  **17-22 min** at a realistic 35-45% MFU (12.8 min even at an optimistic
  60%) vs the **13 min** limit.
- Public: ~115K pairs → ~228K items → 27.7M tok → **~7 min** vs the **6 min**
  limit.

Both stages miss, Public included, before counting parquet IO, tokenization,
or loading 13.8GB of weights. Shipping bf16 in a custom ≤15GB Docker image
(instead of int8 in the zip) would help — bnb LLM.int8 is *slower* than bf16
on H100 — but not by the 1.3-1.7x needed. Realistic paths if this direction is
resumed: a much smaller Russian bi-encoder (`sergeyzh/BERTA`, 128M, the other
ruMTEB pairClass leader), or distilling Giga-3B into one.

Gotchas from the run:
- `torch.cuda.is_bf16_supported()` returns **True on a T4** (sm_75, no bf16
  tensor cores) under torch 2.10, so the dtype guard did not fire and the
  "full precision" pass ran *emulated* bf16 — 5x slower than int8 (1.97 vs
  10.10 items/s). That ordering is a T4 artifact and reverses on an H100.
  Gate on `torch.cuda.get_device_capability()[0] >= 8`, not on that call.
- Both passes fell back to `attn_implementation="eager"`; sdpa failed to load
  and flash-attn needs sm_80+. Throughput here is a floor, not typical.
- Kaggle truncates kernel logs to the tail: thousands of
  `MatMul8bitLt: inputs will be cast ...` warnings from bitsandbytes evicted
  every early `[LOAD]`/`[DATA]` line. Write results to a JSON in
  `/kaggle/working` after each stage (this kernel does) rather than trusting
  stdout to survive.
- The model card pins `transformers==4.51.0` / `sentence-transformers==5.1.1`;
  the custom code is written against them and the ST pipeline is
  Transformer → Pooling(**mean**) → Normalize, not the latent-attention
  pooling the card's prose advertises.

## FRIDA bi-encoder: per-item pre-tokenization caching (pivot from MiniLM continue-SFT)

Continued-SFT kernel v9 (`lora-llm-continue-train`) was running successfully
past both previous failure points when manually stopped: the time budget a
bigger/different encoder would cost turned out to be affordable, so it made
more sense to spend it exploring architecture than to keep grinding the same
MiniLM cross-encoder. That reopened a question asked in passing — can item
text be pre-tokenized and cached per-item, then just concatenated for
training, instead of re-tokenizing every pair from scratch — and the answer
turned out to depend entirely on architecture. For the MiniLM **cross**-encoder
this doesn't really work cleanly: HF's joint pair-truncation
(`truncation="longest_first"`, splitting the budget across `text`/`text_pair`)
would have to be replicated exactly outside the tokenizer to stay
byte-matched, and `truncate_sequences` — the function that implements it —
turned out not to be reachable as an instance method on the loaded tokenizer
(`hasattr(tokenizer, "truncate_sequences")` is `False`; it only exists as a
standalone function elsewhere in `transformers`). For a **bi-encoder**,
though, the problem doesn't exist at all: each item is tokenized
independently to a fixed `max_length`, full stop — no joint truncation, no
`text_pair`, nothing pair-specific to replicate. That's what tipped the
decision to **ai-forever/FRIDA** (0.8B) over continuing to push on MiniLM.

**Model specifics** (checked directly against the loaded model/tokenizer,
not just the model card — the card's prose is partly wrong): FRED-T5-encoder
based, CLS pooling + L2 norm. The card's prose suggests a T5/SentencePiece
tokenizer; loading it directly shows it's actually `RobertaTokenizer`
(byte-level BPE, `vocab_size=93,651`). Requires a task-specific text prefix —
chosen **`paraphrase: `** after checking the model card rather than guessing:
it names "symmetric paraphrasing related tasks (STS, paraphrase mining,
**deduplication**)" explicitly, which is this task exactly (the alternative
prefixes — `categorize:`/`categorize_sentiment:`/`categorize_topic:`/
`categorize_entailment:` for asymmetric classification, `search_query:`/
`search_document:` for asymmetric retrieval — don't fit a symmetric
duplicate-or-not comparison).

**`max_length=256`**, not FRIDA's 512 hard cap — chosen empirically from a
20K-item local sample tokenized with the exact text formula + prefix:
mean=126.6, std=40.4, p99=202.0, p99.9=223.0, true max=272. 256 covers
~99.99% of items at zero truncation, half the storage/RAM of 512.

**Two separate kernels, not one combined script** — `kaggle_kernels/
frida-tokenize-items-human/` (`items_human.parquet`, ~711K items, plain CPU
session, no queue) and `kaggle_kernels/frida-tokenize-items-llm/`
(`items.parquet` semi-joined to `matches_llm.parquet`'s ids, ~12.38M items,
TPU session for the larger RAM allocation — no TPU compute is actually used,
this is pure CPU tokenization, but TPU quota is a separate 20h/week pool
from GPU's 30h/week, so this doesn't eat the budget training needs). Split
so the two scopes can run in parallel, isolate failures independently, and
match hardware to each scope's actual size. Each publishes its own dataset
(`maxamsecret/frida-tokenized-items-human`, `maxamsecret/frida-tokenized-items-llm`
— renamed from an initial `frida-item-tokens-*` naming to match a dataset
the user had already started creating manually under this name)
of `ids.bin`/`input_ids.bin`/`attention_mask.bin` memmaps + `cache_meta.json`
— training later does a pure id lookup, no tokenizer call, no per-chunk
rejoin against the items table (the exact inefficiency that cost the MiniLM
pipeline 9.5h of a 12h session — see above; a bi-encoder's per-item cache
structurally can't repeat that mistake, since there's no pair-level join
left to do at train time at all).

**Gotcha 1**: Kaggle's TPU image does **not** have `polars` preinstalled
(confirmed the hard way — first combined-script version errored with
`ModuleNotFoundError: No module named 'polars'`), unlike the standard GPU
image. Fix is two-part, not just "add polars to the pip install": the
`pip install` call must run **before** the module-level `import polars as pl`
statement, not inside `main()` — the import executes at module-load time,
before `main()` is ever called, so an install placed inside `main()` is
already too late and the script dies at the top-level import first.

**Gotcha 2**: the TPU image's preinstalled `transformers`/`tokenizers` pair
is internally mismatched for FRIDA's `RobertaTokenizer` specifically — real
v1 run on `frida-tokenize-items-llm` died loading the tokenizer with
`TypeError: RobertaProcessing.__new__() got an unexpected keyword argument
'cls'` (the standard CPU image's preinstalled pair loads the same tokenizer
fine, so this is TPU-image-specific, not a FRIDA/transformers-in-general
bug). First fix attempt — `pip install -U transformers tokenizers` (no
version pin), trusting pip's resolver to land on a mutually compatible
modern pair — turned out to be **not actually fixed**: v2's CPU run
happened to resolve `tokenizers==0.22.2` (stable, works), but the TPU run
resolved `tokenizers==0.23.0rc0` (a **pre-release**) and hit the *exact same*
`TypeError`, since the resolved `transformers==5.15.0` predates that rc's
Rust-binding signature change. Same error, two different root-cause layers —
easy to think it's fixed after seeing the CPU image work. Real fix: pin
**exact** versions (`transformers==5.15.0`, `tokenizers==0.22.2` — the pair
empirically confirmed working on the CPU run) instead of an unpinned `-U`,
removing the resolver's freedom to land on a pre-release. Both scripts now
also log `transformers.__version__`/`tokenizers.__version__` on every run,
which is exactly what surfaced the CPU/TPU divergence.

**Gotcha 3**: `kaggle datasets create -p <dir> -t` requires a `licenses` key
in `dataset-metadata.json` — omitting it (both scripts initially wrote only
`{"id": ..., "title": ...}`) fails late, *after* tokenization completes, with
`Key licenses not found in data` (the CLI's `get_or_fail(meta_data,
"licenses")`, confirmed by reading the installed `kaggle` package's own
source). `human` v1 actually finished tokenizing all 711,304 items in ~4min
(2,906 items/sec, RSS 1.7GB→8.5GB→1.7GB, RAM-adaptive chunking scaled
20K→200K cleanly) and only failed on this last step. Fix:
`"licenses": [{"name": "CC0-1.0"}]` in the metadata dict, matching what
`kaggle datasets init` itself templates.

**Gotcha 4**: `_require_disk_space`'s default 35% safety margin is too
conservative for this call site specifically. `human` v2 succeeded (needed
1.24GB against plenty of free space), but `llm` v3 (tokenizer fix applied,
got all the way through loading 12,384,610 items, RSS 20.75GB, fine) failed
at the disk check: `need 21.53GB, have 20.94GB` — the real data is only
15.95GB (`bytes_per_row` is exact: fixed dtypes/shape, zero estimation
uncertainty, unlike e.g. a text-size guess), so the 35% margin alone pushed
the requirement past Kaggle's ~20GB `/kaggle/working` ceiling despite the
actual write fitting with ~5GB to spare. Fixed by passing `margin=1.15` at
this specific call site in both scripts (15% still covers the two tiny JSON
sidecar files and filesystem overhead, without false-failing against a
known-exact byte count).

**Gotcha 5**: the `kaggle` CLI binary itself is not on PATH on the TPU
image (the standard CPU/GPU image has it preinstalled — confirmed, `human`'s
`kaggle datasets create` call worked with zero setup). `llm` v4, with
gotchas 2-4 all fixed, ran the **entire** ~83-minute tokenization of all
12,384,610 items to completion (RSS peaked at a modest 8.14GB) and only
failed at the very last line: `subprocess.check_call(["kaggle", "datasets",
"create", ...])` → `FileNotFoundError: [Errno 2] No such file or directory:
'kaggle'`. Fixed by adding `pip install kaggle` to the same install-before-
work preamble as the other packages. Deliberately *not* recovered from v4's
already-tokenized `/kaggle/working` output via a local download + reupload
— that would route the cache through the local machine, which this project
explicitly avoids for these caches (Kaggle-side only, per the same reasoning
that ruled out a local round-trip for the MiniLM token cache earlier). TPU
quota is a separate, so-far lightly-used pool, so a clean v5 re-run costs
comparatively little.

Status as of 2026-08-13: v1 of both kernels hit gotchas 2 (llm, before any
tokenization work) and 3 (human, after all tokenization work completed);
`human` v2 fixed and published successfully. `llm` took 4 rounds after that:
v2 hit a deeper layer of gotcha 2 (the `-U` fix resolved a `tokenizers`
pre-release specifically on the TPU image), fixed in v3 by pinning exact
versions; v3 then reached gotcha 4 (disk margin) after successfully loading
all 12.38M items; v4, with gotchas 2-4 fixed, ran the full tokenization to
completion and only then hit gotcha 5 (missing `kaggle` CLI) at the very
last step. v5 (all five gotchas fixed) was pushed and sitting `QUEUED`
(zero TPU time spent) when it turned out the user had *already* manually
recovered v4's tokenized output and published it as
`maxamsecret/frida-tokenized-items-llm` — a different slug than the script's
`frida-item-tokens-llm`, so at that point two more kernel pushes (`llm` v6,
`human` v3) were made to rename both scripts' `DATASET_ID` to match, without
first checking whether the manually-created dataset was already complete.
It was — this cost nothing for `human` (v3 was cheap, ~5.5min, and that name
genuinely didn't exist yet under the new scheme) but `llm` v6 would have
burned another ~83min of TPU time re-tokenizing into a dataset that already
existed, only to fail at the same publish step (`datasets create` on an
existing slug). Caught and cancelled via the Kaggle web UI while still
`QUEUED` — confirmed via two independent status pollers both landing on
`CANCEL_ACKNOWLEDGED` without ever passing through `RUNNING`, so **zero**
TPU time was actually lost on `llm` v6. Lesson: check the actual state of a
resource on Kaggle before re-launching work to (re)produce it, especially
once a human is acting on the same resource in parallel. The
already-published `frida-item-tokens-human` from `human` v2 (the pre-rename
slug) is left in place, not deleted, pending the user's own cleanup.

Final state: both `maxamsecret/frida-tokenized-items-human` (711,304 items)
and `maxamsecret/frida-tokenized-items-llm` (12,384,610 items) are published
and ready — `ids.bin`/`input_ids.bin`/`attention_mask.bin` memmaps +
`cache_meta.json` each, `max_length=256`, `paraphrase: ` prefix. The earlier
combined-script attempt (`kaggle_kernels/frida-tokenize-items/`, one
script doing both scopes sequentially) is superseded by the split and
already confirmed `ERROR` on Kaggle (gotcha 1) — no quota lost beyond that
one failed attempt.

Not yet decided: the training architecture on top of these cached
embeddings (e.g. classifier head over `[emb1, emb2, |emb1-emb2|]` vs. plain
cosine similarity) — out of scope until the tokenization caches exist.
