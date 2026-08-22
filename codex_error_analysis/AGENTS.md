# E-CUP 2026 — Hard FP/FN audit project instructions

## Mission

Audit hard false positives / false negatives for champion BERTA and FRIDA.

The goal is NOT to optimize human validation score.
The goal is to discover:
- likely annotation/business-rule issues;
- real recurring model failures;
- truncation-caused failures;
- critical attribute mismatches visible to the model;
- semantic/alias failures;
- how the error mix changes as we move away from the very top of the hard-error ranking.

## Source artifact

Preferred input:

`artifacts/input/all_fp_fn_full_attributes_and_exact_inputs.parquet`

Fallback:

`artifacts/input/all_fp_fn_full_attributes_and_exact_inputs.jsonl`

This artifact must come from the BERTA/FRIDA exact-input explorer and contain:
- row-level BERTA/FRIDA scores;
- human target;
- full original item names/attributes;
- exact visible decoded BERTA and FRIDA inputs.

If neither file exists, STOP and ask the user to place one of them in `artifacts/input/`.
Do not silently substitute `top_errors_*` files: they do not cover the intended severity bands.

## Fixed taxonomy

Read `TAXONOMY_RU.md`.
Do not let individual subagents invent incompatible primary classes.

## Sampling design

Default config is in `config/audit_config.json`.

Four focal groups:
- BERTA FP
- BERTA FN
- FRIDA FP
- FRIDA FN

Four severity bands by rank within that model/error group:
- 1–50
- 51–150
- 151–400
- 401–1000

Default sample: 25 random records per group × band with stable seed.
Expected target if every band is populated: 4 × 4 × 25 = 400 audit assignments.

Why bands instead of only top-N:
we explicitly want to test whether likely label issues are concentrated among the
most extreme errors and real model failures become more common deeper in the ranking.

Do not change the sample merely because some labels look strange.

## Phase 1 — prepare immutable audit sample

Run:

`python scripts/prepare_audit.py`

Then:

`python scripts/validate_audit_sample.py`

Then:

`python scripts/make_batches.py`

Do not start LLM classification until validation passes.

## Phase 2 — subagent classification

Use `.codex/agents/error-auditor.toml`.

Spawn up to 5 `error_auditor` subagents concurrently.
Assign exactly one `audit_batches/batch_XXX/` directory to each active subagent.
When one finishes, assign the next unfinished batch.

Every subagent:
- reads the entire batch;
- classifies every input record;
- writes only `output.jsonl` in its assigned directory;
- uses full original attributes AND focal model visible inputs;
- treats the other model only as supporting evidence;
- does not use web search;
- does not modify source data.

## Phase 3 — validation

After all batches finish:

`python scripts/validate_batch_outputs.py`

If validation fails, repair/re-run only the affected batches.

## Phase 4 — aggregation

Run:

`python scripts/aggregate_audit.py`

Expected outputs include:
- `outputs/classified_audit.parquet`
- `outputs/summary_by_primary_class.csv`
- `outputs/summary_by_severity.csv`
- `outputs/summary_by_category.csv`
- `outputs/summary_by_subtype.csv`
- `outputs/hypothesis_label_noise_by_depth.csv`
- `outputs/manual_review_queue.jsonl`
- `outputs/manual_review_queue.html`
- `outputs/FINAL_REPORT.md`

## Required final main-agent review

Before declaring completion, the main agent must personally inspect:
- all `likely_label_issue` cases with confidence >= 0.90;
- all `truncation_related` cases with confidence >= 0.85;
- all classifications with confidence < 0.70;
- at least 10 random `critical_attribute_mismatch` cases.

Look for systematic subagent misinterpretation or evidence hallucination.

Do NOT train a model, generate synthetic pairs, or change labels in this task.
This task is an error audit only.
