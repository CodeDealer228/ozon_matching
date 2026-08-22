---
name: error-audit-runner
description: Run the repository's hard FP/FN audit pipeline, batch subagent work, validate evidence-grounded outputs, and aggregate severity trends.
---

# Error Audit Runner

Use this skill for the BERTA/FRIDA hard-error audit in this repository.

## Preconditions

One source artifact must exist:

- `artifacts/input/all_fp_fn_full_attributes_and_exact_inputs.parquet`, or
- `artifacts/input/all_fp_fn_full_attributes_and_exact_inputs.jsonl`.

If neither exists, stop and ask the user to provide it.

## Pipeline

Run sequentially:

```bash
python scripts/prepare_audit.py
python scripts/validate_audit_sample.py
python scripts/make_batches.py
```

Only after sample validation succeeds, classify batches with the configured
`error_auditor` subagent, up to 5 concurrently.

Then:

```bash
python scripts/validate_batch_outputs.py
python scripts/aggregate_audit.py
```

## Safety / evidence rules

Never:
- modify source labels;
- resample because labels look odd;
- invent product facts;
- use outside web product knowledge;
- label truncation as causal without comparing FULL ORIGINAL to focal visible input.

Every evidence snippet in subagent output must be copied exactly from supplied input fields.
