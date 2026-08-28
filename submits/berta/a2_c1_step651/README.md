# BERTA + matched control C1, step 651

Local ODS diagnostic submission assembled from the matched C1 arm of the
BERTA A2 experiment. It is intentionally not uploaded automatically.

## Purpose

C1 and T1 used the same sources, row order, weights, seeds, optimizer steps,
and evaluation schedule. C1 replayed the original human-negative B item while
T1 replaced it with synthetic B'. Submitting C1 therefore separates the effect
of the general retraining recipe from the effect of A2 synthetic rewrites.

## Model

- encoder: verified top BERTA checkpoint after the source LoRA merge;
- adapter and classifier: C1 `checkpoint_step_0651_eval`;
- inference: BERTA cross-encoder, mean pooling, pair length 256;
- packaging shell and offline wheels: reused byte-for-byte from the earlier
  ODS-tested BERTA submission with public test score 0.47.

## Validation

At step 651 on the untouched canonical validation:

- macro AP: `0.7463035491`;
- micro AP: `0.8068742173`.

The archive is only a diagnostic candidate. It should be submitted after the
team confirms that spending one of the remaining daily ODS slots is justified.
