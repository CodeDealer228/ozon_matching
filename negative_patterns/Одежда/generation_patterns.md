# Generation patterns — Одежда

These are derived, generator-facing rules.  They reference the immutable sample and do not change its labels.

## Generation pattern 1

**Generalized rule:** An explicit mismatch of garment product type or functional role is a negative signal when the two listings describe distinct apparel products rather than the same identified model or a coordinated set.

**Scope:** Garments whose explicit type/name places them in different functional roles (for example, outerwear versus a sweatshirt, sleepwear versus a skirt, or a shirt versus a polo); it is not a rule for merely different marketing names within one product family.

**Generation operation:** Preserve the intended wearer segment and all non-decisive context, then replace the explicit garment type/function with a distinct apparel role.  Do not retain a common SKU, model identifier, or an indication that the pieces are one set.

**Guards:** Both sides must explicitly identify their garment types/functions.  Exclude coordinated garments and records that share a brand/line/model identity strongly enough to be variants of one offer.  Do not infer a type from an omitted attribute.  The sampled positive shorts-versus-bolero pair shows that a type difference by itself is not unconditional.

**Confidence:** strict

**Negative evidence:** pair_0600, pair_0630, pair_0642, pair_0646, pair_0647

**Positive counterexamples:** pair_0611

**Why reusable:** The decisive axis is the reusable product-type/function field, not any particular blouse, nightdress, jacket, shirt, polo, jeans, or trousers.  The guards narrow the rule to clearly distinct products and exclude the observed cross-type positive.

## Generation pattern 2

**Generalized rule:** A pair is a strict negative when it combines an explicit incompatible intended-wearer segment with an explicit mismatch of garment type/function.

**Scope:** Listings that explicitly distinguish adult women, adult men, or children, and also name different garment types/functions.  This is intentionally conjunctive: the sample does not establish wearer segment alone as a safe negative axis.

**Generation operation:** Preserve the surrounding product context, then change both the explicit intended-wearer segment and the garment type/function to a different, non-set apparel product.

**Guards:** Require explicit values on both sides; never derive sex or age from a size alone.  Keep the type/function mismatch as a second decisive axis and exclude unisex claims, compatible one-set pieces, and shared-SKU/model variants.  If only one of the two axes is available, do not use this rule.

**Confidence:** strict

**Negative evidence:** pair_0613, pair_0644

**Positive counterexamples:** None observed

**Why reusable:** It captures the robust joint distinction between target wearer and product role without claiming that gender, age, or a size notation alone identifies a different item.

## Generation pattern 3

**Generalized rule:** An item explicitly identified as a non-apparel product is negative against an explicitly identified garment.

**Scope:** Category-leakage records where the item type itself names a non-clothing product and the other side names an apparel type.

**Generation operation:** Keep the apparel item unchanged and replace the other item with an explicitly non-apparel product type; do not create this condition from ambiguous text or missing type fields.

**Guards:** This is not a clothing-to-clothing hard-negative template and should normally be excluded from apparel training generation.  Use only for data-quality filtering or if cross-category negatives are deliberately required.

**Confidence:** guarded

**Negative evidence:** pair_0632

**Positive counterexamples:** None observed

**Why reusable:** The reusable signal is an explicit taxonomy conflict, but the evidence is a single category anomaly rather than a normal apparel comparison.

## Rejected generalizations

- **Size is not an unconditional negative axis.** Positives `pair_0616` (different stated sizes) and `pair_0622` (80I versus 75K) contradict it; mixed size systems and incomplete cards also appear throughout the sample.
- **Colour is not an unconditional negative axis.** Positives `pair_0601`, `pair_0619`, and `pair_0622` have different stated colours, so use colour only as a preserved context field, not the decisive generated edit.
- **Brand is not an unconditional negative axis.** `pair_0619` is positive despite the explicit Ava brand on only one side, and many negative pairs combine brand, type, material, and colour differences rather than isolating brand.
- **Material/composition, season, cut, and generic garment-family wording are unsupported as standalone axes.** The negative examples that differ on them also differ in type, colour, size, model, or brand; no safe single-axis rule is isolated.  In particular, do not turn cases such as `pair_0603`, `pair_0604`, `pair_0627`, or `pair_0634` into material-, colour-, or model-only rules.
- **“Any two different garment types are negative” is too broad.** Positive `pair_0611` pairs shorts with a bolero; use the guarded functional-role conditions in Generation pattern 1 instead of an unqualified type mismatch.
- **The non-apparel example is a category anomaly, not a normal clothes pattern.** `pair_0632` contains an explicit diamond-mosaic kit in `Одежда`; it supports filtering, not routine synthetic apparel generation.
