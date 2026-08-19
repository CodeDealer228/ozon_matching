# Synthesis-ready negative patterns — Товары для животных

## Generation pattern 1

**Generalized rule:** Otherwise comparable pet foods with explicitly different named recipes or recipe numbers are different products.

**Scope:** Food of the same species, food form, brand or product line, and life stage, where both listings explicitly name a recipe, principal protein combination, or stable recipe number.

**Generation operation:** Preserve species, life stage, food form, brand/line, unit weight, and pack configuration; replace the explicit recipe/formula value (for example, chicken-and-fish with liver, chicken-and-scallop with chicken-and-beef, or Berkley No. 7 with No. 6).

**Guards:** Change an explicit product recipe or formula identifier, not an inferred ingredient, a broad protein class, or a merely omitted attribute. Do not change pack count, total weight, texture, or intended animal at the same time.

**Confidence:** strict

**Negative evidence:** pair_0800, pair_0810, pair_0833, pair_0842

**Positive counterexamples:** None observed

**Why reusable:** Multiple food families are labelled negative when the named edible formulation differs; the rule operates on the reusable recipe axis while holding the rest of the listing identity fixed.

## Generation pattern 2

**Generalized rule:** For otherwise comparable cat food, an explicit kitten/young versus adult target combined with a different explicit recipe or formula is a negative pair.

**Scope:** Same-brand wet cat-food listings with explicit cat species, food form, and per-serving format, where both life stage and recipe/formula are stated.

**Generation operation:** Preserve brand, species, wet-food form, unit size, and pack configuration; jointly change the life-stage target from kitten/young to adult (or vice versa) and substitute an explicitly named recipe/formula.

**Guards:** Keep this conjunctive: the sample does not isolate life stage from formula. Do not treat an omitted age field, a generic “for cats” field, or a change in pack count alone as a life-stage negative.

**Confidence:** strict

**Negative evidence:** pair_0827, pair_0835

**Positive counterexamples:** None observed

**Why reusable:** The two independent same-brand cat-food examples consistently combine an explicit incompatible life stage with a distinct named offering, yielding a controlled hard-negative transformation rather than assuming that any age text is decisive by itself.

## Generation pattern 3

**Generalized rule:** A switch between explicit, incompatible functional formulas is negative when the named functional purpose and the formula-specific ingredient/recipe both change within the same pet-product family.

**Scope:** Veterinary or functional food, and animal-care products such as shampoo, whose titles or attributes explicitly state a concrete purpose (for example allergy support, hairball removal, hypoallergenic care, or antiparasitic treatment).

**Generation operation:** Preserve product family, brand/line where available, product form, species scope, and amount; jointly replace the explicit functional purpose and its named formula/active ingredient with a different explicit function/formula.

**Guards:** Do not infer function from marketing wording or a generic care category. This is a conjunctive rule because the sampled negatives also change formula-specific content; do not synthesize a negative from a function word alone.

**Confidence:** strict

**Negative evidence:** pair_0801, pair_0809

**Positive counterexamples:** None observed

**Why reusable:** The evidence spans food and shampoo but shares the same decisive structure: explicitly different treatment purposes backed by distinct formulations, not superficial title variation.

## Generation pattern 4

**Generalized rule:** An explicit cat-versus-dog food mismatch can be used as a negative only for the same named formula and food form, with the remaining identity fields held fixed.

**Scope:** Pet-food listings that explicitly name both species and the same brand plus named formula/line, food form, main recipe, and nominal weight.

**Generation operation:** Preserve the named formula/line, food form, recipe, life stage, and weight; change only the explicit intended species between cat and dog.

**Guards:** Require an explicit species value in reliable title or intended-use text on both sides. Do not use generic taxonomy attributes that conflict with the title. The positive cat-versus-dog pair pair_0829 shows that species text alone is not reliable when the formula and target need differ as well.

**Confidence:** guarded

**Negative evidence:** pair_0805

**Positive counterexamples:** pair_0829

**Why reusable:** It retains the strong same-formula cat/dog evidence while narrowing the operation enough to avoid the contradicted unconditional “different species always means negative” rule.

## Generation pattern 5

**Generalized rule:** Different explicitly stated fit-size variants of the same collar or harness family are negative products.

**Scope:** Collars and harnesses with the same product family and intended equipment type, where both listings state a nominal size and/or a concrete neck or body-fit range.

**Generation operation:** Preserve brand, product family, equipment type, material, colour, and attachment style; replace the stated nominal size and corresponding fit interval with another explicit size variant.

**Guards:** Use coherent physical neck/girth ranges or named size variants from the title/specification, not an unreliable generic taxonomy field. Do not derive a negative from animal-size or load-capacity text alone; the labels contain a positive tenfold load-capacity discrepancy.

**Confidence:** strict

**Negative evidence:** pair_0815, pair_0816

**Positive counterexamples:** None observed

**Why reusable:** Two independent equipment families label different fit ranges as separate variants; the rule targets the purchasable fit specification rather than colour or noisy “intended for” metadata.

## Generation pattern 6

**Generalized rule:** For the same pet-bowl mat design, explicit different length-by-width dimensions identify different size variants.

**Scope:** EVA pet-bowl mats with the same named design, material/structure, purpose, and colour, where both dimensions are explicitly stated.

**Generation operation:** Preserve brand, mat purpose, material/structure, colour, and thickness; replace only the explicit length-by-width size with a different stated mat size.

**Guards:** Restrict this rule to the repeated EVA bowl-mat family and complete two-dimensional product sizes. Do not generalize it to litter trays, lights, or arbitrary pet goods, and do not use package dimensions in place of product dimensions.

**Confidence:** strict

**Negative evidence:** pair_0820, pair_0830

**Positive counterexamples:** None observed

**Why reusable:** Both negatives are otherwise the same EVA bowl-mat offering with a deliberately different usable footprint, so the size axis is isolated for this product family.

## Rejected generalizations

- **“Any cat-food listing versus dog-food listing is negative.”** Rejected: pair_0829 is labelled positive despite explicit cat versus dog food. Use only the narrowly guarded same-formula species operation in Generation pattern 4.
- **“Any physical-dimension mismatch is negative.”** Rejected: pair_0821 is labelled positive despite different named litter-tray models and dimensions. The only strict dimension rule is limited to the repeated EVA bowl-mat family.
- **“A different load-capacity or animal-weight value means different equipment.”** Rejected: pair_0840 is labelled positive although its titles say “up to 5 kg” and “up to 50 kg”; this is plausibly a listing or annotation error.
- **“Generic taxonomy attributes can determine species or equipment type.”** Rejected: pair_0845 calls a dog collar “for cats” in an attribute, and pair_0846 types a harness as a leash. Prefer reliable title and concrete specification text.
- **“Different brand/model wording alone is negative.”** Rejected: pair_0814 is positive despite Tetra versus Tetravital/Borinat wording; no synthetic negative should be created from brand or model text alone.
