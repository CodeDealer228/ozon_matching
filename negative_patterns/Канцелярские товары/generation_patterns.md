## Generation pattern 1

**Generalized rule:** An explicit functional size or standardized format mismatch is a negative when it changes the compatible physical variant of the same stationery product family.

**Scope:** Cutting mats, textbook covers, labels, divider sheets, and comparable products whose dimensions or standard format determine fit or usable area.

**Generation operation:** Preserve the product family, material, construction, intended use, and pack configuration; replace one explicitly stated size or format with another valid, incompatible variant (for example A2 to A3, A4 to A4+, or one stated label dimension to another).

**Guards:** Both dimensions/formats must be explicit and must be a product-defining compatibility value. Do not infer a mismatch from a missing dimension, and do not change pack count at the same time.

**Confidence:** strict

**Negative evidence:** pair_0359, pair_0369, pair_0391, pair_0396

**Positive counterexamples:** None observed

**Why reusable:** The evidence covers several independent stationery families in which the differing dimension is stated and changes the physical variant, rather than merely changing wording.

## Generation pattern 2

**Generalized rule:** An explicit colour or named shade mismatch is a negative for otherwise equivalent colour-variant products.

**Scope:** School bags with the same construction, single-colour art supplies, ink refills, and self-adhesive film; use only product families sold as specific colour variants.

**Generation operation:** Preserve product type, brand/model line where known, material/formula, dimensions, capacity, and pack configuration; substitute one explicit colour/shade code with another explicit colour/shade variant.

**Guards:** The colour must be a deliberately selected product variant, not a generic display colour or an omitted attribute. For a multi-attribute product, do not alter any construction, size, pack, or design field together with the colour. Do not rely on the near-duplicate noisy labels in pair_0374, pair_0375, or pair_0377.

**Confidence:** strict

**Negative evidence:** pair_0370, pair_0371, pair_0372, pair_0376, pair_0379

**Positive counterexamples:** None observed

**Why reusable:** Repeated negatives show the same controlled variant axis across bags, paint/ink, and film, including two independently sampled otherwise-equivalent backpacks.

## Generation pattern 3

**Generalized rule:** Different explicit functional text or personalisation makes an otherwise comparable text-bearing stationery item a negative.

**Scope:** Ready-made stamps, safety/information signs, personalised stationery gifts, and purpose-labelled document folders.

**Generation operation:** Preserve the physical product type, format, material, construction, and all non-text specifications; replace the operative printed message, recipient name, or designated document purpose with a different explicit value.

**Guards:** Apply only when the text is the product's operative content or promised personalisation, not incidental marketing copy, a brand, or a decorative theme. Keep the same text-bearing item family; a stamp must remain a stamp, a sign a sign, and so on.

**Confidence:** strict

**Negative evidence:** pair_0360, pair_0364, pair_0381, pair_0394

**Positive counterexamples:** None observed

**Why reusable:** These independent examples all make the literal message or name part of what the buyer receives and uses, so the axis can be changed without conflating it with generic product-description text.

## Generation pattern 4

**Generalized rule:** A daily planner with a different explicit page-organisation specification should be treated as a different variant only as a coherent planner-specification bundle.

**Scope:** Products explicitly described as daily planners/"ежедневник" with stated page ruling and, where supplied, dating and cover specifications.

**Generation operation:** Preserve the planner product type, format, sheet count where known, and brand/model family where known; substitute a fully specified alternative page-organisation bundle (for example ruling together with its stated dating/cover variant), rather than editing an isolated field.

**Guards:** The negative pairs confound ruling with other variant fields: pair_0350 also changes cover properties and pair_0386 also changes dating and cover. Therefore do not generate “ruling alone is negative”, “dating alone is negative”, or “cover alone is negative” from this sample. Require explicit values on both sides and keep the result within the daily-planner family; the positive pair_0363 shows that a similarly named notebook and daily planner can be matched despite different structured descriptions.

**Confidence:** guarded

**Negative evidence:** pair_0350, pair_0386

**Positive counterexamples:** pair_0363

**Why reusable:** Both negatives show that page organisation participates in a fixed planner variant, but the evidence supports only the guarded multi-field specification, not a standalone rule for any one of its fields.

## Generation pattern 5

**Generalized rule:** An explicitly different document-folder construction is a negative only when the construction itself is fully specified and functionally different.

**Scope:** Document folders with stated retention/closure construction, such as fixed file sleeves versus a zip envelope or a lever-arch registrar.

**Generation operation:** Preserve document-storage use, format, material, capacity class where known, and pack configuration; switch to another explicit construction variant with different document-retention behaviour.

**Guards:** Pair_0353 establishes fixed file sleeves versus a zip envelope; pair_0383 also changes spine width, mechanism, material, capacity, and pack size. Consequently use an explicit construction change only when both variants are independently specified, and do not infer a rule from a generic “folder” record (pair_0385) or use pack count/spine width alone.

**Confidence:** guarded

**Negative evidence:** pair_0353, pair_0383

**Positive counterexamples:** None observed

**Why reusable:** Folder construction is a reusable functional axis, but the sampled records do not isolate it cleanly enough to make an unconditional construction rule.

## Rejected generalizations

- **Pack count or quantity is always a negative.** This is contradicted by positive pair_0395 (one named 12-pencil set versus two 12-pencil sets) and positive pair_0384 (one note block versus four sticker blocks). Negative pair_0387 also changes folder construction/capacity, and pair_0383 changes several fields, so neither isolates count.
- **A missing attribute can be turned into a mismatch.** Pairs pair_0362 and pair_0368 compare explicit coloured A4 paper with an underspecified “printer paper”; the missing colour/format on the latter is not evidence for a standalone paper-colour or paper-format rule.
- **Different brands are always different products.** Pair_0398 has different crayon brands, but it does not isolate brand from the product presentation; many positive pairs also have incomplete or inconsistent branding. Brand must be preserved as an identity field, not used as a synthetic negative axis on this evidence.
- **Planner dating, ruling, or cover may each be changed alone.** The negative planner evidence is multi-difference (pair_0350 and pair_0386), while positive pair_0363 has conflicting structured planner/notebook descriptions. Only the guarded coherent bundle in Generation pattern 4 is supported.
- **Chalk colour or cross-section alone is always a negative.** Pair_0388 changes white/coloured composition and round/square cross-section together, so the sample cannot split either field into its own rule.
- **Near-identical textual listings are reliable negatives.** The audit flags pair_0356, pair_0374, pair_0375, pair_0377, pair_0382, pair_0390, and pair_0392 as noisy, contradictory, or insufficiently distinguished; none should seed generation rules.
