## Generation pattern 1

**Generalized rule:** Otherwise matching air fresheners are a negative pair when their explicitly named fragrance variants conflict.

**Scope:** Manual aerosol air fresheners with the same product family, delivery form, nominal volume, and pack configuration.

**Generation operation:** Preserve the brand/product line, aerosol form, volume, and unit count; replace one explicit fragrance-variant value with a different explicit fragrance-variant value.

**Guards:** Use named variants (for example, lily-of-the-valley versus anti-tobacco), not a generic descriptor such as “fresh”. Do not generate from a card whose title and fragrance attribute contradict each other; do not infer a fragrance from colour or a missing field.

**Confidence:** strict

**Negative evidence:** pair_0162, pair_0165, pair_0176, pair_0179

**Positive counterexamples:** None observed

**Why reusable:** The four negatives independently preserve the air-freshener family and 300 ml format while changing the advertised scent variant, so this is a product-variant axis rather than a one-off name difference.

## Generation pattern 2

**Generalized rule:** Otherwise matching laundry detergents are a negative pair when an explicit net mass or volume identifies different consumer pack sizes.

**Scope:** The same detergent brand, product variant, detergent form, and single-pack configuration, with a stated net content on both cards.

**Generation operation:** Preserve brand, product line, formula/target laundry type, form, and pack count; change only the explicit net mass or volume to another marketed pack size.

**Guards:** Both values must be comparable net contents in the same unit (or safely convertible units), not shipping weight, dimensions, a missing value, or a total that also changes the number of units. Do not use this rule when the product variant itself is uncertain.

**Confidence:** strict

**Negative evidence:** pair_0163, pair_0178, pair_0186, pair_0188, pair_0198

**Positive counterexamples:** None observed

**Why reusable:** Several detergent families are labelled negative at materially different explicit pack sizes, including 4.5 versus 3 kg, 2.4 versus 0.4 kg, 3 versus 6 kg, and 2 versus 0.4 kg.

## Generation pattern 3

**Generalized rule:** Otherwise matching household-chemical products are a negative pair when the explicitly stated number of sellable units in the offer differs.

**Scope:** The same product variant and per-unit size, with an explicit unit count or pack count on both cards.

**Generation operation:** Preserve brand, formula/variant, product type, fragrance where applicable, and per-unit quantity; change only the number of units or packs in the offer.

**Guards:** Require two unambiguous numerical counts for the offer. Do not treat omitted counts, shipping dimensions, or an internally inconsistent listing as a count mismatch; do not combine the edit with a fragrance, formula, or per-unit-size change.

**Confidence:** strict

**Negative evidence:** pair_0152, pair_0154, pair_0161, pair_0173, pair_0181, pair_0187, pair_0190, pair_0195

**Positive counterexamples:** None observed

**Why reusable:** Repeated negatives cover floor cleaner, fabric conditioner, moth plates, dish liquid, cleaning wipes, and toilet blocks, while the individual product variant remains identifiable and only the sellable quantity changes.

## Generation pattern 4

**Generalized rule:** Otherwise comparable specialised cleaners are a negative pair when their explicit intended surface or soil-removal task is mutually different.

**Scope:** Cleaning products whose cards explicitly name a specialised target, such as acrylic-bath limescale, glass, kitchen grease, dishwashing, or post-construction cement removal.

**Generation operation:** Preserve the broad cleaning-product context and, where applicable, brand and presentation; replace one explicit specialised purpose with a different non-overlapping purpose/surface/soil-removal task.

**Guards:** Generate only from explicit, non-overlapping functional statements. Do not use broad labels such as “universal”, infer purpose from the name alone, or treat a product marketed for several stated surfaces as incompatible with one of those stated surfaces.

**Confidence:** strict

**Negative evidence:** pair_0164, pair_0170, pair_0191, pair_0193

**Positive counterexamples:** None observed

**Why reusable:** The negative evidence repeatedly separates concrete cleaning jobs rather than merely changing brands or package text: anti-limescale versus degreaser, acrylic-bath cleaner versus degreaser, glass cleaner versus bath cleaner, and dishwashing liquid versus cement remover.

## Generation pattern 5

**Generalized rule:** An air-freshener dispenser device and an explicitly identified compatible refill are negative when sold as distinct product types.

**Scope:** Automatic air-freshener systems where one card is the battery-powered device and the other is a refill canister/block for that system.

**Generation operation:** Preserve brand/system family and nominal refill capacity; change the product role from complete powered dispenser to explicitly labelled refill, or vice versa.

**Guards:** Both roles must be explicit in the cards. Do not apply to two refills, two complete devices, or a card merely described as “automatic”; preserve compatibility-system naming when it is available. The sample also changes fragrance, so this evidence does not isolate product role from every other attribute.

**Confidence:** guarded

**Negative evidence:** pair_0151

**Positive counterexamples:** None observed

**Why reusable:** Device versus consumable is a transferable SKU distinction, but it is retained as guarded because only one sampled negative demonstrates it and that pair contains an additional fragrance difference.

## Generation pattern 6

**Generalized rule:** Different explicit manufacturer product codes within the same professional cleaning line may define different SKUs.

**Scope:** Professional cleaning products whose titles or manufacturer fields carry a structured variant/model code and otherwise identify the same line.

**Generation operation:** Preserve the brand/line, broad application, and presentation; replace one explicit product code with another real code from the same line.

**Guards:** Use only codes explicitly supplied by the source catalog and retain the exact code text. Do not fabricate codes, infer a code from packaging, or use this rule if other decisive changes (formula, net content, or application) cannot be held constant; the sampled pair contains such additional differences.

**Confidence:** guarded

**Negative evidence:** pair_0194

**Positive counterexamples:** None observed

**Why reusable:** Codes are a compact identity field for professional product variants, but the evidence is only one confounded negative, so it is a constrained augmentation signal rather than a strict rule.

## Rejected generalizations

- **Any difference in stated laundry target is negative.** Rejected: `pair_0197` is positive despite “for children's laundry” versus “for coloured fabrics”; the attributes are noisy enough that this axis cannot be unconditional.
- **Any difference in listed appliance compatibility is negative.** Rejected: `pair_0180` is positive although one card says washing machines and the other lists coffee machines, ovens, and ironing systems.
- **Any product-line word, concentration label, or packaging description denotes a different SKU.** Rejected: `pair_0153` is positive for Fairy versus Fairy Platinum at the same named fragrance and volume; `pair_0171`, `pair_0172`, `pair_0177`, `pair_0184`, `pair_0185`, `pair_0189`, `pair_0196`, and `pair_0199` are caveated negatives that look like the same product or a near-duplicate.
- **A missing attribute is a conflicting attribute.** Rejected: the caveated pairs `pair_0171`, `pair_0172`, `pair_0177`, `pair_0184`, `pair_0185`, `pair_0189`, `pair_0196`, and `pair_0199` show that incomplete card fields cannot safely supply a negative edit.
- **Any reported fragrance mismatch is negative.** Rejected without title-level corroboration: `pair_0192` has matching “mystic mulled wine” names but a contradictory “ocean oasis” fragrance attribute on one card.
