# Generation patterns — Хобби и творчество

## Generation pattern 1

**Generalized rule:** Content-defined hobby, educational, and craft products are different offers when both listings explicitly name different designs, subjects, learning targets, or edition titles.

**Scope:** Diamond-painting kits, activity kits, scratch-art kits, speech-therapy games, themed decor kits, stickers, and workbooks whose named content is part of the saleable product identity.

**Generation operation:** Preserve the product family, format, physical configuration, and any shared line or brand; replace one explicit content identity with a different explicit content identity (for example, a pictured subject, phoneme, learning topic, or kit design).

**Guards:** Both sides must contain a specific, conflicting content identifier; do not infer a mismatch from a generic title or an omitted subject. Do not use a mere broad taxonomy label such as “animals” as the content identity.

**Confidence:** strict

**Negative evidence:** pair_0853, pair_0858, pair_0865, pair_0885, pair_0893, pair_0897, pair_0898

**Positive counterexamples:** pair_0859, pair_0889, pair_0894

**Why reusable:** The negative pairs repeatedly distinguish two concrete editions while their generic product format remains comparable. The listed positives have a generic or underspecified side, which is why explicitness is a required guard rather than a reason to discard the rule.

## Generation pattern 2

**Generalized rule:** Precisely specified craft consumables are different variants when their explicit manufacturer colour/article codes conflict, even if the broad product line, size, and pack format agree.

**Scope:** Beads, embroidery floss, sewing thread, and comparable colour-coded consumables sold by a code or swatch identifier.

**Generation operation:** Preserve brand, consumable type, material, nominal size/thickness, length or weight, and pack count; change only an explicit unambiguous colour or article code to another code in the same line.

**Guards:** Require the code or the paired explicit colour values to be present on both sides and genuinely conflicting. Do not generate from truncated, contradictory, or merely broad colour fields; do not treat a colour adjective without a stable code as sufficient when the rest of the record is ambiguous.

**Confidence:** strict

**Negative evidence:** pair_0870, pair_0884, pair_0886

**Positive counterexamples:** None observed

**Why reusable:** These pairs show the same kind of consumable with matching nominal specifications but distinct colour identities. This yields a clean single-axis edit when a reliable code is available.

## Generation pattern 3

**Generalized rule:** A multipack is a different offer when the explicitly stated total quantity of otherwise identical complete kits or consumables differs.

**Scope:** Repeated complete hobby kits and repeatable consumable packs, including modelling compounds and bead packages.

**Generation operation:** Preserve item identity, contents per unit, material, colour assortment, and variant; change the explicit multiplier, unit count, or total amount (for example, one versus two identical 10 g packs).

**Guards:** Calculate and state the total sellable quantity consistently on both sides. Exclude uncertain unit descriptions, component counts that are not sellable units, and listings where one side only reports a per-unit quantity.

**Confidence:** strict

**Negative evidence:** pair_0854, pair_0881

**Positive counterexamples:** None observed

**Why reusable:** The sampled negatives distinguish otherwise matching items by how many complete units the buyer receives, a directly controllable generation axis.

## Generation pattern 4

**Generalized rule:** Rigid craft blanks, surfaces, and components are different products when their explicit functional dimensions or format differ.

**Scope:** Buttons, fixed-shape decorative pieces, jewellery findings, artist canvases, and material sheets whose usable size is the product specification.

**Generation operation:** Preserve product type, material, shape/design, brand or line where known, and pack quantity; replace one explicit usable dimension or standard sheet format with another.

**Guards:** Use dimensions of the item itself, not package dimensions. Restrict this rule to rigid or fixed-format goods; do not apply it to flexible photo corners, elastic, or other listings where a nominal dimension can describe packaging, an orientation, or a non-identity detail.

**Confidence:** strict

**Negative evidence:** pair_0862, pair_0874, pair_0878, pair_0890, pair_0895

**Positive counterexamples:** pair_0868

**Why reusable:** Across several fixed-format products, a size change is an actual change in the usable item. The positive photo-corner pair demonstrates why the scope excludes flexible accessories and avoids treating every number in a listing as an identity field.

## Generation pattern 5

**Generalized rule:** Otherwise comparable knitting-needle sets can be generated as negatives by changing the explicitly stated needle gauge.

**Scope:** Single-purpose knitting-needle sets with a stated construction, length, material, and piece count.

**Generation operation:** Preserve needle type (for example, double-pointed), material, length, and number of needles; change only the explicit diameter.

**Guards:** Both diameters must be explicit and use the same measurement convention. Do not apply this to mixed-gauge sets, unspecified needles, or pairs that also change construction, length, or count.

**Confidence:** guarded

**Negative evidence:** pair_0872

**Positive counterexamples:** None observed

**Why reusable:** Gauge determines how the tool is used and the evidence isolates a 3 mm versus 2 mm change well, but the sample contains only one such needle pair.

## Generation pattern 6

**Generalized rule:** Artist easels with explicitly different structural subtypes are different tools even when their general purpose is the same.

**Scope:** Easels where the listing identifies the construction, such as lyre versus tripod.

**Generation operation:** Preserve the product family, material class, and intended use; replace one explicit structural subtype with another.

**Guards:** Require a named structural form on both sides. Do not use brand, wood shade, height, shelf presence, or a vague “art easel” label as a substitute for an explicit subtype.

**Confidence:** guarded

**Negative evidence:** pair_0857

**Positive counterexamples:** None observed

**Why reusable:** The functional construction changes the tool configuration, but the evidence pair also varies other listing details, so the rule is deliberately guarded.

## Generation pattern 7

**Generalized rule:** A diamond-painting kit is a controlled hard negative when multiple explicit core configuration fields jointly conflict.

**Scope:** Fully specified diamond-painting kits with a stated canvas size, coverage type, and support/base.

**Generation operation:** Preserve the product type and make all edited values explicit; change at least two core configuration axes among canvas size, full versus partial coverage, and supported/framed versus unsupported base.

**Guards:** Do not claim that any one of these axes alone is sufficient from this sample. Do not use generic titles, incomplete listings, or records with contradictory coverage/support fields. Keep named artwork unchanged when testing configuration alone.

**Confidence:** guarded

**Negative evidence:** pair_0869

**Positive counterexamples:** pair_0859, pair_0894

**Why reusable:** The negative pair differs in several material configuration fields, so the safe reusable conclusion is conjunctive. The positives show that a generic diamond-mosaic listing cannot establish a reliable negative solely from a size or configuration reading.

## Generation pattern 8

**Generalized rule:** Mosaic play sets with different explicit component inventories are different sets.

**Scope:** Peg or tile mosaics whose element count, element-size range, and board count are stated as the set contents.

**Generation operation:** Preserve the mosaic type, brand/line where known, and colour assortment; change the explicitly stated internal inventory, such as total elements, element sizes, or number of boards.

**Guards:** Treat the inventory as a conjunction when more than one component field changes. Do not convert a malformed attribute field into a count, and do not apply the rule when only an age recommendation or package size differs.

**Confidence:** guarded

**Negative evidence:** pair_0860

**Positive counterexamples:** None observed

**Why reusable:** The buyer receives a materially different play set when the stated pieces and boards change, but the evidence changes several inventory fields together and therefore does not justify independent unconditional component-count rules.

## Rejected generalizations

- **Any different dimension is a negative.** `pair_0868` is positive despite differing dimension strings for photo corners; flexible/accessory dimensions are not necessarily product identity.
- **Any different picture, theme, or title is a negative.** `pair_0859`, `pair_0889`, and `pair_0894` are positive where one side is generic or underspecified. Generate a content mismatch only when both sides explicitly name conflicting content.
- **Any colour word or colour field is a negative.** `pair_0879` has contradictory colour information, while `pair_0882` and `pair_0883` have otherwise near-identical products with uncertain or incomplete colour evidence. Use only stable explicit code/value conflicts under Generation pattern 2.
- **Matching diamond-mosaic size and configuration proves identity.** `pair_0871`, `pair_0873`, and `pair_0876` are labelled negative despite near-matching or conflicting configuration data; a complete product identity cannot be inferred from these fields alone.
- **A different seller or brand proves a different product.** `pair_0855`, `pair_0880`, and `pair_0899` are positive cross-listings with different brand/seller presentation.
