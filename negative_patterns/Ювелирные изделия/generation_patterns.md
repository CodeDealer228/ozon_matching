# Synthesis-ready generation patterns — Ювелирные изделия

## Generation pattern 1

**Generalized rule:** An explicit base-metal mismatch is a negative signal: gold, silver, and brass are not interchangeable base materials, even where their surface appearance is similarly light or gold-coloured.

**Scope:** Comparable jewellery or jewellery watches with an explicit base material on both records.

**Generation operation:** Preserve the product family, form factor, model/design, stone configuration, fineness where applicable, and all non-material identity fields; change only the explicitly declared base metal (for example, gold to silver or silver to brass).

**Guards:** Use only an explicit base-material field or unambiguous material text. Do not infer metal from colour, plating, rhodium/oxidation, or the word “silver-coloured”; a coating is not the base metal. Do not use an omitted material as a mismatch.

**Confidence:** strict

**Negative evidence:** pair_0967, pair_0969, pair_0985

**Positive counterexamples:** None observed

**Why reusable:** These independent negatives span rings, jewellery watches, and religious jewellery and consistently distinguish the underlying material rather than colour. The rule removes the one-pair “585 white gold versus 925 silver wedding ring” wording while keeping its decisive axis.

## Generation pattern 2

**Generalized rule:** Explicitly different jewellery product forms or intended wearing locations are negative: for example, ring versus pendant, bracelet-for-wrist versus neck chain, or icon versus wearable pendant.

**Scope:** Records whose type/form and, where present, body location are explicit on both sides.

**Generation operation:** Preserve base metal, fineness, brand/model family, stone configuration, and other identity fields; replace the declared product type or intended wearing location with a mutually exclusive form.

**Guards:** Require a true functional/form-factor conflict, not a loose title synonym (for example, “chain necklace” and “neck chain” may describe the same form). Do not generate from missing or ambiguous type fields.

**Confidence:** strict

**Negative evidence:** pair_0983, pair_0985, pair_0989

**Positive counterexamples:** None observed

**Why reusable:** Product form is a catalogue-defining axis across the sampled negatives, whereas shared metal, seller, and packaging did not overcome an explicit form conflict.

## Generation pattern 3

**Generalized rule:** Two different explicit manufacturer article/model codes denote different jewellery products when the code is presented as the item identity, even if type, metal, fineness, and some stones agree.

**Scope:** Same-brand, same product-family records with normalized, fully specified manufacturer article/model codes on both sides.

**Generation operation:** Preserve brand, product type, base metal, fineness, and compatible descriptive fields; substitute the complete manufacturer article/model code with a different valid code for a different product, and align any dependent design fields to that code.

**Guards:** Do not treat size suffixes, seller SKUs, partial codes, formatting variants, or an omitted code as a different article. Do not use this operation to contradict a known cross-listing of the same manufacturer code.

**Confidence:** strict

**Negative evidence:** pair_0964, pair_0988, pair_0992

**Positive counterexamples:** None observed

**Why reusable:** The negatives show different complete Diamant articles within otherwise very similar gold or silver jewellery; this is a controlled identity edit, not a change to an incidental listing attribute.

## Generation pattern 4

**Generalized rule:** For chain jewellery or earrings, an explicitly incompatible construction specification can be used as a guarded negative when it changes the physical product configuration (named chain weave and fixed length, or fastening/style construction).

**Scope:** Same-form chain/bracelet products with named weave plus fixed non-adjustable length, or earrings with explicit construction/fastening and a conflicting silhouette or stone setting.

**Generation operation:** Preserve product family, base metal, fineness, and brand where possible; change a complete construction configuration, such as double-rhomb 65-cm chain to curb 40-cm chain, or dangling oxidized English-lock earrings to stud/screw-lock earrings.

**Guards:** Change a coherent configuration, not a single unverified detail. Length alone is unsafe (see positive size variants); do not regard a different lock alone as decisive without an additional explicit construction difference. Exclude adjustable-length products and missing construction fields.

**Confidence:** guarded

**Negative evidence:** pair_0978, pair_0983, pair_0984

**Positive counterexamples:** pair_0965, pair_0972, pair_0990

**Why reusable:** The evidence repeatedly contains material construction differences, but they are bundled with other differences and the sample contains positive size discrepancies; retaining the bundle makes the generation operation useful without overstating what has been isolated.

## Generation pattern 5

**Generalized rule:** A fully disclosed, incompatible stone-set configuration is a guarded negative when it changes the primary stone material together with a defining configuration detail such as stone count, carat/size, cut, or an additional stone set.

**Scope:** Same jewellery form with explicit base metal and fineness, where both records disclose complete primary stone configurations.

**Generation operation:** Preserve product type, base metal, fineness, and non-stone identity fields; replace the entire explicit stone configuration (for example, agate plus enamel with diamond setting, or prasiolite with a specified multi-diamond setting), rather than changing a gemstone name alone.

**Guards:** Require an explicit complete configuration on both sides; do not infer “no stone” from a missing field. Do not use gemstone-name-only replacement, and preferably pair the edit with a distinct article/model or independently incompatible design detail. Treat this as guarded because positive labels conflict on gemstone names.

**Confidence:** guarded

**Negative evidence:** pair_0964, pair_0980, pair_0986

**Positive counterexamples:** pair_0960, pair_0993

**Why reusable:** The reusable signal is the whole documented setting, not “different gemstone equals different product”; this preserves the negative evidence while respecting the labelled gemstone counterexamples.

## Generation pattern 6

**Generalized rule:** A gold-alloy colour mismatch (white, yellow, red, or combined gold) is a guarded negative only when coupled with another independently explicit product-defining conflict, such as a complete stone-set configuration or a distinct article/model.

**Scope:** Same product form made from explicitly declared gold on both sides.

**Generation operation:** Preserve product form, gold base metal, fineness, and all other identity fields; change gold colour only together with a compatible second identity-level specification, such as the complete stone setting or manufacturer article/model.

**Guards:** Never use colour words alone, surface colour, plating, or rhodium as gold-alloy colour. Do not create a negative merely from white-versus-red/yellow gold; the paired product-defining conflict is required. Keep fineness unchanged—this rule is not a fineness rule.

**Confidence:** guarded

**Negative evidence:** pair_0955, pair_0976, pair_0980

**Positive counterexamples:** None observed

**Why reusable:** The sampled negatives support differing gold-alloy appearance as part of clearly different jewellery specifications, but do not isolate colour as an unconditional identity axis.

## Rejected generalizations

- **Different fineness/proba is always a negative:** rejected. `pair_0957` is positive despite explicit 585 versus 375 red-gold pendants; `pair_0997` also bundles 375 versus 585 with article/design differences. Never make fineness the sole synthetic edit.
- **Different coating/plating is always a negative:** rejected. `pair_0953` is a positive cross-listing of a gold-plated 925-silver bracelet, and `pair_0983` uses gold coating on both sides despite being negative for form/construction. Coating must not be confused with base metal.
- **A different gemstone name is always a negative:** rejected. `pair_0960` and `pair_0993` are positive despite sapphire-versus-iolite descriptions. Use only the guarded complete stone-set rule above.
- **Different size, weight, brand, or seller is always a negative:** rejected. Positive pairs `pair_0965`, `pair_0972`, and `pair_0990` have stated size differences; `pair_0990` also differs in brand. Negative near-duplicates `pair_0966`, `pair_0982`, and `pair_0998` show that weight/size and listing fields are noisy.
- **Different earring lock or chain length alone is always a negative:** rejected. The construction evidence is confounded, while the positive size variants above show that scalar dimensions alone are not safe identity edits.
