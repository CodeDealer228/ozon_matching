# Synthesis-ready negative patterns — Бытовая техника

## Generation pattern 1

**Generalized rule:** For otherwise comparable cooking appliances, an explicit mismatch in the heating-energy configuration is a negative: gas versus induction for a hob, or fully gas versus gas-electric for a freestanding cooker.

**Scope:** Hobs compared with hobs and freestanding cookers compared with freestanding cookers; do not use this rule to change the appliance family itself.

**Generation operation:** Preserve the appliance family, installation form, brand/model family where applicable, number of burners, and cosmetic fields; change the explicit heating technology/configuration from gas to induction, or from fully gas to a stated gas-electric configuration.

**Guards:** Both listings must explicitly state their heating technology or burner/oven configuration. Do not infer a technology from a missing field, and do not substitute a whole hob for a freestanding cooker.

**Confidence:** strict

**Negative evidence:** pair_0125, pair_0139

**Positive counterexamples:** None observed

**Why reusable:** The two independently labelled negatives cover both a hob and a cooker, while isolating a product-defining energy/configuration conflict rather than incidental colour, brand, or control differences.

## Generation pattern 2

**Generalized rule:** Complete appliances with different explicit manufacturer model codes are negative when the code mismatch is accompanied by an explicit product-defining configuration mismatch.

**Scope:** Same-brand major appliances with the same broad product family, specifically the sampled dishwashers, refrigerators, and dryers; exclude ovens and replacement parts.

**Generation operation:** Preserve the brand and broad appliance family; change the model code together with one explicitly stated configuration that distinguishes the SKU, such as dishwasher width/capacity, refrigerator defrosting system/capacity, or dryer energy class/programme set.

**Guards:** Both model codes and the accompanying configuration values must be explicit. Generate the two changes together; neither a model-code edit nor a specification edit alone is supported here. Do not apply to ovens, compatible parts, or listings with sparse specifications.

**Confidence:** strict

**Negative evidence:** pair_0106, pair_0110, pair_0145

**Positive counterexamples:** pair_0148

**Why reusable:** Multiple complete-appliance negatives show that a model identity plus a materially different configuration identifies a distinct SKU. The oven counterexample is excluded from the scope, so the rule does not turn code differences into a universal signal.

## Generation pattern 3

**Generalized rule:** Products with explicitly incompatible primary functions are negative even if their wording, colour, or room-use context overlaps.

**Scope:** Listings whose explicit product type identifies mutually exclusive functions, such as a battery night light versus an air humidifier, or a plumbing/heating-system sectional radiator versus a portable electric oil heater.

**Generation operation:** Preserve only any deliberately shared superficial context (for example colour or home-use context); replace the explicit product type and the corresponding primary-function attributes as one coherent type change.

**Guards:** Both primary product types and functions must be explicit and incompatible. Do not use broad surface words such as “radiator” or “heater” alone; retain the construction/installation or operating-function evidence that establishes the type distinction.

**Confidence:** strict

**Negative evidence:** pair_0147, pair_0149

**Positive counterexamples:** None observed

**Why reusable:** These independent negatives demonstrate that the decisive axis is the primary product function, not shared colour, household context, or an overloaded generic noun.

## Generation pattern 4

**Generalized rule:** A replacement component is a negative when it has an explicit incompatible fitment: non-overlapping supported appliance models together with a conflicting physical-fitment code or dimension.

**Scope:** Replacement seals, filters, bags, gears, and repair kits sold for a named appliance family.

**Generation operation:** Preserve the component class, appliance family, material, and pack count; change the explicit supported-model set together with a fitment-defining part number or physical dimension so that the compatibility sets are non-overlapping.

**Guards:** Do not generate from a missing compatibility field. Exclude listings marked universal and any pair whose supported models may overlap. Require both an explicit model-list conflict and an independently explicit fitment code or dimension; compatibility alone is insufficient.

**Confidence:** guarded

**Negative evidence:** pair_0136

**Positive counterexamples:** pair_0100, pair_0141, pair_0133

**Why reusable:** The refrigerator-seal example supplies a controlled fitment conflict (different HBM models, part codes, and dimensions). The positive compatible-part examples show why this must remain a guarded conjunctive rule.

## Generation pattern 5

**Generalized rule:** Different explicitly stated unit counts are a negative for otherwise identical replacement-consumable packs within the same stated device ecosystem.

**Scope:** Cartridge, filter, bag, or cleaning-consumable packs where the unit count is part of the sold SKU.

**Generation operation:** Preserve the consumable type, formulation/part family, stated compatibility ecosystem, and all non-pack attributes; change only an explicit unit-count field (for example 4 versus 8 cartridges).

**Guards:** Both counts must be explicit and the listings must name the same consumable family and compatibility ecosystem. Do not treat a missing count as a mismatch, and do not combine a pack-count edit with a brand or compatibility edit.

**Confidence:** guarded

**Negative evidence:** pair_0127

**Positive counterexamples:** None observed

**Why reusable:** Pack quantity is a reusable SKU axis, but the sample contains only one clean cartridge-pack instance, so the rule is retained with strict preservation and evidence requirements rather than elevated to strict confidence.

## Rejected generalizations

- **A different model code is always a negative.** Rejected: `pair_0148` is positive despite different Teka oven model codes and major size differences; `pair_0131` is negative despite the same Edisson ES 30 V model string, demonstrating noisy labels in the opposite direction. Model code is usable only with the narrow conjunction in Generation pattern 2.
- **A different compatible-appliance model is always a negative for replacement parts.** Rejected: the explicitly different filter/compatible-model names in positive `pair_0100`, `pair_0141`, and `pair_0133` show that universality, interchangeability, or incomplete fitment data can defeat this inference.
- **A different brand is always a negative.** Rejected: positive `pair_0102`, `pair_0109`, and `pair_0142` use differing or inconsistent brands while describing potentially matching products; brand is not a safe standalone generation axis.
- **Different capacity, dimensions, programme count, or colour is always a negative.** Rejected: positive `pair_0148` has conflicting oven capacity and height, while positive `pair_0107` has differing stated heating coverage/weight and `pair_0102` has different reservoir capacities. These attributes need a supported product-specific conjunction, not unconditional mutation.
- **Sparse near-name matches can establish an attribute rule.** Rejected: `pair_0116`, `pair_0124`, and `pair_0126` lack sufficient decisive detail, so their labels must not be converted into generation rules.
