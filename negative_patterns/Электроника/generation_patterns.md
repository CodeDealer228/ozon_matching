# Generation-ready negative patterns — Электроника

This derived specification keeps the evidence-level files unchanged. It describes only controlled attribute edits for a later generator; it does not create synthetic pairs.

## Generation pattern 1

**Generalized rule:** Two complete electronic products with explicitly different manufacturer model codes or product SKUs are different offers.

**Scope:** Finished devices and modules sold as a particular model: phones, TVs, smart switches, motherboards, and similar hardware.

**Generation operation:** Preserve the product type, brand/family and non-identity presentation fields; replace one explicit offered-product model code/SKU with a different valid code from the same product family.

**Guards:** Both values must identify the offered product itself, not a compatible-device name, seller code, colour variant, or a partial/ambiguous title. Do not generate if the listings share an exact manufacturer part number or model code; do not infer a conflict from an omitted model.

**Confidence:** strict

**Negative evidence:** pair_0901, pair_0915, pair_0944, pair_0948

**Positive counterexamples:** pair_0903, pair_0918, pair_0945

**Why reusable:** The negatives cover several device families with distinct model identifiers. The positives show that matching exact model/part numbers remain matches even when titles, brands, or attribute completeness differ, which makes the rule an identity-code edit rather than a textual-name rule.

## Generation pattern 2

**Generalized rule:** A fixed, prebuilt computer is a different offer when multiple explicitly declared core configuration fields conflict.

**Scope:** Complete desktop/system-unit listings with a fixed configuration rather than separately sold configurable components.

**Generation operation:** Preserve the computer product type, brand/family, form factor and broad use case; change at least two explicit fields among CPU model, GPU model, RAM capacity, SSD/HDD capacity, chipset, or operating-system configuration.

**Guards:** Generate only where both listings are complete prebuilt systems and the changed values are explicit. Do not turn an omitted field into a conflict, and do not treat a single unverified specification difference as sufficient; retain all relevant unit and capacity labels.

**Confidence:** strict

**Negative evidence:** pair_0906, pair_0913, pair_0927, pair_0942

**Positive counterexamples:** None observed

**Why reusable:** Each evidence pair compares complete gaming/system units with several conflicting core components, so the controlled operation transfers across prebuilt-PC listings without claiming that any isolated component attribute always identifies a different item.

## Generation pattern 3

**Generalized rule:** A model-specific accessory or replacement part is negative when its explicitly declared target device/model is different and the two fitment lists do not overlap.

**Scope:** Phone cases and screen protectors, displays, lens caps, model-specific remotes, and other accessories whose function depends on a named compatible device or physical fitment target.

**Generation operation:** Preserve the accessory/part type and its non-fitment characteristics; replace the explicit target model or compatibility list with a disjoint, valid target model/family.

**Guards:** Require an explicit fitment claim on both sides and verify that no exact model, part number, or declared compatibility range overlaps. A brand name alone is not a target. Do not use broad marketing compatibility text as evidence of incompatibility: pair_0903 shares remote code `NH400UD (R20130427)` despite different named brands, and pair_0936 is positive despite different Acer model strings.

**Confidence:** strict

**Negative evidence:** pair_0916, pair_0933, pair_0943, pair_0947, pair_0949

**Positive counterexamples:** pair_0903, pair_0936

**Why reusable:** The same fitment axis occurs in several accessory families, while the guards preserve the observed distinction between a genuinely disjoint target and alternative wording for a shared compatible part.

## Generation pattern 4

**Generalized rule:** An explicitly different number of identical units in a retail pack is a negative variant.

**Scope:** Multi-unit electronics and consumables where the listing states the count: camera sets, batteries, and analogous packs.

**Generation operation:** Preserve the exact item model/type, specification, and packaging context; change only the explicit unit count to another valid count.

**Guards:** The count must be stated as product/pack quantity, not as a count of ports, buttons, or an incidental packaging measurement. Use only where the individual unit is otherwise the same and both counts are explicit.

**Confidence:** strict

**Negative evidence:** pair_0907, pair_0912

**Positive counterexamples:** None observed

**Why reusable:** Both negatives retain the item family while 2 versus 3 cameras and 2 versus 4 AA batteries change the commercial offer itself.

## Generation pattern 5

**Generalized rule:** For an otherwise equivalent cable or mains extension, an explicit cable-length mismatch is a negative variant.

**Scope:** Cables and extension leads with the same declared connector/function family.

**Generation operation:** Preserve the cable/extension type, connector arrangement, electrical rating and other configuration; change one explicit length value and its matching model/SKU where applicable.

**Guards:** Both lengths must use compatible units and identify cable length, not package dimensions. Do not apply to products that differ in connector/interface, number of outlets, or electrical rating as well; that would introduce a confounded multi-axis edit.

**Confidence:** strict

**Negative evidence:** pair_0921, pair_0929

**Positive counterexamples:** None observed

**Why reusable:** The evidence shows 15.2 m versus 4.6 m HDMI cables and 2 m versus 3 m otherwise comparable Lezard extensions; length is a reusable purchasable configuration field in this narrow scope.

## Generation pattern 6

**Generalized rule:** A laptop power adapter with an explicitly incompatible output voltage is a negative variant even when the barrel connector size is the same.

**Scope:** DC laptop power adapters that state output voltage and connector dimensions.

**Generation operation:** Preserve adapter type, connector dimensions/polarity, and output-current/power context; replace the explicit output voltage with a different valid voltage.

**Guards:** Apply only when both output voltages and connector specifications are explicit and the values are not declared as a selectable multi-voltage range. Keep the physical connector fixed; do not infer voltage from wattage, and do not generate from missing electrical attributes.

**Confidence:** guarded

**Negative evidence:** pair_0941

**Positive counterexamples:** None observed

**Why reusable:** The 12 V versus 20 V adapter evidence holds connector size constant, isolating an electrically decisive axis, but the sample supplies only one such negative and therefore warrants guarded use.

## Generation pattern 7

**Generalized rule:** Printer cartridges are different offers when both the explicit compatible printer/platform or cartridge part code and the ink/toner colour channel conflict.

**Scope:** Single-colour compatible ink or toner cartridges for named printer families.

**Generation operation:** Preserve consumable type, pack count and printer technology; replace the explicit printer/platform or cartridge code and the corresponding explicit colour channel with a different valid combination.

**Guards:** Use this as a conjunctive edit, not as a colour-only rule. Require explicit compatible-device information or an unambiguous cartridge code on each side; do not treat a missing compatibility field, generic “black”, or a capacity/resource difference alone as a conflict.

**Confidence:** strict

**Negative evidence:** pair_0925, pair_0934, pair_0937

**Positive counterexamples:** None observed

**Why reusable:** Across the evidence, black/yellow, magenta/black, and magenta/light-magenta cartridges also belong to different stated printer targets or part-code families, supporting a controlled consumable-identity change while avoiding an unsupported universal colour rule.

## Rejected generalizations

- **Any different model-like string is automatically a negative.** Rejected by positive `pair_0903` (the same remote code despite Harper/Nesons wording) and `pair_0936` (different Acer submodel strings with equal battery capacity and voltage). Use only the guarded offered-product or disjoint-fitment rules above.
- **Any single component specification change in a prebuilt PC is enough.** The PC negatives `pair_0906`, `pair_0913`, `pair_0927`, and `pair_0942` contain several simultaneous configuration differences, so they do not isolate CPU, GPU, RAM, or storage as unconditional single-axis rules.
- **A colour, case style, or superficial presentation difference alone makes an accessory negative.** `pair_0947` also changes the target phone model, and the apparently duplicate negatives `pair_0900`, `pair_0902`, `pair_0924`, and `pair_0935` are annotated/noisy cases rather than evidence for visual-variant rules.
- **Any cartridge colour difference is negative.** The available cartridge negatives also change compatible printer/platform or part-code family (`pair_0925`, `pair_0934`, `pair_0937`); colour alone is not isolated.
- **Near-identical names or sparse attributes can be used to manufacture a mismatch.** `pair_0905`, `pair_0911`, `pair_0922`, `pair_0928`, `pair_0931`, `pair_0938`, and `pair_0940` are negative labels with weakly expressed or missing decisive differences, so they must not seed a generator rule.
