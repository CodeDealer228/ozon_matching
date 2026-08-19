# Synthesis-ready negative patterns — Спорт и отдых

## Generation pattern 1

**Generalized rule:** An explicit mismatch in a SKU-defining functional size, resistance, load, or performance rating is a negative for otherwise comparable sports-equipment variants.

**Scope:** The same concrete equipment family: resistance bands, fishing line, manual pumps, protective pads, ice augers, fixed-weight dumbbells, and fishing hooks. The changed value must govern fit, operating capacity, force, bore, gauge, or physical performance rather than a shipping measurement.

**Generation operation:** Preserve the concrete product type, intended use, construction, pack configuration, brand/model family, and all non-target identity fields when known; replace one explicit functional value (for example load, diameter, length, wearer size, drill diameter, dumbbell weight, or hook gauge) with a different valid value for that same family.

**Guards:** Both conflicting values must be explicit and normalized to the same basis. Do not derive a value from a title fragment, substitute package weight for product weight, or combine the size edit with a model, material, colour, or component-type change. Use only values that are genuinely SKU-defining for the stated product family.

**Confidence:** strict

**Negative evidence:** pair_0701, pair_0708, pair_0711, pair_0718, pair_0722, pair_0747, pair_0749

**Positive counterexamples:** None observed

**Why reusable:** Independent negatives repeat the same operational identity principle across exercise, fishing, winter, and protection equipment: a declared functional rating or fit/working size distinguishes a purchasable variant even when the brand and broad product type agree.

## Generation pattern 2

**Generalized rule:** Different explicit manufacturer model codes or named model lines are negative variants within the same product type and brand family.

**Scope:** Pumps, metal detectors, fishing reels, and other sports/outdoor equipment for which both listings explicitly identify a manufacturer model, series, or code.

**Generation operation:** Preserve the manufacturer, concrete product type, product-generation context, and shared high-level specifications; replace one explicit model code or model-line value with another verified distinct value from the same manufacturer.

**Guards:** Require two explicit, non-equivalent model identifiers; an abbreviated title, an omitted code, a generic family name, or a reseller SKU is not a conflict. Do not use model text as a proxy for mere compatibility, colour, or a package configuration, and keep any model-specific technical configuration coherent with the substituted code.

**Confidence:** strict

**Negative evidence:** pair_0711, pair_0723, pair_0737

**Positive counterexamples:** None observed

**Why reusable:** The sampled negatives distinguish Intex 68614 from 68615, XP Deus from Deus 2, and Kaida Gallant from Differ while retaining the broad equipment family; a concrete manufacturer identifier is therefore a controlled cross-product identity axis.

## Generation pattern 3

**Generalized rule:** An explicit colour, print, or manufacturer finish code is a negative when all other SKU-defining identity fields establish the same concrete sporting-good model or form.

**Scope:** Otherwise matched yoga mats, small fitness balls, bicycle valve-cap sets, and fishing lures whose colour/finish code is explicitly declared on both sides.

**Generation operation:** Preserve the product type, manufacturer/model family when present, dimensions or performance rating, material, quantity, and functional construction; replace exactly one explicit colour, print, or finish/code value with another valid variant value.

**Guards:** Do not infer colour from an image, a generic adjective, or a missing field. Keep this rule to catalogue variants with the same concrete form and fixed specifications; do not apply it when a colour phrase also conceals a different lure design, material, size, or product type.

**Confidence:** strict

**Negative evidence:** pair_0703, pair_0709, pair_0712, pair_0729, pair_0732, pair_0736

**Positive counterexamples:** None observed

**Why reusable:** Multiple independent negatives hold the physical form and key dimensions constant while changing an explicit consumer-facing visual variant, so the rule is not tied to one particular mat, ball, lure, or accessory.

## Generation pattern 4

**Generalized rule:** An explicit change to a non-interchangeable primary product type, component role, or operating mechanism is a negative even when the listings share a broad sports domain or compatible platform.

**Scope:** Complete bicycles with different ride architectures, air rifles with different power mechanisms, model-specific vehicle parts with different component roles, and fixed-path strength machines with different exercise functions.

**Generation operation:** Preserve the broad product domain and any intentionally shared brand/model or compatible-host identifier; change one explicit primary type, mechanism, or component role to another mutually non-interchangeable value, while retaining coherent specifications for the resulting product.

**Guards:** Both roles or mechanisms must be explicitly stated and physically non-interchangeable (for example tyre versus rear fork, or spring-piston versus electropneumatic). Do not use a different compatibility target, generic marketplace category, feature list, or brand alone as the type edit; do not create a pair that merely differs in a loose wording synonym.

**Confidence:** strict

**Negative evidence:** pair_0704, pair_0740, pair_0741, pair_0745

**Positive counterexamples:** None observed

**Why reusable:** These negatives make the decisive distinction functional: shared cycling context, rifle caliber, scooter model compatibility, or weight-stack terminology does not make two different product architectures or component roles the same item.

## Generation pattern 5

**Generalized rule:** An explicit different flavour is a negative for an otherwise comparable sports-nutrition powder variant.

**Scope:** The same supplement family and form, particularly comparable whey-protein powders with an explicit flavour on both sides.

**Generation operation:** Preserve supplement type, form, brand/model family, net weight, pack count, and nutrition formulation where known; replace one explicit flavour value with another explicit flavour value.

**Guards:** Require explicit flavour values on both listings and keep the base supplement and sold amount aligned. Do not treat a missing flavour, a marketing phrase, or an unverified recipe/composition change as a flavour mismatch; if composition is also intentionally changed, generate a separate guarded rule rather than claiming a single-axis flavour edit.

**Confidence:** guarded

**Negative evidence:** pair_0728

**Positive counterexamples:** None observed

**Why reusable:** The evidence keeps the RPS whey-product family and 500 g amount while naming different flavours, but one clean sports-nutrition example is insufficient to make flavour universally strict.

## Rejected generalizations

- **Compatibility with a different named platform is automatically negative.** Rejected: positive `pair_0714` pairs a Saiga-308 magazine description with a Vepr-308 description despite the platform wording. Compatibility must not be used as an unconditional identity axis; only an explicit non-interchangeable component role is allowed under Generation pattern 4.
- **Any package count, package weight, or shipping weight mismatch is negative.** Rejected: positive `pair_0727` has inconsistent listed 180 g versus 250 g amounts for closely described Orzax gummies, and positive `pair_0720` shows different item versus packaged weights for the same lantern. The multiple changes in `pair_0708` do not isolate pack count from line specifications.
- **Any broad category, brand, or sparse-title difference is negative.** Rejected: the negatives `pair_0705`, `pair_0721`, `pair_0724`, `pair_0735`, and `pair_0739` are under-specified; `pair_0746` is kiln tubing erroneously present in this category. These are not reliable generators of sports-product negatives.
- **A colour, finish, or visual term by itself is always negative.** Rejected as unconditional: use only explicit value-versus-value colour/finish changes under the same fixed product form in Generation pattern 3; absent or generic visual metadata is not a conflict.
- **A different numeric value anywhere in the listing is negative.** Rejected: `pair_0719` and `pair_0730` show matched items with dimensions expressed differently, while `pair_0727` and `pair_0720` show unreliable weight fields. Only a normalized, item-level functional value within the guarded scope of Generation pattern 1 may be changed.
