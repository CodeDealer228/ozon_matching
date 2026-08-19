## Generation pattern 1

**Generalized rule:** Within a fixed food subtype, an explicit substitution of the recipe's central ingredient or base is a negative when the two named ingredients are incompatible.

**Scope:** Ready porridges, dry broths, and single-ingredient herbs/spices where both listings explicitly identify the same food subtype and the central ingredient or base.

**Generation operation:** Change only the recipe/ingredient axis (for example chicken to fried onion in buckwheat porridge, mushroom to vegetable broth, or mint to basil); preserve the food subtype, preparation form, brand/model family when known, and pack configuration.

**Guards:** Both conflicting values must be explicit ingredients or named bases, not an omitted field, a trace additive, a serving suggestion, or a packaging claim. Do not use this rule for sauces whose shared base remains the same and only a spice descriptor differs.

**Confidence:** strict

**Negative evidence:** pair_0650, pair_0662, pair_0664

**Positive counterexamples:** None observed

**Why reusable:** The evidence covers three different food families, but in each case the decisive axis is the declared edible base or central ingredient rather than a seller, size, or wording difference.

## Generation pattern 2

**Generalized rule:** An edible cake decoration is a negative when its explicit edible substrate changes between sugar-based and wafer-based material, even if the depicted theme is the same.

**Scope:** Flat edible images/decorations for cakes with an explicit substrate in the name or composition.

**Generation operation:** Change the form/material axis from a sugar image/sugar paper to a wafer image/wafer paper, or the reverse; preserve the design theme, intended use, approximate dimensions, and single-item configuration.

**Guards:** Require explicit, conflicting substrate evidence in both listings (for example sugar-containing image versus wafer/vegetable-starch paper). Do not infer substrate from a generic label such as “confectionery decoration,” and do not use image artwork alone as the changed axis.

**Confidence:** strict

**Negative evidence:** pair_0659, pair_0670, pair_0675

**Positive counterexamples:** None observed

**Why reusable:** Three independently sampled pairs keep the cake-decoration use and closely related themes while changing the consumable base, a stable product-form distinction suitable for controlled generation.

## Generation pattern 3

**Generalized rule:** A processed fruit or drink-base item is a negative when both its explicit product form and its named flavour/fruit recipe are changed together.

**Scope:** Branded fruit preserves, dessert bases, beverage concentrates, and syrups where both listings state a concrete product type and a concrete fruit or flavour recipe.

**Generation operation:** Change the conjunctive form-and-recipe axes together (for example confiture to fruit dessert plus peach-only to peach-and-passionfruit, or beverage concentrate plus blueberry to syrup plus bubble-gum); preserve brand, broad use domain, single-unit configuration, and comparable volume/weight.

**Guards:** This is a conjunctive operation: do not generate a negative by changing only form or only flavour from this evidence. Both old and new type and recipe values must be explicit; do not treat an omitted flavour as a mismatch.

**Confidence:** strict

**Negative evidence:** pair_0682, pair_0689

**Positive counterexamples:** None observed

**Why reusable:** In both same-brand comparisons, a type/form mismatch co-occurs with an independently named recipe mismatch, yielding a controlled hard negative without treating either unisolated axis as universally decisive.

## Generation pattern 4

**Generalized rule:** A plain, explicitly single-component tea and an explicitly flavoured or multi-ingredient tea blend are different products.

**Scope:** Leaf tea listings where the tea base and the complete blend/additive status are explicit.

**Generation operation:** Change the recipe/flavour axis from a named pure tea varietal to a named flavoured or botanical blend; preserve tea form (leaf), broad tea type when possible, package configuration, and comparable quantity.

**Guards:** Require explicit composition on both sides: one must state a single tea base and the other must name added flavouring or botanicals. Do not use a vague flavour adjective, a different country of origin, or a different pack size alone; keep this rule separate from tea-bag count.

**Confidence:** guarded

**Negative evidence:** pair_0679

**Positive counterexamples:** None observed

**Why reusable:** The reusable distinction is recipe composition (pure varietal versus declared blend), not the specific Assam or watermelon names; the single observed negative warrants the stated safeguards.

## Generation pattern 5

**Generalized rule:** A standalone food item and a gift bundle that explicitly adds heterogeneous non-identical contents are different sellable products.

**Scope:** Food listings in which one side is a defined food SKU and the other explicitly lists a bundle containing that food plus additional, different product types.

**Generation operation:** Change the bundle-composition axis from the standalone food item to a bundle that adds named heterogeneous components (for example spices, confectionery, or drinkware); preserve the featured food brand/family and do not rely on total weight.

**Guards:** The added contents must be explicit and qualitatively different from merely more units of the same SKU. Do not apply to multipacks, changes in count, or a bundle whose contents are unspecified.

**Confidence:** guarded

**Negative evidence:** pair_0699

**Positive counterexamples:** None observed

**Why reusable:** The evidence isolates a commercially meaningful bundle-composition change, unlike the noisy pack-count differences elsewhere in the sample.

## Generation pattern 6

**Generalized rule:** For otherwise comparable ground coffee, a jointly changed bean-composition, grind, and roast profile can form a negative product variant.

**Scope:** Ground coffee with explicit bean composition, grind setting, and roast setting on both sides.

**Generation operation:** Change the conjunctive recipe-and-processing profile (for example 100% robusta, coarse grind, dark roast to an arabica/robusta blend, medium grind, medium roast); preserve coffee form, brand/family, origin where known, and net weight.

**Guards:** This is not evidence that any one of bean composition, grind, or roast alone is sufficient. Require all three explicit conflicting values and keep coffee form and pack configuration fixed; do not trust a lone contradictory type field.

**Confidence:** guarded

**Negative evidence:** pair_0691

**Positive counterexamples:** None observed

**Why reusable:** The rule turns one multi-attribute coffee mismatch into a safely conjunctive generation operation while avoiding unsupported one-axis coffee rules.

## Generation pattern 7

**Generalized rule:** Within milk chocolate, changing from an explicit sugar-free formulation to a sugared formulation with an additional named nut ingredient is a negative recipe variant.

**Scope:** Milk-chocolate listings with explicit dietary claim and full ingredient composition.

**Generation operation:** Change the conjunctive recipe axis from sugar-free milk chocolate to milk chocolate that explicitly contains sugar and a named nut ingredient; preserve chocolate type, approximate bar size, and single-item configuration.

**Guards:** Require both the explicit “sugar-free/no sugar” claim and an explicit sugar-containing ingredient list with the named nut. Do not reduce this to “different shape,” “different brand,” or an unverified dietary claim; this evidence does not isolate sugar from nut addition.

**Confidence:** guarded

**Negative evidence:** pair_0676

**Positive counterexamples:** None observed

**Why reusable:** It captures a controlled recipe-and-dietary-profile difference while retaining the conjunction that the sample, rather than general product intuition, supports.

## Rejected generalizations

- **Pack count, total weight, or per-pack quantity is always a negative:** rejected. These axes have positive counterexamples with mismatched counts or weights in pair_0660, pair_0668, and pair_0683, while several negatives are otherwise near-duplicates differing mainly in packaging (pair_0656, pair_0657, pair_0666, pair_0678, pair_0685, pair_0688, pair_0692, pair_0694, pair_0695).
- **Different brands or sellers mean different products:** rejected. Different-brand oil is positive in pair_0651, and different-brand sriracha is positive in pair_0665.
- **A generic product-type/form field alone is decisive:** rejected. Pair_0697 is positive although one listing calls the coffee ground and the other has a conflicting “instant coffee” type field; data-field noise makes this unsafe without corroborating explicit recipe/form evidence.
- **Any individual ground-coffee difference in bean blend, grind, or roast is a negative:** rejected. pair_0691 changes all three axes together and therefore does not isolate any one of them; pair_0697 also demonstrates unreliable isolated type metadata.
- **Artwork, title wording, or a nearly identical food listing identifies a different item:** rejected. The sampled negative near-duplicates pair_0653, pair_0667, pair_0671, pair_0677, pair_0680, pair_0681, pair_0687, and pair_0696 are annotation-noise risks rather than safe generation templates.
