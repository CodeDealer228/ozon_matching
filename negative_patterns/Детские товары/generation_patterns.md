## Generation pattern 1

**Generalized rule:** An explicitly different finished-toy size is a negative for an otherwise identical soft-toy design.

**Scope:** Soft toys or soft toy-pillow sets with the same explicitly named character/design and the same stated pack count.

**Generation operation:** Preserve the character/design, toy type, pack count, and all other identity fields; change only the explicit finished-product height or named size.

**Guards:** Both sizes must be explicit, use the same measurement concept (for example, height versus height, not package dimensions), and differ materially. Do not infer a size from an omitted field or use a broad animal name alone as identity.

**Confidence:** strict

**Negative evidence:** pair_0280, pair_0294, pair_0295

**Positive counterexamples:** None observed

**Why reusable:** The evidence covers two-piece heart pillows, a raccoon toy, and the same named Basik edition; the sellable physical size, rather than colour or a generic soft-toy label, is the controlled axis.

## Generation pattern 2

**Generalized rule:** Within an explicitly shared brand, line, or product family, a primary child product and a different explicitly declared product type are negatives.

**Scope:** Listings whose family relation is explicit, including toy lines and child-transport lines.

**Generation operation:** Preserve the brand/line and compatibility context; replace the product-type field with a different standalone type or a dedicated accessory type (for example, crane to toy car, or car seat to its base).

**Guards:** The type difference must be explicit in names or attributes and must change what is being sold. Do not use a taxonomy-label discrepancy alone, and do not call an included accessory a separate product.

**Confidence:** strict

**Negative evidence:** pair_0253, pair_0274

**Positive counterexamples:** None observed

**Why reusable:** In both cases shared line wording could look like identity, but the explicit sellable object is different; this makes a controlled type substitution useful without depending on a brand mismatch.

## Generation pattern 3

**Generalized rule:** A toy scale-model vehicle with a different explicitly named real-vehicle subject is negative.

**Scope:** Single toy-car or scale-model-vehicle listings that state the represented vehicle model.

**Generation operation:** Preserve toy type, scale, mechanism, effects, colour, and pack configuration where available; change only the declared represented vehicle model.

**Guards:** Use fully explicit, non-equivalent vehicle subjects (for example, Chevy Nomad versus Lada, or Tank 300 versus Range Rover Sport). Do not derive a mismatch from a vague label such as “jeep,” and do not combine a vehicle-model change with an unverified scale or pack-count change.

**Confidence:** guarded

**Negative evidence:** pair_0261, pair_0268

**Positive counterexamples:** None observed

**Why reusable:** Both negatives distinguish different real-vehicle subjects despite shared toy-car characteristics, but their source listings also vary on other fields, so generation must isolate the subject-model axis.

## Generation pattern 4

**Generalized rule:** A different declared scale makes an otherwise identical scale-model vehicle a negative.

**Scope:** Toy vehicles with an explicit scale and an explicit identical represented vehicle/service edition.

**Generation operation:** Preserve brand, represented vehicle, edition/theme, vehicle type, colour, mechanism, and pack count; change only the declared scale.

**Guards:** Apply only when both scales are explicit and normalized (for example, 1:32 versus 1:43) and the remaining identity is independently aligned. Do not use physical dimensions as a proxy for scale.

**Confidence:** guarded

**Negative evidence:** pair_0271

**Positive counterexamples:** None observed

**Why reusable:** The UAZ 3962 fire-service pair isolates an explicit scale difference within the same named subject, but the sample supplies only one such clean comparison.

## Generation pattern 5

**Generalized rule:** A pacifier from the same explicit product line is negative when both its exact model code and its non-overlapping stated age band change together.

**Scope:** Pacifiers with a named manufacturer/line, explicit model suffix, explicit age band, and the same pack configuration.

**Generation operation:** Preserve manufacturer, line, material, nipple form, case/accessory configuration, and pack count; replace the model code and its corresponding non-overlapping age band together.

**Guards:** Do not use age alone or a model-code change alone. Both values must be explicit and the age intervals must not overlap.

**Confidence:** guarded

**Negative evidence:** pair_0257

**Positive counterexamples:** None observed

**Why reusable:** The evidence supports a controlled model-and-age variant substitution, while it does not isolate either axis sufficiently to make it unconditional.

## Generation pattern 6

**Generalized rule:** For the same diaper-pants line, a different explicitly declared number of diapers in the sellable pack is negative.

**Scope:** Diaper pants with explicit brand, line, size, weight range, and product type.

**Generation operation:** Preserve brand, line, diaper type, size, weight range, and other variant fields; change only the count of diapers in the pack.

**Guards:** The count must conflict in reliable product text or a canonical count field on both sides. Do not use conflicting noisy auxiliary attributes, and do not simultaneously change size or weight range.

**Confidence:** guarded

**Negative evidence:** pair_0276

**Positive counterexamples:** pair_0251

**Why reusable:** The Pampers pair keeps the named line, size, and weight band aligned while changing 76 to 38, but the positive Helen Harper pair has inconsistent title and attribute counts and therefore requires explicit-source guards.

## Generation pattern 7

**Generalized rule:** A different explicit key count makes an otherwise equivalent child synthesizer configuration negative.

**Scope:** Electronic keyboard/synthesizer toys with an explicit brand and key count.

**Generation operation:** Preserve brand, instrument type, and any known model family; change only the number of keys.

**Guards:** Both key counts must be explicit. Do not treat differences in power supply, included cable, age recommendation, or a missing model code as interchangeable with the key-count axis.

**Confidence:** guarded

**Negative evidence:** pair_0281

**Positive counterexamples:** None observed

**Why reusable:** The two Bondibon synthesizers demonstrate that 61 versus 49 keys identifies distinct configurations, but the source also changes accessories and power details.

## Generation pattern 8

**Generalized rule:** A named costume edition of the same soft-toy character is negative to the plain edition.

**Scope:** Same-brand, same-character soft toys with explicit equal size and an explicit costume/edition descriptor.

**Generation operation:** Preserve brand, character, toy type, and finished size; add or replace only a named costume/edition descriptor.

**Guards:** The costume must be explicitly marketed as part of the toy edition (for example, “в пижамке”), not merely inferred from an image, colour, or included removable accessory.

**Confidence:** guarded

**Negative evidence:** pair_0287

**Positive counterexamples:** None observed

**Why reusable:** The Pomposhki pair keeps character and 21 cm size aligned and supplies a concrete edition-level distinction, but there is one observed example only.

## Generation pattern 9

**Generalized rule:** Educational sets sharing a title phrase are negative when their explicitly stated activity medium or core contents differ.

**Scope:** Learning/activity sets with a shared brand or title fragment but explicit, incompatible core media.

**Generation operation:** Preserve the category, intended skill/theme, and shared title/brand context; change the core activity medium or contents (for example, stamps versus fuzzy wire).

**Guards:** The distinct medium or contents must be explicit on both sides. Do not use title overlap alone, and do not generate from a missing or generic completeness field.

**Confidence:** guarded

**Negative evidence:** pair_0290

**Positive counterexamples:** pair_0263

**Why reusable:** The negative pair shows a real content-level distinction behind the shared “логические задания” wording, while the positive exact “алфавит” pair confirms that wording by itself is not a negative signal.

## Rejected generalizations

- **Colour mismatch is an unconditional negative.** pair_0282 is negative for silver versus gold doll shoes, but pair_0284 is positive despite yellow versus blue crane colours; colour may be an edition axis only when stronger identity fields and clean evidence support it.
- **Different brand or seller name is an unconditional negative.** pair_0293 is positive despite Sluban versus Magic Land naming, and pair_0286 is positive despite one side naming “Технок”; brand text is not a safe generated mismatch by itself.
- **Different diaper size, age range, or any reported pack count is an unconditional negative.** pair_0251 is positive even though the titles name sizes 4 and 5 and the second item's attributes report 54 against title count 80; use only the tightly guarded pack-count rule above.
- **A different model year or version word alone is sufficient.** pair_0259 changes Pro/standard, model year, and colour together, while pair_0265 is an under-specified negative around the same Forward Unit 20 1.0 listing; the sample does not isolate model year or version as a standalone generation axis.
- **Near-identical title text or the absence of an attribute proves a mismatch.** The caveated negatives pair_0250, pair_0252, pair_0255, pair_0260, pair_0262, pair_0269, pair_0270, pair_0272, pair_0273, pair_0288, and pair_0292 are near-duplicates or under-specified, while pair_0254, pair_0263, pair_0266, pair_0283, pair_0291, and pair_0296 are positive near-duplicates.
- **A broad animal/character word, packaging detail, or taxonomy label is sufficient.** pair_0279 is internally inconsistent (a soft-toy name but “одежда для мягкой игрушки” type), and pair_0285 and pair_0289 contain multiple unisolated differences; do not synthesize from these axes without explicit controlled values.
