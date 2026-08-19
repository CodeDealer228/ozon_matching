# Generation-ready negative patterns — Строительство и ремонт

## Generation pattern 1

**Generalized rule:** Within the same construction-component family, an explicit mismatch in the operative interface or standardized working profile is a negative match.

**Scope:** Interchangeable consumables and components whose profile determines the counterpart they work with, such as screwdriver bits and threading dies.

**Generation operation:** Preserve product family, intended tool/system, material, and pack configuration; replace one explicitly stated interface value (for example, PH1 with PZ2, or UNC with a metric thread) with another incompatible standard.

**Guards:** Both values must be explicit, must describe the same interface axis, and must be known incompatible alternatives. Do not infer an interface from a vague product title, a size alone, or a missing field.

**Confidence:** strict

**Negative evidence:** pair_0759, pair_0789

**Positive counterexamples:** None observed

**Why reusable:** These independent negatives show that the decisive identity can be a standardized mating/working profile even where brand, length, quantity, or broad product type are alike.

## Generation pattern 2

**Generalized rule:** For a fixed construction product family, changing the explicitly specified dimensional profile or nominal size creates a negative variant.

**Scope:** Products whose fit or usable result is determined by dimensions: drill bits, towel warmers, windows/soft windows, shower partitions, and sheet/panel materials.

**Generation operation:** Preserve product family, material, functional type, orientation where relevant, and all non-size identity fields; change one explicit required dimension or the ordered width-by-height profile.

**Guards:** Use normalized units and preserve the order of dimensions; do not treat a swapped width and height as equivalent. Generate only when the dimension is a sellable variant, rather than a shipping measurement or an unspecified approximate value.

**Confidence:** strict

**Negative evidence:** pair_0764, pair_0768, pair_0784, pair_0785, pair_0791

**Positive counterexamples:** None observed

**Why reusable:** Across distinct construction families, declared diameter, width, height, and ordered footprint distinguish functional variants despite overlap in brand, material, finish, or use.

## Generation pattern 3

**Generalized rule:** An explicit manufacturer model/SKU mismatch is a negative match when both listings are otherwise the same product family and the codes identify different configurations.

**Scope:** Branded electrical equipment and lighting where the model or part number identifies a concrete variant.

**Generation operation:** Preserve brand, product family, installation class, and common electrical characteristics; substitute a different explicit manufacturer model/SKU and align only the configuration values that that model actually encodes.

**Guards:** The identifiers must be genuine product model/part numbers, not marketplace IDs, source-row IDs, or formatting variants of the same code. Keep the operation within one brand and product family; do not rely on brand alone as a negative signal.

**Confidence:** strict

**Negative evidence:** pair_0755, pair_0779, pair_0781, pair_0793

**Positive counterexamples:** None observed

**Why reusable:** Repeated same-brand comparisons remain negative when explicit codes select different lamp, stabilizer, luminaire, or soft-starter configurations.

## Generation pattern 4

**Generalized rule:** For electrically rated products in one product family, an explicit mismatch in the nominal rating/configuration is a negative variant.

**Scope:** Lamps, voltage stabilizers, motor soft starters, and circuit breakers.

**Generation operation:** Preserve the device family, intended use, electrical system, and pack count; change one declared nominal rating such as wattage/voltage, VA capacity, rated current, or motor power to another valid variant.

**Guards:** Change a named product rating, not a maximum, a marketing claim, or a measurement recorded in incompatible units. Do not combine unrelated family, interface, or pack changes in the same generated pair.

**Confidence:** strict

**Negative evidence:** pair_0753, pair_0779, pair_0793, pair_0796

**Positive counterexamples:** None observed

**Why reusable:** Multiple independently labelled negatives distinguish otherwise similar electrical goods by their declared operating capacity or rating.

## Generation pattern 5

**Generalized rule:** A change between an explicitly recessed and an explicitly surface-mounted installation configuration is a negative variant of a lighting product.

**Scope:** Light fixtures for which the mounting construction is explicitly stated.

**Generation operation:** Preserve fixture family, base/socket, intended room, and the other electrical characteristics; change only the declared installation type between recessed and surface-mounted.

**Guards:** Require unambiguous installation fields or names on both sides. Do not use conflicting marketplace metadata as evidence of a true mounting type, and do not generalize this rule to other product families without direct evidence.

**Confidence:** guarded

**Negative evidence:** pair_0751

**Positive counterexamples:** None observed

**Why reusable:** The pair isolates a real architectural distinction even while socket, color, voltage, protection level, and lamp count overlap; one sample is insufficient for an unguarded category-wide rule.

## Generation pattern 6

**Generalized rule:** For an otherwise confirmed identical item, a different explicit number of units in the sold pack is a negative match.

**Scope:** Multi-packs of the same construction/repair item, especially fixtures and consumables.

**Generation operation:** Preserve the exact product identity, model, finish, dimensions, and unit specification; change only the stated pack count.

**Guards:** Both titles or attributes must explicitly identify the same underlying unit and a different count. Do not infer count from ambiguous completeness text or use it when color, model, or product type changes at the same time.

**Confidence:** guarded

**Negative evidence:** pair_0792

**Positive counterexamples:** None observed

**Why reusable:** The sample directly presents the same named Ecola recessed fixture as packs of three versus five, but the sparse and partially incomplete pack metadata warrants a guarded rule.

## Rejected generalizations

- **Colour/finish alone is always a negative:** reject. `pair_0775` is a negative with nearly identical Rio 4C D100 fan descriptions and different finishes, but the original audit flags it as likely colour-variant noise; other colour-labelled negatives (`pair_0778`, `pair_0780`, `pair_0787`) also contain unisolated model, formula, or specification changes.
- **A different named machine model proves a replacement part is incompatible:** reject. Positive pairs `pair_0783` and `pair_0799` link parts described for different equipment models; compatibility needs an explicit part/application statement, not just a model-name mismatch.
- **Any change in thread size or fitting style can be generated as a standalone negative:** reject. `pair_0752` and `pair_0760` each combine several differences (nominal bore, actuation, fitting form, and connection details), so they do not isolate one safely reusable fitting-compatibility axis.
- **Every difference in decorative motif or surface appearance is a negative:** reject. `pair_0767` combines image subject with height; it does not establish an appearance-only rule.
