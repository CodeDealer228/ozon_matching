# Synthesis-ready negative patterns — Аптека

## Generation pattern 1

**Generalized rule:** Ready-made corrective glasses with an explicit, different spherical optical power are different product variants.

**Scope:** The same explicitly identified ready-made-glasses model or frame/lens family, where both listings state a signed dioptre value.

**Generation operation:** Preserve manufacturer, explicit model code, lens family and treatment, frame variant, PD, pack/accessory configuration, and product type; replace the signed optical-power value with a different explicit value.

**Guards:** Do not infer a power from omitted text, and do not apply to non-corrective frames or to listings whose numeric value belongs to a different measurement. Both values must be explicit dioptres for the finished glasses.

**Confidence:** strict

**Negative evidence:** pair_0050, pair_0056, pair_0082, pair_0084, pair_0094

**Positive counterexamples:** None observed

**Why reusable:** The sample repeatedly labels differing corrective strengths as negatives across several ready-made-glasses lines. Restricting the edit to an explicit prescription value makes a controlled hard negative rather than changing styling or unrelated metadata.

## Generation pattern 2

**Generalized rule:** Different explicit manufacturer frame model codes denote different ready-made-glasses variants.

**Scope:** Listings of finished glasses that name a manufacturer-style frame model code (for example, a numeric model identifier), with the code attached to the glasses rather than to an accessory or seller SKU.

**Generation operation:** Preserve brand, lens family, signed optical power, PD, colour variant, and all non-frame configuration; replace the explicit frame model code with a different valid code from the same manufacturer.

**Guards:** Use only normalized manufacturer model codes that are visibly distinct in the product identity. Do not substitute marketplace offer IDs, internal seller article numbers, or a code whose product role is unclear. Preserve the same product type: do not use this operation to turn ready-made glasses into a bare frame or another optical product.

**Confidence:** strict

**Negative evidence:** pair_0050, pair_0053, pair_0057, pair_0061, pair_0093, pair_0098

**Positive counterexamples:** None observed

**Why reusable:** Model codes are a portable identity axis for the physical frame. The evidence includes multiple brands and shows that shared brand, lens family, and even the same dioptres do not collapse distinct frame models into one item.

## Generation pattern 3

**Generalized rule:** Contact-lens variants in the same explicitly named line are negative when their stated spherical power (SPH/optical power) differs.

**Scope:** Contact lenses with the same named product line, the same lens type and pack configuration, and explicit spherical-power values.

**Generation operation:** Preserve brand, product line, lens type, base curve, diameter, replacement schedule, pack count, and—where applicable—CYL and AX; change only the explicit spherical-power value.

**Guards:** Do not use a value unless it is explicitly marked as SPH, optical power, or an unambiguous signed lens-power value. Never derive it from an omitted attribute or from a model/article code. For toric lenses, retain the existing CYL and AX exactly.

**Confidence:** strict

**Negative evidence:** pair_0063, pair_0064, pair_0081

**Positive counterexamples:** None observed

**Why reusable:** Pair_0064 isolates a sign-and-magnitude SPH change while retaining the Acuvue toric line, CYL, AX, base curve, diameter, and six-lens pack. The same distinction recurs in other named contact-lens lines.

## Generation pattern 4

**Generalized rule:** For a toric contact lens, a different explicit cylinder-axis (AX) value can define a negative prescription variant.

**Scope:** The same named toric lens line with the same explicit SPH, CYL, base curve, diameter, pack count, and replacement interval.

**Generation operation:** Preserve the named line and all prescription components except the explicit AX; substitute a different valid AX value.

**Guards:** This is guarded because the direct evidence is one pair and its material/oxygen-permeability metadata conflicts across listings. Require the core prescription fields named in Scope to agree, use an explicitly labelled AX value, and do not treat a difference in unrelated technical metadata as the generating signal.

**Confidence:** guarded

**Negative evidence:** pair_0089

**Positive counterexamples:** None observed

**Why reusable:** AX is a dedicated toric-prescription coordinate. The restriction to a fully matching named line and remaining prescription tuple prevents this from becoming a vague rule about differing lens metadata.

## Rejected generalizations

- **Different PD range is always negative:** rejected. `pair_0065` is positive despite PD ranges 64–66 and 60–62, as well as differing lens/coating wording.
- **Different coating text or blue-light/computer designation is always negative:** rejected. `pair_0065` is a direct positive conflict; these fields are also inconsistent across listing schemas.
- **Lubricant flavour, warming effect, use area, or composition wording is always negative:** rejected. `pair_0054` is positive although one listing is generic Wet lubricant and the other explicitly names a cherry, warming, anal-use Wet product.
- **Contact-lens replacement schedule is always negative:** rejected. `pair_0059` contains a schedule discrepancy, but the existing evidence audit identifies the schedule metadata as unreliable; it also has multiple other discrepancies.
- **Different toric CYL alone is a strict rule:** rejected as not isolated by this sample. `pair_0060`, `pair_0079`, and `pair_0095` change CYL together with other prescription components, so they do not independently support a CYL-only generator edit.
- **Photochromic versus fixed tint alone is a strict glasses rule:** rejected as not isolated. `pair_0051` and `pair_0058` also differ in model, prescription, PD, and/or lens family.
- **Frame geometry or material alone is always negative:** rejected as not isolated. `pair_0053` and `pair_0067` combine those fields with different models or other frame specifications, and `pair_0082` illustrates cross-listing inconsistency in frame descriptors for the same named model.
