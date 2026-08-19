## Generation pattern 1

**Generalized rule:** Fixed dining or kitchen tables with an explicitly different tabletop footprint are different products when their construction and product family are otherwise held constant.

**Scope:** Non-adjustable dining, kitchen, or coffee tables with explicit tabletop dimensions; use only when the offered table family, material/construction, and table type are preserved.

**Generation operation:** Preserve the table family, use, material, height, and fixed/non-folding configuration; change an explicit tabletop length or width to a materially different offered footprint.

**Guards:** Do not apply to adjustable or extendable tables, and do not treat a different display order of the same dimensions as a change. Do not use generic furniture dimensions or shipment dimensions.

**Confidence:** strict

**Negative evidence:** pair_0455, pair_0464, pair_0475

**Positive counterexamples:** None observed

**Why reusable:** These independent table pairs retain the same broad use and close construction while the explicitly stated tabletop footprint changes, making size a controlled purchasable variant rather than a name-level difference.

## Generation pattern 2

**Generalized rule:** Fixed industrial or kitchen shelving units with a different complete external size configuration are negative variants even when their material and shelf count agree.

**Scope:** Freestanding shelving/rack listings with explicit width, depth, height, and shelf count; preserve the rack type and material class.

**Generation operation:** Preserve the rack product type, material, shelf count, and construction class; change the explicit width/depth/height configuration to another real size variant.

**Guards:** Require an explicit conflict in the product dimensions, not packaging dimensions. Use this only for fixed-size shelving units; it is not a rule for wardrobes, cabinets, or furniture generally.

**Confidence:** strict

**Negative evidence:** pair_0451, pair_0459, pair_0466

**Positive counterexamples:** None observed

**Why reusable:** Each evidence pair is a same-class shelving offering with the same broad construction cues but incompatible external rack dimensions, so the rule can generate controlled size-variant hard negatives.

## Generation pattern 3

**Generalized rule:** Kitchen countertop panels with different explicit cut specifications are different purchasable panels, even when decor and panel family agree.

**Scope:** Rectangular kitchen countertops with explicit finished-panel length and, where applicable, thickness; keep brand/decor/material family and width fixed when those values are available.

**Generation operation:** Preserve the countertop category, decor, material family, width, and edge/pack configuration; change the explicit cut specification by changing panel length, optionally together with thickness when modeling a known SKU variant.

**Guards:** Do not infer a mismatch from an omitted dimension. This evidence supports a different complete cut specification, not thickness alone; retain explicit numeric units and avoid confusing depth with thickness.

**Confidence:** strict

**Negative evidence:** pair_0482, pair_0498

**Positive counterexamples:** None observed

**Why reusable:** Both pairs describe kitchen countertops with matching decor/family cues and a conflicting finished-panel specification; the generator can make a realistic close alternative by altering the explicitly offered cut size.

## Generation pattern 4

**Generalized rule:** Bathroom vanity sets are negative when the explicitly included basin is a different model or has a different stated geometry/size, while the vanity-set configuration is otherwise comparable.

**Scope:** Multi-component bathroom vanity offerings in which a basin is explicitly included and its model, shape, or dimensions are stated.

**Generation operation:** Preserve the vanity family, installation class, included-component count, and basin inclusion; replace the included basin with a different explicitly specified basin model or geometry/size.

**Guards:** Apply only when both listings actually include a basin. Do not derive a mismatch from color alone, and do not change an unmentioned basin field.

**Confidence:** strict

**Negative evidence:** pair_0476, pair_0484

**Positive counterexamples:** None observed

**Why reusable:** The evidence distinguishes otherwise close Bauedge vanity configurations by their named, included basins, so basin identity is a controlled component-level negative axis.

## Generation pattern 5

**Generalized rule:** A garden-furniture cushion and a swing-seat cushion with explicitly incompatible fitted footprints are different products, even when color matches.

**Scope:** Outdoor cushions whose intended fitted form and length/width are explicitly stated.

**Generation operation:** Preserve outdoor use, color, one-item pack count, and cushion material class; change between a general garden-furniture cushion and a swing-seat cushion with a different explicit footprint.

**Guards:** Require both an explicit intended form and incompatible dimensions. Do not use color, filling, or a vague word “cushion” by itself as a negative axis.

**Confidence:** strict

**Negative evidence:** pair_0470, pair_0483

**Positive counterexamples:** None observed

**Why reusable:** The two repeated grey/green examples isolate the fitted product form plus footprint, producing a realistic hard negative without relying on superficial color differences.

## Generation pattern 6

**Generalized rule:** Beds in the same named model line with an explicitly different sleeping-place width can be generated as negative size variants.

**Scope:** A named bed model line where both listings explicitly state sleeping-place width and retain the same mechanism, length, and core construction.

**Generation operation:** Preserve brand/model line, bed type, length, lifting mechanism, and base construction; change only the explicit sleeping-place width.

**Guards:** Use only when the shared model line and the comparable non-width specifications are explicit. Do not generalize to different bed families, and do not use overall bed width when sleeping-place width is absent.

**Confidence:** guarded

**Negative evidence:** pair_0461

**Positive counterexamples:** None observed

**Why reusable:** The Milton pair shows a clear 140×200 versus 80×200 purchasable size split, but the sample provides only one controlled model-line example, so the rule remains deliberately narrow.

## Generation pattern 7

**Generalized rule:** Storage cabinets with the same external dimensions can be negative when their explicit storage construction and intended placement differ.

**Scope:** Cabinet/tumba listings with comparable outer dimensions and explicit information about open shelves, doors/drawers, and room/use.

**Generation operation:** Preserve outer dimensions and broad cabinet class; change the explicit storage construction from an open bedside shelf unit to a drawer-based office cabinet, together with the stated intended placement.

**Guards:** Treat this as a conjunctive transformation: require both an explicit construction difference and a use/placement difference. Do not infer drawer, door, or room mismatches from omitted fields.

**Confidence:** guarded

**Negative evidence:** pair_0468

**Positive counterexamples:** None observed

**Why reusable:** It produces a close visual/dimensional hard negative while preserving size, but one evidence pair is insufficient to split construction and placement into unconditional independent rules.

## Generation pattern 8

**Generalized rule:** An adult dining-table offering and a children's table-and-chair set are negative when intended user and included furniture configuration are explicitly different.

**Scope:** Table offerings with explicit adult dining use versus a children's set that explicitly includes a chair and/or child-oriented adjustment.

**Generation operation:** Preserve the broad furniture domain and, if desired, wood material; change the intended user and included configuration from a standalone adult dining table to a children's table-and-chair set.

**Guards:** Require explicit evidence of the adult dining configuration and the children's set configuration. Do not use a missing age field or a size difference alone to create this negative.

**Confidence:** guarded

**Negative evidence:** pair_0465

**Positive counterexamples:** None observed

**Why reusable:** The decisive distinction combines intended user and included item rather than relying on an unstructured title difference; it is retained as guarded because the sample contains one direct example.

## Rejected generalizations

- **Any color difference means different furniture:** reject. `pair_0452` is a negative with chiefly “слоновая кость” versus “белый снег”, but the category caveat identifies the two listings as otherwise the same E1 Express wardrobe; `pair_0474` also mixes a color change with a small stated height discrepancy. Neither isolates color as a safe unconditional generation axis.
- **Any furniture dimension difference means a negative:** reject. Positive `pair_0494` changes height and internal configuration, positive `pair_0480` changes reported depth and filling, and positive `pair_0499` changes legs/lifting/storage configuration. Dimension rules must stay within the narrow fixed-product scopes above.
- **Any sofa size, mechanism, or attribute difference means a negative:** reject. Category caveats flag `pair_0472` and `pair_0473` as close or matching sofa specifications with incomplete/conflicting metadata, so they cannot support a reliable synthetic axis.
- **Different internal filling, drawers, legs, or lifting mechanism always means a negative:** reject. Positive `pair_0494` and `pair_0499` directly show such bundled feature differences without a negative label.
- **A brand or normalized type-label mismatch means a negative:** reject. Positive `pair_0477` has conflicting brand metadata for an otherwise identical named sofa, and positive `pair_0478` has inconsistent type/feature metadata for the same named chair model.
