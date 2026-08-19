# Synthesis-ready negative patterns — Дом и сад

## Generation pattern 1

**Generalized rule:** An explicit mismatch in finished dimensions is a negative for otherwise comparable home-textile, floor-covering, storage, or tableware variants.

**Scope:** The same concrete product family and sold form: for example entrance mat with entrance mat, ready-made tulle with ready-made tulle, tablecloth with tablecloth, carpet with carpet, or the same kind of storage basket. The rule is about a finished purchasable size, not package dimensions.

**Generation operation:** Preserve the concrete product type, intended use, material/construction, pack configuration, and brand/model family when known; change one or more explicitly stated finished width, length, height, diameter, or the complete dimension tuple to a conflicting SKU value.

**Guards:** Both values must be explicit, normalized to the same unit, and refer to the item rather than its shipping box. Do not apply across different product families or when the apparent size is coupled with an unseparated type, print, or feature change.

**Confidence:** strict

**Negative evidence:** pair_0300, pair_0301, pair_0304, pair_0317, pair_0327, pair_0328, pair_0339

**Positive counterexamples:** None observed

**Why reusable:** Independent negatives cover several comparable product families and repeatedly make the finished size the clear SKU-level difference; this is broader than any one carpet, curtain, basket, or tablecloth listing while retaining a tight same-family guard.

## Generation pattern 2

**Generalized rule:** An explicit capacity mismatch is a negative for otherwise comparable cookware or serving-vessel variants.

**Scope:** Kettles, saucepans, bowls, and kazans compared within the same vessel type; use the rule most confidently when the brand/model family, material, and vessel role agree.

**Generation operation:** Preserve vessel type, material, lid/feature configuration, intended use, and brand/model family where available; change only one explicit volume/capacity value to a different value.

**Guards:** Both capacities must be explicit and measured on the same basis. Do not replace a vessel type (for example, saucepan with kazan) under this rule, and do not infer a capacity from dimensions or a missing field.

**Confidence:** strict

**Negative evidence:** pair_0303, pair_0314, pair_0318, pair_0341, pair_0348

**Positive counterexamples:** None observed

**Why reusable:** Five independent negative pairs show different stated capacities for a comparable vessel, including otherwise closely aligned saucepan and Afghan-kazan listings; the decisive axis is capacity rather than a listing’s incidental colour or origin.

## Generation pattern 3

**Generalized rule:** An explicitly different named visual or cultural content variant is a negative for otherwise comparable decorative goods.

**Scope:** Posters, music posters, decorative-print bedding, icons, and comparable decor whose subject, print name, artist/track, or depicted figure is an explicit version-defining field.

**Generation operation:** Preserve the decorative product type, format/size, medium, framing/presentation, and collection/brand when known; replace one explicit named content value with another incompatible named content value.

**Guards:** Require two explicit, different content values. Do not treat a missing print/subject field, a colour name, or generic decorative wording as a content mismatch; do not use this rule where a print change is inseparable from a product-type change.

**Confidence:** strict

**Negative evidence:** pair_0311, pair_0332, pair_0337, pair_0344

**Positive counterexamples:** None observed

**Why reusable:** The negatives repeat the same identity axis across posters, bedding, and icons: the declared subject/content changes while the broad decorative form remains comparable, so the rule is not tied to a single track, motif, or saint.

## Generation pattern 4

**Generalized rule:** Two bedding listings are negative when they explicitly name non-interchangeable bedding components, such as a fitted sheet versus a duvet cover.

**Scope:** Individual bedding components rather than complete bedding sets.

**Generation operation:** Preserve fabric, broad bedding context, and any intentionally shared size/material information; replace the explicit component type and update its component-specific configuration coherently (for example, fitted-sheet elastic versus duvet-cover closure).

**Guards:** Both component types must be explicit and mutually non-interchangeable. Do not infer a component from a size alone, and do not use the rule for sparse listings or a complete set versus one of its components.

**Confidence:** guarded

**Negative evidence:** pair_0319

**Positive counterexamples:** None observed

**Why reusable:** The sampled negative shows that percale and bedding context do not make a sheet and a duvet cover the same SKU, but only one clean component-level example is available, so the rule remains guarded.

## Generation pattern 5

**Generalized rule:** Seeds of the same explicitly named crop are negative when their explicit cultivar/variety names differ.

**Scope:** Seed offers for the same crop species, with an explicit named cultivar on both sides.

**Generation operation:** Preserve crop species, seed product form, pack configuration, and brand family where available; change only the explicit cultivar/variety value.

**Guards:** Require an explicit crop match and two distinct explicit cultivar names. Do not substitute a cultivar merely because one listing omits it, and do not apply the rule to seedlings, generic plant categories, or broader species changes.

**Confidence:** guarded

**Negative evidence:** pair_0340

**Positive counterexamples:** None observed

**Why reusable:** Cultivar is a reusable agricultural identity field rather than a cosmetic title phrase, but the sample provides one crop-level example only.

## Generation pattern 6

**Generalized rule:** Otherwise comparable tea-light candle packs are negative when both offers explicitly state different fragrance variants.

**Scope:** Same candle form and explicit pack count, specifically comparable scented tea-light packs.

**Generation operation:** Preserve candle form, explicit unit count, and scented-candle product family; replace one explicit fragrance value with another explicit fragrance value.

**Guards:** Require the same candle form and explicit equal pack counts on both sides. Do not use a missing scent as a mismatch and do not extend this rule to decorative versus aromatic candle types, where product form/function also changes.

**Confidence:** guarded

**Negative evidence:** pair_0335

**Positive counterexamples:** None observed

**Why reusable:** The pair keeps the six-piece tea-light configuration while changing the named fragrance from ocean to white lily, but one example is insufficient to elevate scent to a strict rule.

## Rejected generalizations

- **A different brand, country, or material is always a negative.** Rejected: positive `pair_0316` describes matching garlic presses despite differing material and country fields, while positive `pair_0324` describes 33 cm magnetic knife holders despite brand/material-field differences; these attributes are not standalone generation axes.
- **Any coarse product-type wording difference is a negative.** Rejected: positive `pair_0306` uses «овощечистка» versus «фрукто-овощечистка», and positive `pair_0342` uses carpet versus runner wording for same-size floor coverings. A type edit requires an explicit non-interchangeable function/component, not a broad noun mismatch.
- **Any dimension difference across the whole category is a negative.** Rejected as unconditional: `pair_0319`, `pair_0325`, and `pair_0336` combine size with a component, feature-set, or print difference, so they do not isolate size. Dimension changes are allowed only in the same-family scope of Generation pattern 1.
- **An omitted decorative design, scent, or specification is a conflicting value.** Rejected: positive `pair_0322` pairs a detailed «красные ягоды и шарики» wreath listing with a generic wreath listing, and caveat `pair_0323` has near-identical garlands with uneven detail. Generate only explicit value-versus-value conflicts.
- **A product title/model phrase by itself establishes a negative.** Rejected: caveats `pair_0310`, `pair_0321`, `pair_0323`, `pair_0333`, `pair_0346`, and `pair_0349` show same or near-same titles, sparse fields, or unreliable negative labels; an explicit decisive attribute is required.
