## Generation pattern 1

**Generalized rule:** Two otherwise comparable shoes with different explicitly stated canonical brands are different products.

**Scope:** Any shoe type when each side names a complete brand in a dedicated brand field or an unambiguous product title; orthographic variants of the same brand must first be normalized.

**Generation operation:** Preserve shoe type, model/version (if known), material, colour, size, intended user, season, and pack; replace the canonical brand with a different explicit brand.

**Guards:** Do not infer a brand from a stray keyword, and do not treat a shared token as equality (for example, `Nando Muzi` and `Mario Muzi` are distinct). Exclude aliases, spelling variants, licensed collections, and listings where one brand is absent or ambiguous.

**Confidence:** strict

**Negative evidence:** pair_0563, pair_0565, pair_0569, pair_0590, pair_0593, pair_0598, pair_0599

**Positive counterexamples:** None observed

**Why reusable:** Brand is a product-identity axis across footwear families, and these independent negatives cover sneakers, low shoes, loafers, and boots; preserving the remaining identity fields makes the synthetic contrast substantially harder than the sampled evidence.

## Generation pattern 2

**Generalized rule:** A shoe explicitly self-declared as `no brand` is a negative counterpart to an otherwise comparable shoe with a named canonical brand.

**Scope:** Listings whose displayed brand value or leading title explicitly says `no brand`/equivalent, paired with a listing that explicitly states one named brand.

**Generation operation:** Preserve the shoe family and all non-brand attributes; change an explicit canonical brand to an explicit unbranded declaration.

**Guards:** Never substitute a missing brand with `no brand`. Treat the unbranded declaration as decisive even if marketing text later mentions a famous brand; do not use this rule where that declaration could be a marketplace placeholder rather than listing data.

**Confidence:** guarded

**Negative evidence:** pair_0592

**Positive counterexamples:** None observed

**Why reusable:** The explicit provenance conflict is reusable, but the sample has one direct observation and the second title contains brand-like marketing text, so generation must retain the explicit declaration and remain conservative.

## Generation pattern 3

**Generalized rule:** Different explicit model codes or version numbers within the same canonical shoe brand identify different products.

**Scope:** Same-brand shoes with a clearly parsed model family and mutually different model codes or versions, such as `Galaxy 5` versus `Galaxy 6`.

**Generation operation:** Preserve canonical brand, shoe type, colour, material, intended user, season, size convention, and all non-model fields; replace only the explicit model code or version with another valid code from that brand family.

**Guards:** Do not generate from a generic brand-only title, a missing model, a seller SKU, or a cosmetic colourway suffix. Do not collapse repeated model mentions with different punctuation until they are normalized; ambiguous cases such as the sampled New Balance 574 labels remain excluded.

**Confidence:** guarded

**Negative evidence:** pair_0587

**Positive counterexamples:** None observed

**Why reusable:** A version number is an identity-bearing axis rather than a presentation attribute, but only one sampled negative isolates a clear same-brand version contrast.

## Generation pattern 4

**Generalized rule:** For an otherwise fixed shoe listing, an explicit conflict between a natural-leather upper and an artificial-leather/eco-leather upper can indicate different product variants.

**Scope:** Same explicit brand and shoe type, with both upper materials explicitly stated as one of the two base-material groups; use only when no model/version conflict is being introduced.

**Generation operation:** Preserve brand, shoe type, model/version, colour, size, user group, season, sole, and construction; change only the explicit base material of the upper between natural leather and artificial/eco leather.

**Guards:** This is guarded because the evidence also changes other attributes. Do not use a missing material, a mixed composition, a coating, a lining/insock material, textile, mesh, polymer, suede, or a vague field such as `material` as if it were the upper. The positive labels show that material disagreements outside this exact natural-versus-artificial upper contrast are not reliable negatives.

**Confidence:** guarded

**Negative evidence:** pair_0555, pair_0561, pair_0570

**Positive counterexamples:** pair_0562, pair_0580, pair_0589

**Why reusable:** The repeated negative direction is a meaningful variant axis for shoes, while the tight material vocabulary and single-axis generation operation prevent the broader, contradicted claim that any material difference means a different item.

## Rejected generalizations

- **Size mismatch alone:** reject. Positive pairs `pair_0550`, `pair_0553`, `pair_0558`, `pair_0562`, `pair_0578`, `pair_0582`, and `pair_0594` contain different stated sizes, so negatives such as `pair_0575` cannot justify unconditional size edits.
- **Colour mismatch alone:** reject. Positive pairs `pair_0558`, `pair_0562`, `pair_0578`, `pair_0580`, and `pair_0594` contradict it; colour may only remain an unchanged context field for the rules above.
- **Season or intended-gender mismatch alone:** reject. Positive pairs `pair_0552`, `pair_0558`, `pair_0562`, `pair_0578`, and `pair_0580` contain such changes. The many-attribute child-shoe negative `pair_0555` does not isolate either axis.
- **Any material mismatch is negative:** reject. Positives `pair_0562`, `pair_0580`, and `pair_0589` have materially different descriptions; only the narrowly guarded natural-leather versus artificial/eco-leather upper operation is retained.
- **Different sport purpose is negative:** reject. `pair_0571` is a negative Mizuno athletics-versus-badminton pair, but positive `pair_0594` associates badminton and running language with the same brand, and the negative also changes model, material, colour, and size.
- **Near types or close brand spellings are inherently different:** reject. Positives `pair_0553` (мокасины/лоферы), `pair_0562` (кроссовки/кеды), and `pair_0578` (кроссовки/кеды) rule out a broad close-type rule. Negatives `pair_0568`, `pair_0585`, `pair_0586`, and `pair_0595` are noisy or ambiguous, so they must not be used to turn spelling, punctuation, or an identical model name into a negative signal.
