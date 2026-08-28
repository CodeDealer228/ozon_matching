# A2 / Qwen pilot: 30 human-negative sources

Run: 2026-08-24.  Model: `qwen3.8-27b`; 6 workers; temperature 0.25;
reasoning disabled; one paid attempt per source.  Source pool was rebuilt from
canonical human train only: 500 rows, zero validation-item overlap.  This
experiment contains 30 sources: six from each of Автотовары, Галантерея и
аксессуары, Обувь, Одежда, Ювелирные изделия.

## Outcome

- Requests: 30 complete, 0 API errors, 0 uncertain/retried requests.
- Deterministic validator: 8 accepted, 22 rejected.
- Manual audit: 3 probable source-label errors; 4 good rewrites were false
  rejects; 23 decisions were correct.  Therefore 10/30 rewrites are usable
  after manual correction, not the raw 8/30 automatic accepts.

The full prompts (without the system prompt), hidden `attributes1`, raw Qwen
responses and re-evaluated decisions are printed by
`scripts/show_a2_pilot_results.py`.  Raw records and parquet results are in
`artifacts/synthetic_v2/a2_pilot_30/`.

## Manual audit

`M` = probable human-label error; `V` = good generation, false validator
reject; `OK` = source and validator decision are sound.

| # | Verdict | Reason |
|---:|:---:|---|
| 1 | M | Continental tyres have the same model and full size/specification. |
| 2 | V | `RA4041` vs `RA4103` remains intact; Qwen merely aligns word order. |
| 3 | OK | INUVU versus no-brand sunglasses; unchanged reply was rejected. |
| 4 | OK | Stray Kids `#006` remains a meaningful distinction. |
| 5 | OK | Landcruiser Prado 120 versus Landcruiser 100. |
| 6 | V | `1 шт` is confirmed by attributes B, not copied from A. |
| 7 | OK | 5 kg versus 1 kg remains intact; G11 is in attributes B. |
| 8 | V | `кожаная` is supported by B's `натуральная кожа`. |
| 9 | OK | New Balance 9060 versus generic New Balance. |
| 10 | OK | Air Force 1 Low '07 is a model distinction. |
| 11 | OK | Sneakers versus keds remains intact. |
| 12 | OK | Air Force 1 '07 distinction remains intact. |
| 13 | OK | Jordan versus Air Jordan 4. |
| 14 | OK | Adding B's TMI brand makes the pair easier, so reject is correct. |
| 15 | OK | Lemforder articles `2660402` versus `2240201`. |
| 16 | M | Only `не определен` differs; it is not a product fact. |
| 17 | OK | Jacques Lemans model `1-2117A` versus generic title. |
| 18 | OK | Loafers versus keds. |
| 19 | OK | Pepe Jeans variants differ in attributes. |
| 20 | OK | `бельевая` was introduced from A without B evidence. |
| 21 | OK | B attributes say 40 den despite its title saying 20 den. |
| 22 | OK | Bra variants differ by colour and size. |
| 23 | OK | Sundresses differ by brand and attributes. |
| 24 | OK | Mama's Fantasy rewrite preserves B and improves style. |
| 25 | M | Platina tie clips have the same product facts; 5.09/5.19 g alone is weak. |
| 26 | OK | Lucente rings differ by plating, stone colour and size. |
| 27 | OK | 585gold versus VG. |
| 28 | OK | Platina pendant variants have different weights. |
| 29 | V | Word order is closer to A while B's facts remain; similarity rejected it. |
| 30 | OK | Alkor rings differ by size 17.5 versus 19.5. |
