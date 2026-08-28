# Grounded specification v2: full result

## Итог генерации

После провала identity-bundle pilot массовая ветка переведена на не-identity характеристики. Source — только positive human-train пара из canonical item-disjoint split. Source-значение обязано встречаться ровно один раз в каждом title, относиться ровно к одной общей роли атрибута и не быть частью десятичного числа. Target берётся из другой реальной human-train пары той же категории, роли и близкого title-контекста. Свободная генерация строк отсутствует.

Полный output: `artifacts/token_replacement_counterfactual/grounded_spec_v2_full_12k`.

- операций: 6 000;
- synthetic pairs: 12 000;
- positive / negative: 6 000 / 6 000;
- уникальных source pairs: 2 328;
- ordered/unordered duplicates: 0 / 0;
- validation source overlap: 0;
- `synthetic_items.parquet` SHA-256: `2af60109cf71e045dd36b4b96c108c74eb0a267b18b602c6b14d875bf690aabd`;
- `synthetic_pairs.parquet` SHA-256: `19c37860909cf5e390833cb549747aba803c603fe13868ab9f8a4a9171a8ce1a`;
- prepared operations SHA-256: `342489ba289eeb273b406b2eae94d64c594472b0f069f0b9acd9d79b217cd97f`.

Роли: volume 2 798, count 1 563, weight 842, height 260, length 168, diameter 148, width 118, size 97, format 4, resolution 2. Power исключён как зависимая характеристика: отдельная замена мощности может противоречить напряжению и току.

## Quality gate

Три последовательных pilot дали конкретные исправления: запрет фрагментов десятичных чисел, запрет нескольких значений одной роли в title, проверку единицы рядом с числом, запрет экстремального отношения для объёма/веса/габаритов и minimum two-token overlap с evidence-title. В финальном pilot вручную просмотрены 100 из 100 операций по всем доступным категориям; систематических identity/type ошибок не найдено.

В наборе 19 категорий. `Обувь` намеренно не синтезировалась: её positive human-train пары часто уже содержат разные размеры, а размер отсутствует в одном из title. Ослабление gate ради формального покрытия двадцатой категории ухудшило бы метки.

## BERTA experiment contract

C1 и T1 используют один paired manifest. Human train одинаков. Для каждой synthetic replay строки C1 получает реальную human-train пару той же категории и метки, T1 — synthetic pair. Replay factor 4, individual replay weight 0.5, порядок/seed/steps идентичны. Canonical item-disjoint validation (109 648 строк) используется только для evaluation.

- paired rows: 304 128;
- steps: 594 при global batch 512;
- replay rows: 48 000;
- paired manifest SHA-256: `b8a4b01d158eb3a3d3aafdc3391a231a876b8cd2a50abbd0d79fc8fc81a7d28a`;
- validation manifest SHA-256: `98e783670d63e7e43f493c2801dd82c069adf547fe1057be9cd5c7397c7da47f`.
