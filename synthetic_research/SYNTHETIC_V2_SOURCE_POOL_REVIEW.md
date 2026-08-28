# Synthetic v2 source-pool review

## Результат

Source pools построены только из canonical human train. Validation использовался исключительно как deny-list для проверки утечки; validation content, labels, item IDs, пары и значения в генерацию не передавались.

Identity checks:

- split SHA-256: `1e3df234d6cdb8685256d5144b25f9a0a3e3a023f1053a0bd5b16c40e3df8d7e`;
- validation fingerprint: `847d04d081432cc081c2149b75271547bf5e25b14f9d8a5804704a5be2a4b3c8`;
- train rows: `256006`;
- validation rows: `109648`;
- shared train/validation items: `0`;
- unordered train/validation pair overlap: `0`;
- A2 validation item overlap: `0`;
- C validation item overlap: `0`.

Локальный `items_human.parquet`:

- bytes: `214210451`;
- SHA-256: `07d4f09ac8cfa90a2dfc2a7802dec09fc1e188576e1281e781aa1340b3336f4a`.

## A2

Готово `500` source pairs — по `100` human-train negatives для каждой из пяти целевых категорий:

- Автотовары;
- Галантерея и аксессуары;
- Обувь;
- Одежда;
- Ювелирные изделия.

Каждый item используется не более одного раза. Названия имеют умеренно высокое сходство в диапазоне `[0.45, 0.90]`: почти одинаковые названия исключены, поскольку их нельзя содержательно усложнить без изменения identity. В A2 разрешено менять только `name2`; `attributes2`, category и все identity-bearing facts остаются неизменными.

## C

Строгий train gate не прошла ни одна гипотеза. После исправления exact-identifier проверки обнаружилось, что первоначально кажущееся большое семейство ювелирных артикулов было ложным объединением по общему числовому фрагменту. Такое объединение отклонено.

Итог строгой повторной проверки:

- `60` train research candidates;
- `0` approved per-family sources;
- `0` replacement candidates;
- широкая одежная гипотеза `exact name skeleton + color` имеет `53` negative examples, но также `6` human-positive counterexamples, а цвет не подтверждён как видимый в названии;
- гипотезы по обуви и ювелирным изделиям не набрали строгий support после исключения частичных совпадений артикулов.

Автоматически ослаблять C gate запрещено. C остаётся отдельной исследовательской веткой и не входит в текущий pilot. Это не блокирует независимый A2 pilot, как предписывает pipeline v2.

## Воспроизводимость

- builder: `scripts/build_synthetic_v2_source_pools.py`;
- local outputs: `artifacts/synthetic_v2/source_pools`;
- A2 pool SHA-256: `2dee9a0b5b2ef54c4210b07b114346057bddc45effe728e2e43056f42505dd28`;
- C research candidates SHA-256: `37c4727c0e75cd5ea3e4a18578d2ce11a90dfebd1b1ef8ccd8393a07542c6985`;
- C replacement pool SHA-256: `512e89095b3fe44f16a9524cab06e0b5fd4946586489ae369648aeab55cb8785`.

