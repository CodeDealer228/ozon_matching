# Исследование правил для synthetic hard negatives

> Исследование выполнено только на `split == "train"` canonical human split. Human validation не использовался для выбора aliases, правил, порогов или примеров.

## Данные и методика

- Canonical human train до защиты: **292,523** пар.
- Исключено train-пар с validation item ID: **7,727**; пересекающихся item ID: **5,534**.
- Human train после защиты, все категории: **284,796** пар.
- Четыре целевые категории: **61,943** пар.
- Товаров целевых категорий в `items_human`: **121,290**.
- Значения сравнивались после нормализации регистра, `ё/е`, единиц измерения и пунктуации.
- Aliases объединялись только по точным, заранее проверенным ключам; близкие, но разные поля оставлены раздельно.
- Near-identical: сходство названий не ниже заданного порога, не более одного дополнительного semantic-различия и хотя бы одна совпадающая известная характеристика.
- Validation использован только как deny-list item ID; его labels и item content не читались для исследования.
- `safe_negative` назначался только при низкой доле различий в positive и сильном near-identical negative evidence.
- При слабом support или противоречии выставлялся `uncertain`.

### Объём human train

| Категория | Baseline AP | Positive | Negative | Всего |
|---|---:|---:|---:|---:|
| Обувь | 0.513830 | 1,449 | 13,007 | 14,456 |
| Ювелирные изделия | 0.530127 | 1,774 | 12,185 | 13,959 |
| Одежда | 0.543374 | 2,178 | 16,202 | 18,380 |
| Автотовары | 0.694293 | 2,671 | 12,477 | 15,148 |

## Таблица правил

### Обувь

| Semantic attribute | Статус | Known + / − | Diff + / − | P(diff\|+) | P(diff\|−) | Near + / − | В name |
|---|---|---:|---:|---:|---:|---:|---:|
| `brand` | **unsafe_variant** | 599 / 2,629 | 108 / 1,374 | 18.03% | 52.26% | 0 / 0 | 97.86% |
| `article` | **uncertain** | 0 / 0 | 0 / 0 | — | — | 0 / 0 | 21.47% |
| `product_model` | **uncertain** | 1 / 29 | 0 / 17 | 0.00% | 58.62% | 0 / 0 | 80.12% |
| `product_color` | **unsafe_variant** | 1,360 / 12,572 | 355 / 6,855 | 26.10% | 54.53% | 11 / 382 | 15.04% |
| `gender` | **unsafe_variant** | 490 / 1,813 | 171 / 698 | 34.90% | 38.50% | 1 / 1 | 0.59% |
| `target_audience` | **uncertain** | 0 / 1 | 0 / 1 | — | 100.00% | 0 / 0 | 0.00% |
| `russian_size` | **unsafe_variant** | 697 / 6,757 | 581 / 5,949 | 83.36% | 88.04% | 19 / 391 | 5.94% |
| `manufacturer_size` | **unsafe_variant** | 238 / 989 | 229 / 972 | 96.22% | 98.28% | 0 / 0 | 14.91% |
| `generic_size` | **uncertain** | 1 / 1 | 1 / 1 | 100.00% | 100.00% | 0 / 0 | 54.30% |
| `foot_length` | **uncertain** | 18 / 131 | 15 / 99 | 83.33% | 75.57% | 0 / 0 | 0.05% |
| `insole_length` | **unsafe_variant** | 38 / 97 | 33 / 92 | 86.84% | 94.85% | 0 / 0 | 0.03% |
| `shoe_fullness` | **unsafe_variant** | 199 / 2,389 | 119 / 1,755 | 59.80% | 73.46% | 4 / 29 | 0.65% |
| `upper_material` | **unsafe_variant** | 788 / 5,926 | 229 / 2,877 | 29.06% | 48.55% | 12 / 61 | 0.26% |
| `sole_material` | **unsafe_variant** | 797 / 8,500 | 294 / 5,088 | 36.89% | 59.86% | 17 / 92 | 0.08% |
| `insole_material` | **unsafe_variant** | 505 / 4,964 | 153 / 2,579 | 30.30% | 51.95% | 3 / 6 | 0.82% |
| `lining_material` | **unsafe_variant** | 814 / 8,354 | 229 / 4,636 | 28.13% | 55.49% | 4 / 52 | 0.58% |
| `general_material` | **unsafe_variant** | 306 / 5,190 | 122 / 2,743 | 39.87% | 52.85% | 7 / 82 | 1.36% |
| `completeness` | **uncertain** | 0 / 0 | 0 / 0 | — | — | 0 / 0 | 16.45% |
| `season` | **unsafe_variant** | 1,074 / 10,201 | 661 / 7,342 | 61.55% | 71.97% | 18 / 196 | 1.21% |
| `collection` | **unsafe_variant** | 49 / 1,140 | 40 / 1,031 | 81.63% | 90.44% | 5 / 7 | 0.10% |

### Ювелирные изделия

| Semantic attribute | Статус | Known + / − | Diff + / − | P(diff\|+) | P(diff\|−) | Near + / − | В name |
|---|---|---:|---:|---:|---:|---:|---:|
| `brand` | **unsafe_variant** | 628 / 2,268 | 214 / 1,037 | 34.08% | 45.72% | 0 / 0 | 87.36% |
| `article` | **unsafe_variant** | 810 / 3,335 | 375 / 2,838 | 46.30% | 85.10% | 0 / 1 | 14.01% |
| `product_model` | **unsafe_variant** | 530 / 1,481 | 387 / 1,072 | 73.02% | 72.38% | 0 / 0 | 52.06% |
| `product_color` | **unsafe_variant** | 336 / 3,792 | 165 / 2,274 | 49.11% | 59.97% | 6 / 72 | 13.07% |
| `stone_color` | **uncertain** | 0 / 0 | 0 / 0 | — | — | 0 / 0 | 0.17% |
| `metal_color` | **uncertain** | 0 / 0 | 0 / 0 | — | — | 0 / 0 | 0.18% |
| `gender` | **unsafe_variant** | 1,218 / 5,840 | 360 / 1,655 | 29.56% | 28.34% | 0 / 0 | 0.85% |
| `item_size` | **unsafe_variant** | 607 / 2,210 | 345 / 1,516 | 56.84% | 68.60% | 0 / 0 | 27.30% |
| `ring_size` | **uncertain** | 0 / 0 | 0 / 0 | — | — | 0 / 0 | 49.05% |
| `dimensions` | **uncertain** | 6 / 73 | 1 / 43 | 16.67% | 58.90% | 0 / 0 | 1.24% |
| `average_weight` | **unsafe_variant** | 1,093 / 5,533 | 952 / 5,403 | 87.10% | 97.65% | 0 / 14 | 1.30% |
| `item_weight` | **uncertain** | 4 / 83 | 3 / 83 | 75.00% | 100.00% | 0 / 0 | 0.03% |
| `metal_fineness` | **uncertain** | 1,685 / 11,303 | 31 / 363 | 1.84% | 3.21% | 0 / 0 | 66.02% |
| `jewelry_material` | **uncertain** | 901 / 3,592 | 1 / 62 | 0.11% | 1.73% | 0 / 0 | 37.24% |
| `jewelry_composition` | **uncertain** | 233 / 3,027 | 1 / 187 | 0.43% | 6.18% | 0 / 1 | 37.26% |
| `main_insert` | **unsafe_variant** | 1,294 / 8,556 | 423 / 4,460 | 32.69% | 52.13% | 12 / 144 | 46.49% |
| `stone_count` | **uncertain** | 0 / 0 | 0 / 0 | — | — | 0 / 0 | 0.03% |
| `completeness` | **unsafe_variant** | 601 / 5,482 | 424 / 4,797 | 70.55% | 87.50% | 21 / 179 | 8.58% |
| `collection` | **unsafe_variant** | 377 / 1,217 | 274 / 823 | 72.68% | 67.63% | 0 / 0 | 0.26% |
| `item_type` | **unsafe_variant** | 1,284 / 6,490 | 610 / 3,693 | 47.51% | 56.90% | 2 / 16 | 67.89% |

### Одежда

| Semantic attribute | Статус | Known + / − | Diff + / − | P(diff\|+) | P(diff\|−) | Near + / − | В name |
|---|---|---:|---:|---:|---:|---:|---:|
| `brand` | **unsafe_variant** | 239 / 821 | 47 / 565 | 19.67% | 68.82% | 0 / 0 | 89.45% |
| `article` | **uncertain** | 0 / 0 | 0 / 0 | — | — | 0 / 0 | 9.36% |
| `product_model` | **unsafe_variant** | 160 / 528 | 118 / 457 | 73.75% | 86.55% | 0 / 0 | 22.53% |
| `product_color` | **unsafe_variant** | 1,949 / 15,346 | 614 / 9,496 | 31.50% | 61.88% | 44 / 752 | 12.05% |
| `gender` | **unsafe_variant** | 716 / 3,562 | 242 / 1,448 | 33.80% | 40.65% | 1 / 0 | 2.46% |
| `russian_size` | **unsafe_variant** | 772 / 8,588 | 592 / 7,292 | 76.68% | 84.91% | 17 / 345 | 1.88% |
| `manufacturer_size` | **unsafe_variant** | 191 / 683 | 161 / 607 | 84.29% | 88.87% | 0 / 0 | 6.10% |
| `generic_size` | **uncertain** | 0 / 12 | 0 / 11 | — | 91.67% | 0 / 0 | 61.06% |
| `jeans_size` | **uncertain** | 0 / 0 | 0 / 0 | — | — | 0 / 0 | 0.00% |
| `height_size` | **unsafe_variant** | 31 / 94 | 18 / 57 | 58.06% | 60.64% | 0 / 0 | 2.32% |
| `material_composition` | **unsafe_variant** | 1,466 / 12,450 | 294 / 4,484 | 20.05% | 36.02% | 7 / 123 | 0.84% |
| `lining_material` | **unsafe_variant** | 94 / 989 | 41 / 521 | 43.62% | 52.68% | 0 / 6 | 0.65% |
| `package_count` | **uncertain** | 1 / 7 | 1 / 2 | 100.00% | 28.57% | 0 / 0 | 18.82% |
| `completeness` | **unsafe_variant** | 90 / 683 | 86 / 668 | 95.56% | 97.80% | 1 / 5 | 16.46% |
| `season` | **unsafe_variant** | 751 / 6,561 | 591 / 5,566 | 78.70% | 84.83% | 70 / 410 | 0.98% |
| `collection` | **unsafe_variant** | 24 / 1,177 | 19 / 1,042 | 79.17% | 88.53% | 5 / 6 | 0.06% |
| `item_type` | **unsafe_variant** | 914 / 3,713 | 293 / 1,255 | 32.06% | 33.80% | 0 / 1 | 96.98% |

### Автотовары

| Semantic attribute | Статус | Known + / − | Diff + / − | P(diff\|+) | P(diff\|−) | Near + / − | В name |
|---|---|---:|---:|---:|---:|---:|---:|
| `brand` | **unsafe_variant** | 585 / 2,995 | 116 / 1,024 | 19.83% | 34.19% | 3 / 1 | 66.53% |
| `article` | **unsafe_variant** | 1,229 / 5,478 | 448 / 5,394 | 36.45% | 98.47% | 11 / 170 | 40.09% |
| `oem_number` | **unsafe_variant** | 193 / 721 | 84 / 677 | 43.52% | 93.90% | 6 / 6 | 10.64% |
| `alternative_articles` | **uncertain** | 0 / 0 | 0 / 0 | — | — | 0 / 0 | 7.61% |
| `product_model` | **uncertain** | 0 / 0 | 0 / 0 | — | — | 0 / 0 | 40.37% |
| `product_color` | **unsafe_variant** | 540 / 2,803 | 93 / 977 | 17.22% | 34.86% | 8 / 86 | 46.84% |
| `general_material` | **unsafe_variant** | 400 / 1,971 | 80 / 658 | 20.00% | 33.38% | 6 / 15 | 37.58% |
| `package_count` | **unsafe_variant** | 28 / 168 | 23 / 137 | 82.14% | 81.55% | 0 / 20 | 26.54% |
| `completeness` | **unsafe_variant** | 582 / 2,712 | 414 / 2,305 | 71.13% | 84.99% | 20 / 74 | 13.36% |
| `compatibility` | **uncertain** | 0 / 0 | 0 / 0 | — | — | 0 / 0 | 68.63% |
| `car_make` | **unsafe_variant** | 148 / 844 | 104 / 451 | 70.27% | 53.44% | 2 / 3 | 59.60% |
| `car_model` | **unsafe_variant** | 26 / 220 | 22 / 215 | 84.62% | 97.73% | 1 / 2 | 47.06% |
| `car_generation` | **uncertain** | 0 / 0 | 0 / 0 | — | — | 0 / 0 | 0.00% |
| `car_year` | **uncertain** | 0 / 0 | 0 / 0 | — | — | 0 / 0 | 29.66% |
| `engine` | **uncertain** | 4 / 10 | 4 / 10 | 100.00% | 100.00% | 0 / 1 | 0.53% |
| `placement` | **uncertain** | 18 / 77 | 17 / 67 | 94.44% | 87.01% | 0 / 2 | 24.67% |
| `volume` | **unsafe_variant** | 70 / 475 | 61 / 429 | 87.14% | 90.32% | 6 / 36 | 57.74% |
| `diameter` | **unsafe_variant** | 95 / 639 | 17 / 391 | 17.89% | 61.19% | 0 / 0 | 28.83% |
| `length` | **uncertain** | 12 / 41 | 12 / 41 | 100.00% | 100.00% | 0 / 1 | 3.99% |
| `width` | **uncertain** | 9 / 43 | 9 / 43 | 100.00% | 100.00% | 0 / 1 | 3.01% |
| `height` | **uncertain** | 4 / 11 | 4 / 10 | 100.00% | 90.91% | 0 / 0 | 1.60% |
| `power` | **uncertain** | 0 / 1 | 0 / 0 | — | 0.00% | 0 / 0 | 15.62% |
| `tire_width` | **unsafe_variant** | 117 / 814 | 27 / 523 | 23.08% | 64.25% | 0 / 0 | 98.87% |
| `tire_profile` | **unsafe_variant** | 126 / 850 | 23 / 575 | 18.25% | 67.65% | 0 / 0 | 91.72% |
| `tire_diameter` | **uncertain** | 0 / 0 | 0 / 0 | — | — | 0 / 0 | 5.00% |
| `item_type` | **unsafe_variant** | 783 / 3,838 | 466 / 2,277 | 59.51% | 59.33% | 13 / 79 | 43.07% |
| `seasonality` | **unsafe_variant** | 179 / 1,044 | 144 / 950 | 80.45% | 91.00% | 6 / 3 | 2.56% |

## Группы решений

### `safe_negative`

- Нет правил, прошедших консервативные критерии.

### `unsafe_variant`

- **Обувь:** `brand`, `product_color`, `gender`, `russian_size`, `manufacturer_size`, `insole_length`, `shoe_fullness`, `upper_material`, `sole_material`, `insole_material`, `lining_material`, `general_material`, `season`, `collection`
- **Ювелирные изделия:** `brand`, `article`, `product_model`, `product_color`, `gender`, `item_size`, `average_weight`, `main_insert`, `completeness`, `collection`, `item_type`
- **Одежда:** `brand`, `product_model`, `product_color`, `gender`, `russian_size`, `manufacturer_size`, `height_size`, `material_composition`, `lining_material`, `completeness`, `season`, `collection`, `item_type`
- **Автотовары:** `brand`, `article`, `oem_number`, `product_color`, `general_material`, `package_count`, `completeness`, `car_make`, `car_model`, `volume`, `diameter`, `tire_width`, `tire_profile`, `item_type`, `seasonality`

### `uncertain`

- **Обувь:** `article`, `product_model`, `target_audience`, `generic_size`, `foot_length`, `completeness`
- **Ювелирные изделия:** `stone_color`, `metal_color`, `ring_size`, `dimensions`, `item_weight`, `metal_fineness`, `jewelry_material`, `jewelry_composition`, `stone_count`
- **Одежда:** `article`, `generic_size`, `jeans_size`, `package_count`
- **Автотовары:** `alternative_articles`, `product_model`, `compatibility`, `car_generation`, `car_year`, `engine`, `placement`, `length`, `width`, `height`, `power`, `tire_diameter`

## Near-identical evidence

Ниже приведены реальные пары только из human train. Показываются характеристики с наибольшим near-identical support; примеры не являются синтетическими.

### Обувь

#### `russian_size` — unsafe_variant; near +/− = 19/391

- label=0, sim=1.000, ID 11584 ↔ 266287980387; `['43']` → `['44']`; other diffs=['product_color']
  - A: hollywood hills / кеды
  - B: кеды hollywood hills
- label=0, sim=1.000, ID 56403 ↔ 120259143316; `['41']` → `['40']`; other diffs=['shoe_fullness']
  - A: t.taccardi / босоножки
  - B: босоножки t.taccardi
- label=0, sim=1.000, ID 90497 ↔ 661425069336; `['36']` → `['45']`; other diffs=['sole_material']
  - A: ботинки baden
  - B: baden / ботинки

#### `product_color` — unsafe_variant; near +/− = 11/382

- label=0, sim=1.000, ID 11584 ↔ 266287980387; `['рыжий']` → `['голубой']`; other diffs=['russian_size']
  - A: hollywood hills / кеды
  - B: кеды hollywood hills
- label=0, sim=1.000, ID 50550 ↔ 68719606572; `['white']` → `['black']`; other diffs=['upper_material']
  - A: кроссовки anta
  - B: кроссовки anta
- label=0, sim=1.000, ID 68560 ↔ 635655305316; `['темно-бордовый']` → `['black']`; other diffs=[]
  - A: туфли giovanni fabiani
  - B: giovanni fabiani / туфли

#### `season` — unsafe_variant; near +/− = 18/196

- label=0, sim=1.000, ID 99706 ↔ 695784793046; `['демисезон']` → `['на любой сезон']`; other diffs=['shoe_fullness']
  - A: honey girl / кроссовки
  - B: кроссовки honey girl
- label=0, sim=1.000, ID 144070 ↔ 326417577287; `['на любой сезон']` → `['лето']`; other diffs=['product_color']
  - A: сандалии tapiboo
  - B: сандалии tapiboo
- label=0, sim=1.000, ID 144070 ↔ 678604985613; `['на любой сезон']` → `['лето']`; other diffs=['product_color']
  - A: сандалии tapiboo
  - B: сандалии tapiboo

#### `sole_material` — unsafe_variant; near +/− = 17/92

- label=0, sim=1.000, ID 90497 ↔ 661425069336; `['rubber']` → `['polyurethane']`; other diffs=['russian_size']
  - A: ботинки baden
  - B: baden / ботинки
- label=0, sim=1.000, ID 8590007290 ↔ 128849159672; `['tep']` → `['polyurethane']`; other diffs=['product_color']
  - A: кроссовки сказка
  - B: кроссовки сказка
- label=0, sim=1.000, ID 17179892686 ↔ 352187397622; `['polyurethane']` → `['tep']`; other diffs=['season']
  - A: туфли котофей
  - B: туфли котофей

#### `general_material` — unsafe_variant; near +/− = 7/82

- label=0, sim=1.000, ID 35507 ↔ 747324369166; `['textile', 'wool']` → `['войлок']`; other diffs=['upper_material']
  - A: сапоги aurorastar
  - B: сапоги aurorastar
- label=0, sim=1.000, ID 8589949506 ↔ 833223730003; `['leather']` → `['suede']`; other diffs=['product_color']
  - A: туфли tendance
  - B: tendance / туфли
- label=0, sim=1.000, ID 8589955582 ↔ 824633788923; `['polyurethane']` → `['suede']`; other diffs=['russian_size']
  - A: ботфорты tendance
  - B: tendance / ботфорты

#### `upper_material` — unsafe_variant; near +/− = 12/61

- label=0, sim=1.000, ID 35507 ↔ 747324369166; `['textile']` → `['wool']`; other diffs=['general_material']
  - A: сапоги aurorastar
  - B: сапоги aurorastar
- label=0, sim=1.000, ID 50550 ↔ 68719606572; `['комбинированный материал']` → `['материалы полимерные']`; other diffs=['product_color']
  - A: кроссовки anta
  - B: кроссовки anta
- label=0, sim=1.000, ID 145441 ↔ 197568588427; `['faux_leather']` → `['комбинированный материал']`; other diffs=['product_color']
  - A: кроссовки anta
  - B: кроссовки anta

### Ювелирные изделия

#### `completeness` — unsafe_variant; near +/− = 21/179

- label=0, sim=0.839, ID 89363 ↔ 678604989885; `['браслет подарочная упаковка']` → `['браслет упаковка.']`; other diffs=[]
  - A: platina / браслет из серебра
  - B: platina браслет из серебра 925 пробы
- label=0, sim=0.938, ID 8589970919 ↔ 429496746213; `['серьги 1 пара упаковка бирка']` → `['серьги 1 пара упаковка бирка пломба']`; other diffs=['main_insert']
  - A: мзц / серьги классические из серебра
  - B: серьги классические из серебра
- label=0, sim=0.848, ID 8590044428 ↔ 644245204732; `['кольцо бирка пломба']` → `['кольцо', 'кольцо - 1 pcs . коробочка в подарок - 1 pcs .']`; other diffs=['main_insert']
  - A: lucente кольцо серебро 925 женское
  - B: lucente / кольцо серебро 925 пробы

#### `main_insert` — unsafe_variant; near +/− = 12/144

- label=0, sim=0.929, ID 105098 ↔ 438086753545; `['бриллиант']` → `['бриллиант натуральный']`; other diffs=[]
  - A: русские самоцветы  серьги с бриллиантами "мираж"
  - B: русские самоцветы / серьги с бриллиантами
- label=0, sim=0.938, ID 8589970919 ↔ 429496746213; `['кварц натуральный']` → `['no_insert']`; other diffs=['completeness']
  - A: мзц / серьги классические из серебра
  - B: серьги классические из серебра
- label=0, sim=0.848, ID 8590044428 ↔ 644245204732; `['фианит']` → `['кварц искусственный фианит']`; other diffs=['completeness']
  - A: lucente кольцо серебро 925 женское
  - B: lucente / кольцо серебро 925 пробы

#### `product_color` — unsafe_variant; near +/− = 6/72

- label=0, sim=1.000, ID 34359758742 ↔ 51539621660; `['silver']` → `['прозрачный']`; other diffs=['completeness']
  - A: серебро россии / кольцо женское серебро 925 дорожка
  - B: кольцо женское серебро 925 дорожка серебро россии
- label=0, sim=1.000, ID 34359818675 ↔ 730144531623; `['silver']` → `['red']`; other diffs=['completeness']
  - A: lucente кольцо серебряное
  - B: lucente / кольцо серебряное
- label=0, sim=0.883, ID 34359875368 ↔ 807453891487; `['silver']` → `['grey']`; other diffs=['completeness']
  - A: multisilver кольцо серебро 925 родированное с крупным фианитом
  - B: multisilver / кольцо серебро 925 с крупным фианитом

#### `item_type` — unsafe_variant; near +/− = 2/16

- label=0, sim=0.857, ID 25769931266 ↔ 300647785191; `['кольцо']` → `['кольцо ювелирное']`; other diffs=[]
  - A: oriental кольцо из серебра 925 пробы
  - B: кольцо из серебра 925 пробы
- label=0, sim=0.860, ID 111669154933 ↔ 412316907944; `['зажим для галстука']` → `['зажим ювелирный']`; other diffs=['average_weight']
  - A: platina зажим для галстука platina, красное золото, 585 проба
  - B: platina зажим для галстука из красного золота 585 пробы
- label=0, sim=0.841, ID 137439068267 ↔ 188978723282; `['ложка']` → `['сувенир ювелирный']`; other diffs=['completeness']
  - A: аргента серебряная чайная ложка с логотипом "е" позолота
  - B: серебряная чайная ложка с логотипом "телец" с позолотой

#### `average_weight` — unsafe_variant; near +/− = 0/14

- label=0, sim=0.870, ID 25769809346 ↔ 369367247074; `['3.65']` → `['2.99']`; other diffs=[]
  - A: эстет серьги серьги из желтого золота 585 пробы с фианитами 01с1313714
  - B: эстет серьги из желтого золота 585 пробы с фианитом 01с135657
- label=0, sim=0.925, ID 103079226150 ↔ 137439058841; `['3.97']` → `['3.03']`; other diffs=['main_insert']
  - A: кольцо platina из золота 585 пробы с бриллиантом
  - B: кольцо platina из золота 585 пробы с фианитом
- label=0, sim=0.860, ID 111669154933 ↔ 412316907944; `['4.56']` → `['4.95']`; other diffs=['item_type']
  - A: platina зажим для галстука platina, красное золото, 585 проба
  - B: platina зажим для галстука из красного золота 585 пробы

#### `article` — unsafe_variant; near +/− = 0/1

- label=0, sim=0.822, ID 712964672642 ↔ 25769966801; `['55994']` → `['ко0147', 'ко0147170']`; other diffs=['item_type']
  - A: milana silver кольцо milana silver, серебро, 925 проба
  - B: milana silver кольцо серебро 925 пробы

### Одежда

#### `product_color` — unsafe_variant; near +/− = 44/752

- label=0, sim=1.000, ID 19648 ↔ 377957144651; `['blue']` → `['бирюзовый', 'бирюзовый-синий']`; other diffs=[]
  - A: naraswim / купальник слитный
  - B: купальник слитный naraswim
- label=0, sim=1.000, ID 50081 ↔ 420906879596; `['blue']` → `['сиреневый']`; other diffs=['russian_size']
  - A: пуловер u.s. polo assn.
  - B: u.s. polo assn. / пуловер
- label=0, sim=1.000, ID 102924 ↔ 360777380402; `['blue', 'green']` → `['silver']`; other diffs=['russian_size']
  - A: smena / жакет
  - B: жакет smena

#### `season` — unsafe_variant; near +/− = 70/410

- label=0, sim=0.966, ID 2389 ↔ 300647871634; `['постоянный']` → `['на любой сезон']`; other diffs=[]
  - A: пижама lets go
  - B: пижама let's go
- label=0, sim=1.000, ID 53457 ↔ 549755884639; `['постоянный']` → `['на любой сезон']`; other diffs=[]
  - A: джинсы s.oliver
  - B: джинсы s.oliver
- label=1, sim=1.000, ID 111487 ↔ 584115575903; `['на любой сезон']` → `['лето']`; other diffs=['product_color']
  - A: футболка sela
  - B: футболка sela

#### `russian_size` — unsafe_variant; near +/− = 17/345

- label=0, sim=1.000, ID 50081 ↔ 420906879596; `['42']` → `['48']`; other diffs=['product_color']
  - A: пуловер u.s. polo assn.
  - B: u.s. polo assn. / пуловер
- label=0, sim=1.000, ID 102924 ↔ 360777380402; `['158', '164']` → `['140']`; other diffs=['product_color']
  - A: smena / жакет
  - B: жакет smena
- label=0, sim=1.000, ID 8590017389 ↔ 180388739958; `['46']` → `['56']`; other diffs=[]
  - A: louren wilton / куртка
  - B: куртка louren wilton

#### `material_composition` — unsafe_variant; near +/− = 7/123

- label=0, sim=1.000, ID 140056 ↔ 738734455248; `['polyester']` → `['cotton']`; other diffs=['product_color']
  - A: брюки zolla
  - B: брюки zolla
- label=0, sim=1.000, ID 8590044580 ↔ 463856485143; `['30%-50%', 'стрейч-коттон']` → `['cotton']`; other diffs=['russian_size']
  - A: бюстгальтер triumph
  - B: triumph / бюстгальтер
- label=0, sim=1.000, ID 17179932120 ↔ 386547164503; `['cotton', 'трикотаж']` → `['wool']`; other diffs=[]
  - A: шапка
  - B: шапка

#### `collection` — unsafe_variant; near +/− = 5/6

- label=0, sim=1.000, ID 17179961360 ↔ 670015010382; `['осень-зима 2023']` → `['осень-зима 2023 2024']`; other diffs=['product_color']
  - A: рубашка
  - B: рубашка
- label=1, sim=1.000, ID 42949733162 ↔ 755914340632; `['осень-зима 2023 2024']` → `['осень-зима 2023']`; other diffs=['season']
  - A: vittoria vicci джемпер
  - B: джемпер vittoria vicci
- label=1, sim=1.000, ID 103079362569 ↔ 266288093493; `['весна-лето 2024']` → `['осень-зима 2023 2024']`; other diffs=['season']
  - A: платье via appia
  - B: via appia платье

#### `lining_material` — unsafe_variant; near +/− = 0/6

- label=0, sim=1.000, ID 60129632238 ↔ 704374759718; `['без подкладки']` → `['wool']`; other diffs=['product_color']
  - A: crockid / шапка
  - B: шапка crockid
- label=0, sim=1.000, ID 429496769510 ↔ 77309526029; `['без подклада']` → `['без подкладки']`; other diffs=['russian_size']
  - A: платье playtoday
  - B: playtoday / платье
- label=0, sim=1.000, ID 532575954199 ↔ 661425011309; `['polyester']` → `['полиамид']`; other diffs=['material_composition']
  - A: nothing but love / слитный купальник
  - B: купальник слитный nothing but love

### Автотовары

#### `article` — unsafe_variant; near +/− = 11/170

- label=0, sim=0.864, ID 70644 ↔ 240518213890; `['47389']` → `['17790', 'febi17790']`; other diffs=[]
  - A: сайлентблок febi 47389
  - B: сайлентблок febi 17790
- label=0, sim=0.851, ID 89216 ↔ 652835169978; `['77a1165']` → `['77a1154', 'masterkit77a1154']`; other diffs=[]
  - A: ремкомплект тормозного суппорта masterkit 77a1165
  - B: ремкомплект суппорта masterkit 77a1154
- label=0, sim=0.877, ID 90996 ↔ 601295448480; `['pe25069']` → `['pe25006']`; other diffs=[]
  - A: patron датчик парктроника patron pe25069
  - B: датчик парктроника patron pe25006

#### `product_color` — unsafe_variant; near +/− = 8/86

- label=1, sim=0.867, ID 40607 ↔ 446676685402; `['black']` → `['черно-белая']`; other diffs=['item_type']
  - A: оплетка на руль mercedes benz s-classe w 220 (мерседес бенц с-класс 220), м(37-39см), черно-белая
  - B: оплетка на руль мерседес бенц с-класс 221 (mercedes benz s-classe w 221), м(37-39см), черно-белая
- label=1, sim=0.850, ID 149801 ↔ 300647831652; `['black']` → `['шоколадный']`; other diffs=[]
  - A: автопилот накидка защитная на спинку автомобильного сиденья (органайзер) люкс на ниссан мурано (2007 - 2010) внедорожник 5 дверей / nissan murano, экокожа, черный с темно-бежевой …
  - B: накидка защитная на спинку автомобильного сиденья (органайзер) люкс на ниссан мурано (2002 - 2008) внедорожник 5 дверей / nissan murano, экокожа, шоколадный
- label=1, sim=1.000, ID 8590072142 ↔ 747324331924; `['black', 'grey']` → `['blue']`; other diffs=['seasonality']
  - A: перчатки acerbis mx x-h blue/grey xxl
  - B: acerbis перчатки mx x-h blue/grey xxl

#### `completeness` — unsafe_variant; near +/− = 20/74

- label=0, sim=0.909, ID 107406 ↔ 25769861024; `['подголовник']` → `['комплект чехлов']`; other diffs=[]
  - A: чехлы на сиденья киа соул 1 (kia soul 1) с 2009-2013г. am
  - B: / чехлы на сиденья kia soul 1 (киа соул 1) с 2009-2013г.
- label=1, sim=0.918, ID 119415 ↔ 249108171559; `['в состав набора входят флаконы с кисточкой объемом 20мл краска для сколов лак…` → `['грунт краска лак малярный скотч наждачная бумага обезжириватель перчатки поли…`; other diffs=[]
  - A: ремкомплект для ремонта глубоких сколов и царапин color1 для renault, цвет 656 - gris
  - B: ремкомплект для ремонта сколов и царапин color1 для renault, цвет 630 - gris
- label=0, sim=1.000, ID 8589934610 ↔ 25769930943; `['подголовник']` → `['чехлы на переднее сиденье чехол на заднее сиденье задняя спинка подголовники']`; other diffs=[]
  - A: чехлы на сиденья форд рейнджер т6 (ford ranger t6) с 2013-н.в.
  - B: / чехлы на сиденья форд рейнджер т6 ford ranger t6 с 2013-н.в.

#### `item_type` — unsafe_variant; near +/− = 13/79

- label=1, sim=0.867, ID 40607 ↔ 446676685402; `['оплетка на руль']` → `['оплетка чехол']`; other diffs=['product_color']
  - A: оплетка на руль mercedes benz s-classe w 220 (мерседес бенц с-класс 220), м(37-39см), черно-белая
  - B: оплетка на руль мерседес бенц с-класс 221 (mercedes benz s-classe w 221), м(37-39см), черно-белая
- label=0, sim=0.955, ID 17179875278 ↔ 592705526863; `['сальник двигателя']` → `['сальник']`; other diffs=['article']
  - A: сальник kraft kt885202
  - B: сальник kraft kt885210
- label=1, sim=0.889, ID 25769957953 ↔ 687194923957; `['краска автомобильная']` → `['автоэмаль']`; other diffs=['volume']
  - A: аэрозольная краска color1 для nissan, цвет kv6 - silver
  - B: аэрозольная краска color1 для nissan, цвет kas - purplish silver

#### `volume` — unsafe_variant; near +/− = 6/36

- label=0, sim=0.871, ID 79878 ↔ 661424982278; `['520ml']` → `['520']`; other diffs=[]
  - A: автомобильная краска fixpaint spray smart fortwo, код цвета ek0, название sapphire blue, в аэрозольном баллончике 520 мл
  - B: автомобильная краска fixpaint spray suzuki sx4(3), код цвета zq4, название sphere blue, в аэрозольном баллончике 520 мл
- label=0, sim=0.858, ID 25769828574 ↔ 833223764899; `['520ml']` → `['520']`; other diffs=[]
  - A: автомобильная краска fixpaint spray mercedes-benz, код цвета 136u, название gray white, в аэрозольном баллончике 520 мл
  - B: автомобильная краска fixpaint spray mercedes-benz sprinter(910), код цвета 755, название tenorite grey, в аэрозольном баллончике 520 мл
- label=1, sim=0.889, ID 25769957953 ↔ 687194923957; `['520']` → `['520ml']`; other diffs=['item_type']
  - A: аэрозольная краска color1 для nissan, цвет kv6 - silver
  - B: аэрозольная краска color1 для nissan, цвет kas - purplish silver

#### `general_material` — unsafe_variant; near +/− = 6/15

- label=0, sim=0.909, ID 85899354967 ↔ 292057848144; `['faux_leather']` → `['textile']`; other diffs=['completeness']
  - A: / чехлы на сиденья skoda fabia 1 (шкода фабиа 1) с 2001-2007г.
  - B: чехлы на сиденья шкода фабиа 1 (skoda fabia 1) с 2001-2007г. mk1
- label=1, sim=0.836, ID 128849078508 ↔ 214748468583; `['виниловая пленка']` → `['винил']`; other diffs=['item_type']
  - A: вилвинил 2 наклейки "гиря" 14x20см
  - B: 2 наклейки "гиря" 14x20см
- label=0, sim=0.821, ID 154618930703 ↔ 498216283498; `['abs пластик']` → `['rubber']`; other diffs=['item_type']
  - A: багажник на рейлинги volvo 240-260 wagon универсал 1975-1993 - 120см квадратные стальные дуги, интер титан, без замка
  - B: inter багажник на рейлинги ford focus i универсал 1999-2005 - 120см квадратные стальные дуги, интер титан, без замка

## Name ↔ attributes

Доля ниже — среди товаров, где structured attribute известен. Это эвристическое обнаружение нормализованного значения в `name`, поэтому небольшие числа не доказывают отсутствия связи.

| Категория | Semantic attribute | Item support | Найдено в name | Доля | Примеры representations |
|---|---|---:|---:|---:|---|
| Автотовары | `tire_width` | 2,129 | 2,105 | 98.87% | 225: 225; 215: 215; 235: 235 |
| Обувь | `brand` | 17,546 | 17,171 | 97.86% | nike: nike; tendance: tendance; adidas: adidas |
| Одежда | `item_type` | 22,765 | 22,078 | 96.98% | футболка: футболка; бюстгальтер: бюстгальтер; трусы: трусы |
| Автотовары | `tire_profile` | 2,174 | 1,994 | 91.72% | 55: 55; 65: 65; 45: 45 |
| Одежда | `brand` | 19,281 | 17,246 | 89.45% | koton: koton; zara: zara; ava: ava |
| Ювелирные изделия | `brand` | 16,128 | 14,089 | 87.36% | sokolov: sokolov; diamant: diamant; lazurit-online: lazurit-online |
| Обувь | `product_model` | 3,104 | 2,487 | 80.12% | лоферы: лоферы; челси: челси; лодочки: лодочки |
| Автотовары | `compatibility` | 271 | 186 | 68.63% | skoda rapid i ii 40 60 c 14г. vw polo с 20г.: skoda rapid i ii 40 60 c 14г. vw polo с 20г.; lada largus 5 мест 40 60 2012-2021: lada largus 5 мест 40 60 2012-2… |
| Ювелирные изделия | `item_type` | 20,668 | 14,032 | 67.89% | кольцо: кольцо; серьги ювелирные: серьги ювелирные; серьги: серьги |
| Автотовары | `brand` | 18,653 | 12,410 | 66.53% | patron: patron; lynxauto: lynxauto; kiwix: kiwix |
| Ювелирные изделия | `metal_fineness` | 25,500 | 16,834 | 66.02% | 585: 585, 375 585; 925: 925, 925 проба, 585 925; 375: 375 |
| Одежда | `generic_size` | 7,068 | 4,316 | 61.06% | l: l; m: m; s: s |
| Автотовары | `car_make` | 5,867 | 3,497 | 59.60% | toyota: toyota; nissan: nissan; ford: ford |
| Автотовары | `volume` | 2,345 | 1,354 | 57.74% | 20: 20; 40: 40; 520: 520 |
| Обувь | `generic_size` | 6,514 | 3,537 | 54.30% | 39: 39; 41: 41; 37: 37 |
| Ювелирные изделия | `product_model` | 9,400 | 4,894 | 52.06% | обручальное: обручальное; помолвочное: помолвочное; пусеты: пусеты |
| Ювелирные изделия | `ring_size` | 5,246 | 2,573 | 49.05% | 17.5: 17.5; 17: 17; 16.5: 16.5 |
| Автотовары | `car_model` | 4,390 | 2,066 | 47.06% | toyota: toyota; toyota land cruiser prado: toyota land cruiser prado; corolla: corolla |
| Автотовары | `product_color` | 8,810 | 4,127 | 46.84% | black: черный, черный с красной строчкой, черный с серой строчкой; grey: серый; beige: бежевый |
| Ювелирные изделия | `main_insert` | 21,773 | 10,123 | 46.49% | фианит: фианит; бриллиант: бриллиант; топаз: топаз |
| Автотовары | `item_type` | 19,505 | 8,401 | 43.07% | оплетка на руль: оплетка на руль; колодки тормозные: колодки тормозные; сайлентблок: сайлентблок |
| Автотовары | `product_model` | 2,663 | 1,075 | 40.37% | cinturato p7: cinturato p7; hakkapeliitta 9 suv: hakkapeliitta 9 suv; winter i cept iz2 w616: winter i cept iz2 w616 |
| Автотовары | `article` | 20,213 | 8,104 | 40.09% | 6pk1873: 6pk1873; cw0229: cw0229; pgs3060br: pgs3060br |
| Автотовары | `general_material` | 6,360 | 2,390 | 37.58% | faux_leather: экокожа, искусственная кожа; алькантара: алькантара; leather: натуральная кожа |
| Ювелирные изделия | `jewelry_composition` | 11,851 | 4,416 | 37.26% | silver: серебро, серебро родированное, серебро позолоченное; gold: золото красное, золото желтое, золото белое; кольцо ювелирное: ювелирное кольцо |
| Ювелирные изделия | `jewelry_material` | 13,976 | 5,205 | 37.24% | gold: золото, золото красное, золото желтое; silver: серебро, серебро родированное, серебро позолоченное; steel: сталь |
| Автотовары | `car_year` | 145 | 43 | 29.66% | 2001: 2001; 2006: 2006; 2015: 2015, 2015-2022 |
| Автотовары | `diameter` | 3,229 | 931 | 28.83% | 39: 39; 37.39: 37.39; 17: r17 |
| Ювелирные изделия | `item_size` | 12,157 | 3,319 | 27.30% | 17: 17, 17 5; 18: 18, 18 5; 19: 19 |
| Автотовары | `package_count` | 5,723 | 1,519 | 26.54% | 1: 1; 2: 2; 5: 5 |
| Автотовары | `placement` | 1,836 | 453 | 24.67% | передние: передние; багажник: багажник; рейлинг: рейлинг |
| Одежда | `product_model` | 7,866 | 1,772 | 22.53% | слипы: слипы; боксеры: боксеры; стринги: стринги |
| Обувь | `article` | 3,586 | 770 | 21.47% | 9546: 9546-46m; ysm2056a103: ys-m2056a-103; tmh3019f1: tmh3019f-1 |
| Одежда | `package_count` | 7,063 | 1,329 | 18.82% | 1: 1; 3: 3; 2: 2 |
| Одежда | `completeness` | 9,327 | 1,535 | 16.46% | футболка: футболка; куртка: куртка; бюстгальтер: бюстгальтер |
| Обувь | `completeness` | 4,631 | 762 | 16.45% | кроссовки: кроссовки; ботинки: ботинки; туфли: туфли |
| Автотовары | `power` | 32 | 5 | 15.62% | 110: 110; 5: 5; 1000: 1000 |
| Обувь | `product_color` | 27,902 | 4,196 | 15.04% | black: черный, черные, black; white: белый, белые; beige: бежевый, бежевые |
| Обувь | `manufacturer_size` | 12,013 | 1,791 | 14.91% | 40eu: 40 eu; 38eu: 38 eu; 41eu: 41 eu |
| Ювелирные изделия | `article` | 17,329 | 2,427 | 14.01% | 01291200000111011: 01-2912-00-000-1110-11; 100559: 100559; 3922: 01-3922-00-000-1120-11 |

## Неожиданные и рискованные результаты

- **Автотовары / `width`:** различается в 100.00% human-positive (9/9); статус `uncertain`.
- **Автотовары / `length`:** различается в 100.00% human-positive (12/12); статус `uncertain`.
- **Обувь / `manufacturer_size`:** различается в 96.22% human-positive (229/238); статус `unsafe_variant`.
- **Одежда / `completeness`:** различается в 95.56% human-positive (86/90); статус `unsafe_variant`.
- **Автотовары / `placement`:** различается в 94.44% human-positive (17/18); статус `uncertain`.
- **Автотовары / `volume`:** различается в 87.14% human-positive (61/70); статус `unsafe_variant`.
- **Ювелирные изделия / `average_weight`:** различается в 87.10% human-positive (952/1093); статус `unsafe_variant`.
- **Обувь / `insole_length`:** различается в 86.84% human-positive (33/38); статус `unsafe_variant`.
- **Автотовары / `car_model`:** различается в 84.62% human-positive (22/26); статус `unsafe_variant`.
- **Одежда / `manufacturer_size`:** различается в 84.29% human-positive (161/191); статус `unsafe_variant`.
- **Обувь / `russian_size`:** различается в 83.36% human-positive (581/697); статус `unsafe_variant`.
- **Обувь / `foot_length`:** различается в 83.33% human-positive (15/18); статус `uncertain`.
- **Автотовары / `package_count`:** различается в 82.14% human-positive (23/28); статус `unsafe_variant`.
- **Обувь / `collection`:** различается в 81.63% human-positive (40/49); статус `unsafe_variant`.
- **Автотовары / `seasonality`:** различается в 80.45% human-positive (144/179); статус `unsafe_variant`.
- **Одежда / `collection`:** различается в 79.17% human-positive (19/24); статус `unsafe_variant`.
- **Одежда / `season`:** различается в 78.70% human-positive (591/751); статус `unsafe_variant`.
- **Одежда / `russian_size`:** различается в 76.68% human-positive (592/772); статус `unsafe_variant`.
- **Одежда / `product_model`:** различается в 73.75% human-positive (118/160); статус `unsafe_variant`.
- **Ювелирные изделия / `product_model`:** различается в 73.02% human-positive (387/530); статус `unsafe_variant`.

## Ограничения исследования

- Наличие human label не доказывает причинность: даже near-identical фильтр остаётся эвристикой.
- Пропуски и неодинаковое заполнение structured attributes уменьшают support.
- Значения aliases считаются совпадающими при пересечении нормализованных representations; это консервативно, но не является универсальным entity linking.
- `uncertain` нельзя передавать генератору как критическую характеристику без дополнительной проверки.
- Этот отчёт не использует и не оценивает human validation.

Время анализа: **206.1 с**.
