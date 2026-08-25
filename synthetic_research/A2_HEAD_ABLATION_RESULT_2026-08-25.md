# A2 head-only ablation: итог эксперимента

Дата: 2026-08-25  
Kaggle kernel: `rvitvitskii/frida-a2-head-ablation`, version 2  
Public sanitized results kernel:
`rvitvitskii/frida-a2-head-ablation-public-validation-results`, version 2

Kernel commit: `b79ba0f`

## Краткий вывод

A2-синтетика дала микроскопический положительный сдвиг относительно
контрольного дообучения, но итоговая A2-модель осталась хуже исходной модели
Frida. В текущей постановке улучшение эффективности метода не подтверждено.

Не следует загружать A2-архив как ожидаемо более сильное решение. Архив
сохранён как воспроизводимый результат эксперимента.

## Постановка

- 195 отобранных A2 negative pairs;
- точный submitted Frida encoder заморожен;
- обучается только classifier head;
- canonical human-train replay: 10 000 пар;
- A2 и control используют одинаковые seed, порядок, число шагов и веса;
- control повторяет исходные human-negative пары;
- A2 заменяет их соответствующими synthetic pairs;
- A2 sample weight: 0.5;
- repeat: 10;
- validation: untouched canonical 109 648 строк, только для оценки.

## Метрики

| Вариант | Macro AP | Micro AP | Изменение Macro | Изменение Micro |
|---|---:|---:|---:|---:|
| Reference baseline | 0.758880882 | 0.813281913 | — | — |
| Same-GPU baseline | 0.756729236 | 0.811112891 | — | — |
| Control | 0.755536411 | 0.810392583 | -0.001192825 к same-GPU baseline | -0.000720308 |
| A2 | 0.755558897 | 0.810420403 | -0.001170340 к same-GPU baseline | -0.000692488 |
| A2 − Control | — | — | +0.000022486 | +0.000027820 |

Основное сравнение — A2 против control внутри одного запуска. Оно устраняет
разницу окружения между старым reference baseline и текущим GPU runtime.

A2 выше control в 15 из 20 категорий, но изменения очень малы. A2 выше
same-GPU baseline только в 5 из 20 категорий. В четырёх категориях, из которых
пришла синтетика, A2 против control показывает смешанный результат:

- Автотовары: -0.000009140;
- Обувь: -0.000007934;
- Одежда: +0.000046160;
- Ювелирные изделия: +0.000007964.

Это похоже на слабый направленный сигнал, но не на practically meaningful
улучшение модели.

## Проверенная идентичность

- canonical split SHA-256:
  `1e3df234d6cdb8685256d5144b25f9a0a3e3a023f1053a0bd5b16c40e3df8d7e`;
- validation fingerprint:
  `847d04d081432cc081c2149b75271547bf5e25b14f9d8a5804704a5be2a4b3c8`;
- train rows: 256 006;
- validation rows: 109 648;
- validation использовалась для обучения: false;
- exact encoder SHA-256:
  `423c9772de237fb9cc1d797b649dff805b547cb9bdf607ef0616c9ab1c7d0591`;
- starting classifier SHA-256:
  `e661edb385e9ea846e8a8783faec1244e2294f99f56b9e090dd8b7649effb771`;
- A2 classifier SHA-256:
  `5a5df910e103a68c828f129c546018fe7afddb5619ae7d73fb93ede7c47b3b1c`.

## Проверка артефактов

Полный output сохранён локально в
`artifacts/synthetic_v2/frida_a2_head_ablation`.

- baseline/control/A2: по 109 648 predictions в NPY и Parquet;
- NPY и столбец `score` в соответствующем Parquet совпадают точно;
- порядок `id1`, `id2`, `label`, `category` одинаков во всех трёх вариантах;
- все predictions конечные и лежат в допустимом диапазоне;
- сохранены оба classifier checkpoint и оба optimizer state;
- submission ZIP: 1 314 340 585 bytes;
- submission ZIP SHA-256:
  `c48367a953614451972afd1989c2d6f078ccfdb0cb52db262b4db3144509224a`;
- ZIP CRC test: ok;
- внутри ZIP сохранён exact submitted encoder;
- classifier внутри ZIP совпадает с A2 checkpoint;
- структура архива и аргументы `--items_path`, `--matches_path`,
  `--output_path` соответствуют ODS run-контракту.

## Инцидент version 1

Version 1 завершилась до feature extraction: classifier checkpoint ожидал
input dimension 1536, а kernel создавал head с hardcoded dimension 1024.
Минимальное исправление берёт размерность непосредственно из проверенного
checkpoint. Исправление сохранено в commit `b79ba0f`; version 2 завершилась
успешно.

## Решение по A2 v1

Текущий набор из 195 A2-пар и head-only режим не дают полезного улучшения
validation score. Для следующего эксперимента нужно менять хотя бы один из
факторов: объём синтетики, отбор пар, силу/вес synthetic replay либо режим
обучения encoder. Повторять этот же запуск без изменения постановки смысла нет.
