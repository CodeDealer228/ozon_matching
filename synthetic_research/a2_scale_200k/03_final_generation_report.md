# A2 v4 all-category generation — итог квотного прохода

Дата завершения: 2026-08-25

Ветка: `synthetic`

## Итог

Генерация штатно остановилась с `stop_reason =
all_registered_model_quotas_exhausted`: исчерпаны бесплатные allocation-квоты
всех разрешённых моделей из базового списка и двух отдельно разрешённых 35B.
Платный режим не включался.

Цель 200 000 не достигнута. Сохранено **12 833 принятых A2-кандидата** из
**7 462 уникальных human-train source**; не более двух вариантов на source.
Это 6,42% от исходной цели, но данные присутствуют во всех 20 категориях.

## Принятые строки по категориям

| Категория | Accepted |
|---|---:|
| Автотовары | 676 |
| Аптека | 658 |
| Бытовая техника | 807 |
| Бытовая химия | 538 |
| Галантерея и аксессуары | 645 |
| Детские товары | 541 |
| Дом и сад | 648 |
| Канцелярские товары | 468 |
| Красота и гигиена | 622 |
| Мебель | 575 |
| Музыкальные инструменты | 682 |
| Обувь | 698 |
| Одежда | 777 |
| Продукты питания | 635 |
| Спорт и отдых | 508 |
| Строительство и ремонт | 512 |
| Товары для животных | 654 |
| Хобби и творчество | 586 |
| Электроника | 552 |
| Ювелирные изделия | 1 051 |
| **Всего** | **12 833** |

Минимум на категорию — 468, максимум — 1 051. Scheduler балансировал число
реальных попыток; различия accepted связаны с разной долей `skip` и отказов
валидатора.

## Вклад моделей

| Модель | Accepted |
|---|---:|
| `qwen3.6-27b` | 942 |
| `qwen3.5-27b` | 1 209 |
| `qwen3-32b` | 1 668 |
| `qwen3-30b-a3b-instruct-2507` | 763 |
| `qwen3-30b-a3b` | 3 617 |
| `qwen3-14b` | 1 702 |
| `qwen3-8b` | 1 090 |
| `qwen3.6-35b-a3b` | 959 |
| `qwen3.5-35b-a3b` | 883 |

`qwen3.8-27b` сразу вернула allocation-quota error и не дала строк.
`qwen2-7b-instruct` оказалась недоступна аккаунту (`404 model_not_found`).
Её 19 870 pre-inference 404 сохранены в raw audit и не считаются полезными
попытками; соответствующие source затем безопасно обработаны доступными 35B.
Пять запросов при остановке старого процесса остались uncertain и не
повторялись. Ещё одна строка получила terminal `data_inspection_failed` и также
не повторялась.

## Запросы и токены

| Показатель | Значение |
|---|---:|
| Raw terminal records | 37 055 |
| Complete responses | 17 104 |
| Parse errors | 80 |
| Provider errors | 19 871 |
| Prompt tokens | 11 514 802 |
| Completion tokens | 1 434 547 |
| Reasoning tokens | 0 |

## Identity и отсутствие утечки

- Canonical split SHA-256:
  `1e3df234d6cdb8685256d5144b25f9a0a3e3a023f1053a0bd5b16c40e3df8d7e`.
- Validation fingerprint:
  `847d04d081432cc081c2149b75271547bf5e25b14f9d8a5804704a5be2a4b3c8`.
- Source pool SHA-256:
  `61b83e5a50fc96a539430d12571e1de8d3fece4218a8b8f8e920388beecb8a9f`.
- Train/validation item overlap: 0.
- Train/validation unordered-pair overlap: 0.
- Validation content written to provider payload: false.
- IDs sent to provider: false.
- Provider видел только category, name A, name B и attributes B.
- Extension registry SHA-256:
  `e01f88163e62d7be29e29ba25573b0179f400bbfc4b434e66411cd86406c82be`.
- Разрешённый extension содержит только `qwen3.6-35b-a3b` и
  `qwen3.5-35b-a3b`; 80B не использовалась.

## Проверенные артефакты

| Файл | Размер, байт | SHA-256 |
|---|---:|---|
| `accepted_a2.parquet` | 3 185 966 | `c4c0242fd1ae273d3f13c7358d639a503f9b4e2d4efed200d1d6e05d8521ba00` |
| `raw_responses.jsonl` | 50 335 205 | `92cd14c95ce15cc4f25362a9419819492fa625d56ebaab5d0efdbe57a9b045d5` |
| `request_journal.jsonl` | 14 620 562 | `d3dae58c9ecdd3faacc9949b406c473c3feea80a19a23745d888bc37e2b7302b` |
| `model_events.jsonl` | 1 638 | `a5d90937339f32cd224e920e6ea117ad52b3031036dcb4b0837adaa928b76610` |
| `generation_manifest.json` | 3 048 | `d57d596b7468e186a49c1eba1f8abc305e94c7c23b985c8103e419ee9919edb9` |
| `generation_config.json` | 1 397 | `8a1f7dc568b81d2376720b59236a84bbf652ed1af997e6df1cbc4814455802d7` |
| `model_registry_extensions.json` | 232 | `86970ec3ffc240ebf8d525df3bf7649e927ef6628daacbc544e0a92b1aa835ad` |

SHA полей `accepted`, `raw`, `journal` и `model_events` в manifest совпали с
повторно вычисленными значениями. Parquet прочитан полностью: 12 833 строки,
20 категорий, максимум два кандидата на source.

## Что дальше

Это завершённый free-quota generation pass, а не готовое доказательство роста
валидационной метрики. Следующий независимый шаг — собрать из этих строк
воспроизводимый training set, сделать контрольное обучение против human-only
baseline и оценить на неизменённой canonical validation. Для новой генерации
понадобится либо восстановление квот, либо отдельное разрешение на другой класс
моделей/платный режим.
