# kaggle_kernels

Kaggle-кернелы проекта Ozon E-CUP 2026 (matching). Каждый кернел — отдельный
скрипт/ноутбук, который гоняется на инфраструктуре Kaggle (там полный
`items.parquet` и бесплатный GPU) и складывает результат либо в свой `output/`,
либо в отдельный Kaggle-датасет.

Кернелы разведки основной (MiniLM) архитектуры приватные, владелец
`maxamsecret` — `https://www.kaggle.com/code/maxamsecret/<slug>`.

Кернелы разведки новых энкодеров (FRIDA/BERTA, см. ниже) частично лежат на
**втором Kaggle-аккаунте `rays0n`** — отдельная недельная GPU-квота, кернелы
там **публичные** — `https://www.kaggle.com/code/rays0n/<slug>`. Креды обоих
аккаунтов — вне репозитория (`maximus/kaggle creds/`); перед запуском новой
GPU-сессии уточняй, с какого аккаунта.

## Как запускать
```bash
kaggle kernels push   -p kaggle_kernels/<name>                 # залить и запустить
kaggle kernels status maxamsecret/<slug>                       # статус
kaggle kernels output maxamsecret/<slug> -p kaggle_kernels/<name>/output   # забрать результат
```
`<slug>` — поле `id` в `kaggle_kernels/<name>/kernel-metadata.json`.

## Что трекается в git
Только `*.py` / `*.ipynb`, `kernel-metadata.json` и `README.md`. Тяжёлые
`output/`, `pulled/`, `__pycache__/`, `*.bin`, `*.safetensors` — в `.gitignore`
(регенерируются перезапуском кернела или лежат в Kaggle-датасетах).

## Кернелы

### Обучение и сабмит текущего решения (LoRA cross-encoder)
| Кернел | Суть | GPU | Итог |
|---|---|---|---|
| [cross-encoder-sft-pipline](cross-encoder-sft-pipline) | базовое LoRA SFT дообучение | 2×T4 | **0.5282** val / 0.265 LB (epoch 7) |
| [merge-lora](merge-lora) | мёрж адаптера в базу (без `peft` на инференсе) | CPU | датасет `ozon-lora-merged-minilm` |
| [validate-baseline-vs-lora](validate-baseline-vs-lora) | честное сравнение на одном val-сплите | 1×T4 | baseline 0.3710 vs LoRA **0.5282** (Δ+0.157) |
| [lora-timing-dryrun](lora-timing-dryrun) | оффлайн-замер времени сабмита | 1×T4 | 950.6с / 384.7 пар/с — в лимит проходит |
| [val-per-category-metrics](val-per-category-metrics) | разбивка метрики по 20 категориям | CPU | `RUNNING` (pending) |

### Вторая стадия — continued SFT на LLM-метках
| Кернел | Суть | GPU | Итог |
|---|---|---|---|
| [lora-llm-continue-train](lora-llm-continue-train) | дообучение на 11.19M мягких меток + human-mix | 2×T4 | новый best **0.53229** (упёрлись в 12ч лимит) |
| [publish-token-cache](publish-token-cache) | публикация ~9GB токен-кэша как датасета | CPU | датасет `ozon-lora-llm-token-cache` (−9.5ч токенизации) |

### Анализ и исследование новой архитектуры
| Кернел | Суть | GPU | Итог |
|---|---|---|---|
| [item-length-analysis](item-length-analysis) | распределение длин полей / доля обрезки | CPU | `attributes` режется у **47%** товаров |
| [giga3b-quantize-benchmark](giga3b-quantize-benchmark) | проба bi-encoder Giga-3B (корректность/качество/скорость/int8) | T4/P100 | `COMPLETE` — результат ещё не забран в репо |

### Разведка новых энкодеров — FRIDA / BERTA (LoRA SFT)
Замена MiniLM (33M) на что-то более мощное. FRIDA — T5-энкодер 823M;
BERTA — дистиллят FRIDA в BERT-форме, 128.3M. Оба прогона упёрлись в
12-часовой лимит сессии не сойдясь, но уже сильно обгоняют продакшен MiniLM
(0.5282 val). `frida-lora-sft-epoch2` и `berta-lora-sft-human-only-20-epochs`
— на втором аккаунте (`rays0n`), остальное — на `maxamsecret`.

| Кернел | Суть | GPU | Итог |
|---|---|---|---|
| [frida-lora-sft-human-only-10-epochs](frida-lora-sft-human-only-10-epochs) | FRIDA LoRA SFT, эпоха 1 (human-only) | 1×T4 | **0.652** val / **0.3459 LB** @ 60% эпохи (лимит сессии) → датасет `frida-lora`, **подтверждённый (sha256) источник** сабмита `matching-frida_lora_adapter-submit.zip`, прошедшего тестовый стенд |
| [frida-nan-diagnose](frida-nan-diagnose) | диагностика NaN под fp16/DataParallel | 2×T4 | root cause: autocast не пробрасывается в потоки DataParallel-реплик |
| [frida-lora-sft-epoch2](frida-lora-sft-epoch2) (аккаунт `rays0n`) | продолжение SFT FRIDA, эпоха 2 | 1×T4 | **0.674** val @ 40% эпохи (лимит сессии) — лучше эпохи 1, ещё не сабмичен; чекпоинт опубликован отдельным датасетом `frida-lora-sft-epoch2-40` (версией `frida-lora` стать не смог, см. его README) |
| [berta-lora-sft-human-only-20-epochs](berta-lora-sft-human-only-20-epochs) (аккаунт `rays0n`) | BERTA LoRA SFT, план 20 эпох | 1×T4 | **0.7111** val @ 22% плана (лимит сессии) — **лучший результат проекта на сегодня**; чекпоинт опубликован (`berta-lora-sft-5epoch`); правка eval-only-at-epoch-end уже применена в коде (не отдельно "запущена" — следующий прогон резюма получит её автоматически) |
| [frida-lora-sft-epoch3](frida-lora-sft-epoch3) (аккаунт `rays0n`, план) | продолжение epoch2 с фиксом eval (сабсэмпл 25% + 3/эпоху вместо 5) | 1×T4 | **подготовлено, не запущено** — чекпоинт подключён, ждёт добро на GPU-сессию |
| [berta-lora-sft-continue-5epoch](berta-lora-sft-continue-5epoch) (аккаунт `rays0n`, план) | резюм BERTA с 0.7111, ещё 5 эпох (display 5→9 из 20) | 2×T4 | **подготовлено, не запущено** — нашёл и починил баг: `train_state.json` чекпоинта врёт про `completed_epochs=0` (реально `global_step=5016`≈эпоха 4), `start_epoch` теперь считается из `global_step`, не из битого поля |

И FRIDA (`frida-lora-sft-epoch2`), и BERTA упёрлись в 12-часовой лимит
**из-за eval, не обучения**: полный проход по 73,131 val-парам съедал
почти столько же времени, сколько сами шаги обучения между чекпоинтами
(подробный разбор с цифрами — в README `frida-lora-sft-epoch3`). Оба
кернела дальше **не перезапускаются с нуля** — каждый уже сжёг полную
сессию и дал реальный прогресс; продолжение — через новые
кернелы/версии, которые резюмят сохранённое состояние optimizer+adapter.
Оба чекпоинта опубликованы командой как отдельные Kaggle-датасеты
(`maxamsecret/frida-lora-sft-epoch2-40`, `maxamsecret/berta-lora-sft-5epoch`)
— попытка сделать их версиями старых датасетов (`frida-lora`,
`rays0n/berta-lora-sft-checkpoint`) через `kaggle datasets version`
не удалась (`Incompatible Dataset Type`), так и остались отдельными.

Незавершённое в этой ветке: `frida-tokenize-items(-llm/-human)`,
`frida-lora-batch-size-probe`, `giga3b-cross-encoder-classifier-fit` —
вспомогательные/смежные кернелы на `maxamsecret`, ещё не затянуты в
`kaggle_kernels/`.

### Голова поверх кэша + майнинг негативов (в работе)
Идея: кэшировать pooled-выход LoRA-BERTA по всем парам один раз, дальше
экспериментировать с архитектурой головы (MLP вместо `Linear`) без
GPU-энкодера в цикле. Плюс — понять, какие негативы стоит домайнить: не
просто анализ FP, а воспроизведение retrieval-логики организаторов через
recall@k (независимо от target — сам факт присутствия пары в
`matches.parquet` уже свидетельство, что retrieval счёл её кандидатом),
дальше топ-K соседей за вычетом размеченных — hard negatives по
построению.

| Кернел | Суть | GPU | Итог |
|---|---|---|---|
| [berta-lora-pair-cache](berta-lora-pair-cache) | кэш pooled-представлений (768-мерных) по всем 365,654 парам, train/val отдельно | 2×T4 | **отменено пользователем на v4** (после OOM-фикса токенизации, до завершения) |
| [fasttext-category-similarity-probe](fasttext-category-similarity-probe) | отделяет ли дешёвый лексический сигнал (fastText/TF-IDF) позитивы от негативов внутри категории Аптека, без всякой модели | CPU | **готово за 76.8с** — TF-IDF (AP 0.166) обыграл fastText (AP 0.150) и по качеству, и по цене; оба слабые (лифт ×2 над базой) |
| [retrieval-recall-hard-negative-mining](retrieval-recall-hard-negative-mining) | recall@k TF-IDF-ретривала против размеченных пар, все 20 категорий | CPU | **готово** — recall@10 от 0.015 (Ювелирные изделия) до 0.35 (Хобби и творчество) — далеко от цели ~0.9 |
| [rubert-mini-frida-tokenize](rubert-mini-frida-tokenize) | токенизация 711,304 товаров под `rubert-mini-frida`, TPU (ради RAM) | TPU v5e-8 | **готово**, датасет `maxamsecret/rubert-mini-frida-item-tokenized` |
| [rubert-mini-frida-item-embeddings](rubert-mini-frida-item-embeddings) | per-item bi-encoder эмбеддинги (711,304×312) на generic `sergeyzh/rubert-mini-frida` | 2×T4 | **готово**, датасет `maxamsecret/rubert-mini-frida-item-embeddings` |
| [rubert-mini-frida-recall-mining](rubert-mini-frida-recall-mining) | тот же recall@k, но на реальных эмбеддингах вместо TF-IDF | CPU | **готово за 573.7с** — macro recall@1=0.089, recall@10=**0.276** (лучше TF-IDF в 1.5-3× по большинству категорий, но всё ещё далеко от ~0.9); Ювелирные изделия/Обувь/Одежда стабильно самые сложные |
| [berta-lora-item-embeddings](berta-lora-item-embeddings) | то же, но task-tuned `sergeyzh/BERTA`+LoRA (`berta-lora-sft-5epoch`, val 0.7111) вместо generic модели — переиспользует уже готовый item-token-кэш из того же датасета | 2×T4 | **подготовлено, не запущено** |

### Bi-encoder ветка (отдельная git-ветка `bi-encoder`, не здесь)
Параллельная работа тиммейта (`grigoriyyy`) над bi-encoder архитектурой
(Qwen3-Embedding-0.6B + LoRA + MLP-голова) — уже лучший val-результат
проекта (0.726 за 1 сессию), но на другом val-сплите. См. `kaggle_kernels/
qwen3-bi-encoder-lora-sft/README.md` на ветке `bi-encoder`.

## Датасеты, которые производят/потребляют кернелы
- `maxamsecret/ozon-ecup-2026-matching` — исходные данные (вход почти всюду).
- `maxamsecret/ozon-lora-minilm-finetuned` — PEFT-адаптер (выход cross-encoder-sft, вход merge/continue-train).
- `maxamsecret/ozon-lora-merged-minilm` — смёрженные веса (выход merge-lora, вход timing/val-per-category).
- `maxamsecret/ozon-lora-llm-token-cache` — токен-кэш 11.48M (выход publish-token-cache, вход continue-train).
- `maxamsecret/ozon-lora-submission-code` — код сабмита для оффлайн timing-прогона.
- `maxamsecret/frida-item-tokens-human` / `frida-tokenized-items-human` — тот же item-level токен-кэш FRIDA (711,304 товаров), опубликован дважды под разными именами; вход `frida-lora-sft-*`.
- `maxamsecret/frida-lora` — чекпоинт FRIDA LoRA SFT после эпохи 1 (0.652 val); вход `frida-lora-sft-epoch2`.
- `rays0n/berta-lora-sft-checkpoint` — заявлен в коде `berta-lora-sft-human-only-20-epochs` для кросс-сессионного резюма, но ещё не создан (публикация из кернела не сработала — нет Kaggle Secrets).
