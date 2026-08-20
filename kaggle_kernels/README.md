# kaggle_kernels

Kaggle-кернелы проекта Ozon E-CUP 2026 (matching). Каждый кернел — отдельный
скрипт/ноутбук, который гоняется на инфраструктуре Kaggle (там полный
`items.parquet` и бесплатный GPU/TPU) и складывает результат либо в свой
`output/`, либо в отдельный Kaggle-датасет.

**Три Kaggle-аккаунта, три раздельные недельные GPU/TPU-квоты** — не путать,
с какого аккаунта что запущено:
- `maxamsecret` — аккаунт по умолчанию, владеет исходным датасетом
  (`ozon-ecup-2026-matching`) и большей частью FRIDA/Giga3B-кернелов.
- `rays0n` — второй аккаунт, отдельная квота под BERTA-эксперименты
  (`https://www.kaggle.com/code/rays0n/<slug>`).
- `maximmarkovbb` — третий аккаунт, добавлен под FRIDA v2 (чистая, ещё не
  тронутая квота).

Креды всех трёх — вне репозитория; перед запуском новой GPU/TPU-сессии
уточняй, с какого аккаунта. Подробности переключения — CLAUDE.md →
"Environment notes".

## Как запускать
```bash
kaggle kernels push   -p kaggle_kernels/<name>                 # залить и запустить
kaggle kernels status <owner>/<slug>                            # статус
kaggle kernels output <owner>/<slug> -p kaggle_kernels/<name>/output   # забрать результат
```
`<owner>/<slug>` — поле `id` в `kaggle_kernels/<name>/kernel-metadata.json`.

## Что трекается в git
Только `*.py` / `*.ipynb`, `kernel-metadata.json` и `README.md`. Тяжёлые
`output/`, `pulled/`, `__pycache__/`, `*.bin`, `*.safetensors` — в `.gitignore`
(регенерируются перезапуском кернела или лежат в Kaggle-датасетах).

---

## MiniLM cross-encoder (исходный baseline)

| Кернел | Суть | GPU | Итог |
|---|---|---|---|
| [cross-encoder-sft-pipline](cross-encoder-sft-pipline) | базовое LoRA SFT дообучение | 2×T4 | **0.5282** val / 0.265 LB (epoch 7) |
| [merge-lora](merge-lora) | мёрж адаптера в базу (без `peft` на инференсе) | CPU | датасет `ozon-lora-merged-minilm` |
| [validate-baseline-vs-lora](validate-baseline-vs-lora) | честное сравнение на одном val-сплите | 1×T4 | baseline 0.3710 vs LoRA **0.5282** (Δ+0.157) |
| [lora-timing-dryrun](lora-timing-dryrun) | оффлайн-замер времени сабмита | 1×T4 | 950.6с / 384.7 пар/с — в лимит проходит |
| [val-per-category-metrics](val-per-category-metrics) | разбивка метрики по 20 категориям | CPU | готово, см. `reports/val_per_category_lora.md` |
| [lora-llm-continue-train](lora-llm-continue-train) | дообучение на 11.19M мягких меток + human-mix | 2×T4 | новый best **0.53229** (упёрлись в 12ч лимит) |
| [publish-token-cache](publish-token-cache) | публикация ~9GB токен-кэша как датасета | CPU | датасет `ozon-lora-llm-token-cache` (−9.5ч токенизации) |
| [item-length-analysis](item-length-analysis) | распределение длин полей / доля обрезки | CPU | `attributes` режется у **47%** товаров |

## BERTA (`sergeyzh/BERTA`, 128.3M, дистиллят FRIDA в BERT-форме)

Cross-encoder, LoRA SFT. Первый рабочий чекпоинт превзошёл MiniLM с большим
запасом (0.7111 val на 22% плана), дальше разошёлся на несколько
параллельных экспериментов — hard negatives, LLM continue-train (полный и
weak-category), attribute augmentation, две классификационные головы
(линейная и CatBoost), и наконец полномасштабный TPU-прогон.

| Кернел | Суть | GPU/TPU | Итог |
|---|---|---|---|
| [berta-lora-sft-human-only-20-epochs](berta-lora-sft-human-only-20-epochs) | исходный LoRA SFT, план 20 эпох | 1×T4 (`rays0n`) | **0.7111** val @ 22% плана (лимит сессии) |
| [berta-lora-sft-continue-5epoch](berta-lora-sft-continue-5epoch) | резюм с 0.7111, ещё 5 эпох | 2×T4 (`rays0n`) | **0.7246** val (epoch 7) / 0.7229 на факт. сохранённой epoch 9 |
| [berta-merge-lora](berta-merge-lora) | мёрж лучшего адаптера (`rays0n/berta-llm-continue-train`, 0.7255) в базу | CPU | смёрженные веса, без `peft` на инференсе |
| [berta-tokenize-llm-256](berta-tokenize-llm-256) | item-token-кэш на `matches_llm.parquet` + human, `max_length=256`, prefix проверен реально (а не по метаданным) | TPU | датасеты `llm_scope`/`human_scope` (12.38M / 711K items) |
| [berta-llm-continue-prep](berta-llm-continue-prep) | сэмплирует фикс. подмножество `matches_llm.parquet` под GPU-прогон Experiment A | TPU (prep) | prep-кернел, без обучения |
| [berta-llm-continue-train](berta-llm-continue-train) | continue-SFT на LLM-метках + human-mix | 2×T4 | **0.7255** val — на тот момент новый лучший результат проекта |
| [berta-llm-cache-probe](berta-llm-cache-probe) | дешёвая CPU-проверка: реально ли в LLM-scope кэше зашит prefix (метаданные врали) | CPU | подтвердил расхождение, привёл к перетокенизации |
| [berta-hardneg-balanced](berta-hardneg-balanced) | continue-train + mined hard negatives, пропорционально размеру категории | 2×T4 (`rays0n`) | эксперимент B, сравнивается с uniform-вариантом |
| [berta-hardneg-uniform](berta-hardneg-uniform) | то же, но плоское число негативов на категорию | 2×T4 | эксперимент C |
| [berta-weakcat-llm-prep](berta-weakcat-llm-prep) | фильтрует `matches_llm.parquet` до 10 самых слабых по val-PR-AUC категорий, сэмплирует 1M пар | TPU (prep) | prep-кернел под weakcat-эксперимент |
| [berta-weakcat-llm-train](berta-weakcat-llm-train) | continue-train на weak-category сэмпле + полный human-mix | 2×T4 (`rays0n`) | целевой фикс худших категорий |
| [berta-attr-augment](berta-attr-augment) | continue-train с рандомизацией порядка ключей в JSON `attributes` (shuffle/sorted/original) | 2×T4 (`rays0n`) | 2 эпохи на полном human train |
| [berta-embed-extract](berta-embed-extract) | извлекает pooled pair-эмбеддинги (до Linear-головы) лучшего чекпоинта (0.7255) для train+val | 2×T4 | питает CatBoost-трек ниже |
| [berta-catboost-noleak](berta-catboost-noleak) | CatBoost поверх эмбеддингов, train/val строго внутри канонического val-сплита | CPU | **реальный кандидат на сабмит** — см. `submits/berta_catboost` |
| [berta-catboost-leak](berta-catboost-leak) | тот же CatBoost, но train = human-сплит BERTA (заведомая утечка) — количественная оценка завышения | CPU | контрольный эксперимент, не для сабмита |
| [berta-llm-tpu-sft](berta-llm-tpu-sft) | полномасштабный SFT на **всём** `matches_llm.parquet` (без сэмплирования) + human-mix, `PAIR_MAX_LENGTH=256`, TPU v5e-8 | TPU v5e-8 | реальный прогон дошёл до **0.7087** (step 11211) до `BrokenProcessPool`-краша; краш пофикшен (см. ниже) |

`berta-llm-tpu-sft` — первый в проекте настоящий тренировочный (не только
токенизационный) прогон на TPU через `torch_xla`. Крашился несколько раз по
дороге (несовместимые env-переменные Kaggle, не тот тип исключения при
compile-time OOM, probe недооценивал реальную память) — все фиксы описаны в
докстринге файла. Последний найденный и исправленный баг: все 8
fork-процессов независимо перезагружали и конкатенировали item-кэш (~13GB
каждый) и пересчитывали `matches_llm.parquet` — эксгаустило RAM хоста прямо
на границе избыточной второй эпохи (schedule был рассчитан на одну). Фикс:
весь CPU-only сетап переехал в `main()` **до** `torch_xla.launch()`, так что
все реплики шарят один и тот же прочитанный кусок памяти через
copy-on-write; обучение теперь останавливается ровно после одной эпохи.

| Кернел | Суть | GPU/TPU | Итог |
|---|---|---|---|
| [berta-tpu-human-continue-384](berta-tpu-human-continue-384) | резюм топ-чекпоинта, **только** human-разметка (без `matches_llm.parquet`), `PAIR_MAX_LENGTH=384`, 10 эпох, чекпоинт+валидация каждую эпоху | TPU v5e-8 | все 10 эпох прошли; val **монотонно падал** после эпохи 1 (0.7551→0.7501→...→0.7217) — реальное переобучение на маленьком (292K) train; лучший чекпоинт — эпоха 1 |
| [berta-top-embed-extract](berta-top-embed-extract) | эмбеддинг + скор классификатора топ-чекпоинта на val-сплите (73,131 пар, без лика) | 1×T4 | датасет-выход для catboost-трека ниже |
| [tfidf-category-index-fit](tfidf-category-index-fit) | по TF-IDF-индексу на категорию, fit на полном каталоге, хранятся только маленькие fitted-векторайзеры (не per-item векторы) | CPU | ~20 файлов по 5-15МБ, датасет-выход |
| [berta-catboost-top-final](berta-catboost-top-final) | CatBoost на [768 сырых эмбеддингов + classifier_score + tfidf_score + category] | 2×GPU | val **0.7467** (лучше классификатора!), но тест **0.457** — хуже классификатора (0.47); `get_feature_importance()`: 96.11% на эмбеддингах, classifier_score — 1.9%; переобучение на 58,504 строках / 771 фиче |
| `submits/berta_catboost_pca32` (локальный переобуч, кернела нет) | то же, но PCA(32) вместо сырых эмбеддингов + сильнее регуляризация (`depth` 8→6, `l2_leaf_reg` 15→30) | CPU (локально) | val 0.7419, тест **0.469** — почти закрыл разрыв, но всё ещё чуть ниже классификатора; importance classifier_score выросла до 19.3% |
| [berta-embed-cache-all-human](berta-embed-cache-all-human) | кэш эмбеддинга+скора на **всех** 365,654 human-парах (train+val) чекпоинтом `berta-tpu-human-continue-384` — готовит данные под будущую многослойную голову (~5× больше данных без лика, чем нынешний val-срез) | 1-2×T4 (auto-probe batch) | **подготовлено, НЕ запущено** — ждём подтверждения качества `berta-tpu-human-continue-384` на тесте |

## FRIDA (`ai-forever/FRIDA`, 823M, T5-энкодер)

Bi-encoder — item-level токенизация вместо per-pair (нет joint
pair-truncation, которую нужно было бы воспроизводить у cross-encoder'а).

| Кернел | Суть | GPU | Итог |
|---|---|---|---|
| [frida-tokenize-items](frida-tokenize-items) | исходная item-токенизация (human + llm scope), prefix `"paraphrase: "` | CPU | ранняя версия, superseded ниже |
| [frida-tokenize-items-human](frida-tokenize-items-human) | item-кэш, human-scope (711,304 items), `max_length=256` | CPU | датасет `frida-tokenized-items-human` |
| [frida-tokenize-items-llm](frida-tokenize-items-llm) | item-кэш, llm-scope (12,384,610 items) | TPU (ради RAM + отдельная квота) | датасет `frida-tokenized-items-llm` |
| [frida-lora-sft-human-only-10-epochs](frida-lora-sft-human-only-10-epochs) | FRIDA LoRA SFT, эпоха 1 (human-only) | 1×T4 | **0.652** val @ 60% эпохи → источник сабмита `frida_lora_adapter` |
| [frida-batch-probe](frida-batch-probe) | быстрый preflight: макс. micro-batch без реальных данных/обучения | 1×T4 | preflight, минуты вместо часов |
| [frida-nan-diagnose](frida-nan-diagnose) | точечная диагностика NaN-лосса под fp16/DataParallel | 2×T4 | root cause: autocast не пробрасывается в потоки DataParallel-реплик |
| [frida-lora-sft-epoch2](frida-lora-sft-epoch2) | продолжение SFT, эпоха 2 | 1×T4 (`rays0n`) | **0.674** val @ 40% эпохи (лимит сессии) |
| [frida-lora-sft-epoch3](frida-lora-sft-epoch3) | продолжение epoch2 с фиксом eval (сабсэмпл 25%, реже чекпоинты) | 1×T4 (`rays0n`) | подготовлено |
| [frida-lora-continue-2ep](frida-lora-continue-2ep) | дообучение на 2 полные эпохи, best-only чекпоинты с optimizer-state | 1×T4 | дошло до **0.69377** (25% прогресса) до отмены; чекпоинт опубликован как `maximmarkovbb/frida-lora-checkpoint-v1-best` |
| [frida-tokenize-items-v2](frida-tokenize-items-v2) | новая политика упорядочивания `attributes` (демоушен длинных/https/oem-значений в конец + 5% shuffle), два готовых варианта под zero-overhead аугментацию | CPU | датасет `frida-tokenize-items-v2-cache` |
| [frida-lora-v2-epoch](frida-lora-v2-epoch) | текущая версия обучения: 4 метрики в output, time-aware "тихие" чекпоинты (без eval после 11ч), устойчивость к OOM на каждом шаге, eval только в конце эпохи, липкий chunk_size для eval (фикс повторяющегося `eval OOM`-варнинга), новая attribute-policy без оверхеда | 1×T4 (`maximmarkovbb`) | отменено на 10.5ч, step 1127/1143 (98.6% эпохи). Единственный чекпоинт — "тихий" save **без eval**, лог кернела пустой (0 байт, потерян при отмене) → реальное качество неизвестно. Опубликован как `maximmarkovbb/frida-lora-v2-epoch-last` с честной `PROVENANCE.md` |
| [frida-validate-checkpoint](frida-validate-checkpoint) | автономный валидатор любого FRIDA LoRA-чекпоинта: macro PR-AUC на канонической val (73,131 пара), адаптер ищется в `/kaggle/input` по содержимому (`adapter_config.json`), а не по имени датасета. **Считает один и тот же чекпоинт на каждом примонтированном item-кэше** (поддерживает обе раскладки: плоскую v1 и v2 с `variant_base/`) — разрыв между строками и есть цена рассинхрона порядка атрибутов train/serve | 1×T4 (`maximmarkovbb`) | прогон 1 — против `v1-best` (само-проверка валидатора: должен воспроизвести 0.69377). Прогон 2 — против `frida-lora-v2-epoch-last` на обоих кэшах сразу |
| [frida-tpu-human-sft](frida-tpu-human-sft) | FRIDA на TPU v5e-8, human-only, **`PAIR_MAX_LENGTH` 128 → 256** (единственное реальное изменение — 128 обрезал ~94% пар), 4 эпохи, чекпоинты **каждый час** по wall-clock, eval в try/except который не может уронить обучение | TPU v5e-8 (`maxamsecret`) | v1 упал на 47-й секунде (`RobertaProcessing.__new__() got an unexpected keyword argument 'cls'` — рассинхрон transformers/tokenizers на TPU-образе). Фикс: не токенизировать в скрипте вообще, брать готовый кэш `frida-tokenize-items-v2-cache` (256 токенов), `pad_id` выводить из самого кэша. Перезапущено; замеренная скорость ~120 строк/с (полэпохи из 292,523 строк за ~20 мин) — это единственная реальная точка отсчёта throughput'а FRIDA на TPU |

### Пайплайн «многослойная голова + рецепт BERTA» (подготовлено, запускается по цепочке)

Три кернела, которые надо гонять именно в этом порядке — каждый следующий
потребляет датасет предыдущего:

| Кернел | Суть | GPU/TPU | Статус |
|---|---|---|---|
| [frida-tpu-embed-cache-human](frida-tpu-embed-cache-human) | кэш pooled pair-эмбеддингов (CLS, до головы) + логита собственного классификатора чекпоинта по **всем** 365,654 human-парам; `split` **наследует** канонический seed-42 сплит обучения LoRA, в `cache_meta.json` пишется `val_fingerprint`. Шардится по 8 ядрам на непересекающиеся куски одного memmap'а, хвост добивается паддингом до той же статической формы (ни одна строка не теряется), инкрементальный flush + `progress_rank{N}.json` | TPU v5e-8 (`maxamsecret`) | **подготовлено, НЕ запущено** — ждём, какой чекпоинт окажется лучше |
| [frida-head-train](frida-head-train) | обучает `MultiLayerHead` (1536→512→128→1, LayerNorm+GELU+Dropout) поверх кэша. Учится на `split==0`, метрика — на чистом `split==1`, отбор по ней с early stopping. Отдаёт `head.pt` + `val_features.npy` (128-мерные признаки под CatBoost вместо 1536 сырых — ровно та причина, по которой переобучился `berta_catboost_top_final`) | 1×T4 (`maxamsecret`) | подготовлено, минуты работы |
| [frida-llm-tpu-sft](frida-llm-tpu-sft) | **рецепт BERTA, перенесённый на FRIDA**: `matches_llm.parquet` (сэмпл 1.5M) + полный human train вперемешку, `PAIR_MAX_LENGTH=256`, `LR=2e-4`, `WARMUP_RATIO=0.10`, 2 эпохи, многослойная голова с warm-start из предыдущего шага | TPU v5e-8 (`maxamsecret`) | подготовлено |

Почему именно так: единственная конфигурация, давшая лучший тест-скор проекта
(0.47), — это BERTA на `matches_llm`+human вперемешку при `PAIR_MAX_LENGTH=256`.
У FRIDA не было ни того, ни другого (только human, только 128) — и она даёт
0.386 при сопоставимом val. Оба отличия закрываются в `frida-llm-tpu-sft`.
Это осознанно не чистый A/B: меняются сразу LLM-микс, длина пары и голова —
при 5 сабмитах в день и ~8.5ч TPU за прогон дешевле сначала купить всю дельту,
а разбирать вклад по частям потом.

Три вещи, которые в этом пайплайне сделаны намеренно и легко сломать при правке:
- **`optimizer.pt` резюмируемого чекпоинта НЕ загружается** в `frida-llm-tpu-sft`
  — набор параметров изменился (Linear → многослойная голова), моменты Adam
  больше не соответствуют оптимизируемым тензорам. Свежий оптимизатор + 10%
  warmup вместо этого.
- **Оба item-кэша — v1-политики** (`frida-tokenized-items-human` +
  `-llm`). v2-кэш есть только для human-скоупа, а смешивать v2-порядок
  атрибутов у human-пар с v1-порядком у LLM-пар нельзя: порядок атрибутов стал
  бы коррелировать с источником метки. Оба кэша ищутся по `n_items`, не по
  имени папки — иначе `os.walk` мог бы подсунуть не тот.
- **`REQUIRE_PRETRAINED_HEAD=True`** — кернел падает на первых секундах, если
  датасет с головой не примонтирован. Запускать 8.5ч TPU на случайно
  инициализированной голове — ровно тот класс «дорого и впустую», которым полна
  история проекта.

## Giga-Embeddings-instruct (3.45B) — bi-encoder проба, запаркована

Zero-shot cosine similarity уже бьёт обученный baseline, но по FLOPs не
укладывается в лимиты судейского стенда (пары почти не переиспользуют
товары — bi-encoder теряет свой обычный выигрыш). Полный разбор —
`reports/experiment_log.md`.

| Кернел | Суть | GPU | Итог |
|---|---|---|---|
| [giga3b-quantize-benchmark](giga3b-quantize-benchmark) | корректность/качество/скорость + int8-квантизация | T4/P100 | zero-shot 0.4004 val (> обученного baseline), int8 почти бесплатен (0.3973) |
| [giga3b-4bit-validate](giga3b-4bit-validate) | первый реальный GPU-тест переписанного `submits/giga3b_probe` с готовым 4bit-nf4 чекпоинтом и новым batch-size probe | GPU | валидация пайплайна |
| [giga3b-crossencoder-classifier-fit](giga3b-crossencoder-classifier-fit) | дообучает `classifier.joblib` на РЕАЛЬНЫХ int8-эмбеддингах (не синтетических, как раньше — локально нет CUDA под bitsandbytes) | GPU | заменяет synthetic-stub классификатор в сабмите |

## Голова поверх кэша + майнинг негативов

Идея: кэшировать pooled-выход энкодера по всем парам один раз, дальше
экспериментировать с архитектурой головы без GPU-энкодера в цикле. Плюс —
понять, какие негативы стоит домайнить: воспроизведение retrieval-логики
организаторов через recall@k, дальше топ-K соседей за вычетом размеченных —
hard negatives по построению.

| Кернел | Суть | GPU | Итог |
|---|---|---|---|
| [berta-lora-pair-cache](berta-lora-pair-cache) | кэш pooled-представлений (768-мерных) по всем 365,654 парам | 2×T4 | отменено пользователем на v4 (после OOM-фикса токенизации) |
| [fasttext-category-similarity-probe](fasttext-category-similarity-probe) | дешёвый лексический сигнал (fastText/TF-IDF) внутри категории Аптека, без модели | CPU | TF-IDF (AP 0.166) обыграл fastText (AP 0.150), оба слабые |
| [retrieval-recall-hard-negative-mining](retrieval-recall-hard-negative-mining) | recall@k TF-IDF-ретривала против размеченных пар, все 20 категорий | CPU | recall@10 от 0.015 до 0.35 — далеко от цели ~0.9 |
| [rubert-mini-frida-tokenize](rubert-mini-frida-tokenize) | токенизация 711,304 товаров под `rubert-mini-frida`, TPU (ради RAM) | TPU v5e-8 | датасет `rubert-mini-frida-item-tokenized` |
| [rubert-mini-frida-item-embeddings](rubert-mini-frida-item-embeddings) | per-item bi-encoder эмбеддинги на generic `sergeyzh/rubert-mini-frida` | 2×T4 | датасет `rubert-mini-frida-item-embeddings` |
| [rubert-mini-frida-recall-mining](rubert-mini-frida-recall-mining) | тот же recall@k, но на реальных эмбеддингах вместо TF-IDF | CPU | macro recall@10 **0.276** — лучше TF-IDF, всё ещё далеко от цели |
| [berta-lora-item-embeddings](berta-lora-item-embeddings) | то же, но task-tuned BERTA+LoRA (0.7111) вместо generic модели | 2×T4 | подготовлено |
| [negative-pattern-sampling](negative-pattern-sampling) | генерирует по-категорийный набор hard-negative паттернов/пар | CPU | датасет → `negative_patterns/` в репо, 20 категорий |

## Qwen3-Embedding-0.6B bi-encoder — текущий лучший результат

Работа тиммейта (`grigoriyyy`), изначально велась в отдельной ветке
`bi-encoder`, теперь влита в `main`. LoRA + MLP/CatBoost-голова поверх
эмбеддингов Qwen3-Embedding-0.6B. Обучающие кернелы этой ветки в
`kaggle_kernels/` пока не затянуты (код — только в `submits/ecup_qwen_*` и
`submits/matching-qwen-berta-catboost-submit`) — **реальные тест-скоры**
(0.41 лучший, до 0.35 у более ранних вариантов) см. в главном
[README.md](../README.md).

---

## Датасеты, которые производят/потребляют кернелы

- `maxamsecret/ozon-ecup-2026-matching` — исходные данные (вход почти всюду).
- `maxamsecret/ozon-lora-minilm-finetuned` / `ozon-lora-merged-minilm` — PEFT-адаптер / смёрженные веса MiniLM.
- `maxamsecret/ozon-lora-llm-token-cache` — токен-кэш MiniLM continue-train.
- `maxamsecret/berta-tokenized-items-llm` / `-human` — item-token-кэш BERTA, `max_length=256`.
- `rays0n/berta-lora-sft-checkpoint`, `maxamsecret/berta-lora-sft-5epoch` — чекпоинты BERTA LoRA SFT (разные этапы).
- `rays0n/berta-llm-continue-train` — лучший на сегодня BERTA-чекпоинт (0.7255), вход `berta-merge-lora`.
- `maxamsecret/berta-tpu-lora-top-checkpoint` — чекпоинт из упавшего `berta-llm-tpu-sft` (val=0.7087, test=0.47, лучший сабмит проекта) — вход `berta-top-embed-extract`, `berta-tpu-human-continue-384`.
- `maxamsecret/berta-human-continue-384-best-checkpoint` — лучший чекпоинт `berta-tpu-human-continue-384` (эпоха 1, val=0.7551) — вход (пока не запущенного) `berta-embed-cache-all-human`.
- `maxamsecret/frida-tokenized-items-human` / `-llm` — item-token-кэш FRIDA.
- `maximmarkovbb/frida-tokenize-items-v2-cache` — item-token-кэш FRIDA с новой attribute-policy (два варианта: base/shuffled).
- `maxamsecret/frida-lora-sft-epoch2-40` — чекпоинт FRIDA после эпохи 2 (0.67397).
- `maximmarkovbb/frida-lora-checkpoint-v1-best` — чекпоинт FRIDA после отменённого continue-2ep прогона (0.69377, с optimizer-state) — точка резюма для `frida-lora-v2-epoch` и `frida-tpu-human-sft`.
- `maxamsecret/frida-pair-embed-cache-human` — кэш pooled pair-эмбеддингов + classifier-логита FRIDA по всем 365,654 human-парам (выход `frida-tpu-embed-cache-human`, вход `frida-head-train`). Ещё не создан.
- `maxamsecret/frida-multilayer-head` — обученная многослойная голова (выход `frida-head-train`, вход `frida-llm-tpu-sft`). Ещё не создан.
- `maximmarkovbb/frida-lora-v2-epoch-last` — **это ровно те веса, что дали 0.386 на тесте** (sha256 `adapter_model.safetensors` совпадает с сабмитом `submits/Frida Lora V2 Epoch`: `6914cbb181b9c014…`). Чекпоинт FRIDA после `frida-lora-v2-epoch` (98.6% эпохи, с optimizer/scheduler-state). **Не провалидирован**: "тихий" save без eval, `train_state.json: best_macro_pr_auc` — унаследованное 0.69377 с предыдущего чекпоинта, а не метрика этих весов (ровно та ловушка со stale-метрикой, что разбиралась в `reports/berta_tpu_lora_provenance/`). Внутри лежит `PROVENANCE.md` с этим предупреждением.
