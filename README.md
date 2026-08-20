# Ozon E-CUP 2026 — Сопоставление товаров

Классификация дублей пар товаров для хакатона Ozon Tech E-CUP 2026
(контейнерное соревнование — retrieval уже сделан организаторами, этот
репозиторий — только этап попарной классификации/ре-ранжирования).

Проект прошёл несколько архитектурных итераций: от baseline-кроссэнкодера
через LoRA-дообучение к нескольким параллельно исследуемым направлениям —
более крупные энкодеры (BERTA, FRIDA, Giga-3B) и bi-encoder на
Qwen3-Embedding. Актуальный лучший результат на реальном тестовом стенде —
у BERTA cross-encoder + LoRA, дообученной на TPU с `PAIR_MAX_LENGTH=256`
(см. таблицу ниже).

Подробности (точная формула препроцессинга, определение метрики, особенности
железа, полная история экспериментов) — в **[CLAUDE.md](CLAUDE.md)**: этот
README — быстрый старт и карта репозитория, CLAUDE.md — полный референс,
`reports/experiment_log.md` — сырая хронология с разбором багов.

## Текущий результат

Метрика — macro-averaged PR-AUC по 20 категориям (см. CLAUDE.md → "Метрика").
Ниже — **реальные скоры тестового стенда** (не Kaggle val — см. CLAUDE.md про
то, что означает "тест" в этом проекте) для сабмитов, которые дошли до
судейства:

| Сабмит (`submits/<dir>`) | Тест-скор | Архитектура |
|---|---|---|
| **`matching-berta_tpu_lora-submit`** (`submits/berta_tpu_lora`) | **0.47 (лучший)** | BERTA (128.3M) cross-encoder + LoRA, TPU SFT на `matches_llm.parquet`+human, `PAIR_MAX_LENGTH=256` — чекпоинт восстановлен из упавшего рана на 50% эпохи, подробный разбор происхождения в `reports/berta_tpu_lora_provenance/` |
| `berta_tpu_lora_after_llm` | 0.466 | тот же тренировочный кернел, чекпоинт на 100% эпохи (после фикса краша) — val у него реально **ниже** (~0.697 против 0.7087 у 50%-чекпоинта, не просто устаревшая метрика в `train_state.json`) — согласуется с просадкой на тесте |
| `berta_catboost_pca32` | 0.469 | топ-чекпоинт + PCA(32) на эмбеддинге + classifier_score + tf-idf + category → CatBoost — фикс переобучения `berta_catboost_top_final` (см. ниже), почти закрыл разрыв, но всё ещё чуть ниже чистого классификатора |
| `ecup_qwen_stage2_submit` | 0.41 | Qwen3-Embedding-0.6B bi-encoder + LoRA + MLP-голова |
| `berta_catboost_top_final` | 0.457 | топ-чекпоинт + 768 сырых эмбеддингов + classifier_score + tf-idf + category → CatBoost — **хуже чистого классификатора** несмотря на более высокий val (0.7467 против 0.7087); переобучение, см. раздел ниже |
| `berta_catboost` | 0.40 | BERTA (128.3M) cross-encoder + LoRA + CatBoost-голова, `PAIR_MAX_LENGTH=128` |
| `berta_catboost_256_len` | 0.39 | то же, но `PAIR_MAX_LENGTH=256` — скор **ниже** 128-версии, вероятная причина: та BERTA дообучалась на длине пары 128, не 256 |
| `ecup_qwen_bi_encoder` | 0.37 | Qwen3-Embedding bi-encoder, более ранняя версия |
| `ecup_qwen_bi_encoder_128_len` | 0.35 | то же, `max_length=128` |

Два наблюдения, которые пока остаются гипотезами (не подтверждены
контролируемым сравнением):
- **Рост `PAIR_MAX_LENGTH` 128→256, похоже, поднимает тест-скор** —
  `matching-berta_tpu_lora-submit` (256, 0.47) заметно обгоняет
  `berta_catboost` (128, 0.40), хотя это не чистый A/B (разные чекпоинты
  BERTA, разные головы). Правда, у `berta_catboost_256_len` подъём
  `PAIR_MAX_LENGTH` до 256 **не помог** — но там сама BERTA обучалась на
  128, а здесь у BERTA-TPU-чекпоинта тренировка и инференс совпадают по
  длине (256/256), что и должно быть решающим фактором, а не сама цифра.
- **`ecup_qwen_stage2_submit`** — чекпоинт, который просел на val (~1%)
  относительно предыдущего, но вырос на тесте — намекало на "переобучение
  под train помогает тесту". Пара `matching-berta_tpu_lora-submit` /
  `berta_tpu_lora_after_llm` этот паттерн **не повторяет**: там просадка и
  на val, и на тесте идут в одну сторону — обычная, ожидаемая корреляция.
  То есть эффект из qwen-случая не выглядит универсальным правилом, скорее
  специфичен для того конкретного чекпоинта/архитектуры — требует больше
  точек данных, прежде чем на него полагаться.

### Известное ограничение: мало данных без лика для головы поверх BERTA

`berta_catboost_top_final` (CatBoost на 768 сырых эмбеддингах BERTA +
classifier_score + tf-idf + category) вышел **хуже** чистого классификатора
на тесте (0.457 против 0.47), хотя на val было наоборот (0.7467 против
0.7087). `get_feature_importance()` показал причину: 96.11% важности — на
768 сырых измерениях эмбеддинга, размазанных почти плоско (максимум
отдельного измерения 0.43%), а сам `classifier_score` — та единственная
фича, которая одна даёт 0.47 на тесте — получил только 1.9%. Классическое
переобучение: слишком много фич на слишком малом числе строк.

**Почему строк так мало.** Топ-чекпоинт BERTA (`berta-tpu-lora-top-checkpoint`)
обучался на человеческом **train**-сплите (`matches.parquet`, 292,523 строки,
80%) вместе с `matches_llm.parquet`. Значит для головы поверх его
эмбеддингов без лика годится только **val**-сплит — 73,131 строка (20%),
дальше делённая 80/20 на train/val самого catboost → всего 58,504 строки
на обучение catboost. Этого хватает для одной-двух признаков, но не для
771 (768 эмбеддинга + 3). `matches_llm.parquet` мог бы дать на порядок
больше данных, но качество LLM-разметки хуже человеческой и распределение
другое (полностью непересекающийся с `matches.parquet` набор товаров — 0
общих id, см. CLAUDE.md → "Данные") — просто подмешать её нельзя, нужна
отдельная проверка.

**Частичный фикс** — `berta_catboost_pca32`: PCA(32) на эмбеддинге вместо
768 сырых измерений + усиленная регуляризация (`depth` 8→6, `l2_leaf_reg`
15→30). Важность `classifier_score` выросла до 19.3% (в 10 раз), val почти
не просел (0.7419 против 0.7467), тест подрос до 0.469 — но всё ещё чуть
ниже 0.47. Проблема нехватки данных смягчена, но не решена полностью.

**Следующий шаг (в проработке)**: закэшировать эмбеддинги на **всей**
человеческой разметке (все 365,654 пар, не только val-срез), используя
чекпоинт, который для этих конкретных пар не участвовал в обучении —
кандидат: `kaggle_kernels/berta-tpu-human-continue-384` (дообучен уже
после топ-чекпоинта на human-only данных). Дальше — многослойная голова
поверх зафиксированных эмбеддингов (архитектура ещё не выбрана), с
промежуточным слоем как потенциальной фичей для CatBoost. Кэширующий
кернел под это уже подготовлен (`kaggle_kernels/berta-embed-cache-all-human`),
но пока не запущен — ждём подтверждения, что `berta-tpu-human-continue-384`
действительно даёт прирост качества.

Исторически первый работающий сабмит — LoRA-дообученный
`cross-encoder/ms-marco-MiniLM-L-12-v2` (`submits/lora/`):

| Вариант | macro PR-AUC (val) | Mean PR-AUC (лидерборд) |
|---|---|---|
| Baseline (frozen encoder + LogReg) | 0.3710 | 0.1333 |
| LoRA SFT | 0.5282 | 0.265 |

Полная сравнительная таблица по всем архитектурам и чекпоинтам — в
`reports/experiment_log.md`.

## Архитектуры

Коротко, что перепробовано (детали и обоснование выбора — в CLAUDE.md):

- **`submits/lora`** — исходный baseline: `cross-encoder/ms-marco-MiniLM-L-12-v2`
  (33M) + LoRA SFT на `matches.parquet`. Отправная точка проекта.
- **BERTA** (`sergeyzh/BERTA`, 128.3M, дистиллят FRIDA в BERT-форме) —
  LoRA-дообученный cross-encoder, с несколькими вариантами головы: линейная
  (`submits/berta_probe`, `submits/berta_tpu_lora`) и CatBoost поверх
  эмбеддингов (`submits/berta_catboost`, `submits/berta_catboost_256_len`).
  Обучающие кернелы: `kaggle_kernels/berta-lora-sft*`, `berta-merge-lora`,
  `berta-catboost-*`, `berta-llm-tpu-sft` (полномасштабный SFT на TPU
  v5e-8 поверх `matches_llm.parquet`).
- **FRIDA** (`ai-forever/FRIDA`, 823M, T5-энкодер) — bi-encoder,
  LoRA-дообученный. Выбран как более дешёвая альтернатива для
  пере-токенизации (item-level кэш вместо per-pair). Кернелы:
  `kaggle_kernels/frida-tokenize-items*`, `frida-lora-sft*`,
  `frida-lora-continue-2ep`, `frida-lora-v2-epoch` (актуальная версия —
  тайм-аут-устойчивые чекпоинты, новая политика упорядочивания атрибутов).
  Сабмит: `submits/frida_lora_adapter`, `submits/frida_probe`.
- **Giga-Embeddings-instruct** (3.45B, `giga3b_probe`) — проба
  bi-encoder-подхода на ещё большем энкодере; zero-shot cosine similarity
  уже бьёт обученный baseline, но candidate-пары почти не переиспользуют
  товары, так что bi-encoder-выигрыш ("encode once, compare many") не
  реализуется — по FLOPs не укладывается в лимиты судейского стенда.
  Запаркован, см. `reports/experiment_log.md`.
- **`omarelshehy/arabic-english-sts-matryoshka`** (~560M, `matryoshka_probe`) —
  проба "насколько крупный энкодер потянет H100-стенд" в схеме
  baseline-архитектуры (frozen encoder → CLS → LogReg).
- **Qwen3-Embedding-0.6B bi-encoder** (`submits/ecup_qwen_stage2_submit`,
  `ecup_qwen_biencoder_submit_128_length`, `matching-qwen-berta-catboost-submit`) —
  LoRA + MLP/CatBoost голова поверх эмбеддингов. **Текущий лучший результат
  проекта на реальном тесте.**

## Установка

```bash
git clone <this-repo>
cd ozon_matching
python -m venv .venv
.venv\Scripts\activate           # Windows; source .venv/bin/activate на macOS/Linux
pip install polars pandas pyarrow scikit-learn mlflow kaggle torch transformers peft safetensors
```

Kaggle-креды (нужны для любой команды `kaggle datasets ...` / `kaggle
kernels ...` ниже): положите `kaggle.json` в `~/.kaggle/kaggle.json`
(`C:\Users\<вы>\.kaggle\kaggle.json` на Windows) — **никогда не коммитьте
его**. Получить можно на https://www.kaggle.com/settings → API Tokens →
Create Legacy API Key.

В проекте используются **три Kaggle-аккаунта** с раздельными GPU/TPU-квотами
(`maxamsecret`, `rays0n`, `maximmarkovbb`) — подробности и как переключаться
между ними в CLAUDE.md → "Environment notes".

## Данные (не в git, слишком тяжёлые)

| Файл | Размер | Для чего |
|---|---|---|
| `matches.parquet` | 4MB | 365K пар с человеческой разметкой — train/val |
| `items_human.parquet` | 214MB | Каждый товар, на который ссылается `matches.parquet` — **используйте это для локальной разработки/тестов**, а не полный `items.parquet` |
| `matches_llm.parquet` | 104MB | 11.19M пар с LLM-разметкой (слабые/мягкие метки, непрерывные 0..1) — вторая стадия continued LoRA SFT |
| `items.parquet` | 4.1GB | Полный каталог товаров (13M+ строк) — нужен только для полномасштабного прогона |

### Вариант A: прямая ссылка (без Kaggle-аккаунта)

Организаторский bucket на Yandex Cloud, публичный, без креденшлов:

```bash
python scripts/download_data.py                              # все 4 файла в data/
python scripts/download_data.py matches.parquet items_human.parquet   # только нужные
python scripts/download_data.py --overwrite items.parquet     # перекачать заново
```

Скрипт пропускает файл, если он уже лежит в `data/` (кроме `--overwrite`), и
пишет во временный `*.parquet.part`, атомарно переименовывая по завершении —
битый файл в `data/` не останется, даже если скачивание прервётся.

### Вариант B: Kaggle-датасет

Все данные соревнования также лежат в **[maxamsecret/ozon-ecup-2026-matching](https://www.kaggle.com/datasets/maxamsecret/ozon-ecup-2026-matching)** (приватный — нужен доступ):

```bash
mkdir -p data
# Только то, что нужно для локальной разработки (matches.parquet + items_human.parquet):
kaggle datasets download maxamsecret/ozon-ecup-2026-matching -f matches.parquet -p data
kaggle datasets download maxamsecret/ozon-ecup-2026-matching -f items_human.parquet -p data
# Всё целиком, включая items.parquet на 4.1GB:
kaggle datasets download -d maxamsecret/ozon-ecup-2026-matching -p data --unzip
```

Веса моделей публикуются как отдельные Kaggle-датасеты по мере обучения
(смёрженные LoRA-веса, PEFT-адаптеры, item-token-кэши) — полный список
источник/потребитель по каждому кернелу в `kaggle_kernels/README.md`.

## Структура репозитория

```
notebooks/                  тренировочные ноутбуки (Kaggle, LoRA SFT, анализ длин)
kaggle_kernels/              ~45 отдельных Kaggle-кернелов (токенизация, SFT всех
                              архитектур, майнинг негативов, пробы) — см. свой README
scripts/                     локальные dev-скрипты (сплит, сборка zip, smoke-тесты,
                              анализ per-category метрик, догрузка probe-моделей)
data/                        локальные копии parquet-файлов выше (в .gitignore)
data/splits/                 канонический train/val сплит (маленький, в git — см. CLAUDE.md)
negative_patterns/            по-категорийный набор hard-negative пар + описание паттернов
                              (20 категорий, генерируется kaggle_kernels/negative-pattern-sampling)
reports/                     письменные находки: experiment_log.md (полная хронология),
                              per-category val-метрики, анализ пересечения LLM-меток
submits/baseline/            референсное решение от организаторов
submits/lora/                исходный LoRA-сабмит (MiniLM cross-encoder)
submits/berta_*/             BERTA cross-encoder варианты (probe/catboost/TPU)
submits/frida_*/             FRIDA bi-encoder варианты
submits/giga3b_probe/        Giga-3B bi-encoder проба
submits/matryoshka_probe/    матрёшка-энкодер проба (baseline-архитектура)
submits/ecup_qwen_*/,
submits/matching-qwen-*/     Qwen3-Embedding bi-encoder — текущий лучший результат
```

## Пересборка и тестирование сабмита

Паттерн одинаков для любого `submits/<name>/` (конкретные шаги сборки весов
отличаются по архитектуре — см. README/докстринг соответствующего
`kaggle_kernels/*`):

```bash
python scripts/build_submission_zip.py --submission-dir submits/<name>
python scripts/test_submission.py --submission-dir submits/<name> --n-pairs 1000
```

`test_submission.py` собирает небольшой реальный сэмпл из
`data/items_human.parquet` + `data/matches.parquet` и запускает `run.py`
через тот же CLI-контракт, что использует судейский харнесс, проверяя формат
вывода, совпадение числа строк и детерминизм. **Всегда также распаковывайте
собранный zip в чистую папку и тестируйте именно его** (а не просто папку
`submits/<name>/`) — баги упаковки (обёрточная папка в zip, пакет, которого
нет в судейском Docker-образе) проявляются только на реальном архиве.

Для LoRA-сабмита (`submits/lora`) есть отдельный шаг мёржа адаптера:
```bash
python scripts/merge_lora.py                # LoRA-адаптер -> смёрженные веса
```

## Трекинг экспериментов

Локальный mlflow file-store, без сервера: `mlflow ui --backend-store-uri
experiments/mlruns` чтобы посмотреть. На mlflow≥3 обязательна переменная
окружения `MLFLOW_ALLOW_FILE_STORE=true` (см. CLAUDE.md → "Environment
notes").

## Kaggle-кернелы

~45 кернелов, разбитых по архитектурам/задачам (токенизация, SFT, майнинг
негативов, пробы размера энкодера) — полный аннотированный список,
датасеты-источники/потребители и как запускать под каждым из трёх
Kaggle-аккаунтов — в **[kaggle_kernels/README.md](kaggle_kernels/README.md)**.

## Ограничения соревнования, кратко

Полный разбор — в CLAUDE.md, здесь только самые острые углы:

- Судится **офлайн**, в фиксированном Docker-образе
  (`odsai/ecup26-matching-baseline:1.0` по умолчанию) — код может
  импортировать только пакеты, которые реально есть в этом образе
  (`numpy`, `pandas`+`pyarrow`, `torch`, `transformers`; **никакого**
  `polars` — узнали это из упавшего сабмита).
- Железо: 1x H100 80GB, 20 CPU, 200GB RAM.
- 3 отдельных лимита по времени, не один общий бюджет: Check 1 мин
  (1000 пар), Public 6 мин (~115K пар), Private 13 мин (~275K пар).
- **5 сабмитов в день, жёсткий лимит.** Тестируйте локально + на Kaggle GPU
  dry-run прежде чем тратить попытку.
