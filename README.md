# Ozon E-CUP 2026 — Сопоставление товаров

Классификация дублей пар товаров для хакатона Ozon Tech E-CUP 2026
(контейнерное соревнование — retrieval уже сделан организаторами, этот
репозиторий — только этап попарной классификации/ре-ранжирования). Подход:
LoRA-дообученный `cross-encoder/ms-marco-MiniLM-L-12-v2`.

Подробности (точная формула препроцессинга, определение метрики, особенности
железа, история сабмитов/уроки) — в **[CLAUDE.md](CLAUDE.md)**: этот README —
быстрый старт, CLAUDE.md — полный референс.

## Текущий результат

| Вариант | macro PR-AUC (наш val-сплит) | Mean PR-AUC (реальный лидерборд) |
|---|---|---|
| Baseline (frozen encoder + LogReg) | 0.3710 | 0.1333 |
| **LoRA SFT (наш)** | **0.5282** | **0.265** |

Скор на лидерборде — с первого сабмита, прошедшего все стадии проверки (см.
CLAUDE.md → "Submission history"). Он ниже нашего val-скора для обоих
вариантов — товары в скрытом тесте отличаются от всего, что есть в
`matches.parquet`, так что часть разрыва — ожидаемый generalization gap — но
преимущество LoRA над baseline сохранилось: ~2x на лидерборде против ~1.4x на
val.

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

## Данные (не в git, слишком тяжёлые)

| Файл | Размер | Для чего |
|---|---|---|
| `matches.parquet` | 4MB | 365K пар с человеческой разметкой — train/val |
| `items_human.parquet` | 214MB | Каждый товар, на который ссылается `matches.parquet` — **используйте это для локальной разработки/тестов**, а не полный `items.parquet` |
| `matches_llm.parquet` | 104MB | 11.19M пар с LLM-разметкой (слабые/мягкие метки, непрерывные 0..1) — используется для второй стадии continued LoRA SFT, см. ниже |
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

Веса модели (нужны для запуска/пересборки сабмита, не в git):

- **[maxamsecret/ozon-lora-merged-minilm](https://www.kaggle.com/datasets/maxamsecret/ozon-lora-merged-minilm)** — смёрженные LoRA-веса, готовые к инференсу (без `peft`). Это то, что использует `submits/lora/`.
- **[maxamsecret/ozon-lora-minilm-finetuned](https://www.kaggle.com/datasets/maxamsecret/ozon-lora-minilm-finetuned)** — исходный чекпоинт PEFT-адаптера (до merge), если нужно пересобрать merge или посмотреть сам адаптер.

```bash
kaggle datasets download -d maxamsecret/ozon-lora-merged-minilm -p artifacts/lora_merged_dataset --unzip
```

## Структура репозитория

```
notebooks/                  тренировочный ноутбук (запускается на Kaggle, 2xT4, LoRA SFT)
kaggle_kernels/              отдельные Kaggle-кернелы (валидация, LoRA merge, timing dry-run,
                              continued SFT на LLM-размеченных данных)
scripts/                     локальные dev-скрипты (генерация сплита, сборка zip, smoke-тесты)
data/                        локальные копии parquet-файлов выше (в .gitignore)
data/splits/                 канонический train/val сплит (маленький, в git — см. CLAUDE.md)
submits/baseline/            референсное решение от организаторов
submits/lora/                наш сабмит (run.py / src/utils.py / metadata.json + веса модели)
```

## Пересборка и тестирование сабмита

```bash
python scripts/merge_lora.py                # LoRA-адаптер -> смёрженные веса в artifacts/lora_merged/
# скопировать смёрженные файлы в submits/lora/models/lora-merged-ms-marco-MiniLM-L12-v2/ и classifier.pt
python scripts/build_submission_zip.py       # -> submits/lora/matching-lora-submit.zip
python scripts/test_submission.py --submission-dir submits/lora --n-pairs 1000
```

`test_submission.py` собирает небольшой реальный сэмпл из
`data/items_human.parquet` + `data/matches.parquet` и запускает `run.py`
через тот же CLI-контракт, что использует судейский харнесс, проверяя формат
вывода, совпадение числа строк и детерминизм. **Всегда также распаковывайте
собранный zip в чистую папку и тестируйте именно его** (а не просто папку
`submits/lora/`) — в разделе "Submission history" в CLAUDE.md описаны два
реальных бага (обёрточная папка в zip, пакет, которого нет в судейском
Docker-образе), которые проявились именно так.

## В процессе: continued LoRA SFT на `matches_llm.parquet`

Второй этап SFT поверх чекпоинта с val-скором 0.5282 выше, на 11.19M ранее
неиспользованных парах с LLM-разметкой (мягкие, непрерывные метки 0..1) —
1 эпоха, `LR=4e-4` с cosine-расписанием (в 2 раза выше пикового LR исходного
прогона, выбрано по анализу реальной кривой loss исходного прогона — полное
обоснование в CLAUDE.md), валидация на *том же самом* каноническом
val-сплите из 73,131 строки. Эмпирически подтверждено, что
`matches_llm.parquet` полностью не пересекается с `matches.parquet` — ни по
парам, ни по множеству id товаров (0 общих id из 12.4M против 711K), так что
утечки нет, и это реально расширяет покрытие каталога, а не переразмечает те
же пары.

Кернел: [maxamsecret/lora-llm-continue-train](https://www.kaggle.com/code/maxamsecret/lora-llm-continue-train) (приватный) —
`kaggle_kernels/lora-llm-continue-train/`. Валидируется и чекпоинтится каждые
~10% шагов внутри эпохи (не только раз в эпоху), чтобы лучший адаптер всегда
был рядом на диске, плюс throttled push версий датасета для устойчивости к
сбоям за пределами одной Kaggle-сессии.

**Ещё не запущено в полном масштабе — тестовый прогон на 500K строк — это
жёлтый флаг, а не зелёный свет.** Проблемы с памятью из ранних
smoke-итераций исправлены и подтверждены безопасными на масштабе. Но
`val_macro_pr_auc` этого же прогона упал со стартовых 0.5282 до ~0.50-0.51 и
не восстановился даже после полного затухания LR — это реальная регрессия,
а не шум, хотя не обязательно репрезентативная для полного прогона (это был
полный, сжатый cosine-цикл на сэмпле в 500K строк, а не первые 500K строк
реального расписания на 11.19M строк). `LR=4e-4` пересматривается перед тем
как тратить GPU-квоту на полный прогон. Полное описание дизайна, история
отладки по памяти и текущий статус — в CLAUDE.md → "Continued LoRA SFT on
matches_llm.parquet".

## Трекинг экспериментов

Локальный mlflow file-store, без сервера: `mlflow ui --backend-store-uri
experiments/mlruns` чтобы посмотреть. Детали настройки в CLAUDE.md →
"Experiment tracking & caching" (на mlflow≥3 обязательна переменная
окружения `MLFLOW_ALLOW_FILE_STORE=true`).

## Полезные Kaggle-кернелы (приватные)

- [Валидация baseline vs LoRA](https://www.kaggle.com/code/maxamsecret/ozon-baseline-vs-lora-val-macro-pr-auc) — сравнение, стоящее за таблицей результатов выше
- [LoRA merge](https://www.kaggle.com/code/maxamsecret/ozon-merge-lora-adapter-into-base-weights) — производит датасет `ozon-lora-merged-minilm`
- [GPU timing dry-run сабмита](https://www.kaggle.com/code/maxamsecret/ozon-lora-submission-gpu-timing-dry-run-offline) — полномасштабная офлайн-проверка тайминга против судейских лимитов
- [LoRA continue-train на LLM-размеченных парах](https://www.kaggle.com/code/maxamsecret/lora-llm-continue-train) — второй этап SFT на `matches_llm.parquet`, см. выше

## Ограничения соревнования, кратко

Полный разбор — в CLAUDE.md, здесь только самые острые углы:

- Судится **офлайн**, в фиксированном Docker-образе
  (`odsai/ecup26-matching-baseline:1.0` по умолчанию) — код может
  импортировать только пакеты, которые реально есть в этом образе
  (`numpy`, `pandas`+`pyarrow`, `torch`, `transformers`; **никакого**
  `polars` — узнали это из упавшего сабмита).
- 3 отдельных лимита по времени, не один общий бюджет: Check 1 мин
  (1000 пар), Public 6 мин (~115K пар), Private 13 мин (~275K пар).
- **5 сабмитов в день, жёсткий лимит.** Тестируйте локально + на Kaggle GPU
  dry-run прежде чем тратить попытку.
