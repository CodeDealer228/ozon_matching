# BERTA + LoRA, TPU SFT на `matches_llm.parquet`+human (`PAIR_MAX_LENGTH=256`)

**Тест-скор: 0.47 — лучший результат проекта на реальном тестовом стенде.**
Zip: `matching-berta_tpu_lora-submit.zip`.

## Что это за решение

Cross-encoder: `sergeyzh/BERTA` (BertModel, 128.3M параметров, **mean
pooling** — не CLS, это единственное намеренное отличие BERTA от её
родителя FRIDA согласно карточке модели) + смёрженный LoRA (веса слиты в
базу через `merge_and_unload()`, поэтому `peft` на инференсе не нужен) +
линейный классификатор.

## Гиперпараметры обучения

Финальный этап — `kaggle_kernels/berta-llm-tpu-sft` (TPU v5e-8), полный
`matches_llm.parquet` (11.19M строк, без сэмплирования — TPU-сессия это
позволила) вперемешку с человеческим train-сплитом на каждом батче:

| Параметр | Значение |
|---|---|
| LoRA | `r=16, alpha=32, dropout=0.10`, `target_modules=["query","key","value","dense"]` (реальные имена BERT-модулей) |
| LR | `2e-4`, warmup 10% |
| `PAIR_MAX_LENGTH` | **256** (поднято с 128 — тот самый параметр, который отличает этот чекпоинт от более ранних BERTA-сабмитов с тестом 0.39-0.40) |
| Данные | весь `matches_llm.parquet` (мягкие LLM-метки, continuous target в `BCEWithLogitsLoss`, без порога) + весь human train (292,523), человеческие строки — в каждом батче |
| Батч | `PER_CORE_TRAIN_BATCH=64` × 8 TPU-ядер = 512 эффективный |

**Использованный чекпоинт восстановлен из упавшего прогона.** Кернел
`berta-llm-tpu-sft` был первым настоящим (не только токенизационным)
тренировочным прогоном на TPU в проекте и упал с `BrokenProcessPool` (8
fork-процессов независимо грузили ~13GB item-кэша каждый, эксгаустило RAM
хоста). Val **0.7087** зафиксирован на шаге 11211 (~50% эпохи) — это и есть
опубликованный `maxamsecret/berta-tpu-lora-top-checkpoint`. После фикса
краша прогон довели до 100% эпохи — но та версия дала val **ниже** (~0.697,
не устаревшая метрика — измерено заново) и тест **0.466**
(`berta_tpu_lora_after_llm`), то есть 50%-чекпоинт оказался лучше не только
на val, но и на тесте — согласованное, не случайное расхождение.

## Логика обрезки текста item'а

Стандартная формула из корневого `CLAUDE.md` (byte-match обязателен между
обучением и инференсом):

```python
MAX_NAME_CHARS, MAX_CATEGORY_CHARS, MAX_ATTR_CHARS = 128, 64, 384
MAX_ITEM_CHARS = 128 + 64 + 384 + 64  # = 640

text = f"Name: {name[:128]} Category: {category[:64]} Attributes: {attributes[:384]}"[:640]
```

`attributes` — **сырая обрезка строки**, без парсинга JSON (в отличие от
FRIDA-256, у BERTA нет v2-политики переупорядочивания атрибутов — эта ветка
лежит отдельно, см. `../../frida/tpu_human_256_epoch4/README.md`).
Токенизация парная: `tokenizer(text1, text_pair=text2, max_length=256,
padding="max_length", truncation=True)` — обычный джойнт-тюнкейт
токенайзера (`truncation="longest_first"`), не fair-share как у FRIDA
(BERTA — честный BERT-стиль cross-encoder с `[CLS] item1 [SEP] item2
[SEP]` + `token_type_ids`, а не T5 без сегментов).

## Провенанс: последовательность кернелов и Kaggle-датасетов

```
sergeyzh/BERTA (базовая модель, HF)
  │
  ▼
kaggle_kernels/berta-lora-sft-human-only-20-epochs (1×T4, rays0n)
  → val 0.7111 @ 22% плана (упёрлись в лимит сессии)
  │
  ▼
kaggle_kernels/berta-lora-sft-continue-5epoch (2×T4, rays0n)
  → val 0.7246 (эпоха 7) / 0.7229 (фактически сохранённая эпоха 9)
  │
  ▼
kaggle_kernels/berta-llm-continue-train (2×T4)          — continue-SFT на LLM-метках + human-mix
  → val 0.7255 → датасет rays0n/berta-llm-continue-train
  │
  ▼
kaggle_kernels/berta-merge-lora (CPU)                    — мёрж адаптера в базу
  → смёрженные веса, peft больше не нужен на инференсе
  │
  ▼
kaggle_kernels/berta-tokenize-llm-256 (TPU)              — item-token-кэш, max_length=256
  → датасеты llm_scope (12.38M items) / human_scope (711K items)
  │
  ▼
kaggle_kernels/berta-llm-tpu-sft (TPU v5e-8)             — полный matches_llm.parquet + human-mix, PAIR_MAX_LENGTH 128→256
  → val 0.7087 на шаге 11211 (~50% эпохи, до BrokenProcessPool-краша)
  → датасет maxamsecret/berta-tpu-lora-top-checkpoint
```

Полная хронология, все промежуточные эксперименты (hard negatives,
weak-category continue-train, attribute augmentation) и обоснование каждого
решения — `kaggle_kernels/README.md` → раздел «BERTA».

## Известное свойство: PAIR_MAX_LENGTH=256 — вероятный ключевой фактор

Единственный, наравне с LLM-миксом, параметр, который надёжно отличает этот
0.47-чекпоинт от более ранних BERTA-сабмитов на 128 (0.39-0.40). Оба
изменения внесены одновременно в одном прогоне — чистого A/B нет, но
FRIDA-256 (`../../frida/tpu_human_256_epoch4/`) целенаправленно тестирует
влияние именно длины пары на другой архитектуре.
