# FRIDA full fine-tune (без LoRA), обслужено на `PAIR_MAX_LENGTH=180`

**Тест-стенд: 0.481 — новый лучший результат во всём проекте** (обошёл
предыдущий лучший `matching-berta_tpu_lora-submit`, 0.47). Val macro
PR-AUC чекпоинта — **0.7582**, новый рекорд val на момент обучения.
Первый **полный** (не LoRA) файнтюн FRIDA, дошедший до сабмита.

Обучен кернелом `kaggle_kernels/frida/full_sft/full-sft/frida_full_sft.py` (TPU
v5e-8, account `maxamsecret`), веса опубликованы как
`maxamsecret/frida-full-sft-w`. Собран в сабмит кернелом
`kaggle_kernels/frida/full_sft/full-sft-submit-build` (CPU-only, `maxamsecret`) —
zip 1.31GB.

## Архитектура

Cross-encoder: `ai-forever/FRIDA` (`T5EncoderModel`, 823M параметров, **все
обучаемые** — не LoRA, впервые в этом проекте) + `FullSftHead`
(`[1024, 512, 128, 1]`, три блока `Linear→LayerNorm→GELU→Dropout(0.2)`,
сужающие с `hidden_dim=1536` до одного логита) поверх CLS-пулинга
(позиция 0).

## Данные

| | Источник | Строк |
|---|---|---|
| Human train | `data/splits/matches_item_disjoint_70_30_seed42.parquet` (seed=42, union-find по графу пар + стратификация категория×label×терциль длины, **0 общих item id** между train/val) | 256,006 |
| Human val | тот же сплит | 109,648 |
| LLM train | `frida-llm-train-val-split` (no-leak, повторяет байт-в-байт исходную стратифицированную выборку 1M из `matches_llm.parquet`) — **подвыборка, не все 11.19M строк** | 1,000,000 |
| LLM val | тот же сплит, отдельные 100K строк, **никогда не пересекавшиеся** с train-подвыборкой | 100,000 |
| **Итого train (mixed)** | human + llm | **1,256,006** |

`val_fingerprint` (human) = `0fd633cbb571c59a`, LLM val fingerprint =
`d61701d88db0e50d` — зафиксированы в mlflow-стиле для воспроизводимости.

Веса категорий в лоссе (inverse-frequency по **реальному training-миксу**,
не по сырым данным): min=0.946, max=1.011 — почти равномерно, потому что
LLM-подвыборка уже была сбалансирована по категориям на этапе построения
сплита.

## Текстовый препроцессинг (byte-match с `frida-tokenize-items-cpu`, v3-кэш)

```
MAX_NAME_CHARS, MAX_CATEGORY_CHARS, MAX_ATTR_CHARS = 140, 32, 529
MAX_ITEM_CHARS = 731
ITEM_MAX_LENGTH = 256
PREFIX = "paraphrase: "
```

v2-политика переупорядочивания атрибутов (длинные/https/oem — в конец),
`Category:`-спан вырезается по паттерну токенов (нулевой дискриминативный
сигнал), fair-share truncation до `PAIR_MAX_LENGTH`.

**Обучено** на `PAIR_MAX_LENGTH=192`. **Сабмит обслуживает на 180** —
намеренное расхождение train/serve (тот же паттерн, что и у всех LoRA
FRIDA-сабмитов в проекте: 192 не укладывался в бюджет времени на
судейском H100, 180 — ближе к границе с запасом).

## Лосс

**Category-weighted focal BCE** (`gamma=2.0`, дефолт Lin et al./RetinaNet,
не тюнился) + вес категории (inverse-frequency, см. выше) + `HUMAN_WEIGHT=2.0`
(человеческие строки весят вдвое против LLM soft-label строк) — все три
множителя перемножаются в одном взвешенном среднем.

Таргеты: human — бинарные (`target >= 0.5` в сплите). LLM train — **непрерывные**
soft-label (никогда не тресхолдятся). LLM val — бинаризованные (нужен
дискретный ground truth для `average_precision_score`).

```python
def focal_bce_per_sample(logits, targets, gamma):
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    pt = targets * p + (1 - targets) * (1 - p)
    return ((1 - pt).clamp(min=1e-6) ** gamma) * ce
```

## Оптимизатор и LR

**Adafactor** (`transformers.optimization.Adafactor`, НЕ AdamW) —
намеренный выбор: T5-семейство — родная архитектура Adafactor, а Adam'у
для full-FT потребовалось бы fp32-веса + fp32-град + (m,v) = 4x параметров
(13.2GB против 16GB HBM у v5e-8-ядра, до всяких активаций); факторизованный
второй момент Adafactor стоит единицы MB на тензор вместо полного размера.

```python
Adafactor(param_groups, scale_parameter=False, relative_step=False,
          warmup_init=False, weight_decay=0.01)
```

Два param-группы с **противоположно направленным** прогревом:
- **backbone**: `LR=1e-4` (peak), линейный warmup 0→peak на 10% шагов,
  затем косинусный спад до `LR_MIN_RATIO=0.05` от пика.
- **head**: старт на `LR * HEAD_LR_PEAK_MULTIPLIER = 1e-3`, линейно
  **остывает** до `1e-4` за то же окно warmup'а, затем присоединяется к
  тому же косинусному спаду — свежеинициализированная голова быстро
  учится, не доминируя градиентом над бэкбоном на старте.

`WEIGHT_DECAY=0.01`.

## Регуляризация/память

- **Gradient checkpointing** включён:
  `model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True, "preserve_rng_state": False})`
  (после двух неудачных попыток — см. докстринг кернела, "FOURTH/FIFTH REAL
  CRASH": `preserve_rng_state=False` убирает реальный источник
  `AttributeError: module 'torch' has no attribute 'xla'`, `use_reentrant`
  сам по себе не помогал).
- `attn_implementation="eager"` (не sdpa — не тестировался на TPU для этого
  семейства, оставлен как проверенная конвенция).
- bf16 autocast **внутри** `forward()` модели (не снаружи — известный фикс
  NaN на этой архитектуре).

## Батч / железо / бюджет

| | |
|---|---|
| TPU | v5e-8 (8 ядер) |
| `PER_CORE_TRAIN_BATCH` | 20 (после 3 реальных OOM на 64→48→40, см. докстринг) |
| `PER_CORE_EVAL_BATCH` | 72 |
| Глобальный train-батч | 160 (20×8) |
| `EPOCHS` | 3 (пройдены полностью, не обрезаны по времени) |
| `MAX_TRAIN_SECONDS` | 30600 (8.5ч) — реальный прогон уложился в 22,748s (~6.32ч), с запасом |
| Чекпоинты | тихий каждый час (веса+optimizer, без eval, `val: null`, перезаписывает `last/`); полная валидация на границе эпохи → `checkpoint_epochNN/`, если улучшение — `best/` |

## Траектория обучения (реальный прогон, полный лог)

| Эпоха | Шаг | val_macro_pr_auc (human, 109,648) | llm_val_macro_pr_auc (100,000) |
|---|---|---|---|
| 1/3 | 7,850 | 0.7459 | 0.6752 |
| 2/3 | 15,700 | 0.7579 | 0.6910 |
| **3/3 (финал)** | **23,550** | **0.7582** | **0.6916** |

Монотонный рост на всех трёх эпохах, ни одного расхождения (NaN) —
полное wall-clock время 22,748s.

## Per-category val (human, финальный чекпоинт, худшие → лучшие)

| Категория | val PR-AUC |
|---|---|
| Обувь | 0.4697 |
| Ювелирные изделия | 0.5112 |
| Одежда | 0.5196 |
| Галантерея и аксессуары | 0.6730 |
| Автотовары | 0.6849 |
| Мебель | 0.7031 |
| Спорт и отдых | 0.7136 |
| Электроника | 0.7540 |
| Строительство и ремонт | 0.7716 |
| Канцелярские товары | 0.7869 |
| Красота и гигиена | 0.7967 |
| Дом и сад | 0.8012 |
| Бытовая техника | 0.8372 |
| Музыкальные инструменты | 0.8435 |
| Продукты питания | 0.8503 |
| Аптека | 0.8534 |
| Бытовая химия | 0.8807 |
| Товары для животных | 0.8922 |
| Детские товары | 0.8925 |
| Хобби и творчество | 0.9281 |

Тот же кластер слабых категорий (обувь/ювелирка/одежда/галантерея),
что и во всех предыдущих FRIDA/BERTA чекпоинтах проекта — см. CLAUDE.md
"4 категории держат 77% оставшегося запаса роста".

## Сборка сабмита

`kaggle_kernels/frida/full_sft/full-sft-submit-build/frida_full_sft_submit_build.py`
(CPU-only, `maxamsecret`): грузит `frida-full-sft-w/best/{encoder.pt,head.pt}`
(сырые `state_dict`, НЕ HF-директория), строит `T5EncoderModel` из
конфига `ai-forever/FRIDA`, загружает finetuned-веса (`strict=True`,
проверка на mismatch), кастует в bf16, сохраняет через
`model.save_pretrained(..., safe_serialization=True)` (не сырой
`safetensors.save_file` на state_dict напрямую — у T5 `shared.weight`/
`encoder.embed_tokens.weight` — один и тот же tied-параметр, это ловушка).
Голову сохраняет отдельным `classifier.pt` (bf16, `torch.save`) — та же
конвенция, что у всех прежних FRIDA-сабмитов. Никакого `peft`/LoRA —
впервые в проекте сабмит не тащит offline wheels для адаптера.

Обслуживает на `PAIR_MAX_LENGTH=180`, `BATCH_SIZE_FIXED=5248`,
`ITEM_MAX_LENGTH=256`.

## Локальная проверка

CPU sanity-check (batch=8, 200 пар, `.venv`) прошёл end-to-end без ошибок:
предсказания в [0,1], без NaN, mean=0.31/std=0.22 — не вырождены. Полный
`test_submission.py` с реальным `batch_size=5248` на CPU падает
(`RuntimeError: not enough memory: 8.16GB`) — ожидаемо, здесь нет CUDA;
фиксированные под H100 батчи в этом проекте всегда проверялись только на
реальном GPU/тест-стенде, не на CPU (см. `submits/frida/lora/length_256/tpu_human_256_epoch4/README.md`).
