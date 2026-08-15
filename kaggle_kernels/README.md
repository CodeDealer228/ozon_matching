# kaggle_kernels

Kaggle-кернелы проекта Ozon E-CUP 2026 (matching). Каждый кернел — отдельный
скрипт/ноутбук, который гоняется на инфраструктуре Kaggle (там полный
`items.parquet` и бесплатный GPU) и складывает результат либо в свой `output/`,
либо в отдельный Kaggle-датасет.

Все кернелы приватные, владелец `maxamsecret`. Ссылка вида
`https://www.kaggle.com/code/maxamsecret/<slug>`.

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
| [giga3b-quantize-benchmark](giga3b-quantize-benchmark) | проба bi-encoder Giga-3B (корректность/качество/скорость/int8) | T4/P100 | `RUNNING` (pending) |

### Bi-encoder ветка (тиммейт, аккаунт `grigoriyyy`)
| Кернел | Суть | GPU | Итог |
|---|---|---|---|
| [qwen3-bi-encoder-lora-sft](qwen3-bi-encoder-lora-sft) | Qwen3-Embedding-0.6B + LoRA + симметричная MLP-голова | 2×T4 (DDP) | **macro_ap_mlp 0.726** за 1 сессию — лучший val-результат проекта на сегодня (но на другом, item-disjoint 5-fold сплите — не 1:1 сравнимо с остальным) |

## Датасеты, которые производят/потребляют кернелы
- `maxamsecret/ozon-ecup-2026-matching` — исходные данные (вход почти всюду).
- `maxamsecret/ozon-lora-minilm-finetuned` — PEFT-адаптер (выход cross-encoder-sft, вход merge/continue-train).
- `maxamsecret/ozon-lora-merged-minilm` — смёрженные веса (выход merge-lora, вход timing/val-per-category).
- `maxamsecret/ozon-lora-llm-token-cache` — токен-кэш 11.48M (выход publish-token-cache, вход continue-train).
- `maxamsecret/ozon-lora-submission-code` — код сабмита для оффлайн timing-прогона.
