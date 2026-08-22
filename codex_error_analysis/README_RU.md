# Codex starter pack — аудит hard FP/FN BERTA и FRIDA

Этот пакет нужен для автоматизированного **качественного аудита ошибок**, а не для
обучения моделей.

## Что именно проверяем

Для каждой из четырёх групп:

- BERTA FP
- BERTA FN
- FRIDA FP
- FRIDA FN

ошибки ранжируются от самых жёстких к менее жёстким.

По умолчанию берём по 25 случайных ошибок из каждого severity band:

- ranks 1–50
- ranks 51–150
- ranks 151–400
- ranks 401–1000

Итого целевой размер: **400 LLM judgments**.

Это специально сделано не как простой top-N: так можно проверить гипотезу,
что на самом верху hard errors много `label noise / strange business rules`,
а глубже становится больше настоящих model failures.

Если после первого аудита нужно уменьшить шум в оценках, можно увеличить:

`config/audit_config.json` → `"sample_per_band": 50`

Тогда получится до ~800 assignments, а первый band 1–50 будет просмотрен полностью.

## Откуда взять source artifact

Сначала запусти наш CPU notebook:

`ecup_berta_frida_fpfn_exact_input_explorer_v4.ipynb`

Его output содержит:

`all_fp_fn_full_attributes_and_exact_inputs.parquet`

Скопируй этот файл в:

`artifacts/input/all_fp_fn_full_attributes_and_exact_inputs.parquet`

Можно использовать JSONL-версию с тем же basename.

Важно: **не подсовывай `top_errors_with_items.parquet`**.
Для severity bands 151–400 / 401–1000 нужен полный FP/FN artifact.

## Самый простой запуск через Codex

1. Распакуй этот zip в отдельную папку.
2. Положи source artifact в `artifacts/input/`.
3. Открой папку как проект в Codex.
4. Вставь содержимое `MASTER_PROMPT_RU.txt`.
5. Дай Codex пройти pipeline end-to-end.

Main model в config: `gpt-5.6`, reasoning high.
Error-auditor subagents: `gpt-5.6-terra`, reasoning high.
Одновременно максимум 5.

## Что делает pipeline

### 1. Immutable stratified-by-hardness sample

```bash
python scripts/prepare_audit.py
python scripts/validate_audit_sample.py
```

Outputs:

- `artifacts/prepared/audit_sample.jsonl`
- `artifacts/prepared/audit_sample.parquet`
- `artifacts/prepared/coverage.csv`
- `artifacts/prepared/manifest.json`

Каждый audit unit содержит:
- focal model/error type;
- severity rank + band;
- BERTA / FRIDA scores;
- full names and full attributes обоих items;
- focal model exact visible input для item1/item2;
- other model visible input;
- truncation counters, если они есть в source artifact.

### 2. Batches

```bash
python scripts/make_batches.py
```

Default batch size = 20, поэтому 400 judgments → 20 batches.

Батчи специально stable-shuffle'ятся, чтобы один subagent не видел только одну категорию
или только один severity band.

### 3. Subagent classification

Каждый `error_auditor` обязан сначала решить:

**Насколько вообще правдоподобен human label?**

И только затем:
**если label выглядит правильным, в чём ошибка focal-модели?**

Primary classes:

- `likely_label_issue`
- `ambiguous_business_rule`
- `truncation_related`
- `critical_attribute_mismatch`
- `product_type_confusion`
- `semantic_or_alias_failure`
- `variant_policy_failure`
- `missing_information`
- `other_model_error`

Смотри точные определения в `TAXONOMY_RU.md`.

### 4. Grounded validation

```bash
python scripts/validate_batch_outputs.py
```

Validator проверяет не только JSON enums.

Он также требует, чтобы `evidence_original` были **реальными exact substrings**
из исходных name/attributes, а `evidence_model_input` — exact substrings из focal
model input. Это сильно уменьшает шанс LLM hallucinations.

### 5. Aggregation

```bash
python scripts/aggregate_audit.py
```

Главные outputs:

- `outputs/classified_audit.parquet`
- `outputs/summary_by_primary_class.csv`
- `outputs/summary_by_severity.csv`
- `outputs/summary_by_category.csv`
- `outputs/summary_by_subtype.csv`
- `outputs/hypothesis_label_noise_by_depth.csv`
- `outputs/manual_review_queue.html`
- `outputs/FINAL_REPORT.md`

`hypothesis_label_noise_by_depth.csv` специально сравнивает label-issue rate между
верхним и нижним severity bands.

## Почему severity rank лучше raw score

BERTA score и FRIDA score нельзя считать одинаково откалиброванными.

Поэтому для cross-model анализа hard severity используется:

`severity_rank`

внутри конкретной группы `model × FP/FN`, а не утверждение вроде
`BERTA=0.9` эквивалентно `FRIDA=0.9`.

Raw scores всё равно сохраняются как дополнительный контекст.

## Что я бы смотрел в FINAL_REPORT в первую очередь

1. `likely_label_issue` rate по severity bands.
2. `critical_attribute_mismatch` rate для BERTA FP vs FRIDA FP.
3. Реальный `truncation_related` rate, а не просто факт наличия truncation.
4. Top secondary subtypes: model number, article, size, pack count, product type...
5. Доля случаев, где другая модель была правильна при threshold 0.5.
6. Категории, где ошибка специфична для одной архитектуры.
7. Low-confidence / ambiguous cases из `manual_review_queue.html`.

После этого уже можно решать, какие ошибки лечить:
targeted hard negatives, structured mismatch features, preprocessing/truncation или
category-specific logic.


## Зависимости

Если используешь `.parquet` source:

```bash
python -m pip install -r requirements.txt
```

Если не хочется ставить `pyarrow`, можно положить JSONL-версию
`all_fp_fn_full_attributes_and_exact_inputs.jsonl`; подготовка и aggregation смогут
работать без parquet output.
