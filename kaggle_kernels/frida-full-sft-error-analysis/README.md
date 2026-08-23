# FRIDA full-SFT: baseline и анализ ошибок

Validation-only kernel для этапов 0–1 из
`synthetic_research/SYNTHETIC_PIPELINE_v2.md`.

Он:

- byte-for-byte проверяет canonical item-disjoint split;
- загружает и проверяет по SHA-256 exact submission archive текущей FRIDA;
- импортирует preprocessing прямо из этого архива;
- считает untouched validation без обучения;
- сохраняет полные predictions и AP по каждой категории;
- сохраняет rank-based FP/FN candidates с raw items, exact visible token input
  и truncation diagnostics.

Старые human token/feature caches не подключаются и не используются.

## Перед запуском

В `kernel-metadata.json` указан владелец `maxamsecret`, потому что это FRIDA
lineage. Перед `kaggle kernels push` владелец должен быть явно подтверждён и
CLI credentials должны быть проверены read-only. Само наличие этого metadata
файла не означает разрешение на запуск.

Ожидаемые persistent outputs:

- `preflight.json`
- `split_identity.json`
- `model_identity.json`
- `val_predictions.parquet`
- `val_predictions.npy`
- `baseline_metrics.json`
- `per_category_metrics.csv`
- `frida_error_candidates.jsonl`
- `error_candidate_manifest.json`
- `run_manifest.json`
- `run_status.json`
- небольшие `tpu_rank_*` sidecars с предсказаниями и прогрессом каждого ядра.
