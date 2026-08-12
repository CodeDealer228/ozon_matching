# val-per-category-metrics

Подробная разбивка конкурсной метрики (macro PR-AUC) **по каждой из 20
категорий** для текущего решения — смёрженного LoRA-кросс-энкодера.

- **Kaggle:** https://www.kaggle.com/code/maxamsecret/ozon-val-per-category-pr-auc-lora
- **Тип:** script · **GPU:** нет (CPU) · **интернет:** выкл

## Суть
Один макро-скор `0.5282` ничего не говорит о том, **где** модель слабая. Этот
кернел прогоняет merged-LoRA на каноническом seed-42 val-сплите и для каждой
категории считает: AP, `pos_rate` (пол, ниже которого AP не читается — по
категориям 0.07..0.58, поэтому сырые AP между категориями несравнимы),
`roc_auc`, AP наивного name-Jaccard скорера («насколько категория трудная» без
forward pass), bootstrap-95% CI (в категории ~3.5K пар), лучший F1 / precision@recall,
и статистику расхода 128-токенного окна (энкодер — англ. WordPiece, русский текст
крошится, обрезка бьёт по категориям неравномерно).

CPU-only и намеренно RAM-лёгкий: `items_human.parquet` вместо 4.1GB
`items.parquet`, semi-join к нужным id до строковых операций, hash-join +
явный sort (без `maintain_order`).

## Что использовалось
- **Датасеты:** `ozon-ecup-2026-matching`, `ozon-lora-merged-minilm`.

## Результат
Выводы (`/kaggle/working`): `val_scores.parquet` (73 131 строк:
`id1,id2,label,category,prob,naive_jaccard`), `per_category_metrics.csv`
(строка на категорию), `summary.json`. Per-pair скоры сохраняются, чтобы любую
доп. нарезку делать оффлайн без нового forward pass.

> Статус на момент оформления: кернел **выполняется** (`RUNNING`) — результат
> ещё не скачан в `output/`. Забрать после завершения:
> `kaggle kernels output maxamsecret/ozon-val-per-category-pr-auc-lora -p kaggle_kernels/val-per-category-metrics/output`.
