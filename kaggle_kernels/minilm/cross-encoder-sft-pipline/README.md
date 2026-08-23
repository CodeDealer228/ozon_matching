# cross-encoder-sft-pipline

Базовое обучение нашего решения — LoRA SFT кросс-энкодера, которое дало
чекпоинт `0.5282` val / `0.265` leaderboard.

- **Kaggle:** https://www.kaggle.com/code/maxamsecret/cross-encoder-sft-pipline
- **Тип:** notebook · **GPU:** 2×T4 (DataParallel) · **интернет:** вкл

## Суть
Дообучение `cross-encoder/ms-marco-MiniLM-L-12-v2` под задачу матчинга:
LoRA (`r=16, alpha=32, dropout=0.10`, `target_modules=["query","key","value"]`)
+ голова `nn.Linear(hidden, 1)` на `[CLS]`, end-to-end `BCEWithLogitsLoss(pos_weight)`.
RAM-экономная memmap-токенизация (тексты режутся по формуле из CLAUDE.md,
`attributes` — сырой срез без `json.loads`), канонический seed-42 сплит
(292 523 train / 73 131 val).

## Что использовалось
- **Датасет:** `maxamsecret/ozon-ecup-2026-matching` (`items.parquet`, `matches.parquet`).
- **Модель:** `cross-encoder/ms-marco-MiniLM-L-12-v2` (33M).

## Результат
Early stopping на эпохе 7. Лучший чекпоинт по val macro PR-AUC (**per-category**,
исправленная метрика) = **0.5282**, leaderboard **0.265** — против бейзлайна
0.3710 / 0.1333. Адаптер опубликован как датасет
`maxamsecret/ozon-lora-minilm-finetuned` (папка `best/`), позже смёржен в базу
(см. [`../merge-lora`](../merge-lora)).

> Актуальная (правленая в этой сессии) версия ноутбука лежит в репозитории:
> [`notebooks/cross-encoder-sft-pipline (3).ipynb`](../../notebooks) — там
> исправлена метрика на per-category, добавлены авто-подбор батча и mlflow.
> В этой папке трекается только README; `pulled/` (склон с Kaggle) —
> в `.gitignore`.
