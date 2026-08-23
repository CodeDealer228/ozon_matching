# validate-baseline-vs-lora

Честное сравнение бейзлайна и LoRA на **одном** val-сплите по **исправленной**
метрике (per-category macro PR-AUC).

- **Kaggle:** https://www.kaggle.com/code/maxamsecret/ozon-baseline-vs-lora-val-macro-pr-auc
- **Тип:** script · **GPU:** 1×T4 · **интернет:** вкл

## Суть
Раньше метрика в ноутбуке считалась неверно (macro по классам match/non-match,
а не по 20 категориям товаров) и давала несопоставимое `0.76823`. Этот кернел
пересчитывает оба варианта на идентичном seed-42 val-сплите:
1. **baseline_no_lora** — frozen `ms-marco-MiniLM-L-12-v2` → `[CLS]` →
   `StandardScaler` → `LogisticRegression` (LogReg обучается только на train);
2. **lora_finetuned** — дообученный энкодер (`ozon-lora-minilm-finetuned`,
   `best/`) + обученная голова, прямой прогон.

Логирует `val_fingerprint` (sha256 отсортированных пар) и все метрики в mlflow;
эмбеддинги baseline кэшируются, чтобы не пересчитывать.

## Что использовалось
- **Датасеты:** `ozon-ecup-2026-matching`, `ozon-lora-minilm-finetuned`.
- **Модель:** `cross-encoder/ms-marco-MiniLM-L-12-v2` + PEFT-адаптер.

## Результат (`output/val_comparison_results.json`)
| Вариант | macro PR-AUC (per-category) | overall AP |
|---|---|---|
| baseline_no_lora | **0.3710** | 0.4497 |
| lora_finetuned | **0.5282** | 0.6150 |
| **Δ** | **+0.1571** | — |

73 131 val-пара, все 20 категорий с обоими классами. Прогон ~23 мин (вкл.
загрузку моделей). Вывод: прирост LoRA реальный, а не артефакт разной метрики.
