# BERTA сабмиты

| Директория | Что внутри |
|---|---|
| `lora/` | прямые LoRA-сабмиты (`tpu_lora` — **лучший результат BERTA, тест 0.47**; `tpu_lora_after_llm` — тот же прогон позже по эпохе, val реально просел; `human_continue_384`) |
| `catboost/` | LoRA-эмбеддинг + CatBoost-голова поверх него (5 вариантов: base/256_len/pca32/top_final/top_final_v2) — ни один не обошёл чистый классификатор из `lora/tpu_lora` |

Полный разбор происхождения `tpu_lora` (восстановлен из упавшего рана,
byte-verified) — `reports/berta_tpu_lora_provenance/README.md`.
