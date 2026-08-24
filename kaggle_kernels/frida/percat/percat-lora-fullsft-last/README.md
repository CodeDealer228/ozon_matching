# FRIDA per-category LoRA от merged formatting-v4

TPU v5e-8-кернел последовательно обучает 20 независимых LoRA и глубоких
голов — по одной на категорию. Backbone — merged
`matching-frida_format_adapter_v4_best_200-submit.zip`; проверяются SHA256 и
размер трёх реально загружаемых файлов. Входы — только output
`maxamsecret/frida-tokenize-items-v4`: без prefix/category, длина пары 224.
Human train/val использует тот же item-disjoint 70/30 split, что и backbone.

Основной профиль: 8 TPU-ядер, batch 16 на ядро (global 128), eval batch 32,
`num_workers=0`, 28 CPU-потоков на rank, AdamW, bf16 encoder + fp32 LoRA/head.
Используется до 100 000 LLM-пар на категорию плюс human train; ожидаемое время
около 5–6 часов при жёстком останове через 8 ч 15 мин.

Перед обучением выполняются train/eval smoke-test и 48 настоящих обновлений
optimizer preflight. Heartbeat каждые 8 шагов не даёт Kaggle завершить долгий
тихий XLA-этап; обновление выполняет `xm.optimizer_step`.
`catNN_best` содержит только веса и
val-метрики; `catNN_last` каждые 15 минут сохраняет также AdamW, scheduler и
точный batch cursor. Для продолжения нужно подключить output предыдущего запуска:
совместимый `run_contract.json` будет найден и проверен автоматически.
