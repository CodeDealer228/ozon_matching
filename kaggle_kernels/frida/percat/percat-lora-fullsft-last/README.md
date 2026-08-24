# FRIDA per-category LoRA от `continue_c_last_188`

TPU v5e-8-кернел последовательно обучает 20 независимых LoRA и глубоких
голов — по одной на категорию. Исходная модель берётся из
`matching-frida_continue_c_last_188-submit.zip`; проверяются SHA256 и размер
трёх реально загружаемых файлов, поэтому перепаковка ZIP не меняет контракт.
Входы — только `maxamsecret/frida-tokenize-items-v3`, длина пары 188.

Основной профиль: 8 TPU-ядер, batch 16 на ядро (global 128), eval batch 32,
`num_workers=0`, 28 CPU-потоков на rank, AdamW, bf16 encoder + fp32 LoRA/head.
Используется до 100 000 LLM-пар на категорию плюс human train; ожидаемое время
около 5–6 часов при жёстком останове через 8 ч 15 мин.

Перед обучением выполняются train/eval smoke-test и 96 настоящих обновлений
optimizer preflight. Проверки loss и градиентов синхронизированы на TPU без
host-sync на каждом шаге; обновление выполняет `xm.optimizer_step`.
`catNN_best` содержит только веса и
val-метрики; `catNN_last` каждые 15 минут сохраняет также AdamW, scheduler и
точный batch cursor. Для продолжения нужно подключить output предыдущего запуска:
совместимый `run_contract.json` будет найден и проверен автоматически.
