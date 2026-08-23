# FRIDA — первый полный (не LoRA) файнтюн

Первый прогон, обучающий **все** 823M параметров FRIDA (не LoRA-дельту) —
Adafactor + category-weighted focal loss. Val 0.7582, тест 0.481 (был
лучшим результатом проекта до `../full_sft_continue/experiments/c-llm3m`,
0.487).

| Директория | Примечание |
|---|---|
| `full-sft/` | тренировочный кернел (TPU, `frida_full_sft.py`) — полная история 6 реальных крашей в докстринге файла |
| `full-sft-submit-build/` | CPU-сборщик сабмита (веса → bf16 `save_pretrained` → zip) |

Чекпоинт (`maxamsecret/frida-full-sft-w`) — стартовая точка для
`../full_sft_continue/` (все 5 экспериментов резюмируются от него).
