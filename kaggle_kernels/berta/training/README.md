# BERTA — обучение

| Директория | Кол-во | Состав |
|---|---|---|
| `tokenize/` | 5 | токенизация items/LLM-разметки, кэш-пробы, подготовка continue/weakcat-данных |
| `sft/` | 6 | сам SFT (LoRA, human-only и continue на LLM-разметке), включая `llm-tpu-sft` — источник лучшего BERTA-сабмита (`submits/berta/lora/tpu_lora`, тест 0.47) |
| `embeddings/` | 5 | извлечение/кэш эмбеддингов (для CatBoost-голов в `../catboost/` и `submits/qwen/berta_catboost`) |
| `hardneg/` | 2 | сэмплинг hard negatives (balanced/uniform) |
| `misc/` | 2 | `attr-augment` (аугментация атрибутов), `merge-lora` (мерж адаптера в базовые веса) |
