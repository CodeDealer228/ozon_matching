# MiniLM-baseline ветка

`cross-encoder/ms-marco-MiniLM-L-12-v2` — исходный baseline проекта, до
FRIDA/BERTA-пивота. LoRA SFT (val 0.5282, лидерборд 0.265) — самый первый
рабочий результат, задокументирован в корневом `CLAUDE.md` → "Architecture".

| Кернел | Примечание |
|---|---|
| `cross-encoder-sft-pipline` | основной тренировочный ноутбук/пайплайн |
| `lora-llm-continue-train` | продолжение SFT на `matches_llm.parquet` (0.53229, был лучшим до FRIDA) |
| `lora-timing-dryrun` | офлайн-замер таймингов в полном масштабе |
| `merge-lora` | LoRA-адаптер → мерженые веса |
| `validate-baseline-vs-lora` | сравнение baseline vs LoRA на val |

Сабмиты этой линии — `submits/minilm/`.
