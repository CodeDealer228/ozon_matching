# FRIDA — продолжение full-SFT (актуальный лучший результат проекта)

Резюм `../full_sft/full-sft`'s чекпоинта (val 0.7582, тест 0.481) с
варьируемыми гиперпараметрами. **`experiments/c-llm3m` — лучший результат
проекта на тест-стенде (0.487)**.

| Директория | Примечание |
|---|---|
| `full-sft-continue/` | общий код (`frida_full_sft_continue.py`) — НЕ пушится сам, база для 5 копий в `experiments/` |
| `full-sft-continue-submit-build/` | CPU-сборщик сабмитов для всех 5 экспериментов сразу (content-based поиск чекпоинта по `experiment` в `train_state.json`) |
| `experiments/` | 5 копий с разными `EXPERIMENT_NAME`/`LLM_TRAIN_SUBSAMPLE_SIZE`/`HUMAN_WEIGHT`/`FOCAL_GAMMA`/`EPOCHS` — a-baseline, b-gamma3, c-llm3m (лучший), d-llmfull-rebal, e-llmfull-w2 |

Полная матрица экспериментов, resume-механизм (двухрежимный:
cross-experiment vs same-experiment), контракт чекпоинтов
(тихий/валидированный) — `full-sft-continue/README.md`. Тест-скоры и
per-category разбивки по каждому эксперименту —
`submits/frida/sft_continue/README.md`.
