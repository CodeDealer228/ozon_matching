# FRIDA — LoRA SFT (человеческие данные)

Прогрессия LoRA-обучения на `matches.parquet` (human-only), от первого
прогона до финальной 4-эпохной версии. `continue/` — отдельная ветка,
продолжающая обучение на LLM-разметке (`matches_llm.parquet`).

| Кернел | Примечание |
|---|---|
| `lora-sft-human-only-10-epochs` | первый реальный прогон (10 эпох план) |
| `lora-sft-epoch2` | резюм с epoch1, TPU, аккаунт `maximmarkovbb` |
| `lora-sft-epoch3` | резюм с epoch2 |
| `lora-v2-epoch` | v2-переупорядочивание атрибутов, отдельная линия |
| `tpu-human-sft` | 4-эпохная human-only версия, `PAIR_MAX_LENGTH=256` — источник лучших "чисто человеческих" чекпоинтов (val 0.7660) |
| `continue/lora-continue-2ep` | продолжение на LLM-разметке, 2 эпохи |
| `continue/llm-tpu-sft` | почасовые чекпоинты human+LLM continue-SFT (источник `tpu_human_llm_*` сабмитов, лучший тест LoRA-линии — 0.469) |

Итоговые мерженые веса всей LoRA-линии в проекте были заменены full-SFT
подходом (`../full_sft/`, `../full_sft_continue/`) — он и даёт актуальный
лучший результат.
