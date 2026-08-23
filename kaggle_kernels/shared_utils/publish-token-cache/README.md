# publish-token-cache

Публикует ~9GB предтокенизированного memmap-кэша (результат полного 11.48M
прогона continue-train) как отдельный датасет, чтобы больше **никогда не платить
~9.5 ч токенизации**.

- **Kaggle:** https://www.kaggle.com/code/maxamsecret/publish-token-cache
- **Тип:** script · **GPU:** нет (CPU) · **интернет:** вкл

## Суть
Полный continue-train прогон упёрся в 12-часовой лимит сессии, но его
`/kaggle/working` (включая `token_cache_llm_continue/` memmap) сохранился как
output кернела. Этот кернел монтирует тот output read-only через `kernel_sources`
и публикует кэш как датасет — целиком внутри инфраструктуры Kaggle (гонять ~9GB
через локальную машину не вышло: повторяющиеся SSL-обрывы на самом большом файле).

Требует Kaggle Secrets (`KAGGLE_USERNAME`/`KAGGLE_KEY`), т.к. дёргает Datasets
API изнутри кернела.

## Что использовалось
- **kernel_sources:** `maxamsecret/lora-llm-continue-train` (его output-кэш).

## Результат
Датасет `maxamsecret/ozon-lora-llm-token-cache` — «Ozon LoRA continue-train
token cache (11.48M rows)»: train 11 480 303 + val 73 131, `max_length=128`,
поля `*_input_ids/attention_mask/token_type_ids/labels/category.bin`. Теперь
`PRETOKENIZED_CACHE_DIR` в [`../lora-llm-continue-train`](../lora-llm-continue-train)
может указывать на его mount и пропускать загрузку `matches_llm`/`items` и всю
токенизацию.
