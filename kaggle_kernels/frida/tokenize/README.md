# FRIDA — токенизация items

Пре-токенизация каталога товаров в memmap-кэш (`ids.bin`/`input_ids.bin`/
`attention_mask.bin`) — один раз на item, а не на пару, что и делает
FRIDA-подход (bi-encoder-style кэш) практичным для cross-encoder-обучения.

| Кернел | Scope | Примечание |
|---|---|---|
| `tokenize-items` | — | самая ранняя версия |
| `tokenize-items-v2` | — | v2-формула символьной обрезки (140/32/529 + переупорядочивание атрибутов) |
| `tokenize-items-cpu` | human+LLM | **актуальная (v3)** — тот же v2-контракт, CPU-only, без GPU/TPU |
| `tokenize-items-human` | ~711K items | ранняя раздельная human-часть |
| `tokenize-items-llm` | ~12.38M items | ранняя раздельная LLM-часть (TPU, для RAM-запаса) |
| `llm-train-val-split` | — | no-leak train/val сплит поверх `matches_llm.parquet` (не токенизация как таковая, но того же прекурсорного назначения) |

Актуальный опубликованный кэш — `maxamsecret/frida-tokenize-items-v3`
(результат `tokenize-items-cpu`). Остальные — историческая линия, оставлены
для провенанса.
