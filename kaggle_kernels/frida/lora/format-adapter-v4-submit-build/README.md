# FRIDA format-adapter v4 — CPU submit builder

CPU kernel собирает один H100 submission zip для validated `best` LoRA из
`rays0n/frida-format-adapter-v4`. Из приватного input dataset с уже собранным
`continue_c_llm3m_2ep/last` archive он извлекает c-last encoder/head,
накладывает и сливает adapter, сохраняет bf16 encoder и head. В archive кладётся
`utils2.py`-контракт v4 как `src/utils.py`:
без prefix/category, `PAIR_MAX_LENGTH=188`, `BATCH_SIZE_FIXED=5248`.

На выходе только `matching-frida_format_adapter_v4_best_188-submit.zip`.
