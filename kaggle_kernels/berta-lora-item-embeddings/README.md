# berta-lora-item-embeddings

Per-**ITEM** эмбеддинг (bi-encoder режим: один товар за раз, `[CLS] item
[SEP]`) через **текущий лучший чекпоинт BERTA+LoRA**
(`maxamsecret/berta-lora-sft-5epoch`, val macro PR-AUC **0.7111**) — не
сырая BERTA, а адаптер, реально видевший разметку задачи. В отличие от
[`../berta-lora-pair-cache`](../berta-lora-pair-cache) (тот же чекпоинт,
но как cross-encoder — пара кодируется совместно, per-item вектора не
существует), здесь LoRA-веса используются в режиме, для которого их
никто явно не обучал. Гипотеза: раз LoRA видела реальную supervision
задачи (в отличие от generic `sergeyzh/rubert-mini-frida`), её признаки
могут нести больше task-relevant сигнала для retrieval — проверяется
через recall@k в кернеле-преемнике (по образцу
[`../rubert-mini-frida-recall-mining`](../rubert-mini-frida-recall-mining),
у которой macro recall@10=0.2755 — лучше TF-IDF, но далеко от цели ~0.9),
не предполагается заранее.

- **Kaggle:** ещё не запушен
- **Тип:** script · **GPU:** 2×T4 · **интернет:** вкл (нужен для базовых
  весов `sergeyzh/BERTA`)

## Токен-кэш уже есть — новый TPU-кернел не нужен
`maxamsecret/berta-lora-sft-5epoch` уже содержит полный item-token-кэш —
`berta_item_tokens/{ids,input_ids,attention_mask}.bin`. Сверено байт-в-байт
перед тем, как писать код: `attention_mask.bin` ровно 711,304×128 байт
(int8), `input_ids.bin` ровно 711,304×128×4 байт (int32), `ids.bin` ровно
711,304×8 байт (int64) — тот же охват и `ITEM_MAX_LENGTH=128`, что и
везде у BERTA в проекте. Токенизирован **без спецтокенов**
(`add_special_tokens=False` — та же конвенция, что и в item-кэше
`berta-lora-pair-cache`, где CLS/SEP собираются в `collate_fn` под
конкретную пару). Bi-encoder-режим здесь требует своей рамки
`[CLS] + content + [SEP]` — собирается заново из уже готового сырого
контента, **без повторной токенизации**.

Category по-прежнему вырезается из контента (`_find_token_pattern_starts`
+ `build_compact_ids`, портировано один в один из `berta-lora-pair-cache`)
— не потому что она неинформативна для retrieval между категориями (это
не так), а для консистентности с тем, что LoRA реально видела при
дообучении: pair assembly там всегда вырезает Category ещё до энкодера,
так что подать её сейчас — значит скормить модели формат, на котором эти
веса не тренировались.

## Уроки berta-lora-pair-cache, применены сразу
- Пин версий peft/transformers/huggingface-hub (`--no-deps
  --force-reinstall`) — там ловили `ImportError: EmbeddingParallel` от
  непинованного `pip install peft`.
- `torch.distributed.run`, не `mp.spawn` (там падал без трейсбека) и не
  DataParallel/DDP-обёртка (чистый инференс).
- Последовательная загрузка чекпоинта по рангам.
- `BATCH_SIZE_CAP=8192` вместо фактических 1024 — там пробник упёрся в
  придуманный потолок, реального (5.19GB/GPU было далеко от ~11GB target)
  не нашёл.
- Токенизация не нужна вообще (кэш готов) — в отличие от pair-cache, где
  пара может ссылаться на любые два товара и обоим рангам был нужен один
  и тот же общий словарь, здесь каждый ранг берёт свою половину товаров
  независимо — как и в
  [`../rubert-mini-frida-item-embeddings`](../rubert-mini-frida-item-embeddings).

## Что использовалось
- **Датасет:** `maxamsecret/berta-lora-sft-5epoch` (токен-кэш + адаптер,
  оба нужны).
- **Модель:** `sergeyzh/BERTA` + LoRA-адаптер (`is_trainable=False`).

## Что на выходе
`embeddings.npy` (711,304×hidden_dim, float16) + `ids.npy` + `meta.json`.
**Датасет из кернела не публикуется** — забрать через `kaggle kernels
output`, опубликовать вручную.

## Запуск
```bash
kaggle kernels push -p kaggle_kernels/berta-lora-item-embeddings
kaggle kernels status maxamsecret/berta-lora-item-embeddings
kaggle kernels output maxamsecret/berta-lora-item-embeddings -p kaggle_kernels/berta-lora-item-embeddings/output
```

## Заметка о восстановлении
Этот кернел был написан в предыдущей сессии (другое окружение), но так и
не закоммичен до переключения окружения — пересоздан по памяти сессии
1:1 в новой сессии, ещё не запускался ни разу.
