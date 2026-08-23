# rubert-mini-frida-tokenize

Токенизация всех 711,304 товаров `items_human.parquet` под
`sergeyzh/rubert-mini-frida` — **отдельный кернел от вычисления
эмбеддингов** ([`../rubert-mini-frida-item-embeddings`](../rubert-mini-frida-item-embeddings)),
намеренно запущенный на **TPU-инстансе Kaggle** — не ради самих TPU-чипов
(здесь чистая CPU-токенизация, torch вообще не используется), а ради
намного большего host RAM, которым TPU-инстансы располагают.

- **Kaggle:** https://www.kaggle.com/code/maxamsecret/rubert-mini-frida-item-tokenize-tpu-big-ram
- **Тип:** script · **TPU v5e-8** (только ради RAM) · **интернет:** вкл

## Зачем отдельным кернелом и именно на TPU
[`../berta-lora-pair-cache`](../berta-lora-pair-cache) уже словил
**SIGKILL (OOM)** токенизируя этот же по размеру набор товаров на
T4-инстансе — после ~1.8ч, одним нечанкованным вызовом `tokenizer()`,
да ещё дублируя работу на двух ранках сразу (see его README, "v3 OOM
fix"). Здесь решаем не только чанкованием (оно тоже есть, как вторая
линия защиты), а самой машиной — RAM на TPU-инстансе Kaggle на порядок
больше, чем на T4. Плюс разделение на два кернела означает, что
GPU-инференс эмбеддингов не соревнуется за CPU/RAM с токенизацией вообще.

## Формула текста
Канонические бюджеты CLAUDE.md (Name 128 / Category 64 / Attributes 384 /
итого 640), префикс `"paraphrase: "` — тот же, что уже используется для
FRIDA и BERTA (`sergeyzh/BERTA`, тот же автор/линия дистилляции, что и у
`rubert-mini-frida`). **Category здесь не исключена** — в отличие от
частных по-категорийных пробников в этом проекте, этот кэш накрывает весь
каталог сразу (все 20 категорий), где category — не константа.

## Что на выходе
`ids.bin` / `input_ids.bin` / `attention_mask.bin` / `cache_meta.json` —
тот же memmap-формат, что уже используют `frida-item-tokens-human` и
другие токен-кэши в проекте. **Датасет из кернела не публикуется** — забрать
через `kaggle kernels output` и опубликовать вручную, чтобы
`../rubert-mini-frida-item-embeddings` мог примонтировать его как
`dataset_sources`.

## Запуск
```bash
kaggle kernels push -p kaggle_kernels/encoder_probes/rubert_mini_frida/tokenize
kaggle kernels status maxamsecret/rubert-mini-frida-item-tokenize-tpu-big-ram
kaggle kernels output maxamsecret/rubert-mini-frida-item-tokenize-tpu-big-ram -p kaggle_kernels/encoder_probes/rubert_mini_frida/tokenize/output
```
