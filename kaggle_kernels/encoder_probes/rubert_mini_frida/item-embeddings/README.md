# rubert-mini-frida-item-embeddings

Считает per-**ITEM** эмбеддинг (mean-pooled, НЕ per-pair) для каждого из
711,304 товаров — `sergeyzh/rubert-mini-frida` как настоящий bi-encoder:
один независимый вектор на товар, сравнимый косинусом с любым другим —
в отличие от `sergeyzh/BERTA` в этом же проекте, которая используется как
cross-encoder (пара кодируется совместно, per-item вектора не существует).
Без LoRA-адаптера — базовые веса как есть.

- **Kaggle:** https://www.kaggle.com/code/maxamsecret/rubert-mini-frida-item-embeddings-2xt4
  (токен-кэш: [maxamsecret/rubert-mini-frida-item-tokenized](https://www.kaggle.com/datasets/maxamsecret/rubert-mini-frida-item-tokenized))
- **Тип:** script · **GPU:** 2×T4 · **интернет:** вкл

## Зачем
[`../retrieval-recall-hard-negative-mining`](../retrieval-recall-hard-negative-mining)
измерил recall@k TF-IDF-косинуса при попытке воспроизвести retrieval
организаторов — слабо (recall@10 от 0.015 до 0.35 по категориям, далеко
от целевых ~0.9). Чистое лексическое пересечение не ловит перефразировки/
синонимы/порядок слов; обученная embedding-модель должна быть намного
ближе. Этот кернел готовит эмбеддинги, на которых можно повторить ту же
recall@k-проверку — уже с семантическим, не лексическим сигналом.

## Разделение на два кернела
`../berta-lora-pair-cache` словил SIGKILL (OOM) на токенизации того же по
размеру набора товаров на T4. Токенизация здесь — отдельный кернел на
TPU-инстансе (ради RAM, не ради TPU-чипов, см. его README) — этот кернел
просто монтирует уже готовый токен-кэш и делает чистый GPU-инференс, без
конкуренции за CPU/RAM с токенизацией.

## Батч — реально большой, не угаданный
У `berta-lora-pair-cache` VRAM-пробник упёрся в мой собственный
`BATCH_SIZE_CAP=1024`, измерив там всего **5.19GB/GPU** — то есть запас
ещё оставался, потолок был искусственный, не реальный VRAM-лимит. Здесь
`BATCH_SIZE_CAP=8192` — пробник должен найти настоящий потолок, а не
упереться в придуманный.

## Механизм запуска
`torch.distributed.run --standalone --nproc_per_node=N` (не `mp.spawn` —
уже падал в этом проекте без внятного трейсбека; не DataParallel/DDP-
обёртка — чистый инференс, нечего синхронизировать). Каждый ранг берёт
свою непересекающуюся половину товаров — в отличие от `berta-lora-pair-cache`,
здесь рангам не нужно ничего друг у друга (пары не участвуют), поэтому
нет необходимости в токенизации-на-rank0-и-расшаривании.

## Что использовалось
- **Датасет:** `maxamsecret/rubert-mini-frida-item-tokenized` (токен-кэш,
  711,304×256 — размер файлов сверен байт-в-байт с ожидаемым перед
  запуском).
- **Модель:** `sergeyzh/rubert-mini-frida` (тянется по интернету).

## Что на выходе
`embeddings.npy` (N×hidden_dim, float16) + `ids.npy` + `meta.json`
(hidden_dim определяется в рантайме из конфига модели, не угадан
заранее). **Датасет из кернела не публикуется** — забрать через
`kaggle kernels output`, опубликовать вручную.

## Запуск
```bash
kaggle kernels push -p kaggle_kernels/encoder_probes/rubert_mini_frida/item-embeddings
kaggle kernels status maxamsecret/rubert-mini-frida-item-embeddings-2xt4
kaggle kernels output maxamsecret/rubert-mini-frida-item-embeddings-2xt4 -p kaggle_kernels/encoder_probes/rubert_mini_frida/item-embeddings/output
```
