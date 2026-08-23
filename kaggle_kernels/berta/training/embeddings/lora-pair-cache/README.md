# berta-lora-pair-cache

Кэширует pooled-представление (768-мерное, mean-pooling по
`last_hidden_state`) LoRA-адаптированной BERTA для **каждой** из 365,654
человеческих пар (292,523 train + 73,131 val, канонический seed-42 сплит)
— только forward, без градиентов. Не эмбеддинг товара — BERTA здесь
cross-encoder (`[CLS] item1 [SEP] item2 [SEP]`, совместное внимание), так
что закэшированный вектор уже функция конкретной пары, между разными
парами не переиспользуется.

Цель: отвязать эксперименты с архитектурой головы (MLP vs Linear,
активация, глубина, dropout) от прохода энкодера. После кэша обучение
новой головы — чистая матричная арифметика на фиксированной матрице
365,654×768, без GPU-энкодера в цикле.

- **Kaggle:** https://www.kaggle.com/code/maxamsecret/berta-lora-pair-embedding-cache
  (slug определился по заголовку, не по `id` в metadata — разошлось с тем,
  что было в `kernel-metadata.json` при первом пуше, поправлено)
- **Тип:** script · **GPU:** 2×T4 · **интернет:** вкл

## Время (посчитано, не оценено)
Из реального лога `berta-lora-sft-human-only-20-epochs`: один полный
forward-only eval-проход по всем 73,131 val-парам (batch=512 под
DataParallel на 2×T4) занял **14:43 (883с)**, стабильно на трёх разных
чекпоинтах того прогона (5.86-5.88с/it × 143 батча). Все 365,654 пары —
ровно **5× это**. Этот кернел должен уложиться в тот же порядок величины
или быстрее (без DataParallel scatter/gather, без классификационной
головы в forward-проходе) — то есть **порядка часа**, не больше
десятой доли одной 12-часовой сессии.

## Почему 2 независимых процесса, не DataParallel и не DDP-обёртка
Чистый инференс — нигде нет `backward()`, значит синхронизировать
градиенты между репликами нечего. `DistributedDataParallel`-обёртка
добавила бы служебные накладные расходы (регистрация hook'ов на
градиенты) без всякой пользы. `DataParallel` тоже сработал бы (обучение
BERTA уже доказало, что DataParallel + autocast-внутри-forward безопасны
— см. `BertaCrossEncoderClassifier` там и
[`../frida-nan-diagnose`](../frida-nan-diagnose) почему важно именно
"внутри"), но его scatter/replicate/gather на каждый батч — чистые
накладные расходы для embarrassingly-parallel задачи, где GPU вообще не
должны ничем обмениваться до самого конца.

Вместо этого — `torch.multiprocessing.spawn` поднимает 2 полностью
независимых процесса, каждый на своей GPU, с собственной полной копией
модели, обрабатывающих свою непересекающуюся половину данных, каждый
пишет свой шард на диск. `dist.init_process_group` нужен только для
дешёвого barrier перед тем как rank 0 сливает два шарда — не для
синхронизации градиентов. Полезный побочный эффект: каждый rank — это
по-настоящему отдельный OS-процесс со своим Python main thread, так что
конкретная ловушка DataParallel+autocast из `frida-nan-diagnose`
(autocast, открытый в вызывающем потоке, не пробрасывается в потоки
DataParallel-реплик) здесь структурно невозможна вообще — нет никаких
рабочих потоков, только рабочие процессы, у каждого свой нормальный
autocast.

Последовательная загрузка модели по rank (сначала rank 0, потом rank 1,
с barrier между ними) — тот же урок, что уже применялся в других
мульти-GPU кернелах проекта (не материализовывать один и тот же чекпоинт
одновременно).

## Что использовалось
- **Датасеты:** `maxamsecret/ozon-ecup-2026-matching` (`items_human.parquet`,
  `matches.parquet`), `maxamsecret/berta-lora-sft-5epoch` (чекпоинт,
  `lora_berta_sft/checkpoint/adapter/`).
- **Модель:** `sergeyzh/BERTA` + LoRA-адаптер (`is_trainable=False` —
  чистый инференс, без классификационной головы: кэшируется представление
  ДО головы).

## Формат вывода (`/kaggle/working/berta_lora_pair_cache/`)
```
train/embeddings.npy    (292523, 768) float16
train/id1.npy, id2.npy  int64
train/label.npy         float32
train/category_id.npy   int16
val/...                 та же структура, 73131 строк
cache_meta.json         category_vocab, batch_size_used, source_checkpoint, ...
```
**Датасет из кернела не публикуется** (явная инструкция) — забрать через
`kaggle kernels output` и опубликовать вручную, как и с чекпоинтами.

## Запуск
```bash
kaggle kernels push -p kaggle_kernels/berta/training/embeddings/lora-pair-cache
kaggle kernels status maxamsecret/berta-lora-pair-embedding-cache
kaggle kernels output maxamsecret/berta-lora-pair-embedding-cache -p kaggle_kernels/berta/training/embeddings/lora-pair-cache/output
```

## v1 упал — `mp.spawn` → `torch.distributed.run`
Первая версия (`torch.multiprocessing.spawn`) упала: rank 0 умер без
Python-трейсбека прямо во время `PeftModel.from_pretrained(...)` (похоже
на OOM-kill — см. CLAUDE.md "a real Linux OOM-kill isn't a catchable
MemoryError"), rank 1 словил обрыв TCPStore-соединения. Корень не
локализован до конца — переключились на `torch.distributed.run
--standalone --nproc_per_node=N`, тот же механизм запуска, что уже
доказанно работает в этом окружении в
[`../qwen3-bi-encoder-lora-sft`](../qwen3-bi-encoder-lora-sft) (ветка
`bi-encoder`) — не гадание, а известно рабочий вариант.
