# Происхождение `matching-berta_tpu_lora-submit.zip` (тест-скор 0.47)

Восстановлено и **проверено по хешам**, а не по памяти. Из
`submits/matching-berta_tpu_lora-submit.zip` извлечены адаптер и
классификатор, их sha256 сверен с локально сохранённой копией
восстановленного чекпоинта тренировочного рана (`tpu_e9/` — скачан через
`kaggle kernels output` ещё во время исходной работы над BERTA-TPU, до
того как появился описанный ниже фикс краша):

| Файл (внутри zip) | sha256 | Совпадение |
|---|---|---|
| `models/berta-lora-adapter/adapter_model.safetensors` | `26b294123b2445fa7bb46e0c36c2ee5ad259c218ea80e357661f49729800cd34` | ✅ побайтово совпадает с `tpu_e9/lora_berta_tpu_sft/checkpoint/adapter/adapter_model.safetensors` |
| `classifier.pt` | `70ac231c64ad2f70ac73592afbce76c3ee9f0cb804ecb16e8b0f4258325d43fa` | ✅ побайтово совпадает с `tpu_e9/lora_berta_tpu_sft/checkpoint/classifier.pt` |

Это подтверждает с полной уверенностью, какой именно тренировочный ран
дал сданные веса — не "скорее всего этот", а точно.

## Точная последовательность кернелов

1. **`rays0n/berta-llm-continue-train`** (уже существовавший чекпоинт, часть
   более ранней линии BERTA — см. `kaggle_kernels/README.md`) → PEFT-адаптер,
   `global_step=2002`, `val_macro_pr_auc=0.7254629058736792`.
2. **`kaggle_kernels/berta/training/misc/merge-lora`** (`maxamsecret/berta-merge-lora`) —
   загружает базу `sergeyzh/BERTA` + адаптер из (1), `merge_and_unload()`.
   На выходе: `merged_encoder/` (обычный `BertModel`, `peft` не нужен) +
   `classifier.pt` (СТАРАЯ голова классификатора из (1), на этом шаге
   переносится без изменений — не переобучается здесь) + `merge_meta.json`.
   Подтверждено напрямую из собственного лога тренировочного рана:
   `{'base_model': 'sergeyzh/BERTA', 'hidden_dim': 768, 'source_checkpoint':
   'rays0n/berta-llm-continue-train', 'source_global_step': 2002,
   'source_val_macro_pr_auc': 0.7254629058736792}`.
3. **`kaggle_kernels/berta/training/tokenize/tokenize-llm-256`**
   (`maxamsecret/berta-tokenize-llm-256`) — TPU-кэш токенов по товарам,
   `human_scope` (711,304 товаров) + `llm_scope` (12,384,610 товаров),
   `ITEM_MAX_LENGTH=256`, `PREFIX="paraphrase: "`. (Этот кернел раньше в
   проекте перезапускался один раз из-за бага с prefix — здесь
   использован именно исправленный ран, что подтверждается путём монтирования
   в логе тренировки.)
4. **`kaggle_kernels/berta/training/sft/llm-tpu-sft`** (`maxamsecret/berta-llm-tpu-sft`)
   — TPU v5e-8, `torch_xla`. Строит **свежую** LoRA (`get_peft_model`, не
   резюм) поверх мёрженной базы из (2); классификатор **warm-started** от
   `classifier.pt` из (2) (дальше дообучается, не заморожен). Обучен на
   **полном** `matches_llm.parquet` (11,187,780 пар, без сэмплирования) в
   смеси с полным человеческим train-сплитом (292,523) = 11,480,303
   объединённых строк.
   - `PAIR_MAX_LENGTH=256`, LoRA `r=16, lora_alpha=32, lora_dropout=0.15`,
     `target_modules=["key","query","dense","value"]` (взято напрямую из
     сданного `adapter_config.json`).
   - `per_core_train_batch=64 × world_size=8 = 512` эффективный батч,
     `steps_per_epoch_global=22422`, `warmup=2242` (10%),
     `max_train_seconds=28800` (потолок 8ч).
   - **Этот ран упал** (`concurrent.futures.process.BrokenProcessPool`).
     Причина найдена и исправлена в более поздней сессии (см.
     `kaggle_kernels/README.md` → раздел BERTA): все 8 форкнутых TPU-реплик
     независимо перезагружали и конкатенировали полный кэш товаров и
     `matches_llm.parquet`, что эксгаустило RAM хоста прямо на границе
     лишней, не согласованной с расписанием второй эпохи. На валидность
     ИМЕННО ЭТОГО чекпоинта это не влияет — см. примечание ниже.

## Сданный чекпоинт — это чекпоинт на 50% одной эпохи

У упавшего рана был ровно один успешный чекпоинт/eval перед смертью: на
шаге **`global_step=11211`** — ровно 50% от 22,422-шаговой эпохи
(`VAL_EVERY_FRACTION=0.5` → `checkpoint_interval=11211`) —
**`val_macro_pr_auc=0.7087279672868713`**. Логи обучения по шагам
продолжаются и после этой точки примерно до шага 22400 (LR полностью
затух, то есть обучение реально дошло до ~100% эпохи), но чекпоинт на
границе эпохи (100%) не успел сохраниться до краша. Так что чекпоинт на
отметке 50% — единственный, который вообще попал на диск, и, соответственно,
единственный, который можно было восстановить через `kaggle kernels
output` после отмены сессии — и это именно то, что было сдано.

## Шаги сборки (в этом репозитории, не на Kaggle)

5. Восстановленный чекпоинт скопирован в `submits/berta_tpu_lora/`:
   `models/berta-lora-adapter/` (из `adapter/` в (4), без изменений),
   `classifier.pt` (из (4), без изменений, остаётся fp32 — тренировочный
   скрипт никогда не приводит классификатор к bf16, только замороженную
   базу), `models/berta-base/` (bf16-конвертация `merged_encoder/` из
   (2)), плюс `wheels/` (офлайн `peft` + транзитивные зависимости,
   позаимствованы из `submits/berta_catboost/wheels/`), `src/utils.py` /
   `run.py` / `metadata.json` (контракт инференса BERTA cross-encoder,
   `PAIR_MAX_LENGTH=256`, совпадает с тренировкой).
6. `python scripts/build_submission_zip.py --submission-dir
   submits/berta_tpu_lora` → `submits/matching-berta_tpu_lora-submit.zip`
   — **именно этот файл получил скор 0.47.**

## Одна находка, которую стоит обозначить перед разговором про val/тест

`best_val_macro_pr_auc` в `train_state.json` — это **бегущий максимум
метрики**, а не гарантия, что именно СОХРАНЁННЫЕ веса его достигли:
`save_checkpoint()` в `berta_llm_tpu_sft.py` вызывается безусловно на
каждом интервале чекпоинта с теми весами, что есть ПРЯМО СЕЙЧАС, вместе с
ИСТОРИЧЕСКИ лучшим значением метрики — веса с того шага, который реально
дал этот лучший скор, заново не сохраняются. Для ЭТОГО конкретного
сданного чекпоинта проблемы нет (это единственное событие чекпоинта за
весь ран, так что веса и метрика реально относятся к одному и тому же
шагу) — но это настоящая ловушка для любого ДРУГОГО чекпоинта, взятого из
рана с более чем одним интервалом чекпоинта (например, из завершённого
рана с чекпоинтами и на 50%, и на 100%), и напрямую касается
`berta_tpu_lora_after_llm`.
