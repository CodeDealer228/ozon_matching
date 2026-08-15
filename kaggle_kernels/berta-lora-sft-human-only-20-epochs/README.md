# berta-lora-sft-human-only-20-epochs

LoRA SFT `sergeyzh/BERTA` (BertModel, 128.3M) как кросс-энкодера — сиблинг
[`../frida-lora-sft-human-only-10-epochs`](../frida-lora-sft-human-only-10-epochs),
тот же рецепт (человеческая разметка, `PAIR_MAX_LENGTH=128`, LoRA
r16/alpha32), адаптированный под настоящую BERT-архитектуру вместо T5.
BERTA — дистиллят FRIDA, в ~6.5 раза меньше (128.3M против 823M) —
достаточно дёшев, чтобы токенизировать прямо внутри скрипта (не отдельным
кернелом+датасетом, как у FRIDA).

- **Kaggle:** https://www.kaggle.com/code/rays0n/berta-lora-sft-human-only-20-epochs
- **Тип:** script · **GPU:** 1×T4 · **интернет:** вкл · **публичный**, владелец `rays0n`

## Суть
Архитектурные отличия от FRIDA, которые здесь важны:
- Настоящий `BertModel` (12 слоёв, hidden=768, WordPiece, vocab=55,083) —
  канонический формат пары `[CLS] item1 [SEP] item2 [SEP]` +
  `token_type_ids`, а не «две самодостаточные последовательности встык»
  (T5-специфичный обход у FRIDA — у T5 нет CLS/SEP/token_type вообще).
  Кэш товаров хранит сырой контент без спецтокенов
  (`add_special_tokens=False`) — CLS/SEP и token_type собираются в
  `collate_fn` уже для конкретной пары.
- **Mean-pooling, не CLS** — по прямому указанию карточки модели BERTA
  («CLS pooling заменён на mean pooling»), единственное намеренное
  отличие дистилляции от FRIDA.
- `target_modules=["query","key","value","dense"]` — реальные имена
  линейных слоёв BERT (`dense` матчит все три:
  `attention.output.dense`/`intermediate.dense`/`output.dense`).
- `EPOCHS=20`, `PAIR_MAX_LENGTH=128` — обе зафиксированы по прямому
  указанию, не подобраны под эту модель отдельно.

Уроки FRIDA применены сразу, не переоткрывались заново: autocast внутри
`forward()` (не снаружи — см.
[`../frida-nan-diagnose`](../frida-nan-diagnose)), явный `del logits, loss`
в NaN-guard'е (иначе застрявший graph ловит OOM на следующем шаге),
`is_trainable=True` на резюме адаптера.

**Чекпоинты и резюм между сессиями — не опция, а требование**: 20 эпох на
этой скорости — несколько 12-часовых сессий. Каждый чекпоинт (~5/эпоху)
пишет адаптер+classifier+optimizer+scheduler+`train_state.json` в
`/kaggle/working` и (если доступны Kaggle Secrets) публикует их как версию
датасета `rays0n/berta-lora-sft-checkpoint` — **в этом прогоне публикация
не сработала** (`[WARN] Kaggle secrets unavailable ... Connection error`),
поэтому единственная копия чекпоинта сейчас —
[`output/lora_berta_sft/checkpoint/`](output/lora_berta_sft/checkpoint) в
этой папке (восстановлено из output кернела, включая optimizer/scheduler
state).

## Что использовалось
- **Датасет:** `maxamsecret/ozon-ecup-2026-matching`
  (`items_human.parquet` + `matches.parquet`).
- **Модель:** `sergeyzh/BERTA` (тянется по интернету).

## Результат
20 эпох = 22,840 шагов (1142/эпоху), эффективный batch=256 (авто-подобранный
micro=256, eval micro=512 — отдельный auto-probe для eval, т.к. forward-only
без backward даёт больше места в VRAM):

| Эпоха | Шаг | val macro PR-AUC | best на тот момент |
|---|---|---|---|
| 1 (20%) | 228 | 0.350 | 0.350 |
| 1 (100%) | 1142 | 0.620 | 0.621 |
| 2 (100%) | 2284 | 0.678 | 0.678 |
| 3 (100%) | 3426 | 0.698 | 0.699 |
| 4 (100%) | 4568 | 0.709 | 0.709 |
| 5 (~88%) | 5016 | 0.710 | **0.7111** |

Упёрся в 12-часовой лимит сессии (`CANCEL_ACKNOWLEDGED`) на шаге
5016/22840 — **22% от запланированных 20 эпох**, прогон ни разу не вышел
на плато (последние два чекпоинта 0.7111→0.7104 — первый лёгкий откат, но
тренд всё ещё растущий). В отличие от FRIDA чекпоинты здесь быстрые (~29
мин на 228 шагов+eval против ~4.2ч у FRIDA за то же число шагов) — модель
меньше, батч больше.

**Уже лучший результат в истории проекта**: 0.7111 против 0.5282 у
продакшен-чекпоинта MiniLM (+0.183, ~35% относительно) и против 0.674 у
частично обученной FRIDA — и это всего 22% через план, без единой полной
сходимости. Резюм оставшихся ~16 эпох из сохранённого чекпоинта — самый
очевидный следующий шаг.

## Возобновление между сессиями
Опубликовать [`output/lora_berta_sft/checkpoint/`](output/lora_berta_sft/checkpoint)
как датасет `rays0n/berta-lora-sft-checkpoint` (или новую версию, если он
уже существует), добавить его в `dataset_sources` кернела и перезапустить —
`_find_checkpoint_dir()` подхватит `train_state.json` автоматически и
продолжит с `global_step=5016`. Чтобы скрипт публиковал чекпоинт
автоматически на будущих запусках, в настройках кернела нужно включить
Kaggle Secrets (Add-ons → Secrets → `KAGGLE_USERNAME`/`KAGGLE_KEY`).
