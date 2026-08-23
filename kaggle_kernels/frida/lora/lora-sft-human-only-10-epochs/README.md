# frida-lora-sft-human-only-10-epochs

Первый заход LoRA SFT энкодера `ai-forever/FRIDA` (T5EncoderModel, 823M) как
кросс-энкодера — разведка новой, много большей архитектуры взамен/в
дополнение к MiniLM. Обучение только на человеческой разметке
(`matches.parquet`): forward pass FRIDA ~в 25 раз дороже MiniLM, поэтому
двухстадийный рецепт MiniLM (human SFT → continue на 11.19M LLM-меток) не
влезает в недельную GPU-квоту для модели такого размера.

- **Kaggle:** https://www.kaggle.com/code/maxamsecret/frida-lora-sft-human-only-10-epochs
- **Тип:** script · **GPU:** 1×T4 · **интернет:** вкл · приватный, владелец `maxamsecret`

## Суть
- LoRA `r=16, alpha=32, dropout=0.10`, `target_modules=["q","k","v","o","wi_0","wi_1","wo"]`
  — реальные имена модулей T5Attention (q/k/v/o) и gated-GELU FFN
  (wi_0/wi_1/wo), проверено через `named_modules()` на загруженном
  чекпоинте, не угадано по аналогии с BERT. 11,206,656 обучаемых параметров
  (1.34% от 834,607,872).
- Данные — не сырые тексты, а готовый **item-level** токен-кэш
  `maxamsecret/frida-item-tokens-human` (711,304 уникальных товара,
  `"paraphrase: " + текст`, спецтокены bos/eos уже внутри, `max_length=256`).
  Конкатенация пары происходит на лету в `collate_fn` — без вызова
  токенизатора и без per-chunk join пар к `items.parquet` (это был
  задокументированный ~9.5ч-из-12ч боттлнек `lora-llm-continue-train`).
- `PAIR_MAX_LENGTH=128` — намеренный компромисс цена/квота, НЕ безлоссовый
  бюджет (тот — 384: p99 реальных пар, режет только 0.97%). При 128
  обрезается ~94% пар, но поле `Category` (доказано 100%-идентично у обоих
  товаров любой пары, см. CLAUDE.md) вырезается из каждого товара перед
  обрезкой — доля товаров, не дотягивающих вообще ни до какого контента
  `Attributes`, падает с 0.44% до 0.09%.
- Планировалось `EPOCHS=10`; реально в этом кернеле `EPOCHS=1` — более
  ранняя версия на fp16 NaN'ила на первом же чекпоинте (step 229/11430, ещё
  до конца warmup). Корень найден и пофикшен в
  [`../frida-nan-diagnose`](../frida-nan-diagnose) (bf16 вместо fp16 +
  `attn_implementation="eager"`); этот прогон проверяет фикс на одной
  эпохе, прежде чем тратить квоту на все 10.

## Что использовалось
- **Датасеты:** `maxamsecret/ozon-ecup-2026-matching` (только
  `matches.parquet`, для канонического сплита), `maxamsecret/frida-item-tokens-human`
  (токен-кэш товаров).
- **Модель:** `ai-forever/FRIDA` (823M, тянется по интернету).

## Результат
1143 шага (1 эпоха), чекпоинт каждые 229 шагов (~5/эпоху):

| Шаг | Прогресс | train macro PR-AUC | val macro PR-AUC |
|---|---|---|---|
| 229 | 20.0% | 0.352 | **0.526** |
| 458 | 40.1% | 0.554 | **0.622** |
| 687 | 60.1% | 0.635 | **0.652** |

Упёрся в жёсткий 12-часовой лимит сессии Kaggle (`CANCEL_ACKNOWLEDGED`) до
чекпоинта 80%. Лучший чекпоинт (val macro PR-AUC 0.65174 @ 60.1%)
опубликован как датасет `maxamsecret/frida-lora`
(`lora_frida_sft/best/`: adapter + classifier + tokenizer, **без**
optimizer/scheduler state — этот скрипт их ещё не сохранял). Продолжение
обучения — [`../frida-lora-sft-epoch2`](../frida-lora-sft-epoch2) (уже на
аккаунте `rays0n`).

Уже на 60% одной эпохи — **0.652 против 0.5282** у продакшен-чекпоинта
MiniLM (эпоха 7, полностью обученного, за много эпох). Основание считать
переход на FRIDA перспективным.

## Подтверждённый сабмит
`matching-frida_lora_adapter-submit.zip` (внешняя ссылка, скачан локально в
`submits/matching-frida_lora_adapter-submit/`) — **этот** чекпоинт,
проверено побайтово (`sha256` `adapter_model.safetensors` и
`classifier.pt` совпадают с датасетом `maxamsecret/frida-lora` один в
один). Прошёл проверку на тестовом стенде судейства (реальный H100).

Инженерно это **не** тот путь, что у `submits/lora` (MiniLM): адаптер там
не мёржится в базу (осознанное решение) — zip тащит полный `frida-base` в
**bf16** (не fp32 — иначе не влезть в 5GB) отдельно от
`frida-lora-adapter/`, плюс `wheels/` с ~26 офлайн-колёсами (`peft` и его
транзитивные зависимости — их нет в образе `odsai/ecup26-matching-baseline:1.0`,
в отличие от `submits/lora`, который мёржит именно чтобы обойтись без
`peft`). Установка пакетов — `pip install --no-index --target
{tempfile.gettempdir()}` во время `run.py`, не в site-packages: реальный
прогон на стенде поймал `OSError: [Errno 30] Read-only file system` и на
системный site-packages, и на `/root/.local` (pip'овский fallback) —
только `/tmp` оказался гарантированно писуемым в судейском контейнере.

**Препроцессинг пар — не формула из CLAUDE.md** («Text preprocessing must
byte-match...», та формула — контракт конкретно `submits/lora`/MiniLM).
Здесь свой контракт: item-level токенизация со спецтокенами (bos/eos),
вырезание span'а `Category:` (доказано 0%-сигнальное поле) и
fair-share-обрезка пары до 128 токенов — байт-в-байт то же, что
`frida-lora-sft-*` кернелы делают при обучении. Если заводить отдельный
`submits/frida_lora_adapter/` в репозитории (сейчас код сабмита есть только
в скачанном локально zip, не в git) — переносить эту логику, а не формулу
MiniLM.
