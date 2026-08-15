# frida-lora-sft-epoch2

Продолжение LoRA SFT `ai-forever/FRIDA` — резюмирует чекпоинт
[`../frida-lora-sft-human-only-10-epochs`](../frida-lora-sft-human-only-10-epochs)
(`maxamsecret/frida-lora`, val macro PR-AUC 0.652 @ 60% эпохи 1) и продолжает
обучение на том же человеческом train-сплите ещё на одну эпоху. Запущен с
**другого Kaggle-аккаунта** (`rays0n`) — отдельная недельная GPU-квота.

- **Kaggle:** https://www.kaggle.com/code/rays0n/frida-lora-sft-epoch2
- **Тип:** script · **GPU:** 1×T4 · **интернет:** вкл · **публичный**, владелец `rays0n`

## Суть
Архитектура и данные byte-идентичны предыдущему прогону (см. его README) —
те же LoRA-таргеты, тот же `PAIR_MAX_LENGTH=128`, тот же item-level кэш
(`maxamsecret/frida-tokenized-items-human`, повторная публикация того же
кэша, что и `frida-item-tokens-human`). Новое в этом кернеле:
- **Резюм адаптера**: `_find_resume_adapter_dir()` находит
  `adapter_config.json` среди примонтированных датасетов
  (`maxamsecret/frida-lora`) и грузит его через
  `PeftModel.from_pretrained(..., is_trainable=True)` — с явным
  `is_trainable=True`, обходя задокументированную в CLAUDE.md ловушку (без
  этого флага PEFT молча замораживает все LoRA-веса, без ошибки).
- **Сохранение состояния оптимизатора** — по прямому запросу («Сделай
  сохранение оптимизатора в этой новой эпохе»): `save_checkpoint` теперь
  пишет `optimizer.pt`/`scheduler.pt`/`train_state.json` рядом с адаптером,
  а резюм умеет их подхватывать, если они есть. У чекпоинта
  `maxamsecret/frida-lora` их нет (он от прошлого скрипта) — поэтому этот
  прогон стартует со свежим оптимизатором и LR-расписанием, а не
  байт-в-байт продолжением предыдущего.
- bf16 + `attn_implementation="eager"` + autocast **внутри** `forward()` —
  фикс из [`../frida-nan-diagnose`](../frida-nan-diagnose) применён с
  самого начала, не переоткрывался заново.

## Что использовалось
- **Датасеты:** `maxamsecret/frida-lora` (адаптер для резюма),
  `maxamsecret/frida-tokenized-items-human` (токен-кэш),
  `maxamsecret/ozon-ecup-2026-matching`.
- **Модель:** `ai-forever/FRIDA` + LoRA-адаптер сверху.

## Результат
1 эпоха = 1143 шага, чекпоинт каждые 229 (~5/эпоху), эффективный batch=256
(micro=32 × grad_accum=8):

| Шаг | Прогресс | train macro PR-AUC | val macro PR-AUC |
|---|---|---|---|
| 229 | 20.0% | 0.642 | 0.663 |
| 458 | 40.1% | 0.675 | **0.674** ← лучший, сохранён |

Снова упёрся в 12-часовой лимит сессии (`CANCEL_ACKNOWLEDGED`) — оборвался
в середине eval-прохода к чекпоинту 60%. **Узкое место — не обучение, а
eval**: полный проход по 73,131 val-парам при `EVAL_BATCH_SIZE=64` занимает
~2ч (1143 eval-батча × ~6.5с/it), и это отдельный проход на каждый чекпоинт
(~5/эпоху) — грубо говоря, eval съедает бóльшую часть 12-часовой сессии,
оставляя мало времени на собственно обучение. Стоит поднять
`EVAL_MICRO_BATCH_SIZE` (см. как это уже сделано отдельным auto-probe в
[`../berta-lora-sft-human-only-20-epochs`](../berta-lora-sft-human-only-20-epochs))
или считать val на сабсэмпле между промежуточными чекпоинтами.

Лучший чекпоинт (**0.67397 @ 40%**, `lora_frida_sft/best/`: адаптер +
classifier + optimizer + scheduler + tokenizer) восстановлен из output
кернела и лежит в [`output/lora_frida_sft/best/`](output/lora_frida_sft/best)
— **пока не опубликован как Kaggle-датасет**.

> Проверено байт-в-байт (`sha256` адаптера и `classifier.pt`) против
> `submits/matching-frida_lora_adapter-submit/`: сабмит,
> прошедший тестовый стенд, использует **не этот чекпоинт**, а
> предыдущий — эпоху 1 (0.65174 @ 60%), см.
> [`../frida-lora-sft-human-only-10-epochs`](../frida-lora-sft-human-only-10-epochs#подтверждённый-сабмит).
> Этот чекпоинт (0.674) на тестовом стенде ещё не пробовался.

0.674 против **0.5282** у продакшен-чекпоинта MiniLM (эпоха 7, полностью
обученного) — при том что FRIDA прошла едва ли полторы эпохи, и обе эти
эпохи останавливались принудительно по лимиту сессии, а не по сходимости.
