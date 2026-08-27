# Postmortem: `loss=NaN` при обучении 20 FRIDA per-category LoRA на TPU

## Краткий вывод

NaN был не отдельной ошибкой `BCEWithLogitsLoss` и не доказанной нестабильностью FRIDA в bf16. Старый пайплайн соединял уже обученный encoder со случайной линейной головой и обучал её с чрезмерным LR. Один нечисловой gradient/update на отдельном TPU rank мог через distributed all-reduce повредить параметры и AdamW state всех реплик. На следующем batch уже повреждённая модель выдавала `loss=NaN`.

Первый фикс — warmup головы, уменьшение её LR-множителя и локальный gradient clipping — снизил агрессивность, но сохранил случайную голову, высокий head LR и отсутствие replica-synchronous проверки градиентов. Поэтому NaN воспроизводился на тех же категориях и шагах.

Фактически все 20 экспертов удалось обучить после перехода на merged format-v4 backbone с уже обученной глубокой головой, снижения head LR до `1e-5`, LoRA LR до `5e-5`, batch до 16 на TPU core и добавления 48-step preflight полного optimizer path. V12 replica-synchronous guard был полезной защитой, но не является доказанной окончательной причиной успеха: до успешного format-v4 запуска его убрали из основного update path.

---

## 1. Ошибка глобально

### 1.1. В каком месте возникает NaN

Один тренировочный шаг состоит из четырёх стадий:

1. `forward` вычисляет logits и loss;
2. `backward` вычисляет локальные gradients каждого TPU rank;
3. `xm.optimizer_step()` делает cross-rank all-reduce;
4. AdamW обновляет параметры и свои моменты.

Старый код проверял только loss перед `backward()`:

```python
if not torch.isfinite(loss).item():
    raise TrainingDivergedError(...)

loss.backward()
torch.nn.utils.clip_grad_norm_(trainable, 1.0)
xm.optimizer_step(optimizer)
```

Проверка отвечала только на вопрос «параметры ещё позволяют выполнить текущий forward?». Она не проверяла, конечны ли gradients после текущего `backward()` и безопасен ли следующий distributed update.

Если один rank получал `NaN/Inf` в gradient:

- локальный clipping не являлся общим решением восьми реплик;
- `xm.optimizer_step()` включал плохой gradient в all-reduce;
- AdamW state и параметры всех ranks могли стать нечисловыми;
- только следующий forward обнаруживал уже совершившееся повреждение как `loss=NaN`.

Поэтому факт «loss стал NaN до текущего backward» не доказывает, что первопричина находится в текущем forward. Причиной мог быть optimizer update предыдущего шага.

Важно: обычное среднее конечных gradients само по себе не превращается в NaN. Опасность — нечисловой gradient хотя бы на одном rank либо, существенно менее вероятно, overflow во время reduction. Формулировка «конечные локальные gradients стали NaN после усреднения» была неточной.

### 1.2. Почему режим был численно агрессивным

Старый `percat-lora` использовал:

| Параметр | Значение |
|---|---:|
| Head | случайный `LinearHead` |
| LoRA LR | `2e-4` |
| Исходный head LR multiplier | `5.0` |
| Исходный head LR | `1e-3` |
| Head warmup | отсутствовал |
| Gradient clipping | отсутствовал |
| Batch | 24 на TPU core, global 192 |

Таким образом, случайная голова сразу получала большой AdamW update поверх уже обученного FRIDA encoder. В первом полном прогоне 17 из 20 экспертов ушли в NaN; полностью выжили только категории 07, 10 и 14.

### 1.3. Почему первый фикс не решил проблему

Первый набор изменений:

- `HEAD_LR_MULTIPLIER: 5.0 → 3.0`;
- warmup головы 20 шагов;
- `clip_grad_norm_=1.0`;
- немедленный `TrainingDivergedError` при нечисловом loss.

Но head LR всё ещё составлял `2e-4 × 3 = 6e-4`, голова оставалась случайной, а clipping выполнялся до distributed reduction. В следующих независимых запусках 8 из первых 10 категорий снова расходились. Шаги повторялись детерминированно: cat00 около шага 40, cat01 около шага 2, cat02 около шага 5, cat03 около шага 32.

Fail-fast не исправлял обучение — он лишь переставал тратить оставшиеся шаги категории после обнаружения NaN.

### 1.4. Почему отдельный diagnostic kernel не дал ответ

`percat-lora-nan-diagnose` запускался на одном TPU core и сознательно исключал cross-rank all-reduce. При этом он исходил из неверного причинного вывода: раз текущий loss нечисловой до `backward()`, значит виноват текущий bf16 forward.

Но текущий forward мог использовать параметры, повреждённые предыдущим distributed update. Single-core тест не воспроизводил этот участок.

Дополнительно `register_forward_hook` на каждом T5-блоке вызывал `.any()`/`.item()` и host synchronization. На XLA это дробило fused graph и провоцировало множественные компиляции. Такой per-block diagnostic подходит для GPU, но не для production TPU loop.

---

## 2. Конкретный пример: cat00

### 2.1. Где находилась ошибка

В старом `kaggle_kernels/frida/percat/percat-lora/frida_percat_lora.py`:

- `LinearHead` создавался заново;
- `reset_specialist_()` снова случайно инициализировал его для каждой категории;
- loss проверялся до `backward()`;
- gradients клипались локально;
- затем `xm.optimizer_step()` выполнял all-reduce и AdamW update без проверки конечности global update.

У cat00 loss стабильно становился NaN примерно на шаге 40. Это означает не обязательно, что batch 40 был плох для forward. Более вероятная последовательность:

1. шаг 39 начал forward с конечными параметрами;
2. loss шага 39 был конечным и прошёл проверку;
3. backward одного rank дал плохой gradient либо update оказался слишком агрессивным;
4. `xm.optimizer_step()` распространил повреждение;
5. forward шага 40 получил уже нечисловые параметры и вернул `loss=NaN`.

Именно поэтому warmup и локальный clipping могли сдвинуть момент расхождения, но не гарантировали устойчивость.

### 2.2. V12: правильная distributed-защита

В commit `22e4e1f` появился `_safe_optimizer_update()`:

1. все ranks голосовали за конечность loss;
2. выполнялся явный `xm.reduce_gradients(optimizer)`;
3. norm вычислялся и клипался уже на усреднённых gradients;
4. все ranks принимали одинаковое решение — обновлять параметры или пропустить batch;
5. scheduler двигался только при применённом update;
6. preflight выполнял реальные optimizer updates до известной границы cat00.

Это закрывало distributed failure mode, но версия делала host sync почти на каждом шаге и была непрактично медленной. В commit `aff07b4` её заменили on-device голосованием: если один rank сообщал нечисловой loss/gradient norm, gradients обнулялись на всех ranks до обычного `xm.optimizer_step()`.

Это была корректная защитная эволюция, но не финальный production path.

### 2.3. Как cat00 и остальные категории реально удалось обучить

Успешный format-v4 kernel изменил саму численную постановку:

| Параметр | Старый фиксированный режим | Успешный format-v4 |
|---|---:|---:|
| Backbone/head | merged encoder + случайный linear head | merged format-v4 + обученный deep head |
| Head | Linear | `1024 → 512 → 128` |
| LoRA LR | `2e-4` | `5e-5` |
| Head LR | `6e-4` | `1e-5` |
| Batch/core | 24 | 16 |
| LoRA dropout | 0.15 | 0.10 |
| Preflight | forward/backward smoke | 48 настоящих optimizer updates |

`reset_specialist_()` теперь:

- сбрасывал LoRA в identity (`lora_B=0`);
- восстанавливал уже обученную general deep head;
- не создавал случайную классификационную границу заново.

Head LR снизился в 60 раз относительно первого «исправленного» варианта и в 100 раз относительно исходного `1e-3`. LoRA LR снизился в четыре раза. Это устранило агрессивный старт, а 48-step preflight проходил дальше исторического cat00 failure на шаге 40 до начала дорогого 20-category цикла.

Фактический output содержит 20 `last`-чекпоинтов после полной эпохи, по 875–908 шагов. Они затем успешно использовались в submission inference. При этом `val_pr_auc=null`, поэтому доказан факт отсутствия фатального NaN и работоспособность весов, но не улучшение качества. Поле `bad_updates=0` также нельзя считать отдельным доказательством: в финальном V13 production update replica guard уже был удалён.

---

## 3. Отдельный путь появления NaN: повреждённый resume

17 старых `_last`-чекпоинтов были сохранены уже после расхождения. При resume cat00 такой checkpoint давал NaN на первом шаге независимо от нового scheduler.

Это исправили отдельно:

- проверка `torch.isfinite()` для каждого LoRA/head tensor до копирования на XLA;
- отказ от повреждённого checkpoint;
- fallback на чистую инициализацию;
- корректный category-level `try/except`, чтобы один специалист не завершал весь kernel.

Этот баг объясняет мгновенный NaN при resume, но не исходные fresh-init расхождения.

---

## 4. Что считать настоящим решением

### Подтверждённое практикой

1. Не обучать случайную голову с LR порядка `1e-3` поверх сильного backbone.
2. Начинать с уже обученной deep head.
3. Использовать LoRA LR `5e-5`, head LR `1e-5`.
4. Держать batch не выше проверенного `16/core` для этого backbone.
5. Перед полным циклом прогонять минимум 48 настоящих optimizer updates на реальных данных.
6. Проверять checkpoint tensors на конечность до resume.

### Защитный механизм, который стоит сохранить при следующей реализации

Replica-synchronous finite-gradient vote до AdamW update полезен, если он реализован без per-step host sync и не ломает XLA compilation. Один плохой rank должен приводить к одинаковому решению на всех восьми ranks.

### Что не было первопричиной NaN

- host RAM OOM и `BrokenProcessPool` — отдельный класс отказа;
- `num_workers`, pre-fork model loading и `gc.collect()` влияли на RAM, но не создавали loss divergence;
- per-block bf16 hooks не доказали нестабильность FRIDA backbone;
- `BCEWithLogitsLoss` лишь распространял уже появившиеся NaN, а не создавал их сам.

## Итоговая формулировка

> NaN возникал из-за слишком агрессивного optimizer regime для случайной классификационной головы и отсутствия надёжной cross-rank проверки gradients перед TPU all-reduce. Первый warmup/clip фикс был недостаточным. Реально обучение стабилизировалось после перехода на уже обученную format-v4 deep head, снижения head LR до `1e-5`, LoRA LR до `5e-5`, уменьшения batch и обязательного optimizer preflight. Distributed finite-gradient guard был полезной страховкой, но не единственным и не доказанным финальным источником успеха.
