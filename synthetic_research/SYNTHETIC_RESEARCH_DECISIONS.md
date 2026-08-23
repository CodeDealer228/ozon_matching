# Synthetic Research Decision Log

## Назначение документа

Этот документ фиксирует **почему менялась стратегия synthetic data**, что означают варианты A / A2 / B / C, какие идеи были отвергнуты или сохранены и на каких основаниях появился текущий `SYNTHETIC_PIPELINE_v2.md`.

Он нужен, чтобы не восстанавливать весь мыслительный процесс по чатам и комментариям в pull request.

---

## 1. Исходная цель

Нужно добавить hard negative training data для слабых категорий product matching.

Изначальная идея:

1. взять реальный товар из train;
2. найти product-defining attribute;
3. изменить одно значение;
4. считать `(original, synthetic)` negative pair.

Примеры:

```text
iPhone SE 2 256Gb красный
→
iPhone SE 2 512Gb красный
```

```text
джинсы артикул 14881488
→
джинсы артикул 67676767
```

Изначально планировалось получить примерно `1000` synthetic examples на целевую категорию.

Ключевой invariant с самого начала:

> synthetic меняет ровно одну semantic characteristic, но все ее representations в `name` и `attributes` обязаны оставаться синхронизированными.

---

## 2. Предположение pipeline_v1

`pipeline_v1` исходил из того, что удастся найти достаточно надежные правила вида:

```text
category + attribute
→ изменение этого attribute гарантирует negative
```

Candidate attributes:

- color;
- size;
- article/SKU;
- model;
- storage;
- capacity;
- compatibility;
- quantity;
- material.

Планировалось классифицировать правила как:

```text
safe_negative
unsafe_variant
uncertain
```

И использовать для генерации только `safe_negative`.

---

## 3. Почему pipeline_v1 был отвергнут

Первое исследование закончилось главным результатом:

> **Ни одна характеристика не прошла консервативные критерии `safe_negative`.**

Это не означает, что research был бесполезным.

Наоборот, он показал, что исходная абстракция слишком грубая.

В product matching нельзя надежно использовать глобальные правила:

```text
другой цвет
другой размер
другая емкость
другой артикул
```

как универсальный признак different product.

Одинаковое surface difference может:

- означать другой товар в одном product family;
- быть variant-ом того же товара в другом;
- просто коррелировать с negative pair, не являясь decisive difference.

Поэтому нельзя просто ослабить threshold, чтобы искусственно получить `safe_negative`.

Это создало бы noisy labels.

Позднее организаторы уточнили product definition: каждое значение параметра, доступное через переключатель внутри карточки Ozon (размер, цвет и т. п.), считается отдельным товаром. Это позволяет использовать variant dimensions как источник negatives и снимает прежнее опасение, что переключаемый вариант обязательно сохраняет identity.

При этом результат `pipeline_v1` не отменяется. Он проверял, можно ли вывести надежное **глобальное** правило только из `category + attribute`, а не является ли конкретный переключаемый вариант отдельным товаром по правилам соревнования. Human labels, attributes и seller-created variant groups могут быть шумными; продавец способен связать переключателем вообще разные товары. Поэтому новое разъяснение повышает приоритет variant dimensions для C, но не разрешает слепую мутацию любого одноименного поля без train-supported context.

---

## 4. Важное ограничение: semantic characteristic ≠ одно поле

В процессе обсуждения отдельно зафиксировали проблему `name ↔ attributes`.

Одна характеристика часто дублируется:

```text
name:
Apple iPhone 15 256GB Black

attributes:
Объем встроенной памяти = 256 ГБ
```

Если поменять только одно representation:

```text
name = 256GB
attributes = 512 ГБ
```

получается внутренне противоречивый item.

Поэтому единица изменения:

> **одна semantic characteristic, а не один токен и не один JSON field.**

Если characteristic меняется, необходимо согласованно изменить все ее representations внутри item.

Это обязательное правило для C.

---

# 5. Вариант A — identity-preserving rewrite

## Идея

Берется настоящая human negative pair:

```text
(A, B), label = 0
```

После этого B переписывается без изменения identity:

```text
B -> B'
```

Тогда:

```text
(A, B') = negative
```

наследует label от настоящей human pair, если `B'` действительно остается тем же товаром B.

## Почему plain A не стал приоритетом

Обычный paraphrase может дать больше текста, но не обязательно создать более полезный training example.

Если B просто переформулирован и не стал сложнее для модели, сигнал может быть слабым.

Поэтому plain A не выбран как основная ветка.

---

# 6. Вариант A2 — identity-preserving hardening

## Идея

A2 — усиленная версия A.

Берется настоящий negative `(A, B)` и B переписывается так, чтобы:

```text
identity(B') == identity(B)
```

но:

```text
surface_similarity(A, B') > surface_similarity(A, B)
```

Пример:

```text
A:
Фильтр масляный MANN W914/2 для Lada

B:
Масляный фильтр Mahle OC384 ВАЗ

B':
Фильтр масляный Mahle OC384 для ВАЗ
```

## Почему A2 выбран

У A2 два важных плюса:

1. negative label основан на реальной human-negative pair;
2. полезность можно измерить еще до retraining.

Для текущей модели Frida:

```text
old_score = Frida(A, B)
new_score = Frida(A, B')

hardening_delta = new_score - old_score
```

Хороший A2 example должен обычно увеличивать match score модели, оставаясь true negative.

## Главный риск

Qwen может случайно изменить identity B, пытаясь сделать его похожее на A.

Поэтому первый pilot должен быть сильно ограничен:

- разрешено менять только `name`;
- `attributes` immutable;
- factual values должны сохраняться;
- нельзя переносить facts из A;
- подозрительные rewrite rejected.

**Решение: A2 принят как pilot branch.**

---

# 7. Вариант B — conditional rules

## Идея

Вместо глобального:

```text
category + attribute
```

искать более узкое правило:

```text
category
+ product subtype/family
+ semantic attribute
+ context
```

Пример формы:

```text
Автотовары
+ specific part family
+ OEM code
→ different OEM может быть identity-critical
```

## Что полезного в B

B правильно учитывает, что product identity rules локальны, а не глобальны.

Он полезен как **research abstraction** для поиска local critical dimensions.

## Почему B не остается отдельной generation branch

Conditional rule все равно может быть слабым, если найден по корреляции, а не по near-identical human examples.

Если генерировать сразу по каждому plausible conditional rule, можно повторить проблему v1, только на более мелком уровне.

Поэтому полезная часть B переносится в C:

> C должен находить и подтверждать critical dimensions в узком product family/context.

**Решение: B остается research-концепцией, но не отдельным генератором.**

---

# 8. Вариант C — observed minimal-negative patterns

## Первая идея C

Искать real human negatives:

```text
A ≈ B
label = 0
```

где оставшееся различие хорошо интерпретируется.

Пример:

```text
same product family
60Ah vs 70Ah
human negative
```

Это дает evidence, что конкретный semantic difference может быть decisive.

Идея хорошо соответствует реальным FP модели: товары почти одинаковые, но одна узкая характеристика меняет identity.

## Почему C выбран

C основан на human negative behavior, а не на глобальном предположении.

Он специально нацелен на:

> items выглядят почти одинаково, но узкая деталь меняет товар.

Это именно тот тип hard negatives, который reranker/cross-encoder может пропускать.

**Решение: C принят как pilot branch.**

---

# 9. Коррекция C после PR review

Первая формулировка C была слишком строгой.

По сути она означала:

```text
увидели 60Ah vs 70Ah
→ можно генерировать только 60Ah -> 70Ah
```

Это сильно уменьшает coverage и может привести к memorization конкретной пары values.

Новая версия C работает с **critical dimension**, а не с одной observed value pair.

Если доказано:

```text
product_family = batteries
critical_dimension = capacity_Ah
```

то можно создавать несколько synthetic variants:

```text
60Ah -> 65Ah
60Ah -> 70Ah
60Ah -> 75Ah
```

при условии, что replacement values валидны и желательно реально встречаются в human train этого же family.

Тогда модель обучается правилу:

> небольшое изменение по этому semantic dimension может менять identity

а не запоминает конкретно `60` и `70`.

Разъяснение организаторов дополнительно подтверждает, что controlled mutation переключаемого параметра может давать корректный negative. Но pipeline должен либо надежно определить variant family, либо подтвердить dimension на human train: название поля само по себе все еще недостаточно.

---

# 10. Validation-error dataset

Есть специальный dataset:

```text
all_fp_fn_full_attributes_and_exact_inputs.jsonl
```

Внешняя копия:

https://www.kaggle.com/datasets/grigoriyyy/all-fp-and-fn-on-validation

В нем присутствуют:

- human labels;
- categories;
- BERTA и Frida scores/errors;
- полные names и attributes;
- exact model-visible inputs;
- truncation diagnostics.

Этот artifact изменил research strategy: стало можно анализировать **реальные model failure modes**, а не только общие attribute statistics.

Однако он был построен для предыдущей модели/split и теперь является историческим, пока не доказано строгое совпадение identity. Его нельзя автоматически использовать как FP/FN актуальной Frida.

---

# 11. Важная коррекция после PR review: актуальная модель — Frida

В PR было отмечено, что текущая актуальная модель — **Frida**, а не BERTA.

Следовательно, новый research должен опираться на:

- Frida FP/FN taxonomy;
- exact Frida input;
- Frida truncation behavior;
- Frida scores для A2 hardening.

BERTA может использоваться только как исторический/reference comparison.

Целевые weak categories также нужно заново определить по актуальным Frida validation metrics.

После следующего review зафиксирован canonical split актуальной лучшей Frida:

```text
data/splits/matches_item_disjoint_70_30_seed42.parquet
seed = 42
train pairs = 256006
validation pairs = 109648
shared train/validation items = 0
file size = 6192178 bytes
file SHA-256 = 1e3df234d6cdb8685256d5144b25f9a0a3e3a023f1053a0bd5b16c40e3df8d7e
validation fingerprint = 847d04d081432cc081c2149b75271547bf5e25b14f9d8a5804704a5be2a4b3c8
```

Reference submission актуальной Frida:

```text
https://storage.yandexcloud.net/ds-ods/files/submissions/0d0183cf-8864-4331-88be-f78da0c68dd2/8d18ca15/matching-frida_continue_c_llm3m_188-submit.zip
```

Решение: predictions, per-category metrics и FP/FN нужно пересчитать именно на этом split. После пересчета требуется вручную подтвердить, что `critical_attribute_mismatch` остается одним из основных систематических failure modes новой Frida.

---

# 12. Как можно и нельзя использовать validation

Validation разрешено использовать для:

```text
поиска failure modes Frida
классификации ошибок
поиска candidate semantic patterns
разделения semantic errors и truncation errors
```

Validation нельзя использовать как:

```text
source synthetic items
source training pairs
источник validation rewrites для train
прямой источник training examples
```

Правильный flow:

```text
validation error
    ↓
failure-mode hypothesis
    ↓
найти аналогичный evidence в human train
    ↓
генерировать только из human train
```

После review защита от leakage усилена техническим identity-контрактом:

- split фиксируется byte-for-byte по file size и SHA-256;
- сохраняются `seed = 42` и validation fingerprint;
- проверяется нулевой overlap items между train и validation;
- generation manifest связывает каждый synthetic artifact с exact split hash;
- validation IDs, пары и конкретные значения не передаются генератору.

Проверки только filename или row count недостаточно: другой parquet с тем же названием/числом строк считается другим split.

---

# 13. Роли FP, FN и truncation

## FP

Главный источник hard-negative failure modes:

```text
human negative
model thinks match
```

## FN / positive evidence

Контрпримеры против overgeneralization.

Они отвечают на вопрос:

> Могут ли human-matching products также различаться по этому dimension?

Если да, dimension нельзя считать автоматически critical без дополнительного context.

## Truncation

FP может возникнуть потому, что decisive attribute вообще не попал в model input.

Такие кейсы нужно отделять от true semantic failures.

Если critical difference был truncated, добавление новых examples с тем же поздним attribute может не решить настоящую проблему.

Это отдельное направление preprocessing/input prioritization.

---

# 14. Финальное архитектурное решение

Текущий план:

```text
Frida validation FP/FN
        │
        ▼
failure taxonomy
+ positive counterexamples
+ truncation analysis
        │
        ▼
human-train confirmation
       / \
      /   \
     ▼     ▼
    A2      C
     │      │
     ▼      ▼
pilot generation
     │
     ▼
strict validation
     │
     ▼
separate Frida training experiments
```

A2 и C сначала должны оставаться отдельными, чтобы можно было измерить их индивидуальный эффект.

---

# 15. Pilot philosophy

Не генерировать полный volume сразу.

Первый pilot:

- примерно `50–100` A2 candidates на выбранную категорию;
- примерно `50–100` C candidates на выбранную категорию;
- manual inspection;
- rejection statistics;
- A2 `hardening_delta`;
- C distribution по critical dimensions и replacement values.

Только после quality review масштабировать генерацию, например до `1000` accepted synthetics на категорию.

---

# 16. Дополнительные generation guardrails после review

## Compute budget

Generation должна планироваться под две T4 с возможностью параллельного запуска и суммарной квотой порядка `7 × 30` GPU-часов. TPU v5e-8 доступна существенно меньше, поэтому pilot не должен зависеть от большого TPU-бюджета. Конкретную generation model команда выберет отдельно после оценки доступных вариантов.

## Few-shot quality

До pilot нужно собрать и вручную утвердить `3–10` максимально качественных и разнообразных few-shot examples. Они становятся отдельным версионируемым artifact с SHA-256. Без готового few-shot pack массовая generation не начинается.

## Reproducibility и diversity

В generation manifest сохраняются exact model ID/revision, prompt template, few-shot hash, decoding parameters, source IDs и seed каждого shard. Разнообразие обеспечивается заранее зафиксированным набором seeds и source items.

## Reject policy

Если critical attribute отсутствует или Qwen не смог надежно применить одну controlled mutation, example отклоняется и source пропускается. Regeneration того же rejected source не выполняется. Возможный бинарный классификатор наличия critical attribute остается отдельной research hypothesis.

Строгая синхронизация всех representations (`name` и `attributes`) остается намеренным guardrail первого pilot, даже если схематический пример `60Ah -> 70Ah` выглядит искусственно строгим.

---

# 17. Что уже принято, а что еще гипотеза

## Принято

- Текущая reference-модель для этого research — Frida.
- Validation используется только для research; generation — только из train.
- Нет глобального category-level attribute, который заранее считается guaranteed negative.
- A2 и C — две pilot branches.
- A2 обязан сохранять identity товара.
- C меняет одну semantic characteristic и синхронизирует все representations.
- C может генерировать несколько replacements по одному train-supported critical dimension.
- Переключаемые параметры Ozon считаются разными товарами, но глобальное правило только по имени attribute все еще недопустимо.
- Актуальный Frida research использует exact item-disjoint split seed 42 и проверяет его byte-for-byte.
- Старый FP/FN artifact нельзя использовать без строгого совпадения model/split identity.
- После reject source example пропускается без regeneration.
- До generation утверждаются `3–10` diverse few-shot examples.
- A2 и C сначала обучаются отдельно.
- Сначала pilot, потом scale.

## Еще предстоит проверить

- улучшает ли A2 Frida после retraining;
- обобщаются ли выбранные C critical dimensions за пределы observed train pairs;
- коррелирует ли высокий `hardening_delta` с полезностью A2 examples;
- улучшают ли synthetics weak categories без деградации macro AP и других категорий.

Это должны решать эксперименты, а не предположения.

---

# 18. Связь документов

Этот файл — история решений и rationale.

Файл:

```text
SYNTHETIC_PIPELINE_v2.md
```

— актуальный executable research/generation plan.

Decision log отвечает на вопрос **почему v2 устроен именно так**.

Pipeline отвечает на вопрос **что делать дальше**.
