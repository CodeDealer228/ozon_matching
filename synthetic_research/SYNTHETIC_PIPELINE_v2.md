# Pipeline v2: A2 + C Synthetic Hard Negatives для Frida

## 0. Область эксперимента и текущая reference-модель

Этот документ описывает актуальный план эксперимента с synthetic hard negatives после того, как исследование `pipeline_v1` не смогло найти надежные глобальные правила мутации.

**Текущая reference-модель — Frida, а не BERTA.**

Следовательно:

- анализ ошибок строится по FP/FN Frida;
- hardening score для A2 измеряется Frida;
- анализ truncation и реально видимого model input проводится по Frida;
- целевые категории нужно выбирать по актуальным per-category метрикам Frida, а не автоматически наследовать список из старого запуска BERTA.

Предыдущий список слабых категорий (`Обувь`, `Ювелирные изделия`, `Одежда`, `Автотовары`) остается полезной отправной точкой, но перед генерацией Codex должен пересчитать текущий baseline Frida и подтвердить, какие категории действительно являются худшими сейчас.

Актуальная лучшая Frida обучалась и проверялась на другом split, чем модель из предыдущего анализа ошибок. Canonical split для нового research:

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

Существующий датасет ошибок содержит scores/errors моделей и точные model inputs:

- `all_fp_fn_full_attributes_and_exact_inputs.jsonl`
- внешняя копия: https://www.kaggle.com/datasets/grigoriyyy/all-fp-and-fn-on-validation

Но этот artifact нельзя считать актуальным для новой Frida без строгой проверки identity модели и split. Если identity не совпадает, predictions и FP/FN dump нужно пересоздать на canonical split выше. В частности, гипотезу `critical_attribute_mismatch` нужно повторно проверить глазами на ошибках новой Frida и подтвердить, что она по-прежнему является одним из основных failure modes.

Validation используется только для понимания **failure modes**. Synthetic training examples создаются только из **human train**.

---

## 1. Общая схема

Эксперимент состоит из двух независимых веток синтетики:

- **A2** — сделать реальную train-negative пару сложнее, не меняя идентичность товаров;
- **C** — воспроизводить узкие, подтвержденные train-данными identity-critical различия и генерировать несколько контролируемых вариантов по одному critical dimension.

Сначала обе ветки генерируются, валидируются и обучаются отдельно. Совместный эксперимент проводится только после того, как понятен вклад каждой ветки.

```text
                CURRENT FRIDA
                     │
                     ▼
          validation FP + FN dump
                     │
                     ▼
           failure-mode research
          + truncation separation
                     │
                     ▼
          taxonomy of Frida errors
                     │
          НЕ ГЕНЕРИРУЕМ ИЗ VAL
                     │
                     ▼
               HUMAN TRAIN ONLY
                /            \
               /              \
              ▼                ▼
             A2                 C
      harden real          critical-dimension
      train negative          synthesis
             │                │
             ▼                ▼
         validator         validator
             │                │
             ▼                ▼
       Frida old/new       accepted C
       hardening score      synthetics
              \              /
               \            /
                ▼          ▼
            separate training runs
                     │
                     ▼
             SAME UNTOUCHED VAL
                     │
                     ▼
        per-category AP + macro AP
```

---

## 2. Этап 0 — зафиксировать актуальный baseline Frida

Перед новым исследованием и генерацией:

1. Определить точный текущий checkpoint/config Frida, который команда считает актуальным.
2. Проверить canonical split byte-for-byte по размеру и SHA-256, указанным в разделе 0; не продолжать при несовпадении.
3. Пересчитать predictions и per-category AP актуальной Frida на validation из этого split.
4. Сохранить macro AP и AP по всем категориям.
5. Подтвердить целевые категории для synthetic-эксперимента.
6. Построить новый Frida FP/FN subset для этих категорий.
7. Вручную проверить, что `critical_attribute_mismatch` остается актуальным систематическим failure mode новой Frida.
8. Убедиться, что для analysis доступны exact Frida input и truncation diagnostics.

Не использовать старый рейтинг категорий BERTA как неявный источник истины, если актуальная Frida дает другую картину.

---

## 3. Этап 1 — исследование failure modes Frida на validation

### 3.1. FP — основной источник negative failure modes

Главные интересующие случаи:

```text
human label = 0
Frida prediction/score = match-like
```

Это реальные разные товары, которые Frida путает.

Для каждой целевой категории нужно классифицировать FP по минимальному интерпретируемому semantic difference между товарами, например:

- model / model code;
- article / SKU / OEM;
- capacity;
- power;
- compatibility;
- quantity;
- dimensions;
- generation/year;
- product subtype;
- другие category-specific характеристики.

По разъяснению организаторов, каждое значение параметра, доступное через переключатель внутри карточки Ozon (размер, цвет и т. п.), считается отдельным товаром. Поэтому такие variant dimensions можно рассматривать как источник корректных negatives.

Однако это разъяснение не превращает любое поле `color`, `size`, `material` и т. п. в безусловное глобальное правило мутации: продавцы могут зашивать в переключатели вообще другие товары, а human labels и заполнение attributes могут быть шумными. Для генерации по-прежнему нужен подтвержденный train-контекст или надежно определенная variant family; противоречащие human-positive примеры делают правило неоднозначным для текущего датасета.

### 3.2. FN/positive cases — контрпримеры

FN и другие human-positive примеры используются для проверки apparent FP patterns.

Пример:

```text
FP:
разные размеры обуви встречаются среди confusing negatives

но

positive/FN:
разные размеры обуви также встречаются среди human matches
```

Вывод:

```text
shoe size сам по себе не является надежным identity-critical dimension
```

Цель — не получить глобальный список `safe_negative`.

Цель — найти **узкие failure modes** и candidate **critical dimensions внутри конкретного product family/context**.

### 3.3. Отделить semantic errors от truncation errors

Для каждого интересного FP определить:

1. critical difference присутствовал в exact Frida input, но модель его проигнорировала;
2. critical difference существовал только в full item, но был удален preprocessing/truncation.

Для synthetic augmentation приоритет — случай **1**.

Случай **2** в первую очередь относится к preprocessing / attribute prioritization и должен учитываться отдельно.

---

## 4. Результат Этапа 1 — taxonomy ошибок

Для каждого candidate pattern сохранить как минимум:

```text
category
product_family / subtype
candidate_critical_dimension
FP support
positive/FN counterexamples
non-truncated FP support
typical Frida score
examples
verdict
```

Рекомендуемые verdicts:

- `candidate_for_A2`
- `candidate_for_C`
- `preprocessing_issue`
- `ambiguous`
- `reject`

Важно: validation examples являются только **research evidence**.

Они не должны становиться synthetic training examples.

---

## 5. Переход от validation patterns к human train

После обнаружения failure mode на validation:

1. искать аналогичные товары/пары только в human train;
2. заново подтвердить паттерн на train;
3. на train evidence построить source pools для A2/C;
4. не копировать validation item IDs, конкретные пары или значения в synthetic train только потому, что они были замечены на validation.

Перед построением source pools обязательна защита от утечки:

- загрузить только строки `split == train` из exact canonical split;
- проверить размер и SHA-256 parquet byte-for-byte;
- проверить `shared train/validation items == 0`; при item-disjoint split это также гарантирует отсутствие общих unordered pairs;
- сохранить в generation manifest путь split, SHA-256, validation fingerprint и `seed = 42`;
- не передавать генератору validation items, IDs, пары или конкретные значения из validation examples.

Правильная схема:

```text
validation
→ понять тип ошибки
→ найти аналогичный паттерн в train
→ генерировать только из train
```

---

# Ветка A2 — identity-preserving hardening реальных train negatives

## 6. Source pair

Берется реальная human train pair:

```text
(A, B)
label = 0
```

Предпочтительно, чтобы она соответствовала одному из failure modes Frida, найденных на Этапе 1.

A и B уже гарантированно являются разными товарами по human label.

Нужно получить:

```text
B -> B'
```

так, чтобы:

```text
identity(B') == identity(B)
```

но текстовая форма B' стала сложнее для Frida относительно A.

---

## 7. Что Qwen может менять в A2

Для первого pilot:

- `attributes` полностью immutable;
- Qwen может переписывать только `name`;
- вся фактическая информация исходного B должна сохраниться;
- model codes, brands, articles, quantities, sizes, colors, compatibility, numbers, units и другие identity-bearing facts нельзя менять.

Qwen может:

- менять порядок формулировок;
- использовать более похожий синтаксический шаблон;
- делать безопасные paraphrases;
- формулировать уже имеющуюся информацию в стиле, более похожем на A.

Qwen не должен переносить факты из A в B.

Пример:

```text
A:
Фильтр масляный MANN W914/2 для Lada

B:
Масляный фильтр Mahle OC384 ВАЗ

B':
Фильтр масляный Mahle OC384 для ВАЗ
```

Поверхностная форма стала ближе к A, но товар B не изменился.

---

## 8. Валидация A2

Каждый A2 result проходит deterministic / при необходимости LLM-assisted checks на сохранение identity.

Минимальные проверки:

- `attributes` byte-for-byte не изменились;
- category не изменилась;
- protected factual tokens/values сохранились;
- brand/model/article/number из A не скопированы в B';
- B' не идентичен B после trivial normalization;
- подозрительные изменения rejected.

После этого обе пары оцениваются frozen current Frida:

```text
old_score = Frida(A, B)
new_score = Frida(A, B')

hardening_delta = new_score - old_score
```

A2 считается полезным, если synthetic pair становится более match-like для текущей модели, оставаясь истинным negative.

Для pilot:

- сохранять все deltas;
- ранжировать accepted synthetics по `hardening_delta`;
- предпочитать `hardening_delta > 0`;
- анализировать distribution, а не вводить сразу произвольный высокий threshold.

---

# Ветка C — train-supported critical-dimension synthesis

## 9. C — не копирование конкретной пары значений

Первая формулировка C была слишком жесткой:

```text
observed negative: 60Ah vs 70Ah
→ генерировать только 60Ah -> 70Ah
```

Это сильно ограничивает число примеров и может привести к memorization конкретных значений.

Новая интерпретация:

> Если research показывает, что semantic dimension является identity-critical внутри достаточно узкого product family/context, можно генерировать несколько вариантов, меняя этот dimension на разные валидные значения.

Например, если train evidence подтверждает:

```text
product_family = automotive battery
critical_dimension = capacity_Ah
```

то generation policy может разрешить:

```text
60Ah -> 65Ah
60Ah -> 70Ah
60Ah -> 75Ah
```

Цель — научить Frida, что узкое изменение по этому semantic dimension может быть решающим, а не заставить модель запомнить только `60` и `70`.

---

## 10. Как подтверждается critical dimension

Dimension нельзя считать critical только по интуиции.

Он должен быть подтвержден human train в достаточно узком контексте.

Для каждого:

```text
(product_family/context, semantic_dimension)
```

нужно оценить:

- support среди near-identical train negatives;
- positive counterexamples;
- видимость dimension в model input;
- насколько надежно можно исключить другие semantic differences;
- observed value domain в том же train family;
- количество distinct values.

Нужное правило выглядит так:

```text
category + product_family/context + semantic_dimension
```

а не:

```text
category + attribute
```

Примеры формы правила:

```text
Автотовары / battery family / capacity_Ah
Автотовары / specific part family / OEM code
Ювелирные изделия / specific subtype / manufacturer article
```

Это только примеры структуры правила, а не заранее утвержденные истины.

---

## 11. Replacement value pool для C

Для approved critical dimension строится replacement pool только из **реальных значений human train того же узкого family/context**.

Предпочтительный вид:

```text
source value: V0
replacement candidates: V1, V2, V3 ...
```

Все values должны:

- реально встречаться в train;
- быть type-compatible;
- быть unit-compatible;
- быть правдоподобными для того же product family;
- отличаться от source value.

После того как сам **dimension** достаточно надежно подтвержден как critical в этом family, не требуется exact human-negative pair для каждой конкретной мутации `V0 -> Vi`.

Это главное расширение C по сравнению с предыдущим вариантом.

---

## 12. Сколько C-синтетик можно делать с одного source item

На первом проходе:

- один source item может дать несколько C synthetics по одному approved critical dimension;
- replacement values должны быть разными;
- нужен cap на `(source_item, critical_dimension)`, чтобы не концентрировать датасет на одном товаре.

Рекомендуемый pilot cap:

```text
max 3 accepted replacements per (source_item, critical_dimension)
```

Пример:

```text
original capacity = 60Ah

synthetic #1 = 65Ah
synthetic #2 = 70Ah
synthetic #3 = 75Ah
```

Все три примера обучают одной semantic закономерности, но на разных concrete values.

---

## 13. Роль Qwen в C

Qwen НЕ решает:

- является ли dimension critical;
- валидно ли правило для данного product family;
- подходит ли source item.

Это определяет pipeline на основе данных.

Желательно, чтобы pipeline также сам передавал Qwen concrete replacement value из train-derived value pool.

Основная задача Qwen — применить заданную semantic mutation согласованно ко всему item.

Концептуально:

```text
original item
semantic_dimension = capacity_Ah
old_value = 60Ah
new_value = 70Ah
```

Qwen возвращает synthetic item, где изменился только этот semantic dimension.

До запуска pilot нужно собрать и вручную проверить `3–10` максимально качественных и разнообразных few-shot examples. Они должны покрывать разные product families, формы записи units и случаи, когда одна characteristic встречается одновременно в `name` и `attributes`. Few-shot pack хранится как отдельный версионируемый artifact; в generation manifest записываются его путь и SHA-256.

Generation model пока не фиксируется: команда отдельно сравнит доступные варианты. Реальный бюджет планирования — две T4, которые можно запускать параллельно, примерно `7 × 30` GPU-часов квоты; TPU v5e-8 доступна существенно меньше. Pilot не должен зависеть от большого TPU-бюджета.

Для воспроизводимости сохраняются exact model ID/revision, prompt template, few-shot pack, decoding parameters и seed каждого generation shard. Разнообразие создается заранее зафиксированным набором seeds и source items, а не незафиксированной случайностью.

---

## 14. Обязательная синхронизация name ↔ attributes в C

Одна semantic characteristic может быть представлена:

- в `name`;
- в одном structured attribute;
- в нескольких related structured attributes;
- в разных text/unit representations.

Пример:

```text
original:
name = "... 60Ah ..."
attributes.capacity = "60 А·ч"

synthetic:
name = "... 70Ah ..."
attributes.capacity = "70 А·ч"
```

Это одна semantic mutation.

Некорректно:

```text
name = "... 60Ah ..."
attributes.capacity = "70 А·ч"
```

Synthetic item должен оставаться внутренне согласованным.

Пример с `60Ah -> 70Ah` намеренно схематичный и строгий. Для первого pilot это принимается как guardrail, удерживающий Qwen в рамках одной semantic mutation; расширять свободу rewrite можно только после анализа accepted/rejected examples.

---

## 15. Валидация C

Generation rejected, если выполняется хотя бы одно:

- category изменилась;
- target critical dimension не изменился;
- source и replacement values эквивалентны;
- изменился non-target semantic characteristic;
- `name` и `attributes` стали противоречить друг другу;
- old value осталось в конфликтующем representation;
- Qwen придумал unrelated fact;
- synthetic является duplicate;
- correctness нельзя достаточно надежно подтвердить.

При сомнении — reject и перейти к следующему source example. Повторно генерировать тот же rejected example нельзя: отсутствие или ненадежное распознавание critical attribute является допустимой причиной пропуска, а regeneration увеличивает риск неконтролируемой мутации.

Для каждого отказа сохраняется `reject_reason`. Отдельный бинарный классификатор `critical attribute present / absent` можно исследовать позднее, но он не является обязательной частью первого pilot.

---

# Экспериментальный протокол

## 16. A2 и C сначала не смешивать

Первые training runs:

```text
baseline
train + SYN_A2
train + SYN_C
```

Только после понимания обеих веток:

```text
train + SYN_A2 + SYN_C
```

Так можно понять, какая именно ветка дала improvement/regression.

---

## 17. Сначала pilot, потом scale

Не запускать сразу 4000 examples.

Сначала:

1. завершить Frida error taxonomy;
2. собрать и утвердить `3–10` diverse few-shot examples;
3. сгенерировать примерно `50–100` A2 candidates на выбранную категорию;
4. сгенерировать примерно `50–100` C candidates на выбранную категорию;
5. вручную посмотреть accepted/rejected examples;
6. проанализировать A2 `hardening_delta`;
7. проверить distribution C по critical dimensions и replacement values;
8. измерить rejection rate, reject reasons и source-item concentration.

Только после quality review масштабировать генерацию, например до `1000` accepted synthetics на целевую категорию.

---

## 18. Training и evaluation

Во всех сравниваемых запусках использовать одинаковую Frida architecture/training setup, если эксперимент явно не тестирует другую переменную.

Фиксировать:

- canonical split path, file size, SHA-256 и validation fingerprint;
- human train rows из этого split;
- untouched human validation rows из этого же split;
- training hyperparameters;
- split seed, generation seeds и training seed;
- evaluation code.

Перед каждым сравнительным запуском split должен совпасть byte-for-byte с зафиксированным parquet. Проверка только filename, row count или логического состава недостаточна.

Отчет должен содержать:

- macro AP;
- AP по каждой категории;
- delta целевых категорий;
- regressions на остальных категориях;
- количество A2/C samples;
- generation rejection rates;
- A2 hardening statistics;
- C distribution по product family и critical dimension.

Главный вопрос эксперимента:

> Какой synthetic mechanism исправляет какие реальные failure modes Frida, не создавая новых regressions?
