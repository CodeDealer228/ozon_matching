# Codex: design review A2 v3 перед следующим pilot

Нужно **не начинать реализацию сразу**, а сначала критически разобрать новый план A2 generation и вернуть мне техническое заключение.

Текущая задача — улучшить генерацию synthetic hard negatives для product matching так, чтобы:

1. synthetic сохранял identity исходного товара B;
2. negative label оставался корректным;
3. B' становился сложнее для frozen Frida относительно A;
4. большинство оплаченных Qwen requests не уходило на бессмысленные normalization;
5. Qwen не переносил в B факты товара A;
6. максимально возможная часть логики выполнялась локально и детерминированно, без дополнительных LLM-вызовов.

После твоего анализа я отдельно подтвержу или скорректирую план.  
**До этого подтверждения не изменяй код, не запускай Alibaba API и не запускай обучение.**

---

# 1. Текущее состояние

Ветка:

```text
synthetic
```

Текущая ветка synthetic research — **A2**.

Ветка C сейчас остается закрытой: предыдущий строгий train-evidence gate не подтвердил ни одной family. Не ослабляй C и не возвращай ее в работу в рамках этой задачи.

Reference-модель для текущего эксперимента:

```text
Frida
```

Frida во время генерации должна быть **frozen**.

Она используется как judge/scorer:

```text
old_score = Frida(A, B)
new_score = Frida(A, B')

hardening_delta = new_score - old_score
```

Во время построения synthetic dataset веса Frida не меняются.

Отдельный fine-tuning Frida понадобится только после того, как будет зафиксирован достаточно большой и качественный synthetic dataset.

---

# 2. Что такое A2

Исходная пара:

```text
(A, B)
human label = 0
```

A и B — разные товары по human train.

Нужно построить:

```text
B -> B'
```

так, чтобы:

```text
identity(B') == identity(B)
```

но:

```text
Frida(A, B') > Frida(A, B)
```

желательно с положительным и достаточно заметным:

```text
hardening_delta
```

В первом pilot меняется только:

```text
name2
```

Неизменяемы:

```text
category
attributes2
```

---

# 3. Что показал первый pilot

Первый pilot:

```text
100 planned source pairs
99 successful API responses
19 deterministic accepted
80 rejected
```

Основные rejection reasons:

```text
trivial_normalization                    53
not_harder_by_name_similarity            15
copied_unsupported_fact_from_item1        9
copied_numeric_fact_from_item1            4
lost_numeric_fact                         2
```

То есть главные проблемы:

1. Qwen часто не может найти безопасный hardening и делает только:
   - capitalization;
   - punctuation;
   - spaces;
   - trivial reorder.

2. Qwen иногда переносит факт A в B:
   - model;
   - article;
   - number;
   - quantity;
   - specification.

3. Текущий validator местами может давать false positives из-за слишком буквального token matching.

4. Prompt v1 заставляет модель обязательно что-то вернуть: корректного `skip` нет.

5. Few-shot v1 не показывают полный реальный input contract, особенно `attributes2`.

6. Цель «сделать ближе к A» слишком open-ended.

---

# 4. Что уже подготовлено и не нужно переписывать без причины

Уже существует безопасный transport profile v2.

Relevant files:

```text
scripts/generate_a2_pilot_qwen.py
scripts/run_a2_pilot_qwen.ps1
```

Уже подготовлено:

```text
enable_thinking = false
zero automatic retries
one paid request maximum per source pair
bounded concurrency
durable request journal
provider request IDs
HTTP status logging
elapsed-time logging
token-usage logging
separate v2 output directory
```

Не меняй эту часть архитектуры просто ради рефакторинга.

Если новый generation contract действительно потребует изменений — укажи их в плане.

---

# 5. Предлагаемое изменение архитектуры: A2 v3

Главная идея:

> Qwen не должен самостоятельно разбирать факты A/B и решать, что безопасно переносить.

Максимально возможная часть анализа выполняется **локально до API request**.

Qwen используется только как constrained surface realizer.

Предлагаемый flow:

```text
A + B
  │
  ▼
LOCAL FACT EXTRACTION
  │
  ├─ facts_A
  └─ facts_B
       │
       ▼
LOCAL FACT COMPARISON
       │
       ├─ SAFE_SHARED_FACTS
       ├─ NEGATIVE_ANCHORS
       ├─ PROTECTED_B_FACTS
       └─ FORBIDDEN_A_ONLY_FACTS
       │
       ▼
LOCAL HARDENING OPPORTUNITY FILTER
       │
       ├─ no opportunity → LOCAL_SKIP
       │
       └─ useful opportunity
                │
                ▼
          ONE QWEN REQUEST
          thinking = false
                │
                ▼
          1–3 B' candidates
                │
                ▼
        LOCAL VALIDATOR v2
                │
                ▼
         FROZEN FRIDA SCORE
                │
                ▼
       choose best safe B'
```

Проверь, насколько этот pipeline реалистичен в текущем коде.

---

# 6. Локальный fact extraction

Я предлагаю НЕ использовать Qwen для extraction facts per pair.

Это должно быть локально.

Нужно извлекать факты из:

```text
name1
attributes1, если доступны в source
name2
attributes2
```

В первую очередь интересуют легко формализуемые факты:

```text
brand
model
article / SKU / vendor code
numbers
measurements
sizes
units
quantity
color
compatibility
year
capacity
volume
weight
power
memory
other structured attribute values
```

Не нужно строить идеальную universal ontology.

Для первого pilot важнее:

```text
precision >> recall
```

Лучше не распознать часть фактов, чем ошибочно объявить разные факты эквивалентными.

---

# 7. Canonical normalization

Validator и fact comparison не должны работать только по raw substrings.

Например:

```text
20 мл
20мл
20 ML
20 ml
```

должны представлять один факт:

```text
measurement(value=20, unit=ml)
```

А:

```text
215/60 R17
215 / 60 r17
```

должны canonicalize в один tyre-size fact.

Нужно продумать локальную canonical representation хотя бы для:

```text
numbers
measurements
units
product codes
sizes
storage
capacity
weight
volume
quantity
```

Проверь существующий validator и выясни, почему возможен кейс:

```text
64 осталось визуально в B'
```

но был выставлен:

```text
lost_numeric_fact
```

Нужно понять, является ли это реальным bug или следствием normalization/tokenization.

---

# 8. PROTECTED_B_FACTS

Это факты B, которые synthetic не имеет права потерять или изменить.

Примеры:

```text
brand B
model B
article B
numbers B
measurements B
size B
color B
quantity B
compatibility B
other identity-bearing attributes B
```

Важно:

не обязательно требовать сохранения exact formatting.

Например:

```text
20 мл
20ml
```

могут считаться сохранением одного factual value.

Но:

```text
20 мл
→
30 мл
```

является identity violation.

Нужно предложить точную локальную representation и validator logic.

---

# 9. NEGATIVE_ANCHORS — новая сущность

Предлагается отдельно выделить:

```text
NEGATIVE_ANCHORS
```

Это факты B, которые явно отличаются от corresponding facts A и поэтому особенно важны для сохранения negative identity.

Пример:

```text
A.article = 3667
B.article = 2682
```

Тогда:

```text
NEGATIVE_ANCHOR = 2682
```

Или:

```text
A.weight = 5 кг
B.weight = 1 кг
```

Тогда:

```text
NEGATIVE_ANCHOR = 1 кг
```

Идея:

```text
same semantic field known for A and B
+
canonical values differ
=
B value becomes NEGATIVE_ANCHOR
```

Все negative anchors обязаны сохраниться в B'.

Проанализируй:

1. насколько надежно это можно вычислять локально;
2. какие semantic fields можно сопоставлять безопасно;
3. в каких случаях нужно отказаться от определения anchor;
4. чем NEGATIVE_ANCHORS должны отличаться от общего PROTECTED_B_FACTS.

---

# 10. FORBIDDEN_A_ONLY_FACTS

Это identity-sensitive facts, которые:

```text
есть у A
но не подтверждены B
```

Например:

```text
A:
G11, 5 кг

B:
1 кг
```

Если B не содержит evidence для `G11`:

```text
FORBIDDEN_A_ONLY_FACTS:
G11
5 кг
```

Qwen не должен переносить их ни в каком виде.

Предлагается вычислять список полностью локально:

```text
identity_sensitive(facts_A - facts_B)
```

с учетом canonical matching.

Оцени надежность такой схемы.

---

# 11. SAFE_SHARED_FACTS

Это факты, которые:

1. видны в A;
2. независимо подтверждаются B;
3. безопасно использовать для surface hardening B';
4. желательно отсутствуют или представлены менее явно в текущем `name2`.

Пример:

```text
A contains:
Kiwix

B.name:
не содержит Kiwix

B.attributes:
brand = Kiwix
```

Тогда:

```text
Kiwix ∈ SAFE_SHARED_FACTS
```

Предлагается разделить matching на tiers.

### Tier 1 — exact/canonical match

Например:

```text
Continental == continental
20 мл == 20ml
```

Разрешить.

### Tier 2 — controlled aliases

Например заранее проверенный локальный dictionary:

```text
черный ≈ чёрный
для легковых автомобилей ≈ автомобильный
```

Разрешать только консервативные aliases.

### Tier 3 — fuzzy semantic equivalence

Например:

```text
чехол ≈ накидка
```

На первом pilot **не использовать**.

Нужно оценить это решение и предложить конкретный implementation approach.

---

# 12. Не использовать Qwen для per-pair fact extraction

Предлагаемый invariant:

```text
NO LLM CALL
для:
- parsing attributes
- extracting numeric facts
- extracting units
- building PROTECTED_B_FACTS
- building NEGATIVE_ANCHORS
- building FORBIDDEN_A_ONLY_FACTS
- Tier-1 SAFE_SHARED_FACTS
- conservative Tier-2 aliases
```

Если понадобится mapping raw attribute keys:

```text
"Бренд"
"Бренд в одежде и обуви"
"Марка"
```

→

```text
brand
```

его лучше создать один раз как локальный audited artifact:

```text
attribute_aliases.json
```

а не спрашивать Qwen для каждой source pair.

Если считаешь, что часть этого нельзя надежно сделать локально, объясни конкретно какую и почему.

---

# 13. LOCAL HARDENING OPPORTUNITY FILTER

До Qwen нужно отфильтровывать пары, для которых meaningful safe hardening почти невозможен.

Пример:

```text
A:
кроссовки new balance

B:
кроссовки new balance 9060
```

Shared facts уже находятся в B.

Единственное meaningful distinction:

```text
9060
```

является protected / negative anchor.

Изменить wording нетривиально почти невозможно.

Такая source pair должна получить:

```text
LOCAL_SKIP
```

и вообще не отправляться в Alibaba.

Нужно предложить простой interpretable:

```text
hardening_opportunity_score
```

или deterministic eligibility gate.

Возможные признаки:

```text
+ count SAFE_SHARED_FACTS missing from B.name
+ наличие safe structural reorder
+ наличие clear NEGATIVE_ANCHOR
+ lexical relationship A/B
+ исходный Frida score

- ambiguous facts
- отсутствие useful safe operations
- слишком высокий risk identity corruption
```

Не нужен сложный ML classifier.

Нужен cheap, debuggable local filter.

---

# 14. A_SAFE_TEMPLATE / ALLOWED_OPERATIONS

Еще одна предлагаемая защита: не давать Qwen полностью raw A как свободный template.

Есть два возможных подхода.

## Вариант 1 — A_SAFE_TEMPLATE

Из A удалить / замаскировать forbidden facts.

Например:

```text
A:
Kiwix ... Renault Trucks 3667 ... 20 мл
```

если:

```text
3667 = forbidden A-only model/article
```

Qwen получает безопасный template без возможности случайно скопировать значение.

## Вариант 2 — ALLOWED_OPERATIONS

До API код строит небольшой список разрешенных действий:

```text
- add SAFE_SHARED_FACT "Kiwix" as prefix
- move generic product type to prefix
- reorder existing B tokens using A-like ordering
- remove approved marketplace separator/noise
```

Prompt говорит:

> Выполни только одну или несколько операций из ALLOWED_OPERATIONS.

Никаких иных factual edits.

Проанализируй оба подхода.

Нужно сказать:

- какой проще;
- какой безопаснее;
- какой лучше подходит текущему generator;
- стоит ли использовать оба одновременно;
- либо предложить более простой вариант.

---

# 15. Qwen должен иметь право skip

Новый contract обязан позволять:

```json
{
  "status": "skip",
  "reason": "no_safe_nontrivial_hardening"
}
```

Если meaningful hardening невозможен, `skip` считается правильным результатом.

Не нужно заставлять модель генерировать изменение любой ценой.

---

# 16. Несколько candidates за один API request

Вместо одного:

```text
B'
```

предлагается попросить максимум:

```text
3 candidates
```

в одном response.

Все candidates должны удовлетворять одним и тем же restrictions.

Пример schema:

```json
{
  "status": "ok",
  "candidates": [
    {
      "rewritten_name2": "...",
      "used_safe_facts": ["..."],
      "applied_operations": ["..."]
    },
    {
      "rewritten_name2": "...",
      "used_safe_facts": ["..."],
      "applied_operations": ["..."]
    }
  ]
}
```

или:

```json
{
  "status": "skip",
  "reason": "no_safe_nontrivial_hardening"
}
```

После этого:

```text
all candidates
→ local validator
→ frozen Frida
→ select highest Frida score
```

Проанализируй trade-off:

```text
1 candidate
vs
2 candidates
vs
3 candidates
```

с учетом:

- token cost;
- acceptance probability;
- diversity;
- Frida reranking;
- API latency.

Не предлагай несколько отдельных paid requests на одну pair без очень серьезной причины.

---

# 17. used_safe_facts и applied_operations

Предлагается оставить:

```text
used_safe_facts
applied_operations
```

не как источник истины, а как audit metadata.

Validator им не доверяет.

Они нужны, чтобы потом анализировать:

```text
какие types of transformation реально дают высокий Frida delta
```

Например:

```text
brand restoration
generic type insertion
safe reorder
noise removal
```

Если считаешь эти поля лишними — объясни почему.

---

# 18. Validator v3

Validator должен быть независим от Qwen self-report.

Минимально проверить:

```text
attributes2 unchanged
category unchanged

all NEGATIVE_ANCHORS preserved

all critical PROTECTED_B_FACTS preserved

no FORBIDDEN_A_ONLY_FACTS copied

no new unsupported numeric facts

no identity-sensitive B fact changed

B' != B after trivial normalization

all inserted SAFE_SHARED_FACTS have B evidence
```

String similarity:

```text
new_name_similarity > old_name_similarity
```

НЕ предлагается делать главным hard gate.

Можно использовать как:

```text
diagnostic
cheap prefilter
```

Главный hardening metric после identity validation:

```text
hardening_delta =
Frida(A, B') - Frida(A, B)
```

Оцени эту замену.

---

# 19. Frozen Frida — только judge во время generation

Критически важно не менять Frida во время построения synthetic dataset.

Должно быть:

```text
FRIDA_BASELINE_CHECKPOINT = frozen
```

Этот checkpoint:

```text
scores original pair
scores all B' candidates
selects/ranks hard synthetics
```

Он не обновляется после каждых новых synthetics.

После того как synthetic dataset зафиксирован:

```text
dataset manifest
source IDs
SHA-256
generation config
```

только тогда запускается отдельный comparative training run.

Например:

```text
baseline:
same original Frida starting point
+ human train

experiment:
same original Frida starting point
+ human train
+ SYN_A2
```

И обе модели оцениваются на одном untouched validation.

Не начинать это обучение сейчас.

---

# 20. Few-shot v3

Few-shot examples должны обучать **policy**, а не только показывать красивые rewrites.

Предлагаемый состав 6–8 train-only examples.

### OK cases

Например:

1. добавить shared brand, подтвержденный attributes B;
2. безопасно добавить generic product type;
3. перестроить порядок слов без factual changes.

### SKIP cases

Обязательно:

4. B уже практически максимально похож на A;
5. единственная привлекательная информация есть только в A;
6. hardening потребовал бы изменить model/article/size;
7. единственное возможное изменение — capitalization/punctuation.

Каждый few-shot должен иметь тот же schema/context, что production request.

Validation examples использовать нельзя.

Оцени, нужен ли вообще few-shot после сильного local constraint layer или zero/few-shot prompt будет надежнее и дешевле.

---

# 21. Pilot v3

До API всё возможное сделать локально.

Предлагаемый первый pilot:

```text
existing 500 A2 source pairs
        │
        ▼
local extraction / normalization
        │
        ▼
local opportunity gate
        │
        ▼
rank/select top ~30 promising sources
        │
        ▼
Qwen
thinking = false
one HTTP request/source
up to 3 candidates OR skip
        │
        ▼
local validator
        │
        ▼
frozen Frida scoring
        │
        ▼
best safe candidate/source
```

До отдельного моего разрешения **не запускать эти API requests**.

---

# 22. Какие метрики смотреть на pilot

Не ограничиваться одним `accepted rate`.

Нужно сохранить:

```text
local_skip_rate

Qwen_skip_rate

sources_with_at_least_one_valid_candidate

critical_identity_violation_rate

validator_rejection_reasons

candidates_per_source

Frida hardening_delta distribution

fraction with Frida delta > 0

median / p75 / p90 hardening_delta

transformation type statistics
```

Главный safety gate:

```text
critical identity violations = 0
```

Для acceptance/hardening не задавать сейчас произвольный строгий threshold.

Сначала получить distribution.

---

# 23. Что делать после успешного pilot

Если v3 показывает:

```text
0 critical identity violations
разумный valid-candidate rate
заметную долю Frida delta > 0
адекватный API cost
```

тогда:

1. расширить source pool за пределы текущих 500 при необходимости;
2. генерировать несколько сотен разнообразных synthetics;
3. не допускать сильной концентрации на одной category/product pattern;
4. зафиксировать dataset manifest;
5. только после этого запускать comparative Frida fine-tuning.

Текущие 19 examples первого pilot для обучения не использовать как самостоятельный meaningful training experiment.

---

# 24. Возможный fallback: edit-plan вместо full-string generation

Если даже constrained full-string generation v3 продолжит переносить facts или плохо проходить validator, рассмотреть следующий уровень ограничения.

Qwen возвращает не итоговый B', а только edit plan:

```json
{
  "status": "ok",
  "operations": [
    {
      "type": "insert_safe_fact",
      "fact_id": "shared_3",
      "position": "prefix"
    },
    {
      "type": "reorder",
      "pattern": "product_type_brand_model_specs"
    }
  ]
}
```

А итоговый title собирает локальный код.

Это не текущий основной план.

Это fallback, если full-string constrained generation окажется недостаточно безопасным.

Оцени его реалистичность, но не реализуй сейчас.

---

# 25. Что требуется от тебя сейчас

Сейчас нужно только провести design review.

Перед ответом:

1. изучи актуальные relevant файлы в repo;
2. изучи текущий generator;
3. изучи текущий validator/fact extraction;
4. изучи few-shot v1;
5. изучи prepared transport v2;
6. не изменяй файлы;
7. не делай API calls;
8. не запускай Frida training.

После этого дай заключение в следующем формате.

---

# 26. Формат ответа

## A. Общий verdict

Напиши:

```text
A2 v3:
- поддерживаю
- поддерживаю с изменениями
- не поддерживаю
```

и почему.

---

## B. Разбор каждого предлагаемого компонента

Для каждого:

```text
local fact extraction
canonical normalization
SAFE_SHARED_FACTS
NEGATIVE_ANCHORS
PROTECTED_B_FACTS
FORBIDDEN_A_ONLY_FACTS
local opportunity filter
A_SAFE_TEMPLATE
ALLOWED_OPERATIONS
status=skip
multi-candidate generation
used_safe_facts
applied_operations
validator v3
Frida reranking
few-shot v3
edit-plan fallback
```

дать verdict:

```text
keep
change
drop
```

и короткое техническое объяснение.

---

## C. Что можно надежно делать полностью локально

Особенно ответь:

1. Как именно извлекать facts?
2. Какие regex/parsers уже есть в проекте?
3. Как canonicalize numbers/units/models?
4. Как безопасно строить SAFE_SHARED_FACTS?
5. Как строить NEGATIVE_ANCHORS?
6. Где ожидаются false positives/false negatives?
7. Нужен ли вообще LLM где-либо кроме финального surface rewrite?

---

## D. Предлагаемый финальный pipeline

Нарисуй точный pipeline после твоих корректировок:

```text
input
→ ...
→ ...
→ Qwen
→ ...
→ Frida
→ accepted/rejected
```

---

## E. Exact implementation plan после моего approval

Не меняя код сейчас, перечисли:

```text
какие существующие файлы менять
какие новые файлы создавать
какие functions/classes добавлять или менять
```

Не угадывай filenames.

Сначала найди их в реальном repository.

Для каждого файла укажи конкретную ответственность.

---

## F. Pilot plan

Предложи минимальный paid pilot:

```text
сколько source pairs
сколько candidates/request
какие local prefilters
какие validation gates
какие metrics
```

Цель — получить максимум информации при минимальном количестве paid requests.

---

## G. Cost implications

Сравни приблизительно:

```text
v1
vs
предлагаемый v3
```

по:

```text
API requests
prompt length
completion length
reasoning tokens
expected wasted calls
```

Без выдумывания точной стоимости, если price данных нет.

---

## H. Open questions / blockers

Отдельно перечисли решения, которые нельзя надежно принять без моего или тимлида подтверждения.

---

# 27. Стоп-точка

После design review остановись.

НЕ:

```text
редактируй код
делай commit
делай push
запускай Qwen
запускай Frida training
```

пока я явно не подтвержу следующий шаг.