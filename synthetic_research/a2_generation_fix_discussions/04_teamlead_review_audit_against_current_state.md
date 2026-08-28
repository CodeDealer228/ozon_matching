# Аудит замечаний тимлида относительно текущего A2 v3

Дата проверки: 2026-08-24  
Источник: GitHub PR #3, reviewer `CodeDealer228`  
PR: <https://github.com/CodeDealer228/ozon_matching/pull/3>

## 1. Короткий вывод

Большая часть замечаний по первой версии `SYNTHETIC_PIPELINE_v2.md` была
исправлена ещё в PR #3. После commit `c360398` тимлид одобрил PR, и он был
merged.

Однако одобрение архитектуры pipeline не означает, что текущий первый A2 prompt
уже реализует все guardrails. На текущий момент:

- полностью закрыты смена reference-модели на Frida, пересчёт canonical
  validation, byte-for-byte split identity и защита от leakage;
- документально закрыты правила ветки C и разделение A2/C;
- частично закрыты few-shot quality и `skip` policy: правильное направление
  зафиксировано, но production prompt v1 и few-shot v1 ещё нужно заменить;
- замечание о повторном использовании `negative_patterns/` для C не подключено
  к builder и остаётся отдельным незакрытым пунктом;
- первоначальный compute plan с локальной Qwen на T4/TPU был сознательно
  заменён на Alibaba Singapore API с exact model `qwen3.8-27b`.

## 2. Статус каждого замечания

| № | Замечание тимлида | Текущий статус | Проверка |
|---:|---|---|---|
| 1 | Нужен отдельный артефакт: почему отвергнут pipeline v1 и чем отличаются A/B/C | **Закрыто** | `SYNTHETIC_RESEARCH_DECISIONS.md` содержит историю v1, A, A2, B и C, причины выбора и ограничения. |
| 2 | Опираться на актуальную Frida, а не BERTA | **Закрыто** | Pipeline, taxonomy и A2 hardening используют Frida. Baseline пересчитан. |
| 3 | В C не ограничиваться только наблюдавшейся парой `60Ah -> 70Ah`; после подтверждения dimension использовать несколько реальных значений | **Закрыто в дизайне, не дошло до генерации** | В pipeline есть train-derived replacement pool и cap до трёх replacements. Строгий C gate дал 0 approved families, поэтому C не запускалась. |
| 4 | У лучшей Frida другой train/val split; нужно пересчитать validation и глазами подтвердить `critical_attribute_mismatch` | **Закрыто полностью** | Проверены SHA-256 и fingerprint canonical split, получено 109648 predictions; taxonomy заново построена, failure mode подтверждён. |
| 5 | По разъяснению организаторов переключаемые параметры — разные товары | **Учтено в общей политике, нужно явно закрепить в A2 v3 schema** | Текущие C-гипотезы рассматривают размер/цвет как потенциально identity-critical. В A2 v3 эти variant-defining поля нужно включить в protected/anchor whitelist. |
| 6 | Не допустить leakage из validation | **Закрыто полностью** | Validation используется только для исследования и deny-list; shared items и unordered pair overlap равны 0; source pools построены только из canonical human train. |
| 7 | Generation планировалась на 2×T4 либо TPU с ограниченной квотой | **Осознанно superseded** | Позднее команда выбрала Alibaba Singapore API и exact `qwen3.8-27b`. Это изменение среды, а не забытый guardrail. Scoring/training по-прежнему выполняются отдельно. |
| 8 | Нужны 3–10 максимально качественных и разнообразных few-shot | **Частично** | Pack v1 содержит 6 версионированных human-train examples с hash, но pilot показал, что примеры не отражают полный input contract и не обучают `skip`. Нужен новый few-shot v3. |
| 9 | Строгая синхронизация semantic mutation удерживает Qwen в узде | **Закрыто для C в дизайне; усилено для A2 v3** | Для C name/attributes должны меняться согласованно. В A2 v3 Qwen получает только заранее разрешённые операции, а attributes остаются immutable. |
| 10 | Если critical attributes нет или они ненадёжны, пример надо пропустить; после reject не regenerate | **Частично в текущем коде, полностью в плане v3** | Transport v2 уже имеет 0 automatic retries. Но prompt v1 требует rewrite и не умеет корректный `status=skip`; v3 это исправляет. |
| 11 | Фиксировать seed и проверять split byte-for-byte | **Закрыто полностью** | Split seed 42, split SHA, validation fingerprint, generation seed, source/few-shot hashes и request journal фиксируются. |
| 12 | Для C использовать `negative_patterns/*/generation_patterns.md`, особенно strict patterns, как стартовые candidates | **Не закрыто** | Каталог существует, но `build_synthetic_v2_source_pools.py` его не читает. Текущий C gate был построен независимо и завершился с 0 approved families. |

## 3. Что изменилось после ревью

### 3.1. Frida и canonical validation

Текущий baseline:

- validation pairs: `109648`;
- macro AP: `0.7588808815558529`;
- micro AP: `0.8132819127479105`;
- split SHA-256:
  `1e3df234d6cdb8685256d5144b25f9a0a3e3a023f1053a0bd5b16c40e3df8d7e`;
- validation fingerprint:
  `847d04d081432cc081c2149b75271547bf5e25b14f9d8a5804704a5be2a4b3c8`;
- shared train/validation items: `0`;
- unordered train/validation overlap: `0`.

`critical_attribute_mismatch` повторно подтверждён на актуальной Frida, а
truncation отделён от ошибок, где значимое различие действительно было видно
модели.

### 3.2. A2 source pool

Построено 500 human-train negative source pairs, по 100 на пять категорий.
Validation item overlap равен 0. В A2 меняется только `name2`, а attributes и
category остаются неизменными.

### 3.3. Первый A2 pilot

Первый prompt дал 19 deterministic accepted из 100 source pairs. Этого слишком
мало для осмысленного training run. Основные причины отказов подтвердили
опасения тимлида:

- много тривиальной normalization;
- часть изменений не делала пару сложнее;
- встречалось копирование неподтверждённых фактов и чисел;
- prompt не умел честно отказаться от генерации.

Это не опровергает A2, но показывает, что одного общего текстового запрета
недостаточно.

## 4. Что нужно добавить в A2 v3 именно из замечаний тимлида

### 4.1. Явный список variant-defining facts

Разъяснение организаторов нужно превратить в машиночитаемое правило. Размер,
цвет, объём, количество, комплектация, модель/версия и другие переключаемые
параметры могут определять отдельный товар.

Практическое следствие:

- если такой факт уже видим в `name2`, он входит в `PROTECTED_B_FACTS`;
- если значения A и B надёжно различаются, факт может стать
  `NEGATIVE_ANCHOR`;
- перенос значения A в B' запрещён;
- attributes B остаются неизменными независимо от текста.

Это не означает, что любой цвет глобально является надёжным правилом. Для
автоматического anchor по-прежнему нужна корректная category/family schema и
высокая уверенность извлечения.

### 4.2. Настоящий skip до API и внутри prompt

Замечание тимлида про пропуск нужно реализовать на двух уровнях:

1. локальный opportunity filter не отправляет бесполезную source pair в API;
2. Qwen может вернуть `status=skip`, если среди разрешённых операций нет
   безопасного содержательного rewrite.

Rejected response не регенерируется. Transport error не повторяется
автоматически; неясный исход помечается как uncertain.

### 4.3. Few-shot v3

Формальное требование `3–10 examples` выполнено в v1, но требование качества —
нет. Новый pack должен включать полный production schema:

- attributes B или уже построенный fact contract;
- allowed operations;
- protected и forbidden facts;
- три успешных примера;
- три разных `skip`-примера;
- примеры с числами, моделями и variant-defining attributes.

### 4.4. Строгий локальный contract вместо свободной инструкции

Фраза «не переноси факты A» оказалась недостаточной. Текущий проект A2 v3
улучшает исходное требование тимлида:

- факты извлекаются локально;
- Qwen получает замаскированный A;
- Qwen получает конкретные `ALLOWED_OPERATIONS`;
- ответ ссылается на `fact_id` и `operation_id`;
- deterministic validator проверяет результат до Frida scoring.

## 5. Отдельный незакрытый вопрос по ветке C

После merge тимлид предложил брать strict rules из
`negative_patterns/*/generation_patterns.md` как стартовые candidate critical
dimensions. Каталог действительно присутствует и содержит generation
operations, guards, confidence tiers и evidence.

Но текущий `build_synthetic_v2_source_pools.py` эти файлы не использует. Поэтому
вывод `C gate = 0` относится к текущему ограниченному extractor, а не является
окончательным доказательством, что в репозитории нет train-supported C rules.

Рекомендация:

- не открывать C автоматически и не смешивать её с A2;
- после стабилизации A2 отдельно сделать read-only parser/audit strict patterns;
- проверить каждый strict pattern на canonical human train, positive
  counterexamples, split leakage и model-input visibility;
- только после нового gate решать, разрешать ли C pilot.

Это отдельная работа. Она не должна задерживать безопасный локальный этап A2 v3.

## 6. Итоговая оценка

Если оценивать только архитектуру, замечания тимлида учтены хорошо: именно после
их внесения PR был approved. Если оценивать реальный первый A2 generator, ответ
другой: он всё ещё использует prompt/few-shot v1 и поэтому реализует guardrails
не полностью.

Перед следующим paid A2 pilot обязательно закрыть:

1. variant-defining protected/anchor schema;
2. локальный opportunity filter;
3. `status=skip`;
4. few-shot v3 с полным contract;
5. regression tests для чисел, единиц, размеров и кодов;
6. deterministic validator v3;
7. отдельный frozen Frida scoring без старого cache.

После этого A2 v3 будет не только соответствовать смыслу ревью тимлида, но и
технически обеспечивать его требования.
