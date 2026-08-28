# Вариант 3 — hybrid C с мягким gate

## Решение

Использовать короткий список hard safety veto, а доказательность правила,
чистоту family context и model difficulty превращать в soft confidence score и
sample weight. Сомнительные, но правдоподобные строки не удаляются
автоматически — они получают меньший вес.

## Этап 1. Pattern catalog

Парсер читает все `negative_patterns/*/generation_patterns.md` и сохраняет:

- category и pattern id;
- generalized rule, scope и generation operation;
- guards;
- prior confidence (`strict` или `guarded`);
- negative evidence и positive counterexamples;
- проверку ссылок на соответствующий `pairs.jsonl`.

Patterns — только гипотезы. Их старые evidence pairs не используются как
generation source.

## Этап 2. Canonical train audit

Для небольшого структурированного rule registry проверяются только canonical
human-train pairs:

- negative support;
- positive counterexamples;
- visibility target dimension в model input;
- совпадение context concepts;
- количество distinct target values;
- доля пар с дополнительными semantic differences.

Результат — confidence tier, а не бинарное `approved/rejected`:

- `A`: высокая поддержка, рекомендуемый weight `0.5`;
- `B`: правдоподобное правило с ограниченным evidence, weight `0.3`;
- `research`: слабое evidence, weight `0.15`, сначала ручной review.

## Этап 3. Replacement opportunities

Source и donor группируются по реальным context values human train. Из donor
берётся только заранее объявленный target value. Для одного source разрешено не
более трёх distinct replacements.

Opportunity ещё не является synthetic row. Она содержит fact contract:

```text
rule_id
source item
target concepts
old values
replacement values
context values
confidence tier
recommended sample weight
```

## Hard safety veto

Строка всегда исключается, если:

- source или donor не из canonical train;
- найден validation item overlap;
- category изменилась;
- target value отсутствует или фактически не отличается;
- name/attributes после mutation противоречат друг другу;
- old target value осталось в конфликтующем representation;
- изменился незаявленный protected fact;
- результат не парсится или является duplicate.

## Soft signals

Не являются автоматическим reject:

- малое число human negatives;
- один или несколько positive counterexamples;
- неполный family context;
- target dimension отсутствует в title, но присутствует в attributes;
- guarded prior;
- невысокая frozen-Frida difficulty.

Они уменьшают confidence и training weight.

## Генерация

Pipeline сам выбирает concrete replacement value. Qwen не решает, является ли
правило корректным; она только согласованно применяет mutation и делает карточку
естественной. Для простой точной замены генерация может быть полностью
детерминированной без LLM.

## Pilot

1. Получить локальный audit и opportunity pool.
2. Выбрать 5–8 правил tier A/B.
3. Сделать 50–100 synthetics на правило.
4. Вручную проверить случайные accepted и rejected rows.
5. Собрать примерно 500–800 строк.
6. Запустить отдельные `baseline`, `control`, `train + SYN_C`.
7. Масштабировать только при улучшении относительно control.
