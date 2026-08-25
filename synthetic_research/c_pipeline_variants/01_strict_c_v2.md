# Вариант 1 — strict C v2

## Идея

Critical dimension разрешается только внутри очень узкого product family,
если canonical human train содержит несколько почти одинаковых negative pairs,
не содержит positive counterexamples и показывает этот признак в model input.

После одобрения правила replacement values берутся из того же узкого family.
Qwen получает готовую операцию `old_value -> new_value` и синхронно меняет name
и attributes.

## Gate

- exact или почти exact family context;
- минимум несколько human-train negatives;
- ноль positive counterexamples;
- target dimension видим в title;
- почти нет других semantic differences;
- при любом сомнении — reject без regeneration.

## Плюсы

- низкий риск неверной synthetic label;
- легко объяснить каждую принятую строку;
- хорошая воспроизводимость.

## Минусы

- gate слишком чувствителен к неполноте и шуму marketplace cards;
- exact family keys часто отсутствуют или записаны неодинаково;
- небольшой human train редко даёт несколько чистых наблюдений одного правила;
- полезные закономерности превращаются в `0 approved families`.

## Фактический результат

Старый builder нашёл 60 research candidates, но 0 approved families и 0
replacement candidates. Этот результат относится к ограниченному extractor,
который не использовал `negative_patterns/*/generation_patterns.md`.
