# Варианты pipeline для synthetic method C

Method C создаёт hard negatives через контролируемое изменение одного
identity-bearing признака товара. Например, исходная карточка с аккумулятором
`60 Ah` превращается в согласованную карточку `70 Ah`, после чего исходный и
изменённый товары образуют отрицательную пару.

В этой папке варианты фиксируются по порядку, чтобы решения не терялись:

1. [`01_strict_c_v2.md`](01_strict_c_v2.md) — исходный строгий pipeline v2;
2. [`02_relaxed_c.md`](02_relaxed_c.md) — максимально свободный вариант;
3. [`03_hybrid_soft_gate_recommended.md`](03_hybrid_soft_gate_recommended.md) —
   рекомендуемый гибрид `hard safety + soft confidence`;
4. [`04_hybrid_local_audit_result.md`](04_hybrid_local_audit_result.md) —
   фактический результат локального аудита и состав первого пилота;
5. [`05_hybrid_pilot_v1.md`](05_hybrid_pilot_v1.md) — точная исполняемая
   спецификация первого пилота из 100 sources.

Текущая рекомендация — вариант 3 с ограничениями и очередностью из варианта 4.
A2 и C не смешиваются до отдельных сравнительных training runs.

## Общие неизменяемые ограничения

- source items и replacement values берутся только из canonical human train;
- validation используется только как deny-list и для итоговой оценки;
- никакой старый feature/token cache не используется без strict identity match;
- один synthetic меняет только заранее объявленную semantic operation;
- name и attributes после изменения не должны противоречить друг другу;
- все manifests содержат split identity, hashes и leak checks.
