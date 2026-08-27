# Grounded generation v1

Новая генерация не выбирает случайную строковую замену и не ремонтирует результат постфактум.

Для каждой ещё не использованной positive human-train пары код берёт `different_product` примеры из RAG pack, по имени восстанавливает полную локальную human-train карточку и строит до четырёх готовых identity bundles.

Gate до LLM:

- source token обязан относиться к явному brand/model/article полю;
- target token берётся только из явного identity-поля полной реальной карточки;
- article/model замена сохраняет бренд;
- target token присутствует в target title;
- старый бренд, транслитерация и неподтверждённые латинские line/model tokens не остаются;
- общие protected specs source и target не противоречат друг другу;
- меняются только title и identity attributes.

Structured LLM judge может только выбрать предложенный `bundle_id` либо `skip`. Свободные mappings запрещены. Из подтверждённого bundle создаётся одна two-sided positive и одна one-sided negative пара; сторона negative детерминированно чередуется.

Первый запуск — 200 операций, сбалансированных по category/role и не пересекающихся с `fast_v2_full_12k_v1`. Только после ручного quality gate разрешается масштабирование до 5–6 тысяч операций.
