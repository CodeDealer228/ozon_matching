# Taxonomy: аудит hard FP/FN BERTA и FRIDA

## Главный принцип

Не начинай с предположения, что human label правильный, и не начинай с предположения,
что модель ошиблась из-за truncation.

Для каждой пары рассуждай в таком порядке:

1. Прочитай **полные исходные** name/category/attributes обоих товаров.
2. Оцени правдоподобность human label.
3. Если label выглядит явно подозрительно — `likely_label_issue`.
4. Если без бизнес-правил нельзя уверенно решить, должны ли товары матчиться —
   `ambiguous_business_rule`.
5. Только если label выглядит правдоподобно, ищи причину ошибки focal-модели.
6. Сравни полные данные с `*_visible_decoded` именно focal-модели.
7. Называй ошибку `truncation_related` только если критическая информация реально была
   в исходных данных и реально отсутствовала/обрезалась во входе focal-модели.
8. Если критическое различие было видно модели, truncation не является объяснением.

## Primary classes

### `likely_label_issue`

Human target выглядит вероятно неправильным или внутренне противоречивым.

Типичные ситуации:
- `target=0`, но карточки выглядят как один и тот же SKU/товар по доступной информации;
- `target=1`, но видны сильные identity-defining различия, которые трудно совместить
  с точным product matching.

Не используй этот класс просто потому, что модель очень уверена.

### `ambiguous_business_rule`

По карточкам есть реальная неоднозначность: итог зависит от неизвестной policy организаторов.
Например, неясно, считается ли конкретный вариант одной сущностью или отдельным товаром.

Используй этот класс вместо самоуверенного угадывания бизнес-правил.

### `truncation_related`

Label выглядит правдоподобно, и decisive information присутствует в FULL ORIGINAL,
но focal-модель её не получает из-за char/token/pair truncation.

Обязательно укажи, какая информация потеряна.

### `critical_attribute_mismatch`

Обычно FP. Label `0` выглядит правдоподобно.
У товаров есть небольшое, но identity-defining отличие (model/article/size/volume/etc),
и это отличие **видно во входе focal-модели**, но модель всё равно дала высокий match score.

### `product_type_confusion`

Обычно FP. Модель спутала разные типы товара при сильном лексическом сходстве:
например, два разных продуктовых subtype/назначения. Различие видно модели.

### `semantic_or_alias_failure`

Обычно FN. Label `1` выглядит правдоподобно, но одна и та же сущность описана
по-разному: alias, сокращение, перестановка слов, разные формулировки или бедная карточка,
и модель не связала их.

### `variant_policy_failure`

Label выглядит правдоподобно, а ошибка связана с variant dimension:
размер, цвет, комплектность, вкус, материал, упаковка и т.п.
Здесь важно, что проблема именно в policy "что считать тем же товаром", а не просто
в обнаружении любого несовпадения.

### `missing_information`

Данных в двух карточках недостаточно, чтобы обоснованно понять, почему label должен быть
таким. Это не то же самое, что `ambiguous_business_rule`: здесь проблема именно в
недостатке информации.

### `other_model_error`

Label выглядит правдоподобно, ошибка модели реальна, но она не укладывается в классы выше.

## Secondary subtype

`secondary_subtype` — короткая нормализованная строка. Сначала старайся использовать:

- `model_number`
- `article_sku`
- `size_dimension`
- `volume_weight`
- `pack_count`
- `quantity`
- `color`
- `material`
- `brand`
- `product_type`
- `compatibility`
- `gender`
- `flavor_scent`
- `formulation`
- `dosage`
- `optical_parameter`
- `tire_parameter`
- `memory_storage`
- `generation_version`
- `purpose`
- `set_completeness`
- `other_numeric`
- `alias_synonym`
- `missing_attribute`
- `char_truncation`
- `item_token_truncation`
- `pair_budget_truncation`
- `near_exact_duplicate`
- `annotation_inconsistency`
- `other`

Если ни один не подходит, можно дать короткую новую snake_case строку, но не плодить
синонимы без необходимости.

## Label assessment

- `likely_correct`
- `likely_wrong`
- `ambiguous`

## Model failure assessment

- `real_failure`
- `not_model_failure`
- `unclear`

## model_saw_critical_difference

- `yes`: decisive difference явно присутствует в focal `item*_visible_decoded`;
- `no`: decisive difference есть в FULL ORIGINAL, но отсутствует во входе focal-модели;
- `unclear`;
- `not_applicable`.

## truncation_relevance

- `causal`: потеря информации непосредственно объясняет ошибку;
- `possible`: truncation могло помочь, но causal claim слабый;
- `not_relevant`: decisive information и так была видна или truncation отсутствует;
- `unclear`.

## Важное правило precedence

Сначала label quality, затем model failure.

Если label выглядит вероятно неправильным:
`primary_class = likely_label_issue`, даже если у пары есть truncation.

Если label неоднозначен:
`primary_class = ambiguous_business_rule`.

Только после этого классифицируй настоящую ошибку модели.
