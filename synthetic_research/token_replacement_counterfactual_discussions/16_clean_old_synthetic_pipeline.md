# Очистка fast v2 перед новой генерацией

## Решение

Старые 8 812 пар не используются целиком. Brand/article/model операции удаляются независимо от ответа старого генератора. Восемь прошедших grounded-pilot операций остаются только исследовательским материалом: их объём слишком мал, а pilot не был рассчитан как финальный training dataset.

Сохраняются только операции над явными количественными и размерными ролями: volume, weight, count, length, width, height, diameter, power, resolution, format и size.

## Строгая повторная материализация

Старые synthetic items не копируются. Для каждой оставленной операции код заново берёт исходную human-train пару и:

- требует, чтобы source присутствовал в обоих названиях;
- требует source в атрибуте одной и той же роли обеих карточек;
- проверяет совпадение смыслового типа и суффикса единицы измерения;
- заменяет значение в title и только в attributes той же роли;
- не трогает числа в остальных характеристиках;
- создаёт новые synthetic IDs и ровно одну positive + одну negative пару;
- проверяет ordered/unordered duplicates и overlap с canonical validation.

Результат сохраняется отдельно в `artifacts/token_replacement_counterfactual/clean_old_spec_v1` и не смешивается с новой grounded-генерацией до отдельного training-manifest.

## Фактический результат

- вход: 4 406 операций / 8 812 пар;
- оставлено: 507 операций / 1 014 пар;
- баланс: 507 positive / 507 negative;
- удалено 2 938 brand/article операций;
- удалено 930 операций без одного и того же role-атрибута в обеих source-карточках;
- удалена 31 операция с несовпадающими суффиксами единиц;
- ordered/unordered duplicates: 0/0;
- overlap source/synthetic item IDs с canonical validation: 0/0.

Оставшиеся роли: count 150, volume 197, weight 52, length 24, width 14, height 25, diameter 24, size 14, power 5, format 2.

Fingerprints:

- `synthetic_items.parquet`: `a771d1558f2e4a91b504bab8e2a473de234ad36253122bcf2f344cdd309da2c2`;
- `synthetic_pairs.parquet`: `50510cea79d6341e5dfe9f1f3c15c84d12d38f1fdbc26b7b40a00f1118081b1a`.

Этот набор прошёл structural gate, но используется только как небольшой дополнительный spec-блок. Основной объём следующего эксперимента должен прийти из новой grounded-генерации.
