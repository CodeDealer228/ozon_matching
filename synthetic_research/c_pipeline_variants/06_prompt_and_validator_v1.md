# Method C prompt + validator v1

Этот вариант реализует pilot v1 из предыдущего документа.

## Главное упрощение

Qwen не переписывает весь attributes JSON. Она возвращает:

```json
{
  "status": "ok",
  "synthetic_name": "естественное название",
  "attribute_updates": {
    "существующий ключ": "новое значение"
  }
}
```

Код применяет patch к исходным attributes. Поэтому все поля, которые модель не
назвала явно, побитово остаются прежними. Это снимает большую часть старых
фильтров и одновременно не даёт незаметно потерять protected facts.

## Hard validator

Validator проверяет только:

- `status` и JSON schema;
- обновляются только существующие ключи;
- изменённые ключи относятся к target operation или содержат старый target;
- новое target value появилось;
- старое target value исчезло;
- остальные числа из исходного title не потерялись;
- не больше восьми обновлённых attribute keys.

Оценка Frida, stylistic similarity, human evidence count и «слишком необычный
title» не являются автоматическими reject reasons.

## Transport

- exact model: Alibaba Singapore `qwen3.8-27b`;
- reasoning: disabled;
- automatic retries: `0`;
- один оплачиваемый attempt на source;
- checkpoint после каждого terminal response;
- bounded workers: `6`;
- prompt cache marker стоит на стабильной части prompt;
- секрет читается из DPAPI-файла и удаляется из environment в `finally`.

До реального запуска обязательны dry-run, unit tests и сверка SHA prompt/plan.

## Проверенный dry-run

- prompt pack SHA-256:
  `7372cb3cf6b09cc41adc670fbda492787cc357cbfdf4f82c3d4d8d6a3c618699`;
- pilot plan SHA-256:
  `d323311541bed79faf01cfebe3622b78ddc2506ee76b11cddbe9bb5896b72aa5`;
- request pack SHA-256:
  `252f3fb18701b0d2d6af7b29efd1f7e2e6ce5e39ff2c2f915a79e916d866ae1c`;
- planned/remaining: `100/100`;
- uncertain submissions: `0`;
- unit tests: `14 passed`;
- external requests during dry-run: `0`.
