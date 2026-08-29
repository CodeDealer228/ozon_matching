# Разметка few-shot v1

## Что делает сборщик

Сборщик создаёт локальный JSON-шаблон из 60 human-train positive-пар: по 3 пары на каждую из 20 категорий. Полные карточки и ID сохраняются только в `artifacts/`, который не публикуется в GitHub.

## Подготовка шаблона

```powershell
python scripts\generate_token_replacement_counterfactuals.py `
  --count 60 `
  --output-dir artifacts\token_replacement_counterfactual\few_shot_sources

python scripts\prepare_token_replacement_few_shots.py
```

Результат:

```text
artifacts/token_replacement_counterfactual/few_shot_annotation_template_v1.json
```

## Как заполнять один пример

Для хорошей замены:

```json
"decision": {
  "status": "replace",
  "replacements": [
    {
      "source_token": "lynxauto",
      "replacement_token": "Masuma"
    }
  ],
  "skip_reason": ""
}
```

Для пропуска:

```json
"decision": {
  "status": "skip",
  "replacements": [],
  "skip_reason": "Общие значения не задают идентичность товара"
}
```

`source_token` копируется только из `input.offered_tokens`. `replacement_token` должен быть одним естественным токеном без пробелов и того же смыслового типа. В сомнительном случае выбирается `skip`.

Поля `source_reference_local_only` и `reviewer_note_local_only` помогают локальному ревью и никогда не включаются в provider payload. Перед использованием генератор повторно проверяет все решения и отказывается запускаться, если остался placeholder или нарушен контракт.
