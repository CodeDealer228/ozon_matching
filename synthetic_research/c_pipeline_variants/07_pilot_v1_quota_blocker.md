# Method C pilot v1: Alibaba quota blocker

Дата попытки: 2026-08-25.

## Что произошло

После явного разрешения пользователя был запущен один C pilot на 100 planned
rows. Alibaba отклонила все обращения до inference:

```text
HTTP 403
code: AllocationQuota.FreeTierOnly
message: free quota has been exhausted; complete payment information or
disable "use free tier only" mode
```

Итог локального manifest:

- terminal records: `100`;
- request errors: `100`;
- accepted rows: `0`;
- prompt tokens: `0`;
- completion tokens: `0`;
- reasoning tokens: `0`;
- cached tokens: `0`;
- uncertain submissions: `0`;
- automatic retries: `0`.

То есть данные не были сгенерированы и токены не списаны. Это не результат
качества method C и не неудачный ML-эксперимент.

## Почему было 100 одинаковых отказов

Первый запрос выполнялся синхронно для прогрева prompt cache, но generator до
этой попытки не останавливал очередь при его terminal transport error. Остальные
99 запросов получили тот же немедленный 403.

После попытки добавлен fail-fast: если warm-up request завершился ошибкой,
остальная очередь не отправляется.

## Что требуется для продолжения

В Alibaba Cloud Model Studio для Singapore account нужно выполнить одно из двух:

1. отключить режим `use free tier only`, если payment information уже заполнена;
2. либо сначала заполнить payment information / активировать платный биллинг.

После этого нужен новый output directory `c_hybrid_pilot_v1_r2`. Старые 100
terminal error records не удаляются и не смешиваются с новой попыткой. Повтор
разрешён только после подтверждения изменения quota/billing state.
