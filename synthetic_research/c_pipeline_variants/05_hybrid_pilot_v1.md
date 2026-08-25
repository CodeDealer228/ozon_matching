# Method C hybrid pilot v1

Статус: локальный plan готов, внешние запросы ещё не выполнялись.

## Зачем этот пилот

Цель — быстро проверить главный вопрос: даёт ли согласованное изменение
identity-critical факта полезный обучающий сигнал. Пилот не пытается заранее
доказать идеальность каждой возможной семьи товаров.

## Состав

| Правило | Sources | Уровень | Начальный вес |
|---|---:|---|---:|
| `c_auto_part_number` | 40 | A | 0.50 |
| `c_jewelry_base_metal` | 40 | A | 0.50 |
| `c_auto_platform` | 20 | research override | 0.15 |

Всего: `100` sources. Каждый source и каждый replacement donor используются
ровно один раз; пересечений между всеми 200 item IDs нет.

## Почему именно эти правила

- OEM/article у реальной запчасти — точечная identity mutation. Selector
  предпочитает короткие fact contracts и исключает явно шинные названия;
- металл и проба украшения меняются одной согласованной парой, например
  `gold 585 → silver 925`;
- целевая платформа автоаксессуара семантически сильна и хорошо видна в title,
  но не имеет достаточного canonical near-pair evidence. Поэтому она занимает
  только 20% пилота и получает малый вес, а override записан в manifest.

Одежда исключена из pilot v1: замена `поло → пуховое пальто` формально создаёт
negative, но требует переписать слишком много зависимых фактов. Jewelry article
отложен по той же причине.

## Что будущая модель получает

Provider request pack содержит только:

- локальный порядковый номер строки;
- category и rule ID;
- исходные name и attributes;
- разрешённую operation;
- старые и новые target values;
- сохраняемый context;
- схему ответа `ok` или `skip`.

Dataset item IDs, validation content и секреты в pack отсутствуют. Модель не
выбирает новое значение и не решает, корректно ли правило: она только применяет
заданный fact contract и пишет естественную marketplace-карточку.

## Контроль после генерации

Hard reject применяется только если:

- строка не парсится или не имеет terminal status;
- при `ok` отсутствует name/attributes;
- старое target value осталось, новое не появилось;
- name и attributes противоречат друг другу;
- изменился защищённый факт вне объявленной operation;
- появился duplicate или validation overlap.

Низкий Frida score, малое evidence или необычный стиль сами по себе не являются
reject. После validator достаточно быстрого просмотра случайной выборки и всех
research-строк.

## Identity локального плана

- upstream opportunities SHA-256:
  `1fbf7fe291216c4ed2d464c2bd11a6f213fc7331d57fd604cabb90bed8172711`;
- pilot plan SHA-256:
  `d323311541bed79faf01cfebe3622b78ddc2506ee76b11cddbe9bb5896b72aa5`;
- provider request pack SHA-256:
  `252f3fb18701b0d2d6af7b29efd1f7e2e6ce5e39ff2c2f915a79e916d866ae1c`;
- rows: `100`;
- A/research: `80/20`;
- unique sources/donors: `100/100`;
- source/donor reuse: `0`;
- validation item overlap: `0`.

Локальные файлы:

- `artifacts/synthetic_v2/c_hybrid_pilot_plan_v1/manifest.json`;
- `artifacts/synthetic_v2/c_hybrid_pilot_plan_v1/pilot_plan.parquet`;
- `artifacts/synthetic_v2/c_hybrid_pilot_plan_v1/provider_request_pack.jsonl`.

## Решение после пилота

Если validator оставляет около `60+` естественных строк, сразу проводится один
контролируемый training ablation `control vs C`. Масштабирование и добавление
следующих правил разрешаются только после реального validation результата.
