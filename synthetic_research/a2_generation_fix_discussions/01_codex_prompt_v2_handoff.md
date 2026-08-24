# Handoff: A2 synthetic hard negatives и проект prompt v2

Дата состояния: **2026-08-24**  
Репозиторий: `CodeDealer228/ozon_matching`  
Ветка: `synthetic`  
Последний технический commit: `d412508` (`Harden Qwen generation transport`)

Этот документ предназначен для обсуждения с другой LLM. Он содержит текущее
состояние исследования, фактические результаты первого pilot, причины отказов,
предложение по новому промпту и открытые решения. Секретов и API-ключей здесь
нет.

---

## 1. Задача

Решается product matching: по двум карточкам товара нужно определить, являются
ли они одним и тем же товаром.

Ветка **A2** должна создавать сложные отрицательные примеры только из реальных
human-train negative pairs:

```text
A и B — разные товары по human label.
B' — тот же самый товар, что B, но его название переписано так,
     чтобы текущей Frida было легче спутать B' с A.
```

Меняется только `name2`. Категория и `attributes2` неизменяемы. Нельзя переносить
из A в B' модель, артикул, число, размер, цвет, совместимость или иной факт,
которого нет у B.

Главная цель — не просто увеличить строковое сходство, а получить
identity-preserving hard negative:

```text
old_score = frozen_Frida(A, B)
new_score = frozen_Frida(A, B')
hardening_delta = new_score - old_score
```

Полезный A2 должен сохранять отрицательный label и желательно иметь
`hardening_delta > 0`.

---

## 2. Неизменяемые ограничения

1. Synthetic создаётся только из canonical **human train**.
2. Validation используется только для исследования и оценки, никогда как
   источник генерации.
3. Canonical split:
   - train pairs: `256006`;
   - validation pairs: `109648`;
   - split SHA-256:
     `1e3df234d6cdb8685256d5144b25f9a0a3e3a023f1053a0bd5b16c40e3df8d7e`;
   - validation fingerprint:
     `847d04d081432cc081c2149b75271547bf5e25b14f9d8a5804704a5be2a4b3c8`;
   - shared train/validation items: `0`;
   - unordered train/validation pair overlap: `0`.
4. Старый human feature/token cache нельзя использовать для нового текста без
   строгого совпадения identity. У B' новое название, следовательно его нужно
   токенизировать как новый текст.
5. Ветка C сейчас закрыта: строгий train-evidence gate не дал ни одной
   подтверждённой family. Автоматически ослаблять C gate нельзя.

---

## 3. Текущая baseline Frida

Baseline пересчитан на canonical untouched validation и строго проверен:

- macro AP: `0.7588808815558529`;
- micro AP: `0.8132819127479105`;
- validation predictions: `109648`;
- overlap checks: `0`;
- output: `artifacts/synthetic_v2/frida_baseline_error_analysis`.

Актуальные слабые категории, по которым строился A2 source pool:

- Автотовары;
- Галантерея и аксессуары;
- Обувь;
- Одежда;
- Ювелирные изделия.

---

## 4. Что уже подготовлено

### 4.1. A2 source pool

- 500 real human-train negative pairs;
- по 100 на каждую из пяти категорий;
- все leak checks равны нулю;
- файл: `artifacts/synthetic_v2/source_pools/a2_source_pool.parquet`;
- SHA-256:
  `2dee9a0b5b2ef54c4210b07b114346057bddc45effe728e2e43056f42505dd28`.

### 4.2. Few-shot v1

- 6 train-only examples;
- покрывают все пять категорий;
- файл: `synthetic_research/a2_few_shot_v1.json`;
- SHA-256:
  `2aa9c1734fac177782c897a76780bd60b26d01e952b43c0a5a683af5257f4f2a`.

### 4.3. Модель генерации

- exact model: `qwen3.8-27b`;
- Alibaba Cloud Model Studio;
- регион: Singapore;
- OpenAI-compatible Chat Completions endpoint.

---

## 5. Первый pilot: точные результаты

Параметры:

- 100 source pairs;
- 20 на категорию;
- `temperature = 0.25`;
- `top_p = 0.8`;
- `max_tokens = 180`;
- thinking не был явно отключён;
- запросы изначально шли последовательно;
- при транспортных ошибках существовали автоматические повторы.

Результаты:

- planned records: `100`;
- successful API responses: `99`;
- transport errors: `1` (`TimeoutError`);
- deterministic accepted: `19`;
- rejected rows: `80`.

Accepted по категориям:

| Категория | Accepted |
|---|---:|
| Ювелирные изделия | 8 |
| Автотовары | 4 |
| Одежда | 3 |
| Галантерея и аксессуары | 2 |
| Обувь | 2 |

Причины отказа (одна строка может иметь несколько причин):

| Причина | Количество |
|---|---:|
| `trivial_normalization` | 53 |
| `not_harder_by_name_similarity` | 15 |
| `copied_unsupported_fact_from_item1` | 9 |
| `copied_numeric_fact_from_item1` | 4 |
| `lost_numeric_fact` | 2 |

Artifacts:

- `artifacts/synthetic_v2/a2_pilot/raw_responses.jsonl`;
- `artifacts/synthetic_v2/a2_pilot/a2_pilot_results.parquet`;
- `artifacts/synthetic_v2/a2_pilot/generation_manifest.json`;
- raw SHA-256:
  `096b649714abec6827da6295971345d5db0f2d31f621346bb6c1fe0f02869b4a`;
- results SHA-256:
  `edffeb158892d16df6ec1ca106b692a432a9db3d2bfcab5a34de423927177f21`.

Все 19 автоматически принятых примеров были дополнительно просмотрены вручную.
Явного переноса чужой модели, артикула, размера или числа в них не обнаружено.

### Почему на 19 строках пока нельзя обучать

`19 / 256006 = 0.0074%` от human train. По отдельным категориям есть лишь 2–8
примеров. Такое обучение почти наверняка даст изменение validation score в
пределах шума. Сильное повторение этих 19 строк создаст риск запоминания
конкретных пар вместо обучения общему failure mode.

---

## 6. Расход токенов

Локально сохранённый `usage` для 99 успешных ответов:

| Показатель | Значение |
|---|---:|
| prompt tokens total | 108622 |
| prompt tokens average | 1097.19 |
| completion tokens total | 149442 |
| completion tokens average | 1509.52 |
| reasoning tokens total | 146786 |
| reasoning tokens average | 1482.69 |
| visible answer tokens average | около 27 |
| total tokens recorded | 258064 |
| average total per successful pair | 2606.71 |
| median total | 1908 |
| maximum total | 8061 |
| cached prompt tokens | 0 |

Reasoning занял примерно 98% всех completion tokens.

В Alibaba UI пользователь увидел около `360000` токенов. Разница с локально
записанными ответами — `101936`. Наиболее вероятная причина: фактически было
больше 100 HTTP attempts. Один зависший source успел получить до четырёх
попыток до ручного завершения процесса, ещё одна error-пара получила две
попытки. Сервер мог сгенерировать и списать reasoning tokens, хотя клиент из-за
тайм-аута не получил ответ и `usage`.

---

## 7. Текущий prompt v1

```text
Ты делаешь A2 hardening для product matching. Есть два разных товара A и B из
проверенной human-negative пары. Перепиши только название B так, чтобы стиль и
порядок слов стали ближе к A, но identity B не изменился. Attributes B
неизменяемы. Сохрани все бренды, модели, артикулы, числа, размеры, цвета,
количество, совместимость и единицы B. Не переноси ни одного факта из A,
которого нет в B или attributes B. Не удаляй факты из исходного названия B.
Ответь ровно JSON-объектом {"rewritten_name2": "..."}.
```

Dynamic user message:

```text
Категория: {category}
Название A: {name1}
Название B: {name2}
Неизменяемые attributes B: {attributes2}
```

---

## 8. Основные проблемы prompt/few-shot v1

### 8.1. Модель обязана что-то сгенерировать

У неё нет корректного ответа `skip`. Когда безопасного улучшения нет, она
возвращает изменение регистра, пунктуации или пробелов. Это объясняет большую
часть 53 `trivial_normalization`.

### 8.2. Few-shot examples не содержат attributes B

Это наиболее серьёзная структурная проблема. Некоторые few-shot ответы
добавляют в B' бренд или формулировку, видимую в A. Но в few-shot user message
передаются только A и B, без `attributes2`. Модель видит добавление факта из A,
но не видит доказательства, что этот факт принадлежит B. Тем самым примеры
неявно учат копированию из A.

### 8.3. Все few-shot examples являются положительными

Нет примеров правильного отказа, разных моделей/артикулов и случаев, когда
безопасного hardening не существует.

### 8.4. Нет явных allowlist/denylist

Модель самостоятельно решает, какой факт A подтверждён атрибутами B. Это
ненадёжная задача для свободного текста и приводит к копированию чужих чисел
или характеристик.

### 8.5. Цель «стать ближе» не операционализирована

Не объясняется, что изменение только регистра/пунктуации не считается
результатом. Нет требования сохранить конкретное различие, из-за которого A и
B являются разными товарами.

### 8.6. Возможны ложные отказы валидатора

Например, добавление общих слов `чехол` или `накидка` может считаться
`copied_unsupported_fact_from_item1`, хотя это иногда лишь синоним исходного
типа товара. Следовательно, prompt v2 и validator v2 нужно обсуждать вместе.

---

## 9. Репрезентативные примеры

### 9.1. Удачный identity-preserving hardening

```text
A: автомобильные летние шины continental contiecocontact 6 215/60 r17 96h
B: continental contiecocontact 6 шины летние 215/60 r17 96h
B': Автомобильные летние шины Continental ContiEcoContact 6 215/60 R17 96H
```

Слово «автомобильные» поддерживается атрибутом B
`тип = шины для легковых автомобилей`. Все размеры и индексы B сохранены.

```text
A: kiwix / штрих корректор/ маркер 2в1 renault trucks 3667 gris 20 мл
B: штрих корректор/ маркер 2в1 renault trucks 2682 gris 20 мл
B': Kiwix / штрих корректор/маркер 2в1 Renault Trucks 2682 Gris, 20 мл
```

У B сохранён собственный код `2682`, а бренд Kiwix подтверждён атрибутами B.

### 9.2. Бесполезная нормализация

```text
A: кроссовки new balance
B: кроссовки new balance 9060
B': Кроссовки New Balance 9060
```

После нормализации B и B' одинаковы.

```text
A: джинсы pepe jeans london
B: джинсы pepe jeans
B': Джинсы Pepe Jeans
```

Изменён только регистр.

### 9.3. Изменение не стало сложнее

```text
A: invu очки солнцезащитные
B: очки солнцезащитные
B': очки солнцезащитные детские
```

«Детские» подтверждается B, но строковое сходство с A уменьшается. Такой ответ
identity-safe, однако не достигает A2 hardening objective.

### 9.4. Опасное копирование числа

```text
A: охлаждающая жидкость niro coolant green -40c g11 5кг
B: totachi охлаждающая жидкость niro coolant green -40c 1кг
B': totachi охлаждающая жидкость niro coolant green -40c g11 1кг
```

`G11` пришло из A и не было подтверждено B.

```text
A: дивидик губка для обуви 1 шт
B: дивидик / губка для обуви
B': Дивидик / губка для обуви, 1 шт
```

Количество было скопировано из A.

### 9.5. Потеря защищённого факта

```text
A: оплетка на руль для proton wira ... искусственная кожа 13
B: не определен оплетка наруль для proton wira ... искусственная кожа 64
B': Оплетка наруль для Proton Wira ... искусственная кожа 64
```

Этот пример требует отдельного аудита token-level валидатора: визуально `64`
сохранено, но deterministic check сообщил `lost_numeric_fact`. Подобные случаи
нужно рассматривать как потенциальные validator false positives.

---

## 10. Предлагаемая архитектура prompt v2

Рекомендуется не просить Qwen самостоятельно определять допустимые факты.
Перед API-вызовом код должен вычислять:

### `SAFE_SHARED_FACTS`

Факты, которые:

1. видны в A;
2. подтверждены `name2` или `attributes2`;
3. отсутствуют или иначе сформулированы в текущем `name2`;
4. их безопасно использовать для сближения формы B' с A.

### `PROTECTED_B_FACTS`

Факты B, которые обязаны сохраниться дословно или семантически:

- бренд B;
- модель;
- артикул;
- все числа;
- размеры и единицы;
- цвет;
- количество;
- совместимость;
- иные identity-critical значения.

### `FORBIDDEN_A_ONLY_FACTS`

Факты, видимые в A, но не подтверждённые B. Их нельзя переносить ни при каких
условиях.

Если `SAFE_SHARED_FACTS` пуст и одной перестановки слов недостаточно для
нетривиального hardening, пару лучше отфильтровать локально и вообще не
отправлять в API.

---

## 11. Предлагаемый system prompt v2

```text
Ты создаёшь сложную отрицательную пару для модели product matching.

A и B — два разных товара. Их различие подтверждено human-разметкой.
Нужно переписать только название B. Новое название B' должно описывать тот же
самый товар, что B, но по форме и порядку слов стать ближе к A.

Тебе переданы три проверенных кодом списка:

1. SAFE_SHARED_FACTS — факты, которые видны в A и одновременно подтверждены B.
   Только их можно добавлять в название B.
2. PROTECTED_B_FACTS — факты B, которые обязательно должны сохраниться.
3. FORBIDDEN_A_ONLY_FACTS — факты A, которые не принадлежат B. Их нельзя
   добавлять ни в какой форме.

Разрешено:
- использовать стиль и порядок слов A;
- добавлять только факты из SAFE_SHARED_FACTS;
- переставлять уже существующие слова B;
- удалять только явно указанные технические или мусорные префиксы.

Запрещено:
- добавлять факт, отсутствующий в name B, attributes B и SAFE_SHARED_FACTS;
- изменять или удалять PROTECTED_B_FACTS;
- копировать модель, артикул, число, размер, цвет, количество или совместимость
  из FORBIDDEN_A_ONLY_FACTS;
- возвращать только изменение регистра, пробелов, пунктуации или разделителя;
- делать B' эквивалентным B после нормализации;
- превращать B в товар A.

Если безопасного существенного hardening нет, не выдумывай изменение и верни
status = skip.

Перед ответом самостоятельно проверь:
- все PROTECTED_B_FACTS сохранены;
- ни один FORBIDDEN_A_ONLY_FACT не перенесён;
- изменение не является простой нормализацией;
- новое название действительно стало ближе к A.

Не выводи рассуждение. Верни только один JSON-объект:

Успех:
{"status":"ok","rewritten_name2":"...","used_safe_facts":["..."]}

Безопасное преобразование невозможно:
{"status":"skip","reason":"no_safe_nontrivial_hardening"}
```

---

## 12. Предлагаемый dynamic user message v2

```text
Категория: {category}

Название A:
{name1}

Название B:
{name2}

Неизменяемые attributes B:
{attributes2}

SAFE_SHARED_FACTS:
{safe_shared_facts}

PROTECTED_B_FACTS:
{protected_b_facts}

FORBIDDEN_A_ONLY_FACTS:
{forbidden_a_only_facts}

Разрешённые к удалению мусорные префиксы:
{removable_surface_noise}
```

---

## 13. Few-shot v2

Нужно 8–10 примеров, каждый обязательно должен содержать те же поля, что и
реальный запрос, включая `attributes2` и три вычисленных списка.

Рекомендуемый состав:

- 4–5 качественных `status=ok`;
- 3–4 правильных `status=skip`;
- пример с разными артикулами;
- пример с разными моделями или размерами;
- пример, где общий бренд есть в A и attributes B, поэтому его можно добавить;
- пример, где число есть только в A и поэтому его перенос запрещён;
- разные категории и разные формы единиц измерения.

Нельзя использовать validation examples. Все few-shot sources должны быть
train-only и иметь собственный manifest + SHA-256.

---

## 14. Validator v2

После ответа Qwen код всё равно должен независимо проверить:

1. category не изменилась;
2. attributes B byte-for-byte не изменились;
3. все `PROTECTED_B_FACTS` сохранены;
4. ни один `FORBIDDEN_A_ONLY_FACT` не добавлен;
5. не добавлено число из A, которого нет у B;
6. B' не равно B после trivial normalization;
7. name-similarity с A действительно выросла;
8. все добавленные факты имеют evidence в B;
9. morphology/synonym logic не создаёт очевидных false positives;
10. accepted subset проходит ручной audit на малом pilot.

Следует отдельно проверить существующие false-positive candidates, особенно
`lost_numeric_fact` при изменении пробелов/дефисов и generic product-type words.

---

## 15. Уже подготовленный безопасный transport profile

Commit: `d412508`.

Файлы:

- `scripts/generate_a2_pilot_qwen.py`;
- `scripts/run_a2_pilot_qwen.ps1`.

Подготовлено, но **не запускалось**:

- `enable_thinking = false`;
- zero automatic retries;
- максимум один paid request на source pair;
- bounded concurrency: 6 workers;
- первый запрос выполняется отдельно для explicit-cache warm-up;
- затем остальные запросы идут параллельно;
- durable `request_journal.jsonl` фиксирует `submitted` до HTTP call;
- source с `submitted`, но без terminal result считается uncertain и не
  отправляется повторно автоматически;
- сохраняются provider request ID, HTTP status, elapsed time и token usage;
- manifest отдельно считает reasoning, visible completion и cached tokens;
- новый default output: `artifacts/synthetic_v2/a2_pilot_v2`;
- старый pilot и safe-v2 output нельзя случайно смешать.

Статически проверены синтаксис Python и PowerShell. API smoke test не выполнялся
по прямой просьбе пользователя.

### Важная оговорка про кэш

Alibaba explicit cache требует достаточно длинный стабильный prefix (в
актуальной документации минимум 1024 tokens). Текущий v1 stable prefix может
быть короче. Prompt/few-shot v2, вероятно, станет длиннее, но cache hit нужно
обязательно подтвердить маленьким разрешённым smoke test по полю
`usage.prompt_tokens_details.cached_tokens`. Сейчас эффективность кэша не
доказана.

---

## 16. Предлагаемый следующий эксперимент после обсуждения

### Шаг A — полностью локально, без API

1. Утвердить формулировку prompt v2.
2. Определить точные правила построения `SAFE_SHARED_FACTS`,
   `PROTECTED_B_FACTS`, `FORBIDDEN_A_ONLY_FACTS`.
3. Собрать few-shot v2 только из human train.
4. Добавить поддержку `status=skip` в parser/validator.
5. Исправить обнаруженные validator false positives.
6. Выполнить dry-run без ключа и проверить manifests/identity.

### Шаг B — дешёвый pilot v2

После отдельного разрешения пользователя:

- 20–30 source pairs;
- thinking disabled;
- one request/source;
- сначала один cache warm-up request;
- затем ограниченная параллельность;
- без автоматического retry ошибок.

Предлагаемый gate перед масштабированием:

- `0` critical identity violations;
- reasoning tokens = `0`;
- transport attempts не больше числа source pairs;
- cache hits фактически подтверждены или кэш осознанно отключён;
- accepted rate ориентировочно хотя бы `50%`;
- большинство accepted имеет положительный deterministic similarity delta;
- затем frozen-Frida hardening score показывает полезный сигнал.

### Шаг C — масштабирование

Существующий pool содержит только 500 sources. При одном результате на source
теоретический максимум — 500 accepted, практический будет ниже. Для большего
набора нужно выбрать одно из решений:

1. расширять human-train source pool;
2. генерировать несколько строго контролируемых вариантов на source;
3. использовать меньшее число accepted, но обоснованно балансировать и
   взвешивать их при обучении.

Только после получения нескольких сотен качественных, разнообразных и
Frida-hard examples имеет смысл тратить Kaggle quota на comparative training.

---

## 17. Вопросы для обсуждения с GPT

1. Достаточно ли строковой цели `new_name_similarity > old_name_similarity`,
   или generator нужно напрямую оптимизировать под frozen-Frida score?
2. Как лучше формализовать `SAFE_SHARED_FACTS`: exact tokens, morphology,
   normalized attribute values или semantic matcher?
3. Какие факты считать identity-critical по каждой категории?
4. Можно ли безопасно удалять marketplace noise вроде `не определен`,
   seller-prefix и `/`, и когда такое изменение слишком тривиально?
5. Какие `status=skip` examples лучше всего обучат модель не выдумывать ответ?
6. Нужно ли возвращать `used_safe_facts`, или достаточно одного B' и
   независимого валидатора?
7. Стоит ли сначала локально отбрасывать sources без safe shared facts, чтобы
   вообще не платить за бесполезный API call?
8. Сколько accepted synthetics нужно для минимально осмысленного training run
   при 256006 human-train pairs?
9. Делать один вариант на source или несколько вариантов с разными безопасными
   surface transformations?
10. Как балансировать пять категорий и не переобучить модель на ювелирный
    шаблон `металл + проба`?
11. Нужно ли сохранять thinking полностью выключенным или сравнить небольшой
    pilot `thinking=false` с малым `thinking_budget`?
12. Как гарантировать cache prefix >= 1024 tokens без бессмысленного padding и
    без ухудшения качества промпта?

---

## 18. Текущее решение и стоп-точка

Сейчас:

- первая сотня завершена и проверена;
- 19 accepted недостаточно для meaningful training;
- обучение не запускалось;
- дополнительная платная генерация не запускалась;
- transport profile v2 сохранён, но API-тест не выполнялся;
- automation `monitor-frida-synthetic-v2` приостановлена;
- следующий запуск должен происходить только после согласования prompt v2,
  требуемого объёма и бюджета.

