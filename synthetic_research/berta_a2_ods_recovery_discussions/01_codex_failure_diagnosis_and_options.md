# BERTA + A2 после ODS: диагноз Codex и варианты восстановления

Дата: 26 августа 2026.

Это первый документ нового цикла обсуждения. Предлагаемый формат папки:

1. `01_codex_failure_diagnosis_and_options.md` — факты, диагноз и варианты;
2. `02_gpt_design_review.md` — независимое ревью GPT;
3. `03_codex_final_recovery_plan.md` — итоговое решение после обсуждения;
4. последующие файлы — результаты утверждённых экспериментов.

Новые training или ODS-запуски этим документом не разрешаются автоматически.

## 1. Что произошло

ODS submission `matching-berta-a2-t1-step651-submit.zip` завершился успешно, но
получил Mean PR-AUC:

`0.4513363059`

Это примерно на `0.019` ниже исторического результата исходного BERTA
submission (`~0.47`). На локальной historical validation T1, наоборот, выросла:

- source/step 0 macro AP: `0.7086041124`;
- T1 step 651 macro AP: `0.7466399981`;
- source/step 0 micro AP: `0.7759352856`;
- T1 step 651 micro AP: `0.8071911854`.

Следовательно, текущая primary validation неправильно предсказала направление
реального ODS-изменения.

## 2. C1 уже подготовлен, но не отправлен

Локально собран matched-control submission:

`submits/berta/a2_c1_step651/matching-berta-a2-c1-step651-submit.zip`

- размер: `499,549,520` bytes;
- SHA-256:
  `eea006d728fb39f7f9a9d1a71036eab734ebe81483c72eca9a6b4c83d6a8d067`;
- ZIP integrity: passed;
- encoder, C1 adapter и classifier совпадают с проверенными Kaggle outputs;
- на ODS не отправлялся.

C1 нужен для причинного диагноза, но расходовать один из двух оставшихся
суточных submission slots без команды не следует.

Интерпретация будущего C1 score:

| Результат C1 | Вывод |
|---|---|
| C1 тоже около `0.45` | Главный вред даёт общий retraining/replay recipe или неверная validation, не A2 сама по себе |
| C1 около старых `0.47`, T1 `0.451` | Текущая A2 синтетика непосредственно вредит hidden test |
| T1 выше C1, но обе хуже source | A2 частично полезна, но не компенсирует вред общего retraining |
| C1 хуже T1 | A2 даёт реальный положительный эффект, однако training recipe всё равно непригоден |

## 3. Первый установленный дефект: A2 не была сложной для BERTA

Перед обучением все `10,123` clean A2 были оценены frozen source BERTA. Результат
сохранился, но по утверждённому дизайну использовался только как diagnostic и не
фильтровал training data (`filtered_by_hardness = false`).

Фактическое распределение изменения logit synthetic minus original:

- положительный delta: `51.98%`;
- отрицательный delta: `47.81%`;
- median delta: `+0.01758`;
- mean delta: `+0.01071`;
- delta больше `+0.1`: `41.10%`;
- delta больше `+0.5`: только `14.74%`;
- delta меньше `-0.1`: `37.15%`.

Иными словами, почти половина synthetic rows делала отрицательную пару легче
для BERTA, а не сложнее. Большой training набор по числу строк не был большим
набором полезных BERTA-hard примеров.

## 4. Второй дефект: несколько вариантов одного source одновременно шли в train

`10,123` synthetic rows соответствуют только:

- `5,886` unique source orientations;
- `5,818` unique unordered source pairs.

Распределение вариантов:

- `1,649` source orientations имеют один synthetic candidate;
- `4,237` source orientations имеют два candidates;
- оба варианта одного source попадали в train как отдельные строки.

Если для каждого source оставить только лучший по frozen-BERTA delta вариант:

- останется `5,886` строк;
- только `3,530` из них имеют delta больше нуля;
- `2,356` sources вообще не имеют ни одного усиливающего BERTA candidate;
- у positive best-per-source subset mean delta `+0.10775`, median `+0.08398`;
- `1,065` строк имеют delta больше `+0.5`.

Значит, уже существующие данные позволяют бесплатно собрать намного более
сфокусированный BERTA-aware subset примерно из `3,530` строк. Новая Alibaba
generation для первого recovery experiment не обязательна.

## 5. Третий дефект: historical validation имеет общие товары с train

Primary BERTA split `matches_split_seed42.parquet` сохранялся ради точного
сравнения с historical BERTA lineage. Однако этот split является построчным, а
не item-disjoint:

- train rows: `292,523`;
- validation rows: `73,131`;
- unordered pair overlap: `0`;
- общих item IDs между train и validation: `5,534`;
- `7,727` train rows содержат item, встречающийся в validation.

Это не прямой duplicate-pair leakage, но это сильный канал запоминания товаров.
После ещё одной эпохи human replay модель могла заметно улучшиться на знакомых
validation items и одновременно ухудшиться на новых hidden-test items.

В утверждённом дизайне item-disjoint split `256,006 / 109,648` с нулевым item
overlap был назван обязательной secondary diagnostic. Он не был рассчитан для
C1/T1 перед ODS submission. Это главный недостаток нашей evaluation-схемы.

Historical validation всё ещё нужна для сравнимости с source lineage, но больше
не должна единолично выбирать practical submission candidate.

## 6. Что именно произошло при retraining

Training manifest:

- human rows: `292,523`;
- replay rows: `40,492` = `10,123 × 4`;
- synthetic sample weight: `0.5`;
- effective synthetic loss share: примерно `6.5%`;
- полный training: `651` optimizer step.

C1 и T1 на historical validation почти идентичны:

- Pearson correlation final logits: `0.99956`;
- T1 minus C1 macro AP: `+0.00033645`;
- T1 minus C1 micro AP: `+0.00031697`.

Основной локальный рост создала не синтетика, а повторное обучение на historical
human train. C1 уже поднялась с micro AP `0.77594` до `0.80687`.

На historical validation C1:

- уменьшила mean negative logit с `-1.753` до `-2.919`;
- увеличила mean positive logit с `1.543` до `1.927`;
- уменьшила false-positive rate при threshold 0.5 с `19.61%` до `13.55%`;
- одновременно уменьшила true-positive rate с `82.48%` до `80.17%`.

Это сильная перестройка модели, а не малое A2-добавление. Hidden test показал,
что локальное улучшение разделения не перенеслось.

T1 относительно C1 изменила logits совсем немного и в среднем сдвинула их
вверх:

- общий mean shift: `+0.0417`;
- positive rows: `+0.0252`;
- negative rows: `+0.0473`.

То есть A2 effect на BERTA оказался слабым и не сводится к ожидаемому простому
«прижать hard negatives вниз».

## 7. Category и generator diagnostics

Frozen-BERTA hardness по категориям почти не объясняет validation T1-C1 gain:

- correlation category mean delta-logit с category delta-AP: `0.011`;
- correlation positive-delta fraction с category delta-AP: `0.263`.

Это означает, что простой порог delta больше нуля полезен как первый фильтр, но
не гарантирует downstream AP gain сам по себе.

Generator models также неодинаковы. Mean frozen-BERTA delta был отрицательным у:

- `qwen3.5-27b`: `-0.0723`;
- `qwen3-8b`: `-0.0683`;
- `qwen3.6-27b`: `-0.0142`.

Лучшие средние значения дали:

- `qwen3.6-35b-a3b`: `+0.0768`;
- `qwen3-30b-a3b-instruct-2507`: `+0.0477`;
- `qwen3-32b`: `+0.0366`;
- `qwen3-30b-a3b`: `+0.0363`.

Это аргумент за model-aware candidate selection, а не за безусловное смешивание
всех прошедших safety-validator генераций.

## 8. Рабочие гипотезы в порядке вероятности

### H1. Главная проблема — общий retraining на historical human split

Вероятность: высокая.

Обоснование: почти весь локальный gain появился уже в C1; primary validation
делит тысячи товаров с train; ODS рухнул несмотря на большой absolute val gain.

### H2. A2 data была недостаточно BERTA-aware

Вероятность: высокая.

Обоснование: `47.8%` rows снизили frozen-BERTA score; hardness не использовалась
для отбора; несколько candidates одного source одновременно усиливали его вес.

### H3. Текущий A2 лучше соответствует Frida, чем BERTA

Вероятность: средняя.

Обоснование: исходная идея и ранняя effectiveness-проверка строились вокруг
Frida. Но крупная 10k партия уже не проходила Frida selection, поэтому считать
её полностью Frida-targeted тоже нельзя. Нужен frozen-Frida rescore.

### H4. Submission packaging вызвал падение

Вероятность: низкая, но не нулевая.

Обоснование против: archive успешно исполнился; inference code и offline wheels
побайтово взяты из ODS-tested BERTA submission; encoder/adapter/classifier hashes
совпали; training и inference используют mean pooling и pair length 256.

Окончательно исключить packaging можно, если прогнать submission wrapper на
фиксированном локальном/Kaggle probe и сравнить logits с checkpoint evaluation.

## 9. Варианты восстановления

### Вариант A — рекомендованный: сначала починить evaluation и reuse текущей A2

1. Не тратить ODS slot до решения команды по C1.
2. Посчитать step-0, C1 и T1 на item-disjoint validation без общих товаров.
3. Собрать из текущих A2 ровно один best candidate на source и только при
   positive frozen-BERTA delta; стартовый объём около `3,530`.
4. Не повторять полный historical human epoch. Использовать более короткий и
   слабый adaptation recipe с anchor против ухода от source model.
5. Выбирать checkpoint по совокупности historical и item-disjoint metrics, а не
   по одной historical validation.

Плюсы: не требует Alibaba, проверяет главные найденные дефекты, минимальная
стоимость. Минус: positive frozen delta не гарантирует AP gain.

### Вариант B — BERTA-aware regeneration

1. Источники выбирать по ошибкам/уверенности frozen BERTA на human train.
2. Генерировать несколько candidates на source.
3. Safety проверять детерминированно, но оставлять максимум один вариант.
4. Из safety-safe вариантов выбирать candidate, который действительно повышает
   frozen-BERTA score на отрицательной паре.
5. Ограничивать повторы одинаковых типов преобразований и source families.

Плюсы: данные соответствуют обучаемой модели. Минусы: новые расходы Alibaba и
риск переоптимизации под frozen teacher.

### Вариант C — перенести текущую методику на Frida

1. Оценить все clean A2 frozen top Frida.
2. Оставить один positive-delta candidate на source.
3. Сделать matched control/treatment с минимальным уходом от source Frida.
4. Primary practical gate строить на item-disjoint validation.

Плюсы: ближе к исходной идее A2. Минусы: крупная партия не была специально
отобрана Frida, а новый training всё равно может переобучиться.

### Вариант D — только исправить training recipe

Возможные механизмы:

- меньшие LR и меньше шагов;
- lower human replay share вместо полной ещё одной эпохи;
- logit anchoring/distillation к source BERTA на обычных human rows;
- отдельный маленький LoRA для A2 при замороженном classifier либо наоборот;
- checkpoint selection по item-disjoint metric;
- category-aware weights только после устойчивого train-derived evidence.

Этот вариант нужен даже при качественной генерации, потому что T1 test decline
слишком велик, чтобы объяснить его только `6.5%` effective synthetic loss.

## 10. Предварительная рекомендация Codex

До независимого ревью GPT рациональный порядок такой:

1. сохранить C1 ZIP, но не отправлять без команды;
2. подготовить item-disjoint evaluator для source/C1/T1;
3. локально собрать candidate manifest `BERTA-best-one-positive-per-source` из
   существующих данных, ничего не обучая;
4. проверить frozen Frida на тех же clean A2;
5. после этих двух diagnostic axes выбрать BERTA recovery либо Frida route;
6. новую Alibaba generation не начинать, пока не доказано, что существующих
   `3,530` BERTA-hard либо Frida-hard примеров недостаточно.

## 11. Вопросы для независимого GPT review

1. Согласен ли reviewer, что главный evidence указывает на retraining/validation
   failure, а не только на генерацию?
2. Достаточно ли item-disjoint secondary evaluation, учитывая, что source BERTA
   исторически видела часть этих items в другой split lineage?
3. Как лучше построить anchor, чтобы fresh LoRA не разрушала hidden-test
   перенос: logit distillation, L2 к source weights или короткий low-LR run?
4. Следует ли использовать positive frozen-BERTA delta как hard filter либо как
   continuous sample weight?
5. Нужен ли один candidate на source всегда, или можно оставить два при сильно
   разных surface operations?
6. Что информативнее при двух ODS slots: C1 step 651 или новый source-anchored
   BERTA-hard pilot после item-disjoint проверки?
7. Стоит ли сначала проверить текущую A2 на Frida, прежде чем тратить средства
   на BERTA-aware regeneration?
