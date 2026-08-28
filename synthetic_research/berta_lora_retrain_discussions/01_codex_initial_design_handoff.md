# Handoff для design review: BERTA + новая LoRA на human и A2

Дата: **2026-08-25**  
Репозиторий: `CodeDealer228/ozon_matching`  
Ветка: `synthetic`

Этот документ предназначен для обсуждения с другой LLM. Секретов здесь нет.
Никакое новое обучение и никакой Kaggle run на момент его создания не
запускались.

---

## 1. Задание тимлида

> Взять топовую BERTA+LoRA, смёржить LoRA в основные веса, поверх получившейся
> базы создать новую LoRA, продумать несколько конфигураций смешивания
> synthetic/real data и весов loss, затем обучить варианты на TPU.

Цель — проверить уже не «может ли линейная голова слегка сдвинуться», а может ли
полноценное parameter-efficient обучение encoder использовать A2 hard negatives.

---

## 2. Почему первый эксперимент был плохой проверкой идеи

Первый A2-эксперимент был сделан на Frida, а не на BERTA:

- всего 195 синтетических отрицательных пар;
- encoder полностью заморожен;
- обучалась только classifier head;
- synthetic повторялся 10 раз с весом 0.5;
- результат A2 против matched control: всего `+0.000022486` macro AP.

То есть эксперимент проверял очень узкий head-only режим на маленьком наборе.
Он почти ничего не говорил о способности encoder научиться распознавать общий
паттерн A2. Комментарий тимлида, что обучение было сделано плохо, обоснован.

Новый эксперимент не должен повторять эту ошибку: LoRA encoder и classifier
обучаются вместе, а данных теперь на два порядка больше.

---

## 3. Что означает «топовая BERTA+LoRA»: найденная неоднозначность

В проекте есть две разумные трактовки.

### 3.1. Лучший подтверждённый реальный submission

`maxamsecret/berta-tpu-lora-top-checkpoint`:

- exact real test score: `0.47` — лучший подтверждённый BERTA submission;
- `global_step=11211`;
- old validation macro AP: `0.7087279672868713`;
- adapter и classifier побайтово сопоставлены с отправленным submission в
  `reports/berta_tpu_lora_provenance/README.md`.

### 3.2. Лучший checkpoint по корректно сохранённой локальной validation

`maxamsecret/berta-human-continue-384-best-checkpoint`:

- продолжает именно checkpoint выше на human-only данных;
- epoch 1, `global_step=1142`;
- exact `val_macro_pr_auc=0.7550802786405144`;
- эпохи 2–10 монотонно ухудшались, поэтому epoch 1 действительно лучший;
- датасет на Kaggle существует и содержит adapter config, adapter weights,
  classifier и точный `train_state.json`;
- такого же подтверждённого real test score у него нет.

### Первоначальная рекомендация Codex

Для задачи тимлида использовать **human-continue epoch 1**: это наиболее свежая
и лучшая по чистой validation BERTA+LoRA, а её Kaggle dataset прямо называется
`best-checkpoint`. Реальный test-top 0.47 сохранить как внешний anchor и явно не
утверждать заранее, что epoch 1 лучше на лидерборде.

Если под словом «топовая» тимлид имел в виду строго модель с test=0.47, базу
нужно заменить до реализации. Смешивать две базы внутри одного сравнения нельзя:
тогда эффект A2 будет спутан с эффектом checkpoint.

---

## 4. Как правильно сделать merge

1. Загрузить исходную BERTA-базу, на которой обучался выбранный adapter.
2. Загрузить PEFT adapter в режиме inference.
3. Выполнить `merge_and_unload(safe_merge=True)` в fp32.
4. Сохранить plain encoder, tokenizer, source adapter/classifier SHA-256 и
   `merge_meta.json`.
5. Classifier не выбрасывать: warm-start из выбранного checkpoint нужен, чтобы
   модель на шаге 0 воспроизводила исходные logits.
6. Поверх merged encoder создать **свежую** LoRA с одинаковой инициализацией во
   всех сравниваемых конфигурациях.

Для рекомендуемого human-continue checkpoint пункт 1 особенно важен: его LoRA
обучалась не прямо поверх `sergeyzh/BERTA`, а поверх уже смёрженного encoder из
`maxamsecret/berta-merge-lora`. Сначала нужно загрузить именно тот
`merged_encoder`, затем применить human-continue adapter и только после этого
сделать второй merge. Применение adapter прямо к исходной `sergeyzh/BERTA`
сломает цепочку весов, даже если формы тензоров формально совпадут.

Предлагаемый LoRA contract, пока GPT не предложит лучше:

- `r=16`;
- `alpha=32`;
- dropout `0.15`;
- target modules: `query`, `key`, `value`, `dense`;
- LoRA и classifier trainable, merged base frozen;
- pair length `384`;
- per-core batch `32` на TPU v5e-8, global batch `256`.

### Обязательная merge-equivalence проверка

До обучения сравнить исходную adapter-модель и merged-модель:

- фиксированный probe минимум 512 пар;
- одинаковый tokenizer/pair assembly;
- max absolute logit difference должен быть малым и заранее записанным;
- full validation AP до и после merge должна совпадать в пределах численной
  погрешности.

Если эта проверка не проходит, TPU training запускать нельзя.

---

## 5. Данные и критически важный split

Историческая BERTA обучалась и оценивалась на split:

- human train: `292523`;
- validation: `73131`;
- train labels: 217411 negative / 75112 positive;
- validation labels: 54353 negative / 18778 positive;
- human-train `pos_weight = 2.894490893598893`.

Split уже сохранён как `data/splits/matches_split_seed42.parquet` и должен быть
использован байт-в-байт для сопоставимости с BERTA lineage.

Новая A2 generation дала 12833 accepted строк во всех 20 категориях. После
сопоставления source-пар с историческим BERTA split:

- 10123 synthetic rows происходят из BERTA human train;
- 2710 происходят из BERTA validation и **обязаны быть исключены** из обучения;
- чистые 10123 строки соответствуют 5886 уникальным source orientations;
- в каждой категории после фильтра остаётся 364–795 строк.

Это не quality-фильтр и не попытка снова чрезмерно отбраковывать синтетику. Это
обычная защита от прямой утечки: нельзя учиться на переписанной версии пары,
которая находится в validation.

Текущий item-disjoint split 256006/109648 полезен как вторичная диагностика, но
не является чистой primary validation для исторического BERTA checkpoint:
старые BERTA-веса уже обучались на другом 80/20 row split и могли видеть часть
его пар/товаров. Primary comparison должен использовать старые untouched 73131.

Synthetic item tokens нельзя брать из старого cache: `name2_prime` — новый
текст. Их нужно токенизировать отдельно с тем же prefix и tokenizer contract.

---

## 6. Предлагаемый причинный эксперимент

Главная опасность — спутать пользу A2 с дополнительными шагами на отрицательных
примерах. Поэтому synthetic treatment должен иметь matched control.

### Общие настройки всех обучаемых вариантов

- одна и та же merged base;
- одна и та же classifier warm-start;
- одна и та же свежая LoRA initialization и seed;
- одинаковый human train;
- одинаковый порядок human-части;
- одинаковое число replay-строк, шагов и sample weights;
- `pos_weight` фиксируется по исходному human train (`2.894490893598893`) и не
  пересчитывается после добавления synthetic negatives;
- unreduced BCE сначала учитывает `pos_weight`, затем умножается на per-row
  `sample_weight`, нормируется по сумме весов.

### C0 — step-0 / no-training anchor

Merged model + нулевая свежая LoRA + исходный classifier. Нужен для проверки,
что merge и создание новой LoRA сами по себе ничего не испортили.

### C1 — matched human-source replay

- полный human train;
- детерминированная replay-последовательность исходных negative source pairs;
- 4 replay exposure на synthetic row;
- replay weight `0.5`.

### T1 — A2 replay, умеренный вес

Полностью совпадает с C1 по source, порядку, label, числу строк и весу, но
исходный `name2` заменён на `name2_prime`.

Это главное причинное сравнение: **T1 − C1**.

### T2 — A2 replay, сильный вес

То же, что T1, но synthetic weight `1.0`. Этот вариант проверяет, не слишком ли
сильно 0.5 подавляет сигнал. Он имеет больший эффективный вклад A2 и потому
сравнивается прежде всего с T1, а не только с C1.

### T3 — category-balanced A2 (второй этап)

Если T1 даёт правильное направление, построить одинаковое число replay draws
на каждую из 20 категорий, с replacement внутри малых категорий. Для честного
контроля рядом нужен category-balanced вариант исходных source pairs. Не стоит
сразу сжигать TPU-квоту на T3, пока T1 не показал хотя бы сигнал.

### Почему не просто concat 10123 строк один раз

Это только 3.46% от human train и, вероятно, слишком слабый сигнал. Четыре
exposure дают 40492 replay rows; при весе 0.5 эффективная масса равна примерно
20246 обычным строкам. При этом matched source replay отделяет эффект нового
текста от эффекта дополнительных negative steps.

---

## 7. Оптимизация и остановка

Human-continue checkpoint уже начинал переобучаться после одной эпохи. Поэтому
не следует снова планировать 10 эпох.

Первоначальное предложение:

- максимум одна эпоха общего датасета;
- LR LoRA/classifier `5e-5` как более осторожный, чем исторические `2e-4`;
- cosine schedule, warmup 10%;
- validation на 25%, 50%, 75% и 100% эпохи;
- каждый checkpoint в отдельной директории;
- best weights сохраняются отдельно, metric никогда не переносится от других
  weights;
- early stop, если два последовательных eval хуже step-0/C1 и тренд падающий.

LR здесь сознательно не превращён в ещё одну ось матрицы: сначала нужно
изолировать data effect. Если GPT считает `5e-5` слишком маленьким для свежей
LoRA, нужно предложить один общий LR для всех вариантов и обосновать его.

---

## 8. Метрики и критерий решения

Для каждого checkpoint сохранить:

- 73131 validation predictions в одинаковом порядке;
- macro AP и micro AP;
- AP по 20 категориям;
- loss и learning rate;
- source/base/adapter/classifier/data/split SHA-256;
- paired differences T1−C1 и T2−T1;
- paired bootstrap по validation predictions, чтобы отличать сигнал от шума.

Успех A2 нельзя объявлять только по training loss. Основной вопрос:

```text
T1 лучше C1 на macro AP и не создаёт широкую деградацию категорий?
```

Если T1 и T2 оба хуже C1, текущий A2 dataset/recipe не подтверждён. Если T1
лучше, но T2 хуже, сигнал полезен только при слабом весе. Если T2 лучше T1,
можно переходить к category balancing и/или большему объёму.

---

## 9. Как запускать на Kaggle после согласования

Предпочтительный безопасный порядок:

1. Маленький merge kernel создаёт один immutable merged checkpoint dataset и
   equivalence report.
2. Один общий training kernel template параметризуется `C1/T1/T2`, но каждый
   вариант запускается отдельной версией/отдельным kernel output из одной и той
   же базы.
3. Сначала C1 и T1. T2 запускается после sanity-check первых двух, T3 — только
   при положительном сигнале.
4. Никогда не запускать duplicate, пока TPU run QUEUED/RUNNING.
5. Все checkpoints, predictions, manifests и метрики сохранять как persistent
   output и скачивать после COMPLETE.

Отдельные TPU runs безопаснее одного процесса, который последовательно создаёт
несколько XLA-моделей: в истории проекта уже был cross-epoch HBM creep.

---

## 10. Вопросы внешнему GPT

Пожалуйста, выступи как строгий design reviewer и ответь конкретно:

1. Какой source checkpoint считать «топовым»: test-proven 0.47 или
   human-continue epoch1 с val 0.75508? Почему?
2. Правильно ли warm-start classifier после merge, или его следует обнулить?
3. Достаточно ли причинного сравнения C1/T1/T2? Что убрать или добавить при
   ограниченной TPU-квоте?
4. Разумны ли 4 exposure и веса 0.5/1.0 для 10123 clean synthetic rows?
5. Нужно ли сразу балансировать категории или сначала сохранить естественное
   распределение accepted A2?
6. Какой единый LR и число шагов выбрать для свежей LoRA поверх уже сильной
   merged base?
7. Есть ли утечка или иной confound, который пропущен в этом плане?
8. Предложи финальную минимальную матрицу запусков и чёткое правило выбора
   победителя.

Важно: не предлагай использовать 2710 synthetic rows, происходящих из BERTA
validation, и не оценивай эксперимент только по текущему item-disjoint split,
который historical checkpoint уже частично видел.
