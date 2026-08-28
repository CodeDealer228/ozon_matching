# Design review: обучение BERTA на A2 synthetic hard negatives

Дата состояния: **2026-08-25**  
Репозиторий: `CodeDealer228/ozon_matching`  
Ветка: `synthetic`

Этот документ предназначен для обсуждения training design с Codex **до запуска TPU-обучения**.

На текущем шаге Codex должен выступить как строгий design reviewer: изучить актуальный код/артефакты, проверить предложенный эксперимент, найти confound'ы и вернуть заключение.  
**Не запускать обучение и не менять production/training code до отдельного подтверждения пользователя.**

---

# 1. Source checkpoint: использовать модель по ссылке Макса

Макс явно указал, что для нового эксперимента нужно использовать модель из этого submission:

https://storage.yandexcloud.net/ds-ods/files/submissions/0d0183cf-8864-4331-88be-f78da0c68dd2/2e2ad7b0/matching-berta_tpu_lora-submit.zip

Это считается source-of-truth со стороны тимлида.

В текущем research handoff эта lineage соответствует BERTA+LoRA, а не Frida:

```text
maxamsecret/berta-tpu-lora-top-checkpoint
```

Для неё ранее был зафиксирован лучший подтверждённый BERTA real submission score `0.47`.

Перед merge Codex обязан проверить реальную provenance chain в repository:

- какая exact BERTA base использовалась;
- какой exact LoRA adapter породил submission;
- какой classifier использовался;
- какие tokenizer/pair-assembly settings использовались;
- какие SHA / manifests / provenance reports уже есть.

Нельзя подменять эту модель:

- `human-continue epoch1`;
- Frida;
- другой BERTA checkpoint с более высокой локальной validation;
- checkpoint с похожим названием без доказанной связи с submission Макса.

Если сам submission ZIP содержит только predictions/submission artifact, а не обучаемые веса, нужно восстановить exact training checkpoint по repository provenance и убедиться, что он действительно породил этот submission.

---

# 2. Что мы хотим проверить

Теперь задача не в том, чтобы проверить, может ли маленькая classifier head слегка подстроиться под A2.

Мы хотим проверить:

> Может ли полноценное parameter-efficient обучение encoder через новую LoRA извлечь пользу из большого набора A2 hard negatives?

Предыдущий Frida experiment был слишком слабой проверкой:

- около 195 synthetic rows;
- encoder frozen;
- обучалась только classifier head;
- synthetic сильно повторялся;
- improvement относительно matched control был практически нулевой.

Новый эксперимент должен быть принципиально сильнее:

```text
strong BERTA checkpoint
→ merge old LoRA into encoder
→ fresh LoRA
→ train fresh LoRA + classifier
→ compare matched control vs A2 treatment
```

---

# 3. Synthetic dataset: что реально доступно для BERTA training

Всего A2 generation дала:

```text
12833 accepted synthetic rows
```

Но historical BERTA train/validation split отличается от item-disjoint split, который использовался при generation research.

После сопоставления с historical BERTA split:

```text
10123 synthetic rows originate from BERTA human train
2710 synthetic rows originate from BERTA validation
```

В training можно использовать только:

```text
10123
```

Строки `2710` должны быть исключены **только по причине leakage**.

Это не quality filtering.

Нельзя обучать BERTA на synthetic rewrite пары, source которой находится в primary BERTA validation.

Оставшиеся `10123` synthetic rows соответствуют:

```text
5886 unique source orientations
```

После leakage filtering во всех 20 категориях остаются synthetic examples.

---

# 4. Очень важный принцип: больше не делать жёсткий quality-gate перед первым нормальным training experiment

Мы уже несколько дней слишком строго фильтровали synthetic data и рискуем снова прийти к слишком маленькому dataset, на котором невозможно проверить идею.

Поэтому для первого полноценного BERTA training experiment:

> **все 10123 train-clean synthetic rows идут в training treatment.**

Не делать перед этим новый gate вида:

```text
keep only BERTA delta > 0
keep only top-X% hardest
keep only high similarity
keep only top categories
keep only one variant/source
```

Исключать только:

- leakage;
- явную corruption / broken record;
- технически невалидный input;
- доказанную identity violation, если такая найдена отдельным audit.

Низкая hardness по frozen BERTA сама по себе **не является причиной reject**.

---

# 5. При этом BERTA-rescore нужен — но только как диагностика

A2 generation изначально исследовалась вокруг Frida.

Теперь обучаться будет BERTA.

Поэтому перед TPU-training полезно прогнать все `10123` clean synthetic rows через frozen top BERTA.

Для каждой source/synthetic пары вычислить:

```text
old_logit = BERTA(A, B)
new_logit = BERTA(A, B')
delta_berta = new_logit - old_logit
```

Также можно сохранить probabilities, но главным diagnostic hardness measure должен быть logit delta.

Сохранить минимум:

```text
source_id
category
name1
name2
name2_prime
old_logit
new_logit
delta_berta
old_probability
new_probability
```

Агрегаты:

```text
fraction(delta_berta > 0)
median delta
p25
p75
p90
distribution by category
distribution by number of variants/source
```

## Критически важно

Этот rescore:

```text
НЕ является training filter
```

То есть:

```text
delta_berta <= 0
```

не означает reject.

Причины:

1. A2 может учить полезной инвариантности к surface form даже там, где текущая BERTA уже классифицирует пару хорошо.
2. Filtering только по ошибкам текущей BERTA слишком сильно подгонит dataset под blind spots одного checkpoint.
3. Наша первая задача — наконец провести полноценный encoder-level experiment на полном clean A2 pool.
4. Rescore нужен для интерпретации результата, а не для очередного сокращения data.

Rescore может остановить experiment только если обнаруживает **технический баг**, например:

- scoring идёт не тем checkpoint;
- `name2_prime` не попадает в model input;
- пары перепутаны местами;
- tokenizer/pair assembly отличается от BERTA training contract;
- logits не соответствуют expected model.

---

# 6. Merge top BERTA+LoRA

Исходная модель концептуально:

```text
BERTA base
+
old trained LoRA
+
trained classifier
```

Нужно:

```text
BERTA + old LoRA
        ↓
      merge
        ↓
BERTA_merged
```

После этого old LoRA больше не является отдельным trainable adapter: её effect встроен в base weights.

Рекомендуемый merge process:

1. загрузить exact source BERTA base;
2. загрузить exact old LoRA adapter;
3. применить adapter к правильной base lineage;
4. выполнить safe merge;
5. сохранить merged encoder;
6. сохранить tokenizer;
7. сохранить source checkpoint / adapter / classifier provenance;
8. сохранить `merge_meta.json`;
9. сохранить SHA-256 relevant artifacts.

Classifier **не обнулять**.

Он должен warm-start из top checkpoint Макса.

---

# 7. Merge-equivalence check — обязательный стоп-гейт

До обучения нужно доказать, что merge ничего не сломал.

Сравнить:

```text
OLD:
base + old LoRA + old classifier

NEW:
merged base + same old classifier
```

Сначала deterministic probe минимум на 512 парах:

```text
max_abs_logit_diff
mean_abs_logit_diff
```

Затем полный historical BERTA validation.

Должны практически совпасть:

```text
macro AP
micro AP
per-category AP
```

в пределах численной погрешности.

Если equivalence не проходит:

```text
НЕ запускать TPU training
```

Сначала найти ошибку в provenance / base / merge / tokenizer / classifier.

---

# 8. После merge создать новую fresh LoRA

Training model:

```text
BERTA_merged   → frozen
fresh LoRA     → trainable
classifier     → trainable
```

Fresh LoRA должна стартовать одинаково во всех сравниваемых arms.

Все arms используют:

- один и тот же merged base;
- один и тот же classifier warm-start;
- одну и ту же fresh-LoRA architecture;
- одинаковую initialization;
- одинаковый seed;
- одинаковый tokenizer;
- одинаковый pair assembly;
- одинаковый historical human split.

---

# 9. LoRA architecture: не изобретать новую без причины

В старом handoff предварительно обсуждалась конфигурация:

```text
r = 16
alpha = 32
dropout = 0.15
target_modules = query, key, value, dense
```

Но это пока не считается окончательным решением.

Перед фиксацией новой LoRA Codex должен открыть exact `adapter_config.json` top BERTA Макса.

Default recommendation:

> Если historical top LoRA использовала разумную и совместимую архитектуру, fresh LoRA должна повторять её topology/config.

Причина:

мы хотим исследовать **data effect**, а не одновременно менять:

```text
data
+
LoRA rank
+
target modules
+
dropout
```

Если Codex считает, что historical config нельзя/не стоит повторять, он должен объяснить конкретную техническую причину.

---

# 10. Historical BERTA split — primary split

Primary comparison должен использовать historical BERTA split:

```text
human train: 292523
validation:   73131
```

Не использовать item-disjoint split как primary metric для этого checkpoint.

Причина:

historical BERTA уже обучалась в другой split lineage и могла видеть часть items/pairs из нового item-disjoint split.

Item-disjoint split можно использовать только как secondary diagnostic.

---

# 11. Main causal experiment: C1 vs T1

Это сердце experiment.

Нельзя просто сделать:

```text
human train
+
A2
```

и сравнить с old checkpoint.

Иначе невозможно понять:

- полезна ли A2;
- или помогли просто дополнительные negative updates.

Поэтому делаем matched control.

---

# 12. C0 — step-0 anchor

C0:

```text
merged BERTA
+
fresh new LoRA
+
warm-start classifier
+
zero training
```

C0 нужен, чтобы проверить:

- fresh LoRA creation ничего не испортила;
- step-0 logits ожидаемые;
- training template собирает ту же модель;
- validation до обучения совпадает с merged anchor.

C0 не требует отдельного полноценного training run, если это можно корректно проверить до optimizer step.

---

# 13. C1 — matched original-source replay

C1 получает:

```text
full historical human train
+
replay original source negatives (A, B)
```

Для каждой synthetic row T1 должна существовать matched control row C1 с тем же source.

Если один source породил несколько synthetic variants:

```text
(A, B'1)
(A, B'2)
(A, B'3)
```

то C1 получает соответствующую multiplicity:

```text
(A, B)
(A, B)
(A, B)
```

Это нормально.

В первом causal run **не нужно дополнительно схлопывать variants/source**.

Почему:

такая multiplicity полностью matched между C1 и T1 и поэтому не является confound.

---

# 14. T1 — A2 treatment

T1 получает:

```text
full historical human train
+
replay A2 rows (A, B')
```

Всё остальное совпадает с C1:

```text
same human train
same source identity distribution
same number of replay rows
same replay multiplicity
same replay order
same labels
same sample weights
same number of optimizer steps
same fresh LoRA init
same seed
same optimizer settings
same eval steps
```

Единственная meaningful treatment difference:

```text
C1 uses original name2
T1 uses synthetic name2_prime
```

Главный causal estimate:

```text
T1 - C1
```

Если T1 выигрывает C1, появляется evidence, что пользу приносит именно A2 rewrite/hardening, а не просто дополнительное повторение negatives.

---

# 15. Replay recipe для первого run

Предлагаемый первый recipe:

```text
4 replay exposures per synthetic row
sample_weight = 0.5
```

Для `10123` rows:

```text
10123 × 4 = 40492 replay presentations
```

При `weight = 0.5` effective contribution примерно соответствует:

```text
~20246 normal-weight rows
```

На фоне `292523` human train это достаточно заметный, но не доминирующий treatment.

Одинаковый replay recipe применяется и в C1, и в T1.

---

# 16. pos_weight

Использовать historical human-train:

```text
pos_weight = 2.894490893598893
```

Не пересчитывать `pos_weight` после добавления replay negatives.

Иначе synthetic/replay изменит не только data stream, но и сам class-weighting objective.

Loss contract:

```text
unreduced BCEWithLogits(pos_weight=human_pos_weight)
→ multiply by per-row sample_weight
→ normalize by sum(sample_weight)
```

Codex должен проверить корректность этой реализации в текущем training code.

---

# 17. Category balancing — не в первом run

После leakage filtering synthetic остаются во всех 20 категориях.

Первый experiment должен проверить:

> полезен ли получившийся A2 dataset в своём естественном accepted distribution?

Поэтому сейчас не делать:

```text
category oversampling
equal examples/category
replacement-based category balancing
category-specific weights
```

Если T1 даёт положительный сигнал, category balancing можно вынести во второй этап.

---

# 18. Learning rate: текущее рабочее предложение

Предлагаемый starting recipe:

```text
LoRA LR       = 1e-4
classifier LR = 5e-5
```

Логика:

- fresh LoRA должна реально успеть адаптироваться;
- classifier уже хороший и warm-started, поэтому его двигаем осторожнее;
- `5e-5` для fresh LoRA может быть слишком маленьким;
- `2e-4` может быть слишком агрессивным поверх уже сильной merged base.

Но это **не окончательная догма**.

Перед фиксацией Codex должен проверить:

- optimizer config top BERTA;
- historical LoRA LR;
- classifier LR;
- scheduler;
- warmup;
- global batch;
- effective step count.

Если есть сильная техническая причина выбрать другой LR, Codex должен предложить **один общий recipe для C1 и T1**, а не LR sweep.

Первый causal experiment не должен превращаться в hyperparameter search.

---

# 19. Scheduler / warmup

Рабочее предложение:

```text
cosine scheduler
warmup = 10%
```

Но exact scheduler лучше сверить с historical top BERTA training recipe.

Главное invariant:

```text
C1 и T1 используют абсолютно одинаковый optimizer/scheduler contract.
```

---

# 20. Batch / TPU

Если текущая infrastructure подтверждает стабильность:

```text
TPU v5e-8
per-core batch = 32
global batch = 256
```

Но Codex должен проверить реальные ограничения current kernel и memory.

Не менять batch между C1/T1.

---

# 21. Число эпох

Для первого run:

```text
maximum = 1 epoch
```

Не планировать 10 epochs.

Historical continuation уже показывал degradation после первой эпохи.

Evaluation:

```text
step 0
25%
50%
75%
100%
```

---

# 22. Не делать asymmetric early stopping

Не использовать automatic rule:

```text
если два eval хуже → остановить arm
```

потому что тогда:

```text
C1 может закончиться на 75%
T1 — на 100%
```

и сравнение перестаёт быть идеально matched.

C1 и T1 должны пройти одинаковое число optimizer steps.

Остановить run раньше можно только при настоящей технической аварии:

```text
NaN
loss explosion
broken data
OOM/HBM failure
TPU runtime error
```

---

# 23. Primary comparison — одинаковые training steps

Для causal conclusion сравнивать:

```text
C1@0    vs T1@0
C1@25%  vs T1@25%
C1@50%  vs T1@50%
C1@75%  vs T1@75%
C1@100% vs T1@100%
```

Считать:

```text
delta_A2(step) =
macro_AP_T1(step) - macro_AP_C1(step)
```

Не использовать как главный causal estimate:

```text
best(C1) vs best(T1)
```

потому что каждый arm тогда отдельно выбирает шумный best checkpoint на одной и той же validation.

Best checkpoint можно выбрать позже для practical model selection, но вывод о влиянии A2 должен опираться на paired same-step trajectory.

---

# 24. Metrics

Для каждого checkpoint сохранить:

```text
full 73131 validation predictions
macro AP
micro AP
AP by 20 categories
validation loss
training loss
learning rate
global step
```

Плюс provenance:

```text
source submission URL
source checkpoint
base SHA
old adapter SHA
classifier SHA
merged encoder SHA
fresh LoRA config
training manifest
human split SHA
synthetic dataset SHA
seed
optimizer config
```

---

# 25. Paired error analysis

После C1/T1 сохранить для каждой validation row:

```text
row_id
category
label
C1 score/logit
T1 score/logit
difference
```

Нужно посмотреть:

- где T1 исправляет C1 false positives;
- где создаёт новые false positives;
- что происходит на near-identical negatives;
- что происходит по категориям;
- не ухудшаются ли positives/FN;
- соответствует ли эффект тому failure mode, ради которого создавалась A2.

---

# 26. Paired bootstrap

После получения predictions провести paired bootstrap C1 vs T1 на одной и той же validation.

Так как основная metric — macro AP по категориям, bootstrap должен учитывать category structure / стратификацию, а не бездумно смешивать всё в один micro sample.

Bootstrap нужен как uncertainty diagnostic.

Он не заменяет training-seed robustness.

Если improvement микроскопический, не объявлять A2 победителем только потому, что bootstrap выглядит красиво.

---

# 27. Что считать первым успешным сигналом

Не фиксировать сейчас искусственный threshold типа:

```text
+0.001 обязательно
```

Смотреть одновременно:

1. `T1 - C1` macro AP;
2. trajectory на 25/50/75/100%;
3. число категорий с улучшением/деградацией;
4. размер худших regressions;
5. paired bootstrap;
6. error-type analysis;
7. secondary item-disjoint diagnostics;
8. связь результата с BERTA-rescore groups.

Сильный результат выглядит примерно так:

```text
T1 consistently > C1
на нескольких одинаковых training steps
+
macro AP растёт
+
нет широкой category regression
+
изменения логически соответствуют hard-negative objective
```

---

# 28. T2 сейчас не запускать

Первоначально обсуждался:

```text
T2 = A2 replay weight 1.0
```

Но standalone T2 создаёт confound, если рядом нет matched control с тем же weight.

Если T1 показывает полезный сигнал, второй stage лучше строить так:

```text
C2:
original source replay
weight = 1.0

T2:
A2 replay
weight = 1.0
```

Главное comparison:

```text
T2 - C2
```

Пока T1 не показал сигнал, TPU-квоту на C2/T2 не тратить.

---

# 29. T3 category-balanced — только потом

Если natural-distribution T1 показывает пользу, тогда можно тестировать:

```text
category-balanced C3
vs
category-balanced T3
```

с одинаковым control/treatment sampling.

До этого category balancing не нужен.

---

# 30. Минимальная первая матрица запусков

Обязательные:

```text
C0 — no-training anchor / equivalence
C1 — human + matched original-source replay
T1 — human + matched A2 replay
```

Не запускать сразу:

```text
T2
T3
LR sweep
multiple replay weights
multiple category samplers
full fine-tuning
```

---

# 31. Рекомендуемый порядок работ

```text
1. Resolve exact checkpoint from Max submission provenance
2. Verify base + old LoRA + classifier chain
3. Merge old LoRA
4. Merge-equivalence probe
5. Full historical validation equivalence
6. Create fresh LoRA
7. Build/verify 10123 clean train-only A2 manifest
8. Frozen BERTA rescore all 10123
9. Save rescore diagnostics
10. DO NOT filter dataset by BERTA hardness
11. Build matched C1/T1 replay manifests
12. Verify equal rows/steps/weights/source multiplicity
13. Dry-run training pipeline without expensive full run
14. Launch C1
15. Launch T1
16. Evaluate at same checkpoints
17. Paired analysis
18. Decide whether C2/T2 or balancing are justified
```

---

# 32. Что требуется от Codex сейчас

Сейчас **не реализовывать и не запускать обучение**.

Сначала провести design review.

Codex должен изучить actual repository и ответить по следующим вопросам.

## A. Source checkpoint

1. Подтверждает ли repository provenance, что submission Макса:
   `matching-berta_tpu_lora-submit.zip`
   действительно соответствует `maxamsecret/berta-tpu-lora-top-checkpoint`?
2. Какой exact base checkpoint нужен для merge?
3. Какие exact adapter/classifier files нужно использовать?
4. Есть ли chain-of-merges или continuation, которую нельзя пропустить?

## B. Merge

1. Корректен ли предложенный merge?
2. Какую numerical tolerance поставить для logit equivalence?
3. Какие exact files/manifests сохранить?

## C. Fresh LoRA

1. Нужно ли копировать topology historical top LoRA?
2. Если нет — почему?
3. Какие exact `r/alpha/dropout/target_modules` рекомендуются?

## D. BERTA-rescore

1. Реализуем ли frozen rescore всех 10123?
2. Правильно ли использовать его только как diagnostic, без data filtering?
3. Какие дополнительные hardness diagnostics стоит сохранить?

## E. C1/T1

1. Чистое ли causal comparison?
2. Есть ли скрытый confound в matched multiplicity?
3. Нужно ли сохранять все 10123 rows без дополнительного filtering?
4. Верны ли `4 exposures × weight 0.5`?

## F. Optimizer

1. Разумны ли:
   `LoRA LR=1e-4`, `classifier LR=5e-5`?
2. Или historical recipe требует другой scale?
3. Какой один recipe лучше выбрать без sweep?

## G. Training duration

1. Одна epoch максимум — разумно?
2. Правильно ли eval на `0/25/50/75/100%`?
3. Согласен ли Codex не делать asymmetric early stopping?

## H. Statistics

1. Правильно ли сравнивать same-step C1/T1 trajectory?
2. Как лучше сделать category-aware paired bootstrap?
3. Какой минимальный effect считать практически интересным, не выдумывая threshold заранее?

## I. Hidden problems

Отдельно поискать:

- validation leakage;
- stale token cache;
- wrong synthetic tokenization;
- mismatch pair assembly;
- classifier mismatch;
- scheduler/step-count mismatch;
- replay-weight bug;
- duplicated rows;
- sample-weight normalization bugs;
- source multiplicity mismatch;
- TPU/XLA issues;
- nondeterministic data order;
- different number of optimizer steps C1/T1.

---

# 33. Формат ответа Codex

Codex должен вернуть:

## 1. Verdict

```text
support
support with changes
do not support
```

## 2. Critical blockers

Только вещи, которые реально нельзя игнорировать перед training.

## 3. Keep / Change / Drop

Для:

```text
source checkpoint selection
merge
classifier warm-start
fresh LoRA
BERTA-rescore
no-hardness-filter policy
4x replay
weight 0.5
C1/T1
LR recipe
1 epoch
same-step eval
bootstrap
```

## 4. Exact implementation plan

После design review, но **не выполняя его**, перечислить:

```text
какие existing files менять
какие new files создавать
какие functions/classes нужны
какие manifests/artifacts появятся
```

Не угадывать filenames: сначала найти их в repository.

## 5. Minimal TPU run plan

Только:

```text
C0
C1
T1
```

если нет критической причины добавить что-то ещё.

## 6. Open questions

Только вопросы, которые действительно требуют решения пользователя/тимлида.

---

# 34. Стоп-точка

После design review остановиться.

До явного approval пользователя:

```text
НЕ запускать TPU
НЕ запускать Kaggle training
НЕ создавать новые expensive runs
НЕ менять source checkpoint
НЕ фильтровать 10123 clean synthetics по BERTA hardness
НЕ запускать T2/T3
НЕ делать LR sweep
НЕ пушить архитектурные изменения без согласования
```
