# Итог Codex после GPT-review: BERTA + fresh LoRA + A2

Дата: **2026-08-25**  
Статус: **design approved with changes; обучение не запускалось**

## 1. Verdict

**Support with changes.**

Основная конструкция GPT правильная:

```text
exact BERTA submission Макса
→ merge его LoRA в правильную base lineage
→ fresh LoRA + warm-start classifier
→ matched C1/T1
→ одинаковые шаги и validation checkpoints
```

Источник зафиксирован пользователем окончательно: используется только модель из
ссылки Макса, соответствующая
`maxamsecret/berta-tpu-lora-top-checkpoint` и реальному test score `0.47`.
`human-continue epoch1` в этот эксперимент не входит.

---

## 2. Что подтверждено по репозиторию и Kaggle

Repository provenance подтверждает цепочку submission:

1. `sergeyzh/BERTA` + ранняя LoRA `rays0n/berta-llm-continue-train` были
   смёржены кернелом `maxamsecret/berta-merge-lora`.
2. На его plain `merged_encoder` была создана свежая LoRA в
   `maxamsecret/berta-llm-tpu-sft`.
3. Checkpoint step 11211 содержит именно adapter/classifier отправленного
   `matching-berta_tpu_lora-submit.zip`.
4. Точное совпадение уже доказано SHA-256 в
   `reports/berta_tpu_lora_provenance/README.md`:
   - adapter: `26b294123b2445fa7bb46e0c36c2ee5ad259c218ea80e357661f49729800cd34`;
   - classifier: `70ac231c64ad2f70ac73592afbce76c3ee9f0cb804ecb16e8b0f4258325d43fa`.
5. Kaggle dataset source checkpoint существует и содержит adapter config,
   adapter weights, classifier и train state.
6. Его `adapter_config.json` прямо указывает base:
   `/kaggle/input/notebooks/maxamsecret/berta-merge-lora/berta_merged/merged_encoder`.

Следовательно, merge нельзя делать как `sergeyzh/BERTA + top adapter`.
Правильная операция:

```text
maxamsecret/berta-merge-lora/merged_encoder
+ maxamsecret/berta-tpu-lora-top-checkpoint/adapter
→ second-stage merged BERTA
```

Classifier из top checkpoint переносится без обнуления.

---

## 3. Critical blockers перед training

### B1. Exact base lineage

Новый merge kernel должен находить и hash-проверять именно output
`maxamsecret/berta-merge-lora`. Старый `berta-merge-lora.py` нельзя просто
перезапустить: он мёржит более раннюю LoRA и решает предыдущую задачу.

### B2. Merge equivalence

До TPU-run обязательны unmerged-vs-merged probe и full historical validation.
Если они расходятся выше tolerance, training не запускается.

Предлагаемые stop thresholds:

- fp32 probe: `max_abs_logit_diff <= 1e-4`,
  `mean_abs_logit_diff <= 1e-5`;
- реальный bf16 inference probe: `max_abs_logit_diff <= 2e-2`,
  `mean_abs_logit_diff <= 2e-3`;
- full validation: absolute macro/micro AP delta `<= 1e-4`;
- любое превышение не «расширяет tolerance», а запускает разбор dtype, base,
  tokenizer, pair assembly и classifier identity.

### B3. Leakage filtering

Из 12833 accepted A2 разрешены только 10123 строки, source которых находится в
historical BERTA train. 2710 validation-source строк исключаются. Других
hardness/quality-фильтров перед первым training experiment не добавляется.

### B4. Уникальная адресация synthetic title tokens

Старый BERTA item cache индексируется по item ID. Это нельзя напрямую применить
к A2: один `id2` может иметь несколько разных `name2_prime`. Если положить их в
cache под исходным `id2`, варианты столкнутся, перезапишутся или все пары увидят
один и тот же title.

Каждая synthetic row получает отдельный cache key, например:

```text
(source_orientation_sha256, candidate_index)
```

Для неё заново токенизируется только `name2_prime`. Category и attributes
берутся от исходного B и остаются неизменными. Real human cache не изменяется.

### B5. Одинаковая fresh-LoRA initialization на всех TPU replicas

Исторические TPU scripts вызывают `set_seed(SEED + rank)` до создания модели.
Для свежей LoRA это опасно: восемь replicas могут получить разные случайные
LoRA-A matrices. `xm.optimizer_step()` синхронизирует gradients, но не исправляет
разные исходные параметры.

Новый код обязан:

1. поставить одинаковый `MODEL_INIT_SEED` на всех ranks;
2. создать fresh LoRA;
3. проверить hash/checksum trainable parameters между ranks;
4. только после этого поставить `DATA_SEED + rank` для dropout/data randomness.

### B6. Matched manifests, order и step count

C1 и T1 должны читать один paired manifest. Он содержит обе версии title, а arm
выбирает `name2` либо `name2_prime`. Row IDs, labels, weights, multiplicity,
shuffle seed и число шагов совпадают.

---

## 4. Keep / Change / Drop

| Решение GPT | Итог Codex | Комментарий |
|---|---|---|
| Source = submission Макса | **Keep** | Пользователь подтвердил окончательно |
| Safe merge | **Keep** | Но base только из `maxamsecret/berta-merge-lora` |
| Classifier warm-start | **Keep** | Нужен для step-0 equivalence |
| Fresh LoRA | **Keep** | Exact historical topology |
| BERTA-rescore 10123 | **Keep** | Только диагностика, не фильтр |
| Не фильтровать по hardness | **Keep** | Исключаются только leak/corruption |
| 4 exposures × weight 0.5 | **Keep** | Одинаково в C1/T1 |
| C1 vs T1 | **Keep** | Главное причинное сравнение |
| LoRA LR `1e-4`, classifier `5e-5` | **Keep** | Один recipe, sweep не нужен |
| Одна эпоха | **Keep** | Исторически дальнейшее обучение переобучало |
| Eval 0/25/50/75/100% | **Keep** | Сравнивать только same-step |
| Asymmetric early stopping | **Drop** | Arms проходят одинаковые шаги |
| T2/T3 сейчас | **Drop** | Только после результата C1/T1 |
| Category balancing сейчас | **Drop** | Сначала natural accepted distribution |
| `PAIR_MAX_LENGTH=384` | **Change to 256** | Source submission обучался и работал на 256 |
| per-core train batch 32 | **Change to 64** | Для pair=256 это exact proven configuration |
| Loss normalization по sum weights | **Change** | Использовать weighted mean по fixed batch |

---

## 5. Exact model/training contract

Для изоляции data effect повторяется contract source model:

- prefix: `paraphrase: `;
- item max length: `256`;
- pair max length: `256`;
- classifier: `Dropout(0.10) -> Linear(768, 1)`;
- LoRA `r=16`, alpha `32`, dropout `0.15`;
- target modules: `key`, `query`, `dense`, `value`;
- merged base frozen; fresh LoRA и classifier trainable;
- TPU v5e-8;
- per-core train batch `64`, global batch `512`;
- per-core eval batch `128`;
- AdamW, weight decay `0.04`;
- LoRA LR `1e-4`;
- classifier LR `5e-5`;
- cosine schedule, warmup 10%;
- human-train pos_weight фиксирован:
  `2.894490893598893`.

Переход на 384 сейчас запрещён как confound. Он одновременно меняет truncation,
compiled shapes, batch и inference contract, поэтому эффект уже нельзя будет
приписать A2.

---

## 6. Loss contract

Для каждой строки:

```python
per_row = binary_cross_entropy_with_logits(
    logits,
    labels,
    pos_weight=human_pos_weight,
    reduction="none",
)
loss = (per_row * sample_weight).mean()
```

Почему не local `sum / sum(weights)`: при разном количестве replay-строк на
разных TPU ranks локальные знаменатели могут слегка менять objective. Fixed
batch mean сохраняет абсолютный смысл веса 0.5; C1 и T1 имеют одинаковые
weight positions, поэтому gradient scale и scheduler остаются matched.

Перед запуском нужны unit tests на ручном тензоре и проверка одинаковой суммы
весов по global step C1/T1.

---

## 7. BERTA-rescore

Rescore всех 10123 train-clean строк поддерживается, но не является gate по
hardness. Используется exact unmerged test-top checkpoint и contract 256.

Сохранить:

- source/candidate stable IDs;
- category и generator model;
- old/new logits и probabilities;
- `delta_logit`;
- variant count per source;
- агрегаты и quantiles overall/by category/by generator model/by multiplicity;
- долю `delta > 0`;
- технический manifest tokenizer/pair/checkpoint hashes.

Остановить дальнейший pipeline rescore может только при доказанном техническом
несоответствии. `delta <= 0` строки остаются в T1.

---

## 8. Paired C1/T1 manifests

Clean A2 rows: 10123. Replay factor: 4.

```text
human rows       = 292523
replay rows      = 10123 × 4 = 40492
total before pad = 333015
```

При global batch 512 нужно добавить одинаковые 297 deterministic human pad
rows в оба arms:

```text
333312 rows / 512 = 651 exact global steps
```

Это позволяет не терять строки через `drop_last` и гарантирует одинаковые
steps. Pad rows и их позиции фиксируются в manifest.

C1 replay использует `(A, B)`, T1 — `(A, B')`. Если source имеет два variants,
C1 получает две matched копии оригинала. Дополнительное схлопывание не нужно.

---

## 9. Evaluation и решение

Primary validation: historical 73131 rows. Checkpoints:

```text
step 0
step 163  (~25%)
step 326  (~50%)
step 488  (~75%)
step 651  (100%)
```

Главные paired comparisons:

```text
T1(step) - C1(step)
```

Нельзя объявлять causal effect по `best(T1) - best(C1)`. Best checkpoint можно
выбрать позднее только для практического submission.

Нужно разделять два вывода:

1. **Научный:** A2 помогает, если T1 устойчиво лучше matched C1 на одинаковых
   шагах и эффект соответствует hard-negative errors.
2. **Практический:** новая модель — кандидат на submission, только если T1 также
   не хуже step-0/source anchor и не создаёт широкую category regression.

Bootstrap: fixed-seed paired resampling внутри каждого `category × label`, затем
AP по категориям и macro-average. Минимум 2000 resamples. Это uncertainty
diagnostic, а не замена повторному seed-run при микроскопическом эффекте.

---

## 10. Exact implementation plan — пока не выполнять

### Existing files, используемые как source/reference

- `reports/berta_tpu_lora_provenance/README.md`;
- `kaggle_kernels/berta-merge-lora/berta_merge_lora.py`;
- `kaggle_kernels/berta-llm-tpu-sft/berta_llm_tpu_sft.py`;
- `data/a2_synthetic_all_categories_v4.parquet`;
- `data/a2_synthetic_all_categories_v4.meta.json`;
- `data/splits/matches_split_seed42.parquet`.

### Новые локальные файлы

- `scripts/build_berta_a2_replay_manifests.py` — leak filter, unique synthetic
  keys, C1/T1 paired rows, replay/pad и identities;
- `scripts/compare_berta_a2_c1_t1.py` — same-step metrics, category analysis и
  paired bootstrap;
- `tests/test_berta_a2_replay_manifests.py` — leakage, multiplicity, order,
  step count, sample weights;
- `artifacts/synthetic_v2/berta_a2_training_v1/` — generated private manifests,
  diagnostics и SHA reports.

### Новые Kaggle components

- `kaggle_kernels/berta-a2-merge-top/` — second-stage merge и equivalence;
- `kaggle_kernels/berta-a2-rescore/` — diagnostic old/new scoring;
- `kaggle_kernels/berta-a2-tpu-c1/` — matched original-source replay;
- `kaggle_kernels/berta-a2-tpu-t1/` — matched A2 replay.

C1/T1 training code должен происходить из одного общего template и иметь
одинаковый code SHA; различается только immutable arm config. Persistent output:
checkpoints на пяти шагах, predictions parquet/npy, metrics JSON, train manifest,
model/data identities, optimizer/scheduler state и failure/status JSON.

---

## 11. Minimal run plan

1. Подготовить и проверить clean/paired manifests локально.
2. Merge exact Max checkpoint на небольшом GPU kernel.
3. Выполнить probe и full-validation equivalence. При fail остановиться.
4. Выполнить BERTA-rescore как diagnostic; ничего не фильтровать.
5. Dry-run/preflight C1/T1.
6. Запустить C1 на TPU, скачать и проверить output.
7. Запустить T1 на TPU с тем же contract, скачать и проверить output.
8. Провести same-step paired analysis.
9. Только после результата решать, нужен ли C2/T2 или balancing.

T2, T3, LR sweep и дополнительные модели сейчас не запускаются.

---

## 12. Open questions

Source checkpoint уже разрешён пользователем: только submission Макса/test-top
0.47. Технических вопросов, требующих угадывания, не осталось.

Перед реализацией требуется одно подтверждение пользователя:

> Принять этот `support with changes` план, включая contract 256,
> C1/T1-only и отдельный diagnostic rescore, после чего Codex может менять код,
> публиковать необходимые private Kaggle datasets и запускать последовательные
> merge/GPU и C1/T1 TPU runs.
