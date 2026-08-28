# BERTA + A2: результаты matched C1/T1 обучения

Дата завершения: 26 августа 2026.

## Короткий вывод

Эксперимент завершён полностью. В сравнении с matched control синтетика A2 дала
**маленькое, но устойчиво положительное улучшение** на одной и той же untouched
validation:

- final macro AP: `0.7463035491` (C1) → `0.7466399981` (T1),
  delta `+0.0003364490`;
- final micro AP: `0.8068742173` (C1) → `0.8071911854` (T1),
  delta `+0.0003169681`;
- T1 выше C1 по macro и micro на каждом ненулевом checkpoint;
- улучшились 13 из 20 категорий, ухудшились 7 из 20.

Это подтверждает, что эффект даёт именно A2 rewrite, а не просто дополнительное
повторение исходных negative-пар: C1 и T1 совпадали по source, порядку, весам,
числу строк, seed, optimizer steps и всем настройкам, кроме выбора original B
в C1 против synthetic B' в T1.

Сигнал всё же небольшой. Для macro AP 95% bootstrap-интервал пересекает ноль,
поэтому сильное утверждение о гарантированном macro-улучшении пока делать нельзя.
Для micro AP интервал полностью положительный.

## Ссылки

- [Matched C1/T1 TPU kernel, versions 5 and 6](https://www.kaggle.com/code/rvitvitskii/berta-a2-tpu-train)
- [Safe merge and equivalence kernel, version 4](https://www.kaggle.com/code/rvitvitskii/berta-a2-merge-top-and-equivalence)
- [Утверждённый source checkpoint Макса](https://www.kaggle.com/code/maxamsecret/berta-tpu-lora-top-checkpoint)
- [Base merged BERTA](https://www.kaggle.com/code/maxamsecret/berta-merge-lora)

### Kaggle datasets

- [Private paired replay dataset с clean A2 и immutable C1/T1 manifest](https://www.kaggle.com/datasets/rvitvitskii/berta-a2-replay-v1)
- [Source BERTA checkpoint dataset](https://www.kaggle.com/datasets/maxamsecret/berta-tpu-lora-top-checkpoint)
- [Исходные данные Ozon Matching](https://www.kaggle.com/datasets/maxamsecret/ozon-ecup-2026-matching)

Kaggle kernel и replay dataset приватные. Все persistent outputs скачаны и
проверены локально.

## Дизайн эксперимента

- source: exact top BERTA+LoRA checkpoint Макса;
- source adapter SHA-256:
  `26b294123b2445fa7bb46e0c36c2ee5ad259c218ea80e357661f49729800cd34`;
- source classifier SHA-256:
  `70ac231c64ad2f70ac73592afbce76c3ee9f0cb804ecb16e8b0f4258325d43fa`;
- LoRA сначала безопасно смёржена в BERTA base; equivalence gate пройден;
- clean A2: `10,123` строк;
- paired training manifest: `333,312` строк;
- paired manifest SHA-256:
  `e3ff627896bacd6d0c50438bfd959b11c2a3013878c64be426c95e06fd4eb225`;
- validation: `73,131` строк, только evaluation;
- validation SHA-256:
  `a48993ca6f54d0b7dde7fbe3f6e637a737ee6cb0c55f33b21e4970c40b6e8bb9`;
- unordered train/validation overlap: `0`;
- global batch: `512`, TPU cores: `8`, fixed per-core batch: `64`;
- optimizer steps: `651`;
- checkpoints/evaluation: `0`, `163`, `326`, `488`, `651`;
- synthetic/control replay sample weight: `0.5`;
- C1 и T1 имеют идентичный step-0 adapter, classifier и validation logits.

## Траектория validation

| Step | C1 macro | T1 macro | T1−C1 | C1 micro | T1 micro | T1−C1 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.7086041124 | 0.7086041124 | 0 | 0.7759352856 | 0.7759352856 | 0 |
| 163 | 0.7406743571 | 0.7410067690 | +0.0003324119 | 0.8020245894 | 0.8022151567 | +0.0001905673 |
| 326 | 0.7446515319 | 0.7452865729 | +0.0006350410 | 0.8064209053 | 0.8069584929 | +0.0005375876 |
| 488 | 0.7459877663 | 0.7465070942 | +0.0005193279 | 0.8067767126 | 0.8072072988 | +0.0004305863 |
| 651 | 0.7463035491 | 0.7466399981 | +0.0003364490 | 0.8068742173 | 0.8071911854 | +0.0003169681 |

Лучший measured checkpoint у обеих веток — step 651. Результат сравнивается на
одинаковом step, а не как несогласованный best-vs-best выбор.

## Paired bootstrap

Использован paired stratified row bootstrap на тех же `73,131` validation rows:
`1,000` повторов, seed `20260826`, размеры категорий в каждом повторе сохранены.

| Метрика T1−C1 | Bootstrap mean | 95% CI | P(T1>C1) | two-sided sign p |
|---|---:|---:|---:|---:|
| Macro AP | +0.0003466424 | [-0.0001544512; +0.0008488274] | 0.919 | 0.162 |
| Micro AP | +0.0003151245 | [+0.0000706921; +0.0005674239] | 0.996 | 0.008 |

Интерпретация:

- micro-улучшение статистически устойчиво в этом bootstrap;
- macro-направление положительное, но неопределённость ещё слишком велика;
- эксперимент подтверждает полезный сигнал, но не большой скачок качества.

## Категории

Самые большие final улучшения:

| Категория | T1−C1 AP |
|---|---:|
| Галантерея и аксессуары | +0.0029124412 |
| Обувь | +0.0025261186 |
| Мебель | +0.0010250091 |
| Бытовая техника | +0.0008655673 |
| Красота и гигиена | +0.0008256274 |

Самые большие final ухудшения:

| Категория | T1−C1 AP |
|---|---:|
| Одежда | -0.0015168971 |
| Строительство и ремонт | -0.0007743215 |
| Спорт и отдых | -0.0005514090 |
| Продукты питания | -0.0004984658 |
| Аптека | -0.0004619745 |

Только у «Галантерея и аксессуары» отдельный category-level 95% bootstrap CI
полностью выше нуля. Остальные category-level эффекты пока недостаточно точны.

## Сохранённые outputs

### Merge

`artifacts/synthetic_v2/berta_a2_merge_top_utf8/berta_a2_merge_top`

- 19 файлов, `527,762,384` bytes;
- equivalence gate passed;
- merged encoder SHA-256:
  `b7fd9ae3fb50f5df95468c9d8671e5a9e8ea733d18ac6fd1090b8b6c76c55cdb`.

### C1 final, Kaggle version 5

`artifacts/synthetic_v2/berta_a2_tpu_c1_complete/berta_a2_tpu_c1`

- 48 файлов, `151,911,893` bytes, пустых файлов нет;
- final adapter SHA-256:
  `a314e32f7a03404850c95f05f54f798321ad529209b88326e731c5d31aa3d0eb`;
- final classifier SHA-256:
  `1c91e8a026a9375400a44aa4752b08d4a6def540cb1110abb2844908aa182f93`;
- final optimizer SHA-256:
  `eaa432ebfbd1be9b0d5bcc7544769294c08b1d66e3ebbd19dadc0cf82e81a722`;
- final predictions SHA-256:
  `9c737ffefa750a661b34403d2457a183f1fbe5689445320e58f45bba76db6b52`.

### T1 final, Kaggle version 6

`artifacts/synthetic_v2/berta_a2_tpu_t1/berta_a2_tpu_t1`

- 48 файлов, `151,912,016` bytes, пустых файлов нет;
- final adapter SHA-256:
  `1e48b001899ccf755ec5e31f96a1c02f0e0603cf9b30c688d2e2bb6832b7fa3c`;
- final classifier SHA-256:
  `cee80410e938f0a5faab826a93f39f4e930a89b20a713ac81e81222470f815fd`;
- final optimizer SHA-256:
  `12211cbe6037dfc34b7b7124e206f5ff9a2594453a6df62042abee8e36718402`;
- final predictions SHA-256:
  `cc83c711e51449008ebb0840e197d56d685839fd15b0e184d414977574a26f4a`.

В обоих outputs есть adapter, classifier, optimizer, scheduler, train state,
metrics, logits и row-level parquet predictions для всех пяти checkpoints.

### Сравнение

`artifacts/synthetic_v2/berta_a2_c1_t1_comparison`

- `comparison_manifest.json`;
- `metrics_by_step.csv`;
- `per_category_final.csv`;
- `paired_bootstrap_final.json`.

### ODS submission: T1 step 651

Собран локальный архив:

`submits/berta/a2_t1_step651/matching-berta-a2-t1-step651-submit.zip`

- размер ZIP: `499,549,071` bytes (`499.5` MB decimal);
- файлов в архиве: `40`;
- распакованный размер: `538,033,750` bytes;
- ZIP SHA-256:
  `11fda7b7e4a77398197b5cab409a8c9f4e1dc4aacfc0d49b7d92210b64bf4337`;
- merged encoder SHA-256:
  `b7fd9ae3fb50f5df95468c9d8671e5a9e8ea733d18ac6fd1090b8b6c76c55cdb`;
- T1 adapter SHA-256:
  `1e48b001899ccf755ec5e31f96a1c02f0e0603cf9b30c688d2e2bb6832b7fa3c`;
- T1 classifier SHA-256:
  `cee80410e938f0a5faab826a93f39f4e930a89b20a713ac81e81222470f815fd`.

Оболочка инференса и offline wheels взяты побайтово из ранее прошедшего ODS
BERTA submission с test score `0.47`. `run.py` и `src/utils.py` совпадают с
проверенным вариантом по SHA-256. Веса внутри ZIP побайтово совпадают с
проверенными Kaggle outputs; `ZipFile.testzip()` вернул `None`, обязательные
файлы, tokenizer, config, encoder, adapter и classifier присутствуют. Реальный
ODS score появится только после загрузки этого архива на стенд.

## Технические сбои и исправления

Все сбои произошли до дорогой части либо были обнаружены строгим output audit:

1. merge v1: optional torchao/PEFT incompatibility;
2. merge v2: отсутствующая ветка no-truncation для короткой пары;
3. merge v3: один fp16 runtime outlier; tolerance не расширялась, v4 отдельно
   перепроверила outlier в fp32 и прошла gate;
4. TPU v1: PEFT отсутствовала в TPU image;
5. TPU v2: несовместимые версии PEFT/transformers;
6. TPU v3: checksum tensor требовал `.detach()` перед NumPy;
7. TPU v4: обучение прошло, но аудит обнаружил, что `xm.save` не сохранил
   optimizer/scheduler;
8. TPU v5: финальный воспроизводимый C1 со всеми состояниями;
9. TPU v6: matched T1, отличается от v5 только `ARM = "T1"`.

## Решение

Текущий A2 pipeline **подтверждён как полезный, но эффект небольшой**. T1 можно
считать более сильным кандидатом, чем matched C1. Submission для выбранного T1
checkpoint собран и проверен локально, но результат ещё не доказывает улучшение
реального ODS leaderboard score: для этого архив нужно прогнать на стенде.

Рациональный следующий эксперимент — не безусловно увеличивать общий synthetic
weight, а отдельно проверить category-aware sampling/weighting, потому что
польза сконцентрирована неравномерно и «Одежда» заметно ухудшилась. Такой запуск
в этот отчёт не входит и автоматически не выполнялся.

## Git status

Код и отчёт зафиксированы локальными commits в ветке `synthetic`. По отдельной
команде пользователя этот итоговый отчёт отправлен в `origin/synthetic`.
Build source T1, тяжёлые веса, offline wheels и ZIP остаются локальными; веса,
wheels и ZIP игнорируются Git.
