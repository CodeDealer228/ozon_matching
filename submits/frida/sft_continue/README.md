# FRIDA full-SFT — continue-эксперименты

Продолжение обучения `kaggle_kernels/frida/full_sft/full-sft` (val 0.7582, тест 0.481)
через `kaggle_kernels/frida/full_sft_continue/full-sft-continue` — общий код,
`EXPERIMENT_NAME`/`LLM_TRAIN_SUBSAMPLE_SIZE`/`HUMAN_WEIGHT`/`FOCAL_GAMMA`/
`EPOCHS` различаются по копии. Полное описание рецепта, resume-механизма и
матрицы экспериментов — `kaggle_kernels/frida/full_sft_continue/full-sft-continue/README.md`.

## Тест-стенд — реальные скоры

| Директория | Эксперимент | `PAIR_MAX_LENGTH` | val_macro_pr_auc | Тест-скор |
|---|---|---|---|---|
| `continue_c_llm3m` (файл `..._188-submit.zip`) | `continue_c_llm3m_2ep` | **188** | **0.7595** | **0.487 (лучший в проекте)** |
| `continue_a_baseline` | `continue_a_baseline_3ep` | 180 | 0.7555 | 0.478 |
| `continue_c_llm3m` (файл без суффикса) | `continue_c_llm3m_2ep` | 180 | 0.7595 | не тестировался отдельно |
| `continue_b_gamma3` | `continue_b_gamma3_3ep` | 180 | 0.7550 | не тестировался |
| `continue_b_gamma3_184` | `continue_b_gamma3_3ep` | 184 | 0.7550 | не тестировался |

**`continue_c_llm3m` при `PAIR_MAX_LENGTH=188` — новый лучший результат
проекта** (0.487, обошёл `full_sft` 0.481). Ключевое: (c) прервался на
epoch 1/2 (не успел завершить план), но уже превосходит **полностью
завершённый** `continue_a_baseline` (3/3 эпохи) и по val, и по тесту —
единственное реальное отличие рецептов (c) от (a) — LLM-подвыборка
1M→3M + `HUMAN_WEIGHT` пересчитан на 5.86 (баланс градиентов human:llm
≈1:2 при новом объёме) — значит именно это отличие, а не просто «больше
шагов continue-стадии», двигает результат.

**`PAIR_MAX_LENGTH=188` подтверждён на реальном стенде**: и по скору выше
180, и по времени — frida+head прогон укладывается в ~16-19 минут при
`BATCH_SIZE_FIXED=5248`. `continue_a_baseline` при 180 отдельно замерен на
16/20 минут — то есть у 188 всё ещё есть запас. **188 — новый стандарт
`PAIR_MAX_LENGTH` для всех continue-сабмитов**, зашит по умолчанию в оба
CPU-сборщика (`frida-full-sft-submit-build`,
`frida-full-sft-continue-submit-build`).

## Per-category разбивка по val (обучение)

Раньше `train_state.json` хранил только macro-среднее — per-category dict
печатался в лог (`[METRICS] per_category: {...}`), но нигде не
сохранялся. Исправлено в `frida_full_sft_continue.py` (2026-08-23,
`save_checkpoint` теперь пишет `val_per_category`/`llm_val_per_category`
в `train_state.json` для всех БУДУЩИХ прогонов) — но для уже
завершённых (a)/(b) пришлось вытащить из сырых логов кернелов вручную:

- `continue_a_baseline/per_category_by_epoch.json` — 3 чекпоинта (эпохи 1-3), val + llm_val, по 20 категориям каждый.
- `continue_b_gamma3/per_category_by_epoch.json` — то же, 3 чекпоинта.
- **`continue_c_llm3m` — не восстановлено.** `kaggle kernels logs`/`output`
  отдают только ПОСЛЕДНЮЮ запушенную версию кернела; после эпохи 1 (v2,
  тот самый прогон, что дал 0.7595) была запушена v3 (позже отменена
  пользователем), которая навсегда закрыла доступ к v2-логу через API.
  Macro-числа сохранились (в опубликованном датасете
  `maximmarkovbb/frida-sft-experiment-c-llm-3m`), per-category — нет.

## Сборка сабмитов

`kaggle_kernels/frida/full_sft_continue/full-sft-continue-submit-build` (CPU, приватный,
`maxamsecret`) — content-based поиск чекпоинта по `experiment` в
`train_state.json` (best > last), по `kernel_sources`
(`frida-full-sft-continue-a-baseline`, `-b-gamma3`) и `dataset_sources`
(`maximmarkovbb/frida-sft-experiment-c-llm-3m` — (c) публичным сделать не
получилось из-за телефон-верификации/TPU-сессии, поэтому чекпоинт залит
как датасет). Устойчив к недоступному источнику — остальные всё равно
собираются, кернел просто явно падает в конце, если что-то не собралось.
