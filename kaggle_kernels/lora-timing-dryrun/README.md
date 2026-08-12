# lora-timing-dryrun

Оффлайн-замер времени сабмита на Kaggle GPU — чтобы не жечь одну из 5 попыток
в день на стенде вслепую.

- **Kaggle:** https://www.kaggle.com/code/maxamsecret/ozon-lora-submission-gpu-timing-dry-run-offline
- **Тип:** script · **GPU:** 1×T4 · **интернет:** **выкл** (как на реальном судействе)

## Суть
Собирает точную структуру сабмита (`run.py`, `src/utils.py`, `classifier.pt`,
`models/lora-merged-.../`) из приложенных датасетов и запускает `run.py` через
**тот же CLI-контракт**, что и judging-харнесс (`--items_path --matches_path
--output_path`) — не переписанная копия логики, а реальный сабмит. Полностью
оффлайн (`HF_HUB_OFFLINE=1`) — заодно доказывает, что решение работает без сети.

## Что использовалось
- **Датасеты:** `ozon-ecup-2026-matching`, `ozon-lora-merged-minilm`,
  `ozon-lora-submission-code`.

## Результат
Полный прогон на **365 654** парах: **950.6 с, 384.7 пар/с** на T4 (T4 медленнее
стендовой H100). Лимиты Public (6 мин / ~115K) и Private (13 мин / ~275K)
проходятся с запасом даже на более медленном GPU. Вывод — `output/predictions.csv`
(365 654 строки, формат `id1,id2,predict`).

## Запуск
```bash
kaggle kernels push -p kaggle_kernels/lora-timing-dryrun --accelerator NvidiaTeslaT4
```
