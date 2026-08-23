# merge-lora

Мёрж LoRA-адаптера в базовые веса, чтобы на инференсе не тащить зависимость
`peft`.

- **Kaggle:** https://www.kaggle.com/code/maxamsecret/ozon-merge-lora-adapter-into-base-weights
- **Тип:** script · **GPU:** нет (CPU) · **интернет:** вкл

## Суть
`PeftModel.from_pretrained(base, adapter).merge_and_unload()` над 33M-моделью —
это просто арифметика над весами, GPU не нужен. На выходе — обычный
`transformers.AutoModel` + отдельный `classifier.pt`, который грузится штатно,
без `peft`. Так сабмит становится легче и быстрее.

## Что использовалось
- **Датасет:** `maxamsecret/ozon-lora-minilm-finetuned` (адаптер, `best/`).
- **База:** `cross-encoder/ms-marco-MiniLM-L-12-v2` (тянется по интернету).

## Результат (`output/lora_merged/`)
Смёрженные веса: `model.safetensors`, `config.json`, `classifier.pt`,
`tokenizer*`. Скачиваются локально и публикуются как датасет
`maxamsecret/ozon-lora-merged-minilm` — именно он идёт в сабмит и в
[`../lora-timing-dryrun`](../lora-timing-dryrun).

## Запуск
```bash
kaggle kernels push -p kaggle_kernels/minilm/merge-lora
kaggle kernels output maxamsecret/ozon-merge-lora-adapter-into-base-weights -p kaggle_kernels/minilm/merge-lora/output
```
