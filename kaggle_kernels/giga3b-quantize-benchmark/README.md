# giga3b-quantize-benchmark

Проверка гипотезы «уйти в bi-encoder и взять модель побольше»: тянет ли
`ai-sage/Giga-Embeddings-instruct` (3.45B, decoder, MIT) — по корректности,
качеству, скорости и упаковке — в замену текущему LoRA-кросс-энкодеру.

- **Kaggle:** https://www.kaggle.com/code/maxamsecret/giga3b-quantize-benchmark-bi-encoder-probe
- **Тип:** script · **GPU:** T4/P100 (что выдаст Kaggle) · **интернет:** вкл

## Суть — 4 вопроса за один прогон
1. **CORRECTNESS** — грузится и кодирует ли связка `trust_remote_code +
   sentence-transformers` в full-precision и (главное) int8.
2. **QUALITY** — сырой (без дообучения) косинус macro PR-AUC per-category на
   стратифицированном сабсэмпле нашего val-сплита. Если сырой 3B bi-encoder
   уже около/выше 0.5282 — дообучать стоит; если ~0.15 — нет.
3. **SPEED** — throughput (items/s, tokens/s) + пиковый VRAM, проекция на
   ~500–680K уникальных товаров приватной фазы → уложимся ли в лимит 13 мин на
   стендовой H100.
4. **PACKAGING** — сохраняет int8-веса (~3.5GB, влезают в 5GB zip) и печатает
   их размер на диске.

Версии пиннятся под карточку модели (`transformers==4.51.0`,
`sentence-transformers==5.1.1`); flash-attn не обязателен (есть sdpa-fallback,
T4/P100 всё равно без bf16-тензоров → full-precision идёт в fp16).

## Что использовалось
- **Датасет:** `maxamsecret/ozon-ecup-2026-matching` (val-сабсэмпл из `matches`
  + тексты из `items_human.parquet`).
- **Модель:** `ai-sage/Giga-Embeddings-instruct` (тянется с HF).

## Результат
Пишет `output/giga3b_probe_results.json` (env, bf16/fp16 и int8: macro_pr_auc,
overall_ap, items/s, peak_gb, проекции на H100) и сохраняет int8-веса в
`/kaggle/working/giga_int8` для сборки zip. Связан с локальным сабмит-каркасом
[`submits/giga3b_probe/`](../../submits/giga3b_probe).

> Статус на момент оформления: кернел **выполняется** (`RUNNING`) — числа ещё
> не скачаны. Забрать после завершения:
> `kaggle kernels output maxamsecret/giga3b-quantize-benchmark-bi-encoder-probe -p kaggle_kernels/giga3b-quantize-benchmark/output`.
