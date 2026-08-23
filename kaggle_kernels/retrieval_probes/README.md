# Retrieval / hard-negative пробы

Не привязаны к конкретному энкодеру — работа с самим графом кандидатных
пар и retrieval-сигналом, а не с обучением классификатора.

| Кернел | Назначение |
|---|---|
| `negative-pattern-sampling` | rule-based майнинг hard-negative паттернов по категориям (`negative_patterns/`) |
| `retrieval-recall-hard-negative-mining` | замер recall retrieval-этапа + майнинг hard negatives из него |
| `fasttext-category-similarity-probe` | проба fastText для категориальной похожести |
| `tfidf-category-index-fit` | TF-IDF индекс по категориям (используется в CatBoost-головах, `submits/*/catboost*`) |
