# rubert-mini-frida-recall-mining

Тот же recall@k / hard-negative mining тест, что и
[`../retrieval-recall-hard-negative-mining`](../retrieval-recall-hard-negative-mining),
но на реальных семантических эмбеддингах `sergeyzh/rubert-mini-frida`
вместо TF-IDF. TF-IDF вышел слабым (recall@10 от 0.015 до 0.35 по
категориям, далеко от цели ~0.9) — чистая лексика не ловит
перефразировки/синонимы. Прямое сравнение на тех же 20 категориях, той же
методике.

- **Kaggle:** https://www.kaggle.com/code/maxamsecret/rubert-mini-frida-recall-hard-negative-mining
- **Тип:** script · **CPU only** · **интернет:** выкл — эмбеддинги уже
  посчитаны, тут только `NearestNeighbors`

## Предпосылка
Эмбеддинги готовы и опубликованы:
[maxamsecret/rubert-mini-frida-item-embeddings](https://www.kaggle.com/datasets/maxamsecret/rubert-mini-frida-item-embeddings)
— 711,304×312, float16 (проверено по размеру файла перед запуском, сходится).

## Логика — без изменений от TF-IDF-версии
Единственное отличие — источник вектора товара: не `TfidfVectorizer.fit_transform()`
по тексту категории, а срез из уже готовой глобальной матрицы эмбеддингов
по id. Дальше всё то же: `NearestNeighbors(metric="cosine")` внутри
категории, recall@{1,5,10,20,50} по всем размеченным парам (независимо
от target — обе метки одинаково свидетельствуют, что organizers'
retrieval счёл пару кандидатом), топ-K соседей за вычетом уже
размеченных → `mined_hard_negatives.parquet`.

Категории — от меньшей к большей, та же осторожность, что и в TF-IDF-версии:
полный N-к-N поиск на плотных 312-мерных векторах не измерялся на этом
масштабе (должен быть не медленнее разреженного TF-IDF — плотный matmul
на такой размерности обычно быстрее, — но не проверено).

## Что на выходе
- `recall_summary.json` — recall@k по категориям, оба направления + лучшее.
- `mined_hard_negatives.parquet` — `(id1, id2, embedding_cosine, category)`.

## Запуск
```bash
kaggle kernels push -p kaggle_kernels/rubert-mini-frida-recall-mining
kaggle kernels status maxamsecret/rubert-mini-frida-recall-hard-negative-mining
kaggle kernels output maxamsecret/rubert-mini-frida-recall-hard-negative-mining -p kaggle_kernels/rubert-mini-frida-recall-mining/output
```
