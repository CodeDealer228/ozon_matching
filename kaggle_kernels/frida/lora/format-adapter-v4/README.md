# FRIDA format adapter v4

Небольшая LoRA адаптирует лучший checkpoint `continue_c_llm3m_2ep/last`
к item-format v4. Исходный encoder заморожен; исходный `FullSftHead` также
заморожен и всегда работает без dropout.

Обучение повторяет data/loss-контракт experiment C: human item-disjoint
train, 3 млн LLM train без пересечения с LLM val, focal loss (`gamma=2`),
`human_weight=5.86`, веса категорий и баланс 0/1 внутри категории. Новый
v4-cache уже не содержит prefix и category в тексте.

LoRA: `r=8`, `alpha=16`, dropout `0.10`, только `q` и `v`, LR `7e-5`, одна
эпоха, `PAIR_MAX_LENGTH=224`. TPU batch фиксирован на `32/64` на rank
(train/eval) и проверяется одним smoke ровно production-пути. Этот shape
прошёл на всех 8 rank в v2; последующий бинарный перебор нескольких shapes
оставил XLA executable резидентными и завершился падением rank7, поэтому
автоподбор на TPU удалён. Human val считается
каждые 100 тыс. обучающих пар. После достижения
0.758 обучение продолжается до первого снижения следующей метрики; при
результате ниже 0.700 одна следующая проверка пропускается.

Выход: `best/` и `last/`, каждый содержит adapter, optimizer, scheduler и
`train_state.json` с macro PR-AUC, словарём метрик по категориям, выбранными
batch-размерами и наблюдаемым peak HBM. HBM также логируется после каждого
smoke, каждых 500 train-step и полного human-val; значение allocator-based и
может не видеть память резидентных XLA executable.
