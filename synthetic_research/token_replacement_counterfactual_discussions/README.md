# Контрфактуальная синтетика заменой токенов

В этой папке последовательно фиксируется обсуждение нового метода синтетики из PR №7.

Порядок документов:

1. [`00_teamlead_proposal.md`](00_teamlead_proposal.md) — исходная идея и последующие уточнения тимлида.
2. [`01_codex_initial_review.md`](01_codex_initial_review.md) — первичная оценка метода, риски и рекомендуемый пилот.
3. [`02_implementation_v1.md`](02_implementation_v1.md) — контракт первой реализации в ветке `simple_sintize_and_augmentation`.
4. [`03_few_shot_annotation_workflow.md`](03_few_shot_annotation_workflow.md) — подготовка и проверка human few-shot.
5. [`04_global_five_shot_template.md`](04_global_five_shot_template.md) — сокращённый глобальный шаблон.
6. [`05_teamlead_architecture_options.md`](05_teamlead_architecture_options.md) — generator-validator, ensemble, search и локальный Qwen.
7. [`06_codex_architecture_review.md`](06_codex_architecture_review.md) — первая оценка этих вариантов.
8. [`07_kaggle_hardware_review.md`](07_kaggle_hardware_review.md) — актуальное железо Kaggle и применимость для LLM.
9. [`08_quality_first_experiment_matrix.md`](08_quality_first_experiment_matrix.md) — quality-first матрица конкурирующих экспериментов.
10. [`09_web_search_agent_contract.md`](09_web_search_agent_contract.md) — точный контракт поискового инструмента и prompts.

Документы описывают решения, но сами по себе не разрешают платные API-запросы или обучение.
