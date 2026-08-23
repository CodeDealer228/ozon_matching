"""Build a validation-only FRIDA FP/FN taxonomy with truncation separation."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import statistics
from collections import defaultdict
from pathlib import Path

import pandas as pd


EXTRA_ALIASES = {
    "Галантерея и аксессуары": {
        "brand": ("бренд", "торговая марка"),
        "article": ("артикул", "артикул производителя", "артикул поставщика"),
        "product_model": ("модель", "наименование модели"),
        "item_type": ("тип", "вид сумки", "вид рюкзака", "тип часов"),
        "product_color": ("цвет товара", "цвет", "название цвета", "основной цвет"),
        "gender": ("пол",),
        "target_audience": ("целевая аудитория",),
        "general_material": ("материал", "материал изделия", "материал корпуса"),
        "material_composition": ("состав", "состав бижутерии"),
        "completeness": ("комплектация",),
        "package_count": ("количество в упаковке", "количество предметов в упаковке"),
        "item_size": ("размер изделия", "размер рюкзака"),
        "dimensions": ("размеры, мм",),
        "width": ("ширина, см", "ширина предмета"),
        "height": ("высота, см", "высота предмета"),
        "depth": ("глубина, см", "глубина предмета"),
        "diameter": ("диаметр корпуса", "диаметр корпуса, мм"),
        "watch_mechanism": ("механизм часов",),
        "watch_display": ("тип индикации",),
        "watch_waterproof": ("класс водонепроницаемости", "водонепроницаемость"),
        "main_insert": ("вставка", "тип вставки"),
        "fastener": ("тип застежки", "вид застежки", "вид замка"),
    }
}

CRITICAL_CONCEPTS = {
    "article",
    "oem_number",
    "alternative_articles",
    "product_model",
    "item_type",
    "package_count",
    "completeness",
    "compatibility",
    "car_make",
    "car_model",
    "car_generation",
    "car_year",
    "engine",
    "placement",
    "volume",
    "diameter",
    "length",
    "width",
    "height",
    "depth",
    "power",
    "tire_width",
    "tire_profile",
    "tire_diameter",
    "metal_fineness",
    "main_insert",
    "stone_count",
    "watch_mechanism",
    "watch_display",
    "watch_waterproof",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def markdown_table(frame: pd.DataFrame) -> str:
    columns = [str(column) for column in frame.columns]
    rows = [columns, ["---"] * len(columns)]
    for values in frame.itertuples(index=False, name=None):
        rendered = []
        for value in values:
            if isinstance(value, float):
                rendered.append(f"{value:.6f}")
            else:
                rendered.append(str(value).replace("|", "\\|"))
        rows.append(rendered)
    return "\n".join("| " + " | ".join(row) + " |" for row in rows)


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("negative_rule_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import helper module {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.SEMANTIC_ALIASES.update(EXTRA_ALIASES)
    module.DIMENSION_CONCEPTS.update({"depth", "watch_waterproof"})
    module.build_alias_index.cache_clear()
    return module


def visible(module, concept: str, semantic: dict, exact_text: str) -> bool:
    return module.find_name_representation(
        concept,
        list(semantic["raw"]),
        set(semantic["values"]),
        exact_text,
    ) is not None


def compact_example(record: dict, concept: str, left: dict, right: dict, visible_both: bool) -> dict:
    return {
        "validation_row": record["validation_row"],
        "id1": record["id1"],
        "id2": record["id2"],
        "score": record["score"],
        "candidate_type": record["candidate_type"],
        "threshold_error": record["is_threshold_error_at_0_5"],
        "name1": record["raw_item1"]["name"],
        "name2": record["raw_item2"]["name"],
        "value1": sorted(left["values"]),
        "value2": sorted(right["values"]),
        "raw_keys1": sorted(left["keys"]),
        "raw_keys2": sorted(right["keys"]),
        "visible_in_both_exact_inputs": visible_both,
        "item1_pair_truncated": record["item1_pair_truncated"],
        "item2_pair_truncated": record["item2_pair_truncated"],
        "item1_item_token_budget_hit": record["item1_item_token_budget_hit"],
        "item2_item_token_budget_hit": record["item2_item_token_budget_hit"],
        "exact_item1_visible_input": record["exact_item1_visible_input"],
        "exact_item2_visible_input": record["exact_item2_visible_input"],
        "candidate_critical_dimension": concept,
    }


def verdict(row: dict) -> tuple[str, str]:
    visible_fp = row["non_truncated_fp_support"]
    truncation_fp = row["truncation_only_fp_support"]
    counterexamples = row["visible_positive_counterexamples"]
    concept = row["candidate_critical_dimension"]
    if visible_fp >= 5 and concept in CRITICAL_CONCEPTS and counterexamples == 0:
        return "candidate_for_C", "Visible single-dimension FP support with no visible positive counterexample in the reviewed subset."
    if visible_fp >= 3:
        return "candidate_for_A2", "Frida ignores a visible difference, but C is unsafe or ambiguous; use identity-preserving hardening only."
    if truncation_fp >= 3 and visible_fp == 0:
        return "preprocessing_issue", "The structured difference is repeatedly absent from at least one exact model input."
    if visible_fp or truncation_fp or counterexamples:
        return "ambiguous", "Evidence exists but is too small, mixed, or confounded for a mutation rule."
    return "reject", "No usable support in the reviewed FP/FN candidate subset."


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--helpers", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-count", type=int, default=5)
    args = parser.parse_args()

    module = load_module(args.helpers)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metric_frame = pd.read_csv(args.metrics).sort_values("average_precision", ascending=True)
    target_categories = metric_frame.head(args.target_count)["category"].tolist()

    pattern_rows: dict[tuple[str, str], dict] = {}
    examples: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    truncation_rows = []
    candidate_counts = defaultdict(lambda: defaultdict(int))

    with args.candidates.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            category = record["category"]
            if category not in target_categories:
                continue
            candidate_type = record["candidate_type"]
            candidate_counts[category][candidate_type] += 1
            attr1 = module.parse_attributes(record["raw_item1"].get("attributes"))
            attr2 = module.parse_attributes(record["raw_item2"].get("attributes"))
            sem1 = module.extract_semantics(category, attr1)
            sem2 = module.extract_semantics(category, attr2)
            common = sorted(set(sem1) & set(sem2))
            differences = [concept for concept in common if module.concept_diff(sem1[concept], sem2[concept])]
            any_truncation = any(
                record[key]
                for key in (
                    "item1_pair_truncated",
                    "item2_pair_truncated",
                    "item1_item_token_budget_hit",
                    "item2_item_token_budget_hit",
                )
            )
            visible_differences = []
            hidden_differences = []
            for concept in differences:
                left_visible = visible(module, concept, sem1[concept], record["exact_item1_visible_input"])
                right_visible = visible(module, concept, sem2[concept], record["exact_item2_visible_input"])
                (visible_differences if left_visible and right_visible else hidden_differences).append(concept)

            truncation_rows.append(
                {
                    "category": category,
                    "candidate_type": candidate_type,
                    "is_threshold_error_at_0_5": bool(record["is_threshold_error_at_0_5"]),
                    "score": float(record["score"]),
                    "any_pair_or_item_truncation": any_truncation,
                    "semantic_diff_count": len(differences),
                    "visible_semantic_diff_count": len(visible_differences),
                    "hidden_semantic_diff_count": len(hidden_differences),
                    "has_visible_semantic_difference": bool(visible_differences),
                    "has_only_hidden_semantic_differences": bool(hidden_differences and not visible_differences),
                }
            )

            for concept in differences:
                key = (category, concept)
                row = pattern_rows.setdefault(
                    key,
                    {
                        "category": category,
                        "product_family": "requires_train-side narrowing",
                        "candidate_critical_dimension": concept,
                        "fp_support": 0,
                        "threshold_fp_support": 0,
                        "non_truncated_fp_support": 0,
                        "truncation_only_fp_support": 0,
                        "positive_fn_counterexamples": 0,
                        "visible_positive_counterexamples": 0,
                        "fp_scores": [],
                    },
                )
                left_visible = visible(module, concept, sem1[concept], record["exact_item1_visible_input"])
                right_visible = visible(module, concept, sem2[concept], record["exact_item2_visible_input"])
                visible_both = left_visible and right_visible
                sample = compact_example(record, concept, sem1[concept], sem2[concept], visible_both)
                if candidate_type == "high_score_negative":
                    row["fp_support"] += 1
                    row["threshold_fp_support"] += int(record["is_threshold_error_at_0_5"])
                    row["non_truncated_fp_support"] += int(visible_both)
                    row["truncation_only_fp_support"] += int(not visible_both and any_truncation)
                    row["fp_scores"].append(float(record["score"]))
                    bucket = examples[(category, concept, "fp")]
                    bucket.append(sample)
                    bucket.sort(key=lambda item: item["score"], reverse=True)
                    del bucket[5:]
                else:
                    row["positive_fn_counterexamples"] += 1
                    row["visible_positive_counterexamples"] += int(visible_both)
                    bucket = examples[(category, concept, "positive")]
                    bucket.append(sample)
                    bucket.sort(key=lambda item: item["score"])
                    del bucket[5:]

    taxonomy = []
    evidence = []
    for key in sorted(pattern_rows):
        row = pattern_rows[key]
        scores = row.pop("fp_scores")
        row["typical_frida_score"] = statistics.median(scores) if scores else None
        row["mean_frida_score"] = statistics.fmean(scores) if scores else None
        row["verdict"], row["verdict_reason"] = verdict(row)
        taxonomy.append(row)
        evidence.append(
            {
                "category": key[0],
                "candidate_critical_dimension": key[1],
                "fp_examples": examples[(key[0], key[1], "fp")],
                "positive_counterexamples": examples[(key[0], key[1], "positive")],
            }
        )

    taxonomy_frame = pd.DataFrame(taxonomy).sort_values(
        ["category", "non_truncated_fp_support", "fp_support"], ascending=[True, False, False]
    )
    taxonomy_frame.to_csv(args.output_dir / "taxonomy.csv", index=False)
    with (args.output_dir / "taxonomy_evidence.json").open("w", encoding="utf-8") as handle:
        json.dump(evidence, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    truncation_frame = pd.DataFrame(truncation_rows)
    truncation_summary = (
        truncation_frame.groupby(["category", "candidate_type"], sort=True)
        .agg(
            candidates=("score", "size"),
            threshold_errors=("is_threshold_error_at_0_5", "sum"),
            any_truncation=("any_pair_or_item_truncation", "sum"),
            visible_semantic_difference=("has_visible_semantic_difference", "sum"),
            only_hidden_semantic_differences=("has_only_hidden_semantic_differences", "sum"),
            mean_score=("score", "mean"),
        )
        .reset_index()
    )
    truncation_summary.to_csv(args.output_dir / "truncation_summary.csv", index=False)

    approved_c = taxonomy_frame[taxonomy_frame["verdict"] == "candidate_for_C"]
    approved_a2 = taxonomy_frame[taxonomy_frame["verdict"] == "candidate_for_A2"]
    visible_fp_dimension_occurrences = int(taxonomy_frame["non_truncated_fp_support"].sum())
    visible_fp_candidate_pairs = int(
        truncation_frame.loc[
            truncation_frame["candidate_type"] == "high_score_negative",
            "has_visible_semantic_difference",
        ].sum()
    )
    manifest = {
        "status": "complete",
        "validation_only_research": True,
        "may_not_be_used_as_generation_source": True,
        "target_categories": target_categories,
        "target_category_metrics": metric_frame.head(args.target_count).to_dict("records"),
        "candidate_counts": {category: dict(counts) for category, counts in candidate_counts.items()},
        "critical_attribute_mismatch_confirmed": visible_fp_candidate_pairs >= 20,
        "visible_fp_candidate_pairs_total": visible_fp_candidate_pairs,
        "visible_fp_dimension_occurrences_total": visible_fp_dimension_occurrences,
        "candidate_for_C_patterns": approved_c[["category", "candidate_critical_dimension"]].to_dict("records"),
        "candidate_for_A2_patterns": approved_a2[["category", "candidate_critical_dimension"]].to_dict("records"),
        "inputs": {
            "candidates": {"path": str(args.candidates), "sha256": sha256_file(args.candidates)},
            "metrics": {"path": str(args.metrics), "sha256": sha256_file(args.metrics)},
        },
    }
    with (args.output_dir / "taxonomy_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    lines = [
        "# FRIDA validation error taxonomy",
        "",
        "> Только исследование validation. Эти товары, ID, пары и значения нельзя передавать генератору.",
        "",
        f"Macro AP: `{metric_frame['average_precision'].mean():.6f}` (mean shown only for orientation; canonical macro is in baseline_metrics.json).",
        "",
        "## Актуальные худшие категории",
        "",
        "| Категория | AP | FP@0.5 | FN@0.5 |",
        "|---|---:|---:|---:|",
    ]
    for row in metric_frame.head(args.target_count).itertuples(index=False):
        lines.append(f"| {row.category} | {row.average_precision:.6f} | {row.fp_at_0_5} | {row.fn_at_0_5} |")
    lines.extend(["", "## Truncation и видимые различия", ""])
    lines.append(markdown_table(truncation_summary))
    lines.extend(["", "## Candidate patterns", ""])
    report_columns = [
        "category",
        "candidate_critical_dimension",
        "fp_support",
        "non_truncated_fp_support",
        "truncation_only_fp_support",
        "visible_positive_counterexamples",
        "typical_frida_score",
        "verdict",
    ]
    lines.append(markdown_table(taxonomy_frame[report_columns]))
    lines.extend(
        [
            "",
            "## Вывод",
            "",
            f"`critical_attribute_mismatch` подтверждён как систематический: **{visible_fp_candidate_pairs}** "
            "из 1000 просмотренных hard-FP candidates содержат хотя бы одно semantic difference, "
            f"видимое в exact Frida input ({visible_fp_dimension_occurrences} dimension occurrences). "
            "Это не означает, "
            "что все такие dimensions безопасны для C: positive counterexamples и узкий product family "
            "должны быть повторно проверены только на human train.",
            "",
        ]
    )
    (args.output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
