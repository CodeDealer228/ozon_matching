"""Audit method-C pattern priors against canonical human train.

This script is read-only with respect to source data.  It does not call an LLM,
generate synthetic text, score with Frida, or train a model.  The historical
``negative_patterns`` catalog supplies hypotheses only; support and replacement
opportunities are reconstructed exclusively from canonical human train.
Validation contributes only item/pair deny-lists and its frozen fingerprint.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import math
import os
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SPLIT_BYTES = 6_192_178
EXPECTED_SPLIT_SHA256 = "1e3df234d6cdb8685256d5144b25f9a0a3e3a023f1053a0bd5b16c40e3df8d7e"
EXPECTED_VAL_FINGERPRINT = "847d04d081432cc081c2149b75271547bf5e25b14f9d8a5804704a5be2a4b3c8"
EXPECTED_TRAIN_ROWS = 256_006
EXPECTED_VAL_ROWS = 109_648

PATTERN_HEADER = re.compile(r"^## Generation pattern\s+(\d+)\s*$", re.IGNORECASE)
FIELD_HEADER = re.compile(r"^\*\*([^*]+?):\*\*\s*(.*)$")
PAIR_REFERENCE = re.compile(r"\bpair_\d+\b")


def file_sha256(path: Path, chunk_bytes: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(*values: object) -> str:
    payload = "\x1f".join(str(value) for value in values).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    os.replace(temporary, path)


def fingerprint_pairs(frame: pd.DataFrame) -> str:
    payload = "\n".join(
        sorted(f"{int(left)}-{int(right)}" for left, right in zip(frame.id1, frame.id2))
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def unordered_pairs(frame: pd.DataFrame) -> set[tuple[int, int]]:
    return {
        (min(int(left), int(right)), max(int(left), int(right)))
        for left, right in zip(frame.id1, frame.id2)
    }


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_pattern_markdown(path: Path, category: str) -> list[dict[str, object]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    try:
        source_path = str(path.relative_to(REPO_ROOT))
    except ValueError:
        source_path = str(path)
    starts = [index for index, line in enumerate(lines) if PATTERN_HEADER.match(line.strip())]
    patterns: list[dict[str, object]] = []
    for position, start in enumerate(starts):
        match = PATTERN_HEADER.match(lines[start].strip())
        if match is None:
            continue
        end = starts[position + 1] if position + 1 < len(starts) else len(lines)
        fields: dict[str, str] = {}
        current: str | None = None
        chunks: list[str] = []

        def flush() -> None:
            nonlocal current, chunks
            if current is not None:
                fields[current] = " ".join(value for value in chunks if value).strip()
            current, chunks = None, []

        for line in lines[start + 1 : end]:
            stripped = line.strip()
            field = FIELD_HEADER.match(stripped)
            if field:
                flush()
                current = field.group(1).strip().lower().replace(" ", "_")
                chunks = [field.group(2).strip()]
            elif current is not None and stripped and not stripped.startswith("##"):
                chunks.append(stripped)
        flush()
        pattern_index = int(match.group(1))
        required = {
            "generalized_rule",
            "scope",
            "generation_operation",
            "guards",
            "confidence",
            "negative_evidence",
            "positive_counterexamples",
        }
        patterns.append(
            {
                "category": category,
                "pattern_index": pattern_index,
                "pattern_id": f"{category}::{pattern_index}",
                "source_path": source_path,
                **fields,
                "missing_required_fields": sorted(required - set(fields)),
                "negative_evidence_ids": sorted(
                    set(PAIR_REFERENCE.findall(fields.get("negative_evidence", "")))
                ),
                "positive_counterexample_ids": sorted(
                    set(PAIR_REFERENCE.findall(fields.get("positive_counterexamples", "")))
                ),
            }
        )
    return patterns


def load_pattern_catalog(root: Path) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for path in sorted(root.glob("*/generation_patterns.md")):
        category_file = path.parent / "CATEGORY.txt"
        category = (
            category_file.read_text(encoding="utf-8").strip()
            if category_file.is_file()
            else path.parent.name
        )
        output.extend(parse_pattern_markdown(path, category))
    return output


def audit_catalog_references(
    catalog: list[dict[str, object]], root: Path
) -> list[dict[str, object]]:
    pairs_by_category: dict[str, dict[str, int]] = {}
    for category_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        pairs_path = category_dir / "pairs.jsonl"
        if not pairs_path.is_file():
            continue
        pair_targets: dict[str, int] = {}
        with pairs_path.open(encoding="utf-8") as stream:
            for line in stream:
                record = json.loads(line)
                pair_targets[str(record["sample_pair_id"])] = int(float(record["target"]))
        pairs_by_category[category_dir.name] = pair_targets

    rows: list[dict[str, object]] = []
    for pattern in catalog:
        category = str(pattern["category"])
        targets = pairs_by_category.get(category, {})
        negative_ids = list(pattern["negative_evidence_ids"])
        positive_ids = list(pattern["positive_counterexample_ids"])
        missing_negative = sorted(value for value in negative_ids if value not in targets)
        missing_positive = sorted(value for value in positive_ids if value not in targets)
        wrong_negative = sorted(value for value in negative_ids if targets.get(value) == 1)
        wrong_positive = sorted(value for value in positive_ids if targets.get(value) == 0)
        valid = not (
            pattern["missing_required_fields"]
            or missing_negative
            or missing_positive
            or wrong_negative
            or wrong_positive
        )
        rows.append(
            {
                **pattern,
                "negative_evidence_count": len(negative_ids),
                "positive_counterexample_count": len(positive_ids),
                "missing_negative_references": missing_negative,
                "missing_positive_references": missing_positive,
                "wrong_label_negative_references": wrong_negative,
                "wrong_label_positive_references": wrong_positive,
                "catalog_reference_audit_passed": valid,
            }
        )
    return rows


def load_registry(path: Path) -> dict[str, object]:
    registry = json.loads(path.read_text(encoding="utf-8"))
    rules = registry.get("rules")
    if not isinstance(rules, list) or not rules:
        raise RuntimeError("C rule registry has no rules")
    seen: set[str] = set()
    for rule in rules:
        rule_id = str(rule.get("rule_id", ""))
        if not rule_id or rule_id in seen:
            raise RuntimeError(f"Invalid or duplicate rule_id: {rule_id!r}")
        seen.add(rule_id)
        if rule.get("target_mode") not in {"all", "any"}:
            raise RuntimeError(f"Invalid target_mode for {rule_id}")
        if not rule.get("target_concepts") or not rule.get("context_concepts"):
            raise RuntimeError(f"Missing concepts for {rule_id}")
    return registry


def semantic_values(builder, helpers, item: dict[str, object], concept: str) -> set[str]:
    entry = item["semantics"].get(concept)
    if entry is None:
        return set()
    return builder.semantic_values(concept, entry, helpers)


def identifier_like(value: str) -> bool:
    compact = re.sub(r"[^0-9a-zа-я]+", "", str(value).lower().replace("ё", "е"))
    return 3 <= len(compact) <= 48 and any(character.isdigit() for character in compact)


def apply_target_value_policy(values: set[str], rule: dict[str, object]) -> set[str]:
    policy = rule.get("target_value_policy")
    if policy is None:
        return values
    if policy == "identifier_like":
        return {value for value in values if identifier_like(value)}
    raise RuntimeError(f"Unknown target_value_policy: {policy}")


def item_type_allowed(builder, helpers, item: dict[str, object], rule: dict[str, object]) -> bool:
    pattern = rule.get("item_type_allow_regex")
    if not pattern:
        return True
    values = semantic_values(builder, helpers, item, "item_type")
    return any(re.search(str(pattern), value, flags=re.IGNORECASE) for value in values)


def item_scope_allowed(builder, helpers, item: dict[str, object], rule: dict[str, object]) -> bool:
    if not item_type_allowed(builder, helpers, item, rule):
        return False
    name = str(item.get("name", ""))
    allow_pattern = rule.get("item_name_allow_regex")
    if allow_pattern and not re.search(str(allow_pattern), name, flags=re.IGNORECASE):
        return False
    deny_pattern = rule.get("item_name_deny_regex")
    return not (deny_pattern and re.search(str(deny_pattern), name, flags=re.IGNORECASE))


def compare_items_for_rule(
    builder,
    helpers,
    item1: dict[str, object],
    item2: dict[str, object],
    rule: dict[str, object],
) -> dict[str, object] | None:
    if not item_scope_allowed(builder, helpers, item1, rule) or not item_scope_allowed(
        builder, helpers, item2, rule
    ):
        return None
    target_details: dict[str, dict[str, list[str]]] = {}
    available_targets = 0
    different_targets: list[str] = []
    any_overlap = False
    for concept in rule["target_concepts"]:
        values1 = apply_target_value_policy(
            semantic_values(builder, helpers, item1, str(concept)), rule
        )
        values2 = apply_target_value_policy(
            semantic_values(builder, helpers, item2, str(concept)), rule
        )
        if not values1 or not values2:
            if rule["target_mode"] == "all":
                return None
            continue
        available_targets += 1
        overlap = values1 & values2
        any_overlap = any_overlap or bool(overlap)
        if not overlap:
            different_targets.append(str(concept))
        target_details[str(concept)] = {
            "left": sorted(values1),
            "right": sorted(values2),
        }
    if available_targets == 0:
        return None
    if len(different_targets) < int(rule["min_target_differences"]):
        return None
    required_differences = set(rule.get("required_different_concepts", []))
    if not required_differences.issubset(different_targets):
        return None
    if bool(rule.get("disallow_any_target_overlap")) and any_overlap:
        return None

    matched_context: dict[str, list[str]] = {}
    for concept in rule["context_concepts"]:
        values1 = semantic_values(builder, helpers, item1, str(concept))
        values2 = semantic_values(builder, helpers, item2, str(concept))
        overlap = values1 & values2
        if overlap:
            matched_context[str(concept)] = sorted(overlap)
    if len(matched_context) < int(rule["min_context_matches"]):
        return None
    required_context = set(rule.get("required_context_concepts", []))
    if not required_context.issubset(matched_context):
        return None

    visible = []
    for concept in different_targets:
        visible.append(
            builder.visible_in_name(item1, concept, helpers)
            or builder.visible_in_name(item2, concept, helpers)
        )
    return {
        "different_target_concepts": different_targets,
        "target_values": target_details,
        "matched_context": matched_context,
        "matched_context_count": len(matched_context),
        "target_visible": any(visible),
    }


def confidence_result(
    rule: dict[str, object],
    catalog_pattern: dict[str, object],
    rows: list[dict[str, object]],
) -> dict[str, object]:
    negatives = [row for row in rows if int(row["label"]) == 0]
    positives = [row for row in rows if int(row["label"]) == 1]
    total = len(rows)
    purity = len(negatives) / total if total else 0.0
    visibility = (
        sum(bool(row["target_visible"]) for row in negatives) / len(negatives)
        if negatives
        else 0.0
    )
    average_context = (
        sum(int(row["matched_context_count"]) for row in negatives) / len(negatives)
        if negatives
        else 0.0
    )
    prior = 0.20 if str(rule["prior_confidence"]).lower() == "strict" else 0.12
    support = 0.30 * min(math.log2(len(negatives) + 1) / 3.0, 1.0)
    purity_score = 0.25 * purity
    visibility_score = 0.10 * visibility
    context_score = 0.10 * min(
        average_context / max(int(rule["min_context_matches"]), 1), 1.0
    )
    catalog_score = 0.05 if catalog_pattern["catalog_reference_audit_passed"] else 0.0
    score = max(
        0.0,
        min(1.0, prior + support + purity_score + visibility_score + context_score + catalog_score),
    )
    counterexample_rate = len(positives) / total if total else 1.0
    if (
        str(rule["prior_confidence"]).lower() == "strict"
        and len(negatives) >= 3
        and score >= 0.70
        and counterexample_rate <= 0.20
    ):
        tier, weight = "A", 0.5
    elif len(negatives) >= 1 and score >= 0.45 and counterexample_rate <= 0.40:
        tier, weight = "B", 0.3
    else:
        tier, weight = "research", 0.15
    return {
        "negative_support": len(negatives),
        "positive_counterexamples": len(positives),
        "counterexample_rate": counterexample_rate,
        "negative_visibility_rate": visibility,
        "average_matched_contexts": average_context,
        "confidence_score": score,
        "confidence_tier": tier,
        "recommended_sample_weight": weight,
        "generation_allowed_without_manual_override": tier in {"A", "B"},
    }


def canonical_rule_audit(
    train_pairs: pd.DataFrame,
    records: dict[int, dict[str, object]],
    rules: list[dict[str, object]],
    catalog: list[dict[str, object]],
    builder,
    helpers,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    catalog_by_key = {
        (str(row["category"]), int(row["pattern_index"])): row for row in catalog
    }
    evidence_by_rule: dict[str, list[dict[str, object]]] = defaultdict(list)
    rules_by_category: dict[str, list[dict[str, object]]] = defaultdict(list)
    for rule in rules:
        rules_by_category[str(rule["category"])].append(rule)

    for pair in train_pairs.itertuples(index=False):
        item1 = records.get(int(pair.id1))
        item2 = records.get(int(pair.id2))
        if item1 is None or item2 is None or item1["category"] != item2["category"]:
            continue
        for rule in rules_by_category.get(str(item1["category"]), []):
            comparison = compare_items_for_rule(builder, helpers, item1, item2, rule)
            if comparison is None:
                continue
            evidence_by_rule[str(rule["rule_id"])].append(
                {
                    "rule_id": str(rule["rule_id"]),
                    "category": str(rule["category"]),
                    "id1": int(pair.id1),
                    "id2": int(pair.id2),
                    "label": int(pair.label),
                    "source_pair_sha256": stable_hash(
                        min(int(pair.id1), int(pair.id2)),
                        max(int(pair.id1), int(pair.id2)),
                        int(pair.label),
                    ),
                    **comparison,
                }
            )

    audit_rows: list[dict[str, object]] = []
    evidence_rows: list[dict[str, object]] = []
    for rule in rules:
        rule_id = str(rule["rule_id"])
        catalog_key = (str(rule["category"]), int(rule["catalog_pattern"]))
        if catalog_key not in catalog_by_key:
            raise RuntimeError(f"Registry rule {rule_id} points to missing pattern {catalog_key}")
        pattern = catalog_by_key[catalog_key]
        rows = evidence_by_rule.get(rule_id, [])
        result = confidence_result(rule, pattern, rows)
        audit_rows.append(
            {
                "rule_id": rule_id,
                "category": str(rule["category"]),
                "catalog_pattern": int(rule["catalog_pattern"]),
                "prior_confidence": str(rule["prior_confidence"]),
                "target_concepts": json.dumps(rule["target_concepts"], ensure_ascii=False),
                "context_concepts": json.dumps(rule["context_concepts"], ensure_ascii=False),
                "operation": str(rule["operation"]),
                "catalog_negative_evidence": int(pattern["negative_evidence_count"]),
                "catalog_positive_counterexamples": int(pattern["positive_counterexample_count"]),
                "catalog_reference_audit_passed": bool(pattern["catalog_reference_audit_passed"]),
                **result,
            }
        )
        for row in rows:
            evidence_rows.append(
                {
                    **row,
                    "different_target_concepts": json.dumps(
                        row["different_target_concepts"], ensure_ascii=False
                    ),
                    "target_values": json.dumps(row["target_values"], ensure_ascii=False),
                    "matched_context": json.dumps(row["matched_context"], ensure_ascii=False),
                }
            )
    return pd.DataFrame(audit_rows), pd.DataFrame(evidence_rows)


def item_context_keys(
    builder,
    helpers,
    item: dict[str, object],
    rule: dict[str, object],
) -> list[tuple[tuple[str, tuple[str, ...]], ...]]:
    available: list[tuple[str, tuple[str, ...]]] = []
    for concept in rule["context_concepts"]:
        values = semantic_values(builder, helpers, item, str(concept))
        if values:
            available.append((str(concept), tuple(sorted(values))))
    needed = int(rule["min_context_matches"])
    if len(available) < needed:
        return []
    required = set(rule.get("required_context_concepts", []))
    return [
        tuple(values)
        for values in itertools.combinations(available, needed)
        if required.issubset({concept for concept, _values in values})
    ]


def item_has_targets(builder, helpers, item, rule) -> bool:
    if not item_scope_allowed(builder, helpers, item, rule):
        return False
    present = sum(
        bool(
            apply_target_value_policy(
                semantic_values(builder, helpers, item, str(concept)), rule
            )
        )
        for concept in rule["target_concepts"]
    )
    if rule["target_mode"] == "all":
        return present == len(rule["target_concepts"])
    return present >= int(rule["min_target_differences"])


def build_opportunities(
    records: dict[int, dict[str, object]],
    rules: list[dict[str, object]],
    audit: pd.DataFrame,
    builder,
    helpers,
    max_per_rule: int,
    max_replacements_per_source: int,
) -> pd.DataFrame:
    audit_by_rule = {str(row.rule_id): row._asdict() for row in audit.itertuples(index=False)}
    rows: list[dict[str, object]] = []
    for rule in rules:
        rule_id = str(rule["rule_id"])
        audit_row = audit_by_rule[rule_id]
        groups: dict[tuple[tuple[str, tuple[str, ...]], ...], list[dict[str, object]]] = defaultdict(list)
        for item in records.values():
            if item["category"] != rule["category"] or not item_has_targets(
                builder, helpers, item, rule
            ):
                continue
            for context_key in item_context_keys(builder, helpers, item, rule):
                groups[context_key].append(item)

        rule_rows: list[dict[str, object]] = []
        seen: set[str] = set()
        for context_key, members in sorted(groups.items(), key=lambda value: repr(value[0])):
            if len(members) < 2:
                continue
            ordered = sorted(members, key=lambda item: int(item["id"]))
            for source in ordered:
                accepted_for_source = 0
                donors = sorted(
                    (item for item in ordered if int(item["id"]) != int(source["id"])),
                    key=lambda item: stable_hash(rule_id, source["id"], item["id"]),
                )
                for donor in donors:
                    comparison = compare_items_for_rule(
                        builder, helpers, source, donor, rule
                    )
                    if comparison is None:
                        continue
                    source_values = {
                        concept: sorted(
                            apply_target_value_policy(
                                semantic_values(builder, helpers, source, concept), rule
                            )
                        )
                        for concept in comparison["different_target_concepts"]
                    }
                    replacement_values = {
                        concept: sorted(
                            apply_target_value_policy(
                                semantic_values(builder, helpers, donor, concept), rule
                            )
                        )
                        for concept in comparison["different_target_concepts"]
                    }
                    identity = stable_hash(
                        rule_id,
                        source["id"],
                        json.dumps(replacement_values, sort_keys=True, ensure_ascii=False),
                    )
                    if identity in seen:
                        continue
                    seen.add(identity)
                    rule_rows.append(
                        {
                            "branch": "C",
                            "rule_id": rule_id,
                            "category": str(rule["category"]),
                            "confidence_tier": str(audit_row["confidence_tier"]),
                            "confidence_score": float(audit_row["confidence_score"]),
                            "recommended_sample_weight": float(
                                audit_row["recommended_sample_weight"]
                            ),
                            "generation_allowed_without_manual_override": bool(
                                audit_row["generation_allowed_without_manual_override"]
                            ),
                            "source_id": int(source["id"]),
                            "source_name": str(source["name"]),
                            "source_attributes": str(source["attributes"]),
                            "replacement_donor_id": int(donor["id"]),
                            "changed_target_concepts": json.dumps(
                                comparison["different_target_concepts"], ensure_ascii=False
                            ),
                            "source_values": json.dumps(source_values, ensure_ascii=False),
                            "replacement_values": json.dumps(
                                replacement_values, ensure_ascii=False
                            ),
                            "context_values": json.dumps(
                                {concept: list(values) for concept, values in context_key},
                                ensure_ascii=False,
                            ),
                            "target_visible_in_source_or_donor_name": bool(
                                comparison["target_visible"]
                            ),
                            "operation": str(rule["operation"]),
                            "opportunity_sha256": identity,
                        }
                    )
                    accepted_for_source += 1
                    if accepted_for_source >= max_replacements_per_source:
                        break
                if len(rule_rows) >= max_per_rule:
                    break
            if len(rule_rows) >= max_per_rule:
                break
        rows.extend(rule_rows[:max_per_rule])
    return pd.DataFrame(rows)


def write_report(
    catalog: list[dict[str, object]],
    audit: pd.DataFrame,
    opportunities: pd.DataFrame,
    manifest: dict[str, object],
) -> str:
    strict_count = sum(str(row.get("confidence", "")).lower() == "strict" for row in catalog)
    guarded_count = sum(str(row.get("confidence", "")).lower() == "guarded" for row in catalog)
    lines = [
        "# Method C catalog and canonical-train audit",
        "",
        "> Это read-only audit. Synthetic text не создавался, Alibaba и Kaggle не запускались.",
        "",
        "## Identity",
        "",
        f"- split SHA-256: `{manifest['split']['sha256']}`;",
        f"- validation fingerprint: `{manifest['split']['validation_fingerprint']}`;",
        f"- train rows: `{manifest['split']['train_rows']}`;",
        f"- validation rows: `{manifest['split']['validation_rows']}`;",
        f"- shared train/validation items: `{manifest['leak_checks']['shared_train_validation_items']}`;",
        f"- unordered pair overlap: `{manifest['leak_checks']['unordered_train_validation_overlap']}`.",
        "",
        "## Pattern catalog",
        "",
        f"Parsed patterns: `{len(catalog)}` (`strict={strict_count}`, `guarded={guarded_count}`).",
        f"Broken evidence references: `{manifest['catalog']['failed_reference_audits']}`.",
        "",
        "## Structured rules",
        "",
        "| Rule | Category | Negative support | Positive counterexamples | Score | Tier | Weight | Opportunities |",
        "|---|---|---:|---:|---:|---|---:|---:|",
    ]
    opportunity_counts = (
        opportunities["rule_id"].value_counts().to_dict() if not opportunities.empty else {}
    )
    for row in audit.sort_values(["confidence_tier", "confidence_score"], ascending=[True, False]).itertuples(index=False):
        lines.append(
            f"| {row.rule_id} | {row.category} | {row.negative_support} | "
            f"{row.positive_counterexamples} | {row.confidence_score:.3f} | "
            f"{row.confidence_tier} | {row.recommended_sample_weight:.2f} | "
            f"{int(opportunity_counts.get(row.rule_id, 0))} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Tier A/B — candidates for a manually reviewed pilot. Tier research remains visible in the output and is not silently deleted, but requires an explicit override before generation.",
            "",
            f"Total replacement opportunities: `{len(opportunities)}`.",
            f"Generation-ready A/B opportunities: `{int(opportunities['generation_allowed_without_manual_override'].sum()) if not opportunities.empty else 0}`.",
            "",
            "The next step is representative manual review and a deterministic mutation validator; this audit alone does not authorize paid generation.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split-path",
        type=Path,
        default=REPO_ROOT / "data/splits/matches_item_disjoint_70_30_seed42.parquet",
    )
    parser.add_argument(
        "--items-path",
        type=Path,
        default=REPO_ROOT / "data/items_human.parquet",
    )
    parser.add_argument(
        "--patterns-root", type=Path, default=REPO_ROOT / "negative_patterns"
    )
    parser.add_argument(
        "--registry-path",
        type=Path,
        default=REPO_ROOT
        / "synthetic_research/c_pipeline_variants/c_rule_registry_v1.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "artifacts/synthetic_v2/c_pattern_audit_v1",
    )
    parser.add_argument("--max-opportunities-per-rule", type=int, default=5000)
    parser.add_argument("--max-replacements-per-source", type=int, default=3)
    args = parser.parse_args()
    started = time.time()

    split_path = args.split_path.resolve()
    items_path = args.items_path.resolve()
    if split_path.stat().st_size != EXPECTED_SPLIT_BYTES:
        raise RuntimeError(f"Canonical split size mismatch: {split_path.stat().st_size}")
    split_sha = file_sha256(split_path)
    if split_sha != EXPECTED_SPLIT_SHA256:
        raise RuntimeError(f"Canonical split SHA-256 mismatch: {split_sha}")

    pairs = pd.read_parquet(split_path, columns=["id1", "id2", "label", "split"])
    train = pairs[pairs.split.eq("train")].copy()
    validation = pairs[pairs.split.eq("val")].copy()
    if len(train) != EXPECTED_TRAIN_ROWS or len(validation) != EXPECTED_VAL_ROWS:
        raise RuntimeError(f"Unexpected split counts: train={len(train)} val={len(validation)}")
    val_fingerprint = fingerprint_pairs(validation)
    if val_fingerprint != EXPECTED_VAL_FINGERPRINT:
        raise RuntimeError(f"Validation fingerprint mismatch: {val_fingerprint}")
    train_ids = set(train.id1.astype(int)) | set(train.id2.astype(int))
    val_ids = set(validation.id1.astype(int)) | set(validation.id2.astype(int))
    shared_items = train_ids & val_ids
    pair_overlap = unordered_pairs(train) & unordered_pairs(validation)
    if shared_items or pair_overlap:
        raise RuntimeError(
            f"Canonical split leak: shared_items={len(shared_items)} pairs={len(pair_overlap)}"
        )

    registry = load_registry(args.registry_path)
    rules = list(registry["rules"])
    categories = {str(rule["category"]) for rule in rules}
    catalog = audit_catalog_references(
        load_pattern_catalog(args.patterns_root), args.patterns_root
    )
    if not catalog:
        raise RuntimeError("No generation patterns parsed")

    builder = load_module(
        "c_audit_source_builder", REPO_ROOT / "scripts/build_synthetic_v2_source_pools.py"
    )
    helpers = builder.load_rule_helpers()
    unsupported_categories = categories - set(helpers.SEMANTIC_ALIASES)
    if unsupported_categories:
        raise RuntimeError(f"Registry categories lack semantic aliases: {unsupported_categories}")

    items = pd.read_parquet(items_path, columns=["id", "name", "attributes", "category"])
    if items.id.duplicated().any():
        raise RuntimeError("items_human contains duplicate IDs")
    items = items[items.id.isin(train_ids) & items.category.isin(categories)].copy()
    records, bad_json = builder.build_item_records(items, helpers)
    target_train = train[
        train.id1.astype(int).isin(records) & train.id2.astype(int).isin(records)
    ].copy()

    audit, evidence = canonical_rule_audit(
        target_train, records, rules, catalog, builder, helpers
    )
    opportunities = build_opportunities(
        records,
        rules,
        audit,
        builder,
        helpers,
        args.max_opportunities_per_rule,
        args.max_replacements_per_source,
    )
    if not opportunities.empty:
        opportunity_ids = set(opportunities.source_id.astype(int)) | set(
            opportunities.replacement_donor_id.astype(int)
        )
        if opportunity_ids - train_ids or opportunity_ids & val_ids:
            raise RuntimeError("C opportunity source/donor leak check failed")
        if opportunities.opportunity_sha256.duplicated().any():
            raise RuntimeError("Duplicate C opportunity identity")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    catalog_path = args.output_dir / "pattern_catalog.json"
    audit_path = args.output_dir / "canonical_rule_audit.csv"
    evidence_path = args.output_dir / "canonical_rule_evidence.parquet"
    opportunities_path = args.output_dir / "replacement_opportunities.parquet"
    report_path = args.output_dir / "report.md"
    manifest_path = args.output_dir / "manifest.json"
    write_json(catalog_path, catalog)
    audit.to_csv(audit_path, index=False, encoding="utf-8")
    evidence.to_parquet(evidence_path, index=False)
    opportunities.to_parquet(opportunities_path, index=False)

    manifest = {
        "status": "complete",
        "profile": "c_hybrid_soft_gate_audit_v2",
        "scope": "canonical_human_train_only",
        "synthetic_generation_performed": False,
        "llm_calls_performed": False,
        "model_scoring_performed": False,
        "training_performed": False,
        "validation_content_used": False,
        "validation_used_as_denylist_only": True,
        "split": {
            "path": str(split_path),
            "bytes": split_path.stat().st_size,
            "sha256": split_sha,
            "train_rows": len(train),
            "validation_rows": len(validation),
            "validation_fingerprint": val_fingerprint,
        },
        "items": {
            "path": str(items_path),
            "bytes": items_path.stat().st_size,
            "sha256": file_sha256(items_path),
            "canonical_train_items_audited": len(records),
            "bad_attribute_json": bad_json,
        },
        "leak_checks": {
            "shared_train_validation_items": len(shared_items),
            "unordered_train_validation_overlap": len(pair_overlap),
            "opportunity_validation_item_overlap": 0,
        },
        "catalog": {
            "patterns": len(catalog),
            "strict": sum(str(row.get("confidence", "")).lower() == "strict" for row in catalog),
            "guarded": sum(str(row.get("confidence", "")).lower() == "guarded" for row in catalog),
            "failed_reference_audits": sum(
                not bool(row["catalog_reference_audit_passed"]) for row in catalog
            ),
        },
        "registry": {
            "path": str(args.registry_path.resolve()),
            "sha256": file_sha256(args.registry_path),
            "version": registry["version"],
            "rules": len(rules),
        },
        "audit": {
            "tier_counts": {
                str(key): int(value)
                for key, value in audit.confidence_tier.value_counts().sort_index().items()
            },
            "evidence_rows": len(evidence),
        },
        "opportunities": {
            "rows": len(opportunities),
            "generation_ready_rows": int(
                opportunities.generation_allowed_without_manual_override.sum()
            )
            if not opportunities.empty
            else 0,
            "unique_sources": int(opportunities.source_id.nunique())
            if not opportunities.empty
            else 0,
            "max_source_reuse": int(opportunities.source_id.value_counts().max())
            if not opportunities.empty
            else 0,
            "counts_by_rule": {
                str(key): int(value)
                for key, value in opportunities.rule_id.value_counts().sort_index().items()
            }
            if not opportunities.empty
            else {},
        },
        "outputs": {},
        "elapsed_seconds": time.time() - started,
    }
    report_path.write_text(
        write_report(catalog, audit, opportunities, manifest), encoding="utf-8"
    )
    for name, path in (
        ("pattern_catalog", catalog_path),
        ("canonical_rule_audit", audit_path),
        ("canonical_rule_evidence", evidence_path),
        ("replacement_opportunities", opportunities_path),
        ("report", report_path),
    ):
        manifest["outputs"][name] = {
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
