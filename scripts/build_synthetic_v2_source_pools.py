"""Build leak-checked A2/C source pools from canonical human train only.

Validation is read only to construct a deny-list and to prove zero overlap.  No
validation item content, label, pair, or value enters either source pool.

Outputs (ignored local artifacts):

    artifacts/synthetic_v2/source_pools/a2_source_pool.parquet
    artifacts/synthetic_v2/source_pools/c_candidate_pool.parquet
    artifacts/synthetic_v2/source_pools/c_train_research_candidates.parquet
    artifacts/synthetic_v2/source_pools/c_replacement_source_pool.parquet
    artifacts/synthetic_v2/source_pools/c_family_evidence.csv
    artifacts/synthetic_v2/source_pools/source_pool_manifest.json
    artifacts/synthetic_v2/source_pools/report.md

The C output is deliberately a candidate pool.  A family becomes eligible for
the pilot only when narrow human-train evidence passes the thresholds below.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
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

TARGET_CATEGORIES = (
    "Обувь",
    "Ювелирные изделия",
    "Одежда",
    "Галантерея и аксессуары",
    "Автотовары",
)

C_HYPOTHESES = {
    "Обувь": ("russian_size", "manufacturer_size", "generic_size"),
    "Ювелирные изделия": ("item_size", "ring_size", "article"),
    "Одежда": ("product_color", "russian_size", "manufacturer_size", "generic_size"),
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(*values: object) -> str:
    payload = "\x1f".join(str(value) for value in values).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def strict_identifier_values(entry: dict[str, object], helpers) -> set[str]:
    """Return full identifiers, excluding short token fragments.

    The generic research helper intentionally retains digit-bearing fragments
    for recall.  They are unsafe as exact family keys (for example many
    unrelated jewelry articles share ``1110``), so C uses full raw values.
    """

    result: set[str] = set()
    for raw in entry.get("raw", []):
        normalized = helpers.normalize_text(raw)
        compact = re.sub(r"[^0-9a-zа-я]+", "", normalized)
        if len(compact) >= 6 and any(char.isdigit() for char in compact):
            result.add(compact)
    return result


def strict_item_size_values(entry: dict[str, object]) -> set[str]:
    result: set[str] = set()
    for raw in entry.get("raw", []):
        text = str(raw).lower().replace("ё", "е")
        for value in re.findall(r"(?<!\d)\d{1,2}(?:[.,]\d+)?(?!\d)", text):
            normalized = value.replace(",", ".").lstrip("0") or "0"
            if normalized.endswith(".0"):
                normalized = normalized[:-2]
            number = float(normalized)
            if 10 <= number <= 40:
                result.add(normalized)
    return result


def semantic_values(concept: str, entry: dict[str, object], helpers) -> set[str]:
    if concept == "article":
        return strict_identifier_values(entry, helpers)
    if concept in {"item_size", "ring_size"}:
        return strict_item_size_values(entry)
    return {str(value) for value in entry.get("values", set())}


def unordered_pairs(frame: pd.DataFrame) -> set[tuple[int, int]]:
    return {
        (min(int(left), int(right)), max(int(left), int(right)))
        for left, right in zip(frame["id1"], frame["id2"])
    }


def fingerprint_pairs(frame: pd.DataFrame) -> str:
    payload = "\n".join(
        sorted(f"{int(left)}-{int(right)}" for left, right in zip(frame["id1"], frame["id2"]))
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_rule_helpers():
    path = REPO_ROOT / "scripts" / "analyze_synthetic_negative_rules.py"
    spec = importlib.util.spec_from_file_location("synthetic_negative_rules_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import helper module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def exact_context(item1: dict[str, object], item2: dict[str, object], concept: str, helpers) -> tuple[str, str] | None:
    semantics1 = item1["semantics"]
    semantics2 = item2["semantics"]

    for context_concept in ("product_model", "article"):
        if context_concept == concept:
            continue
        if context_concept not in semantics1 or context_concept not in semantics2:
            continue
        shared = sorted(
            semantic_values(context_concept, semantics1[context_concept], helpers)
            & semantic_values(context_concept, semantics2[context_concept], helpers)
        )
        if shared:
            value_hash = stable_hash(*shared)[:20]
            return f"shared_{context_concept}", value_hash

    skeleton1 = name_skeleton(item1, concept, helpers)
    skeleton2 = name_skeleton(item2, concept, helpers)
    if skeleton1 and skeleton1 == skeleton2 and len(skeleton1) >= 8:
        return "exact_name_skeleton", stable_hash(skeleton1)[:20]
    return None


def name_skeleton(item: dict[str, object], concept: str, helpers) -> str:
    text = helpers.normalize_text(item["name"])
    entry = item["semantics"].get(concept)
    if not text or not entry:
        return text

    candidates: set[str] = set()
    for raw in entry["raw"]:
        normalized = helpers.normalize_text(raw)
        if normalized:
            candidates.add(normalized)
            candidates.add(normalized.replace(" ", ""))
    if concept in helpers.COLOR_CONCEPTS:
        for canonical in entry["values"]:
            for alias in helpers.COLOR_GROUPS.get(canonical, ()):
                candidates.add(helpers.normalize_text(alias))

    for value in sorted(candidates, key=len, reverse=True):
        if not value:
            continue
        text = re.sub(rf"(?<![0-9a-zа-я]){re.escape(value)}(?![0-9a-zа-я])", " ", text)
    text = re.sub(r"\s+", " ", text).strip(" -_,.;/")
    return text


def visible_in_name(item: dict[str, object], concept: str, helpers) -> bool:
    entry = item["semantics"].get(concept)
    if not entry:
        return False
    return helpers.find_name_representation(
        concept,
        entry["raw"],
        entry["values"],
        item["name"],
    ) is not None


def build_item_records(items: pd.DataFrame, helpers) -> tuple[dict[int, dict[str, object]], dict[str, int]]:
    records: dict[int, dict[str, object]] = {}
    bad_json: Counter[str] = Counter()
    for row in items.itertuples(index=False):
        attributes = helpers.parse_attributes(row.attributes)
        if row.attributes and not attributes:
            bad_json[str(row.category)] += 1
        semantics = (
            helpers.extract_semantics(row.category, attributes)
            if row.category in helpers.SEMANTIC_ALIASES
            else {}
        )
        records[int(row.id)] = {
            "id": int(row.id),
            "name": str(row.name or ""),
            "attributes": str(row.attributes or ""),
            "category": str(row.category),
            "semantics": semantics,
        }
    return records, dict(bad_json)


def build_a2_pool(
    train_pairs: pd.DataFrame,
    records: dict[int, dict[str, object]],
    helpers,
    per_category: int,
) -> pd.DataFrame:
    candidates: list[dict[str, object]] = []
    negatives = train_pairs[train_pairs["label"].eq(0)]
    for row in negatives.itertuples(index=False):
        item1 = records.get(int(row.id1))
        item2 = records.get(int(row.id2))
        if item1 is None or item2 is None or item1["category"] != item2["category"]:
            continue
        category = item1["category"]
        if category not in TARGET_CATEGORIES:
            continue
        similarity, sequence, jaccard = helpers.name_similarity(item1["name"], item2["name"])
        # Near-identical titles leave no meaningful room for A2 hardening and
        # are more likely to be noisy human negatives.  Keep a difficult but
        # genuinely rewritable band instead.
        if not (0.45 <= similarity <= 0.90):
            continue
        if len(helpers.normalize_text(item1["name"])) < 8 or len(helpers.normalize_text(item2["name"])) < 8:
            continue
        candidates.append(
            {
                "branch": "A2",
                "category": category,
                "id1": int(row.id1),
                "id2": int(row.id2),
                "label": 0,
                "name1": item1["name"],
                "name2": item2["name"],
                "attributes1": item1["attributes"],
                "attributes2": item2["attributes"],
                "name_similarity": float(similarity),
                "sequence_similarity": float(sequence),
                "token_jaccard": float(jaccard),
                "source_pair_sha256": stable_hash(min(int(row.id1), int(row.id2)), max(int(row.id1), int(row.id2)), 0),
            }
        )

    candidates.sort(
        key=lambda value: (
            value["category"],
            -float(value["name_similarity"]),
            str(value["source_pair_sha256"]),
        )
    )
    selected: list[dict[str, object]] = []
    item_use: Counter[tuple[str, int]] = Counter()
    category_use: Counter[str] = Counter()
    for value in candidates:
        category = str(value["category"])
        if category_use[category] >= per_category:
            continue
        keys = ((category, int(value["id1"])), (category, int(value["id2"])))
        if any(item_use[key] >= 1 for key in keys):
            continue
        selected.append(value)
        category_use[category] += 1
        for key in keys:
            item_use[key] += 1
    return pd.DataFrame(selected)


def build_c_candidates(
    train_pairs: pd.DataFrame,
    records: dict[int, dict[str, object]],
    helpers,
    min_negative_support: int,
    max_positive_counterexamples: int,
    per_category: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw: list[dict[str, object]] = []
    for row in train_pairs.itertuples(index=False):
        item1 = records.get(int(row.id1))
        item2 = records.get(int(row.id2))
        if item1 is None or item2 is None or item1["category"] != item2["category"]:
            continue
        category = item1["category"]
        concepts = C_HYPOTHESES.get(category)
        if not concepts:
            continue
        semantics1 = item1["semantics"]
        semantics2 = item2["semantics"]
        common = sorted(set(semantics1) & set(semantics2))
        differences = [
            candidate
            for candidate in common
            if not (
                semantic_values(candidate, semantics1[candidate], helpers)
                & semantic_values(candidate, semantics2[candidate], helpers)
            )
        ]
        equal_other = len(common) - len(differences)
        similarity, sequence, jaccard = helpers.name_similarity(item1["name"], item2["name"])
        if similarity < 0.65 or equal_other < 2:
            continue
        for concept in concepts:
            if concept not in differences:
                continue
            other_differences = [value for value in differences if value != concept]
            if len(other_differences) > 1:
                continue
            context = exact_context(item1, item2, concept, helpers)
            if context is None:
                continue
            context_type, context_hash = context
            visible1 = visible_in_name(item1, concept, helpers)
            visible2 = visible_in_name(item2, concept, helpers)
            raw.append(
                {
                    "branch": "C",
                    "category": category,
                    "critical_dimension": concept,
                    "family_context_type": context_type,
                    "family_context_hash": context_hash,
                    "id1": int(row.id1),
                    "id2": int(row.id2),
                    "label": int(row.label),
                    "name1": item1["name"],
                    "name2": item2["name"],
                    "attributes1": item1["attributes"],
                    "attributes2": item2["attributes"],
                    "value1": json.dumps(sorted(semantic_values(concept, semantics1[concept], helpers)), ensure_ascii=False),
                    "value2": json.dumps(sorted(semantic_values(concept, semantics2[concept], helpers)), ensure_ascii=False),
                    "raw_value1": json.dumps(semantics1[concept]["raw"], ensure_ascii=False),
                    "raw_value2": json.dumps(semantics2[concept]["raw"], ensure_ascii=False),
                    "dimension_visible_in_name1": bool(visible1),
                    "dimension_visible_in_name2": bool(visible2),
                    "name_similarity": float(similarity),
                    "sequence_similarity": float(sequence),
                    "token_jaccard": float(jaccard),
                    "equal_other_known_semantics": int(equal_other),
                    "other_semantic_differences": json.dumps(other_differences, ensure_ascii=False),
                    "source_pair_sha256": stable_hash(min(int(row.id1), int(row.id2)), max(int(row.id1), int(row.id2)), int(row.label)),
                }
            )

    raw_frame = pd.DataFrame(raw)
    evidence_rows: list[dict[str, object]] = []
    if raw_frame.empty:
        return raw_frame, pd.DataFrame(evidence_rows), raw_frame

    family_columns = ["category", "critical_dimension", "family_context_type", "family_context_hash"]
    eligibility: dict[tuple[object, ...], bool] = {}
    for key, group in raw_frame.groupby(family_columns, sort=True, dropna=False):
        negative = group[group["label"].eq(0)]
        positive = group[group["label"].eq(1)]
        visible_negative = negative[
            negative["dimension_visible_in_name1"] | negative["dimension_visible_in_name2"]
        ]
        eligible = (
            len(negative) >= min_negative_support
            and len(positive) <= max_positive_counterexamples
            and len(visible_negative) >= min_negative_support
        )
        eligibility[tuple(key)] = bool(eligible)
        evidence_rows.append(
            {
                **dict(zip(family_columns, key)),
                "negative_support": int(len(negative)),
                "positive_counterexamples": int(len(positive)),
                "visible_negative_support": int(len(visible_negative)),
                "distinct_negative_items": int(len(set(negative["id1"]) | set(negative["id2"]))),
                "eligible_for_pilot": bool(eligible),
                "verdict": "candidate_for_C" if eligible else "reject_or_more_train_evidence_needed",
            }
        )

    raw_frame["eligible_for_pilot"] = [
        eligibility[tuple(row[column] for column in family_columns)]
        for _, row in raw_frame.iterrows()
    ]
    candidates = raw_frame[
        raw_frame["label"].eq(0) & raw_frame["eligible_for_pilot"]
    ].copy()
    candidates.sort_values(
        ["category", "critical_dimension", "name_similarity", "source_pair_sha256"],
        ascending=[True, True, False, True],
        inplace=True,
    )
    selected_parts: list[pd.DataFrame] = []
    for _category, group in candidates.groupby("category", sort=True):
        selected_parts.append(group.head(per_category))
    selected = pd.concat(selected_parts, ignore_index=True) if selected_parts else candidates.head(0)
    evidence = pd.DataFrame(evidence_rows).sort_values(
        ["eligible_for_pilot", "category", "critical_dimension", "negative_support"],
        ascending=[False, True, True, False],
    )
    return selected, evidence, raw_frame


def build_c_replacement_pool(
    records: dict[int, dict[str, object]],
    train_research: pd.DataFrame,
    helpers,
    per_category: int,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    """Mine replacement candidates only after a narrow rule passes train evidence.

    The first supported rule is deliberately specific: jewelry rings sharing
    an exact manufacturer article, with item size as the only controlled
    dimension.  Each replacement value and its donor item occur in human train.
    """

    columns = [
        "branch", "category", "critical_dimension", "rule_context", "family_context_hash",
        "source_id", "source_name", "source_attributes", "source_value",
        "replacement_value", "replacement_donor_id", "family_item_count",
        "family_distinct_values", "human_negative_rule_support",
        "human_positive_rule_counterexamples", "source_value_visible_in_name",
        "source_candidate_sha256",
    ]
    if train_research.empty:
        return pd.DataFrame(columns=columns), []

    rule_evidence: list[dict[str, object]] = []
    approved_rules: set[tuple[str, str, str]] = set()
    grouped = train_research.groupby(
        ["category", "critical_dimension", "family_context_type"],
        sort=True,
        dropna=False,
    )
    for key, group in grouped:
        negative = group[group["label"].eq(0)]
        positive = group[group["label"].eq(1)]
        visible_negative = negative[
            negative["dimension_visible_in_name1"] | negative["dimension_visible_in_name2"]
        ]
        approved = len(negative) >= 3 and len(positive) == 0 and len(visible_negative) >= 3
        rule_evidence.append(
            {
                "category": key[0],
                "critical_dimension": key[1],
                "family_context_type": key[2],
                "negative_support": int(len(negative)),
                "positive_counterexamples": int(len(positive)),
                "visible_negative_support": int(len(visible_negative)),
                "approved_for_replacement_mining": bool(approved),
            }
        )
        if approved:
            approved_rules.add(tuple(key))

    jewelry_rule = ("Ювелирные изделия", "item_size", "shared_article")
    if jewelry_rule not in approved_rules:
        return pd.DataFrame(columns=columns), rule_evidence

    support = next(value for value in rule_evidence if (
        value["category"], value["critical_dimension"], value["family_context_type"]
    ) == jewelry_rule)
    family_items: dict[str, list[tuple[dict[str, object], str]]] = defaultdict(list)
    for item in records.values():
        if item["category"] != "Ювелирные изделия":
            continue
        semantics = item["semantics"]
        if "article" not in semantics or "item_size" not in semantics:
            continue
        normalized_name = helpers.normalize_text(item["name"])
        item_types = set(semantics.get("item_type", {}).get("values", set()))
        is_ring = "кольц" in normalized_name or any("кольц" in value for value in item_types)
        if not is_ring:
            continue
        if not visible_in_name(item, "item_size", helpers):
            continue
        for article in strict_identifier_values(semantics["article"], helpers):
            family_hash = stable_hash("Ювелирные изделия", "ring_exact_article", article)[:20]
            family_items[family_hash].append((item, article))

    rows: list[dict[str, object]] = []
    for family_hash, members in sorted(family_items.items()):
        unique_members = {int(item["id"]): item for item, _article in members}
        values_to_donors: dict[str, list[int]] = defaultdict(list)
        for item in unique_members.values():
            for value in strict_item_size_values(item["semantics"]["item_size"]):
                values_to_donors[str(value)].append(int(item["id"]))
        if len(unique_members) < 2 or len(values_to_donors) < 2:
            continue
        for source_id, item in sorted(unique_members.items()):
            source_values = strict_item_size_values(item["semantics"]["item_size"])
            replacements = sorted(set(values_to_donors) - source_values)
            for replacement in replacements[:3]:
                donor_id = min(values_to_donors[replacement])
                rows.append(
                    {
                        "branch": "C",
                        "category": "Ювелирные изделия",
                        "critical_dimension": "item_size",
                        "rule_context": "ring_exact_article",
                        "family_context_hash": family_hash,
                        "source_id": source_id,
                        "source_name": item["name"],
                        "source_attributes": item["attributes"],
                        "source_value": json.dumps(sorted(source_values), ensure_ascii=False),
                        "replacement_value": replacement,
                        "replacement_donor_id": donor_id,
                        "family_item_count": len(unique_members),
                        "family_distinct_values": len(values_to_donors),
                        "human_negative_rule_support": int(support["negative_support"]),
                        "human_positive_rule_counterexamples": int(support["positive_counterexamples"]),
                        "source_value_visible_in_name": True,
                        "source_candidate_sha256": stable_hash(source_id, "item_size", replacement, family_hash),
                    }
                )

    rows.sort(key=lambda value: (
        -int(value["family_distinct_values"]),
        -int(value["family_item_count"]),
        str(value["source_candidate_sha256"]),
    ))
    return pd.DataFrame(rows[:per_category], columns=columns), rule_evidence


def counts_by_category(frame: pd.DataFrame) -> dict[str, int]:
    if frame.empty or "category" not in frame:
        return {}
    return {str(key): int(value) for key, value in frame["category"].value_counts().sort_index().items()}


def write_report(manifest: dict[str, object], evidence: pd.DataFrame) -> str:
    lines = [
        "# Synthetic v2 human-train source pools",
        "",
        "> Источник — только canonical human train. Validation использован только как deny-list для проверки утечки.",
        "",
        "## Identity checks",
        "",
        f"- split SHA-256: `{manifest['split']['sha256']}`;",
        f"- train rows: `{manifest['split']['train_rows']}`;",
        f"- validation rows: `{manifest['split']['validation_rows']}`;",
        f"- shared train/validation items: `{manifest['leak_checks']['shared_train_validation_items']}`;",
        f"- unordered train/validation overlap: `{manifest['leak_checks']['unordered_train_validation_overlap']}`;",
        "",
        "## A2",
        "",
        f"Selected rows: `{manifest['a2']['rows']}`.",
        "",
    ]
    for category, count in manifest["a2"]["counts_by_category"].items():
        lines.append(f"- {category}: `{count}`")
    lines.extend([
        "",
        "## C",
        "",
        f"Selected observed-pair rows under the per-family gate: `{manifest['c']['rows']}`.",
        f"Replacement candidates under an approved train-level narrow rule: `{manifest['c']['replacement_candidate_rows']}`.",
        "",
    ])
    for category, count in manifest["c"]["counts_by_category"].items():
        lines.append(f"- {category}: `{count}`")
    for category, count in manifest["c"]["replacement_counts_by_category"].items():
        lines.append(f"- replacement candidates / {category}: `{count}`")
    eligible = evidence[evidence["eligible_for_pilot"]] if not evidence.empty else evidence
    lines.extend(["", f"Eligible narrow families: `{len(eligible)}`.", ""])
    if not eligible.empty:
        lines.extend([
            "| Категория | Dimension | Context | Negative support | Positive counterexamples |",
            "|---|---|---|---:|---:|",
        ])
        for row in eligible.itertuples(index=False):
            lines.append(
                f"| {row.category} | {row.critical_dimension} | {row.family_context_type} | "
                f"{row.negative_support} | {row.positive_counterexamples} |"
            )
    else:
        lines.append("Ни одна narrow family пока не прошла строгий C gate; ослаблять gate автоматически запрещено.")
    lines.extend([
        "",
        "C candidates остаются source evidence, а не готовой синтетикой. Перед pilot требуется few-shot pack и controlled mutation validator.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split-path",
        type=Path,
        default=REPO_ROOT / "data" / "splits" / "matches_item_disjoint_70_30_seed42.parquet",
    )
    parser.add_argument(
        "--items-path",
        type=Path,
        default=REPO_ROOT / "data" / "items_human.parquet",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "artifacts" / "synthetic_v2" / "source_pools",
    )
    parser.add_argument("--a2-per-category", type=int, default=100)
    parser.add_argument("--c-per-category", type=int, default=100)
    parser.add_argument("--c-min-negative-support", type=int, default=2)
    parser.add_argument("--c-max-positive-counterexamples", type=int, default=0)
    args = parser.parse_args()

    started = time.time()
    split_path = args.split_path.resolve()
    items_path = args.items_path.resolve()
    if split_path.stat().st_size != EXPECTED_SPLIT_BYTES:
        raise RuntimeError(f"Canonical split size mismatch: {split_path.stat().st_size}")
    split_sha = file_sha256(split_path)
    if split_sha != EXPECTED_SPLIT_SHA256:
        raise RuntimeError(f"Canonical split SHA-256 mismatch: {split_sha}")

    all_pairs = pd.read_parquet(split_path, columns=["id1", "id2", "label", "split"])
    train_pairs = all_pairs[all_pairs["split"].eq("train")].copy()
    val_refs = all_pairs[all_pairs["split"].eq("val")].copy()
    if len(train_pairs) != EXPECTED_TRAIN_ROWS or len(val_refs) != EXPECTED_VAL_ROWS:
        raise RuntimeError(f"Unexpected split counts: train={len(train_pairs)} val={len(val_refs)}")
    if not set(all_pairs["label"].dropna().unique()).issubset({0, 1}):
        raise RuntimeError("Human labels are not binary")

    train_ids = set(train_pairs["id1"]) | set(train_pairs["id2"])
    val_ids = set(val_refs["id1"]) | set(val_refs["id2"])
    shared_items = train_ids & val_ids
    pair_overlap = unordered_pairs(train_pairs) & unordered_pairs(val_refs)
    if shared_items or pair_overlap:
        raise RuntimeError(
            f"Leak check failed: shared_items={len(shared_items)} pair_overlap={len(pair_overlap)}"
        )
    val_fingerprint = fingerprint_pairs(val_refs)
    if val_fingerprint != EXPECTED_VAL_FINGERPRINT:
        raise RuntimeError(f"Validation fingerprint mismatch: {val_fingerprint}")

    items = pd.read_parquet(items_path, columns=["id", "name", "attributes", "category"])
    if items["id"].duplicated().any():
        raise RuntimeError("items_human contains duplicate item IDs")
    items = items[items["id"].isin(train_ids) & items["category"].isin(TARGET_CATEGORIES)].copy()
    category_by_id = items.set_index("id")["category"]
    train_pairs["category1"] = train_pairs["id1"].map(category_by_id)
    train_pairs["category2"] = train_pairs["id2"].map(category_by_id)
    target_pairs = train_pairs[
        train_pairs["category1"].isin(TARGET_CATEGORIES)
        & train_pairs["category1"].eq(train_pairs["category2"])
    ].copy()
    target_ids = set(target_pairs["id1"]) | set(target_pairs["id2"])
    missing = target_ids - set(items["id"])
    if missing:
        raise RuntimeError(f"Missing target item rows: {len(missing)}")

    helpers = load_rule_helpers()
    records, bad_json = build_item_records(items, helpers)
    a2 = build_a2_pool(target_pairs, records, helpers, args.a2_per_category)
    c_pool, c_evidence, c_research = build_c_candidates(
        target_pairs,
        records,
        helpers,
        args.c_min_negative_support,
        args.c_max_positive_counterexamples,
        args.c_per_category,
    )
    c_replacements, c_rule_evidence = build_c_replacement_pool(
        records,
        c_research,
        helpers,
        args.c_per_category,
    )

    for name, frame in (("A2", a2), ("C", c_pool)):
        if frame.empty:
            continue
        source_ids = set(frame["id1"]) | set(frame["id2"])
        if not source_ids.issubset(train_ids) or source_ids & val_ids:
            raise RuntimeError(f"{name} source pool failed item leak check")
        if set(frame["label"].unique()) != {0}:
            raise RuntimeError(f"{name} source pool contains non-negative labels")
    if not c_replacements.empty:
        replacement_ids = set(c_replacements["source_id"]) | set(c_replacements["replacement_donor_id"])
        if not replacement_ids.issubset(train_ids) or replacement_ids & val_ids:
            raise RuntimeError("C replacement source pool failed item leak check")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    a2_path = args.output_dir / "a2_source_pool.parquet"
    c_path = args.output_dir / "c_candidate_pool.parquet"
    c_research_path = args.output_dir / "c_train_research_candidates.parquet"
    c_replacements_path = args.output_dir / "c_replacement_source_pool.parquet"
    c_rule_evidence_path = args.output_dir / "c_rule_evidence.json"
    evidence_path = args.output_dir / "c_family_evidence.csv"
    a2.to_parquet(a2_path, index=False)
    c_pool.to_parquet(c_path, index=False)
    c_research.to_parquet(c_research_path, index=False)
    c_replacements.to_parquet(c_replacements_path, index=False)
    c_rule_evidence_path.write_text(
        json.dumps(c_rule_evidence, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    c_evidence.to_csv(evidence_path, index=False, encoding="utf-8")

    manifest = {
        "status": "complete",
        "scope": "canonical_human_train_only",
        "validation_content_used": False,
        "validation_used_as_denylist_only": True,
        "split": {
            "path": str(split_path),
            "bytes": split_path.stat().st_size,
            "sha256": split_sha,
            "seed": 42,
            "train_rows": len(train_pairs),
            "validation_rows": len(val_refs),
            "validation_fingerprint": val_fingerprint,
        },
        "items": {
            "path": str(items_path),
            "bytes": items_path.stat().st_size,
            "sha256": file_sha256(items_path),
            "target_train_items": len(items),
            "bad_attribute_json": bad_json,
        },
        "leak_checks": {
            "shared_train_validation_items": len(shared_items),
            "unordered_train_validation_overlap": len(pair_overlap),
            "a2_validation_item_overlap": len((set(a2.get("id1", [])) | set(a2.get("id2", []))) & val_ids),
            "c_validation_item_overlap": len((set(c_pool.get("id1", [])) | set(c_pool.get("id2", []))) & val_ids),
            "c_replacement_validation_item_overlap": len(
                (set(c_replacements.get("source_id", [])) | set(c_replacements.get("replacement_donor_id", [])))
                & val_ids
            ),
        },
        "a2": {
            "rows": len(a2),
            "counts_by_category": counts_by_category(a2),
            "attributes_immutable": True,
            "mutable_fields": ["name2"],
            "selection": "name similarity in [0.45, 0.90], then highest deterministic similarity; each item used at most once",
        },
        "c": {
            "rows": len(c_pool),
            "train_research_candidate_rows": len(c_research),
            "replacement_candidate_rows": len(c_replacements),
            "replacement_counts_by_category": counts_by_category(c_replacements),
            "counts_by_category": counts_by_category(c_pool),
            "eligible_families": int(c_evidence["eligible_for_pilot"].sum()) if not c_evidence.empty else 0,
            "hypotheses": C_HYPOTHESES,
            "min_negative_support": args.c_min_negative_support,
            "max_positive_counterexamples": args.c_max_positive_counterexamples,
            "requires_dimension_visible_in_name": True,
            "global_category_attribute_rules_allowed": False,
        },
        "outputs": {},
        "elapsed_seconds": time.time() - started,
    }
    for label, path in (
        ("a2", a2_path),
        ("c", c_path),
        ("c_train_research_candidates", c_research_path),
        ("c_replacement_sources", c_replacements_path),
        ("c_rule_evidence", c_rule_evidence_path),
        ("c_family_evidence", evidence_path),
    ):
        manifest["outputs"][label] = {
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }

    manifest_path = args.output_dir / "source_pool_manifest.json"
    report_path = args.output_dir / "report.md"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_path.write_text(write_report(manifest, c_evidence), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
