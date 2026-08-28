"""Assemble one identity-safe A2 training set without rescoring batch 200.

The two earlier batches use their frozen-Frida-selected rows.  The final
200-source continuation uses one deterministically accepted, title-like
candidate per source, preferring the minimal placement variant.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SPLIT_SHA256 = (
    "1e3df234d6cdb8685256d5144b25f9a0a3e3a023f1053a0bd5b16c40e3df8d7e"
)
EXPECTED_VAL_FINGERPRINT = (
    "847d04d081432cc081c2149b75271547bf5e25b14f9d8a5804704a5be2a4b3c8"
)
META_PATTERN = re.compile(
    r"(?:<|>|\{|\}|```|fact[_ -]?id|operation[_ -]?id|товар\s+[abаб]\b|name2)",
    re.IGNORECASE,
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pair_fingerprint(frame: pd.DataFrame) -> str:
    payload = "\n".join(
        sorted(f"{int(left)}-{int(right)}" for left, right in zip(frame.id1, frame.id2))
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def naturalness_reasons(name: str, original: str) -> list[str]:
    value = str(name).strip()
    reasons: list[str] = []
    if len(value) < 8:
        reasons.append("too_short")
    if len(value) > 240:
        reasons.append("too_long")
    if "\n" in value or "\r" in value:
        reasons.append("multiline")
    if META_PATTERN.search(value):
        reasons.append("meta_or_template_marker")
    if value.casefold() == str(original).strip().casefold():
        reasons.append("unchanged")
    tokens = re.findall(r"[\w.-]+", value.casefold(), flags=re.UNICODE)
    content = [token for token in tokens if len(token) >= 3]
    if content and max(content.count(token) for token in set(content)) > 3:
        reasons.append("excessive_token_repetition")
    return reasons


def load_selected(path: Path, source_path: Path, origin: str) -> pd.DataFrame:
    selected = pd.read_parquet(path)
    source = pd.read_parquet(source_path)
    identity = source[
        ["source_pair_sha256", "id1", "id2", "category", "name1", "name2", "attributes2"]
    ]
    missing = [column for column in ("id1", "id2") if column not in selected]
    if missing:
        selected = selected.drop(
            columns=[column for column in identity.columns if column in selected.columns and column not in {"source_pair_sha256", "category"}]
        ).merge(identity, on=["source_pair_sha256", "category"], validate="one_to_one")
    selected["selection_origin"] = origin
    selected["naturalness_reasons_json"] = "[]"
    return selected


def select_batch_200() -> tuple[pd.DataFrame, dict[str, int]]:
    root = REPO_ROOT / "artifacts/synthetic_v2"
    candidates = pd.read_parquet(
        root / "a2_v31_batch_200_02/candidate_results.parquet"
    )
    source = pd.read_parquet(root / "a2_v31_batch_200_02_plan/source_pool.parquet")
    accepted = candidates[candidates["accepted_by_deterministic_validator"]].copy()
    accepted = accepted.merge(
        source[
            ["source_pair_sha256", "id1", "id2", "category", "name1", "name2", "attributes2"]
        ],
        on=["source_pair_sha256", "category"],
        validate="many_to_one",
    )
    accepted["naturalness_reasons"] = accepted.apply(
        lambda row: naturalness_reasons(row["name2_prime"], row["name2"]), axis=1
    )
    accepted["naturalness_reasons_json"] = accepted["naturalness_reasons"].map(
        lambda values: json.dumps(values, ensure_ascii=False)
    )
    natural = accepted[accepted["naturalness_reasons"].map(len).eq(0)].copy()
    natural["variant_priority"] = natural["variant_kind"].map(
        {"minimal": 0, "alternate_placement": 1}
    ).fillna(2)
    natural = natural.sort_values(
        ["source_pair_sha256", "variant_priority", "name_similarity_delta"],
        ascending=[True, True, False],
    )
    chosen = natural.groupby("source_pair_sha256", as_index=False, sort=False).first()
    chosen["selection_origin"] = "batch_200_deterministic_naturalness"
    stats = {
        "deterministic_accepted_candidates": int(len(accepted)),
        "sources_with_deterministic_candidate": int(accepted.source_pair_sha256.nunique()),
        "candidates_rejected_by_naturalness": int(len(accepted) - len(natural)),
        "selected_sources": int(len(chosen)),
    }
    return chosen, stats


def main() -> None:
    root = REPO_ROOT / "artifacts/synthetic_v2"
    split_path = REPO_ROOT / "data/splits/matches_item_disjoint_70_30_seed42.parquet"
    if file_sha256(split_path) != EXPECTED_SPLIT_SHA256:
        raise RuntimeError("Canonical split SHA-256 mismatch")
    split = pd.read_parquet(split_path, columns=["id1", "id2", "label", "split"])
    train = split[split.split.eq("train")]
    val = split[split.split.eq("val")]
    if pair_fingerprint(val) != EXPECTED_VAL_FINGERPRINT:
        raise RuntimeError("Validation fingerprint mismatch")

    pilot = load_selected(
        root / "a2_pilot_v31_r2_frida_scores/selected_synthetics.parquet",
        root / "a2_v31_pilot_plan/pilot_source_pool.parquet",
        "v31_pilot_frozen_frida",
    )
    batch_100 = load_selected(
        root / "a2_v31_batch_100_01_frida_scores/selected_synthetics.parquet",
        root / "a2_v31_batch_100_01_plan/source_pool.parquet",
        "batch_100_frozen_frida",
    )
    batch_200, batch_200_stats = select_batch_200()
    selected = pd.concat((pilot, batch_100, batch_200), ignore_index=True, sort=False)
    if selected.source_pair_sha256.duplicated().any():
        raise RuntimeError("Duplicate source pair in selected A2")
    original_ids = [int(value) for column in ("id1", "id2") for value in selected[column]]
    if len(original_ids) != len(set(original_ids)):
        raise RuntimeError("Original item reused across selected A2")

    train_ids = set(map(int, train.id1)) | set(map(int, train.id2))
    val_ids = set(map(int, val.id1)) | set(map(int, val.id2))
    if set(original_ids) - train_ids:
        raise RuntimeError("Selected A2 contains a non-train source item")
    if set(original_ids) & val_ids:
        raise RuntimeError("Selected A2 overlaps validation items")

    selected = selected.sort_values("source_pair_sha256").reset_index(drop=True)
    selected["synthetic_id2"] = -(selected.index + 1).astype("int64")
    selected["label"] = 0
    selected["sample_weight"] = 0.5
    selected["recommended_repeat"] = 10

    output_dir = root / "a2_training_set_v1"
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_path = output_dir / "selected_a2.parquet"
    selected.to_parquet(selected_path, index=False)
    pairs = selected[
        ["id1", "synthetic_id2", "label", "category", "source_pair_sha256", "sample_weight", "recommended_repeat"]
    ].rename(columns={"synthetic_id2": "id2"})
    pairs["branch"] = "A2"
    pairs_path = output_dir / "synthetic_pairs.parquet"
    pairs.to_parquet(pairs_path, index=False)
    items = selected[
        ["synthetic_id2", "name2_prime", "attributes2", "category", "source_pair_sha256"]
    ].rename(
        columns={"synthetic_id2": "id", "name2_prime": "name", "attributes2": "attributes"}
    )
    items_path = output_dir / "synthetic_items.parquet"
    items.to_parquet(items_path, index=False)

    manifest = {
        "status": "complete",
        "profile": "a2_training_set_v1",
        "canonical_split_sha256": EXPECTED_SPLIT_SHA256,
        "validation_fingerprint": EXPECTED_VAL_FINGERPRINT,
        "validation_used_as_source": False,
        "selected_rows": int(len(selected)),
        "selected_by_origin": {
            str(key): int(value)
            for key, value in selected.selection_origin.value_counts().sort_index().items()
        },
        "selected_by_category": {
            str(key): int(value)
            for key, value in selected.category.value_counts().sort_index().items()
        },
        "batch_200_selection": batch_200_stats,
        "source_duplicates": int(selected.source_pair_sha256.duplicated().sum()),
        "original_item_reuse": len(original_ids) - len(set(original_ids)),
        "validation_item_overlap": len(set(original_ids) & val_ids),
        "label": 0,
        "sample_weight": 0.5,
        "recommended_repeat": 10,
        "effective_a2_weight_per_row": 5.0,
        "selected_a2_sha256": file_sha256(selected_path),
        "synthetic_pairs_sha256": file_sha256(pairs_path),
        "synthetic_items_sha256": file_sha256(items_path),
    }
    manifest_path = output_dir / "training_set_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
