"""Sanitized public report for the verified FRIDA A2 head-only ablation.

The expensive experiment ran in private kernel version 2 because it consumes a
private human-train/A2 dataset. This public CPU kernel exposes the exact
verified validation metrics and provenance without publishing dataset rows,
item IDs, labels, predictions, or model weights.
"""

from __future__ import annotations

import json
from pathlib import Path


OUTPUT_DIR = Path("/kaggle/working")
SOURCE_KERNEL = "https://www.kaggle.com/code/rvitvitskii/frida-a2-head-ablation"
TRAINING_CODE = (
    "https://github.com/CodeDealer228/ozon_matching/blob/b79ba0f/"
    "kaggle_kernels/frida-a2-head-ablation/frida_a2_head_ablation.py"
)
SPLIT_SHA256 = "1e3df234d6cdb8685256d5144b25f9a0a3e3a023f1053a0bd5b16c40e3df8d7e"
VALIDATION_FINGERPRINT = (
    "847d04d081432cc081c2149b75271547bf5e25b14f9d8a5804704a5be2a4b3c8"
)

AGGREGATE = {
    "reference_baseline": {
        "macro_average_precision": 0.7588808815558529,
        "micro_average_precision": 0.8132819127479105,
    },
    "same_gpu_baseline": {
        "macro_average_precision": 0.7567292364833378,
        "micro_average_precision": 0.8111128907763334,
    },
    "control": {
        "macro_average_precision": 0.7555364111139332,
        "micro_average_precision": 0.8103925829146043,
    },
    "a2": {
        "macro_average_precision": 0.7555588967052053,
        "micro_average_precision": 0.81042040279828,
    },
}

TARGET_CATEGORIES = {
    "Автотовары": {"control": 0.6858223742417431, "a2": 0.6858132338144906},
    "Обувь": {"control": 0.4667728597969624, "a2": 0.46676492586943275},
    "Одежда": {"control": 0.5203515618370615, "a2": 0.5203977219636198},
    "Ювелирные изделия": {
        "control": 0.5083564321528325,
        "a2": 0.5083643962910117,
    },
}


def metric_delta(left: str, right: str, metric: str) -> float:
    return AGGREGATE[left][metric] - AGGREGATE[right][metric]


def build_report() -> dict[str, object]:
    a2_minus_control_macro = metric_delta("a2", "control", "macro_average_precision")
    a2_minus_control_micro = metric_delta("a2", "control", "micro_average_precision")
    a2_minus_same_gpu_macro = metric_delta(
        "a2", "same_gpu_baseline", "macro_average_precision"
    )
    a2_minus_same_gpu_micro = metric_delta(
        "a2", "same_gpu_baseline", "micro_average_precision"
    )
    assert abs(a2_minus_control_macro - 2.2485591272047856e-05) < 1e-15
    assert abs(a2_minus_control_micro - 2.7819883675728185e-05) < 1e-15
    assert abs(a2_minus_same_gpu_macro - (-0.001170339778132523)) < 1e-15
    assert abs(a2_minus_same_gpu_micro - (-0.0006924879780534354)) < 1e-15

    categories = {
        category: {
            **metrics,
            "a2_minus_control": metrics["a2"] - metrics["control"],
        }
        for category, metrics in TARGET_CATEGORIES.items()
    }
    return {
        "status": "complete",
        "report_kind": "sanitized_public_mirror_of_verified_private_run",
        "source_kernel": SOURCE_KERNEL,
        "source_kernel_version": 2,
        "training_code": TRAINING_CODE,
        "experiment": {
            "mode": "frozen_encoder_head_only_control_vs_a2",
            "canonical_human_train_replay_rows": 10000,
            "unique_a2_rows": 195,
            "a2_repeat": 10,
            "a2_sample_weight": 0.5,
            "validation_rows": 109648,
            "validation_used_for_training": False,
        },
        "identity": {
            "canonical_split_sha256": SPLIT_SHA256,
            "validation_fingerprint": VALIDATION_FINGERPRINT,
        },
        "aggregate": AGGREGATE,
        "comparison": {
            "a2_minus_control_macro": a2_minus_control_macro,
            "a2_minus_control_micro": a2_minus_control_micro,
            "a2_minus_same_gpu_baseline_macro": a2_minus_same_gpu_macro,
            "a2_minus_same_gpu_baseline_micro": a2_minus_same_gpu_micro,
            "a2_improved_over_control": a2_minus_control_macro > 0,
            "practically_meaningful_improvement_confirmed": False,
        },
        "target_categories": categories,
        "privacy": {
            "raw_dataset_rows_published": False,
            "item_ids_published": False,
            "validation_predictions_published": False,
            "model_weights_published": False,
        },
    }


def markdown(report: dict[str, object]) -> str:
    aggregate = report["aggregate"]
    comparison = report["comparison"]
    lines = [
        "# FRIDA A2 head-only ablation: public validation results",
        "",
        "This is a sanitized public mirror of the verified private Kaggle run (version 2).",
        "The private run is required because its A2/human-train input is non-public.",
        "",
        "## Aggregate validation metrics",
        "",
        "| Variant | Macro AP | Micro AP |",
        "|---|---:|---:|",
    ]
    labels = {
        "reference_baseline": "Reference baseline",
        "same_gpu_baseline": "Same-GPU baseline",
        "control": "Control fine-tuning",
        "a2": "A2 fine-tuning",
    }
    for key in ("reference_baseline", "same_gpu_baseline", "control", "a2"):
        values = aggregate[key]
        lines.append(
            f"| {labels[key]} | {values['macro_average_precision']:.9f} | "
            f"{values['micro_average_precision']:.9f} |"
        )
    lines.extend(
        [
            "",
            "## Main comparison",
            "",
            f"- A2 - control Macro AP: `{comparison['a2_minus_control_macro']:+.9f}`",
            f"- A2 - control Micro AP: `{comparison['a2_minus_control_micro']:+.9f}`",
            f"- A2 - same-GPU baseline Macro AP: "
            f"`{comparison['a2_minus_same_gpu_baseline_macro']:+.9f}`",
            f"- A2 - same-GPU baseline Micro AP: "
            f"`{comparison['a2_minus_same_gpu_baseline_micro']:+.9f}`",
            "- Conclusion: A2 is microscopically above control, but remains below the "
            "same-GPU baseline. A practically meaningful improvement is not confirmed.",
            "",
            "## A2 target categories",
            "",
            "| Category | Control AP | A2 AP | Delta |",
            "|---|---:|---:|---:|",
        ]
    )
    for category, values in report["target_categories"].items():
        lines.append(
            f"| {category} | {values['control']:.9f} | {values['a2']:.9f} | "
            f"{values['a2_minus_control']:+.9f} |"
        )
    lines.extend(
        [
            "",
            "## Provenance and privacy",
            "",
            f"- Source run: [private Kaggle kernel version 2]({SOURCE_KERNEL})",
            f"- Training code: [GitHub commit b79ba0f]({TRAINING_CODE})",
            f"- Validation rows: `{report['experiment']['validation_rows']}`",
            f"- Split SHA-256: `{SPLIT_SHA256}`",
            f"- Validation fingerprint: `{VALIDATION_FINGERPRINT}`",
            "- No raw rows, item IDs, predictions, or weights are published here.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report = build_report()
    report_path = OUTPUT_DIR / "public_validation_results.json"
    markdown_path = OUTPUT_DIR / "PUBLIC_RESULTS.md"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    text = markdown(report)
    markdown_path.write_text(text, encoding="utf-8")
    print(text, flush=True)
    print(json.dumps(report["comparison"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
