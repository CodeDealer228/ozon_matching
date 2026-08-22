from __future__ import annotations

import json
from pathlib import Path

from common import ROOT, read_jsonl, contains_exactish


PRIMARY = {
    "likely_label_issue", "ambiguous_business_rule", "truncation_related",
    "critical_attribute_mismatch", "product_type_confusion",
    "semantic_or_alias_failure", "variant_policy_failure",
    "missing_information", "other_model_error"
}
LABEL = {"likely_correct", "likely_wrong", "ambiguous"}
FAILURE = {"real_failure", "not_model_failure", "unclear"}
SAW = {"yes", "no", "unclear", "not_applicable"}
TRUNC = {"causal", "possible", "not_relevant", "unclear"}


def evidence_haystacks(inp: dict):
    original = "\n".join([
        str(inp.get("item1_name_full", "")),
        str(inp.get("item2_name_full", "")),
        str(inp.get("item1_attributes_full", "")),
        str(inp.get("item2_attributes_full", "")),
    ])
    focal = "\n".join([
        str(inp.get("focal_item1_visible_input", "")),
        str(inp.get("focal_item2_visible_input", "")),
    ])
    return original, focal


def validate_one(inp: dict, out: dict) -> list[str]:
    e = []
    uid = inp["audit_unit_id"]

    required = [
        "audit_unit_id", "primary_class", "secondary_subtype",
        "label_assessment", "model_failure_assessment",
        "model_saw_critical_difference", "truncation_relevance",
        "critical_difference", "reason_ru", "confidence",
        "evidence_original", "evidence_model_input", "notes_ru"
    ]
    for k in required:
        if k not in out:
            e.append(f"{uid}: missing field {k}")

    if e:
        return e

    if out["audit_unit_id"] != uid:
        e.append(f"{uid}: output audit_unit_id mismatch")
    if out["primary_class"] not in PRIMARY:
        e.append(f"{uid}: invalid primary_class={out['primary_class']}")
    if out["label_assessment"] not in LABEL:
        e.append(f"{uid}: invalid label_assessment")
    if out["model_failure_assessment"] not in FAILURE:
        e.append(f"{uid}: invalid model_failure_assessment")
    if out["model_saw_critical_difference"] not in SAW:
        e.append(f"{uid}: invalid model_saw_critical_difference")
    if out["truncation_relevance"] not in TRUNC:
        e.append(f"{uid}: invalid truncation_relevance")

    try:
        conf = float(out["confidence"])
        if not 0 <= conf <= 1:
            e.append(f"{uid}: confidence outside [0,1]")
    except Exception:
        e.append(f"{uid}: confidence is not numeric")

    if not isinstance(out["reason_ru"], str) or len(out["reason_ru"].strip()) < 20:
        e.append(f"{uid}: reason_ru too short")

    # Decision consistency.
    pc = out["primary_class"]
    la = out["label_assessment"]
    mf = out["model_failure_assessment"]

    if pc == "likely_label_issue" and la != "likely_wrong":
        e.append(f"{uid}: likely_label_issue requires label_assessment=likely_wrong")
    if pc == "ambiguous_business_rule" and la != "ambiguous":
        e.append(f"{uid}: ambiguous_business_rule requires label_assessment=ambiguous")
    if pc in {
        "truncation_related", "critical_attribute_mismatch", "product_type_confusion",
        "semantic_or_alias_failure", "variant_policy_failure", "other_model_error"
    }:
        if la != "likely_correct":
            e.append(f"{uid}: real model-error class requires label_assessment=likely_correct")
        if mf != "real_failure":
            e.append(f"{uid}: real model-error class requires model_failure_assessment=real_failure")

    if pc == "truncation_related" and out["truncation_relevance"] not in {"causal", "possible"}:
        e.append(f"{uid}: truncation_related requires causal/possible truncation_relevance")

    # Evidence must be grounded in the supplied record.
    original_h, focal_h = evidence_haystacks(inp)

    eo = out["evidence_original"]
    em = out["evidence_model_input"]
    if not isinstance(eo, list) or not (1 <= len(eo) <= 6):
        e.append(f"{uid}: evidence_original must be a list of 1..6 strings")
    else:
        for s in eo:
            if not isinstance(s, str) or not contains_exactish(original_h, s):
                e.append(f"{uid}: original evidence is not an exact supplied substring: {s!r}")

    if not isinstance(em, list) or len(em) > 6:
        e.append(f"{uid}: evidence_model_input must be a list of <=6 strings")
    else:
        for s in em:
            if not isinstance(s, str) or not contains_exactish(focal_h, s):
                e.append(f"{uid}: model-input evidence is not an exact supplied substring: {s!r}")

    return e


def main():
    batch_dirs = sorted(p for p in (ROOT / "audit_batches").glob("batch_*") if p.is_dir())
    if not batch_dirs:
        raise SystemExit("No audit batches found. Run make_batches.py.")

    all_errors = []
    total = 0

    for d in batch_dirs:
        inp_p = d / "input.jsonl"
        out_p = d / "output.jsonl"
        if not out_p.exists():
            all_errors.append(f"{d.name}: missing output.jsonl")
            continue

        inputs = read_jsonl(inp_p)
        outputs = read_jsonl(out_p)

        inp_by = {r["audit_unit_id"]: r for r in inputs}
        out_by = {}
        for r in outputs:
            uid = r.get("audit_unit_id")
            if uid in out_by:
                all_errors.append(f"{d.name}: duplicate output audit_unit_id={uid}")
            out_by[uid] = r

        missing = set(inp_by) - set(out_by)
        extra = set(out_by) - set(inp_by)
        if missing:
            all_errors.append(f"{d.name}: missing outputs: {sorted(missing)[:10]}")
        if extra:
            all_errors.append(f"{d.name}: extra outputs: {sorted(extra)[:10]}")

        for uid in set(inp_by) & set(out_by):
            all_errors.extend(validate_one(inp_by[uid], out_by[uid]))
            total += 1

    if all_errors:
        print("\n".join(all_errors[:200]))
        raise SystemExit(f"Batch validation FAILED with {len(all_errors)} issue(s).")

    print(f"Batch validation PASSED: {total} classifications.")


if __name__ == "__main__":
    main()
