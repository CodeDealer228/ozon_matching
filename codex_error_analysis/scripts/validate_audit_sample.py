from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from common import ROOT, load_config, read_jsonl, sha256_file


def main():
    cfg = load_config()
    p = ROOT / "artifacts" / "prepared" / "audit_sample.jsonl"
    manifest_p = ROOT / "artifacts" / "prepared" / "manifest.json"

    if not p.exists() or not manifest_p.exists():
        raise SystemExit("Run scripts/prepare_audit.py first.")

    rows = read_jsonl(p)
    manifest = json.loads(manifest_p.read_text(encoding="utf-8"))
    errors = []

    ids = [r.get("audit_unit_id") for r in rows]
    if len(ids) != len(set(ids)):
        errors.append("audit_unit_id is not unique")

    required_text = [
        "item1_name_full", "item2_name_full",
        "item1_attributes_full", "item2_attributes_full",
        "focal_item1_visible_input", "focal_item2_visible_input",
    ]

    bands = {b["name"]: b for b in cfg["severity_bands"]}

    for i, r in enumerate(rows, start=1):
        prefix = f"line {i} / {r.get('audit_unit_id')}"
        if r.get("focal_model") not in {"berta", "frida"}:
            errors.append(f"{prefix}: invalid focal_model")
        if r.get("error_type") not in {"FP", "FN"}:
            errors.append(f"{prefix}: invalid error_type")
        if r.get("error_type") == "FP" and int(r.get("target", -1)) != 0:
            errors.append(f"{prefix}: FP must have target=0")
        if r.get("error_type") == "FN" and int(r.get("target", -1)) != 1:
            errors.append(f"{prefix}: FN must have target=1")

        band_name = r.get("severity_band")
        if band_name not in bands:
            errors.append(f"{prefix}: unknown severity_band={band_name}")
        else:
            rank = int(r["severity_rank"])
            b = bands[band_name]
            if not (int(b["start_rank"]) <= rank <= int(b["end_rank"])):
                errors.append(f"{prefix}: rank {rank} outside band {band_name}")

        for c in required_text:
            if c not in r or r[c] is None:
                errors.append(f"{prefix}: missing {c}")

    if manifest.get("audit_sample_jsonl_sha256") != sha256_file(p):
        errors.append("audit_sample.jsonl hash differs from manifest")

    if errors:
        print("\n".join(errors[:100]))
        raise SystemExit(f"Sample validation FAILED with {len(errors)} issue(s).")

    print(f"Sample validation PASSED: {len(rows)} audit units.")


if __name__ == "__main__":
    main()
