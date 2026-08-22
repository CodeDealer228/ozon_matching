from __future__ import annotations

from pathlib import Path
import hashlib
import json
import math
import re
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def load_config() -> dict:
    return json.loads((ROOT / "config" / "audit_config.json").read_text(encoding="utf-8"))


def find_source_artifact() -> Path:
    candidates = [
        ROOT / "artifacts" / "input" / "all_fp_fn_full_attributes_and_exact_inputs.parquet",
        ROOT / "artifacts" / "input" / "all_fp_fn_full_attributes_and_exact_inputs.jsonl",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Missing source artifact. Put all_fp_fn_full_attributes_and_exact_inputs.parquet "
        "or .jsonl into artifacts/input/."
    )


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        try:
            return pd.read_parquet(path)
        except ImportError as e:
            raise RuntimeError(
                "Reading parquet requires pyarrow/fastparquet. Run "
                "`python -m pip install -r requirements.txt`, or provide the JSONL "
                "source artifact instead."
            ) from e
    if path.suffix.lower() in {".jsonl", ".json"}:
        return pd.read_json(path, lines=True)
    raise ValueError(f"Unsupported input format: {path}")


def write_jsonl(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in df.to_dict(orient="records"):
            f.write(json.dumps(rec, ensure_ascii=False, default=_json_default) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except Exception as e:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {e}") from e
    return rows


def _json_default(x: Any):
    if pd.isna(x):
        return None
    if hasattr(x, "item"):
        return x.item()
    return str(x)


def stable_seed(*parts: str, base_seed: int = 0) -> int:
    s = "||".join(map(str, parts)) + f"||{base_seed}"
    h = hashlib.sha256(s.encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def sha256_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def canonical_text(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and math.isnan(x):
        return ""
    return str(x)


def contains_exactish(haystack: str, needle: str) -> bool:
    """Evidence validator: exact substring after only whitespace normalization."""
    haystack = re.sub(r"\s+", " ", canonical_text(haystack)).strip()
    needle = re.sub(r"\s+", " ", canonical_text(needle)).strip()
    return bool(needle) and needle in haystack


def wilson_interval(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n <= 0:
        return (float("nan"), float("nan"))
    phat = k / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = z * math.sqrt((phat * (1 - phat) / n) + z * z / (4 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def markdown_table(df: pd.DataFrame, max_rows: int = 30) -> str:
    x = df.head(max_rows).copy()
    if len(x) == 0:
        return "_No rows._"
    cols = [str(c) for c in x.columns]
    def fmt(v):
        if pd.isna(v):
            return ""
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v).replace("|", "\\|").replace("\n", " ")
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, r in x.iterrows():
        lines.append("| " + " | ".join(fmt(r[c]) for c in x.columns) + " |")
    return "\n".join(lines)
