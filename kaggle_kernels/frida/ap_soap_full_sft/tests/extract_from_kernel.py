"""Pulls exact source text for named top-level classes/constants out of the
real frida_ap_soap_full_sft.py kernel via `ast`, so anything built from it is
byte-for-byte identical to what actually gets pushed to Kaggle -- never a
hand-retyped, driftable rewrite.
"""
import ast
from pathlib import Path

KERNEL_PATH = Path(__file__).resolve().parent.parent / "frida_ap_soap_full_sft.py"


def extract(*names):
    """Returns {name: exact_source_text} for each top-level class or simple
    `NAME = ...` assignment in the kernel whose name is in `names`."""
    source = KERNEL_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(KERNEL_PATH))
    found = {}
    wanted = set(names)
    for node in tree.body:
        node_name = None
        if isinstance(node, ast.ClassDef):
            node_name = node.name
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            node_name = node.targets[0].id
        if node_name in wanted:
            found[node_name] = ast.get_source_segment(source, node)
    missing = wanted - found.keys()
    if missing:
        raise RuntimeError(f"could not find {sorted(missing)} as a top-level class/constant in {KERNEL_PATH}")
    return found
