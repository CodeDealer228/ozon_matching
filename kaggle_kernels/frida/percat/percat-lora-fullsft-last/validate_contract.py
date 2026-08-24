"""Cheap local pre-push contract check; no ML dependencies required."""
from pathlib import Path
import ast
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[3]
SOURCE = ROOT / "frida_percat_fullsft_last.py"
META = ROOT / "kernel-metadata.json"
ARCHIVE = (PROJECT / "submits/frida/lora/format_adapter_v4/"
           "matching-frida_format_adapter_v4_best_200-submit.zip")
EXPECTED_SHA = "b3736c61b7a3d16d77f6708cb5f5e4db7cc1c97b849433d7d759bb62406b5ea2"

text = SOURCE.read_text(encoding="utf-8")
tree = ast.parse(text)
meta = json.loads(META.read_text(encoding="utf-8"))
assert meta["enable_tpu"] and not meta["enable_gpu"]
assert meta["id"].startswith("kirpich682/") and meta["is_private"] is False
assert "kirpich682/frida-format-adapter-v4-merged" in meta["dataset_sources"]
assert "maxamsecret/frida-item-disjoint-split" in meta["dataset_sources"]
assert meta["kernel_sources"] == ["maxamsecret/frida-tokenize-items-v4"]
assert "PAIR_MAX_LENGTH = 224" in text
assert "PER_CORE_TRAIN_BATCH = 16" in text
assert "PER_CORE_EVAL_BATCH = 32" in text
assert "PREFLIGHT_STEPS = 48" in text
assert "PREFLIGHT_HEARTBEAT = 8" in text
assert "num_workers=0" in text and "drop_last=True" in text
assert "xmp.MpModelWrapper" in text and 'start_method="fork"' in text
assert "xm.optimizer_step(optimizer)" in text
assert not any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
               and isinstance(node.func.value, ast.Name)
               and node.func.value.id == "optimizer" and node.func.attr == "step"
               for node in ast.walk(tree))
assert "torch.stack(flags).amin(dim=0)" in text
safe_update = next(node for node in tree.body
                   if isinstance(node, ast.FunctionDef)
                   and node.name == "_optimizer_update")
assert not any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
               and node.func.attr == "item" for node in ast.walk(safe_update))
assert not any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
               and node.func.attr == "all_reduce" for node in ast.walk(safe_update))
assert "item-disjoint 70/30 seed42" in text
assert 'meta.get("format_version") != 4' in text
assert 'meta.get("prefix") != ""' in text
assert 'meta.get("category_in_text") is not False' in text
assert 'if kind == "last":' in text
assert '"optimizer.pt"' in text and '"scheduler.pt"' in text
assert "_to_cpu_state(optimizer.state_dict())" in text
assert "_to_cpu_state(scheduler.state_dict())" in text
assert "MAX_WALL_SECONDS = 29_700" in text
assert "LAST_SAVE_SECONDS = 900" in text
assert '"percat_results.json"' in text
digest = hashlib.sha256()
with ARCHIVE.open("rb") as handle:
    for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
        digest.update(chunk)
assert digest.hexdigest() == EXPECTED_SHA
with zipfile.ZipFile(ARCHIVE) as archive:
    names = set(archive.namelist())
    expected = {
        "classifier.pt": (4_340_725, "0fd9134e8cd4a9b945aee7fb3d12d991208c9ff93d293e958c0ec3b35eb94612"),
        "models/frida-format-adapter-v4/config.json": (846, "f42bc8b9f9706487667056e8bb5c2151a995fbf7d056c2461936d147e24e3651"),
        "models/frida-format-adapter-v4/model.safetensors": (1_646_829_336, "3978a7b9c0641ab75248257adc0abe2b2c0007a98202b2ce22040bf40b42b3d1"),
    }
    assert set(expected) <= names
    for name, (size, expected_digest) in expected.items():
        member_digest = hashlib.sha256()
        with archive.open(name) as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                member_digest.update(chunk)
        assert archive.getinfo(name).file_size == size
        assert member_digest.hexdigest() == expected_digest
print("contract OK")
