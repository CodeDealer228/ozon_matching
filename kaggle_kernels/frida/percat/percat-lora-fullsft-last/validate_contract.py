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
ARCHIVE = (PROJECT / "submits/frida/sft_continue/continue_c_llm3m/"
           "matching-frida_continue_c_last_188-submit.zip")
EXPECTED_SHA = "6622022adf93bd159ebd9be2c938feed6cf5e606ce56f0e99c598a02dff4cb73"

text = SOURCE.read_text(encoding="utf-8")
ast.parse(text)
meta = json.loads(META.read_text(encoding="utf-8"))
assert meta["enable_tpu"] and not meta["enable_gpu"]
assert "maxamsecret/frida-tokenize-items-v3" in meta["dataset_sources"]
assert all("tokenize-items-v4" not in item for item in meta["dataset_sources"])
assert any(s.startswith("maxamsecret/frida-full-sft-continue-submit-build") for s in meta["kernel_sources"])
assert "PAIR_MAX_LENGTH = 188" in text
assert "PER_CORE_TRAIN_BATCH = 16" in text
assert "PER_CORE_EVAL_BATCH = 32" in text
assert "num_workers=0" in text and "drop_last=True" in text
assert "xmp.MpModelWrapper" in text and 'start_method="fork"' in text
assert "xm.reduce_gradients(optimizer)" in text
assert "torch.stack(flags).amin(dim=0)" in text
assert 'if kind == "last":' in text
assert '"optimizer.pt"' in text and '"scheduler.pt"' in text
digest = hashlib.sha256()
with ARCHIVE.open("rb") as handle:
    for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
        digest.update(chunk)
assert digest.hexdigest() == EXPECTED_SHA
with zipfile.ZipFile(ARCHIVE) as archive:
    names = set(archive.namelist())
    assert {"classifier.pt", "models/frida-full-sft/config.json",
            "models/frida-full-sft/model.safetensors"} <= names
print("contract OK")
