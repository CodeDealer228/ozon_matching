"""CPU builder: merge the validated FRIDA v4 LoRA into its exact c-last base.

The resulting archive is self-contained for the offline H100 stand: after the
one-time CPU merge it needs no PEFT/wheels at inference.  `utils2.py` is the
new v4 serving contract and is copied verbatim as src/utils.py.
"""
import json
import os
import shutil
import subprocess
import sys
import zipfile

# Kaggle CPU image ships torchao 0.10.0, while current PEFT refuses to use
# any torchao <0.16 even for ordinary nn.Linear LoRA.  This builder neither
# quantizes nor needs torchao, so remove the incompatible optional backend.
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"],
               check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--upgrade", "peft", "safetensors"])

import torch
from peft import PeftModel
from transformers import AutoTokenizer, T5Config, T5EncoderModel

MODEL_NAME = "ai-forever/FRIDA"
BASE_EXPERIMENT = "continue_c_llm3m_2ep"
BASE_LAST_STEP = 13_474
ADAPTER_EXPERIMENT = "frida_format_adapter_v4_c_last"
WORKING = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
STAGE = os.path.join(WORKING, "_frida_format_adapter_v4_submit")
ZIP_PATH = os.path.join(WORKING, "matching-frida_format_adapter_v4_best_188-submit.zip")


def log(message):
    print(f"[BUILD] {message}", flush=True)


def find_base():
    found = []
    for root, _dirs, files in os.walk("/kaggle/input"):
        if {"encoder.pt", "head.pt", "train_state.json"} <= set(files):
            with open(os.path.join(root, "train_state.json"), encoding="utf-8") as fh:
                state = json.load(fh)
            if (state.get("experiment") == BASE_EXPERIMENT
                    and int(state.get("global_step", -1)) == BASE_LAST_STEP
                    and os.path.basename(root) == "last"):
                found.append(root)
    if len(found) != 1:
        raise RuntimeError(f"Expected one c-last step={BASE_LAST_STEP}, got {found}")
    return found[0]


def find_best_adapter():
    found = []
    for root, _dirs, files in os.walk("/kaggle/input"):
        adapter = os.path.join(root, "adapter")
        if (os.path.basename(root) == "best" and "train_state.json" in files
                and os.path.isfile(os.path.join(adapter, "adapter_model.safetensors"))):
            with open(os.path.join(root, "train_state.json"), encoding="utf-8") as fh:
                state = json.load(fh)
            if state.get("experiment") == ADAPTER_EXPERIMENT:
                found.append((root, state))
    if len(found) != 1:
        raise RuntimeError(f"Expected one validated v4 best adapter, got {[p for p, _ in found]}")
    return found[0]


def bf16_state(path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    return {name: value.to(torch.bfloat16) if torch.is_tensor(value) else value
            for name, value in state.items()}


def write_text(path, text):
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)


def main():
    if os.path.exists(STAGE):
        shutil.rmtree(STAGE)
    if os.path.exists(ZIP_PATH):
        os.remove(ZIP_PATH)

    base_dir = find_base()
    adapter_root, adapter_state = find_best_adapter()
    log(f"base={base_dir}")
    log(f"adapter={adapter_root}; val={adapter_state.get('val_macro_pr_auc')}")

    model = T5EncoderModel(T5Config.from_pretrained(MODEL_NAME))
    incompatible = model.load_state_dict(torch.load(os.path.join(base_dir, "encoder.pt"),
                                                     map_location="cpu", weights_only=True))
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"base state mismatch: {incompatible}")
    merged = PeftModel.from_pretrained(model, os.path.join(adapter_root, "adapter")).merge_and_unload()
    merged = merged.to(dtype=torch.bfloat16)

    model_dir = os.path.join(STAGE, "models", "frida-format-adapter-v4")
    os.makedirs(model_dir, exist_ok=True)
    merged.save_pretrained(model_dir, safe_serialization=True)
    AutoTokenizer.from_pretrained(MODEL_NAME).save_pretrained(model_dir)
    torch.save(bf16_state(os.path.join(base_dir, "head.pt")), os.path.join(STAGE, "classifier.pt"))

    os.makedirs(os.path.join(STAGE, "src"), exist_ok=True)
    shutil.copy2(os.path.join(os.path.dirname(__file__), "utils2.py"),
                 os.path.join(STAGE, "src", "utils.py"))
    write_text(os.path.join(STAGE, "run.py"), '''import argparse
import os
from src.utils import predict_pipeline

ROOT = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument("--output_path", required=True)
parser.add_argument("--items_path", required=True)
parser.add_argument("--matches_path", required=True)
args = parser.parse_args()
predict_pipeline(args.items_path, args.matches_path,
                 os.path.join(ROOT, "models", "frida-format-adapter-v4"),
                 os.path.join(ROOT, "classifier.pt"), args.output_path)
''')
    write_text(os.path.join(STAGE, "metadata.json"),
               json.dumps({"image": "odsai/ecup26-matching-baseline:1.0", "entry_point": "python -u run.py"}))

    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
        for root, _dirs, files in os.walk(STAGE):
            for name in files:
                path = os.path.join(root, name)
                archive.write(path, os.path.relpath(path, STAGE))
    shutil.rmtree(STAGE)
    log(f"wrote only {ZIP_PATH} ({os.path.getsize(ZIP_PATH) / 2**30:.2f} GiB)")


if __name__ == "__main__":
    main()
