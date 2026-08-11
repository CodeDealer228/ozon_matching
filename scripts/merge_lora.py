"""
Merge the trained LoRA adapter into the base cross-encoder so the final
submission runs plain transformers.AutoModel with no `peft` dependency
(faster load, one less runtime requirement -- see CLAUDE.md Architecture).

Requires local internet access to pull the base model from the HF hub --
this is a build-time step, not part of the offline inference container.

Run once locally:
    python scripts/merge_lora.py

Reads adapter files from artifacts/lora_adapter/ (downloaded from the Kaggle
dataset maxamsecret/ozon-lora-minilm-finetuned, best/ checkpoint) and writes
the merged encoder to artifacts/lora_merged/.
"""

import os
import shutil

import torch
from transformers import AutoModel, AutoTokenizer

try:
    import peft.import_utils
    import peft.tuners.lora.torchao

    peft.import_utils.is_torchao_available = lambda: False
    peft.tuners.lora.torchao.is_torchao_available = lambda: False
except Exception:
    pass

from peft import PeftModel

BASE_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-12-v2"
ADAPTER_DIR = os.path.join("artifacts", "lora_adapter")
OUT_DIR = os.path.join("artifacts", "lora_merged")


def main():
    print(f"[MERGE] loading base model {BASE_MODEL_NAME} ...")
    base_model = AutoModel.from_pretrained(BASE_MODEL_NAME)

    print(f"[MERGE] loading tokenizer from {ADAPTER_DIR} (the one the LoRA run trained/saved with) ...")
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)

    print(f"[MERGE] wrapping base model with adapter from {ADAPTER_DIR} ...")
    peft_model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)

    print("[MERGE] merge_and_unload() ...")
    merged = peft_model.merge_and_unload()

    os.makedirs(OUT_DIR, exist_ok=True)
    try:
        merged.save_pretrained(OUT_DIR, safe_serialization=True)
    except Exception as e:
        # merge_and_unload() occasionally leaves tensors sharing memory,
        # which safetensors rejects -- fall back to the plain pickle format.
        print(f"[MERGE] safetensors save failed ({e}); retrying with safe_serialization=False")
        merged.save_pretrained(OUT_DIR, safe_serialization=False)
    tokenizer.save_pretrained(OUT_DIR)

    classifier_src = os.path.join(ADAPTER_DIR, "classifier.pt")
    classifier_dst = os.path.join(OUT_DIR, "classifier.pt")
    shutil.copyfile(classifier_src, classifier_dst)

    print(f"[MERGE] saved merged encoder + tokenizer + classifier.pt to {OUT_DIR}")
    for f in sorted(os.listdir(OUT_DIR)):
        print(f"  {f} ({os.path.getsize(os.path.join(OUT_DIR, f)):,} bytes)")


if __name__ == "__main__":
    main()
