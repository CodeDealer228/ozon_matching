"""
Merge the trained LoRA adapter (maxamsecret/ozon-lora-minilm-finetuned,
best/ checkpoint) into the base cross-encoder so the final submission runs
plain transformers.AutoModel with no `peft` dependency at inference.

Runs on Kaggle (internet enabled, no GPU needed -- this is just weight
arithmetic on a 33M param model). Output goes to /kaggle/working/lora_merged/
and gets downloaded + published as a Kaggle dataset from the local machine
after this kernel completes (see CLAUDE.md: all GPU work + dataset publishing
conventions).
"""

import os
import shutil
import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "peft"])

import torch
from transformers import AutoModel, AutoTokenizer

try:
    import peft.import_utils
    import peft.tuners.lora.torchao

    peft.import_utils.is_torchao_available = lambda: False
    peft.tuners.lora.torchao.is_torchao_available = lambda: False
    print("[INFO] torchao disabled for PEFT")
except Exception as e:
    print(f"[WARN] torchao patch failed (likely fine): {e}")

from peft import PeftModel

BASE_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-12-v2"
ADAPTER_DIR = "/kaggle/input/datasets/maxamsecret/ozon-lora-minilm-finetuned/lora_mini_self_attention_memmap/best"
OUT_DIR = "/kaggle/working/lora_merged"

for _root, _dirs, _files in os.walk("/kaggle/input"):
    depth = _root[len("/kaggle/input"):].count(os.sep)
    if depth <= 4:
        print(f"[DEBUG] {_root} -> dirs={_dirs} files={_files}")

print(f"[MERGE] loading base model {BASE_MODEL_NAME} ...")
base_model = AutoModel.from_pretrained(BASE_MODEL_NAME)

print(f"[MERGE] loading tokenizer from {ADAPTER_DIR} ...")
tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)

print(f"[MERGE] wrapping base model with adapter from {ADAPTER_DIR} ...")
peft_model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
peft_model.eval()

# Sanity check BEFORE merging: merge_and_unload() must not change behavior.
# Run a few dummy batches through the PEFT-wrapped model, merge, run the same
# batches through the merged model, and assert the outputs match closely.
print("[CHECK] capturing PEFT-wrapped outputs on dummy batches ...")
torch.manual_seed(0)
dummy_batches = [
    {
        "input_ids": torch.randint(0, 1000, (4, 128), dtype=torch.long),
        "attention_mask": torch.ones((4, 128), dtype=torch.long),
        "token_type_ids": torch.zeros((4, 128), dtype=torch.long),
    }
    for _ in range(3)
]

with torch.no_grad():
    pre_merge_outputs = [
        peft_model(**batch).last_hidden_state.clone() for batch in dummy_batches
    ]

print("[MERGE] merge_and_unload() ...")
merged = peft_model.merge_and_unload()
merged.eval()

print("[CHECK] comparing merged model outputs on the same dummy batches ...")
with torch.no_grad():
    post_merge_outputs = [
        merged(**batch).last_hidden_state.clone() for batch in dummy_batches
    ]

max_diff = max(
    (a - b).abs().max().item()
    for a, b in zip(pre_merge_outputs, post_merge_outputs)
)
print(f"[CHECK] max |pre_merge - post_merge| = {max_diff:.8f}")
assert max_diff < 1e-3, "merge_and_unload() changed model behavior beyond floating-point noise"
print("[CHECK] PASSED -- merged model is behaviorally identical to the PEFT-wrapped model")

os.makedirs(OUT_DIR, exist_ok=True)
try:
    merged.save_pretrained(OUT_DIR, safe_serialization=True)
except Exception as e:
    print(f"[MERGE] safetensors save failed ({e}); retrying with safe_serialization=False")
    merged.save_pretrained(OUT_DIR, safe_serialization=False)
tokenizer.save_pretrained(OUT_DIR)

classifier_src = os.path.join(ADAPTER_DIR, "classifier.pt")
classifier_dst = os.path.join(OUT_DIR, "classifier.pt")
shutil.copyfile(classifier_src, classifier_dst)

print(f"[DONE] saved merged encoder + tokenizer + classifier.pt to {OUT_DIR}")
for f in sorted(os.listdir(OUT_DIR)):
    print(f"  {f} ({os.path.getsize(os.path.join(OUT_DIR, f)):,} bytes)")
