"""
Merges the current best BERTA LoRA adapter (rays0n/berta-llm-continue-train,
val_macro_pr_auc=0.7255 -- the checkpoint the real test-stand submission
matching-berta_catboost-submit.zip used, which scored 0.4 on the actual
judging environment) into the base sergeyzh/BERTA weights via peft's
merge_and_unload(). Output is a plain BertModel checkpoint (no peft/LoRA
machinery needed to load it) -- meant to be published by hand as a Kaggle
dataset (project convention: manual publish via the web UI, not an in-kernel
`kaggle datasets create` call) and used as the base for a fresh LoRA SFT run
on matches_llm.parquet (see ../berta-llm-tpu-sft), replacing the resumed-
adapter pattern used by every other BERTA experiment so far -- merging first
gives that next round a clean full-precision starting point instead of
compounding a second LoRA adapter on top of the first.

The classifier head (ClassifierHead: Dropout -> Linear, see
../berta-hardneg-balanced's module docstring for why the checkpoint has this
wrapper) is copied alongside as classifier.pt for reference, but is NOT
merged into anything (LoRA never touched it) -- a fresh downstream SFT run
is expected to initialize its own new classifier head from scratch, same as
every fresh (non-resumed) run in this project's history.
"""
import os
import json
import time

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

try:
    import peft.import_utils
    import peft.tuners.lora.torchao
    peft.import_utils.is_torchao_available = lambda: False
    peft.tuners.lora.torchao.is_torchao_available = lambda: False
except Exception as e:
    print(f"[WARN] torchao patch failed (likely fine): {e}")

from peft import PeftModel

MODEL_NAME = "sergeyzh/BERTA"
CLASSIFIER_DROPOUT = 0.10

WORKING_DIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
OUTPUT_DIR = os.path.join(WORKING_DIR, "berta_merged")

t_script_start = time.perf_counter()


def _elapsed():
    return time.perf_counter() - t_script_start


class ClassifierHead(nn.Module):
    def __init__(self, hidden_dim, dropout_p):
        super().__init__()
        self.dropout = nn.Dropout(dropout_p)
        self.linear = nn.Linear(hidden_dim, 1)


def _find_checkpoint_dir():
    if os.path.isdir("/kaggle/input"):
        for root, _dirs, files in os.walk("/kaggle/input"):
            if "train_state.json" in files and "adapter" in _dirs:
                print(f"[FIND] found source checkpoint at {root}")
                return root
    raise RuntimeError("Source checkpoint not found under /kaggle/input -- expected "
                        "rays0n/berta-llm-continue-train mounted (kernel_sources).")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[CONFIG] model={MODEL_NAME} device={device}")

    if os.path.isdir("/kaggle/input"):
        for _root, _dirs, _files in os.walk("/kaggle/input"):
            depth = _root[len("/kaggle/input"):].count(os.sep)
            if depth <= 3:
                print(f"[DEBUG] {_root} -> dirs={_dirs} files={_files}")

    checkpoint_dir = _find_checkpoint_dir()
    with open(os.path.join(checkpoint_dir, "train_state.json")) as f:
        source_state = json.load(f)
    print(f"[SOURCE] global_step={source_state['global_step']} "
          f"best_val_macro_pr_auc={source_state['best_val_macro_pr_auc']}")

    print("[LOAD] base encoder (fp32 for a numerically clean merge) ...")
    base_model = AutoModel.from_pretrained(MODEL_NAME)
    hidden_dim = base_model.config.hidden_size
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    adapter_dir = os.path.join(checkpoint_dir, "adapter")
    print(f"[LOAD] LoRA adapter from {adapter_dir} ...")
    peft_model = PeftModel.from_pretrained(base_model, adapter_dir, is_trainable=False)
    peft_model.print_trainable_parameters()

    print("[MERGE] merge_and_unload() ...")
    merged_model = peft_model.merge_and_unload()
    print(f"[MERGE] done ({_elapsed():.1f}s) -- merged_model is a plain "
          f"{type(merged_model).__name__}, no peft/LoRA machinery left")

    merged_dir = os.path.join(OUTPUT_DIR, "merged_encoder")
    os.makedirs(merged_dir, exist_ok=True)
    merged_model.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)
    print(f"[SAVE] merged encoder + tokenizer -> {merged_dir}")

    classifier_head = ClassifierHead(hidden_dim, CLASSIFIER_DROPOUT)
    raw_state = torch.load(os.path.join(checkpoint_dir, "classifier.pt"), map_location="cpu")
    if "linear.weight" in raw_state:
        classifier_head.load_state_dict(raw_state)
    else:
        classifier_head.linear.load_state_dict(raw_state)
    torch.save(classifier_head.state_dict(), os.path.join(OUTPUT_DIR, "classifier.pt"))
    print(f"[SAVE] classifier head copied (reference only, LoRA never touched it) -> "
          f"{OUTPUT_DIR}/classifier.pt")

    with open(os.path.join(OUTPUT_DIR, "merge_meta.json"), "w") as f:
        json.dump({
            "base_model": MODEL_NAME, "hidden_dim": hidden_dim,
            "source_checkpoint": "rays0n/berta-llm-continue-train",
            "source_global_step": source_state["global_step"],
            "source_val_macro_pr_auc": source_state["best_val_macro_pr_auc"],
            "note": "merged_encoder/ is a plain BertModel checkpoint (config.json + "
                    "safetensors + tokenizer files) -- load with AutoModel.from_pretrained, "
                    "no peft needed. classifier.pt is the OLD classifier head, kept for "
                    "reference only -- a fresh downstream SFT run should init a new one.",
        }, f, indent=2)

    print(f"[DONE] merge complete, total wall-clock: {_elapsed():.0f}s")


if __name__ == "__main__":
    main()
