# REAL scoring attempt. Adapter+head here are the "checkpoint_hour08"
# checkpoint (global_step=31093, val_macro_pr_auc=0.6754) from
# kaggle_kernels/frida-llm-tpu-sft's LLM+human mixed run -- the most-trained
# checkpoint before the session was cut off (CANCEL_ACKNOWLEDGED at ~8h),
# NOT the human-only tpu_human_256_epoch4 lineage this submission's own
# PAIR_MAX_LENGTH=176/BATCH_SIZE_FIXED=5120 pipeline was proven against
# (0.4087 on ODS). Same proven seq_len/batch config, deliberately reused
# rather than re-derived, but a genuinely different checkpoint+head
# architecture (MultiLayerHead (1024,256), see src/utils.py).
#
# Deliberate pairing with submits/frida/tpu_human_llm_best_176_5120 (the
# same run's peak-val checkpoint, val=0.7060 at step 5177): this run's val
# fell sharply right after that peak and only slowly climbed back toward it
# over the following ~7h without fully recovering (see README.md) -- the
# same "best vs last" A/B this project has already used for BERTA/Qwen,
# testing whether more LLM+human exposure helps or hurts on the real
# judging stand despite scoring lower on val.
#
# No diagnostic self-crash -- an earlier version of the human-only lineage
# this file is copied from carried one (progress-every-20%, one VRAM log, a
# forced RuntimeError at a wall-clock deadline), but real ODS results showed
# it still firing on Private even after tuning -- removed there and never
# re-added here. See README.md's "Как дёшево профилировать реальный
# H100-стенд" for why the mechanism existed in the first place.
import argparse
import os

from src.utils import predict_pipeline

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_MODEL_PATH = os.path.join(BASE_DIR, "models", "frida-base")
ADAPTER_PATH = os.path.join(BASE_DIR, "models", "frida-lora-adapter")
CLASSIFIER_PATH = os.path.join(BASE_DIR, "classifier.pt")
WHEELS_DIR = os.path.join(BASE_DIR, "wheels")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_path", type=str, help="output file")
    parser.add_argument("--items_path", type=str, default=None, help="test items data path")
    parser.add_argument("--matches_path", type=str, default=None, help="test matches data path")
    args = parser.parse_args()

    predict_pipeline(
        items_path=args.items_path,
        matches_path=args.matches_path,
        base_model_path=BASE_MODEL_PATH,
        adapter_path=ADAPTER_PATH,
        classifier_path=CLASSIFIER_PATH,
        output_csv_path=args.output_path,
        wheels_dir=WHEELS_DIR,
        # batch_size intentionally omitted -- uses predict_pipeline's own
        # BATCH_SIZE_FIXED=5120 default, already proven safe at this
        # PAIR_MAX_LENGTH=176 on the real judging stand.
    )


if __name__ == "__main__":
    main()
