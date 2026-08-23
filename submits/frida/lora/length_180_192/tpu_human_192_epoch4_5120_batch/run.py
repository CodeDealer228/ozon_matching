# DIAGNOSTIC SUBMISSION -- not a real scoring attempt. Deliberately forces a
# logged, non-zero-exit failure at ~5 minutes into the pipeline (see
# DIAGNOSTIC_FORCE_FAIL_AT_SECONDS in src/utils.py) so Private-stage progress
# is visible in the logs instead of being silently swallowed by a real
# "Container did not finish in time" timeout, which shows nothing. Also logs
# peak VRAM once at ~3 minutes. See README.md's "Как дёшево профилировать
# реальный H100-стенд" for the general technique.
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
        # BATCH_SIZE_FIXED=5120 default. This run WILL fail on purpose --
        # see the DIAGNOSTIC SUBMISSION note above.
    )


if __name__ == "__main__":
    main()
