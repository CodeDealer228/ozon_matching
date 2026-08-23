# REAL scoring attempt -- seq=128 already passed for real (0.398 test score),
# so this shorter-than-256-training length is no longer assumed to fail. Still
# carries the diagnostic safety net (progress-every-20%, one VRAM log,
# self-forced failure) from the earlier pure-diagnostic runs, but the deadline
# was raised (DIAGNOSTIC_FORCE_FAIL_AT_SECONDS in src/utils.py, now 700s) so it
# can no longer convert a real pass within budget into a false failure -- it
# only fires if Private is genuinely about to overrun. See README.md's
# "Как дёшево профилировать реальный H100-стенд" for the technique's origin.
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
