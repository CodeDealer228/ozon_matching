import argparse

from src.utils import predict_pipeline

# Cross-encoder-baseline-style pipeline (frozen encoder -> CLS -> Standard
# Scaler -> LogisticRegression), swapped from cross-encoder/ms-marco-MiniLM-
# L-12-v2 (33M) to omarelshehy/arabic-english-sts-matryoshka (xlm-roberta-
# large, ~560M) to probe how much bigger an encoder the test stand (1x H100
# 80GB / 20 CPU / 200GB RAM) can absorb within the timing budget.
# batch_size=512 (the organizer baseline's default) starved the H100; 4096 +
# adaptive OOM fallback (see src/utils.py) actually uses the hardware.
MODEL_PATH = "models/arabic-english-sts-matryoshka"
CLASSIFIER_PATH = "classifier.joblib"
BATCH_SIZE = 4096


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items_path", type=str, required=True)
    parser.add_argument("--matches_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    predict_pipeline(
        items_path=args.items_path,
        matches_path=args.matches_path,
        model_path=MODEL_PATH,
        classifier_path=CLASSIFIER_PATH,
        output_csv_path=args.output_path,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
