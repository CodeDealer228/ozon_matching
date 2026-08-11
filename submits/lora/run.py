import argparse

from src.utils import predict_pipeline

CLASSIFIER_PATH = "classifier.pt"
MODEL_CE_PATH = "models/lora-merged-ms-marco-MiniLM-L12-v2"


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_path", type=str, help="output file")
    parser.add_argument("--items_path", type=str, default=None, help="test items data path")
    parser.add_argument("--matches_path", type=str, default=None, help="test matches data path")
    args = parser.parse_args()

    predict_pipeline(
        data_path=args.items_path,
        match_path=args.matches_path,
        model_path=MODEL_CE_PATH,
        classifier_path=CLASSIFIER_PATH,
        output_csv_path=args.output_path,
        batch_size=1024,
    )


if __name__ == "__main__":
    main()
