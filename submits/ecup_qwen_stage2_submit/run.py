import argparse
import os

from src.utils import predict_pipeline

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "qwen3-embedding-0.6b-merged")
HEAD_PATH = os.path.join(BASE_DIR, "pair_head.pt")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, required=True, help="output CSV")
    parser.add_argument("--items_path", type=str, default=None, help="test items parquet")
    parser.add_argument("--matches_path", type=str, default=None, help="test matches parquet")
    args = parser.parse_args()

    predict_pipeline(
        items_path=args.items_path,
        matches_path=args.matches_path,
        model_path=MODEL_PATH,
        head_path=HEAD_PATH,
        output_csv_path=args.output_path,
    )


if __name__ == "__main__":
    main()
