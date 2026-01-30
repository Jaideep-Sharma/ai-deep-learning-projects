import argparse
import pandas as pd
from pathlib import Path

from src.inference import predict


def main():
    parser = argparse.ArgumentParser(description="Run churn prediction on new customers")

    parser.add_argument(
        "--input-csv",
        required=True,
        help="Path to raw input CSV (same format as original data)"
    )

    parser.add_argument(
        "--model-dir",
        default="models/v1",
        help="Directory containing model.keras, scaler.pkl, metadata.json"
    )

    parser.add_argument(
        "--output-csv",
        default="predictions.csv",
        help="Path to save predictions"
    )

    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    if not input_csv.exists():
        raise FileNotFoundError(f"Input file not found: {input_csv}")

    # Load raw input
    raw_df = pd.read_csv(input_csv)

    # Run inference
    probs, preds = predict(raw_df, args.model_dir)

    # Save results
    result_df = raw_df.copy()
    result_df["churn_probability"] = probs
    result_df["churn_prediction"] = preds

    result_df.to_csv(args.output_csv, index=False)

    print(f"Predictions saved to {args.output_csv}")


if __name__ == "__main__":
    main()
