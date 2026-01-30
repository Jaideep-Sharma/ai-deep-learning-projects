import argparse
import joblib
from pathlib import Path

from src.data_preprocessing import preprocess_and_save
from src.train import train
from src.evaluate import evaluate


# ---------------------------------------------------
# Resolve project root safely (works from any folder)
# ---------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent

DEFAULT_RAW_DATA = PROJECT_ROOT / "data" / "raw" / "synthetic_customer_churn_100k.csv"
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "v1"


# ---------------------------------------------------
# CLI
# ---------------------------------------------------
parser = argparse.ArgumentParser(description="Churn Prediction Pipeline")

parser.add_argument(
    "command",
    choices=["preprocess", "train", "evaluate"],
    help="Pipeline step to execute"
)

parser.add_argument(
    "--raw-data",
    default=str(DEFAULT_RAW_DATA),
    help="Path to raw CSV data"
)

parser.add_argument(
    "--processed-dir",
    default=str(DEFAULT_PROCESSED_DIR),
    help="Directory to save processed datasets"
)

parser.add_argument(
    "--model-dir",
    default=str(DEFAULT_MODEL_DIR),
    help="Directory to save/load model artifacts"
)

args = parser.parse_args()


# ---------------------------------------------------
# Ensure directories exist
# ---------------------------------------------------
Path(args.processed_dir).mkdir(parents=True, exist_ok=True)
Path(args.model_dir).mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------
# Execute command
# ---------------------------------------------------
if args.command == "preprocess":
    scaler = preprocess_and_save(
        raw_csv_path=args.raw_data,
        processed_dir=args.processed_dir,
    )
    joblib.dump(scaler, f"{args.model_dir}/scaler.pkl")
    print("Preprocessing completed")

elif args.command == "train":
    train(
        processed_dir=args.processed_dir,
        model_dir=args.model_dir,
    )
    print("Training completed")

elif args.command == "evaluate":
    evaluate(
        processed_dir=args.processed_dir,
        model_dir=args.model_dir,
    )
    print("Evaluation completed")
