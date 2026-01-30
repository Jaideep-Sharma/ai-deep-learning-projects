import json
import joblib
import pandas as pd
from pathlib import Path
from tensorflow.keras.models import load_model

from src.data_preprocessing import transform_raw_features, NUMERIC_COLS


def predict(raw_df: pd.DataFrame, model_dir: str):
    model_dir = Path(model_dir)

    model = load_model(model_dir / "model.keras")
    scaler = joblib.load(model_dir / "scaler.pkl")

    # Apply SAME preprocessing as training
    X = transform_raw_features(raw_df)

    # Scale numeric columns
    X[NUMERIC_COLS] = scaler.transform(X[NUMERIC_COLS])

    # Predict
    probs = model.predict(X.astype("float32").to_numpy()).ravel()
    preds = (probs >= 0.5).astype(int)

    return probs, preds