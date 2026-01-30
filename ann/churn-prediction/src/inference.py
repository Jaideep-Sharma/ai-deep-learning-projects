import json
import joblib
import pandas as pd
from pathlib import Path
from tensorflow.keras.models import load_model
import tensorflow as tf

from src.data_preprocessing import transform_raw_features, NUMERIC_COLS


def predict(raw_df: pd.DataFrame, model_dir: str):
    model_dir = Path(model_dir)

    # Try to load model with compatibility settings
    try:
        # For Keras 3.x compatibility
        with tf.keras.config.enable_unsafe_deserialization():
            model = load_model(model_dir / "model.keras")
    except Exception as e:
        # If that fails, try standard loading
        try:
            model = load_model(model_dir / "model.keras")
        except Exception as load_error:
            raise ValueError(
                f"Failed to load model: {load_error}\n"
                "The model was likely saved with an incompatible Keras version. "
                "Please retrain the model with: python run.py train"
            )
    
    scaler = joblib.load(model_dir / "scaler.pkl")

    # Apply SAME preprocessing as training
    X = transform_raw_features(raw_df)

    # Scale numeric columns
    X[NUMERIC_COLS] = scaler.transform(X[NUMERIC_COLS])

    # Predict
    probs = model.predict(X.astype("float32").to_numpy()).ravel()
    preds = (probs >= 0.5).astype(int)

    return probs, preds