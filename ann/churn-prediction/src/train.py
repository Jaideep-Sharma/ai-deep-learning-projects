import json
import pandas as pd
import joblib
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from src.build_model import build_model


def train(processed_dir: str, model_dir: str):
    X_train = pd.read_csv(f"{processed_dir}/X_train.csv").astype("float32").to_numpy()
    y_train = pd.read_csv(f"{processed_dir}/y_train.csv").values.ravel()

    model = build_model(X_train.shape[1])

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, min_delta=1e-4, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", patience=2, factor=0.5, min_lr=1e-5),
    ]

    model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=50,
        batch_size=256,
        callbacks=callbacks,
        verbose=1,
    )

    model.save(f"{model_dir}/model.keras")

    print("Model training completed and saved")
