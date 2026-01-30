import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from tensorflow.keras.models import load_model


def evaluate(processed_dir: str, model_dir: str):
    X_test = pd.read_csv(f"{processed_dir}/X_test.csv").astype("float32").to_numpy()
    y_test = pd.read_csv(f"{processed_dir}/y_test.csv").values.ravel()

    model = load_model(f"{model_dir}/model.keras")

    y_prob = model.predict(X_test).ravel()
    y_pred = (y_prob > 0.5).astype(int)

    print(classification_report(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, y_prob))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
