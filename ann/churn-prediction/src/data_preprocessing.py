import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# -------------------------
# Feature configuration
# -------------------------

NUMERIC_COLS = ["Age", "Tenure", "MonthlyCharges", "TotalCharges"]

PAYMENT_METHOD_CATS = [
    "Bank transfer",
    "Credit card",
    "Electronic check",
    "Mailed check",
]

CONTRACT_CATS = [
    "Month-to-month",
    "One year",
    "Two year",
]

EXPECTED_FEATURE_ORDER = (
    NUMERIC_COLS
    + ["Gender"]
    + [f"PaymentMethod_{c}" for c in PAYMENT_METHOD_CATS]
    + [f"Contract_{c}" for c in CONTRACT_CATS]
)

# ---------------------------------------------------
# Shared preprocessing (TRAINING + INFERENCE)
# ---------------------------------------------------
def transform_raw_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms raw input data into model-ready numeric features.
    MUST be identical for training and inference.
    """
    df = df.copy()

    # Binary encoding
    df["Gender"] = df["Gender"].map({"Female": 0, "Male": 1}).fillna(0)

    # One-hot encoding with FIXED schema
    payment_dummies = (
        pd.get_dummies(df["PaymentMethod"], prefix="PaymentMethod")
        .reindex(
            columns=[f"PaymentMethod_{c}" for c in PAYMENT_METHOD_CATS],
            fill_value=0,
        )
    )

    contract_dummies = (
        pd.get_dummies(df["Contract"], prefix="Contract")
        .reindex(
            columns=[f"Contract_{c}" for c in CONTRACT_CATS],
            fill_value=0,
        )
    )

    df = pd.concat([df, payment_dummies, contract_dummies], axis=1)

    # Drop unused raw columns
    df = df.drop(columns=["CustomerID", "PaymentMethod", "Contract"], errors="ignore")

    # Safety: fill missing numeric values
    df[NUMERIC_COLS] = df[NUMERIC_COLS].fillna(0)

    # 🔒 Enforce exact feature order
    return df[EXPECTED_FEATURE_ORDER]


# ---------------------------------------------------
# Training-only preprocessing
# ---------------------------------------------------
def preprocess_and_save(
    raw_csv_path: str,
    processed_dir: str,
    test_size: float = 0.2,
    random_state: int = 42,
):
    # Load raw data
    data = pd.read_csv(raw_csv_path)

    # Target encoding
    data["Churn"] = data["Churn"].map({"No": 0, "Yes": 1})

    # Feature engineering
    X = transform_raw_features(data)
    y = data["Churn"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    # Scaling (fit ONLY on train)
    scaler = StandardScaler()
    X_train[NUMERIC_COLS] = scaler.fit_transform(X_train[NUMERIC_COLS])
    X_test[NUMERIC_COLS] = scaler.transform(X_test[NUMERIC_COLS])

    # Save artifacts
    X_train.to_csv(f"{processed_dir}/X_train.csv", index=False)
    X_test.to_csv(f"{processed_dir}/X_test.csv", index=False)
    y_train.to_csv(f"{processed_dir}/y_train.csv", index=False)
    y_test.to_csv(f"{processed_dir}/y_test.csv", index=False)

    joblib.dump(scaler, f"{processed_dir}/scaler.pkl")

    return scaler
