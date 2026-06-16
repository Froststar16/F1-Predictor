"""
Trains the position-prediction model and stores the residual
distribution alongside it, since the residuals are what
model/simulate.py uses to turn point predictions into probabilities.

Run from the project root:
    python model/train.py
"""
import os
import sys
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from features.build_features import build_feature_matrix, FEATURE_COLUMNS

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "race_history.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "saved", "model.pkl")


def train():
    history = pd.read_csv(DATA_PATH)
    X, y, meta = build_feature_matrix(history)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model = lgb.LGBMRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=4, random_state=42, verbose=-1
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_val)
    residuals = (y_val.values - preds)
    mae = np.mean(np.abs(residuals))
    print(f"Validation MAE: {mae:.2f} positions")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump({
        "model": model,
        "residuals": residuals,
        "feature_columns": FEATURE_COLUMNS,
    }, MODEL_PATH)
    print(f"Saved model + residuals to {MODEL_PATH}")


if __name__ == "__main__":
    train()
