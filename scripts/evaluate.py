#!/usr/bin/env python3
# evaluate.py  (no argparse, just hardcoded paths)

from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from lightgbm import LGBMRegressor

BASE_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
TEST_CSV   = PROJECT_ROOT / "data" / "test.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "lgbm_model_cap.pkl"              # e.g., test_folder/lgbm_model_cap.pkl
OUT_PREDS  = PROJECT_ROOT / "results" / "predictions.csv"

def wmape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.abs(y_true).sum() + 1e-9
    return float(np.abs(y_true - y_pred).sum() / denom)

def main():
    # Sanity checks
    if not TEST_CSV.exists():
        raise FileNotFoundError(f"Missing test CSV at: {TEST_CSV}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model file at: {MODEL_PATH}")

    print(f"Reading test set from: {TEST_CSV}")
    df = pd.read_csv(TEST_CSV)

    # Categorical columns must be category dtype (same as training)
    cat_cols = ["pdv","produto","premise","categoria_pdv","categoria","tipos","label","subcategoria","marca","fabricante"]
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].astype("category")

    # Exact training feature order
    feature_cols = [
        "pdv","produto","week_ord",
        "lag1","lag2","lag3","lag4","rmean4","rmean12",
        "price_gross","price_net","margin","disc","taxes",
        "price_gross_lag1","price_net_lag1","margin_lag1","disc_lag1","taxes_lag1",
        "store_rmean4","prod_rmean4",
        "premise","categoria_pdv","categoria","tipos","label","subcategoria","marca","fabricante",
        "zipcode",
    ]

    # Ensure ALL expected features exist and in the exact order (add missing as NaN)
    for col in feature_cols:
        if col not in df.columns:
            df[col] = np.nan
    X = df.reindex(columns=feature_cols)

    # Use only rows where we have ground truth for evaluation
    if "target_next" not in df.columns:
        raise ValueError("Column 'target_next' not found in test.csv (did preprocessing include it?).")

    valid = df["target_next"].notna()
    if valid.sum() == 0:
        raise ValueError(
            "No rows with non-NaN target_next in test.csv. "
            "Generate test.csv with January + some history so next-week targets exist."
        )

    Xv = X.loc[valid]
    y_true = df.loc[valid, "target_next"].astype(float).values

    print(f"Loading model from: {MODEL_PATH}")
    model: LGBMRegressor = joblib.load(MODEL_PATH)

    print("Predicting…")
    y_pred = model.predict(Xv)

    score = wmape(y_true, y_pred)
    print(f"WMAPE on {valid.sum()} rows: {score:.6f}")

    OUT_PREDS.parent.mkdir(parents=True, exist_ok=True)
    out = df.loc[valid, ["week_end","pdv","produto","quantidade","target_next"]].copy()
    out["pred"] = y_pred
    out.to_csv(OUT_PREDS, index=False)
    print(f"Saved predictions to: {OUT_PREDS}")

if __name__ == "__main__":
    main()
