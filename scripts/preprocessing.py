#!/usr/bin/env python3
# preprocessing.py
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

def add_feats(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["pdv","produto","week_end"]).copy()
    g = df.groupby(["pdv","produto"], observed=True)

    # quantity lags
    df["lag1"] = g["quantidade"].shift(1)
    df["lag2"] = g["quantidade"].shift(2)
    df["lag3"] = g["quantidade"].shift(3)
    df["lag4"] = g["quantidade"].shift(4)

    # rolling means (past-only)
    df["rmean4"] = (
        g["quantidade"].rolling(4, min_periods=1).mean()
         .reset_index(level=[0,1], drop=True).shift(1)
    )
    df["rmean12"] = (
        g["quantidade"].rolling(12, min_periods=1).mean()
         .reset_index(level=[0,1], drop=True).shift(1)
    )

    # price/margin lags
    for c in ["price_gross","price_net","margin","disc","taxes"]:
        if c in df.columns:
            df[f"{c}_lag1"] = g[c].shift(1)

    # context fallbacks
    df["store_rmean4"] = (
        df.groupby("pdv", observed=True)["quantidade"]
          .rolling(4, min_periods=1).mean()
          .reset_index(level=0, drop=True).shift(1)
    )
    df["prod_rmean4"] = (
        df.groupby("produto", observed=True)["quantidade"]
          .rolling(4, min_periods=1).mean()
          .reset_index(level=0, drop=True).shift(1)
    )
    return df

def build_weekly_from_tx(tx: pd.DataFrame) -> pd.DataFrame:
    tx = tx.copy()
    eps = 1e-9
    tx["unit_price_gross"] = tx["gross_value"] / (tx["quantity"] + eps)
    tx["unit_price_net"]   = tx["net_value"]   / (tx["quantity"] + eps)
    tx["unit_margin"]      = tx["gross_profit"]/ (tx["quantity"] + eps)
    tx["week_end"] = pd.to_datetime(tx["dt"]).dt.to_period("W-SAT").dt.end_time

    weekly = (
        tx.groupby(["internal_store_id","internal_product_id","week_end"], as_index=False)
          .agg(
              quantidade=("quantity","sum"),
              price_gross=("unit_price_gross","mean"),
              price_net=("unit_price_net","mean"),
              margin=("unit_margin","mean"),
              disc=("discount","mean"),
              taxes=("taxes","mean"),
              premise=("premise","first"),
              categoria_pdv=("categoria_pdv","first"),
              zipcode=("zipcode","first"),
              categoria=("categoria","first"),
              tipos=("tipos","first"),
              label=("label","first"),
              subcategoria=("subcategoria","first"),
              marca=("marca","first"),
              fabricante=("fabricante","first"),
          )
          .sort_values(["internal_store_id","internal_product_id","week_end"])
          .reset_index(drop=True)
    )
    weekly = weekly.rename(columns={
        "internal_store_id":"pdv",
        "internal_product_id":"produto"
    })

    # numeric coercion (same spirit as training)
    for c in ["price_gross","price_net","margin","disc","taxes","zipcode"]:
        if c in weekly.columns:
            weekly[c] = pd.to_numeric(weekly[c], errors="coerce").fillna(0)
    return weekly

def main():
    ap = argparse.ArgumentParser(description="Create test.csv (features + target_next) for Jan 2023.")
    ap.add_argument("--in-dir", default=".", help="Dir with part27_jan.parquet, part71_jan.parquet, part51_jan.parquet")
    ap.add_argument("--out-csv", default="test.csv", help="Output CSV path")
    ap.add_argument("--history-weekly", default=None,
                    help="Optional path to weekly history parquet (e.g., 2022 weekly) to compute proper lags/targets")
    args = ap.parse_args()

    BASE_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = BASE_DIR.parent
    IN_DIR = PROJECT_ROOT / "data" / "jan23_simul"
    # Load inputs
    df27 = pd.read_parquet(IN_DIR / "part27_jan.parquet")
    df71 = pd.read_parquet(IN_DIR / "part71_jan.parquet")
    df51 = pd.read_parquet(IN_DIR / "part51_jan.parquet")

    # Transactions: align with training steps
    tx = df51.rename(columns={"transaction_date":"dt"}).copy()
    tx["dt"] = pd.to_datetime(tx["dt"], errors="coerce")
    tx = tx.dropna(subset=["dt"])
    tx["quantity"] = pd.to_numeric(tx["quantity"], errors="coerce").fillna(0.0).clip(lower=0)
    for c in ["gross_value","net_value","gross_profit","discount","taxes"]:
        if c in tx.columns:
            tx[c] = pd.to_numeric(tx[c], errors="coerce").fillna(0.0)

    # Dimensions: left join (same as training)
    stores = df27.rename(columns={"pdv":"internal_store_id"}).copy()
    prods  = df71.rename(columns={"produto":"internal_product_id"}).copy()
    tx = tx.merge(stores, on="internal_store_id", how="left")
    tx = tx.merge(prods,  on="internal_product_id", how="left")

    # Build Jan 2023 weekly
    weekly_jan = build_weekly_from_tx(tx)
    weekly_jan["week_end"] = pd.to_datetime(weekly_jan["week_end"])

    # If history provided, concat to compute lags and a proper target_next
    if args.history_weekly:
        hist = pd.read_parquet(args.history_weekly).copy()
        hist["week_end"] = pd.to_datetime(hist["week_end"])
        combo = pd.concat([hist, weekly_jan], ignore_index=True, sort=False)
    else:
        combo = weekly_jan.copy()

    combo = combo.sort_values(["pdv","produto","week_end"]).reset_index(drop=True)

    # Feature engineering
    feats = add_feats(combo)

    # Target = next week's quantidade (to evaluate 1-step ahead)
    feats["target_next"] = feats.groupby(["pdv","produto"])["quantidade"].shift(-1)

    # Keep only January 2023 rows for testing
    jan_mask = (feats["week_end"] >= pd.Timestamp("2023-01-01")) & (feats["week_end"] <= pd.Timestamp("2023-01-31") + pd.offsets.MonthEnd(0))
    test = feats.loc[jan_mask].copy()

    # Match training dtypes for categoricals
    cat_cols = ["pdv","produto","premise","categoria_pdv","categoria","tipos","label","subcategoria","marca","fabricante"]
    for c in cat_cols:
        if c in test.columns:
            test[c] = test[c].astype("category")

    # week_ord like training
    test["week_ord"] = pd.to_datetime(test["week_end"]).view("int64") // 10**9

    # Final feature list (same as training)
    feature_cols = [
        "pdv","produto","week_ord",
        "lag1","lag2","lag3","lag4","rmean4","rmean12",
        "price_gross","price_net","margin","disc","taxes",
        "price_gross_lag1","price_net_lag1","margin_lag1","disc_lag1","taxes_lag1",
        "store_rmean4","prod_rmean4",
        "premise","categoria_pdv","categoria","tipos","label","subcategoria","marca","fabricante",
        "zipcode",
    ]
    keep_cols = ["week_end","quantidade","target_next"] + [c for c in feature_cols if c in test.columns]
    test = test[keep_cols].copy()

    # Downcast numerics
    for c in test.select_dtypes(include=["float64"]).columns:
        test[c] = pd.to_numeric(test[c], downcast="float")
    for c in test.select_dtypes(include=["int64"]).columns:
        test[c] = pd.to_numeric(test[c], downcast="integer")

    # Save CSV
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    test.to_csv(args.out_csv, index=False)
    print(f"Saved test set with features + target to: {args.out_csv}")
    print(test.head(3))

if __name__ == "__main__":
    main()
