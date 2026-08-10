"""Manual Random Forest return predictor using final-data master panel.

Reads (via dissertation-models config):
  data/final data for AI/07_panel/master_panel.parquet

Uses prepare_xy feature sets:
  traditional | with_alt_data | all_indicators
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
DISS_MODELS = ROOT.parent / "dissertation-models"
FINAL_DATA = ROOT.parent.parent / "data" / "final data for AI"
MASTER_PANEL = FINAL_DATA / "07_panel" / "master_panel.parquet"

# Ticker filter — pick ONE of these forms:
#   None                          → all overlapping UK tickers in the Bloomberg panel
#   "CRST.LN"                     → one company
#   ["CRST.LN", "HSBA.LN"]        → several companies (must use square brackets)
TICKERS: str | list[str] | None = None
# traditional | with_alt_data | all_indicators
FEATURE_SET = "all_indicators"

N_ESTIMATORS = 200
MAX_DEPTH = 8


def _configure_paths() -> None:
    if not MASTER_PANEL.is_file():
        raise FileNotFoundError(f"Master panel not found: {MASTER_PANEL}")
    os.environ["DISSLAB_MASTER_PANEL"] = str(MASTER_PANEL.resolve())
    os.environ["DISSLAB_DATASET_LABEL"] = "final_data_master_panel"
    diss = str(DISS_MODELS.resolve())
    if diss not in sys.path:
        sys.path.insert(0, diss)


def _normalize_tickers(tickers: str | list[str] | None) -> list[str] | None:
    if tickers is None:
        return None
    if isinstance(tickers, str):
        return [tickers.upper()]
    return [str(t).upper() for t in tickers]


def run_random_forest_forecast(
    tickers: str | list[str] | None = TICKERS,
    feature_set: str = FEATURE_SET,
) -> dict:
    _configure_paths()
    warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)

    from data.build_panel import build_panel
    from data.features import prepare_xy

    selected = _normalize_tickers(tickers)
    _, splits = build_panel()
    train = splits["train"].copy()
    test = splits["test"].copy()

    if selected:
        train = train[train["Ticker"].isin(selected)]
        test = test[test["Ticker"].isin(selected)]
        missing = sorted(set(selected) - set(train["Ticker"].unique()) - set(test["Ticker"].unique()))
        if train.empty or test.empty:
            available = sorted(splits["test"]["Ticker"].unique())
            raise ValueError(
                f"No train/test rows for tickers {selected}. "
                f"Missing from panel: {missing or selected}. "
                f"Available: {available}"
            )

    X_tr, y_tr, cols = prepare_xy(train, feature_set)
    X_te, y_te, _ = prepare_xy(test, feature_set)
    if X_tr.empty or X_te.empty:
        raise ValueError("Empty feature matrix after prepare_xy — check master panel and split dates in config.yaml")

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    model = RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_tr_s, y_tr)
    pred = model.predict(X_te_s)

    y_arr = y_te.values
    mse = float(mean_squared_error(y_arr, pred))
    mae = float(mean_absolute_error(y_arr, pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_arr, pred))

    imp = sorted(zip(cols, model.feature_importances_), key=lambda x: -x[1])

    te_idx = X_te.index
    preview = pd.DataFrame({
        "Ticker": test.loc[te_idx, "Ticker"].astype(str).values,
        "month": pd.to_datetime(test.loc[te_idx, "month"]).dt.strftime("%Y-%m-%d").values,
        "y_true": y_arr,
        "y_pred": pred,
        "error": y_arr - pred,
    }).sort_values(["Ticker", "month"])
    preview_display = (
        preview.groupby("Ticker", sort=True).tail(1).reset_index(drop=True)
        if preview["Ticker"].nunique() > 1
        else preview.tail(5)
    )

    tickers_used = sorted(train["Ticker"].unique().tolist())
    label = (
        f"ALL ({len(tickers_used)} tickers)"
        if selected is None
        else ", ".join(tickers_used)
    )
    data_path = splits.get("master_panel") or splits.get("features_csv") or str(MASTER_PANEL)
    return {
        "ticker": label,
        "tickers": tickers_used,
        "feature_set": feature_set,
        "features_csv": str(data_path),
        "n_features": len(cols),
        "train_rows": len(y_tr),
        "test_rows": len(y_te),
        "train_start": pd.to_datetime(train["month"]).min().date().isoformat(),
        "train_end": pd.to_datetime(train["month"]).max().date().isoformat(),
        "test_start": pd.to_datetime(test["month"]).min().date().isoformat(),
        "test_end": pd.to_datetime(test["month"]).max().date().isoformat(),
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "feature_importance": imp,
        "preview": preview,
        "preview_display": preview_display,
        "test_rows_by_ticker": preview.groupby("Ticker").size().to_dict(),
    }


if __name__ == "__main__":
    result = run_random_forest_forecast(TICKERS, FEATURE_SET)

    print(f"Random Forest excess-return forecast | {result['ticker']}")
    print(f"Data : {Path(result['features_csv']).name} (final data master panel)")
    print(f"Split: train {result['train_start']} .. {result['train_end']} ({result['train_rows']} rows)")
    print(f"       test  {result['test_start']} .. {result['test_end']} ({result['test_rows']} rows)")
    print(f"Features ({result['feature_set']}): {result['n_features']} columns")
    print("\nOut-of-sample metrics (next-month excess return, ticker-months):")
    print(f"  MSE  : {result['mse']:.6f}")
    print(f"  MAE  : {result['mae']:.6f}")
    print(f"  RMSE : {result['rmse']:.6f}")
    print(f"  R²   : {result['r2']:.4f}")

    print("\nTop feature importance:")
    for name, score in result["feature_importance"][:15]:
        print(f"  {name:40s} {score:.4f}")

    if result["test_rows_by_ticker"]:
        print("\nTest rows by ticker:")
        for t, n in sorted(result["test_rows_by_ticker"].items())[:20]:
            print(f"  {t}: {n} months")
        extra = len(result["test_rows_by_ticker"]) - 20
        if extra > 0:
            print(f"  ... +{extra} more tickers")

    label = "Latest test month per ticker (sample):" if len(result["test_rows_by_ticker"]) > 1 else "Last 5 test predictions:"
    print(f"\n{label}")
    print(result["preview_display"].head(15).to_string(index=False))
