"""Stacked ensemble: saved XGB+LSTM forecasts -> ranked book -> local trading PPO (Arch B).

================================================================================
EDIT ONLY THE CONFIG BLOCK BELOW — all legs read settings from here
================================================================================

Architecture B (dissertation FTSE):
  Stage 1: load saved XGB + LSTM walk-forward forecasts -> y_pred -> ranked_signals.csv
  Stage 2: trading_ppo/PPO_fixed.py times BUY/SELL/HOLD on that book (local only)

Run:
  python stacked_ensemble.py
  run_ensemble.bat

Outputs: esemble/saved_data/
"""
from __future__ import annotations

# =============================================================================
# CONFIG — change everything here (single edit point)
# =============================================================================

# Default: reuse already-run FTSE XGB/LSTM (no Yahoo retrain).
# Legacy: STAGE1_MODE = "retrain" with DATA_SOURCE yahoo|bloomberg
STAGE1_MODE = "saved_forecasts"  # "saved_forecasts" | "retrain"
DATA_SOURCE = "bloomberg"  # only used when STAGE1_MODE == "retrain"
TICKERS: str | list[str] | None = None  # FTSE: None = all names in saved forecasts
FEATURE_SET = "traditional"
INCLUDE_ALT_DATA = True
YAHOO_REFRESH_IF_MISSING = False

# --- Saved walk-forward forecast roots (Stage 1) ---
# XGB: prefer full preview.csv (written by xgb_walk_forward); falls back to top50_picks.csv
XGB_FORECAST_ROOT = None  # set below after Path imports if None
LSTM_FORECAST_ROOT = None
BLEND_MODE = "ic_weight"  # "ic_weight" | "fixed"
FIXED_W_XGB = 0.7

# --- Yahoo date splits (retrain legacy only) ---
YAHOO_BASE_TRAIN_END = "2018-12-31"
YAHOO_VAL_START = "2019-01-01"
YAHOO_VAL_END = "2020-12-31"
YAHOO_TEST_START = "2021-01-01"
YAHOO_TEST_END = "2025-12-31"

# --- XGBoost / LSTM retrain knobs (legacy STAGE1_MODE=retrain only) ---
XGB_N_ESTIMATORS = 100
XGB_MAX_DEPTH = 4
XGB_LEARNING_RATE = 0.05
SEQ_LEN = 6
LSTM_HIDDEN_SIZE = 32
LSTM_EPOCHS = 15
LSTM_BATCH_SIZE = 64

# --- Legacy retrain-only: tiny XGB/LSTM blend meta (NOT trading PPO; unused in saved_forecasts) ---
META_PPO_TIMESTEPS = 2000
PPO_N_STEPS = 64
PPO_BATCH_SIZE = 32
PPO_LEARNING_RATE = 3e-4
PPO_GAMMA = 0.99

# --- Architecture B Stage 2: trading PPO — ONE knob ---
AUTO_START_TRADING_PPO = True
TRADING_PPO_TIMESTEPS = 100000  # <<< only timestep that matters for Stage 2

# Curated reward / training overrides applied into local dynamic_config at Stage-2 launch
STAGE2_REWARD_OVERRIDES: dict = {
    "pnl_reward_scale": 72.0,
    "base_trading_incentive": 0.18,
    "sell_base_incentive": 0.18,
    "sell_unconditional_base": 0.08,
    "wrong_regime_buy_penalty": -0.15,
    "profit_target_pct": 0.03,
    "profit_target_5pct_bonus": 0.28,
    "ent_coef": 0.15,
    "learning_rate": 5e-4,
}

# =============================================================================
# END CONFIG
# =============================================================================

import json
import os
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from attribution_report import build_attribution, print_summary
from features_core import prepare_xy
from forecast_loader import load_and_blend_forecasts
from lstm_core import build_sequences, fit_predict_lstm
from ppo_meta import fallback_blend, predict_ppo_meta, sb3_available, train_ppo_meta
from signal_interface import build_ranked_signals, save_ranked_signals, signal_summary

ROOT = Path(__file__).resolve().parent
MANUAL = ROOT.parent
DISS_MODELS = MANUAL.parent / "dissertation-models"
SAVE_DIR = ROOT / "saved_data"
TRADING_PPO_DIR = ROOT / "trading_ppo"
TRADING_PPO_SCRIPT = TRADING_PPO_DIR / "PPO_fixed.py"
RANKED_SIGNALS_PATH = SAVE_DIR / "ranked_signals.csv"
FINAL_DATA = MANUAL.parent.parent / "data" / "final data for AI"
MASTER_PANEL = FINAL_DATA / "07_panel" / "master_panel.parquet"
FUNDAMENTALS_CSV = FINAL_DATA / "02_fundamentals" / "Fundamental_Data_Full.csv"
RISK_FREE_CSV = FINAL_DATA / "05_macro" / "risk_free_monthly.csv"

# Default forecast locations (FTSE walk-forward outputs)
if XGB_FORECAST_ROOT is None:
    # Ensemble-local copy of XGB walk-forward (user-maintained folder)
    XGB_FORECAST_ROOT = ROOT / "xgboost data" / "walk_forward_baseline"
if LSTM_FORECAST_ROOT is None:
    # Ensemble-local copy of LSTM walk-forward (user-maintained folder)
    LSTM_FORECAST_ROOT = ROOT / "lstm" / "walk_forward_baseline"


def get_ppo_config() -> dict[str, Any]:
    return {
        "ppo_meta": {
            "total_timesteps": META_PPO_TIMESTEPS,
            "n_steps": PPO_N_STEPS,
            "batch_size": PPO_BATCH_SIZE,
            "learning_rate": PPO_LEARNING_RATE,
            "gamma": PPO_GAMMA,
        }
    }


def _ensure_paths() -> None:
    if str(MANUAL) not in sys.path:
        sys.path.insert(0, str(MANUAL))
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))


def _normalize_tickers(tickers: str | list[str] | None) -> list[str] | None:
    if tickers is None:
        return None
    if isinstance(tickers, str):
        return [tickers.upper()]
    return [str(t).upper() for t in tickers]


def _leg_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mse = float(mean_squared_error(y_true, y_pred))
    mae = float(mean_absolute_error(y_true, y_pred))
    return {
        "mse": mse,
        "mae": mae,
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(y_true, y_pred)),
        "n": int(len(y_true)),
    }


def _print_leg(name: str, split: str, metrics: dict[str, float], extra: str = "") -> None:
    suffix = f" | {extra}" if extra else ""
    print(
        f"  {name} ({split}, n={metrics['n']}): "
        f"MSE={metrics['mse']:.6f} MAE={metrics['mae']:.6f} "
        f"RMSE={metrics['rmse']:.6f} R²={metrics['r2']:.4f}{suffix}"
    )


def _configure_bloomberg() -> None:
    if not MASTER_PANEL.is_file():
        raise FileNotFoundError(f"Master panel not found: {MASTER_PANEL}")
    os.environ["DISSLAB_MASTER_PANEL"] = str(MASTER_PANEL.resolve())
    os.environ["DISSLAB_DATASET_LABEL"] = "final_data_master_panel"
    if FUNDAMENTALS_CSV.is_file():
        os.environ["DISSLAB_FUNDAMENTALS_CSV"] = str(FUNDAMENTALS_CSV.resolve())
    if RISK_FREE_CSV.is_file():
        os.environ["DISSLAB_RISK_FREE_CSV"] = str(RISK_FREE_CSV.resolve())
    if str(DISS_MODELS) not in sys.path:
        sys.path.insert(0, str(DISS_MODELS))
    print(f"  Master panel           : {MASTER_PANEL.name}")
    print(f"  Fundamentals (ref)     : {FUNDAMENTALS_CSV.name}")


def _split_yahoo(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    m = pd.to_datetime(panel["month"])
    train = panel.loc[m <= pd.Timestamp(YAHOO_BASE_TRAIN_END)].copy()
    val = panel.loc[(m >= pd.Timestamp(YAHOO_VAL_START)) & (m <= pd.Timestamp(YAHOO_VAL_END))].copy()
    test = panel.loc[(m >= pd.Timestamp(YAHOO_TEST_START)) & (m <= pd.Timestamp(YAHOO_TEST_END))].copy()
    return train, val, test


def _pred_xgb_panel(train: pd.DataFrame, df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame | None:
    target = "target_excess_return"
    tr = train.dropna(subset=[target])
    cols = [c for c in feature_cols if c in tr.columns]
    if tr.empty or not cols:
        return None

    X_tr = tr[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y_tr = tr[target]
    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr)

    xgb = XGBRegressor(
        n_estimators=XGB_N_ESTIMATORS,
        max_depth=XGB_MAX_DEPTH,
        learning_rate=XGB_LEARNING_RATE,
        random_state=42,
        n_jobs=1,
    )
    xgb.fit(X_tr_s, y_tr)

    te = df.dropna(subset=[target]).copy()
    if te.empty:
        return None
    X_te = te[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return pd.DataFrame({
        "Ticker": te["Ticker"].astype(str).values,
        "month": pd.to_datetime(te["month"]).dt.strftime("%Y-%m-%d").values,
        "y": te[target].values,
        "xgb_pred": xgb.predict(sc.transform(X_te)),
    })


def _pred_xgb_bloomberg(train: pd.DataFrame, df: pd.DataFrame, feature_set: str) -> pd.DataFrame | None:
    X_tr, y_tr, _ = prepare_xy(train, feature_set)
    if X_tr.empty:
        return None
    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr)
    xgb = XGBRegressor(
        n_estimators=XGB_N_ESTIMATORS,
        max_depth=XGB_MAX_DEPTH,
        learning_rate=XGB_LEARNING_RATE,
        random_state=42,
        n_jobs=1,
    )
    xgb.fit(X_tr_s, y_tr)

    X_df, y_df, _ = prepare_xy(df, feature_set)
    if X_df.empty:
        return None
    idx = X_df.index
    return pd.DataFrame({
        "Ticker": df.loc[idx, "Ticker"].astype(str).values,
        "month": pd.to_datetime(df.loc[idx, "month"]).dt.strftime("%Y-%m-%d").values,
        "y": y_df.values,
        "xgb_pred": xgb.predict(sc.transform(X_df)),
    })


def _lstm_preds(
    train: pd.DataFrame,
    df: pd.DataFrame,
    feature_cols: list[str] | None,
    feature_set: str,
    bloomberg: bool,
) -> pd.DataFrame | None:
    if bloomberg:
        X_tr, y_tr, _, _ = build_sequences(train, SEQ_LEN, feature_cols=None, feature_set=feature_set)
        X_df, _, _, keys = build_sequences(df, SEQ_LEN, feature_cols=None, feature_set=feature_set)
    else:
        X_tr, y_tr, _, _ = build_sequences(train, SEQ_LEN, feature_cols=feature_cols or [], feature_set=feature_set)
        X_df, _, _, keys = build_sequences(df, SEQ_LEN, feature_cols=feature_cols or [], feature_set=feature_set)

    if X_tr is None or X_df is None or not keys:
        return None
    pred = fit_predict_lstm(X_tr, y_tr, X_df, LSTM_HIDDEN_SIZE, LSTM_EPOCHS, LSTM_BATCH_SIZE)
    tickers, months = zip(*keys)
    return pd.DataFrame({"Ticker": list(tickers), "month": list(months), "lstm_pred": pred})


def run_stacked_ensemble_core(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str] | None,
    feature_set: str,
    bloomberg: bool,
    data_label: str,
    tickers_used: list[str],
) -> dict[str, Any]:
    print("\n[1/3] XGBoost supervised leg")
    if bloomberg:
        xgb_val = _pred_xgb_bloomberg(train, val, feature_set)
        xgb_test = _pred_xgb_bloomberg(train, test, feature_set)
        lstm_val = _lstm_preds(train, val, None, feature_set, bloomberg=True)
        lstm_test = _lstm_preds(train, test, None, feature_set, bloomberg=True)
    else:
        xgb_val = _pred_xgb_panel(train, val, feature_cols or [])
        xgb_test = _pred_xgb_panel(train, test, feature_cols or [])
        lstm_val = _lstm_preds(train, val, feature_cols, feature_set, bloomberg=False)
        lstm_test = _lstm_preds(train, test, feature_cols, feature_set, bloomberg=False)

    if xgb_val is None or xgb_test is None:
        raise ValueError("XGBoost leg produced no rows — check features and splits.")

    xgb_val_m = _leg_metrics(xgb_val["y"].values, xgb_val["xgb_pred"].values)
    xgb_test_m = _leg_metrics(xgb_test["y"].values, xgb_test["xgb_pred"].values)
    _print_leg("XGBoost", "validation", xgb_val_m)
    _print_leg("XGBoost", "test", xgb_test_m)

    print("\n[2/3] LSTM supervised leg")
    lstm_val_m = lstm_test_m = None
    if lstm_val is not None and lstm_test is not None:
        # LSTM rows are a subset — align on keys for fair metrics
        lv = xgb_val.merge(lstm_val, on=["Ticker", "month"], how="inner")
        lt = xgb_test.merge(lstm_test, on=["Ticker", "month"], how="inner")
        if not lv.empty:
            lstm_val_m = _leg_metrics(lv["y"].values, lv["lstm_pred"].values)
            _print_leg("LSTM", "validation", lstm_val_m, f"{len(lv)} ticker-months")
        if not lt.empty:
            lstm_test_m = _leg_metrics(lt["y"].values, lt["lstm_pred"].values)
            _print_leg("LSTM", "test", lstm_test_m, f"{len(lt)} ticker-months")
    else:
        print("  LSTM skipped — insufficient sequence history per ticker")

    meta_val = xgb_val.merge(lstm_val, on=["Ticker", "month"], how="left") if lstm_val is not None else xgb_val.copy()
    meta_test = xgb_test.merge(lstm_test, on=["Ticker", "month"], how="left") if lstm_test is not None else xgb_test.copy()

    use_lstm = (
        lstm_val is not None
        and lstm_test is not None
        and meta_val["lstm_pred"].notna().sum() >= 5
        and meta_test["lstm_pred"].notna().sum() >= 1
    )

    cfg = get_ppo_config()
    print("\n[3/3] PPO meta-learner (blend XGB + LSTM on validation)")
    if not sb3_available():
        print("  WARNING: stable-baselines3 not installed — pip install stable-baselines3 gymnasium")
    print(f"  validation rows for PPO: {len(meta_val)} | total_timesteps={META_PPO_TIMESTEPS}")
    ppo_model = train_ppo_meta(meta_val, use_lstm, cfg) if len(meta_val) >= 5 else None
    if ppo_model is not None:
        pred_test, blend_stats = predict_ppo_meta(ppo_model, meta_test, use_lstm)
        meta_method = "ppo"
    else:
        pred_test = fallback_blend(meta_test, use_lstm)
        blend_stats = None
        meta_method = "fallback_equal_weight" if use_lstm else "xgb_only"
        if not sb3_available():
            print(f"[info] stable-baselines3 not installed — using {meta_method}")

    y_test = meta_test["y"].values
    mse = float(mean_squared_error(y_test, pred_test))
    mae = float(mean_absolute_error(y_test, pred_test))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_test, pred_test))

    frame = meta_test.copy()
    frame = frame.rename(columns={"y": "y_true"})
    frame["y_pred"] = pred_test
    frame["error"] = frame["y_true"] - frame["y_pred"]
    frame["abs_error"] = frame["error"].abs()
    frame["month"] = pd.to_datetime(frame["month"])

    preview = frame.sort_values(["Ticker", "month"])
    preview_display = (
        preview.groupby("Ticker", sort=True).tail(1).reset_index(drop=True)
        if preview["Ticker"].nunique() > 1
        else preview.tail(5)
    )

    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    pred_path = SAVE_DIR / "ensemble_predictions.csv"
    frame.to_csv(pred_path, index=False)

    export = frame[["month", "Ticker", "xgb_pred", "lstm_pred", "y_pred", "y_true"]].copy()
    export = export.rename(columns={"month": "date", "y_pred": "ensemble_pred"})
    export_path = SAVE_DIR / "ppo_forecast_export.csv"
    export.to_csv(export_path, index=False)

    # Architecture B: rank + size from y_pred (Stage 2 trading PPO times these weights)
    ranked = build_ranked_signals(frame)
    ranked_path = save_ranked_signals(ranked, SAVE_DIR / "ranked_signals.csv")
    ranked_info = signal_summary(ranked)

    metrics = {
        "data_source": DATA_SOURCE,
        "data_label": data_label,
        "tickers": tickers_used,
        "meta_method": meta_method,
        "used_lstm": bool(use_lstm),
        "supervised_legs": ["xgboost", "lstm"] if use_lstm else ["xgboost"],
        "base_train_end": str(pd.to_datetime(train["month"]).max().date()),
        "val_start": str(pd.to_datetime(val["month"]).min().date()) if not val.empty else None,
        "val_end": str(pd.to_datetime(val["month"]).max().date()) if not val.empty else None,
        "test_start": str(pd.to_datetime(test["month"]).min().date()),
        "test_end": str(pd.to_datetime(test["month"]).max().date()),
        "test_rows": int(len(frame)),
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "blend_stats": blend_stats,
        "xgb_val": xgb_val_m,
        "xgb_test": xgb_test_m,
        "lstm_val": lstm_val_m,
        "lstm_test": lstm_test_m,
        "ppo_meta_total_timesteps": META_PPO_TIMESTEPS,
        "ppo_meta_n_steps": PPO_N_STEPS,
        "ppo_config": cfg["ppo_meta"],
        "architecture_b": ranked_info,
        "trading_ppo": {
            "local_copy": str(TRADING_PPO_DIR),
            "timesteps": TRADING_PPO_TIMESTEPS,
            "signal_file": str(ranked_path),
            "note": "Architecture B — Stage 2 times ranked book (local trading_ppo)",
        },
    }
    metrics_path = SAVE_DIR / "ensemble_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    ppo_path = SAVE_DIR / "ppo_meta_blend.zip"
    if ppo_model is not None:
        ppo_model.save(str(ppo_path))

    # Stage-1 attribution (PPO section filled / refreshed after Stage 2)
    try:
        attr = build_attribution(ranked_path=ranked_path, metrics_path=metrics_path, out_dir=SAVE_DIR)
        attr_paths = attr.get("paths", {})
    except Exception as attr_err:
        attr_paths = {"attribution_error": str(attr_err)}

    return {
        **metrics,
        "preview": preview,
        "preview_display": preview_display,
        "test_rows_by_ticker": preview.groupby("Ticker").size().to_dict(),
        "paths": {
            "predictions": str(pred_path),
            "metrics": str(metrics_path),
            "export": str(export_path),
            "ranked_signals": str(ranked_path),
            "ppo_model": str(ppo_path) if ppo_model is not None else None,
            "trading_ppo": str(TRADING_PPO_DIR),
            "attribution_md": attr_paths.get("attribution_md"),
            "attribution_json": attr_paths.get("attribution_json"),
            "trade_attribution": attr_paths.get("trade_attribution_csv"),
        },
    }


def run_yahoo(tickers: list[str] | None) -> dict[str, Any]:
    from yahoo_panel import DEFAULT_TICKERS, build_yahoo_panel, yahoo_feature_columns

    selected = tickers or DEFAULT_TICKERS
    selected_u = [str(t).upper() for t in selected]
    panel = build_yahoo_panel(selected_u, refresh=False)
    have = set(panel["Ticker"].astype(str).str.upper().unique()) if not panel.empty else set()
    missing = [t for t in selected_u if t not in have]
    if missing and YAHOO_REFRESH_IF_MISSING:
        print(f"  Yahoo cache missing {len(missing)} tickers — refreshing: {missing[:8]}{'...' if len(missing) > 8 else ''}")
        panel = build_yahoo_panel(selected_u, refresh=True)
    panel = panel[panel["Ticker"].astype(str).str.upper().isin(selected_u)].copy()
    train, val, test = _split_yahoo(panel)
    if train.empty or val.empty or test.empty:
        raise ValueError(
            f"Empty yahoo split: train={len(train)} val={len(val)} test={len(test)}. "
            "Set YAHOO_REFRESH_IF_MISSING=True or run ../fetch_yahoo_data.py first."
        )

    feature_cols = yahoo_feature_columns(include_alt=INCLUDE_ALT_DATA, panel=panel)
    return run_stacked_ensemble_core(
        train, val, test, feature_cols, FEATURE_SET, bloomberg=False,
        data_label=str(MANUAL / "saved_data" / "yahoo" / "panel_monthly.csv"),
        tickers_used=sorted(panel["Ticker"].astype(str).str.upper().unique().tolist()),
    )


def launch_trading_ppo() -> int:
    """Stage 2: run local trading_ppo/PPO_fixed.py after ranked_signals.csv exists."""
    if not TRADING_PPO_SCRIPT.is_file():
        raise FileNotFoundError(f"Trading PPO not found: {TRADING_PPO_SCRIPT}")
    if not RANKED_SIGNALS_PATH.is_file():
        raise FileNotFoundError(f"Ranked signals missing: {RANKED_SIGNALS_PATH}")

    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    overrides_path = SAVE_DIR / "stage2_overrides.json"
    ts = int(TRADING_PPO_TIMESTEPS)
    payload = {
        "start_timesteps": ts,
        "total_timesteps": ts,
        "reward_overrides": dict(STAGE2_REWARD_OVERRIDES),
        "ranked_signals": str(RANKED_SIGNALS_PATH.resolve()),
        "note": "Written by stacked_ensemble.launch_trading_ppo — local esemble only",
    }
    overrides_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\n" + "=" * 72)
    print("STAGE 2: Launching trading PPO (Architecture B timing | FTSE local)")
    print(f"  script : {TRADING_PPO_SCRIPT}")
    print(f"  signals: {RANKED_SIGNALS_PATH}")
    print(f"  overrides: {overrides_path}")
    print(f"  timesteps: {ts:,}")
    print(f"  cwd    : {TRADING_PPO_DIR}")
    print("=" * 72)

    env = os.environ.copy()
    # Prefer local trading_ppo package (config.py + ppo/) — never Pathrise github private
    env["PYTHONPATH"] = str(TRADING_PPO_DIR) + os.pathsep + env.get("PYTHONPATH", "")
    env["ESEMBLE_RANKED_SIGNALS"] = str(RANKED_SIGNALS_PATH.resolve())
    env["ESEMBLE_ARCH_B"] = "1"
    env["ESEMBLE_START_TIMESTEPS"] = str(ts)
    env["ESEMBLE_TOTAL_TIMESTEPS"] = str(ts)
    env["ESEMBLE_REWARD_OVERRIDES"] = str(overrides_path.resolve())
    env["DISS_FINAL_DATA"] = str(FINAL_DATA.resolve())
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:512")

    # Prefer Python 3.11 + CUDA for Stage 2 when available (3.13 often CPU-only here)
    py311 = Path(r"C:\Users\Faiaz\AppData\Local\Programs\Python\Python311\python.exe")
    stage2_py = str(py311) if py311.is_file() else sys.executable
    print(f"  python : {stage2_py}")
    try:
        import torch as _torch
        print(f"  cuda   : {_torch.cuda.is_available()} | {(_torch.cuda.get_device_name(0) if _torch.cuda.is_available() else 'n/a')}")
    except Exception as _e:
        print(f"  cuda   : check failed ({_e})")

    completed = subprocess.run(
        [stage2_py, str(TRADING_PPO_SCRIPT)],
        cwd=str(TRADING_PPO_DIR),
        env=env,
    )
    print(f"\n[Stage 2] trading PPO exit code: {completed.returncode}")
    return int(completed.returncode)


def run_from_saved_forecasts() -> dict[str, Any]:
    """Stage 1 from saved XGB/LSTM walk-forward — no retrain."""
    xgb_root = Path(XGB_FORECAST_ROOT)
    lstm_root = Path(LSTM_FORECAST_ROOT)
    print(f"[Stage 1] Loading saved forecasts")
    print(f"  XGB : {xgb_root}")
    print(f"  LSTM: {lstm_root}")
    print(f"  blend: {BLEND_MODE}")

    frame, meta = load_and_blend_forecasts(
        xgb_root,
        lstm_root,
        blend_mode=BLEND_MODE,
        fixed_w_xgb=FIXED_W_XGB,
    )
    print(
        f"  merged rows={meta['n_rows']} tickers={meta['n_tickers']} "
        f"months={meta['month_min']}..{meta['month_max']} w_xgb={meta['w_xgb']:.3f}"
    )
    print(f"  xgb_source={meta['xgb_source']} | lstm_source={meta['lstm_source']}")

    y_true = frame["y_true"].to_numpy(dtype=float)
    y_pred = frame["y_pred"].to_numpy(dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 5:
        raise ValueError("Too few finite y_true/y_pred rows after blend")
    mse = float(mean_squared_error(y_true[mask], y_pred[mask]))
    mae = float(mean_absolute_error(y_true[mask], y_pred[mask]))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true[mask], y_pred[mask]))

    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    pred_path = SAVE_DIR / "ensemble_predictions.csv"
    frame.to_csv(pred_path, index=False)

    export = frame[["month", "Ticker", "xgb_pred", "lstm_pred", "y_pred", "y_true"]].copy()
    export = export.rename(columns={"month": "date", "y_pred": "ensemble_pred"})
    export_path = SAVE_DIR / "ppo_forecast_export.csv"
    export.to_csv(export_path, index=False)

    ranked = build_ranked_signals(frame)
    ranked_path = save_ranked_signals(ranked, SAVE_DIR / "ranked_signals.csv")
    ranked_info = signal_summary(ranked)

    tickers_used = sorted(frame["Ticker"].astype(str).unique().tolist())
    preview = frame.sort_values(["Ticker", "month"])
    preview_display = (
        preview.groupby("Ticker", sort=True).tail(1).reset_index(drop=True)
        if preview["Ticker"].nunique() > 1
        else preview.tail(5)
    )

    metrics = {
        "stage1_mode": "saved_forecasts",
        "data_source": "ftse_saved_walk_forward",
        "data_label": f"xgb={xgb_root} | lstm={lstm_root}",
        "tickers": tickers_used,
        "meta_method": f"saved_{BLEND_MODE}",
        "used_lstm": True,
        "supervised_legs": ["xgboost", "lstm"],
        "base_train_end": None,
        "val_start": None,
        "val_end": None,
        "test_start": meta["month_min"],
        "test_end": meta["month_max"],
        "test_rows": int(len(frame)),
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "blend_stats": {
            "mean_blend_weight_on_xgb": meta["w_xgb"],
            "std_blend_weight_on_xgb": 0.0,
            "w_xgb": meta["w_xgb"],
            "xgb_source": meta["xgb_source"],
            "lstm_source": meta["lstm_source"],
        },
        "forecast_meta": meta,
        "architecture_b": ranked_info,
        "trading_ppo": {
            "local_copy": str(TRADING_PPO_DIR),
            "timesteps": TRADING_PPO_TIMESTEPS,
            "signal_file": str(ranked_path),
            "reward_overrides": STAGE2_REWARD_OVERRIDES,
            "note": "Architecture B — Stage 2 times ranked FTSE book (local trading_ppo)",
        },
    }
    metrics_path = SAVE_DIR / "ensemble_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")

    try:
        attr = build_attribution(ranked_path=ranked_path, metrics_path=metrics_path, out_dir=SAVE_DIR)
        attr_paths = attr.get("paths", {})
    except Exception as attr_err:
        attr_paths = {"attribution_error": str(attr_err)}

    return {
        **metrics,
        "preview": preview,
        "preview_display": preview_display,
        "test_rows_by_ticker": preview.groupby("Ticker").size().to_dict(),
        "paths": {
            "predictions": str(pred_path),
            "metrics": str(metrics_path),
            "export": str(export_path),
            "ranked_signals": str(ranked_path),
            "ppo_model": None,
            "trading_ppo": str(TRADING_PPO_DIR),
            "attribution_md": attr_paths.get("attribution_md"),
            "attribution_json": attr_paths.get("attribution_json"),
            "trade_attribution": attr_paths.get("trade_attribution_csv"),
        },
    }


def run_bloomberg(tickers: list[str] | None) -> dict[str, Any]:
    _configure_bloomberg()
    from data.build_panel import build_panel

    _, splits = build_panel()
    train, val, test = splits["train"].copy(), splits["val"].copy(), splits["test"].copy()
    if tickers:
        train = train[train["Ticker"].isin(tickers)]
        val = val[val["Ticker"].isin(tickers)]
        test = test[test["Ticker"].isin(tickers)]
    if train.empty or val.empty or test.empty:
        raise ValueError("Empty bloomberg split — check tickers and config.yaml")

    tickers_used = sorted(set(train["Ticker"]) | set(test["Ticker"]))
    return run_stacked_ensemble_core(
        train, val, test, None, FEATURE_SET, bloomberg=True,
        data_label=str(MASTER_PANEL), tickers_used=tickers_used,
    )


def run_stacked_ensemble(tickers: str | list[str] | None = TICKERS) -> dict[str, Any]:
    warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
    _ensure_paths()
    if STAGE1_MODE == "saved_forecasts":
        return run_from_saved_forecasts()
    selected = _normalize_tickers(tickers)
    if DATA_SOURCE == "yahoo":
        return run_yahoo(selected)
    return run_bloomberg(selected)


if __name__ == "__main__":
    print("Config snapshot:")
    print(f"  STAGE1_MODE              : {STAGE1_MODE}")
    print(f"  DATA_SOURCE              : {DATA_SOURCE}")
    print(f"  TICKERS                  : {TICKERS if TICKERS is not None else 'ALL (from forecasts)'}")
    print(f"  XGB_FORECAST_ROOT        : {XGB_FORECAST_ROOT}")
    print(f"  LSTM_FORECAST_ROOT       : {LSTM_FORECAST_ROOT}")
    print(f"  BLEND_MODE               : {BLEND_MODE}")
    print(f"  Auto-start trading PPO   : {AUTO_START_TRADING_PPO}")
    print(f"  Trading PPO (local copy) : {TRADING_PPO_DIR}")
    print(f"  Trading PPO timesteps    : {TRADING_PPO_TIMESTEPS:,}")
    print(f"  Stage2 reward overrides  : {list(STAGE2_REWARD_OVERRIDES.keys())}")
    print()

    if STAGE1_MODE == "saved_forecasts":
        print("Loading saved FTSE XGB + LSTM walk-forward forecasts...")
    elif DATA_SOURCE == "bloomberg":
        print("Retrain path: Loading Bloomberg panel...")
        _configure_bloomberg()
    else:
        print("Retrain path: Loading Yahoo US panel...")
    print()

    result = run_stacked_ensemble(TICKERS)

    n = len(result["tickers"])
    print(f"Stacked ensemble | Arch B | {n} tickers")
    print(f"Mode  : {result.get('stage1_mode', STAGE1_MODE)} | meta: {result['meta_method']}")
    print(f"Legs  : {', '.join(result['supervised_legs'])}")
    print(f"Tickers: {', '.join(result['tickers'][:12])}{'...' if n > 12 else ''}")
    print(
        f"OOS   : {result['test_start']}..{result['test_end']} ({result['test_rows']} rows)"
    )
    if result.get("blend_stats"):
        bs = result["blend_stats"]
        w = bs.get("mean_blend_weight_on_xgb", bs.get("w_xgb"))
        print(f"Blend weight on XGB: {w:.3f}")
    print("\nOut-of-sample ENSEMBLE metrics:")
    print(f"  MSE  : {result['mse']:.6f}")
    print(f"  MAE  : {result['mae']:.6f}")
    print(f"  RMSE : {result['rmse']:.6f}")
    print(f"  R²   : {result['r2']:.4f}")

    if result.get("architecture_b"):
        ab = result["architecture_b"]
        print("\nArchitecture B ranked book (for trading PPO timing):")
        print(f"  months={ab['months']} top_k={ab['top_k']} "
              f"avg_longs/month={ab['avg_long_candidates_per_month']:.1f}")

    print("\nSaved:")
    for k, v in result["paths"].items():
        if v:
            print(f"  {k}: {v}")

    print("\nLatest test month per ticker (sample):")
    print(result["preview_display"].head(20).to_string(index=False))

    if AUTO_START_TRADING_PPO:
        code = launch_trading_ppo()
        if code != 0:
            raise SystemExit(code)
        try:
            attr = build_attribution(out_dir=SAVE_DIR)
            print_summary(attr)
        except Exception as attr_err:
            print(f"\n[Attribution] post-Stage-2 refresh failed: {attr_err}")
    else:
        print(
            "\n[Stage 2] Skipped (AUTO_START_TRADING_PPO=False). "
            f'Set True in CONFIG or run: python "{TRADING_PPO_SCRIPT}"'
        )
        try:
            attr = build_attribution(out_dir=SAVE_DIR)
            print_summary(attr)
        except Exception as attr_err:
            print(f"\n[Attribution] failed: {attr_err}")
