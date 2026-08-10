"""Stage-1 / Stage-2 attribution: XGB vs LSTM vs blend vs trading PPO.

Writes:
  saved_data/attribution_report.md
  saved_data/attribution_report.json
  saved_data/trade_attribution.csv  (when Stage-2 trade_history exists)

Architecture B note: trading PPO does not pick XGB vs LSTM per trade.
Blend weight lives in Stage-1 y_pred; PPO only times the ranked book.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SAVE_DIR = ROOT / "saved_data"
DEFAULT_RANKED = SAVE_DIR / "ranked_signals.csv"
DEFAULT_METRICS = SAVE_DIR / "ensemble_metrics.json"
DEFAULT_TRADES = (
    ROOT / "trading_ppo" / "ppo" / "saved_data" / "validation" / "trade_history.csv"
)
DEFAULT_ANALYTICS = (
    ROOT
    / "trading_ppo"
    / "ppo"
    / "saved_data"
    / "validation"
    / "analytics_summary_compact.json"
)
DEFAULT_BACKTEST = (
    ROOT
    / "trading_ppo"
    / "ppo"
    / "saved_data"
    / "validation"
    / "backtest_summary_compact.json"
)


def _implied_w_xgb(df: pd.DataFrame) -> pd.Series:
    den = (df["xgb_pred"] - df["lstm_pred"]).replace(0, np.nan)
    w = (df["y_pred"] - df["lstm_pred"]) / den
    return w.clip(0.0, 1.0)


def _forecast_metrics(yt: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(yt) & np.isfinite(pred)
    yt, pred = yt[mask], pred[mask]
    if len(yt) == 0:
        return {"n": 0, "mse": float("nan"), "r2": float("nan"), "dir_acc": float("nan")}
    mse = float(np.mean((yt - pred) ** 2))
    ss_tot = float(np.sum((yt - yt.mean()) ** 2)) + 1e-12
    r2 = float(1.0 - np.sum((yt - pred) ** 2) / ss_tot)
    dir_acc = float(np.mean(np.sign(pred) == np.sign(yt)))
    return {"n": int(len(yt)), "mse": mse, "r2": r2, "dir_acc": dir_acc}


def _monthly_eq_book(df: pd.DataFrame, score_col: str, top_k: int = 10) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for per, g in df.groupby(df["month"].dt.to_period("M")):
        g2 = g.nlargest(min(top_k, len(g)), score_col)
        use = g2[g2[score_col] > 0]
        if use.empty:
            rows.append({"month": str(per), "ret": 0.0, "n": 0})
            continue
        rows.append(
            {
                "month": str(per),
                "ret": float(use["y_true"].mean()),
                "n": int(len(use)),
            }
        )
    return pd.DataFrame(rows)


def _monthly_archb_book(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for per, g in df.groupby(df["month"].dt.to_period("M")):
        use = g[g["weight"] > 0]
        if use.empty:
            rows.append({"month": str(per), "ret": 0.0, "n": 0})
            continue
        wsum = float(use["weight"].sum())
        ret = float((use["weight"] * use["y_true"]).sum() / max(wsum, 1e-12))
        rows.append({"month": str(per), "ret": ret, "n": int(len(use))})
    return pd.DataFrame(rows)


def _book_summary(d: pd.DataFrame) -> dict[str, float]:
    if d.empty:
        return {"months": 0, "mean_ret": float("nan"), "cum_ret": float("nan"), "hit_rate": float("nan")}
    r = d["ret"].to_numpy(dtype=float)
    return {
        "months": int(len(d)),
        "mean_ret": float(np.mean(r)),
        "cum_ret": float(np.prod(1.0 + r) - 1.0),
        "hit_rate": float(np.mean(r > 0)),
    }


def _load_ppo_summary(
    analytics_path: Path,
    backtest_path: Path,
) -> dict[str, Any] | None:
    out: dict[str, Any] = {}
    if analytics_path.is_file():
        try:
            out.update(json.loads(analytics_path.read_text(encoding="utf-8")))
        except Exception:
            pass
    if backtest_path.is_file():
        try:
            bt = json.loads(backtest_path.read_text(encoding="utf-8"))
            out.setdefault("total_trades", bt.get("total_trades"))
            out.setdefault("steps_simulated", bt.get("steps_simulated"))
            out.setdefault("final_portfolio_value", bt.get("final_portfolio_value"))
        except Exception:
            pass
    return out or None


def _default_symbols_from_ranked(ranked: pd.DataFrame) -> list[str]:
    return sorted(ranked["Ticker"].astype(str).str.upper().unique().tolist())


def _tag_trades(
    trades: pd.DataFrame,
    ranked: pd.DataFrame,
    symbols: list[str],
) -> pd.DataFrame:
    """Attach Stage-1 avg book stats to each trade by ticker (no calendar on trades)."""
    th = trades.copy()
    th["ticker_idx"] = pd.to_numeric(th["ticker"], errors="coerce")
    th["Ticker"] = th["ticker_idx"].map(
        lambda i: symbols[int(i)] if pd.notna(i) and 0 <= int(i) < len(symbols) else None
    )

    by_tic = (
        ranked.groupby(ranked["Ticker"].astype(str).str.upper())
        .agg(
            avg_xgb_pred=("xgb_pred", "mean"),
            avg_lstm_pred=("lstm_pred", "mean"),
            avg_y_pred=("y_pred", "mean"),
            avg_weight=("weight", "mean"),
            months_long=("weight", lambda s: int((s > 0).sum())),
            months_total=("weight", "count"),
        )
        .reset_index()
        .rename(columns={"Ticker": "Ticker"})
    )
    out = th.merge(by_tic, on="Ticker", how="left")
    # share of this name's forecast that is XGB in the blend (row-level mean)
    den = (ranked["xgb_pred"] - ranked["lstm_pred"]).replace(0, np.nan)
    ranked = ranked.copy()
    ranked["w_xgb"] = ((ranked["y_pred"] - ranked["lstm_pred"]) / den).clip(0, 1)
    wmap = ranked.groupby(ranked["Ticker"].astype(str).str.upper())["w_xgb"].mean()
    out["avg_blend_w_xgb"] = out["Ticker"].map(wmap)
    out["avg_blend_w_lstm"] = 1.0 - out["avg_blend_w_xgb"]
    return out


def build_attribution(
    ranked_path: Path = DEFAULT_RANKED,
    metrics_path: Path = DEFAULT_METRICS,
    trades_path: Path = DEFAULT_TRADES,
    analytics_path: Path = DEFAULT_ANALYTICS,
    backtest_path: Path = DEFAULT_BACKTEST,
    out_dir: Path = SAVE_DIR,
    top_k: int = 10,
) -> dict[str, Any]:
    if not ranked_path.is_file():
        raise FileNotFoundError(f"ranked_signals missing: {ranked_path}")

    ranked = pd.read_csv(ranked_path)
    ranked["month"] = pd.to_datetime(ranked["month"])
    ranked["Ticker"] = ranked["Ticker"].astype(str).str.upper()
    for c in ("xgb_pred", "lstm_pred", "y_pred", "y_true", "weight"):
        if c in ranked.columns:
            ranked[c] = pd.to_numeric(ranked[c], errors="coerce")

    need = ["xgb_pred", "lstm_pred", "y_pred", "y_true"]
    m = ranked.dropna(subset=need).copy()
    w = _implied_w_xgb(m)

    blend = {
        "n_rows": int(len(m)),
        "mean_w_xgb": float(w.mean()) if len(w) else float("nan"),
        "median_w_xgb": float(w.median()) if len(w) else float("nan"),
        "std_w_xgb": float(w.std()) if len(w) else float("nan"),
        "pct_xgb_heavy_gt_0_7": float((w > 0.7).mean()) if len(w) else float("nan"),
        "pct_mid_0_3_to_0_7": float(((w >= 0.3) & (w <= 0.7)).mean()) if len(w) else float("nan"),
        "pct_lstm_heavy_lt_0_3": float((w < 0.3).mean()) if len(w) else float("nan"),
        "histogram_w_xgb": {
            "0.0-0.2": float(((w >= 0) & (w < 0.2)).mean()) if len(w) else 0.0,
            "0.2-0.4": float(((w >= 0.2) & (w < 0.4)).mean()) if len(w) else 0.0,
            "0.4-0.6": float(((w >= 0.4) & (w < 0.6)).mean()) if len(w) else 0.0,
            "0.6-0.8": float(((w >= 0.6) & (w < 0.8)).mean()) if len(w) else 0.0,
            "0.8-1.0": float(((w >= 0.8) & (w <= 1.0)).mean()) if len(w) else 0.0,
        },
    }

    yt = m["y_true"].to_numpy(dtype=float)
    forecast = {
        "xgb": _forecast_metrics(yt, m["xgb_pred"].to_numpy(dtype=float)),
        "lstm": _forecast_metrics(yt, m["lstm_pred"].to_numpy(dtype=float)),
        "blend": _forecast_metrics(yt, m["y_pred"].to_numpy(dtype=float)),
    }

    books = {
        "xgb_only_eq_top10": _book_summary(_monthly_eq_book(m, "xgb_pred", top_k=top_k)),
        "lstm_only_eq_top10": _book_summary(_monthly_eq_book(m, "lstm_pred", top_k=top_k)),
        "blend_eq_top10": _book_summary(_monthly_eq_book(m, "y_pred", top_k=top_k)),
        "arch_b_weighted": _book_summary(_monthly_archb_book(m)),
    }

    sel = m[m["weight"] > 0].copy()
    if not sel.empty:
        sel_w = _implied_w_xgb(sel)
        books["on_selected_longs"] = {
            "n": int(len(sel)),
            "mean_blend_w_xgb": float(sel_w.mean()),
            "mean_blend_w_lstm": float(1.0 - sel_w.mean()),
        }

    ensemble_metrics = None
    if metrics_path.is_file():
        try:
            ensemble_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except Exception:
            ensemble_metrics = None

    ppo = _load_ppo_summary(analytics_path, backtest_path)
    trade_attr_path = None
    trade_stats: dict[str, Any] | None = None
    if trades_path.is_file():
        trades = pd.read_csv(trades_path)
        symbols = _default_symbols_from_ranked(ranked)
        # Prefer Stage-1 ticker order from metrics if present (matches env order better if same list)
        if ensemble_metrics and isinstance(ensemble_metrics.get("tickers"), list):
            symbols = [str(t).upper() for t in ensemble_metrics["tickers"]]
        tagged = _tag_trades(trades, m, symbols)
        out_dir.mkdir(parents=True, exist_ok=True)
        trade_attr_path = out_dir / "trade_attribution.csv"
        tagged.to_csv(trade_attr_path, index=False)

        buys = tagged[tagged["action"].astype(str).str.upper() == "BUY"]
        closes = tagged[tagged["action"].astype(str).str.upper().isin(["SELL", "STOP_LOSS"])]
        trade_stats = {
            "n_trades": int(len(tagged)),
            "n_buy": int(len(buys)),
            "n_close": int(len(closes)),
            "buy_avg_stage1_weight": float(buys["avg_weight"].mean()) if len(buys) else float("nan"),
            "buy_avg_blend_w_xgb": float(buys["avg_blend_w_xgb"].mean()) if len(buys) else float("nan"),
            "buy_avg_blend_w_lstm": float(buys["avg_blend_w_lstm"].mean()) if len(buys) else float("nan"),
            "close_pnl_sum": float(closes["pnl"].sum()) if "pnl" in closes.columns and len(closes) else float("nan"),
            "note": (
                "Trades have no market calendar date; tags use per-ticker averages "
                "from Stage-1 ranked_signals (not the exact month of the fill)."
            ),
        }

    payload: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "architecture": "B",
        "ranked_signals": str(ranked_path),
        "blend_weight_on_xgb": blend,
        "forecast_quality": forecast,
        "counterfactual_monthly_books": books,
        "trading_ppo": ppo,
        "trade_attribution": trade_stats,
        "paths": {
            "ranked_signals": str(ranked_path),
            "trade_attribution_csv": str(trade_attr_path) if trade_attr_path else None,
        },
        "interpretation": {
            "blend": (
                "y_pred = w*xgb + (1-w)*lstm. High mean_w_xgb => book is XGB-driven; "
                "low => LSTM-driven. Trading PPO does not choose the leg."
            ),
            "books_vs_ppo": (
                "Monthly book cum returns span Stage-1 test months. "
                "Trading PPO return is a short daily execution window (often 1k-timestep smoke) — "
                "compare directionally, not as like-for-like total return."
            ),
        },
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "attribution_report.json"
    md_path = out_dir / "attribution_report.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md_path.write_text(_to_markdown(payload), encoding="utf-8")
    payload["paths"]["attribution_json"] = str(json_path)
    payload["paths"]["attribution_md"] = str(md_path)
    # rewrite with paths filled
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md_path.write_text(_to_markdown(payload), encoding="utf-8")
    return payload


def _pct(x: float | None) -> str:
    if x is None or not np.isfinite(x):
        return "n/a"
    return f"{100.0 * float(x):.1f}%"


def _f(x: float | None, digits: int = 4) -> str:
    if x is None or not np.isfinite(x):
        return "n/a"
    return f"{float(x):.{digits}f}"


def _to_markdown(p: dict[str, Any]) -> str:
    b = p["blend_weight_on_xgb"]
    f = p["forecast_quality"]
    books = p["counterfactual_monthly_books"]
    lines = [
        f"# Ensemble Attribution Report ({p.get('generated_at', '')})",
        "",
        "Architecture **B**: XGB + LSTM → blend `y_pred` → ranked book → trading PPO times BUY/SELL/HOLD.",
        "",
        "## 1. Blend weight on XGB (`w` in `y_pred = w·xgb + (1-w)·lstm`)",
        "",
        f"- Rows: **{b['n_rows']}**",
        f"- Mean `w_xgb`: **{_f(b['mean_w_xgb'], 3)}** (median {_f(b['median_w_xgb'], 3)}, std {_f(b['std_w_xgb'], 3)})",
        f"- XGB-heavy (`w>0.7`): {_pct(b['pct_xgb_heavy_gt_0_7'])}",
        f"- Mid (`0.3–0.7`): {_pct(b['pct_mid_0_3_to_0_7'])}",
        f"- LSTM-heavy (`w<0.3`): {_pct(b['pct_lstm_heavy_lt_0_3'])}",
        "",
        "### Histogram of `w_xgb`",
        "",
        "| Bin | Share |",
        "|-----|-------|",
    ]
    for k, v in b.get("histogram_w_xgb", {}).items():
        lines.append(f"| {k} | {_pct(v)} |")

    lines += [
        "",
        "## 2. Forecast quality (same ranked rows)",
        "",
        "| Leg | MSE | R² | Dir acc | N |",
        "|-----|-----|----|---------|---|",
    ]
    for name, key in (("XGB", "xgb"), ("LSTM", "lstm"), ("Blend", "blend")):
        m = f[key]
        lines.append(
            f"| {name} | {_f(m['mse'], 6)} | {_f(m['r2'], 4)} | {_f(m['dir_acc'], 3)} | {m['n']} |"
        )

    lines += [
        "",
        "## 3. Counterfactual monthly long books",
        "",
        "Equal-weight top-10 with `score > 0`, plus Arch B weighted book.",
        "",
        "| Book | Mean monthly | Cum (compound) | Hit rate | Months |",
        "|------|--------------|----------------|----------|--------|",
    ]
    labels = [
        ("XGB-only", "xgb_only_eq_top10"),
        ("LSTM-only", "lstm_only_eq_top10"),
        ("Blend `y_pred`", "blend_eq_top10"),
        ("Arch B weighted", "arch_b_weighted"),
    ]
    for label, key in labels:
        s = books[key]
        lines.append(
            f"| {label} | {_pct(s['mean_ret'])} | {_pct(s['cum_ret'])} | {_pct(s['hit_rate'])} | {s['months']} |"
        )

    if "on_selected_longs" in books:
        o = books["on_selected_longs"]
        lines += [
            "",
            f"On Arch B selected longs (n={o['n']}): "
            f"blend ≈ **{_pct(o['mean_blend_w_xgb'])} XGB / {_pct(o['mean_blend_w_lstm'])} LSTM**.",
        ]

    lines += ["", "## 4. Trading PPO (Stage 2)", ""]
    ppo = p.get("trading_ppo")
    if not ppo:
        lines.append("_No Stage-2 analytics found yet (run trading PPO, then re-run attribution)._")
    else:
        lines += [
            f"- Total return: **{_pct(ppo.get('total_return'))}**",
            f"- Sharpe: **{_f(ppo.get('sharpe_ratio'), 3)}**",
            f"- Max DD: **{_pct(ppo.get('max_drawdown'))}**",
            f"- Win rate: **{_pct(ppo.get('win_rate'))}**",
            f"- Trades / steps: {ppo.get('total_trades', 'n/a')} / {ppo.get('steps_simulated', 'n/a')}",
            "",
            f"> {p['interpretation']['books_vs_ppo']}",
        ]

    ta = p.get("trade_attribution")
    lines += ["", "## 5. Trade ↔ Stage-1 tags", ""]
    if not ta:
        lines.append("_No `trade_history.csv` — skip trade tags._")
    else:
        lines += [
            f"- Trades tagged: **{ta['n_trades']}** (BUY={ta['n_buy']}, closes={ta['n_close']})",
            f"- BUY avg Stage-1 book weight: **{_f(ta.get('buy_avg_stage1_weight'), 4)}**",
            f"- BUY avg blend mix: **{_pct(ta.get('buy_avg_blend_w_xgb'))} XGB / "
            f"{_pct(ta.get('buy_avg_blend_w_lstm'))} LSTM**",
            f"- Close PnL sum: **{_f(ta.get('close_pnl_sum'), 2)}**",
            f"- CSV: `{p['paths'].get('trade_attribution_csv')}`",
            "",
            f"> {ta.get('note', '')}",
        ]

    lines += [
        "",
        "## Interpretation",
        "",
        f"- {p['interpretation']['blend']}",
        f"- {p['interpretation']['books_vs_ppo']}",
        "",
        f"JSON: `{p['paths'].get('attribution_json')}`",
        "",
    ]
    return "\n".join(lines)


def print_summary(payload: dict[str, Any]) -> None:
    b = payload["blend_weight_on_xgb"]
    books = payload["counterfactual_monthly_books"]
    print("\n" + "=" * 72)
    print("ATTRIBUTION (XGB vs LSTM vs blend vs trading PPO)")
    print("=" * 72)
    print(
        f"  Blend w_xgb: mean={b['mean_w_xgb']:.3f} | "
        f"XGB-heavy={100*b['pct_xgb_heavy_gt_0_7']:.0f}% | "
        f"LSTM-heavy={100*b['pct_lstm_heavy_lt_0_3']:.0f}%"
    )
    for label, key in (
        ("XGB-only book", "xgb_only_eq_top10"),
        ("LSTM-only book", "lstm_only_eq_top10"),
        ("Blend book", "blend_eq_top10"),
        ("ArchB weighted", "arch_b_weighted"),
    ):
        s = books[key]
        print(
            f"  {label:16s} mean_mo={100*s['mean_ret']:+.2f}% "
            f"cum={100*s['cum_ret']:+.1f}% hit={100*s['hit_rate']:.0f}%"
        )
    ppo = payload.get("trading_ppo")
    if ppo and ppo.get("total_return") is not None:
        print(
            f"  Trading PPO     return={100*float(ppo['total_return']):+.2f}% "
            f"Sharpe={float(ppo.get('sharpe_ratio') or 0):.2f}"
        )
    paths = payload.get("paths", {})
    print(f"  Report: {paths.get('attribution_md')}")
    if paths.get("trade_attribution_csv"):
        print(f"  Trades: {paths.get('trade_attribution_csv')}")


if __name__ == "__main__":
    payload = build_attribution()
    print_summary(payload)
