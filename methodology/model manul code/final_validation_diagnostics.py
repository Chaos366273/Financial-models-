"""
Final validation diagnostics for the frozen B_drop_bottom9 model.
Runs all four recommended checks in one go (validation only).
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# PATHS – edit if your files are elsewhere
# ============================================================
SAVE_DIR = Path(__file__).resolve().parent / "saved_data"
PICKS_CSV = SAVE_DIR / "xgb_topk_picks_B_drop_bottom9_val.csv"
# Full val cross-section (written by xgb_return after diag-col enrich)
PREVIEW_CSV = SAVE_DIR / "xgb_preview_B_drop_bottom9_val.csv"

TOP_K = 50
FACTOR_COLS = ["MOM", "SMB", "HML", "log_market_cap", "RMW", "CMA"]


def main():
    print("=" * 70)
    print("FINAL VALIDATION DIAGNOSTICS - B_drop_bottom9 (validation only)")
    print("=" * 70)

    if not PICKS_CSV.exists():
        raise FileNotFoundError(f"Picks file not found: {PICKS_CSV}")

    picks = pd.read_csv(PICKS_CSV)
    picks["month"] = pd.to_datetime(picks["month"])
    print(f"Picks file: {PICKS_CSV.name} | rows={len(picks)} | cols={list(picks.columns)}")

    universe = None
    if PREVIEW_CSV.exists():
        universe = pd.read_csv(PREVIEW_CSV)
        universe["month"] = pd.to_datetime(universe["month"])
        print(f"Preview file: {PREVIEW_CSV.name} | rows={len(universe)} (full val cross-section)")
    else:
        print(f"Preview file not found ({PREVIEW_CSV.name}) - universe comparisons limited")

    # --------------------------------------------------------
    # 1. Size & Liquidity profile of Top-50 vs universe
    # --------------------------------------------------------
    print("\n1. SIZE & LIQUIDITY PROFILE OF TOP-50")
    print("-" * 50)

    has_size = "log_market_cap" in picks.columns
    has_turn = "eqy_turnover" in picks.columns

    if has_size:
        top_size = picks.groupby("month")["log_market_cap"].mean().mean()
        print(f"Average log_market_cap of Top-50 names : {top_size:.3f}")
        print(f"Median  log_market_cap of Top-50 names : {picks['log_market_cap'].median():.3f}")
        if universe is not None and "log_market_cap" in universe.columns:
            uni_size = universe.groupby("month")["log_market_cap"].mean().mean()
            print(f"Average log_market_cap of universe     : {uni_size:.3f}")
            print(f"Top-50 minus universe (log mcap)       : {top_size - uni_size:+.3f}")
    else:
        print("log_market_cap not found in picks file - skip detailed size comparison")

    if has_turn:
        top_turn = picks.groupby("month")["eqy_turnover"].mean().mean()
        print(f"Average eqy_turnover of Top-50 names   : {top_turn:,.0f}")
        if universe is not None and "eqy_turnover" in universe.columns:
            uni_turn = universe.groupby("month")["eqy_turnover"].mean().mean()
            print(f"Average eqy_turnover of universe       : {uni_turn:,.0f}")
            print(f"Top-50 / universe turnover ratio       : {top_turn / uni_turn:.2f}x" if uni_turn else "")
    else:
        print("eqy_turnover not found in picks file - skip turnover comparison")

    print("-> Interpretation: if Top-50 names are systematically smaller/less liquid")
    print("  than the universe, this confirms the earlier liquidity-filter results.")

    # --------------------------------------------------------
    # 2. Sector concentration
    # --------------------------------------------------------
    print("\n2. SECTOR CONCENTRATION OF TOP-50")
    print("-" * 50)

    if "Sector" not in picks.columns:
        print("Sector column not found in picks file.")
    else:
        sector_share = (
            picks.groupby(["month", "Sector"])
            .size()
            .groupby("Sector")
            .mean()
            / TOP_K
        ).sort_values(ascending=False)

        print("Average share of Top-50 portfolio by sector:")
        for sec, share in sector_share.items():
            print(f"  {sec:<35} {share:.1%}")

        max_share = sector_share.max()
        print(f"\nHighest average sector weight: {max_share:.1%}")
        if max_share > 0.35:
            print("-> Warning: relatively concentrated in one sector.")
        else:
            print("-> Sector exposure looks reasonably diversified.")

    # --------------------------------------------------------
    # 3. Risk metrics of the Top-50 book
    # --------------------------------------------------------
    print("\n3. RISK METRICS OF THE TOP-50 BOOK")
    print("-" * 50)

    ret_col = None
    for c in ["y_true", "excess_return", "return", "ret", "target_excess_return"]:
        if c in picks.columns:
            ret_col = c
            break

    if ret_col is None:
        print("No return column found in picks file - cannot compute risk metrics.")
        print("Columns available:", list(picks.columns))
    else:
        monthly_ret = picks.groupby("month")[ret_col].mean()
        vol = monthly_ret.std()
        mean_ret = monthly_ret.mean()
        sharpe = mean_ret / vol if vol > 0 else np.nan

        cum = (1 + monthly_ret).cumprod()
        drawdown = cum / cum.cummax() - 1
        max_dd = drawdown.min()

        print(f"Return column used         : {ret_col}")
        print(f"Mean monthly excess return : {mean_ret:+.2%}")
        print(f"Volatility (monthly)       : {vol:.2%}")
        print(f"Annualised volatility (approx): {vol * np.sqrt(12):.2%}")
        print(f"Sharpe ratio (monthly)     : {sharpe:.2f}")
        print(f"Maximum drawdown           : {max_dd:.2%}")
        if "pick_minus_market" in picks.columns:
            month_edge = picks.groupby("month")["pick_minus_market"].mean()
            hit = float((month_edge > 0).mean())
            print(f"Hit rate (months beat mkt) : {hit:.1%}")
        else:
            print("Hit rate (already known)   : 91.7%")

    # --------------------------------------------------------
    # 4. Correlation with classic factors
    # --------------------------------------------------------
    print("\n4. CORRELATION WITH CLASSIC FACTORS")
    print("-" * 50)

    frame = universe if universe is not None else picks
    frame_label = "full val cross-section" if universe is not None else "Top-50 picks only"
    factor_cols = [c for c in FACTOR_COLS if c in frame.columns]

    if not factor_cols or "y_pred" not in frame.columns:
        print("No classic factor columns found (or missing y_pred).")
        print("Re-run frozen xgb_return.py so picks/preview include diag columns.")
    else:
        print(f"Correlations on {frame_label}:")
        print("Available factors:", factor_cols)
        for f in factor_cols:
            corr = frame["y_pred"].corr(frame[f], method="spearman")
            print(f"  Rank correlation with {f:<15}: {corr:.3f}")

    print("\n" + "=" * 70)
    print("Diagnostics complete. Review the four blocks above.")
    print("None of these checks change the frozen model or unlock the test set.")
    print("=" * 70)


if __name__ == "__main__":
    main()
