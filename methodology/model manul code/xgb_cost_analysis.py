"""
Post-processing: Transaction cost analysis for the frozen Top-50 strategy.
Uses measured turnover + credible one-way cost assumptions.
Does not touch training / golden rule / test set.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# SETTINGS – edit these paths / values if needed
# ============================================================
PICKS_CSV = Path(__file__).resolve().parent / "saved_data" / "xgb_topk_picks_B_drop_bottom9_val.csv"

# Measured from your data (frozen no-filter val 2023)
GROSS_TOP50_EDGE = 0.0162      # +1.62% per month
GROSS_LONG_SHORT = 0.0260      # +2.60% per month (approx)
AVG_TURNOVER = 0.62            # 62% monthly turnover

# Credible one-way cost scenarios (basis points)
COST_SCENARIOS_BP = {
    "Optimistic (15 bp)": 15,
    "Base (25 bp)": 25,
    "Base (30 bp)": 30,
    "Conservative (40 bp)": 40,
    "Stress (50 bp)": 50,
}

# ============================================================
# FUNCTIONS
# ============================================================
def compute_turnover(picks: pd.DataFrame) -> float:
    """Average monthly turnover = average fraction of new names."""
    picks = picks.copy()
    picks["month"] = pd.to_datetime(picks["month"])
    months = sorted(picks["month"].unique())
    turnovers = []
    for i in range(1, len(months)):
        prev = set(picks.loc[picks["month"] == months[i - 1], "Ticker"])
        curr = set(picks.loc[picks["month"] == months[i], "Ticker"])
        new_names = len(curr - prev)
        turnovers.append(new_names / 50)
    return float(np.mean(turnovers)) if turnovers else np.nan


def net_edge(gross_edge: float, turnover: float, one_way_bp: float) -> float:
    """
    Simple round-trip cost model:
    Net = Gross - (Turnover × one-way cost × 2)
    """
    one_way = one_way_bp / 10000.0
    round_trip = turnover * one_way * 2
    return gross_edge - round_trip


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("=" * 70)
    print("TRANSACTION COST ANALYSIS – Frozen B_drop_bottom9 Top-50")
    print("=" * 70)

    if PICKS_CSV.exists():
        df = pd.read_csv(PICKS_CSV)
        measured_to = compute_turnover(df)
        print(f"Measured average monthly turnover : {measured_to:.1%}")
        turnover = measured_to
    else:
        print(f"Picks file not found – using stored turnover {AVG_TURNOVER:.1%}")
        turnover = AVG_TURNOVER

    print(f"Gross Top-50 edge                  : {GROSS_TOP50_EDGE:+.2%}")
    print(f"Gross Long-Short (approx)          : {GROSS_LONG_SHORT:+.2%}")
    print()

    print(f"{'Scenario':<22} {'1-way':>8} {'Round-trip':>12} {'Net Top-50':>12} {'Net LS*':>12}")
    print("-" * 70)

    for name, bp in COST_SCENARIOS_BP.items():
        net = net_edge(GROSS_TOP50_EDGE, turnover, bp)
        net_ls = net_edge(GROSS_LONG_SHORT, turnover, bp)
        rt = turnover * (bp / 10000) * 2
        print(f"{name:<22} {bp:>6} bp {rt:>11.2%} {net:>11.2%} {net_ls:>11.2%}")

    print()
    print("Notes:")
    print("• Round-trip cost = Turnover × one-way cost × 2")
    print("• LS* applies the same one-book TO formula (approx; shorts add more cost)")
    print("• UK Stamp Duty (50 bp on purchases) is one reason higher costs are realistic")
    print("• Primary model keeps LIQUIDITY_FILTER = False")
    print("• 10% and 20% liquidity cuts were already tested as robustness only")
    print("=" * 70)
