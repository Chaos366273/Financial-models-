"""
Post-processing: Transaction costs + simple capacity estimate
for the frozen Top-50 strategy (B_drop_bottom9).
Does not touch training / golden rule / test set.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# SETTINGS
# ============================================================
PICKS_CSV = Path(__file__).resolve().parent / "saved_data" / "xgb_topk_picks_B_drop_bottom9_val.csv"

GROSS_TOP50_EDGE = 0.0162          # +1.62% per month (gross)
AVG_TURNOVER = 0.62                # 62% monthly turnover

# Cost scenarios (one-way, basis points)
COST_SCENARIOS_BP = {
    "Optimistic (15 bp)": 15,
    "Base (25 bp)": 25,
    "Base (30 bp)": 30,
    "Conservative (40 bp)": 40,
    "Stress (50 bp)": 50,
}

# ---- Simple capacity assumptions (illustrative) ----
# We assume market impact rises with the square-root of participation.
# These are deliberately conservative and transparent for a dissertation.
BASE_ONE_WAY_BP = 25               # starting one-way cost at very small size
IMPACT_COEFF_BP = 8                # additional bp when participation doubles
MAX_PARTICIPATION = 0.10           # we never want to be >10% of ADV (rule of thumb)
ASSUMED_AVG_ADV_GBP = 5_000_000    # illustrative average daily volume per stock (£5m)
                                   # (replace with real ADV if you have it)

# ============================================================
# FUNCTIONS
# ============================================================
def compute_turnover(picks: pd.DataFrame) -> float:
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
    one_way = one_way_bp / 10000.0
    round_trip = turnover * one_way * 2
    return gross_edge - round_trip


def capacity_table(
    gross_edge: float,
    turnover: float,
    base_bp: float = 25,
    impact_coeff: float = 8,
    max_part: float = 0.10,
    adv: float = 5_000_000,
):
    """
    Very simple illustrative capacity model.
    We increase AUM and let one-way cost rise with sqrt(participation).
    """
    results = []
    # Test a range of strategy AUM
    for aum_m in [10, 25, 50, 100, 200, 500, 1000]:  # £ million
        aum = aum_m * 1_000_000
        # Approximate average trade size per rebalance (half the portfolio turns over)
        trade_value = aum * turnover / 2          # rough
        participation = trade_value / (50 * adv)  # very rough average participation

        # Square-root market impact (common simple form)
        extra_bp = impact_coeff * np.sqrt(max(participation, 0.001) / 0.01)
        one_way_bp = base_bp + extra_bp

        net = net_edge(gross_edge, turnover, one_way_bp)
        results.append({
            "AUM (£m)": aum_m,
            "Approx participation": f"{participation:.1%}",
            "Implied 1-way (bp)": round(one_way_bp, 1),
            "Net monthly edge": f"{net:+.2%}",
            "Still positive?": "Yes" if net > 0 else "No",
            "Above 10% ADV rule?": "Yes" if participation > max_part else "No",
        })
    return pd.DataFrame(results)


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("=" * 70)
    print("TRANSACTION COSTS + SIMPLE CAPACITY ESTIMATE")
    print("Frozen model: B_drop_bottom9 | Top-50 | Monthly rebalance")
    print("=" * 70)

    # Turnover
    if PICKS_CSV.exists():
        df = pd.read_csv(PICKS_CSV)
        turnover = compute_turnover(df)
        print(f"Measured monthly turnover          : {turnover:.1%}")
    else:
        turnover = AVG_TURNOVER
        print(f"Using stored turnover              : {turnover:.1%}")

    print(f"Gross Top-50 edge                  : {GROSS_TOP50_EDGE:+.2%}")
    print()

    # ---- Cost scenarios ----
    print("1. Net edge under fixed cost assumptions")
    print(f"{'Scenario':<22} {'1-way':>8} {'Round-trip':>12} {'Net Top-50':>12}")
    print("-" * 58)
    for name, bp in COST_SCENARIOS_BP.items():
        net = net_edge(GROSS_TOP50_EDGE, turnover, bp)
        rt = turnover * (bp / 10000) * 2
        print(f"{name:<22} {bp:>6} bp {rt:>11.2%} {net:>11.2%}")

    print()
    print("2. Illustrative capacity (square-root impact model)")
    print("   (Assumptions are transparent and conservative – replace ADV if you have real data)")
    print()
    cap = capacity_table(
        GROSS_TOP50_EDGE,
        turnover,
        base_bp=BASE_ONE_WAY_BP,
        impact_coeff=IMPACT_COEFF_BP,
        max_part=MAX_PARTICIPATION,
        adv=ASSUMED_AVG_ADV_GBP,
    )
    print(cap.to_string(index=False))

    print()
    print("Interpretation notes for the dissertation:")
    print("• Costs rise with the square-root of participation (standard simple form).")
    print("• Strategy remains positive at moderate AUM under base-case costs.")
    print("• High turnover (62%) and the earlier liquidity-filter results both")
    print("  indicate that capacity is limited – the edge is partly carried by")
    print("  less liquid names.")
    print("• Primary model keeps LIQUIDITY_FILTER = False; 10%/20% cuts are")
    print("  robustness checks only.")
    print("=" * 70)
