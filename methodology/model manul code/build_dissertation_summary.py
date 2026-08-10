"""
Build a clean, dissertation-ready results summary for frozen XGB B_drop_bottom9.

Writes:
  saved_data/XGB_DISSERTATION_RESULTS_SUMMARY.csv
  saved_data/XGB_DISSERTATION_RESULTS_SUMMARY.md

Re-run after --final-test to fill the Test column from the tracker / lock file.
Does not train models or unlock the test set.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
SAVE = ROOT / "saved_data"
FINAL_TEST_DIR = ROOT / "final test results" / "xgboost"
# Prefer final-test folder once unlocked; else validation saved_data
TRACKER = (
    FINAL_TEST_DIR / "xgb_results_tracker.csv"
    if (FINAL_TEST_DIR / "xgb_results_tracker.csv").is_file()
    else SAVE / "xgb_results_tracker.csv"
)
LOCK = (
    FINAL_TEST_DIR / "GOLDEN_RULE_test_already_scored.txt"
    if (FINAL_TEST_DIR / "GOLDEN_RULE_test_already_scored.txt").is_file()
    else SAVE / "GOLDEN_RULE_test_already_scored.txt"
)
OUT_CSV = SAVE / "XGB_DISSERTATION_RESULTS_SUMMARY.csv"
OUT_MD = SAVE / "XGB_DISSERTATION_RESULTS_SUMMARY.md"
# Mirror summary into final-test folder after unlock
OUT_CSV_FINAL = FINAL_TEST_DIR / "XGB_DISSERTATION_RESULTS_SUMMARY.csv"
OUT_MD_FINAL = FINAL_TEST_DIR / "XGB_DISSERTATION_RESULTS_SUMMARY.md"


def _fmt_pct(x, digits=2):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "—"
    return f"{100.0 * float(x):.{digits}f}%"


def _fmt_num(x, digits=4):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "—"
    return f"{float(x):.{digits}f}"


def _latest_test_from_tracker() -> dict:
    if not TRACKER.is_file():
        return {}
    df = pd.read_csv(TRACKER)
    test = df[df["evaluated_on"].astype(str).str.lower() == "test"]
    if test.empty:
        return {}
    row = test.iloc[-1]
    return {
        "test_ic": row.get("test_ic"),
        "test_ls": row.get("test_ls"),
        "test_topk_edge": row.get("test_topk_edge"),
        "test_rmse": row.get("test_rmse"),
        "run_id": row.get("run_id"),
    }


def build_tables() -> tuple[list[dict], list[dict], list[dict]]:
    """Return (main_metrics, robustness, costs) row dicts."""
    test = _latest_test_from_tracker()
    test_ready = bool(test) and LOCK.is_file()

    # Frozen validation baseline (agreed numbers)
    val = {
        "ic": 0.0444,
        "ls": 0.0260,
        "topk_edge": 0.0162,
        "topk_hit": 0.9167,
        "turnover": 0.62,
        "rmse": 0.1011,
    }

    main = [
        {
            "Metric": "Spearman IC (monthly mean)",
            "Validation (2023)": _fmt_num(val["ic"]),
            "Test (2024–2025)": _fmt_num(test.get("test_ic")) if test_ready else "Locked / not yet run",
        },
        {
            "Metric": "Long–short (top–bottom 10%)",
            "Validation (2023)": _fmt_pct(val["ls"]),
            "Test (2024–2025)": _fmt_pct(test.get("test_ls")) if test_ready else "Locked / not yet run",
        },
        {
            "Metric": "Top-50 edge vs equal-weight market",
            "Validation (2023)": _fmt_pct(val["topk_edge"]),
            "Test (2024–2025)": _fmt_pct(test.get("test_topk_edge")) if test_ready else "Locked / not yet run",
        },
        {
            "Metric": "Top-50 hit rate (months beat market)",
            "Validation (2023)": _fmt_pct(val["topk_hit"], 1),
            "Test (2024–2025)": "See test log / diagnostics" if test_ready else "Locked / not yet run",
        },
        {
            "Metric": "Top-50 avg monthly turnover",
            "Validation (2023)": _fmt_pct(val["turnover"], 0),
            "Test (2024–2025)": "Recompute from *_test.csv after unlock" if test_ready else "Locked / not yet run",
        },
        {
            "Metric": "RMSE (level)",
            "Validation (2023)": _fmt_num(val["rmse"]),
            "Test (2024–2025)": _fmt_num(test.get("test_rmse")) if test_ready else "Locked / not yet run",
        },
    ]

    robustness = [
        {"Check": "Feature set", "Spec / result": "B_drop_bottom9 (19 feats from h1_clean); A_top16 secondary only"},
        {"Check": "Hyper-parameters", "Spec / result": "depth=4, lr=0.03, min_child_weight=8, trees=178"},
        {"Check": "Target (frozen)", "Spec / result": "Sector-neutral excess return; winsor 1%/99%"},
        {"Check": "Exp1 market-neutral Y only", "Spec / result": "IC 0.0503; Top-50 edge 0.99%; hit 66.7% → reject for book"},
        {"Check": "Exp2 tighter winsor 0.5%/99.5%", "Spec / result": "IC noisier; imp. stability weaker → keep 1%/99%"},
        {"Check": "Exp3 no demean", "Spec / result": "IC 0.0236; Top-50 edge ~0; hit 41.7% → demeaning required"},
        {"Check": "Liquidity (primary)", "Spec / result": "LIQUIDITY_FILTER=False (full universe)"},
        {"Check": "Exp4 drop bottom 10%", "Spec / result": "IC 0.0338; Top-50 edge 0.81%; hit 66.7% (robustness only)"},
        {"Check": "Exp5 drop bottom 20%", "Spec / result": "IC 0.0299; Top-50 edge 0.47%; hit 50.0% (robustness only)"},
        {"Check": "Top-50 size/liquidity (val)", "Spec / result": "Smaller & less liquid than universe (log mcap −0.62; turnover 0.50×)"},
        {"Check": "Sector concentration (val)", "Spec / result": "Max avg sector weight Real Estate 20.2% (diversified)"},
        {"Check": "Top-50 risk (val)", "Spec / result": "Mean +1.46%/mo; vol 5.34%; Sharpe 0.27; max DD −7.9%"},
        {"Check": "Factor corr of y_pred (val)", "Spec / result": "MOM +0.21; SMB −0.19; HML −0.14; log mcap ~0"},
    ]

    # Net edge = gross - turnover * 2 * one_way
    gross = val["topk_edge"]
    to = val["turnover"]
    costs = []
    for name, bp in [
        ("Optimistic", 15),
        ("Base", 25),
        ("Base", 30),
        ("Conservative", 40),
        ("Stress", 50),
    ]:
        rt = to * (bp / 10000.0) * 2
        net = gross - rt
        costs.append({
            "Scenario": f"{name} ({bp} bp one-way)",
            "Round-trip cost / month": _fmt_pct(rt),
            "Net Top-50 edge / month": _fmt_pct(net),
        })

    return main, robustness, costs, test_ready, test


def write_outputs() -> None:
    SAVE.mkdir(parents=True, exist_ok=True)
    main, robustness, costs, test_ready, test = build_tables()

    # Flat CSV: three sections stacked with a section column
    flat = []
    for r in main:
        flat.append({"Section": "Main metrics", **r, "Check": "", "Spec / result": "", "Scenario": "", "Round-trip cost / month": "", "Net Top-50 edge / month": ""})
    # Cleaner: write three CSVs? User asked one table - write one MD with three tables + one CSV for main metrics only, plus robustness CSV.

    main_df = pd.DataFrame(main)
    rob_df = pd.DataFrame(robustness)
    cost_df = pd.DataFrame(costs)

    main_df.to_csv(OUT_CSV, index=False)
    rob_df.to_csv(SAVE / "XGB_DISSERTATION_ROBUSTNESS_SUMMARY.csv", index=False)
    cost_df.to_csv(SAVE / "XGB_DISSERTATION_COST_SUMMARY.csv", index=False)

    lines = [
        "# XGBoost dissertation results summary",
        "",
        "**Model:** B_drop_bottom9 (19 features from h1_clean)  ",
        "**Hyper-parameters:** depth=4, learning_rate=0.03, min_child_weight=8, trees=178  ",
        "**Target:** Sector-neutral excess return, winsorised at 1%/99%  ",
        "**Liquidity (primary):** no filter (`LIQUIDITY_FILTER=False`)  ",
        f"**Test status:** {'UNLOCKED — scored once' if test_ready else 'LOCKED (run `run_xgb.bat --final-test` once when ready)'}  ",
        "",
        "## Table 1 — Main ranking / portfolio metrics",
        "",
        main_df.to_markdown(index=False),
        "",
        "## Table 2 — Validation robustness (do not retune on these)",
        "",
        rob_df.to_markdown(index=False),
        "",
        "## Table 3 — Transaction costs (val Top-50; turnover 62%)",
        "",
        cost_df.to_markdown(index=False),
        "",
        "### Capacity note (illustrative)",
        "Under a simple square-root impact model (£5m ADV assumption), practical AUM is roughly",
        "**£50–100m** before the 10% ADV participation rule is breached. See `cost_and_capacity_analysis.py`.",
        "",
        "### Files",
        f"- Main metrics CSV: `{OUT_CSV.name}`",
        "- Robustness CSV: `XGB_DISSERTATION_ROBUSTNESS_SUMMARY.csv`",
        "- Cost CSV: `XGB_DISSERTATION_COST_SUMMARY.csv`",
        "- Tracker: `xgb_results_tracker.csv`",
        "- Frozen spec: `xgb_frozen_spec.json`",
        "",
    ]
    if test_ready:
        lines.extend([
            "### Test unlock reference",
            f"- Tracker run_id: {test.get('run_id')}",
            f"- Lock file: `{LOCK.name}`",
            "",
        ])

    md_text = "\n".join(lines)
    OUT_MD.write_text(md_text, encoding="utf-8")
    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {SAVE / 'XGB_DISSERTATION_ROBUSTNESS_SUMMARY.csv'}")
    print(f"Wrote {SAVE / 'XGB_DISSERTATION_COST_SUMMARY.csv'}")
    print(f"Wrote {OUT_MD}")
    if test_ready:
        FINAL_TEST_DIR.mkdir(parents=True, exist_ok=True)
        main_df.to_csv(OUT_CSV_FINAL, index=False)
        OUT_MD_FINAL.write_text(md_text, encoding="utf-8")
        print(f"Also wrote summaries into final-test folder: {FINAL_TEST_DIR}")
    print()
    print(main_df.to_string(index=False))


if __name__ == "__main__":
    # Prefer tabulate for markdown; fall back if missing
    try:
        import tabulate  # noqa: F401
    except ImportError:
        def _to_markdown(self, index=False):  # type: ignore
            cols = list(self.columns)
            header = "| " + " | ".join(cols) + " |"
            sep = "| " + " | ".join("---" for _ in cols) + " |"
            rows = []
            for _, row in self.iterrows():
                rows.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
            return "\n".join([header, sep, *rows])

        pd.DataFrame.to_markdown = _to_markdown  # type: ignore
    write_outputs()
