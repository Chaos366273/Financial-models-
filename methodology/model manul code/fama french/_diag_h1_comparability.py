"""One-off H1 comparability diagnostics (FF5 vs XGB vs LSTM)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "diagnostics"
OUT.mkdir(parents=True, exist_ok=True)


def rmse(a, b) -> float:
    e = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    return float(np.sqrt(np.mean(e**2)))


def medae(a, b) -> float:
    return float(np.median(np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))))


def main() -> None:
    ff5 = pd.read_csv(ROOT / "fama french" / "walk_forward_baseline" / "pooled_oos_predictions.csv")
    ff5["month"] = pd.to_datetime(ff5["month"])
    ff5["yk"] = ff5["month"].dt.year

    xgb = pd.read_csv(
        ROOT / "final test results" / "xgboost 2" / "walk_forward_baseline" / "walk_forward_baseline_summary.csv"
    )
    xgb_ok = xgb[xgb["status"] == "ok"].copy()
    xgb_ok["eval_year"] = xgb_ok["eval_year"].astype(int)

    lstm_s = pd.read_csv(
        ROOT / "saved_data" / "lstm_walk_forward" / "walk_forward_baseline" / "summary_metrics.csv"
    )
    lstm_ok = lstm_s[lstm_s["status"].astype(str).str.lower() == "ok"].copy()
    lstm_ok["eval_year"] = lstm_ok["eval_year"].astype(int)

    cmp = pd.DataFrame(
        {
            "ff5_n": ff5.groupby("yk").size(),
            "xgb_n": xgb_ok.set_index("eval_year")["n_eval_rows"].astype(int),
            "lstm_n": lstm_ok.set_index("eval_year")["n_eval_rows"].astype(int),
        }
    ).sort_index()
    cmp["ff5_minus_xgb"] = cmp["ff5_n"] - cmp["xgb_n"]
    cmp["ff5_minus_lstm"] = cmp["ff5_n"] - cmp["lstm_n"]
    print(cmp.to_string())
    print(
        "FF5 total",
        len(ff5),
        "XGB total",
        int(xgb_ok["n_eval_rows"].sum()),
        "LSTM total",
        int(lstm_ok["n_eval_rows"].sum()),
    )

    lstm_parts = []
    for p in sorted(
        (ROOT / "saved_data" / "lstm_walk_forward" / "walk_forward_baseline").glob("step_*/preview.csv")
    ):
        d = pd.read_csv(p)
        d["month"] = pd.to_datetime(d["month"])
        d["source"] = p.parent.name
        lstm_parts.append(d)
    lstm = pd.concat(lstm_parts, ignore_index=True)
    print("LSTM pooled preview rows", len(lstm), "tickers", lstm["Ticker"].nunique())

    ff5_k = ff5.copy()
    ff5_k["key"] = ff5_k["Ticker"].astype(str) + "|" + ff5_k["month"].dt.strftime("%Y-%m-%d")
    lstm_k = lstm.copy()
    lstm_k["key"] = lstm_k["Ticker"].astype(str) + "|" + lstm_k["month"].dt.strftime("%Y-%m-%d")
    inter = ff5_k.merge(
        lstm_k[["key", "y_pred", "y_true"]].rename(
            columns={"y_pred": "lstm_pred", "y_true": "lstm_y"}
        ),
        on="key",
        how="inner",
    )
    print("Intersection FF5 n LSTM", len(inter))
    print(
        "y_true match rate",
        float(np.isclose(inter["y_true"], inter["lstm_y"], equal_nan=True).mean()),
    )
    print(
        "Aligned RMSE FF5",
        rmse(inter["y_true"], inter["y_pred"]),
        "LSTM",
        rmse(inter["lstm_y"], inter["lstm_pred"]),
    )
    print(
        "Aligned MedAE FF5",
        medae(inter["y_true"], inter["y_pred"]),
        "LSTM",
        medae(inter["lstm_y"], inter["lstm_pred"]),
    )

    rows = []
    for y, g in inter.groupby(inter["month"].dt.year):
        rows.append(
            dict(
                year=int(y),
                n=len(g),
                ff5_rmse=rmse(g["y_true"], g["y_pred"]),
                lstm_rmse=rmse(g["lstm_y"], g["lstm_pred"]),
                ff5_medae=medae(g["y_true"], g["y_pred"]),
                lstm_medae=medae(g["lstm_y"], g["lstm_pred"]),
                max_abs_y=float(g["y_true"].abs().max()),
            )
        )
    yr = pd.DataFrame(rows)
    print(yr.to_string(index=False))
    print("mean year RMSE FF5", yr["ff5_rmse"].mean(), "LSTM", yr["lstm_rmse"].mean())
    print("median year RMSE FF5", yr["ff5_rmse"].median(), "LSTM", yr["lstm_rmse"].median())

    # FF5 year metrics on full sample
    ff5_yr = []
    for y, g in ff5.groupby("yk"):
        ff5_yr.append(
            dict(
                year=int(y),
                n=len(g),
                rmse=rmse(g["y_true"], g["y_pred"]),
                medae=medae(g["y_true"], g["y_pred"]),
                max_abs_y=float(g["y_true"].abs().max()),
            )
        )
    ff5_yr_df = pd.DataFrame(ff5_yr)

    y14 = ff5[ff5["yk"] == 2014].copy()
    desc = y14["y_true"].describe(percentiles=[0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999])
    print("\n2014 y_true describe\n", desc.to_string())
    ext = y14.reindex(y14["y_true"].abs().sort_values(ascending=False).index)
    print("\nTop 20 |y_true| 2014")
    print(ext.head(20)[["Ticker", "month", "y_true", "y_pred"]].to_string(index=False))
    sse = (y14["y_true"] - y14["y_pred"]) ** 2
    total_sse = float(sse.sum())
    top5_sse = float(((ext.head(5)["y_true"] - ext.head(5)["y_pred"]) ** 2).sum())
    top1_sse = float((ext.iloc[0]["y_true"] - ext.iloc[0]["y_pred"]) ** 2)
    print("Top5 share of 2014 SSE", top5_sse / total_sse)
    print("Top1 share", top1_sse / total_sse)
    for lo, hi in [(0.01, 0.99), (0.005, 0.995)]:
        q = y14["y_true"].quantile([lo, hi])
        m = y14["y_true"].between(q.iloc[0], q.iloc[1])
        g = y14[m]
        print(
            f"2014 RMSE keep {lo}-{hi}",
            rmse(g["y_true"], g["y_pred"]),
            "n",
            len(g),
            "dropped",
            int((~m).sum()),
        )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    clip = y14["y_true"].clip(-1, 1)
    axes[0].hist(clip, bins=60, color="#2f4b7c", edgecolor="white")
    axes[0].set_title("2014 target_excess_return (clipped ±1 for display)")
    axes[0].set_xlabel("target_excess_return")
    axes[0].set_ylabel("count")
    axes[1].scatter(range(len(ext.head(30))), ext.head(30)["y_true"].abs().values, color="#c0392b")
    axes[1].set_yscale("log")
    axes[1].set_title("2014 top 30 |target| (log scale)")
    axes[1].set_xlabel("rank")
    axes[1].set_ylabel("|target_excess_return|")
    fig.tight_layout()
    fig.savefig(OUT / "2014_target_distribution.png", dpi=140)

    cmp.to_csv(OUT / "n_obs_by_year.csv")
    yr.to_csv(OUT / "aligned_ff5_lstm_rmse_by_year.csv", index=False)
    ff5_yr_df.to_csv(OUT / "ff5_rmse_by_year_full.csv", index=False)
    ext.head(50).to_csv(OUT / "2014_extreme_observations.csv", index=False)
    print("diagnostics written to", OUT)


if __name__ == "__main__":
    main()
