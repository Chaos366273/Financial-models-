Markov-switching — regime-dependent FF5 walk-forward (H1)
=========================================================
Created: 2026-08-20T18:22:18
Log: C:\Users\Faiaz\Desktop\Courses\sem2\dessertation\methodology\model manul code\marvok\markov_run_20260820_182001.txt
Active: k=4 window=rolling assign=soft train_probs=filtered market=mkt_rf fallback=True

Primary (H1): soft + filtered + rolling + mkt_rf + pooled fallback
Hygiene: AIS.LN × April 2014 dropped (one stock-month; unadjusted corporate action).
H1 metrics aligned with fama5.py; WLS/pooled OLS HAC(4).
FF5 summary: C:\Users\Faiaz\Desktop\Courses\sem2\dessertation\methodology\model manul code\fama french\walk_forward_baseline\walk_forward_baseline_summary.csv
R1: --train-assign hard
R2: --train-prob-source smoothed
R4: --market-series ew_excess
R5: --no-pooled-fallback
Expanding: --window expanding

Commands:
  python markov_switching.py --no-pooled-fallback --k-regimes 2
  python markov_switching.py --no-pooled-fallback --k-regimes 4
  python markov_switching.py --compare-fallback

Outputs include:
  walk_forward_nofallback_k2/ walk_forward_nofallback_k4/
  fallback_sensitivity.md

All Markov artefacts root: C:\Users\Faiaz\Desktop\Courses\sem2\dessertation\methodology\model manul code\marvok
