Markov-switching — regime-dependent FF5 walk-forward (H1)
=========================================================
Created: 2026-08-04T14:24:34
Log: C:\Users\Faiaz\Desktop\Courses\sem2\dessertation\methodology\model manul code\marvok\markov_run_20260804_142358.txt
Active: k=2 window=rolling assign=soft train_probs=filtered market=mkt_rf fallback=True

Primary (H1): soft + filtered + rolling + mkt_rf + pooled fallback
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
