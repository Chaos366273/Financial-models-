# Reward Configuration Changelog

**Purpose**: Single source of truth for all reward-related changes. Use this document to track what was changed, when, and why.

---

## Document Structure

- **History**: Chronological list of all changes
- **Current Values**: Snapshot of key parameters (synced from `dynamic_config.py`)
- **Diagnostics**: Why SELL without position? Why not using technical/fundamental indicators?

---

## History of Changes

### 2026-03-07 (Positive Return with Zero Shorts – Validation)

**Context:** After 2026-03-06 reward changes and `short_signal_min_threshold` 0.68, run achieved **+2.84% return** with **0 short trades** (BUY: 35, SELL: 19, STOP_LOSS: 7). Win rate 53.8%, Sharpe 1.17, Paper Trading Ready: YES.

**Findings:**
- Zero shorts confirms `short_signal_min_threshold` 0.68 is effective (no weak bearish shorts).
- 2026-03-06 entry-quality and profit-taking changes improved win rate (33% → 53.8%) and realized return (-0.56% → +2.84%).
- SHAP diagnostic added to unified report: Python executable path + SHAP status. Report shows SHAP Unavailable when run with different Python; tip: `python3.13.exe PPO_fixed.py` or `python3.13.exe -m pip install shap`.

**Report:** `ppo/daily_reports/UNIFIED_REPORT_2026-03-07_12-40-16.md`

**Source:** `ppo/evaluation/unified_report_generator.py`, `PPO_fixed.py` (`_collect_python_shap_diagnostic`)

---

### 2026-03-06 (Win Rate & Realized Return: Entry Quality + Profit-Taking)

**Context:** Win rate 33.3%, realized return -0.56%, 92 shorts (83.6%). Goal: improve win rate and realized return via stricter entry quality, stronger profit-taking rewards, and harsher loss penalties.

**Changes:**

| Parameter | Before | After | Reason |
|-----------|--------|-------|--------|
| `wrong_signal_buy_threshold` | 0.35 | **0.45** | Only BUY when tech score stronger (entry quality) |
| `wrong_signal_buy_penalty` | -0.08 | **-0.12** | Stronger penalty for weak-signal BUY |
| `wrong_regime_buy_penalty` | -0.10 | **-0.15** | Stronger penalty for BUY in bear regime |
| `profit_target_pct` | 0.05 | **0.03** | Easier 3% target for intraday profit-taking |
| `profit_target_5pct_bonus` | 0.20 | **0.28** | Stronger bonus for SELL at ≥5% gain |
| `sell_profit_taking_bonus` | 0.80 | **0.90** | Stronger reward for profitable SELLs |
| `signal_alignment_bonus` | 0.24 | **0.30** | Stronger reward for signal-aligned profitable SELLs |
| `progressive_loss_penalty_base` | -0.15 | **-0.20** | Stronger first-loss penalty |
| `progressive_loss_penalty_step` | -0.05 | **-0.08** | Faster escalation for repeated losses |
| `hold_with_5pct_unrealized_in_bull_penalty` | -0.02 | **-0.03** | Stronger nudge to realize gains in bull |

**Retrain:** From scratch or from latest checkpoint. Compare win rate and realized return before/after.

**Source:** `ppo/utils/dynamic_config.py`

---

### 2026-02-28 (Balanced SELL Reward: Unconditional Base + Stronger Behavior-Adaptive SELL Mult)

**Context:** At 150k timesteps the policy becomes BUY-heavy (64% executed BUY, 14% voluntary SELL). Reward tuning alone failed because SELL rewards were mostly outcome-conditional (big bonuses only when `realized_pnl > 0`), so E[reward|SELL] was low/negative. User wanted **balanced** reward: agent should make **profitable trades**, not buy only for rewards; no gapping BUY.

**Changes:**

| Parameter / Location | Before | After | Reason |
|----------------------|--------|-------|--------|
| `sell_unconditional_base` (new) | — | **0.08** | Small unconditional positive on every SELL with position; outcome bonuses (sell_base_incentive, gain tiers) unchanged when profitable. SELL is no longer "0.06 or big negative." |
| Behavior-adaptive `sell_base_incentive_mult` cap | 0.75–**1.25** | 0.75–**1.55** | Computed boost (1.5x/1.8x when SELL ratio low) was clipped to 1.25x; now up to 1.55x so adaptive can meaningfully boost SELL vs BUY stack. |

**Design:** BUY unchanged. Every executed SELL with position gets: `sell_attempt_bonus` (0.06) + `sell_unconditional_base` (0.08 × behavior_mult) + outcome terms (sell_base_incentive, gain bonuses when profitable; loss penalty when not). Profitable trades still get the full outcome stack; balanced reward, not buy-only for rewards.

**Source:** `ppo/utils/dynamic_config.py`, `ppo/models/trading_env.py`, `ppo/docs/BUY_HEAVY_150K_ANALYSIS_2026-02-28.md`

---

### 2026-02-28 (Adaptive LR/Entropy Callback & Behavior-Adaptive Fixes)

**Context:** Adaptive LR and entropy callbacks were not triggering ("CALLBACK DID NOT TRIGGER", "NO ENTROPY EVENTS RECORDED"). With BUY 91.6% and entropy 0.26, entropy boost should have fired but did not. Behavior-adaptive SELL boost was too late (policy already BUY-heavy). Investigation: `ppo/docs/ADAPTIVE_CALLBACK_INVESTIGATION_2026-02-28.md`.

**Changes:**

| Parameter / Location | Before | After | Reason |
|----------------------|--------|-------|--------|
| `_continue_training` (PPO_fixed.py) | No adaptive callbacks | **AdaptiveHyperparameterCallback + EarlyStoppingCallback** when eval_env available | Continue path previously omitted adaptive callbacks; report showed empty logs |
| Entropy trigger: BUY-dominant | Not present | **buy_pct > 85%** triggers entropy boost | 91.6% BUY policy was not caught by existing triggers |
| `entropy_ratio` threshold | 0.2 | **0.3** (`adaptive_ent_entropy_ratio_threshold`) | 0.26 entropy_ratio did not trigger with 0.2; relaxed to catch BUY-heavy |
| `adaptive_lr_warmup_steps` | 50000 | **25000** | Catch collapse earlier; was blocking first 50k steps |
| `adaptive_ent_buy_high_threshold` (new) | — | **85.0** | BUY-dominant collapse detection |
| `adaptive_ent_entropy_ratio_threshold` (new) | — | **0.3** | Policy too deterministic (was hardcoded 0.2) |
| `adaptive_sell_early_boost_steps` (new) | — | **30000** | Config for early proactive SELL boost |
| `adaptive_sell_early_boost_mult` (new) | — | **1.25** | SELL mult during early phase |
| Behavior-adaptive (trading_env.py) | Executed trades only | **+ BUY-dominant** (raw buy_action_ratio > 0.85 → SELL boost 1.35x) | Proactive SELL boost when policy is BUY-heavy |
| Behavior-adaptive (trading_env.py) | — | **+ Early phase** (first 20% of episode → SELL mult 1.25x) | Prevent early BUY convergence |

**Verification:** `python -m ppo.tests.test_adaptive_callback_fixes_2026_02_28` — all tests pass.

**Source:** `PPO_fixed.py`, `ppo/models/trading_env.py`, `ppo/utils/dynamic_config.py`, `ppo/docs/ADAPTIVE_CALLBACK_INVESTIGATION_2026-02-28.md`, `ppo/tests/test_adaptive_callback_fixes_2026_02_28.py`

---

### 2026-02-28 (Regime + Indicator–Driven Rewards & 5% Profit-Taking)

**Context:** Agent should rely on **regime** and **technical/fundamental indicators** to decide when to BUY/SELL. In bull, buy then **SELL after ~5% return**. Wrong regime/signal should carry a **real cost**, not just a small multiplier cut.

**Implemented (from REWARD_PLAN_REGIME_INDICATORS_2026-02-28.md):**

| Parameter | Default | Purpose |
|-----------|--------|---------|
| `wrong_regime_buy_penalty` | -0.10 | BUY in bear regime |
| `wrong_signal_buy_threshold` | 0.35 | Tech BUY score below this → penalty if BUY |
| `wrong_signal_buy_penalty` | -0.08 | BUY when tech says no buy |
| `wrong_signal_sell_penalty` | -0.04 | SELL at loss when tech was strongly bullish |
| `profit_target_pct` | 0.05 | Target return for profit-taking in bull (5%) |
| `profit_target_5pct_bonus` | 0.20 | Bonus for SELL in bull with realized return ≥ 5% |
| `overbought_rsi_threshold` | 0.65 | RSI (0..1) above this = overbought for BUY discount |
| `overbought_buy_discount` | 0.85 | Multiply BUY mult by this when overbought in bull |
| `weak_signal_bull_buy_discount` | 0.5 | In bull, if BUY tech_score < 0.4, apply to bull BUY boost |
| `contra_signal_buy_block_threshold_sell` | 0.6 | SELL tech > this + low BUY → penalize BUY |
| `contra_signal_buy_block_threshold_buy` | 0.3 | BUY tech < this (with high SELL) → penalize BUY |
| `contra_signal_buy_penalty` | -0.08 | Penalty when BUY against clear sell signal |
| `hold_with_5pct_unrealized_in_bull_penalty` | -0.02 | HOLD in bull with ≥5% unrealized (nudge to SELL at 5%) |

**Logic:** Regime outputs now include `bull_bear`; BUY path applies wrong-regime penalty (bear), contra-signal penalty (BUY when SELL score high + BUY score low), wrong-signal penalty (BUY when tech < threshold), and in bull applies overbought/weak-signal discount to BUY multiplier. SELL path adds 5% profit-taking bonus in bull and wrong-signal SELL penalty when selling at loss with strong bullish tech. HOLD path adds optional penalty when in bull with any position ≥5% unrealized.

**Retrain:** From scratch. Track in Performance Tracking below.

**Source:** `ppo/docs/REWARD_PLAN_REGIME_INDICATORS_2026-02-28.md`, `ppo/utils/dynamic_config.py`, `ppo/models/trading_env.py`

---

### 2026-02-27 (Intraday Philosophy: Teach BUY→SELL Profitable Cycle)

**Context:** User clarified reward philosophy: teach agent to buy AND sell profitable trades; don't randomly stop buying; if profitable signal → BUY (provided signal to SELL in time); intraday = buy and sell same day.

**Philosophy:**
- **Do NOT** penalize or cap BUY — agent should buy when it sees profitable signal.
- **Do** reward completing the cycle: BUY → SELL when gains exist or signal turns bearish.
- **Intraday** — same-day round-trips; `min_hold_steps=4` (20 min) is appropriate; quick profitable exits should not be penalized.

**Changes (Before → After):**

| Parameter | Before (in code) | After (planned) | Reason |
|-----------|------------------|-----------------|--------|
| `sell_attempt_bonus` | 0.04 | **0.06** | Stronger reward for voluntary SELL with position |
| `unrealized_gain_bonus` | 0.12 | **0.18** | Stronger incentive to SELL when position has gains |
| `quick_flip_penalty_profitable` | -0.05 | **0.0** | Intraday: don't penalize quick profitable exits |
| Regime: `hold_mult` when has_positions (bull) | 0.92 | **cap ≥ 1.0** | Don't make HOLD too cheap; encourage SELL when gains exist |
| ~~Consecutive BUY penalty~~ | — | **DROPPED** | Would block valid intraday buys |

**Implementation status:** ✅ IMPLEMENTED (2026-02-27)

**Retrain:** From scratch after applying. Track results in Performance Tracking below.

**Source:** `ppo/docs/LIVE_TRADING_REWARD_PLAN_2026-02-27.md`

---

### 2026-02-26 (Changelog sync with code – documentation only)

**Context:** Reward parameters were changed in `ppo/utils/dynamic_config.py` (2026-02-21 and 2026-02-25) but the changelog’s “Current Key Values” table was not updated.

**Updates in this doc only (no code changes):**
- **Current Key Values** table synced to match `dynamic_config.py`: `pnl_reward_scale` 72, `profitable_trade_bonus` / `sell_profit_taking_bonus` 0.80, `profitable_trade_bonus_multiplier` 55, `profitable_close_min_return_pct` 0.012, gain tiers (0.72/0.52/0.32/0.15), `technical_signal_weight` 1.0, `fundamental_signal_weight` 0.10, `base_trading_incentive` / `sell_base_incentive` 0.18, and added `profitable_trade_bonus_max`, hold grace params, `eval_hold_penalty`.

**Source:** `ppo/utils/dynamic_config.py` (code is source of truth).

---

### 2026-02-25 (Balance Policy: Fix "Always BUY" for Paper Trading / TG5)

**Context:** In live/paper (TG5), model outputs BUY (action 1) every step; no voluntary HOLD/SELL. Same BUY-heavy collapse as 150k (99% BUY). Goal: reward structure that allows HOLD and voluntary SELL so the policy can produce all three actions appropriately.

**Root cause (from Diagnostics):** Strong HOLD penalty when flat (`hold_penalty_no_positions` -0.08) and strong general `hold_penalty` (-0.18) make "do something" dominant; BUY rewards fire often (bull/signal), SELL rewards are outcome-conditional (realized_pnl > 0), so agent learns BUY is safest. No penalty for excessive BUY.

**Changes to make (in `ppo/utils/dynamic_config.py`):**

| Parameter | Before | After | Reason |
|-----------|--------|-------|--------|
| `hold_penalty_no_positions` | -0.08 | **-0.05** | Softer nudge when flat; HOLD when no signal is acceptable (reduces "must BUY every step") |
| `hold_penalty` | -0.18 | **-0.12** | Still costly to HOLD but not crushing; leaves room for HOLD when uncertain |
| `hold_penalty_with_positions` | -0.05 | **-0.04** | Slightly less punishment for HOLD with position (encourage hold-then-sell, not only BUY) |
| `sell_attempt_bonus` | 0.02 | **0.04** | Small reward for SELL with position (even before outcome); encourages voluntary SELL |
| `unrealized_gain_bonus` | 0.08 | **0.12** | Stronger incentive to SELL when position has unrealized gains (realize gains) |
| `eval_hold_penalty` | -0.06 | **-0.05** | Slight relax in eval so deterministic policy isn't over-pushed to trade |

**Optional (requires `trading_env.py`):** Add **consecutive BUY penalty** after N BUYs (e.g. 15): e.g. `consecutive_buy_penalty_after: int = 15`, `consecutive_buy_penalty: float = -0.02` so "always BUY" becomes mildly costly. Implement only if above config changes are insufficient after retrain.

**Retrain:** After changing these, **retrain from scratch or from latest checkpoint**; ensure evaluation reports policy action distribution (BUY/HOLD/SELL %). Target: BUY 30–50%, HOLD 40–55%, SELL 10–25% (not 99% BUY).

**Source:** `ppo/utils/dynamic_config.py` (and optionally `ppo/models/trading_env.py` for consecutive BUY penalty).

---

### 2026-02-20 (Entropy Collapse Fix – Hold Penalty Grace + Entropy Floor)

**Root cause:** `hold_penalty_with_positions = -0.08` was too strong. After BUY (+0.2), each HOLD step = -0.08. So 3 holds = -0.24, net = -0.04. Agent learned: "don't BUY, avoid hold penalty." Policy collapsed to HOLD 97%+.

**Fixes:**
- `hold_penalty_with_positions`: -0.08 → **-0.04** (less crushing after BUY)
- **hold_penalty_with_positions_grace**: **-0.02** for first N steps after BUY
- **hold_penalty_grace_steps**: **3** — grace period so agent isn't punished for holding a position just opened
- `ent_coef`: 0.05 → **0.12** (prevent entropy collapse)
- **ent_coef_min**: **0.08** — adaptive callback never restores below this floor

**Removed:** Stochastic evaluation (not realistic for paper trading; single deterministic result only).

**Source**: `ppo/utils/dynamic_config.py`, `ppo/models/trading_env.py`, `PPO_fixed.py`, `ppo/evaluation/unified_report_generator.py`

---

### 2026-02-21 (Plan: Train Until 5%, Block 97% HOLD, Entropy for 300k)

**Goals**: Train until 5% positive return; 97% HOLD not allowed; maintain exploration through 300k steps.

**Hold penalty changes (block 97% HOLD):**

| Parameter | Before | After | Reason |
|-----------|--------|-------|--------|
| `hold_penalty` | -0.10 | **-0.18** | Stronger HOLD deterrent |
| `hold_penalty_no_positions` | -0.10 | **-0.08** | Encourage BUY when flat |
| `hold_penalty_with_positions` | -0.04 | **-0.05** | Encourage SELL or purposeful hold |
| `eval_hold_penalty` | -0.05 | **-0.06** | Match train intent in eval |

**Entropy changes (300k training):**

| Parameter | Before | After | Reason |
|-----------|--------|-------|--------|
| `ent_coef` | 0.12 | **0.15** | Maintain exploration through 300k |
| `ent_coef_min` | 0.08 | **0.10** | Floor for adaptive entropy |
| `adaptive_ent_boost_factor` | 2.0 | **2.5** | Stronger boost when collapsed |

**Other**: Stochastic eval as primary when deterministic trades < 10; `--train-until-return 0.05` CLI; cap 300k.

**BUY-heavy collapse fix (trading_env.py)**: When STOP_LOSS executes, reset `_consecutive_buys` to 0. Previously, stop loss did not reset the counter, so 100 consecutive BUYs ended the episode before the agent could learn to SELL.

**Source**: `ppo/utils/dynamic_config.py`, `ppo/models/trading_env.py`, `ppo/training/ppo_trainer.py`, `PPO_fixed.py`

---

### 2026-02-21 (Higher Win Rate, Alpha, Realized Return, Positive P&L Incentives)

**Goal**: Increase reward incentives for higher win rate, alpha return, realized return, and more positive P&L.

| Parameter | Before | After | Reason |
|-----------|--------|-------|--------|
| `pnl_reward_scale` | 60 | **72** | Stronger positive P&L signal |
| `profitable_trade_bonus` | 0.70 | **0.80** | Higher win rate incentive |
| `sell_profit_taking_bonus` | 0.70 | **0.80** | Higher realized return incentive |
| `profitable_trade_bonus_multiplier` | 50 | **55** | 1% profit = 0.55 bonus |
| `profitable_trade_bonus_max` | 1.2 | **1.35** | Allow bigger rewards for big wins |
| `gain_bonus_value_10pct` | 0.65 | **0.72** | Reinforce big wins |
| `gain_bonus_value_5pct` | 0.45 | **0.52** | |
| `gain_bonus_value_2pct` | 0.28 | **0.32** | |
| `gain_bonus_value_1pct` | 0.12 | **0.15** | Reward small profits |
| `unrealized_gain_bonus` | 0.05 | **0.08** | Stronger incentive to realize gains |
| `unrealized_gain_threshold_2pct` | 0.15 | **0.18** | Encourage capturing gains |
| `unrealized_gain_threshold_5pct` | 0.30 | **0.35** | |
| `unrealized_gain_threshold_10pct` | 0.50 | **0.58** | |
| `win_rate_bonus` | 1.5 | **2.0** | Higher win rate incentive |
| `profitable_close_min_return_pct` | 0.015 | **0.012** | More closes qualify as profitable |

**Source**: `ppo/utils/dynamic_config.py`

---

### 2026-02-21 Run Results Summary

| Timesteps | Return | Notes |
|-----------|--------|------|
| 50k | Positive | Good early signal |
| 100k | -1.04% (run 1) | Negative |
| 100k | +1.09% (run 2) | Positive; variance across seeds |
| 150k | +6.71% | **Significant improvement** but mostly unrealized (Realized: -3.18%, Unrealized: +8.33%) |

**150k behavior**: Policy outputs 99% BUY, 0 voluntary SELL. All 44 exits are STOP_LOSS (automatic). Episode ends at 100 consecutive BUYs. See Diagnostics: Why 150k Model Not Selling.

---

### 2026-02-20 (Dead Code Audit + Deterministic Collapse Fix)

**Root cause diagnosed:** Deterministic eval showed 96.2% HOLD / 6 trades / -0.08% return. Three code bugs silently killed up to +1.85 reward per trade. Combined with regime-controller HOLD discount introduced 2026-02-19, the policy collapsed to near-deterministic HOLD.

**Critical code bugs fixed (trading_env.py):**

| Bug | Location | Impact |
|-----|----------|--------|
| `allow_positive_signal = portfolio_change >= 0` always False at BUY time (fees make change negative) | line ~5395 | Up to +0.80 BUY reward never fired |
| Unrealized gain checks ran after `shares` already set to 0 by sale execution | lines ~6312, ~6381 | +0.50/+0.30/+0.15 SELL threshold bonuses never fired |
| `reward = sell_attempt_bonus` then overwritten by `reward = normalized_pnl * ...` | lines ~5774, ~6135 | +0.05 per SELL attempt lost |

**Fixes:**
- `allow_positive_signal = True` — signal bonuses now fire for BUY
- Pre-capture `_pre_sale_unrealized_gain_pct` before position is cleared; use at both unrealized gain checks
- Changed `reward = normalized_pnl * ...` to `reward += normalized_pnl * ...`

**Config sync gaps fixed (trading_env.py):**

| Issue | Fix |
|-------|-----|
| `hold_progressive_start` not in sync overlay → fallback 30 used instead of dynamic_config's 10 | Added to sync overlay; fallback → 10 |
| `hold_penalty_transition_threshold` code fallback 0.95 vs dynamic_config 0.9 | Fallback → 0.9 |

**Reward parameter changes (dynamic_config.py):**

| Parameter | Before | After | Reason |
|-----------|--------|-------|--------|
| `quick_flip_penalty_profitable` | -0.35 | **-0.05** | Quick profitable exits were net-negative; killed entire fast-exit lifecycle |
| `min_hold_steps` | 8 | **4** | 40min minimum was HFT-inappropriate; 20min more realistic |
| `weak_signal_penalty` | -0.06 | **-0.02** | Medium-confidence trades barely profitable; too much friction |
| `progressive_loss_penalty_base` | -0.3 | **-0.15** | First loss too catastrophic in volatile market |
| `progressive_loss_penalty_step` | -0.2 | **-0.05** | Spiral was -0.3, -0.5, -0.7 (total -1.5 for 3 losses); now -0.15, -0.20, -0.25 (total -0.60) |
| `no_positions_bonus` | 0.01 | **0.05** | Negligible BUY incentive when flat; increased to meaningful nudge |
| `eval_hold_penalty` | -0.03 | **-0.05** | HOLD near-free in eval; stronger deterrent needed |
| `eval_hold_penalty_performance_based` | True | **False** | Free pass when growing caused HOLD collapse in eval |
| `eval_hold_penalty_minimum_floor` | -0.03 | **-0.05** | Match new eval_hold_penalty |

**Regime controller fix (regime_controller.py):**
- Removed `hold_mult *= 0.75` in `high_vol` branch — this change (added 2026-02-19) caused HOLD penalty to be only -0.075 instead of -0.10 in volatile regimes, making HOLD the cheapest action. Since training vol = 2.11% (above 2% threshold), it applied on nearly every step.

**Cleanup:**
- Removed two `tech_score_for_checks = original_tech_score` assignments (variable never read)
- Removed duplicate `self._recent_masked_sell = False` (already cleared 20 lines earlier)

**Tooling:**
- Installed `shap==0.50.0` — SHAP explainability reports now show real feature importances instead of fallback 1.0

**Expected outcome after retraining:**

| Metric | Before | Expected |
|--------|--------|----------|
| Deterministic trades | 4–6 | 30–70 |
| HOLD% (deterministic) | 96.2% | 55–70% |
| BUY% (deterministic) | 1.0% | 15–25% |
| Win rate | 0% | 45–60% |
| Realized return | -0.08% | +0.5–2.0% |

**Source**: `ppo/models/trading_env.py`, `ppo/utils/dynamic_config.py`, `ppo/utils/regime_controller.py`

---

### 2026-02-19 (Deterministic returns – stronger HOLD penalty & incentives)

| Parameter | Before | After | Reason |
|-----------|--------|-------|--------|
| `base_trading_incentive` | 0.12 | **0.18** | Improve deterministic returns; encourage more activity |
| `sell_base_incentive` | 0.12 | **0.18** | Same |
| `hold_penalty` | -0.05 | **-0.10** | Stronger; relax only in high vol (adaptive multiplier) |
| `hold_penalty_with_positions` | -0.04 | **-0.08** | Stronger; HOLD with position less attractive vs SELL |
| `hold_penalty_no_positions` | -0.05 | **-0.10** | Stronger; relax only in high volatility |

**Other changes (non-reward):**
- **Hold penalty in volatility**: `_calculate_adaptive_hold_penalty()` – high vol now *reduces* penalty (0.5x–1.0x) so HOLD is acceptable in volatile regimes; low vol = full penalty.
- **Entropy boost**: Re-applied every eval when imbalance detected (sustained boost until balanced), not just once.
- **SHAP explainer**: Pass `action_masks` to `model.predict()` so explanations match masked behavior (no SELL when flat).
- **Adaptive base incentive**: Reward incentive now changes on its own in two ways:
  - **Per step**: Existing behavior-based multipliers (`base_trading_incentive_mult`, `sell_base_incentive_mult`, `buy_base_incentive_mult`) still scale the base every step (e.g. 1.0x–1.5x for trading, 0.4x–1.8x for SELL).
  - **Across training**: When the adaptive callback detects action imbalance or low trades, it bumps `base_trading_incentive` and `sell_base_incentive` in config by `adaptive_incentive_step` (default 0.02), capped at `adaptive_incentive_cap` (0.25). Env syncs via `_on_config_change` so all envs see the new base immediately.

**Source**: `ppo/utils/dynamic_config.py`, `ppo/models/trading_env.py`, `PPO_fixed.py`, `ppo/utils/agent_explainability.py`

---

### 2026-02-19 (Policy Collapse & Report Consistency)

| Parameter | Before | After | Reason |
|-----------|--------|-------|--------|
| `base_trading_incentive` | 0.08 | **0.12** | Reduce policy collapse; encourage more activity |
| `sell_base_incentive` | 0.08 | **0.12** | Same |
| `hold_penalty_no_positions` | -0.08 | **-0.05** | Less extreme so policy doesn’t collapse to HOLD |
| `base_stop_loss_pct` | 0.06 | **0.08** | Slightly wider to avoid premature exits in volatile regimes |

**Other changes (non-reward):**
- **Position shares mismatch**: Clear `position_lots[ticker]` when stop loss executes; sync `position_lots` with `positions[ticker]['shares']` after BUY (single source of truth).
- **Unified report**: Headline and Key Performance Metrics use **deterministic** eval only (no stochastic overwrite).

**Source**: `ppo/utils/dynamic_config.py`, `ppo/models/trading_env.py`, `ppo/evaluation/unified_report_generator.py`

---

### 2026-02-15 (Batch 2 – Win Rate & Realized Return)

| Parameter | Before | After | Reason |
|-----------|--------|-------|--------|
| `profitable_trade_bonus` | 0.65 | **0.70** | Stronger reinforcement for winning trades |
| `sell_profit_taking_bonus` | 0.65 | **0.70** | Stronger reinforcement for profitable SELLs |
| `hold_winner_bonus` | 0.05 | **0.08** | Reward holding winners longer |
| `hold_time_bonus` | 0.12 | **0.15** | Reward hold-then-sell pattern |
| `sell_hold_time_bonus_base` | 0.15 | **0.18** | Reward holding then selling |
| `gain_bonus_value_10pct` | 0.60 | **0.65** | Reinforce big wins |
| `gain_bonus_value_5pct` | 0.40 | **0.45** | Reinforce medium wins |
| `gain_bonus_value_2pct` | 0.25 | **0.28** | Reinforce small wins |
| `gain_bonus_value_1pct` | (new) | **0.12** | Reward 1%+ gains (small profits count) |
| `profitable_trade_bonus_multiplier` | 45 | **50** | 1% profit = 0.50 bonus |
| `profitable_close_min_return_pct` | 0.02 | **0.015** | More closes qualify as profitable |
| `failed_sell_penalty` | -0.001 | **-0.02** | Teach policy: SELL when empty = bad (SHAP habit fix) |
| `failed_sell_consecutive_penalty` | -0.0005 | **-0.01** | Stronger growth for repeated mistakes |
| `failed_sell_penalty_cap` | (new) | **-0.10** | Allow stronger penalty to be felt |

**Source**: `ppo/utils/dynamic_config.py`, `ppo/models/trading_env.py`

---

### 2026-02-15 (Batch 1 – Outcome-Based SELL Incentives)

| Parameter | Before | After | Reason |
|-----------|--------|-------|--------|
| `sell_base_incentive` | Unconditional | **Conditional on realized_pnl > 0** | Only reward profitable SELLs |
| `adaptive_sell_bonus` | Unconditional | **Conditional on realized_pnl > 0** | Same |
| `trading_frequency_bonus` | Unconditional | **Conditional on realized_pnl > 0** | Same |
| `progressive_bonus` | Unconditional | **Conditional on realized_pnl > 0** | Same |
| `exploration_bonus` | Unconditional | **Conditional on realized_pnl > 0** | Same |
| `loss_scale_multiplier` | 2.0 | **2.5** | Losses hurt more |
| `profitable_trade_bonus` | 0.55 | **0.65** | Reinforce positive trades |
| `sell_profit_taking_bonus` | 0.55 | **0.65** | Same |
| `pnl_reward_scale` | 50 | **60** | Stronger PnL signal |
| `profitable_trade_bonus_multiplier` | 40 | **45** | Same |
| `profitable_close_min_return_pct` | 0.05 | **0.02** | More closes qualify |
| `gain_bonus_value_10pct` | 0.55 | **0.60** | Big wins |
| `gain_bonus_value_5pct` | 0.35 | **0.40** | Medium wins |
| `gain_bonus_value_2pct` | 0.20 | **0.25** | Small wins |
| `gain_bonus_threshold_1pct` | (new) | **0.01** | New tier |
| `gain_bonus_value_1pct` | (new) | **0.12** | Reward 1%+ gains |

**Source**: `ppo/utils/dynamic_config.py`, `ppo/models/trading_env.py`

---

### 2026-02-14 (Policy Collapse Fixes)

| Parameter | Before | After | Reason |
|-----------|--------|-------|--------|
| Entropy: `model.ent_coef` priority | `policy.ent_coef` first | **`model.ent_coef` first** | SB3 compatibility |
| Entropy when trades=0 | 1.5x boost | **`adaptive_ent_max` (0.08)** | Max exploration when collapsed |
| Action entropy collapse trigger | None | **Trigger when entropy_ratio < 0.2** | Detect policy too deterministic |
| `eval_freq` (Adaptive callback) | 10000 | **5000** | Detect collapse earlier |
| `tech_only_soft_gating` | Required `reward_mode='technical_only_v1'` | **Enable when `fundamental_signal_weight < 0.2`** | Reduce unintended trade blocking |

**Source**: `PPO_fixed.py`, `ppo/models/trading_env.py`

---

### 2026-02-10 (Phase 2 – PnL-First)

| Parameter | Before | After | Reason |
|-----------|--------|-------|--------|
| `base_trading_incentive` | 0.25 | **0.08** | PnL must dominate |
| `sell_base_incentive` | 0.25 | **0.08** | Same |
| `trading_frequency_bonus` | 0.01 | **0.0** | Reward quality not quantity |
| `exploration_bonus` | 0.05 | **0.0** | Use signal alignment instead |
| `learning_bonus` | 0.03 | **0.0** | PnL is teacher |

**Source**: `ppo/utils/dynamic_config.py`

---

### 2026-02-09 (SELL Signal Alignment)

| Parameter | Before | After | Reason |
|-----------|--------|-------|--------|
| `signal_alignment_bonus` (SELL) | Unconditional | **Tied to realized_pnl > 0** | Full bonus only when profitable |
| `hold_winner_bonus` | 0.03 | **0.05** | Strengthen hold-winner behavior |

**Source**: `ppo/models/trading_env.py`

---

### 2026-01-18 (Hold Penalty & Failed SELL)

| Parameter | Before | After | Reason |
|-----------|--------|-------|--------|
| `hold_penalty` | -0.0001 | **-0.05** | HOLD must be costly |
| `hold_penalty_no_positions` | -0.005 | **-0.08** | Strong nudge when flat |
| `failed_sell_penalty` | -0.005 | **-0.001** | Minimal (was discouraging SELL attempts) |

**Source**: `ppo/utils/dynamic_config.py`, `ppo/models/trading_env.py`

---

## Diagnostics: Why 150k Model Not Selling (BUY-Heavy Collapse)

**Observed**: 150k model shows 6.71% total return but 0 voluntary SELL trades. All 44 exits are STOP_LOSS (environment-triggered). Policy: 99% BUY, 1% SELL, 0% HOLD.

**Root cause**: Policy collapsed to "always BUY" — opposite of prior "always HOLD" collapse. Agent learned BUY is more rewarding than SELL in training.

**Why no voluntary SELL?**

1. **SELL rewards are outcome-conditional**: `sell_base_incentive`, `profitable_trade_bonus` only fire when `realized_pnl > 0`. Losing SELLs get penalties. Agent may have learned "SELL often loses" in training.
2. **BUY rewards fire more often**: Bull market / high vol boosts BUY. Agent keeps buying; stop loss forces exits; freed cash enables more BUYs.
3. **`_consecutive_buys` not reset by STOP_LOSS**: When stop loss executes, `_consecutive_buys` is NOT reset (only SELL action resets it). So 100 consecutive BUYs → episode ends before agent can learn to SELL.
4. **Behavior adaptive tried**: Report shows "Low SELL ratio detected - system increased SELL rewards" (1.12x–1.25x) but policy was already too deterministic (entropy 0.05).

**Recommended fixes** (in order):

1. **Reset `_consecutive_buys` when STOP_LOSS triggers** — ✅ DONE 2026-02-21. Treat stop loss as an exit so episode doesn't end prematurely.
2. **Strengthen voluntary SELL incentives** — e.g. `unrealized_gain_bonus` when SELL with gains; or small bonus for any SELL with position (not just profitable).
3. **Penalty for "never SELL"** — if agent has unrealized gains > X% and never chooses SELL, apply missed-sell penalty (already exists but may need tuning).
4. **Consider lowering consecutive-BUY limit** — 100 may be too high; agent hits it before learning. Or make it configurable.

---

## Diagnostics: Why SELL Without Position?

### What Signals the Agent to Predict SELL When It Has No Positions?

**SHAP report (fallback method)** shows top features: `close_AMZN_t*`, `vol_ratio_20_AMZN_t*`, `close_NVDA_t*`, `vol_ratio_20_NVDA_t*`, `Portfolio_Cash_Pct`.

**Important caveat**: The SHAP fallback (`_explain_with_signals`) ranks features by **raw observation magnitude**, not causal importance. Features with high values (e.g. close ~200–400, vol_ratio ~0.5–2.0) appear first after clipping to [-1, 1]. So the report does **not** prove these features cause SELL.

**Likely causes**:

1. **Spurious correlation**: Policy may have learned “high cash + certain price/vol patterns → SELL” from training data where SELL was often correct. At episode start (100% cash), those patterns still appear, so it outputs SELL.
2. **Position state under-weighted**: Observation = [market_features..., portfolio_state]. Portfolio state (cash%, pos_0%, pos_1%, …) is at the **end**. The network may weight market features more than position features.
3. **Weak penalty**: `failed_sell_penalty` was -0.001, so the policy had little incentive to avoid SELL when empty.

**Fix applied (2026-02-15)**:
- `failed_sell_penalty`: -0.001 → **-0.02**
- `failed_sell_consecutive_penalty`: -0.0005 → **-0.01**
- `failed_sell_penalty_cap`: **-0.10**

---

## Diagnostics: Why Not Using Technical and Fundamental Indicators?

### Is the Agent Using Technical Indicators?

**Yes.** The observation includes technical indicators from the feature matrix:

- **Observation structure**: `[market_features (window × tickers × num_features), portfolio_state (cash%, pos_0%, …)]`
- **Feature columns** (NEW_CSV path): `close`, `ret_1`, `rsi_14`, `macd`, `macd_signal`, `macd_hist`, `bb_z`, `bb_width`, `atr_14`, `vol_ratio_20`, `tod_sin`, `tod_cos`, `spy_ret_1`, `spy_rsi_14`

So the policy **does** receive RSI, MACD, Bollinger Bands, ATR, volume ratio, and SPY context. The SHAP fallback highlights `close` and `vol_ratio_20` because they have the largest magnitudes, not because they are the only drivers.

### Is the Agent Using Fundamental Indicators?

**Partially.** Fundamental data is **not** in the observation. It is used in:

1. **Reward**: `_calculate_fundamental_signal_quality()` → `signal_alignment_bonus` when tech + fund align
2. **Gating**: `_should_gate_trade_by_signal_confidence(tech_score, fund_score, action)` can block or down-weight trades

So the policy learns from fundamentals **indirectly** via reward and gating, not from direct input. If fundamentals are weak or missing, the policy has no direct fundamental signal to use.

**Recommendation**: To make fundamentals more influential, either:
- Add fundamental metrics to the observation (e.g. P/E, revenue growth), or
- Increase `fundamental_signal_weight` and ensure the fundamental cache is populated.

---

## Performance Tracking (What's Working)

*Record run results here to track which reward configs work. Update after each retrain.*

### Latest Run (Before 2026-02-27 Changes)

| Date | Timesteps | Total Return | Realized | Unrealized | Voluntary SELL% | BUY% | Notes |
|------|-----------|--------------|----------|------------|-----------------|------|-------|
| 2026-02-27 | ~150k | 8.40% | 8.81% | -0.67% | 5.4% (18/90) | 93% | BUY-heavy; 72 stop losses; unrealized negative (open positions underwater) |

**Issue:** Agent buys but rarely sells voluntarily; most exits are stop loss. Need to teach BUY→SELL profitable cycle.

### After 2026-02-27 Changes (To Fill After Retrain)

| Date | Timesteps | Total Return | Realized | Unrealized | Voluntary SELL% | BUY% | Notes |
|------|-----------|--------------|----------|------------|-----------------|------|-------|
| — | — | — | — | — | — | — | Retrain from scratch; apply A2–A5 |

### After 2026-02-28 Regime/Indicator Changes (To Fill After Retrain)

| Date | Timesteps | Total Return | Realized | Unrealized | Voluntary SELL% | BUY% | SELL@5%+ in bull | Notes |
|------|-----------|--------------|----------|------------|-----------------|------|------------------|-------|
| — | — | — | — | — | — | — | — | Retrain from scratch; check wrong-regime/contra-signal/wrong-signal penalties fire; 5% bonus in bull |

**What to check:** Policy action mix (BUY 25–45%, HOLD 40–55%, SELL 15–30%); voluntary SELL% of exits; whether SELL at 5%+ in bull gets bonus (logs `[PROFIT_TARGET_5PCT]`); whether BUY in bear gets penalty; gating/contra-signal blocks or penalizes BUY when indicators say sell.

---

## Current Key Values (Snapshot)

*Synced from `ppo/utils/dynamic_config.py` (code is source of truth).*

### In Code Now (After 2026-02-27 Implementation)

| Category | Parameter | Value |
|----------|-----------|-------|
| **PnL** | `pnl_reward_scale` | 72 |
| | `loss_scale_multiplier` | 2.5 |
| **Profitable SELL** | `profitable_trade_bonus` | 0.80 |
| | `sell_profit_taking_bonus` | 0.80 |
| | `sell_attempt_bonus` | **0.06** |
| | `sell_unconditional_base` | **0.08** |
| | `unrealized_gain_bonus` | **0.18** |
| | `profitable_trade_bonus_multiplier` | 55 |
| | `profitable_trade_bonus_max` | 1.35 |
| | `profitable_close_min_return_pct` | 0.012 |
| **Gain tiers** | `gain_bonus_value_10pct` | 0.72 |
| | `gain_bonus_value_5pct` | 0.52 |
| | `gain_bonus_value_2pct` | 0.32 |
| | `gain_bonus_value_1pct` | 0.18 |
| **Hold** | `hold_penalty` | -0.12 |
| | `hold_penalty_no_positions` | -0.05 |
| | `hold_penalty_with_positions` | -0.04 |
| | `hold_penalty_with_positions_grace` | -0.02 |
| | `hold_penalty_grace_steps` | 3 |
| | `hold_winner_bonus` | 0.08 |
| | `hold_time_bonus` | 0.15 |
| | `eval_hold_penalty` | -0.05 |
| **Quick flip** | `quick_flip_penalty` | -0.15 |
| | `quick_flip_penalty_profitable` | **0.0** |
| | `min_hold_steps` | 4 |
| **Failed SELL** | `failed_sell_penalty` | -0.02 |
| | `failed_sell_consecutive_penalty` | -0.01 |
| | `failed_sell_penalty_cap` | -0.10 |
| **Base / activity** | `base_trading_incentive` | 0.18 |
| | `sell_base_incentive` | 0.18 |
| **Signal** | `signal_alignment_bonus` | 0.24 |
| | `technical_signal_weight` | 1.0 |
| | `fundamental_signal_weight` | 0.10 |
| **Regime/indicator (2026-02-28)** | `wrong_regime_buy_penalty` | -0.10 |
| | `wrong_signal_buy_threshold` | 0.35 |
| | `wrong_signal_buy_penalty` | -0.08 |
| | `wrong_signal_sell_penalty` | -0.04 |
| | `profit_target_pct` | 0.05 |
| | `profit_target_5pct_bonus` | 0.20 |
| | `overbought_rsi_threshold` | 0.65 |
| | `overbought_buy_discount` | 0.85 |
| | `weak_signal_bull_buy_discount` | 0.5 |
| | `contra_signal_buy_penalty` | -0.08 |
| | `hold_with_5pct_unrealized_in_bull_penalty` | -0.02 |
| **Adaptive callback (2026-02-28)** | `adaptive_lr_warmup_steps` | 25000 |
| | `adaptive_ent_buy_high_threshold` | 85.0 |
| | `adaptive_ent_entropy_ratio_threshold` | 0.3 |
| | `adaptive_sell_early_boost_steps` | 30000 |
| | `adaptive_sell_early_boost_mult` | 1.25 |

### Applied (2026-02-27)

| Parameter | Before | After |
|-----------|--------|-------|
| `sell_attempt_bonus` | 0.04 | **0.06** |
| `unrealized_gain_bonus` | 0.12 | **0.18** |
| `quick_flip_penalty_profitable` | -0.05 | **0.0** |
| Regime `hold_mult` (bull, has_positions) | 0.92 | **unchanged (no discount)** |

---

## How to Update This Document

When changing reward parameters:

1. Add a new dated entry under **History of Changes**
2. Include: parameter name, before, after, reason
3. Update **Current Key Values** if needed (and **Planned** when implementing)
4. Add **Diagnostics** notes if the change addresses a specific behavior
5. **After each retrain:** Update **Performance Tracking** with run results (timesteps, return, voluntary SELL%, BUY%, notes) so you can see what's working

---

---

## 2026-02-15: Fundamental Metrics in Observation

| Change | Detail |
|-------|--------|
| **Added** | 7 fundamental metrics per ticker in observation: `pe_ratio`, `pb_ratio`, `debt_equity`, `roe`, `current_ratio`, `revenue_growth`, `earnings_growth` |
| **Config** | `include_fundamental_in_observation: bool = True` in `dynamic_config.py` |
| **Normalization** | Metrics scaled to ~[-1, 1] for network stability |
| **When unavailable** | Zeros used (e.g. cache empty, time-filtered in test) |

**Note**: Enabling this changes observation size. **Retrain from scratch** (do not load old model).

---

## Should the Agent See All Companies from the Data?

**Recommendation: Yes.** The agent should observe all tickers in the trading universe.

| Aspect | Recommendation |
|--------|----------------|
| **Tickers in universe** | Include all. The agent must see each ticker's data to decide BUY/SELL/HOLD for that ticker. |
| **Market context** | Already included via SPY (spy_ret_1, spy_rsi_14). Optional: add sector ETFs for broader context. |
| **Observation size** | More tickers = larger obs. With 11–12 tickers, size is manageable. 20+ tickers may need observation pruning. |

**Do not** add tickers the agent cannot trade (e.g. indexes only for context) unless you explicitly add a "context-only" slot. Keep the observation aligned with the action space (which ticker to trade).

**Implementation (2026-02-15):** The agent is configured to see all companies from the data.

| Item | Detail |
|------|--------|
| **Config** | `use_all_data_tickers: bool = True` in `ppo/utils/dynamic_config.py` |
| **Behavior** | When True, `expected_tickers` = all tickers present in the loaded data (e.g. CSV/cache), not only `config['tickers']`. |
| **Sync** | `self.config['tickers']` is set to that list so env, SHAP, and reports use the same names. |
| **Env names** | After creating the env, `env.unique_tickers` and `env.ticker_symbols` are set to the data ticker list so labels match matrix columns. |
| **Disable** | Set `use_all_data_tickers = False` to restrict to config tickers only (e.g. for a fixed small universe). |

*Last updated: 2026-02-15*
