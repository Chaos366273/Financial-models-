# Detailed Plan: Regime + Indicator–Driven Rewards & 5% Profit-Taking

**Goal:** Agent should rely on **regime** and **technical/fundamental indicators** to decide when to BUY and SELL. In bull markets it will BUY, but should **SELL after ~5% return** (profit-taking). Wrong regime/signal should carry a **real cost**, not just a small multiplier cut.

**Design principles:**
- **Bonus when right** (already present) + **strong cost when wrong** (to add).
- **Regime balance:** In bull, reward SELL (profit-taking) explicitly; don’t only boost BUY; reduce BUY when overbought/weak technical.
- **Indicator veto or strong penalty:** When tech/fundamental clearly say “sell” or “no buy,” BUY is blocked or heavily penalized.

---

## 1. Stronger “Wrong Regime / Wrong Signal” Cost

**Current state:** Regime and indicators mostly **scale** rewards (0.7x–1.6x). There is no explicit penalty for “BUY in bear” or “BUY when signals say sell.”

**Target:** When regime or signals say “don’t buy” or “sell,” **expected reward for BUY** should drop (e.g. explicit penalty or large discount) so that E[BUY] < E[SELL/HOLD] in those states.

### 1.1 Wrong-regime penalties (new)

| Condition | Action | Proposed change |
|-----------|--------|------------------|
| **Bear regime** (RegimeController `bull_bear == "bear"`) | BUY | Apply a **wrong_regime_buy_penalty** (e.g. -0.08 to -0.12) in addition to existing buy_mult 0.92. So BUY in bear is clearly worse than HOLD/SELL. |
| **Bear regime** | SELL | No extra penalty (SELL in bear is correct). Optionally small bonus for “right regime” (e.g. +0.02) to reinforce. |
| **Bull regime** | SELL | No penalty (profit-taking in bull is desired). Already have `sell_bull_market_boost`; keep or slightly increase. |
| **Bull regime** | BUY | Keep current boost; add **overbought/weak-tech** discount below (so BUY in bull but overbought gets reduced). |

**Where:** In the BUY reward path in `trading_env.py`, after regime outputs are applied: if `_get_regime_outputs()` or equivalent says `bull_bear == "bear"`, add a configurable `wrong_regime_buy_penalty` (e.g. in `dynamic_config`: `wrong_regime_buy_penalty: float = -0.10`). Same idea for “BUY in strong downtrend” if you have `spy_r <= -0.02` available at step level.

### 1.2 Wrong-signal penalties (strengthen existing)

| Condition | Action | Proposed change |
|-----------|--------|------------------|
| **Tech says SELL** (e.g. tech_score for BUY < 0.3 or SELL score > 0.6) | BUY | Today: gating can block (return -0.02 or -0.05). **Option A:** Keep block. **Option B:** If you allow trade with reduced mult, add **wrong_signal_buy_penalty** (e.g. -0.06 to -0.10) when BUY with weak/bearish tech so that “BUY against signal” is clearly bad. |
| **Tech says BUY** (e.g. BUY score high) | SELL | Add **wrong_signal_sell_penalty** when SELL with strong bullish tech and no profit (e.g. -0.04) so that “SELL a winner too early with no gain” is slightly penalized; keep profit-taking bonuses as-is. |

**Where:** In BUY reward block: after computing `tech_score` / `fund_score` for the chosen ticker, if `tech_score < wrong_signal_buy_threshold` (e.g. 0.35), add `wrong_signal_buy_penalty` (e.g. -0.08). In SELL block: if position is closed at a loss and tech was strongly bullish for that ticker, add small `wrong_signal_sell_penalty`. Config: `wrong_signal_buy_threshold`, `wrong_signal_buy_penalty`, `wrong_signal_sell_penalty` in `dynamic_config`.

---

## 2. Regime Balance (Bull: Don’t Only Boost BUY; Reward SELL Profit-Taking)

**Current state:** In bull, RegimeController does `buy_mult *= 1.12`, `sell_mult *= 1.04`. So BUY is boosted more than SELL. You also have `sell_bull_market_boost` in the env for SELL in bull.

**Target:** In bull, SELL (profit-taking) should be **explicitly rewarded**; BUY should not be boosted when price is **overbought** or technical is weak.

### 2.1 Bull SELL (profit-taking) – explicit 5% target

| Item | Proposal |
|------|----------|
| **5% profit-taking bonus** | When regime is bull **and** SELL is executed **and** realized return on that position ≥ 5%: add a **profit_target_5pct_bonus** (e.g. +0.15 to +0.25). This teaches “in bull, buy then sell at ~5% gain.” |
| **Unrealized gain ≥ 5% in bull** | When in bull and agent **holds** a position with unrealized gain ≥ 5%, you can add a small **missed_profit_target_penalty** (e.g. -0.02 per step) so that holding past 5% in bull without selling is mildly costly. Optional; start with the bonus only. |

**Where:** In SELL reward path: after computing `gain_pct` (realized_pnl / trade_value), if `bull_bear == "bull"` (from RegimeController or env’s bull/bear detection) and `gain_pct >= 0.05`, add `profit_target_5pct_bonus`. Config: `profit_target_5pct_bonus: float = 0.20`, `profit_target_pct: float = 0.05`.

### 2.2 Bull BUY – reduce boost when overbought / weak technical

| Condition | Proposal |
|-----------|----------|
| **Bull + overbought** | If RSI (or equivalent) for the ticker is above an overbought threshold (e.g. RSI > 65 or 70), **cap or reduce** the BUY regime multiplier (e.g. buy_mult *= 0.85) so BUY in bull but overbought is less rewarded. |
| **Bull + weak BUY signal** | If tech_score for BUY < 0.4 in bull, do not apply the full bull BUY boost (e.g. apply only 50% of the boost). So “BUY in bull on weak signal” is less attractive. |

**Where:** In BUY reward path: after getting `buy_market_multiplier` from regime/market condition, get ticker’s RSI (or reuse `_step_signals_cache` / technical scores). If overbought, multiply buy_market_multiplier by e.g. `overbought_buy_discount = 0.85`. If tech_score (BUY) < 0.4, multiply by e.g. `weak_signal_bull_buy_discount = 0.5`. Config: `overbought_rsi_threshold: 65`, `overbought_buy_discount: 0.85`, `weak_signal_bull_buy_discount: 0.5`.

---

## 3. Indicator Veto or Strong Penalty

**Current state:** Gating can **block** BUY/SELL when confidence is too low (return -0.02 or -0.05). Reports showed “Gating blocked: 0,” so either thresholds are low or blocking is rare. There is also `weak_signal_penalty` for trading with low/medium confidence.

**Target:** When technical (and fundamental) **clearly** say “sell” or “no buy,” BUY should be either **blocked** or **heavily penalized**.

### 3.1 Stricter gating (veto) when signal is clearly against

| Scenario | Current | Proposal |
|----------|---------|----------|
| **BUY when tech strongly bearish** | Gate uses confidence; may allow with 0.3x mult. | If **SELL tech score** (for that ticker) > 0.6 **and** BUY tech score < 0.3: **hard block** BUY (return penalty e.g. -0.08 or use existing blocked_penalty -0.05). So “BUY when indicators say sell” is vetoed or very costly. |
| **SELL when tech strongly bullish and position in profit** | Not explicitly penalized. | Optional: if SELL with small profit (e.g. < 2%) and BUY tech score > 0.6, small penalty (e.g. -0.03) to discourage “sell winner too early.” Do not penalize SELL when profit ≥ 5% (we want 5% profit-taking). |

**Where:** In `_should_gate_trade_by_signal_confidence` or in the BUY branch before executing: add a **contra-signal check**. If for the chosen ticker `tech_sell_score > 0.6 and tech_buy_score < 0.3`, treat as “indicators say no buy” and return False (block) or apply strong penalty. Config: `contra_signal_buy_block_threshold_sell: 0.6`, `contra_signal_buy_block_threshold_buy: 0.3`, `contra_signal_buy_penalty: -0.08`.

### 3.2 Strong penalty when trading against signal (if not blocking)

If you prefer **soft veto** (allow but punish): when BUY is allowed but tech_score (BUY) < 0.35 **or** fund_score (BUY) < 0.3 (when fundamentals enabled), add **wrong_signal_buy_penalty** (e.g. -0.08). So E[BUY] in those states drops below HOLD/SELL.

---

## 4. 5% Return Profit-Taking (Bull)

You want: **in bull, agent buys then sells after 5% return.**

### 4.1 Already present

- **Gain tiers:** You have `gain_bonus_value_5pct` (e.g. 0.52) when `gain_pct > threshold_5pct` (0.05) on SELL. So SELL at 5%+ profit already gets a bonus.
- **Unrealized gain bonus:** `unrealized_gain_threshold_5pct` and bonus when SELL with unrealized gain > 5%.
- **Bull SELL boost:** `sell_bull_market_boost` when SPY uptrend and SELL.

### 4.2 Reinforce 5% in bull

- Add **profit_target_5pct_bonus** (Section 2.1) when **bull + SELL with realized return ≥ 5%**. This makes “sell at 5% in bull” a clear, extra reward.
- Optionally: **hold penalty with position in bull when unrealized gain ≥ 5%:** add a small step penalty (e.g. -0.02) when in bull, has position, unrealized gain ≥ 5%, and agent HOLDs. So the policy learns “in bull, once up 5%, prefer to SELL.”

**Where:** SELL path: if `bull_bear == "bull"` and `gain_pct >= 0.05`, add `profit_target_5pct_bonus`. HOLD path (optional): if bull, has_positions, and any position has unrealized_gain_pct >= 0.05, add `hold_with_5pct_unrealized_in_bull_penalty: -0.02`. Config: `profit_target_5pct_bonus: 0.20`, `hold_with_5pct_unrealized_in_bull_penalty: -0.02` (optional).

---

## 5. Config Parameters Summary (to add in `dynamic_config.py`)

| Parameter | Suggested default | Purpose |
|-----------|-------------------|---------|
| `wrong_regime_buy_penalty` | -0.10 | BUY in bear (or strong downtrend) |
| `wrong_signal_buy_threshold` | 0.35 | Tech BUY score below this → penalty if BUY |
| `wrong_signal_buy_penalty` | -0.08 | BUY when tech says no buy |
| `wrong_signal_sell_penalty` | -0.04 | SELL with no profit when tech was strongly bullish (optional) |
| `profit_target_pct` | 0.05 | Target return for profit-taking in bull |
| `profit_target_5pct_bonus` | 0.20 | Bonus for SELL in bull with realized return ≥ 5% |
| `overbought_rsi_threshold` | 65 | RSI above this = overbought for BUY discount |
| `overbought_buy_discount` | 0.85 | Multiply BUY mult by this when overbought |
| `weak_signal_bull_buy_discount` | 0.5 | In bull, if BUY tech_score < 0.4, apply this to bull BUY boost |
| `contra_signal_buy_block_threshold_sell` | 0.6 | SELL tech score above this + low BUY score → block/penalize BUY |
| `contra_signal_buy_block_threshold_buy` | 0.3 | BUY tech score below this (with high SELL score) → block/penalize BUY |
| `contra_signal_buy_penalty` | -0.08 | Penalty when BUY against clear sell signal (if not hard block) |
| `hold_with_5pct_unrealized_in_bull_penalty` | -0.02 | Optional: HOLD in bull with ≥5% unrealized gain |

---

## 6. Implementation Order (recommended)

1. **Regime wrong-side penalty** – Add `wrong_regime_buy_penalty` in BUY path when `bull_bear == "bear"`. Quick win, clear signal.
2. **5% profit-taking in bull** – Add `profit_target_5pct_bonus` in SELL path when bull + `gain_pct >= 0.05`. Reinforces “buy in bull, sell at 5%.”
3. **Wrong-signal BUY penalty** – Add `wrong_signal_buy_penalty` when BUY with tech_score (BUY) < threshold. Reduces E[BUY] when indicators say no.
4. **Contra-signal block or strong penalty** – In BUY path, if SELL score > 0.6 and BUY score < 0.3, block or apply `contra_signal_buy_penalty`.
5. **Bull BUY overbought/weak-signal discount** – In BUY path, in bull, apply overbought and weak-signal discounts to BUY multiplier.
6. **Optional:** HOLD penalty when in bull with ≥5% unrealized; optional wrong_signal_sell_penalty.

---

## 7. How This Meets Your Goals

| Goal | How the plan addresses it |
|------|----------------------------|
| **Agent relies on regime** | Wrong-regime BUY penalty and bull-only 5% bonus make regime matter for expected reward. |
| **Agent relies on indicators** | Wrong-signal and contra-signal penalties/blocks tie reward to tech (and optional fund) scores. |
| **Buy in bull** | BUY in bull stays boosted; overbought/weak-tech discount avoids “buy anything in bull.” |
| **Sell after 5% return** | `profit_target_5pct_bonus` + existing gain tiers + optional hold penalty when holding past 5% in bull. |
| **Balanced; not buy-only** | SELL in bull and at 5% is rewarded; BUY in bear or against signal is penalized so E[BUY] drops in those states. |

---

## 8. Files to Touch (when you implement)

- **`ppo/utils/dynamic_config.py`** – Add new parameters.
- **`ppo/models/trading_env.py`** – BUY path: regime check + wrong-regime penalty; overbought/weak-signal discount; contra-signal check + block/penalty; wrong_signal_buy_penalty. SELL path: bull + 5% bonus; optional wrong_signal_sell_penalty. HOLD path (optional): bull + 5% unrealized penalty.
- **`ppo/utils/regime_controller.py`** – No change required; env already uses `_get_regime_outputs()` and bull_bear. Ensure env has access to `bull_bear` at step level (from RegimeState or from its own bull/bear detection).
- **`ppo/docs/REWARD_CHANGELOG.md`** – Log all new params and rationale after implementation.

No code is implemented in this document; it is a detailed plan only.
