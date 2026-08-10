from __future__ import annotations

from dataclasses import dataclass


def _clip(x: float, lo: float, hi: float) -> float:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


@dataclass(frozen=True)
class RegimeState:
    """
    Lightweight, environment-agnostic snapshot of market conditions.

    Notes:
    - `avg_volatility` is realized vol over a short window (e.g. 20 bars), in return space (0.02 = 2%).
    - `spy_return_lookback` is SPY (or proxy) return over a short window (e.g. 10 bars), in return space.
    """

    market_regime: str  # "trending" | "mean_reversion" | "unknown"
    bull_bear: str  # "bull" | "bear" | "neutral"
    avg_volatility: float
    spy_return_lookback: float
    vol_ratio: float
    has_positions: bool


@dataclass(frozen=True)
class RegimeOutputs:
    """
    Primary (market-first) adaptive outputs.
    Multipliers are intended to be applied to the *base* shaped reward terms.
    """

    buy_reward_mult: float
    sell_reward_mult: float
    hold_penalty_mult: float  # Multiplies the magnitude of HOLD penalties ( >1 = harsher HOLD )
    gate_high_threshold: float
    gate_medium_threshold: float
    stop_loss_mult: float
    trailing_stop_mult: float


class RegimeController:
    """
    Market-first adaptation controller.

    Design goals:
    - Deterministic (paper-trading) behavior should be driven primarily by market regime,
      not by behavior-adaptive self-rescue knobs.
    - HOLD should be acceptable mainly in high volatility / low edge regimes OR when holding winners in a trend.
    - BUY should be easier (lower gating thresholds) in bull/uptrend regimes, especially when flat.
    - SELL should be easier (lower gating thresholds for SELL) in bear/downtrend regimes and when risk is high.
    """

    def compute(self, s: RegimeState) -> RegimeOutputs:
        # Base thresholds (matches existing env defaults).
        gate_high = 0.25
        gate_med = 0.10

        buy_mult = 1.0
        sell_mult = 1.0
        hold_mult = 1.0
        stop_mult = 1.0
        trail_mult = 1.0

        vol = float(s.avg_volatility or 0.0)
        spy_r = float(s.spy_return_lookback or 0.0)

        high_vol = vol >= 0.020

        # 1) Volatility is the primary driver for HOLD permissiveness and stop width.
        if high_vol:
            # 2026-02-20: Removed hold_mult *= 0.75 — discounting HOLD penalty in volatile regimes caused
            # deterministic collapse (96% HOLD). Stops still widen; HOLD remains fully penalised.
            stop_mult *= 1.20
            trail_mult *= 1.20
            # Require higher confidence to trade in choppy regimes.
            gate_high += 0.03
            gate_med += 0.02
        else:
            # In calm regimes, HOLD should be less attractive (avoid deterministic HOLD collapse).
            hold_mult *= 1.10

        # 2) Bull/bear tilt (secondary driver).
        if s.bull_bear == "bull":
            buy_mult *= 1.12
            # Encourage profit-taking slightly in bull.
            sell_mult *= 1.04
            # Easier to BUY in bull, especially when flat.
            gate_high -= 0.02
            gate_med -= 0.01
            if not s.has_positions:
                hold_mult *= 1.12
        elif s.bull_bear == "bear":
            buy_mult *= 0.92
            sell_mult *= 1.15
            gate_high += 0.02
            gate_med += 0.01
            hold_mult *= 1.15

        # 3) Trend direction (continuous). When flat in uptrend, penalize HOLD more.
        if spy_r >= 0.02:
            if s.has_positions:
                # 2026-02-27: Don't reduce hold penalty when has positions — encourages SELL when gains exist (intraday profit-taking)
                pass  # hold_mult unchanged; was 0.92x which made HOLD too cheap vs SELL
            else:
                hold_mult *= 1.20
            buy_mult *= 1.06
        elif spy_r <= -0.02:
            hold_mult *= 1.20
            sell_mult *= 1.08
            buy_mult *= 0.95
            stop_mult *= 0.90
            trail_mult *= 0.90

        # 4) Regime label (coarse).
        if s.market_regime == "trending":
            buy_mult *= 1.04
            sell_mult *= 1.04
        elif s.market_regime == "mean_reversion":
            # Slightly higher gating in mean-reversion to avoid chasing noise.
            gate_high += 0.01
            gate_med += 0.005

        # Final clamps (keep consistent with env multipliers)
        buy_mult = _clip(buy_mult, 0.70, 1.60)
        sell_mult = _clip(sell_mult, 0.70, 1.60)
        hold_mult = _clip(hold_mult, 0.50, 1.50)
        gate_high = _clip(gate_high, 0.10, 0.45)
        gate_med = _clip(gate_med, 0.05, 0.30)
        stop_mult = _clip(stop_mult, 0.70, 1.50)
        trail_mult = _clip(trail_mult, 0.70, 1.50)

        return RegimeOutputs(
            buy_reward_mult=float(buy_mult),
            sell_reward_mult=float(sell_mult),
            hold_penalty_mult=float(hold_mult),
            gate_high_threshold=float(gate_high),
            gate_medium_threshold=float(gate_med),
            stop_loss_mult=float(stop_mult),
            trailing_stop_mult=float(trail_mult),
        )

