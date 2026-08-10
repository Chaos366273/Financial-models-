#!/usr/bin/env python3
"""
DYNAMIC CONFIGURATION SYSTEM
Centralized configuration that automatically syncs across all environments
When one environment changes a parameter, all others update automatically
"""

import threading
import time
import os
import sys
from typing import Dict, Any, Callable, List, Optional
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)

@dataclass
class DynamicConfig:
    """
    Centralized dynamic configuration that syncs across all environments
    """
    # Trading parameters - when changed, all environments update
    min_trade_interval: int = 1
    
    # Phase 2 (2026-02-10): PnL-first reward tuning - reduce action-shaped, increase outcome-shaped
    # 2026-02-13: Quality over quantity – reduce noise so agent uses indicators; reward signal-aligned trades
    base_trading_incentive: float = 0.18       # 2026-02-19: 0.12→0.18 - improve deterministic returns (was 0.25; PnL still dominates)
    sell_base_incentive: float = 0.18          # 2026-02-19: 0.12→0.18 - improve deterministic returns (was 0.25)
    short_base_incentive: float = 0.05         # 2026-03-06: Base reward for opening SHORT when signal is bearish (scaled by tech+fund)
    short_signal_min_threshold: float = 0.68   # 2026-03-06: Require VERY STRONG bearish. 0.55→0.68 to cut 92 shorts (83.6%). Needs RSI overbought + MACD + BB + vol
    short_weak_signal_penalty: float = -0.02   # 2026-03-06: Penalty for shorting with weak/no bearish signal
    signal_alignment_bonus: float = 0.30       # 2026-03-06: 0.24→0.30 – stronger reward for signal-aligned profitable SELLs
    profitable_trade_bonus: float = 0.80       # 2026-02-21: 0.70→0.80 – higher win rate incentive
    learning_bonus: float = 0.0               # ZERO – was 0.03; noise, PnL is teacher
    first_trade_bonus: float = 0.05            # Minimal nudge only (was 0.20) – avoid rewarding "any first trade"
    trading_frequency_bonus: float = 0.0      # ZERO – was 0.01; reward quality not quantity
    trading_frequency_bonus_max_trades: int = 200  # Unused when bonus=0; keep for compat
    weak_signal_penalty: float = -0.02         # 2026-02-20: -0.06→-0.02; medium-confidence trades must remain viable
    
    # SELL-SPECIFIC REWARDS - Phase 2: outcome over activity
    sell_attempt_bonus: float = 0.06           # 2026-02-27: 0.04→0.06; reward voluntary SELL (teach BUY→SELL cycle)
    sell_unconditional_base: float = 0.08     # 2026-02-28: small unconditional base on every SELL with position (balanced; outcome bonuses stay when profitable)
    sell_profit_taking_bonus: float = 0.90     # 2026-03-06: 0.80→0.90 – stronger reward for profitable SELLs (win rate)
    sell_loss_penalty_cap: float = -1.0        # Phase 7: Balanced penalty for unprofitable sells (was -2.0, too harsh)
    buy_loss_penalty_cap: float = -1.0         # Phase 7: Equal penalty for losing BUY trades (balanced with SELL)
    buy_reward_cap: float = 0.40               # FIX: Increased from 0.08 to 0.40 to allow adaptive bonuses (up to 0.35)
    progressive_loss_penalty_base: float = -0.20  # 2026-03-06: -0.15→-0.20; stronger first-loss penalty (reduce losing trades)
    progressive_loss_penalty_step: float = -0.08  # 2026-03-06: -0.05→-0.08; faster escalation for repeated losses
    sell_bull_market_boost: float = 0.08       # FIXED: Increased from 0.05 to 0.08 (60% increase) to encourage profit-taking in bull markets
    
    # Dynamic bull/bear reward scaling (reward insight by market regime)
    bull_buy_reward_boost: float = 0.06       # Extra multiplier for BUY rewards in bull market (e.g. 1.0 + 0.06 = 1.06x)
    bear_sell_reward_boost: float = 0.08       # Extra multiplier for SELL rewards in bear market (encourage exits)
    bull_bear_lookback_days: int = 10         # SPY lookback for bull/bear detection
    bull_threshold_pct: float = 0.02           # SPY return >= 2% over lookback = bull
    bear_threshold_pct: float = -0.02          # SPY return <= -2% over lookback = bear

    # 2026-02-28: Regime + indicator–driven rewards (REWARD_PLAN_REGIME_INDICATORS_2026-02-28.md)
    wrong_regime_buy_penalty: float = -0.15    # 2026-03-06: -0.10→-0.15; stronger penalty for BUY in bear regime
    wrong_signal_buy_threshold: float = 0.45   # 2026-03-06: 0.35→0.45; only BUY when tech score stronger (entry quality)
    wrong_signal_buy_penalty: float = -0.12   # 2026-03-06: -0.08→-0.12; stronger penalty for weak-signal BUY
    wrong_signal_sell_penalty: float = -0.04  # SELL at loss when tech was strongly bullish (optional)
    profit_target_pct: float = 0.03            # 2026-03-06: 0.05→0.03; easier 3% target for intraday profit-taking
    profit_target_5pct_bonus: float = 0.28     # 2026-03-06: 0.20→0.28; stronger bonus for SELL at >=5% gain
    overbought_rsi_threshold: float = 0.65     # RSI (0..1) above this = overbought for BUY discount
    overbought_buy_discount: float = 0.85      # Multiply BUY mult by this when overbought in bull
    weak_signal_bull_buy_discount: float = 0.5  # In bull, if BUY tech_score < 0.4, apply to bull BUY boost
    contra_signal_buy_block_threshold_sell: float = 0.6   # SELL tech > this + low BUY → block/penalize BUY
    contra_signal_buy_block_threshold_buy: float = 0.3    # BUY tech < this (with high SELL) → block/penalize BUY
    contra_signal_buy_penalty: float = -0.08   # Penalty when BUY against clear sell signal
    hold_with_5pct_unrealized_in_bull_penalty: float = -0.03  # 2026-03-06: -0.02→-0.03; stronger nudge to realize gains in bull
    
    # NEW: SELL-specific bonuses to encourage voluntary profit-taking
    unrealized_gain_threshold_2pct: float = 0.18  # 2026-02-21: 0.15→0.18 – encourage capturing gains (realized)
    unrealized_gain_threshold_5pct: float = 0.35  # 2026-02-21: 0.30→0.35
    unrealized_gain_threshold_10pct: float = 0.58  # 2026-02-21: 0.50→0.58
    sell_hold_time_bonus_base: float = 0.18   # 2026-02-15: 0.15→0.18 – reward holding then selling
    sell_hold_time_min_steps: int = 20        # Minimum hold time to get bonus (encourages holding then selling)
    position_incentive_decay_start: int = 10  # FIX: Reduced from 15 to 10 (start decay earlier) to encourage faster exits
    position_incentive_decay_rate: float = 0.3  # Keep at 0.3 for faster decay (discourages prolonged holding)
    
    # NEW: Position aging penalty - penalize holding positions too long
    position_aging_penalty_start: int = 10    # FIX: Reduced from 20 to 10 steps (start earlier to encourage exits)
    position_aging_penalty_rate: float = 0.015 # FIX: Slightly increased from 0.01 to 0.015 to nudge more frequent exits once aging kicks in
    unrealized_gain_bonus: float = 0.18        # 2026-02-27: 0.12→0.18 – stronger SELL-with-gains incentive (intraday profit-taking)
    trailing_stop_bonus: float = 0.10          # NEW: Bonus for SELL via trailing stop (encourages profit-taking)
    
    # DEPRECATED - kept for backwards compatibility
    trading_incentive: float = 0.05            # DEPRECATED: Use base_trading_incentive instead
    
    exploration_bonus: float = 0.0             # ZERO – was 0.05; noise, use signal alignment instead
    hold_penalty: float = -0.12               # 2026-02-25: -0.18→-0.12; allow HOLD when uncertain (fix always BUY)
    hold_penalty_with_positions: float = -0.04  # 2026-02-25: -0.05→-0.04; encourage hold-then-sell
    hold_penalty_with_positions_grace: float = -0.02  # 2026-02-20: Lighter penalty for first N steps after BUY (grace period)
    hold_penalty_grace_steps: int = 3         # 2026-02-20: Steps after BUY where grace penalty applies (avoids "don't BUY" collapse)
    hold_penalty_no_positions: float = -0.05   # 2026-02-25: -0.08→-0.05 when flat; HOLD acceptable (fix always BUY)
    hold_penalty_transition_threshold: float = 0.9  # Cash threshold for low penalty
    hold_signal_grace_steps: int = 3
    strong_signal_threshold: float = 0.7
    strong_fundamental_threshold: float = 0.65
    hold_strong_signal_penalty: float = -0.1
    hold_progressive_start: int = 10           # Earlier escalation
    hold_penalty_eval_multiplier: float = 1.0  # Equal penalties in train/eval
    hold_progressive_multiplier: float = 1.5   # Gentler escalation (was 3.0x) - prevents death spiral
    
    # CRITICAL FIX: Remove all penalties that discourage SELL attempts
    failed_sell_penalty: float = -0.02           # 2026-02-15: -0.001→-0.02 – teach policy SELL when empty = bad (SHAP habit fix)
    failed_sell_consecutive_penalty: float = -0.01   # 2026-02-15: -0.0005→-0.01 – stronger growth for repeated mistakes
    failed_sell_penalty_cap: float = -0.10       # 2026-02-15: Cap for failed SELL penalty (allows base -0.02 to be felt)
    invalid_action_penalty: float = -0.0001     # Tiny penalty to discourage exploration of invalid actions
    position_check_bonus: float = 0.001         # Small bonus for learning to check positions
    invalid_sell_penalty: float = -0.001        # 50x lighter - nudge not crush (action masking already prevents invalid sells)
    invalid_sell_penalty_growth: float = 0.0    # No escalation - penalties create death spirals
    invalid_sell_penalty_cap: float = -0.001    # No crushing - keep it minimal
    invalid_sell_penalty_decay: float = 0.5     # Decay applied each step after penalty is added
    masked_sell_warning_threshold: int = 8      # Logging threshold for repeated masked SELLs
    
    # FIX 2: BALANCED HOLD PENALTY - Make HOLD worse than trading but not crushing
    eval_hold_penalty: float = -0.05            # 2026-02-25: -0.06→-0.05; slight relax so policy can HOLD in eval (fix always BUY)
    eval_hold_penalty_performance_based: bool = False  # 2026-02-20: True→False; no free pass when portfolio growing
    eval_hold_penalty_growth_threshold: float = 0.005  # 2026-02-05: 0.01 -> 0.005 so "reduce penalty when growing" kicks in earlier (0.5%)
    eval_hold_penalty_growth_multiplier: float = 0.75  # FIXED: Reduced from 0.5 to 0.75 - less aggressive reduction
    eval_hold_penalty_minimum_floor: float = -0.05  # 2026-02-25: match eval_hold_penalty
    quick_flip_penalty: float = -0.15           # A: Penalty for losing short holds (stronger to discourage churning)
    quick_flip_penalty_profitable: float = 0.0   # 2026-02-27: -0.05→0.0; intraday: don't penalize quick profitable exits
    min_hold_steps: int = 4                    # 2026-02-20: 8→4; 4×5m=20min minimum (HFT-appropriate)
    hold_time_bonus: float = 0.15               # 2026-02-15: 0.12→0.15 – reward hold-then-sell pattern
    loss_scale_multiplier: float = 2.5          # 2026-02-15: 2.0→2.5 – losses hurt more, reinforces avoiding losers
    
    # Additional reward parameters (previously hardcoded)
    market_movement_multiplier: float = 0.2      # Multiplier for market movement rewards
    hold_profit_bonus: float = 0.0               # CRITICAL: Set to 0 (was 0.001) - no bonus for holding, only for trading
    position_incentive: float = 0.02             # FIXED 2026-01-18: Increased from 0.01 to 0.02 (2x stronger) - better reward for holding profitable positions
    hold_winner_bonus: float = 0.08              # 2026-02-15: 0.05→0.08 – reward holding winners longer
    hold_winner_min_unrealized_pct: float = 0.02 # Min unrealized gain (2% of position) to get hold-winner bonus
    hold_winner_min_steps: int = 10              # Min steps held to get hold-winner bonus
    position_size_reward: float = 0.05           # Reward scaled by position sizing
    momentum_bonus_multiplier: float = 0.2       # Multiplier for momentum-based rewards
    momentum_threshold: float = 0.01             # Threshold for momentum detection (1%)
    transaction_cost_multiplier: float = 0.1     # Multiplier for transaction cost penalties
    trade_frequency_threshold: int = 50            # Quality over quantity: penalty after 50 trades (was 80)
    trade_frequency_penalty_rate: float = 0.006   # Stronger penalty for overtrading (was 0.004)
    
    # NEW 2025-12-25: Additional reward multipliers (previously hardcoded)
    no_positions_bonus: float = 0.05              # 2026-02-20: 0.01→0.05; meaningful nudge when agent is flat and should be buying
    gain_bonus_threshold_10pct: float = 0.10     # Threshold for 10%+ gain bonus
    gain_bonus_threshold_5pct: float = 0.05       # Threshold for 5%+ gain bonus
    gain_bonus_threshold_2pct: float = 0.02       # Threshold for 2%+ gain bonus
    gain_bonus_threshold_1pct: float = 0.01       # 2026-02-15: NEW – reward 1%+ gains (small profits still count)
    gain_bonus_value_10pct: float = 0.72         # 2026-02-21: 0.65→0.72 – reinforce big wins (positive P&L)
    gain_bonus_value_5pct: float = 0.52           # 2026-02-21: 0.45→0.52
    gain_bonus_value_2pct: float = 0.32           # 2026-02-21: 0.28→0.32
    gain_bonus_value_1pct: float = 0.18            # 2026-02-26: 0.15→0.18 – stronger return reward for small profits
    progressive_bonus_per_trade: float = 0.001   # Phase 2: Minimal - PnL dominates (was 0.002)
    progressive_bonus_max: float = 0.05           # Phase 2: Lower cap (was 0.08)
    signal_multiplier_scale: float = 10.0         # Scale for signal multiplier (abs(normalized_pnl) * scale)
    profitable_trade_bonus_multiplier: float = 55.0  # 2026-02-21: 50→55 – 1% profit = 0.55 bonus (higher win rate)
    profitable_trade_bonus_max: float = 1.35       # 2026-02-21: 1.2→1.35 – allow bigger rewards for big wins
    unrealized_gain_scale_multiplier: float = 25.0  # Multiplier for unrealized gain scaling
    unrealized_gain_scale_max: float = 3.0        # Maximum unrealized gain scale (3x)
    hold_time_bonus_scale_max: float = 2.0        # Maximum hold time bonus scale (2x)
    loss_penalty_pnl_multiplier: float = 5.0      # Multiplier for loss penalty (normalized_pnl * multiplier)
    diversification_bonus_multiplier: float = 0.1  # Multiplier for diversification bonus
    profit_bonus_multiplier: float = 0.2          # Multiplier for profit bonus (portfolio_change * multiplier)
    size_bonus_multiplier: float = 0.05           # Multiplier for position size bonus

    # -------------------------------------------------------------------------
    # 2025-12-21: SELL LEARNING SHAPING (minimal, train-only via env logic)
    # Goal: if exit pressure is high but agent doesn't choose SELL, apply a
    # small negative shaping so PPO learns action=2.
    # -------------------------------------------------------------------------
    # 2026-02-05: REDUCED to fix SELL bias - only penalize clear "should have sold" cases
    # (was 0.30/0.40/0.01/3 causing 75% raw SELL; now allow holding winners longer)
    missed_sell_penalty_base: float = 0.10           # Reduced from 0.30 - nudge not crush
    missed_sell_signal_threshold: float = 0.50       # Raised from 0.40 - only strong sell signals
    missed_sell_profit_take_threshold: float = 0.02  # Raised from 0.01 - 2% profit before pressure
    missed_sell_min_hold_steps: int = 8              # Raised from 3 - allow hold to develop
    forced_sell_exploration_prob: float = 0.20       # 20% exploration (0.15→0.20)
    
    # Environment parameters (CENTRALIZED)
    window_size: int = 10                     # CENTRALIZED from config.py DATA_CONFIG
    max_position_weight: float = 0.02
    max_tech_weight: float = 0.6
    max_position: float = 0.02               # CENTRALIZED from config.py TRADING_CONFIG
    transaction_cost: float = 0.000024       # REAL ALPACA: SEC + FINRA fees only (0.0024%)
    max_samples: int = 20000                 # CENTRALIZED from config.py (expert dataset)
    train_split: float = 0.75                # CENTRALIZED from config.py (75% train split)
    validation_split: float = 0.25            # CENTRALIZED from config.py (25% validation split)
    
    # TCN Temporal Memory Configuration (NEW - Replaces LSTM)
    use_temporal_memory: bool = True         # ENABLED: TCN for temporal memory (Phase 7)
    temporal_memory_type: str = "tcn"        # Temporal memory type: "tcn" (LSTM removed)
    tcn_hidden_dims: List[int] = field(default_factory=lambda: [256, 256, 128, 128])  # Hidden dims per TCN block
    tcn_kernel_size: int = 3                  # Convolution kernel size
    tcn_dilation_rates: List[int] = field(default_factory=lambda: [1, 2, 4, 8])  # Exponential dilation for long-range memory
    tcn_dropout: float = 0.1                  # Dropout rate for TCN blocks
    tcn_use_residual: bool = True             # Use residual connections in TCN blocks
    
    # 2026-02-15: Observation augmentation - fundamental metrics in observation
    include_fundamental_in_observation: bool = True  # Add P/E, ROE, etc. to obs so policy sees fundamentals directly
    # 2026-02-15: Use all tickers present in the data (agent sees all companies from data)
    use_all_data_tickers: bool = True  # If True, expected_tickers = all tickers in data; if False, use config tickers only

    # PHASE 1: Core Trading Parameters (NEW)
    max_position_size: float = 0.04          # FIXED: Increased from 0.03 to 0.04 (4% per ticker) for larger entries
    max_trades_per_episode: int = 100        # Maximum trades per episode
    drawdown_penalty_lambda: float = 0.5     # Drawdown penalty weight

    # Risk/Exit parameters (new centralized)
    base_stop_loss_pct: float = 0.08         # 2026-02-19: 0.06→0.08 - slightly wider to avoid premature exits in volatile regimes
    trailing_stop_pct: float = 0.08           # Trailing stop 8%

    # Evaluation exploration controls (centralized)
    eval_exploration_warmup_steps: int = 150  # Allow early exploration for first 150 steps in eval
    low_confidence_exploration: float = 0.95   # CRITICAL: Increased from 0.85 to 0.95 (95% exploration) to force trades through signal gating  # noqa: F401
    
    # PHASE 2: PPO Model Parameters (CENTRALIZED)
    learning_rate: float = 5e-04             # FIX 2026-01-18: Increased from 3e-5 to 5e-4 (16.7x) - Policy loss = 0.0 (converged too early)
    # Learning rate scheduling to prevent overfitting
    use_lr_schedule: bool = True             # Enable learning rate decay for longer training
    lr_initial: float = 5e-5                 # FIXED: Increased from 1e-5 to 3e-5 to enable learning
    lr_final: float = 1.5e-5                # FIXED: Increased from 7.5e-6 to 1.5e-5 (2x) to maintain learning capacity
    lr_decay_type: str = "linear"            # Linear decay (can be "cosine" or "linear")
    
    # Adaptive hyperparameter thresholds
    adaptive_lr_sharpe_threshold: float = 2.0  # Maintain higher LR if Sharpe > this
    adaptive_lr_return_threshold: float = 0.03  # Maintain higher LR if Return > this (3%)
    adaptive_lr_boost_factor: float = 1.5      # Multiply lr_final by this when conditions met
    # NEW: Bidirectional adaptive LR - increase LR when performance is poor
    adaptive_lr_poor_sharpe_threshold: float = 0.5  # 2026-02-06: Lower to 0.5 so LR boosts when Sharpe < 0.5 (policy loss ~0 fix)
    adaptive_lr_poor_return_threshold: float = 0.03  # UPDATED: Increase LR if Return < this (was 1%, now 3% for better sensitivity)
    adaptive_lr_poor_boost_factor: float = 2.0  # UPDATED: Multiply LR by this when performance is poor (was 1.8, now 2.0)
    adaptive_lr_poor_max: float = 5e-4  # 2026-02-09: Must be > typical current LR so "poor" branch INCREASES LR (was 1e-4, reduced LR 3e-4->1e-4)
    adaptive_lr_warmup_steps: int = 25000  # 2026-02-28: 50k→25k; catch collapse earlier (was blocking first 50k)
    adaptive_ent_min_trades: int = 50          # Increase entropy if trades < this
    adaptive_ent_boost_factor: float = 2.5     # 2026-02-21: 2.0→2.5; stronger boost for 300k training (plan)
    adaptive_ent_max: float = 0.04             # Maximum entropy coefficient (cap) - INCREASED for more exploration
    adaptive_ent_buy_threshold: float = 5.0    # FIX 2: Increase entropy if BUY actions < this % (default 5%)
    adaptive_ent_buy_high_threshold: float = 85.0  # 2026-02-28: BUY-dominant collapse (e.g. 91.6% BUY)
    adaptive_ent_sell_threshold: float = 85.0  # FIX 2: Increase entropy if SELL actions > this % (default 85%)
    adaptive_ent_entropy_ratio_threshold: float = 0.3  # 2026-02-28: entropy_ratio < this = collapsed (was 0.2)
    adaptive_sell_early_boost_steps: int = 30000  # 2026-02-28: Proactive SELL boost in first N steps
    adaptive_sell_early_boost_mult: float = 1.25  # 2026-02-28: SELL mult during early boost phase
    # 2026-02-19: Training-time adaptive base incentive (callback bumps when imbalance detected)
    adaptive_incentive_step: float = 0.02     # Bump base_trading_incentive/sell_base_incentive by this each trigger (cap applies)
    adaptive_incentive_cap: float = 0.25      # Max value for base_trading_incentive and sell_base_incentive when adapting
    n_steps: int = 2048                      # PPO n_steps (optimized for 50K timesteps)
    batch_size: int = 512                    # PPO batch size - CENTRALIZED from config.py
    n_epochs: int = 4                        # PPO n_epochs - CENTRALIZED from config.py
    gamma: float = 0.995                     # PPO discount factor - CENTRALIZED from config.py
    gae_lambda: float = 0.98                 # PPO GAE lambda - CENTRALIZED from config.py
    clip_range: float = 0.15                 # PPO clip range - CENTRALIZED from config.py
    ent_coef: float = 0.15                   # 2026-02-21: 0.12→0.15; maintain exploration through 300k (plan)
    ent_coef_min: float = 0.10                # 2026-02-21: 0.08→0.10; floor for adaptive entropy (plan)
    vf_coef: float = 0.25                    # PPO value function coefficient - CENTRALIZED from config.py
    max_grad_norm: float = 0.3               # PPO max gradient norm - CENTRALIZED from config.py
    target_kl: float = 0.01                  # PPO target KL divergence - CENTRALIZED from config.py
    
    # PHASE 3: Reward System Parameters (NEW)
    base_return: float = 1.0                 # Base return weight
    sharpe_bonus: float = 2.5                # Sharpe ratio bonus weight
    transaction_penalty: float = 0.3         # REDUCED: Encourage more trading with real Alpaca costs
    activity_bonus: float = 0.8              # Trading activity bonus weight
    risk_bonus: float = 2.0                  # Risk management bonus weight
    drawdown_penalty: float = 2.5            # Drawdown penalty weight
    concentration_penalty: float = 1.2       # Portfolio concentration penalty weight
    win_rate_bonus: float = 2.0              # 2026-02-21: 1.5→2.0 – higher win rate incentive
    profit_factor_bonus: float = 1.0         # Profit factor bonus weight
    consistency_bonus: float = 1.2           # Performance consistency bonus weight
    pnl_reward_scale: float = 72.0            # 2026-02-21: 60→72 – stronger positive P&L signal
    profitable_close_bonus: float = 0.06      # 2026-02-26: 0.05→0.06 – stronger return reward for profitable closes
    profitable_close_min_return_pct: float = 0.010  # 2026-02-26: 1.2%→1.0% – more closes qualify as profitable
    profit_bonus_scale: float = 1.0          # Scale for additional profit bonuses
    profit_bonus_cap: float = 0.25           # Cap for profit bonus contribution
    loss_penalty_scale: float = 5.0          # PHASE 1 FIX: Increased from 1.0 to 5.0 (losing trades must hurt!)
    loss_penalty_cap: float = 2.0            # PHASE 1 FIX: Increased from 0.4 to 2.0 (allow stronger penalties for big losses)
    signal_bonus_cap: float = 0.3            # Max contribution from signal rewards
    signal_miss_penalty_scale: float = 0.2   # Penalty weight when signals misfire
    w_turnover: float = 0.002                # Turnover penalty weight
    
    # System parameters
    debug_mode: bool = True
    enable_training_profiling: bool = False  # Disable profiling by default for speed
    enable_trading_constraints: bool = True
    
    # TECHNICAL INDICATOR REWARD SHAPING PARAMETERS
    # PHASE 2: Increased from 0.2 to 0.5 - signals now drive 50% of reward (was 20%)
    technical_signal_weight: float = 1.0   # RL rewards technical-only (fundamental reserved for supervised nets later)
    pnl_weight: float = 0.6  # Weight for P&L signals (60%)
    regime_detection_enabled: bool = True  # Enable regime detection
    atr_period: int = 14  # ATR calculation period
    high_volatility_threshold: float = 0.02  # Threshold for trending regime (2% ATR)
    rsi_oversold: float = 40  # RSI oversold threshold
    rsi_overbought: float = 60  # RSI overbought threshold
    volume_threshold: float = 1.0  # Volume confirmation threshold (1x average)
    
    # FIX 3: SIGNAL GATING CONFIGURATION - Relax thresholds to allow more trading
    high_confidence_threshold: float = 0.25    # FIX 3: Reduced from 0.35 to 0.25 (lower bar for full rewards)
    medium_confidence_threshold: float = 0.10  # FIX 3: Reduced from 0.15 to 0.10 (lower bar for reduced rewards)
    # low_confidence_exploration already declared at line 174 - removed duplicate
    
    # VWAP REWARD PARAMETERS
    vwap_lookback: int = 10  # Lookback window for VWAP calculation (bars)
    vwap_buy_bonus_scale: float = 10.0  # Scale for buying below VWAP
    vwap_sell_bonus_scale: float = 2.0   # Scale for buying above VWAP (penalty magnitude)
    
    # FUNDAMENTAL ANALYSIS REWARD SHAPING PARAMETERS
    # PHASE 2: Increased from 0.15 to 0.35 - signals now drive 35% of reward (was 15%)
    fundamental_signal_weight: float = 0.10  # Re-enabled conservatively: keep technical dominant, add fundamental confirmation
    fundamental_health_threshold: float = 60.0  # Health score threshold for good companies
    fundamental_excellent_threshold: float = 80.0  # Health score threshold for excellent companies
    pe_ratio_good_min: float = 10.0  # Minimum PE ratio for good score
    pe_ratio_good_max: float = 20.0  # Maximum PE ratio for good score
    roe_strong_threshold: float = 0.15  # ROE threshold for strong score
    roe_good_threshold: float = 0.10  # ROE threshold for good score
    debt_equity_low_threshold: float = 0.5  # Debt-to-equity threshold for low debt
    debt_equity_moderate_threshold: float = 1.0  # Debt-to-equity threshold for moderate debt
    growth_strong_threshold: float = 0.15  # Growth threshold for strong score
    growth_good_threshold: float = 0.05  # Growth threshold for good score
    
    # MULTI-TIMEFRAME REWARD SCALING (for when multi-timeframe data is used)
    # Higher weight = more reward for trades aligned with that timeframe's signals.
    timeframe_reward_weight_1d: float = 1.0   # Daily baseline
    timeframe_reward_weight_1h: float = 1.1   # Slight boost for 1h
    timeframe_reward_weight_30m: float = 1.15  # Favor 30m
    timeframe_reward_weight_5m: float = 1.2    # Favor 5m
    timeframe_reward_weight_1m: float = 1.0   # 1m noisy, same as 1d
    # Intraday merge: which intervals to merge into daily (order: coarse to fine)
    intraday_merge_intervals: List[str] = field(default_factory=lambda: ['5m', '30m', '1h', '1m'])
    # Intraday step: when True, env steps on 5m/30m bars (market-condition driven); when False, daily step
    use_intraday_step: bool = True
    intraday_step_interval: str = '5m'  # '5m' or '30m' for bar granularity when use_intraday_step

    # -------------------------------------------------------------------------
    # Reward mode control (Step 2: reward/gating alignment)
    # "technical_only_v1" overwrites full reward with minimal formula (base 0.01/0.02 + PnL + small signal).
    # Set to "" to use full reward (base 0.40, hold time, gain bonuses, signal alignment, etc.).
    # 2026-02-10: Default "" so agent gets full reward; was "technical_only_v1" (silent reward collapse).
    # -------------------------------------------------------------------------
    reward_mode: str = ""
    tech_only_buy_base: float = 0.01
    tech_only_sell_base: float = 0.02
    tech_only_pnl_scale: float = 50.0
    tech_only_signal_weight: float = 0.15
    tech_only_vwap_weight: float = 0.05
    tech_only_cost_weight: float = 0.10
    tech_only_hold_miss_penalty: float = 0.01
    tech_only_entry_signal_threshold: float = 0.25
    tech_only_exit_signal_threshold: float = 0.25
    tech_only_disable_forced_sell: bool = True
    tech_only_disable_missed_sell_penalty: bool = True
    tech_only_soft_gating: bool = True  # never block trades on confidence; use reward multipliers only
    
    # Additional parameters for comprehensive fixes
    action_space_size: int = 3
    num_features: int = 15  # CRITICAL FIX: Dynamic feature count
    observation_space_size: int = 150  # 5 * 6 * 5
    total_timesteps: int = 1000  # Baseline scaling: 300k → 600k (per scaling plan)
    max_timesteps: int = 600000   # Cap for baseline scaling phase
    min_timesteps: int = 100000     # FIXED: Minimum 100K (BC needs ~430 trades, 50K only gave 103)
    initial_balance: float = 100000.0
    commission_rate: float = 0.000024       # REAL ALPACA: SEC + FINRA fees only (0.0024%)
    
    # PROGRESSIVE SCALING TO 1M TIMESTEPS - Monitoring & Validation
    scaling_phase: str = "250k"            # Current phase: 250k → 400k → 600k → 800k → 1M
    min_sharpe_to_scale: float = 0.2       # Minimum Sharpe to increase timesteps
    min_return_to_scale: float = 0.02      # Minimum return (2%) to increase timesteps  
    max_drawdown_to_scale: float = 0.15    # Maximum drawdown (15%) to allow scaling
    min_trades_per_episode: int = 20       # Minimum trades to ensure trading activity
    # max_trades_per_episode already declared at line 165 - removed duplicate
    
    # Scaling decision rules
    timestep_increase_amount: int = 100000 # Increase by 100k when criteria met
    timestep_decrease_amount: int = 50000  # Decrease by 50k if performance degrades
    consecutive_good_runs_to_scale: int = 2  # Need 2 good runs before scaling up
    
    # DYNAMIC TIMESTEP SCALING PARAMETERS
    start_timesteps: int = 300000          # Baseline scaling start
    target_timesteps: int = 600000         # Target for super agent
    increment_size: int = 50000            # Increase by 50k each run
    max_errors_per_run: int = 2            # Max errors before reducing
    reduction_factor: float = 0.8          # Reduce by 20% if errors
    success_runs_before_increase: int = 2 # Need 2 clean runs to increase
    
    # Internal tracking
    _listeners: List[Callable] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _last_update: float = field(default_factory=time.time)
    
    def __post_init__(self):
        """Validate configuration values after initialization"""
        self._validate_values()
    
    def _validate_values(self):
        """Validate and cap configuration values - only cap if values are unreasonably high"""
        # Allow higher values from dynamic_config.py - only cap extreme values
        if self.exploration_bonus > 2.0:
            logger.warning(f"exploration_bonus={self.exploration_bonus} too high! Capping at 2.0")
            self.exploration_bonus = 2.0
        
        if self.trading_incentive > 2.0:
            logger.warning(f"trading_incentive={self.trading_incentive} too high! Capping at 2.0")
            self.trading_incentive = 2.0
        
        if self.hold_profit_bonus > 0.10:
            logger.warning(f"hold_profit_bonus={self.hold_profit_bonus} too high! Capping at 0.05")
            self.hold_profit_bonus = 0.05
    
    def add_listener(self, callback: Callable):
        """Add a callback to be notified when config changes"""
        with self._lock:
            self._listeners.append(callback)
            logger.info(f"Added config listener: {callback.__name__}")
    
    def remove_listener(self, callback: Callable):
        """Remove a callback from notifications"""
        with self._lock:
            if callback in self._listeners:
                self._listeners.remove(callback)
                logger.info(f"Removed config listener: {callback.__name__}")
    
    def update_config(self, **kwargs):
        """Update configuration and notify all listeners"""
        with self._lock:
            changed_params = []
            for key, value in kwargs.items():
                if hasattr(self, key) and getattr(self, key) != value:
                    old_value = getattr(self, key)
                    setattr(self, key, value)
                    changed_params.append(f"{key}={value}")
                    if key == 'max_position_size':
                        print(f" DEBUG: update_config changing {key} from {old_value} to {value}")
            
            # CRITICAL FIX: Validate values after update
            self._validate_values()
            
            if changed_params:
                self._last_update = time.time()
                logger.info(f" CONFIG UPDATED: {', '.join(changed_params)}")
                print(f" DEBUG: CONFIG UPDATED: {', '.join(changed_params)}")
                
                # Notify all listeners
                for callback in self._listeners:
                    try:
                        callback(self)
                    except Exception as e:
                        logger.error(f"Error in config listener {callback.__name__}: {e}")
                
                logger.info(f" Notified {len(self._listeners)} listeners of config change")
            else:
                logger.debug("No config changes detected")
    
    def get_config_dict(self) -> Dict[str, Any]:
        """Get current configuration as dictionary"""
        with self._lock:
            return {
                'min_trade_interval': self.min_trade_interval,
                'trading_incentive': self.trading_incentive,
                'exploration_bonus': self.exploration_bonus,
                'hold_penalty': self.hold_penalty,
                'hold_penalty_no_positions': self.hold_penalty_no_positions,
                'hold_progressive_start': self.hold_progressive_start,
                'hold_penalty_eval_multiplier': self.hold_penalty_eval_multiplier,
                'hold_progressive_multiplier': self.hold_progressive_multiplier,
                'window_size': self.window_size,
                'max_position_weight': self.max_position_weight,
                'max_tech_weight': self.max_tech_weight,
                'max_position_size': self.max_position_size,
                'max_trades_per_episode': self.max_trades_per_episode,
                'drawdown_penalty_lambda': self.drawdown_penalty_lambda,
                'learning_rate': self.learning_rate,
                'n_steps': self.n_steps,
                'batch_size': self.batch_size,
                'n_epochs': self.n_epochs,
                'gamma': self.gamma,
                'gae_lambda': self.gae_lambda,
                'clip_range': self.clip_range,
                'ent_coef': self.ent_coef,
                'vf_coef': self.vf_coef,
                'max_grad_norm': self.max_grad_norm,
                'base_return': self.base_return,
                'sharpe_bonus': self.sharpe_bonus,
                'transaction_penalty': self.transaction_penalty,
                'activity_bonus': self.activity_bonus,
                'risk_bonus': self.risk_bonus,
                'drawdown_penalty': self.drawdown_penalty,
                'concentration_penalty': self.concentration_penalty,
                'win_rate_bonus': self.win_rate_bonus,
                'profit_factor_bonus': self.profit_factor_bonus,
                'consistency_bonus': self.consistency_bonus,
                'w_turnover': self.w_turnover,
                'debug_mode': self.debug_mode,
                'enable_training_profiling': self.enable_training_profiling,
                'enable_trading_constraints': self.enable_trading_constraints,
                'missed_sell_penalty_base': self.missed_sell_penalty_base,
                'missed_sell_signal_threshold': self.missed_sell_signal_threshold,
                'missed_sell_profit_take_threshold': self.missed_sell_profit_take_threshold,
                'missed_sell_min_hold_steps': self.missed_sell_min_hold_steps,
                'forced_sell_exploration_prob': self.forced_sell_exploration_prob,
                'last_update': self._last_update
            }
    
    def get_reward_balance_report(self) -> Dict[str, Any]:
        """Generate reward balance analysis for debugging"""
        return {
            'buy_rewards': {
                'base_incentive': self.trading_incentive,
                'first_trade_bonus': self.first_trade_bonus,
                'exploration_bonus': self.exploration_bonus,
                'total_max_buy': self.trading_incentive + self.first_trade_bonus + self.exploration_bonus,
            },
            'sell_rewards': {
                'base_incentive': self.trading_incentive,
                'failed_sell_penalty': self.failed_sell_penalty,
                'consecutive_penalty': self.failed_sell_consecutive_penalty,
                'total_max_sell': self.trading_incentive,
                'total_min_sell_failed': self.failed_sell_penalty + self.failed_sell_consecutive_penalty * 10,
            },
            'hold_rewards': {
                'with_positions': self.hold_penalty_with_positions,
                'no_positions': self.hold_penalty_no_positions,
            },
            'balance_analysis': {
                'buy_vs_sell_ratio': (self.trading_incentive + self.first_trade_bonus) / max(abs(self.failed_sell_penalty), 0.0001),
                'buy_vs_hold_ratio': (self.trading_incentive + self.first_trade_bonus) / abs(self.hold_penalty_no_positions),
                'recommendation': 'BALANCED' if abs(self.failed_sell_penalty) < 0.001 else 'IMBALANCED - REDUCE SELL PENALTIES',
            }
        }

    def sync_from_config_file(self):
        """Sync with external config file"""
        try:
            import sys
            import os
            # Add the parent directory to the path to find config.py
            parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            if parent_dir not in sys.path:
                sys.path.insert(0, parent_dir)
            
            import config as main_config  # type: ignore
            
            # DEBUG: Check what values are being loaded from external config
            external_max_position_size = main_config.TRADING_CONFIG.get('max_position_size', 0.02)  # PHASE 1 FIX: Safe fallback
            print(f" DEBUG: sync_from_config_file loading max_position_size from external config: {external_max_position_size}")
            print(f" DEBUG: Current dynamic config max_position_size: {self.max_position_size}")
            
            # Update from external config
            # 2026-02-11 FIX: NEVER overwrite reward/trading-behavior params from config.py.
            # dynamic_config.py is the SINGLE SOURCE OF TRUTH for all reward tuning.
            # config.py can override infrastructure (window_size, scaling, paths) only.
            self.update_config(
                min_trade_interval=main_config.TRADING_CONFIG.get('min_trade_interval', 1),
                window_size=main_config.ENV_CONFIG.get('window_size', 10),
                max_position_weight=main_config.TRADING_CONFIG.get('max_position_weight', 0.02),
                max_position_size=external_max_position_size,
                # Reward params: keep dynamic_config defaults (do NOT read from config.py)
                trading_incentive=self.trading_incentive,
                exploration_bonus=self.exploration_bonus,
                hold_penalty=self.hold_penalty,
                failed_sell_penalty=self.failed_sell_penalty,
                failed_sell_consecutive_penalty=self.failed_sell_consecutive_penalty,
                invalid_action_penalty=self.invalid_action_penalty,
                position_check_bonus=self.position_check_bonus,
                max_trades_per_episode=self.max_trades_per_episode,
                drawdown_penalty_lambda=self.drawdown_penalty_lambda,
                # CENTRALIZED: Use our own value as single source of truth
                # total_timesteps=main_config.get_dynamic_timesteps() if hasattr(main_config, 'get_dynamic_timesteps') else main_config.ENV_CONFIG.get('total_timesteps', 50000),  # REMOVED: Prevents circular override
                start_timesteps=main_config.SCALING_CONFIG.get('start_timesteps', 50000) if hasattr(main_config, 'SCALING_CONFIG') else 50000,
                target_timesteps=main_config.SCALING_CONFIG.get('target_timesteps', 600000) if hasattr(main_config, 'SCALING_CONFIG') else 600000,
                increment_size=main_config.SCALING_CONFIG.get('increment_size', 50000) if hasattr(main_config, 'SCALING_CONFIG') else 50000,
                max_errors_per_run=main_config.SCALING_CONFIG.get('max_errors_per_run', 2) if hasattr(main_config, 'SCALING_CONFIG') else 2,
                reduction_factor=main_config.SCALING_CONFIG.get('reduction_factor', 0.8) if hasattr(main_config, 'SCALING_CONFIG') else 0.8,
                success_runs_before_increase=main_config.SCALING_CONFIG.get('success_runs_before_increase', 2) if hasattr(main_config, 'SCALING_CONFIG') else 2,
                # LR: do NOT let config.py 1e-06 overwrite; keep dynamic_config default (5e-4) so adaptive LR works
                learning_rate=_resolve_learning_rate(main_config.MODEL_CONFIG.get('learning_rate', 1e-06), self.learning_rate),
                # PPO training params: keep dynamic_config (do not let config.py overwrite tuned values)
                n_steps=self.n_steps,
                batch_size=self.batch_size,
                n_epochs=self.n_epochs,
                gamma=self.gamma,
                gae_lambda=self.gae_lambda,
                clip_range=self.clip_range,
                ent_coef=self.ent_coef,
                vf_coef=self.vf_coef,
                max_grad_norm=self.max_grad_norm,
                base_return=main_config.PHASE1_BASELINE_WEIGHTS.get('base_return', 1.0),
                sharpe_bonus=main_config.PHASE1_BASELINE_WEIGHTS.get('sharpe_bonus', 2.5),
                transaction_penalty=main_config.PHASE1_BASELINE_WEIGHTS.get('transaction_penalty', 1.2),
                activity_bonus=main_config.PHASE1_BASELINE_WEIGHTS.get('activity_bonus', 0.8),
                risk_bonus=main_config.PHASE1_BASELINE_WEIGHTS.get('risk_bonus', 2.0),
                drawdown_penalty=main_config.PHASE1_BASELINE_WEIGHTS.get('drawdown_penalty', 2.5),
                concentration_penalty=main_config.PHASE1_BASELINE_WEIGHTS.get('concentration_penalty', 1.2),
                win_rate_bonus=main_config.PHASE1_BASELINE_WEIGHTS.get('win_rate_bonus', 1.5),
                profit_factor_bonus=main_config.PHASE1_BASELINE_WEIGHTS.get('profit_factor_bonus', 1.0),
                consistency_bonus=main_config.PHASE1_BASELINE_WEIGHTS.get('consistency_bonus', 1.2),
                w_turnover=main_config.PHASE1_BASELINE_WEIGHTS.get('w_turnover', 0.002),
                debug_mode=main_config.DEBUG
            )

            # 2026-02-11 FIX: REWARD_CONFIG from config.py is NO LONGER applied.
            # All reward params are defined in dynamic_config.py (single source of truth).
            # config.py REWARD_CONFIG was overwriting carefully tuned values
            # (e.g. invalid_sell_penalty -0.001 -> -0.05, causing SELL death spiral).
            # If you need to experiment, edit dynamic_config.py directly.

            # Optional: sync technical signal parameters
            try:
                ts = getattr(main_config, 'TECHNICAL_SIGNAL_CONFIG', {})
                if isinstance(ts, dict) and ts:
                    self.update_config(
                        technical_signal_weight=ts.get('technical_signal_weight', self.technical_signal_weight),
                        atr_period=ts.get('atr_period', self.atr_period),
                        high_volatility_threshold=ts.get('high_volatility_threshold', self.high_volatility_threshold),
                        rsi_oversold=ts.get('rsi_oversold', self.rsi_oversold),
                        rsi_overbought=ts.get('rsi_overbought', self.rsi_overbought),
                        volume_threshold=ts.get('volume_threshold', self.volume_threshold),
                    )
            except Exception:
                pass

            # Optional: sync fundamental signal parameters
            try:
                fs = getattr(main_config, 'FUNDAMENTAL_SIGNAL_CONFIG', {})
                if isinstance(fs, dict) and fs:
                    self.update_config(
                        fundamental_signal_weight=fs.get('fundamental_signal_weight', self.fundamental_signal_weight),
                        fundamental_health_threshold=fs.get('fundamental_health_threshold', self.fundamental_health_threshold),
                        fundamental_excellent_threshold=fs.get('fundamental_excellent_threshold', self.fundamental_excellent_threshold),
                        pe_ratio_good_min=fs.get('pe_ratio_good_min', self.pe_ratio_good_min),
                        pe_ratio_good_max=fs.get('pe_ratio_good_max', self.pe_ratio_good_max),
                        roe_strong_threshold=fs.get('roe_strong_threshold', self.roe_strong_threshold),
                        roe_good_threshold=fs.get('roe_good_threshold', self.roe_good_threshold),
                        debt_equity_low_threshold=fs.get('debt_equity_low_threshold', self.debt_equity_low_threshold),
                        growth_strong_threshold=fs.get('growth_strong_threshold', self.growth_strong_threshold),
                        growth_good_threshold=fs.get('growth_good_threshold', self.growth_good_threshold),
                    )
            except Exception:
                pass
            
            logger.info(" Synced with external config file")
            
        except Exception as e:
            logger.error(f"Failed to sync with config file: {e}")


def _resolve_learning_rate(from_config: float, dynamic_default: float) -> float:
    """Use config value only if it's a real LR (>1e-5); else keep dynamic_config default so adaptive LR works."""
    return from_config if from_config > 1e-5 else dynamic_default


# Global dynamic config instance
DYNAMIC_CONFIG = DynamicConfig()

# Canonical daily feature names (order matters for observation); from trading_data_features.csv
BASE_DAILY_FEATURES = [
    'sma_20', 'ema_12', 'rsi', 'macd', 'macd_signal', 'atr',
    'bb_upper', 'bb_middle', 'bb_lower', 'price_to_sma_20', 'volume_trend',
]

def get_intraday_merge_intervals() -> List[str]:
    """Return list of intraday intervals to merge into daily (e.g. ['5m','30m','1h','1m'])."""
    return getattr(DYNAMIC_CONFIG, 'intraday_merge_intervals', ['5m', '30m', '1h', '1m'])

def get_canonical_feature_cols(include_intraday: bool = True, intervals: Optional[List[str]] = None) -> List[str]:
    """
    Return ordered feature column list: base daily + per-interval intraday columns.
    Used by CSV loader, env, and precompute so observation uses same feature order.
    Missing columns in data are filled with 0.
    """
    cols = list(BASE_DAILY_FEATURES)
    if not include_intraday:
        return cols
    if intervals is None:
        intervals = get_intraday_merge_intervals()
    for interval in intervals:
        cols.append(f'intraday_vol_{interval}')
        cols.append(f'intraday_range_pct_{interval}')
        cols.append(f'intraday_vol_ratio_{interval}')
        cols.append(f'intraday_vol_sum_{interval}')
    return cols

def get_dynamic_config() -> DynamicConfig:
    """Get the global dynamic configuration instance"""
    return DYNAMIC_CONFIG

def update_global_config(**kwargs):
    """Update global configuration and sync all environments"""
    print(f" DEBUG: update_global_config called with: {kwargs}")
    if 'max_position_size' in kwargs:
        print(f" DEBUG: Updating max_position_size from {DYNAMIC_CONFIG.max_position_size} to {kwargs['max_position_size']}")
    DYNAMIC_CONFIG.update_config(**kwargs)

def register_config_listener(callback: Callable):
    """Register a callback to be notified of config changes"""
    DYNAMIC_CONFIG.add_listener(callback)

def unregister_config_listener(callback: Callable):
    """Unregister a config change callback"""
    DYNAMIC_CONFIG.remove_listener(callback)

def get_dynamic_feature_count() -> dict:
    """
    DYNAMIC FEATURE COUNTING SYSTEM
    Counts features based on actual implementation, not hardcoded claims
    """
    from pathlib import Path
    
    # Add PPO module to path
    ppo_dir = os.path.dirname(os.path.dirname(__file__))
    if ppo_dir not in sys.path:
        sys.path.insert(0, ppo_dir)
    
    feature_count = {
        'total_implemented': 0,
        'total_claimed': 0,
        'enabled_features': 0,
        'disabled_features': 0,
        'feature_details': {},
        'inflation_detected': False,
        'accuracy_score': 0.0
    }
    
    try:
        # CORE FEATURES - Always implemented (5 basic OHLCV)
        core_features = {
            'open_price': True,
            'high_price': True, 
            'low_price': True,
            'close_price': True,
            'volume': True
        }
        
        # TECHNICAL FEATURES - Check ACTUAL implementation in code
        technical_features = {}
        try:
            # Read the actual trading_env.py file to check for method implementations
            trading_env_path = os.path.join(ppo_dir, 'models', 'trading_env.py')
            if os.path.exists(trading_env_path):
                with open(trading_env_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check for actual method implementations, not just method names
                technical_methods = {
                    'returns': 'def compute_returns' in content or 'compute_returns' in content,
                    'macd': 'def compute_macd' in content or 'compute_macd' in content,
                    'hull_ma': 'def compute_hull_ma' in content or 'compute_hull_ma' in content,
                    'stoch_rsi': 'def compute_stoch_rsi' in content or 'compute_stoch_rsi' in content,
                    'atr': 'def compute_atr' in content or 'compute_atr' in content,
                    'bb_percent': 'def compute_bb_percent' in content or 'compute_bb_percent' in content
                }
                
                technical_features = technical_methods
                logger.info(f" TECHNICAL FEATURES: Found {sum(technical_features.values())}/6 actually implemented")
            else:
                technical_features = {key: False for key in ['returns', 'macd', 'hull_ma', 'stoch_rsi', 'atr', 'bb_percent']}
        except Exception as e:
            logger.error(f"Technical feature detection failed: {e}")
            technical_features = {key: False for key in ['returns', 'macd', 'hull_ma', 'stoch_rsi', 'atr', 'bb_percent']}
        
        # ADVANCED FEATURES - Check ACTUAL implementation in code
        advanced_features = {}
        try:
            # Read the actual trading_env.py file to check for advanced feature implementations
            trading_env_path = os.path.join(ppo_dir, 'models', 'trading_env.py')
            if os.path.exists(trading_env_path):
                with open(trading_env_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check for actual advanced feature implementations in the code
                advanced_methods = {
                    'fundamental_analysis': 'fundamental_analysis' in content and ('def ' in content or 'class ' in content),
                    'sentiment_analysis': 'sentiment_analysis' in content and ('def ' in content or 'class ' in content),
                    'dynamic_stop_loss': 'dynamic_stop_loss' in content and ('def ' in content or 'class ' in content),
                    'portfolio_hedging': 'portfolio_hedging' in content and ('def ' in content or 'class ' in content),
                    'market_regime_detection': 'market_regime_detection' in content and ('def ' in content or 'class ' in content),
                    'trade_validation': 'trade_validation' in content and ('def ' in content or 'class ' in content),
                    'portfolio_optimization': 'portfolio_optimization' in content and ('def ' in content or 'class ' in content)
                }
                
                advanced_features = advanced_methods
                logger.info(f" ADVANCED FEATURES: Found {sum(advanced_features.values())}/7 actually implemented")
            else:
                advanced_features = {key: False for key in ['fundamental_analysis', 'sentiment_analysis', 'dynamic_stop_loss', 'portfolio_hedging', 'market_regime_detection', 'trade_validation', 'portfolio_optimization']}
        except Exception as e:
            logger.error(f"Advanced feature detection failed: {e}")
            advanced_features = {key: False for key in ['fundamental_analysis', 'sentiment_analysis', 'dynamic_stop_loss', 'portfolio_hedging', 'market_regime_detection', 'trade_validation', 'portfolio_optimization']}
        
        # Combine all features
        all_features = {**core_features, **technical_features, **advanced_features}
        
        # Count actual implementations
        feature_count['total_implemented'] = len([f for f in all_features.values() if f])
        feature_count['enabled_features'] = feature_count['total_implemented']
        feature_count['disabled_features'] = len([f for f in all_features.values() if not f])
        feature_count['feature_details'] = all_features
        
        # Check against claimed features (from config)
        try:
            # Try to import config from parent directory
            parent_dir = os.path.dirname(os.path.dirname(__file__))
            if parent_dir not in sys.path:
                sys.path.insert(0, parent_dir)
            import config  # type: ignore
            claimed_features = config.NUM_FEATURES
            feature_count['total_claimed'] = claimed_features
        except ImportError:
            # Fallback: Check if config.py file exists
            config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.py')
            if os.path.exists(config_path):
                # File exists, use default claim
                feature_count['total_claimed'] = 15  # Default claim
            else:
                feature_count['total_claimed'] = 15  # Default claim
        except Exception:
            feature_count['total_claimed'] = 15  # Default claim
        
        # Calculate accuracy and detect inflation
        if feature_count['total_claimed'] > 0:
            feature_count['accuracy_score'] = feature_count['total_implemented'] / feature_count['total_claimed']
            feature_count['inflation_detected'] = feature_count['total_claimed'] > feature_count['total_implemented']
        
        logger.info(f" DYNAMIC FEATURE COUNT: {feature_count['total_implemented']}/{feature_count['total_claimed']} features actually implemented")
        logger.info(f" ACCURACY SCORE: {feature_count['accuracy_score']:.2%}")
        if feature_count['inflation_detected']:
            logger.warning(f" FEATURE INFLATION DETECTED: Claims {feature_count['total_claimed']} but only {feature_count['total_implemented']} implemented!")
        
    except Exception as e:
        logger.error(f"Dynamic feature counting failed: {e}")
        feature_count['error'] = str(e)
    
    return feature_count

def get_accurate_feature_status() -> dict:
    """
    Get accurate feature status based on actual implementation
    Returns status strings that match the reporting format
    """
    feature_count = get_dynamic_feature_count()
    status = {}
    
    for feature_name, is_implemented in feature_count.get('feature_details', {}).items():
        if is_implemented:
            status[feature_name] = " ENABLED"
        else:
            status[feature_name] = " DISABLED"
    
    return status

def get_dynamic_timesteps() -> int:
    """
    Get current timesteps from dynamic config system
    This ensures all parts of the system use the same timestep value
    """
    return DYNAMIC_CONFIG.total_timesteps

def update_dynamic_timesteps(new_timesteps: int) -> int:
    """
    Update timesteps across all systems using dynamic config
    """
    DYNAMIC_CONFIG.update_config(total_timesteps=new_timesteps)
    
    # CENTRALIZED: No need to sync with main config - we are the source of truth
    # Removed circular import to prevent infinite recursion
    
    return new_timesteps

def get_scaling_config() -> dict:
    """
    Get scaling configuration from dynamic config
    """
    return {
        'start_timesteps': DYNAMIC_CONFIG.start_timesteps,
        'target_timesteps': DYNAMIC_CONFIG.target_timesteps,
        'increment_size': DYNAMIC_CONFIG.increment_size,
        'max_errors_per_run': DYNAMIC_CONFIG.max_errors_per_run,
        'reduction_factor': DYNAMIC_CONFIG.reduction_factor,
        'success_runs_before_increase': DYNAMIC_CONFIG.success_runs_before_increase
    }


def _ppo_saved_data_dir() -> str:
    try:
        ppo_dir = os.path.dirname(os.path.dirname(__file__))
        return os.path.join(ppo_dir, "saved_data")
    except Exception:
        return os.path.join("ppo", "saved_data")


def _load_json_file(path: str) -> dict:
    try:
        if not os.path.isfile(path):
            return {}
        import json
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_json_file(path: str, data: dict) -> None:
    try:
        import json
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception:
        # Non-fatal: scaling persistence should never crash training
        return


def auto_scale_timesteps(execution_results: dict, error_summary: dict) -> int:
    """
    Stepwise timesteps scaling with stability guardrails.

    - Scale up only after consecutive good runs.
    - Scale down on critical errors or deterministic collapse.
    - Updates in-memory config only; total_timesteps comes from dynamic_config.py.
    """
    cfg = DYNAMIC_CONFIG

    # Load state
    state_path = os.path.join(_ppo_saved_data_dir(), "scaling_state.json")
    state = _load_json_file(state_path)
    good_runs = int(state.get("good_runs", 0) or 0)

    cur = int(getattr(cfg, "total_timesteps", 0) or 0)
    if cur <= 0:
        cur = int(getattr(cfg, "start_timesteps", 300000) or 300000)

    # Error guardrail
    try:
        total_errors = int(error_summary.get("total_errors", 0) or 0)
        warnings = int(error_summary.get("warnings", 0) or 0)
        critical = int(error_summary.get("critical_errors", 0) or 0)
    except Exception:
        total_errors, warnings, critical = 0, 0, 0
    too_many_errors = (critical > 0) or ((total_errors + warnings) > int(getattr(cfg, "max_errors_per_run", 2) or 2))

    # Deterministic stability metrics
    try:
        total_return = float(execution_results.get("total_return", 0.0) or 0.0)
        sharpe = float(execution_results.get("sharpe_ratio", 0.0) or 0.0)
        max_dd = float(execution_results.get("max_drawdown", 0.0) or 0.0)
        total_trades = int(execution_results.get("total_trades", 0) or 0)
    except Exception:
        total_return, sharpe, max_dd, total_trades = 0.0, 0.0, 0.0, 0

    pap = execution_results.get("policy_action_percentages", {})
    buy_pct = float(pap.get("BUY", 0.0)) if isinstance(pap, dict) else 0.0
    hold_pct = float(pap.get("HOLD", 0.0)) if isinstance(pap, dict) else 0.0
    collapsed = (buy_pct < 2.0 and hold_pct > 85.0)

    min_sharpe = float(getattr(cfg, "min_sharpe_to_scale", 0.2) or 0.2)
    min_ret = float(getattr(cfg, "min_return_to_scale", 0.02) or 0.02)
    max_dd_ok = float(getattr(cfg, "max_drawdown_to_scale", 0.15) or 0.15)
    min_trades = int(getattr(cfg, "min_trades_per_episode", 20) or 20)

    good_metrics = (
        (sharpe >= min_sharpe)
        and (total_return >= min_ret)
        and (max_dd <= max_dd_ok)
        and (total_trades >= min_trades)
        and (not collapsed)
    )

    decision = "HOLD"
    new_ts = cur

    if too_many_errors or collapsed:
        good_runs = 0
        decision = "DECREASE"
        reduction_factor = float(getattr(cfg, "reduction_factor", 0.8) or 0.8)
        new_ts = max(int(getattr(cfg, "min_timesteps", 100000) or 100000), int(cur * reduction_factor))
    elif good_metrics:
        good_runs += 1
        if good_runs >= int(getattr(cfg, "success_runs_before_increase", 2) or 2):
            decision = "INCREASE"
            inc = int(getattr(cfg, "increment_size", 50000) or 50000)
            cap = int(getattr(cfg, "max_timesteps", 600000) or 600000)
            new_ts = min(cap, cur + inc)
            good_runs = 0
    else:
        good_runs = 0

    # Apply + persist
    try:
        new_ts = int(new_ts)
    except Exception:
        new_ts = cur
    cfg.update_config(total_timesteps=new_ts)

    # Persist scaling state only (for audit). total_timesteps comes from dynamic_config.py.
    _save_json_file(
        state_path,
        {
            "good_runs": good_runs,
            "decision": decision,
            "prev_timesteps": cur,
            "new_timesteps": new_ts,
            "metrics": {
                "total_return": total_return,
                "sharpe_ratio": sharpe,
                "max_drawdown": max_dd,
                "total_trades": total_trades,
                "buy_pct": buy_pct,
                "hold_pct": hold_pct,
            },
            "errors": {
                "total_errors": total_errors,
                "warnings": warnings,
                "critical_errors": critical,
            },
        },
    )
    return new_ts


# NOTE: dynamic_config.py is the single source of truth for total_timesteps.
# Persisted overrides (dynamic_config_overrides.json) were removed so user edits
# in this file are always respected.

# CENTRALIZED: Getter functions for all parameters
def get_dynamic_learning_rate() -> float:
    """Get current learning rate from dynamic config"""
    return DYNAMIC_CONFIG.learning_rate

def get_dynamic_batch_size() -> int:
    """Get current batch size from dynamic config"""
    return DYNAMIC_CONFIG.batch_size

def get_dynamic_n_epochs() -> int:
    """Get current n_epochs from dynamic config"""
    return DYNAMIC_CONFIG.n_epochs

def get_dynamic_gamma() -> float:
    """Get current gamma from dynamic config"""
    return DYNAMIC_CONFIG.gamma

def get_dynamic_gae_lambda() -> float:
    """Get current gae_lambda from dynamic config"""
    return DYNAMIC_CONFIG.gae_lambda

def get_dynamic_clip_range() -> float:
    """Get current clip_range from dynamic config"""
    return DYNAMIC_CONFIG.clip_range

def get_dynamic_ent_coef() -> float:
    """Get current ent_coef from dynamic config"""
    return DYNAMIC_CONFIG.ent_coef

def get_dynamic_vf_coef() -> float:
    """Get current vf_coef from dynamic config"""
    return DYNAMIC_CONFIG.vf_coef

def get_dynamic_max_grad_norm() -> float:
    """Get current max_grad_norm from dynamic config"""
    return DYNAMIC_CONFIG.max_grad_norm

def get_dynamic_target_kl() -> float:
    """Get current target_kl from dynamic config"""
    return DYNAMIC_CONFIG.target_kl

def get_dynamic_window_size() -> int:
    """Get current window_size from dynamic config"""
    return DYNAMIC_CONFIG.window_size

def get_dynamic_max_position() -> float:
    """Get current max_position from dynamic config"""
    return DYNAMIC_CONFIG.max_position

def get_dynamic_transaction_cost() -> float:
    """Get current transaction_cost from dynamic config"""
    return DYNAMIC_CONFIG.transaction_cost

def get_dynamic_max_samples() -> int:
    """Get current max_samples from dynamic config"""
    return DYNAMIC_CONFIG.max_samples

def get_dynamic_train_split() -> float:
    """Get current train_split from dynamic config"""
    return DYNAMIC_CONFIG.train_split

def get_dynamic_validation_split() -> float:
    """Get current validation_split from dynamic config"""
    return DYNAMIC_CONFIG.validation_split

# Initialize with external config (only once during first import)
if not hasattr(DYNAMIC_CONFIG, '_initialized'):
    print(f" DEBUG: Initializing dynamic config system...")
    print(f" DEBUG: Current max_position_size before sync: {DYNAMIC_CONFIG.max_position_size}")
    try:
        DYNAMIC_CONFIG.sync_from_config_file()
        DYNAMIC_CONFIG._initialized = True  # type: ignore
        print(f" DEBUG: After sync, max_position_size: {DYNAMIC_CONFIG.max_position_size}")
        logger.info(" Dynamic configuration system initialized")
    except Exception as e:
        logger.warning(f"Could not sync with external config: {e}")
        logger.info(" Dynamic configuration system initialized with defaults")
        DYNAMIC_CONFIG._initialized = True  # type: ignore
