"""
Advanced Reward Calculator for PPO Trading Agent
Implements sophisticated reward functions including risk-adjusted metrics
"""

import numpy as np
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class RewardComponents:
    """Components of the reward function"""
    base_return: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    drawdown_penalty: float = 0.0
    activity_bonus: float = 0.0
    regime_alignment: float = 0.0
    volatility_penalty: float = 0.0
    transaction_cost_penalty: float = 0.0  # NEW: Transaction cost penalty
    total_reward: float = 0.0
    # Minute-level components (normalized to equity)
    realized_pnl_frac: float = 0.0
    m2m_frac: float = 0.0
    exec_cost_frac: float = 0.0
    turnover_frac: float = 0.0
    inventory_frac: float = 0.0
    minute_reward_unclipped: float = 0.0

class AdvancedRewardCalculator:
    """Advanced reward calculator with risk-adjusted metrics"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.logger = logger
        self.portfolio_history = []
        self.return_history = []
        self.market_regime_history = []
        self.trade_history = []
        self.current_drawdown = 0.0
        
        # PRIORITY 1 & 2 FIXES: Updated reward weights for better trading activity and win rate
        self.weights = {
            'base_return': 0.4,  # Increased from 0.35 - PRIORITY 2: Better win rate
            'sharpe_ratio': 0.25,  # Increased from 0.2 - PRIORITY 2: Higher Sharpe
            'sortino_ratio': 0.2,  # Increased from 0.15 - PRIORITY 2: Better risk-adjusted returns
            'drawdown_penalty': 0.1,  # Maintained - Risk management
            'activity_bonus': 0.05,  # RE-ENABLED - PRIORITY 1: Enable trading activity
            'regime_alignment': 0.05,  # Maintained
            'volatility_penalty': 0.05,  # Maintained
            'transaction_cost_penalty': 0.03  # Reduced from 0.05 - PRIORITY 1: Less penalty for trading
        }
        
        # Update weights from config if available
        if 'REWARD_WEIGHTS' in self.config:
            self.weights.update(self.config['REWARD_WEIGHTS'])

        # PERFORMANCE: Cache expensive metrics and update periodically
        try:
            cfg_interval = None
            if isinstance(self.config, dict):
                cfg_interval = self.config.get('REWARD_UPDATE_INTERVAL', None)
                if isinstance(self.config.get('ENV_CONFIG', {}), dict) and cfg_interval is None:
                    cfg_interval = self.config['ENV_CONFIG'].get('reward_update_interval', None)
            self.update_interval = int(cfg_interval) if cfg_interval is not None else 5  # PRIORITY 2: More frequent updates
        except Exception:
            self.update_interval = 5  # PRIORITY 2: More frequent updates
        self._last_update_step: int = -1
        self._cached_sharpe: float = 0.0
        self._cached_sortino: float = 0.0
        self._cached_volatility: float = 0.0
        # PRIORITY 1 & 2 FIXES: Reduced cost parameters to enable more trading activity
        self.cost_params = {
            'lambda_turnover': 5e-5,  # Reduced from 1e-4 - PRIORITY 1: Less turnover penalty
            'k_inventory': 2e-4,  # Reduced from 5e-4 - PRIORITY 1: Less inventory penalty
            'k_var': 2e-4,  # Reduced from 5e-4 - PRIORITY 1: Less VaR penalty
            'cost_penalty_weight': 0.5,  # Reduced from 1.0 - PRIORITY 1: Less cost penalty
            'risk_lookback': 30  # Reduced from 60 - PRIORITY 2: Shorter lookback for faster adaptation
        }
        try:
            if 'REWARD_COST_PARAMS' in (self.config or {}):
                self.cost_params.update(self.config['REWARD_COST_PARAMS'])
        except Exception:
            pass
    
    def calculate_reward(self, portfolio_return: float, trades: List[Dict], 
                        portfolio_value: float, benchmark_return: float = 0.0) -> float:
        """CRITICAL FIX: Calculate comprehensive reward with better trading incentives"""
        try:
            # Base return component (40% weight)
            base_return_reward = portfolio_return * self.weights['base_return']
            
            # CRITICAL FIX: Enhanced trading activity bonus
            activity_bonus = self._calculate_activity_bonus(trades) * self.weights['activity_bonus']
            
            # Risk-adjusted return components
            sharpe_bonus = 0.0
            sortino_bonus = 0.0
            
            if len(self.return_history) >= 20:  # Need enough history for risk metrics
                # Calculate Sharpe ratio bonus
                returns_array = np.array(self.return_history[-20:])
                if np.std(returns_array) > 0:
                    sharpe_ratio = float(np.mean(returns_array) / np.std(returns_array))
                    sharpe_bonus = max(0.0, sharpe_ratio) * self.weights['sharpe_ratio']
                
                # Calculate Sortino ratio bonus
                negative_returns = returns_array[returns_array < 0]
                if len(negative_returns) > 0 and np.std(negative_returns) > 0:
                    sortino_ratio = float(np.mean(returns_array) / np.std(negative_returns))
                    sortino_bonus = max(0.0, sortino_ratio) * self.weights['sortino_ratio']
            
            # CRITICAL FIX: Drawdown penalty (reduced for better performance)
            drawdown_penalty = 0.0
            if self.current_drawdown < -0.05:  # Only penalize >5% drawdown
                drawdown_penalty = self.current_drawdown * self.weights['drawdown_penalty']
            
            # CRITICAL FIX: Transaction cost penalty (reduced)
            transaction_penalty = 0.0
            if trades:
                total_commission = sum(abs(trade.get('commission', 0.0)) for trade in trades)
                transaction_penalty = -(total_commission / portfolio_value) * 0.5  # Reduced penalty
            
            # Combine all components
            total_reward = (base_return_reward + 
                          activity_bonus + 
                          sharpe_bonus + 
                          sortino_bonus + 
                          drawdown_penalty + 
                          transaction_penalty)
            
            # CRITICAL FIX: Ensure reasonable reward range
            total_reward = np.clip(total_reward, -0.5, 0.5)
            
            # Update history
            self.return_history.append(portfolio_return)
            if len(self.return_history) > 100:
                self.return_history.pop(0)
            
            return float(total_reward)
            
        except Exception as e:
            self.logger.error(f"Reward calculation failed: {e}")
            return 0.0
    
    def _calculate_sharpe_ratio(self, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio from return history"""
        if len(self.return_history) < 2:
            return 0.0
        
        returns = np.array(self.return_history)
        excess_returns = returns - (risk_free_rate / 252)  # Daily risk-free rate
        
        if np.std(excess_returns) == 0:
            return 0.0
        
        sharpe = np.mean(excess_returns) / np.std(excess_returns)
        return float(np.clip(sharpe, -2.0, 2.0))  # Clip to reasonable range
    
    def _calculate_sortino_ratio(self, risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio from return history"""
        if len(self.return_history) < 2:
            return 0.0
        
        returns = np.array(self.return_history)
        excess_returns = returns - (risk_free_rate / 252)
        
        # Only consider downside returns
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0 or np.std(downside_returns) == 0:
            return 0.0
        
        sortino = np.mean(excess_returns) / np.std(downside_returns)
        return float(np.clip(sortino, -2.0, 2.0))
    
    def _calculate_drawdown_penalty(self, current_value: float) -> float:
        """Calculate drawdown penalty - OPTIMIZED to avoid repeated max() calls"""
        if len(self.portfolio_history) < 2:
            return 0.0
        
        # OPTIMIZATION: Cache peak value and only recalculate when needed
        if not hasattr(self, '_cached_peak_value') or current_value > self._cached_peak_value:
            self._cached_peak_value = current_value
        else:
            # Use cached peak value if current value is lower
            peak_value = self._cached_peak_value
            if peak_value <= 0:
                return 0.0
            
            drawdown = (current_value - peak_value) / peak_value
            
            # Progressive penalty: more severe for larger drawdowns
            if drawdown > -0.05:  # Less than 5% drawdown
                return 0.0
            elif drawdown > -0.10:  # 5-10% drawdown
                return drawdown * 0.5
            elif drawdown > -0.20:  # 10-20% drawdown
                return drawdown * 1.0
            else:  # >20% drawdown
                return drawdown * 2.0
        
        return 0.0  # No drawdown if we're at a new peak
    
    def _calculate_activity_bonus(self, recent_trades: List) -> float:
        """CRITICAL FIX: Calculate activity bonus to encourage trading"""
        try:
            if not recent_trades:
                return 0.0
            
            # CRITICAL FIX: More aggressive activity bonus
            num_trades = len(recent_trades)
            
            # Base activity bonus
            activity_bonus = num_trades * 0.01  # 1% bonus per trade
            
            # Volume bonus for larger trades
            total_volume = sum(abs(trade.get('value', 0.0)) for trade in recent_trades)
            volume_bonus = min(total_volume / 10000.0, 0.05)  # Up to 5% volume bonus
            
            # Diversity bonus for trading multiple assets
            unique_tickers = len(set(trade.get('ticker', '') for trade in recent_trades))
            diversity_bonus = unique_tickers * 0.005  # 0.5% per unique ticker
            
            total_bonus = activity_bonus + volume_bonus + diversity_bonus
            
            # CRITICAL FIX: Cap the bonus to prevent excessive trading
            return min(total_bonus, 0.1)  # Max 10% activity bonus
            
        except Exception as e:
            self.logger.error(f"Activity bonus calculation failed: {e}")
            return 0.0

    def _compute_cost_aware_components(self, recent_trades: List, prev_value: float, ctx: Optional[Dict]) -> Dict:
        """Compute normalized execution cost, turnover, inventory, and risk proxies.

        Returns a dict with keys: execution_cost_ratio, turnover, inventory, portfolio_var.
        All values are normalized to portfolio value where applicable.
        """
        try:
            prev_v = float(max(prev_value, 1e-9))
        except Exception:
            prev_v = 1.0

        # Aggregate trade stats (numpy reductions)
        total_commission = 0.0
        total_slippage = 0.0
        total_trade_value = 0.0
        if recent_trades:
            try:
                arr = lambda k: np.asarray([float(t.get(k, 0.0)) for t in recent_trades], dtype=float)
                total_commission = float(np.sum(arr('commission')))
                total_slippage = float(np.sum(arr('slippage')))
                # Prefer provided trade_value; fallback to shares*price
                tv = np.asarray([
                    (float(t.get('trade_value')) if ('trade_value' in t) else (abs(float(t.get('shares', 0.0)) * float(t.get('price', 0.0)))))
                    for t in recent_trades
                ], dtype=float)
                tv[~np.isfinite(tv)] = 0.0
                total_trade_value = float(np.sum(tv))
            except Exception:
                pass

        # Estimated market impact proportional to traded notional
        market_impact_factor = 0.0
        try:
            market_impact_factor = float(self.config.get('TRADING_CONFIG', {}).get('market_impact_factor', 0.0)) if isinstance(self.config, dict) else 0.0
        except Exception:
            market_impact_factor = 0.0
        est_market_impact_value = market_impact_factor * total_trade_value

        execution_cost_ratio = (total_commission + total_slippage + est_market_impact_value) / prev_v

        # Turnover normalized by portfolio value
        turnover = total_trade_value / prev_v

        # Inventory (gross exposure)  prefer ctx if provided
        inventory = 0.0
        try:
            if ctx and 'inventory' in ctx:
                inventory = float(ctx.get('inventory', 0.0))
        except Exception:
            inventory = 0.0

        # OPTIMIZATION: Cache risk calculation to avoid repeated computations
        if not hasattr(self, '_cached_risk_lookback') or self._cached_risk_lookback != self.cost_params.get('risk_lookback', 60):
            self._cached_risk_lookback = self.cost_params.get('risk_lookback', 60)
            self._cached_portfolio_var = None
        
        # Risk proxy: rolling variance of recent returns history
        lookback = int(max(1, self._cached_risk_lookback))
        if len(self.return_history) >= 2 and self._cached_portfolio_var is None:
            rets = np.asarray(self.return_history[-lookback:], dtype=float)
            rets = rets[np.isfinite(rets)]
            self._cached_portfolio_var = float(np.var(rets)) if rets.size > 0 else 0.0
        
        portfolio_var = self._cached_portfolio_var if self._cached_portfolio_var is not None else 0.0

        return {
            'execution_cost_ratio': max(execution_cost_ratio, 0.0),
            'turnover': max(turnover, 0.0),
            'inventory': max(inventory, 0.0),
            'portfolio_var': max(portfolio_var, 0.0)
        }
    
    def _calculate_regime_alignment(self, base_return: float, market_regime: Dict) -> float:
        """Calculate bonus for market regime alignment"""
        if not market_regime:
            return 0.0
        
        regime = market_regime.get('regime', 'unknown')
        
        # Bonus for good performance in bull markets
        if regime in ['bull_stable', 'bull_volatile'] and base_return > 0:
            return 0.002  # Small bonus
        
        # Bonus for avoiding losses in bear markets
        elif regime in ['bear_stable', 'bear_volatile'] and base_return < 0:
            return 0.001  # Small bonus for avoiding losses
        
        return 0.0
    
    def _calculate_volatility_penalty(self) -> float:
        """Calculate penalty for excessive volatility"""
        # Deprecated in favor of cached volatility computation for performance
        if len(self.return_history) < 20:
            return 0.0
        vol = self._cached_volatility if self._cached_volatility is not None else 0.0
        if vol > 0.03:
            return -0.001 * (vol - 0.03)
        return 0.0
    
    def _calculate_transaction_cost_penalty(self, recent_trades: List) -> float:
        """Calculate transaction cost penalty to discourage overtrading"""
        try:
            if not recent_trades:
                return 0.0
            
            # Calculate total transaction costs
            total_commission = sum(trade.get('commission', 0.0) for trade in recent_trades)
            total_slippage = sum(trade.get('slippage', 0.0) for trade in recent_trades)
            total_costs = total_commission + total_slippage
            
            # Get portfolio value for normalization
            if self.portfolio_history:
                portfolio_value = self.portfolio_history[-1]
                if portfolio_value > 0:
                    # Penalty based on transaction costs as percentage of portfolio
                    cost_ratio = total_costs / portfolio_value
                    penalty = -cost_ratio * 10  # Scale penalty
                    return max(penalty, -0.02)  # Cap penalty at -2%
            
            return 0.0
            
        except Exception as e:
            logger.warning(f" Transaction cost penalty calculation error: {e}")
            return 0.0
    
    def reset(self):
        """Reset calculator state"""
        self.portfolio_history = []
        self.return_history = []
        self.market_regime_history = []
        self.trade_history = []
    
    def get_reward_summary(self) -> Dict:
        """Get summary of reward components"""
        if not self.return_history:
            return {}
        
        return {
            'avg_sharpe_ratio': self._calculate_sharpe_ratio(),
            'avg_sortino_ratio': self._calculate_sortino_ratio(),
            'total_volatility': np.std(self.return_history) if self.return_history else 0.0,
            'max_drawdown': self._calculate_max_drawdown(),
            'reward_history_length': len(self.return_history)
        }
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown from portfolio history"""
        if len(self.portfolio_history) < 2:
            return 0.0
        
        portfolio_values = np.array(self.portfolio_history)
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - peak) / peak
        
        return float(np.min(drawdown))

# Backward compatibility alias
RewardCalculator = AdvancedRewardCalculator
