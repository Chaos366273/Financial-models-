"""
Performance metrics calculation for trading system
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class PerformanceMetrics:
    """Comprehensive performance metrics calculation"""
    
    def __init__(self):
        self.portfolio_values = []
        self.returns = []
        self.trades = []
        self.benchmark_returns = None
    
    def add_portfolio_value(self, value: float):
        """Add portfolio value for tracking"""
        self.portfolio_values.append(value)
    
    def add_return(self, return_val: float):
        """Add return for tracking"""
        self.returns.append(return_val)
    
    def add_trade(self, trade: Dict):
        """Add trade for tracking"""
        self.trades.append(trade)
    
    def set_benchmark_returns(self, benchmark_returns: List[float]):
        """Set benchmark returns for comparison"""
        self.benchmark_returns = benchmark_returns
    
    def calculate_all_metrics(self, training_results: Optional[Dict] = None) -> Dict:
        """Calculate comprehensive performance metrics"""
        metrics = {}
        
        try:
            # Basic metrics
            if len(self.portfolio_values) > 1:
                total_return = (self.portfolio_values[-1] - self.portfolio_values[0]) / self.portfolio_values[0]
                metrics['total_return'] = total_return
                
                # Calculate returns if not provided
                if not self.returns and len(self.portfolio_values) > 1:
                    self.returns = list(np.diff(self.portfolio_values) / self.portfolio_values[:-1])
                
                if self.returns:
                    returns_array = np.array(self.returns)
                    
                    # Risk metrics
                    metrics['volatility'] = np.std(returns_array) * np.sqrt(252)  # Annualized
                    metrics['sharpe_ratio'] = self._calculate_sharpe_ratio(returns_array)
                    metrics['sortino_ratio'] = self._calculate_sortino_ratio(returns_array)
                    metrics['max_drawdown'] = self._calculate_max_drawdown()
                    metrics['calmar_ratio'] = self._calculate_calmar_ratio(returns_array)
                    
                    # VaR and CVaR
                    metrics['var_95'] = np.percentile(returns_array, 5)
                    metrics['cvar_95'] = returns_array[returns_array <= metrics['var_95']].mean()
                    
                    # Win rate and profit factor
                    if self.trades:
                        metrics['win_rate'] = self._calculate_win_rate()
                        metrics['profit_factor'] = self._calculate_profit_factor()
                    
                    # Benchmark comparison
                    if self.benchmark_returns is not None:
                        benchmark_metrics = self._calculate_benchmark_metrics(returns_array)
                        metrics.update(benchmark_metrics)
            
            # Training metrics
            if training_results:
                metrics.update(self._extract_training_metrics(training_results))
            
            logger.info(f" Calculated {len(metrics)} performance metrics")
            
        except Exception as e:
            logger.error(f" Error calculating metrics: {e}")
            metrics = {'error': str(e)}
        
        return metrics
    
    def _calculate_sharpe_ratio(self, returns: np.ndarray) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0
        return np.mean(returns) / np.std(returns) * np.sqrt(252)
    
    def _calculate_sortino_ratio(self, returns: np.ndarray) -> float:
        """Calculate Sortino ratio"""
        if len(returns) == 0:
            return 0.0
        downside_returns = returns[returns < 0]
        if len(downside_returns) == 0:
            return np.inf if np.mean(returns) > 0 else 0.0
        downside_deviation = np.std(downside_returns)
        if downside_deviation == 0:
            return 0.0
        return np.mean(returns) / downside_deviation * np.sqrt(252)
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown"""
        if len(self.portfolio_values) < 2:
            return 0.0
        
        peak = self.portfolio_values[0]
        max_dd = 0.0
        
        for value in self.portfolio_values:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            if dd > max_dd:
                max_dd = dd
        
        return max_dd
    
    def _calculate_calmar_ratio(self, returns: np.ndarray) -> float:
        """Calculate Calmar ratio"""
        max_dd = self._calculate_max_drawdown()
        if max_dd == 0:
            return 0.0
        annual_return = np.mean(returns) * 252
        return float(annual_return / max_dd)
    
    def _calculate_win_rate(self) -> float:
        """Calculate win rate from trades"""
        if not self.trades:
            return 0.0
        
        winning_trades = sum(1 for trade in self.trades if trade.get('pnl', 0) > 0)
        return winning_trades / len(self.trades)
    
    def _calculate_profit_factor(self) -> float:
        """Calculate profit factor"""
        if not self.trades:
            return 0.0
        
        gross_profit = sum(trade.get('pnl', 0) for trade in self.trades if trade.get('pnl', 0) > 0)
        gross_loss = abs(sum(trade.get('pnl', 0) for trade in self.trades if trade.get('pnl', 0) < 0))
        
        if gross_loss == 0:
            return np.inf if gross_profit > 0 else 0.0
        
        return gross_profit / gross_loss
    
    def _calculate_benchmark_metrics(self, returns: np.ndarray) -> Dict:
        """Calculate benchmark comparison metrics"""
        if self.benchmark_returns is None or len(self.benchmark_returns) != len(returns):
            return {}
        
        benchmark_returns = np.array(self.benchmark_returns)
        
        # Alpha and Beta
        covariance = np.cov(returns, benchmark_returns)[0, 1]
        benchmark_variance = np.var(benchmark_returns)
        
        if benchmark_variance == 0:
            beta = 0.0
        else:
            beta = covariance / benchmark_variance
        
        alpha = np.mean(returns) - beta * np.mean(benchmark_returns)
        
        # Information ratio
        excess_returns = returns - benchmark_returns
        tracking_error = np.std(excess_returns)
        
        if tracking_error == 0:
            information_ratio = 0.0
        else:
            information_ratio = np.mean(excess_returns) / tracking_error
        
        return {
            'alpha': alpha,
            'beta': beta,
            'information_ratio': information_ratio
        }
    
    def _extract_training_metrics(self, training_results: Dict) -> Dict:
        """Extract training-specific metrics"""
        metrics = {}
        
        if 'episode_rewards' in training_results:
            metrics['mean_episode_reward'] = np.mean(training_results['episode_rewards'])
            metrics['std_episode_reward'] = np.std(training_results['episode_rewards'])
        
        if 'losses' in training_results:
            metrics['final_loss'] = training_results['losses'][-1] if training_results['losses'] else 0.0
        
        return metrics
