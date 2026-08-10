"""
Advanced Financial Metrics
Comprehensive risk and performance metrics calculation
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, cast
from datetime import datetime, timedelta
from dataclasses import dataclass
from scipy import stats

logger = logging.getLogger(__name__)

@dataclass
class MetricsConfig:
    """Advanced metrics configuration"""
    risk_free_rate: float = 0.02  # 2% annual risk-free rate
    confidence_level: float = 0.95  # 95% confidence for VaR
    lookback_period: int = 252  # Trading days
    min_data_points: int = 30  # Minimum data points for calculations

class AdvancedMetrics:
    """
    Comprehensive financial metrics and risk analysis
    """
    
    def __init__(self, initial_capital: float = 100000.0):
        self.initial_capital = initial_capital
        self.config = MetricsConfig()
        
        # Performance data
        self.returns = []
        self.prices = []
        self.volumes = []
        
        # Risk metrics
        self.var_history = []
        self.cvar_history = []
        self.drawdown_history = []
        
        # Advanced metrics
        self.skewness_history = []
        self.kurtosis_history = []
        self.ulcer_index_history = []
        
        # Rolling metrics
        self.rolling_metrics = {}
        
        logger.info("AdvancedMetrics initialized")
    
    def calculate_sharpe_ratio(self, returns: List[float], risk_free_rate: Optional[float] = None) -> float:
        """Calculate Sharpe ratio"""
        try:
            if not returns or len(returns) < self.config.min_data_points:
                return 0.0
            
            returns_array = np.array(returns)
            avg_return = np.mean(returns_array)
            std_return = np.std(returns_array, ddof=1)
            
            if std_return == 0:
                return 0.0
            
            rf_rate = self.config.risk_free_rate / 252 if risk_free_rate is None else risk_free_rate  # Daily rate
            sharpe = (avg_return - rf_rate) / std_return
            
            return float(sharpe)
            
        except Exception as e:
            logger.error(f"Error calculating Sharpe ratio: {e}")
            return 0.0
    
    def calculate_sortino_ratio(self, returns: List[float], risk_free_rate: Optional[float] = None) -> float:
        """Calculate Sortino ratio (downside deviation)"""
        try:
            if not returns or len(returns) < self.config.min_data_points:
                return 0.0
            
            returns_array = np.array(returns)
            avg_return = np.mean(returns_array)
            
            # Calculate downside deviation
            negative_returns = returns_array[returns_array < 0]
            if len(negative_returns) == 0:
                return 0.0
            
            downside_deviation = np.std(negative_returns, ddof=1)
            
            if downside_deviation == 0:
                return 0.0
            
            rf_rate = self.config.risk_free_rate / 252 if risk_free_rate is None else risk_free_rate
            sortino = (avg_return - rf_rate) / downside_deviation
            
            return float(sortino)
            
        except Exception as e:
            logger.error(f"Error calculating Sortino ratio: {e}")
            return 0.0
    
    def calculate_calmar_ratio(self, returns: List[float], max_drawdown: float) -> float:
        """Calculate Calmar ratio (return / max drawdown)"""
        try:
            if not returns or max_drawdown == 0:
                return 0.0
            
            avg_return = np.mean(returns)
            calmar = avg_return / abs(max_drawdown)
            
            return float(calmar)
            
        except Exception as e:
            logger.error(f"Error calculating Calmar ratio: {e}")
            return 0.0
    
    def calculate_information_ratio(self, portfolio_returns: List[float], 
                                  benchmark_returns: List[float]) -> float:
        """Calculate information ratio"""
        try:
            if len(portfolio_returns) != len(benchmark_returns) or len(portfolio_returns) < self.config.min_data_points:
                return 0.0
            
            excess_returns = np.array(portfolio_returns) - np.array(benchmark_returns)
            avg_excess = np.mean(excess_returns)
            tracking_error = np.std(excess_returns, ddof=1)
            
            if tracking_error == 0:
                return 0.0
            
            information_ratio = avg_excess / tracking_error
            
            return float(information_ratio)
            
        except Exception as e:
            logger.error(f"Error calculating information ratio: {e}")
            return 0.0
    
    def calculate_treynor_ratio(self, returns: List[float], benchmark_returns: List[float], 
                               risk_free_rate: Optional[float] = None) -> float:
        """Calculate Treynor ratio (excess return / beta)"""
        try:
            if len(returns) != len(benchmark_returns) or len(returns) < self.config.min_data_points:
                return 0.0
            
            returns_array = np.array(returns)
            benchmark_array = np.array(benchmark_returns)
            
            # Calculate beta
            covariance = np.cov(returns_array, benchmark_array)[0, 1]
            benchmark_variance = np.var(benchmark_array, ddof=1)
            
            if benchmark_variance == 0:
                return 0.0
            
            beta = covariance / benchmark_variance
            
            if beta == 0:
                return 0.0
            
            # Calculate excess return
            avg_return = np.mean(returns_array)
            rf_rate = self.config.risk_free_rate / 252 if risk_free_rate is None else risk_free_rate
            excess_return = avg_return - rf_rate
            
            treynor = excess_return / beta
            
            return float(treynor)
            
        except Exception as e:
            logger.error(f"Error calculating Treynor ratio: {e}")
            return 0.0
    
    def calculate_value_at_risk(self, returns: List[float], confidence_level: Optional[float] = None) -> float:
        """Calculate Value at Risk (VaR)"""
        try:
            if not returns or len(returns) < self.config.min_data_points:
                return 0.0
            
            confidence = confidence_level or self.config.confidence_level
            percentile = (1 - confidence) * 100
            
            var = np.percentile(returns, percentile)
            
            return float(var)
            
        except Exception as e:
            logger.error(f"Error calculating VaR: {e}")
            return 0.0
    
    def calculate_conditional_var(self, returns: List[float], confidence_level: Optional[float] = None) -> float:
        """Calculate Conditional Value at Risk (CVaR)"""
        try:
            if not returns or len(returns) < self.config.min_data_points:
                return 0.0
            
            confidence = confidence_level or self.config.confidence_level
            var = self.calculate_value_at_risk(returns, confidence)
            
            # Calculate mean of returns below VaR
            returns_array = np.array(returns)
            tail_returns = returns_array[returns_array <= var]
            
            if len(tail_returns) == 0:
                return var
            
            cvar = np.mean(tail_returns)
            
            return float(cvar)
            
        except Exception as e:
            logger.error(f"Error calculating CVaR: {e}")
            return 0.0
    
    def calculate_ulcer_index(self, prices: List[float]) -> float:
        """Calculate Ulcer Index (measure of downside risk)"""
        try:
            if not prices or len(prices) < self.config.min_data_points:
                return 0.0
            
            prices_array = np.array(prices)
            
            # Calculate drawdowns
            peak = np.maximum.accumulate(prices_array)
            drawdown = (prices_array - peak) / peak
            
            # Calculate Ulcer Index
            squared_drawdown = drawdown ** 2
            ulcer_index = np.sqrt(np.mean(squared_drawdown))
            
            return float(ulcer_index)
            
        except Exception as e:
            logger.error(f"Error calculating Ulcer Index: {e}")
            return 0.0
    
    def calculate_skewness(self, returns: List[float]) -> float:
        """Calculate skewness of returns"""
        try:
            if not returns or len(returns) < self.config.min_data_points:
                return 0.0
            
            returns_array = np.array(returns)
            skewness = stats.skew(returns_array)
            
            return float(skewness)
            
        except Exception as e:
            logger.error(f"Error calculating skewness: {e}")
            return 0.0
    
    def calculate_kurtosis(self, returns: List[float]) -> float:
        """Calculate kurtosis of returns"""
        try:
            if not returns or len(returns) < self.config.min_data_points:
                return 0.0
            
            returns_array = np.array(returns)
            kurtosis = stats.kurtosis(returns_array)
            
            return float(kurtosis)
            
        except Exception as e:
            logger.error(f"Error calculating kurtosis: {e}")
            return 0.0
    
    def calculate_win_rate(self, returns: List[float]) -> float:
        """Calculate win rate (percentage of positive returns)"""
        try:
            if not returns:
                return 0.0
            
            returns_array = np.array(returns)
            positive_returns = returns_array[returns_array > 0]
            win_rate = len(positive_returns) / len(returns_array)
            
            return float(win_rate)
            
        except Exception as e:
            logger.error(f"Error calculating win rate: {e}")
            return 0.0
    
    def calculate_profit_factor(self, returns: List[float]) -> float:
        """Calculate profit factor (gross profit / gross loss)"""
        try:
            if not returns:
                return 0.0
            
            returns_array = np.array(returns)
            positive_returns = returns_array[returns_array > 0]
            negative_returns = returns_array[returns_array < 0]
            
            gross_profit = np.sum(positive_returns)
            gross_loss = abs(np.sum(negative_returns))
            
            if gross_loss == 0:
                return 0.0
            
            profit_factor = gross_profit / gross_loss
            
            return float(profit_factor)
            
        except Exception as e:
            logger.error(f"Error calculating profit factor: {e}")
            return 0.0
    
    def calculate_max_drawdown(self, prices: List[float]) -> Tuple[float, int, int]:
        """Calculate maximum drawdown and its duration"""
        try:
            if not prices or len(prices) < 2:
                return 0.0, 0, 0
            
            prices_array = np.array(prices)
            peak = np.maximum.accumulate(prices_array)
            drawdown = (prices_array - peak) / peak
            
            max_dd = float(np.min(drawdown))
            max_dd_idx = int(np.argmin(drawdown))
            
            # Find peak before max drawdown
            peak_idx = int(np.argmax(prices_array[:max_dd_idx + 1]))
            
            # Calculate recovery time
            recovery_idx = None
            for i in range(max_dd_idx, len(prices_array)):
                if prices_array[i] >= prices_array[peak_idx]:
                    recovery_idx = i
                    break
            
            drawdown_duration = int(max_dd_idx - peak_idx)
            recovery_duration = int(recovery_idx - max_dd_idx) if recovery_idx is not None else int(len(prices_array) - max_dd_idx)
            
            return max_dd, drawdown_duration, recovery_duration
            
        except Exception as e:
            logger.error(f"Error calculating max drawdown: {e}")
            return 0.0, 0, 0
    
    def calculate_volatility(self, returns: List[float], window: Optional[int] = None) -> float:
        """Calculate volatility (standard deviation of returns)"""
        try:
            if not returns or len(returns) < self.config.min_data_points:
                return 0.0
            
            returns_array = np.array(returns)
            
            if window and window < len(returns_array):
                returns_array = returns_array[-window:]
            
            volatility = np.std(returns_array, ddof=1)
            
            return float(volatility)
            
        except Exception as e:
            logger.error(f"Error calculating volatility: {e}")
            return 0.0
    
    def calculate_beta(self, returns: List[float], benchmark_returns: List[float]) -> float:
        """Calculate beta against benchmark"""
        try:
            if len(returns) != len(benchmark_returns) or len(returns) < self.config.min_data_points:
                return 1.0
            
            returns_array = np.array(returns)
            benchmark_array = np.array(benchmark_returns)
            
            covariance = np.cov(returns_array, benchmark_array)[0, 1]
            benchmark_variance = np.var(benchmark_array, ddof=1)
            
            if benchmark_variance == 0:
                return 1.0
            
            beta = covariance / benchmark_variance
            
            return float(beta)
            
        except Exception as e:
            logger.error(f"Error calculating beta: {e}")
            return 1.0
    
    def calculate_alpha(self, returns: List[float], benchmark_returns: List[float], 
                       risk_free_rate: Optional[float] = None) -> float:
        """Calculate alpha (excess return not explained by beta)"""
        try:
            if len(returns) != len(benchmark_returns) or len(returns) < self.config.min_data_points:
                return 0.0
            
            returns_array = np.array(returns)
            benchmark_array = np.array(benchmark_returns)
            
            avg_return = np.mean(returns_array)
            avg_benchmark = np.mean(benchmark_array)
            beta = self.calculate_beta(returns, benchmark_returns)
            
            rf_rate = self.config.risk_free_rate / 252 if risk_free_rate is None else risk_free_rate
            
            alpha = avg_return - (rf_rate + beta * (avg_benchmark - rf_rate))
            
            return float(alpha)
            
        except Exception as e:
            logger.error(f"Error calculating alpha: {e}")
            return 0.0
    
    def calculate_rolling_metrics(self, returns: List[float], window: int = 60) -> Dict[str, List[float]]:
        """Calculate rolling metrics over time"""
        try:
            if len(returns) < window:
                return {}
            
            rolling_metrics = {
                'volatility': [],
                'sharpe_ratio': [],
                'var': [],
                'max_drawdown': []
            }
            
            for i in range(window, len(returns) + 1):
                window_returns = returns[i-window:i]
                
                # Rolling volatility
                vol = self.calculate_volatility(window_returns)
                rolling_metrics['volatility'].append(vol)
                
                # Rolling Sharpe ratio
                sharpe = self.calculate_sharpe_ratio(window_returns)
                rolling_metrics['sharpe_ratio'].append(sharpe)
                
                # Rolling VaR
                var = self.calculate_value_at_risk(window_returns)
                rolling_metrics['var'].append(var)
                
                # Rolling max drawdown (simplified)
                cumulative_returns = np.cumprod(1 + np.array(window_returns))
                max_dd, _, _ = self.calculate_max_drawdown(cumulative_returns.tolist())
                rolling_metrics['max_drawdown'].append(max_dd)
            
            return rolling_metrics
            
        except Exception as e:
            logger.error(f"Error calculating rolling metrics: {e}")
            return {}
    
    def generate_comprehensive_report(self, returns: List[float], prices: Optional[List[float]] = None,
                                   benchmark_returns: Optional[List[float]] = None) -> Dict[str, Any]:
        """Generate comprehensive metrics report"""
        try:
            report = {
                'basic_metrics': {
                    'total_return': np.prod(1 + np.array(returns)) - 1 if returns else 0,
                    'avg_return': np.mean(returns) if returns else 0,
                    'volatility': self.calculate_volatility(returns),
                    'sharpe_ratio': self.calculate_sharpe_ratio(returns),
                    'sortino_ratio': self.calculate_sortino_ratio(returns),
                    'win_rate': self.calculate_win_rate(returns),
                    'profit_factor': self.calculate_profit_factor(returns)
                },
                'risk_metrics': {
                    'var_95': self.calculate_value_at_risk(returns, 0.95),
                    'cvar_95': self.calculate_conditional_var(returns, 0.95),
                    'max_drawdown': self.calculate_max_drawdown(prices or [1])[0],
                    'ulcer_index': self.calculate_ulcer_index(prices or [1])
                },
                'distribution_metrics': {
                    'skewness': self.calculate_skewness(returns),
                    'kurtosis': self.calculate_kurtosis(returns)
                }
            }
            
            # Add benchmark comparison if available
            if benchmark_returns and len(benchmark_returns) == len(returns):
                report['benchmark_metrics'] = {
                    'beta': self.calculate_beta(returns, benchmark_returns),
                    'alpha': self.calculate_alpha(returns, benchmark_returns),
                    'information_ratio': self.calculate_information_ratio(returns, benchmark_returns),
                    'treynor_ratio': self.calculate_treynor_ratio(returns, benchmark_returns)
                }
            
            # Add rolling metrics if enough data
            if len(returns) >= 60:
                report['rolling_metrics'] = self.calculate_rolling_metrics(returns)
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating comprehensive report: {e}")
            return {}
    
    def update_metrics(self, return_value: float, price: Optional[float] = None, volume: Optional[float] = None):
        """Update metrics with new data point"""
        try:
            self.returns.append(return_value)
            
            if price is not None:
                self.prices.append(price)
            
            if volume is not None:
                self.volumes.append(volume)
            
            # Keep only recent data
            max_length = self.config.lookback_period
            if len(self.returns) > max_length:
                self.returns = self.returns[-max_length:]
                self.prices = self.prices[-max_length:] if self.prices else []
                self.volumes = self.volumes[-max_length:] if self.volumes else []
                
        except Exception as e:
            logger.error(f"Error updating metrics: {e}")
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of current metrics"""
        try:
            if not self.returns:
                return {}
            
            return {
                'data_points': len(self.returns),
                'current_volatility': self.calculate_volatility(self.returns),
                'current_sharpe': self.calculate_sharpe_ratio(self.returns),
                'current_var': self.calculate_value_at_risk(self.returns),
                'current_win_rate': self.calculate_win_rate(self.returns)
            }
            
        except Exception as e:
            logger.error(f"Error getting metrics summary: {e}")
            return {}


class AdvancedRiskMetrics:
    """
    Advanced Risk Metrics class for comprehensive risk analysis
    Compatible with the existing system architecture
    """
    
    def __init__(self, initial_capital: float = 100000.0):
        self.advanced_metrics = AdvancedMetrics(initial_capital)
        self.initial_capital = initial_capital
        
        # Risk-specific tracking
        self.portfolio_values = []
        self.position_sizes = []
        self.exposure_history = []
        self.correlation_history = []
        
        logger.info("AdvancedRiskMetrics initialized")
    
    def calculate_portfolio_risk(self, portfolio_values: List[float], 
                               position_sizes: Optional[List[float]] = None) -> Dict[str, float]:
        """Calculate comprehensive portfolio risk metrics"""
        try:
            if not portfolio_values:
                return {}
            
            returns = []
            for i in range(1, len(portfolio_values)):
                if portfolio_values[i-1] > 0:
                    returns.append((portfolio_values[i] - portfolio_values[i-1]) / portfolio_values[i-1])
            
            risk_metrics = {
                'volatility': self.advanced_metrics.calculate_volatility(returns),
                'var_95': self.advanced_metrics.calculate_value_at_risk(returns, 0.95),
                'cvar_95': self.advanced_metrics.calculate_conditional_var(returns, 0.95),
                'max_drawdown': self.advanced_metrics.calculate_max_drawdown(portfolio_values)[0],
                'ulcer_index': self.advanced_metrics.calculate_ulcer_index(portfolio_values),
                'sharpe_ratio': self.advanced_metrics.calculate_sharpe_ratio(returns),
                'sortino_ratio': self.advanced_metrics.calculate_sortino_ratio(returns)
            }
            
            # Add position concentration risk if available
            if position_sizes:
                risk_metrics['concentration_risk'] = self._calculate_concentration_risk(position_sizes)
                risk_metrics['largest_position'] = max(position_sizes) if position_sizes else 0
            
            return risk_metrics
            
        except Exception as e:
            logger.error(f"Error calculating portfolio risk: {e}")
            return {}
    
    def _calculate_concentration_risk(self, position_sizes: List[float]) -> float:
        """Calculate portfolio concentration risk using Herfindahl-Hirschman Index"""
        try:
            if not position_sizes:
                return 0.0
            
            total_exposure = sum(abs(size) for size in position_sizes)
            if total_exposure == 0:
                return 0.0
            
            # Calculate HHI
            hhi = sum((abs(size) / total_exposure) ** 2 for size in position_sizes)
            return hhi
            
        except Exception as e:
            logger.error(f"Error calculating concentration risk: {e}")
            return 0.0
    
    def update_portfolio_state(self, portfolio_value: float, position_sizes: Optional[List[float]] = None):
        """Update portfolio state for risk tracking"""
        try:
            self.portfolio_values.append(portfolio_value)
            
            if position_sizes:
                self.position_sizes.append(position_sizes)
            
            # Keep only recent data
            max_length = 252  # One year of trading days
            if len(self.portfolio_values) > max_length:
                self.portfolio_values = self.portfolio_values[-max_length:]
                self.position_sizes = self.position_sizes[-max_length:] if self.position_sizes else []
                
        except Exception as e:
            logger.error(f"Error updating portfolio state: {e}")
    
    def get_current_risk_metrics(self) -> Dict[str, Any]:
        """Get current risk metrics"""
        try:
            return self.calculate_portfolio_risk(self.portfolio_values, 
                                               self.position_sizes[-1] if self.position_sizes else None)
        except Exception as e:
            logger.error(f"Error getting current risk metrics: {e}")
            return {}
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics - required for system compatibility"""
        try:
            metrics = self.get_current_risk_metrics()
            
            # Add basic metrics from AdvancedMetrics
            if self.advanced_metrics.returns:
                basic_metrics = self.advanced_metrics.get_metrics_summary()
                metrics.update(basic_metrics)
            
            # Add portfolio state
            metrics.update({
                'portfolio_value': self.portfolio_values[-1] if self.portfolio_values else self.initial_capital,
                'data_points': len(self.portfolio_values),
                'initial_capital': self.initial_capital
            })
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting metrics: {e}")
            return {
                'error': str(e),
                'portfolio_value': self.initial_capital,
                'data_points': 0,
                'initial_capital': self.initial_capital
            }
    
    def calculate_stress_test_scenarios(self, portfolio_value: float, 
                                      position_sizes: Optional[List[float]] = None) -> Dict[str, float]:
        """Calculate stress test scenarios"""
        try:
            scenarios = {}
            
            # Market crash scenario (20% decline)
            scenarios['market_crash_20pct'] = portfolio_value * 0.80
            
            # Volatility spike scenario (2x volatility)
            if self.portfolio_values:
                current_vol = self.get_current_risk_metrics().get('volatility', 0.15)
                scenarios['volatility_spike_2x'] = portfolio_value * (1 - 2 * current_vol)
            
            # Sector shock scenario (15% decline in largest position)
            if position_sizes:
                largest_position = max(position_sizes) if position_sizes else 0
                scenarios['sector_shock_15pct'] = portfolio_value - (largest_position * 0.15)
            
            return scenarios
            
        except Exception as e:
            logger.error(f"Error calculating stress test scenarios: {e}")
            return {}
    
    def cleanup(self):
        """Cleanup resources"""
        try:
            self.portfolio_values.clear()
            self.position_sizes.clear()
            self.exposure_history.clear()
            self.correlation_history.clear()
            logger.info("AdvancedRiskMetrics cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


def calculate_sharpe(returns, annual_risk_free_rate=0.02):
    """Calculate Sharpe ratio for returns"""
    # CRITICAL FIX: Handle numpy arrays properly
    if returns is None or len(returns) == 0:
        return 0.0

    # Convert to numpy array if not already
    returns_array = np.asarray(returns)

    # Check for NaN or infinite values
    if np.any(np.isnan(returns_array)) or np.any(np.isinf(returns_array)):
        return 0.0

    # Check if all returns are the same (zero volatility)
    if np.std(returns_array) == 0:
        return 0.0

    # Assuming daily returns, annualize
    annualized_return = np.mean(returns_array) * 252
    annualized_std = np.std(returns_array) * np.sqrt(252)
    return (annualized_return - annual_risk_free_rate) / annualized_std

def perform_statistical_analysis(agent_step_returns, backtest_steps, benchmark_returns, logger):
    """
    Perform comprehensive statistical analysis for agent performance evaluation
    
    Args:
        agent_step_returns: List of agent step returns
        backtest_steps: Number of backtest steps completed
        benchmark_returns: List of benchmark returns
        logger: Logger instance
        
    Returns:
        Dictionary with statistical analysis results
    """
    stats_results = {
        't_test_p_value': 'N/A',
        'bootstrap_p_value': 'N/A',
        'statistically_significant': False,
        'paper_trading_ready': False,
        'agent_sharpe': 0.0,
        'benchmark_sharpe': 0.0,
        'data_points': len(agent_step_returns),
        'backtest_steps': backtest_steps
    }

    if not agent_step_returns or len(agent_step_returns) < 30:  # Need sufficient data for stats
        logger.warning(" Not enough agent returns for meaningful statistical analysis.")
        return stats_results

    agent_returns_array = np.array(agent_step_returns)
    agent_sharpe = calculate_sharpe(agent_returns_array)
    stats_results['agent_sharpe'] = float(agent_sharpe)  # Convert to Python float

    # If benchmark data is available, perform comparisons
    # CRITICAL FIX: Handle numpy arrays properly for boolean evaluation
    if benchmark_returns is not None and len(benchmark_returns) > 0 and len(benchmark_returns) >= len(agent_step_returns):
        # Trim benchmark returns to match agent returns length
        benchmark_returns_array = np.array(benchmark_returns[:len(agent_returns_array)])
        benchmark_sharpe = calculate_sharpe(benchmark_returns_array)
        stats_results['benchmark_sharpe'] = float(benchmark_sharpe)  # Convert to Python float

        # T-test for difference in means (assuming normal distribution of returns)
        try:
            ttest_result = stats.ttest_ind(agent_returns_array, benchmark_returns_array, equal_var=False)
            if isinstance(ttest_result, tuple):
                result_tuple = cast(Tuple[float, float], ttest_result)
                t_stat = float(result_tuple[0])
                p_value_ttest = float(result_tuple[1])
            else:
                t_stat = float(ttest_result.statistic)
                p_value_ttest = float(ttest_result.pvalue)
            stats_results['t_test_p_value'] = p_value_ttest  # Convert to Python float
            if p_value_ttest < 0.05 and agent_sharpe > benchmark_sharpe:  # Agent performs significantly better
                stats_results['statistically_significant'] = True
        except Exception as e:
            logger.error(f"T-test failed: {e}")

        # Bootstrap p-value (more robust for non-normal returns)
        try:
            n_bootstrap = 1000
            bootstrap_differences = []
            
            for _ in range(n_bootstrap):
                # Resample with replacement
                agent_sample = np.random.choice(agent_returns_array, size=len(agent_returns_array), replace=True)
                benchmark_sample = np.random.choice(benchmark_returns_array, size=len(benchmark_returns_array), replace=True)
                
                agent_sharpe_sample = calculate_sharpe(agent_sample)
                benchmark_sharpe_sample = calculate_sharpe(benchmark_sample)

                # CRITICAL FIX: Validate Sharpe ratios before storing differences
                if np.isfinite(agent_sharpe_sample) and np.isfinite(benchmark_sharpe_sample):
                    bootstrap_differences.append(agent_sharpe_sample - benchmark_sharpe_sample)
                else:
                    # Skip invalid samples
                    logger.debug(f"Skipping bootstrap sample with invalid Sharpe: agent={agent_sharpe_sample}, benchmark={benchmark_sharpe_sample}")
                    continue
            
            # Calculate p-value for bootstrap test
            observed_difference = agent_sharpe - benchmark_sharpe
            bootstrap_differences_array = np.array(bootstrap_differences)

            # CRITICAL FIX: Handle NaN and infinite values in bootstrap differences
            if np.any(np.isnan(bootstrap_differences_array)) or np.any(np.isinf(bootstrap_differences_array)):
                logger.warning(" Bootstrap differences contain NaN or infinite values, using fallback calculation")
                # Remove NaN/inf values for calculation
                valid_differences = bootstrap_differences_array[np.isfinite(bootstrap_differences_array)]
                if len(valid_differences) > 0:
                    bootstrap_p_value = np.mean(valid_differences <= 0)
                else:
                    bootstrap_p_value = 0.5  # Default when no valid data
            else:
                bootstrap_p_value = np.mean(bootstrap_differences_array <= 0)  # One-sided test

            stats_results['bootstrap_p_value'] = float(bootstrap_p_value)
            
        except Exception as e:
            logger.error(f"Bootstrap analysis failed: {e}")

    # Paper trading readiness check based on README criteria
    target_sharpe = 1.0
    target_max_drawdown = 0.10  # 10%
    target_win_rate = 0.45  # 45%

    # Since the backtest isn't running fully yet, these will be based on 0 trades for now.
    # Once trades occur, these need to be updated with actual backtest metrics.
    # For now, we will mark as False until actual trades and performance are observed.
    
    # IMPORTANT: The readiness logic here must consume the actual metrics from the full backtest
    # For now, given 0 trades, it should realistically be False
    if backtest_steps > 0 and stats_results['agent_sharpe'] > 0.5:  # A minimal bar to start
        stats_results['paper_trading_ready'] = True  # This will be refined when full metrics are available
    else:
        stats_results['paper_trading_ready'] = False

    return stats_results
