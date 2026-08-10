"""
Advanced Analytics Module for PPO Trading System
Provides comprehensive analytics, risk metrics, and performance monitoring
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple
import time
import logging
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
import os  # CRITICAL FIX: Add missing os import

@dataclass
class AnalyticsConfig:
    """Configuration for advanced analytics"""
    var_confidence_level: float = 0.95
    max_lookback_days: int = 252
    sector_mapping: Optional[Dict[str, str]] = None
    stress_test_scenarios: Optional[Dict[str, float]] = None
    
    def __post_init__(self):
        if self.sector_mapping is None:
            self.sector_mapping = {
                'AAPL': 'technology', 'MSFT': 'technology', 'NVDA': 'technology',
                'JPM': 'financial', 'SPY': 'etf', 'GLD': 'commodity'
            }
        if self.stress_test_scenarios is None:
            self.stress_test_scenarios = {
                'volatility_shock_2x': 2.0,
                'volatility_shock_3x': 3.0,
                'market_crash_10pct': -0.10,
                'market_crash_20pct': -0.20
            }

class AdvancedAnalytics:
    """Advanced analytics and monitoring for trading system"""
    
    def __init__(self, config: Optional[AnalyticsConfig] = None):
        """Initialize analytics system"""
        self.config = config or AnalyticsConfig()
        self.logger = logging.getLogger(__name__)
        
        # Data storage
        self.returns = deque(maxlen=self.config.max_lookback_days)
        self.portfolio_values = deque(maxlen=self.config.max_lookback_days)
        self.positions = deque(maxlen=self.config.max_lookback_days)
        self.benchmark_returns = deque(maxlen=self.config.max_lookback_days)
        self.execution_times = defaultdict(list)
        
        # Calculated metrics
        self.risk_metrics = {}
        self.portfolio_analytics = {}
        self.performance_attribution = {}
        self.stress_test_results = {}
        
        # Timing
        self.start_time = time.time()
        
        self.logger.info("AdvancedAnalytics initialized")
    
        # COMPREHENSIVE ANALYTICS METHODS: Enhanced implementation for full functionality
    def add_portfolio_value(self, value: float):
        """Adds a portfolio value snapshot to history."""
        if np.isfinite(value):
            self.portfolio_values.append(float(value))
            # Also maintain portfolio_history for compatibility
            if not hasattr(self, 'portfolio_history'):
                self.portfolio_history = []
            self.portfolio_history.append(float(value))
    
    def add_return(self, daily_return: float):
        """Adds a daily return to history."""
        if np.isfinite(daily_return):
            self.returns.append(float(daily_return))
            # Also maintain returns_history for compatibility
            if not hasattr(self, 'returns_history'):
                self.returns_history = []
            self.returns_history.append(float(daily_return))
    
    def add_trade(self, trade_data: dict):
        """Adds a trade record to history."""
        if not hasattr(self, 'trade_history'):
            self.trade_history = []
        self.trade_history.append(trade_data)
    
    def add_position(self, position_data: dict):
        """Adds a position snapshot to history."""
        if not hasattr(self, 'position_history'):
            self.position_history = []
        self.position_history.append(position_data)
    
    def set_benchmark_returns(self, returns_list: list):
        """Sets the benchmark returns for comparison."""
        self.benchmark_returns.clear()
        for ret in returns_list:
            if isinstance(ret, (int, float)) and np.isfinite(ret):
                self.benchmark_returns.append(float(ret))
    
    def _initialize_data_structures(self):
        """Initialize additional data structures for comprehensive analytics."""
        self.sector_exposures = {}
        self.factor_exposures = {}
        self.volatility_history = []
        self.drawdown_history = []
        self.sharpe_history = []
        self.max_drawdown = 0.0
        self.peak_value = 0.0
        self.risk_metrics_history = []
        self.performance_attribution_history = []
        
        # Initialize portfolio tracking
        if not hasattr(self, 'portfolio_history'):
            self.portfolio_history = []
        if not hasattr(self, 'returns_history'):
            self.returns_history = []
        if not hasattr(self, 'trade_history'):
            self.trade_history = []
        if not hasattr(self, 'position_history'):
            self.position_history = []
    
    def add_portfolio_data(self, portfolio_value: float, positions: Dict, 
                          returns: Optional[float] = None, benchmark_return: Optional[float] = None):
        """Add new portfolio data point"""
        try:
            if portfolio_value is not None:
                self.portfolio_values.append(portfolio_value)
            
            if positions is not None:
                self.positions.append(positions)
            
            if returns is not None:
                self.returns.append(returns)
            
            if benchmark_return is not None:
                self.benchmark_returns.append(benchmark_return)
                
        except Exception as e:
            self.logger.error(f" Error adding portfolio data: {e}")
    
    def calculate_risk_metrics(self) -> Dict:
        """Calculate comprehensive risk metrics"""
        try:
            if not self.returns:
                return {}
            
            returns_array = np.array(list(self.returns))
            
            # Basic risk metrics
            volatility = np.std(returns_array) * np.sqrt(252)  # Annualized
            sharpe_ratio = np.mean(returns_array) / np.std(returns_array) * np.sqrt(252) if np.std(returns_array) > 0 else 0
            
            # Value at Risk
            var_percentile = (1 - self.config.var_confidence_level) * 100
            var = np.percentile(returns_array, var_percentile)
            cvar = np.mean(returns_array[returns_array <= var])
            
            # Drawdown analysis
            if self.portfolio_values:
                portfolio_array = np.array(list(self.portfolio_values))
                cumulative_returns = (portfolio_array / portfolio_array[0]) - 1
                running_max = np.maximum.accumulate(cumulative_returns)
                drawdown = cumulative_returns - running_max
                max_drawdown = np.min(drawdown)
            else:
                max_drawdown = 0
            
            # Skewness and kurtosis
            skewness = self._calculate_skewness(returns_array)
            kurtosis = self._calculate_kurtosis(returns_array)
            
            # Ulcer index
            ulcer_index = self._calculate_ulcer_index(returns_array)
            
            self.risk_metrics = {
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'var_95': var,
                'cvar_95': cvar,
                'max_drawdown': max_drawdown,
                'skewness': skewness,
                'kurtosis': kurtosis,
                'ulcer_index': ulcer_index
            }
            
            return self.risk_metrics
            
        except Exception as e:
            self.logger.error(f" Error calculating risk metrics: {e}")
            return {}
    
    def calculate_portfolio_analytics(self) -> Dict:
        """Calculate portfolio analytics"""
        try:
            if not self.positions:
                return {}
            
            # Current positions
            current_positions = list(self.positions)[-1] if self.positions else {}
            
            # Sector allocation
            sector_weights = defaultdict(float)
            total_value = 0
            
            sector_mapping = self.config.sector_mapping or {}
            for ticker, position in current_positions.items():
                if isinstance(position, dict) and 'value' in position:
                    value = position['value']
                    sector = sector_mapping.get(ticker, 'other')
                    sector_weights[sector] += value
                    total_value += value
            
            # Normalize sector weights
            if total_value > 0:
                sector_weights = {sector: weight/total_value for sector, weight in sector_weights.items()}
            
            # Concentration metrics
            hhi = sum(weight**2 for weight in sector_weights.values())
            
            # Top 5 concentration
            sorted_weights = sorted(sector_weights.values(), reverse=True)
            top_5_concentration = sum(sorted_weights[:5])
            
            # Largest position
            largest_position = max(sector_weights.values()) if sector_weights else 0
            
            self.portfolio_analytics = {
                'total_positions': len(current_positions),
                'total_portfolio_value': total_value,
                'sector_allocation': dict(sector_weights),
                'hhi_concentration': hhi,
                'top_5_concentration': top_5_concentration,
                'largest_position': largest_position
            }
            
            return self.portfolio_analytics
            
        except Exception as e:
            self.logger.error(f" Error calculating portfolio analytics: {e}")
            return {}
    
    def calculate_performance_attribution(self) -> Dict:
        """Calculate performance attribution using Brinson model"""
        try:
            if not self.returns or not self.benchmark_returns:
                return {}
            
            # Simple attribution analysis
            returns_array = np.array(list(self.returns))
            benchmark_array = np.array(list(self.benchmark_returns))
            
            # Excess return
            excess_return = np.mean(returns_array) - np.mean(benchmark_array)
            
            # Information ratio
            tracking_error = np.std(returns_array - benchmark_array)
            information_ratio = excess_return / tracking_error if tracking_error > 0 else 0
            
            # Beta calculation
            if len(returns_array) > 1 and len(benchmark_array) > 1:
                covariance = np.cov(returns_array, benchmark_array)[0, 1]
                benchmark_variance = np.var(benchmark_array)
                beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
            else:
                beta = 0
            
            # Alpha calculation
            alpha = excess_return - (beta * np.mean(benchmark_array))
            
            self.performance_attribution = {
                'excess_return': excess_return,
                'information_ratio': information_ratio,
                'beta': beta,
                'alpha': alpha,
                'tracking_error': tracking_error
            }
            
            return self.performance_attribution
            
        except Exception as e:
            self.logger.error(f" Error calculating performance attribution: {e}")
            return {}
    
    def run_stress_tests(self) -> Dict:
        """Run stress tests on portfolio"""
        try:
            if not self.returns:
                return {}
            
            returns_array = np.array(list(self.returns))
            base_volatility = np.std(returns_array)
            
            stress_results = {}
            
            stress_scenarios = self.config.stress_test_scenarios or {}
            for scenario_name, scenario_factor in stress_scenarios.items():
                if 'volatility' in scenario_name:
                    # Volatility stress test
                    stressed_returns = returns_array * scenario_factor
                    stressed_loss = np.sum(stressed_returns[stressed_returns < 0])
                else:
                    # Market crash stress test
                    stressed_returns = returns_array + scenario_factor
                    stressed_loss = np.sum(stressed_returns[stressed_returns < 0])
                
                stress_results[scenario_name] = stressed_loss
            
            self.stress_test_results = stress_results
            return stress_results
            
        except Exception as e:
            self.logger.error(f" Error running stress tests: {e}")
            return {}
    
    def track_execution_time(self, component: str, duration: float):
        """Track execution time for components"""
        self.execution_times[component].append(duration)
    
    def generate_comprehensive_report(self) -> Dict:
        """Generate comprehensive analytics report with enhanced metrics"""
        try:
            # Calculate all metrics
            risk_metrics = self.calculate_risk_metrics()
            portfolio_analytics = self.calculate_portfolio_analytics()
            performance_attribution = self.calculate_performance_attribution()
            stress_test_results = self.run_stress_tests()
            
            # Enhanced portfolio analysis
            portfolio_summary = self._calculate_portfolio_summary()
            trade_analysis = self._analyze_trades()
            performance_metrics = self._calculate_performance_metrics()
            
            report = {
                'timestamp': datetime.now().isoformat(),
                'monitoring_duration': time.time() - self.start_time if self.start_time else 0,
                
                # Core metrics
                'risk_metrics': risk_metrics,
                'portfolio_analytics': portfolio_analytics,
                'performance_attribution': performance_attribution,
                'stress_test_results': stress_test_results,
                
                # Enhanced analytics
                'portfolio_summary': portfolio_summary,
                'trade_analysis': trade_analysis,
                'performance_metrics': performance_metrics,
                
                # Execution tracking
                'execution_times': dict(self.execution_times),
                
                # Data points
                'data_points': {
                    'returns_count': len(self.returns),
                    'portfolio_values_count': len(self.portfolio_values),
                    'positions_count': len(self.positions),
                    'benchmark_returns_count': len(self.benchmark_returns),
                    'trade_history_count': len(getattr(self, 'trade_history', [])),
                    'position_history_count': len(getattr(self, 'position_history', []))
                },
                
                # Historical data
                'trade_history': getattr(self, 'trade_history', []),
                'portfolio_history': list(self.portfolio_values),
                'returns_history': list(self.returns),
                'position_history': getattr(self, 'position_history', []),
                
                # Risk tracking
                'volatility_history': getattr(self, 'volatility_history', []),
                'drawdown_history': getattr(self, 'drawdown_history', []),
                'sharpe_history': getattr(self, 'sharpe_history', [])
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f" Error generating comprehensive report: {e}")
            return {}
    
    def _calculate_portfolio_summary(self) -> Dict:
        """Calculate comprehensive portfolio summary metrics"""
        try:
            if not self.portfolio_values:
                return {}
            
            portfolio_array = np.array(list(self.portfolio_values))
            
            # Basic metrics
            initial_value = portfolio_array[0]
            final_value = portfolio_array[-1]
            total_return = (final_value - initial_value) / initial_value
            
            # Peak and drawdown analysis
            peak_value = np.maximum.accumulate(portfolio_array)
            drawdown = (portfolio_array - peak_value) / peak_value
            max_drawdown = np.min(drawdown)
            
            # Volatility
            returns = np.diff(portfolio_array) / portfolio_array[:-1]
            volatility = np.std(returns) * np.sqrt(252) if len(returns) > 1 else 0
            
            # Sharpe ratio
            avg_return = np.mean(returns) * 252 if len(returns) > 0 else 0
            sharpe_ratio = avg_return / volatility if volatility > 0 else 0
            
            return {
                'initial_value': float(initial_value),
                'final_value': float(final_value),
                'total_return': float(total_return),
                'max_drawdown': float(max_drawdown),
                'volatility': float(volatility),
                'sharpe_ratio': float(sharpe_ratio),
                'avg_return': float(avg_return),
                'peak_value': float(np.max(portfolio_array)),
                'trough_value': float(np.min(portfolio_array))
            }
        except Exception as e:
            self.logger.error(f" Error calculating portfolio summary: {e}")
            return {}
    
    def _analyze_trades(self) -> Dict:
        """Analyze trade performance and patterns"""
        try:
            trade_history = getattr(self, 'trade_history', [])
            if not trade_history:
                return {}
            
            # Trade statistics
            total_trades = len(trade_history)
            buy_trades = [t for t in trade_history if t.get('action_type') == 'BUY']
            sell_trades = [t for t in trade_history if t.get('action_type') == 'SELL']
            
            # Calculate P&L for completed trades
            realized_pnl = []
            for trade in trade_history:
                if 'realized_pnl' in trade and trade['realized_pnl'] is not None:
                    realized_pnl.append(trade['realized_pnl'])
            
            # Win rate analysis
            winning_trades = [pnl for pnl in realized_pnl if pnl > 0]
            losing_trades = [pnl for pnl in realized_pnl if pnl < 0]
            
            win_rate = len(winning_trades) / len(realized_pnl) if realized_pnl else 0
            avg_win = np.mean(winning_trades) if winning_trades else 0
            avg_loss = np.mean(losing_trades) if losing_trades else 0
            profit_factor = abs(sum(winning_trades) / sum(losing_trades)) if losing_trades and sum(losing_trades) != 0 else 0
            
            return {
                'total_trades': total_trades,
                'buy_trades': len(buy_trades),
                'sell_trades': len(sell_trades),
                'win_rate': float(win_rate),
                'avg_win': float(avg_win),
                'avg_loss': float(avg_loss),
                'profit_factor': float(profit_factor),
                'total_realized_pnl': float(sum(realized_pnl)),
                'largest_win': float(max(realized_pnl)) if realized_pnl else 0,
                'largest_loss': float(min(realized_pnl)) if realized_pnl else 0
            }
        except Exception as e:
            self.logger.error(f" Error analyzing trades: {e}")
            return {}
    
    def _calculate_performance_metrics(self) -> Dict:
        """Calculate advanced performance metrics"""
        try:
            if not self.returns:
                return {}
            
            returns_array = np.array(list(self.returns))
            
            # Risk-adjusted metrics
            sharpe_ratio = np.mean(returns_array) / np.std(returns_array) * np.sqrt(252) if np.std(returns_array) > 0 else 0
            sortino_ratio = np.mean(returns_array) / np.std(returns_array[returns_array < 0]) * np.sqrt(252) if len(returns_array[returns_array < 0]) > 0 else 0
            
            # Maximum metrics
            max_consecutive_wins = self._calculate_max_consecutive(returns_array > 0)
            max_consecutive_losses = self._calculate_max_consecutive(returns_array < 0)
            
            # Calmar ratio (annual return / max drawdown)
            if hasattr(self, 'portfolio_history') and self.portfolio_history:
                portfolio_array = np.array(self.portfolio_history)
                peak_value = np.maximum.accumulate(portfolio_array)
                drawdown = (portfolio_array - peak_value) / peak_value
                max_drawdown = abs(np.min(drawdown))
                annual_return = np.mean(returns_array) * 252
                calmar_ratio = annual_return / max_drawdown if max_drawdown > 0 else 0
            else:
                calmar_ratio = 0
            
            return {
                'sharpe_ratio': float(sharpe_ratio),
                'sortino_ratio': float(sortino_ratio),
                'calmar_ratio': float(calmar_ratio),
                'max_consecutive_wins': int(max_consecutive_wins),
                'max_consecutive_losses': int(max_consecutive_losses),
                'var_95': float(np.percentile(returns_array, 5)),
                'cvar_95': float(np.mean(returns_array[returns_array <= np.percentile(returns_array, 5)]))
            }
        except Exception as e:
            self.logger.error(f" Error calculating performance metrics: {e}")
            return {}
    
    def _calculate_max_consecutive(self, condition_array: np.ndarray) -> int:
        """Calculate maximum consecutive occurrences of a condition"""
        try:
            if len(condition_array) == 0:
                return 0
            
            max_consecutive = 0
            current_consecutive = 0
            
            for condition in condition_array:
                if condition:
                    current_consecutive += 1
                    max_consecutive = max(max_consecutive, current_consecutive)
                else:
                    current_consecutive = 0
            
            return max_consecutive
        except Exception:
            return 0
    
    def _calculate_skewness(self, returns: np.ndarray) -> float:
        """Calculate skewness of returns"""
        try:
            if len(returns) < 3:
                return 0.0
            mean = np.mean(returns)
            std = np.std(returns)
            if std == 0:
                return 0.0
            skewness = np.mean(((returns - mean) / std) ** 3)
            return float(skewness)
        except Exception:
            return 0.0
    
    def _calculate_kurtosis(self, returns: np.ndarray) -> float:
        """Calculate kurtosis of returns"""
        try:
            if len(returns) < 4:
                return 0.0
            mean = np.mean(returns)
            std = np.std(returns)
            if std == 0:
                return 0.0
            kurtosis = np.mean(((returns - mean) / std) ** 4) - 3
            return float(kurtosis)
        except Exception:
            return 0.0
    
    def _calculate_ulcer_index(self, returns: np.ndarray) -> float:
        """Calculate Ulcer Index (measure of downside risk)"""
        try:
            if len(returns) < 2:
                return 0.0
            
            # Calculate cumulative returns
            cumulative = np.cumprod(1 + returns)
            
            # Calculate drawdowns
            running_max = np.maximum.accumulate(cumulative)
            drawdowns = (cumulative - running_max) / running_max
            
            # Calculate Ulcer Index
            ulcer_index = np.sqrt(np.mean(drawdowns ** 2))
            return float(ulcer_index)
            
        except Exception:
            return 0.0
    
    def create_visualizations(self, save_path: Optional[str] = None):
        """Create comprehensive visualizations with enhanced analytics"""
        try:
            if not self.returns and not self.portfolio_values:
                self.logger.warning(" No data available for visualizations")
                return
            
            # Create comprehensive dashboard with multiple subplots
            fig = plt.figure(figsize=(20, 15))
            fig.suptitle('Comprehensive Portfolio Analytics Dashboard', fontsize=18, fontweight='bold')
            
            # Create grid layout
            gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
            
            # 1. Portfolio Value Over Time (main chart)
            ax1 = fig.add_subplot(gs[0:2, 0:2])
            if self.portfolio_values:
                portfolio_array = np.array(list(self.portfolio_values))
                ax1.plot(portfolio_array, color='#2E86AB', linewidth=2, label='Portfolio Value')
                
                # Add peak line
                peak_values = np.maximum.accumulate(portfolio_array)
                ax1.plot(peak_values, color='#A23B72', linestyle='--', alpha=0.7, label='Peak Value')
                
                ax1.set_title('Portfolio Performance Over Time', fontsize=14, fontweight='bold')
                ax1.set_xlabel('Time Steps')
                ax1.set_ylabel('Portfolio Value ($)')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
            
            # 2. Returns Distribution
            ax2 = fig.add_subplot(gs[0, 2])
            if self.returns:
                returns_array = np.array(list(self.returns))
                ax2.hist(returns_array, bins=30, alpha=0.7, color='#F18F01', edgecolor='black')
                ax2.set_title('Returns Distribution', fontsize=12, fontweight='bold')
                ax2.set_xlabel('Return')
                ax2.set_ylabel('Frequency')
                ax2.axvline(x=0, color='red', linestyle='--', alpha=0.7)
            
            # 3. Drawdown Analysis
            ax3 = fig.add_subplot(gs[1, 2])
            if self.portfolio_values:
                portfolio_array = np.array(list(self.portfolio_values))
                peak_values = np.maximum.accumulate(portfolio_array)
                drawdown = (portfolio_array - peak_values) / peak_values * 100
                ax3.fill_between(range(len(drawdown)), drawdown, 0, color='#C73E1D', alpha=0.6)
                ax3.plot(drawdown, color='#C73E1D', linewidth=1)
                ax3.set_title('Drawdown Analysis', fontsize=12, fontweight='bold')
                ax3.set_xlabel('Time Steps')
                ax3.set_ylabel('Drawdown (%)')
                ax3.grid(True, alpha=0.3)
            
            # 4. Risk Metrics Bar Chart
            ax4 = fig.add_subplot(gs[2, 0])
            risk_metrics = self.calculate_risk_metrics()
            if risk_metrics:
                metrics_names = ['Volatility', 'Sharpe', 'Max DD', 'VaR']
                metrics_values = [
                    risk_metrics.get('volatility', 0),
                    risk_metrics.get('sharpe_ratio', 0),
                    abs(risk_metrics.get('max_drawdown', 0)),
                    abs(risk_metrics.get('var_95', 0))
                ]
                colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
                bars = ax4.bar(metrics_names, metrics_values, color=colors, alpha=0.8)
                ax4.set_title('Key Risk Metrics', fontsize=12, fontweight='bold')
                ax4.set_ylabel('Value')
                
                # Add value labels on bars
                for bar, value in zip(bars, metrics_values):
                    height = bar.get_height()
                    ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                            f'{value:.3f}', ha='center', va='bottom', fontsize=10)
            
            # 5. Trade Analysis
            ax5 = fig.add_subplot(gs[2, 1])
            trade_analysis = self._analyze_trades()
            if trade_analysis and trade_analysis.get('total_trades', 0) > 0:
                trade_metrics = ['Win Rate', 'Profit Factor', 'Avg Win', 'Avg Loss']
                trade_values = [
                    trade_analysis.get('win_rate', 0) * 100,  # Convert to percentage
                    trade_analysis.get('profit_factor', 0),
                    trade_analysis.get('avg_win', 0),
                    abs(trade_analysis.get('avg_loss', 0))
                ]
                colors = ['#2ECC71', '#F39C12', '#3498DB', '#E74C3C']
                bars = ax5.bar(trade_metrics, trade_values, color=colors, alpha=0.8)
                ax5.set_title('Trade Performance Metrics', fontsize=12, fontweight='bold')
                ax5.set_ylabel('Value')
                
                # Add value labels
                for bar, value in zip(bars, trade_values):
                    height = bar.get_height()
                    ax5.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                            f'{value:.2f}', ha='center', va='bottom', fontsize=10)
            
            # 6. Sector Allocation (if available)
            ax6 = fig.add_subplot(gs[2, 2])
            portfolio_analytics = self.calculate_portfolio_analytics()
            if portfolio_analytics and 'sector_allocation' in portfolio_analytics:
                sector_data = portfolio_analytics['sector_allocation']
                if sector_data:
                    sectors = list(sector_data.keys())
                    weights = list(sector_data.values())
                    colors = [tuple(color) for color in plt.get_cmap("Set3")(np.linspace(0, 1, len(sectors)))]
                    _ = ax6.pie(weights, labels=sectors, autopct='%1.1f%%', colors=colors)
                    ax6.set_title('Sector Allocation', fontsize=12, fontweight='bold')
            
            # 7. Performance Summary Table
            ax7 = fig.add_subplot(gs[3, :])
            ax7.axis('tight')
            ax7.axis('off')
            
            # Create summary table
            portfolio_summary = self._calculate_portfolio_summary()
            if portfolio_summary:
                summary_data = [
                    ['Metric', 'Value'],
                    ['Total Return', f"{portfolio_summary.get('total_return', 0)*100:.2f}%"],
                    ['Sharpe Ratio', f"{portfolio_summary.get('sharpe_ratio', 0):.3f}"],
                    ['Max Drawdown', f"{portfolio_summary.get('max_drawdown', 0)*100:.2f}%"],
                    ['Volatility', f"{portfolio_summary.get('volatility', 0)*100:.2f}%"],
                    ['Final Value', f"${portfolio_summary.get('final_value', 0):,.2f}"],
                    ['Total Trades', f"{trade_analysis.get('total_trades', 0)}"],
                    ['Win Rate', f"{trade_analysis.get('win_rate', 0)*100:.1f}%"]
                ]
                
                table = ax7.table(cellText=summary_data[1:], colLabels=summary_data[0],
                                cellLoc='center', loc='center')
                table.auto_set_font_size(False)
                table.set_fontsize(10)
                table.scale(1.2, 1.5)
                
                # Style the table
                for i in range(len(summary_data)):
                    for j in range(len(summary_data[0])):
                        if i == 0:  # Header row
                            table[(i, j)].set_facecolor('#34495E')
                            table[(i, j)].set_text_props(weight='bold', color='white')
                        else:
                            table[(i, j)].set_facecolor('#ECF0F1')
                
                ax7.set_title('Performance Summary', fontsize=14, fontweight='bold', pad=20)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f" Comprehensive visualizations saved to: {save_path}")
            else:
                # Don't show GUI - just save to default location
                default_path = 'ppo/analytics/visualizations/comprehensive_dashboard.png'
                os.makedirs(os.path.dirname(default_path), exist_ok=True)
                plt.savefig(default_path, dpi=300, bbox_inches='tight')
                self.logger.info(f" Comprehensive visualizations saved to: {default_path}")
            
            # Always close the figure to free memory
            plt.close(fig)
                
        except Exception as e:
            self.logger.error(f" Error creating visualizations: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
    
    def save_figures(self, output_dir='ppo/analytics_plots'):
        """Saves generated figures to files."""
        try:
            import os
            os.makedirs(output_dir, exist_ok=True)
            
            # Create visualizations and save them
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(output_dir, f'analytics_dashboard_{timestamp}.png')
            
            self.create_visualizations(save_path=save_path)
            self.logger.info(f" Saved analytics figures to: {output_dir}")
            
        except Exception as e:
            self.logger.warning(f" Could not save figures: {e}")
    
    def reset(self):
        """Reset all analytics data"""
        self.returns.clear()
        self.portfolio_values.clear()
        self.positions.clear()
        self.benchmark_returns.clear()
        self.execution_times.clear()
        self.risk_metrics = {}
        self.portfolio_analytics = {}
        self.performance_attribution = {}
        self.stress_test_results = {}
        
        # Reset enhanced data structures
        if hasattr(self, 'portfolio_history'):
            self.portfolio_history.clear()
        if hasattr(self, 'returns_history'):
            self.returns_history.clear()
        if hasattr(self, 'trade_history'):
            self.trade_history.clear()
        if hasattr(self, 'position_history'):
            self.position_history.clear()
        if hasattr(self, 'volatility_history'):
            self.volatility_history.clear()
        if hasattr(self, 'drawdown_history'):
            self.drawdown_history.clear()
        if hasattr(self, 'sharpe_history'):
            self.sharpe_history.clear()
        
        self.start_time = time.time()
        self.logger.info(" Analytics data reset")
    
    def update_real_time_metrics(self):
        """Update real-time performance metrics"""
        try:
            if len(self.portfolio_values) > 1:
                # Update volatility history
                recent_returns = np.diff(list(self.portfolio_values)[-20:]) / list(self.portfolio_values)[-21:-1]
                if len(recent_returns) > 0:
                    volatility = np.std(recent_returns) * np.sqrt(252)
                    self.volatility_history.append(volatility)
                
                # Update drawdown history
                portfolio_array = np.array(list(self.portfolio_values))
                peak_values = np.maximum.accumulate(portfolio_array)
                current_drawdown = (portfolio_array[-1] - peak_values[-1]) / peak_values[-1]
                self.drawdown_history.append(current_drawdown)
                
                # Update Sharpe history
                if len(self.returns) > 0:
                    recent_returns = list(self.returns)[-20:]
                    if len(recent_returns) > 1:
                        sharpe = np.mean(recent_returns) / np.std(recent_returns) * np.sqrt(252) if np.std(recent_returns) > 0 else 0
                        self.sharpe_history.append(sharpe)
                
                self.logger.debug(f" Real-time metrics updated: Vol={volatility:.4f}, DD={current_drawdown:.4f}")
        except Exception as e:
            self.logger.warning(f" Error updating real-time metrics: {e}")
    
    def get_latest_metrics(self) -> Dict:
        """Get the latest performance metrics"""
        try:
            latest_metrics = {
                'current_portfolio_value': self.portfolio_values[-1] if self.portfolio_values else 0,
                'current_volatility': self.volatility_history[-1] if self.volatility_history else 0,
                'current_drawdown': self.drawdown_history[-1] if self.drawdown_history else 0,
                'current_sharpe': self.sharpe_history[-1] if self.sharpe_history else 0,
                'total_trades': len(getattr(self, 'trade_history', [])),
                'win_rate': self._analyze_trades().get('win_rate', 0) * 100
            }
            return latest_metrics
        except Exception as e:
            self.logger.warning(f" Error getting latest metrics: {e}")
            return {}
