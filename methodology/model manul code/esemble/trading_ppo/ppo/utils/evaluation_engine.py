"""
Advanced Evaluation Engine
Based on TG4.1 generate_evaluation_summary implementation
"""

import os
import time
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)

class EvaluationEngine:
    """Advanced evaluation engine like TG4.1"""
    
    def __init__(self):
        self.evaluation_results = {}
        self.benchmark_data = {}
        self.portfolio_metrics = {}
        
        # Evaluation settings
        self.risk_free_rate = 0.02  # 2% risk-free rate
        self.trading_days_per_year = 252
        
        logger.info(" EvaluationEngine initialized")
    
    def generate_evaluation_summary(self, results: Dict, benchmark_returns: Optional[np.ndarray] = None) -> Dict:
        """Generate comprehensive evaluation summary like TG4.1"""
        try:
            logger.info(" Generating comprehensive evaluation summary...")
            
            # Extract basic results
            total_return = results.get('total_return', 0.0)
            sharpe_ratio = results.get('sharpe_ratio', 0.0)
            max_drawdown = results.get('max_drawdown', 0.0)
            total_trades = results.get('total_trades', 0)
            total_commission = results.get('total_commission', 0.0)
            final_portfolio = results.get('final_portfolio_value', 0.0)
            initial_portfolio = results.get('initial_portfolio_value', 0.0)
            
            # Calculate additional metrics
            additional_metrics = self._calculate_additional_metrics(results)
            
            # Benchmark comparison
            benchmark_analysis = self._analyze_benchmark_comparison(results, benchmark_returns)
            
            # Risk analysis
            risk_analysis = self._analyze_risk_metrics(results)
            
            # Performance attribution
            performance_attribution = self._analyze_performance_attribution(results)
            
            # Create comprehensive summary
            summary = {
                # Basic metrics
                'total_return': total_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'total_trades': total_trades,
                'total_commission': total_commission,
                'final_portfolio_value': final_portfolio,
                'initial_portfolio_value': initial_portfolio,
                
                # Additional metrics
                'additional_metrics': additional_metrics,
                
                # Benchmark analysis
                'benchmark_analysis': benchmark_analysis,
                
                # Risk analysis
                'risk_analysis': risk_analysis,
                
                # Performance attribution
                'performance_attribution': performance_attribution,
                
                # Metadata
                'evaluation_timestamp': datetime.now().isoformat(),
                'evaluation_version': 'TG4.1_compatible'
            }
            
            # Store results
            self.evaluation_results = summary
            
            logger.info(" Evaluation summary generated successfully")
            return summary
            
        except Exception as e:
            logger.error(f" Evaluation summary generation failed: {e}")
            return {'error': str(e)}
    
    def _calculate_additional_metrics(self, results: Dict) -> Dict:
        """Calculate additional performance metrics"""
        try:
            portfolio_history = results.get('portfolio_history', [])
            trade_history = results.get('trade_history', [])
            
            if not portfolio_history:
                return {}
            
            # Calculate returns
            returns = np.diff(portfolio_history) / portfolio_history[:-1]
            
            # Basic metrics
            avg_return = np.mean(returns) if len(returns) > 0 else 0.0
            return_std = np.std(returns) if len(returns) > 0 else 0.0
            return_skew = self._calculate_skewness(returns)
            return_kurtosis = self._calculate_kurtosis(returns)
            
            # Volatility metrics
            annualized_volatility = return_std * np.sqrt(self.trading_days_per_year)
            
            # Risk-adjusted metrics
            sortino_ratio = self._calculate_sortino_ratio(returns)
            calmar_ratio = self._calculate_calmar_ratio(results)
            information_ratio = self._calculate_information_ratio(returns)
            
            # Trading metrics
            win_rate = self._calculate_win_rate(trade_history)
            avg_trade_return = self._calculate_avg_trade_return(trade_history)
            profit_factor = self._calculate_profit_factor(trade_history)
            
            # Drawdown metrics
            drawdown_metrics = self._calculate_drawdown_metrics(portfolio_history)
            
            return {
                'avg_return': avg_return,
                'return_std': return_std,
                'return_skew': return_skew,
                'return_kurtosis': return_kurtosis,
                'annualized_volatility': annualized_volatility,
                'sortino_ratio': sortino_ratio,
                'calmar_ratio': calmar_ratio,
                'information_ratio': information_ratio,
                'win_rate': win_rate,
                'avg_trade_return': avg_trade_return,
                'profit_factor': profit_factor,
                'drawdown_metrics': drawdown_metrics
            }
            
        except Exception as e:
            logger.error(f" Additional metrics calculation error: {e}")
            return {}
    
    def _analyze_benchmark_comparison(self, results: Dict, benchmark_returns: Optional[np.ndarray]) -> Dict:
        """Analyze benchmark comparison"""
        try:
            if benchmark_returns is None:
                return {'status': 'No benchmark data available'}
            
            portfolio_history = results.get('portfolio_history', [])
            if len(portfolio_history) < 2:
                return {'status': 'Insufficient portfolio data'}
            
            # Calculate portfolio returns
            portfolio_returns = np.diff(portfolio_history) / portfolio_history[:-1]
            
            # Align lengths
            min_length = min(len(portfolio_returns), len(benchmark_returns))
            portfolio_returns = portfolio_returns[:min_length]
            benchmark_returns = benchmark_returns[:min_length]
            
            # Calculate metrics
            alpha = np.mean(portfolio_returns) - np.mean(benchmark_returns)
            beta = self._calculate_beta(portfolio_returns, benchmark_returns)
            correlation = np.corrcoef(portfolio_returns, benchmark_returns)[0, 1]
            
            # Tracking error
            tracking_error = np.std(portfolio_returns - benchmark_returns)
            
            # Information ratio
            information_ratio = alpha / tracking_error if tracking_error > 0 else 0.0
            
            # Treynor ratio
            treynor_ratio = alpha / beta if beta != 0 else 0.0
            
            return {
                'alpha': alpha,
                'beta': beta,
                'correlation': correlation,
                'tracking_error': tracking_error,
                'information_ratio': information_ratio,
                'treynor_ratio': treynor_ratio,
                'benchmark_return': np.sum(benchmark_returns),
                'portfolio_return': np.sum(portfolio_returns),
                'excess_return': np.sum(portfolio_returns) - np.sum(benchmark_returns)
            }
            
        except Exception as e:
            logger.error(f" Benchmark analysis error: {e}")
            return {'status': f'Benchmark analysis failed: {str(e)}'}
    
    def _analyze_risk_metrics(self, results: Dict) -> Dict:
        """Analyze risk metrics"""
        try:
            portfolio_history = results.get('portfolio_history', [])
            if len(portfolio_history) < 2:
                return {'status': 'Insufficient portfolio data'}
            
            returns = np.diff(portfolio_history) / portfolio_history[:-1]
            
            # Value at Risk (VaR)
            var_95 = np.percentile(returns, 5)
            var_99 = np.percentile(returns, 1)
            
            # Conditional Value at Risk (CVaR)
            cvar_95 = np.mean(returns[returns <= var_95])
            cvar_99 = np.mean(returns[returns <= var_99])
            
            # Maximum drawdown
            max_dd = results.get('max_drawdown', 0.0)
            
            # Downside deviation
            downside_returns = returns[returns < 0]
            downside_deviation = np.std(downside_returns) if len(downside_returns) > 0 else 0.0
            
            # Semi-deviation
            semi_deviation = np.std(returns[returns < np.mean(returns)])
            
            # Ulcer index
            ulcer_index = self._calculate_ulcer_index(portfolio_history)
            
            return {
                'var_95': var_95,
                'var_99': var_99,
                'cvar_95': cvar_95,
                'cvar_99': cvar_99,
                'max_drawdown': max_dd,
                'downside_deviation': downside_deviation,
                'semi_deviation': semi_deviation,
                'ulcer_index': ulcer_index
            }
            
        except Exception as e:
            logger.error(f" Risk analysis error: {e}")
            return {'status': f'Risk analysis failed: {str(e)}'}
    
    def _analyze_performance_attribution(self, results: Dict) -> Dict:
        """Analyze performance attribution"""
        try:
            trade_history = results.get('trade_history', [])
            if not trade_history:
                return {'status': 'No trade data available'}
            
            # Analyze by ticker
            ticker_performance = {}
            for trade in trade_history:
                ticker = trade.get('ticker', 'unknown')
                if ticker not in ticker_performance:
                    ticker_performance[ticker] = {
                        'trades': 0,
                        'total_return': 0.0,
                        'total_commission': 0.0,
                        'wins': 0,
                        'losses': 0
                    }
                
                ticker_performance[ticker]['trades'] += 1
                ticker_performance[ticker]['total_commission'] += trade.get('commission', 0.0)
                
                # Calculate trade return
                shares = trade.get('shares', 0)
                price = trade.get('price', 0)
                if shares != 0 and price != 0:
                    trade_return = shares * price
                    ticker_performance[ticker]['total_return'] += trade_return
                    
                    if trade_return > 0:
                        ticker_performance[ticker]['wins'] += 1
                    else:
                        ticker_performance[ticker]['losses'] += 1
            
            # Calculate attribution metrics
            total_trades = sum(tp['trades'] for tp in ticker_performance.values())
            total_return = sum(tp['total_return'] for tp in ticker_performance.values())
            
            attribution = {}
            for ticker, metrics in ticker_performance.items():
                if total_trades > 0:
                    attribution[ticker] = {
                        'trade_contribution': metrics['trades'] / total_trades,
                        'return_contribution': metrics['total_return'] / total_return if total_return != 0 else 0,
                        'win_rate': metrics['wins'] / metrics['trades'] if metrics['trades'] > 0 else 0,
                        'avg_commission': metrics['total_commission'] / metrics['trades'] if metrics['trades'] > 0 else 0
                    }
            
            return {
                'ticker_attribution': attribution,
                'total_trades': total_trades,
                'total_return': total_return
            }
            
        except Exception as e:
            logger.error(f" Performance attribution error: {e}")
            return {'status': f'Performance attribution failed: {str(e)}'}
    
    def _calculate_skewness(self, returns: np.ndarray) -> float:
        """Calculate skewness"""
        if len(returns) < 3:
            return 0.0
        return float(np.mean(((returns - np.mean(returns)) / np.std(returns)) ** 3))
    
    def _calculate_kurtosis(self, returns: np.ndarray) -> float:
        """Calculate kurtosis"""
        if len(returns) < 4:
            return 0.0
        return float(np.mean(((returns - np.mean(returns)) / np.std(returns)) ** 4) - 3)
    
    def _calculate_sortino_ratio(self, returns: np.ndarray) -> float:
        """Calculate Sortino ratio"""
        if len(returns) == 0:
            return 0.0
        
        avg_return = np.mean(returns)
        downside_returns = returns[returns < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 1e-8
        
        return float((avg_return - self.risk_free_rate / self.trading_days_per_year) / downside_std)
    
    def _calculate_calmar_ratio(self, results: Dict) -> float:
        """Calculate Calmar ratio"""
        total_return = results.get('total_return', 0.0)
        max_drawdown = abs(results.get('max_drawdown', 0.0))
        
        if max_drawdown == 0:
            return 0.0
        
        return total_return / max_drawdown
    
    def _calculate_information_ratio(self, returns: np.ndarray) -> float:
        """Calculate information ratio"""
        if len(returns) == 0:
            return 0.0
        
        excess_returns = returns - self.risk_free_rate / self.trading_days_per_year
        return float(np.mean(excess_returns) / np.std(excess_returns)) if np.std(excess_returns) > 0 else 0.0
    
    def _calculate_win_rate(self, trade_history: List[Dict]) -> float:
        """Calculate win rate"""
        if not trade_history:
            return 0.0
        
        wins = sum(1 for trade in trade_history if trade.get('shares', 0) > 0)
        return wins / len(trade_history)
    
    def _calculate_avg_trade_return(self, trade_history: List[Dict]) -> float:
        """Calculate average trade return"""
        if not trade_history:
            return 0.0
        
        total_return = sum(trade.get('shares', 0) * trade.get('price', 0) for trade in trade_history)
        return total_return / len(trade_history)
    
    def _calculate_profit_factor(self, trade_history: List[Dict]) -> float:
        """Calculate profit factor"""
        if not trade_history:
            return 0.0
        
        gross_profit = sum(trade.get('shares', 0) * trade.get('price', 0) 
                          for trade in trade_history 
                          if trade.get('shares', 0) * trade.get('price', 0) > 0)
        
        gross_loss = abs(sum(trade.get('shares', 0) * trade.get('price', 0) 
                            for trade in trade_history 
                            if trade.get('shares', 0) * trade.get('price', 0) < 0))
        
        return gross_profit / gross_loss if gross_loss > 0 else 0.0
    
    def _calculate_drawdown_metrics(self, portfolio_history: List[float]) -> Dict:
        """Calculate drawdown metrics"""
        if len(portfolio_history) < 2:
            return {}
        
        try:
            # Calculate drawdowns
            peak = portfolio_history[0]
            drawdowns = []
            current_dd = 0
            
            for value in portfolio_history:
                if value > peak:
                    peak = value
                    current_dd = 0
                else:
                    current_dd = (value - peak) / peak
                    drawdowns.append(current_dd)
            
            if not drawdowns:
                return {}
            
            return {
                'max_drawdown': min(drawdowns),
                'avg_drawdown': np.mean(drawdowns),
                'drawdown_std': np.std(drawdowns),
                'drawdown_count': len([dd for dd in drawdowns if dd < -0.05])  # Count significant drawdowns
            }
            
        except Exception as e:
            logger.error(f" Drawdown metrics calculation error: {e}")
            return {}
    
    def _calculate_beta(self, portfolio_returns: np.ndarray, benchmark_returns: np.ndarray) -> float:
        """Calculate beta"""
        if len(portfolio_returns) != len(benchmark_returns) or len(portfolio_returns) < 2:
            return 1.0
        
        covariance = np.cov(portfolio_returns, benchmark_returns)[0, 1]
        benchmark_variance = np.var(benchmark_returns)
        
        return covariance / benchmark_variance if benchmark_variance > 0 else 1.0
    
    def _calculate_ulcer_index(self, portfolio_history: List[float]) -> float:
        """Calculate Ulcer Index"""
        if len(portfolio_history) < 2:
            return 0.0
        
        try:
            # Calculate drawdowns
            peak = portfolio_history[0]
            squared_drawdowns = []
            
            for value in portfolio_history:
                if value > peak:
                    peak = value
                    squared_drawdowns.append(0)
                else:
                    drawdown = (value - peak) / peak
                    squared_drawdowns.append(drawdown ** 2)
            
            return np.sqrt(np.mean(squared_drawdowns))
            
        except Exception as e:
            logger.error(f" Ulcer index calculation error: {e}")
            return 0.0
    
    def save_evaluation_results(self, filepath: str = 'ppo/saved_models/evaluation_results.json'):
        """Save evaluation results"""
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            with open(filepath, 'w') as f:
                json.dump(self.evaluation_results, f, indent=2, default=str)
            
            logger.info(f" Evaluation results saved to {filepath}")
            
        except Exception as e:
            logger.error(f" Failed to save evaluation results: {e}")
    
    def load_evaluation_results(self, filepath: str = 'ppo/saved_models/evaluation_results.json') -> bool:
        """Load evaluation results"""
        try:
            if not os.path.exists(filepath):
                logger.warning(f"  No evaluation results found at {filepath}")
                return False
            
            with open(filepath, 'r') as f:
                self.evaluation_results = json.load(f)
            
            logger.info(f" Evaluation results loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f" Failed to load evaluation results: {e}")
            return False
