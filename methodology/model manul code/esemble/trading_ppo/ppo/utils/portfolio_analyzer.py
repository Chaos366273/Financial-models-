"""
Portfolio Analyzer
Based on TG4.1 portfolio analysis implementation
"""

import logging
import numpy as np
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class PortfolioAnalyzer:
    """Advanced portfolio analysis like TG4.1"""
    
    def __init__(self):
        self.analysis_results = {}
        logger.info(" PortfolioAnalyzer initialized")
    
    def analyze_portfolio(self, portfolio_history: List[float], trade_history: List[Dict]) -> Dict:
        """Comprehensive portfolio analysis"""
        try:
            if not portfolio_history:
                return {'status': 'No portfolio data'}
            
            # Basic metrics
            total_return = (portfolio_history[-1] - portfolio_history[0]) / portfolio_history[0]
            volatility = np.std(np.diff(portfolio_history) / portfolio_history[:-1])
            
            # Risk metrics
            max_drawdown = self._calculate_max_drawdown(portfolio_history)
            sharpe_ratio = self._calculate_sharpe_ratio(portfolio_history)
            
            # Trading metrics
            trading_metrics = self._analyze_trading_metrics(trade_history)
            
            return {
                'total_return': total_return,
                'volatility': volatility,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe_ratio,
                'trading_metrics': trading_metrics
            }
            
        except Exception as e:
            logger.error(f" Portfolio analysis error: {e}")
            return {'status': f'Analysis failed: {str(e)}'}
    
    def _calculate_max_drawdown(self, portfolio_history: List[float]) -> float:
        """Calculate maximum drawdown"""
        if len(portfolio_history) < 2:
            return 0.0
        
        peak = portfolio_history[0]
        max_dd = 0.0
        
        for value in portfolio_history:
            if value > peak:
                peak = value
            else:
                dd = (value - peak) / peak
                max_dd = min(max_dd, dd)
        
        return max_dd
    
    def _calculate_sharpe_ratio(self, portfolio_history: List[float]) -> float:
        """Calculate Sharpe ratio"""
        if len(portfolio_history) < 2:
            return 0.0
        
        returns = np.diff(portfolio_history) / portfolio_history[:-1]
        avg_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0.0
        
        return avg_return / std_return * np.sqrt(252)
    
    def _analyze_trading_metrics(self, trade_history: List[Dict]) -> Dict:
        """Analyze trading metrics"""
        if not trade_history:
            return {}
        
        total_trades = len(trade_history)
        profitable_trades = sum(1 for trade in trade_history if trade.get('shares', 0) > 0)
        win_rate = profitable_trades / total_trades if total_trades > 0 else 0.0
        
        return {
            'total_trades': total_trades,
            'profitable_trades': profitable_trades,
            'win_rate': win_rate
        }
