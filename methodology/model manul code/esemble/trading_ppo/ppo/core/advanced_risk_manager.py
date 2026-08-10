"""
Advanced Risk Management System
Comprehensive risk controls and position management
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class RiskLimits:
    """Risk limit configuration"""
    max_position_size: float = 0.1  # 10% of portfolio
    max_portfolio_risk: float = 0.02  # 2% max portfolio risk
    max_drawdown: float = 0.15  # 15% max drawdown
    stop_loss_pct: float = 0.05  # 5% stop loss
    take_profit_pct: float = 0.10  # 10% take profit
    max_correlation: float = 0.7  # Max correlation between positions
    circuit_breaker_pct: float = 0.20  # 20% circuit breaker
    volatility_threshold: float = 0.03  # 3% volatility threshold

class AdvancedRiskManager:
    """
    Advanced risk management system with comprehensive controls
    """
    
    def __init__(self, initial_capital: float = 100000.0):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.peak_capital = initial_capital
        
        # Risk limits
        self.risk_limits = RiskLimits()
        
        # Position tracking
        self.positions = {}
        self.position_history = []
        self.risk_metrics = {}
        
        # Circuit breakers
        self.circuit_breaker_active = False
        self.circuit_breaker_triggered_at = None
        self.circuit_breaker_recovery_time = timedelta(hours=1)
        
        # Performance tracking
        self.daily_returns = []
        self.volatility_window = 20
        self.correlation_matrix = None
        
        # Risk alerts
        self.risk_alerts = []
        
        logger.info("AdvancedRiskManager initialized")
    
    def update_capital(self, new_capital: float):
        """Update current capital and track peak"""
        self.current_capital = new_capital
        self.peak_capital = max(self.peak_capital, new_capital)
        
        # Check circuit breaker
        self._check_circuit_breaker()
    
    def calculate_position_size(self, symbol: str, price: float, confidence: float = 1.0) -> float:
        """Calculate optimal position size using Kelly Criterion and risk limits"""
        try:
            # Get historical volatility
            volatility = self._calculate_volatility(symbol)
            
            # Kelly Criterion calculation
            win_rate = self._estimate_win_rate(symbol)
            avg_win = self._estimate_avg_win(symbol)
            avg_loss = self._estimate_avg_loss(symbol)
            
            if avg_loss == 0:
                kelly_fraction = 0.0
            else:
                kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
            
            # Apply risk limits
            max_position_value = self.current_capital * self.risk_limits.max_position_size
            max_shares = max_position_value / price
            
            # Volatility adjustment
            volatility_adjustment = max(0.1, 1 - volatility / self.risk_limits.volatility_threshold)
            
            # Confidence adjustment
            confidence_adjustment = max(0.1, confidence)
            
            # Final position size
            optimal_shares = max_shares * kelly_fraction * volatility_adjustment * confidence_adjustment
            
            # Ensure minimum position
            min_position_value = 100.0  # $100 minimum
            min_shares = min_position_value / price
            
            return max(min_shares, optimal_shares)
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0.0
    
    def validate_trade(self, symbol: str, action: int, shares: float, price: float) -> Tuple[bool, str]:
        """Validate if trade meets risk requirements"""
        try:
            # Check circuit breaker
            if self.circuit_breaker_active:
                return False, "Circuit breaker active"
            
            # Check position limits
            if action > 0:  # Buy
                position_value = shares * price
                if position_value > self.current_capital * self.risk_limits.max_position_size:
                    return False, f"Position size exceeds limit: ${position_value:.2f}"
            
            # Check portfolio risk
            portfolio_risk = self._calculate_portfolio_risk()
            if portfolio_risk > self.risk_limits.max_portfolio_risk:
                return False, f"Portfolio risk too high: {portfolio_risk:.4f}"
            
            # Check correlation limits
            if not self._check_correlation_limits(symbol):
                return False, "Correlation limit exceeded"
            
            # Check drawdown
            current_drawdown = (self.peak_capital - self.current_capital) / self.peak_capital
            if current_drawdown > self.risk_limits.max_drawdown:
                return False, f"Drawdown limit exceeded: {current_drawdown:.4f}"
            
            return True, "Trade validated"
            
        except Exception as e:
            logger.error(f"Error validating trade: {e}")
            return False, f"Validation error: {e}"
    
    def apply_stop_loss_take_profit(self, symbol: str, current_price: float) -> Tuple[bool, str]:
        """Check and apply stop-loss/take-profit orders"""
        if symbol not in self.positions:
            return False, "No position"
        
        position = self.positions[symbol]
        entry_price = position['entry_price']
        shares = position['shares']
        
        # Calculate P&L
        if shares > 0:  # Long position
            pnl_pct = (current_price - entry_price) / entry_price
        else:  # Short position
            pnl_pct = (entry_price - current_price) / entry_price
        
        # Check stop loss
        if pnl_pct < -self.risk_limits.stop_loss_pct:
            return True, f"Stop loss triggered: {pnl_pct:.4f}"
        
        # Check take profit
        if pnl_pct > self.risk_limits.take_profit_pct:
            return True, f"Take profit triggered: {pnl_pct:.4f}"
        
        return False, "No stop/take triggered"
    
    def update_position(self, symbol: str, shares: float, price: float, action: str):
        """Update position tracking"""
        try:
            if symbol not in self.positions:
                self.positions[symbol] = {
                    'shares': 0,
                    'entry_price': 0,
                    'total_cost': 0,
                    'last_update': datetime.now()
                }
            
            position = self.positions[symbol]
            
            if action == 'buy':
                # Calculate new average price
                total_shares = position['shares'] + shares
                total_cost = position['total_cost'] + (shares * price)
                
                if total_shares > 0:
                    position['entry_price'] = total_cost / total_shares
                
                position['shares'] = total_shares
                position['total_cost'] = total_cost
                
            elif action == 'sell':
                position['shares'] -= shares
                if position['shares'] <= 0:
                    # Position closed
                    del self.positions[symbol]
                    return
            
            position['last_update'] = datetime.now()
            
            # Update position history
            self.position_history.append({
                'symbol': symbol,
                'shares': shares,
                'price': price,
                'action': action,
                'timestamp': datetime.now()
            })
            
        except Exception as e:
            logger.error(f"Error updating position: {e}")
    
    def _check_circuit_breaker(self):
        """Check if circuit breaker should be triggered"""
        if self.circuit_breaker_active:
            # Check if recovery time has passed
            if (datetime.now() - self.circuit_breaker_triggered_at) > self.circuit_breaker_recovery_time:
                self.circuit_breaker_active = False
                logger.info("Circuit breaker deactivated")
            return
        
        # Check for circuit breaker conditions
        current_drawdown = (self.peak_capital - self.current_capital) / self.peak_capital
        
        if current_drawdown > self.risk_limits.circuit_breaker_pct:
            self.circuit_breaker_active = True
            self.circuit_breaker_triggered_at = datetime.now()
            logger.warning(f"Circuit breaker triggered: {current_drawdown:.4f} drawdown")
    
    def _calculate_volatility(self, symbol: str) -> float:
        """Calculate historical volatility for symbol"""
        try:
            # This would typically use historical price data
            # For now, return a default value
            return 0.02  # 2% default volatility
        except Exception as e:
            logger.error(f"Error calculating volatility: {e}")
            return 0.02
    
    def _estimate_win_rate(self, symbol: str) -> float:
        """Estimate win rate for symbol"""
        try:
            # This would use historical trade data
            # For now, return a default value
            return 0.55  # 55% default win rate
        except Exception as e:
            logger.error(f"Error estimating win rate: {e}")
            return 0.55
    
    def _estimate_avg_win(self, symbol: str) -> float:
        """Estimate average win for symbol"""
        try:
            # This would use historical trade data
            return 0.05  # 5% default avg win
        except Exception as e:
            logger.error(f"Error estimating avg win: {e}")
            return 0.05
    
    def _estimate_avg_loss(self, symbol: str) -> float:
        """Estimate average loss for symbol"""
        try:
            # This would use historical trade data
            return 0.03  # 3% default avg loss
        except Exception as e:
            logger.error(f"Error estimating avg loss: {e}")
            return 0.03
    
    def _calculate_portfolio_risk(self) -> float:
        """Calculate current portfolio risk"""
        try:
            if not self.positions:
                return 0.0
            
            total_risk = 0.0
            for symbol, position in self.positions.items():
                position_value = abs(position['shares'] * position.get('current_price', 0))
                position_risk = position_value / self.current_capital
                total_risk += position_risk
            
            return total_risk
            
        except Exception as e:
            logger.error(f"Error calculating portfolio risk: {e}")
            return 0.0
    
    def _check_correlation_limits(self, symbol: str) -> bool:
        """Check if new position violates correlation limits"""
        try:
            if len(self.positions) < 2:
                return True
            
            # This would calculate correlation with existing positions
            # For now, return True (allow trade)
            return True
            
        except Exception as e:
            logger.error(f"Error checking correlation limits: {e}")
            return True
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """Get comprehensive risk summary"""
        try:
            current_drawdown = (self.peak_capital - self.current_capital) / self.peak_capital
            portfolio_risk = self._calculate_portfolio_risk()
            
            return {
                'current_capital': self.current_capital,
                'peak_capital': self.peak_capital,
                'current_drawdown': current_drawdown,
                'portfolio_risk': portfolio_risk,
                'circuit_breaker_active': self.circuit_breaker_active,
                'active_positions': len(self.positions),
                'risk_alerts': len(self.risk_alerts),
                'risk_limits': {
                    'max_drawdown': self.risk_limits.max_drawdown,
                    'max_portfolio_risk': self.risk_limits.max_portfolio_risk,
                    'stop_loss_pct': self.risk_limits.stop_loss_pct,
                    'take_profit_pct': self.risk_limits.take_profit_pct
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting risk summary: {e}")
            return {}
    
    def add_risk_alert(self, alert_type: str, message: str, severity: str = 'warning'):
        """Add risk alert"""
        alert = {
            'type': alert_type,
            'message': message,
            'severity': severity,
            'timestamp': datetime.now()
        }
        self.risk_alerts.append(alert)
        logger.warning(f"Risk alert: {message}")
    
    def get_risk_alerts(self) -> List[Dict]:
        """Get recent risk alerts"""
        return self.risk_alerts[-10:]  # Last 10 alerts
