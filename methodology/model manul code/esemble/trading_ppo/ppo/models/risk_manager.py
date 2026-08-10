"""
Risk manager for PPO Trading Agent
Handles position sizing, stop-loss, and risk limits
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import logging

from ..core.exceptions import ValidationException

class RiskManager:
    """Risk management for trading positions"""
    
    def __init__(self, trading_config: Dict):
        """
        Initialize risk manager
        
        Args:
            trading_config: Trading configuration dictionary
        """
        self.config = trading_config
        self.logger = logging.getLogger(__name__)
        
        # Risk parameters
        self.max_position_risk_pct = trading_config.get('max_position_risk_pct', 0.25)
        self.max_portfolio_risk_pct = trading_config.get('max_portfolio_risk_pct', 0.60)
        self.stop_loss_pct = trading_config.get('stop_loss_pct', 0.05)
        self.take_profit_pct = trading_config.get('take_profit_pct', 0.10)
        self.max_drawdown_pct = trading_config.get('max_drawdown_pct', 0.10)
        
        # Position tracking
        self.position_entries = {}  # Track entry prices
        self.daily_pnl = 0.0
        self.peak_portfolio_value = 0.0
        
    def check_position_risk(self, action: np.ndarray, current_prices: np.ndarray, 
                           portfolio_value: float, positions: np.ndarray, 
                           strict: bool = False) -> Tuple[np.ndarray, List[str]]:
        """
        Check and adjust actions based on risk limits
        
        Args:
            action: Action array from agent
            current_prices: Current prices for all tickers
            portfolio_value: Current portfolio value
            positions: Current positions
            strict: Whether to apply strict risk limits
            
        Returns:
            Tuple of (adjusted_action, risk_violations)
        """
        adjusted_action = action.copy()
        violations = []
        
        try:
            # Calculate position values
            position_values = positions * current_prices
            total_position_value = np.sum(np.abs(position_values))
            effective_portfolio_value = max(float(portfolio_value), 1e-9)
            
            # Check portfolio risk limit
            if total_position_value > effective_portfolio_value * self.max_portfolio_risk_pct:
                violations.append(f"Portfolio risk limit exceeded: {total_position_value:.2f} > {effective_portfolio_value * self.max_portfolio_risk_pct:.2f}")
                if strict:
                    # Reduce all actions proportionally
                    reduction_factor = (effective_portfolio_value * self.max_portfolio_risk_pct) / total_position_value
                    adjusted_action *= reduction_factor
            
            # Check individual position limits
            for i, (pos_value, price) in enumerate(zip(position_values, current_prices)):
                if price > 0:
                    position_pct = abs(pos_value) / effective_portfolio_value
                    if position_pct > self.max_position_risk_pct:
                        violations.append(f"Position {i} risk limit exceeded: {position_pct:.2%} > {self.max_position_risk_pct:.2%}")
                        if strict:
                            # Reduce action for this position
                            adjusted_action[i] *= self.max_position_risk_pct / position_pct
            
            # Check drawdown limit
            if self.peak_portfolio_value > 0:
                current_drawdown = (self.peak_portfolio_value - portfolio_value) / self.peak_portfolio_value
                if current_drawdown > self.max_drawdown_pct:
                    violations.append(f"Drawdown limit exceeded: {current_drawdown:.2%} > {self.max_drawdown_pct:.2%}")
                    if strict:
                        # Stop trading if drawdown limit exceeded
                        adjusted_action *= 0.0
            
        except Exception as e:
            self.logger.error(f"Risk check failed: {e}")
            violations.append(f"Risk check error: {e}")
        
        return adjusted_action, violations
    
    def apply_stop_loss_take_profit(self, action: np.ndarray, current_prices: np.ndarray,
                                   positions: np.ndarray, portfolio_value: float) -> np.ndarray:
        """
        Apply stop-loss and take-profit orders
        
        Args:
            action: Action array from agent
            current_prices: Current prices for all tickers
            positions: Current positions
            portfolio_value: Current portfolio value
            
        Returns:
            Adjusted action array
        """
        adjusted_action = action.copy()
        
        try:
            for i, (position, price) in enumerate(zip(positions, current_prices)):
                if position != 0 and price > 0:
                    entry_price = self.position_entries.get(i, price)
                    
                    if position > 0:  # Long position
                        # Check stop loss
                        if price < entry_price * (1 - self.stop_loss_pct):
                            adjusted_action[i] = -1.0  # Force sell
                            self.logger.info(f"Stop loss triggered for position {i}: {price:.2f} < {entry_price * (1 - self.stop_loss_pct):.2f}")
                        
                        # Check take profit
                        elif price > entry_price * (1 + self.take_profit_pct):
                            adjusted_action[i] = -0.5  # Partial sell
                            self.logger.info(f"Take profit triggered for position {i}: {price:.2f} > {entry_price * (1 + self.take_profit_pct):.2f}")
                    
                    elif position < 0:  # Short position
                        # Check stop loss
                        if price > entry_price * (1 + self.stop_loss_pct):
                            adjusted_action[i] = 1.0  # Force buy
                            self.logger.info(f"Stop loss triggered for short position {i}: {price:.2f} > {entry_price * (1 + self.stop_loss_pct):.2f}")
                        
                        # Check take profit
                        elif price < entry_price * (1 - self.take_profit_pct):
                            adjusted_action[i] = 0.5  # Partial buy
                            self.logger.info(f"Take profit triggered for short position {i}: {price:.2f} < {entry_price * (1 - self.take_profit_pct):.2f}")
        
        except Exception as e:
            self.logger.error(f"Stop-loss/take-profit application failed: {e}")
        
        return adjusted_action
    
    def update_position_entry(self, position_idx: int, price: float, position: float):
        """
        Update entry price for a position
        
        Args:
            position_idx: Position index
            price: Current price
            position: Current position size
        """
        try:
            if position != 0:
                self.position_entries[position_idx] = price
            else:
                # Clear entry price if position is closed
                self.position_entries.pop(position_idx, None)
        except Exception as e:
            self.logger.error(f"Failed to update position entry: {e}")
    
    def update_portfolio_peak(self, portfolio_value: float):
        """Update peak portfolio value for drawdown calculation"""
        if portfolio_value > self.peak_portfolio_value:
            self.peak_portfolio_value = portfolio_value
    
    def get_risk_metrics(self, portfolio_value: float, positions: np.ndarray, 
                        current_prices: np.ndarray) -> Dict:
        """
        Get current risk metrics
        
        Args:
            portfolio_value: Current portfolio value
            positions: Current positions
            current_prices: Current prices
            
        Returns:
            Dictionary of risk metrics
        """
        try:
            position_values = positions * current_prices
            total_position_value = np.sum(np.abs(position_values))
            
            # Calculate risk metrics
            effective_portfolio_value = max(float(portfolio_value), 1e-9)
            portfolio_risk_pct = total_position_value / effective_portfolio_value
            max_position_pct = np.max(np.abs(position_values)) / effective_portfolio_value
            current_drawdown = (self.peak_portfolio_value - portfolio_value) / self.peak_portfolio_value if self.peak_portfolio_value > 0 else 0
            
            return {
                'portfolio_risk_pct': portfolio_risk_pct,
                'max_position_pct': max_position_pct,
                'current_drawdown': current_drawdown,
                'total_position_value': total_position_value,
                'num_positions': np.count_nonzero(positions)
            }
            
        except Exception as e:
            self.logger.error(f"Risk metrics calculation failed: {e}")
            return {}
    
    def should_stop_trading(self, portfolio_value: float) -> bool:
        """
        Check if trading should be stopped due to risk limits
        
        Args:
            portfolio_value: Current portfolio value
            
        Returns:
            True if trading should be stopped
        """
        try:
            # Check drawdown limit
            if self.peak_portfolio_value > 0:
                current_drawdown = (self.peak_portfolio_value - portfolio_value) / self.peak_portfolio_value
                if current_drawdown > self.max_drawdown_pct:
                    self.logger.warning(f"Trading stopped due to drawdown limit: {current_drawdown:.2%}")
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Stop trading check failed: {e}")
            return False
    
    def calculate_portfolio_hedge(self, positions: np.ndarray, current_prices: np.ndarray,
                                portfolio_value: float, ticker_names: Optional[List[str]] = None) -> Dict:
        """
        Calculate portfolio hedging positions to reduce risk
        
        Args:
            positions: Current positions
            current_prices: Current prices
            portfolio_value: Current portfolio value
            ticker_names: List of ticker names
            
        Returns:
            Dictionary with hedging recommendations
        """
        try:
            # Initialize hedging parameters
            hedge_threshold = self.config.get('hedge_threshold', 0.02)  # 2% correlation threshold
            max_hedge_pct = self.config.get('max_hedge_pct', 0.15)  # 15% max hedge size
            
            # Calculate position correlations
            position_values = positions * current_prices
            total_position_value = np.sum(np.abs(position_values))
            
            if total_position_value == 0:
                return {'hedge_positions': [], 'hedge_cost': 0.0, 'risk_reduction': 0.0}
            
            # Calculate portfolio beta (simplified)
            portfolio_beta = np.sum(position_values) / total_position_value if total_position_value > 0 else 0
            
            # Determine hedge positions
            hedge_positions = []
            hedge_cost = 0.0
            
            # If portfolio has high positive beta, hedge with inverse positions
            if abs(portfolio_beta) > hedge_threshold:
                # Calculate hedge size
                hedge_size = min(abs(portfolio_beta) * total_position_value * 0.5, 
                               portfolio_value * max_hedge_pct)
                
                # Create hedge positions (simplified - use first few tickers as hedge)
                num_hedge_tickers = min(3, len(positions))
                hedge_per_ticker = hedge_size / num_hedge_tickers
                
                for i in range(num_hedge_tickers):
                    if current_prices[i] > 0:
                        hedge_shares = -hedge_per_ticker / current_prices[i]  # Inverse position
                        hedge_positions.append({
                            'ticker': ticker_names[i] if ticker_names else f'TICKER_{i}',
                            'shares': hedge_shares,
                            'value': hedge_per_ticker,
                            'type': 'correlation_hedge'
                        })
                        hedge_cost += abs(hedge_per_ticker) * 0.0029  # Commission cost
            
            # Calculate risk reduction
            risk_reduction = abs(portfolio_beta) * 0.3 if abs(portfolio_beta) > hedge_threshold else 0.0
            
            return {
                'hedge_positions': hedge_positions,
                'hedge_cost': hedge_cost,
                'risk_reduction': risk_reduction,
                'portfolio_beta': portfolio_beta,
                'total_hedge_value': sum([h['value'] for h in hedge_positions])
            }
            
        except Exception as e:
            self.logger.error(f"Portfolio hedge calculation failed: {e}")
            return {'hedge_positions': [], 'hedge_cost': 0.0, 'risk_reduction': 0.0}
    
    def apply_dynamic_stop_loss(self, positions: np.ndarray, current_prices: np.ndarray,
                               volatility_window: int = 20) -> Dict:
        """
        Apply dynamic stop-loss based on volatility
        
        Args:
            positions: Current positions
            current_prices: Current prices
            volatility_window: Window for volatility calculation
            
        Returns:
            Dictionary with stop-loss recommendations
        """
        try:
            # Initialize stop-loss tracking
            if not hasattr(self, 'stop_loss_levels'):
                self.stop_loss_levels = {}
            
            # Calculate volatility-adjusted stop-loss levels
            stop_loss_recommendations = []
            total_stop_loss_value = 0.0
            
            for i, (position, price) in enumerate(zip(positions, current_prices)):
                if abs(position) > 0 and price > 0:
                    # Calculate position-specific volatility (simplified)
                    volatility = 0.02  # Default 2% volatility
                    
                    # Adjust stop-loss based on volatility
                    if volatility > 0.03:  # High volatility
                        stop_loss_pct = self.stop_loss_pct * 1.5  # Wider stop
                    elif volatility < 0.01:  # Low volatility
                        stop_loss_pct = self.stop_loss_pct * 0.7  # Tighter stop
                    else:
                        stop_loss_pct = self.stop_loss_pct
                    
                    # Calculate stop-loss level
                    if position > 0:  # Long position
                        stop_loss_price = price * (1 - stop_loss_pct)
                    else:  # Short position
                        stop_loss_price = price * (1 + stop_loss_pct)
                    
                    # Store stop-loss level
                    self.stop_loss_levels[i] = {
                        'price': stop_loss_price,
                        'type': 'long' if position > 0 else 'short',
                        'volatility': volatility,
                        'stop_loss_pct': stop_loss_pct
                    }
                    
                    stop_loss_recommendations.append({
                        'position_idx': i,
                        'stop_loss_price': stop_loss_price,
                        'current_price': price,
                        'position': position,
                        'stop_loss_pct': stop_loss_pct
                    })
                    
                    total_stop_loss_value += abs(position * price)
            
            return {
                'stop_loss_recommendations': stop_loss_recommendations,
                'total_stop_loss_value': total_stop_loss_value,
                'num_active_stops': len(stop_loss_recommendations)
            }
            
        except Exception as e:
            self.logger.error(f"Dynamic stop-loss calculation failed: {e}")
            return {'stop_loss_recommendations': [], 'total_stop_loss_value': 0.0, 'num_active_stops': 0}
