"""
Trade Validator Module

Validates trades before execution to ensure they meet all constraints and requirements.
"""

import logging
from typing import Dict, Any, Tuple, Optional
import numpy as np

logger = logging.getLogger(__name__)

class TradeValidator:
    """
    Validates trades before execution to prevent invalid operations
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize trade validator with configuration
        
        Args:
            config: Configuration dictionary with validation parameters
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Validation parameters
        self.min_trade_size = self.config.get('min_trade_size', 20.0)
        self.min_cash_balance = self.config.get('min_cash_balance', 100.0)
        self.max_position_size = self.config.get('max_position_size', 0.02)
        self.enable_validation = self.config.get('enable_trade_validation', True)
        
        self.logger.info(f"TradeValidator initialized with min_trade_size={self.min_trade_size}, "
                        f"min_cash_balance={self.min_cash_balance}, max_position_size={self.max_position_size}")
    
    def validate_buy_order(self, ticker: int, shares: float, price: float, 
                          cash: float, current_positions: Dict[int, Dict[str, Any]], 
                          portfolio_value: float) -> Tuple[bool, str]:
        """
        Validate a BUY order before execution
        
        Args:
            ticker: Ticker index
            shares: Number of shares to buy
            price: Price per share
            cash: Available cash
            current_positions: Current positions dictionary
            portfolio_value: Total portfolio value
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not self.enable_validation:
            return True, ""
        
        try:
            # Check 1: Sufficient cash
            total_cost = shares * price
            if total_cost > cash:
                return False, f"Insufficient cash: need ${total_cost:.2f}, have ${cash:.2f}"
            
            # Check 2: Minimum trade size
            if total_cost < self.min_trade_size:
                return False, f"Trade too small: ${total_cost:.2f} < ${self.min_trade_size:.2f}"
            
            # Check 3: Minimum cash balance after trade
            remaining_cash = cash - total_cost
            if remaining_cash < self.min_cash_balance:
                return False, f"Insufficient cash after trade: ${remaining_cash:.2f} < ${self.min_cash_balance:.2f}"
            
            # Check 4: Position size limit
            current_position_value = current_positions.get(ticker, {'shares': 0.0})['shares'] * price
            new_position_value = current_position_value + total_cost
            position_weight = new_position_value / portfolio_value if portfolio_value > 0 else 0
            
            if position_weight > self.max_position_size:
                return False, f"Position too large: {position_weight:.2%} > {self.max_position_size:.2%}"
            
            # Check 5: Valid ticker
            if ticker < 0 or ticker >= 6:  # Assuming 6 tickers (AAPL, MSFT, NVDA, JPM, SPY, GLD)
                return False, f"Invalid ticker: {ticker}"
            
            # Check 6: Valid price
            if price <= 0:
                return False, f"Invalid price: ${price:.2f}"
            
            # Check 7: Valid shares
            if shares <= 0:
                return False, f"Invalid shares: {shares:.4f}"
            
            return True, ""
            
        except Exception as e:
            self.logger.error(f"Error validating buy order: {e}")
            return False, f"Validation error: {str(e)}"
    
    def validate_sell_order(self, ticker: int, shares: float, price: float,
                           current_positions: Dict[int, Dict[str, Any]]) -> Tuple[bool, str]:
        """
        Validate a SELL order before execution
        
        Args:
            ticker: Ticker index
            shares: Number of shares to sell
            price: Price per share
            current_positions: Current positions dictionary
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not self.enable_validation:
            return True, ""
        
        try:
            # Check 1: Position exists
            if ticker not in current_positions:
                return False, f"No position in ticker {ticker}"
            
            current_shares = current_positions[ticker]['shares']
            if current_shares <= 0:
                return False, f"No shares to sell in ticker {ticker}"
            
            # Check 2: Sufficient shares
            if shares > current_shares:
                return False, f"Insufficient shares: trying to sell {shares:.4f}, have {current_shares:.4f}"
            
            # Check 3: Minimum trade size
            total_proceeds = shares * price
            if total_proceeds < self.min_trade_size:
                return False, f"Trade too small: ${total_proceeds:.2f} < ${self.min_trade_size:.2f}"
            
            # Check 4: Valid ticker
            if ticker < 0 or ticker >= 6:  # Assuming 6 tickers
                return False, f"Invalid ticker: {ticker}"
            
            # Check 5: Valid price
            if price <= 0:
                return False, f"Invalid price: ${price:.2f}"
            
            # Check 6: Valid shares
            if shares <= 0:
                return False, f"Invalid shares: {shares:.4f}"
            
            return True, ""
            
        except Exception as e:
            self.logger.error(f"Error validating sell order: {e}")
            return False, f"Validation error: {str(e)}"
    
    def validate_trade_interval(self, current_step: int, last_trade_step: int, 
                               min_interval: int) -> Tuple[bool, str]:
        """
        Validate trade interval constraints
        
        Args:
            current_step: Current step number
            last_trade_step: Last step when a trade was executed
            min_interval: Minimum steps between trades
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not self.enable_validation:
            return True, ""
        
        try:
            steps_since_last_trade = current_step - last_trade_step
            if steps_since_last_trade < min_interval:
                return False, f"Trade interval violation: {steps_since_last_trade} < {min_interval} steps"
            
            return True, ""
            
        except Exception as e:
            self.logger.error(f"Error validating trade interval: {e}")
            return False, f"Validation error: {str(e)}"
    
    def validate_cooldown(self, ticker: int, current_step: int, 
                         cooldown_periods: Dict[int, int]) -> Tuple[bool, str]:
        """
        Validate cooldown constraints for a ticker
        
        Args:
            ticker: Ticker index
            current_step: Current step number
            cooldown_periods: Dictionary of ticker -> cooldown_end_step
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not self.enable_validation:
            return True, ""
        
        try:
            if ticker in cooldown_periods:
                cooldown_end = cooldown_periods[ticker]
                if current_step < cooldown_end:
                    remaining_steps = cooldown_end - current_step
                    return False, f"Ticker {ticker} in cooldown: {remaining_steps} steps remaining"
            
            return True, ""
            
        except Exception as e:
            self.logger.error(f"Error validating cooldown: {e}")
            return False, f"Validation error: {str(e)}"
    
    def validate_trade(self, action: int, ticker: int, shares: float, price: float,
                      cash: float, current_positions: Dict[int, Dict[str, Any]],
                      portfolio_value: float, current_step: int, last_trade_step: int,
                      min_interval: int, cooldown_periods: Dict[int, int]) -> Tuple[bool, str]:
        """
        Comprehensive trade validation
        
        Args:
            action: Action type (0=hold, 1=buy, 2=sell)
            ticker: Ticker index
            shares: Number of shares
            price: Price per share
            cash: Available cash
            current_positions: Current positions
            portfolio_value: Total portfolio value
            current_step: Current step
            last_trade_step: Last trade step
            min_interval: Minimum trade interval
            cooldown_periods: Cooldown periods
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not self.enable_validation:
            return True, ""
        
        try:
            # Skip validation for hold actions
            if action == 0:
                return True, ""
            
            # Validate trade interval
            is_valid, error_msg = self.validate_trade_interval(current_step, last_trade_step, min_interval)
            if not is_valid:
                return False, error_msg
            
            # Validate cooldown
            is_valid, error_msg = self.validate_cooldown(ticker, current_step, cooldown_periods)
            if not is_valid:
                return False, error_msg
            
            # Validate specific action
            if action == 1:  # Buy
                return self.validate_buy_order(ticker, shares, price, cash, current_positions, portfolio_value)
            elif action == 2:  # Sell
                return self.validate_sell_order(ticker, shares, price, current_positions)
            else:
                return False, f"Invalid action: {action}"
                
        except Exception as e:
            self.logger.error(f"Error in comprehensive trade validation: {e}")
            return False, f"Validation error: {str(e)}"
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """
        Get validation statistics
        
        Returns:
            Dictionary with validation statistics
        """
        return {
            'min_trade_size': self.min_trade_size,
            'min_cash_balance': self.min_cash_balance,
            'max_position_size': self.max_position_size,
            'validation_enabled': self.enable_validation
        }
