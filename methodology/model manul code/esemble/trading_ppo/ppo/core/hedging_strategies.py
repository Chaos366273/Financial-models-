"""
Advanced Hedging Strategies
Pairs trading, spread strategies, and dynamic hedging
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class HedgingConfig:
    """Hedging strategy configuration"""
    pairs_window: int = 60  # Days for pairs calculation
    z_score_threshold: float = 2.0  # Z-score for pairs trading
    correlation_threshold: float = 0.8  # Minimum correlation for pairs
    hedge_ratio: float = 0.5  # Default hedge ratio
    max_spread: float = 0.05  # Maximum spread for pairs
    rebalance_frequency: int = 5  # Days between rebalancing

class HedgingStrategies:
    """
    Advanced hedging strategies including pairs trading and dynamic hedging
    """
    
    def __init__(self, initial_capital: float = 100000.0):
        self.initial_capital = initial_capital
        self.config = HedgingConfig()
        
        # Pairs tracking
        self.pairs_data = {}
        self.active_pairs = {}
        self.pairs_history = []
        
        # Hedging positions
        self.hedge_positions = {}
        self.hedge_ratios = {}
        
        # Performance tracking
        self.hedge_returns = []
        self.spread_analysis = {}
        
        # Market data cache
        self.price_cache = {}
        self.correlation_matrix = None
        
        logger.info("HedgingStrategies initialized")
    
    def identify_pairs(self, symbols: List[str], price_data: Dict[str, pd.DataFrame]) -> List[Tuple[str, str]]:
        """Identify potential trading pairs based on correlation"""
        try:
            pairs = []
            
            # Calculate correlation matrix
            returns_data = {}
            for symbol in symbols:
                if symbol in price_data and len(price_data[symbol]) > self.config.pairs_window:
                    returns = price_data[symbol]['Close'].pct_change().dropna()
                    returns_data[symbol] = returns
            
            if len(returns_data) < 2:
                return pairs
            
            # Create correlation matrix
            returns_df = pd.DataFrame(returns_data)
            correlation_matrix = returns_df.corr()
            
            # Find highly correlated pairs
            for i, symbol1 in enumerate(returns_df.columns):
                for j, symbol2 in enumerate(returns_df.columns[i+1:], i+1):
                    correlation = correlation_matrix.iloc[i, j]
                    
                    if abs(correlation) > self.config.correlation_threshold:
                        pairs.append((symbol1, symbol2))
                        logger.info(f"Identified pair: {symbol1}-{symbol2} (correlation: {correlation:.3f})")
            
            return pairs
            
        except Exception as e:
            logger.error(f"Error identifying pairs: {e}")
            return []
    
    def calculate_spread(self, symbol1: str, symbol2: str, price_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Calculate spread between two symbols"""
        try:
            if symbol1 not in price_data or symbol2 not in price_data:
                return {}
            
            # Get price data
            prices1 = price_data[symbol1]['Close'].iloc[-self.config.pairs_window:]
            prices2 = price_data[symbol2]['Close'].iloc[-self.config.pairs_window:]
            
            # Calculate hedge ratio (beta)
            returns1 = prices1.pct_change().dropna()
            returns2 = prices2.pct_change().dropna()
            
            # Align data
            common_index = returns1.index.intersection(returns2.index)
            returns1 = returns1[common_index]
            returns2 = returns2[common_index]
            
            if len(returns1) < 10:
                return {}
            
            # Calculate beta (hedge ratio)
            covariance = np.cov(returns1, returns2)[0, 1]
            variance = np.var(returns2)
            
            if variance == 0:
                return {}
            
            beta = covariance / variance
            
            # Calculate spread
            spread = prices1 - (beta * prices2)
            
            # Calculate z-score
            spread_mean = spread.mean()
            spread_std = spread.std()
            
            if spread_std == 0:
                z_score = 0
            else:
                z_score = (spread.iloc[-1] - spread_mean) / spread_std
            
            return {
                'symbol1': symbol1,
                'symbol2': symbol2,
                'beta': beta,
                'spread': spread.iloc[-1],
                'spread_mean': spread_mean,
                'spread_std': spread_std,
                'z_score': z_score,
                'correlation': returns1.corr(returns2),
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error calculating spread: {e}")
            return {}
    
    def check_pairs_signal(self, symbol1: str, symbol2: str, price_data: Dict[str, pd.DataFrame]) -> Tuple[bool, str, float]:
        """Check for pairs trading signal"""
        try:
            spread_info = self.calculate_spread(symbol1, symbol2, price_data)
            
            if not spread_info:
                return False, "Insufficient data", 0.0
            
            z_score = spread_info['z_score']
            
            # Trading signals based on z-score
            if z_score > self.config.z_score_threshold:
                # Spread is wide - short symbol1, long symbol2
                return True, "SHORT_LONG", z_score
            elif z_score < -self.config.z_score_threshold:
                # Spread is narrow - long symbol1, short symbol2
                return True, "LONG_SHORT", z_score
            else:
                return False, "No signal", z_score
                
        except Exception as e:
            logger.error(f"Error checking pairs signal: {e}")
            return False, "Error", 0.0
    
    def calculate_position_sizes(self, symbol1: str, symbol2: str, spread_info: Dict[str, Any], 
                               available_capital: float) -> Tuple[float, float]:
        """Calculate position sizes for pairs trade"""
        try:
            beta = spread_info.get('beta', 1.0)
            z_score = abs(spread_info.get('z_score', 0))
            
            # Base position size
            base_size = available_capital * 0.1  # 10% of capital per pair
            
            # Adjust for z-score strength
            z_score_multiplier = min(2.0, z_score / self.config.z_score_threshold)
            adjusted_size = base_size * z_score_multiplier
            
            # Calculate individual position sizes
            # For beta-hedged positions, size2 = size1 * beta
            size1 = adjusted_size / 2  # Half for each side
            size2 = size1 * beta
            
            return size1, size2
            
        except Exception as e:
            logger.error(f"Error calculating position sizes: {e}")
            return 0.0, 0.0
    
    def execute_pairs_trade(self, symbol1: str, symbol2: str, signal: str, 
                           size1: float, size2: float, current_prices: Dict[str, float]) -> Dict[str, Any]:
        """Execute pairs trading strategy"""
        try:
            if signal not in ["SHORT_LONG", "LONG_SHORT"]:
                return {"success": False, "error": "Invalid signal"}
            
            price1 = current_prices.get(symbol1, 0)
            price2 = current_prices.get(symbol2, 0)
            
            if price1 == 0 or price2 == 0:
                return {"success": False, "error": "Missing price data"}
            
            # Calculate shares
            shares1 = size1 / price1
            shares2 = size2 / price2
            
            # Execute trades based on signal
            trades = []
            
            if signal == "SHORT_LONG":
                # Short symbol1, Long symbol2
                trades.extend([
                    {"symbol": symbol1, "action": "sell", "shares": shares1, "price": price1},
                    {"symbol": symbol2, "action": "buy", "shares": shares2, "price": price2}
                ])
            else:  # LONG_SHORT
                # Long symbol1, Short symbol2
                trades.extend([
                    {"symbol": symbol1, "action": "buy", "shares": shares1, "price": price1},
                    {"symbol": symbol2, "action": "sell", "shares": shares2, "price": price2}
                ])
            
            # Record trade
            trade_record = {
                "symbol1": symbol1,
                "symbol2": symbol2,
                "signal": signal,
                "trades": trades,
                "timestamp": datetime.now(),
                "size1": size1,
                "size2": size2
            }
            
            self.pairs_history.append(trade_record)
            
            # Update active pairs
            pair_key = f"{symbol1}_{symbol2}"
            self.active_pairs[pair_key] = {
                "symbol1": symbol1,
                "symbol2": symbol2,
                "signal": signal,
                "entry_time": datetime.now(),
                "size1": size1,
                "size2": size2
            }
            
            logger.info(f"Executed pairs trade: {signal} {symbol1}/{symbol2}")
            
            return {
                "success": True,
                "trades": trades,
                "pair_key": pair_key
            }
            
        except Exception as e:
            logger.error(f"Error executing pairs trade: {e}")
            return {"success": False, "error": str(e)}
    
    def close_pairs_position(self, pair_key: str, current_prices: Dict[str, float]) -> Dict[str, Any]:
        """Close an active pairs position"""
        try:
            if pair_key not in self.active_pairs:
                return {"success": False, "error": "Position not found"}
            
            position = self.active_pairs[pair_key]
            symbol1 = position["symbol1"]
            symbol2 = position["symbol2"]
            signal = position["signal"]
            
            price1 = current_prices.get(symbol1, 0)
            price2 = current_prices.get(symbol2, 0)
            
            if price1 == 0 or price2 == 0:
                return {"success": False, "error": "Missing price data"}
            
            # Calculate shares
            shares1 = position["size1"] / price1
            shares2 = position["size2"] / price2
            
            # Close trades (opposite of entry)
            trades = []
            
            if signal == "SHORT_LONG":
                # Buy back symbol1, Sell symbol2
                trades.extend([
                    {"symbol": symbol1, "action": "buy", "shares": shares1, "price": price1},
                    {"symbol": symbol2, "action": "sell", "shares": shares2, "price": price2}
                ])
            else:  # LONG_SHORT
                # Sell symbol1, Buy back symbol2
                trades.extend([
                    {"symbol": symbol1, "action": "sell", "shares": shares1, "price": price1},
                    {"symbol": symbol2, "action": "buy", "shares": shares2, "price": price2}
                ])
            
            # Calculate P&L
            entry_time = position["entry_time"]
            holding_period = (datetime.now() - entry_time).days
            
            # Remove from active pairs
            del self.active_pairs[pair_key]
            
            logger.info(f"Closed pairs position: {pair_key}")
            
            return {
                "success": True,
                "trades": trades,
                "holding_period": holding_period
            }
            
        except Exception as e:
            logger.error(f"Error closing pairs position: {e}")
            return {"success": False, "error": str(e)}
    
    def calculate_dynamic_hedge_ratio(self, portfolio_symbols: List[str], 
                                     benchmark_symbol: str, price_data: Dict[str, pd.DataFrame]) -> float:
        """Calculate dynamic hedge ratio for portfolio hedging"""
        try:
            if benchmark_symbol not in price_data:
                return 1.0
            
            # Calculate portfolio beta against benchmark
            portfolio_returns = []
            benchmark_returns = price_data[benchmark_symbol]['Close'].pct_change().dropna()
            
            for symbol in portfolio_symbols:
                if symbol in price_data:
                    symbol_returns = price_data[symbol]['Close'].pct_change().dropna()
                    portfolio_returns.append(symbol_returns)
            
            if not portfolio_returns:
                return 1.0
            
            # Calculate portfolio beta
            portfolio_returns_series = pd.concat(portfolio_returns, axis=1).mean(axis=1)
            
            # Align with benchmark
            common_index = portfolio_returns_series.index.intersection(benchmark_returns.index)
            portfolio_returns_aligned = portfolio_returns_series[common_index]
            benchmark_returns_aligned = benchmark_returns[common_index]
            
            if len(portfolio_returns_aligned) < 20:
                return 1.0
            
            # Calculate beta
            covariance = np.cov(portfolio_returns_aligned, benchmark_returns_aligned)[0, 1]
            variance = np.var(benchmark_returns_aligned)
            
            if variance == 0:
                return 1.0
            
            beta = covariance / variance
            
            return max(0.0, min(2.0, beta))  # Clamp between 0 and 2
            
        except Exception as e:
            logger.error(f"Error calculating dynamic hedge ratio: {e}")
            return 1.0
    
    def get_hedging_summary(self) -> Dict[str, Any]:
        """Get comprehensive hedging summary"""
        try:
            return {
                "active_pairs": len(self.active_pairs),
                "total_pairs_trades": len(self.pairs_history),
                "hedge_positions": len(self.hedge_positions),
                "config": {
                    "pairs_window": self.config.pairs_window,
                    "z_score_threshold": self.config.z_score_threshold,
                    "correlation_threshold": self.config.correlation_threshold,
                    "hedge_ratio": self.config.hedge_ratio
                },
                "active_pairs_list": list(self.active_pairs.keys())
            }
            
        except Exception as e:
            logger.error(f"Error getting hedging summary: {e}")
            return {}
    
    def update_price_cache(self, symbol: str, price: float):
        """Update price cache for real-time calculations"""
        if symbol not in self.price_cache:
            self.price_cache[symbol] = []
        
        self.price_cache[symbol].append({
            "price": price,
            "timestamp": datetime.now()
        })
        
        # Keep only recent prices
        cutoff_time = datetime.now() - timedelta(days=30)
        self.price_cache[symbol] = [
            p for p in self.price_cache[symbol] 
            if p["timestamp"] > cutoff_time
        ]
    
    def cleanup_old_data(self):
        """Clean up old data to prevent memory issues"""
        try:
            # Clean up old pairs history
            cutoff_time = datetime.now() - timedelta(days=90)
            self.pairs_history = [
                record for record in self.pairs_history
                if record["timestamp"] > cutoff_time
            ]
            
            # Clean up price cache
            cutoff_time = datetime.now() - timedelta(days=30)
            for symbol in list(self.price_cache.keys()):
                self.price_cache[symbol] = [
                    p for p in self.price_cache[symbol]
                    if p["timestamp"] > cutoff_time
                ]
                
                if not self.price_cache[symbol]:
                    del self.price_cache[symbol]
            
            logger.info("Cleaned up old hedging data")
            
        except Exception as e:
            logger.error(f"Error cleaning up old data: {e}")

# Backwards-compatible alias expected by other modules
class HedgingManager(HedgingStrategies):
    pass