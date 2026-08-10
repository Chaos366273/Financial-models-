"""
Regime Manager - Auto-Promotion System
Handles automatic switching between regime models based on performance
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, cast, Any
from datetime import datetime, timedelta
import os
import json

class RegimeManager:
    """
    Market Regime Manager for PPO Trading Agent
    
    Handles:
    - Market regime detection (bull, bear, sideways, volatile)
    - Regime-specific model training and switching
    - Performance tracking across regimes
    - Auto-promotion of best performing models
    """
    
    def __init__(self, config: Dict):
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        # Regime configuration
        self.enable_regime_training = config.get('enable_regime_training', True)
        self.regime_auto_switch = config.get('regime_switch', True)
        self.regime_confidence_threshold = config.get('regime_confidence_threshold', 0.8)
        self.regime_switch_notification = config.get('regime_switch_notification', True)
        self.regime_training_epochs = config.get('regime_training_epochs', 1000)
        self.regime_validation_period = config.get('regime_validation_period', 500)
        self.enable_market_regime_detection = config.get('enable_market_regime_detection', True)
        
        # Regime state
        self.active_regime = 'current_training'
        self.regime_models = {}  # regime_name -> model_path
        self.performance_history = {}  # regime_name -> performance_metrics
        self.regime_detection_history = []
        
        # Regime detection parameters
        self.volatility_threshold = 0.02  # 2% daily volatility
        self.trend_threshold = 0.01  # 1% daily trend
        self.correlation_threshold = 0.7  # 70% correlation for regime change
        
        # Performance tracking
        self.current_performance = {
            'total_return': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0,
            'total_trades': 0
        }
        
        self.logger.info("RegimeManager initialized with regime training enabled")
    
    def detect_market_regime(self, price_data: pd.DataFrame, volume_data: pd.DataFrame) -> str:
        """
        Detect current market regime based on price and volume patterns
        
        Returns:
            str: 'bull', 'bear', 'sideways', 'volatile', 'unknown'
        """
        try:
            if not self.enable_market_regime_detection:
                return 'unknown'
            
            # CRITICAL FIX: Handle different data structures
            # If price_data has ticker columns (from environment), use first ticker as proxy
            if len(price_data.columns) > 0 and not 'close' in price_data.columns:
                # Use first ticker as market proxy (typically SPY or major index)
                first_ticker = price_data.columns[0]
                returns = price_data[first_ticker].pct_change().dropna()
                
                # Use first ticker volume as proxy
                if len(volume_data.columns) > 0:
                    first_volume_col = volume_data.columns[0]
                    volume_series = volume_data[first_volume_col]
                else:
                    volume_series = pd.Series(np.ones(len(price_data)) * 1000000)
            else:
                # Original logic for 'close' column
                returns = price_data['close'].pct_change().dropna()
                volume_series = volume_data['volume'] if 'volume' in volume_data.columns else pd.Series(np.ones(len(price_data)) * 1000000)
            
            # Calculate key metrics
            returns_series: pd.Series = pd.Series(returns)
            volume_series_pd: pd.Series = pd.Series(volume_series)
            returns_rolling = cast(Any, returns_series).rolling(window=20)
            volume_rolling = cast(Any, volume_series_pd).rolling(window=20).mean()
            volatility = returns_rolling.std().iloc[-1] if len(returns_series) >= 20 else 0.02
            trend = returns_rolling.mean().iloc[-1] if len(returns_series) >= 20 else 0.0
            volume_trend = volume_rolling.pct_change().iloc[-1] if len(volume_series_pd) >= 20 else 0.0
            
            # Regime detection logic
            if volatility > self.volatility_threshold:
                if abs(trend) > self.trend_threshold:
                    regime = 'volatile_trend'
                else:
                    regime = 'volatile_sideways'
            elif trend > self.trend_threshold:
                regime = 'bull'
            elif trend < -self.trend_threshold:
                regime = 'bear'
            else:
                regime = 'sideways'
            
            # Store detection history
            self.regime_detection_history.append({
                'timestamp': datetime.now(),
                'regime': regime,
                'volatility': volatility,
                'trend': trend,
                'volume_trend': volume_trend
            })
            
            self.logger.info(f"Market regime detected: {regime} (vol: {volatility:.4f}, trend: {trend:.4f})")
            return regime
            
        except Exception as e:
            self.logger.error(f"Regime detection failed: {e}")
            return 'unknown'
    
    def register_regime_model(self, regime_name: str, model_path: str):
        """Register a model for a specific regime"""
        try:
            self.regime_models[regime_name] = model_path
            self.logger.info(f"Registered model for regime '{regime_name}': {model_path}")
        except Exception as e:
            self.logger.error(f"Failed to register regime model: {e}")
    
    def update_regime_performance(self, regime_name: str, performance_metrics: Dict):
        """Update performance metrics for a regime"""
        try:
            if regime_name not in self.performance_history:
                self.performance_history[regime_name] = []
            
            self.performance_history[regime_name].append({
                'timestamp': datetime.now(),
                'metrics': performance_metrics.copy()
            })
            
            # Keep only recent history (last 10 entries)
            if len(self.performance_history[regime_name]) > 10:
                self.performance_history[regime_name] = self.performance_history[regime_name][-10:]
            
            self.logger.info(f"Updated performance for regime '{regime_name}': {performance_metrics}")
            
        except Exception as e:
            self.logger.error(f"Failed to update regime performance: {e}")
    
    def auto_promote_regime(self) -> Tuple[bool, str]:
        """
        Auto-promote best performing regime model
        
        Returns:
            Tuple[bool, str]: (promotion_occurred, new_active_regime)
        """
        try:
            if not self.regime_auto_switch or len(self.performance_history) < 2:
                return False, self.active_regime
            
            # Calculate average performance for each regime
            regime_performance = {}
            for regime, history in self.performance_history.items():
                if len(history) > 0:
                    avg_sharpe = np.mean([h['metrics'].get('sharpe_ratio', 0) for h in history])
                    avg_return = np.mean([h['metrics'].get('total_return', 0) for h in history])
                    regime_performance[regime] = {
                        'avg_sharpe': avg_sharpe,
                        'avg_return': avg_return,
                        'confidence': len(history) / 10.0  # More history = higher confidence
                    }
            
            # Find best performing regime
            best_regime = None
            best_score = -np.inf
            
            for regime, perf in regime_performance.items():
                # Combined score: 70% Sharpe + 30% Return
                score = 0.7 * perf['avg_sharpe'] + 0.3 * perf['avg_return']
                
                # Apply confidence threshold
                if perf['confidence'] >= self.regime_confidence_threshold:
                    if score > best_score:
                        best_score = score
                        best_regime = regime
            
            # Switch if better regime found
            if best_regime and best_regime != self.active_regime:
                old_regime = self.active_regime
                self.active_regime = best_regime
                
                if self.regime_switch_notification:
                    self.logger.info(f" REGIME SWITCH: {old_regime}  {best_regime}")
                    self.logger.info(f"   Score: {best_score:.4f} (Sharpe: {regime_performance[best_regime]['avg_sharpe']:.3f}, Return: {regime_performance[best_regime]['avg_return']:.3f})")
                
                return True, best_regime
            
            return False, self.active_regime
            
        except Exception as e:
            self.logger.error(f"Auto-promotion failed: {e}")
            return False, self.active_regime
    
    def get_regime_performance_summary(self) -> Dict:
        """Get summary of regime performance"""
        try:
            summary = {
                'active_regime': self.active_regime,
                'total_regimes': len(self.regime_models),
                'regime_performance': {}
            }
            
            for regime, history in self.performance_history.items():
                if len(history) > 0:
                    avg_sharpe = np.mean([h['metrics'].get('sharpe_ratio', 0) for h in history])
                    avg_return = np.mean([h['metrics'].get('total_return', 0) for h in history])
                    summary['regime_performance'][regime] = {
                        'avg_sharpe': avg_sharpe,
                        'avg_return': avg_return,
                        'history_count': len(history)
                    }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Failed to get regime summary: {e}")
            return {'active_regime': self.active_regime, 'total_regimes': 0}
    
    def get_active_model_path(self) -> Optional[str]:
        """Get path to active regime model"""
        return self.regime_models.get(self.active_regime)
    
    def should_switch_regime(self, current_performance: Dict) -> bool:
        """Determine if regime switch is needed based on current performance"""
        try:
            if not self.regime_auto_switch:
                return False
            
            # Check if current performance is poor
            current_sharpe = current_performance.get('sharpe_ratio', 0)
            current_return = current_performance.get('total_return', 0)
            
            # Switch if performance is significantly poor
            if current_sharpe < 0.5 or current_return < -0.1:
                self.logger.warning(f"Poor performance detected - considering regime switch (Sharpe: {current_sharpe:.3f}, Return: {current_return:.3f})")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Regime switch check failed: {e}")
            return False
    
    def save_regime_state(self, filepath: str):
        """Save regime manager state to file"""
        try:
            state = {
                'active_regime': self.active_regime,
                'regime_models': self.regime_models,
                'performance_history': self.performance_history,
                'regime_detection_history': self.regime_detection_history
            }
            
            with open(filepath, 'w') as f:
                json.dump(state, f, default=str, indent=2)
            
            self.logger.info(f"Regime state saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to save regime state: {e}")
    
    def load_regime_state(self, filepath: str):
        """Load regime manager state from file"""
        try:
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    state = json.load(f)
                
                self.active_regime = state.get('active_regime', 'current_training')
                self.regime_models = state.get('regime_models', {})
                self.performance_history = state.get('performance_history', {})
                self.regime_detection_history = state.get('regime_detection_history', [])
                
                self.logger.info(f"Regime state loaded from {filepath}")
            else:
                self.logger.warning(f"Regime state file not found: {filepath}")
                
        except Exception as e:
            self.logger.error(f"Failed to load regime state: {e}")
