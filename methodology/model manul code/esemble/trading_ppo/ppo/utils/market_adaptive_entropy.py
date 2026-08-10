#!/usr/bin/env python3
"""Market-Adaptive Entropy System"""
import numpy as np
import logging
from typing import Tuple, Dict, Any
from ppo.utils.dynamic_config import get_dynamic_config
logger = logging.getLogger(__name__)
class MarketAdaptiveEntropy:
    def __init__(self, base_entropy: float = 0.02):
        self.base_entropy = base_entropy
        self.config = get_dynamic_config()
        self.high_volatility_threshold = 0.03
        self.low_volatility_threshold = 0.01
        self.strong_trend_threshold = 0.7
        self.weak_trend_threshold = 0.3
        self.volatility_multiplier_range = (0.8, 1.5)
        self.regime_multiplier_range = (0.9, 1.3)
        self.trend_multiplier_range = (0.85, 1.2)
        logger.info(f"[MARKET ADAPTIVE] Initialized with base_entropy={base_entropy:.4f}")
    def calculate(self, price_matrix: np.ndarray, volume_matrix: np.ndarray) -> Tuple[float, Dict[str, Any]]:
        try:
            volatility = self._calculate_volatility(price_matrix)
            regime = self._detect_regime(price_matrix, volume_matrix)
            trend_strength = self._calculate_trend_strength(price_matrix)
            uncertainty = self._calculate_uncertainty(price_matrix, volume_matrix)
            vol_multiplier = self._get_volatility_multiplier(volatility)
            regime_multiplier = self._get_regime_multiplier(regime)
            trend_multiplier = self._get_trend_multiplier(trend_strength)
            uncertainty_multiplier = self._get_uncertainty_multiplier(uncertainty)
            total_multiplier = (vol_multiplier * regime_multiplier * trend_multiplier * uncertainty_multiplier) ** 0.25
            adaptive_entropy = self.base_entropy * total_multiplier
            adaptive_entropy = np.clip(adaptive_entropy, self.base_entropy * 0.5, getattr(self.config, 'adaptive_ent_max', 0.04))
            market_info = {'volatility': volatility, 'regime': regime, 'trend_strength': trend_strength, 'uncertainty': uncertainty, 'volatility_multiplier': vol_multiplier, 'regime_multiplier': regime_multiplier, 'trend_multiplier': trend_multiplier, 'uncertainty_multiplier': uncertainty_multiplier, 'total_multiplier': total_multiplier, 'base_entropy': self.base_entropy, 'adaptive_entropy': adaptive_entropy}
            return float(adaptive_entropy), market_info
        except Exception as e:
            logger.warning(f"[MARKET ADAPTIVE] Error: {e}, using base entropy")
            return self.base_entropy, {'volatility': 0.0, 'regime': 'unknown', 'trend_strength': 0.0, 'uncertainty': 0.0, 'total_multiplier': 1.0, 'adaptive_entropy': self.base_entropy}
    def _calculate_volatility(self, price_matrix: np.ndarray) -> float:
        if price_matrix.shape[0] < 2:
            return 0.0
        returns = np.diff(price_matrix, axis=0) / price_matrix[:-1]
        volatility = np.std(returns, axis=0)
        return float(np.mean(volatility))
    def _detect_regime(self, price_matrix: np.ndarray, volume_matrix: np.ndarray) -> str:
        if price_matrix.shape[0] < 20:
            return 'stable'
        high_low = np.max(price_matrix[-20:], axis=0) - np.min(price_matrix[-20:], axis=0)
        atr = np.mean(high_low / price_matrix[-20:], axis=0)
        avg_atr = np.mean(atr)
        returns = np.diff(price_matrix[-20:], axis=0) / price_matrix[-20:-1]
        if returns.shape[0] > 1:
            trend_strength = np.abs(np.corrcoef(returns[:-1].flatten(), returns[1:].flatten())[0, 1])
            if np.isnan(trend_strength):
                trend_strength = 0.0
        else:
            trend_strength = 0.0
        if avg_atr > 0.05:
            return 'volatile'
        elif trend_strength > 0.5:
            return 'trending'
        elif trend_strength < 0.2:
            return 'mean_reversion'
        else:
            return 'stable'
    def _calculate_trend_strength(self, price_matrix: np.ndarray) -> float:
        if price_matrix.shape[0] < 20:
            return 0.5
        returns = np.diff(price_matrix[-20:], axis=0) / price_matrix[-20:-1]
        if returns.shape[0] > 1:
            autocorr = np.corrcoef(returns[:-1].flatten(), returns[1:].flatten())[0, 1]
            if np.isnan(autocorr):
                autocorr = 0.0
            trend_strength = abs(autocorr)
        else:
            trend_strength = 0.5
        return float(np.clip(trend_strength, 0.0, 1.0))
    def _calculate_uncertainty(self, price_matrix: np.ndarray, volume_matrix: np.ndarray) -> float:
        if price_matrix.shape[0] < 10:
            return 0.5
        price_vol = np.std(np.diff(price_matrix[-10:], axis=0) / price_matrix[-10:-1], axis=0)
        price_uncertainty = np.mean(price_vol) / 0.02
        if volume_matrix.shape[0] >= 10:
            volume_cv = np.std(volume_matrix[-10:], axis=0) / (np.mean(volume_matrix[-10:], axis=0) + 1e-8)
            volume_uncertainty = np.mean(volume_cv)
        else:
            volume_uncertainty = 0.5
        price_change = np.abs(np.diff(price_matrix[-10:], axis=0))
        volume_change = np.abs(np.diff(volume_matrix[-10:], axis=0))
        if np.sum(volume_change) > 0:
            divergence = 1.0 - np.corrcoef(price_change.flatten(), volume_change.flatten())[0, 1]
            if np.isnan(divergence):
                divergence = 0.5
        else:
            divergence = 0.5
        uncertainty = (price_uncertainty * 0.4 + volume_uncertainty * 0.3 + abs(divergence) * 0.3)
        return float(np.clip(uncertainty, 0.0, 1.0))
    def _get_volatility_multiplier(self, volatility: float) -> float:
        if volatility > self.high_volatility_threshold:
            multiplier = 1.2 + (volatility - self.high_volatility_threshold) * 10
        elif volatility < self.low_volatility_threshold:
            multiplier = 0.8 + (volatility / self.low_volatility_threshold) * 0.1
        else:
            multiplier = 1.0
        return float(np.clip(multiplier, self.volatility_multiplier_range[0], self.volatility_multiplier_range[1]))
    def _get_regime_multiplier(self, regime: str) -> float:
        regime_multipliers = {'volatile': 1.3, 'trending': 0.95, 'mean_reversion': 1.1, 'stable': 0.9, 'unknown': 1.0}
        multiplier = regime_multipliers.get(regime, 1.0)
        return float(np.clip(multiplier, self.regime_multiplier_range[0], self.regime_multiplier_range[1]))
    def _get_trend_multiplier(self, trend_strength: float) -> float:
        if trend_strength > self.strong_trend_threshold:
            multiplier = 0.85 + (trend_strength - self.strong_trend_threshold) * 0.33
        elif trend_strength < self.weak_trend_threshold:
            multiplier = 1.1 + (1.0 - trend_strength / self.weak_trend_threshold) * 0.1
        else:
            multiplier = 1.0
        return float(np.clip(multiplier, self.trend_multiplier_range[0], self.trend_multiplier_range[1]))
    def _get_uncertainty_multiplier(self, uncertainty: float) -> float:
        multiplier = 0.9 + uncertainty * 0.3
        return float(np.clip(multiplier, 0.85, 1.2))
    def log_market_conditions(self, market_info: Dict[str, Any]) -> None:
        logger.info(f"[MARKET ADAPTIVE] Volatility={market_info['volatility']:.2%}, Regime={market_info['regime']}, Trend={market_info['trend_strength']:.2f}, Uncertainty={market_info['uncertainty']:.2f}, Entropy={market_info['adaptive_entropy']:.4f} (multiplier={market_info['total_multiplier']:.2f}x)")