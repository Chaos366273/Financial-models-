"""
Expert Dataset Module for PPO Trading System
Builds expert (observation, action) pairs for behavior cloning pretraining
"""

import numpy as np
import logging
from typing import Tuple, Optional
import os
import pickle
import gzip

logger = logging.getLogger(__name__)

def build_expert_dataset(
    feature_matrix: np.ndarray,
    price_matrix: np.ndarray,
    volume_matrix: np.ndarray,
    window_size: int,
    max_samples: int = 10000,
    mode: str = 'vwap'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build expert dataset from historical data
    
    Args:
        feature_matrix: (num_dates, num_tickers, num_features)
        price_matrix: (num_dates, num_tickers)
        volume_matrix: (num_dates, num_tickers)
        window_size: Observation window size
        max_samples: Maximum number of samples to generate
        mode: Expert strategy ('vwap', 'twap', 'pov')

    Returns:
        observations: (num_samples, window_size * num_tickers * num_features + 1 + num_tickers)
                     Market features (900) + Portfolio state (7: cash% + 6 position sizes) = 907
        actions: (num_samples, num_tickers)
    """
    try:
        logger.info(f" Building expert dataset: {mode} mode, max {max_samples} samples")
        
        num_dates, num_tickers, num_features = feature_matrix.shape
        
        # Calculate valid start indices (need window_size + 1 for next price)
        valid_starts = range(window_size, num_dates - 1)
        
        # Sample indices
        if len(valid_starts) > max_samples:
            indices = np.random.choice(valid_starts, max_samples, replace=False)
        else:
            indices = valid_starts
        
        observations = []
        actions = []
        
        for idx in indices:
            # Extract observation window (market features)
            obs_window = feature_matrix[idx-window_size:idx]
            obs_flat = obs_window.flatten()
            
            # PROPER FIX: Append portfolio state (7 dims: cash% + 6 position sizes)
            # For BC training, simulate starting portfolio (all cash, no positions)
            # This matches the expanded observation space in trading_env (900 → 907)
            portfolio_state = np.zeros(1 + num_tickers, dtype=np.float32)
            portfolio_state[0] = 1.0  # 100% cash initially
            # positions are all 0 (no shares held)
            
            # Concatenate market features + portfolio state
            obs_flat = np.concatenate([obs_flat, portfolio_state])
            
            # Generate expert action based on mode
            if mode == 'vwap':
                action = _generate_vwap_action(
                    price_matrix[idx-window_size:idx+1],
                    volume_matrix[idx-window_size:idx+1],
                    num_tickers
                )
            elif mode == 'twap':
                action = _generate_twap_action(
                    price_matrix[idx-window_size:idx+1],
                    num_tickers
                )
            elif mode == 'pov':
                action = _generate_pov_action(
                    price_matrix[idx-window_size:idx+1],
                    volume_matrix[idx-window_size:idx+1],
                    num_tickers
                )
            else:
                # Random action as fallback
                action = np.random.uniform(-0.1, 0.1, num_tickers)
            
            observations.append(obs_flat)
            actions.append(action)
        
        observations = np.array(observations)
        actions = np.array(actions)
        
        logger.info(f" Expert dataset built: {len(observations)} samples")
        logger.info(f" Observation shape: {observations.shape}")
        logger.info(f" Action shape: {actions.shape}")
        
        return observations, actions

    except Exception as e:
        logger.error(f" Error building expert dataset: {e}")
        # Return empty arrays as fallback
        return np.array([]), np.array([])

def _generate_vwap_action(
    price_window: np.ndarray,
    volume_window: np.ndarray,
    num_tickers: int
) -> np.ndarray:
    """Generate VWAP-based expert actions"""
    try:
        actions = np.zeros(num_tickers)
        
        for ticker in range(num_tickers):
            prices = price_window[:, ticker]
            volumes = volume_window[:, ticker]
            
            if len(prices) < 2 or np.all(volumes == 0):
                continue
                
            # Calculate VWAP
            vwap = np.sum(prices * volumes) / np.sum(volumes)
            current_price = prices[-1]
            
            # Generate action based on price vs VWAP
            if current_price > vwap * 1.01:  # 1% above VWAP
                actions[ticker] = -0.05  # Sell
            elif current_price < vwap * 0.99:  # 1% below VWAP
                actions[ticker] = 0.05   # Buy
            else:
                actions[ticker] = 0.0    # Hold
                
        return actions
        
    except Exception as e:
        logger.warning(f" VWAP action generation failed: {e}")
        return np.zeros(num_tickers)

def _generate_twap_action(
    price_window: np.ndarray,
    num_tickers: int
) -> np.ndarray:
    """Generate TWAP-based expert actions"""
    try:
        actions = np.zeros(num_tickers)
        
        for ticker in range(num_tickers):
            prices = price_window[:, ticker]
            
            if len(prices) < 2:
                continue
                
            # Calculate TWAP (simple average)
            twap = np.mean(prices)
            current_price = prices[-1]
            
            # Generate action based on price vs TWAP
            if current_price > twap * 1.005:  # 0.5% above TWAP
                actions[ticker] = -0.03  # Sell
            elif current_price < twap * 0.995:  # 0.5% below TWAP
                actions[ticker] = 0.03   # Buy
            else:
                actions[ticker] = 0.0    # Hold
                
        return actions
        
    except Exception as e:
        logger.warning(f" TWAP action generation failed: {e}")
        return np.zeros(num_tickers)

def _generate_pov_action(
    price_window: np.ndarray,
    volume_window: np.ndarray,
    num_tickers: int
) -> np.ndarray:
    """Generate POV-based expert actions"""
    try:
        actions = np.zeros(num_tickers)
        
        for ticker in range(num_tickers):
            prices = price_window[:, ticker]
            volumes = volume_window[:, ticker]
            
            if len(prices) < 3 or np.all(volumes == 0):
                continue
                
            # Calculate volume-weighted momentum
            price_changes = np.diff(prices)
            volume_weights = volumes[1:] / np.sum(volumes[1:])
            momentum = np.sum(price_changes * volume_weights)
            
            # Generate action based on momentum
            if momentum > 0.001:  # Positive momentum
                actions[ticker] = 0.04   # Buy
            elif momentum < -0.001:  # Negative momentum
                actions[ticker] = -0.04  # Sell
            else:
                actions[ticker] = 0.0    # Hold
                
        return actions
        
    except Exception as e:
        logger.warning(f" POV action generation failed: {e}")
        return np.zeros(num_tickers)

def save_expert_dataset(observations: np.ndarray, actions: np.ndarray, 
                       filepath: str = 'ppo/saved_data/expert_dataset.pkl'):
    """Save expert dataset to file"""
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        data = {
            'observations': observations,
            'actions': actions,
            'timestamp': np.datetime64('now')
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info(f" Expert dataset saved to: {filepath}")
        logger.info(f" Saved {len(observations)} samples")
        
    except Exception as e:
        logger.error(f" Error saving expert dataset: {e}")

def load_expert_dataset(filepath: str = 'ppo/saved_data/expert_dataset.pkl') -> Tuple[np.ndarray, np.ndarray]:
    """Load expert dataset from file"""
    try:
        if not os.path.exists(filepath):
            logger.warning(f" Expert dataset file not found: {filepath}")
            return np.array([]), np.array([])
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        observations = data['observations']
        actions = data['actions']
        
        logger.info(f" Expert dataset loaded from: {filepath}")
        logger.info(f" Loaded {len(observations)} samples")
        
        return observations, actions
        
    except Exception as e:
        logger.error(f" Error loading expert dataset: {e}")
        return np.array([]), np.array([])

def validate_expert_dataset(observations: np.ndarray, actions: np.ndarray) -> bool:
    """Validate expert dataset quality"""
    try:
        if len(observations) == 0 or len(actions) == 0:
            logger.warning(" Empty expert dataset")
            return False
        
        # Check shapes
        if observations.shape[0] != actions.shape[0]:
            logger.error(" Mismatched observation and action counts")
            return False
        
        # Check for NaN values
        if np.isnan(observations).any():
            logger.error(" NaN values found in observations")
            return False
        
        if np.isnan(actions).any():
            logger.error(" NaN values found in actions")
            return False
        
        # Check action ranges
        action_min = np.min(actions)
        action_max = np.max(actions)
        if action_min < -1.0 or action_max > 1.0:
            logger.warning(f" Actions outside expected range: [{action_min:.3f}, {action_max:.3f}]")
        
        logger.info(" Expert dataset validation passed")
        return True
        
    except Exception as e:
        logger.error(f" Error validating expert dataset: {e}")
        return False


