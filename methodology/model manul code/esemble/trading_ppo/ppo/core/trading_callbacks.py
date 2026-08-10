"""
Advanced Trading Callbacks
Based on TG4.1 TradingCallback implementation
"""

import os
import time
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from collections import deque

# Stable-baselines3 imports
try:
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.logger import configure
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    logging.warning("  Stable-baselines3 not available for callbacks")

logger = logging.getLogger(__name__)

class TradingCallback(BaseCallback):
    """Advanced trading callback like TG4.1"""
    
    def __init__(self, verbose: int = 1):
        super().__init__(verbose)
        
        # Performance tracking
        self.training_logs = []
        self.episode_rewards = []
        self.episode_lengths = []
        self.trading_metrics = []
        
        # Real-time monitoring
        self.current_episode = 0
        self.episode_start_time = None
        self.last_log_time = time.time()
        self.log_interval = 1000  # Log every 1000 steps
        
        # Performance thresholds
        self.performance_threshold = -0.05
        self.improvement_threshold = 0.01
        self.patience = 10  # Episodes without improvement
        
        # Early stopping
        self.best_performance = -np.inf
        self.episodes_without_improvement = 0
        self.should_stop = False
        
        logger.info(" TradingCallback initialized")
    
    def _on_training_start(self) -> None:
        """Called at the start of training"""
        logger.info(" Training started - TradingCallback active")
        self.episode_start_time = time.time()
    
    def _on_step(self) -> bool:
        """Called after each step during training"""
        try:
            # Track episode progress
            if self.locals.get('dones', [False])[0]:
                self._on_episode_end()
            
            # Log periodically
            if self.num_timesteps % self.log_interval == 0:
                self._log_training_progress()
            
            # Check for early stopping
            if self.should_stop:
                logger.info(" Early stopping triggered")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f" TradingCallback step error: {e}")
            return True
    
    def _on_episode_end(self) -> None:
        """Called at the end of each episode"""
        try:
            # Get episode info
            episode_reward = self.locals.get('rewards', [0.0])[0]
            episode_length = self.locals.get('ep_infos', [{}])[0].get('l', 0)
            
            # Calculate episode metrics
            episode_time = time.time() - self.episode_start_time if self.episode_start_time else 0
            
            episode_metrics = {
                'episode': self.current_episode,
                'reward': episode_reward,
                'length': episode_length,
                'time': episode_time,
                'timesteps': self.num_timesteps,
                'timestamp': datetime.now().isoformat()
            }
            
            # Store metrics
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            self.training_logs.append(episode_metrics)
            
            # Update performance tracking
            self._update_performance_tracking(episode_reward)
            
            # Log episode summary
            if self.verbose > 0:
                logger.info(f" Episode {self.current_episode}: Reward={episode_reward:.4f}, Length={episode_length}")
            
            # Reset for next episode
            self.current_episode += 1
            self.episode_start_time = time.time()
            
        except Exception as e:
            logger.error(f" TradingCallback episode end error: {e}")
    
    def _update_performance_tracking(self, episode_reward: float):
        """Update performance tracking and check for early stopping"""
        # Calculate moving average
        if len(self.episode_rewards) >= 10:
            recent_rewards = self.episode_rewards[-10:]
            avg_reward = np.mean(recent_rewards)
            
            # Check for improvement
            if avg_reward > self.best_performance + self.improvement_threshold:
                self.best_performance = avg_reward
                self.episodes_without_improvement = 0
                logger.info(f" New best performance: {avg_reward:.4f}")
            else:
                self.episodes_without_improvement += 1
            
            # Check for early stopping
            if self.episodes_without_improvement >= self.patience:
                logger.warning(f"  No improvement for {self.patience} episodes - considering early stop")
                if avg_reward < self.performance_threshold:
                    self.should_stop = True
    
    def _log_training_progress(self):
        """Log detailed training progress"""
        try:
            current_time = time.time()
            time_since_last_log = current_time - self.last_log_time
            
            # Calculate metrics
            if len(self.episode_rewards) > 0:
                recent_rewards = self.episode_rewards[-10:]
                avg_reward = np.mean(recent_rewards)
                std_reward = np.std(recent_rewards)
                min_reward = np.min(recent_rewards)
                max_reward = np.max(recent_rewards)
            else:
                avg_reward = std_reward = min_reward = max_reward = 0.0
            
            # Log progress
            logger.info(f" Training Progress:")
            logger.info(f"   - Timesteps: {self.num_timesteps}")
            logger.info(f"   - Episodes: {self.current_episode}")
            logger.info(f"   - Avg Reward (last 10): {avg_reward:.4f}")
            logger.info(f"   - Reward Std: {std_reward:.4f}")
            logger.info(f"   - Reward Range: [{min_reward:.4f}, {max_reward:.4f}]")
            logger.info(f"   - Time since last log: {time_since_last_log:.1f}s")
            
            self.last_log_time = current_time
            
        except Exception as e:
            logger.error(f" Progress logging error: {e}")
    
    def get_training_summary(self) -> Dict:
        """Get comprehensive training summary"""
        if not self.training_logs:
            return {'status': 'No training data'}
        
        try:
            rewards = [log['reward'] for log in self.training_logs]
            lengths = [log['length'] for log in self.training_logs]
            
            summary = {
                'total_episodes': len(self.training_logs),
                'total_timesteps': self.num_timesteps,
                'avg_reward': np.mean(rewards),
                'std_reward': np.std(rewards),
                'min_reward': np.min(rewards),
                'max_reward': np.max(rewards),
                'avg_episode_length': np.mean(lengths),
                'best_performance': self.best_performance,
                'episodes_without_improvement': self.episodes_without_improvement,
                'training_time': time.time() - self.episode_start_time if self.episode_start_time else 0,
                'last_updated': datetime.now().isoformat()
            }
            
            return summary
            
        except Exception as e:
            logger.error(f" Training summary error: {e}")
            return {'status': 'Error generating summary'}
    
    def save_training_logs(self, filepath: str = 'ppo/saved_models/training_logs.pkl'):
        """Save training logs"""
        try:
            import pickle
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            logs_data = {
                'training_logs': self.training_logs,
                'episode_rewards': self.episode_rewards,
                'episode_lengths': self.episode_lengths,
                'trading_metrics': self.trading_metrics,
                'summary': self.get_training_summary(),
                'timestamp': datetime.now().isoformat()
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(logs_data, f)
            
            logger.info(f" Training logs saved to {filepath}")
            
        except Exception as e:
            logger.error(f" Failed to save training logs: {e}")
    
    def load_training_logs(self, filepath: str = 'ppo/saved_models/training_logs.pkl') -> bool:
        """Load training logs"""
        try:
            import pickle
            
            if not os.path.exists(filepath):
                logger.warning(f"  No training logs found at {filepath}")
                return False
            
            with open(filepath, 'rb') as f:
                logs_data = pickle.load(f)
            
            self.training_logs = logs_data.get('training_logs', [])
            self.episode_rewards = logs_data.get('episode_rewards', [])
            self.episode_lengths = logs_data.get('episode_lengths', [])
            self.trading_metrics = logs_data.get('trading_metrics', [])
            
            logger.info(f" Training logs loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f" Failed to load training logs: {e}")
            return False

class PerformanceMonitor:
    """Real-time performance monitoring like TG4.1"""
    
    def __init__(self):
        self.metrics_history = deque(maxlen=1000)
        self.performance_alerts = []
        self.monitoring_active = False
        
        # Performance thresholds
        self.reward_threshold = -0.1
        self.volatility_threshold = 0.5
        self.drawdown_threshold = -0.2
        
        logger.info(" PerformanceMonitor initialized")
    
    def start_monitoring(self):
        """Start real-time performance monitoring"""
        self.monitoring_active = True
        logger.info(" Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop real-time performance monitoring"""
        self.monitoring_active = False
        logger.info(" Performance monitoring stopped")
    
    def update_metrics(self, metrics: Dict):
        """Update performance metrics"""
        if not self.monitoring_active:
            return
        
        try:
            # Add timestamp
            metrics['timestamp'] = datetime.now().isoformat()
            
            # Store metrics
            self.metrics_history.append(metrics)
            
            # Check for alerts
            self._check_performance_alerts(metrics)
            
        except Exception as e:
            logger.error(f" Performance metrics update error: {e}")
    
    def _check_performance_alerts(self, metrics: Dict):
        """Check for performance alerts"""
        try:
            reward = metrics.get('reward', 0.0)
            volatility = metrics.get('volatility', 0.0)
            drawdown = metrics.get('drawdown', 0.0)
            
            alerts = []
            
            # Check reward threshold
            if reward < self.reward_threshold:
                alerts.append(f"  Low reward: {reward:.4f}")
            
            # Check volatility threshold
            if volatility > self.volatility_threshold:
                alerts.append(f"  High volatility: {volatility:.4f}")
            
            # Check drawdown threshold
            if drawdown < self.drawdown_threshold:
                alerts.append(f"  High drawdown: {drawdown:.4f}")
            
            # Log alerts
            for alert in alerts:
                logger.warning(alert)
                self.performance_alerts.append({
                    'alert': alert,
                    'timestamp': datetime.now().isoformat(),
                    'metrics': metrics
                })
                
        except Exception as e:
            logger.error(f" Performance alert check error: {e}")
    
    def get_performance_summary(self) -> Dict:
        """Get performance monitoring summary"""
        if not self.metrics_history:
            return {'status': 'No performance data'}
        
        try:
            recent_metrics = list(self.metrics_history)[-100:]  # Last 100 metrics
            
            rewards = [m.get('reward', 0.0) for m in recent_metrics]
            volatilities = [m.get('volatility', 0.0) for m in recent_metrics]
            drawdowns = [m.get('drawdown', 0.0) for m in recent_metrics]
            
            summary = {
                'total_metrics': len(self.metrics_history),
                'recent_metrics': len(recent_metrics),
                'avg_reward': np.mean(rewards),
                'avg_volatility': np.mean(volatilities),
                'avg_drawdown': np.mean(drawdowns),
                'alerts_count': len(self.performance_alerts),
                'monitoring_active': self.monitoring_active,
                'last_updated': datetime.now().isoformat()
            }
            
            return summary
            
        except Exception as e:
            logger.error(f" Performance summary error: {e}")
            return {'status': 'Error generating summary'}
