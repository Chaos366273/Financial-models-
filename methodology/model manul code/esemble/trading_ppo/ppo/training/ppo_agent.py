"""
PPO Agent implementation for trading
"""

import os
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import logging

class TradingCallback(BaseCallback):
    """Custom callback for tracking trading performance during training"""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.training_logs = {
            'portfolio_values': [],
            'trades_executed': [],
            'rewards': [],
            'actions': []
        }
    
    def _on_step(self) -> bool:
        """Log training metrics at each step"""
        if self.locals.get('infos'):
            for info in self.locals['infos']:
                if 'portfolio_value' in info:
                    self.training_logs['portfolio_values'].append(info['portfolio_value'])
                if 'trades_executed' in info:
                    self.training_logs['trades_executed'].append(info['trades_executed'])
        
        if self.locals.get('rewards'):
            self.training_logs['rewards'].extend(self.locals['rewards'])
        
        if self.locals.get('actions'):
            self.training_logs['actions'].extend(self.locals['actions'])
        
        return True

class PPOAgent:
    """PPO Agent for trading with GPU acceleration"""
    
    def __init__(self, config, logger=None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.model = None
        self.callback = None
    
    def create_model(self, env, model_params=None):
        """Create PPO model with specified parameters"""
        if model_params is None:
            model_params = {
                'learning_rate': self.config.LEARNING_RATE,
                'n_steps': 2048,
                'batch_size': self.config.BATCH_SIZE,
                'n_epochs': self.config.N_EPOCHS,
                'gamma': self.config.GAMMA,
                'gae_lambda': self.config.GAE_LAMBDA,
                'clip_range': self.config.CLIP_RATIO,
                'ent_coef': self.config.ENTROPY_COEF,
                'value_coef': self.config.VALUE_LOSS_COEF,
                'verbose': 1
            }
        
        self.logger.info("Creating PPO model with GPU acceleration...")
        
        # Create callback for tracking
        self.callback = TradingCallback(verbose=0)
        
        # Create PPO model
        self.model = PPO(
            "MlpPolicy",
            env,
            device='auto',  # Enable GPU acceleration
            **model_params
        )
        
        self.logger.info(f"PPO model created successfully")
        self.logger.info(f"Model Device: {self.model.device}")
        
        return self.model
    
    def train(self, total_timesteps, callback=None):
        """Train the PPO agent"""
        if self.model is None:
            raise ValueError("Model not created. Call create_model() first.")
        
        if callback is None:
            callback = self.callback
        
        self.logger.info(f"Training PPO agent for {total_timesteps} timesteps...")
        
        # Train the model
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=True
        )
        
        self.logger.info("Training completed successfully")
        
        return self.callback.training_logs if self.callback else {}
    
    def predict(self, observation, deterministic=True):
        """Predict action for given observation"""
        if self.model is None:
            raise ValueError("Model not created. Call create_model() first.")
        
        return self.model.predict(observation, deterministic=deterministic)
    
    def save(self, path):
        """Save the trained model"""
        if self.model is None:
            raise ValueError("Model not created. Call create_model() first.")
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)
        self.logger.info(f"Model saved to: {path}")
    
    def load(self, path):
        """Load a trained model"""
        self.model = PPO.load(path, device='auto')
        self.logger.info(f"Model loaded from: {path}")
        return self.model