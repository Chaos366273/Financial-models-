"""
Training loop for PPO Trading Agent
Handles experience collection and training iterations
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any, cast
import logging
import time
from collections import deque
import pickle
import os

from ..core.exceptions import TrainingException, ValidationException
from .ppo_agent import PPOAgent
from ..models.trading_env import FixedTradingEnv
from ..models.risk_manager import RiskManager
from ..models.reward_calculator import RewardCalculator

class TrainingLoop:
    """Training loop for PPO agent"""

    def __init__(self, config: Dict, agent: PPOAgent, env: FixedTradingEnv):
        """
        Initialize training loop

        Args:
            config: Configuration dictionary
            agent: PPO agent instance
            env: Trading environment instance
        """
        self.config = config
        self.agent = agent
        self.env = env
        self.logger = logging.getLogger(__name__)

        # Training parameters
        self.max_episodes = config.get('max_episodes', 1000)
        self.max_steps_per_episode = config.get('max_steps_per_episode', 1000)
        self.batch_size = config.get('batch_size', 64)
        self.update_frequency = config.get('update_frequency', 10)
        self.save_frequency = config.get('save_frequency', 100)
        self.eval_frequency = config.get('eval_frequency', 50)

        # Experience buffer
        self.experience_buffer = deque(maxlen=10000)
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)

        # Performance tracking
        self.best_reward = float('-inf')
        self.training_start_time = time.time()
        self.total_steps = 0

        # Initialize components
        self.risk_manager = RiskManager(config.get('trading_config', {}))
        self.reward_calculator = RewardCalculator(config.get('reward_config', {}))

    def collect_experience(self, num_steps: int) -> List[Dict]:
        """
        Collect experience from environment

        Args:
            num_steps: Number of steps to collect

        Returns:
            List of experience dictionaries
        """
        experiences = []
        step_count = 0

        try:
            env = cast(Any, self.env)
            agent = cast(Any, self.agent)
            while step_count < num_steps:
                # Reset environment if episode is done
                max_steps = getattr(env, "max_steps", float("inf"))
                if env.current_step >= max_steps:
                    obs, info = env.reset()
                else:
                    obs = env._get_observation()

                # Get action from agent
                action, log_prob, value = agent.get_action(obs)

                # Apply risk management
                current_prices = env.price_matrix[env.current_step]
                adjusted_action, violations = self.risk_manager.check_position_risk(
                    action, current_prices, env.portfolio_value, 
                    cast(Any, env.positions), strict=False
                )

                # Apply stop-loss/take-profit
                adjusted_action = self.risk_manager.apply_stop_loss_take_profit(
                    adjusted_action, current_prices, cast(Any, env.positions), 
                    env.portfolio_value
                )

                # Take step in environment
                next_obs, reward, done, truncated, info = env.step(adjusted_action)

                # Store experience
                experience = {
                    'observation': obs,
                    'action': adjusted_action,
                    'log_prob': log_prob,
                    'value': value,
                    'reward': reward,
                    'next_observation': next_obs,
                    'done': done,
                    'info': info
                }
                experiences.append(experience)

                step_count += 1
                self.total_steps += 1

                # Update risk manager
                self.risk_manager.update_portfolio_peak(env.portfolio_value)

                # Check if trading should be stopped
                if self.risk_manager.should_stop_trading(env.portfolio_value):
                    self.logger.warning("Trading stopped due to risk limits")
                    break

        except Exception as e:
            self.logger.error(f"Experience collection failed: {e}")
            raise TrainingException(f"Experience collection failed: {e}")

        return experiences

    def compute_advantages(self, experiences: List[Dict], gamma: float = 0.99, 
                          lambda_: float = 0.95) -> List[Dict]:
        """
        Compute advantages for experiences

        Args:
            experiences: List of experience dictionaries
            gamma: Discount factor
            lambda_: GAE lambda parameter

        Returns:
            List of experiences with advantages and returns
        """
        try:
            # Extract values and rewards
            values = [exp['value'] for exp in experiences]
            rewards = [exp['reward'] for exp in experiences]
            dones = [exp['done'] for exp in experiences]

            # Compute returns and advantages
            returns = []
            advantages = []
            
            # Last value (bootstrap)
            last_value = values[-1] if not dones[-1] else 0.0
            
            # Compute returns and advantages backwards
            for i in reversed(range(len(experiences))):
                if dones[i]:
                    returns.insert(0, rewards[i])
                    advantages.insert(0, rewards[i] - values[i])
                else:
                    if i == len(experiences) - 1:
                        next_value = last_value
                    else:
                        next_value = values[i + 1]
                    
                    delta = rewards[i] + gamma * next_value - values[i]
                    returns.insert(0, rewards[i] + gamma * next_value)
                    advantages.insert(0, delta + gamma * lambda_ * (advantages[0] if advantages else 0))

            # Normalize advantages
            advantages = np.array(advantages)
            if len(advantages) > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Add advantages and returns to experiences
            for i, exp in enumerate(experiences):
                exp['advantage'] = advantages[i]
                exp['return'] = returns[i]

            return experiences

        except Exception as e:
            self.logger.error(f"Advantage computation failed: {e}")
            raise TrainingException(f"Advantage computation failed: {e}")

    def create_batch(self, experiences: List[Dict]) -> Dict:
        """
        Create training batch from experiences

        Args:
            experiences: List of experience dictionaries

        Returns:
            Training batch dictionary
        """
        try:
            batch = {
                'observations': np.array([exp['observation'] for exp in experiences]),
                'actions': np.array([exp['action'] for exp in experiences]),
                'log_probs': np.array([exp['log_prob'] for exp in experiences]),
                'values': np.array([exp['value'] for exp in experiences]),
                'advantages': np.array([exp['advantage'] for exp in experiences]),
                'returns': np.array([exp['return'] for exp in experiences])
            }

            return batch

        except Exception as e:
            self.logger.error(f"Batch creation failed: {e}")
            raise TrainingException(f"Batch creation failed: {e}")

    def train_episode(self) -> Dict:
        """
        Train for one episode

        Returns:
            Episode training metrics
        """
        try:
            env = cast(Any, self.env)
            agent = cast(Any, self.agent)
            # Reset environment
            obs, info = env.reset()
            episode_reward = 0.0
            episode_steps = 0
            episode_trades = 0

            # Collect experience for this episode
            experiences = []
            
            while episode_steps < self.max_steps_per_episode:
                # Get action
                action, log_prob, value = agent.get_action(obs)

                # Apply risk management
                current_prices = env.price_matrix[env.current_step]
                adjusted_action, violations = self.risk_manager.check_position_risk(
                    action, current_prices, env.portfolio_value, 
                    cast(Any, env.positions), strict=False
                )

                # Take step
                next_obs, reward, done, truncated, info = env.step(adjusted_action)

                # Store experience
                experience = {
                    'observation': obs,
                    'action': adjusted_action,
                    'log_prob': log_prob,
                    'value': value,
                    'reward': reward,
                    'next_observation': next_obs,
                    'done': done,
                    'info': info
                }
                experiences.append(experience)

                # Update tracking
                episode_reward += reward
                episode_steps += 1
                episode_trades += info.get('num_trades', 0)

                # Update observation
                obs = next_obs

                # Check if done
                if done or truncated:
                    break

                # Check risk limits
                if self.risk_manager.should_stop_trading(env.portfolio_value):
                    self.logger.warning("Episode stopped due to risk limits")
                    break

            # Compute advantages
            experiences = self.compute_advantages(experiences)

            # Update model if enough experience
            if len(experiences) >= self.batch_size:
                batch = self.create_batch(experiences[:self.batch_size])
                metrics = agent.update(batch)
            else:
                metrics = {}

            # Update tracking
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_steps)

            # Update best reward
            if episode_reward > self.best_reward:
                self.best_reward = episode_reward
                agent.update_best_reward(episode_reward)

            episode_metrics = {
                'episode_reward': episode_reward,
                'episode_steps': episode_steps,
                'episode_trades': episode_trades,
                'portfolio_value': env.portfolio_value,
                'cash': env.cash,
                'drawdown': env.current_drawdown,
                'best_reward': self.best_reward,
                'avg_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0,
                'avg_length': np.mean(self.episode_lengths) if self.episode_lengths else 0
            }

            # Add training metrics if available
            if metrics:
                episode_metrics.update(metrics)

            return episode_metrics

        except Exception as e:
            self.logger.error(f"Episode training failed: {e}")
            raise TrainingException(f"Episode training failed: {e}")

    def train(self, save_dir: str = "models") -> Dict:
        """
        Main training loop

        Args:
            save_dir: Directory to save models

        Returns:
            Training summary
        """
        try:
            agent = cast(Any, self.agent)
            self.logger.info("Starting training loop")
            training_start_time = time.time()

            # Create save directory
            os.makedirs(save_dir, exist_ok=True)

            # Training loop
            for episode in range(self.max_episodes):
                try:
                    # Train one episode
                    episode_metrics = self.train_episode()

                    # Log progress
                    if episode % 10 == 0:
                        self.logger.info(
                            f"Episode {episode}/{self.max_episodes} - "
                            f"Reward: {episode_metrics['episode_reward']:.4f} - "
                            f"Portfolio: ${episode_metrics['portfolio_value']:.2f} - "
                            f"Trades: {episode_metrics['episode_trades']}"
                        )

                    # Save model periodically
                    if episode % self.save_frequency == 0 and episode > 0:
                        model_path = os.path.join(save_dir, f"ppo_model_episode_{episode}.pth")
                        agent.save_model(model_path)

                    # Evaluate periodically
                    if episode % self.eval_frequency == 0 and episode > 0:
                        eval_metrics = self.evaluate()
                        self.logger.info(f"Evaluation - Reward: {eval_metrics['avg_reward']:.4f}")

                except Exception as e:
                    self.logger.error(f"Episode {episode} failed: {e}")
                    continue

            # Save final model
            final_model_path = os.path.join(save_dir, "ppo_model_final.pth")
            agent.save_model(final_model_path)

            # Training summary
            training_time = time.time() - training_start_time
            training_summary = {
                'total_episodes': self.max_episodes,
                'total_steps': self.total_steps,
                'best_reward': self.best_reward,
                'avg_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0,
                'training_time': training_time,
                'steps_per_second': self.total_steps / training_time if training_time > 0 else 0
            }

            self.logger.info(f"Training completed: {training_summary}")
            return training_summary

        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            raise TrainingException(f"Training failed: {e}")

    def evaluate(self, num_episodes: int = 10) -> Dict:
        """
        Evaluate the trained agent

        Args:
            num_episodes: Number of evaluation episodes

        Returns:
            Evaluation metrics
        """
        try:
            env = cast(Any, self.env)
            agent = cast(Any, self.agent)
            eval_rewards = []
            eval_portfolio_values = []
            eval_trades = []

            for _ in range(num_episodes):
                # Reset environment
                obs, info = env.reset()
                episode_reward = 0.0
                episode_trades = 0

                max_steps = getattr(env, "max_steps", float("inf"))
                while env.current_step < max_steps:
                    # Get action (no exploration during evaluation)
                    action, _, _ = agent.get_action(obs)

                    # Apply risk management
                    current_prices = env.price_matrix[env.current_step]
                    adjusted_action, _ = self.risk_manager.check_position_risk(
                        action, current_prices, env.portfolio_value, 
                        cast(Any, env.positions), strict=True
                    )

                    # Take step
                    next_obs, reward, done, truncated, info = env.step(adjusted_action)

                    episode_reward += reward
                    episode_trades += info.get('num_trades', 0)
                    obs = next_obs

                    if done or truncated:
                        break

                eval_rewards.append(episode_reward)
                eval_portfolio_values.append(env.portfolio_value)
                eval_trades.append(episode_trades)

            eval_metrics = {
                'avg_reward': np.mean(eval_rewards),
                'std_reward': np.std(eval_rewards),
                'avg_portfolio_value': np.mean(eval_portfolio_values),
                'avg_trades': np.mean(eval_trades),
                'num_episodes': num_episodes
            }

            return eval_metrics

        except Exception as e:
            self.logger.error(f"Evaluation failed: {e}")
            raise TrainingException(f"Evaluation failed: {e}")

    def get_training_stats(self) -> Dict:
        """Get current training statistics"""
        return {
            'total_steps': self.total_steps,
            'best_reward': self.best_reward,
            'avg_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0,
            'avg_episode_length': np.mean(self.episode_lengths) if self.episode_lengths else 0,
            'training_time': time.time() - self.training_start_time,
            'episodes_completed': len(self.episode_rewards)
        }
