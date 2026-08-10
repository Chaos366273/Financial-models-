"""
Enhanced Model Save Functions - PyCapsule Error Fixes
"""

import torch
import json
import zipfile
import tempfile
import os
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

def enhanced_model_save(model, path):
    """Enhanced model saving that completely avoids PyCapsule issues"""
    try:
        logger.info(" Using enhanced model save to avoid PyCapsule issues...")
        
        # CRITICAL: Don't modify the original model, save components directly
        try:
            # Save policy weights (most important)
            policy_path = f"{path}_policy.pt"
            torch.save(model.policy.state_dict(), policy_path)
            logger.info(f" Policy saved: {policy_path}")
            
            # Save value function if available
            try:
                value_path = f"{path}_value.pt"
                torch.save(model.policy.value_net.state_dict(), value_path)
                logger.info(f" Value function saved: {value_path}")
            except Exception as vf_error:
                logger.warning(f" Value function save failed: {vf_error}")
            
            # Save optimizer state if available
            try:
                optimizer_path = f"{path}_optimizer.pt"
                torch.save(model.policy.optimizer.state_dict(), optimizer_path)
                logger.info(f" Optimizer saved: {optimizer_path}")
            except Exception as opt_error:
                logger.warning(f" Optimizer save failed: {opt_error}")
            
            # Save model configuration
            config = {
                'model_type': 'PPO',
                'timestamp': datetime.now().isoformat(),
                'save_method': 'enhanced_component_based',
                'components': ['policy', 'value_function', 'optimizer'],
                'policy_kwargs': getattr(model.policy, 'policy_kwargs', {}),
                'observation_space': str(model.observation_space),
                'action_space': str(model.action_space),
                'learning_rate': getattr(model, 'learning_rate', 3e-4),
                'n_steps': getattr(model, 'n_steps', 2048),
                'batch_size': getattr(model, 'batch_size', 64),
                'n_epochs': getattr(model, 'n_epochs', 10),
                'gamma': getattr(model, 'gamma', 0.99),
                'gae_lambda': getattr(model, 'gae_lambda', 0.95),
                'clip_range': getattr(model, 'clip_range', 0.2),
                'ent_coef': getattr(model, 'ent_coef', 0.01),
                'vf_coef': getattr(model, 'vf_coef', 0.5),
                'max_grad_norm': getattr(model, 'max_grad_norm', 0.5),
                'target_kl': getattr(model, 'target_kl', None),
                'tensorboard_log': None,
                'policy_kwargs': getattr(model.policy, 'policy_kwargs', {}),
                'verbose': 1,
                'seed': getattr(model, 'seed', 42),
                'device': str(getattr(model, 'device', 'cpu')),
                '_init_setup_model': True
            }
            
            config_path = f"{path}_config.json"
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            logger.info(f" Config saved: {config_path}")
            
            # Create a simple zip file with all components
            zip_path = f"{path}.zip"
            with zipfile.ZipFile(zip_path, 'w') as zipf:
                # Add all component files
                for component_file in [policy_path, value_path, optimizer_path, config_path]:
                    if os.path.exists(component_file):
                        zipf.write(component_file, os.path.basename(component_file))
                        logger.info(f" Added to zip: {os.path.basename(component_file)}")
            
            logger.info(f" Enhanced model save successful: {zip_path}")
            return True
            
        except Exception as save_error:
            logger.error(f" Enhanced save failed: {save_error}")
            return False
            
    except Exception as e:
        logger.error(f" Enhanced model save failed: {e}")
        return False

def load_enhanced_model(model_path, env):
    """Load a model saved with enhanced_model_save"""
    try:
        logger.info(f" Loading enhanced model from: {model_path}")
        
        # Extract components from zip file
        with zipfile.ZipFile(model_path, 'r') as zipf:
            # Extract to temporary directory
            with tempfile.TemporaryDirectory() as temp_dir:
                zipf.extractall(temp_dir)
                
                # Load config
                config_path = os.path.join(temp_dir, 'ppo_trading_agent_config.json')
                if os.path.exists(config_path):
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                    logger.info(" Loaded model configuration")
                else:
                    logger.warning(" No config file found, using defaults")
                    config = {}
                
                # Create new PPO model with same parameters
                from stable_baselines3 import PPO
                
                model = PPO(
                    "MlpPolicy",
                    env,
                    learning_rate=config.get('learning_rate', 3e-4),
                    n_steps=config.get('n_steps', 2048),
                    batch_size=config.get('batch_size', 64),
                    n_epochs=config.get('n_epochs', 10),
                    gamma=config.get('gamma', 0.99),
                    gae_lambda=config.get('gae_lambda', 0.95),
                    clip_range=config.get('clip_range', 0.2),
                    ent_coef=config.get('ent_coef', 0.01),
                    vf_coef=config.get('vf_coef', 0.5),
                    max_grad_norm=config.get('max_grad_norm', 0.5),
                    target_kl=config.get('target_kl', None),
                    tensorboard_log=None,
                    policy_kwargs=config.get('policy_kwargs', {}),
                    verbose=1,
                    seed=config.get('seed', 42),
                    device=config.get('device', 'cpu'),
                    _init_setup_model=True
                )
                
                # Load policy weights
                policy_path = os.path.join(temp_dir, 'ppo_trading_agent_policy.pt')
                if os.path.exists(policy_path):
                    model.policy.load_state_dict(torch.load(policy_path, map_location='cpu'))
                    logger.info(" Loaded policy weights")
                
                # Load value function weights
                value_path = os.path.join(temp_dir, 'ppo_trading_agent_value.pt')
                if os.path.exists(value_path):
                    model.policy.value_net.load_state_dict(torch.load(value_path, map_location='cpu'))
                    logger.info(" Loaded value function weights")
                
                # Load optimizer state
                optimizer_path = os.path.join(temp_dir, 'ppo_trading_agent_optimizer.pt')
                if os.path.exists(optimizer_path):
                    model.policy.optimizer.load_state_dict(torch.load(optimizer_path, map_location='cpu'))
                    logger.info(" Loaded optimizer state")
                
                logger.info(" Enhanced model loaded successfully")
                return model
                
    except Exception as e:
        logger.error(f" Enhanced model loading failed: {e}")
        return None
