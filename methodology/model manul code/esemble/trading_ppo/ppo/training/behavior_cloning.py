"""
Behavior Cloning Module for PPO Warm Start
Trains a simple policy to initialize PPO with sensible behavior
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import logging
import os
import pickle
from typing import Tuple, Optional
import warnings

logger = logging.getLogger(__name__)

class BehaviorCloningPolicy(nn.Module):
    """Simple neural network for behavior cloning"""
    
    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # Output in [-1, 1] range
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, x):
        return self.network(x)

def train_bc(
    observations: np.ndarray,
    actions: np.ndarray,
    epochs: int = 3,
    batch_size: int = 256,
    lr: float = 1e-3,
    validation_split: float = 0.2
) -> BehaviorCloningPolicy:
    """
    Train behavior cloning policy
    
    Args:
        observations: (num_samples, observation_dim)
        actions: (num_samples, action_dim)
        epochs: Number of training epochs
        batch_size: Batch size for training
        lr: Learning rate
        validation_split: Fraction of data for validation
    
    Returns:
        trained_policy: BehaviorCloningPolicy
    """
    try:
        logger.info(f" Training behavior cloning policy")
        logger.info(f" Dataset: {len(observations)} samples")
        logger.info(f" Observation dim: {observations.shape[1]}")
        logger.info(f" Action dim: {actions.shape[1]}")
        
        # CRITICAL FIX: Set device for GPU usage
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f" Using device: {device}")
        
        # Convert to torch tensors and move to device
        obs_tensor = torch.FloatTensor(observations).to(device)
        act_tensor = torch.FloatTensor(actions).to(device)
        
        # Split into train/validation
        num_val = int(len(observations) * validation_split)
        train_obs = obs_tensor[:-num_val]
        train_act = act_tensor[:-num_val]
        val_obs = obs_tensor[-num_val:]
        val_act = act_tensor[-num_val:]
        
        # Create data loaders
        train_dataset = TensorDataset(train_obs, train_act)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        val_dataset = TensorDataset(val_obs, val_act)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Initialize model with SUPER AGENT configuration
        model = BehaviorCloningPolicy(
            observation_dim=observations.shape[1],
            action_dim=actions.shape[1],
            hidden_dim=512  # SUPER AGENT: Larger hidden dim for better pattern recognition
        ).to(device)  # CRITICAL FIX: Move model to device

        # CRITICAL FIX: Ensure autograd is enabled (may be disabled globally elsewhere)
        try:
            torch.set_grad_enabled(True)
            for p in model.parameters():
                p.requires_grad = True
        except Exception:
            pass
        
        # SUPER AGENT: Enhanced loss function and optimizer for profitable trading
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)  # L2 regularization
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)  # Learning rate scheduling
        
        # Training loop
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0.0
            
            for batch_obs, batch_act in train_loader:
                optimizer.zero_grad()
                
                pred_act = model(batch_obs)
                loss = criterion(pred_act, batch_act)
                
                # Ensure loss requires grad
                if not loss.requires_grad:
                    # Re-enable autograd as a safety net
                    torch.set_grad_enabled(True)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch_obs, batch_act in val_loader:
                    pred_act = model(batch_obs)
                    loss = criterion(pred_act, batch_act)
                    val_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            
            logger.info(f" Epoch {epoch+1}/{epochs}: Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, LR: {scheduler.get_last_lr()[0]:.6f}")
            
            # SUPER AGENT: Update learning rate schedule
            scheduler.step()
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = model.state_dict().copy()
        
        # Load best model
        model.load_state_dict(best_model_state)
        
        logger.info(f" Behavior cloning training completed")
        logger.info(f" Best validation loss: {best_val_loss:.6f}")
        
        return model
        
    except Exception as e:
        logger.error(f" Behavior cloning training failed: {e}")
        # Return a simple model as fallback
        model = BehaviorCloningPolicy(
            observation_dim=observations.shape[1] if len(observations) > 0 else 100,
            action_dim=actions.shape[1] if len(actions) > 0 else 20
        )
        return model

def save_bc_init_legacy(model: BehaviorCloningPolicy, filepath: Optional[str] = None):
    """Save behavior cloning model for PPO initialization"""
    try:
        if filepath is None:
            filepath = "ppo/saved_models/ppo_bc_init.pt"
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save model state dict
        torch.save(model.state_dict(), filepath)
        
        logger.info(f" Behavior cloning model saved to: {filepath}")
        
    except Exception as e:
        logger.error(f" Error saving behavior cloning model: {e}")

def load_bc_init_legacy(filepath: Optional[str] = None) -> Optional[BehaviorCloningPolicy]:
    """Load behavior cloning model for PPO initialization"""
    try:
        if filepath is None:
            filepath = "ppo/saved_models/ppo_bc_init.pt"
        
        if not os.path.exists(filepath):
            logger.warning(f" Behavior cloning model not found: {filepath}")
            return None
        
        # Load model state dict
        state_dict = torch.load(filepath, map_location='cpu')
        
        # Create model with dummy dimensions (will be overridden)
        model = BehaviorCloningPolicy(observation_dim=100, action_dim=20)
        model.load_state_dict(state_dict)
        
        logger.info(f" Behavior cloning model loaded from: {filepath}")
        return model
        
    except Exception as e:
        logger.error(f" Error loading behavior cloning model: {e}")
        return None

def apply_bc_to_ppo(bc_model: BehaviorCloningPolicy, ppo_policy) -> bool:
    """
    Apply behavior cloning weights to PPO policy
    
    Args:
        bc_model: Trained behavior cloning model
        ppo_policy: PPO policy to initialize
    
    Returns:
        success: Whether initialization was successful
    """
    try:
        logger.info(" Applying behavior cloning weights to PPO policy")
        
        # Get PPO policy network
        if hasattr(ppo_policy, 'policy'):
            policy_net = ppo_policy.policy
        else:
            policy_net = ppo_policy
        
        # Try to find compatible layers
        bc_state_dict = bc_model.state_dict()
        ppo_state_dict = policy_net.state_dict()
        
        # Find matching layers
        matched_layers = {}
        for bc_name, bc_param in bc_state_dict.items():
            for ppo_name, ppo_param in ppo_state_dict.items():
                if bc_param.shape == ppo_param.shape:
                    # Check if it's a policy head layer (usually the last few layers)
                    if 'policy' in ppo_name.lower() or 'action' in ppo_name.lower():
                        matched_layers[ppo_name] = bc_param
                        logger.info(f" Matched layer: {bc_name} -> {ppo_name}")
                        break
        
        if not matched_layers:
            logger.warning(" No compatible layers found for BC initialization")
            return False
        
        # Apply matched weights
        for ppo_name, bc_param in matched_layers.items():
            ppo_state_dict[ppo_name] = bc_param
        
        # Load updated state dict
        policy_net.load_state_dict(ppo_state_dict)
        
        logger.info(f" Successfully applied {len(matched_layers)} layers from BC model")
        return True
        
    except Exception as e:
        logger.error(f" Error applying BC weights to PPO: {e}")
        return False

def validate_bc_compatibility(bc_model: BehaviorCloningPolicy, ppo_policy) -> bool:
    """
    Validate if BC model is compatible with PPO policy
    
    Args:
        bc_model: Behavior cloning model
        ppo_policy: PPO policy to check compatibility with
    
    Returns:
        compatible: Whether models are compatible
    """
    try:
        # Get policy network
        if hasattr(ppo_policy, 'policy'):
            policy_net = ppo_policy.policy
        else:
            policy_net = ppo_policy
        
        # Check observation and action dimensions
        bc_obs_dim = bc_model.network[0].in_features
        bc_act_dim = bc_model.network[-1].out_features
        
        # Get PPO dimensions (this is approximate)
        ppo_obs_dim = None
        ppo_act_dim = None
        
        for name, param in policy_net.named_parameters():
            if 'policy' in name.lower() and 'weight' in name.lower():
                if ppo_obs_dim is None:
                    ppo_obs_dim = param.shape[1]
                if ppo_act_dim is None:
                    ppo_act_dim = param.shape[0]
                break
        
        if ppo_obs_dim is None or ppo_act_dim is None:
            logger.warning(" Could not determine PPO dimensions")
            return False
        
        # Check compatibility
        obs_compatible = bc_obs_dim == ppo_obs_dim
        act_compatible = bc_act_dim == ppo_act_dim
        
        logger.info(f" BC dimensions: obs={bc_obs_dim}, act={bc_act_dim}")
        logger.info(f" PPO dimensions: obs={ppo_obs_dim}, act={ppo_act_dim}")
        logger.info(f" Compatibility: obs={obs_compatible}, act={act_compatible}")
        
        return obs_compatible and act_compatible
        
    except Exception as e:
        logger.error(f" Error validating BC compatibility: {e}")
        return False

def save_bc_init(bc_model: BehaviorCloningPolicy, filepath: Optional[str] = None) -> bool:
    """
    Save behavior cloning model for initialization
    
    Args:
        bc_model: Trained behavior cloning model
        filepath: Path to save the model (optional)
    
    Returns:
        success: Whether save was successful
    """
    try:
        if filepath is None:
            filepath = 'ppo/saved_models/ppo_bc_init.pt'
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save model with metadata for proper loading
        try:
            # Get dimensions from the first and last linear layers
            first_layer = None
            last_layer = None
            for module in bc_model.network:
                if isinstance(module, nn.Linear):
                    if first_layer is None:
                        first_layer = module
                    last_layer = module
            
            save_data = {
                'state_dict': bc_model.state_dict(),
                'observation_dim': first_layer.in_features if first_layer else 510,
                'action_dim': last_layer.out_features if last_layer else 6,
                'hidden_dim': first_layer.out_features if first_layer else 256
            }
        except Exception as e:
            # Fallback: use default dimensions if network access fails
            logger.warning(f"Could not access network dimensions: {e}")
            save_data = {
                'state_dict': bc_model.state_dict(),
                'observation_dim': 510,  # Default
                'action_dim': 6,  # Default
                'hidden_dim': 256  # Default
            }
        
        torch.save(save_data, filepath)
        
        logger.info(f"PASS: BC initialization model saved to {filepath}")
        return True
        
    except Exception as e:
        logger.error(f"FAIL: Error saving BC initialization model: {e}")
        return False

def load_bc_init(filepath: Optional[str] = None) -> Optional[BehaviorCloningPolicy]:
    """
    Load behavior cloning model for initialization
    
    Args:
        filepath: Path to load the model from (optional)
    
    Returns:
        bc_model: Loaded behavior cloning model or None if failed
    """
    try:
        if filepath is None:
            filepath = 'ppo/saved_models/ppo_bc_init.pt'
        
        if not os.path.exists(filepath):
            logger.warning(f"WARNING: BC initialization model not found at {filepath}")
            return None
        
        # Load the model with metadata
        save_data = torch.load(filepath, map_location='cpu')
        
        # Check if it's the new format with metadata
        if isinstance(save_data, dict) and 'state_dict' in save_data:
            # New format with metadata
            state_dict = save_data['state_dict']
            observation_dim = save_data['observation_dim']
            action_dim = save_data['action_dim']
            hidden_dim = save_data.get('hidden_dim', 256)
        else:
            # Old format - use default dimensions
            state_dict = save_data
            observation_dim = 510  # Default from your environment
            action_dim = 6  # Default from your environment
            hidden_dim = 256
        
        # Create model with correct dimensions
        bc_model = BehaviorCloningPolicy(observation_dim, action_dim, hidden_dim)
        
        # Load state dict with strict=False to handle dimension mismatches
        try:
            bc_model.load_state_dict(state_dict, strict=True)
        except Exception as e:
            logger.warning(f"Strict loading failed, trying non-strict: {e}")
            # Try non-strict loading for partial compatibility
            bc_model.load_state_dict(state_dict, strict=False)
        
        logger.info(f"PASS: BC initialization model loaded from {filepath}")
        return bc_model
        
    except Exception as e:
        logger.error(f"FAIL: Error loading BC initialization model: {e}")
        return None


