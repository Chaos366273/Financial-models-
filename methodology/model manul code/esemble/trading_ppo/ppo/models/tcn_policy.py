"""
Temporal Convolutional Network (TCN) Policy for PPO Trading Agent
More powerful than LSTM for temporal pattern recognition in financial markets
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable, Type
import gymnasium as gym
from gymnasium import spaces
import logging

logger = logging.getLogger(__name__)

BasePolicy: Any = Any
BaseFeaturesExtractor: Any = Any

def _noop_register_policy(*args, **kwargs) -> None:
    return None

register_policy: Callable[[str, Any], None] = _noop_register_policy

try:
    from stable_baselines3.common.policies import BasePolicy as _BasePolicy
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor as _BaseFeaturesExtractor
    try:
        from stable_baselines3.common.policies import register_policy as _register_policy  # type: ignore
        REGISTER_POLICY_AVAILABLE = True
    except ImportError:
        REGISTER_POLICY_AVAILABLE = False
        # Try alternative registration methods
        try:
            from stable_baselines3.common.policies import POLICY_REGISTRY as _POLICY_REGISTRY  # type: ignore
            def _registry_register_policy(name, policy_class) -> None:
                _POLICY_REGISTRY[name] = policy_class
                logger.info(f"Registered TCN policy via POLICY_REGISTRY: {name}")
            REGISTER_POLICY_AVAILABLE = True
        except ImportError:
            # Fallback: manual registration via get_policy_from_name
            def _fallback_register_policy(name, policy_class) -> None:
                # Store in module-level dict for manual lookup
                if not hasattr(_fallback_register_policy, '_registry'):
                    _fallback_register_policy._registry = {}
                _fallback_register_policy._registry[name] = policy_class
                logger.info(f"Registered TCN policy in fallback registry: {name}")
            REGISTER_POLICY_AVAILABLE = True
    BasePolicy = _BasePolicy
    BaseFeaturesExtractor = _BaseFeaturesExtractor
    if '_register_policy' in locals():
        register_policy = _register_policy
    elif '_registry_register_policy' in locals():
        register_policy = _registry_register_policy
    elif '_fallback_register_policy' in locals():
        register_policy = _fallback_register_policy
    SB3_AVAILABLE = True
except ImportError as e:
    SB3_AVAILABLE = False
    REGISTER_POLICY_AVAILABLE = False
    logger.warning(f"Stable-Baselines3 not available - TCN policy cannot be used: {e}")
    register_policy = _noop_register_policy


class TCNBlock(nn.Module):
    """
    Temporal Convolutional Network Block with dilated convolutions
    Provides long-range temporal memory through exponential dilation
    """
    
    def __init__(self, 
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 dilation: int = 1,
                 dropout: float = 0.1,
                 use_residual: bool = True):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.use_residual = use_residual and (in_channels == out_channels)
        
        # Padding to maintain sequence length
        padding = (kernel_size - 1) * dilation
        
        # Dilated convolution
        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding
        )
        
        # Weight normalization for stability
        self.conv1 = nn.utils.weight_norm(self.conv1)
        
        # Activation and dropout
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        # Residual connection (if channels match)
        if self.use_residual:
            self.residual_conv = nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1
            ) if in_channels != out_channels else nn.Identity()
        else:
            self.residual_conv = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through TCN block
        
        Args:
            x: Input tensor of shape (batch, in_channels, sequence_length)
        
        Returns:
            Output tensor of shape (batch, out_channels, sequence_length)
        """
        # Store input for residual connection
        residual = x if self.use_residual else None
        
        # Dilated convolution
        out = self.conv1(x)
        
        # Remove padding from the right to maintain sequence length
        if self.dilation > 0:
            out = out[:, :, :-self.dilation * (self.kernel_size - 1)]
        
        # Activation and dropout
        out = self.activation(out)
        out = self.dropout(out)
        
        # Residual connection
        if self.use_residual and residual is not None:
            if self.residual_conv is not None:
                residual = self.residual_conv(residual)
            out = out + residual
        
        return out


class TCNFeatureExtractor(BaseFeaturesExtractor if SB3_AVAILABLE else nn.Module):
    """
    TCN Feature Extractor for temporal sequence processing.
    Reshapes flat observations to temporal sequences and processes with TCN.
    Compatible with MaskableActorCriticPolicy via features_extractor_class.
    Dimensions derived from observation_space (e.g. 1442 = 1430 market + 12 portfolio).
    """
    
    def __init__(self,
                 observation_space: gym.spaces.Space,
                 window_size: int = 10,
                 portfolio_features: int = 12,
                 tcn_hidden_dims: List[int] = [256, 256, 128, 128],
                 tcn_kernel_size: int = 3,
                 tcn_dilation_rates: List[int] = [1, 2, 4, 8],
                 tcn_dropout: float = 0.1,
                 tcn_use_residual: bool = True,
                 features_dim: int = 128,
                 **kwargs):
        obs_shape = observation_space.shape or ()
        obs_size = int(np.prod(obs_shape))
        market_features = obs_size - portfolio_features
        market_features_per_timestep = market_features // max(1, window_size)
        features_per_timestep = market_features_per_timestep + 1
        out_features_dim = tcn_hidden_dims[-1]
        if SB3_AVAILABLE:
            super().__init__(observation_space, features_dim=out_features_dim)
        else:
            nn.Module.__init__(self)
            self._observation_space = observation_space
            self._features_dim = out_features_dim
        
        self.window_size = window_size
        self.obs_size = obs_size
        self.market_features = market_features
        self.portfolio_features = portfolio_features
        self.market_features_per_timestep = market_features_per_timestep
        self.features_per_timestep = features_per_timestep
        
        self.input_projection = nn.Linear(self.features_per_timestep, tcn_hidden_dims[0])
        
        self.tcn_blocks = nn.ModuleList()
        in_channels = tcn_hidden_dims[0]
        for hidden_dim, dilation in zip(tcn_hidden_dims, tcn_dilation_rates):
            self.tcn_blocks.append(
                TCNBlock(
                    in_channels=in_channels,
                    out_channels=hidden_dim,
                    kernel_size=tcn_kernel_size,
                    dilation=dilation,
                    dropout=tcn_dropout,
                    use_residual=tcn_use_residual
                )
            )
            in_channels = hidden_dim
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
    
    @property
    def features_dim(self) -> int:
        return getattr(self, '_features_dim', 128)
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size = observations.shape[0]
        market_features = observations[:, :self.market_features]
        portfolio_features = observations[:, self.market_features:]
        market_reshaped = market_features.view(batch_size, self.window_size, self.market_features_per_timestep)
        portfolio_normalized = portfolio_features[:, 0:1]
        portfolio_broadcast = portfolio_normalized.unsqueeze(1).expand(-1, self.window_size, -1)
        tcn_input = torch.cat([market_reshaped, portfolio_broadcast], dim=2)
        tcn_input = self.input_projection(tcn_input)
        tcn_input = tcn_input.transpose(1, 2)
        x = tcn_input
        for tcn_block in self.tcn_blocks:
            x = tcn_block(x)
        x = self.global_pool(x)
        x = x.squeeze(-1)
        return x


if SB3_AVAILABLE:
    class TCNTradingPolicy(BasePolicy):
        """
        TCN Policy for PPO Trading Agent
        Uses Temporal Convolutional Network for temporal memory
        More powerful than LSTM, faster on CPU, better long-range memory
        """
        
        def __init__(self,
                     observation_space: gym.spaces.Space,
                     action_space: gym.spaces.Space,
                     lr_schedule: Callable[[float], float],
                     net_arch: Optional[List[int]] = None,
                     activation_fn: Type[nn.Module] = nn.ReLU,
                     tcn_hidden_dims: List[int] = [256, 256, 128, 128],
                     tcn_kernel_size: int = 3,
                     tcn_dilation_rates: List[int] = [1, 2, 4, 8],
                     tcn_dropout: float = 0.1,
                     tcn_use_residual: bool = True,
                     window_size: int = 10,
                     **kwargs):
            # Extract TCN-specific kwargs before passing to super
            tcn_kwargs = {
                'tcn_hidden_dims': tcn_hidden_dims,
                'tcn_kernel_size': tcn_kernel_size,
                'tcn_dilation_rates': tcn_dilation_rates,
                'tcn_dropout': tcn_dropout,
                'tcn_use_residual': tcn_use_residual,
                'window_size': window_size,
            }
            # Remove TCN kwargs and other unsupported args from kwargs dict
            # BasePolicy doesn't accept use_sde, log_std_init, etc. for discrete actions
            filtered_kwargs = {}
            for k, v in kwargs.items():
                if k.startswith('tcn_') or k == 'window_size':
                    continue
                # Filter out continuous action space args for discrete actions
                if isinstance(action_space, spaces.Discrete):
                    if k in ['use_sde', 'log_std_init', 'sde_net_arch', 'use_expln', 'ortho_init']:
                        continue
                filtered_kwargs[k] = v
            
            super().__init__(
                observation_space,
                action_space,
                lr_schedule,
                **filtered_kwargs
            )
            self.action_space = action_space
            
            # TCN feature extractor: derive portfolio_features from obs so reshape matches env
            _obs_size = int(np.prod(observation_space.shape))
            _win = tcn_kwargs['window_size']
            _pf = _obs_size - _win * (_obs_size // _win)
            portfolio_features = _pf if 1 <= _pf <= 64 else 4
            self.features_extractor = TCNFeatureExtractor(
                observation_space=observation_space,
                window_size=tcn_kwargs['window_size'],
                portfolio_features=portfolio_features,
                tcn_hidden_dims=tcn_kwargs['tcn_hidden_dims'],
                tcn_kernel_size=tcn_kwargs['tcn_kernel_size'],
                tcn_dilation_rates=tcn_kwargs['tcn_dilation_rates'],
                tcn_dropout=tcn_kwargs['tcn_dropout'],
                tcn_use_residual=tcn_kwargs['tcn_use_residual'],
                features_dim=tcn_kwargs['tcn_hidden_dims'][-1]
            )
            
            # Policy and value heads
            features_dim = self.features_extractor.features_dim
            
            # Get action space size
            if isinstance(action_space, spaces.Discrete):
                action_dim = int(action_space.n)
            else:
                action_shape = action_space.shape or ()
                action_dim = int(np.prod(action_shape))
            
            # Policy head (action distribution)
            if net_arch is None:
                net_arch = [64]
            
            policy_layers = []
            in_dim = features_dim
            for out_dim in net_arch:
                policy_layers.append(nn.Linear(in_dim, out_dim))
                policy_layers.append(activation_fn())
                in_dim = out_dim
            
            # Output layer for action logits
            policy_layers.append(nn.Linear(in_dim, action_dim))
            self.action_net = nn.Sequential(*policy_layers)
            
            # Value head
            value_layers = []
            in_dim = features_dim
            for out_dim in net_arch:
                value_layers.append(nn.Linear(in_dim, out_dim))
                value_layers.append(activation_fn())
                in_dim = out_dim
            
            value_layers.append(nn.Linear(in_dim, 1))
            self.value_net = nn.Sequential(*value_layers)
            
            # Build optimizer (required by SB3)
            self._build(lr_schedule)
        
        def _build(self, lr_schedule: Callable[[float], float]) -> None:
            """
            Build the policy network and optimizer (called by BasePolicy)
            This is required for Stable-Baselines3 to properly initialize the optimizer
            """
            # Skip if optimizer already exists (may be called multiple times by SB3)
            if hasattr(self, 'optimizer') and self.optimizer is not None:
                return
            
            # Collect all trainable parameters
            trainable_params = list(self.features_extractor.parameters())
            trainable_params.extend(list(self.action_net.parameters()))
            trainable_params.extend(list(self.value_net.parameters()))
            
            # Create optimizer with learning rate schedule
            # Use Adam optimizer (standard for PPO)
            # Note: SB3's PPO algorithm will update the learning rate via lr_schedule
            import torch.optim as optim
            initial_lr = float(lr_schedule(1))
            self.optimizer = optim.Adam(
                trainable_params,
                lr=initial_lr,  # Initial learning rate
                eps=1e-5
            )
        
        def forward(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """
            Forward pass through TCN policy
            
            Args:
                obs: Observations tensor
                deterministic: Whether to use deterministic action selection
            
            Returns:
                action, value, log_prob
            """
            # Extract features using TCN
            features = self.features_extractor(obs)
            
            # Policy head
            action_logits = self.action_net(features)
            
            # Value head
            value = self.value_net(features)
            
            # Create action distribution
            if isinstance(self.action_space, spaces.Discrete):
                action_dist = torch.distributions.Categorical(logits=action_logits)
            else:
                # Continuous action space (not used in trading env, but for compatibility)
                mean = action_logits
                log_std = torch.zeros_like(mean)
                action_dist = torch.distributions.Normal(mean, log_std.exp())
            
            # Sample action
            if deterministic:
                if isinstance(self.action_space, spaces.Discrete):
                    action = torch.argmax(action_logits, dim=1)
                else:
                    action = mean
            else:
                action = action_dist.sample()
            
            # Log probability
            log_prob = action_dist.log_prob(action)
            if len(log_prob.shape) > 1:
                log_prob = log_prob.sum(dim=1)
            
            return action, value.squeeze(-1), log_prob
        
        def _predict(self, observation: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
            """Predict action for given observation"""
            action, _, _ = self.forward(observation, deterministic=deterministic)
            return action
        
        def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """
            Evaluate actions for given observations
            
            Args:
                obs: Observations tensor
                actions: Actions tensor
            
            Returns:
                value, log_prob, entropy
            """
            # Extract features
            features = self.features_extractor(obs)
            
            # Value
            value = self.value_net(features)
            
            # Action logits
            action_logits = self.action_net(features)
            
            # Create action distribution
            if isinstance(self.action_space, spaces.Discrete):
                action_dist = torch.distributions.Categorical(logits=action_logits)
            else:
                mean = action_logits
                log_std = torch.zeros_like(mean)
                action_dist = torch.distributions.Normal(mean, log_std.exp())
            
            # Log probability
            log_prob = action_dist.log_prob(actions)
            if len(log_prob.shape) > 1:
                log_prob = log_prob.sum(dim=1)
            
            # Entropy
            entropy = action_dist.entropy()
            if len(entropy.shape) > 1:
                entropy = entropy.sum(dim=1)
            
            return value.squeeze(-1), log_prob, entropy
        
        def predict_values(self, obs: torch.Tensor) -> torch.Tensor:
            """
            Predict values for given observations (required by Stable-Baselines3)
            
            Args:
                obs: Observations tensor
            
            Returns:
                values: Value predictions
            """
            # Extract features
            features = self.features_extractor(obs)
            
            # Value head
            value = self.value_net(features)
            
            return value.squeeze(-1)


def register_tcn_policy():
    """Register TCN policy with Stable-Baselines3"""
    if not SB3_AVAILABLE:
        logger.warning("Stable-Baselines3 not available - cannot register TCN policy")
        return False
    
    try:
        register_policy("TCNTradingPolicy", TCNTradingPolicy)
        logger.info("✅ TCN policy registered successfully with Stable-Baselines3")
        return True
    except Exception as e:
        logger.error(f"Failed to register TCN policy: {e}")
        return False


# Auto-register on import
if SB3_AVAILABLE:
    register_tcn_policy()

