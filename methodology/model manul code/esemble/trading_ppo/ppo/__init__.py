"""
PPO Trading System Package
Modular implementation of TG4.1 trading system
"""

__version__ = "1.0.0"
__author__ = "Trading System"

# Import key components for easy access
try:
    from .core.logging_setup import setup_logging, get_logger
except ImportError:
    # Fallback if core modules not available
    import logging

    def setup_logging(*args, **kwargs) -> logging.Logger:
        return logging.getLogger(__name__)
    def get_logger(*args, **kwargs):
        return logging.getLogger(__name__)

# Import exceptions first (they have no dependencies)
try:
    from .core.exceptions import TradingAgentException, DataException, ModelException  # type: ignore[assignment]
except ImportError as e:
    print(f"Warning: Core exceptions not available: {e}")
    # Create fallback exception classes
    class TradingAgentException(Exception): pass
    class DataException(Exception): pass
    class ModelException(Exception): pass

# Import other components with proper error handling
try:
    from .models.trading_env import FixedTradingEnv
except ImportError as e:
    print(f"Warning: Trading environment not available: {e}")
    FixedTradingEnv = None

try:
    from .data.alpaca_client import AlpacaDataClient
except ImportError as e:
    print(f"Warning: Alpaca client not available: {e}")
    AlpacaDataClient = None

try:
    from .models.tcn_policy import TCNTradingPolicy, register_tcn_policy
except ImportError as e:
    print(f"Warning: TCN policy not available: {e}")
    TCNTradingPolicy = None
    register_tcn_policy = None

try:
    from .training.ppo_agent import PPOAgent
except ImportError as e:
    print(f"Warning: PPO agent not available: {e}")
    PPOAgent = None

try:
    from . import constants
except ImportError as e:
    print(f"Warning: Constants not available: {e}")
    constants = None

__all__ = [
    'setup_logging',
    'get_logger', 
    'TradingAgentException',
    'DataException',
    'ModelException',
    'AlpacaDataClient',
    'FixedTradingEnv',
    'TCNTradingPolicy',
    'register_tcn_policy',
    'PPOAgent'
]
