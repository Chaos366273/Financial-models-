"""
Custom exceptions for PPO Trading Agent
Provides specific exception types for different error scenarios
"""

class TradingAgentException(Exception):
    """Base exception for trading agent errors"""
    pass

class DataException(TradingAgentException):
    """Exception for data-related errors"""
    pass

class ModelException(TradingAgentException):
    """Exception for model-related errors"""
    pass

class ConfigurationException(TradingAgentException):
    """Exception for configuration errors"""
    pass

class APIException(TradingAgentException):
    """Exception for API-related errors"""
    pass

class ValidationException(TradingAgentException):
    """Exception for validation errors"""
    pass

class MemoryException(TradingAgentException):
    """Exception for memory-related errors"""
    pass

class PerformanceException(TradingAgentException):
    """Exception for performance-related errors"""
    pass

class TrainingException(TradingAgentException):
    """Exception for training-related errors"""
    pass

class BacktestingException(TradingAgentException):
    """Exception for backtesting-related errors"""
    pass

class LiveTradingException(TradingAgentException):
    """Exception for live trading errors"""
    pass
