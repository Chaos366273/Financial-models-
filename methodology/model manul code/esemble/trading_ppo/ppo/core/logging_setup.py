"""
Logging setup for PPO Trading Agent
Handles console and file logging with performance optimizations
"""

import logging
import os
import sys
from datetime import datetime
from typing import Optional

def setup_logging(
    console_level: str = 'WARNING',
    file_level: str = 'INFO',
    log_format: str = '%(asctime)s - %(levelname)s - %(message)s',
    max_bytes: int = 5 * 1024 * 1024,  # 5MB
    backup_count: int = 2,
    disable_file_logging: bool = True,
    use_buffered_handler: bool = True,
    flush_interval: int = 10000,
    disable_console_logging: bool = False
) -> logging.Logger:
    """
    Setup logging configuration for the trading agent
    
    Args:
        console_level: Console log level
        file_level: File log level  
        log_format: Log message format
        max_bytes: Max file size before rotation
        backup_count: Number of backup files
        disable_file_logging: Disable file logging for performance
        use_buffered_handler: Use buffered handler for faster I/O
        flush_interval: Flush logs every N steps
        disable_console_logging: Disable console logging
        
    Returns:
        Configured logger instance
    """
    
    # Create logger
    logger = logging.getLogger('trading_agent')
    logger.setLevel(logging.DEBUG)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    if not disable_console_logging:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, console_level.upper()))
        console_formatter = logging.Formatter(log_format)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    # File handler (optional for performance)
    if not disable_file_logging:
        # Ensure logs directory exists under project root
        os.makedirs('ppo/logs', exist_ok=True)
        
        # Create rotating file handler
        from logging.handlers import RotatingFileHandler
        file_handler = RotatingFileHandler(
            'ppo/logs/trading_agent.log',
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        file_handler.setLevel(getattr(logging, file_level.upper()))
        file_formatter = logging.Formatter(log_format)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    # Performance optimization: Disable propagation to root logger
    logger.propagate = False
    
    return logger

def get_logger(name: str = 'trading_agent') -> logging.Logger:
    """
    Get a logger instance
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)

def log_performance_metrics(logger: logging.Logger, step: int, metrics: dict):
    """
    Log performance metrics at specified intervals
    
    Args:
        logger: Logger instance
        step: Current training step
        metrics: Dictionary of metrics to log
    """
    if step % 5000 == 0:  # Log every 5000 steps
        logger.info(f"Step {step}: {metrics}")

def log_memory_usage(logger: logging.Logger, memory_usage_mb: float):
    """
    Log memory usage with warnings
    
    Args:
        logger: Logger instance
        memory_usage_mb: Memory usage in MB
    """
    if memory_usage_mb > 3000:  # 3GB threshold
        logger.warning(f"High memory usage: {memory_usage_mb:.1f} MB")
    elif memory_usage_mb > 4000:  # 4GB threshold
        logger.error(f"Critical memory usage: {memory_usage_mb:.1f} MB")
