"""
Logging utilities for PPO Trading Agent
Centralized logging configuration and utilities
"""

import os
import logging
from pathlib import Path
from datetime import datetime

def setup_logging(log_level=logging.WARNING, log_file='ppo/logs/ppo_trading.log'):
    """
    Setup logging configuration with Unicode support
    
    Args:
        log_level: Logging level (default: WARNING to prevent massive log files)
        log_file: Path to log file
    """
    # Create logs directory if it doesn't exist
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure logging with Unicode support
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, encoding='utf-8')
        ]
    )
    
    return logging.getLogger(__name__)

def get_logger(name, log_level=None):
    """
    Get a logger with optional custom log level
    
    Args:
        name: Logger name
        log_level: Optional custom log level
    """
    logger = logging.getLogger(name)
    if log_level is not None:
        logger.setLevel(log_level)
    return logger

def log_performance_metrics(logger, metrics, stage):
    """
    Log performance metrics for a specific stage
    
    Args:
        logger: Logger instance
        metrics: Performance metrics dictionary
        stage: Stage name
    """
    if stage in metrics:
        duration = metrics[stage].get('duration', 0)
        logger.info(f"{stage.replace('_', ' ').title()} completed in {duration:.2f} seconds")
    else:
        logger.warning(f"No metrics found for stage: {stage}")

def log_training_progress(logger, step, total_steps, reward, loss=None):
    """
    Log training progress with key metrics
    
    Args:
        logger: Logger instance
        step: Current training step
        total_steps: Total training steps
        reward: Current reward
        loss: Optional loss value
    """
    progress_pct = (step / total_steps) * 100 if total_steps > 0 else 0
    log_msg = f"Training progress: {step}/{total_steps} ({progress_pct:.1f}%) - Reward: {reward:.4f}"
    
    if loss is not None:
        log_msg += f" - Loss: {loss:.4f}"
    
    logger.info(log_msg)

def log_backtest_results(logger, results):
    """
    Log comprehensive backtest results
    
    Args:
        logger: Logger instance
        results: Backtest results dictionary
    """
    logger.info(" BACKTEST RESULTS:")
    logger.info(f"   Total Return: {results.get('total_return_pct', 0):.2f}%")
    logger.info(f"   Sharpe Ratio: {results.get('sharpe_ratio', 0):.3f}")
    logger.info(f"   Max Drawdown: {results.get('max_drawdown_pct', 0):.2f}%")
    logger.info(f"   Win Rate: {results.get('win_rate_pct', 0):.1f}%")
    logger.info(f"   Total Trades: {results.get('num_trades', 0)}")
    logger.info(f"   Steps Completed: {results.get('steps', 0)}")

def log_risk_management_status(logger, risk_status):
    """
    Log hard risk management status
    
    Args:
        logger: Logger instance
        risk_status: Risk management status dictionary
    """
    if risk_status:
        logger.info(" RISK MANAGEMENT STATUS:")
        logger.info(f"   Trading Halted: {risk_status.get('trading_halted', False)}")
        logger.info(f"   Risk Violations: {risk_status.get('risk_violations', 0)}")
        logger.info(f"   Stop Loss K: {risk_status.get('stop_loss_k', 2.0)}")
        logger.info(f"   Volatility: {risk_status.get('rolling_volatility', 0)*100:.2f}%")
        
        if risk_status.get('last_violation'):
            last_violation = risk_status['last_violation']
            logger.warning(f"   Last Violation: {last_violation.get('type', 'Unknown')}")

def log_statistical_analysis(logger, stats):
    """
    Log statistical analysis results
    
    Args:
        logger: Logger instance
        stats: Statistical analysis results dictionary
    """
    if stats:
        logger.info(" STATISTICAL ANALYSIS:")
        logger.info(f"   T-Test p-value: {stats.get('t_test_p_value', 'N/A')}")
        logger.info(f"   Bootstrap p-value: {stats.get('bootstrap_p_value', 'N/A')}")
        logger.info(f"   Statistically Significant: {stats.get('statistically_significant', 'N/A')}")
        logger.info(f"   Paper Trading Ready: {stats.get('paper_trading_ready', 'N/A')}")
        logger.info(f"   Data Points: {stats.get('data_points', 'N/A')}")
