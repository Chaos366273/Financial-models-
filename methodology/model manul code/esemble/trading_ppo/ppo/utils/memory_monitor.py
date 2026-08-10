"""
Memory Monitoring System for PPO Trading
Tracks memory usage during large dataset processing
"""

import psutil
import logging
import time
from typing import Dict, List, Optional
import numpy as np

logger = logging.getLogger(__name__)

class MemoryMonitor:
    """
    Monitors system memory usage during PPO training and data processing
    """
    
    def __init__(self, warning_threshold: float = 0.8, critical_threshold: float = 0.9):
        """
        Initialize memory monitor
        
        Args:
            warning_threshold: Memory usage percentage that triggers warning (default: 80%)
            critical_threshold: Memory usage percentage that triggers critical alert (default: 90%)
        """
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.memory_history = []
        self.start_time = time.time()
        
    def get_memory_usage(self) -> Dict:
        """
        Get current memory usage statistics
        
        Returns:
            Dictionary with memory usage information
        """
        try:
            memory = psutil.virtual_memory()
            return {
                'total_gb': memory.total / (1024**3),
                'available_gb': memory.available / (1024**3),
                'used_gb': memory.used / (1024**3),
                'percent_used': memory.percent / 100,
                'free_gb': memory.free / (1024**3)
            }
        except Exception as e:
            logger.warning(f" Could not get memory usage: {e}")
            return {}
    
    def check_memory_status(self) -> str:
        """
        Check current memory status and return status level
        
        Returns:
            Status string: 'OK', 'WARNING', or 'CRITICAL'
        """
        memory_info = self.get_memory_usage()
        if not memory_info:
            return 'UNKNOWN'
        
        percent_used = memory_info['percent_used']
        
        if percent_used >= self.critical_threshold:
            return 'CRITICAL'
        elif percent_used >= self.warning_threshold:
            return 'WARNING'
        else:
            return 'OK'
    
    def log_memory_status(self, context: str = ""):
        """
        Log current memory status
        
        Args:
            context: Additional context for the log message
        """
        memory_info = self.get_memory_usage()
        if not memory_info:
            return
        
        status = self.check_memory_status()
        context_str = f" [{context}]" if context else ""
        
        if status == 'CRITICAL':
            logger.critical(f" CRITICAL MEMORY USAGE{context_str}: {memory_info['percent_used']:.1%} ({memory_info['used_gb']:.1f}GB / {memory_info['total_gb']:.1f}GB)")
        elif status == 'WARNING':
            logger.warning(f" HIGH MEMORY USAGE{context_str}: {memory_info['percent_used']:.1%} ({memory_info['used_gb']:.1f}GB / {memory_info['total_gb']:.1f}GB)")
        else:
            logger.info(f" MEMORY OK{context_str}: {memory_info['percent_used']:.1%} ({memory_info['used_gb']:.1f}GB / {memory_info['total_gb']:.1f}GB)")
        
        # Store in history
        self.memory_history.append({
            'timestamp': time.time(),
            'memory_info': memory_info,
            'status': status,
            'context': context
        })
    
    def estimate_memory_for_dataset(self, num_records: int, num_tickers: int, num_features: int) -> Dict:
        """
        Estimate memory requirements for dataset processing
        
        Args:
            num_records: Number of data records
            num_tickers: Number of tickers
            num_features: Number of features per ticker
            
        Returns:
            Dictionary with memory estimates
        """
        # Estimate memory for different data structures
        float32_size = 4  # bytes
        float64_size = 8  # bytes
        
        # DataFrame memory (assuming float64)
        df_memory_mb = (num_records * (num_features + 5) * float64_size) / (1024**2)
        
        # Matrix memory (float32 for efficiency)
        feature_matrix_mb = (num_records * num_tickers * num_features * float32_size) / (1024**2)
        price_matrix_mb = (num_records * num_tickers * float32_size) / (1024**2)
        volume_matrix_mb = (num_records * num_tickers * float32_size) / (1024**2)
        
        total_estimated_mb = df_memory_mb + feature_matrix_mb + price_matrix_mb + volume_matrix_mb
        total_estimated_gb = total_estimated_mb / 1024
        
        return {
            'dataframe_mb': df_memory_mb,
            'feature_matrix_mb': feature_matrix_mb,
            'price_matrix_mb': price_matrix_mb,
            'volume_matrix_mb': volume_matrix_mb,
            'total_estimated_mb': total_estimated_mb,
            'total_estimated_gb': total_estimated_gb
        }
    
    def check_dataset_memory_requirements(self, num_records: int, num_tickers: int, num_features: int) -> bool:
        """
        Check if system has enough memory for dataset processing
        
        Args:
            num_records: Number of data records
            num_tickers: Number of tickers
            num_features: Number of features per ticker
            
        Returns:
            True if enough memory available, False otherwise
        """
        memory_info = self.get_memory_usage()
        if not memory_info:
            return True  # Assume OK if can't check
        
        estimates = self.estimate_memory_for_dataset(num_records, num_tickers, num_features)
        available_gb = memory_info['available_gb']
        required_gb = estimates['total_estimated_gb']
        
        # Require at least 2x the estimated memory to be available
        safety_factor = 2.0
        required_with_safety = required_gb * safety_factor
        
        if available_gb < required_with_safety:
            logger.warning(f" INSUFFICIENT MEMORY: Need {required_with_safety:.1f}GB, Available: {available_gb:.1f}GB")
            logger.warning(f"   Dataset: {num_records:,} records  {num_tickers} tickers  {num_features} features")
            return False
        else:
            logger.info(f" SUFFICIENT MEMORY: Need {required_with_safety:.1f}GB, Available: {available_gb:.1f}GB")
            return True
    
    def get_memory_summary(self) -> Dict:
        """
        Get memory usage summary
        
        Returns:
            Dictionary with memory summary statistics
        """
        if not self.memory_history:
            return {}
        
        memory_percentages = [entry['memory_info']['percent_used'] for entry in self.memory_history]
        
        return {
            'total_checks': len(self.memory_history),
            'avg_memory_usage': np.mean(memory_percentages),
            'max_memory_usage': np.max(memory_percentages),
            'min_memory_usage': np.min(memory_percentages),
            'critical_count': sum(1 for entry in self.memory_history if entry['status'] == 'CRITICAL'),
            'warning_count': sum(1 for entry in self.memory_history if entry['status'] == 'WARNING'),
            'runtime_seconds': time.time() - self.start_time
        }
