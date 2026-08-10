"""
Memory optimization functionality for the PPO Trading Agent
Extracted from TG4.1.py - MemoryOptimizer class and related utilities
"""

import os
import sys
import warnings
import logging
import pickle
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd

class MemoryOptimizer:
    """Advanced memory optimization system for trading agent"""
    
    def __init__(self, config=None):
        self.config = config or {}
        self.memory_usage = []
        self.gc_counter = 0
        self.cache_clear_counter = 0
        self.optimization_interval = 100  # Optimize every 100 steps
        self.max_memory_gb = self.config.get('max_memory_usage_gb', 20)
        self.gc_freq = self.config.get('garbage_collection_freq', 100)
        self.clear_cache_freq = self.config.get('clear_cache_freq', 500)  # OPTIMIZED: Reduced from 50 to 500
        
        # Memory thresholds
        self.warning_threshold = 0.7  # 70% of max memory
        self.critical_threshold = 0.9  # 90% of max memory
        
        # Initialize memory monitoring
        self._setup_memory_monitoring()
    
    def _setup_memory_monitoring(self):
        """Setup memory monitoring and optimization"""
        try:
            import psutil
            self.psutil = psutil
            self.total_memory = psutil.virtual_memory().total
            print(f"[MEMORY] Total RAM: {self.total_memory / (1024**3):.1f}GB")
        except ImportError:
            print("[WARNING] psutil not available for memory monitoring")
            self.psutil = None
    
    def get_memory_usage(self):
        """Get current memory usage in GB"""
        if self.psutil is None:
            return 0.0
        
        try:
            memory = self.psutil.virtual_memory()
            return memory.used / (1024**3)  # Convert to GB
        except Exception:
            return 0.0
    
    def get_memory_percentage(self):
        """Get memory usage as percentage"""
        if self.psutil is None:
            return 0.0
        
        try:
            memory = self.psutil.virtual_memory()
            return memory.percent / 100.0
        except Exception:
            return 0.0
    
    def optimize_memory(self, step_count):
        """Perform memory optimization based on current usage"""
        if step_count % self.optimization_interval != 0:
            return
        
        current_memory_gb = self.get_memory_usage()
        memory_percentage = self.get_memory_percentage()
        
        # Log memory usage
        self.memory_usage.append({
            'step': step_count,
            'memory_gb': current_memory_gb,
            'percentage': memory_percentage
        })
        
        # Perform garbage collection if needed
        if self.gc_counter % self.gc_freq == 0:
            self._perform_garbage_collection()
        
        # Clear caches if needed
        if self.cache_clear_counter % self.clear_cache_freq == 0:
            self._clear_caches()
        
        # Aggressive optimization if memory usage is high
        if memory_percentage > self.critical_threshold:
            self._aggressive_optimization()
        elif memory_percentage > self.warning_threshold:
            self._moderate_optimization()
        
        # Log memory status
        if step_count % 1000 == 0:
            print(f"[MEMORY] Step {step_count}: {current_memory_gb:.1f}GB ({memory_percentage:.1%})")
    
    def _perform_garbage_collection(self):
        """Perform garbage collection"""
        try:
            import gc
            collected = gc.collect()
            self.gc_counter += 1
            if collected > 0:
                print(f"[MEMORY] Garbage collection: {collected} objects collected")
        except Exception as e:
            print(f"[ERROR] Garbage collection failed: {e}")
    
    def _clear_caches(self):
        """Clear various caches to free memory"""
        try:
            # Clear NumPy cache
            import numpy as np
            if hasattr(np, 'clear_cache'):
                np.clear_cache()
            
            # Clear PyTorch cache if available
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            
            # Clear matplotlib cache
            try:
                import matplotlib.pyplot as plt
                plt.close('all')
            except ImportError:
                pass
            
            self.cache_clear_counter += 1
            print(f"[MEMORY] Caches cleared")
        except Exception as e:
            print(f"[ERROR] Cache clearing failed: {e}")
    
    def _moderate_optimization(self):
        """Perform moderate memory optimization"""
        try:
            # Force garbage collection
            import gc
            gc.collect()
            
            # Clear some caches
            self._clear_caches()
            
            print(f"[MEMORY] Moderate optimization performed")
        except Exception as e:
            print(f"[ERROR] Moderate optimization failed: {e}")
    
    def _aggressive_optimization(self):
        """Perform aggressive memory optimization"""
        try:
            # Force multiple garbage collections
            import gc
            for _ in range(3):
                gc.collect()
            
            # Clear all caches
            self._clear_caches()
            
            # Clear matplotlib figures
            try:
                import matplotlib.pyplot as plt
                plt.close('all')
            except ImportError:
                pass
            
            print(f"[MEMORY] Aggressive optimization performed")
        except Exception as e:
            print(f"[ERROR] Aggressive optimization failed: {e}")
    
    def get_memory_summary(self):
        """Get memory usage summary"""
        if not self.memory_usage:
            return "No memory data available"
        
        memory_values = [m['memory_gb'] for m in self.memory_usage]
        return {
            'current_gb': self.get_memory_usage(),
            'max_gb': max(memory_values),
            'avg_gb': sum(memory_values) / len(memory_values),
            'min_gb': min(memory_values),
            'total_samples': len(self.memory_usage)
        }
