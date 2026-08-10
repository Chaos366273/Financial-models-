#!/usr/bin/env python3
"""
Comprehensive Performance Profiling Module for BC Training
Tracks timing, memory usage, GPU utilization, and identifies bottlenecks
"""

import time
import psutil
import json
import os
from pathlib import Path
from functools import wraps
from typing import Dict, List, Any, Optional
import threading
import gc

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import GPUtil  # type: ignore
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

class PerformanceProfiler:
    """Comprehensive performance profiler for BC training"""
    
    def __init__(self, output_dir: str = "ppo/profiling"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Profiling data storage
        self.timings: Dict[str, List[float]] = {}
        self.memory_usage: Dict[str, List[float]] = {}
        self.gpu_usage: Dict[str, List[float]] = {}
        self.function_calls: Dict[str, int] = {}
        self.bottlenecks: List[Dict[str, Any]] = []
        
        # ENHANCED: Track actual execution time
        self.start_time = time.time()
        self.end_time = None
        self.total_execution_time = None
        
        # Thread-safe lock
        self._lock = threading.Lock()
        
        # Start monitoring
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_system, daemon=True)
        self._monitor_thread.start()
        
        print(f"📊 Performance Profiler initialized")
        print(f"   Output directory: {self.output_dir}")
        print(f"   GPU monitoring: {'✅ Enabled' if GPU_AVAILABLE else '❌ Disabled'}")
        print(f"   PyTorch monitoring: {'✅ Enabled' if TORCH_AVAILABLE else '❌ Disabled'}")
        print(f"   🕐 Started at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.start_time))}")
    
    def stop_profiling(self):
        """Stop profiling and calculate total execution time"""
        self.end_time = time.time()
        self.total_execution_time = self.end_time - self.start_time
        self._monitoring = False
        
        print(f"🕐 Profiling stopped at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.end_time))}")
        print(f"⏱️  TOTAL EXECUTION TIME: {self.total_execution_time:.2f} seconds ({self.total_execution_time/60:.2f} minutes)")
        
        return self.total_execution_time
    
    def get_execution_time(self):
        """Get current execution time"""
        if self.end_time:
            return self.total_execution_time
        else:
            return time.time() - self.start_time
    
    def _monitor_system(self):
        """Background monitoring thread"""
        while self._monitoring:
            try:
                # Monitor system resources
                cpu_percent = psutil.cpu_percent()
                memory = psutil.virtual_memory()
                
                # Monitor GPU if available
                if GPU_AVAILABLE:
                    try:
                        gpus = GPUtil.getGPUs()
                        if gpus:
                            gpu = gpus[0]  # Use first GPU
                            gpu_util = gpu.load * 100
                            gpu_memory = gpu.memoryUtil * 100
                        else:
                            gpu_util = 0
                            gpu_memory = 0
                    except:
                        gpu_util = 0
                        gpu_memory = 0
                else:
                    gpu_util = 0
                    gpu_memory = 0
                
                # Store system metrics
                with self._lock:
                    if 'system_cpu' not in self.timings:
                        self.timings['system_cpu'] = []
                        self.memory_usage['system_memory'] = []
                        self.gpu_usage['system_gpu'] = []
                    
                    self.timings['system_cpu'].append(cpu_percent)
                    self.memory_usage['system_memory'].append(memory.percent)
                    self.gpu_usage['system_gpu'].append(gpu_util)
                
                time.sleep(1.0)  # Monitor every second
                
            except Exception as e:
                print(f"⚠️ Profiler monitoring error: {e}")
                time.sleep(5.0)
    
    def track_time(self, function_name: str, duration: float):
        """Track function execution time"""
        with self._lock:
            if function_name not in self.timings:
                self.timings[function_name] = []
            self.timings[function_name].append(duration)
    
    def track_memory(self, function_name: str, memory_mb: float):
        """Track memory usage for a function"""
        with self._lock:
            if function_name not in self.memory_usage:
                self.memory_usage[function_name] = []
            self.memory_usage[function_name].append(memory_mb)
    
    def track_gpu(self, function_name: str, gpu_percent: float):
        """Track GPU utilization for a function"""
        with self._lock:
            if function_name not in self.gpu_usage:
                self.gpu_usage[function_name] = []
            self.gpu_usage[function_name].append(gpu_percent)
    
    def track_call(self, function_name: str):
        """Track function call count"""
        with self._lock:
            if function_name not in self.function_calls:
                self.function_calls[function_name] = 0
            self.function_calls[function_name] += 1
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024  # Convert to MB
    
    def get_gpu_usage(self) -> float:
        """Get current GPU usage percentage"""
        if not GPU_AVAILABLE:
            return 0.0
        
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                return gpus[0].load * 100
            return 0.0
        except:
            return 0.0
    
    def identify_bottlenecks(self) -> List[Dict[str, Any]]:
        """Identify performance bottlenecks"""
        bottlenecks = []
        
        with self._lock:
            # Analyze timing data
            for func_name, times in self.timings.items():
                if len(times) > 0:
                    avg_time = sum(times) / len(times)
                    total_time = sum(times)
                    max_time = max(times)
                    call_count = self.function_calls.get(func_name, len(times))
                    
                    bottlenecks.append({
                        'function': func_name,
                        'type': 'timing',
                        'avg_time': avg_time,
                        'total_time': total_time,
                        'max_time': max_time,
                        'call_count': call_count,
                        'efficiency_score': call_count / total_time if total_time > 0 else 0
                    })
            
            # Analyze memory data
            for func_name, memory_data in self.memory_usage.items():
                if len(memory_data) > 0:
                    avg_memory = sum(memory_data) / len(memory_data)
                    max_memory = max(memory_data)
                    
                    bottlenecks.append({
                        'function': func_name,
                        'type': 'memory',
                        'avg_memory_mb': avg_memory,
                        'max_memory_mb': max_memory,
                        'memory_efficiency': 1.0 / (avg_memory + 1)  # Higher is better
                    })
        
        # Sort by impact (combination of time and memory)
        bottlenecks.sort(key=lambda x: x.get('total_time', 0) * x.get('avg_memory_mb', 1), reverse=True)
        
        self.bottlenecks = bottlenecks[:10]  # Top 10 bottlenecks
        return self.bottlenecks
    
    def generate_report(self) -> str:
        """Generate comprehensive profiling report"""
        report_path = self.output_dir / "bc_profiling_report.json"
        
        # Identify bottlenecks
        bottlenecks = self.identify_bottlenecks()
        
        # Calculate summary statistics
        total_functions = len(self.timings)
        total_calls = sum(self.function_calls.values())
        total_time = sum(sum(times) for times in self.timings.values())
        
        # Memory statistics
        memory_stats = {}
        for func_name, memory_data in self.memory_usage.items():
            if memory_data:
                memory_stats[func_name] = {
                    'avg_mb': sum(memory_data) / len(memory_data),
                    'max_mb': max(memory_data),
                    'min_mb': min(memory_data)
                }
        
        # GPU statistics
        gpu_stats = {}
        for func_name, gpu_data in self.gpu_usage.items():
            if gpu_data:
                gpu_stats[func_name] = {
                    'avg_percent': sum(gpu_data) / len(gpu_data),
                    'max_percent': max(gpu_data),
                    'min_percent': min(gpu_data)
                }
        
        # ENHANCED: Add actual execution time to report
        actual_execution_time = self.get_execution_time() or 0.0
        
        # Create comprehensive report
        report = {
            'summary': {
                'total_functions_profiled': total_functions,
                'total_function_calls': total_calls,
                'total_execution_time': total_time,
                'actual_execution_time_seconds': actual_execution_time,
            'actual_execution_time_minutes': actual_execution_time / 60,
            'actual_execution_time_hours': actual_execution_time / 3600,
                'profiling_duration': time.time() - getattr(self, '_start_time', time.time()),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'bottlenecks': bottlenecks,
            'timing_analysis': {
                func_name: {
                    'avg_time': sum(times) / len(times) if times else 0,
                    'total_time': sum(times),
                    'max_time': max(times) if times else 0,
                    'min_time': min(times) if times else 0,
                    'call_count': self.function_calls.get(func_name, 0)
                }
                for func_name, times in self.timings.items()
            },
            'memory_analysis': memory_stats,
            'gpu_analysis': gpu_stats,
            'recommendations': self._generate_recommendations(bottlenecks)
        }
        
        # Save report
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📊 Profiling report saved to: {report_path}")
        print(f"🔗 View profiling stats at: {report_path}")
        
        return str(report_path)
    
    def _generate_recommendations(self, bottlenecks: List[Dict[str, Any]]) -> List[str]:
        """Generate optimization recommendations based on bottlenecks"""
        recommendations = []
        
        # Analyze timing bottlenecks
        timing_bottlenecks = [b for b in bottlenecks if b['type'] == 'timing']
        if timing_bottlenecks:
            worst_timing = timing_bottlenecks[0]
            if worst_timing['avg_time'] > 1.0:  # More than 1 second average
                recommendations.append(f"🚨 CRITICAL: {worst_timing['function']} takes {worst_timing['avg_time']:.2f}s average - consider optimization")
            elif worst_timing['avg_time'] > 0.5:  # More than 0.5 seconds
                recommendations.append(f"⚠️ SLOW: {worst_timing['function']} takes {worst_timing['avg_time']:.2f}s average - consider caching or parallelization")
        
        # Analyze memory bottlenecks
        memory_bottlenecks = [b for b in bottlenecks if b['type'] == 'memory']
        if memory_bottlenecks:
            worst_memory = memory_bottlenecks[0]
            if worst_memory['avg_memory_mb'] > 1000:  # More than 1GB
                recommendations.append(f"🚨 HIGH MEMORY: {worst_memory['function']} uses {worst_memory['avg_memory_mb']:.1f}MB average - consider memory optimization")
            elif worst_memory['avg_memory_mb'] > 500:  # More than 500MB
                recommendations.append(f"⚠️ MEMORY: {worst_memory['function']} uses {worst_memory['avg_memory_mb']:.1f}MB average - consider lazy loading")
        
        # General recommendations
        if len(timing_bottlenecks) > 5:
            recommendations.append("💡 Consider parallel processing for multiple slow functions")
        
        if any(b['call_count'] > 1000 for b in bottlenecks):
            recommendations.append("💡 High call count functions should be optimized or cached")
        
        return recommendations
    
    def stop_monitoring(self):
        """Stop background monitoring"""
        self._monitoring = False
        if self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=2.0)

# Global profiler instance
_profiler = None

def get_profiler() -> PerformanceProfiler:
    """Get global profiler instance"""
    global _profiler
    if _profiler is None:
        _profiler = PerformanceProfiler()
    return _profiler

def profile_function(name: Optional[str] = None, track_memory: bool = True, track_gpu: bool = True):
    """Decorator to profile function execution"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            profiler = get_profiler()
            func_name = name or f"{func.__module__}.{func.__name__}"
            
            # Track function call
            profiler.track_call(func_name)
            
            # Get initial state
            start_time = time.time()
            start_memory = profiler.get_memory_usage() if track_memory else 0
            start_gpu = profiler.get_gpu_usage() if track_gpu else 0
            
            try:
                # Execute function
                result = func(*args, **kwargs)
                
                # Calculate metrics
                end_time = time.time()
                duration = end_time - start_time
                
                # Track metrics
                profiler.track_time(func_name, duration)
                
                if track_memory:
                    end_memory = profiler.get_memory_usage()
                    memory_used = end_memory - start_memory
                    profiler.track_memory(func_name, memory_used)
                
                if track_gpu:
                    end_gpu = profiler.get_gpu_usage()
                    gpu_used = end_gpu - start_gpu
                    profiler.track_gpu(func_name, gpu_used)
                
                return result
                
            except Exception as e:
                # Still track timing even if function fails
                end_time = time.time()
                duration = end_time - start_time
                profiler.track_time(func_name, duration)
                raise e
        
        return wrapper
    return decorator

def generate_profiling_report() -> str:
    """Generate and save profiling report"""
    profiler = get_profiler()
    return profiler.generate_report()

def cleanup_profiler():
    """Cleanup profiler resources"""
    global _profiler
    if _profiler:
        _profiler.stop_monitoring()
        _profiler = None
