"""
Advanced Performance Monitoring System
Comprehensive profiling and execution time tracking
"""

import time
import psutil
import threading
import io
import cProfile
import pstats
import numpy as np
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class PerformanceConfig:
    """Performance monitoring configuration"""
    enable_profiling: bool = True
    enable_resource_monitoring: bool = True
    profiling_interval: int = 10000  # Steps between profiling reports
    resource_monitoring_interval: int = 5  # Seconds between resource checks
    memory_threshold_gb: float = 20.0  # Memory threshold for warnings
    cpu_threshold_percent: float = 90.0  # CPU threshold for warnings

class PerformanceMonitor:
    """
    Advanced performance monitoring with cProfile integration
    Matches TG4.1's comprehensive profiling capabilities
    """
    
    def __init__(self, config: Optional[PerformanceConfig] = None):
        self.config = config or PerformanceConfig()
        
        # Timing data
        self.start_times = {}
        self.execution_times = {}
        
        # Resource monitoring
        self.memory_usage = []
        self.cpu_usage = []
        self.monitoring = False
        self.monitor_thread = None
        
        # Profiling
        self.profiler = None
        self.profile_stats = None
        self.profiling_active = False
        
        # Performance tracking
        self.performance_history = []
        self.bottlenecks = []
        
        # Enhanced monitoring features
        self.alert_system = None
        self.real_time_monitor = None
        self.alert_history = []
        
        logger.info("PerformanceMonitor initialized")
    
    def start_timer(self, name: str):
        """Start timing an operation"""
        self.start_times[name] = time.time()
    
    def end_timer(self, name: str) -> float:
        """End timing an operation and record execution time"""
        if name in self.start_times:
            execution_time = time.time() - self.start_times[name]
            if name not in self.execution_times:
                self.execution_times[name] = []
            self.execution_times[name].append(execution_time)
            del self.start_times[name]
            return execution_time
        return 0.0
    
    def get_average_time(self, name: str) -> float:
        """Get average execution time for an operation"""
        if name in self.execution_times and self.execution_times[name]:
            return sum(self.execution_times[name]) / len(self.execution_times[name])
        return 0.0
    
    def start_monitoring(self):
        """Start system resource monitoring"""
        if not self.monitoring:
            self.monitoring = True
            self.monitor_thread = threading.Thread(target=self._monitor_resources)
            self.monitor_thread.daemon = True
            self.monitor_thread.start()
            logger.info("Started resource monitoring")
    
    def stop_monitoring(self):
        """Stop system resource monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1)
        logger.info("Stopped resource monitoring")
    
    def start_profiling(self):
        """Start cProfile profiling"""
        if self.config.enable_profiling and not self.profiling_active:
            self.profiler = cProfile.Profile()
            self.profiler.enable()
            self.profiling_active = True
            logger.info("Started cProfile profiling")
    
    def stop_profiling(self):
        """Stop cProfile profiling and print results"""
        if self.profiler and self.profiling_active:
            self.profiler.disable()
            self.profiling_active = False

            # Build stats to an in-memory buffer first
            cumulative_buffer = io.StringIO()
            ps = pstats.Stats(self.profiler, stream=cumulative_buffer).sort_stats('cumulative')

            # Prepare separate buffer for time-per-call view
            time_buffer = io.StringIO()
            ps_time = pstats.Stats(self.profiler, stream=time_buffer).sort_stats('time')

            # Render selected summaries into buffers
            ps.print_stats(20)
            ps_time.print_stats(10)

            # Print bounded summaries to console
            print("\n" + "="*60)
            print(" PROFILER: TOP 20 FUNCTIONS BY CUMULATIVE TIME")
            print("="*60)
            print(cumulative_buffer.getvalue())

            print("\n" + "="*60)
            print(" PROFILER: TOP 10 FUNCTIONS BY TIME PER CALL")
            print("="*60)
            print(time_buffer.getvalue())

            # Save detailed stats to file
            timestamp = int(time.time())
            stats_file = f"profiling_stats_{timestamp}.txt"
            with open(stats_file, 'w') as f:
                # Reuse cumulative view for file output (full stats)
                ps = pstats.Stats(self.profiler, stream=f).sort_stats('cumulative')
                ps.print_stats()
            print(f" PROFILER: Detailed stats saved to: {stats_file}")

            # Store stats for later analysis
            self.profile_stats = ps
            self.profiler = None

            print("="*60)
            logger.info("Stopped cProfile profiling")
    
    def get_profiling_summary(self) -> Optional[Dict[str, Any]]:
        """Get a quick profiling summary"""
        if self.profile_stats:
            return {
                'total_calls': getattr(self.profile_stats, 'total_calls', 0),
                'total_time': getattr(self.profile_stats, 'total_tt', 0.0),
                'functions_profiled': len(getattr(self.profile_stats, 'stats', {}))
            }
        return None
    
    def get_execution_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get detailed execution time breakdown"""
        summary = {}
        for name, times in self.execution_times.items():
            if times:
                summary[name] = {
                    'total_time': sum(times),
                    'avg_time': sum(times) / len(times),
                    'min_time': min(times),
                    'max_time': max(times),
                    'count': len(times),
                    'percentage': 0  # Will calculate below
                }
        
        # Calculate percentages
        total_time = sum(data['total_time'] for data in summary.values())
        for name in summary:
            if total_time > 0:
                summary[name]['percentage'] = (summary[name]['total_time'] / total_time * 100)
        
        return summary
    
    def print_execution_summary(self):
        """Print detailed execution time breakdown"""
        summary = self.get_execution_summary()
        if not summary:
            print("  No timing data available")
            return
        
        print("\n" + "="*60)
        print("  DETAILED EXECUTION TIME BREAKDOWN")
        print("="*60)
        
        # Sort by total time (descending)
        sorted_summary = sorted(summary.items(), key=lambda x: x[1]['total_time'], reverse=True)
        
        total_time = sum(data['total_time'] for data in summary.values())
        
        for name, data in sorted_summary:
            print(f" {name}:")
            print(f"     Total: {data['total_time']:.2f}s ({data['percentage']:.1f}%)")
            print(f"    Avg: {data['avg_time']:.3f}s")
            print(f"    Count: {data['count']}")
            print(f"    Range: {data['min_time']:.3f}s - {data['max_time']:.3f}s")
        
        print(f"\n TOTAL EXECUTION TIME: {total_time:.2f}s")
        print("="*60)
    
    def _monitor_resources(self):
        """Monitor system resources in background with hardware-specific tracking"""
        while self.monitoring:
            try:
                memory = psutil.virtual_memory()
                cpu = psutil.cpu_percent(interval=1)
                cpu_count = psutil.cpu_count()
                cpu_freq = psutil.cpu_freq()
                
                # Enhanced hardware-specific monitoring for i5-12500H
                self.memory_usage.append({
                    'timestamp': time.time(),
                    'percent': memory.percent,
                    'used_gb': memory.used / (1024**3),
                    'available_gb': memory.available / (1024**3),
                    'total_gb': memory.total / (1024**3)
                })
                
                self.cpu_usage.append({
                    'timestamp': time.time(),
                    'percent': cpu,
                    'cores': cpu_count,
                    'freq_mhz': cpu_freq.current if cpu_freq else 0
                })
                
                # Check for resource warnings
                if memory.percent > self.config.cpu_threshold_percent:
                    logger.warning(f"  High memory usage: {memory.percent:.1f}%")
                
                if cpu > self.config.cpu_threshold_percent:
                    logger.warning(f"  High CPU usage: {cpu:.1f}%")
                
                # Log hardware utilization every 30 seconds
                if len(self.cpu_usage) % 30 == 0:
                    avg_cpu = np.mean([u['percent'] for u in self.cpu_usage[-30:]])
                    avg_memory_gb = np.mean([u['used_gb'] for u in self.memory_usage[-30:]])
                    # Reduce console chatter in production; prefer logger at DEBUG
                    logger.debug(f"HARDWARE: CPU={avg_cpu:.1f}%, RAM={avg_memory_gb:.1f}GB/{memory.total/(1024**3):.1f}GB")
                
                time.sleep(self.config.resource_monitoring_interval)
            except Exception as e:
                logger.error(f"Error monitoring resources: {e}")
                break
    
    def get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB"""
        try:
            memory = psutil.virtual_memory()
            return memory.used / (1024**2)  # Convert to MB
        except Exception as e:
            logger.error(f"Error getting memory usage: {e}")
            return 0.0
    
    def get_cpu_usage_percent(self) -> float:
        """Get current CPU usage percentage"""
        try:
            return psutil.cpu_percent()
        except Exception as e:
            logger.error(f"Error getting CPU usage: {e}")
            return 0.0
    
    def get_resource_warnings(self) -> List[str]:
        """Get resource usage warnings"""
        warnings = []
        try:
            memory = psutil.virtual_memory()
            cpu = psutil.cpu_percent()
            
            if memory.percent > self.config.memory_threshold_gb:
                warnings.append(f"High memory usage: {memory.percent:.1f}%")
            
            if cpu > self.config.cpu_threshold_percent:
                warnings.append(f"High CPU usage: {cpu:.1f}%")
                
        except Exception as e:
            logger.error(f"Error getting resource warnings: {e}")
        
        return warnings
    
    def get_resource_summary(self) -> Dict[str, Any]:
        """Get current resource usage summary"""
        try:
            memory = psutil.virtual_memory()
            cpu = psutil.cpu_percent()
            
            return {
                'memory': {
                    'percent': memory.percent,
                    'used_gb': memory.used / (1024**3),
                    'available_gb': memory.available / (1024**3),
                    'total_gb': memory.total / (1024**3)
                },
                'cpu': {
                    'percent': cpu,
                    'cores': psutil.cpu_count(),
                    'freq_mhz': psutil.cpu_freq().current if psutil.cpu_freq() else 0
                }
            }
        except Exception as e:
            logger.error(f"Error getting resource summary: {e}")
            return {}
    
    def identify_bottlenecks(self) -> List[str]:
        """Identify performance bottlenecks"""
        bottlenecks = []
        summary = self.get_execution_summary()
        
        if summary:
            # Find operations taking more than 10% of total time
            total_time = sum(data['total_time'] for data in summary.values())
            for name, data in summary.items():
                if data['percentage'] > 10.0:
                    bottlenecks.append(f"{name}: {data['percentage']:.1f}% of total time")
                
                # Check for slow operations
                if data['avg_time'] > 0.1:  # More than 100ms average
                    bottlenecks.append(f"{name}: slow average time ({data['avg_time']:.3f}s)")
        
        return bottlenecks
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        report = {
            'execution_summary': self.get_execution_summary(),
            'resource_summary': self.get_resource_summary(),
            'bottlenecks': self.identify_bottlenecks(),
            'profiling_summary': self.get_profiling_summary()
        }
        
        return report
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get all performance metrics"""
        return {
            'execution_times': self.execution_times,
            'average_times': {name: self.get_average_time(name) for name in self.execution_times},
            'memory_usage': self.memory_usage[-10:] if self.memory_usage else [],
            'cpu_usage': self.cpu_usage[-10:] if self.cpu_usage else [],
            'bottlenecks': self.bottlenecks,
            'profile_stats': self.profile_stats
        }
    
    def track_memory_usage(self):
        """Track current memory usage"""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            self.memory_usage.append(memory_mb)
            
            # Keep only last 100 measurements
            if len(self.memory_usage) > 100:
                self.memory_usage = self.memory_usage[-100:]
                
        except Exception as e:
            logger.warning(f"Failed to track memory usage: {e}")
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            return memory_info.rss / 1024 / 1024
        except Exception as e:
            logger.warning(f"Failed to get memory usage: {e}")
            return 0.0
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary - required for test compatibility"""
        try:
            summary = {
                'total_execution_time': sum(sum(times) for times in self.execution_times.values()) if self.execution_times else 0.0,
                'operation_count': len(self.execution_times),
                'memory_mb': self.get_memory_usage_mb(),
                'cpu_percent': self.get_cpu_usage_percent(),
                'profiling_active': self.profiling_active,
                'monitoring_active': self.monitoring,
                'bottlenecks': self.identify_bottlenecks(),
                'resource_warnings': self.get_resource_warnings()
            }
            
            # Add average times for each operation
            for name, times in self.execution_times.items():
                if times:
                    summary[f'{name}_avg'] = sum(times) / len(times)
                    summary[f'{name}_total'] = sum(times)
                    summary[f'{name}_count'] = len(times)
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting summary: {e}")
            return {
                'error': str(e),
                'total_execution_time': 0.0,
                'operation_count': 0,
                'memory_mb': 0.0,
                'cpu_percent': 0.0
            }
    
    def cleanup(self):
        """Cleanup resources"""
        self.stop_monitoring()
        if self.profiling_active:
            self.stop_profiling()
        logger.info("PerformanceMonitor cleanup completed")


class RealTimeMonitor:
    """Real-time monitoring for live trading"""
    
    def __init__(self):
        self.start_time = None
        self.monitoring_active = False
        self.portfolio_history = []
        self.trade_history = []
        self.performance_metrics = {}
        
    def start_monitoring(self):
        """Start real-time monitoring"""
        self.start_time = time.time()
        self.monitoring_active = True
        logger.info("Real-time monitoring started")
        
    def stop_monitoring(self):
        """Stop real-time monitoring"""
        self.monitoring_active = False
        logger.info("Real-time monitoring stopped")
        
    def update_portfolio(self, portfolio_value: float):
        """Update portfolio value"""
        if self.monitoring_active:
            self.portfolio_history.append({
                'timestamp': time.time(),
                'portfolio_value': portfolio_value
            })
            
    def update_trade(self, trade_data: Dict[str, Any]):
        """Update trade information"""
        if self.monitoring_active:
            self.trade_history.append({
                'timestamp': time.time(),
                **trade_data
            })
            
    def get_monitoring_duration(self) -> float:
        """Get monitoring duration in seconds"""
        if self.start_time:
            return time.time() - self.start_time
        return 0.0
        
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current monitoring metrics"""
        duration = self.get_monitoring_duration()
        
        return {
            'monitoring_duration': duration,
            'portfolio_history_count': len(self.portfolio_history),
            'trade_history_count': len(self.trade_history),
            'monitoring_active': self.monitoring_active
        }


# Enhanced Performance Alert System
class PerformanceAlert:
    """Performance alert system for trading metrics"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.alert_history = []
        self.alert_thresholds = {
            'sharpe_ratio': 1.0,
            'max_drawdown': 0.10,
            'win_rate': 0.50,
            'portfolio_value': 0.95,  # 95% of initial value
            'volatility': 0.30,
            'correlation': 0.80
        }
    
    def check_alerts(self, metrics: Dict) -> List[Dict]:
        """Check for performance alerts"""
        alerts = []
        
        try:
            # Sharpe ratio alert
            if metrics.get('sharpe_ratio', 0) < self.alert_thresholds['sharpe_ratio']:
                alerts.append({
                    'type': 'sharpe_alert',
                    'message': f"Low Sharpe ratio: {metrics.get('sharpe_ratio', 0):.3f}",
                    'severity': 'warning',
                    'timestamp': datetime.now()
                })
            
            # Drawdown alert
            if abs(metrics.get('max_drawdown', 0)) > self.alert_thresholds['max_drawdown']:
                alerts.append({
                    'type': 'drawdown_alert',
                    'message': f"High drawdown: {metrics.get('max_drawdown', 0):.2%}",
                    'severity': 'critical',
                    'timestamp': datetime.now()
                })
            
            # Win rate alert
            if metrics.get('win_rate', 0) < self.alert_thresholds['win_rate']:
                alerts.append({
                    'type': 'win_rate_alert',
                    'message': f"Low win rate: {metrics.get('win_rate', 0):.2%}",
                    'severity': 'warning',
                    'timestamp': datetime.now()
                })
            
            # Store alerts
            self.alert_history.extend(alerts)
            
            # Log alerts
            for alert in alerts:
                logger.warning(f" {alert['severity'].upper()}: {alert['message']}")
            
        except Exception as e:
            logger.error(f" Alert check error: {e}")
        
        return alerts
