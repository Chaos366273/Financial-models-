"""
Core trading components
"""

try:
    from .trading_callbacks import TradingCallback, PerformanceMonitor
except ImportError as e:
    print(f"Warning: Some core components not available: {e}")

# Define fallback classes for missing components
class TradingCallback:
    """Fallback trading callback class"""
    pass

class PerformanceMonitor:
    """Fallback performance monitor class"""
    pass

__all__ = [
    'TradingCallback',
    'PerformanceMonitor'
]
