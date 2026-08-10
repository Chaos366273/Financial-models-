"""
PPO Utils Package
Utility functions and classes for the PPO trading system
"""

# Import key utility classes for easy access with error handling
try:
    from .report_generator import ReportGenerator  # type: ignore[reportUnusedImport]
except ImportError:
    ReportGenerator = None

try:
    from .advanced_metrics import AdvancedRiskMetrics
except ImportError:
    AdvancedRiskMetrics = None

try:
    from .performance_monitor import PerformanceMonitor
except ImportError:
    PerformanceMonitor = None

try:
    from .memory_optimizer import MemoryOptimizer  # type: ignore[reportUnusedImport]
except ImportError:
    MemoryOptimizer = None

try:
    from .logging_utils import setup_logging, get_logger
except ImportError:
    setup_logging = get_logger = None

try:
    from .data_validator import DataValidator
except ImportError:
    DataValidator = None

try:
    from .benchmark_evaluator import BenchmarkEvaluator
except ImportError:
    BenchmarkEvaluator = None

try:
    from .performance_attribution import PerformanceAttribution
except ImportError:
    PerformanceAttribution = None

try:
    from .portfolio_analyzer import PortfolioAnalyzer
except ImportError:
    PortfolioAnalyzer = None

try:
    from .evaluation_engine import EvaluationEngine
except ImportError:
    EvaluationEngine = None

try:
    from .model_cleanup import ModelCleanup
except ImportError:
    ModelCleanup = None

try:
    from .parameter_manager import ParameterManager
except ImportError:
    ParameterManager = None

try:
    from .config_manager import ConfigManager
except ImportError:
    ConfigManager = None

try:
    from .realtime_stream import AlpacaNewsStream
except ImportError:
    AlpacaNewsStream = None

try:
    from .model_readiness_assessor import ModelReadinessAssessor
except ImportError:
    ModelReadinessAssessor = None

try:
    from .emergency_mode_manager import emergency_manager
except ImportError:
    emergency_manager = None

try:
    from .market_adaptive_entropy import MarketAdaptiveEntropy
except ImportError:
    MarketAdaptiveEntropy = None

# Build __all__ list based on successful imports
__all__ = []
_items = list(globals().items())  # Create a static list to avoid dictionary changed size error
for name, obj in _items:
    if not name.startswith('_') and obj is not None and name not in ['sys', 'os', 'ImportError']:
        __all__.append(name)
