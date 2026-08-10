"""
Data processing components
"""

try:
    from .alpaca_client import AlpacaDataClient
    from .data_processor import DataProcessor
    from .feature_engineer import FeatureEngineer
    from .news_stream import AlpacaNewsStream
    from .sentiment_analyzer import SentimentAnalyzer
    from .alpaca_news_stream import AlpacaNewsStream as AlpacaNewsStreamV2
    from .technical_indicators import TechnicalIndicators
    from .cache_manager import CacheManager
    from .benchmark_data import BenchmarkData
    from .realtime_data import RealtimeData
except ImportError:
    # Optional modules may be missing; keep silent to avoid noisy warnings in console
    pass

__all__ = [
    'AlpacaDataClient',
    'DataProcessor',
    'FeatureEngineer',
    'AlpacaNewsStream',
    'SentimentAnalyzer',
    'AlpacaNewsStreamV2',
    'TechnicalIndicators',
    'CacheManager',
    'BenchmarkData',
    'RealtimeData'
]
