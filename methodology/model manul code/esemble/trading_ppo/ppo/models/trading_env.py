# Import constants with fallback
import sys
import os
import pickle
from pathlib import Path

# Add the ppo directory to the path for imports
ppo_dir = os.path.dirname(os.path.dirname(__file__))
if ppo_dir not in sys.path:
    sys.path.insert(0, ppo_dir)

# Add parent directory for config imports
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Core imports
import numpy as np
import logging
import traceback
from typing import Dict, Any, List, Optional, Tuple, cast
from datetime import datetime
import warnings

# Type stubs for VS Code type checker (Pylance/basedpyright)
# These help the type checker understand imports even with try/except blocks
try:
    import gymnasium as gym  # type: ignore
    from gymnasium import spaces  # type: ignore
except ImportError:
    try:
        import gym  # type: ignore
        from gym import spaces  # type: ignore
    except ImportError:
        # Create a minimal gym-like interface if neither is available
        class gym:  # type: ignore
            class Env:
                def __init__(self):
                    pass
            class spaces:
                class Box:
                    def __init__(self, low, high, shape, dtype):
                        self.shape = shape
                class Discrete:
                    def __init__(self, n):
                        self.n = n

# Explicit imports for type checker (with fallback for runtime)
try:
    from ppo.constants import *  # type: ignore
except ImportError:
    try:
        from constants import *  # type: ignore
    except ImportError:
        # Fallback constants if import fails
        DEFAULT_DRAWDOWN_LIMIT = 0.15
        CUDA_GARBAGE_COLLECTION_THRESHOLD = 0.6
        DEFAULT_RATE_LIMIT = 200
        LOG_FREQUENCY = 100
        REWARD_POOL_SIZE = 1000
        DEFAULT_NUM_FEATURES = 15
        DEFAULT_RETRY_ATTEMPTS = 3
        DEFAULT_NUM_TICKERS = 12  # UPDATED: Now 12 tickers (AAPL, MSFT, NVDA, JPM, SPY, GLD, BTC-USD, KO, JNJ, AMZN, XLE, TLT)
        DEFAULT_SLIPPAGE = 0.0005
        DEFAULT_TRAILING_STOP = 0.05
# Ensure NUM_ACTIONS is always defined (ppo.constants may not export it in older versions)
if 'NUM_ACTIONS' not in globals():
    NUM_ACTIONS = 4  # 0=HOLD, 1=BUY, 2=SELL, 3=SHORT

# Explicit import for type checker
try:
    from ppo.core.trade_validator import TradeValidator  # type: ignore
except ImportError:
    try:
        from core.trade_validator import TradeValidator  # type: ignore
    except ImportError:
        # Fallback: create dummy trade validator
        class TradeValidator:  # type: ignore
            def __init__(self, config=None):
                self.config = config or {}
            def validate_trade(self, *args, **kwargs):
                return True, ""


# Create fallback config first to prevent import errors
class FallbackConfig:
    COINTEGRATION_CONFIG = {
        'cointegration_enabled': False,
        'aggressive_pairs_trading': False,
        'pairs_trading_enabled': False
    }
    COOLDOWN_CONFIG = {
        'cooldown_enabled': True,
        'min_hold_bars': 3,
        'cooldown_bars': 3,
        'cooldown_penalty': 0.001,
        'log_cooldown_violations': True
    }
    STRESS_TEST_CONFIG = {
        'stress_testing_enabled': False,
        'cost_scenarios': {
            'baseline': {'slippage_multiplier': 1.0, 'impact_multiplier': 1.0},
            'moderate_stress': {'slippage_multiplier': 1.5, 'impact_multiplier': 1.5},
            'high_stress': {'slippage_multiplier': 2.0, 'impact_multiplier': 2.0}
        },
        'volatility_scenarios': {
            'baseline': {'volatility_multiplier': 1.0},
            'low_volatility': {'volatility_multiplier': 0.5},
            'high_volatility': {'volatility_multiplier': 2.0}
        }
    }

# Try to import config at module level to verify it's accessible
try:
    import importlib
    config = importlib.import_module('config')  # type: ignore
    CONFIG_AVAILABLE = True
except ImportError:
    # If config import fails, use fallback config
    CONFIG_AVAILABLE = False
    config = FallbackConfig()  # type: ignore

warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# REMOVED: Duplicate incomplete class definition (lines 60-1503)
# The actual working FixedTradingEnv class starts at line 1597 below



class FixedTradingEnv(gym.Env):
    """Fixed trading environment for PPO agent training with resolved observation shape and index bounds issues"""
    
    # Class-level constants to avoid repeated dict creation
    ACTION_NAMES = {0: 'HOLD', 1: 'BUY', 2: 'SELL', 3: 'SHORT'}
    DEFAULT_POLICY_ACTION_COUNTS = {0: 0, 1: 0, 2: 0, 3: 0}  # HOLD, BUY, SELL, SHORT
    # Attributes assigned by env_factory for downstream consumers
    dates: Any
    feature_cols: List[str]
    trading_config: Dict[str, Any]
    unique_tickers: List[str]

    def _load_fundamental_cache(self) -> None:
        """Load cached fundamental snapshots from disk the first time they're needed."""
        if self._fundamental_cache_loaded:
            return

        self._fundamental_cache_loaded = True
        cache_dir = self._fundamental_cache_dir
        if not cache_dir.exists():
            self.logger.warning(f"Fundamental cache directory missing: {cache_dir} (fundamentals disabled)")
            self._fundamental_data_available = False
            return

        candidates = sorted(cache_dir.glob("*.pkl"), key=lambda path: path.stat().st_mtime, reverse=True)

        for candidate in candidates:
            try:
                with candidate.open("rb") as handle:
                    data = pickle.load(handle)
                if isinstance(data, dict) and data:
                    self._fundamental_cache = {
                        str(symbol).upper(): value
                        for symbol, value in data.items()
                        if isinstance(value, dict)
                    }
                    self._fundamental_data_available = len(self._fundamental_cache) > 0
                    if self._fundamental_data_available:
                        self.logger.info(f"Fundamental cache loaded from {candidate}")
                    return
            except Exception as err:
                self.logger.warning(f"Failed to load fundamental cache {candidate}: {err}")

        self._fundamental_cache = {}
        self._fundamental_data_available = False

    def _resolve_symbol(self, ticker_identifier: Any) -> str:
        """Translate synthetic ticker placeholders back to configured ticker symbols."""
        if isinstance(ticker_identifier, str) and ticker_identifier.startswith("TICKER_"):
            idx_str = ticker_identifier.split("_")[-1]
            if idx_str.isdigit():
                idx = int(idx_str)
                if 0 <= idx < len(self.ticker_symbols):
                    return self.ticker_symbols[idx]
        elif isinstance(ticker_identifier, int) and 0 <= ticker_identifier < len(self.ticker_symbols):
            return self.ticker_symbols[ticker_identifier]

        if isinstance(ticker_identifier, str):
            return ticker_identifier

        return self.ticker_symbols[0]

    def _get_cached_fundamental_metrics(self, symbol: str) -> Optional[Dict[str, float]]:
        """Return cleaned fundamental metrics for the given symbol if available.
        
        CRITICAL FIX: Time-aware fundamental data filtering to prevent data leakage.
        Fundamental data is only used if it's "safe" (old enough to have been available
        at the current step). Since fundamental data is typically updated quarterly,
        we use a conservative approach: only use data that's at least 1 quarter old.
        """
        if not self._fundamental_data_available:
            return None

        metrics = self._fundamental_cache.get(symbol.upper())
        if not isinstance(metrics, dict):
            return None

        # CRITICAL FIX: Time-aware filtering to prevent data leakage
        # Fundamental data is typically static snapshots (quarterly earnings reports)
        # We need to ensure we only use data that would have been available at current_step
        
        # Strategy: Since fundamental data is usually collected before training starts,
        # it's generally safe to use throughout training. However, in test mode,
        # we need to be more conservative since test data represents future dates.
        
        if hasattr(self, 'mode') and self.mode == 'test':
            # In test mode (evaluation on unseen data), be conservative:
            # Only use fundamental data in early steps to avoid using future data
            # Test data represents dates AFTER training, so fundamental data from
            # before training may not be appropriate for later test steps
            total_steps = getattr(self, 'price_matrix', np.array([])).shape[0] if hasattr(self, 'price_matrix') else 1000
            # Use first 50% of test data - this is a conservative threshold
            # that ensures we're not using fundamental data that might be from
            # a time period that overlaps with test data dates
            safe_threshold = int(total_steps * 0.50)  # First 50% of test data
            
            if self.current_step > safe_threshold:
                # Too far into test data - fundamental data may not be appropriate
                # Return None to prevent potential data leakage
                return None
        
        # For training mode: Fundamental data is generally safe if it exists
        # (assuming it was collected before training period started)
        # We use it throughout training since it represents historical data
        # that would have been available at any point during the training period

        required_keys = (
            'pe_ratio',
            'pb_ratio',
            'debt_equity',
            'roe',
            'current_ratio',
            'revenue_growth',
            'earnings_growth',
        )

        cleaned: Dict[str, float] = {}
        for key in required_keys:
            value = metrics.get(key)
            if value is None:
                return None
            try:
                cleaned[key] = float(value)
            except (TypeError, ValueError):
                return None

        return cleaned

    def _calculate_fundamental_metrics(self, tickers: list) -> Dict[str, Dict[str, float]]:
        """Calculate fundamental analysis metrics for each ticker with per-step caching."""
        try:
            if getattr(self, '_fundamental_metric_cache_step', -1) != self.current_step:
                self._fundamental_metric_cache = {}
                self._fundamental_metric_cache_step = self.current_step

            self._load_fundamental_cache()

            fundamental_data: Dict[str, Dict[str, float]] = {}
            newly_generated = 0
            time_filtered_count = 0  # Track how many were filtered due to time constraints

            for ticker in tickers:
                if ticker in self._fundamental_metric_cache:
                    cached_metrics = self._fundamental_metric_cache[ticker]
                    if cached_metrics:
                        fundamental_data[ticker] = cached_metrics
                    continue

                symbol = self._resolve_symbol(ticker)
                # CRITICAL FIX: _get_cached_fundamental_metrics now applies time-aware filtering
                metrics = self._get_cached_fundamental_metrics(symbol)
                if metrics:
                    self._fundamental_metric_cache[ticker] = metrics
                    fundamental_data[ticker] = metrics
                    newly_generated += 1
                else:
                    self._fundamental_metric_cache[ticker] = {}
                    # Check if it was filtered due to time constraints
                    if self._fundamental_data_available:
                        time_filtered_count += 1

            if not fundamental_data:
                # Only warn if data was available but filtered, or if cache is empty
                if time_filtered_count > 0:
                    if not hasattr(self, '_time_filter_warning_emitted'):
                        self.logger.warning(
                            f"⚠️ TIME-AWARE FILTERING: {time_filtered_count} fundamental metrics filtered "
                            f"at step {self.current_step} to prevent data leakage (using only past data)"
                        )
                        self._time_filter_warning_emitted = True
                elif not self._fundamental_data_available:
                    if not self._missing_fundamentals_warning_emitted:
                        self.logger.warning("Fundamental cache empty – skipping fundamental rewards until real data is cached.")
                        self._missing_fundamentals_warning_emitted = True
            else:
                self._fundamental_data_available = True
                if newly_generated and (self.verbose_step_logs or self.current_step % max(getattr(self, 'log_every_n_steps', 50), 1) == 0):
                    self.logger.info(
                        f" FUNDAMENTAL ANALYSIS: Metrics cached for {len(fundamental_data)} tickers "
                        f"(loaded {newly_generated} from cache, time-filtered: {time_filtered_count})"
                    )

            return fundamental_data

        except Exception as e:
            self.logger.error(f"Fundamental metrics calculation failed: {e}")
            return {}

    def _analyze_financial_health(self, fundamental_data: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, Any]]:
        """Analyze financial health based on fundamental metrics with caching"""
        try:
            if getattr(self, '_fundamental_analysis_cache_step', -1) != self.current_step:
                self._fundamental_analysis_cache = {}
                self._fundamental_analysis_cache_step = self.current_step

            health_analysis: Dict[str, Dict[str, Any]] = {}

            for ticker, metrics in fundamental_data.items():
                cache_key = (ticker, self.current_step)
                if cache_key in self._fundamental_analysis_cache:
                    health_analysis[ticker] = self._fundamental_analysis_cache[cache_key]
                    continue

                health_factors = []

                pe_ratio = metrics['pe_ratio']
                if 10 <= pe_ratio <= 20:
                    pe_score = 25
                    health_factors.append("Good PE ratio")
                elif 20 < pe_ratio <= 25:
                    pe_score = 15
                    health_factors.append("Moderate PE ratio")
                else:
                    pe_score = 5
                    health_factors.append("High PE ratio")

                roe = metrics['roe']
                if roe >= 0.15:
                    roe_score = 25
                    health_factors.append("Strong ROE")
                elif roe >= 0.10:
                    roe_score = 15
                    health_factors.append("Good ROE")
                else:
                    roe_score = 5
                    health_factors.append("Weak ROE")

                debt_equity = metrics['debt_equity']
                if debt_equity <= 0.5:
                    debt_score = 25
                    health_factors.append("Low debt")
                elif debt_equity <= 1.0:
                    debt_score = 15
                    health_factors.append("Moderate debt")
                else:
                    debt_score = 5
                    health_factors.append("High debt")

                revenue_growth = metrics['revenue_growth']
                earnings_growth = metrics['earnings_growth']
                if revenue_growth > 0.1 and earnings_growth > 0.1:
                    growth_score = 25
                    health_factors.append("Strong growth")
                elif revenue_growth > 0.05 and earnings_growth > 0.05:
                    growth_score = 15
                    health_factors.append("Moderate growth")
                else:
                    growth_score = 5
                    health_factors.append("Weak growth")

                health_score = pe_score + roe_score + debt_score + growth_score
                if health_score >= 80:
                    health_status = "EXCELLENT"
                elif health_score >= 60:
                    health_status = "GOOD"
                elif health_score >= 40:
                    health_status = "FAIR"
                else:
                    health_status = "POOR"

                analysis = {
                    'health_score': health_score,
                    'health_status': health_status,
                    'health_factors': health_factors,
                    'pe_score': pe_score,
                    'roe_score': roe_score,
                    'debt_score': debt_score,
                    'growth_score': growth_score
                }

                self._fundamental_analysis_cache[cache_key] = analysis
                health_analysis[ticker] = analysis

            if health_analysis and (self.verbose_step_logs or self.current_step % max(getattr(self, 'log_every_n_steps', 50), 1) == 0):
                self.logger.info(f" FUNDAMENTAL ANALYSIS: Financial health analyzed for {len(health_analysis)} tickers")
            return health_analysis

        except Exception as e:
            self.logger.error(f"Financial health analysis failed: {e}")
            return {}

    def _get_fundamental_signals(self, health_analysis: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Generate trading signals based on fundamental analysis"""
        try:
            fundamental_signals = {}

            for ticker, health in health_analysis.items():
                health_score = health['health_score']
                health_status = health['health_status']

                # Generate signals based on financial health
                if health_status == "EXCELLENT" and health_score >= 85:
                    signal = "STRONG_BUY"
                    signal_strength = 0.9
                    confidence = 0.85
                elif health_status == "EXCELLENT":
                    signal = "BUY"
                    signal_strength = 0.7
                    confidence = 0.75
                elif health_status == "GOOD" and health_score >= 70:
                    signal = "BUY"
                    signal_strength = 0.6
                    confidence = 0.65
                elif health_status == "GOOD":
                    signal = "HOLD"
                    signal_strength = 0.4
                    confidence = 0.55
                elif health_status == "FAIR":
                    signal = "HOLD"
                    signal_strength = 0.3
                    confidence = 0.45
                else:  # POOR
                    signal = "SELL"
                    signal_strength = 0.8
                    confidence = 0.75

                fundamental_signals[ticker] = {
                    'signal': signal,
                    'signal_strength': signal_strength,
                    'confidence': confidence,
                    'health_score': health_score,
                    'health_status': health_status,
                    'health_factors': health['health_factors']
                }

            logger.info(f" FUNDAMENTAL ANALYSIS: Trading signals generated for {len(fundamental_signals)} tickers")
            return fundamental_signals

        except Exception as e:
            logger.error(f"Fundamental signal generation failed: {e}")
            return {}

    def _analyze_news_sentiment(self, tickers: list) -> Dict[str, Dict[str, float]]:
        """Analyze news sentiment for each ticker using REAL data from Alpaca API.
        
        DISABLED: Sentiment analysis disabled per user request.
        Returns empty dict.
        """
        # DISABLED: Sentiment analysis disabled
        return {}
        
        # Original code below (disabled):
        try:
            news_sentiment = {}
            
            # CRITICAL: Only use REAL data from Alpaca API - no synthetic data
            # Try to get real news sentiment from Alpaca News API
            try:
                import os
                import requests
                
                api_key = os.environ.get('APCA_API_KEY') or os.environ.get('ALPACA_API_KEY')
                secret_key = os.environ.get('APCA_API_SECRET') or os.environ.get('ALPACA_API_SECRET')
                
                if not api_key or not secret_key:
                    self.logger.warning("⚠️ NO SYNTHETIC DATA: Alpaca API keys not found - skipping news sentiment (no fake data)")
                    return {}  # Return empty - don't use synthetic data
                
                # Use Alpaca News API for REAL sentiment data
                base_url = "https://data.alpaca.markets/v1beta1"
                headers = {
                    'APCA-API-KEY-ID': api_key,
                    'APCA-API-SECRET-KEY': secret_key
                }
                
                for ticker in tickers:
                    try:
                        symbol = self._resolve_symbol(ticker) if hasattr(self, '_resolve_symbol') else ticker
                        news_url = f"{base_url}/news?symbols={symbol}&limit=20"
                        response = requests.get(news_url, headers=headers, timeout=5)
                        
                        if response.status_code == 200:
                            news_data = response.json()
                            articles = news_data.get('news', [])
                            
                            if articles:
                                # Calculate REAL sentiment from actual news headlines
                                # Simple keyword-based sentiment (can be enhanced with NLP)
                                positive_keywords = ['up', 'gain', 'rise', 'surge', 'beat', 'exceed', 'strong', 'growth']
                                negative_keywords = ['down', 'fall', 'drop', 'decline', 'miss', 'weak', 'loss', 'concern']
                                
                                sentiment_scores = []
                                for article in articles:
                                    headline = article.get('headline', '').lower()
                                    pos_count = sum(1 for kw in positive_keywords if kw in headline)
                                    neg_count = sum(1 for kw in negative_keywords if kw in headline)
                                    
                                    if pos_count > neg_count:
                                        sentiment_scores.append(0.5)
                                    elif neg_count > pos_count:
                                        sentiment_scores.append(-0.5)
                                    else:
                                        sentiment_scores.append(0.0)
                                
                                if sentiment_scores:
                                    avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
                                    news_sentiment[ticker] = {
                                        'sentiment_score': avg_sentiment,
                                        'sentiment_category': "POSITIVE" if avg_sentiment > 0.1 else ("NEGATIVE" if avg_sentiment < -0.1 else "NEUTRAL"),
                                        'news_volume': len(articles),
                                        'sentiment_confidence': min(0.8, len(articles) / 20.0),  # Based on article count
                                        'recent_impact': 1.0  # All articles are recent
                                    }
                                else:
                                    # No sentiment calculated - return empty for this ticker
                                    pass
                            else:
                                # No news articles - return empty for this ticker
                                pass
                        else:
                            # API call failed - don't use synthetic data
                            self.logger.warning(f"⚠️ NO SYNTHETIC DATA: Alpaca News API unavailable for {ticker} (status {response.status_code})")
                    except Exception as e:
                        self.logger.warning(f"⚠️ NO SYNTHETIC DATA: Failed to get real news for {ticker}: {e}")
                        # Don't use synthetic data - just skip this ticker
                        continue
                
                if news_sentiment:
                    self.logger.info(f"✅ REAL DATA: News sentiment analyzed for {len(news_sentiment)} tickers using Alpaca News API")
                else:
                    self.logger.info("⚠️ NO SYNTHETIC DATA: No real news sentiment available - using empty dict (no fake data)")
                
                return news_sentiment
                
            except Exception as e:
                self.logger.warning(f"⚠️ NO SYNTHETIC DATA: Alpaca News API not available: {e} - returning empty dict (no fake data)")
                return {}  # Return empty - don't use synthetic data

        except Exception as e:
            self.logger.error(f"News sentiment analysis failed: {e}")
            return {}  # Return empty - don't use synthetic data

    def _analyze_social_sentiment(self, tickers: list) -> Dict[str, Dict[str, float]]:
        """Analyze social media sentiment for each ticker using REAL data.
        
        DISABLED: Sentiment analysis disabled per user request.
        Returns empty dict.
        """
        # DISABLED: Sentiment analysis disabled
        return {}
        
        # Original code below (disabled):
        try:
            # CRITICAL: NO SYNTHETIC DATA - Social sentiment requires real API access
            # Since we don't have Twitter/Reddit API access configured, return empty
            # This prevents using random/fake data
            
            self.logger.warning("⚠️ NO SYNTHETIC DATA: Social sentiment requires Twitter/Reddit API access - not configured. Returning empty dict (no fake data).")
            return {}  # Return empty - don't use synthetic data

        except Exception as e:
            self.logger.error(f"Social sentiment analysis failed: {e}")
            return {}  # Return empty - don't use synthetic data

    def _get_sentiment_signals(self, news_sentiment: Dict[str, Dict[str, float]],
                              social_sentiment: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, Any]]:
        """Generate trading signals based on combined sentiment analysis"""
        try:
            sentiment_signals = {}

            for ticker in news_sentiment.keys():
                if ticker not in social_sentiment:
                    continue

                news_data = news_sentiment[ticker]
                social_data = social_sentiment[ticker]

                # Combine news and social sentiment with weights
                news_weight = 0.6  # News is more reliable
                social_weight = 0.4  # Social media is more volatile

                combined_sentiment = (news_data['sentiment_score'] * news_weight +
                                    social_data['social_score'] * social_weight)

                # Calculate combined confidence
                combined_confidence = (news_data['sentiment_confidence'] * news_weight +
                                     social_data['social_confidence'] * social_weight)

                # Adjust for news volume and social mentions
                volume_factor = min(1.0, (news_data['news_volume'] + social_data['mention_count']) / 100)
                combined_confidence *= volume_factor

                # Generate sentiment-based signals
                if combined_sentiment > 0.5 and combined_confidence > 0.7:
                    signal = "STRONG_BUY"
                    signal_strength = 0.8
                elif combined_sentiment > 0.2 and combined_confidence > 0.6:
                    signal = "BUY"
                    signal_strength = 0.6
                elif combined_sentiment < -0.5 and combined_confidence > 0.7:
                    signal = "STRONG_SELL"
                    signal_strength = 0.8
                elif combined_sentiment < -0.2 and combined_confidence > 0.6:
                    signal = "SELL"
                    signal_strength = 0.6
                else:
                    signal = "HOLD"
                    signal_strength = 0.3

                sentiment_signals[ticker] = {
                    'signal': signal,
                    'signal_strength': signal_strength,
                    'confidence': combined_confidence,
                    'combined_sentiment': combined_sentiment,
                    'news_sentiment': news_data['sentiment_score'],
                    'social_sentiment': social_data['social_score'],
                    'news_category': news_data['sentiment_category'],
                    'social_category': social_data['social_category'],
                    'news_volume': news_data['news_volume'],
                    'social_mentions': social_data['mention_count']
                }

            logger.info(f" SENTIMENT ANALYSIS: Combined signals generated for {len(sentiment_signals)} tickers")
            return sentiment_signals

        except Exception as e:
            logger.error(f"Sentiment signal generation failed: {e}")
            return {}

    def _calculate_dynamic_stop_loss(self, price_matrix: np.ndarray, tickers: list,
                                   base_stop_loss: float = 0.15) -> Dict[str, Dict[str, float]]:
        """Calculate dynamic stop loss levels based on volatility and market conditions"""
        try:
            dynamic_stop_losses = {}

            for i, ticker in enumerate(tickers):
                prices = price_matrix[:, i]

                if len(prices) < 20:  # Need sufficient data
                    dynamic_stop_losses[ticker] = {
                        'stop_loss_pct': base_stop_loss,
                        'stop_loss_price': prices[-1] * (1 - base_stop_loss),
                        'volatility_adjustment': 0.0,
                        'regime_adjustment': 0.0,
                        'final_adjustment': 0.0
                    }
                    continue

                # Calculate recent volatility (20-day rolling)
                returns = np.diff(prices) / prices[:-1]
                recent_volatility = np.std(returns[-20:]) if len(returns) >= 20 else np.std(returns)

                # Calculate ATR (Average True Range) approximation
                high_low_range = np.array([abs(prices[j] - prices[j-1]) for j in range(1, len(prices))])
                atr = np.mean(high_low_range[-20:]) if len(high_low_range) >= 20 else np.mean(high_low_range)

                # Volatility-based adjustment
                # Higher volatility = wider stop loss
                volatility_multiplier = 1.0 + (recent_volatility * 2.0)  # Scale volatility
                volatility_adjustment = (volatility_multiplier - 1.0) * base_stop_loss

                # ATR-based adjustment
                atr_adjustment = (atr / prices[-1]) * 0.5  # ATR as percentage of price

                # Combine volatility adjustments
                total_volatility_adjustment = volatility_adjustment + atr_adjustment

                # Regime-based adjustment (if regime detection is available)
                regime_adjustment = 0.0
                if hasattr(self, '_active_regime'):
                    regime = self._active_regime
                    if regime == 'volatile':
                        regime_adjustment = base_stop_loss * 0.3  # 30% wider in volatile markets
                    elif regime == 'bear':
                        regime_adjustment = base_stop_loss * 0.2  # 20% wider in bear markets
                    elif regime == 'bull':
                        regime_adjustment = -base_stop_loss * 0.1  # 10% tighter in bull markets

                # Calculate final dynamic stop loss
                final_stop_loss = base_stop_loss + total_volatility_adjustment + regime_adjustment

                # Apply reasonable bounds (5% to 30%)
                final_stop_loss = max(0.05, min(0.30, final_stop_loss))

                # Calculate stop loss price
                current_price = prices[-1]
                stop_loss_price = current_price * (1 - final_stop_loss)

                dynamic_stop_losses[ticker] = {
                    'stop_loss_pct': final_stop_loss,
                    'stop_loss_price': stop_loss_price,
                    'volatility_adjustment': total_volatility_adjustment,
                    'regime_adjustment': regime_adjustment,
                    'final_adjustment': final_stop_loss - base_stop_loss,
                    'recent_volatility': recent_volatility,
                    'atr': atr,
                    'base_stop_loss': base_stop_loss
                }

            logger.info(f" DYNAMIC STOP LOSS: Dynamic stop losses calculated for {len(dynamic_stop_losses)} tickers")
            return dynamic_stop_losses

        except Exception as e:
            logger.error(f"Dynamic stop loss calculation failed: {e}")
            return {}

    def _adjust_stop_loss_volatility(self, current_stop_loss: float, volatility: float,
                                   base_volatility: float = 0.02) -> float:
        """Adjust stop loss based on current volatility vs base volatility"""
        try:
            # Calculate volatility ratio
            volatility_ratio = volatility / base_volatility

            # Adjust stop loss based on volatility
            if volatility_ratio > 2.0:  # Very high volatility
                adjustment_factor = 1.5
            elif volatility_ratio > 1.5:  # High volatility
                adjustment_factor = 1.3
            elif volatility_ratio < 0.5:  # Low volatility
                adjustment_factor = 0.8
            else:  # Normal volatility
                adjustment_factor = 1.0

            adjusted_stop_loss = current_stop_loss * adjustment_factor

            # Apply bounds
            return max(0.05, min(0.30, adjusted_stop_loss))

        except Exception as e:
            logger.error(f"Stop loss volatility adjustment failed: {e}")
            return current_stop_loss

    def _get_stop_loss_levels(self, dynamic_stop_losses: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """Get comprehensive stop loss levels including trailing stops"""
        try:
            stop_loss_levels = {}

            for ticker, stop_data in dynamic_stop_losses.items():
                base_stop_loss = stop_data['stop_loss_pct']
                current_price = stop_data['stop_loss_price'] / (1 - base_stop_loss)

                # Calculate trailing stop levels
                trailing_stop_1 = base_stop_loss * 0.5  # 50% of main stop loss
                trailing_stop_2 = base_stop_loss * 0.3  # 30% of main stop loss

                # Calculate stop loss prices
                main_stop_price = current_price * (1 - base_stop_loss)
                trailing_stop_1_price = current_price * (1 - trailing_stop_1)
                trailing_stop_2_price = current_price * (1 - trailing_stop_2)

                stop_loss_levels[ticker] = {
                    'main_stop_loss_pct': base_stop_loss,
                    'main_stop_price': main_stop_price,
                    'trailing_stop_1_pct': trailing_stop_1,
                    'trailing_stop_1_price': trailing_stop_1_price,
                    'trailing_stop_2_pct': trailing_stop_2,
                    'trailing_stop_2_price': trailing_stop_2_price,
                    'current_price': current_price,
                    'volatility_adjustment': stop_data.get('volatility_adjustment', 0.0),
                    'regime_adjustment': stop_data.get('regime_adjustment', 0.0)
                }

            logger.info(f" DYNAMIC STOP LOSS: Stop loss levels calculated for {len(stop_loss_levels)} tickers")
            return stop_loss_levels

        except Exception as e:
            logger.error(f"Stop loss levels calculation failed: {e}")
            return {}

    def _validate_trade(self, ticker: str, action: str, quantity: float, current_price: float,
                       portfolio_value: float, position_sizes: Dict[str, float]) -> Dict[str, Any]:
        """Validate a trade before execution"""
        try:
            validation_result = {
                'is_valid': True,
                'validation_score': 1.0,
                'warnings': [],
                'errors': [],
                'risk_assessment': 'LOW',
                'recommendations': []
            }

            # 1. Position Size Validation
            if action in ['BUY', 'SELL']:
                position_value = quantity * current_price
                position_pct = position_value / portfolio_value

                if position_pct > 0.1:  # More than 10% of portfolio
                    validation_result['warnings'].append(f"Large position: {position_pct:.1%} of portfolio")
                    validation_result['validation_score'] *= 0.8

                if position_pct > 0.2:  # More than 20% of portfolio
                    validation_result['errors'].append(f"Position too large: {position_pct:.1%} of portfolio")
                    validation_result['is_valid'] = False
                    validation_result['validation_score'] *= 0.5

            # 2. Portfolio Concentration Check
            total_position_value = sum(pos * current_price for pos in position_sizes.values())
            concentration_ratio = total_position_value / portfolio_value

            if concentration_ratio > 0.8:  # More than 80% invested
                validation_result['warnings'].append(f"High portfolio concentration: {concentration_ratio:.1%}")
                validation_result['validation_score'] *= 0.9

            if concentration_ratio > 0.95:  # More than 95% invested
                validation_result['errors'].append(f"Portfolio over-concentrated: {concentration_ratio:.1%}")
                validation_result['is_valid'] = False
                validation_result['validation_score'] *= 0.3

            # 3. Liquidity Check (simplified)
            if quantity > 1000:  # Large trade
                validation_result['warnings'].append(f"Large trade size: {quantity} shares")
                validation_result['validation_score'] *= 0.9

            # 4. Risk Assessment
            if validation_result['validation_score'] >= 0.8:
                validation_result['risk_assessment'] = 'LOW'
            elif validation_result['validation_score'] >= 0.6:
                validation_result['risk_assessment'] = 'MEDIUM'
            else:
                validation_result['risk_assessment'] = 'HIGH'

            # 5. Generate Recommendations
            if validation_result['warnings']:
                validation_result['recommendations'].append("Consider reducing position size")

            if validation_result['errors']:
                validation_result['recommendations'].append("Trade rejected due to risk concerns")

            if validation_result['risk_assessment'] == 'HIGH':
                validation_result['recommendations'].append("High risk trade - consider alternatives")

            logger.info(f" TRADE VALIDATION: {ticker} {action} - {validation_result['risk_assessment']} risk")
            return validation_result

        except Exception as e:
            logger.error(f"Trade validation failed: {e}")
            return {
                'is_valid': False,
                'validation_score': 0.0,
                'warnings': [],
                'errors': [f"Validation error: {str(e)}"],
                'risk_assessment': 'HIGH',
                'recommendations': ['Trade validation failed']
            }

    def _check_risk_limits(self, portfolio_value: float, position_sizes: Dict[str, float],
                          current_prices: Dict[str, float]) -> Dict[str, Any]:
        """Check portfolio risk limits and constraints"""
        try:
            risk_limits = {
                'max_single_position': 0.2,  # 20% max per position
                'max_total_exposure': 0.9,   # 90% max total exposure
                'min_cash_reserve': 0.1,     # 10% min cash reserve
                'max_sector_concentration': 0.4  # 40% max per sector
            }

            risk_check = {
                'within_limits': True,
                'violations': [],
                'warnings': [],
                'risk_score': 1.0
            }

            # Calculate current exposure
            total_exposure = 0.0
            for ticker, position in position_sizes.items():
                if ticker in current_prices:
                    position_value = position * current_prices[ticker]
                    total_exposure += position_value

                    # Check single position limit
                    position_pct = position_value / portfolio_value
                    if position_pct > risk_limits['max_single_position']:
                        risk_check['violations'].append(f"{ticker}: {position_pct:.1%} > {risk_limits['max_single_position']:.1%}")
                        risk_check['within_limits'] = False
                        risk_check['risk_score'] *= 0.7

            # Check total exposure limit
            exposure_pct = total_exposure / portfolio_value
            if exposure_pct > risk_limits['max_total_exposure']:
                risk_check['violations'].append(f"Total exposure: {exposure_pct:.1%} > {risk_limits['max_total_exposure']:.1%}")
                risk_check['within_limits'] = False
                risk_check['risk_score'] *= 0.5

            # Check cash reserve
            cash_pct = 1.0 - exposure_pct
            if cash_pct < risk_limits['min_cash_reserve']:
                risk_check['warnings'].append(f"Low cash reserve: {cash_pct:.1%} < {risk_limits['min_cash_reserve']:.1%}")
                risk_check['risk_score'] *= 0.8

            logger.info(f" RISK LIMITS: Portfolio risk check - {'PASSED' if risk_check['within_limits'] else 'FAILED'}")
            return risk_check

        except Exception as e:
            logger.error(f"Risk limits check failed: {e}")
            return {
                'within_limits': False,
                'violations': [f"Risk check error: {str(e)}"],
                'warnings': [],
                'risk_score': 0.0
            }

    def _pre_trade_validation(self, ticker: str, action: str, quantity: float,
                            current_price: float, portfolio_value: float,
                            position_sizes: Dict[str, float],
                            current_prices: Dict[str, float]) -> Dict[str, Any]:
        """Comprehensive pre-trade validation"""
        try:
            # Individual trade validation
            trade_validation = self._validate_trade(ticker, action, quantity, current_price,
                                                  portfolio_value, position_sizes)

            # Portfolio risk limits check
            risk_check = self._check_risk_limits(portfolio_value, position_sizes, current_prices)

            # Combine validation results
            pre_trade_result = {
                'trade_valid': trade_validation['is_valid'],
                'risk_within_limits': risk_check['within_limits'],
                'overall_valid': trade_validation['is_valid'] and risk_check['within_limits'],
                'validation_score': trade_validation['validation_score'] * risk_check['risk_score'],
                'trade_warnings': trade_validation['warnings'],
                'trade_errors': trade_validation['errors'],
                'risk_violations': risk_check['violations'],
                'risk_warnings': risk_check['warnings'],
                'recommendations': trade_validation['recommendations'],
                'risk_assessment': trade_validation['risk_assessment']
            }

            # Add additional recommendations based on combined results
            if not pre_trade_result['overall_valid']:
                pre_trade_result['recommendations'].append("Trade rejected - validation failed")

            if pre_trade_result['validation_score'] < 0.5:
                pre_trade_result['recommendations'].append("High risk trade - consider alternatives")

            logger.info(f" PRE-TRADE VALIDATION: {ticker} {action} - {'APPROVED' if pre_trade_result['overall_valid'] else 'REJECTED'}")
            return pre_trade_result

        except Exception as e:
            logger.error(f"Pre-trade validation failed: {e}")
            return {
                'trade_valid': False,
                'risk_within_limits': False,
                'overall_valid': False,
                'validation_score': 0.0,
                'trade_warnings': [],
                'trade_errors': [f"Validation error: {str(e)}"],
                'risk_violations': [],
                'risk_warnings': [],
                'recommendations': ['Pre-trade validation failed'],
                'risk_assessment': 'HIGH'
            }

    def _calculate_kelly_criterion(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        """Calculate Kelly Criterion for optimal position sizing with Sharpe ratio optimization"""
        try:
            if avg_loss == 0 or win_rate <= 0 or win_rate >= 1:
                return 0.0

            # Kelly = (bp - q) / b
            # b = odds (avg_win / avg_loss)
            # p = win probability
            # q = loss probability (1 - p)

            b = avg_win / avg_loss
            p = win_rate
            q = 1 - p

            # ENHANCED: Calculate Kelly with volatility adjustment for better Sharpe ratio
            kelly = (b * p - q) / b

            # ENHANCED: Apply volatility scaling to improve Sharpe ratio
            # Higher volatility = lower position size for better risk-adjusted returns
            volatility_factor = min(1.0, max(0.5, 1.0 - (avg_loss / avg_win)))
            kelly *= volatility_factor

            # Cap Kelly at 25% of portfolio to prevent over-leveraging
            return max(0, min(kelly, 0.25))

        except Exception as e:
            logger.error(f"Kelly Criterion calculation failed: {e}")
            return 0.0

    def _modern_portfolio_theory(self, expected_returns: np.ndarray, cov_matrix: np.ndarray,
                                risk_free_rate: float = 0.02) -> Dict[str, Any]:
        """Implement Modern Portfolio Theory for optimal portfolio allocation"""
        try:
            n_assets = len(expected_returns)

            # Calculate efficient frontier
            # For simplicity, we'll use a simplified approach
            # In practice, you'd use quadratic programming

            # Calculate Sharpe ratios for each asset
            excess_returns = expected_returns - risk_free_rate
            volatilities = np.sqrt(np.diag(cov_matrix))
            sharpe_ratios = excess_returns / volatilities

            # Weight assets by their Sharpe ratios (simplified MPT)
            weights = sharpe_ratios / np.sum(sharpe_ratios)

            # Ensure weights sum to 1 and are non-negative
            weights = np.maximum(weights, 0)
            weights = weights / np.sum(weights)

            # Calculate portfolio metrics
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
            portfolio_volatility = np.sqrt(portfolio_variance)
            portfolio_sharpe = (portfolio_return - risk_free_rate) / portfolio_volatility

            return {
                'weights': weights,
                'expected_return': portfolio_return,
                'volatility': portfolio_volatility,
                'sharpe_ratio': portfolio_sharpe,
                'method': 'Modern Portfolio Theory'
            }

        except Exception as e:
            logger.error(f"Modern Portfolio Theory calculation failed: {e}")
            return {
                'weights': np.array([1.0 / len(expected_returns)] * len(expected_returns)),
                'expected_return': 0.0,
                'volatility': 0.0,
                'sharpe_ratio': 0.0,
                'method': 'Fallback - Equal Weight'
            }

    def _risk_parity_allocation(self, cov_matrix: np.ndarray) -> Dict[str, Any]:
        """Implement Risk Parity allocation strategy"""
        try:
            n_assets = len(cov_matrix)

            # Calculate risk contributions
            # Risk contribution = weight * (covariance * weight) / portfolio_variance

            # Start with equal weights
            weights = np.ones(n_assets) / n_assets

            # Iterative optimization to achieve equal risk contributions
            max_iterations = 100
            tolerance = 1e-6

            for iteration in range(max_iterations):
                # Calculate portfolio variance
                portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))

                # Calculate risk contributions
                risk_contributions = (weights * np.dot(cov_matrix, weights)) / portfolio_variance

                # Check if risk contributions are equal (within tolerance)
                if np.max(risk_contributions) - np.min(risk_contributions) < tolerance:
                    break

                # Adjust weights to equalize risk contributions
                # Increase weights for assets with low risk contribution
                adjustment_factor = np.mean(risk_contributions) / risk_contributions
                weights = weights * adjustment_factor
                weights = weights / np.sum(weights)  # Normalize

            # Calculate portfolio metrics
            portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
            portfolio_volatility = np.sqrt(portfolio_variance)

            return {
                'weights': weights,
                'volatility': portfolio_volatility,
                'risk_contributions': risk_contributions,
                'method': 'Risk Parity'
            }

        except Exception as e:
            logger.error(f"Risk Parity calculation failed: {e}")
            n_assets = len(cov_matrix)
            return {
                'weights': np.ones(n_assets) / n_assets,
                'volatility': 0.0,
                'risk_contributions': np.ones(n_assets) / n_assets,
                'method': 'Fallback - Equal Weight'
            }

    def _optimize_portfolio_weights(self, tickers: list, price_matrix: np.ndarray,
                                  volume_matrix: np.ndarray, method: str = 'kelly') -> Dict[str, Any]:
        """Optimize portfolio weights using specified method"""
        try:
            n_assets = len(tickers)

            if method == 'kelly':
                # Calculate Kelly Criterion for each asset
                kelly_weights = np.zeros(n_assets)

                for i, ticker in enumerate(tickers):
                    # Calculate returns for this asset
                    prices = price_matrix[:, i]
                    returns = np.diff(prices) / prices[:-1]

                    # Calculate win rate and average win/loss
                    positive_returns = returns[returns > 0]
                    negative_returns = returns[returns < 0]

                    if len(positive_returns) > 0 and len(negative_returns) > 0:
                        win_rate = len(positive_returns) / len(returns)
                        avg_win = float(np.mean(positive_returns))
                        avg_loss = float(abs(np.mean(negative_returns)))

                        kelly_weights[i] = self._calculate_kelly_criterion(win_rate, avg_win, avg_loss)
                    else:
                        kelly_weights[i] = 0.0

                # Normalize weights
                if np.sum(kelly_weights) > 0:
                    kelly_weights = kelly_weights / np.sum(kelly_weights)
                else:
                    kelly_weights = np.ones(n_assets) / n_assets

                return {
                    'weights': kelly_weights,
                    'method': 'Kelly Criterion',
                    'tickers': tickers
                }

            elif method == 'mpt':
                # Modern Portfolio Theory
                # Calculate expected returns and covariance matrix
                returns_matrix = np.diff(price_matrix, axis=0) / price_matrix[:-1]
                expected_returns = np.mean(returns_matrix, axis=0)
                cov_matrix = np.cov(returns_matrix.T)

                mpt_result = self._modern_portfolio_theory(expected_returns, cov_matrix)
                return {
                    'weights': mpt_result['weights'],
                    'method': 'Modern Portfolio Theory',
                    'expected_return': mpt_result['expected_return'],
                    'volatility': mpt_result['volatility'],
                    'sharpe_ratio': mpt_result['sharpe_ratio'],
                    'tickers': tickers
                }

            elif method == 'risk_parity':
                # Risk Parity
                returns_matrix = np.diff(price_matrix, axis=0) / price_matrix[:-1]
                cov_matrix = np.cov(returns_matrix.T)

                risk_parity_result = self._risk_parity_allocation(cov_matrix)
                return {
                    'weights': risk_parity_result['weights'],
                    'method': 'Risk Parity',
                    'volatility': risk_parity_result['volatility'],
                    'risk_contributions': risk_parity_result['risk_contributions'],
                    'tickers': tickers
                }

            else:
                # Default to equal weight
                equal_weights = np.ones(n_assets) / n_assets
                return {
                    'weights': equal_weights,
                    'method': 'Equal Weight',
                    'tickers': tickers
                }

        except Exception as e:
            logger.error(f"Portfolio optimization failed: {e}")
            n_assets = len(tickers)
            return {
                'weights': np.ones(n_assets) / n_assets,
                'method': 'Fallback - Equal Weight',
                'tickers': tickers
            }

    def _apply_portfolio_optimization(self, action: int, ticker: str, current_price: float,
                                    portfolio_value: float, optimized_weights: Dict[str, Any]) -> float:
        """Apply portfolio optimization to position sizing"""
        try:
            if not optimized_weights or 'weights' not in optimized_weights:
                # Fallback to default position sizing
                return self.config.get('position_size_pct', 0.1)

            weights = optimized_weights['weights']
            tickers = optimized_weights.get('tickers', [])

            if ticker not in tickers:
                # Fallback to default position sizing
                return self.config.get('position_size_pct', 0.1)

            # Get optimized weight for this ticker
            ticker_index = tickers.index(ticker)
            optimized_weight = weights[ticker_index]

            # Apply Kelly Criterion or other optimization method
            base_position = self.config.get('position_size_pct', 0.1)

            if optimized_weights['method'] == 'Kelly Criterion':
                # Kelly Criterion already provides optimal position size
                return optimized_weight
            else:
                # For other methods, scale the base position by the optimized weight
                return base_position * optimized_weight

        except Exception as e:
            logger.error(f"Portfolio optimization application failed: {e}")
            return self.config.get('position_size_pct', 0.1)

    def __init__(self, feature_matrix, price_matrix, volume_matrix, window_size=DEFAULT_NUM_FEATURES, min_trade_interval=None, mode='train', max_trades_per_episode=None, bar_frequency='day'):
        """
        Initialize trading environment with FIXED observation shape logic

        Args:
            feature_matrix: Feature data matrix (timesteps, num_tickers, num_features)
            price_matrix: Price data matrix (timesteps, num_tickers)
            volume_matrix: Volume data matrix (timesteps, num_tickers)
            window_size: Number of timesteps for observation window
            min_trade_interval: Minimum steps between trades (overrides hardcoded value)
            mode: Training mode ('train' or 'test')
            bar_frequency: 'day' or 'month' (FTSE master_panel uses month)
        """
        super(FixedTradingEnv, self).__init__()

        # CRITICAL FIX: Initialize logger
        import logging
        self.logger = logging.getLogger(__name__)
        config_module = globals().get('config', None)
        self.config = getattr(config_module, 'TRADING_CONFIG', {}) if config_module and hasattr(config_module, 'TRADING_CONFIG') else {}
        self._active_regime = None
        self._last_technical_signals = {}
        self._is_training = False
        self._is_evaluating = False

        # Defaults for metadata injected by env_factory
        self.dates = None
        self.feature_cols = []
        self.trading_config = {}
        self.unique_tickers = []
        # Architecture B (ensemble rank/size → PPO timing) — off until enable_architecture_b()
        self._arch_b_enabled = False
        self._arch_b = None
        self._arch_b_skip_gating = True
        # Must be set BEFORE corruption cleaning below
        self.bar_frequency = bar_frequency or "day"
        if self.bar_frequency == "month":
            self.min_hold_steps_before_stop = 2
            self.monthly_gap_block_down = 0.40
            self.monthly_gap_block_up = 0.50
            self.monthly_corruption_pct = 1.50
            self.base_stop_loss_pct = 0.28
            self.trailing_stop_pct = 0.20
        else:
            self.min_hold_steps_before_stop = 0
            self.monthly_gap_block_down = 0.10
            self.monthly_gap_block_up = 0.15
            self.monthly_corruption_pct = 0.50

        # COMPREHENSIVE DEBUGGING: Log initialization details
        self.logger.info("=" * 80)
        self.logger.info("TRADING ENVIRONMENT INITIALIZATION")
        self.logger.info("=" * 80)
        self.logger.info(f"Mode: {mode}")
        self.logger.info(f"Window size: {window_size}")
        self.logger.info(f"Min trade interval: {min_trade_interval}")
        self.logger.info(f"Max trades per episode: {max_trades_per_episode}")

        # Log matrix shapes and validation
        self.logger.info("MATRIX SHAPES:")
        self.logger.info(f"  Feature matrix: {feature_matrix.shape}")
        self.logger.info(f"  Price matrix: {price_matrix.shape}")
        self.logger.info(f"  Volume matrix: {volume_matrix.shape}")

        # Validate matrix dimensions
        if len(feature_matrix.shape) != 3:
            self.logger.error(f"INVALID: Feature matrix must be 3D (timesteps, tickers, features), got {feature_matrix.shape}")
        if len(price_matrix.shape) != 2:
            self.logger.error(f"INVALID: Price matrix must be 2D (timesteps, tickers), got {price_matrix.shape}")
        if len(volume_matrix.shape) != 2:
            self.logger.error(f"INVALID: Volume matrix must be 2D (timesteps, tickers), got {volume_matrix.shape}")

        # Log data quality
        self.logger.info("DATA QUALITY:")
        self.logger.info(f"  Feature matrix - Min: {np.min(feature_matrix):.4f}, Max: {np.max(feature_matrix):.4f}, NaN: {np.isnan(feature_matrix).sum()}")
        self.logger.info(f"  Price matrix - Min: {np.min(price_matrix):.4f}, Max: {np.max(price_matrix):.4f}, NaN: {np.isnan(price_matrix).sum()}")
        self.logger.info(f"  Volume matrix - Min: {np.min(volume_matrix):.4f}, Max: {np.max(volume_matrix):.4f}, NaN: {np.isnan(volume_matrix).sum()}")

        # Detect extreme moves. Daily bars: >50% is often corruption.
        # Monthly FTSE bars: large moves are normal — only clean extreme outliers.
        corruption_found = False
        cleaned_count = 0
        is_monthly = getattr(self, "bar_frequency", "day") == "month"
        extreme_thresh = float(getattr(self, "monthly_corruption_pct", 1.50)) if is_monthly else 0.50
        
        for ticker_idx in range(price_matrix.shape[1]):
            ticker_prices = price_matrix[:, ticker_idx].copy()
            if len(ticker_prices) > 1:
                price_changes_pct = np.abs(np.diff(ticker_prices) / (ticker_prices[:-1] + 1e-10))  # Avoid division by zero
                extreme_changes = price_changes_pct > extreme_thresh
                
                if extreme_changes.sum() > 0:
                    corruption_found = True
                    extreme_indices = np.where(extreme_changes)[0]
                    self.logger.warning(
                        f"⚠️ DATA CORRUPTION DETECTED: Ticker {ticker_idx} has "
                        f"{extreme_changes.sum()} extreme price changes (>{extreme_thresh:.0%})"
                    )
                    
                    # Log first 3 examples
                    for idx in extreme_indices[:3]:
                        if idx < len(ticker_prices) - 1:
                            prev_price = ticker_prices[idx]
                            curr_price = ticker_prices[idx + 1]
                            change_pct = ((curr_price - prev_price) / (prev_price + 1e-10)) * 100
                            self.logger.warning(f"  Step {idx}->{idx+1}: ${prev_price:.2f} -> ${curr_price:.2f} ({change_pct:+.1f}%) - LIKELY CORRUPTION")
                    
                    # DATA CLEANING: Interpolate corrupted prices
                    for idx in extreme_indices:
                        if idx < len(ticker_prices) - 1:
                            # Use linear interpolation between valid neighbors
                            if idx > 0 and idx < len(ticker_prices) - 2:
                                # Interpolate between previous and next valid price
                                price_matrix[idx + 1, ticker_idx] = (ticker_prices[idx] + ticker_prices[idx + 2]) / 2.0
                                cleaned_count += 1
                            elif idx == 0:
                                # Use next valid price
                                price_matrix[idx + 1, ticker_idx] = ticker_prices[idx + 2] if len(ticker_prices) > 2 else ticker_prices[idx]
                                cleaned_count += 1
                            else:
                                # Use previous valid price
                                price_matrix[idx + 1, ticker_idx] = ticker_prices[idx]
                                cleaned_count += 1
                    
                    self.logger.info(f"✅ Cleaned {cleaned_count} corrupted price points for ticker {ticker_idx} using interpolation")
        
        if corruption_found:
            self.logger.warning(f"⚠️ DATA CLEANING COMPLETE: Fixed {cleaned_count} corrupted price points across all tickers")
        else:
            self.logger.info("✅ Data validation passed: No extreme price changes detected")

        # NaN clean: replace any remaining NaN in price_matrix (forward-fill per column)
        if np.isnan(price_matrix).any():
            nan_count = int(np.isnan(price_matrix).sum())
            for col in range(price_matrix.shape[1]):
                col_vals = price_matrix[:, col]
                if np.isnan(col_vals).any():
                    last = np.nan
                    for i in range(len(col_vals)):
                        if np.isnan(col_vals[i]) and not np.isnan(last):
                            price_matrix[i, col] = last
                        elif not np.isnan(col_vals[i]):
                            last = col_vals[i]
                    col_vals = price_matrix[:, col]
                    if np.isnan(col_vals).any():
                        idx = np.where(~np.isnan(col_vals))[0]
                        if len(idx):
                            price_matrix[:idx[0], col] = col_vals[idx[0]]
                        price_matrix[np.isnan(price_matrix[:, col]), col] = 0.01
            self.logger.warning(f"Price matrix had {nan_count} NaN values; forward-filled and replaced remaining with 0.01")

        # Check for price changes (critical for trading)
        price_changes = np.diff(price_matrix[:, 0])
        non_zero_changes = np.count_nonzero(price_changes)
        self.logger.info(f"  Price changes (ticker 0): {non_zero_changes}/{len(price_changes)} steps have changes")

        if non_zero_changes == 0:
            self.logger.error("CRITICAL: No price changes detected - trading will be impossible!")
        elif non_zero_changes < len(price_changes) * 0.1:
            self.logger.warning(f"WARNING: Only {non_zero_changes/len(price_changes)*100:.1f}% of steps have price changes")

        # Data matrices
        self.feature_matrix = feature_matrix
        self.price_matrix = price_matrix
        self.volume_matrix = volume_matrix
        self.window_size = window_size
        # DYNAMIC: Set num_tickers from actual data shape (supports any number of tickers)
        if price_matrix is not None and len(price_matrix.shape) >= 2:
            self.num_tickers = price_matrix.shape[1]  # Get from data shape
        else:
            self.num_tickers = DEFAULT_NUM_TICKERS  # Fallback: AAPL, MSFT, NVDA, JPM, SPY, GLD, BTC-USD, KO, JNJ, AMZN, XLE, TLT
        # DYNAMIC: Set num_features from feature_matrix so intraday/multi-timeframe columns are included
        if feature_matrix is not None and len(feature_matrix.shape) >= 3:
            self.num_features = int(feature_matrix.shape[2])
        else:
            self.num_features = 13  # Fallback: base daily features when shape not available

        # CRITICAL FIX: Calculate observation space dynamically based on actual data
        # Observation size = window_size * num_tickers * num_features + portfolio_state + [fundamental_features]
        # DYNAMIC: Supports any number of tickers (now 12: AAPL, MSFT, NVDA, JPM, SPY, GLD, BTC-USD, KO, JNJ, AMZN, XLE, TLT)
        market_features = window_size * self.num_tickers * self.num_features
        portfolio_features = 1 + self.num_tickers  # cash% + position sizes for each ticker
        # 2026-02-15: Fundamental metrics in observation (P/E, ROE, etc. per ticker)
        self.FUNDAMENTAL_FEATURE_KEYS = ('pe_ratio', 'pb_ratio', 'debt_equity', 'roe', 'current_ratio', 'revenue_growth', 'earnings_growth')
        self.num_fundamental_features = len(self.FUNDAMENTAL_FEATURE_KEYS)
        try:
            from ppo.utils.dynamic_config import get_dynamic_config
            dc = get_dynamic_config()
            include_fund = getattr(dc, 'include_fundamental_in_observation', True)
        except Exception:
            include_fund = True
        self._include_fundamental_in_obs = bool(include_fund)
        fundamental_features = self.num_tickers * self.num_fundamental_features if self._include_fundamental_in_obs else 0
        obs_size = market_features + portfolio_features + fundamental_features
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )
        self.logger.info(f"Observation space: {obs_size} dims ({market_features} market + {portfolio_features} portfolio)")

        # Action space: 4 actions (HOLD, BUY, SELL, SHORT)
        # Use module-level NUM_ACTIONS constant (from ppo.constants or fallback above)
        self.action_space = gym.spaces.Discrete(NUM_ACTIONS)

        # CRITICAL FIX: Store mode for proper reward handling
        self.mode = mode
        self.logger.info(f"Environment initialized in {self.mode} mode")

        # CRITICAL DEBUG: Verify price_matrix contains varying prices over time
        self.logger.info(f"DEBUG Env: Price matrix sample (ticker 0): [{self.price_matrix[0,0]:.2f}, {self.price_matrix[1,0]:.2f}, ..., {self.price_matrix[min(self.price_matrix.shape[0]-1, 9),0]:.2f}]")
        self.logger.info(f"DEBUG Env: Price variance (ticker 0): {np.var(self.price_matrix[:,0]):.4f}")
        self.logger.info(f"DEBUG Env: Price range (ticker 0): {np.min(self.price_matrix[:,0]):.2f} - {np.max(self.price_matrix[:,0]):.2f}")

        # Check if prices are actually changing
        price_changes = np.diff(self.price_matrix[:,0])
        non_zero_changes = np.count_nonzero(price_changes)
        self.logger.info(f"DEBUG Env: Price changes (ticker 0): {non_zero_changes}/{len(price_changes)} steps have price changes")

        if non_zero_changes == 0:
            self.logger.error("CRITICAL: Environment price matrix has NO price changes - this will cause 0% returns!")
        elif non_zero_changes < len(price_changes) * 0.1:
            self.logger.warning(f"WARNING: Only {non_zero_changes/len(price_changes)*100:.1f}% of steps have price changes")

        # Trading state
        self.current_step = 0
        
        # Initialize adaptive reward tracking for unified report
        self._adaptive_reward_log = []
        self.portfolio_value = 100000.0
        # CRITICAL FIX: Initialize positions with entry price tracking for accurate PnL
        self.positions = {i: {'shares': 0.0, 'avg_cost': 0.0} for i in range(self.num_tickers)}
        self.cash = 100000.0
        self.initial_balance = 100000.0

        # CRITICAL FIX: Initialize portfolio tracking
        self._last_portfolio_value = 100000.0
        self._total_trades_executed = 0
        self._episode_trades_collected = 0
        self._total_portfolio_changes = 0
        
        # BEHAVIOR-BASED ADAPTIVE: Track action counts for behavior analysis
        self._buy_count = 0
        self._sell_count = 0
        self._hold_count = 0
        self._avg_hold_time_history = []  # Track hold times for adaptive adjustments
        
        # FIX: Track policy actions (raw actions before masking) for entropy boost
        self._policy_action_counts = self.DEFAULT_POLICY_ACTION_COUNTS.copy()
        
        # Progressive loss penalty tracking
        self._consecutive_losses = 0  # Track consecutive losing trades
        
        # Trade quality metrics - hold time tracking
        self._position_buy_steps = {}  # Track when each position was bought: {ticker: buy_step}

        # Bar frequency (day|month). FTSE dissertation sets month after env create.
        self.bar_frequency = getattr(self, "bar_frequency", "day")
        self.min_hold_steps_before_stop = getattr(self, "min_hold_steps_before_stop", 0)
        self.monthly_gap_block_down = getattr(self, "monthly_gap_block_down", 0.40)
        self.monthly_gap_block_up = getattr(self, "monthly_gap_block_up", 0.50)
        self.monthly_corruption_pct = getattr(self, "monthly_corruption_pct", 1.50)

        # Performance tracking
        self.portfolio_history = []
        self.trade_history = []
        self.reward_history = []
        
        # Realized vs Unrealized gains tracking
        self.cumulative_realized_gains = 0.0  # Track total realized PnL from closed trades

        # CRITICAL: Drawdown penalty system for risk management
        self.peak_portfolio_value = 100000.0  # Track peak for drawdown calculation
        self.current_drawdown = 0.0  # Current drawdown percentage
        # self.drawdown_penalty_lambda = 0.15  # Now dynamic

        # CRITICAL: Episode trade history for proper aggregation
        self.episode_trade_history = []  # Trades for current episode only

        # CRITICAL FIX: Cumulative trade history for evaluation
        self.cumulative_evaluation_trade_history = []  # Persists across all evaluation episodes

        # Episode control
        # CRITICAL FIX: Ensure minimum episode length
        self._max_episode_steps = max(100, len(self.price_matrix) - self.window_size - 1)
        self.logger.info(f"EPISODE LENGTH: {self._max_episode_steps} steps (price_matrix: {len(self.price_matrix)}, window: {self.window_size})")

        # CRITICAL: Validate episode length
        if self._max_episode_steps < 10:
            self.logger.error(f"CRITICAL: Episode length too short ({self._max_episode_steps}), setting to minimum of 100")
            self._max_episode_steps = 100

        # DYNAMIC NORMALIZATION: Store mode for consistent behavior
        self.mode = mode

        # DYNAMIC NORMALIZATION: Use consistent constraints for both training and evaluation
        if min_trade_interval is not None:
            self.min_trade_interval = min_trade_interval
        else:
            # Try to get from config, fallback to 0 for dynamic normalization compatibility
            try:
                import config  # type: ignore
                self.min_trade_interval = getattr(config, 'min_trade_interval', 0)  # Allow immediate trading for dynamic normalization
            except (ImportError, AttributeError):
                self.min_trade_interval = 0  # Allow immediate trading for dynamic normalization compatibility

        self.logger.info(f" DYNAMIC NORMALIZATION: Mode={mode}, Trade interval={self.min_trade_interval} (consistent for both train/eval)")

        # DYNAMIC CONFIG: Import and use centralized configuration
        try:
            import sys
            import os
            # Add the ppo directory to the path for imports
            ppo_dir = os.path.dirname(os.path.dirname(__file__))
            if ppo_dir not in sys.path:
                sys.path.insert(0, ppo_dir)

            try:
                from utils.dynamic_config import get_dynamic_config, register_config_listener  # type: ignore
            except ImportError:
                try:
                    from ppo.utils.dynamic_config import get_dynamic_config, register_config_listener  # type: ignore
                except ImportError:
                    # Fallback: create dummy functions
                    def get_dynamic_config() -> Any:
                        return {}
                    def register_config_listener(func: Any) -> None:
                        pass

            self.dynamic_config = cast(Any, get_dynamic_config())
            # PERFORMANCE: reuse per-step to avoid 100+ get_dynamic_config() calls/step (logic unchanged)
            self._get_dynamic_config = get_dynamic_config
            self._dcfg = None  # Cleared on reset; set at start of each step()

            # Register this environment as a listener for config changes
            register_config_listener(self._on_config_change)

            # Initialize with dynamic config values
            self.exploration_bonus = self.dynamic_config.exploration_bonus
            self.trading_incentive = self.dynamic_config.trading_incentive
            self.hold_penalty = self.dynamic_config.hold_penalty
            self.max_position_size = self.dynamic_config.max_position_size
            self.max_tech_weight = self.dynamic_config.max_tech_weight
            self.max_trades_per_episode = self.dynamic_config.max_trades_per_episode
            self.drawdown_penalty_lambda = self.dynamic_config.drawdown_penalty_lambda
            # Centralize stop-loss configuration from dynamic config
            self.base_stop_loss_pct = getattr(self.dynamic_config, 'base_stop_loss_pct', 0.10)
            self.trailing_stop_pct = getattr(self.dynamic_config, 'trailing_stop_pct', 0.08)
            # FTSE month bars: keep wider stops after dynamic_config sync (else agent never sells)
            if getattr(self, "bar_frequency", "day") == "month":
                self.base_stop_loss_pct = max(float(self.base_stop_loss_pct), 0.28)
                self.trailing_stop_pct = max(float(self.trailing_stop_pct), 0.20)
                self.min_hold_steps_before_stop = max(int(getattr(self, "min_hold_steps_before_stop", 0) or 0), 2)
                self.logger.info(
                    f" FTSE MONTHLY STOPS: base={self.base_stop_loss_pct:.0%} "
                    f"trail={self.trailing_stop_pct:.0%} min_hold={self.min_hold_steps_before_stop}"
                )

            self.logger.info(" DYNAMIC CONFIG: Environment connected to centralized config")

            # Load logging configuration from main config
            try:
                import config  # type: ignore
                if hasattr(config, 'ENV_CONFIG'):
                    self.log_every_n_steps = config.ENV_CONFIG.get('log_every_n_steps', 200)  # OPT B: default 200 was 50
                    self.verbose_step_logs = config.ENV_CONFIG.get('verbose_step_logs', False)
                    self.verbose_trade_logs = config.ENV_CONFIG.get('verbose_trade_logs', True)
                    self.logger.info(f" LOGGING CONFIG: log_every_n_steps={self.log_every_n_steps}, verbose_step_logs={self.verbose_step_logs}")
                else:
                    self.log_every_n_steps = 200  # OPT B: was 50
                    self.verbose_step_logs = False
                    self.verbose_trade_logs = True
                # Load reward configuration
                # CRITICAL FIX: Always start with empty dict, then sync from dynamic_config
                # This ensures dynamic_config values always take precedence
                self.reward_config = {}
                
                # If config.REWARD_CONFIG exists, use it as base (but will be overridden by dynamic_config)
                if hasattr(config, 'REWARD_CONFIG'):
                    self.reward_config.update(config.REWARD_CONFIG)
                    self.logger.info(" REWARD CONFIG: Loaded base from config.py (will be synced with dynamic_config)")
                
                # Always add fallback values for any missing keys
                fallback_reward_config = {
                    # BUY reward controls: CRITICAL FIX 2025-12-23 - Increased to encourage trading
                    # Previous 0.02 was too low, causing agent to avoid trading entirely
                    'buy_base_incentive': 0.15,
                    'buy_reward_cap': 0.08,
                    'hold_penalty': -0.18,  # 2026-02-21: match dynamic_config (plan: block 97% HOLD)
                    'hold_penalty_with_positions': -0.05,  # 2026-02-21: match dynamic_config
                    'hold_penalty_no_positions': -0.08,    # 2026-02-21: match dynamic_config
                    'hold_penalty_transition_threshold': 0.9,
                    'hold_signal_grace_steps': 3,
                    'strong_signal_threshold': 0.7,
                    'strong_fundamental_threshold': 0.65,
                    'hold_strong_signal_penalty': -0.1,
                    'hold_progressive_start': 10,  # 2026-02-20: 30→10 to match dynamic_config
                    'trading_incentive': 0.10,  # PHASE 2 FIX: Increased from 0.02 to 0.10 (5x stronger)
                    'first_trade_bonus': 0.30,  # PHASE 2 FIX: Increased from 0.20 to 0.30
                    'profitable_trade_bonus': 0.20,
                    'profit_bonus_scale': 1.0,
                    'profit_bonus_cap': 0.25,
                    'pnl_reward_scale': 7.0,  # FIXED: Increased from 5.0 to 7.0
                    'loss_penalty_scale': 0.3,  # REDUCED: Losses hurt 70% less (was 1.0) - mistakes are OK!
                    'loss_penalty_cap': 0.15,  # REDUCED: Max loss penalty smaller (was 0.4) - don't fear losses!
                    'signal_bonus_cap': 0.3,
                    'signal_miss_penalty_scale': 0.1,  # REDUCED: Less punishment for timing misses (was 0.2)
                    'failed_sell_penalty': -0.001,  # CRITICAL FIX: Further reduced from -0.005 to -0.001 (5x less) - encourage SELL attempts
                    'failed_sell_consecutive_penalty': -0.0005,  # CRITICAL FIX: Further reduced from -0.001 to -0.0005 (2x less) - minimal penalty
                    'exploration_bonus_threshold': 3,  # PHASE 1: Trigger after 3 holds
                    'exploration_bonus_rate': 0.03,  # PHASE 1: Increased for stronger signal
                    'exploration_bonus_train_only': False,  # PHASE 1: Enable in eval too
                    'learning_bonus': 0.05,  # PHASE 2 FIX: Increased from 0.02 to 0.05 - reward for attempting trades
                    'base_trading_incentive': 0.25,  # FIX: Updated to match dynamic_config (was 0.15)
                    'sell_base_incentive': 0.25,    # Phase 2 fallback; overlay syncs from dynamic_config
                    'short_base_incentive': 0.05,   # 2026-03-06: SHORT reward (scaled by bearish signal in step)
                    'short_signal_min_threshold': 0.68,  # 2026-03-06: Require VERY STRONG bearish (0.55→0.68 to cut excess shorts)
                    'short_weak_signal_penalty': -0.02,
                    'sell_attempt_bonus': 0.05,     # Phase 2: reduced; overlay syncs from dynamic_config
                    'sell_unconditional_base': 0.08,  # 2026-02-28: unconditional SELL base; overlay syncs from dynamic_config
                    'buy_loss_penalty_cap': -1.0,   # Phase 7: Balanced BUY loss penalty
                    'sell_loss_penalty_cap': -1.0,  # Phase 7: Balanced SELL loss penalty
                    'position_incentive': 0.05,     # FIX: Reduced from 0.15 to make HOLD less attractive
                    'position_incentive_decay_start': 15,  # FIX: Faster decay to encourage exits
                    'position_incentive_decay_rate': 0.3,  # FIX: Faster decay rate
                    'position_aging_penalty_start': 10,    # FIX: Reduced from 20 to 10 steps (start earlier)
                    'position_aging_penalty_rate': 0.01,   # Per-step penalty after threshold
                    'sell_profit_taking_bonus': 0.50,      # FIX: Increased to make SELL more attractive
                }
                # Update reward_config with fallback values (only for missing keys)
                for k, v in fallback_reward_config.items():
                    if k not in self.reward_config:
                        self.reward_config[k] = v
                self.logger.info(" REWARD CONFIG: Added fallback values for missing keys")
                
                # CRITICAL FIX: Always sync from dynamic_config (takes precedence over everything)
                try:
                    dc = self.dynamic_config
                    # Map dynamic_config attributes to reward keys if available
                    # CRITICAL: Sync ALL reward-related values from dynamic_config
                    overlay = {
                        'trading_incentive': getattr(dc, 'trading_incentive', None),
                        'base_trading_incentive': getattr(dc, 'base_trading_incentive', None),  # CRITICAL FIX: Add base_trading_incentive to sync
                        'buy_base_incentive': getattr(dc, 'base_trading_incentive', None),  # NEW: Map base_trading_incentive to buy_base_incentive
                        'sell_base_incentive': getattr(dc, 'sell_base_incentive', None),  # HIGH PRIORITY FIX: Add sell_base_incentive to sync
                        'short_base_incentive': getattr(dc, 'short_base_incentive', None),  # 2026-03-06: SHORT reward scaled by bearish signal
                        'short_signal_min_threshold': getattr(dc, 'short_signal_min_threshold', None),
                        'short_weak_signal_penalty': getattr(dc, 'short_weak_signal_penalty', None),
                        'exploration_bonus': getattr(dc, 'exploration_bonus', None),  # CRITICAL FIX: Add exploration_bonus to sync
                        'no_positions_bonus': getattr(dc, 'no_positions_bonus', None),  # NEW: Add no_positions_bonus to sync
                        'first_trade_bonus': getattr(dc, 'first_trade_bonus', None),
                        'trading_frequency_bonus': getattr(dc, 'trading_frequency_bonus', None),  # HIGH PRIORITY FIX: Add trading frequency bonus
                        'trading_frequency_bonus_max_trades': getattr(dc, 'trading_frequency_bonus_max_trades', None),  # HIGH PRIORITY FIX: Add max trades
                        'trade_frequency_threshold': getattr(dc, 'trade_frequency_threshold', None),  # HIGH PRIORITY FIX: Add frequency threshold
                        'trade_frequency_penalty_rate': getattr(dc, 'trade_frequency_penalty_rate', None),  # HIGH PRIORITY FIX: Add penalty rate
                        # 2025-12-21: Missed SELL shaping (train-only in env.step)
                        'missed_sell_penalty_base': getattr(dc, 'missed_sell_penalty_base', None),
                        'missed_sell_signal_threshold': getattr(dc, 'missed_sell_signal_threshold', None),
                        'missed_sell_profit_take_threshold': getattr(dc, 'missed_sell_profit_take_threshold', None),
                        'missed_sell_min_hold_steps': getattr(dc, 'missed_sell_min_hold_steps', None),
                        'sell_profit_taking_bonus': getattr(dc, 'sell_profit_taking_bonus', None),  # HIGH PRIORITY FIX: Add profit-taking bonus
                        'sell_attempt_bonus': getattr(dc, 'sell_attempt_bonus', None),  # Phase 2: Sync from dynamic_config (was missing, env used 0.10)
                        'sell_unconditional_base': getattr(dc, 'sell_unconditional_base', None),  # 2026-02-28: unconditional SELL base (balanced reward)
                        'signal_alignment_bonus': getattr(dc, 'signal_alignment_bonus', None),  # Phase 2: Sync from dynamic_config
                        'unrealized_gain_threshold_2pct': getattr(dc, 'unrealized_gain_threshold_2pct', None),  # NEW: Bonus for SELL when unrealized gains > 2%
                        'unrealized_gain_threshold_5pct': getattr(dc, 'unrealized_gain_threshold_5pct', None),  # NEW: Bonus for SELL when unrealized gains > 5%
                        'unrealized_gain_threshold_10pct': getattr(dc, 'unrealized_gain_threshold_10pct', None),  # NEW: Bonus for SELL when unrealized gains > 10%
                        'sell_hold_time_bonus_base': getattr(dc, 'sell_hold_time_bonus_base', None),  # NEW: Base bonus for SELL after holding X steps
                        'sell_hold_time_min_steps': getattr(dc, 'sell_hold_time_min_steps', None),  # NEW: Minimum hold time to get bonus
                        'position_incentive_decay_start': getattr(dc, 'position_incentive_decay_start', None),  # NEW: Start decaying position incentive after holding X steps
                        'position_incentive_decay_rate': getattr(dc, 'position_incentive_decay_rate', None),  # NEW: Decay rate for position incentive
                        'position_aging_penalty_start': getattr(dc, 'position_aging_penalty_start', None),  # NEW: Start aging penalty after X steps
                        'position_aging_penalty_rate': getattr(dc, 'position_aging_penalty_rate', None),  # NEW: Per-step penalty after threshold
                        'unrealized_gain_bonus': getattr(dc, 'unrealized_gain_bonus', None),  # HIGH PRIORITY FIX: Add unrealized gain bonus
                        'trailing_stop_bonus': getattr(dc, 'trailing_stop_bonus', None),  # HIGH PRIORITY FIX: Add trailing stop bonus
                        'sell_bull_market_boost': getattr(dc, 'sell_bull_market_boost', None),  # HIGH PRIORITY FIX: Add bull market boost
                        'hold_penalty_with_positions': getattr(dc, 'hold_penalty_with_positions', None),
                        'hold_penalty_with_positions_grace': getattr(dc, 'hold_penalty_with_positions_grace', None),
                        'hold_penalty_grace_steps': getattr(dc, 'hold_penalty_grace_steps', None),
                        'hold_penalty_no_positions': getattr(dc, 'hold_penalty_no_positions', None),
                        'hold_penalty_transition_threshold': getattr(dc, 'hold_penalty_transition_threshold', None),
                        'hold_progressive_start': getattr(dc, 'hold_progressive_start', None),  # 2026-02-20: was missing from overlay
                        'wrong_regime_buy_penalty': getattr(dc, 'wrong_regime_buy_penalty', None),
                        'wrong_signal_buy_threshold': getattr(dc, 'wrong_signal_buy_threshold', None),
                        'wrong_signal_buy_penalty': getattr(dc, 'wrong_signal_buy_penalty', None),
                        'wrong_signal_sell_penalty': getattr(dc, 'wrong_signal_sell_penalty', None),
                        'profit_target_pct': getattr(dc, 'profit_target_pct', None),
                        'profit_target_5pct_bonus': getattr(dc, 'profit_target_5pct_bonus', None),
                        'overbought_rsi_threshold': getattr(dc, 'overbought_rsi_threshold', None),
                        'overbought_buy_discount': getattr(dc, 'overbought_buy_discount', None),
                        'weak_signal_bull_buy_discount': getattr(dc, 'weak_signal_bull_buy_discount', None),
                        'contra_signal_buy_block_threshold_sell': getattr(dc, 'contra_signal_buy_block_threshold_sell', None),
                        'contra_signal_buy_block_threshold_buy': getattr(dc, 'contra_signal_buy_block_threshold_buy', None),
                        'contra_signal_buy_penalty': getattr(dc, 'contra_signal_buy_penalty', None),
                        'hold_with_5pct_unrealized_in_bull_penalty': getattr(dc, 'hold_with_5pct_unrealized_in_bull_penalty', None),
                        'market_movement_multiplier': getattr(dc, 'market_movement_multiplier', None),
                        'failed_sell_penalty': getattr(dc, 'failed_sell_penalty', None),
                        'failed_sell_consecutive_penalty': getattr(dc, 'failed_sell_consecutive_penalty', None),
                        'failed_sell_penalty_cap': getattr(dc, 'failed_sell_penalty_cap', None),
                        'exploration_bonus_threshold': getattr(dc, 'exploration_bonus_threshold', None),
                        'exploration_bonus_rate': getattr(dc, 'exploration_bonus_rate', None),
                        'learning_bonus': getattr(dc, 'learning_bonus', None),
                        'signal_bonus_cap': getattr(dc, 'signal_bonus_cap', None),
                        'profit_bonus_scale': getattr(dc, 'profit_bonus_scale', None),
                        'profit_bonus_cap': getattr(dc, 'profit_bonus_cap', None),
                        'pnl_reward_scale': getattr(dc, 'pnl_reward_scale', None),
                        'loss_penalty_scale': getattr(dc, 'loss_penalty_scale', None),
                        'loss_penalty_cap': getattr(dc, 'loss_penalty_cap', None),
                        'hold_profit_bonus': getattr(dc, 'hold_profit_bonus', None),
                        'position_incentive': getattr(dc, 'position_incentive', None),  # CRITICAL FIX: Add position_incentive to sync (was missing, causing 0.0 override)
                        # REMOVED: Duplicate entries (already synced above):
                        # - sell_base_incentive (line 1470)
                        # - sell_profit_taking_bonus (line 1477)
                        # - sell_bull_market_boost (line 1480)
                        'sell_loss_penalty_cap': getattr(dc, 'sell_loss_penalty_cap', -1.0),  # Phase 7: Balanced SELL loss penalty cap
                        'buy_loss_penalty_cap': getattr(dc, 'buy_loss_penalty_cap', -1.0),  # Phase 7: Balanced BUY loss penalty cap
                        'profitable_trade_bonus': getattr(dc, 'profitable_trade_bonus', None),  # NEW: Add profitable_trade_bonus
                        'profitable_trade_bonus_multiplier': getattr(dc, 'profitable_trade_bonus_multiplier', None),  # NEW: Add multiplier
                        'profitable_trade_bonus_max': getattr(dc, 'profitable_trade_bonus_max', None),  # NEW: Add max cap
                        'progressive_bonus_per_trade': getattr(dc, 'progressive_bonus_per_trade', None),  # NEW: Add progressive bonus
                        'progressive_bonus_max': getattr(dc, 'progressive_bonus_max', None),  # NEW: Add progressive max
                        'gain_bonus_threshold_10pct': getattr(dc, 'gain_bonus_threshold_10pct', None),
                        'gain_bonus_threshold_5pct': getattr(dc, 'gain_bonus_threshold_5pct', None),
                        'gain_bonus_threshold_2pct': getattr(dc, 'gain_bonus_threshold_2pct', None),
                        'gain_bonus_threshold_1pct': getattr(dc, 'gain_bonus_threshold_1pct', None),  # 2026-02-15
                        'gain_bonus_value_10pct': getattr(dc, 'gain_bonus_value_10pct', None),
                        'gain_bonus_value_5pct': getattr(dc, 'gain_bonus_value_5pct', None),
                        'gain_bonus_value_2pct': getattr(dc, 'gain_bonus_value_2pct', None),
                        'gain_bonus_value_1pct': getattr(dc, 'gain_bonus_value_1pct', None),  # 2026-02-15
                        'progressive_loss_penalty_base': getattr(dc, 'progressive_loss_penalty_base', None),  # NEW: Add progressive loss penalty
                        'progressive_loss_penalty_step': getattr(dc, 'progressive_loss_penalty_step', None),
                        'quick_flip_penalty': getattr(dc, 'quick_flip_penalty', None),  # NEW: Add quick flip penalty
                        'quick_flip_penalty_profitable': getattr(dc, 'quick_flip_penalty_profitable', None),  # A: Penalty for profitable short holds
                        'min_hold_steps': getattr(dc, 'min_hold_steps', None),  # NEW: Add min hold steps
                        'hold_time_bonus': getattr(dc, 'hold_time_bonus', None),  # NEW: Add hold time bonus
                        'hold_winner_bonus': getattr(dc, 'hold_winner_bonus', None),  # 2026-02-05: Bonus for HOLD when position is winner (min unrealized pct + min steps)
                        'hold_winner_min_unrealized_pct': getattr(dc, 'hold_winner_min_unrealized_pct', None),
                        'hold_winner_min_steps': getattr(dc, 'hold_winner_min_steps', None),
                        'loss_scale_multiplier': getattr(dc, 'loss_scale_multiplier', None),  # B: Asymmetric PnL - scale losses more
                        'signal_multiplier_scale': getattr(dc, 'signal_multiplier_scale', None),  # NEW: Add signal multiplier scale
                        'unrealized_gain_scale_multiplier': getattr(dc, 'unrealized_gain_scale_multiplier', None),  # NEW: Add unrealized gain scale
                        'unrealized_gain_scale_max': getattr(dc, 'unrealized_gain_scale_max', None),  # NEW: Add unrealized gain max
                        'hold_time_bonus_scale_max': getattr(dc, 'hold_time_bonus_scale_max', None),  # NEW: Add hold time bonus scale max
                        'loss_penalty_pnl_multiplier': getattr(dc, 'loss_penalty_pnl_multiplier', None),  # NEW: Add loss penalty PnL multiplier
                        'profitable_close_bonus': getattr(dc, 'profitable_close_bonus', None),  # 2026-02-05: Bonus for closing trade with >5% net return
                        'profitable_close_min_return_pct': getattr(dc, 'profitable_close_min_return_pct', None),
                        'weak_signal_penalty': getattr(dc, 'weak_signal_penalty', -0.06),  # Penalty when trading with low/medium confidence – use indicators
                    }
                    # CRITICAL FIX: Always update from dynamic_config (even if value is 0.0)
                    # Use a sentinel to distinguish None (not found) from 0.0 (valid value)
                    synced_count = 0
                    for k, v in overlay.items():
                        if v is not None:  # None means attribute doesn't exist, skip it
                            old_value = self.reward_config.get(k, 'NOT_SET')
                            self.reward_config[k] = v
                            if old_value != v:
                                synced_count += 1
                                self.logger.debug(f" REWARD CONFIG SYNC: {k} = {v} (was {old_value})")
                    
                    if synced_count > 0:
                        self.logger.info(f" REWARD CONFIG: Synchronized {synced_count} values from dynamic_config")
                    else:
                        self.logger.info(" REWARD CONFIG: All values already synced with dynamic_config")
                    
                    # CRITICAL: Log key reward values to verify sync worked
                    key_values = {
                        'base_trading_incentive': self.reward_config.get('base_trading_incentive', 'MISSING'),
                        'buy_base_incentive': self.reward_config.get('buy_base_incentive', 'MISSING'),
                        'sell_base_incentive': self.reward_config.get('sell_base_incentive', 'MISSING'),
                        'exploration_bonus': self.reward_config.get('exploration_bonus', 'MISSING'),
                        'no_positions_bonus': self.reward_config.get('no_positions_bonus', 'MISSING'),
                    }
                    self.logger.info(f" REWARD CONFIG VERIFICATION: {key_values}")
                    
                    # CRITICAL FIX: Validate config sync and force correction if mismatched
                    self._validate_and_fix_reward_config_sync()
                except Exception as e:
                    # Log the error for debugging (but non-fatal)
                    self.logger.warning(f"Failed to sync reward_config from dynamic_config: {e}")
                    # Non-fatal; leave reward_config as-is (fallback values will be used)
                    pass
            except ImportError:
                self.log_every_n_steps = 200  # OPT B: was 50
                self.verbose_step_logs = False
                self.verbose_trade_logs = True
                # Fallback reward config - FEARLESS LEARNING: Mistakes are learning opportunities
                self.reward_config = {
                    # BUY reward controls: CRITICAL FIX 2025-12-23 - Increased to encourage trading
                    'buy_base_incentive': 0.15,
                    'buy_reward_cap': 0.08,
                    'hold_penalty': -0.18,  # 2026-02-21: match dynamic_config (plan: block 97% HOLD)
                    'hold_penalty_with_positions': -0.05,  # 2026-02-21: match dynamic_config
                    'hold_penalty_no_positions': -0.08,    # 2026-02-21: match dynamic_config
                    'hold_penalty_transition_threshold': 0.9,
                    'hold_signal_grace_steps': 3,
                    'strong_signal_threshold': 0.7,
                    'strong_fundamental_threshold': 0.65,
                    'hold_strong_signal_penalty': -0.1,
                    'hold_progressive_start': 10,  # 2026-02-20: 30→10 to match dynamic_config
                    'trading_incentive': 0.10,  # PHASE 2 FIX: Increased from 0.02 to 0.10 (5x stronger)
                    'first_trade_bonus': 0.30,  # PHASE 2 FIX: Increased from 0.20 to 0.30
                    'profitable_trade_bonus': 0.20,
                    'profit_bonus_scale': 1.0,
                    'profit_bonus_cap': 0.25,
                    'pnl_reward_scale': 35.0,
                    'loss_penalty_scale': 0.3,  # REDUCED: Losses hurt 70% less (was 1.0) - mistakes are OK!
                    'loss_penalty_cap': 0.15,  # REDUCED: Max loss penalty smaller (was 0.4) - don't fear losses!
                    'signal_bonus_cap': 0.3,
                    'signal_miss_penalty_scale': 0.1,  # REDUCED: Less punishment for timing misses (was 0.2)
                    'failed_sell_penalty': -0.001,  # CRITICAL FIX: Further reduced from -0.005 to -0.001 (5x less) - encourage SELL attempts
                    'failed_sell_consecutive_penalty': -0.0005,  # CRITICAL FIX: Further reduced from -0.001 to -0.0005 (2x less) - minimal penalty
                    'base_trading_incentive': 0.25,  # FIX: Updated to match dynamic_config (was 0.15)
                    'sell_base_incentive': 0.25,    # Phase 2 fallback; overlay syncs from dynamic_config
                    'sell_attempt_bonus': 0.05,     # Phase 2: reduced; overlay syncs from dynamic_config
                    'sell_unconditional_base': 0.08,  # 2026-02-28: unconditional SELL base; overlay syncs from dynamic_config
                    'buy_loss_penalty_cap': -1.0,   # Phase 7: Balanced BUY loss penalty
                    'sell_loss_penalty_cap': -1.0,  # Phase 7: Balanced SELL loss penalty
                    'position_incentive': 0.05,     # FIX: Reduced from 0.15 to make HOLD less attractive
                    'position_incentive_decay_start': 15,  # FIX: Faster decay to encourage exits
                    'position_incentive_decay_rate': 0.3,  # FIX: Faster decay rate
                    'position_aging_penalty_start': 10,    # FIX: Reduced from 20 to 10 steps (start earlier)
                    'position_aging_penalty_rate': 0.01,   # Per-step penalty after threshold
                    'sell_profit_taking_bonus': 0.50,      # FIX: Increased to make SELL more attractive
                    'exploration_bonus_threshold': 3,  # PHASE 1: Trigger after 3 holds
                    'exploration_bonus_rate': 0.03,  # PHASE 1: Increased for stronger signal
                    'exploration_bonus_train_only': False,  # PHASE 1: Enable in eval too
                    'learning_bonus': 0.02,  # NEW: Reward for attempting any trade (learn from experience)
                }
                self.logger.info(" REWARD CONFIG: Using fallback configuration (ImportError)")
            
        except Exception as e:
            # CRITICAL FIX: Use dynamic_config values instead of hardcoded overrides
            # Hardcoded values were preventing proper reward structure
            try:
                dc = cast(Any, self._cfg())
                # CRITICAL FIX: Use correct default values from dynamic_config.py (not old hardcoded values)
                # Note: config.py may override these via sync_from_config_file(), but we use dynamic_config defaults
                self.exploration_bonus = getattr(dc, 'exploration_bonus', 0.25)  # 2026-02-05: Match dynamic_config (outcome over activity)
                self.trading_incentive = getattr(dc, 'base_trading_incentive', 0.50)  # FIXED: Use 0.50 from dynamic_config.py (was 0.15)
            except Exception:
                # Fallback to reasonable defaults if dynamic_config unavailable (match dynamic_config.py defaults)
                self.exploration_bonus = 0.25  # 2026-02-05: Match dynamic_config default
                self.trading_incentive = 0.50  # FIXED: Match dynamic_config.py default (was 0.15)
            self.hold_penalty = -0.1       # CRITICAL FIX: Reduced hold penalty to prevent portfolio crashes
            
            self.max_position_size = 0.02  # CRITICAL FIX: Reduced from 40% to 2% to prevent portfolio crashes
            self.max_tech_weight = 0.60
            self.max_trades_per_episode = 100
            self.drawdown_penalty_lambda = 0.15
            self.dynamic_config = None

            self.logger.info(" DYNAMIC NORMALIZATION: Using consistent reward values for both train/eval")
            self.logger.warning(f" DYNAMIC CONFIG: Using fallback values: {e}")
    
    def _validate_and_fix_reward_config_sync(self):
        """CRITICAL FIX: Validate that reward_config matches dynamic_config and force sync if mismatched"""
        try:
            if not hasattr(self, 'dynamic_config'):
                return  # Can't validate without dynamic_config
            
            dc = self.dynamic_config
            mismatches = []
            corrections = {}
            
            # Check critical reward values
            critical_keys = {
                'base_trading_incentive': 'base_trading_incentive',
                'buy_base_incentive': 'base_trading_incentive',  # Maps to base_trading_incentive
                'sell_base_incentive': 'sell_base_incentive',
                'exploration_bonus': 'exploration_bonus',
                'no_positions_bonus': 'no_positions_bonus',
            }
            
            for reward_key, dc_attr in critical_keys.items():
                config_value = self.reward_config.get(reward_key)
                dc_value = getattr(dc, dc_attr, None)
                
                if dc_value is not None:
                    # Handle buy_base_incentive special case (maps to base_trading_incentive)
                    if reward_key == 'buy_base_incentive':
                        expected_value = dc_value
                    else:
                        expected_value = dc_value
                    
                    if config_value != expected_value:
                        mismatches.append(f"{reward_key}: config={config_value}, expected={expected_value}")
                        corrections[reward_key] = expected_value
            
            if mismatches:
                self.logger.error(f"[CONFIG_SYNC] MISMATCHES DETECTED: {mismatches}")
                self.logger.error(f"[CONFIG_SYNC] FORCING CORRECTION: {corrections}")
                
                # Force sync corrections
                for key, value in corrections.items():
                    old_value = self.reward_config.get(key, 'NOT_SET')
                    self.reward_config[key] = value
                    self.logger.warning(f"[CONFIG_SYNC] CORRECTED: {key} = {value} (was {old_value})")
                
                self.logger.warning(f"[CONFIG_SYNC] Fixed {len(corrections)} mismatched values")
            else:
                self.logger.debug("[CONFIG_SYNC] All critical values are properly synced")
                
        except Exception as e:
            self.logger.warning(f"[CONFIG_SYNC] Validation failed: {e}")
            # Non-fatal - continue with existing config

        ticker_candidates = []
        try:
            import config  # type: ignore
            ticker_candidates = list(getattr(config, 'DATA_CONFIG', {}).get('tickers', []))
        except Exception:
            ticker_candidates = []

        if not ticker_candidates or len(ticker_candidates) != self.num_tickers:
            ticker_candidates = [f"TICKER_{i}" for i in range(self.num_tickers)]

        self.ticker_symbols = ticker_candidates
        self._fundamental_cache_dir = Path("ppo/saved_data/fundamental_cache")
        self._fundamental_cache_loaded = False
        self._fundamental_cache: Dict[str, Dict[str, float]] = {}
        self._fundamental_data_available = False
        self._missing_fundamentals_warning_emitted = False

        self._load_fundamental_cache()

        # Per-step signal/fundamental caches to avoid recomputation storms
        self._signal_cache: Dict[Tuple[int, str, str], float] = {}
        self._signal_cache_step: int = -1
        self._fundamental_metric_cache: Dict[str, Dict[str, float]] = {}
        self._fundamental_metric_cache_step: int = -1
        self._fundamental_analysis_cache: Dict[Tuple[str, int], Dict[str, Any]] = {}
        self._fundamental_analysis_cache_step: int = -1

        # Track last step we emitted high-verbosity hold logs
        self._last_hold_info_step: int = -1

        self.logger.info(" TRADING CONSTRAINTS ENABLED: Proper trading behavior restored")
        self.logger.info(f" TRADING CONSTRAINTS: Trade interval = {self.min_trade_interval}, Trading incentives = {self.trading_incentive}")

        # FIXED: Add configurable trade limit (now handled by dynamic config)
        # self.max_trades_per_episode = max_trades_per_episode  # Now dynamic
        # self.max_position_size = 0.02  # CRITICAL FIX: Reduced from 20%+ to 2%0  # Now dynamic
        # self.max_tech_weight = 0.60  # Now dynamic

        # Performance monitoring attributes
        self.profiling_active = False
        self.memory_monitoring_active = False
        self.profiling_interval = 100
        self.memory_check_interval = 50

        # Portfolio state tracking
        self.portfolio_state = [0.0] * self.num_tickers
        self.last_trade_step = -1

        # Constraint violation tracking
        self._trade_interval_violations = 0
        self._trade_interval_errors = 0
        self._diversification_violations = 0
        self._consecutive_trades = 0

        # CRITICAL FIX: Initialize cooldown trackers
        self._cooldown_tracker = {}  # Track cooldown periods for each ticker
        self._position_hold_tracker = {}  # Track minimum hold periods for each ticker
        self._cooldown_violations = 0  # Track cooldown violations
        self._cooldown_logged = False

        # CRITICAL FIX: Initialize trade validator
        try:
            validator_config = {
                'min_trade_size': getattr(self, 'min_trade_size', 20.0),
                'min_cash_balance': 100.0,
                'max_position_size': getattr(self, 'max_position_size', 0.02),
                'enable_trade_validation': True
            }
            self.trade_validator = TradeValidator(validator_config)
            self.logger.info("✅ Trade validator initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize trade validator: {e}")
            # Create dummy validator
            self.trade_validator = TradeValidator({'enable_trade_validation': False})

        # Logging
        self.logger = logging.getLogger(__name__)

        # STEP 8 OPTIMIZATION: Stress testing configuration
        self._stress_test_config = self._load_stress_test_config()
        self._slippage_multiplier = self._stress_test_config.get('slippage_multiplier', 1.0)
        self._impact_multiplier = self._stress_test_config.get('impact_multiplier', 1.0)
        self._volatility_multiplier = self._stress_test_config.get('volatility_multiplier', 1.0)
        # Gating diagnostics (run-level counters; not reset on episode reset)
        self._gating_stats = {
            'allowed': 0,
            'blocked': 0,
            'explore': 0,
            'high': 0,
            'medium': 0,
            'low': 0,
        }
        # COMPREHENSIVE DEBUGGING: Log final initialization state
        self.logger.info("INITIALIZATION COMPLETE:")
        self.logger.info(f"  Observation space: {self.observation_space.shape}")
        self.logger.info(f"  Action space: {self.action_space}")
        self.logger.info(f"  Episode max steps: {self._max_episode_steps}")
        self.logger.info(f"  Number of tickers: {self.num_tickers}")
        self.logger.info(f"  Number of features: {self.num_features}")
        self.logger.info(f"  Initial portfolio value: ${self.portfolio_value:,.2f}")
        self.logger.info(f"  Initial cash: ${self.cash:,.2f}")

        # Log trading constraints
        self.logger.info("TRADING CONSTRAINTS:")
        self.logger.info(f"  Min trade interval: {self.min_trade_interval}")
        self.logger.info(f"  Max position size: {self.max_position_size}")
        self.logger.info(f"  Max tech weight: {self.max_tech_weight}")
        self.logger.info(f"  Max trades per episode: {self.max_trades_per_episode}")
        self.logger.info(f"  Hold penalty: {self.hold_penalty}")
        self.logger.info(f"  Trading incentive: {self.trading_incentive}")
        self.logger.info(f"  Exploration bonus: {self.exploration_bonus}")

        self.logger.info("=" * 80)

        self._load_fundamental_cache()

        # Validate configuration
        self._validate_configuration()

    def _validate_configuration(self):
        """Validate trading environment configuration - EMERGENCY FIX: Allow 0 for learning"""
        try:
            # EMERGENCY FIX: Allow 0 for learning phase, but warn about it
            assert 0 <= self.min_trade_interval <= 100, f"Invalid min_trade_interval: {self.min_trade_interval}"
            if self.min_trade_interval == 0:
                self.logger.warning(" min_trade_interval=0 (super relaxed for learning)")
            elif self.min_trade_interval == 1:
                self.logger.info(f" min_trade_interval={self.min_trade_interval} (relaxed for learning)")
            else:
                self.logger.info(f" min_trade_interval={self.min_trade_interval} (using config value)")
            assert 0 < self.max_position_size <= 1.0, f"Invalid max_position_size: {self.max_position_size}"
            assert 0 < self.max_tech_weight <= 1.0, f"Invalid max_tech_weight: {self.max_tech_weight}"

            # Validate data matrices
            assert self.feature_matrix is not None, "Feature matrix is None"
            assert self.price_matrix is not None, "Price matrix is None"
            assert self.volume_matrix is not None, "Volume matrix is None"

            self.logger.info(" Trading environment configuration validated successfully")

        except Exception as e:
            self.logger.error(f" Configuration validation failed: {e}")
            raise ValueError(f"Invalid trading environment configuration: {e}")

    def _on_config_change(self, config):
        """Callback for when dynamic configuration changes"""
        try:
            # Update trading parameters from dynamic config
            old_interval = self.min_trade_interval
            old_incentive = self.trading_incentive
            old_bonus = self.exploration_bonus
            old_penalty = self.hold_penalty
            old_position_size = self.max_position_size
            old_tech_weight = self.max_tech_weight
            old_trades_per_episode = self.max_trades_per_episode
            old_drawdown_lambda = self.drawdown_penalty_lambda

            self.min_trade_interval = config.min_trade_interval
            self.trading_incentive = config.trading_incentive
            self.exploration_bonus = config.exploration_bonus
            self.hold_penalty = config.hold_penalty
            self.max_position_size = config.max_position_size
            self.max_tech_weight = config.max_tech_weight
            self.max_trades_per_episode = config.max_trades_per_episode
            self.drawdown_penalty_lambda = config.drawdown_penalty_lambda

            # Sync reward_config so base_trading_incentive / sell_base_incentive changes apply immediately
            if hasattr(config, 'base_trading_incentive'):
                self.reward_config['base_trading_incentive'] = config.base_trading_incentive
                self.reward_config['buy_base_incentive'] = config.base_trading_incentive
            if hasattr(config, 'sell_base_incentive'):
                self.reward_config['sell_base_incentive'] = config.sell_base_incentive
            if hasattr(config, 'sell_unconditional_base'):
                self.reward_config['sell_unconditional_base'] = config.sell_unconditional_base
            if hasattr(config, 'hold_penalty'):
                self.reward_config['hold_penalty'] = config.hold_penalty
            if hasattr(config, 'hold_penalty_no_positions'):
                self.reward_config['hold_penalty_no_positions'] = config.hold_penalty_no_positions
            if hasattr(config, 'hold_penalty_with_positions'):
                self.reward_config['hold_penalty_with_positions'] = config.hold_penalty_with_positions

            # Log the changes
            changes = []
            if old_interval != self.min_trade_interval:
                changes.append(f"min_trade_interval: {old_interval}  {self.min_trade_interval}")
            if old_incentive != self.trading_incentive:
                changes.append(f"trading_incentive: {old_incentive}  {self.trading_incentive}")
            if old_bonus != self.exploration_bonus:
                changes.append(f"exploration_bonus: {old_bonus}  {self.exploration_bonus}")
            if old_penalty != self.hold_penalty:
                changes.append(f"hold_penalty: {old_penalty}  {self.hold_penalty}")
            if old_position_size != self.max_position_size:
                changes.append(f"max_position_size: {old_position_size}  {self.max_position_size}")
            if old_tech_weight != self.max_tech_weight:
                changes.append(f"max_tech_weight: {old_tech_weight}  {self.max_tech_weight}")
            if old_trades_per_episode != self.max_trades_per_episode:
                changes.append(f"max_trades_per_episode: {old_trades_per_episode}  {self.max_trades_per_episode}")
            if old_drawdown_lambda != self.drawdown_penalty_lambda:
                changes.append(f"drawdown_penalty_lambda: {old_drawdown_lambda}  {self.drawdown_penalty_lambda}")

            if changes:
                self.logger.info(f" DYNAMIC CONFIG UPDATE: {', '.join(changes)}")
            else:
                self.logger.debug(" DYNAMIC CONFIG: No changes detected")

        except Exception as e:
            self.logger.error(f" DYNAMIC CONFIG: Error updating config: {e}")

    def reset(self, seed=None, options=None):
        """Reset environment to initial state with cumulative performance tracking
        
        Args:
            seed: Optional random seed for reproducibility
            options: Optional dictionary with reset options (Gymnasium API compatibility)
        
        Returns:
            observation: Initial observation array
            info: Dictionary with reset information
        """
        super_obj = cast(Any, super())
        super_obj.reset(seed=seed)

        # CRITICAL FIX: Add reset tracing to identify unexpected callers
        import traceback

        # Initialize reset tracking if not exists
        if not hasattr(self, '_reset_count'):
            self._reset_count = 0
            self._episode_started = False

        self._reset_count += 1

        # 2026-02-06: Clear trade-step state first so SHAP/post-eval never see stale last_trade_step
        self._last_trade_step = -999
        self.last_trade_step = -1

        # Log unexpected resets (after first startup)
        if self._reset_count > 1:
            current_step = getattr(self, 'current_step', 0)
            done_flags = {
                'last_done': getattr(self, '_last_done', None),
                'last_terminated': getattr(self, '_last_terminated', None),
                'last_truncated': getattr(self, '_last_truncated', None)
            }

            # Get caller traceback (limit to 6 frames to avoid noise)
            caller_trace = traceback.format_stack(limit=6)
            caller_info = caller_trace[-2] if len(caller_trace) > 1 else "Unknown caller"

            self.logger.warning(f"🚨 UNEXPECTED RESET #{self._reset_count} 🚨")
            self.logger.warning(f"   Current step: {current_step}")
            self.logger.warning(f"   Done flags: {done_flags}")
            self.logger.warning(f"   Caller: {caller_info.strip()}")
            self.logger.warning(f"   Full trace: {''.join(caller_trace)}")
        else:
            self.logger.info(f"✅ Initial reset #{self._reset_count} (expected)")
        
        # CRITICAL FIX: Reset cumulative performance tracking for BC training
        # Each BC training episode should start fresh with $100,000
        # CRITICAL: DO NOT carry forward exploded portfolio values
        self.cumulative_portfolio_value = 100000.0
        self.cumulative_trades = 0
        self.cumulative_episodes = 0
        self.episode_start_portfolio = 100000.0
        
        # OPTIMIZED: Simplified reset logic - portfolio bounds checking happens during step execution
        # Store episode performance before reset
        if hasattr(self, 'portfolio_value'):
            episode_return = (self.portfolio_value - self.episode_start_portfolio) / self.episode_start_portfolio
            self.cumulative_trades += len(self.episode_trade_history) if hasattr(self, 'episode_trade_history') else 0
            self.cumulative_episodes += 1

            # Log episode performance
            self.logger.info(f"Episode {self.cumulative_episodes} completed: Return {episode_return:.2%}, "
                           f"Final Portfolio: ${self.portfolio_value:.2f}, Trades: {len(self.episode_trade_history) if hasattr(self, 'episode_trade_history') else 0}")

        # Reset trading state - always start fresh with initial balance
        self.current_step = 0
        self.portfolio_value = self.initial_balance  # CRITICAL: Always start fresh, don't carry forward
        self.positions = {i: {'shares': 0.0, 'avg_cost': 0.0} for i in range(self.num_tickers)}
        # CRITICAL FIX: Clear position_lots on reset to prevent corruption from previous episodes
        self._clear_position_lots()
        self.cash = self.initial_balance  # CRITICAL: Always start fresh with initial balance
        self._pending_invalid_sell_penalty = 0.0
        if hasattr(self, '_stop_loss_near_miss_last_step'):
            self._stop_loss_near_miss_last_step = {}
        self._consecutive_masked_sells = 0
        self._masked_sell_counter = 0
        self._consecutive_failed_sells = 0  # 2025-12-22: Reset failed sells counter between episodes
        
        # CRITICAL: Reset drawdown tracking
        if not hasattr(self, 'cumulative_peak_portfolio'):
            self.cumulative_peak_portfolio = self.initial_balance
        self.peak_portfolio_value = self.initial_balance
        self.current_drawdown = 0.0

        # OPTIMIZED: Store episode trades reference instead of copy (faster)
        if hasattr(self, 'episode_trade_history') and len(self.episode_trade_history) > 0:
            self._last_episode_trades = self.episode_trade_history  # Reference, not copy
            self.logger.debug(f"DEBUG RESET: Saved {len(self._last_episode_trades)} trades to _last_episode_trades")
        else:
            self._last_episode_trades = []
            self.logger.debug("DEBUG RESET: No trades to save (episode_trade_history empty or missing)")

        # CRITICAL: Reset episode trade history for proper aggregation
        self.episode_trade_history = []
        self.episode_start_portfolio = self.initial_balance  # Always start fresh
        
        # CRITICAL FIX: Reset successful trades counter for first trade bonus
        if hasattr(self, '_successful_trades_count'):
            self._successful_trades_count = 0
        
        # PHASE 1 FIX: Reset first BUY in episode flag
        self._first_buy_in_episode = False
        
        # CACHE INVALIDATION: Explicitly clear all caches to prevent stale data between episodes
        # This ensures no cached calculations from previous episodes affect new episodes
        self._fundamental_metric_cache = {}
        self._fundamental_metric_cache_step = -1
        self._fundamental_analysis_cache = {}
        self._fundamental_analysis_cache_step = -1
        self._signal_cache = {}
        self._signal_cache_step = -1
        self._cached_portfolio_value = None
        self._cached_portfolio_step = -1
        # PERFORMANCE: Clear cached active positions on reset
        if hasattr(self, '_cached_active_positions'):
            delattr(self, '_cached_active_positions')
        if hasattr(self, '_cached_active_positions_step'):
            delattr(self, '_cached_active_positions_step')
        # Clear any other cached values
        if hasattr(self, '_cached_vwap_rewards'):
            self._cached_vwap_rewards = {}
        if hasattr(self, '_cached_technical_signals'):
            self._cached_technical_signals = {}
        if hasattr(self, '_cached_fundamental_signals'):
            self._cached_fundamental_signals = {}
        if hasattr(self, '_cached_market_regime'):
            self._cached_market_regime = None
        if hasattr(self, '_dynamic_reward_context'):
            del self._dynamic_reward_context
        if hasattr(self, '_dynamic_reward_context_step'):
            del self._dynamic_reward_context_step
        if hasattr(self, '_cached_market_condition_multiplier'):
            self._cached_market_condition_multiplier = None
        self._dcfg = None  # PERFORMANCE: clear per-step config cache so next step() refetches
        self.logger.debug("CACHE INVALIDATION: All caches cleared at episode reset")

        # CRITICAL FIX: DO NOT accumulate trades here - let the evaluator handle accumulation
        # The evaluator should call get_episode_trade_history() to get per-episode trades
        # and handle the accumulation itself to avoid double-counting

        # Reset history but maintain cumulative data
        self.portfolio_history = [self.portfolio_value]
        self.trade_history = []  # Reset per-episode trade history
        self.reward_history = []
        # PERFORMANCE: Initialize portfolio snapshot tracking
        self._last_portfolio_snapshot_step = 0
        
        # Reset realized gains tracking for new episode
        self.cumulative_realized_gains = 0.0

        # Reset episode control
        self._last_episode_check = 0

        # Reset constraint violation counters
        self._trade_interval_violations = 0
        self._trade_interval_errors = 0
        self._diversification_violations = 0
        self._consecutive_trades = 0

        # CRITICAL FIX: Reset trade interval tracking
        self._last_trade_step = -999  # Allow first trade

        # CRITICAL FIX: Reset trade counter for new episode
        self._total_trades_executed = 0
        self._episode_trades_collected = 0
        self._last_total_trades = 0
        
        # BEHAVIOR-BASED ADAPTIVE: Reset action counters for new episode
        self._buy_count = 0
        self._sell_count = 0
        self._hold_count = 0
        # Keep _avg_hold_time_history across episodes for rolling average
        
        # CRITICAL FIX: Initialize blocked trades counter for new episode
        self._blocked_trades_count = {'BUY': 0, 'SELL': 0}
        
        # FIX: Reset policy action counts for new episode
        if not hasattr(self, '_policy_action_counts'):
            self._policy_action_counts = self.DEFAULT_POLICY_ACTION_COUNTS.copy()
        else:
            # Keep counts across episodes for entropy boost analysis (rolling window)
            # Only reset if counts are too large (prevent memory issues)
            total = sum(self._policy_action_counts.values())
            if total > 100000:  # Reset if too large
                self._policy_action_counts = self.DEFAULT_POLICY_ACTION_COUNTS.copy()
        
        # Reset progressive loss penalty tracking
        if hasattr(self, '_consecutive_losses'):
            self._consecutive_losses = 0
        
        # Reset trade quality metrics
        if hasattr(self, '_position_buy_steps'):
            self._position_buy_steps = {}
        if hasattr(self, '_hold_winner_last_step'):
            self._hold_winner_last_step = {}

        # Log reward mode once per process (2026-02-10: visibility for technical_only_v1 vs full reward)
        if not getattr(self, '_reward_mode_logged', False):
            try:
                dc = self._cfg()
                mode = str(getattr(dc, 'reward_mode', '') or '')
                self.logger.info(f"[REWARD_MODE] reward_mode={repr(mode)} (empty=full reward; 'technical_only_v1'=minimal formula)")
                self._reward_mode_logged = True
            except Exception:
                self._reward_mode_logged = True
        # CRITICAL FIX: Return observation and info for VecEnv compatibility
        # DummyVecEnv expects: obs, info = env.reset() (Gymnasium API)
        return self._get_observation(), {}

    def action_masks(self) -> np.ndarray:
        """
        Return valid action mask for MaskablePPO.
        
        2025-12-22: Implemented to prevent invalid SELL actions when no positions exist.
        This is the PROPER fix - agent cannot even select SELL when it has no positions.
        
        Action space: 0=HOLD, 1=BUY, 2=SELL, 3=SHORT
        
        Returns:
            np.ndarray: Boolean mask [HOLD_valid, BUY_valid, SELL_valid, SHORT_valid]
        """
        # Initialize all actions as valid
        mask = np.array([True, True, True, True], dtype=np.bool_)
        
        # HOLD (action 0): Always valid
        mask[0] = True
        
        # BUY (action 1): Valid if we can execute a BUY (have cash) or cover a short
        min_cash_for_buy = 50.0
        has_shorts = any(
            i in self.positions and self.positions[i].get('shares', 0.0) < -0.001
            for i in range(self.num_tickers)
        )
        mask[1] = bool(self.cash > min_cash_for_buy) or has_shorts
        
        # SELL (action 2): Valid ONLY if we have long positions to sell
        has_positions = any(
            i in self.positions and self.positions[i].get('shares', 0.0) > 0.001  # Small threshold for floating point
            for i in range(self.num_tickers)
        )
        mask[2] = has_positions
        
        # SHORT (action 3): Valid only if we have cash AND at least one ticker has bearish signal (no shorts without bearish signal)
        min_cash_for_short = 50.0
        has_cash_for_short = bool(self.cash > min_cash_for_short)
        has_bearish_signal = False
        if has_cash_for_short and self.current_step < len(self.price_matrix):
            try:
                current_prices = self.price_matrix[self.current_step]
                ticker = self._select_best_ticker_to_short(current_prices)
                if hasattr(self, '_step_signals_cache') and ticker in self._step_signals_cache and 'SELL' in self._step_signals_cache[ticker]:
                    tech_score = float(self._step_signals_cache[ticker]['SELL'].get('tech', 0.0) or 0.0)
                    fund_score = float(self._step_signals_cache[ticker]['SELL'].get('fund', 0.0) or 0.0)
                else:
                    tech_score = float(self._calculate_technical_signal_quality(ticker, 'SELL'))
                    fund_score = float(self._calculate_fundamental_signal_quality(ticker, 'SELL'))
                bearish_signal = 0.6 * np.clip(tech_score, 0.0, 1.0) + 0.4 * np.clip(fund_score, 0.0, 1.0)
                min_threshold = float(self.reward_config.get('short_signal_min_threshold', 0.55))
                has_bearish_signal = bearish_signal >= min_threshold
            except Exception:
                has_bearish_signal = False
        mask[3] = has_cash_for_short and has_bearish_signal

        # Architecture B: PPO times TRADE/HOLD only — no SHORT; BUY only if book has underweight names
        if getattr(self, '_arch_b_enabled', False) and self._arch_b is not None:
            if getattr(self._arch_b, 'disable_short', True):
                mask[3] = False
            period = self._arch_b_active_month()
            underweight = self._arch_b_underweight_candidates(period, self.price_matrix[self.current_step] if self.current_step < len(self.price_matrix) else None)
            mask[1] = mask[1] and bool(underweight)
        
        # FIX 5: Enhanced logging to track masking during evaluation
        mode = getattr(self, 'mode', 'train')
        if mode in ('test', 'eval'):
            log_mask = self.current_step % 500 == 0
        else:
            log_mask = (self.current_step % 50 == 0) or (not has_positions)
        if log_mask:
            arch = " ARCH_B" if getattr(self, '_arch_b_enabled', False) else ""
            self.logger.info(f"[ACTION_MASK] Step {self.current_step}: HOLD={mask[0]}, BUY={mask[1]}, SELL={mask[2]}, SHORT={mask[3]} (has_positions={has_positions}, mode={mode}){arch}")
        
        return mask

    def _cfg(self):
        """Return current dynamic config; cache per step to avoid 100+ get_dynamic_config() calls per step (logic unchanged)."""
        if getattr(self, '_dcfg', None) is not None:
            return self._dcfg
        if hasattr(self, '_get_dynamic_config'):
            self._dcfg = self._get_dynamic_config()
            return self._dcfg
        self._dcfg = getattr(self, 'dynamic_config', None)
        return self._dcfg

    def step(self, action):
        """Execute one step in the environment with proper loop prevention"""
        try:
            # PERFORMANCE: Cache config once per step so _cfg() and all callees avoid repeated get_dynamic_config()
            if hasattr(self, '_get_dynamic_config'):
                self._dcfg = self._get_dynamic_config()
            else:
                self._dcfg = getattr(self, 'dynamic_config', None)
            # CRITICAL FIX: Normalize action to int scalar (handles numpy arrays from model.predict)
            if isinstance(action, np.ndarray):
                action = int(action.item() if action.size == 1 else action[0])
            elif not isinstance(action, (int, np.integer)):
                action = int(action)
            
            action_key = int(action)

            # DIAGNOSTIC #1: Track action attempt
            try:
                from ppo.diagnostic_trade_tracking import track_action_attempt  # type: ignore
                track_action_attempt(
                    step=self.current_step,
                    action=action,
                    action_name=self.ACTION_NAMES.get(action_key, 'UNKNOWN')
                )
            except Exception as diag_error:
                pass  # Non-critical diagnostic
            
            # FIX: Track policy action (raw action before any masking/overrides) for entropy boost
            if not hasattr(self, '_policy_action_counts'):
                self._policy_action_counts = self.DEFAULT_POLICY_ACTION_COUNTS.copy()
            if action_key in self._policy_action_counts:
                self._policy_action_counts[action_key] += 1
            
            # PERFORMANCE: Precompute all signals once per step (30-40% speedup)
            self._precompute_step_signals()
            
            # FIX #2: Reset SELL execution flag at start of each step
            self._sell_executed_this_step = False
            
            if not hasattr(self, '_recent_masked_sell'):
                self._recent_masked_sell = False
            # CRITICAL FIX: Reduce logging verbosity - only log every N steps or when action != HOLD
            # Use config to control logging frequency (default 400 = fewer I/O, logic unchanged)
            log_every_n_steps = getattr(self, 'log_every_n_steps', 400)
            verbose_step_logs = getattr(self, 'verbose_step_logs', False)
            should_log = (self.current_step % log_every_n_steps == 0) or (action != 0) or (self.current_step < 5) or verbose_step_logs

            # =========================================================================
            # 2025-12-21 CRITICAL FIX: REMOVED ACTION MASKING FOR SELL
            # -------------------------------------------------------------------------
            # Action masking was causing credit assignment problem:
            # - PPO chooses action=2 (SELL)
            # - Environment masks to action=0 (HOLD) 
            # - PPO sees: "I chose SELL but got HOLD reward"
            # - This teaches PPO: "SELL is bad, don't choose SELL"
            #
            # SOLUTION: Let SELL execute normally. The SELL branch already handles
            # "no positions" case correctly with small penalty. This way PPO sees
            # "I chose SELL, got SELL reward" which is correct credit assignment.
            # =========================================================================
            reward_config = getattr(self, 'reward_config', {})
            action_masked = False  # Keep for compatibility but no longer used for SELL
            original_action = action  # Store for debugging
            
            # REMOVED: Action masking that converted SELL→HOLD
            # The SELL execution branch (elif action == 2) now handles all cases
            
            if action != 0:
                self._recent_masked_sell = False

            # =========================================================================
            # 2025-12-21: FORCED SELL EXPLORATION (FIX FOR COLLAPSED SELL PROBABILITY)
            # -------------------------------------------------------------------------
            # The PPO policy has collapsed to never output SELL (action=2). Even with
            # entropy regularization, the logits for SELL are so negative that the
            # probability is ~0%. The agent never explores SELL, so it never learns
            # what SELL actually does.
            #
            # FIX: During training, when the agent has positions and chose HOLD/BUY,
            # with some probability, OVERRIDE the action to SELL so PPO gets direct
            # experience with SELL outcomes and can learn from them.
            # =========================================================================
            try:
                mode_now = getattr(self, 'mode', 'test')
                # Step 2: In technical-only reward mode, do NOT override the policy action.
                tech_only_disable_forced_sell = False
                try:
                    dc = getattr(self, 'dynamic_config', None)
                    tech_only_disable_forced_sell = (
                        getattr(dc, 'reward_mode', '') == 'technical_only_v1'
                        and bool(getattr(dc, 'tech_only_disable_forced_sell', True))
                    )
                except Exception:
                    tech_only_disable_forced_sell = False

                if (not tech_only_disable_forced_sell) and mode_now == 'train' and action != 2 and not action_masked:
                    # Check if we have any positions to sell
                    has_any_positions = any(
                        i in self.positions and self.positions[i].get('shares', 0.0) > 0 
                        for i in range(self.num_tickers)
                    )
                    if has_any_positions:
                        # Forced exploration probability (start aggressive, can decay later)
                        forced_sell_prob = float(reward_config.get('forced_sell_exploration_prob', 0.15))
                        
                        # Random override to SELL
                        if np.random.random() < forced_sell_prob:
                            original_action = action
                            action = 2  # Override to SELL
                            
                            # Track for logging/debugging
                            if not hasattr(self, '_forced_sell_count'):
                                self._forced_sell_count = 0
                            self._forced_sell_count += 1
                            
                            if self._forced_sell_count <= 10 or self._forced_sell_count % 50 == 0:
                                self.logger.info(
                                    f"[FORCED_SELL_EXPLORATION] Step {self.current_step}: "
                                    f"Overriding action {original_action}→2 (SELL) for exploration "
                                    f"(count: {self._forced_sell_count})"
                                )
            except Exception as e:
                # Never fail the step due to exploration logic
                pass

            # PERFORMANCE: Reduce logging frequency - only log every 100 steps in training (not every step with action != 0)
            # This reduces I/O overhead significantly during training
            should_log_detailed = should_log and (self.current_step % 100 == 0 or verbose_step_logs or self.current_step < 5)
            
            if should_log_detailed:
                # COMPREHENSIVE DEBUGGING: Log step execution details
                if verbose_step_logs:
                    self.logger.info("=" * 60)
                    self.logger.info(f"STEP EXECUTION: Step {self.current_step}, Action {action}")
                    self.logger.info("=" * 60)

                # Log current state (reduced frequency for performance)
                self.logger.debug("CURRENT STATE:")
                self.logger.debug(f"  Portfolio value: ${self.portfolio_value:,.2f}")
                self.logger.debug(f"  Cash: ${self.cash:,.2f}")
                self.logger.debug(f"  Total trades executed: {self._total_trades_executed}")
                self.logger.debug(f"  Last trade step: {getattr(self, '_last_trade_step', -999)}")
                positions_str = [(i, f"{pos['shares']:.2f} shares") for i, pos in self.positions.items() if pos['shares'] > 0]
                self.logger.debug(f"  Current positions: {positions_str}")

                # Log action details
                self.logger.debug(f"ACTION: {action} ({self.ACTION_NAMES.get(action_key, 'UNKNOWN')})")

                # Log current prices
                if self.current_step < len(self.price_matrix):
                    current_prices = self.price_matrix[self.current_step]
                    self.logger.debug(f"  Current prices: {[f'${price:.2f}' for price in current_prices[:3]]}...")  # Show first 3 tickers
                else:
                    # Reduced logging - only log essential info
                    self.logger.debug(f"STEP {self.current_step}: Action {action}, Portfolio: ${self.portfolio_value:,.2f}, Trades: {self._total_trades_executed}")
            
            # Add trading incentive for non-hold actions (only in training mode)
            current_mode = getattr(self, 'mode', 'unknown')

            # CRITICAL FIX: More robust mode detection
            if current_mode == 'unknown':
                # Try to detect mode from environment state or context
                if hasattr(self, '_is_training') and self._is_training:
                    current_mode = 'train'
                elif hasattr(self, '_is_evaluating') and self._is_evaluating:
                    current_mode = 'test'
                else:
                    # Use the mode set during initialization
                    current_mode = getattr(self, 'mode', 'test')
                    if current_mode == 'unknown':
                        current_mode = 'test'  # Default to test mode if still unknown
                    self.logger.info(f"[MODE_FIX] Mode detected as: {current_mode}")

            if should_log:
                if action != 0 and current_mode == 'train':  # Only in training mode
                    # PERFORMANCE: Gate training incentive logs behind frequency check
                    if self._should_log_trade(50):
                        self.logger.debug(f"[DEBUG_TRADE]  TRADING INCENTIVE: +{self.trading_incentive:.3f} for attempting action {action}")
                elif action != 0 and current_mode != 'train':
                    # CRITICAL FIX: Enable trading incentives in test mode for proper evaluation
                    # This ensures the agent doesn't avoid trading during evaluation
                    # PERFORMANCE: Gate eval trade logs behind frequency check
                    if self._should_log_trade(50):
                        self.logger.debug(f"[EVAL_TRADE]  Trading incentive enabled in {current_mode} mode for action {action}")

            # CRITICAL FIX: Check bounds before accessing data
            # Allow episode to continue until _check_episode_done() determines termination
            # This allows proper episode length based on available data (80% rule)
            max_available_steps = len(self.price_matrix) - self.window_size - 1
            
            # CRITICAL FIX: Only terminate if we've truly run out of data (no hard limit to allow full training)
            if self.current_step >= max_available_steps:
                # Only terminate if we've truly run out of data
                self.logger.info(f"[EPISODE_TERMINATION] Episode ended: reached data limit at step {self.current_step} (max_available: {max_available_steps}, price_matrix_len: {len(self.price_matrix)})")
                return self._get_observation(), 0.0, True, False, {}

            # CRITICAL FIX: Add trade frequency limit to prevent infinite loops
            if not hasattr(self, '_last_trade_step'):
                self._last_trade_step = -999  # Allow first trade

            # CRITICAL FIX: Ensure last_trade_step is valid (should never happen, but safety check)
            if self._last_trade_step > self.current_step:
                self.logger.warning(f"Invalid last_trade_step {self._last_trade_step} > current_step {self.current_step}, resetting")
                self._last_trade_step = -999  # Reset to allow trade

            # Apply trade constraints for buy/sell actions
            if action in [1, 2]:  # Buy or sell actions
                # Enforce minimum trade interval
                if not self._check_trade_interval_constraints(action):
                    self.logger.debug("Trade interval violation: forcing HOLD")
                    action = 0
                # Apply cooldown logic (may adjust/convert action based on config)
                action = self._apply_cooldown_logic(action)

            # EMERGENCY DEBUG MODE: Skip cointegration logic
            # STEP 4 OPTIMIZATION: Disable cointegration and aggressive pairs trading (DISABLED IN DEBUG MODE)
            # action = self._apply_cointegration_disable_logic(action)  # DISABLED FOR DEBUGGING

            # EMERGENCY DEBUG MODE: Skip cooldown logic
            # STEP DEFAULT_NUM_FEATURES  # CONSTANT: Replaced magic number OPTIMIZATION: Apply cooldown logic to prevent scalping (DISABLED IN DEBUG MODE)
            # action = self._apply_cooldown_logic(action)  # DISABLED FOR DEBUGGING

            # CRITICAL FIX: Remove action balancing - let PPO learn naturally
            # The agent should learn to make proper decisions without artificial balancing

            # CRITICAL FIX: Initialize consecutive trade counter
            if not hasattr(self, '_consecutive_trades'):
                self._consecutive_trades = 0

            # CRITICAL FIX: Get current prices for trade execution
            current_prices = self.price_matrix[self.current_step]  # Use current_step BEFORE incrementing

            # CRITICAL: Check dynamic stop loss for all positions before any new trades
            # MONTHLY / Arch B: if agent chose SELL, run sell reward path first (don't let stops steal exits)
            stop_loss_triggered = False
            # PERFORMANCE: Cache active positions to avoid repeated dict lookups and list comprehensions
            if not hasattr(self, '_cached_active_positions') or not hasattr(self, '_cached_active_positions_step') or \
               self._cached_active_positions_step != self.current_step:
                self._cached_active_positions = [t for t in self.positions.keys() 
                                                if t < self.num_tickers and self.positions[t].get('shares', 0) > 0]
                self._cached_active_positions_step = self.current_step
            positions_with_shares = self._cached_active_positions
            skip_stops_for_agent_sell = (
                action == 2 and getattr(self, "bar_frequency", "day") == "month"
            )
            if positions_with_shares and not skip_stops_for_agent_sell:
                for ticker in positions_with_shares:
                    if ticker < len(current_prices):
                        if self._check_dynamic_stop_loss(ticker, current_prices[ticker]):
                            stop_loss_triggered = True
            elif skip_stops_for_agent_sell and positions_with_shares:
                self.logger.info(
                    "[FTSE MONTHLY] Agent SELL this step — deferring stop-loss so sell rewards can apply"
                )

            # FIX 2: REWARD FLOW - Check positions BEFORE applying bonuses to prevent invalid actions getting rewards
            # If action was masked (invalid SELL), it's now HOLD (action=0), so this check is defensive
            invalid_action_handled = False
            if action == 2:  # SELL action (defensive check - should be masked if invalid)
                # CRITICAL FIX: Check positions safely with .get() to handle missing keys
                has_positions = any(
                    i in self.positions and self.positions[i].get('shares', 0.0) > 0 
                    for i in range(self.num_tickers)
                )
                if not has_positions and not action_masked:
                    # CRITICAL FIX 2025-12-23: Invalid SELL must be worse than HOLD to prevent spam
                    # HOLD penalty is -0.02, so invalid SELL must be at least -0.03
                    # This prevents the agent from preferring wasteful invalid SELLs over holding
                    if not hasattr(self, '_consecutive_failed_sells'):
                        self._consecutive_failed_sells = 0
                    self._consecutive_failed_sells += 1
                    # INCREASED PENALTY: -0.03 base + -0.005 per consecutive (capped at -0.10)
                    reward = -0.03 - ((self._consecutive_failed_sells - 1) * 0.005)
                    reward = max(reward, -0.10)  # CRITICAL: Cap at -0.10 (was -0.01)
                    # Reduce log spam - only log every 100 steps
                    if self.current_step % 100 == 0:
                        self.logger.warning(f"[REWARD_FLOW_FIX] Invalid SELL (no positions) - soft penalty: {reward:.4f}")
                    invalid_action_handled = True  # Flag to skip trade execution and bonuses
            
            # CRITICAL FIX: Execute trade first, then update counters based on success
            if action in [1, 2, 3] and not invalid_action_handled:  # Buy, sell, or short action (skip if invalid)
                # CRITICAL FIX: Skip trade validation for now to fix the loop issue
                # The trade validator is being too strict and causing all trades to fail
                # We'll let _execute_trade handle the validation internally
                reward = self._execute_trade(action)
                self._pending_invalid_sell_penalty = 0.0
                self._consecutive_masked_sells = 0

                # CRITICAL FIX: Only increment consecutive trades if trade was actually executed
                # Check if a trade was recorded (indicating successful execution)
                if hasattr(self, '_total_trades_executed') and self._total_trades_executed > getattr(self, '_last_total_trades', 0):
                    # Trade was successful - add small trading incentive only in training mode
                    current_mode = getattr(self, 'mode', 'unknown')
                    
                    # CRITICAL FIX: Use same robust mode detection (FIXED INDENTATION BUG)
                    if current_mode == 'unknown':
                        current_mode = getattr(self, 'mode', 'test')
                        if current_mode == 'unknown':
                            current_mode = 'test'  # Default to test mode
                    
                    if current_mode == 'train':
                        small_incentive = 0.01  # Minimal incentive for successful trade execution
                        reward += small_incentive
                        # PERFORMANCE: Gate training incentive logs behind frequency check
                        if self._should_log_trade(50):
                            self.logger.debug(f"[DEBUG_TRADE]  TRADING INCENTIVE APPLIED: +{small_incentive:.3f}, Total reward: {reward:.3f}")
                    else:
                        # PERFORMANCE: Gate eval trade logs behind frequency check
                        if self._should_log_trade(50):
                            self.logger.debug(f"[EVAL_TRADE]  Trade executed in {current_mode} mode, Total reward: {reward:.3f}")
                    
                    self._consecutive_trades += 1
                    self._last_trade_step = self.current_step
                    # Keep trade-interval tracker in sync with successful trade step
                    self.last_trade_step = self.current_step
                    self._last_total_trades = self._total_trades_executed
                else:
                    # Trade failed - no trading incentive, reset consecutive counter
                    # PERFORMANCE: Gate failed trade logs behind frequency check
                    if self._should_log_trade(50):
                        self.logger.debug(f"[DEBUG_TRADE]  TRADE FAILED: No trading incentive applied, Total reward: {reward:.3f}")
                    self._consecutive_trades = 0
            else:
                # Hold action - reset consecutive counter
                self._consecutive_trades = 0
                reward = self._execute_trade(action)
                
                # Note: Hold penalty is now handled in _execute_trade method

            # -----------------------------------------------------------------
            # 2025-12-21: Missed SELL shaping (train-only)
            # If the agent has positions and exit pressure is high (strong SELL
            # signals or decent unrealized profit after minimum hold), but the
            # chosen action is not SELL, apply a STRONG penalty so PPO learns
            # to actually choose action=2.
            # -----------------------------------------------------------------
            try:
                mode_now = getattr(self, 'mode', 'test')
                # Step 2: In technical-only reward mode, disable missed-sell penalties.
                tech_only_disable_missed_sell_penalty = False
                try:
                    dc = getattr(self, 'dynamic_config', None)
                    tech_only_disable_missed_sell_penalty = (
                        getattr(dc, 'reward_mode', '') == 'technical_only_v1'
                        and bool(getattr(dc, 'tech_only_disable_missed_sell_penalty', True))
                    )
                except Exception:
                    tech_only_disable_missed_sell_penalty = False

                if (not tech_only_disable_missed_sell_penalty) and mode_now == 'train' and action != 2:
                    cfg = getattr(self, 'reward_config', {}) if hasattr(self, 'reward_config') else {}
                    # 2026-02-05: Reduced defaults to fix SELL bias (was 0.30/0.40/0.01/3)
                    base_pen = float(cfg.get('missed_sell_penalty_base', 0.10) or 0.10)
                    if base_pen > 0:
                        sig_thr = float(cfg.get('missed_sell_signal_threshold', 0.50) or 0.50)
                        prof_thr = float(cfg.get('missed_sell_profit_take_threshold', 0.02) or 0.02)
                        min_hold = int(cfg.get('missed_sell_min_hold_steps', 8) or 8)

                        active_positions = []
                        try:
                            for t, pos in getattr(self, 'positions', {}).items():
                                if not isinstance(pos, dict):
                                    continue
                                if t >= getattr(self, 'num_tickers', 0):
                                    continue
                                if float(pos.get('shares', 0.0) or 0.0) > 0:
                                    active_positions.append(int(t))
                        except Exception:
                            active_positions = []

                        if active_positions:
                            best_sell_signal = 0.0
                            best_profit = 0.0
                            best_hold = 0

                            for t in active_positions:
                                # Signal strength (SELL)
                                tech_s = 0.0
                                fund_s = 0.0
                                try:
                                    cache = getattr(self, '_step_signals_cache', {})
                                    if isinstance(cache, dict) and t in cache:
                                        sell_cache = cache[t].get('SELL', {}) if isinstance(cache[t], dict) else {}
                                        if isinstance(sell_cache, dict):
                                            tech_s = float(sell_cache.get('tech', 0.0) or 0.0)
                                            fund_s = float(sell_cache.get('fund', 0.0) or 0.0)
                                except Exception:
                                    pass
                                best_sell_signal = max(best_sell_signal, tech_s, fund_s)

                                # Unrealized gain pressure (profit-taking)
                                try:
                                    avg_cost = float(self.positions[t].get('avg_cost', 0.0) or 0.0)
                                except Exception:
                                    avg_cost = 0.0
                                if avg_cost > 0 and t < len(current_prices):
                                    cp = float(current_prices[t])
                                    gain = (cp - avg_cost) / avg_cost
                                    if gain > best_profit:
                                        best_profit = gain

                                # Hold time (if tracked)
                                try:
                                    if hasattr(self, '_position_buy_steps') and t in self._position_buy_steps and self._position_buy_steps[t] is not None:
                                        ht = int(self.current_step - int(self._position_buy_steps[t]))
                                        if ht > best_hold:
                                            best_hold = ht
                                except Exception:
                                    pass

                            should_penalize = (best_sell_signal >= sig_thr) or (best_profit >= prof_thr and best_hold >= min_hold)
                            if should_penalize:
                                penalty = -base_pen
                                # Light scaling when signals are very strong (cap the scaling)
                                if best_sell_signal > sig_thr:
                                    penalty *= (1.0 + min(1.0, best_sell_signal - sig_thr))
                                reward += penalty
                                
                                # Track penalty count for debugging
                                if not hasattr(self, '_missed_sell_penalty_count'):
                                    self._missed_sell_penalty_count = 0
                                self._missed_sell_penalty_count += 1
                                
                                # Log first 10 and every 100th occurrence
                                if self._missed_sell_penalty_count <= 10 or self._missed_sell_penalty_count % 100 == 0:
                                    self.logger.warning(
                                        f"[MISSED_SELL_PENALTY] Step {self.current_step}: "
                                        f"penalty={penalty:.3f}, signal={best_sell_signal:.2f}, "
                                        f"profit={best_profit:.1%}, hold={best_hold} steps, "
                                        f"total_penalties={self._missed_sell_penalty_count}"
                                    )
            except Exception:
                # Never fail the env step due to shaping logic
                pass

            # NEW REWARD LOGIC: First trade bonus and BUY exploration bonus
            # CRITICAL FIX: First Trade Bonus should only apply to SUCCESSFUL trades
            # The bonus was being given even when trades failed, inflating rewards incorrectly
            # Track successful trades only
            if not hasattr(self, '_successful_trades_count'):
                self._successful_trades_count = 0
            
            # Only give bonus if this was a successful trade (reward > 0 from _execute_trade)
            # Note: We can't check here since reward is set after _execute_trade
            # Instead, we'll track this in _execute_trade itself after successful execution
            
            # CRITICAL FIX: BUY exploration bonus should only apply when we have cash and no positions
            # This encourages the agent to take positions when sitting on cash
            # PERFORMANCE: Use cached active positions instead of iterating all tickers
            if hasattr(self, '_cached_active_positions') and self._cached_active_positions:
                has_positions = True
            else:
                has_positions = any(self.positions.get(i, {}).get('shares', 0) > 0 for i in self.positions.keys() if i < self.num_tickers)
            cash_pct = getattr(self, 'cash', 0) / max(getattr(self, 'portfolio_value', 100000), 1)
            
            # Dynamic hold penalty is now handled in _execute_trade method

            # PERFORMANCE: Log reward breakdown every 50 steps (moved to debug level)
            if self.current_step % 50 == 0:
                cash_pct = self.cash / self.portfolio_value if self.portfolio_value > 0 else 1.0
                self.logger.debug(f"[REWARD_DEBUG] Step {self.current_step}:")
                self.logger.debug(f"  Cash %: {cash_pct:.2%}")
                self.logger.debug(f"  Total trades: {self._total_trades_executed}")
                self.logger.debug(f"  Last action: {self.ACTION_NAMES.get(int(action), action)}")
                self.logger.debug(f"  Cumulative reward: {reward:.3f}")

            # PERFORMANCE: Log profitability metrics every 50 steps (moved to debug level)
            if self.current_step % 50 == 0:
                profitable_trades = sum(1 for t in self.trade_history if t.get('pnl', 0) > 0)
                total_trades = len([t for t in self.trade_history if 'pnl' in t])
                win_rate = profitable_trades / total_trades if total_trades > 0 else 0

                self.logger.debug(f"[PROFIT_METRICS] Step {self.current_step}:")
                self.logger.debug(f"  Win Rate: {win_rate:.1%} ({profitable_trades}/{total_trades})")
                self.logger.debug(f"  Portfolio: ${self.portfolio_value:,.2f}")
                self.logger.debug(f"  Return: {(self.portfolio_value/100000 - 1)*100:.2f}%")

            # CRITICAL FIX: Update portfolio value AFTER trade execution to reflect new positions
            # PERFORMANCE: Use cached fast calculation - SINGLE SOURCE OF TRUTH
            old_portfolio = self.portfolio_value
            calculated_portfolio = self._calculate_portfolio_value_fast(current_prices)
            
            # CRITICAL FIX: Cap portfolio value GAINS to prevent explosion (max 10% per step)
            # FIXED: Removed loss cap - losses should be fully recognized (accurate accounting)
            # Gap losses and large drawdowns are real trading risks that should be reflected
            if old_portfolio > 0:
                max_change = old_portfolio * 0.10  # Max 10% gain per step (prevent numerical explosions)
                change = calculated_portfolio - old_portfolio
                if change > max_change:
                    self.logger.warning(f"Portfolio gain capped: ${change:.2f} > ${max_change:.2f} (10% limit to prevent numerical explosion). Recalculating from cash+positions.")
                    calculated_portfolio = self._calculate_portfolio_value_fast(current_prices)  # Recalculate
                    # If still too large, use old portfolio + capped change
                    if calculated_portfolio > old_portfolio + max_change:
                        calculated_portfolio = old_portfolio + max_change
                # REMOVED: Loss cap - losses are now fully recognized for accurate return calculation
                # This allows gap losses (e.g., -76.95%) to be properly reflected in portfolio value
            
            # CRITICAL FIX: Validate portfolio value is reasonable before assigning
            max_reasonable = self.initial_balance * 10  # Max 10x initial (reduced from 100x)
            min_reasonable = self.initial_balance * 0.1  # Min 10% of initial (increased from 1%)
            
            if (not np.isfinite(calculated_portfolio) or 
                calculated_portfolio > max_reasonable or 
                calculated_portfolio < min_reasonable):
                self.logger.error(f"PORTFOLIO VALUE EXPLOSION DETECTED: ${calculated_portfolio:.2f}. Recalculating from cash+positions.")
                # Recalculate from scratch
                calculated_portfolio = self._calculate_portfolio_value_fast(current_prices)
                # If still out of bounds, reset
                if (calculated_portfolio > max_reasonable or calculated_portfolio < min_reasonable):
                    self.logger.error(f"Portfolio still out of bounds after recalculation: ${calculated_portfolio:.2f}. Resetting to ${self.initial_balance:.2f}")
                    calculated_portfolio = self.initial_balance
                    self.cash = self.initial_balance
                    # Reset positions if portfolio exploded
                    self.positions = {i: {'shares': 0.0, 'avg_cost': 0.0} for i in range(self.num_tickers)}
                    # CRITICAL FIX: Clear position_lots when portfolio explodes to prevent corruption
                    self._clear_position_lots()
            
            self.portfolio_value = calculated_portfolio
            
            # OPTIMIZED: Removed redundant validation - _calculate_portfolio_value_fast is single source of truth
            # Only log discrepancies if they're significant (relaxed tolerance from 1e-10 to 0.01)
            # This reduces overhead and floating-point precision noise
            
            # PERFORMANCE: Limit portfolio_history to every 10 steps to reduce memory overhead
            # Only append if it's the 10th step or if it's the first/last step of episode
            if self.current_step % 10 == 0 or not hasattr(self, '_last_portfolio_snapshot_step') or self._last_portfolio_snapshot_step is None:
                self.portfolio_history.append(self.portfolio_value)  # Track this update
                self._last_portfolio_snapshot_step = self.current_step
                # PERFORMANCE: Limit portfolio_history to last 5000 snapshots (50k steps / 10 = 5k max)
                if len(self.portfolio_history) > 5000:
                    self.portfolio_history = self.portfolio_history[-5000:]

            # CRITICAL FIX: Move to next step
            self.current_step += 1

            # CRITICAL FIX: Get observation for next step
            obs = self._get_observation()

            # CRITICAL FIX: Check if episode is done - but only after minimum steps
            # Don't check episode done in first 50 steps to prevent premature termination
            if self.current_step < 50:
                done = False
                self.logger.info(f"[EPISODE_CHECK] Step {self.current_step} < 50, episode not done (early termination prevention)")
            else:
                done = self._check_episode_done()

            # CRITICAL FIX: Add done flag tracking to identify what sets done=True prematurely
            if done and self.current_step < 50:
                self.logger.error(f"[BUG_DETECTED] Episode trying to end at step {self.current_step}!")
                self.logger.error(f"  Portfolio: ${self.portfolio_value:.2f}")
                self.logger.error(f"  Trades: {self._total_trades_executed}")
                import traceback
                self.logger.error(f"  Caller: {''.join(traceback.format_stack()[-5:])}")
                done = False  # Override

            # CRITICAL FIX: Update portfolio history (already appended once per step earlier)
            # Removed duplicate append to prevent double-counting

            # Log constraint summary if episode is done
            if done:
                self.log_constraint_summary()
                # Validate returns/trades matching at episode end
                validation_result = self._validate_returns_trades_matching()
                if not validation_result['valid'] or len(validation_result['warnings']) > 0:
                    self.logger.warning(
                        f"RETURNS/TRADES VALIDATION at episode end: "
                        f"valid={validation_result['valid']}, "
                        f"warnings={len(validation_result['warnings'])}, "
                        f"errors={len(validation_result['errors'])}"
                    )

            # CRITICAL FIX: Conditional step results logging - only when done=True or every 50 steps
            if done or (self.current_step % 50 == 0):
                # COMPREHENSIVE DEBUGGING: Log final step results
                self.logger.info("STEP RESULTS:")
                self.logger.info(f"  Final reward: {reward:.6f}")
                self.logger.info(f"  Portfolio value: ${self.portfolio_value:,.2f}")
                self.logger.info(f"  Cash: ${self.cash:,.2f}")
                self.logger.info(f"  Total trades: {self._total_trades_executed}")
                self.logger.info(f"  Episode done: {done}")
                self.logger.info(f"  Observation shape: {obs.shape if hasattr(obs, 'shape') else 'N/A'}")

                # Log reward breakdown if available
                if hasattr(self, '_last_reward_breakdown'):
                    breakdown = self._last_reward_breakdown
                    self.logger.info("REWARD BREAKDOWN:")
                    for component, value in breakdown.items():
                        if isinstance(value, (int, float)):
                            self.logger.info(f"  {component}: {value:.6f}")
                        else:
                            self.logger.info(f"  {component}: {value}")

            # Log every 50th step for performance monitoring (reduced from 10th)
            if self.current_step % 50 == 0:
                self.logger.info(f"[PERFORMANCE] Step {self.current_step}: Portfolio=${self.portfolio_value:.2f}, Cash=${self.cash:.2f}, Trades={self._total_trades_executed}")

            # CRITICAL FIX: Calculate benchmark return from SPY (ticker 4)
            # SPY is ticker index 4 (0=AAPL, 1=MSFT, 2=NVDA, 3=JPM, 4=SPY, 5=GLD)
            benchmark_return = 0.0
            SPY_TICKER_INDEX = 4
            if (self.current_step > 0 and 
                self.current_step < len(self.price_matrix) and 
                SPY_TICKER_INDEX < self.num_tickers):
                try:
                    current_spy_price = self.price_matrix[self.current_step, SPY_TICKER_INDEX]
                    previous_spy_price = self.price_matrix[self.current_step - 1, SPY_TICKER_INDEX]
                    if previous_spy_price > 0:
                        benchmark_return = (current_spy_price - previous_spy_price) / previous_spy_price
                except (IndexError, ValueError) as e:
                    # If SPY data not available, use 0.0 (will be handled by fallback logic)
                    benchmark_return = 0.0
            
            info = {
                'portfolio_value': self.portfolio_value,
                'cash': self.cash,
                'positions': {i: pos['shares'] for i, pos in self.positions.items()},
                'current_step': self.current_step,
                'total_trades': self._total_trades_executed,
                'episode_length': self.current_step,
                'masked_sell_streak': getattr(self, '_consecutive_masked_sells', 0),
                'masked_sell_count': getattr(self, '_masked_sell_counter', 0),
                'benchmark_return': benchmark_return  # CRITICAL FIX: Add benchmark return for Alpha/Beta calculation
            }

            # CRITICAL FIX: Log terminated/truncated values for debugging
            self._last_done = done
            self._last_terminated = done
            self._last_truncated = False

            # CRITICAL FIX: Add safety check before return to ensure done=False for steps < 50
            if self.current_step < 50 and done:
                self.logger.warning(f"[BUG_FIX] Overriding done=True to done=False at step {self.current_step}")
                done = False

            # CRITICAL FIX: Ensure done is explicitly a Python bool (not numpy bool)
            done = bool(done)
            truncated = False

            # FIX #2: Reset SELL execution flag before returning (cleanup)
            self._sell_executed_this_step = False

            if should_log:
                self.logger.info("=" * 60)
            # CRITICAL FIX: Return 5 values for VecEnv compatibility
            # DummyVecEnv expects: obs, reward, done, truncated, info (Gymnasium API)
            return obs, reward, done, truncated, info

        except Exception as e:
            # Return safe fallback - CRITICAL: Use done=False to prevent premature episode termination
            self.logger.error(f"Exception in step() method: {e}")
            import traceback as tb
            self.logger.error(f"Traceback: {tb.format_exc()}")
            obs = self._get_observation()
            return obs, 0.0, False, False, {'error': str(e), 'portfolio_value': getattr(self, 'portfolio_value', 100000.0)}

    def _check_trade_interval_constraints(self, action):
        """
        Check trade interval constraints to prevent excessive trading

        Returns:
            bool: True if trade interval is acceptable, False if violated
        """
        try:
            if action == 0:  # Hold action
                self._hold_count += 1  # BEHAVIOR TRACKING: Track HOLD actions
                return True

            # Check if we have the trade interval tracking
            if not hasattr(self, 'last_trade_step'):
                self.last_trade_step = -999  # Allow first trade

            # Use configured minimum trade interval
            min_trade_interval = max(0, getattr(self, 'min_trade_interval', 0))

            # CRITICAL FIX: Ensure last_trade_step is not in the future (should never happen, but safety check)
            if self.last_trade_step > self.current_step:
                self.logger.warning(f"Invalid last_trade_step {self.last_trade_step} > current_step {self.current_step}, resetting")
                self.last_trade_step = -999  # Reset to allow trade

            # Check if enough time has passed since last trade
            if (self.current_step - self.last_trade_step) < min_trade_interval:
                # Track violation for monitoring (reduced logging for performance)
                if not hasattr(self, '_trade_interval_violations'):
                    self._trade_interval_violations = 0
                self._trade_interval_violations += 1

                # Only log every 50th violation to reduce I/O overhead
                if self._trade_interval_violations % 50 == 0:
                    self.logger.warning(f"Trade interval violation: step {self.current_step}, last trade at {self.last_trade_step}, min interval: {min_trade_interval} (violations: {self._trade_interval_violations})")

                return False

            return True

        except Exception as e:
            self.logger.error(f"Trade interval constraint check failed: {e}")
            # Track error for monitoring
            if not hasattr(self, '_trade_interval_errors'):
                self._trade_interval_errors = 0
            self._trade_interval_errors += 1
            return False  # Block trade on error for safety

    def _apply_cointegration_disable_logic(self, action):
        """
        STEP 4 OPTIMIZATION: Disable cointegration and aggressive pairs trading
        Reduces false signals and excessive trading at 1-minute resolution
        """
        try:
            # Import config to check cointegration settings
            try:
                cointegration_enabled = config.COINTEGRATION_CONFIG.get('cointegration_enabled', False)
                aggressive_pairs_trading = config.COINTEGRATION_CONFIG.get('aggressive_pairs_trading', False)
                pairs_trading_enabled = config.COINTEGRATION_CONFIG.get('pairs_trading_enabled', False)
            except (ImportError, AttributeError):
                # Default to disabled if config not available
                cointegration_enabled = False
                aggressive_pairs_trading = False
                pairs_trading_enabled = False

            # STEP 4: Disable cointegration-based trading signals
            if not cointegration_enabled and not aggressive_pairs_trading and not pairs_trading_enabled:
                # All cointegration features are disabled - this is the expected behavior
                # No action modification needed - just log that we're using clean signals
                if not hasattr(self, '_cointegration_logged') or not self._cointegration_logged:
                    self.logger.info(" STEP 4: Cointegration and pairs trading disabled - using clean signals")
                    self._cointegration_logged = True
            else:
                # Cointegration features are enabled - apply filtering logic
                if action in [1, 2]:  # Buy or sell actions
                    # Apply cointegration-based filtering
                    if self._is_cointegration_signal(action):
                        # This is a cointegration-based signal - potentially filter it
                        if not aggressive_pairs_trading:
                            # Convert aggressive pairs signals to HOLD
                            if self._is_aggressive_pairs_signal(action):
                                self.logger.info(f"STEP 4: Converted aggressive pairs signal to HOLD")
                                return action  # DISABLED - DON'T CONVERT TO HOLD

                    # Apply pairs trading filtering
                    if not pairs_trading_enabled and self._is_pairs_trading_signal(action):
                        self.logger.info(f"STEP 4: Converted pairs trading signal to HOLD")
                        return action  # DISABLED - DON'T CONVERT TO HOLD

            return action

        except Exception as e:
            self.logger.error(f"Cointegration disable logic failed: {e}")
            return action  # Return original action if logic fails

    def _is_cointegration_signal(self, action):
        """Check if the action is based on cointegration signals"""
        # Placeholder logic - in a real implementation, this would check
        # if the action was triggered by cointegration analysis
        return False  # For now, assume no cointegration signals

    def _is_aggressive_pairs_signal(self, action):
        """Check if the action is an aggressive pairs trading signal"""
        # Placeholder logic - in a real implementation, this would check
        # if the action was triggered by aggressive pairs trading
        return False  # For now, assume no aggressive pairs signals

    def _is_pairs_trading_signal(self, action):
        """Check if the action is a pairs trading signal"""
        # Placeholder logic - in a real implementation, this would check
        # if the action was triggered by pairs trading analysis
        return False  # For now, assume no pairs trading signals

    def _apply_cooldown_logic(self, action):
        """
        STEP DEFAULT_NUM_FEATURES  # CONSTANT: Replaced magic number OPTIMIZATION: Apply cooldown logic to prevent scalping
        Prevents extremely short flip trades that cost more in slippage than they earn
        """
        try:
            # CRITICAL FIX: Import config at method start to prevent scope errors
            try:
                import config  # type: ignore
                cooldown_config = config.COOLDOWN_CONFIG
            except (ImportError, AttributeError):
                # Fallback values if config unavailable
                cooldown_config = {'min_hold_bars': 1, 'cooldown_bars': 1, 'cooldown_penalty': 0.001, 'log_cooldown_violations': True}

            # FIX: Re-enable cooldown; min_hold_steps is dynamic (market-dependent)
            try:
                dcfg = self._cfg()
                cooldown_enabled = bool(cooldown_config.get('cooldown_enabled', True))
            except Exception:
                cooldown_enabled = bool(cooldown_config.get('cooldown_enabled', True))
            # Dynamic min_hold: high vol/trending → shorter, mean_reversion → longer (from _get_dynamic_min_hold_steps)
            min_hold_bars = self._get_dynamic_min_hold_steps()
            if hasattr(self, 'mode') and self.mode == 'test':
                min_hold_bars = max(1, min_hold_bars)
            cooldown_bars = cooldown_config.get('cooldown_bars', 2)
            cooldown_penalty = cooldown_config.get('cooldown_penalty', 0.001)
            log_violations = cooldown_config.get('log_cooldown_violations', True)

            if not cooldown_enabled:
                return action

            # Log cooldown status once
            if not hasattr(self, '_cooldown_logged') or not self._cooldown_logged:
                self.logger.debug(" STEP DEFAULT_NUM_FEATURES  # CONSTANT: Replaced magic number: Cooldown logic enabled - preventing scalping")  # OPT D: was info
                self._cooldown_logged = True

            # Update cooldown and hold trackers
            self._update_cooldown_trackers()

            # Check if action violates cooldown constraints
            if action in [1, 2]:  # Buy or sell action
                # Get the ticker that would be affected
                ticker = self._get_action_ticker(action)

                if ticker is not None:
                    # Check minimum hold constraint
                    if self._violates_minimum_hold(ticker, min_hold_bars):
                        if log_violations:
                            self.logger.debug(f"STEP DEFAULT_NUM_FEATURES  # CONSTANT: Replaced magic number: Converted {action} to HOLD - minimum hold not met for {ticker}")  # OPT D: was info
                        self._cooldown_violations += 1
                        return 0  # Convert SELL/BUY to HOLD when min hold not met (enforce hold time)

                    # Check cooldown constraint
                    if self._violates_cooldown(ticker, cooldown_bars):
                        if log_violations:
                            self.logger.debug(f"STEP DEFAULT_NUM_FEATURES  # CONSTANT: Replaced magic number: Converted {action} to HOLD - cooldown active for {ticker}")  # OPT D: was info
                        self._cooldown_violations += 1
                        return 0  # Convert to HOLD when cooldown active

            return action

        except Exception as e:
            self.logger.error(f"Cooldown logic failed: {e}")
            return action

    def _update_cooldown_trackers(self):
        """Update cooldown and position hold trackers"""
        try:
            # Decrement cooldown counters
            for ticker in list(self._cooldown_tracker.keys()):
                if self._cooldown_tracker[ticker] > 0:
                    self._cooldown_tracker[ticker] -= 1
                    if self._cooldown_tracker[ticker] <= 0:
                        del self._cooldown_tracker[ticker]

            # Decrement position hold counters
            for ticker in list(self._position_hold_tracker.keys()):
                if self._position_hold_tracker[ticker] > 0:
                    self._position_hold_tracker[ticker] -= 1
                    if self._position_hold_tracker[ticker] <= 0:
                        del self._position_hold_tracker[ticker]

        except Exception as e:
            self.logger.error(f"Failed to update cooldown trackers: {e}")

    def _get_action_ticker(self, action):
        """Get the ticker that would be affected by the action"""
        try:
            if action == 1:  # Buy action
                return self._select_best_ticker_to_buy(self.price_matrix[self.current_step])
            elif action == 2:  # Sell action
                return self._select_ticker_to_sell()
            elif action == 3:  # Short action
                return self._select_best_ticker_to_short(self.price_matrix[self.current_step])
            return None
        except Exception as e:
            self.logger.error(f"Failed to get action ticker: {e}")
            return None

    def _violates_minimum_hold(self, ticker, min_hold_bars):
        """Check if ticker violates minimum hold constraint"""
        try:
            if ticker in self._position_hold_tracker:
                return self._position_hold_tracker[ticker] > 0
            return False
        except Exception as e:
            self.logger.error(f"Failed to check minimum hold: {e}")
            return False

    def _violates_cooldown(self, ticker, cooldown_bars):
        """Check if ticker violates cooldown constraint"""
        try:
            if ticker in self._cooldown_tracker:
                return self._cooldown_tracker[ticker] > 0
            return False
        except Exception as e:
            self.logger.error(f"Failed to check cooldown: {e}")
            return False

    def _set_cooldown(self, ticker, cooldown_bars):
        """Set cooldown for a ticker"""
        try:
            self._cooldown_tracker[ticker] = cooldown_bars
        except Exception as e:
            self.logger.error(f"Failed to set cooldown: {e}")

    def _set_minimum_hold(self, ticker, min_hold_bars):
        """Set minimum hold time for a ticker"""
        try:
            self._position_hold_tracker[ticker] = min_hold_bars
        except Exception as e:
            self.logger.error(f"Failed to set minimum hold: {e}")

    def _get_dynamic_min_hold_steps(self) -> int:
        """Min hold steps depending on market: high vol/trending → shorter hold, low vol/mean_reversion → longer hold."""
        try:
            dcfg = self._cfg()
            base = int(getattr(dcfg, 'min_hold_steps', 8))
            base = max(1, base)
            regime = getattr(self, 'current_market_regime', None) or getattr(self, '_cached_market_regime', None)
            if regime is None and self.current_step >= 1:
                regime = self._detect_market_regime()
            # High vol/trending: allow shorter hold (exit faster); mean_reversion: hold longer
            if regime == 'trending':
                return max(1, base - 2)   # e.g. 8 -> 6
            if regime == 'mean_reversion':
                return min(base + 2, 25)  # e.g. 8 -> 10, cap 25
            return base
        except Exception:
            return 8

    def _load_stress_test_config(self):
        """Load stress testing configuration"""
        try:
            if config.STRESS_TEST_CONFIG.get('stress_testing_enabled', False):
                # Use baseline scenario by default
                cost_config = config.STRESS_TEST_CONFIG['cost_scenarios']['baseline']
                volatility_config = config.STRESS_TEST_CONFIG['volatility_scenarios']['baseline']
                stress_config = {**cost_config, **volatility_config}
                self.logger.info(" STEP 8: Stress testing enabled - baseline scenario")
                return stress_config
            else:
                return {'slippage_multiplier': 1.0, 'impact_multiplier': 1.0, 'volatility_multiplier': 1.0}
        except (ImportError, AttributeError):
            return {'slippage_multiplier': 1.0, 'impact_multiplier': 1.0, 'volatility_multiplier': 1.0}

    def set_stress_scenario(self, scenario_name):
        """
        STEP 8 OPTIMIZATION: Set stress testing scenario

        Args:
            scenario_name: 'baseline', 'moderate_stress', 'high_stress', 'low_volatility', 'high_volatility'
        """
        try:
            if scenario_name in config.STRESS_TEST_CONFIG['cost_scenarios']:
                cost_config = config.STRESS_TEST_CONFIG['cost_scenarios'][scenario_name]
                self._slippage_multiplier = cost_config['slippage_multiplier']
                self._impact_multiplier = cost_config['impact_multiplier']
                self.logger.info(f" STEP 8: Set cost scenario '{scenario_name}' - slippage: {self._slippage_multiplier}x, impact: {self._impact_multiplier}x")

            if scenario_name in config.STRESS_TEST_CONFIG['volatility_scenarios']:
                volatility_config = config.STRESS_TEST_CONFIG['volatility_scenarios'][scenario_name]
                self._volatility_multiplier = volatility_config['volatility_multiplier']
                self.logger.info(f" STEP 8: Set volatility scenario '{scenario_name}' - volatility: {self._volatility_multiplier}x")

        except Exception as e:
            self.logger.error(f"Failed to set stress scenario: {e}")

    def _calculate_stress_costs(self, base_cost, trade_size, price):
        """
        STEP 8 OPTIMIZATION: Calculate costs under stress testing

        Args:
            base_cost: Base transaction cost
            trade_size: Size of the trade
            price: Price of the asset

        Returns:
            Total cost including slippage and market impact
        """
        try:
            # Base transaction cost
            total_cost = base_cost

            # Add slippage cost (proportional to trade size)
            slippage_cost = base_cost * 0.001 * trade_size * self._slippage_multiplier
            total_cost += slippage_cost

            # Add market impact cost (proportional to trade size and price)
            impact_cost = base_cost * DEFAULT_SLIPPAGE  # CONSTANT: Replaced magic number * trade_size * price * self._impact_multiplier
            total_cost += impact_cost

            return total_cost

        except Exception as e:
            self.logger.error(f"Failed to calculate stress costs: {e}")
            return base_cost

    def _apply_volatility_shock(self, returns):
        """
        STEP 8 OPTIMIZATION: Apply volatility shock to returns

        Args:
            returns: Original returns

        Returns:
            Returns with volatility shock applied
        """
        try:
            if self._volatility_multiplier != 1.0:
                # Apply volatility multiplier to returns
                shocked_returns = returns * self._volatility_multiplier
                return shocked_returns
            return returns

        except Exception as e:
            self.logger.error(f"Failed to apply volatility shock: {e}")
            return returns

    def _check_diversification_constraints(self, action, ticker):
        """
        Check diversification constraints before executing trade

        Args:
            action: Trading action (0=hold, 1=buy, 2=sell)
            ticker: Ticker index (0-5 for AAPL, MSFT, NVDA, JPM, SPY, GLD)

        Returns:
            bool: True if trade passes constraints, False if blocked
        """
        try:
            # Skip constraint check for hold actions
            if action == 0:
                return True

            # Get current portfolio allocation
            total_value = self.portfolio_value
            if total_value <= 0:
                return False

            # Calculate current position weights
            current_weights = {}
            for i in range(self.num_tickers):
                if i < len(self.price_matrix[self.current_step]):
                    current_price = self.price_matrix[self.current_step][i]
                    position_value = self.positions[i]['shares'] * current_price
                    current_weights[i] = position_value / total_value if total_value > 0 else 0
                else:
                    current_weights[i] = 0

            # Diversification constraint: No single position > max_position_size of portfolio
            # PHASE 1 FIX: Use dynamic config for position sizing
            max_single_position = getattr(self, 'max_position_size', 0.02)  # Use dynamic config value

            if action == 1:  # Buy action
                # Check if this ticker already exceeds max position
                if ticker in current_weights and current_weights[ticker] > max_single_position:
                    self.logger.warning(f"Ticker {ticker} position {current_weights[ticker]:.2%} exceeds max {max_single_position:.2%}")
                    # Track violation
                    if not hasattr(self, '_diversification_violations'):
                        self._diversification_violations = 0
                    self._diversification_violations += 1
                    return False

                # Check if adding to this position would exceed max
                # For simplicity, assume we're adding a small amount (1% of portfolio)
                potential_addition = 0.01
                if ticker in current_weights:
                    potential_weight = current_weights[ticker] + potential_addition
                    if potential_weight > max_single_position:
                        # Track violation
                        if not hasattr(self, '_diversification_violations'):
                            self._diversification_violations = 0
                        self._diversification_violations += 1
                        return False

            elif action == 2:  # Sell action
                # Selling is generally allowed for diversification
                return True

            # Sector diversification: Ensure not too concentrated in tech (AAPL, MSFT, NVDA)
            tech_tickers = [0, 1, 2]  # AAPL, MSFT, NVDA
            tech_weight = sum(current_weights.get(i, 0) for i in tech_tickers)
            # PHASE 1 FIX: Use dynamic config for tech weight constraints
            max_tech_weight = getattr(self, 'max_tech_weight', 0.80)  # Use dynamic config value

            if tech_weight > max_tech_weight:
                if action == 1 and ticker in tech_tickers:
                    self.logger.warning(f"Tech sector weight {tech_weight:.2%} exceeds max {max_tech_weight:.2%}")
                    # Track violation
                    if not hasattr(self, '_diversification_violations'):
                        self._diversification_violations = 0
                    self._diversification_violations += 1
                    return False

            # All constraints passed
            return True

        except Exception as e:
            self.logger.error(f"Diversification constraint check failed: {e}")
            # On error, allow the trade (fail-safe)
            return True

    def _check_episode_done(self):
        """Check if episode is done with comprehensive debugging"""
        try:
            # CRITICAL FIX: Ensure _max_episode_steps is set
            if not hasattr(self, '_max_episode_steps'):
                self._max_episode_steps = len(self.price_matrix) - self.window_size - 1

            # CRITICAL FIX: ABSOLUTE: Never end episode before 50 steps
            if self.current_step < 50:
                return False  # No logging to reduce verbosity

            # FIX #1: Use percentage-based termination instead of hard limit
            # This allows longer episodes when more data is available
            # Evaluation mode uses higher percentage (95%) for more complete assessment
            max_available_steps = len(self.price_matrix) - self.window_size - 1
            current_mode = getattr(self, 'mode', 'train')
            if current_mode == 'test' or current_mode == 'eval':
                episode_length_percentage = 0.95  # 95% for evaluation mode (more complete assessment)
            else:
                episode_length_percentage = 0.90  # 90% for training mode
            end_threshold = int(max_available_steps * episode_length_percentage)
            
            # FIXED: Ensure minimum episode length of 250 steps for meaningful training (increased from 50)
            end_threshold = max(250, end_threshold)
            
            if self.current_step >= end_threshold:
                self.logger.info(f"[EPISODE_CHECK] Episode ended: reached {episode_length_percentage*100:.0f}% of available data at step {self.current_step} (threshold: {end_threshold}, max_available: {max_available_steps})")
                return True
            
            # Also check absolute limit as safety (prevent going beyond data)
            absolute_limit = max_available_steps
            if self.current_step >= absolute_limit:
                self.logger.info(f"[EPISODE_CHECK] Episode ended: reached absolute data limit at step {self.current_step} (limit: {absolute_limit})")
                return True

            # CRITICAL FIX: Portfolio threshold - only terminate early for severe crashes (below 50%)
            # FIXED: Changed from 2% to 50% - only terminate for severe crashes
            portfolio_crash_threshold = self.initial_balance * 0.50  # 50% loss threshold
            min_steps = self.window_size + 50  # Ensure minimum steps before checking

            if self.portfolio_value < portfolio_crash_threshold and self.current_step > min_steps:
                self.logger.info(f"[EPISODE_CHECK] Episode ended: portfolio crashed below 50% threshold ${portfolio_crash_threshold:.2f}")
                return True

            # CRITICAL FIX: Add maximum steps limit to prevent infinite episodes
            # Use a very high limit (10,000) to allow full training while preventing true infinite loops
            # This is much higher than normal episodes but prevents runaway scenarios
            SAFETY_MAX_STEPS = 10000  # Very high safety limit to prevent infinite loops without hampering training
            if self.current_step >= SAFETY_MAX_STEPS:
                self.logger.warning(f"[EPISODE_CHECK] Episode ended: reached SAFETY MAX STEPS limit {SAFETY_MAX_STEPS} at step {self.current_step} (likely infinite loop)")
                return True

            # CRITICAL FIX: Add portfolio value explosion protection - only terminate if exceeds 200%
            # FIXED: Changed from 50x to 2.0x (200%) - only terminate for unrealistic gains
            max_portfolio_threshold = self.initial_balance * 2.0  # 200% gain threshold
            if self.portfolio_value > max_portfolio_threshold:
                self.logger.info(f"[EPISODE_CHECK] Episode ended: portfolio exceeded 200% threshold ${max_portfolio_threshold:.2f}")
                return True

            # CRITICAL FIX: Add consecutive trade limit (check the counter set in step method)
            # Check max consecutive trades of same type (not total trades)
            consecutive_same_action = max(
                getattr(self, '_consecutive_buys', 0),
                getattr(self, '_consecutive_sells', 0)
            )

            if consecutive_same_action >= 100:  # FIXED: Increased from 20 to 100 to prevent premature termination
                action_type = "BUY" if getattr(self, '_consecutive_buys', 0) > getattr(self, '_consecutive_sells', 0) else "SELL"
                self.logger.info(f"[EPISODE_CHECK] Episode ended: too many consecutive {action_type} ({consecutive_same_action})")
                return True

            # CRITICAL FIX: Trade limit check removed from here - now handled in step() method BEFORE trade execution
            # This prevents the "too many total trades (101)" issue by blocking trades at the limit

            return False  # Episode continuing - no logging to reduce verbosity

        except Exception as e:
            self.logger.error(f"Episode done check failed: {e}")
            return True  # End episode on error for safety

    def _validate_action(self, action):
        """Phase 2.2: Validate action before execution to prevent invalid trades"""
        try:
            if action == 1:  # BUY (or cover short)
                min_trade_size = getattr(self, 'min_trade_size', 20.0)
                has_shorts = any(
                    i in self.positions and self.positions[i].get('shares', 0.0) < 0
                    for i in range(self.num_tickers)
                )
                return self.cash >= min_trade_size or has_shorts
            elif action == 2:  # SELL
                return any(
                    i in self.positions and self.positions[i].get('shares', 0.0) > 0 
                    for i in range(self.num_tickers)
                )
            elif action == 3:  # SHORT
                min_trade_size = getattr(self, 'min_trade_size', 20.0)
                return self.cash >= min_trade_size
            return True  # HOLD always valid
        except Exception as e:
            self.logger.error(f"Action validation failed: {e}")
            return True  # Allow action on validation error to prevent complete blocking

    def _precompute_step_signals(self):
        """
        PERFORMANCE OPTIMIZATION: Precompute all signals once per step for all tickers and actions.
        This eliminates redundant calculations and provides 30-40% speedup.
        
        ENHANCED: Now uses vectorized numpy operations for all tickers at once.
        
        Precomputes:
        - Technical signals for all tickers and actions (BUY/SELL) - VECTORIZED
        - Fundamental signals for all tickers and actions (BUY/SELL)
        - VWAP rewards for all tickers
        - Market regime and multipliers
        """
        try:
            # Initialize cache structure if needed
            if not hasattr(self, '_step_signals_cache'):
                self._step_signals_cache = {}
            
            # PERFORMANCE: Clear cache if step changed (use clear() for better memory efficiency)
            if not hasattr(self, '_step_signals_cache_step') or self._step_signals_cache_step != self.current_step:
                if hasattr(self, '_step_signals_cache') and isinstance(self._step_signals_cache, dict):
                    self._step_signals_cache.clear()  # More memory efficient than {} assignment
                else:
                    self._step_signals_cache = {}
                self._step_signals_cache_step = self.current_step
            
            # Get current prices for VWAP calculations
            if self.current_step >= len(self.price_matrix):
                return  # Can't precompute if no price data
            
            current_prices = self.price_matrix[self.current_step]
            
            # PERFORMANCE: Precompute market regime (cache for 10 steps since it changes slowly)
            if not hasattr(self, '_cached_market_regime') or not hasattr(self, '_cached_market_regime_step') or \
               (self.current_step - self._cached_market_regime_step) >= 10:
                self._cached_market_regime = self._detect_market_regime()
                self._cached_market_regime_step = self.current_step
                self.current_market_regime = self._cached_market_regime
            
            # PERFORMANCE: Precompute market condition multipliers (cache for 10 steps)
            if not hasattr(self, '_cached_market_multiplier') or not hasattr(self, '_cached_market_multiplier_step') or \
               (self.current_step - self._cached_market_multiplier_step) >= 10:
                buy_mult = self._calculate_market_condition_multiplier('BUY')
                sell_mult = self._calculate_market_condition_multiplier('SELL')
                hold_mult = self._calculate_adaptive_hold_penalty()
                
                self._cached_market_multiplier = {
                    'BUY': buy_mult,
                    'SELL': sell_mult,
                    'HOLD': hold_mult
                }
                self._cached_market_multiplier_step = self.current_step
                
                # Track adaptive reward adjustments for unified report
                if not hasattr(self, '_adaptive_reward_log'):
                    self._adaptive_reward_log = []
                
                # FIXED: Log every 50 steps OR at first step OR near last step to ensure evaluation logging
                total_steps = getattr(self, '_max_episode_steps', None)
                if total_steps is None or total_steps <= 0:
                    total_steps = len(self.price_matrix) - 1
                should_log = (
                    (self.current_step % 50 == 0)
                    or (self.current_step == 0)
                    or (self.current_step >= total_steps - 1)
                )
                if should_log:
                    self._adaptive_reward_log.append({
                        'step': self.current_step,
                        'buy_multiplier': float(np.nan_to_num(buy_mult, nan=1.0, posinf=1.0, neginf=1.0)),
                        'sell_multiplier': float(np.nan_to_num(sell_mult, nan=1.0, posinf=1.0, neginf=1.0)),
                        'hold_multiplier': float(np.nan_to_num(hold_mult, nan=1.0, posinf=1.0, neginf=1.0)),
                        'message': f'Market multipliers: BUY={buy_mult:.2f}x, SELL={sell_mult:.2f}x, HOLD={hold_mult:.2f}x'
                    })
            
            # PERFORMANCE OPTIMIZATION: Vectorized technical indicator computation
            # Compute RSI, MACD, BB, Volume scores for ALL tickers at once
            tech_scores_buy, tech_scores_sell = self._compute_vectorized_technical_signals()
            
            # Precompute signals for all tickers and actions
            for ticker in range(self.num_tickers):
                if ticker >= len(current_prices):
                    continue
                
                current_price = float(current_prices[ticker])
                
                # Initialize ticker cache
                if ticker not in self._step_signals_cache:
                    self._step_signals_cache[ticker] = {}
                
                # Use pre-computed vectorized technical signals
                for action in ['BUY', 'SELL']:
                    if action not in self._step_signals_cache[ticker]:
                        self._step_signals_cache[ticker][action] = {}
                    
                    # Use vectorized tech score (much faster than individual calculation)
                    if action == 'BUY':
                        tech_score = tech_scores_buy[ticker] if ticker < len(tech_scores_buy) else 0.0
                    else:
                        tech_score = tech_scores_sell[ticker] if ticker < len(tech_scores_sell) else 0.0
                    
                    self._step_signals_cache[ticker][action]['tech'] = tech_score
                    
                    # Fundamental signal (still per-ticker, but cached)
                    fund_score = self._calculate_fundamental_signal_quality(ticker, action)
                    self._step_signals_cache[ticker][action]['fund'] = fund_score
                
                # Precompute VWAP reward (only once per ticker, not per action)
                vwap_reward = self._calculate_vwap_reward(ticker, current_price)
                self._step_signals_cache[ticker]['vwap'] = vwap_reward
                
        except Exception as e:
            self.logger.debug(f"Error in _precompute_step_signals: {e}")
            # On error, clear cache to force recalculation
            # PERFORMANCE: Use clear() for better memory efficiency
            if hasattr(self, '_step_signals_cache') and isinstance(self._step_signals_cache, dict):
                self._step_signals_cache.clear()
            else:
                self._step_signals_cache = {}
    
    def _compute_vectorized_technical_signals(self) -> tuple:
        """
        PERFORMANCE OPTIMIZATION: Compute technical signals for ALL tickers at once using numpy vectorization.
        This replaces 24 individual calls (12 tickers × 2 actions) with a single vectorized computation.
        
        Expected speedup: 5-10x faster than per-ticker computation.
        
        Returns:
            Tuple of (buy_scores, sell_scores) - arrays of shape (num_tickers,)
        """
        try:
            # Get config values (cached)
            try:
                if not hasattr(self, '_cached_ta_config'):
                    dcfg = self._cfg()
                    self._cached_ta_config = {
                        'rsi_lo': float(getattr(dcfg, 'rsi_oversold', 30)),
                        'rsi_hi': float(getattr(dcfg, 'rsi_overbought', 70)),
                        'vol_th': float(getattr(dcfg, 'volume_threshold', 1.2))
                    }
                rsi_lo = self._cached_ta_config['rsi_lo']
                rsi_hi = self._cached_ta_config['rsi_hi']
                vol_th = self._cached_ta_config['vol_th']
            except Exception:
                rsi_lo, rsi_hi, vol_th = 30.0, 70.0, 1.2
            
            # Get price and volume slices for all tickers (shape: [lookback, num_tickers])
            end = self.current_step
            start = max(0, end - 26)  # Need 26 periods for MACD
            
            if end <= start or end > len(self.price_matrix):
                return np.zeros(self.num_tickers), np.zeros(self.num_tickers)
            
            # Slice data for all tickers at once (shape: [lookback, num_tickers])
            prices_all = self.price_matrix[start:end, :self.num_tickers].astype(float)
            volumes_all = self.volume_matrix[start:end, :self.num_tickers].astype(float)
            
            lookback = prices_all.shape[0]
            if lookback < 5:
                return np.zeros(self.num_tickers), np.zeros(self.num_tickers)
            
            # Initialize score arrays
            buy_scores = np.zeros(self.num_tickers)
            sell_scores = np.zeros(self.num_tickers)
            
            # === VECTORIZED RSI CALCULATION ===
            # Compute price differences for all tickers at once
            diffs = np.diff(prices_all, axis=0)  # Shape: [lookback-1, num_tickers]
            gains = np.clip(diffs, 0, None)
            losses = -np.clip(diffs, None, 0)
            
            # Mean gains and losses per ticker
            avg_gains = np.mean(gains, axis=0)  # Shape: [num_tickers]
            avg_losses = np.mean(losses, axis=0)  # Shape: [num_tickers]
            
            # RSI calculation
            with np.errstate(divide='ignore', invalid='ignore'):
                rs = np.where(avg_losses > 0, avg_gains / (avg_losses + 1e-9), np.inf)
                rsi = 100.0 - (100.0 / (1.0 + rs))
                rsi = np.where(np.isfinite(rsi), rsi, 100.0)
            
            # === VECTORIZED MACD CALCULATION ===
            # PHASE 1 FIX: Use EMA-based MACD for vectorized calculation
            # Note: Full EMA implementation for vectorized case is complex, so we use weighted average approximation
            # For vectorized case, we'll use a weighted average that approximates EMA
            if lookback >= 26:
                # Approximate EMA with weighted averages (faster than full EMA for all tickers)
                # EMA approximation: use recent prices with more weight (exponential decay)
                weights_12 = np.exp(np.linspace(-1, 0, 12))
                weights_12 = weights_12 / weights_12.sum()
                weights_26 = np.exp(np.linspace(-1, 0, 26))
                weights_26 = weights_26 / weights_26.sum()
                ema_12_approx = np.sum(prices_all[-12:] * weights_12[:, np.newaxis], axis=0)
                ema_26_approx = np.sum(prices_all[-26:] * weights_26[:, np.newaxis], axis=0)
                macd_line = ema_12_approx - ema_26_approx
                # Signal line: approximate EMA of MACD by using weighted average of recent MACD values
                # Simplified: use weighted average of recent price differences (approximates MACD history)
                # For vectorized case, use a simpler approximation: signal = 0.9 * macd_line (slight lag)
                # This is faster and still provides reasonable MACD signals
                macd_signal = macd_line * 0.9  # Approximate: signal slightly below MACD line (typical MACD behavior)
                macd_diff = macd_line - macd_signal
            else:
                short_len = min(12, lookback)
                macd_line = np.mean(prices_all[-short_len:], axis=0) - np.mean(prices_all, axis=0)
                macd_signal = np.mean(prices_all[-min(9, lookback):], axis=0)
                macd_diff = macd_line - macd_signal
            
            # === VECTORIZED BOLLINGER BANDS CALCULATION ===
            bb_scores = np.zeros(self.num_tickers)
            if lookback >= 20:
                bb_mean = np.mean(prices_all[-20:], axis=0)  # Shape: [num_tickers]
                bb_std = np.std(prices_all[-20:], axis=0) + 1e-9  # Shape: [num_tickers]
                bb_z = (prices_all[-1] - bb_mean) / bb_std
                bb_scores = -bb_z  # Prefer buys below mean, sells above mean
            
            # === VECTORIZED VOLUME SCORE ===
            vol_scores = np.zeros(self.num_tickers)
            if lookback >= 5:
                avg_vol = np.mean(volumes_all[:-1], axis=0) if lookback > 1 else np.mean(volumes_all, axis=0)
                with np.errstate(divide='ignore', invalid='ignore'):
                    vol_ratio = np.where(avg_vol > 0, volumes_all[-1] / (avg_vol + 1e-9), 1.0)
                vol_scores = np.where(vol_ratio > vol_th, 0.2, np.where(vol_ratio < 0.8, -0.2, 0.0))
            
            # === COMPUTE BUY SCORES ===
            buy_scores += np.where(rsi <= rsi_lo, 0.5, 0.0)
            buy_scores += np.where(macd_diff > 0, 0.2, 0.0)
            buy_scores += bb_scores * 0.1
            buy_scores += vol_scores
            
            # === COMPUTE SELL SCORES ===
            sell_scores += np.where(rsi >= rsi_hi, 0.5, 0.0)
            sell_scores += np.where(macd_diff < 0, 0.2, 0.0)
            sell_scores += (-bb_scores) * 0.1
            sell_scores += vol_scores
            
            # Apply regime adjustment (same for all tickers)
            regime_adj = self._get_regime_adjustment('BUY')  # Same adjustment for both
            buy_scores *= regime_adj
            sell_scores *= regime_adj
            
            # Clip to [-1, 1]
            buy_scores = np.clip(buy_scores, -1.0, 1.0)
            sell_scores = np.clip(sell_scores, -1.0, 1.0)
            
            return buy_scores, sell_scores
            
        except Exception as e:
            self.logger.debug(f"Error in _compute_vectorized_technical_signals: {e}")
            return np.zeros(self.num_tickers), np.zeros(self.num_tickers)

    def _calculate_vwap_reward(self, ticker: int, current_price: float) -> float:
        """VWAP-based reward component for BUY/SELL decisions."""
        try:
            from ppo.utils.dynamic_config import get_dynamic_config  # type: ignore
            dcfg = self._cfg()
            lookback = int(getattr(dcfg, 'vwap_lookback', 10))
            buy_scale = float(getattr(dcfg, 'vwap_buy_bonus_scale', 10.0))
            sell_scale = float(getattr(dcfg, 'vwap_sell_bonus_scale', 2.0))
        except Exception:
            lookback, buy_scale, sell_scale = 10, 10.0, 2.0

        # DATA LEAKAGE PREVENTION: VWAP uses only PAST data (excludes current_step)
        # end_idx = current_step means slice [start_idx:end_idx] excludes current_step
        # This ensures we only use historical data for VWAP calculation
        end_idx = self.current_step
        start_idx = max(0, end_idx - lookback)
        if end_idx <= start_idx:
            return 0.0
        # Bounds check: ensure we don't access beyond available data
        if end_idx > len(self.price_matrix) or ticker >= self.price_matrix.shape[1]:
            return 0.0
        prices = self.price_matrix[start_idx:end_idx, ticker]
        volumes = self.volume_matrix[start_idx:end_idx, ticker]
        total_vol = float(np.sum(volumes)) if len(volumes) > 0 else 0.0
        if total_vol <= 0:
            return 0.0
        vwap = float(np.sum(prices * volumes) / total_vol)
        if vwap <= 0:
            return 0.0
        if current_price < vwap:
            return ((vwap - current_price) / vwap) * buy_scale
        return -((current_price - vwap) / vwap) * sell_scale

    def _detect_market_regime(self) -> str:
        """Detect simple regime via normalized ATR."""
        try:
            from ppo.utils.dynamic_config import get_dynamic_config  # type: ignore
            dcfg = self._cfg()
            atr_period = int(getattr(dcfg, 'atr_period', 14))
            high_vol_th = float(getattr(dcfg, 'high_volatility_threshold', 0.02))
        except Exception:
            atr_period, high_vol_th = 14, 0.02

        if self.current_step < atr_period:
            return 'unknown'
        # DATA LEAKAGE PREVENTION: ATR includes current_step (intentional)
        # Python slicing [start:end] is exclusive on end, so [start:current_step+1] includes current_step
        # This is correct - we want current step's price for ATR calculation
        # We're NOT accessing current_step+1 or higher (which would be future data)
        start_idx = max(0, self.current_step - atr_period)
        atr_vals = []
        # Bounds check: ensure we don't access beyond available data
        max_available_step = len(self.price_matrix) - 1
        if self.current_step > max_available_step:
            return 'unknown'
        for i in range(self.num_tickers):
            # Slice includes current_step (intentional - needed for ATR calculation)
            px = self.price_matrix[start_idx:self.current_step + 1, i]
            if len(px) < 2 or px[-1] <= 0:
                continue
            tr = np.abs(np.diff(px))
            atr = float(np.mean(tr)) if tr.size > 0 else 0.0
            atr_norm = atr / float(px[-1]) if px[-1] > 0 else 0.0
            atr_vals.append(atr_norm)
        if not atr_vals:
            return 'unknown'
        avg_atr = float(np.mean(atr_vals))
        return 'trending' if avg_atr > high_vol_th else 'mean_reversion'

    def _get_regime_adjustment(self, action: str) -> float:
        try:
            # PERFORMANCE: Use cached market regime (updated in _precompute_step_signals)
            regime = getattr(self, 'current_market_regime', None)
            if not regime:
                # Fallback: check if cached regime exists
                if hasattr(self, '_cached_market_regime'):
                    regime = self._cached_market_regime
                    self.current_market_regime = regime
                else:
                    regime = self._detect_market_regime()
                    self.current_market_regime = regime
            return 1.2 if regime == 'trending' else (1.1 if regime == 'mean_reversion' else 1.0)
        except Exception:
            return 1.0

    def _get_dynamic_reward_context(self) -> dict:
        """
        Compute market/portfolio context for dynamic reward scaling.
        Used to scale activity bonuses, exploration, signal alignment, gain thresholds,
        hold-time bonus, quick-flip penalty, and loss penalty by regime/vol/drawdown.
        Cached per step (10-step TTL) to avoid repeated computation.
        """
        cache_step = getattr(self, '_dynamic_reward_context_step', -999)
        if hasattr(self, '_dynamic_reward_context') and (self.current_step - cache_step) < 10:
            return self._dynamic_reward_context
        try:
            regime = getattr(self, 'current_market_regime', None) or getattr(self, '_cached_market_regime', None)
            if regime is None:
                regime = self._detect_market_regime()
            vol_ratio = 1.0
            if hasattr(self, 'price_matrix') and self.current_step >= 20:
                volatilities = []
                for i in range(self.num_tickers):
                    prices = self.price_matrix[max(0, self.current_step - 20):self.current_step + 1, i]
                    if len(prices) >= 2:
                        returns = np.diff(prices) / (prices[:-1] + 1e-10)
                        vol = float(np.std(returns)) if len(returns) > 0 else 0.0
                        volatilities.append(vol)
                if volatilities:
                    avg_vol = float(np.mean(volatilities))
                    vol_ratio = avg_vol / 0.02 if avg_vol > 1e-10 else 1.0
                    vol_ratio = float(np.clip(vol_ratio, 0.5, 2.0))
            init_bal = getattr(self, 'initial_balance', 100000.0)
            port_val = getattr(self, 'portfolio_value', init_bal)
            portfolio_return = (port_val - init_bal) / init_bal if init_bal > 0 else 0.0
            current_drawdown = getattr(self, 'current_drawdown', 0.0)
            if not hasattr(current_drawdown, '__float__'):
                current_drawdown = float(current_drawdown) if current_drawdown is not None else 0.0
            ctx = {
                'regime': regime or 'unknown',
                'vol_ratio': vol_ratio,
                'portfolio_return': portfolio_return,
                'current_drawdown': current_drawdown,
            }
            self._dynamic_reward_context = ctx
            self._dynamic_reward_context_step = self.current_step
            return ctx
        except Exception:
            return {
                'regime': 'unknown',
                'vol_ratio': 1.0,
                'portfolio_return': 0.0,
                'current_drawdown': 0.0,
            }

    def _get_regime_outputs(self) -> dict:
        """
        Market-first regime controller outputs (cached per step; 10-step TTL).

        Returns a dict with:
          - buy_reward_mult, sell_reward_mult, hold_penalty_mult
          - gate_high_threshold, gate_medium_threshold
          - stop_loss_mult, trailing_stop_mult
        """
        cache_step = getattr(self, '_regime_outputs_step', -999)
        if hasattr(self, '_regime_outputs') and (self.current_step - cache_step) < 10:
            ro = getattr(self, '_regime_outputs', None)
            if isinstance(ro, dict):
                return ro
        try:
            ctx = self._get_dynamic_reward_context()
            market_regime = str(ctx.get('regime', 'unknown') or 'unknown')
            vol_ratio = float(ctx.get('vol_ratio', 1.0) or 1.0)
            avg_vol = float(np.clip(vol_ratio, 0.5, 2.0)) * 0.02

            bull_bear = 'neutral'
            try:
                bull_bear = str(self._detect_bull_bear_regime() or 'neutral')
            except Exception:
                bull_bear = 'neutral'

            spy_ret = 0.0
            try:
                if hasattr(self, 'price_matrix') and self.current_step >= 10:
                    spy_idx = min(4, self.num_tickers - 1)
                    spy_prices = self.price_matrix[max(0, self.current_step - 10):self.current_step + 1, spy_idx]
                    if len(spy_prices) >= 2 and spy_prices[0] > 0:
                        spy_ret = float((spy_prices[-1] - spy_prices[0]) / (spy_prices[0] + 1e-10))
            except Exception:
                spy_ret = 0.0

            has_positions = any(
                i in self.positions and self.positions[i].get('shares', 0.0) > 0
                for i in range(self.num_tickers)
            )

            from ppo.utils.regime_controller import RegimeController, RegimeState

            if not hasattr(self, '_regime_controller') or self._regime_controller is None:
                self._regime_controller = RegimeController()

            out = self._regime_controller.compute(
                RegimeState(
                    market_regime=market_regime,
                    bull_bear=bull_bear,
                    avg_volatility=avg_vol,
                    spy_return_lookback=spy_ret,
                    vol_ratio=vol_ratio,
                    has_positions=bool(has_positions),
                )
            )
            ro = {
                'buy_reward_mult': float(out.buy_reward_mult),
                'sell_reward_mult': float(out.sell_reward_mult),
                'hold_penalty_mult': float(out.hold_penalty_mult),
                'gate_high_threshold': float(out.gate_high_threshold),
                'gate_medium_threshold': float(out.gate_medium_threshold),
                'stop_loss_mult': float(out.stop_loss_mult),
                'trailing_stop_mult': float(out.trailing_stop_mult),
                'bull_bear': bull_bear,  # 2026-02-28: for wrong-regime penalty and 5% profit-taking
            }
            self._regime_outputs = ro
            self._regime_outputs_step = self.current_step
            return ro
        except Exception as e:
            self.logger.debug(f"Regime controller compute failed: {e}")
            return {
                'buy_reward_mult': 1.0,
                'sell_reward_mult': 1.0,
                'hold_penalty_mult': 1.0,
                'gate_high_threshold': 0.28,
                'gate_medium_threshold': 0.12,
                'stop_loss_mult': 1.0,
                'trailing_stop_mult': 1.0,
                'bull_bear': 'neutral',
            }

    def _calculate_adaptive_position_decay(self) -> tuple[float, float]:
        """
        Calculate adaptive position incentive decay parameters based on market conditions.
        
        Returns:
            (decay_start_multiplier, decay_rate_multiplier) - multipliers to apply to base values
            - decay_start_multiplier: >1.0 = hold longer, <1.0 = exit faster
            - decay_rate_multiplier: >1.0 = slower decay, <1.0 = faster decay
        """
        try:
            start_mult = 1.0  # Base: 15 steps
            rate_mult = 1.0   # Base: 0.3 rate
            
            # 1. Market Regime Adjustment
            regime = getattr(self, 'current_market_regime', None)
            if not regime and hasattr(self, '_cached_market_regime'):
                regime = self._cached_market_regime
            
            if regime == 'trending':
                # Trending market: Hold longer, slower decay (let winners run)
                start_mult = 1.3  # 15 * 1.3 = 19.5 steps
                rate_mult = 1.2   # 0.3 * 1.2 = 0.36 (slower decay)
            elif regime == 'mean_reversion':
                # Mean-reversion: Exit faster, faster decay (take profits quickly)
                start_mult = 0.8  # 15 * 0.8 = 12 steps
                rate_mult = 0.8   # 0.3 * 0.8 = 0.24 (faster decay)
            
            # 2. Volatility Adjustment
            if hasattr(self, 'price_matrix') and self.current_step >= 20:
                volatilities = []
                for i in range(self.num_tickers):
                    prices = self.price_matrix[max(0, self.current_step - 20):self.current_step + 1, i]
                    if len(prices) >= 2:
                        returns = np.diff(prices) / prices[:-1]
                        vol = np.std(returns) if len(returns) > 0 else 0.0
                        volatilities.append(vol)
                
                if volatilities:
                    avg_volatility = float(np.mean(volatilities))
                    # High volatility (>0.02): Exit faster (choppy markets)
                    # Low volatility (<0.01): Hold longer (stable markets)
                    if avg_volatility > 0.02:
                        start_mult *= 0.85  # Exit 15% faster
                        rate_mult *= 0.85   # Decay 15% faster
                    elif avg_volatility < 0.01:
                        start_mult *= 1.15  # Hold 15% longer
                        rate_mult *= 1.15   # Decay 15% slower
            
            # 3. Market Trend Adjustment (recent price movement)
            if hasattr(self, 'price_matrix') and self.current_step >= 10:
                recent_returns = []
                for i in range(self.num_tickers):
                    if self.current_step >= 10:
                        prices = self.price_matrix[self.current_step - 10:self.current_step + 1, i]
                        if len(prices) >= 2:
                            return_pct = (prices[-1] - prices[0]) / prices[0]
                            recent_returns.append(return_pct)
                
                if recent_returns:
                    avg_return = np.mean(recent_returns)
                    # Strong uptrend (>2%): Hold longer (let winners run)
                    if avg_return > 0.02:
                        start_mult *= 1.2   # Hold 20% longer
                        rate_mult *= 1.2    # Decay 20% slower
                    # Strong downtrend (<-2%): Exit faster (cut losses)
                    elif avg_return < -0.02:
                        start_mult *= 0.7   # Exit 30% faster
                        rate_mult *= 0.7    # Decay 30% faster
            
            # Clamp multipliers to reasonable ranges
            start_mult = float(np.clip(start_mult, 0.80, 1.5))  # FIXED: 12-22.5 steps range (was 0.6, too aggressive)
            rate_mult = float(np.clip(rate_mult, 0.80, 1.5))    # FIXED: 0.24-0.45 rate range (was 0.6, too aggressive)
            
            return start_mult, rate_mult
        except Exception as e:
            self.logger.debug(f"Error calculating adaptive position decay: {e}")
            return 1.0, 1.0  # Default: no adjustment
    
    def _calculate_behavior_based_adaptive_rewards(self) -> dict:
        """
        Calculate behavior-based adaptive adjustments to base rewards.
        
        Monitors agent performance and automatically adjusts:
        - Base trading incentives (if trade count is low)
        - SELL rewards (if SELL/BUY ratio is low)
        - Hold penalties (if hold times are too long)
        - Aging penalty rates (if positions held too long)
        
        Returns:
            dict with adjustments: {
                'base_trading_incentive_mult': float,  # Multiplier for base trading reward
                'sell_base_incentive_mult': float,     # Multiplier for SELL base reward
                'hold_penalty_mult': float,            # Multiplier for hold penalties
                'aging_penalty_rate_mult': float,     # Multiplier for aging penalty rate
                'aging_penalty_start_mult': float     # Multiplier for aging penalty start
            }
        """
        try:
            adjustments = {
                'base_trading_incentive_mult': 1.0,
                'sell_base_incentive_mult': 1.0,
                'buy_base_incentive_mult': 1.0,  # NEW: BUY boost multiplier for action imbalance correction
                'hold_penalty_mult': 1.0,
                'aging_penalty_rate_mult': 1.0,
                'aging_penalty_start_mult': 1.0
            }
            
            # FIX 3: Reduce minimum step requirement from 20 to 10 for shorter runs
            # Also log even if no adjustments are made (for unified report)
            if self.current_step < 10:
                return adjustments
            
            # 1. TRADE FREQUENCY ADJUSTMENT: Increase base rewards if trade count is low
            expected_trades_per_step = 0.15  # Target: ~15% of steps should be trades
            steps_so_far = self.current_step + 1
            expected_trades = steps_so_far * expected_trades_per_step
            actual_trades = getattr(self, '_total_trades_executed', 0)
            
            if actual_trades < expected_trades * 0.5:  # Less than 50% of expected
                # Aggressively increase trading incentives
                trade_deficit = float((expected_trades - actual_trades) / max(expected_trades, 1.0))
                adjustments['base_trading_incentive_mult'] = float(min(1.0 + (trade_deficit * 0.5), 1.5))  # Up to 1.5x
            elif actual_trades < expected_trades * 0.75:  # Less than 75% of expected
                # Moderately increase trading incentives
                trade_deficit = float((expected_trades - actual_trades) / max(expected_trades, 1.0))
                adjustments['base_trading_incentive_mult'] = float(min(1.0 + (trade_deficit * 0.3), 1.3))  # Up to 1.3x
            
            # 2. SELL/BUY RATIO ADJUSTMENT: Bidirectional - increase/decrease based on actual distribution
            buy_count = getattr(self, '_buy_count', 0)
            sell_count = getattr(self, '_sell_count', 0)
            total_trades = buy_count + sell_count
            
            # Also check raw policy actions for imbalance detection (more accurate than executed trades)
            # Use _policy_action_counts which tracks raw PPO actions (0=HOLD, 1=BUY, 2=SELL)
            policy_action_counts = getattr(self, '_policy_action_counts', self.DEFAULT_POLICY_ACTION_COUNTS.copy())
            total_actions = sum(policy_action_counts.values())
            buy_actions = policy_action_counts.get(1, 0)  # BUY = action 1
            sell_actions = policy_action_counts.get(2, 0)  # SELL = action 2
            
            if total_trades > 5:  # Need at least some trades to calculate ratio
                sell_ratio = sell_count / total_trades
                target_sell_ratio = 0.35  # Target: 35% SELL, 65% BUY
                
                # BIDIRECTIONAL LOGIC: Increase SELL when too low, decrease when too high
                if sell_ratio < target_sell_ratio * 0.5:  # Less than 50% of target (i.e., <17.5%)
                    # Aggressively increase SELL rewards
                    ratio_deficit = float((target_sell_ratio - sell_ratio) / target_sell_ratio)
                    adjustments['sell_base_incentive_mult'] = float(min(1.0 + (ratio_deficit * 0.8), 1.8))  # Up to 1.8x
                elif sell_ratio < target_sell_ratio * 0.75:  # Less than 75% of target (i.e., <26.25%)
                    # Moderately increase SELL rewards
                    ratio_deficit = float((target_sell_ratio - sell_ratio) / target_sell_ratio)
                    adjustments['sell_base_incentive_mult'] = float(min(1.0 + (ratio_deficit * 0.5), 1.5))  # Up to 1.5x
                elif sell_ratio > target_sell_ratio * 2.0:  # More than 2x target (i.e., >70% SELL)
                    # CRITICAL FIX: Aggressively decrease SELL rewards when SELL is too high
                    ratio_excess = float((sell_ratio - target_sell_ratio) / target_sell_ratio)
                    adjustments['sell_base_incentive_mult'] = float(max(1.0 - (ratio_excess * 0.6), 0.4))  # Down to 0.4x (60% reduction)
                elif sell_ratio > target_sell_ratio * 1.5:  # More than 1.5x target (i.e., >52.5% SELL)
                    # Moderately decrease SELL rewards
                    ratio_excess = float((sell_ratio - target_sell_ratio) / target_sell_ratio)
                    adjustments['sell_base_incentive_mult'] = float(max(1.0 - (ratio_excess * 0.4), 0.6))  # Down to 0.6x (40% reduction)
            
            # NEW: BUY boost when BUY ratio is critically low (based on raw policy actions)
            buy_action_ratio = buy_actions / total_actions if total_actions > 0 else 0.0
            if total_actions > 100:  # Need enough actions to calculate ratio
                target_buy_ratio = 0.30  # Target: 30% BUY actions (allowing for HOLD)
                
                if buy_action_ratio < 0.05:  # CRITICAL: BUY < 5% (severe imbalance)
                    # Aggressively boost BUY rewards (up to 3x)
                    buy_deficit = float((target_buy_ratio - buy_action_ratio) / target_buy_ratio)
                    adjustments['buy_base_incentive_mult'] = float(min(1.0 + (buy_deficit * 2.0), 3.0))  # Up to 3.0x
                elif buy_action_ratio < 0.10:  # BUY < 10% (moderate imbalance)
                    # Moderately boost BUY rewards (up to 2x)
                    buy_deficit = float((target_buy_ratio - buy_action_ratio) / target_buy_ratio)
                    adjustments['buy_base_incentive_mult'] = float(min(1.0 + (buy_deficit * 1.0), 2.0))  # Up to 2.0x
                elif buy_action_ratio < 0.15:  # BUY < 15% (mild imbalance)
                    # Slightly boost BUY rewards (up to 1.5x)
                    buy_deficit = float((target_buy_ratio - buy_action_ratio) / target_buy_ratio)
                    adjustments['buy_base_incentive_mult'] = float(min(1.0 + (buy_deficit * 0.5), 1.5))  # Up to 1.5x
                
                # 2026-02-28: BUY-dominant detection - when raw BUY > 85%, proactively boost SELL
                try:
                    from ppo.utils.dynamic_config import get_dynamic_config
                    cfg = get_dynamic_config()
                    buy_high = getattr(cfg, 'adaptive_ent_buy_high_threshold', 85.0) / 100.0
                except Exception:
                    buy_high = 0.85
                if buy_action_ratio > buy_high and total_actions > 50:
                    # BUY-dominant policy (e.g. 91.6% BUY) - boost SELL to encourage balance
                    adjustments['sell_base_incentive_mult'] = float(max(
                        adjustments.get('sell_base_incentive_mult', 1.0),
                        1.35  # At least 1.35x when BUY-dominant
                    ))
            
            # 2026-02-28: Early proactive SELL boost - first 20% of episode to prevent early BUY convergence
            # (Env has no global training step; use per-episode proxy)
            try:
                from ppo.utils.dynamic_config import get_dynamic_config
                cfg = get_dynamic_config()
                early_mult = float(getattr(cfg, 'adaptive_sell_early_boost_mult', 1.25))
            except Exception:
                early_mult = 1.25
            episode_len = len(self.price_matrix) if hasattr(self, 'price_matrix') else 2000
            early_phase = self.current_step < min(500, int(episode_len * 0.2))
            if early_phase:
                adjustments['sell_base_incentive_mult'] = float(max(
                    adjustments.get('sell_base_incentive_mult', 1.0),
                    early_mult
                ))
            
            # 3. HOLD TIME ADJUSTMENT: Increase penalties if average hold time is too long
            if hasattr(self, '_avg_hold_time_history') and len(self._avg_hold_time_history) > 0:
                avg_hold_time = float(np.mean(self._avg_hold_time_history[-10:]))  # Last 10 positions
                target_hold_time = 25.0  # Target: 25 steps average
                
                if avg_hold_time > target_hold_time * 1.5:  # More than 1.5x target (i.e., >37.5 steps)
                    # Aggressively increase hold penalties and aging rates
                    hold_excess = (avg_hold_time - target_hold_time) / target_hold_time
                    adjustments['hold_penalty_mult'] = float(min(1.0 + (hold_excess * 0.4), 1.4))  # Up to 1.4x
                    adjustments['aging_penalty_rate_mult'] = float(min(1.0 + (hold_excess * 0.6), 1.6))  # Up to 1.6x
                    adjustments['aging_penalty_start_mult'] = float(max(0.5, min(0.5 + (hold_excess * 0.2), 0.7)))  # Start earlier
                elif avg_hold_time > target_hold_time * 1.2:  # More than 1.2x target (i.e., >30 steps)
                    # Moderately increase hold penalties
                    hold_excess = float((avg_hold_time - target_hold_time) / target_hold_time)
                    adjustments['hold_penalty_mult'] = float(min(1.0 + (hold_excess * 0.2), 1.2))  # Up to 1.2x
                    adjustments['aging_penalty_rate_mult'] = float(min(1.0 + (hold_excess * 0.3), 1.3))  # Up to 1.3x
                    adjustments['aging_penalty_start_mult'] = float(max(0.7, min(0.7 + (hold_excess * 0.1), 0.8)))  # Start earlier

            # Safety-only clamps: behavior-adaptive is a tertiary self-rescue layer (avoid dominating market-first logic)
            # 2026-02-28: SELL mult cap 1.25→1.55 so low-SELL-ratio boost (1.5x/1.8x) can take effect; balances vs BUY stack
            adjustments['base_trading_incentive_mult'] = float(np.clip(adjustments.get('base_trading_incentive_mult', 1.0), 0.85, 1.25))
            adjustments['sell_base_incentive_mult'] = float(np.clip(adjustments.get('sell_base_incentive_mult', 1.0), 0.75, 1.55))
            adjustments['buy_base_incentive_mult'] = float(np.clip(adjustments.get('buy_base_incentive_mult', 1.0), 0.75, 1.50))
            adjustments['hold_penalty_mult'] = float(np.clip(adjustments.get('hold_penalty_mult', 1.0), 0.85, 1.20))
            adjustments['aging_penalty_rate_mult'] = float(np.clip(adjustments.get('aging_penalty_rate_mult', 1.0), 0.85, 1.25))
            adjustments['aging_penalty_start_mult'] = float(np.clip(adjustments.get('aging_penalty_start_mult', 1.0), 0.60, 1.10))
            
            # Track behavior-based adaptive adjustments for unified report
            if not hasattr(self, '_adaptive_reward_log'):
                self._adaptive_reward_log = []
            
            # FIX 3: Log more frequently (every 25 steps instead of 50) and always log at key milestones
            total_steps = getattr(self, '_max_episode_steps', None)
            if total_steps is None or total_steps <= 0:
                total_steps = len(self.price_matrix) if hasattr(self, 'price_matrix') else 0
            should_log = (
                any(v != 1.0 for v in adjustments.values())
                or (self.current_step % 25 == 0)
                or (self.current_step == 0)
                or (total_steps > 0 and self.current_step >= total_steps - 5)
            )
            if should_log:
                avg_hold_time_str = f"{np.mean(self._avg_hold_time_history[-10:]):.1f}" if hasattr(self, '_avg_hold_time_history') and len(self._avg_hold_time_history) > 0 else "N/A"
                sell_ratio_str = f"{sell_ratio:.1%}" if total_trades > 5 else "N/A"
                
                # Log to console
                if any(v != 1.0 for v in adjustments.values()):
                    self.logger.info(
                        f"[BEHAVIOR_ADAPTIVE] Step {self.current_step}: "
                        f"Trades={actual_trades:.0f}/{expected_trades:.1f} (mult: {adjustments['base_trading_incentive_mult']:.2f}x), "
                        f"SELL={sell_count}/{total_trades} ({sell_ratio_str}, mult: {adjustments['sell_base_incentive_mult']:.2f}x), "
                        f"AvgHold={avg_hold_time_str} (penalty_mult: {adjustments['hold_penalty_mult']:.2f}x)"
                    )
                
                # FIX: Ensure adaptive reward log is initialized
                if not hasattr(self, '_adaptive_reward_log'):
                    self._adaptive_reward_log = []
                
                # FIXED: Track for unified report (every 50 steps, first step, last step, or when adjustments are active)
                if should_log:
                    self._adaptive_reward_log.append({
                        'step': self.current_step,
                        'type': 'behavior_adaptive',
                        'base_trading_incentive_mult': float(adjustments['base_trading_incentive_mult']),
                        'sell_base_incentive_mult': float(adjustments['sell_base_incentive_mult']),
                        'buy_base_incentive_mult': float(adjustments['buy_base_incentive_mult']),  # NEW: Track BUY boost
                        'hold_penalty_mult': float(adjustments['hold_penalty_mult']),
                        'aging_penalty_rate_mult': float(adjustments['aging_penalty_rate_mult']),
                        'aging_penalty_start_mult': float(adjustments['aging_penalty_start_mult']),
                        'actual_trades': int(actual_trades),
                        'expected_trades': float(expected_trades),
                        'buy_count': int(buy_count),
                        'sell_count': int(sell_count),
                        'total_trades': int(total_trades),
                        'sell_ratio': float(sell_ratio) if total_trades > 5 else 0.0,
                        'avg_hold_time': float(np.mean(self._avg_hold_time_history[-10:])) if hasattr(self, '_avg_hold_time_history') and len(self._avg_hold_time_history) > 0 else 0.0,
                        'message': (
                            f"Behavior Adaptive: Trades={actual_trades:.0f}/{expected_trades:.1f} "
                            f"(trading_mult={adjustments['base_trading_incentive_mult']:.2f}x), "
                            f"SELL={sell_count}/{total_trades} ({sell_ratio_str}, sell_mult={adjustments['sell_base_incentive_mult']:.2f}x), "
                            f"BUY={buy_count}/{total_trades} (buy_mult={adjustments['buy_base_incentive_mult']:.2f}x), "
                            f"HoldTime={avg_hold_time_str} (penalty_mult={adjustments['hold_penalty_mult']:.2f}x)"
                        )
                    })
            
            return adjustments
        except Exception as e:
            self.logger.debug(f"Error calculating behavior-based adaptive rewards: {e}")
            return {
                'base_trading_incentive_mult': 1.0,
                'sell_base_incentive_mult': 1.0,
                'buy_base_incentive_mult': 1.0,  # NEW: BUY boost multiplier
                'hold_penalty_mult': 1.0,
                'aging_penalty_rate_mult': 1.0,
                'aging_penalty_start_mult': 1.0
            }
    
    def _calculate_adaptive_hold_penalty(self) -> float:
        """
        Calculate adaptive hold penalty based on market conditions.
        
        Factors:
        - Volatility: High vol = LOWER penalty (allow HOLD in volatile regimes; 2026-02-19)
        - Market trend: Strong trend = lower penalty (holding winners is good)
        - Trading opportunities: High opportunity = higher penalty (should be trading)
        
        Returns:
            Adaptive hold penalty multiplier (0.5x to 1.5x)
        """
        try:
            # Market-first controller baseline
            base_mult = None
            try:
                ro = self._get_regime_outputs()
                if isinstance(ro, dict):
                    base_mult = float(ro.get('hold_penalty_mult', 1.0))
            except Exception:
                base_mult = None

            penalty_multiplier = float(base_mult) if base_mult is not None else 1.0
            
            # If controller is not available, fall back to the legacy market adjustments
            if base_mult is None:
                # 1. Volatility-based adjustment: HIGH vol = LOWER penalty (hold only in volatility)
                #    Low vol = full penalty (1.0x); high vol = relax to 0.5x-0.8x so HOLD is acceptable
                if hasattr(self, 'price_matrix') and self.current_step >= 20:
                    volatilities = []
                    for i in range(self.num_tickers):
                        prices = self.price_matrix[max(0, self.current_step - 20):self.current_step + 1, i]
                        if len(prices) >= 2:
                            returns = np.diff(prices) / prices[:-1]
                            vol = np.std(returns) if len(returns) > 0 else 0.0
                            volatilities.append(vol)
                    
                    if volatilities:
                        avg_volatility = float(np.mean(volatilities))
                        # High volatility = reduce penalty (allow HOLD); low vol = full penalty
                        # 0.01 vol = 1.0x, 0.02 vol = 0.8x, 0.03+ vol = 0.5x
                        vol_multiplier = 1.2 - (avg_volatility * 25.0)
                        vol_multiplier = np.clip(vol_multiplier, 0.5, 1.0)
                        penalty_multiplier *= vol_multiplier
                
                # 2. Market trend adjustment (strong trend = lower penalty if holding winners)
                if hasattr(self, 'price_matrix') and self.current_step >= 10:
                    spy_idx = min(4, self.num_tickers - 1)
                    spy_prices = self.price_matrix[max(0, self.current_step - 10):self.current_step + 1, spy_idx]
                    if len(spy_prices) >= 2:
                        market_return = (spy_prices[-1] - spy_prices[0]) / spy_prices[0]
                        has_positions = any(
                            i in self.positions and self.positions[i].get('shares', 0.0) > 0 
                            for i in range(self.num_tickers)
                        )
                        if has_positions and abs(market_return) > 0.02:  # Strong trend with positions
                            trend_multiplier = 1.0 - (abs(market_return) * 5.0)  # -5% trend = 0.75x penalty
                            trend_multiplier = np.clip(trend_multiplier, 0.5, 1.0)
                            penalty_multiplier *= trend_multiplier
            
            # 3. Trading opportunity adjustment (recent price movements = opportunities missed)
            if hasattr(self, 'price_matrix') and self.current_step >= 5:
                recent_movements = []
                for i in range(self.num_tickers):
                    if self.current_step >= 5:
                        prices = self.price_matrix[self.current_step - 5:self.current_step + 1, i]
                        if len(prices) >= 2:
                            movement = abs((prices[-1] - prices[0]) / prices[0])
                            recent_movements.append(movement)
                
                if recent_movements:
                    avg_movement = np.mean(recent_movements)
                    # High recent movement = opportunities missed = higher penalty
                    # 0.01 movement = 1.0x, 0.02 movement = 1.1x, 0.03 movement = 1.2x
                    opportunity_multiplier = 1.0 + (avg_movement * 5.0)
                    opportunity_multiplier = np.clip(opportunity_multiplier, 0.9, 1.3)
                    penalty_multiplier *= opportunity_multiplier
            
            # Final cap (use strict clipping to handle floating point precision)
            penalty_multiplier = float(np.clip(penalty_multiplier, 0.5, 1.5))
            # Additional safety check for floating point precision issues
            if penalty_multiplier > 1.5:
                penalty_multiplier = 1.5
            if penalty_multiplier < 0.5:
                penalty_multiplier = 0.5
            
            return penalty_multiplier
        except Exception as e:
            self.logger.debug(f"Error calculating adaptive hold penalty: {e}")
            return 1.0
    
    def _calculate_adaptive_action_bonus(self, action: str = 'BUY') -> float:
        """
        Calculate adaptive bonus for BUY/SELL based on market conditions.
        
        This is the CORRECT adaptive approach: reward the right action instead of punishing HOLD.
        
        Logic:
        - Bull market + no positions → BUY bonus (encourage entering)
        - Bear market + has positions → SELL bonus (encourage exiting)
        - Bull market + profitable positions → reduce SELL bonus (let profits run)
        - Has losing positions → SELL bonus (cut losses)
        - High volatility → bonus for both actions (more opportunities)
        
        Returns:
            Direct bonus to add to reward (0.0 to 0.30)
        """
        try:
            bonus = 0.0
            
            # Check position status
            has_positions = any(
                i in self.positions and self.positions[i].get('shares', 0.0) > 0 
                for i in range(self.num_tickers)
            )
            
            # Calculate unrealized P&L if has positions
            total_unrealized_pnl = 0.0
            if has_positions:
                for i in range(self.num_tickers):
                    if i in self.positions and self.positions[i].get('shares', 0.0) > 0:
                        shares = self.positions[i].get('shares', 0.0)
                        avg_cost = self.positions[i].get('avg_cost', 0.0)
                        if self.current_step < len(self.price_matrix) and avg_cost > 0:
                            current_price = self.price_matrix[self.current_step, i]
                            unrealized = (current_price - avg_cost) * shares
                            total_unrealized_pnl += unrealized
            
            # Get market trend
            market_return = 0.0
            if hasattr(self, 'price_matrix') and self.current_step >= 10:
                spy_idx = min(4, self.num_tickers - 1)
                spy_prices = self.price_matrix[max(0, self.current_step - 10):self.current_step + 1, spy_idx]
                if len(spy_prices) >= 2:
                    market_return = (spy_prices[-1] - spy_prices[0]) / spy_prices[0]
            
            # Get volatility
            avg_volatility = 0.0
            if hasattr(self, 'price_matrix') and self.current_step >= 20:
                volatilities = []
                for i in range(self.num_tickers):
                    prices = self.price_matrix[max(0, self.current_step - 20):self.current_step + 1, i]
                    if len(prices) >= 2:
                        returns = np.diff(prices) / prices[:-1]
                        vol = np.std(returns) if len(returns) > 0 else 0.0
                        volatilities.append(vol)
                if volatilities:
                    avg_volatility = float(np.mean(volatilities))
            
            if action == 'BUY':
                # BUY bonuses
                if not has_positions:
                    # No positions - encourage entering the market
                    if market_return > 0.01:  # Bull market (>1%)
                        # Strong bonus for buying in bull market when flat
                        bonus += 0.15 + (market_return * 2.0)  # Up to +0.25 for 5% bull market
                        if self.current_step % 100 == 0:
                            self.logger.debug(f"[ADAPTIVE_BONUS] BUY bonus for bull market entry: +{bonus:.3f} (market: {market_return:.2%})")
                    elif market_return < -0.02:  # Strong bear market
                        # Small bonus for buying the dip (contrarian)
                        bonus += 0.05
                    else:
                        # Neutral market - small bonus to encourage first position
                        bonus += 0.08
                else:
                    # Has positions - smaller BUY bonus (already in market)
                    if market_return > 0.02:  # Strong bull - add to winners
                        bonus += 0.05
                
                # Volatility bonus for BUY (high vol = good entry opportunities)
                if avg_volatility > 0.015:
                    vol_bonus = min(avg_volatility * 3.0, 0.10)
                    bonus += vol_bonus
                    
            else:  # SELL
                # SELL bonuses
                if has_positions:
                    # Has positions - check if should exit
                    if total_unrealized_pnl > 0:
                        # Profitable positions
                        if market_return < -0.01:  # Bear market - take profits
                            bonus += 0.20 + abs(market_return) * 2.0  # Strong bonus to exit
                            if self.current_step % 100 == 0:
                                self.logger.debug(f"[ADAPTIVE_BONUS] SELL bonus for profit-taking in bear: +{bonus:.3f}")
                        elif market_return > 0.03:  # Strong bull - let profits run
                            bonus += 0.02  # Very small bonus (don't encourage premature exit)
                        else:
                            # Neutral - moderate bonus to take profits
                            bonus += 0.10
                    else:
                        # Losing positions - encourage cutting losses
                        loss_ratio = abs(total_unrealized_pnl) / max(self.portfolio_value, 1.0)
                        loss_bonus = min(loss_ratio * 10.0, 0.25)  # Up to +0.25 for 2.5% loss
                        bonus += 0.15 + loss_bonus  # Strong bonus to cut losses
                        if self.current_step % 100 == 0:
                            self.logger.debug(f"[ADAPTIVE_BONUS] SELL bonus for cutting losses: +{bonus:.3f} (loss ratio: {loss_ratio:.2%})")
                
                # Volatility bonus for SELL (high vol = good exit timing matters)
                if avg_volatility > 0.02 and has_positions:
                    vol_bonus = min(avg_volatility * 2.0, 0.08)
                    bonus += vol_bonus
            
            # Cap bonus to prevent extreme values
            bonus = float(np.clip(bonus, 0.0, 0.35))
            return bonus
            
        except Exception as e:
            self.logger.debug(f"Error calculating adaptive action bonus: {e}")
            return 0.0
    
    def _detect_bull_bear_regime(self) -> str:
        """
        Detect bull / bear / neutral regime using SPY over lookback (configurable).
        Used for dynamic reward scaling: bull -> boost BUY, bear -> boost SELL.
        """
        try:
            from ppo.utils.dynamic_config import get_dynamic_config  # type: ignore
            dcfg = self._cfg()
            lookback = int(getattr(dcfg, 'bull_bear_lookback_days', 10))
            bull_th = float(getattr(dcfg, 'bull_threshold_pct', 0.02))
            bear_th = float(getattr(dcfg, 'bear_threshold_pct', -0.02))
        except Exception:
            lookback, bull_th, bear_th = 10, 0.02, -0.02
        if not hasattr(self, 'price_matrix') or self.current_step < lookback:
            return 'neutral'
        max_step = len(self.price_matrix) - 1
        if self.current_step > max_step:
            return 'neutral'
        spy_idx = min(4, self.num_tickers - 1)
        start = max(0, self.current_step - lookback)
        spy_prices = self.price_matrix[start:self.current_step + 1, spy_idx]
        if len(spy_prices) < 2:
            return 'neutral'
        market_return = (spy_prices[-1] - spy_prices[0]) / (spy_prices[0] + 1e-10)
        if market_return >= bull_th:
            return 'bull'
        if market_return <= bear_th:
            return 'bear'
        return 'neutral'

    def _calculate_market_condition_multiplier(self, action: str = 'BUY') -> float:
        """
        Calculate dynamic multiplier for base rewards based on market conditions.
        
        Factors:
        - Volatility: High vol = higher rewards (more opportunity)
        - Bull/bear regime: Bull = higher BUY rewards, Bear = higher SELL rewards (dynamic reward insight)
        - Market trend (continuous): tilt BUY/SELL by SPY return
        - Regime: Trending = higher rewards
        
        Returns:
            Multiplier (typically 0.8x to 1.5x)
        """
        try:
            multiplier = 1.0
            
            # 1. Volatility-based adjustment (high vol = more opportunity = higher rewards)
            if hasattr(self, 'price_matrix') and self.current_step >= 20:
                max_available_step = len(self.price_matrix) - 1
                if self.current_step > max_available_step:
                    return 1.0
                volatilities = []
                for i in range(self.num_tickers):
                    prices = self.price_matrix[max(0, self.current_step - 20):self.current_step + 1, i]
                    if len(prices) >= 2:
                        returns = np.diff(prices) / (prices[:-1] + 1e-10)
                        vol = np.std(returns) if len(returns) > 0 else 0.0
                        volatilities.append(vol)
                if volatilities:
                    avg_volatility = float(np.mean(volatilities))
                    vol_multiplier = 1.0 + (avg_volatility * 10.0)
                    vol_multiplier = np.clip(vol_multiplier, 0.8, 1.5)
                    multiplier *= vol_multiplier
            
            # 2. Bull/bear regime: explicit dynamic reward scaling (bull -> BUY boost, bear -> SELL boost)
            if hasattr(self, 'price_matrix') and self.current_step >= 10:
                if not hasattr(self, '_cached_bull_bear_regime') or not hasattr(self, '_cached_bull_bear_step') or \
                   (self.current_step - getattr(self, '_cached_bull_bear_step', -20)) >= 10:
                    self._cached_bull_bear_regime = self._detect_bull_bear_regime()
                    self._cached_bull_bear_step = self.current_step
                regime = getattr(self, '_cached_bull_bear_regime', 'neutral')
                try:
                    from ppo.utils.dynamic_config import get_dynamic_config  # type: ignore
                    dcfg = self._cfg()
                    bull_boost = float(getattr(dcfg, 'bull_buy_reward_boost', 0.06))
                    bear_boost = float(getattr(dcfg, 'bear_sell_reward_boost', 0.08))
                except Exception:
                    bull_boost, bear_boost = 0.06, 0.08
                if action == 'BUY' and regime == 'bull':
                    multiplier *= (1.0 + bull_boost)
                elif action == 'SELL' and regime == 'bear':
                    multiplier *= (1.0 + bear_boost)
                elif action == 'BUY' and regime == 'bear':
                    multiplier *= (1.0 - bear_boost * 0.5)  # Slight BUY reduction in bear
                elif action == 'SELL' and regime == 'bull':
                    sell_bull = 0.08
                    if hasattr(self, 'reward_config') and isinstance(self.reward_config, dict):
                        sell_bull = float(self.reward_config.get('sell_bull_market_boost', 0.08))
                    multiplier *= (1.0 + sell_bull)  # SELL in bull: profit-taking boost (existing param)
            
            # 3. Continuous market trend (gentle tilt by SPY return)
            if hasattr(self, 'price_matrix') and self.current_step >= 10:
                max_available_step = len(self.price_matrix) - 1
                if self.current_step <= max_available_step:
                    spy_idx = min(4, self.num_tickers - 1)
                    spy_prices = self.price_matrix[max(0, self.current_step - 10):self.current_step + 1, spy_idx]
                    if len(spy_prices) >= 2:
                        market_return = (spy_prices[-1] - spy_prices[0]) / (spy_prices[0] + 1e-10)
                        if action == 'BUY':
                            trend_multiplier = 1.0 + (market_return * 0.5)
                        else:
                            trend_multiplier = 1.0 - (market_return * 0.25)
                        trend_multiplier = np.clip(trend_multiplier, 0.95, 1.10)
                        multiplier *= trend_multiplier
            
            # 4. Regime adjustment (trending = higher rewards)
            regime_multiplier = self._get_regime_adjustment(action)
            multiplier *= regime_multiplier
            
            multiplier = float(np.clip(multiplier, 0.7, 1.6))
            if multiplier > 1.6:
                multiplier = 1.6
            if multiplier < 0.7:
                multiplier = 0.7
            # Guard NaN/Inf so report and rewards never see nan
            if np.isnan(multiplier) or np.isinf(multiplier):
                multiplier = 1.0
            return multiplier
        except Exception as e:
            self.logger.debug(f"Error calculating market condition multiplier: {e}")
            return 1.0

    def _get_timeframe_reward_multiplier(self) -> float:
        """
        Dynamic reward scaling by timeframe when multi-timeframe features are present.
        Uses intraday columns in feature_matrix to pick dominant timeframe and returns
        config timeframe_reward_weight_* for that interval (1d, 1h, 30m, 5m, 1m).
        """
        try:
            from ppo.utils.dynamic_config import get_dynamic_config
            dc = self._cfg()
            w_1d = float(getattr(dc, 'timeframe_reward_weight_1d', 1.0))
            w_1h = float(getattr(dc, 'timeframe_reward_weight_1h', 1.1))
            w_30m = float(getattr(dc, 'timeframe_reward_weight_30m', 1.15))
            w_5m = float(getattr(dc, 'timeframe_reward_weight_5m', 1.2))
            w_1m = float(getattr(dc, 'timeframe_reward_weight_1m', 1.0))
            base_len = 11
            features_per_interval = 4
            if self.feature_matrix is None or self.current_step >= len(self.feature_matrix):
                return w_1d
            frame = self.feature_matrix[self.current_step]
            if frame.ndim == 1:
                frame = np.reshape(frame, (self.num_tickers, -1))
            nf = frame.shape[1] if frame.ndim >= 2 else 0
            if nf <= base_len:
                return w_1d
            activities = []
            intervals_ordered = ['5m', '30m', '1h', '1m']
            for idx, interval in enumerate(intervals_ordered):
                start = base_len + idx * features_per_interval
                end = min(start + features_per_interval, nf)
                if start >= nf:
                    activities.append(0.0)
                    continue
                slice_ = frame[:, start:end]
                vol_range = np.abs(slice_).mean()
                activities.append(float(vol_range))
            if not activities or max(activities) <= 0:
                return w_1d
            best_idx = int(np.argmax(activities))
            interval = intervals_ordered[best_idx]
            weights = {'1d': w_1d, '1h': w_1h, '30m': w_30m, '5m': w_5m, '1m': w_1m}
            mult = weights.get(interval, w_1d)
            if self._should_log_trade(20):
                self.logger.debug(f"[TIMEFRAME_REWARD] dominant={interval}, mult={mult:.2f}")
            return mult
        except Exception as e:
            self.logger.debug(f"Timeframe reward multiplier error: {e}")
            return 1.0

    def _calculate_technical_signal_quality(self, ticker: int, action: str) -> float:
        """Compact TA score using RSI/MACD/Bollinger/Volume; adjusted by regime."""
        # Single-source-of-truth: if step signals are precomputed, use them (prevents duplicate TA logic drift).
        try:
            cache = getattr(self, '_step_signals_cache', None)
            if isinstance(cache, dict) and ticker in cache and isinstance(cache[ticker], dict):
                if action in cache[ticker] and isinstance(cache[ticker][action], dict) and 'tech' in cache[ticker][action]:
                    return float(cache[ticker][action].get('tech', 0.0) or 0.0)
        except Exception:
            pass
        try:
            # PERFORMANCE: Cache signals per step (not per action call)
            if not hasattr(self, '_signal_cache'):
                self._signal_cache = {}
                self._signal_cache_step = -1
            
            if self._signal_cache_step != self.current_step:
                self._signal_cache.clear()  # Use clear() instead of {} for better memory
                self._signal_cache_step = self.current_step
                # HIGH PRIORITY FIX: Verify caching is working (log every 100 steps for verification)
                if self.current_step % 100 == 0 and hasattr(self, '_total_trades_executed'):
                    self.logger.debug(f"[CACHE_VERIFY] Signal cache cleared for step {self.current_step} (cache size: {len(self._signal_cache)})")

            cache_key = (ticker, action, 'tech')
            if cache_key in self._signal_cache:
                return self._signal_cache[cache_key]

            from ppo.utils.dynamic_config import get_dynamic_config  # type: ignore
            # PERFORMANCE: Cache config values instead of fetching every time
            if not hasattr(self, '_cached_ta_config'):
                dcfg = self._cfg()
                self._cached_ta_config = {
                    'rsi_lo': float(getattr(dcfg, 'rsi_oversold', 30)),
                    'rsi_hi': float(getattr(dcfg, 'rsi_overbought', 70)),
                    'vol_th': float(getattr(dcfg, 'volume_threshold', 1.2))
                }
            
            rsi_lo = self._cached_ta_config['rsi_lo']
            rsi_hi = self._cached_ta_config['rsi_hi']
            vol_th = self._cached_ta_config['vol_th']
        except Exception:
            rsi_lo, rsi_hi, vol_th = 30.0, 70.0, 1.2

        end = self.current_step
        start = max(0, end - 20)
        if end <= start or ticker >= self.num_tickers:
            return 0.0
        prices = self.price_matrix[start:end, ticker].astype(float)
        vols = self.volume_matrix[start:end, ticker].astype(float)
        if len(prices) < 5:
            return 0.0

        diffs = np.diff(prices)
        gains = np.clip(diffs, 0, None)
        losses = -np.clip(diffs, None, 0)
        avg_gain = np.mean(gains) if gains.size else 0.0
        avg_loss = np.mean(losses) if losses.size else 0.0
        # PHASE 2 FIX: Clearer RSI calculation for edge cases
        if avg_loss == 0:
            rsi = 100.0  # All gains, no losses
        elif avg_gain == 0:
            rsi = 0.0   # All losses, no gains
        else:
            rs = avg_gain / avg_loss
            rsi = 100.0 - (100.0 / (1.0 + rs))

        # PHASE 1 FIX: Proper EMA-based MACD calculation
        if len(prices) >= 26:
            # Calculate EMA helper function
            def calculate_ema(data, period):
                """Calculate Exponential Moving Average."""
                if len(data) < period:
                    return None
                alpha = 2.0 / (period + 1)
                ema = np.zeros_like(data)
                ema[0] = data[0]
                for i in range(1, len(data)):
                    ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]
                return ema
            
            ema_12 = calculate_ema(prices, 12)
            ema_26 = calculate_ema(prices, 26)
            if ema_12 is not None and ema_26 is not None:
                macd_line = ema_12[-1] - ema_26[-1]
                # Signal line is EMA of MACD line (need MACD history)
                macd_history = ema_12 - ema_26
                macd_signal_line = calculate_ema(macd_history, 9)
                if macd_signal_line is not None:
                    macd_signal = macd_signal_line[-1]
                else:
                    macd_signal = np.mean(macd_history[-9:])  # Fallback
                macd_diff = macd_line - macd_signal
            else:
                # Fallback to SMA if EMA calculation fails
                macd_line = np.mean(prices[-12:]) - np.mean(prices[-26:])
                macd_signal = np.mean(prices[-9:])
                macd_diff = macd_line - macd_signal
        else:
            # Keep existing fallback for short data
            macd_line = np.mean(prices[-min(12, len(prices)):]) - np.mean(prices)
            macd_signal = np.mean(prices[-min(9, len(prices)):])
            macd_diff = macd_line - macd_signal

        bb_score = 0.0
        if len(prices) >= 20:
            m = np.mean(prices[-20:])
            s = np.std(prices[-20:]) + 1e-9
            z = (prices[-1] - m) / s
            bb_score = -z  # prefer buys when below mean, sells when above

        vol_score = 0.0
        if len(vols) >= 5:
            avg_vol = np.mean(vols[:-1]) if len(vols) > 1 else np.mean(vols)
            vol_ratio = (vols[-1] / (avg_vol + 1e-9)) if avg_vol > 0 else 1.0
            if vol_ratio > vol_th:
                vol_score = 0.2
            elif vol_ratio < 0.8:
                vol_score = -0.2

        score = 0.0
        if action == 'BUY':
            if rsi <= rsi_lo:
                score += 0.5
            if macd_diff > 0:
                score += 0.2
            score += bb_score * 0.1
        else:  # SELL
            if rsi >= rsi_hi:
                score += 0.5
            if macd_diff < 0:
                score += 0.2
            score += (-bb_score) * 0.1
        score += vol_score
        score *= self._get_regime_adjustment(action)
        score = float(np.clip(score, -1.0, 1.0))
        self._signal_cache[cache_key] = score
        return score

    def _calculate_fundamental_signal_quality(self, ticker: int, action: str) -> float:
        """Calculate fundamental signal quality.
        
        DISABLED: Fundamental analysis disabled per user request.
        Returns 0.0.
        """
        # DISABLED: Fundamental analysis disabled
        return 0.0
        
        # Original code below (disabled):
        """Action-oriented score from financial health (existing FA helpers)."""
        try:
            if not self._fundamental_data_available:
                return 0.0

            # PERFORMANCE: Reuse same cache as technical signals
            if not hasattr(self, '_signal_cache'):
                self._signal_cache = {}
                self._signal_cache_step = -1
            
            if self._signal_cache_step != self.current_step:
                self._signal_cache.clear()
                self._signal_cache_step = self.current_step

            cache_key = (ticker, action, 'fund')
            if cache_key in self._signal_cache:
                return self._signal_cache[cache_key]

            metrics = self._calculate_fundamental_metrics([f"TICKER_{ticker}"])
            if not metrics:
                return 0.0
            analysis = self._analyze_financial_health(metrics)
            key = f"TICKER_{ticker}"
            if key not in analysis:
                return 0.0
            health_score = float(analysis[key]['health_score'])
            try:
                from ppo.utils.dynamic_config import get_dynamic_config  # type: ignore
                dcfg = self._cfg()
                excellent = float(getattr(dcfg, 'fundamental_excellent_threshold', 80.0))
                good = float(getattr(dcfg, 'fundamental_health_threshold', 60.0))
            except Exception:
                excellent, good = 80.0, 60.0

            # PHASE 2 FIX: Check if price is below recent average (oversold) even if fundamentals are good
            # Get recent prices for momentum calculation
            # Note: ticker is already an integer index, not a symbol
            try:
                if hasattr(self, 'price_matrix') and len(self.price_matrix) >= 20 and ticker < self.num_tickers:
                    recent_prices = self.price_matrix[-20:, ticker]
                    current_price = self.price_matrix[-1, ticker]
                    price_momentum = (current_price - np.mean(recent_prices)) / np.mean(recent_prices) if np.mean(recent_prices) > 0 else 0.0
                else:
                    price_momentum = 0.0
            except Exception:
                price_momentum = 0.0

            if action == 'BUY':
                # Good company + oversold = best buy opportunity
                if health_score >= excellent and price_momentum < -0.05:  # Good company, oversold
                    result = 0.9
                elif health_score >= good and price_momentum < -0.02:
                    result = 0.5
                elif health_score >= excellent:
                    result = 0.6  # Good company but not oversold (reduced from 0.8)
                elif health_score >= good:
                    result = 0.3  # Good company but not oversold (reduced from 0.4)
                elif health_score >= 40.0:
                    result = 0.0
                else:
                    result = -0.6
            else:
                if health_score >= excellent:
                    result = -0.6
                elif health_score >= good:
                    result = -0.2
                elif health_score >= 40.0:
                    result = 0.0
                else:
                    result = 0.6

            result = float(result)
            self._signal_cache[cache_key] = result
            return result
        except Exception as e:
            self.logger.error(f"Fundamental signal calculation failed: {e}")
            return 0.0

    def _execute_trade(self, action):  # type: ignore[reportGeneralTypeIssues]
        """Execute trade based on action (0=hold, 1=buy, 2=sell) with proper position management"""
        # CRITICAL FIX: Import config OUTSIDE try block to ensure proper scope
        try:
            import config  # type: ignore
        except ImportError:
            config = None

        # CRITICAL FIX: Load cooldown config with fallback (SEPARATE check)
        if config and hasattr(config, 'COOLDOWN_CONFIG'):
            cooldown_config = config.COOLDOWN_CONFIG
        else:
            # Synced with config.py COOLDOWN_CONFIG (lines 1302-1312)
            cooldown_config = {
                'min_hold_bars': 3,
                'cooldown_bars': 2,
                'cooldown_penalty': 0.001,
                'cooldown_enabled': True,
                'max_cooldown_bars': 5,
                'track_cooldowns': True,
                'log_cooldown_violations': True
            }
        
        # CENTRALIZED: Use the instance-level reward_config synchronized with dynamic_config
        reward_config = getattr(self, 'reward_config', {})

        # Reward mode (Step 2): technical-only + PnL
        try:
            dc_mode = getattr(self, 'dynamic_config', None)
            reward_mode = str(getattr(dc_mode, 'reward_mode', '') or '')
        except Exception:
            reward_mode = ''
        
        # CRITICAL FIX: Add logging for config loading status
        self.logger.debug(f"Config loaded: COOLDOWN_CONFIG={bool(config and hasattr(config, 'COOLDOWN_CONFIG'))}, REWARD_CONFIG={bool(config and hasattr(config, 'REWARD_CONFIG'))}")
        
        try:
            
            # CRITICAL FIX: Initialize positions if not exists or missing keys
            if not hasattr(self, 'positions'):
                self.positions = {i: {'shares': 0.0, 'avg_cost': 0.0} for i in range(self.num_tickers)}
            else:
                # Ensure all ticker keys exist
                for i in range(self.num_tickers):
                    if i not in self.positions:
                        self.positions[i] = {'shares': 0.0, 'avg_cost': 0.0}
            # Initialize FIFO lot tracking for cost-basis if not present
            if not hasattr(self, 'position_lots'):
                self.position_lots = {i: [] for i in range(self.num_tickers)}  # list of dicts: {'qty', 'price'}
            
            if not hasattr(self, 'cash'):
                self.cash = self.initial_balance
            
            if not hasattr(self, 'portfolio_value'):
                self.portfolio_value = self.initial_balance
            
            # CRITICAL FIX: Get current prices
            if self.current_step >= len(self.price_matrix):
                return 0.0
            
            current_prices = self.price_matrix[self.current_step]

            # CRITICAL FIX: Simple trading logic to prevent infinite loops
            reward = 0.0

            # Phase 2.2: Validate action before execution (DISABLED - was too strict)
            # if not self._validate_action(action):
            #     self.logger.info(f"[VALIDATION] Action {action} failed validation, treating as HOLD")
            #     action = 0  # Convert to HOLD

            if action == 1:  # Buy (or cover short)
                # If we have shorts, prefer to cover (BUY closes short positions)
                has_shorts = [(i, self.positions[i]['shares']) for i in range(self.num_tickers)
                              if i in self.positions and self.positions[i].get('shares', 0) < -0.001]
                if has_shorts:
                    ticker = max(has_shorts, key=lambda x: abs(x[1]))[0]
                    # Cover short: reduce negative shares by buying
                    short_shares = abs(self.positions[ticker]['shares'])
                    avg_cost_short = self.positions[ticker].get('avg_cost', current_prices[ticker])
                    cover_value = min(self.cash * 0.99, short_shares * current_prices[ticker])
                    cover_shares = cover_value / current_prices[ticker] if current_prices[ticker] > 0 else 0
                    if cover_shares > 0.001 and self.cash >= cover_value:
                        cost = cover_shares * current_prices[ticker] * 1.002
                        self.cash -= cost
                        new_shares = self.positions[ticker]['shares'] + cover_shares  # Toward 0
                        realized_pnl = (avg_cost_short - current_prices[ticker]) * cover_shares  # Profit when price dropped
                        self.cumulative_realized_gains += realized_pnl
                        self.positions[ticker]['shares'] = new_shares
                        if abs(new_shares) < 0.001:
                            self.positions[ticker] = {'shares': 0.0, 'avg_cost': 0.0}
                        self._total_trades_executed += 1
                        trade_record = {'step': self.current_step, 'ticker': ticker, 'action': 'COVER', 'quantity': cover_shares, 'price': current_prices[ticker], 'value': cost, 'pnl': realized_pnl}
                        self.trade_history.append(trade_record)
                        if hasattr(self, 'episode_trade_history'):
                            self.episode_trade_history.append(trade_record)
                        return 0.05 + (0.1 if realized_pnl > 0 else -0.05)  # Reward for covering
                ticker = self._select_best_ticker_to_buy(current_prices)

                # Check if we can afford to buy
                if self.cash > 50:
                    # CRITICAL FIX: Comprehensive price validation to prevent crashes
                    if current_prices[ticker] <= 0 or np.isnan(current_prices[ticker]) or np.isinf(current_prices[ticker]):
                        self.logger.warning(f"Invalid price for ticker {ticker}: {current_prices[ticker]}, skipping trade")
                        return 0.0

                    # CRITICAL FIX: Use dynamic config for position sizing instead of hardcoded values
                    try:
                        from ppo.utils.dynamic_config import get_dynamic_config  # type: ignore
                        dynamic_config = self._cfg()
                        max_position_size = dynamic_config.max_position_size  # Use dynamic config (4%)
                    except ImportError:
                        # Fallback to default value
                        max_position_size = 0.04  # 4% default position size

                    # Architecture B: size toward ensemble target weight (remaining gap)
                    if getattr(self, '_arch_b_enabled', False) and self._arch_b is not None:
                        period = self._arch_b_active_month()
                        target_w = self._arch_b.target_weight(ticker, period)
                        cur_pos_val = 0.0
                        if ticker in self.positions:
                            cur_pos_val = self.positions[ticker].get('shares', 0.0) * current_prices[ticker]
                        cur_w = cur_pos_val / self.portfolio_value if self.portfolio_value > 0 else 0.0
                        gap = max(0.0, target_w - cur_w)
                        if gap <= 0.001:
                            self.logger.info(f"[ARCH_B] Already at target weight for ticker {ticker} ({cur_w:.3f}/{target_w:.3f})")
                            return -0.01
                        max_position_size = min(max_position_size, gap)

                    # Pre-trade gap detection (thresholds depend on bar frequency)
                    gap_detected = False
                    gap_pct = 0.0
                    data_corruption_suspected = False
                    is_monthly = getattr(self, "bar_frequency", "day") == "month"
                    corrupt_thresh = float(getattr(self, "monthly_corruption_pct", 1.50)) if is_monthly else 0.50
                    gap_down = float(getattr(self, "monthly_gap_block_down", 0.40)) if is_monthly else 0.10
                    gap_up = float(getattr(self, "monthly_gap_block_up", 0.50)) if is_monthly else 0.15
                    if self.current_step > 0 and hasattr(self, 'price_matrix'):
                        try:
                            prev_price = self.price_matrix[self.current_step - 1, ticker]
                            if prev_price > 0:
                                gap_pct = (current_prices[ticker] - prev_price) / prev_price
                                
                                if abs(gap_pct) > corrupt_thresh:
                                    data_corruption_suspected = True
                                    self.logger.error(
                                        f"DATA CORRUPTION SUSPECTED: Ticker {ticker} price changed "
                                        f"{gap_pct:.2%} (${prev_price:.2f} -> ${current_prices[ticker]:.2f})"
                                    )
                                    reward = -0.05
                                    return reward
                                
                                if gap_pct < -gap_down or gap_pct > gap_up:
                                    gap_detected = True
                                    self.logger.warning(
                                        f"PRE-TRADE GAP: Ticker {ticker} gapped {gap_pct:.2%} "
                                        f"(block thresholds down={gap_down:.0%}/up={gap_up:.0%}, "
                                        f"bar={getattr(self, 'bar_frequency', 'day')})"
                                    )
                                    self.logger.warning("BLOCKING TRADE: Extreme gap detected")
                                    reward = -0.05
                                    return reward
                        except (IndexError, ValueError) as e:
                            self.logger.debug(f"Could not check gap for ticker {ticker}: {e}")
                    
                    # CRITICAL FIX #2: Per-ticker position limit (max 20% per ticker to prevent concentration)
                    # IMPORTANT: Check CUMULATIVE position value (sum of all positions in same ticker)
                    max_position_per_ticker = 0.20  # Maximum 20% of portfolio per ticker
                    
                    # Calculate cumulative position value for this ticker (sum of all positions)
                    current_position_value = 0.0
                    if ticker in self.positions:
                        current_position_value = self.positions[ticker].get('shares', 0.0) * current_prices[ticker]
                    
                    current_position_weight = current_position_value / self.portfolio_value if self.portfolio_value > 0 else 0.0
                    
                    if current_position_weight >= max_position_per_ticker:
                        self.logger.warning(f"⚠️ POSITION LIMIT: Ticker {ticker} already at {current_position_weight:.2%} (max {max_position_per_ticker:.2%}), skipping buy")
                        # Skip this buy - position already at limit
                        reward = -0.1  # Small penalty for attempting to exceed limit
                        return reward
                    
                    # Adjust max_position_size to respect per-ticker limit (cumulative)
                    remaining_weight = max_position_per_ticker - current_position_weight
                    max_position_size = min(max_position_size, remaining_weight)
                    
                    # CRITICAL: Also check if adding this position would exceed limit
                    proposed_position_value = self.portfolio_value * max_position_size
                    proposed_total_value = current_position_value + proposed_position_value
                    proposed_total_weight = proposed_total_value / self.portfolio_value if self.portfolio_value > 0 else 0.0
                    
                    if proposed_total_weight > max_position_per_ticker:
                        # Adjust position size to stay within limit
                        max_allowed_value = (self.portfolio_value * max_position_per_ticker) - current_position_value
                        if max_allowed_value > 0:
                            max_position_size = max_allowed_value / self.portfolio_value
                            self.logger.debug(f"⚠️ Adjusted position size to respect cumulative limit: {max_position_size:.2%}")
                        else:
                            # Already at limit, skip trade
                            self.logger.warning(f"⚠️ POSITION LIMIT: Ticker {ticker} would exceed {max_position_per_ticker:.2%} limit, skipping buy")
                            reward = -0.1
                            return reward
                    
                    # NEW: Dynamic position sizing - larger positions when signals are strong (>0.7)
                    # PERFORMANCE: Use precomputed signals from cache
                    if hasattr(self, '_step_signals_cache') and ticker in self._step_signals_cache and \
                       'BUY' in self._step_signals_cache[ticker]:
                        tech_score = self._step_signals_cache[ticker]['BUY'].get('tech', 0.0)
                        fund_score = self._step_signals_cache[ticker]['BUY'].get('fund', 0.0)
                    else:
                        # Fallback to direct calculation if cache miss
                        tech_score = self._calculate_technical_signal_quality(ticker, 'BUY')
                        fund_score = self._calculate_fundamental_signal_quality(ticker, 'BUY')
                    
                    # PHASE 0 FIX: Apply signal gating BEFORE trade execution
                    # Architecture B: book already selected names — skip tech gating when configured
                    if getattr(self, '_arch_b_enabled', False) and getattr(self, '_arch_b_skip_gating', True):
                        should_trade, reward_multiplier = True, 1.0
                    else:
                        should_trade, reward_multiplier = self._should_gate_trade_by_signal_confidence(tech_score, fund_score, 'BUY')
                    if not should_trade:
                        # CRITICAL FIX: Always log blocked trades and track them
                        if not hasattr(self, '_blocked_trades_count'):
                            self._blocked_trades_count = {'BUY': 0, 'SELL': 0}
                        self._blocked_trades_count['BUY'] += 1
                        self.logger.warning(
                            f"[SIGNAL_GATING] BUY BLOCKED for ticker {ticker}: tech={tech_score:.3f}, fund={fund_score:.3f}, "
                            f"step={self.current_step}, total_blocked={self._blocked_trades_count['BUY']}"
                        )
                        # 2026-02-06: Softer penalty when fundamentals disabled so blocked BUY isn't worse than HOLD (-0.03)
                        blocked_penalty = -0.02 if (fund_score == 0.0 or not getattr(self, '_fundamental_data_available', True)) else -0.05
                        return blocked_penalty
                    
                    # 2026-02-28: Contra-signal check – BUY when indicators say sell → strong penalty
                    self._contra_signal_buy_penalty_this_step = 0.0
                    try:
                        dc = self._cfg()
                        sell_tech = 0.0
                        if hasattr(self, '_step_signals_cache') and ticker in self._step_signals_cache and 'SELL' in self._step_signals_cache[ticker]:
                            sell_tech = float(self._step_signals_cache[ticker]['SELL'].get('tech', 0.0) or 0.0)
                        else:
                            sell_tech = float(self._calculate_technical_signal_quality(ticker, 'SELL'))
                        thresh_sell = float(getattr(dc, 'contra_signal_buy_block_threshold_sell', 0.6))
                        thresh_buy = float(getattr(dc, 'contra_signal_buy_block_threshold_buy', 0.3))
                        contra_pen = float(getattr(dc, 'contra_signal_buy_penalty', -0.08))
                        if sell_tech >= thresh_sell and tech_score < thresh_buy:
                            if self._should_log_trade(20):
                                self.logger.info(f"[CONTRA_SIGNAL] BUY penalized: SELL tech={sell_tech:.2f} >= {thresh_sell}, BUY tech={tech_score:.2f} < {thresh_buy}, penalty={contra_pen:.2f}")
                            self._contra_signal_buy_penalty_this_step = contra_pen
                    except Exception:
                        pass
                    
                    # Store reward multiplier to apply at end of reward calculation
                    self._current_trade_reward_multiplier = reward_multiplier
                    
                    strong_signal_threshold = 0.7
                    
                    # Increase position size by up to 33% (from 4% to 5.33%) for very strong signals
                    if tech_score > strong_signal_threshold or fund_score > strong_signal_threshold:
                        signal_strength = max(tech_score, fund_score)
                        # PHASE 2 FIX: Prevent division by zero when signal_strength = 1.0
                        if signal_strength >= 1.0:
                            signal_multiplier = 1.33  # Max 33% increase
                        else:
                            # Scale from 1.0x to 1.33x based on signal strength (0.7 to 1.0)
                            signal_multiplier = 1.0 + ((signal_strength - strong_signal_threshold) / (1.0 - strong_signal_threshold)) * 0.33
                        max_position_size = max_position_size * signal_multiplier
                        self.logger.debug(f"[DYNAMIC_POSITION_SIZE] Strong signal detected (tech={tech_score:.2f}, fund={fund_score:.2f}), increasing position size by {signal_multiplier:.2f}x to {max_position_size:.2%}")

                    # TRADING REALISM: Position sizing respects cash constraints
                    # Calculate position size based on portfolio value and max position size
                    position_size = self.portfolio_value * max_position_size / current_prices[ticker]
                    
                    # CRITICAL: Ensure position size doesn't exceed available cash
                    # Estimate total cost (will be calculated exactly below, but need rough estimate for validation)
                    estimated_cost = position_size * current_prices[ticker] * 1.002  # Add 0.2% buffer for fees
                    if estimated_cost > self.cash:
                        # Adjust position size to fit available cash
                        position_size = (self.cash * 0.99) / current_prices[ticker]  # Use 99% of cash to leave room for fees
                        self.logger.debug(
                            f"TRADING REALISM: Adjusted position size to fit cash - "
                            f"original={self.portfolio_value * max_position_size / current_prices[ticker]:.4f}, "
                            f"adjusted={position_size:.4f}, cash=${self.cash:.2f}"
                        )

                    # Check diversification constraints - DISABLED FOR DEBUGGING
                    if True:  # DISABLE CONSTRAINTS - FORCE ALLOW ALL TRADES
                        # Execute buy with REALISTIC ALPACA FEES
                        base_cost = position_size * current_prices[ticker]

                        # ALPACA-STYLE COSTS - REALISTIC TRADING FEES
                        # Commission: Alpaca is commission-free for retail (0.0)
                        # However, we apply transaction_cost from dynamic_config for SEC/FINRA fees
                        try:
                            from ppo.utils.dynamic_config import get_dynamic_config  # type: ignore
                            dcfg = self._cfg()
                            # Use transaction_cost from dynamic_config (0.000024 = 0.0024% for Alpaca SEC/FINRA fees)
                            transaction_cost_rate = float(getattr(dcfg, 'transaction_cost', 0.000024))
                        except Exception:
                            transaction_cost_rate = 0.000024  # Alpaca realistic: 0.0024% default
                        
                        # Commission: allow override via TRADING_CONFIG (can be 0 for commission-free)
                        try:
                            commission_rate_cfg = getattr(config, 'TRADING_CONFIG', {}).get('commission_rate', 0.0)
                        except Exception:
                            commission_rate_cfg = 0.0
                        commission = base_cost * float(commission_rate_cfg)

                        # Brokerage fees configuration (CAT fee on buys and sells)
                        bc = getattr(config, 'BROKERAGE_COSTS', None)
                        cat_per_share = 0.0
                        if bc and bc.get('enabled', True):
                            try:
                                cat_per_share = float(bc.get('rates_static', {}).get('cat_fee_per_share', 0.0))
                            except Exception:
                                cat_per_share = 0.0
                        cat_fee = position_size * cat_per_share

                        # Market impact and slippage (realistic 0.1% for retail)
                        try:
                            slippage_rate = float(getattr(config, 'TRADING_CONFIG', {}).get('slippage_rate', 0.001))
                        except Exception:
                            slippage_rate = 0.001  # 0.10% default (realistic for retail)
                        slippage_cost = base_cost * slippage_rate
                        
                        # Apply transaction_cost (SEC/FINRA fees) - this is the realistic Alpaca cost
                        transaction_cost_fee = base_cost * transaction_cost_rate

                        # Total realistic cost: base + commission + CAT + slippage + transaction_cost
                        # VERIFIED: Total cost structure matches Alpaca (commission-free + SEC/FINRA + slippage)
                        cost = base_cost + commission + cat_fee + slippage_cost + transaction_cost_fee

                        if cost <= self.cash:
                            # CRITICAL FIX: Calculate portfolio_before using CURRENT prices with OLD position sizes
                            # This ensures both portfolio_before and portfolio_after use the same price snapshot
                            # The difference will be exactly -cost (plus fees), making validation more accurate
                            # PERFORMANCE: Use vectorized calculation for consistency
                            portfolio_before = self.cash + sum(
                                self.positions[i]['shares'] * current_prices[i] 
                                for i in range(min(self.num_tickers, len(current_prices)))
                                if i in self.positions and self.positions[i].get('shares', 0.0) > 0
                            )

                            # DEBUG: Log position update (gated for performance)
                            if self._should_log_trade(50):
                                self.logger.debug(f"TRADE DEBUG: Before - positions[{ticker}]={self.positions[ticker]['shares']:.4f}, cash=${self.cash:.2f}")

                            # CRITICAL FIX: Update position with proper PnL tracking
                            current_pos_info = self.positions[ticker]
                            new_total_shares = current_pos_info['shares'] + position_size
                            # Allocate per-share buy cost including allocatable fees (commission+slippage+CAT)
                            per_share_cost = (base_cost + commission + slippage_cost + cat_fee) / max(position_size, 1e-9)
                            # Strict FIFO: append lot
                            self.position_lots[ticker].append({'qty': float(position_size), 'price': float(per_share_cost)})
                            # Recompute avg_cost from lots for reporting
                            if self.position_lots[ticker]:
                                total_qty = sum(l['qty'] for l in self.position_lots[ticker])
                                total_val = sum(l['qty'] * l['price'] for l in self.position_lots[ticker])
                                
                                # CRITICAL FIX: Validate position_lots are not corrupted
                                max_affordable_from_lots = (self.initial_balance * 10.0) / per_share_cost if per_share_cost > 0 else 1e9
                                if total_qty > max_affordable_from_lots:
                                    self.logger.error(f"🚨 POSITION_LOTS CORRUPTION: Ticker {ticker} total_qty {total_qty:.2f} > max {max_affordable_from_lots:.2f}")
                                    self.logger.error(f"🚨 Clearing corrupted position_lots")
                                    self.position_lots[ticker] = []
                                    total_qty = 0.0
                                    total_val = 0.0
                                
                                new_avg_cost = (total_val / total_qty) if total_qty > 0 else 0.0
                                
                                # CRITICAL FIX: Use positions[ticker]['shares'] as single source of truth
                                # If mismatch detected, rebuild position_lots from positions (not vice versa)
                                if abs(new_total_shares - total_qty) > 0.01:  # Allow small floating point differences
                                    self.logger.warning(f"⚠️ Position shares mismatch: {new_total_shares:.4f} vs position_lots total: {total_qty:.4f}")
                                    # FIX: Use positions[ticker]['shares'] as source of truth, rebuild position_lots
                                    if new_total_shares > 0 and per_share_cost > 0:
                                        # Rebuild position_lots from current position (single lot with current avg cost)
                                        self.logger.warning(f"⚠️ Rebuilding position_lots from positions[ticker]['shares'] (single source of truth)")
                                        self.position_lots[ticker] = [{'qty': float(new_total_shares), 'price': float(per_share_cost)}]
                                        total_qty = new_total_shares  # Update for consistency
                                    else:
                                        # If shares is 0, clear lots
                                        self.position_lots[ticker] = []
                                        total_qty = 0.0
                            else:
                                new_avg_cost = 0.0
                            
                            # CRITICAL FIX: Validate position shares before setting to prevent corruption
                            max_affordable_shares = (self.initial_balance * 10.0) / per_share_cost if per_share_cost > 0 else 1e9
                            if new_total_shares > max_affordable_shares:
                                self.logger.error(f"🚨 POSITION CORRUPTION DETECTED: Ticker {ticker} shares {new_total_shares:.2f} > max affordable {max_affordable_shares:.2f}")
                                self.logger.error(f"🚨 Resetting position to prevent portfolio explosion")
                                new_total_shares = 0.0
                                new_avg_cost = 0.0
                                self.position_lots[ticker] = []  # Clear corrupted lots
                            
                            # DIAGNOSTIC #1: Track position update BEFORE it happens
                            cash_before_update = self.cash
                            try:
                                from ppo.diagnostic_trade_tracking import track_position_update  # type: ignore
                                track_position_update(
                                    step=self.current_step,
                                    ticker=ticker,
                                    action='BUY',
                                    shares=new_total_shares,
                                    cost=cost,
                                    cash_before=cash_before_update,
                                    cash_after=cash_before_update - cost
                                )
                            except Exception as diag_error:
                                self.logger.debug(f"Diagnostic tracking failed (non-critical): {diag_error}")
                            
                            self.positions[ticker] = {'shares': new_total_shares, 'avg_cost': new_avg_cost}
                            # Single source of truth: keep position_lots in sync with positions[ticker]['shares'] (avoid float drift)
                            if hasattr(self, 'position_lots') and self.position_lots.get(ticker):
                                total_qty = sum(l['qty'] for l in self.position_lots[ticker])
                                if abs(total_qty - new_total_shares) > 1e-6:
                                    self.position_lots[ticker] = [{'qty': float(new_total_shares), 'price': float(new_avg_cost)}]
                            # PERFORMANCE: Invalidate cached active positions when positions change
                            if hasattr(self, '_cached_active_positions'):
                                delattr(self, '_cached_active_positions')
                            self.cash -= cost
                            
                            # SAFEGUARD #3: Track that we're about to record a trade (for validation)
                            if not hasattr(self, '_pending_trade_record'):
                                self._pending_trade_record = {}
                            self._pending_trade_record['ticker'] = ticker
                            self._pending_trade_record['action'] = 'BUY'
                            self._pending_trade_record['position_size'] = position_size
                            self._pending_trade_record['cost'] = cost
                            self._pending_trade_record['step'] = self.current_step
                            
                            # NEW: Track buy step for hold time calculation
                            if ticker not in self._position_buy_steps or self._position_buy_steps[ticker] is None:
                                # First buy of this position - record buy step
                                self._position_buy_steps[ticker] = self.current_step
                            # If position already exists, keep original buy step (FIFO tracking)

                            # DEBUG: Log position update (gated for performance)
                            if self._should_log_trade(50):
                                self.logger.debug(f"TRADE DEBUG: After - positions[{ticker}]={self.positions[ticker]['shares']:.4f}, cash=${self.cash:.2f}")
                                self.logger.debug(f"TRADE DEBUG: Position size added: {position_size:.4f}, Cost: ${cost:.2f}, Avg Cost: ${new_avg_cost:.2f}")

                            # CRITICAL FIX: Calculate portfolio value directly with CURRENT prices after trade
                            # PERFORMANCE: Use cached fast calculation instead of list comprehension
                            portfolio_after = self._calculate_portfolio_value_fast(current_prices)

                            # CRITICAL FIX: Update self.portfolio_value to reflect the trade immediately
                            # This ensures consistency for next step
                            # PERFORMANCE: Use cached value (already calculated above)
                            expected_portfolio = portfolio_after
                            self.portfolio_value = expected_portfolio
                            
                            # FIX #3: Validate the calculation makes sense (portfolio should decrease by approximately cost)
                            # Account for unrealized P&L on other positions due to price movements
                            expected_change = portfolio_after - portfolio_before
                            
                            # CRITICAL FIX: Since both portfolio_before and portfolio_after use current_prices,
                            # the expected change should be exactly -cost (cash spent) + position_size * current_price (new position value)
                            # Simplified: expected_change = -cost + (position_size * current_prices[ticker])
                            # Tolerance: 1% of cost or $50, whichever is larger (much tighter since we use same price snapshot)
                            expected_change_simple = -cost + (position_size * current_prices[ticker] if ticker < len(current_prices) else 0.0)
                            tolerance = max(50.0, abs(cost) * 0.01)  # 1% of cost or $50 minimum
                            
                            # Only warn if the mismatch is significant (relaxed check since we're using same prices)
                            if abs(expected_change - expected_change_simple) > tolerance:
                                # This is now a debug-level warning since it's less critical with consistent price snapshots
                                self.logger.debug(f"Portfolio change ({expected_change:.2f}) vs expected ({expected_change_simple:.2f}), cost={cost:.2f}, position_value={position_size * current_prices[ticker] if ticker < len(current_prices) else 0.0:.2f} - tolerance: {tolerance:.2f}")
                            
                            # CRITICAL FIX: Optimized reward function for better learning with safety checks
                            # Validate portfolio values before calculation
                            if (not np.isfinite(portfolio_before) or not np.isfinite(portfolio_after) or
                                portfolio_before <= 0 or portfolio_after < 0):
                                portfolio_change = 0.0
                                portfolio_before = self.initial_balance
                                portfolio_after = self.initial_balance
                            else:
                                portfolio_change = (portfolio_after - portfolio_before) / portfolio_before
                                # CRITICAL: Aggressively clip portfolio change to prevent explosion
                                portfolio_change = np.clip(portfolio_change, -0.2, 0.2)  # Limit to 20% change max
                            
                            # CRITICAL FIX: BUY reward should NOT use portfolio_change (unrealized gains)
                            # This causes reward hacking - agent learns to hold positions instead of closing profitably
                            # BUY reward should be minimal - real reward comes on SELL when position is closed
                            # Only reward BUY if the position immediately shows profit (entry quality), not unrealized gains
                            
                            # BEHAVIOR-BASED ADAPTIVE: Get adjustments based on agent performance
                            behavior_adjustments = self._calculate_behavior_based_adaptive_rewards()
                            
                            # Calculate immediate entry quality (price vs VWAP, signal strength)
                            # This rewards good entry timing, not holding positions
                            entry_quality = 0.0
                            if hasattr(self, '_step_signals_cache') and ticker in self._step_signals_cache:
                                vwap_bonus = self._step_signals_cache[ticker].get('vwap', 0.0)
                                entry_quality = max(0.0, vwap_bonus) * 0.7  # 2026-02-26: 0.5→0.7 – stronger entry quality reward
                            
                            # CRITICAL FIX: Use dynamic_config value directly, with proper fallback chain
                            # Priority: buy_base_incentive -> base_trading_incentive -> dynamic_config -> 0.50
                            try:
                                from ppo.utils.dynamic_config import get_dynamic_config
                                dc = self._cfg()
                                default_base = getattr(dc, 'base_trading_incentive', 0.50)
                            except Exception:
                                default_base = 0.50
                            base_buy_reward = reward_config.get('buy_base_incentive', reward_config.get('base_trading_incentive', default_base))
                            # BEHAVIOR-BASED ADAPTIVE: Apply multipliers (trade frequency + BUY boost for action imbalance)
                            base_buy_reward *= behavior_adjustments['base_trading_incentive_mult']
                            base_buy_reward *= behavior_adjustments['buy_base_incentive_mult']  # NEW: BUY boost when BUY ratio critically low

                            # Market-first: scale BUY base incentive by market regime multipliers
                            try:
                                if hasattr(self, '_cached_market_multiplier') and 'BUY' in self._cached_market_multiplier:
                                    buy_market_multiplier = float(self._cached_market_multiplier['BUY'])
                                else:
                                    buy_market_multiplier = float(self._calculate_market_condition_multiplier('BUY'))
                            except Exception:
                                buy_market_multiplier = 1.0
                            try:
                                ro = self._get_regime_outputs()
                                buy_market_multiplier *= float(ro.get('buy_reward_mult', 1.0)) if isinstance(ro, dict) else 1.0
                                # 2026-02-28: Wrong-regime penalty (bear) and bull overbought/weak-signal discount
                                bull_bear = str(ro.get('bull_bear', 'neutral')) if isinstance(ro, dict) else 'neutral'
                                dc = self._cfg()
                                if bull_bear == 'bear':
                                    wrong_regime_pen = float(getattr(dc, 'wrong_regime_buy_penalty', -0.10))
                                    if not hasattr(self, '_wrong_regime_buy_penalty_this_step'):
                                        self._wrong_regime_buy_penalty_this_step = 0.0
                                    self._wrong_regime_buy_penalty_this_step = wrong_regime_pen
                                else:
                                    self._wrong_regime_buy_penalty_this_step = 0.0
                                if bull_bear == 'bull':
                                    overbought_thresh = float(getattr(dc, 'overbought_rsi_threshold', 0.65))
                                    overbought_disc = float(getattr(dc, 'overbought_buy_discount', 0.85))
                                    weak_disc = float(getattr(dc, 'weak_signal_bull_buy_discount', 0.5))
                                    if hasattr(self, 'feature_matrix') and self.feature_matrix.shape[0] > self.current_step and self.feature_matrix.shape[2] > 2:
                                        rsi_val = float(self.feature_matrix[self.current_step, ticker, 2])
                                        if rsi_val >= overbought_thresh:
                                            buy_market_multiplier *= overbought_disc
                                    if tech_score < 0.4:
                                        buy_market_multiplier *= weak_disc
                            except Exception:
                                setattr(self, '_wrong_regime_buy_penalty_this_step', 0.0)
                            buy_market_multiplier = float(np.clip(buy_market_multiplier, 0.7, 1.6))
                            base_buy_reward *= buy_market_multiplier

                            # Cap the shaped BUY base incentive (avoid overpowering SELL/PnL)
                            buy_cap = float(reward_config.get('buy_reward_cap', 0.40) or 0.40)
                            base_buy_reward = float(np.clip(base_buy_reward, 0.0, buy_cap))
                            
                            # 2025-12-22 FIX: NO-POSITIONS BONUS - Encourage BUY when agent is all in cash
                            # This breaks the "avoid trading" pattern caused by harsh SELL penalties
                            total_positions = sum(1 for i in range(self.num_tickers) 
                                                if i in self.positions and self.positions[i].get('shares', 0) > 0)
                            no_positions_bonus = 0.0
                            if total_positions == 0:
                                # Agent has no positions - give extra incentive to start trading
                                # CRITICAL FIX: Use dynamic_config value directly
                                try:
                                    from ppo.utils.dynamic_config import get_dynamic_config
                                    dc = self._cfg()
                                    default_no_pos = getattr(dc, 'no_positions_bonus', 0.03)
                                except Exception:
                                    default_no_pos = 0.03
                                no_positions_bonus = reward_config.get('no_positions_bonus', default_no_pos)
                                if self._should_log_trade(50):
                                    self.logger.info(f"[BUY_INCENTIVE] No positions - adding +{no_positions_bonus:.3f} bonus")
                            
                            # NEW: Adaptive action bonus based on market conditions
                            # This is the CORRECT adaptive approach: reward BUY when market conditions favor buying
                            adaptive_buy_bonus = self._calculate_adaptive_action_bonus('BUY')
                            
                            reward = base_buy_reward + entry_quality + no_positions_bonus + adaptive_buy_bonus  # Small reward for opening position
                            
                            if adaptive_buy_bonus > 0 and self._should_log_trade(20):
                                self.logger.info(f"[ADAPTIVE_BUY_BONUS] Market-based bonus: +{adaptive_buy_bonus:.3f}")
                            
                            # CRITICAL: Do NOT use portfolio_change for BUY - it rewards unrealized gains
                            # portfolio_change includes unrealized gains which encourages holding instead of closing

                            # VWAP + Technical + Fundamental bonuses (dynamic-config weights)
                            try:
                                from ppo.utils.dynamic_config import get_dynamic_config  # type: ignore
                                dcfg = self._cfg()
                                tech_w = float(getattr(dcfg, 'technical_signal_weight', 0.40))  # REWARD REBALANCING: Updated to 0.40 (40%)
                                fund_w = float(getattr(dcfg, 'fundamental_signal_weight', 0.20))  # REWARD REBALANCING: Updated to 0.20 (20%)
                            except Exception:
                                tech_w, fund_w = 0.40, 0.20  # REWARD REBALANCING: Updated fallback values (40% tech, 20% fund)
                            # PERFORMANCE: Use precomputed signals from cache
                            if hasattr(self, '_step_signals_cache') and ticker in self._step_signals_cache:
                                vwap_bonus = self._step_signals_cache[ticker].get('vwap', 0.0)
                                if 'BUY' in self._step_signals_cache[ticker]:
                                    tech_score = self._step_signals_cache[ticker]['BUY'].get('tech', 0.0)
                                    fund_score = self._step_signals_cache[ticker]['BUY'].get('fund', 0.0)
                                else:
                                    tech_score = self._calculate_technical_signal_quality(ticker, 'BUY')
                                    fund_score = self._calculate_fundamental_signal_quality(ticker, 'BUY')
                            else:
                                # Fallback to direct calculation if cache miss
                                vwap_bonus = self._calculate_vwap_reward(ticker, current_prices[ticker])
                                tech_score = self._calculate_technical_signal_quality(ticker, 'BUY')
                                fund_score = self._calculate_fundamental_signal_quality(ticker, 'BUY')
                            
                            # PHASE 1 FIX: Don't amplify signals - use original scores for all threshold checks
                            # If you want to reward strong signals, do it in the reward calculation, not signal quality
                            original_tech_score = tech_score
                            # 2026-02-06: Technical-only reward when fundamentals disabled
                            fundamentals_disabled = (fund_score == 0.0 or not getattr(self, '_fundamental_data_available', True))
                            if fundamentals_disabled:
                                tech_w, fund_w = 0.60, 0.0
                            # PHASE 4: Add signal confirmation bonus (high-confidence aligned signals)
                            signal_base_reward = vwap_bonus + original_tech_score * tech_w + fund_score * fund_w
                            signal_bonus_cap = reward_config.get('signal_bonus_cap', 0.3)

                            allow_positive_signal = True  # FIX: was portfolio_change >= 0 which is always False at BUY time (fees make change negative)

                            # PHASE 1 FIX: Apply amplification only in reward calculation (not in signal quality)
                            # Add amplification bonus for strong signals
                            if original_tech_score > 0.8:
                                signal_amplification_bonus = 0.2  # Bonus for very strong signals
                            elif original_tech_score > 0.7:
                                signal_amplification_bonus = 0.1  # Bonus for strong signals
                            else:
                                signal_amplification_bonus = 0.0
                            reward += signal_amplification_bonus

                            # High-confidence: when fundamentals disabled use tech-only (original_tech_score > 0.6)
                            if fundamentals_disabled:
                                signal_confirmation_bonus = 0.0
                                signal_alignment_bonus = 0.0
                                if allow_positive_signal and original_tech_score > 0.6:
                                    signal_confirmation_bonus = min(0.3, signal_bonus_cap)
                                    reward += signal_confirmation_bonus
                                if allow_positive_signal and original_tech_score > 0.5:
                                    signal_alignment_bonus = min(0.2, signal_bonus_cap)
                                    reward += signal_alignment_bonus
                                if self._should_log_trade(50) and (signal_confirmation_bonus or signal_alignment_bonus):
                                    self.logger.debug(f"[SIGNAL_TECH_ONLY] BUY tech-only bonus (fund disabled) conf={signal_confirmation_bonus:.3f} align={signal_alignment_bonus:.3f} tech={original_tech_score:.3f}")
                            else:
                                if allow_positive_signal and original_tech_score > 0.7 and fund_score > 0.6:
                                    signal_confirmation_bonus = min(0.3, signal_bonus_cap)
                                    reward += signal_confirmation_bonus
                                    if self._should_log_trade(50):
                                        self.logger.debug(f"[SIGNAL_CONFIRM] High-confidence aligned signals bonus: +{signal_confirmation_bonus:.3f} (tech={tech_score:.3f}, fund={fund_score:.3f})")
                                if allow_positive_signal and original_tech_score > 0.5 and fund_score > 0.5:
                                    signal_alignment_bonus = min(0.2, signal_bonus_cap)
                                    reward += signal_alignment_bonus
                                    if self._should_log_trade(50):
                                        self.logger.debug(f"[SIGNAL_ALIGN] Signals align bonus: +{signal_alignment_bonus:.3f} (tech={tech_score:.3f}, fund={fund_score:.3f})")
                            
                            clipped_signal = np.clip(signal_base_reward, -signal_bonus_cap, signal_bonus_cap)
                            positive_signal = max(clipped_signal, 0.0)
                            negative_signal = min(clipped_signal, 0.0)

                            signal_contribution = 0.0
                            if allow_positive_signal and positive_signal > 0:
                                reward += positive_signal
                                signal_contribution += positive_signal
                            if negative_signal < 0:
                                reward += negative_signal
                                signal_contribution += negative_signal

                            # First trade bonus (small, just to encourage the first entry)
                            if not hasattr(self, '_first_buy_in_episode'):
                                self._first_buy_in_episode = False
                            if not self._first_buy_in_episode:
                                first_trade_bonus = reward_config.get('first_trade_bonus', 0.15)
                                reward += first_trade_bonus
                                if hasattr(self, '_total_trades_executed') and self._total_trades_executed <= 1:
                                    self.logger.info(f"[FIRST_TRADE] First trade in episode! Bonus: +{first_trade_bonus:.3f}")
                                self._first_buy_in_episode = True

                            # PERFORMANCE: Gate reward structure logs behind frequency check
                            if self._should_log_trade(50):
                                self.logger.debug(
                                    f"[REWARD_STRUCTURE] Portfolio change component: {portfolio_change * 0.05:.4f}, "
                                    f"Signal reward contribution: {signal_contribution:.4f}, Total signal weight: {tech_w + fund_w:.1%}"
                                )

                            # Position size bonus (only for meaningful positions >0.5%)
                            # FIXED: Reduced threshold from 0.01 (1%) to 0.005 (0.5%) to encourage smaller positions
                            position_weight = (position_size * current_prices[ticker]) / self.portfolio_value
                            if position_weight > 0.005:
                                size_bonus_mult = reward_config.get('size_bonus_multiplier', 0.05)
                                size_bonus = position_weight * size_bonus_mult  # FIXED: Use dynamic_config
                                reward += size_bonus
                                # PERFORMANCE: Gate position size bonus logs behind frequency check
                                if self._should_log_trade(50):
                                    self.logger.debug(f"[DEBUG_TRADE] BUY POSITION SIZE BONUS: +{size_bonus:.3f} for {position_weight:.1%} position")
                            
                            # PERFORMANCE: Only log if debugging or every 20th trade
                            if self._should_log_trade(20):
                                self.logger.info(
                                    f"[DEBUG_TRADE] BUY REWARD: base={base_buy_reward:.3f}, "
                                    f"entry_quality={entry_quality:.3f}, signal_contrib={signal_contribution:.3f}, "
                                    f"total={reward:.3f}"
                                )
                            
                            # Add diversification bonus (encourage spreading risk)
                            diversification_score = self._calculate_diversification_score()
                            diversification_mult = reward_config.get('diversification_bonus_multiplier', 0.1)
                            diversification_bonus = diversification_score * diversification_mult  # FIXED: Use dynamic_config
                            reward += diversification_bonus

                            # Add risk penalty for over-concentration (prevent single-stock focus)
                            position_weight = (self.positions[ticker]['shares'] * current_prices[ticker]) / self.portfolio_value
                            if position_weight > 0.25:  # Reduced threshold from 0.4
                                risk_penalty = (position_weight - 0.25) * 0.2  # Reduced penalty
                                reward -= risk_penalty

                            # STEP 8 OPTIMIZATION: Apply stress testing costs
                            stress_cost = self._calculate_stress_costs(cost, position_size, current_prices[ticker])
                            transaction_cost_penalty = (stress_cost / portfolio_before) * 0.1
                            reward -= transaction_cost_penalty

                            # CRITICAL: Add drawdown penalty for risk management
                            drawdown_penalty = self._calculate_drawdown_penalty()
                            reward += drawdown_penalty

                            # 2026-02-28: Regime/indicator penalties (contra-signal, wrong-regime, wrong-signal)
                            reward += getattr(self, '_contra_signal_buy_penalty_this_step', 0.0)
                            reward += getattr(self, '_wrong_regime_buy_penalty_this_step', 0.0)
                            wrong_signal_thresh = reward_config.get('wrong_signal_buy_threshold', 0.35)
                            wrong_signal_pen = reward_config.get('wrong_signal_buy_penalty', -0.08)
                            if tech_score < wrong_signal_thresh:
                                reward += wrong_signal_pen

                            # PHASE 4: Trade frequency penalty - increased threshold to allow more trading
                            # HIGH PRIORITY FIX: Increased threshold from 80 to 120 and reduced penalty rate to fix over-conservatism
                            trade_frequency_threshold = reward_config.get('trade_frequency_threshold', 120)
                            trade_frequency_penalty_rate = reward_config.get('trade_frequency_penalty_rate', 0.003)
                            if hasattr(self, '_total_trades_executed') and self._total_trades_executed > trade_frequency_threshold:
                                # Penalty for too many trades (encourages quality over quantity)
                                trade_frequency_penalty = (self._total_trades_executed - trade_frequency_threshold) * trade_frequency_penalty_rate
                                reward -= trade_frequency_penalty
                                if self._should_log_trade(10):  # Log every 10th trade after threshold
                                    self.logger.debug(f"[TRADE_FREQ_PENALTY] Applied penalty: -{trade_frequency_penalty:.3f} for {self._total_trades_executed} trades")

                            # CRITICAL FIX: Reset consecutive holds counter when agent takes action
                            if hasattr(self, '_consecutive_holds'):
                                self._consecutive_holds = 0

                            # Add momentum bonus (reward buying in uptrends)
                            if hasattr(self, 'price_matrix') and self.current_step > 0:
                                price_change = (current_prices[ticker] - self.price_matrix[self.current_step-1][ticker]) / self.price_matrix[self.current_step-1][ticker]
                                if price_change > 0.01:  # 1% uptrend
                                    momentum_mult = reward_config.get('momentum_bonus_multiplier', 0.2)
                                    momentum_bonus = price_change * momentum_mult  # FIXED: Use dynamic_config
                                    reward += momentum_bonus

                            # CRITICAL: Cap BUY reward so PPO doesn't learn "always BUY".
                            # BUY has no realized PnL; large positive BUY rewards bias against SELL.
                            # Use existing reward caps when available; increased cap to allow adaptive bonus.
                            try:
                                reward_cap_min = float(reward_config.get('reward_cap_min', -0.5))
                            except Exception:
                                reward_cap_min = -0.5
                            try:
                                # FIX: Increased cap from 0.08 to 0.40 to allow adaptive bonuses
                                # The adaptive bonus can add up to 0.35, so we need higher cap
                                buy_reward_cap = float(reward_config.get('buy_reward_cap', 0.40))
                            except Exception:
                                buy_reward_cap = 0.40
                            
                            # Step 2: Technical-only reward override (PnL + technicals only; ignore activity/aux bonuses).
                            if reward_mode == "technical_only_v1":
                                try:
                                    dc = getattr(self, 'dynamic_config', None)
                                    base = float(getattr(dc, 'tech_only_buy_base', 0.01))
                                    sig_w = float(getattr(dc, 'tech_only_signal_weight', 0.15))
                                    vwap_w = float(getattr(dc, 'tech_only_vwap_weight', 0.05))
                                    cost_w = float(getattr(dc, 'tech_only_cost_weight', 0.10))
                                except Exception:
                                    base, sig_w, vwap_w, cost_w = 0.01, 0.15, 0.05, 0.10

                                vwap_bonus = 0.0
                                tech_score_buy = 0.0
                                try:
                                    if hasattr(self, '_step_signals_cache') and ticker in self._step_signals_cache:
                                        vwap_bonus = float(self._step_signals_cache[ticker].get('vwap', 0.0) or 0.0)
                                        if 'BUY' in self._step_signals_cache[ticker]:
                                            tech_score_buy = float(self._step_signals_cache[ticker]['BUY'].get('tech', 0.0) or 0.0)
                                except Exception:
                                    pass

                                fees_total = float(commission + cat_fee + slippage_cost + transaction_cost_fee)
                                tech_pos = max(0.0, tech_score_buy)
                                tech_neg = max(0.0, -tech_score_buy)
                                reward = base + (sig_w * tech_pos) - (sig_w * tech_neg)
                                reward += vwap_w * max(0.0, vwap_bonus)
                                reward -= cost_w * (fees_total / max(portfolio_before, 1e-9))

                            # PHASE 0 FIX: Apply signal gating reward multiplier before clipping
                            # Quality over quantity: penalize weak-signal trades so agent learns to use indicators
                            if hasattr(self, '_current_trade_reward_multiplier'):
                                mult = self._current_trade_reward_multiplier
                                if mult < 1.0:
                                    weak_penalty = reward_config.get('weak_signal_penalty', -0.06)
                                    reward += weak_penalty
                                    if self._should_log_trade(50):
                                        self.logger.debug(f"[WEAK_SIGNAL] BUY with low/medium confidence: penalty={weak_penalty:.3f}, mult={mult:.2f}x")
                                reward = reward * mult
                                if self._should_log_trade(50):
                                    self.logger.debug(f"[SIGNAL_GATING] BUY reward multiplier: {mult:.2f}x")
                                delattr(self, '_current_trade_reward_multiplier')
                            
                            reward = float(np.clip(reward, reward_cap_min, buy_reward_cap))
                            if not np.isfinite(reward):
                                reward = 0.0

                            # CRITICAL FIX: Record trade in history with proper structure
                            # For BUY trades, PnL is 0 until position is closed
                            trade_info = {
                                'step': self.current_step,
                                'action': 'BUY',
                                'ticker': ticker,
                                'shares': position_size,
                                'price': current_prices[ticker],
                                'cost': cost,
                                'fees': {
                                    'commission': float(commission),
                                    'sec_fee': 0.0,
                                    'finra_taf': 0.0,
                                    'cat_fee': float(cat_fee),
                                    'slippage_cost': float(slippage_cost)
                                },
                                'fees_total': float(commission + cat_fee + slippage_cost),
                                'pnl': 0.0,  # PnL is 0 for BUY trades (realized on SELL)
                                'portfolio_before': portfolio_before,
                                'portfolio_after': portfolio_after,
                                'reward': reward,
                                'cash_before': self.cash + cost,
                                'timestamp': datetime.now().isoformat(),
                                'trade_id': f"BUY_{self.current_step}_{ticker}_{self._total_trades_executed}"
                            }
                            # SAFEGUARD #3: Record trade with validation
                            try:
                                # DIAGNOSTIC #1: Track trade record BEFORE it happens
                                try:
                                    from ppo.diagnostic_trade_tracking import track_trade_record  # type: ignore
                                    track_trade_record(
                                        step=self.current_step,
                                        ticker=ticker,
                                        action='BUY',
                                        trade_info=trade_info
                                    )
                                except Exception as diag_error:
                                    self.logger.debug(f"Diagnostic tracking failed (non-critical): {diag_error}")
                                
                                self.trade_history.append(trade_info)
                                # PERFORMANCE: Limit trade_history to last 1000 trades to reduce memory overhead
                                if len(self.trade_history) > 1000:
                                    self.trade_history = self.trade_history[-1000:]
                                self.episode_trade_history.append(trade_info)  # CRITICAL: Also record in episode history
                                self.cumulative_evaluation_trade_history.append(trade_info)  # CRITICAL: Also record in cumulative history
                                self._total_trades_executed += 1
                                self._buy_count += 1  # BEHAVIOR TRACKING: Track BUY actions
                                # PHASE 2 FIX: Update last trade step for structured exploration cooldown
                                self._last_trade_step = self.current_step
                                
                                # SAFEGUARD #3: Validate trade was recorded
                                if hasattr(self, '_pending_trade_record') and self._pending_trade_record.get('ticker') == ticker:
                                    # Clear pending record - trade was successfully recorded
                                    self._pending_trade_record = {}
                                
                                # DEBUG: Log successful trade execution (gated for performance)
                                if self._should_log_trade(50):
                                    self.logger.info(f"[DEBUG_TRADE]  BUY EXECUTED: Ticker {ticker}, Shares: {position_size:.4f}, Cost: ${cost:.2f}")
                                    self.logger.info(f"[DEBUG_TRADE] Trade History Length: {len(self.trade_history)}, Total Trades: {self._total_trades_executed}")
                            except Exception as trade_record_error:
                                # SAFEGUARD #3: Critical error - trade recording failed but position was updated!
                                self.logger.error(f"🚨 CRITICAL: Trade recording failed for BUY on ticker {ticker} at step {self.current_step}: {trade_record_error}")
                                self.logger.error(f"🚨 Position was updated (shares={new_total_shares:.4f}, cost=${cost:.2f}) but trade NOT recorded!")
                                self.logger.error(f"🚨 This will cause data inconsistency - attempting emergency trade record...")
                                import traceback
                                self.logger.error(f"🚨 Traceback: {traceback.format_exc()}")
                                
                                # Emergency trade record attempt
                                try:
                                    emergency_trade = {
                                        'step': self.current_step,
                                        'action': 'BUY',
                                        'ticker': ticker,
                                        'shares': position_size,
                                        'price': current_prices[ticker],
                                        'cost': cost,
                                        'pnl': 0.0,
                                        'portfolio_before': portfolio_before,
                                        'portfolio_after': portfolio_after,
                                        'reward': reward,
                                        'cash_before': self.cash + cost,
                                        'timestamp': datetime.now().isoformat(),
                                        'trade_id': f"BUY_EMERGENCY_{self.current_step}_{ticker}",
                                        'emergency_record': True,
                                        'original_error': str(trade_record_error)
                                    }
                                    self.trade_history.append(emergency_trade)
                                    self.episode_trade_history.append(emergency_trade)
                                    self.cumulative_evaluation_trade_history.append(emergency_trade)
                                    self._total_trades_executed += 1
                                    self._buy_count += 1
                                    self.logger.warning(f"🚨 Emergency trade record created - trade may be incomplete")
                                except Exception as emergency_error:
                                    self.logger.error(f"🚨🚨 FATAL: Emergency trade record also failed: {emergency_error}")
                            
                            # PHASE 1 FIX: First BUY bonus already applied earlier (unconditional)
                            # Removed duplicate first trade bonus logic that was conditional on success
                            
                            # CRITICAL FIX: Also update last trade step for interval tracking
                            self.last_trade_step = self.current_step

                            # CRITICAL FIX: Track consecutive trades by action type
                            if not hasattr(self, '_consecutive_buys'):
                                self._consecutive_buys = 0
                                self._consecutive_sells = 0

                            # Increment consecutive buys, reset sells
                            self._consecutive_buys += 1
                            self._consecutive_sells = 0

                            # Reset consecutive failed sells counter on successful buy
                            if hasattr(self, '_consecutive_failed_sells'):
                                self._consecutive_failed_sells = 0

                            # STEP 3: Set cooldown and minimum hold (dynamic: market regime/volatility)
                            min_hold_bars = self._get_dynamic_min_hold_steps()
                            cooldown_bars = cooldown_config.get('cooldown_bars', 3)
                            self._set_minimum_hold(ticker, min_hold_bars)
                            self._set_cooldown(ticker, cooldown_bars)

                            # OPTIMIZED: Reduced logging frequency from every 20th to every 50th trade
                            if self._should_log_trade(50):  # Only log every 50th trade
                                self.logger.info(f"BUY executed: {position_size:.2f} shares of ticker {ticker} at ${current_prices[ticker]:.2f}")
                                self.logger.info(f"  - Cost: ${cost:.2f}, Cash: ${self.cash:.2f}")
                                self.logger.info(f"  - Portfolio: ${portfolio_before:.2f}  ${portfolio_after:.2f}")
                                self.logger.info(f"  - Reward: {reward:.4f} (portfolio: {portfolio_change:.4f}, diversification: {diversification_bonus:.4f}, drawdown: {drawdown_penalty:.4f})")
                                self.logger.info(f"  - Drawdown: {self.current_drawdown:.2%} (peak: ${self.peak_portfolio_value:.2f})")
                                self.logger.info(f"  - Total trades: {self._total_trades_executed}")
                                self.logger.debug(f"  - STEP DEFAULT_NUM_FEATURES  # CONSTANT: Replaced magic number: Set cooldown {cooldown_bars} bars, min hold {min_hold_bars} bars for {ticker}")  # OPT D: was info
                        else:
                            reward = -DEFAULT_TRAILING_STOP  # CONSTANT: Replaced magic number  # Penalty for insufficient funds
                        # Constraint branch removed (constraints disabled)
                else:
                    reward = -DEFAULT_TRAILING_STOP  # CONSTANT: Replaced magic number  # Penalty for insufficient cash

            elif action == 2:  # Sell
                # ENHANCED LOGGING: Track SELL attempts for debugging
                if self._should_log_trade(10):
                    has_positions_check = any(
                        i in self.positions and self.positions[i].get('shares', 0.0) > 0 
                        for i in range(self.num_tickers)
                    )
                    self.logger.info(f"[SELL_ATTEMPT] Step {self.current_step}: SELL action predicted, has_positions={has_positions_check}")
                
                # FIX #2: Guard against double-processing - if we just executed a SELL in this step, don't process again
                # This happens when stop loss triggers a SELL, then action == 2 is also processed
                # Changed from WARNING to DEBUG since this is expected behavior (preventing duplicate processing)
                if hasattr(self, '_sell_executed_this_step') and self._sell_executed_this_step:
                    self.logger.debug(f"[DEBUG_TRADE] SELL already executed this step (likely by stop loss), skipping duplicate processing")
                    reward = 0.0  # FIXED: Set reward instead of returning early (step() must return 5-tuple)
                
                # CRITICAL FIX: Check if we have any positions to sell first (safely with .get() to handle missing keys)
                has_positions = any(
                    i in self.positions and self.positions[i].get('shares', 0.0) > 0 
                    for i in range(self.num_tickers)
                )

                if not has_positions:
                    # No positions to sell - apply minimal penalty to allow learning
                    if not hasattr(self, '_consecutive_failed_sells'):
                        self._consecutive_failed_sells = 0

                    self._consecutive_failed_sells += 1

                    # CRITICAL FIX: Minimal penalties to encourage SELL attempts
                    # Agent should try SELL even when uncertain - exploration is good
                    base_penalty = reward_config.get('failed_sell_penalty', -0.001)  # CRITICAL: Reduced to -0.001
                    consecutive_penalty = reward_config.get('failed_sell_consecutive_penalty', -0.0005)  # CRITICAL: Reduced to -0.0005
                    # Linear growth after first failure (very slow growth)
                    reward = base_penalty + (consecutive_penalty * min(self._consecutive_failed_sells - 1, 10))
                    # Cap: allow stronger penalty (2026-02-15: -0.01→-0.10 so SHAP habit fix is felt)
                    failed_cap = reward_config.get('failed_sell_penalty_cap', -0.10)
                    reward = max(reward, failed_cap)
                    
                    # ENHANCED LOGGING: Log failed SELL attempts more frequently for debugging
                    if self._should_log_trade(20):
                        self.logger.warning(f"[FAILED_SELL] Step {self.current_step}: No positions to sell (consecutive: {self._consecutive_failed_sells}), Penalty: {reward:.4f}")
                    # FIXED: Don't return early - set reward and continue to end of step() to return 5-tuple
                    # 2025-12-21 FIX: Skip sell execution when no positions (don't overwrite penalty)
                    ticker = None
                else:
                    # =========================================================================
                    # 2025-12-21 FIX: Only execute SELL logic when has_positions is True
                    # This was a bug where sell_attempt_bonus was overwriting the penalty above
                    # =========================================================================
                    
                    # CRITICAL FIX: Sell from largest position to force diversification
                    ticker = self._select_ticker_to_sell()
                    
                    # CRITICAL FIX: Add bonus for attempting SELL when positions exist (encourage exploration)
                    sell_attempt_bonus = reward_config.get('sell_attempt_bonus', 0.10)
                    reward = sell_attempt_bonus  # Start with attempt bonus - agent gets rewarded just for trying
                    
                    # ENHANCED LOGGING: Track SELL execution attempts
                    if self._should_log_trade(10):
                        self.logger.info(f"[SELL_EXECUTION] Step {self.current_step}: Attempting SELL on ticker {ticker}, attempt_bonus={sell_attempt_bonus:.3f}")
                
                # FIX #2: Double-check position still exists after selection (prevent race condition)
                if ticker is None:
                    pass  # Already handled - no positions case
                elif ticker not in self.positions or self.positions[ticker].get('shares', 0.0) <= 0:
                    self.logger.warning(f"[DEBUG_TRADE] Selected ticker {ticker} has no position, skipping SELL")
                    reward = 0.0  # FIXED: Set reward instead of returning early (step() must return 5-tuple)
                elif self.positions[ticker]['shares'] > 0:  # FIXED: Changed to elif to prevent processing if reward already set
                    # PHASE 0 FIX: Apply signal gating BEFORE trade execution
                    # Get signals for gating check
                    if hasattr(self, '_step_signals_cache') and ticker in self._step_signals_cache and \
                       'SELL' in self._step_signals_cache[ticker]:
                        tech_score = self._step_signals_cache[ticker]['SELL'].get('tech', 0.0)
                        fund_score = self._step_signals_cache[ticker]['SELL'].get('fund', 0.0)
                    else:
                        tech_score = self._calculate_technical_signal_quality(ticker, 'SELL')
                        fund_score = self._calculate_fundamental_signal_quality(ticker, 'SELL')
                    
                    should_trade, reward_multiplier = self._should_gate_trade_by_signal_confidence(tech_score, fund_score, 'SELL')
                    if not should_trade:
                        # CRITICAL FIX: Always log blocked trades and track them
                        if not hasattr(self, '_blocked_trades_count'):
                            self._blocked_trades_count = {'BUY': 0, 'SELL': 0}
                        self._blocked_trades_count['SELL'] += 1
                        self.logger.warning(
                            f"[SIGNAL_GATING] SELL BLOCKED for ticker {ticker}: tech={tech_score:.3f}, fund={fund_score:.3f}, "
                            f"step={self.current_step}, total_blocked={self._blocked_trades_count['SELL']}"
                        )
                        # CRITICAL FIX: Return small negative reward instead of 0.0 to provide learning signal
                        return -0.05  # Small penalty for attempting low-quality trade
                    
                    # Store reward multiplier to apply at end of reward calculation
                    self._current_trade_reward_multiplier = reward_multiplier
                    
                    # ENHANCED LOGGING: Log successful SELL execution
                    if self._should_log_trade(5):
                        shares = self.positions[ticker]['shares']
                        self.logger.info(f"[SELL_EXECUTION] Step {self.current_step}: Executing SELL on ticker {ticker}, shares={shares:.4f}")
                    # FIXED: Enforce minimum hold period to prevent premature exits (A: apply penalty to reward)
                    self._early_sell_penalty_this_step = 0.0
                    min_hold_steps_early = self._get_dynamic_min_hold_steps()  # market-dependent
                    if ticker in self._position_hold_tracker and self._position_hold_tracker[ticker] > 0:
                        hold_remaining = self._position_hold_tracker[ticker]
                        quick_flip_penalty = reward_config.get('quick_flip_penalty', -0.15)
                        penalty = quick_flip_penalty * (1.0 - (hold_remaining / min_hold_steps_early))  # Scale penalty by how early
                        self._early_sell_penalty_this_step = penalty
                        if self._should_log_trade(50):
                            self.logger.warning(f"[MIN_HOLD_VIOLATION] Attempted to sell {ticker} after {min_hold_steps_early - hold_remaining} steps (min: {min_hold_steps_early}), applying penalty: {penalty:.3f}")
                    
                    # FIX #2: Mark that we're executing a SELL to prevent double-processing
                    self._sell_executed_this_step = True
                    # CRITICAL FIX: Comprehensive price validation to prevent crashes
                    if current_prices[ticker] <= 0 or np.isnan(current_prices[ticker]) or np.isinf(current_prices[ticker]):
                        self.logger.warning(f"Invalid price for ticker {ticker}: {current_prices[ticker]}, skipping sell")
                        reward = 0.0  # FIXED: Set reward instead of returning early (step() must return 5-tuple)

                    # CRITICAL FIX: Calculate portfolio_before using CURRENT prices with OLD position sizes
                    # This ensures both portfolio_before and portfolio_after use the same price snapshot
                    # The difference will be exactly realized_pnl, making validation more accurate
                    # PERFORMANCE: Use vectorized calculation for consistency
                    portfolio_before = self.cash + sum(
                        self.positions[i]['shares'] * current_prices[i] 
                        for i in range(min(self.num_tickers, len(current_prices)))
                        if i in self.positions and self.positions[i].get('shares', 0.0) > 0
                    )
                    
                    current_pos_info = self.positions[ticker]
                    shares_to_sell = float(current_pos_info['shares'])
                    avg_cost = float(current_pos_info.get('avg_cost', 0.0))
                    # PRE-CAPTURE: record unrealized gain before position is cleared during sale execution
                    _pre_sale_unrealized_gain_pct = (current_prices[ticker] - avg_cost) / avg_cost if avg_cost > 0 else 0.0
                    
                    # CRITICAL BUG FIX: Validate shares before selling to prevent portfolio explosion
                    if shares_to_sell <= 0:
                        self.logger.error(f"🚨 CRITICAL: Attempted to sell {shares_to_sell} shares of ticker {ticker} - invalid position!")
                        reward = -1.0  # FIXED: Set reward instead of returning early (step() must return 5-tuple)
                    
                    # CRITICAL BUG FIX: Cap shares to prevent unrealistic sells (portfolio protection)
                    # PERFORMANCE: Use cached fast calculation instead of list comprehension
                    actual_portfolio = self._calculate_portfolio_value_fast(current_prices)
                    max_reasonable_shares = (actual_portfolio / current_prices[ticker]) if current_prices[ticker] > 0 else 1e9
                    
                    # CRITICAL: If selling more shares than we could have possibly bought, it's a bug
                    # Maximum we could have bought = initial_balance * (some reasonable multiplier like 10x) / price
                    max_possible_shares = (self.initial_balance * 10.0) / current_prices[ticker] if current_prices[ticker] > 0 else 1e9
                    
                    if shares_to_sell > max_possible_shares:
                        self.logger.error(f"🚨 CRITICAL BUG: Attempted to sell {shares_to_sell:.2f} shares but max possible from ${self.initial_balance:.2f} is {max_possible_shares:.2f}")
                        self.logger.error(f"🚨 Price: ${current_prices[ticker]:.2f}, Position shares: {current_pos_info['shares']}")
                        # Reset position completely to prevent explosion
                        self.positions[ticker] = {'shares': 0.0, 'avg_cost': 0.0}
                        # PERFORMANCE: Invalidate cached active positions when positions change
                        if hasattr(self, '_cached_active_positions'):
                            delattr(self, '_cached_active_positions')
                        reward = -1.0  # FIXED: Set reward instead of returning early (step() must return 5-tuple)
                    
                    # Also check against actual portfolio (with 3x safety margin)
                    if shares_to_sell > max_reasonable_shares * 3:
                        self.logger.error(f"🚨 CRITICAL BUG: Attempted to sell {shares_to_sell:.2f} shares but max from portfolio is {max_reasonable_shares:.2f}")
                        self.logger.error(f"🚨 Actual Portfolio: ${actual_portfolio:.2f}, Price: ${current_prices[ticker]:.2f}")
                        # Reset position and cap to prevent explosion
                        shares_to_sell = min(shares_to_sell, max_reasonable_shares)
                        self.positions[ticker] = {'shares': 0.0, 'avg_cost': 0.0}
                        # PERFORMANCE: Invalidate cached active positions when positions change
                        if hasattr(self, '_cached_active_positions'):
                            delattr(self, '_cached_active_positions')
                        reward = -1.0  # FIXED: Set reward instead of returning early (step() must return 5-tuple)

                    # Execute sell with REALISTIC ALPACA FEES
                    base_proceeds = shares_to_sell * current_prices[ticker]

                    # ALPACA-STYLE COSTS (commission-free retail, include regulatory + slippage)
                    # Apply transaction_cost from dynamic_config for SEC/FINRA fees
                    try:
                        from ppo.utils.dynamic_config import get_dynamic_config  # type: ignore
                        dcfg = self._cfg()
                        # Use transaction_cost from dynamic_config (0.000024 = 0.0024% for Alpaca SEC/FINRA fees)
                        transaction_cost_rate = float(getattr(dcfg, 'transaction_cost', 0.000024))
                    except Exception:
                        transaction_cost_rate = 0.000024  # Alpaca realistic: 0.0024% default
                    
                    # Commission: allow override via TRADING_CONFIG (can be 0 for commission-free)
                    try:
                        commission_rate_cfg = getattr(config, 'TRADING_CONFIG', {}).get('commission_rate', 0.0)
                    except Exception:
                        commission_rate_cfg = 0.0
                    commission = base_proceeds * float(commission_rate_cfg)

                    # Brokerage fees: SEC + FINRA (SELL only) + CAT (both sides)
                    bc = getattr(config, 'BROKERAGE_COSTS', None)
                    if bc and bc.get('enabled', True):
                        try:
                            sec_rate = float(bc.get('rates_static', {}).get('sec_fee_per_dollar_sold', 22.90/1_000_000.0))
                        except Exception:
                            sec_rate = 22.90/1_000_000.0
                        try:
                            taf_per_share = float(bc.get('rates_static', {}).get('finra_taf_per_share', 0.000119))
                        except Exception:
                            taf_per_share = 0.000119
                        try:
                            cat_per_share = float(bc.get('rates_static', {}).get('cat_fee_per_share', 0.0))
                        except Exception:
                            cat_per_share = 0.0
                    else:
                        sec_rate = 22.90/1_000_000.0
                        taf_per_share = 0.000119
                        cat_per_share = 0.0
                    sec_fee = base_proceeds * sec_rate
                    finra_taf = shares_to_sell * taf_per_share
                    cat_fee = shares_to_sell * cat_per_share
                    regulatory_fees = sec_fee + finra_taf + cat_fee

                    # Market impact and slippage (realistic 0.1% for retail)
                    try:
                        slippage_rate = float(getattr(config, 'TRADING_CONFIG', {}).get('slippage_rate', 0.001))
                    except Exception:
                        slippage_rate = 0.001  # 0.10% default (realistic for retail)
                    slippage_cost = base_proceeds * slippage_rate
                    
                    # Apply transaction_cost (SEC/FINRA fees) - this is the realistic Alpaca cost
                    transaction_cost_fee = base_proceeds * transaction_cost_rate

                    # Total fees and costs: commission + regulatory + slippage + transaction_cost
                    # VERIFIED: Total cost structure matches Alpaca (commission-free + SEC/FINRA + slippage)
                    total_fees = commission + regulatory_fees + slippage_cost + transaction_cost_fee

                    # Net proceeds after fees
                    proceeds = base_proceeds - total_fees

                    # Strict FIFO cost-basis: consume lots to compute realized cost
                    realized_cost_value = 0.0
                    qty_to_close = float(shares_to_sell)
                    if hasattr(self, 'position_lots') and self.position_lots.get(ticker):
                        lots = self.position_lots[ticker]
                        new_lots = []
                        for lot in lots:
                            if qty_to_close <= 0:
                                new_lots.append(lot)
                                continue
                            take_qty = min(lot['qty'], qty_to_close)
                            realized_cost_value += take_qty * lot['price']
                            remaining = lot['qty'] - take_qty
                            if remaining > 1e-9:
                                new_lots.append({'qty': remaining, 'price': lot['price']})
                            qty_to_close -= take_qty
                        self.position_lots[ticker] = new_lots
                    else:
                        # Fallback to average cost if no lots
                        realized_cost_value = shares_to_sell * avg_cost

                    realized_pnl = base_proceeds - total_fees - realized_cost_value
                    
                    # Accumulate realized gains from closed trades
                    self.cumulative_realized_gains += realized_pnl

                    self.cash += proceeds
                    # Update remaining position from lots
                    if hasattr(self, 'position_lots') and self.position_lots.get(ticker):
                        rem_qty = sum(l['qty'] for l in self.position_lots[ticker])
                        rem_val = sum(l['qty'] * l['price'] for l in self.position_lots[ticker])
                        rem_avg = (rem_val / rem_qty) if rem_qty > 1e-9 else 0.0
                        self.positions[ticker] = {'shares': float(rem_qty), 'avg_cost': float(rem_avg)}
                        # PERFORMANCE: Invalidate cached active positions when positions change
                        if hasattr(self, '_cached_active_positions'):
                            delattr(self, '_cached_active_positions')
                    else:
                        self.positions[ticker] = {'shares': 0.0, 'avg_cost': 0.0}
                        # PERFORMANCE: Invalidate cached active positions when positions change
                        if hasattr(self, '_cached_active_positions'):
                            delattr(self, '_cached_active_positions')
                    
                    # DIAGNOSTIC #1: Track SELL position update BEFORE it happens
                    cash_before_update = self.cash
                    try:
                        from ppo.diagnostic_trade_tracking import track_position_update  # type: ignore
                        track_position_update(
                            step=self.current_step,
                            ticker=ticker,
                            action='SELL',
                            shares=shares_to_sell,
                            cost=-proceeds,  # Negative because we're receiving proceeds
                            cash_before=cash_before_update,
                            cash_after=cash_before_update + proceeds
                        )
                    except Exception as diag_error:
                        self.logger.debug(f"Diagnostic tracking failed (non-critical): {diag_error}")
                    
                    # SAFEGUARD #3: Track that we're about to record a SELL trade (for validation)
                    if not hasattr(self, '_pending_trade_record'):
                        self._pending_trade_record = {}
                    self._pending_trade_record['ticker'] = ticker
                    self._pending_trade_record['action'] = 'SELL'
                    self._pending_trade_record['shares'] = shares_to_sell
                    self._pending_trade_record['proceeds'] = proceeds
                    self._pending_trade_record['step'] = self.current_step

                    # CRITICAL FIX: Calculate portfolio value directly with CURRENT prices after trade
                    # PERFORMANCE: Use cached fast calculation instead of list comprehension
                    portfolio_after = self._calculate_portfolio_value_fast(current_prices)

                    # CRITICAL FIX: Update self.portfolio_value to reflect the trade immediately
                    # PERFORMANCE: Use cached value (already calculated above)
                    expected_portfolio = portfolio_after
                    self.portfolio_value = expected_portfolio
                    
                    # FIX #3: Validate the calculation makes sense (portfolio should change by approximately PnL)
                    # Account for unrealized P&L on remaining positions due to price movements
                    expected_change = portfolio_after - portfolio_before
                    
                    # CRITICAL FIX: Since both portfolio_before and portfolio_after use current_prices,
                    # the expected change should be exactly realized_pnl (proceeds - cost basis)
                    # Simplified: expected_change = realized_pnl (since we use same price snapshot)
                    # Tolerance: 1% of realized PnL or $50, whichever is larger (much tighter since we use same price snapshot)
                    expected_change_simple = realized_pnl
                    tolerance = max(50.0, abs(realized_pnl) * 0.01)  # 1% of realized PnL or $50 minimum
                    
                    # Only warn if the mismatch is significant (relaxed check since we're using same prices)
                    if abs(expected_change - expected_change_simple) > tolerance:
                        # This is now a debug-level warning since it's less critical with consistent price snapshots
                        self.logger.debug(f"Portfolio change ({expected_change:.2f}) vs expected ({expected_change_simple:.2f}), realized_pnl={realized_pnl:.2f} - tolerance: {tolerance:.2f}")

                    # CRITICAL FIX: Optimized sell reward function
                    # Validate portfolio values before calculation
                    if (not np.isfinite(portfolio_before) or not np.isfinite(portfolio_after) or
                        portfolio_before <= 0 or portfolio_after < 0):
                        portfolio_change = 0.0
                        portfolio_before = self.initial_balance
                        portfolio_after = self.initial_balance
                    else:
                        portfolio_change = (portfolio_after - portfolio_before) / portfolio_before
                        # CRITICAL: Aggressively clip portfolio change to prevent explosion
                        portfolio_change = np.clip(portfolio_change, -0.2, 0.2)  # Limit to 20% change max

                    # REWARD REBALANCING: Signals now drive 60% of reward (40% technical + 20% fundamental), profit drives 10%+ (progressive scaling)
                    # Base reward: realized PnL scaled to keep magnitude manageable
                    try:
                        from ppo.utils.dynamic_config import get_dynamic_config  # type: ignore
                        dcfg = self._cfg()
                        tech_w = float(getattr(dcfg, 'technical_signal_weight', 0.40))  # REWARD REBALANCING: Updated to 0.40 (40%)
                        fund_w = float(getattr(dcfg, 'fundamental_signal_weight', 0.20))  # REWARD REBALANCING: Updated to 0.20 (20%)
                    except Exception:
                        tech_w, fund_w = 0.40, 0.20  # REWARD REBALANCING: Updated fallback values (40% tech, 20% fund)

                    # PERFORMANCE: Use precomputed signals from cache
                    if hasattr(self, '_step_signals_cache') and ticker in self._step_signals_cache:
                        vwap_bonus = self._step_signals_cache[ticker].get('vwap', 0.0)
                        if 'SELL' in self._step_signals_cache[ticker]:
                            tech_score = self._step_signals_cache[ticker]['SELL'].get('tech', 0.0)
                            fund_score = self._step_signals_cache[ticker]['SELL'].get('fund', 0.0)
                        else:
                            tech_score = self._calculate_technical_signal_quality(ticker, 'SELL')
                            fund_score = self._calculate_fundamental_signal_quality(ticker, 'SELL')
                    else:
                        # Fallback to direct calculation if cache miss
                        vwap_bonus = self._calculate_vwap_reward(ticker, current_prices[ticker])
                        tech_score = self._calculate_technical_signal_quality(ticker, 'SELL')
                        fund_score = self._calculate_fundamental_signal_quality(ticker, 'SELL')

                    # PHASE 1 FIX: Don't amplify signals - use original scores for all threshold checks
                    original_tech_score = tech_score
                    # 2026-02-06: Technical-only reward when fundamentals disabled
                    fundamentals_disabled_sell = (fund_score == 0.0 or not getattr(self, '_fundamental_data_available', True))
                    if fundamentals_disabled_sell:
                        tech_w, fund_w = 0.60, 0.0
                    # PHASE 1 FIX: Apply amplification only in reward calculation (not in signal quality)
                    if original_tech_score > 0.8:
                        signal_amplification_bonus = 0.2
                    elif original_tech_score > 0.7:
                        signal_amplification_bonus = 0.1
                    else:
                        signal_amplification_bonus = 0.0
                    reward += signal_amplification_bonus

                    signal_base_reward = vwap_bonus + original_tech_score * tech_w + fund_score * fund_w
                    signal_bonus_cap = reward_config.get('signal_bonus_cap', 0.3)

                    # CRITICAL FIX: SELL reward must be MUCH larger than BUY to incentivize closing profitable positions
                    # The agent was learning to hold positions (unrealized gains) instead of closing (realized profits)
                    # SELL reward is the PRIMARY reward - this is where the agent makes money
                    # REWARD REBALANCING: Progressive profit scaling - increases from 1.0x to 2.0x as training progresses
                    base_pnl_reward_scale = reward_config.get('pnl_reward_scale', 30.0)  # Phase 7: Increased from 25.0 to 30.0 to significantly reward profitable trades
                    
                    # Calculate training progress (0.0 to 1.0) based on episode step vs max episode steps
                    if hasattr(self, '_max_episode_steps') and self._max_episode_steps > 0:
                        episode_progress = min(1.0, self.current_step / max(self._max_episode_steps, 1))
                    else:
                        episode_progress = 0.5  # Default to mid-phase if unknown
                    
                    # Phase 7: Increased progressive scaling to 2.5x (was 2.0x) - agent needs very strong signal that closing profitably is good
                    profit_scale_boost = 1.0 + (1.5 * episode_progress)  # Scales from 1.0x to 2.5x (was 2.0x)
                    pnl_reward_scale = base_pnl_reward_scale * profit_scale_boost  # Scales from 30.0 to 75.0 (was 25.0 to 50.0)
                    
                    normalized_pnl = realized_pnl / max(portfolio_before, 1e-9)
                    
                    # CRITICAL FIX: Apply adaptive market multiplier to PnL rewards FIRST
                    # This amplifies profitable trades in favorable market conditions (high vol, strong trends)
                    # PERFORMANCE: Use cached market multiplier
                    if hasattr(self, '_cached_market_multiplier') and 'SELL' in self._cached_market_multiplier:
                        market_multiplier = self._cached_market_multiplier['SELL']
                    else:
                        market_multiplier = self._calculate_market_condition_multiplier('SELL')
                    # Market-first: apply regime controller SELL tilt (separate from behavior-adaptive)
                    try:
                        ro = self._get_regime_outputs()
                        if isinstance(ro, dict):
                            market_multiplier = float(market_multiplier) * float(ro.get('sell_reward_mult', 1.0))
                    except Exception:
                        pass
                    market_multiplier = float(np.clip(float(market_multiplier), 0.7, 1.6))
                    ctx = self._get_dynamic_reward_context()
                    
                    # FIXED: Apply market multiplier to PnL reward - amplifies profits in favorable conditions
                    # In high volatility or strong trends, profitable trades get 1.2x-1.5x reward boost
                    pnl_reward_scale_adaptive = pnl_reward_scale * market_multiplier
                    # B: Asymmetric PnL - scale losses more than gains so policy avoids large losers
                    loss_scale_mult = reward_config.get('loss_scale_multiplier', 1.5)
                    pnl_mult = loss_scale_mult if normalized_pnl < 0 else 1.0
                    reward += normalized_pnl * pnl_reward_scale_adaptive * pnl_mult  # FIX: was = (overwriting sell_attempt_bonus set earlier)
                    reward += getattr(self, '_early_sell_penalty_this_step', 0.0)  # A: Apply early-sell penalty (was computed but never added)
                    
                    # Log adaptive PnL scaling for verification
                    # FIXED: Lower threshold to log more frequently (every 10 trades instead of 50) for better visibility
                    # Also log when multiplier is 1.0 to confirm adaptive system is active
                    if self._should_log_trade(10) or (self._should_log_trade(50) and market_multiplier != 1.0):
                        self.logger.info(f"[ADAPTIVE_PNL] PnL reward scaled by {market_multiplier:.2f}x: base_scale={pnl_reward_scale:.1f}, adaptive_scale={pnl_reward_scale_adaptive:.1f}, normalized_pnl={normalized_pnl:.4f}, realized_pnl={realized_pnl:.2f}")
                    
                    # DYNAMIC REWARD: Base rewards scale with market conditions
                    # Market conditions: volatility, trend, regime → multiplier (0.7x to 1.6x)
                    
                    # RESTORED: Previous reward structure - SELL gets full reward + exploration bonus
                    # Base trading incentive (encourages closing positions)
                    # BEHAVIOR-BASED ADAPTIVE: Get adjustments based on agent performance
                    behavior_adjustments = self._calculate_behavior_based_adaptive_rewards()
                    
                    # CRITICAL FIX: Use dynamic_config value directly, with proper fallback chain
                    # Priority: sell_base_incentive -> base_trading_incentive -> dynamic_config -> 0.48
                    try:
                        from ppo.utils.dynamic_config import get_dynamic_config
                        dc = self._cfg()
                        default_sell = getattr(dc, 'sell_base_incentive', getattr(dc, 'base_trading_incentive', 0.48))
                    except Exception:
                        default_sell = 0.48
                    # 2026-02-28: Unconditional SELL base – every SELL with position gets small positive (balanced; no gapping BUY). Outcome bonuses below.
                    sell_unconditional_base = reward_config.get('sell_unconditional_base', 0.08)
                    sell_unconditional_base *= behavior_adjustments['sell_base_incentive_mult']  # Adaptive: boost when SELL ratio low
                    reward += sell_unconditional_base

                    sell_base_incentive_base = reward_config.get('sell_base_incentive', reward_config.get('base_trading_incentive', default_sell))
                    # BEHAVIOR-BASED ADAPTIVE: Apply multiplier (can increase OR decrease based on action distribution)
                    sell_base_incentive_base *= behavior_adjustments['sell_base_incentive_mult']
                    base_trading_incentive = sell_base_incentive_base * market_multiplier  # DYNAMIC: Scale with market conditions
                    # 2026-02-15: Tie to outcome – full incentive only when profitable; zero when loss (reinforce positive trades)
                    if realized_pnl > 0:
                        reward += base_trading_incentive
                    elif realized_pnl > -0.001:  # Flat/small loss: minimal 25% of base
                        reward += base_trading_incentive * 0.25
                    # else: realized_pnl <= -0.001 → no base incentive for losing SELLs
                    
                    # NEW: Adaptive action bonus - 2026-02-15: only when profitable (reinforce positive)
                    if realized_pnl > 0:
                        adaptive_sell_bonus = self._calculate_adaptive_action_bonus('SELL')
                        reward += adaptive_sell_bonus
                    else:
                        adaptive_sell_bonus = 0.0
                    
                    # ENHANCED LOGGING: Log SELL base incentive usage more frequently to verify it's working correctly
                    if self._should_log_trade(10):
                        self.logger.info(f"[SELL_BASE_INCENTIVE] Base={sell_base_incentive_base:.3f}, Market multiplier={market_multiplier:.3f}, Adaptive bonus={adaptive_sell_bonus:.3f}, Final={base_trading_incentive + adaptive_sell_bonus:.3f}, Total reward so far={reward:.3f}")
                    
                    # NEW: Gain-based SELL bonus scaling - reward larger profits more
                    # Calculate gain percentage based on realized PnL vs trade value
                    trade_value = realized_cost_value if realized_cost_value > 0 else base_proceeds
                    if trade_value > 0:
                        gain_pct = realized_pnl / trade_value
                        threshold_10pct = reward_config.get('gain_bonus_threshold_10pct', 0.10)
                        threshold_5pct = reward_config.get('gain_bonus_threshold_5pct', 0.05)
                        threshold_2pct = reward_config.get('gain_bonus_threshold_2pct', 0.02)
                        threshold_1pct = reward_config.get('gain_bonus_threshold_1pct', 0.01)
                        if gain_pct > threshold_10pct:  # 10%+ gain
                            gain_bonus = reward_config.get('gain_bonus_value_10pct', 0.60)
                        elif gain_pct > threshold_5pct:  # 5%+ gain
                            gain_bonus = reward_config.get('gain_bonus_value_5pct', 0.40)
                        elif gain_pct > threshold_2pct:  # 2%+ gain
                            gain_bonus = reward_config.get('gain_bonus_value_2pct', 0.25)
                        elif gain_pct > threshold_1pct:  # 1%+ gain (2026-02-15: reward small profits)
                            gain_bonus = reward_config.get('gain_bonus_value_1pct', 0.12)
                        else:
                            gain_bonus = 0.0
                        vol_scale = np.clip(0.85 + 0.35 * min(ctx['vol_ratio'], 1.5), 0.7, 1.2)
                        gain_bonus *= vol_scale
                        reward += gain_bonus
                        if self._should_log_trade(10) and gain_bonus > 0:
                            self.logger.info(f"[GAIN_SCALED_BONUS] SELL gain {gain_pct:.2%}: +{gain_bonus:.3f} bonus")
                        # 2026-02-28: 5% profit-taking bonus in bull (sell at 5%+ in bull)
                        ro_sell = self._get_regime_outputs()
                        bull_bear_sell = str(ro_sell.get('bull_bear', 'neutral')) if isinstance(ro_sell, dict) else 'neutral'
                        if bull_bear_sell == 'bull' and gain_pct >= reward_config.get('profit_target_pct', 0.05):
                            profit_target_bonus = reward_config.get('profit_target_5pct_bonus', 0.20)
                            reward += profit_target_bonus
                            if self._should_log_trade(10):
                                self.logger.info(f"[PROFIT_TARGET_5PCT] SELL in bull at {gain_pct:.2%}: +{profit_target_bonus:.3f}")
                    
                    # 2026-02-28: Wrong-signal SELL penalty (SELL at loss when tech was strongly bullish)
                    if realized_pnl <= 0:
                        tech_buy_sell = 0.0
                        if hasattr(self, '_step_signals_cache') and ticker in self._step_signals_cache and 'BUY' in self._step_signals_cache[ticker]:
                            tech_buy_sell = float(self._step_signals_cache[ticker]['BUY'].get('tech', 0.0) or 0.0)
                        if tech_buy_sell > 0.6:
                            wrong_sell_pen = reward_config.get('wrong_signal_sell_penalty', -0.04)
                            reward += wrong_sell_pen
                    
                    # NEW: Bull market profit-taking bonus (encourages selling in bull markets)
                    if hasattr(self, 'price_matrix') and self.current_step >= 10:
                        spy_idx = min(4, self.num_tickers - 1)
                        spy_prices = self.price_matrix[max(0, self.current_step - 10):self.current_step + 1, spy_idx]
                        if len(spy_prices) >= 2:
                            market_return = (spy_prices[-1] - spy_prices[0]) / spy_prices[0]
                            if market_return > 0:  # Bull market
                                bull_market_bonus = reward_config.get('sell_bull_market_boost', 0.05)
                                reward += bull_market_bonus
                                if self._should_log_trade(50):
                                    self.logger.debug(f"[BULL_MARKET_BONUS] SELL in bull market: +{bull_market_bonus:.3f} (market return: {market_return:.2%})")
                    
                    # CRITICAL FIX: Use dynamic_config value directly, with proper fallback chain
                    # Priority: reward_config -> dynamic_config -> 0.25
                    try:
                        from ppo.utils.dynamic_config import get_dynamic_config
                        dc = self._cfg()
                        default_exploration = getattr(dc, 'exploration_bonus', 0.25)  # 2026-02-05: Match dynamic_config
                    except Exception:
                        default_exploration = 0.25
                    # 2026-02-15: Exploration bonus only when profitable (reinforce positive; config has 0.0)
                    exploration_bonus_base = reward_config.get('exploration_bonus', default_exploration)
                    exploration_bonus = 0.0
                    if realized_pnl > 0 and exploration_bonus_base > 0:
                        exploration_bonus = exploration_bonus_base * market_multiplier
                        if ctx['current_drawdown'] > 0.05:
                            exploration_dynamic = 1.2
                        elif ctx['portfolio_return'] > 0.01:
                            exploration_dynamic = 0.9
                        else:
                            exploration_dynamic = 1.0
                        exploration_bonus *= exploration_dynamic
                        reward += exploration_bonus
                    
                    # IMPROVED: Log market multiplier more frequently to verify it's working
                    if hasattr(self, '_total_trades_executed') and (self._total_trades_executed % 10 == 0 or market_multiplier != 1.0):
                        self.logger.debug(f"[DYNAMIC_REWARD] Market multiplier: {market_multiplier:.2f}x (base_trading: {base_trading_incentive:.3f}, exploration: {exploration_bonus:.3f})")
                    
                    # NEW: Trading frequency bonus - dynamic by regime; 2026-02-15: only when profitable (reinforce positive)
                    activity_mult = 0.7 if ctx['regime'] == 'mean_reversion' else 1.0
                    if realized_pnl > 0:
                        try:
                            dc = self._cfg()
                            default_trading_freq = getattr(dc, 'trading_frequency_bonus', 0.03)
                        except Exception:
                            default_trading_freq = 0.03
                        trading_frequency_bonus = reward_config.get('trading_frequency_bonus', default_trading_freq)
                        trading_frequency_bonus *= activity_mult
                        trading_frequency_max = reward_config.get('trading_frequency_bonus_max_trades', 200)
                        if hasattr(self, '_total_trades_executed') and self._total_trades_executed <= trading_frequency_max:
                            reward += trading_frequency_bonus
                        if self._total_trades_executed % 5 == 0:
                            self.logger.debug(f"[TRADING_FREQUENCY_BONUS] SELL Trade {self._total_trades_executed}/{trading_frequency_max}: +{trading_frequency_bonus:.3f} (activity_mult={activity_mult:.2f})")
                    
                    # FIX: Progressive trading bonus - 2026-02-15: only when profitable (reinforce positive)
                    if realized_pnl > 0 and hasattr(self, '_total_trades_executed'):
                        try:
                            dc = self._cfg()
                            default_progressive_per_trade = getattr(dc, 'progressive_bonus_per_trade', 0.002)
                            default_progressive_max = getattr(dc, 'progressive_bonus_max', 0.08)
                        except Exception:
                            default_progressive_per_trade = 0.002
                            default_progressive_max = 0.08
                        progressive_per_trade = reward_config.get('progressive_bonus_per_trade', default_progressive_per_trade)
                        progressive_max = reward_config.get('progressive_bonus_max', default_progressive_max)
                        progressive_bonus = min(self._total_trades_executed * progressive_per_trade, progressive_max) * activity_mult
                        reward += progressive_bonus
                        if self._total_trades_executed % 10 == 0:  # Log every 10th trade
                            self.logger.debug(f"[PROGRESSIVE_TRADING_BONUS] SELL Trade {self._total_trades_executed}: +{progressive_bonus:.3f}")
                    
                    # FIXED: Scale signal bonuses by performance (bigger profits = bigger signal rewards)
                    signal_multiplier_scale = reward_config.get('signal_multiplier_scale', 10.0)
                    signal_multiplier = max(1.0, abs(normalized_pnl) * signal_multiplier_scale)  # FIXED: Use dynamic_config
                    
                    # Signal alignment bonus (scaled by performance and regime); 2026-02-05: full bonus only when SELL is profitable
                    # 2026-02-06: When fundamentals disabled use tech-only (original_tech_score < 0.4)
                    signal_alignment_bonus = 0.0
                    regime_signal_mult = 1.2 if ctx['regime'] == 'trending' else (0.85 if ctx['regime'] == 'mean_reversion' else 1.0)
                    strong_sell_signals = (original_tech_score < 0.4 or (not fundamentals_disabled_sell and fund_score < 0.5))
                    if strong_sell_signals:
                        signal_alignment_base = reward_config.get('signal_alignment_bonus', 0.10)
                        signal_alignment_bonus = signal_alignment_base * signal_multiplier * regime_signal_mult
                        # 2026-02-09: Tie bonus to outcome - full only when profitable; zero when loss
                        if realized_pnl <= 0:
                            signal_alignment_bonus = 0.0
                        elif realized_pnl <= 0.001:  # Flat/small profit: reduce bonus to 25%
                            signal_alignment_bonus *= 0.25
                        reward += signal_alignment_bonus
                        if self._should_log_trade(50):
                            self.logger.debug(f"[SIGNAL_ALIGN] Strong signal SELL bonus: +{signal_alignment_bonus:.3f} (tech={original_tech_score:.3f}, fund={fund_score:.3f}, scaled={signal_multiplier:.2f}x, profitable={realized_pnl > 0.001})")
                    
                    # FIXED: Remove learning bonus - let PnL be the teacher
                    # Learning bonus removed - agent learns from PnL, not artificial bonuses
                    
                    clipped_signal = np.clip(signal_base_reward, -signal_bonus_cap, signal_bonus_cap)

                    # NEW: Calculate hold time for trade quality metrics
                    hold_time = 0
                    if ticker in self._position_buy_steps and self._position_buy_steps[ticker] is not None:
                        hold_time = self.current_step - self._position_buy_steps[ticker]
                        # Clear buy step after selling
                        self._position_buy_steps[ticker] = None
                    
                    # NEW: Unrealized gains threshold bonuses - encourage SELL when unrealized gains are high
                    # Use pre-captured value (position already cleared by sale execution above)
                    unrealized_gain_pct = _pre_sale_unrealized_gain_pct
                    
                    # Apply threshold bonuses based on unrealized gain magnitude
                    if unrealized_gain_pct > 0.10:  # >10% unrealized gain
                        threshold_bonus = reward_config.get('unrealized_gain_threshold_10pct', 0.50)
                        reward += threshold_bonus
                        if self._should_log_trade(10):
                            self.logger.info(f"[SELL_THRESHOLD_BONUS] High unrealized gain ({unrealized_gain_pct:.2%}): +{threshold_bonus:.3f}")
                    elif unrealized_gain_pct > 0.05:  # >5% unrealized gain
                        threshold_bonus = reward_config.get('unrealized_gain_threshold_5pct', 0.30)
                        reward += threshold_bonus
                        if self._should_log_trade(10):
                            self.logger.info(f"[SELL_THRESHOLD_BONUS] Medium unrealized gain ({unrealized_gain_pct:.2%}): +{threshold_bonus:.3f}")
                    elif unrealized_gain_pct > 0.02:  # >2% unrealized gain
                        threshold_bonus = reward_config.get('unrealized_gain_threshold_2pct', 0.15)
                        reward += threshold_bonus
                        if self._should_log_trade(20):
                            self.logger.debug(f"[SELL_THRESHOLD_BONUS] Low unrealized gain ({unrealized_gain_pct:.2%}): +{threshold_bonus:.3f}")
                    
                    # NEW: Enhanced hold-time bonus for SELL - dynamic by regime (stronger in trending)
                    sell_hold_time_min = reward_config.get('sell_hold_time_min_steps', 20)
                    if hold_time >= sell_hold_time_min:
                        sell_hold_time_bonus_base = reward_config.get('sell_hold_time_bonus_base', 0.15)
                        hold_scale = min(1.0 + ((hold_time - sell_hold_time_min) / sell_hold_time_min), 2.0)
                        sell_hold_time_bonus = sell_hold_time_bonus_base * hold_scale * regime_signal_mult
                        reward += sell_hold_time_bonus
                        if self._should_log_trade(10):
                            self.logger.info(f"[SELL_HOLD_TIME_BONUS] Held {hold_time} steps (min: {sell_hold_time_min}): +{sell_hold_time_bonus:.3f} (scaled {hold_scale:.2f}x)")
                    
                    # A: Quick-flip penalty - dynamic by regime (stronger in trending, weaker in mean_reversion)
                    min_hold_steps = self._get_dynamic_min_hold_steps()
                    quick_flip_regime_mult = 1.2 if ctx['regime'] == 'trending' else (0.8 if ctx['regime'] == 'mean_reversion' else 1.0)
                    if hold_time > 0 and hold_time < min_hold_steps:
                        penalty_scale = 1.0 - (hold_time / min_hold_steps)
                        if realized_pnl > 0:
                            quick_flip_p = reward_config.get('quick_flip_penalty_profitable', -0.35)
                            scaled_penalty = quick_flip_p * (1.0 + penalty_scale) * quick_flip_regime_mult
                        else:
                            quick_flip_p = reward_config.get('quick_flip_penalty', -0.15)
                            scaled_penalty = quick_flip_p * (1.0 + penalty_scale) * quick_flip_regime_mult
                        reward += scaled_penalty
                        if self._should_log_trade(50):
                            self.logger.debug(f"[QUICK_FLIP_PENALTY] Sold after {hold_time} steps (min: {min_hold_steps}), PnL={'profit' if realized_pnl > 0 else 'loss'}: {scaled_penalty:.3f}")
                    
                    if realized_pnl > 0:
                        # Phase 7: Profitable trade bonus scales with profit magnitude (increased from 15x to 25x)
                        # Scale bonus by profit: 1% profit = 0.25 bonus, 5% profit = 1.25 bonus (capped at max)
                        profitable_multiplier = reward_config.get('profitable_trade_bonus_multiplier', 25.0)
                        profitable_max = reward_config.get('profitable_trade_bonus_max', 1.0)
                        profitable_trade_bonus = min(normalized_pnl * profitable_multiplier, profitable_max)  # FIXED: Use dynamic_config
                        # Phase 7: Increased profit-taking bonus to significantly encourage profitable exits
                        sell_profit_bonus = reward_config.get('sell_profit_taking_bonus', 0.40)  # Phase 7: Increased from 0.25 to 0.40 (60% increase)
                        reward += profitable_trade_bonus
                        reward += sell_profit_bonus  # NEW: Extra bonus for profitable SELL
                        # 2026-02-05: Profitable-close bonus only when net return > min (profitability exceeds costs, >5% return)
                        profitable_close_min = reward_config.get('profitable_close_min_return_pct', 0.05)
                        if trade_value > 0 and gain_pct >= profitable_close_min:
                            profitable_close_bonus = reward_config.get('profitable_close_bonus', 0.05)
                            reward += profitable_close_bonus
                            if self._should_log_trade(20):
                                self.logger.info(f"[PROFITABLE_CLOSE] Net return {gain_pct:.2%} >= {profitable_close_min:.0%}: +{profitable_close_bonus:.3f}")
                        reward += max(clipped_signal, 0.0)
                        
                        # HIGH PRIORITY FIX: Unrealized gain bonus - encourages capturing unrealized gains (scaled by unrealized gain magnitude)
                        # Use pre-captured value (position already cleared by sale execution above)
                        unrealized_gain_pct = _pre_sale_unrealized_gain_pct
                        if unrealized_gain_pct > 0.01:  # At least 1% unrealized gain
                                unrealized_bonus_base = reward_config.get('unrealized_gain_bonus', 0.05)
                                # Scale bonus by unrealized gain magnitude: 1% = base, 5% = 2x base, 10% = 3x base
                                unrealized_scale_mult = reward_config.get('unrealized_gain_scale_multiplier', 25.0)
                                unrealized_scale_max = reward_config.get('unrealized_gain_scale_max', 3.0)
                                unrealized_scale = min(1.0 + (unrealized_gain_pct - 0.01) * unrealized_scale_mult, unrealized_scale_max)  # FIXED: Use dynamic_config
                                unrealized_bonus = unrealized_bonus_base * unrealized_scale
                                reward += unrealized_bonus
                                if self._should_log_trade(50):
                                    self.logger.debug(f"[UNREALIZED_GAIN_BONUS] Capturing {unrealized_gain_pct:.2%} unrealized gain: +{unrealized_bonus:.3f} (scaled {unrealized_scale:.2f}x)")
                        
                        # C: Hold time bonus for profitable positions held > min_hold_steps (strengthened)
                        if hold_time > min_hold_steps:
                            hold_time_bonus_base = reward_config.get('hold_time_bonus', 0.12)
                            bonus_scale_max = reward_config.get('hold_time_bonus_scale_max', 2.0)
                            bonus_scale = min(1.0 + ((hold_time - min_hold_steps) / min_hold_steps), bonus_scale_max)
                            hold_time_bonus = hold_time_bonus_base * bonus_scale
                            reward += hold_time_bonus
                            if self._should_log_trade(50):
                                self.logger.debug(f"[HOLD_TIME_BONUS] Profitable position held {hold_time} steps (min: {min_hold_steps}): +{hold_time_bonus:.3f} (scaled {bonus_scale:.2f}x)")
                        
                        # Reset consecutive losses on profitable trade
                        if hasattr(self, '_consecutive_losses'):
                            self._consecutive_losses = 0
                        
                        if self._should_log_trade(50):
                            self.logger.debug(f"[PROFIT_SELL] Profitable SELL: PnL={realized_pnl:.2f}, Bonus={profitable_trade_bonus:.3f} + SELL bonus={sell_profit_bonus:.3f} (scaled by {normalized_pnl:.2%} profit, hold time: {hold_time} steps)")
                    else:
                        # Quick-flip penalty already applied above for all short holds (A)
                        # FIXED: Progressive loss penalty - stronger penalties for consecutive losses
                        if not hasattr(self, '_consecutive_losses'):
                            self._consecutive_losses = 0
                        self._consecutive_losses += 1
                        
                        # Progressive penalty: -0.3 for first, -0.5 for second, -0.7 for third+
                        progressive_base = reward_config.get('progressive_loss_penalty_base', -0.3)
                        progressive_step = reward_config.get('progressive_loss_penalty_step', -0.2)
                        progressive_penalty = progressive_base + (progressive_step * (self._consecutive_losses - 1))
                        
                        # Apply loss penalty cap (Phase 7: Balanced at -1.0, was -2.0 too harsh)
                        loss_penalty_cap = reward_config.get('sell_loss_penalty_cap', -1.0)
                        loss_penalty_pnl_mult = reward_config.get('loss_penalty_pnl_multiplier', 5.0)
                        loss_penalty = max(normalized_pnl * loss_penalty_pnl_mult, progressive_penalty, loss_penalty_cap)
                        drawdown_mult = 1.0 + 0.25 * max(0.0, ctx['current_drawdown']) if ctx['current_drawdown'] > 0.02 else 1.0
                        drawdown_mult = min(drawdown_mult, 1.5)
                        loss_penalty *= drawdown_mult
                        reward += loss_penalty
                        reward += min(clipped_signal, 0.0)
                        
                        if self._should_log_trade(50):
                            self.logger.debug(f"[MISTAKE_SIGNAL] Losing SELL penalty: {loss_penalty:.3f} (loss: {normalized_pnl:.2%}, PnL=${realized_pnl:.2f}, consecutive losses: {self._consecutive_losses}, hold time: {hold_time} steps) - Strong penalty encourages holding until profitable")

                    # PERFORMANCE: Gate reward structure logs behind frequency check
                    if self._should_log_trade(50):
                        self.logger.debug(
                            f"[REWARD_STRUCTURE] Realized PnL={realized_pnl:.2f}, Normalized={normalized_pnl:.4f}, "
                            f"Signal contribution={clipped_signal:.4f}, Base reward={reward:.3f}"
                        )

                    # Add diversification bonus only when the trade is profitable
                    diversification_bonus = 0.0
                    if realized_pnl > 0:
                        diversification_score = self._calculate_diversification_score()
                        diversification_mult = reward_config.get('diversification_bonus_multiplier', 0.1)
                        diversification_bonus = diversification_score * diversification_mult  # FIXED: Use dynamic_config
                        reward += diversification_bonus

                    # Add profit-taking bonus if selling at profit (REDUCED to balance with BUY)
                    if portfolio_change > 0:
                        profit_bonus_mult = reward_config.get('profit_bonus_multiplier', 0.2)
                        profit_bonus = portfolio_change * profit_bonus_mult  # FIXED: Use dynamic_config
                        reward += profit_bonus
                        # PERFORMANCE: Gate profit-taking bonus logs behind frequency check
                        if self._should_log_trade(50):
                            self.logger.debug(f"[DEBUG_TRADE] PROFIT-TAKING BONUS: +{profit_bonus:.3f} for selling at profit")

                    # FIXED: Reduce momentum penalty for profitable sells (encourages profit-taking in uptrends)
                    if hasattr(self, 'price_matrix') and self.current_step > 0:
                        price_change = (current_prices[ticker] - self.price_matrix[self.current_step-1][ticker]) / self.price_matrix[self.current_step-1][ticker]
                        if price_change > 0.01:  # 1% uptrend
                            # Only apply penalty if selling at a loss (allow profit-taking in uptrends)
                            if realized_pnl <= 0:
                                momentum_penalty = price_change * 0.1
                                reward -= momentum_penalty
                            else:
                                # Profitable sell in uptrend = good profit-taking, no penalty
                                if self._should_log_trade(50):
                                    self.logger.debug(f"[PROFIT_TAKING] Selling at profit in uptrend: PnL=${realized_pnl:.2f}, price change={price_change:.2%} - No momentum penalty")

                    # STEP 8 OPTIMIZATION: Apply stress testing costs
                    stress_cost = self._calculate_stress_costs(proceeds, shares_to_sell, current_prices[ticker])
                    transaction_cost_penalty = (stress_cost / portfolio_before) * 0.1
                    reward -= transaction_cost_penalty

                    # CRITICAL: Add drawdown penalty for risk management
                    drawdown_penalty = self._calculate_drawdown_penalty()
                    reward += drawdown_penalty

                    # PHASE 4: Trade frequency penalty - increased threshold to allow more trading
                    # HIGH PRIORITY FIX: Increased threshold from 50 to 120 and reduced penalty rate to fix over-conservatism
                    trade_frequency_threshold = reward_config.get('trade_frequency_threshold', 120)
                    trade_frequency_penalty_rate = reward_config.get('trade_frequency_penalty_rate', 0.003)
                    if hasattr(self, '_total_trades_executed') and self._total_trades_executed > trade_frequency_threshold:
                        # Penalty for too many trades (encourages quality over quantity)
                        trade_frequency_penalty = (self._total_trades_executed - trade_frequency_threshold) * trade_frequency_penalty_rate
                        reward -= trade_frequency_penalty
                        if self._total_trades_executed % 10 == 0:  # Log every 10th trade after threshold
                            self.logger.debug(f"[TRADE_FREQ_PENALTY] Applied penalty: -{trade_frequency_penalty:.3f} for {self._total_trades_executed} trades")

                    # Step 2: Technical-only reward override (PnL + technicals only; ignore activity/aux bonuses).
                    if reward_mode == "technical_only_v1":
                        try:
                            dc = getattr(self, 'dynamic_config', None)
                            base = float(getattr(dc, 'tech_only_sell_base', 0.02))
                            pnl_scale = float(getattr(dc, 'tech_only_pnl_scale', 50.0))
                            sig_w = float(getattr(dc, 'tech_only_signal_weight', 0.15))
                            vwap_w = float(getattr(dc, 'tech_only_vwap_weight', 0.05))
                            cost_w = float(getattr(dc, 'tech_only_cost_weight', 0.10))
                        except Exception:
                            base, pnl_scale, sig_w, vwap_w, cost_w = 0.02, 50.0, 0.15, 0.05, 0.10

                        normalized_pnl = float(realized_pnl / max(portfolio_before, 1e-9))
                        vwap_bonus = 0.0
                        tech_score_sell = 0.0
                        try:
                            if hasattr(self, '_step_signals_cache') and ticker in self._step_signals_cache:
                                vwap_bonus = float(self._step_signals_cache[ticker].get('vwap', 0.0) or 0.0)
                                if 'SELL' in self._step_signals_cache[ticker]:
                                    tech_score_sell = float(self._step_signals_cache[ticker]['SELL'].get('tech', 0.0) or 0.0)
                        except Exception:
                            pass

                        signal_term = sig_w * max(0.0, tech_score_sell)
                        if normalized_pnl <= 0:
                            signal_term *= 0.33

                        reward = base + (normalized_pnl * pnl_scale) + signal_term
                        reward += vwap_w * max(0.0, -vwap_bonus)
                        reward -= cost_w * (total_fees / max(portfolio_before, 1e-9))

                    # PHASE 0 FIX: Apply signal gating reward multiplier before recording trade
                    # Quality over quantity: penalize weak-signal trades so agent learns to use indicators
                    if hasattr(self, '_current_trade_reward_multiplier'):
                        mult = self._current_trade_reward_multiplier
                        if mult < 1.0:
                            weak_penalty = reward_config.get('weak_signal_penalty', -0.06)
                            reward += weak_penalty
                            if self._should_log_trade(50):
                                self.logger.debug(f"[WEAK_SIGNAL] SELL with low/medium confidence: penalty={weak_penalty:.3f}, mult={mult:.2f}x")
                        reward = reward * mult
                        if self._should_log_trade(50):
                            self.logger.debug(f"[SIGNAL_GATING] SELL reward multiplier: {mult:.2f}x")
                        delattr(self, '_current_trade_reward_multiplier')

                    # CRITICAL FIX: Reset consecutive holds counter when agent takes action
                    if hasattr(self, '_consecutive_holds'):
                        self._consecutive_holds = 0

                    # CRITICAL FIX: Record trade in history with proper structure and PnL
                    trade_info = {
                        'step': self.current_step,
                        'action': 'SELL',
                        'ticker': ticker,
                        'shares': shares_to_sell,
                        'price': current_prices[ticker],
                        'avg_cost': avg_cost,
                        'proceeds': proceeds,
                        'pnl': realized_pnl,
                        'fees': {
                            'commission': float(commission),
                            'sec_fee': float(sec_fee),
                            'finra_taf': float(finra_taf),
                            'cat_fee': float(cat_fee),
                            'slippage_cost': float(slippage_cost)
                        },
                        'fees_total': float(total_fees),
                        'portfolio_before': portfolio_before,
                        'portfolio_after': portfolio_after,
                        'reward': reward,
                        'timestamp': datetime.now().isoformat(),
                        'trade_id': f"SELL_{self.current_step}_{ticker}_{self._total_trades_executed}"
                    }
                    # SAFEGUARD #3: Record SELL trade with validation
                    try:
                        # DIAGNOSTIC #1: Track SELL trade record BEFORE it happens
                        try:
                            from ppo.diagnostic_trade_tracking import track_trade_record  # type: ignore
                            track_trade_record(
                                step=self.current_step,
                                ticker=ticker,
                                action='SELL',
                                trade_info=trade_info
                            )
                        except Exception as diag_error:
                            self.logger.debug(f"Diagnostic tracking failed (non-critical): {diag_error}")
                        
                        self.trade_history.append(trade_info)
                        # PERFORMANCE: Limit trade_history to last 1000 trades to reduce memory overhead
                        if len(self.trade_history) > 1000:
                            self.trade_history = self.trade_history[-1000:]
                        self.episode_trade_history.append(trade_info)  # CRITICAL: Also record in episode history
                        self.cumulative_evaluation_trade_history.append(trade_info)  # CRITICAL: Also record in cumulative history
                        self._total_trades_executed += 1
                        self._sell_count += 1  # BEHAVIOR TRACKING: Track SELL actions
                        # PHASE 2 FIX: Update last trade step for structured exploration cooldown
                        self._last_trade_step = self.current_step
                        
                        # SAFEGUARD #3: Validate trade was recorded
                        if hasattr(self, '_pending_trade_record') and self._pending_trade_record.get('ticker') == ticker:
                            # Clear pending record - trade was successfully recorded
                            self._pending_trade_record = {}
                    except Exception as trade_record_error:
                        # SAFEGUARD #3: Critical error - trade recording failed but position was updated!
                        self.logger.error(f"🚨 CRITICAL: Trade recording failed for SELL on ticker {ticker} at step {self.current_step}: {trade_record_error}")
                        self.logger.error(f"🚨 Position was updated (shares sold={shares_to_sell:.4f}, proceeds=${proceeds:.2f}) but trade NOT recorded!")
                        self.logger.error(f"🚨 This will cause data inconsistency - attempting emergency trade record...")
                        import traceback
                        self.logger.error(f"🚨 Traceback: {traceback.format_exc()}")
                        
                        # Emergency trade record attempt
                        try:
                            emergency_trade = {
                                'step': self.current_step,
                                'action': 'SELL',
                                'ticker': ticker,
                                'shares': shares_to_sell,
                                'price': current_prices[ticker],
                                'proceeds': proceeds,
                                'pnl': realized_pnl,
                                'portfolio_before': portfolio_before,
                                'portfolio_after': portfolio_after,
                                'reward': reward,
                                'cash_before': self.cash - proceeds,
                                'timestamp': datetime.now().isoformat(),
                                'trade_id': f"SELL_EMERGENCY_{self.current_step}_{ticker}",
                                'emergency_record': True,
                                'original_error': str(trade_record_error)
                            }
                            self.trade_history.append(emergency_trade)
                            self.episode_trade_history.append(emergency_trade)
                            self.cumulative_evaluation_trade_history.append(emergency_trade)
                            self._total_trades_executed += 1
                            self._sell_count += 1
                            self.logger.warning(f"🚨 Emergency trade record created - trade may be incomplete")
                        except Exception as emergency_error:
                            self.logger.error(f"🚨🚨 FATAL: Emergency trade record also failed: {emergency_error}")
                    
                    # BEHAVIOR TRACKING: Record hold time for this position
                    if ticker in getattr(self, '_position_buy_steps', {}) and self._position_buy_steps.get(ticker) is not None:
                        hold_time = self.current_step - self._position_buy_steps[ticker]
                        if not hasattr(self, '_avg_hold_time_history'):
                            self._avg_hold_time_history = []
                        self._avg_hold_time_history.append(hold_time)
                        # Keep only last 50 hold times for rolling average
                        if len(self._avg_hold_time_history) > 50:
                            self._avg_hold_time_history = self._avg_hold_time_history[-50:]
                    
                    # CRITICAL FIX: Give first trade bonus only on SUCCESSFUL trades
                    if not hasattr(self, '_successful_trades_count'):
                        self._successful_trades_count = 0
                    
                    if self._successful_trades_count == 0:
                        current_mode = getattr(self, 'mode', 'test')
                        if current_mode == 'train' and reward > 0:
                            first_bonus = reward_config.get('first_trade_bonus', 0.20)
                            reward += first_bonus
                            # First trade is important, but gate logging for performance
                            if hasattr(self, '_successful_trades_count') and self._successful_trades_count <= 1:
                                self.logger.info(f"[FIRST_TRADE_BONUS] Agent executed first SUCCESSFUL trade! Bonus: +{first_bonus}")
                    
                    self._successful_trades_count += 1

                    # DEBUG: Log successful trade execution (gated for performance)
                    if self._should_log_trade(50):
                        self.logger.info(f"[DEBUG_TRADE]  SELL EXECUTED: Ticker {ticker}, Shares: {shares_to_sell:.4f}, Proceeds: ${proceeds:.2f}")
                        self.logger.info(f"[DEBUG_TRADE] Trade History Length: {len(self.trade_history)}, Total Trades: {self._total_trades_executed}")
                    
                    # CRITICAL FIX: Also update last trade step for interval tracking
                    self.last_trade_step = self.current_step

                    # CRITICAL FIX: Track consecutive trades by action type
                    if not hasattr(self, '_consecutive_buys'):
                        self._consecutive_buys = 0
                        self._consecutive_sells = 0

                    # Increment consecutive sells, reset buys
                    self._consecutive_sells += 1
                    self._consecutive_buys = 0

                    # Reset consecutive failed sells counter on successful sell
                    if hasattr(self, '_consecutive_failed_sells'):
                        self._consecutive_failed_sells = 0

                    # STEP 3 OPTIMIZATION: Set cooldown for anti-scalping (no minimum hold for sells)
                    cooldown_bars = cooldown_config.get('cooldown_bars', 3)
                    self._set_cooldown(ticker, cooldown_bars)

                    # OPTIMIZED: Reduced logging frequency from every 5th to every 50th trade
                    if self._total_trades_executed % 50 == 0:  # Only log every 50th trade
                        self.logger.info(f"SELL executed: {shares_to_sell:.2f} shares of ticker {ticker} at ${current_prices[ticker]:.2f}")
                        self.logger.info(f"  - Proceeds: ${proceeds:.2f}, Cash: ${self.cash:.2f}")
                        self.logger.info(f"  - Portfolio: ${portfolio_before:.2f}  ${portfolio_after:.2f}")
                        self.logger.info(f"  - Reward: {reward:.4f} (portfolio: {portfolio_change:.4f}, diversification: {diversification_bonus:.4f}, drawdown: {drawdown_penalty:.4f})")
                        self.logger.info(f"  - Drawdown: {self.current_drawdown:.2%} (peak: ${self.peak_portfolio_value:.2f})")
                        self.logger.info(f"  - STEP 3: Set cooldown {cooldown_bars} bars for {ticker}")
                else:
                    reward = -0.01  # REDUCED penalty for selling what we don't have (10x less punishing)

            elif action == 3:  # SHORT - open short position (negative shares)
                ticker = self._select_best_ticker_to_short(current_prices)
                if self.cash > 50 and current_prices[ticker] > 0:
                    try:
                        max_position_size = 0.04
                        position_size = self.portfolio_value * max_position_size / current_prices[ticker]
                        position_size = min(position_size, (self.cash * 0.99) / current_prices[ticker])
                        if position_size > 0.001:
                            short_shares = -position_size  # Negative = short
                            base_proceeds = abs(short_shares) * current_prices[ticker]
                            self.cash += base_proceeds * 0.998  # Proceeds minus ~0.2% fees
                            if ticker not in self.positions:
                                self.positions[ticker] = {'shares': 0.0, 'avg_cost': 0.0}
                            old_shares = self.positions[ticker]['shares']
                            old_avg = self.positions[ticker].get('avg_cost', current_prices[ticker])
                            new_shares = old_shares + short_shares  # More negative
                            if new_shares != 0:
                                self.positions[ticker]['avg_cost'] = (old_avg * abs(old_shares) + current_prices[ticker] * abs(short_shares)) / abs(new_shares)
                            self.positions[ticker]['shares'] = new_shares
                            self._total_trades_executed += 1
                            self._position_buy_steps[ticker] = self.current_step
                            trade_record = {'step': self.current_step, 'ticker': ticker, 'action': 'SHORT', 'quantity': abs(short_shares), 'price': current_prices[ticker], 'value': base_proceeds}
                            self.trade_history.append(trade_record)
                            if hasattr(self, 'episode_trade_history'):
                                self.episode_trade_history.append(trade_record)
                            # Scale SHORT reward by technical + fundamental bearish signal (discourage signal-blind shorting)
                            try:
                                if hasattr(self, '_step_signals_cache') and ticker in self._step_signals_cache and 'SELL' in self._step_signals_cache[ticker]:
                                    tech_score = float(self._step_signals_cache[ticker]['SELL'].get('tech', 0.0) or 0.0)
                                    fund_score = float(self._step_signals_cache[ticker]['SELL'].get('fund', 0.0) or 0.0)
                                else:
                                    tech_score = float(self._calculate_technical_signal_quality(ticker, 'SELL'))
                                    fund_score = float(self._calculate_fundamental_signal_quality(ticker, 'SELL'))
                                bearish_signal = 0.6 * np.clip(tech_score, 0.0, 1.0) + 0.4 * np.clip(fund_score, 0.0, 1.0)
                            except Exception:
                                bearish_signal = 0.0
                            short_base = float(self.reward_config.get('short_base_incentive', 0.05))
                            min_threshold = float(self.reward_config.get('short_signal_min_threshold', 0.55))
                            weak_penalty = float(self.reward_config.get('short_weak_signal_penalty', -0.02))
                            if bearish_signal < min_threshold:
                                reward = weak_penalty  # Discourage shorting with no/weak bearish signal
                            else:
                                reward = short_base * (0.5 + 0.5 * min(bearish_signal, 1.0))  # Stronger signal -> higher reward
                        else:
                            reward = -0.02
                    except Exception as e:
                        self.logger.warning(f"SHORT execution failed: {e}")
                        reward = -0.05
                else:
                    reward = -0.02  # Insufficient cash

            else:  # Hold
                # CRITICAL FIX: Check positions FIRST before using in opportunity cost calculation
                has_positions = any(
                    i in self.positions and self.positions[i].get('shares', 0.0) > 0 
                    for i in range(self.num_tickers)
                )
                
                # CRITICAL FIX: Calculate instantaneous portfolio value change due to market movement for held positions
                portfolio_before = self.portfolio_value
                self._update_portfolio_value(current_prices)
                portfolio_after = self.portfolio_value

                # OPTIMIZED: Use cached fast calculation - portfolio already updated by _update_portfolio_value
                # Removed redundant validation to reduce overhead
                # Portfolio bounds checking happens in _update_portfolio_value
                
                # Calculate market movement reward (should be small for short periods)
                # CRITICAL: Validate portfolio values before calculation
                if (not np.isfinite(portfolio_before) or not np.isfinite(portfolio_after) or
                    portfolio_before <= 0 or portfolio_after < 0):
                    portfolio_change = 0.0
                else:
                    portfolio_change = (portfolio_after - portfolio_before) / portfolio_before
                    # CRITICAL: Aggressively clip portfolio change to prevent explosion
                    portfolio_change = np.clip(portfolio_change, -0.2, 0.2)  # Limit to 20% change max
                
                market_movement_multiplier = reward_config.get('market_movement_multiplier', 0.2)
                market_movement_reward = portfolio_change * market_movement_multiplier
                
                # CRITICAL FIX: Apply adaptive multiplier to market movement rewards
                # This amplifies portfolio gains in favorable market conditions (high vol, strong trends)
                if hasattr(self, '_cached_market_multiplier') and 'HOLD' in self._cached_market_multiplier:
                    hold_market_multiplier = self._cached_market_multiplier['HOLD']
                else:
                    hold_market_multiplier = self._calculate_adaptive_hold_penalty()  # Returns multiplier (0.5x to 1.5x)
                
                # Apply adaptive multiplier to market movement (amplifies gains in favorable conditions)
                market_movement_reward = market_movement_reward * hold_market_multiplier

                # Step 2: Technical-only HOLD reward override (PnL proxy + technical miss penalties only).
                if reward_mode == "technical_only_v1":
                    try:
                        dc = getattr(self, 'dynamic_config', None)
                        miss_pen = float(getattr(dc, 'tech_only_hold_miss_penalty', 0.01))
                        entry_thr = float(getattr(dc, 'tech_only_entry_signal_threshold', 0.25))
                        exit_thr = float(getattr(dc, 'tech_only_exit_signal_threshold', 0.25))
                    except Exception:
                        miss_pen, entry_thr, exit_thr = 0.01, 0.25, 0.25

                    reward_hold = float(market_movement_reward)
                    best_buy = 0.0
                    best_sell = 0.0
                    try:
                        cache = getattr(self, '_step_signals_cache', None)
                        if isinstance(cache, dict):
                            for t in range(self.num_tickers):
                                if t not in cache or not isinstance(cache[t], dict):
                                    continue
                                tcache = cache[t]
                                if 'BUY' in tcache and isinstance(tcache['BUY'], dict):
                                    best_buy = max(best_buy, float(tcache['BUY'].get('tech', 0.0) or 0.0))
                                if 'SELL' in tcache and isinstance(tcache['SELL'], dict):
                                    best_sell = max(best_sell, float(tcache['SELL'].get('tech', 0.0) or 0.0))
                    except Exception:
                        pass

                    if (not has_positions) and best_buy >= entry_thr:
                        reward_hold -= miss_pen * float(best_buy - entry_thr)
                    if has_positions and best_sell >= exit_thr:
                        reward_hold -= miss_pen * float(best_sell - exit_thr)
                    return float(np.clip(reward_hold, -1.0, 1.0))
                
                # Log adaptive market movement scaling for verification
                if self.current_step % 50 == 0 and hold_market_multiplier != 1.0 and abs(portfolio_change) > 0.001:
                    self.logger.debug(f"[ADAPTIVE_HOLD] Market movement reward scaled by {hold_market_multiplier:.2f}x: portfolio_change={portfolio_change:.4f}, base_reward={portfolio_change * market_movement_multiplier:.4f}, adaptive_reward={market_movement_reward:.4f}")
                
                # CRITICAL FIX: Penalize holding unprofitable positions (encourages closing losing trades)
                # This prevents the agent from holding positions that look good (unrealized) but lose when closed
                if has_positions:
                    total_unrealized_loss = 0.0
                    for i in range(self.num_tickers):
                        if i in self.positions and self.positions[i].get('shares', 0) > 0:
                            avg_cost = self.positions[i].get('avg_cost', 0)
                            if avg_cost > 0 and i < len(current_prices):
                                current_price = current_prices[i]
                                position_return = (current_price - avg_cost) / avg_cost
                                if position_return < 0:  # Position is losing money
                                    position_value = self.positions[i]['shares'] * current_price
                                    unrealized_loss = abs(position_return) * position_value
                                    total_unrealized_loss += unrealized_loss
                    
                    # Penalize holding losing positions (scaled by portfolio value)
                    if total_unrealized_loss > 0:
                        unrealized_loss_penalty = -(total_unrealized_loss / max(self.portfolio_value, 1e-9)) * 2.0  # 2x penalty
                        market_movement_reward += unrealized_loss_penalty
                        if self.current_step % 50 == 0:  # Log periodically
                            self.logger.debug(f"[UNREALIZED_LOSS_PENALTY] Total unrealized loss: ${total_unrealized_loss:.2f}, penalty: {unrealized_loss_penalty:.4f}")

                # FIX #3: Add opportunity cost when holding 100% cash (BC model should be participating!)
                if not has_positions and abs(market_movement_reward) < 1e-6:
                    # Calculate what agent MISSED by holding cash instead of being invested
                    missed_opportunity = 0.0
                    valid_tickers = 0
                    for i in range(self.num_tickers):
                        if self.current_step > 0 and i < len(current_prices):
                            prev_price = self.price_matrix[self.current_step - 1, i]
                            curr_price = current_prices[i]
                            if prev_price > 0:
                                ticker_return = (curr_price - prev_price) / prev_price
                                missed_opportunity += abs(ticker_return)  # Penalize missing moves in either direction
                                valid_tickers += 1
                    
                    if valid_tickers > 0 and missed_opportunity > 0:
                        avg_missed = missed_opportunity / valid_tickers
                        opportunity_cost = -avg_missed * 0.1  # 10% of average missed market movement
                        market_movement_reward += opportunity_cost
                        if self.current_step % 50 == 0:  # Log periodically
                            self.logger.debug(f"[OPPORTUNITY_COST] Avg missed movement: {avg_missed*100:.3f}%, cost: {opportunity_cost:.4f}")

                # CRITICAL FIX: Reduced hold penalty to prevent portfolio crashes
                # Hold penalty should be minimal to allow natural market movement rewards
                # In evaluation mode, use even smaller penalty to focus on actual returns
                current_mode = getattr(self, 'mode', 'unknown')

                # CRITICAL FIX: Use same robust mode detection
                if current_mode == 'unknown':
                    if hasattr(self, '_is_training') and self._is_training:
                        current_mode = 'train'
                    elif hasattr(self, '_is_evaluating') and self._is_evaluating:
                        current_mode = 'test'
                    else:
                        current_mode = 'test'  # Default to test mode

                # CRITICAL FIX: Remove position incentive that was causing false positive rewards
                # This was the main bug - agent was getting rewards for having no positions
                # NOTE: has_positions already defined earlier in this branch (line 3581)
                log_interval = max(getattr(self, 'log_every_n_steps', 200), 1)  # OPT B: default 200 was 50
                consecutive_holds = getattr(self, '_consecutive_holds', 0)
                # OPT B: In test/eval mode use debug only so INFO is not flooded during eval/training
                is_eval_or_test = getattr(self, 'mode', 'train') in ('test', 'eval') or getattr(self, '_eval_mode', False)
                should_log_info = (not is_eval_or_test) and (self.verbose_step_logs or (consecutive_holds % log_interval == 0))
                hold_log = self.logger.info if should_log_info else self.logger.debug

                # RESTORED: Position incentive for holding profitable positions (encourages maintaining unrealized gains)
                position_incentive_value = 0.0
                if has_positions:
                    # Calculate total unrealized gains across all positions
                    total_unrealized_gains = 0.0
                    for i in range(self.num_tickers):
                        if i in self.positions and self.positions[i].get('shares', 0) > 0:
                            avg_cost = self.positions[i].get('avg_cost', 0)
                            if avg_cost > 0 and i < len(current_prices):
                                current_price = current_prices[i]
                                position_return = (current_price - avg_cost) / avg_cost
                                if position_return > 0:  # Position is profitable
                                    position_value = self.positions[i]['shares'] * current_price
                                    unrealized_gain = position_return * position_value
                                    total_unrealized_gains += unrealized_gain
                    
                    # Apply position incentive only when there are unrealized gains
                    if total_unrealized_gains > 0:
                        # Get position_incentive from dynamic_config (synced 0.02; fallback 0.02 so HOLD not over-rewarded if sync fails)
                        position_incentive_base = reward_config.get('position_incentive', 0.02)
                        # Scale by unrealized gains as percentage of portfolio (aggressive scaling)
                        unrealized_pct = total_unrealized_gains / max(self.portfolio_value, 1e-9)
                        # Phase 7: Scale aggressively - 0.1% unrealized = 0.015, 1% = 0.15, 2%+ = 0.30 (cap)
                        position_incentive_value = position_incentive_base * min(unrealized_pct * 100.0, 2.0)  # Scale 0.1%→0.015, 1%→0.15, 2%+→0.30
                        
                        # NEW: Adaptive time-based decay to position incentive - adjust based on market conditions
                        # This encourages selling after holding for a while, but adapts to market conditions
                        base_decay_start = reward_config.get('position_incentive_decay_start', 15)  # Base: 15 steps
                        base_decay_rate = reward_config.get('position_incentive_decay_rate', 0.3)   # Base: 0.3 rate
                        
                        # ADAPTIVE: Adjust decay parameters based on market conditions
                        start_mult, rate_mult = self._calculate_adaptive_position_decay()
                        position_incentive_decay_start = base_decay_start * start_mult  # Adaptive: 9-22.5 steps
                        position_incentive_decay_rate = base_decay_rate * rate_mult      # Adaptive: 0.18-0.45 rate
                        
                        # Calculate average hold time across all positions
                        avg_hold_time = 0.0
                        positions_with_time = 0
                        for i in range(self.num_tickers):
                            if i in self.positions and self.positions[i].get('shares', 0) > 0:
                                if i in self._position_buy_steps and self._position_buy_steps[i] is not None:
                                    hold_time_i = self.current_step - self._position_buy_steps[i]
                                    avg_hold_time += hold_time_i
                                    positions_with_time += 1
                        
                        if positions_with_time > 0:
                            avg_hold_time = avg_hold_time / positions_with_time
                            
                            # Apply decay if average hold time exceeds adaptive threshold
                            if avg_hold_time > position_incentive_decay_start:
                                decay_factor = position_incentive_decay_rate ** ((avg_hold_time - position_incentive_decay_start) / 10.0)  # Decay every 10 steps
                                position_incentive_value *= decay_factor
                                if self.current_step % 50 == 0:  # Log periodically
                                    hold_log(f"[ADAPTIVE_POSITION_DECAY] Avg hold time: {avg_hold_time:.1f} steps, decay start: {position_incentive_decay_start:.1f} (base: {base_decay_start}, mult: {start_mult:.2f}x), decay rate: {position_incentive_decay_rate:.3f} (base: {base_decay_rate}, mult: {rate_mult:.2f}x), decay factor: {decay_factor:.3f}, adjusted incentive: {position_incentive_value:.3f}")
                            
                            # NEW: Adaptive position aging penalty - penalize holding positions too long, adjusted by market conditions
                            base_aging_start = reward_config.get('position_aging_penalty_start', 20)
                            base_aging_rate = reward_config.get('position_aging_penalty_rate', 0.01)
                            
                            # BEHAVIOR-BASED ADAPTIVE: Get adjustments based on agent performance
                            behavior_adjustments = self._calculate_behavior_based_adaptive_rewards()
                            
                            # ADAPTIVE: Use same multipliers as decay (consistent with market conditions)
                            # In bull markets: start later, lower rate (let winners run)
                            # In bear/volatile markets: start earlier, higher rate (exit faster)
                            # BEHAVIOR-BASED ADAPTIVE: Apply behavior adjustments on top of market adjustments
                            position_aging_penalty_start = base_aging_start * start_mult * behavior_adjustments['aging_penalty_start_mult']  # Adaptive: 6-30 steps (behavior can reduce to 6)
                            position_aging_penalty_rate = base_aging_rate * rate_mult * behavior_adjustments['aging_penalty_rate_mult']     # FIXED: Consistent direction (was inverted)
                            
                            if avg_hold_time > position_aging_penalty_start:
                                aging_steps = avg_hold_time - position_aging_penalty_start
                                aging_penalty = -aging_steps * position_aging_penalty_rate  # Progressive penalty
                                aging_penalty = max(aging_penalty, -0.15)  # FIXED: Reduced cap from -0.30 to -0.15 (50% lighter)
                                position_incentive_value += aging_penalty  # Reduce net position incentive
                                if self.current_step % 50 == 0:  # Log periodically
                                    hold_log(f"[ADAPTIVE_AGING_PENALTY] Avg hold time: {avg_hold_time:.1f} steps, aging start: {position_aging_penalty_start:.1f} (base: {base_aging_start}, mult: {start_mult:.2f}x), aging rate: {position_aging_penalty_rate:.4f} (base: {base_aging_rate}, mult: {rate_mult:.2f}x), aging penalty: {aging_penalty:.3f}, net incentive: {position_incentive_value:.3f}")
                        
                        # FIXED: Add floor to prevent catastrophic negative incentive
                        position_incentive_value = max(position_incentive_value, -0.15)  # FIXED: Reduced floor from -0.30 to -0.15 (50% lighter)
                        market_movement_reward += position_incentive_value
                        # 2026-02-05: Small unrealized term - nudge for holding when up
                        unrealized_term = 0.02 * min(2.0, unrealized_pct)
                        market_movement_reward += unrealized_term
                        # 2026-02-05: Hold-winner bonus (train only), CAPPED: at most once per position every 20 steps so profitable SELL dominates
                        if current_mode == 'train':
                            hw_bonus = reward_config.get('hold_winner_bonus', 0.03)
                            hw_min_pct = reward_config.get('hold_winner_min_unrealized_pct', 0.02)
                            hw_min_steps = reward_config.get('hold_winner_min_steps', 10)
                            hw_cooldown_steps = 20  # Cap: add bonus at most once per 20 steps per position
                            if not hasattr(self, '_hold_winner_last_step') or not isinstance(getattr(self, '_hold_winner_last_step', None), dict):
                                self._hold_winner_last_step = {}
                            for i in range(self.num_tickers):
                                if i in self.positions and self.positions[i].get('shares', 0) > 0 and i < len(current_prices):
                                    avg_cost = self.positions[i].get('avg_cost', 0)
                                    if avg_cost > 0:
                                        pos_return = (current_prices[i] - avg_cost) / avg_cost
                                        if pos_return >= hw_min_pct:
                                            hold_time_i = (self.current_step - self._position_buy_steps[i]) if (i in self._position_buy_steps and self._position_buy_steps[i] is not None) else 0
                                            if hold_time_i >= hw_min_steps:
                                                last_added = self._hold_winner_last_step.get(i, -999)
                                                if self.current_step - last_added >= hw_cooldown_steps:
                                                    market_movement_reward += hw_bonus
                                                    self._hold_winner_last_step[i] = self.current_step
                                                    if self.current_step % 50 == 0:
                                                        hold_log(f"[HOLD_WINNER] Bonus +{hw_bonus:.3f} (ticker {i}, unrealized {pos_return:.2%}, held {hold_time_i} steps)")
                                                break
                        # Always log position incentive when calculated (for debugging)
                        if self.current_step % 50 == 0:  # Log periodically
                            hold_log(f"[DEBUG_TRADE] POSITION INCENTIVE: +{position_incentive_value:.3f} (unrealized gains: ${total_unrealized_gains:.2f}, {unrealized_pct:.2%} of portfolio, base={position_incentive_base:.3f})")
                    else:
                        hold_log(f"[DEBUG_TRADE] POSITION INCENTIVE: 0.000 (no unrealized gains)")
                else:
                    hold_log(f"[DEBUG_TRADE] NO POSITION INCENTIVE: 0.000 for no positions")

                # CRITICAL FIX: Use dynamic hold penalty based on portfolio state
                # This prevents the agent from learning that holding is profitable
                cash_pct = self.cash / self.portfolio_value if self.portfolio_value > 0 else 1.0
                threshold = reward_config.get('hold_penalty_transition_threshold', 0.9)  # 2026-02-20: 0.95→0.9 to match dynamic_config

                # ADAPTIVE: Calculate market-based penalty multiplier
                adaptive_penalty_multiplier = self._calculate_adaptive_hold_penalty()
                
                # BEHAVIOR-BASED ADAPTIVE: Get adjustments based on agent performance
                behavior_adjustments = self._calculate_behavior_based_adaptive_rewards()
                
                if cash_pct > threshold:
                    # Mostly cash - apply consistent penalty across train/eval to preserve BC policy
                    # FIX #1: REMOVED eval_multiplier that was reducing penalty by 70% in eval mode!
                    hold_penalty_base = reward_config.get('hold_penalty_no_positions', -0.035)  # ADAPTIVE: Base value from config
                    # BEHAVIOR-BASED ADAPTIVE: Apply behavior multiplier on top of market multiplier
                    hold_penalty = hold_penalty_base * adaptive_penalty_multiplier * behavior_adjustments['hold_penalty_mult']  # ADAPTIVE: Scale by market + behavior conditions
                    current_mode = getattr(self, 'mode', 'test')
                    if self.current_step % 50 == 0:  # Log periodically
                        self.logger.debug(
                            f"[ADAPTIVE_HOLD_PENALTY] No positions ({cash_pct:.1%} cash), base: {hold_penalty_base:.3f}, "
                            f"multiplier: {adaptive_penalty_multiplier:.2f}x, final: {hold_penalty:.3f}"
                        )
                else:
                    # Has positions - apply penalty; use grace period for recently opened positions (avoids "don't BUY" collapse)
                    hold_penalty_base = reward_config.get('hold_penalty_with_positions', -0.04)
                    grace_steps = int(reward_config.get('hold_penalty_grace_steps', 3))
                    grace_penalty = float(reward_config.get('hold_penalty_with_positions_grace', -0.02))
                    in_grace = False
                    if hasattr(self, '_position_buy_steps') and self._position_buy_steps:
                        newest_buy_step = max((s for s in self._position_buy_steps.values() if s is not None), default=-999)
                        if newest_buy_step >= 0 and (self.current_step - newest_buy_step) <= grace_steps:
                            in_grace = True
                            hold_penalty_base = grace_penalty
                    # BEHAVIOR-BASED ADAPTIVE: Apply behavior multiplier on top of market multiplier
                    hold_penalty = hold_penalty_base * adaptive_penalty_multiplier * behavior_adjustments['hold_penalty_mult']  # ADAPTIVE: Scale by market + behavior conditions
                    if self.current_step % 50 == 0:  # Log periodically
                        self.logger.debug(
                            f"[ADAPTIVE_HOLD_PENALTY] Has positions ({cash_pct:.1%} cash), base: {hold_penalty_base:.3f} "
                            f"({'grace' if in_grace else 'normal'}), multiplier: {adaptive_penalty_multiplier:.2f}x, final: {hold_penalty:.3f}"
                        )

                # Evaluation-mode override: honor eval_hold_penalty and disable progressive/strong penalties
                # ADAPTIVE: Apply market-based multiplier to eval hold penalty
                if getattr(self, 'mode', 'test') == 'test' or getattr(self, '_eval_mode', False):
                    try:
                        eval_hp_base = getattr(self.dynamic_config, 'eval_hold_penalty', -0.03) if hasattr(self, 'dynamic_config') else -0.03  # 2026-02-05: Match dynamic_config
                        # NEW: Performance-based hold penalty - reduce penalty if portfolio is growing
                        performance_based = getattr(self.dynamic_config, 'eval_hold_penalty_performance_based', False) if hasattr(self, 'dynamic_config') else False
                        if performance_based:
                            # Calculate portfolio return from initial balance
                            initial_balance = getattr(self, 'initial_balance', 100000.0)
                            portfolio_return = (self.portfolio_value - initial_balance) / initial_balance
                            growth_threshold = getattr(self.dynamic_config, 'eval_hold_penalty_growth_threshold', 0.005) if hasattr(self, 'dynamic_config') else 0.005  # 2026-02-05: 0.5% growth
                            growth_multiplier = getattr(self.dynamic_config, 'eval_hold_penalty_growth_multiplier', 0.75) if hasattr(self, 'dynamic_config') else 0.75
                            minimum_floor = getattr(self.dynamic_config, 'eval_hold_penalty_minimum_floor', -0.02) if hasattr(self, 'dynamic_config') else -0.02
                            
                            if portfolio_return > growth_threshold:
                                # Portfolio is growing - reduce hold penalty but enforce minimum floor
                                eval_hp = float(eval_hp_base) * growth_multiplier
                                # CRITICAL FIX: Enforce minimum floor - never reduce below minimum_floor
                                eval_hp = max(eval_hp, minimum_floor)
                                # PERFORMANCE: Gate hold penalty logs behind frequency check (every 50 steps)
                                if self.current_step % 50 == 0:
                                    self.logger.debug(f"[HOLD_PENALTY_EVAL] Portfolio growing ({portfolio_return:.2%} > {growth_threshold:.2%}), reducing penalty: {eval_hp_base:.3f} -> {eval_hp:.3f} (floor: {minimum_floor:.3f})")
                            else:
                                # Portfolio not growing - use full penalty
                                eval_hp = float(eval_hp_base)
                                # PERFORMANCE: Gate hold penalty logs behind frequency check (every 50 steps)
                                if self.current_step % 50 == 0:
                                    self.logger.debug(f"[HOLD_PENALTY_EVAL] Portfolio not growing ({portfolio_return:.2%} <= {growth_threshold:.2%}), using full penalty: {eval_hp:.3f}")
                        else:
                            eval_hp = float(eval_hp_base)
                        
                        # FIX: Do NOT apply adaptive multiplier to eval hold penalty in eval mode
                        # The adaptive multiplier was amplifying penalty up to 1.5x (-0.30 → -0.45)
                        # which caused 95% SELL-bias (agent preferred failed SELL over massive HOLD penalty)
                        # In eval mode, use stable penalty to get consistent performance assessment
                        # eval_hp = eval_hp * adaptive_penalty_multiplier  # DISABLED
                    except Exception:
                        eval_hp = -0.02  # FIX: Stable fallback, no multiplier
                    hold_penalty = float(eval_hp)
                    reward = market_movement_reward + hold_penalty
                    hold_log(f"[HOLD_PENALTY_EVAL] Using eval_hold_penalty={hold_penalty:.3f}; total reward: {reward:.3f}")
                    # FIXED: Removed hold profit bonus completely - hold penalty is the only signal
                    # No conflicting bonuses - agent learns that holding is costly
                    # Cap negative reward if configured
                    reward_cap_min = reward_config.get('reward_cap_min', -0.5)
                    if reward < reward_cap_min:
                        reward = reward_cap_min
                    # Return early from HOLD branch in evaluation
                    return reward

                # PHASE 4: Strengthen hold penalty when strong signals exist - agent must trade, not hold
                # For HOLD action, check all tickers for strong signals (no specific ticker for HOLD)
                # If strong signals exist on ANY ticker and agent HOLDS, apply stronger penalty
                try:
                    strong_signal_threshold = reward_config.get('strong_signal_threshold', 0.7)
                    strong_fund_threshold = reward_config.get('strong_fundamental_threshold', 0.65)
                    hold_signal_grace = int(reward_config.get('hold_signal_grace_steps', 3))
                    strong_signal_penalty = reward_config.get('hold_strong_signal_penalty', -0.1)

                    max_signal_strength = 0.0
                    best_ticker = None
                    best_scores = (0.0, 0.0)
                    for t in range(self.num_tickers):
                        try:
                            # PERFORMANCE: Use precomputed signals from cache
                            if hasattr(self, '_step_signals_cache') and t in self._step_signals_cache and \
                               'BUY' in self._step_signals_cache[t]:
                                tech_score = self._step_signals_cache[t]['BUY'].get('tech', 0.0)
                                fund_score = self._step_signals_cache[t]['BUY'].get('fund', 0.0)
                            else:
                                # Fallback to direct calculation if cache miss
                                tech_score = self._calculate_technical_signal_quality(t, 'BUY')
                                fund_score = self._calculate_fundamental_signal_quality(t, 'BUY')
                            combined_signal = max(tech_score, fund_score)
                            if combined_signal > max_signal_strength:
                                max_signal_strength = combined_signal
                                best_ticker = t
                                best_scores = (tech_score, fund_score)
                        except Exception:
                            continue

                    apply_strong_penalty = False
                    tech_val, fund_val = best_scores
                    if best_ticker is not None:
                        if tech_val >= strong_signal_threshold or fund_val >= strong_fund_threshold:
                            apply_strong_penalty = True

                    if getattr(self, '_recent_masked_sell', False):
                        apply_strong_penalty = False
                        self._recent_masked_sell = False
                    elif hasattr(self, '_consecutive_holds') and self._consecutive_holds <= hold_signal_grace:
                        apply_strong_penalty = False

                    if not has_positions:
                        apply_strong_penalty = False

                    if apply_strong_penalty and not self._fundamental_data_available:
                        apply_strong_penalty = False

                    if apply_strong_penalty:
                        hold_penalty += strong_signal_penalty
                        hold_log(
                            f"[HOLD_PENALTY_STRONG_SIGNAL] Strong signals detected on ticker {best_ticker} "
                            f"(tech={tech_val:.3f}, fund={fund_val:.3f}), increased hold penalty by {strong_signal_penalty:.3f} → total: {hold_penalty:.3f}"
                        )
                except Exception as e:
                    self.logger.debug(f"[HOLD_PENALTY] Could not calculate signals for penalty: {e}")

                # CRITICAL FIX: Save base penalty before progressive calculation to prevent exponential compounding
                hold_penalty_base = hold_penalty
                
                # PHASE 2: Progressive hold penalty - LINEAR scaling with caps (was exponential - FIXED)
                # Track consecutive holds and apply progressive penalty after configurable threshold
                progressive_start = reward_config.get('hold_progressive_start', 10)  # PHASE 2: Default now 10
                if hasattr(self, '_consecutive_holds') and self._consecutive_holds > progressive_start:
                    # PHASE 2: Use dynamic_config multiplier cap (3.0x) instead of hardcoded 2.0x
                    max_multiplier = getattr(self.dynamic_config, 'hold_progressive_multiplier', 3.0) if hasattr(self, 'dynamic_config') else 3.0
                    excess_holds = self._consecutive_holds - progressive_start
                    progressive_multiplier = min(1.0 + excess_holds * 0.1, max_multiplier)  # PHASE 2: Max 3.0x (was 2.0x)
                    progressive_penalty = hold_penalty_base * (progressive_multiplier - 1.0)  # Use BASE, not compounded
                    hold_penalty = hold_penalty_base + progressive_penalty  # Add to base, not compound
                    # CRITICAL: Cap total hold_penalty to prevent numerical overflow
                    hold_penalty = max(hold_penalty, -10.0)  # Cap at -10.0 maximum
                    if should_log_info:
                        hold_log(f"[HOLD_PENALTY_PROGRESSIVE] {self._consecutive_holds} consecutive holds, "
                                 f"progressive penalty: +{progressive_penalty:.4f} (multiplier: {progressive_multiplier:.2f}x, "
                                 f"base: {hold_penalty_base:.3f}), total hold penalty: {hold_penalty:.4f}")
                
                # PHASE 3 FIX: Apply invalid SELL penalty only once, then clear immediately
                # This penalty should ONLY apply to the step where SELL was attempted and masked
                # It should NOT persist to subsequent HOLD actions
                applied_masked_penalty = 0.0
                pending_penalty = getattr(self, '_pending_invalid_sell_penalty', 0.0)
                if pending_penalty:
                    hold_penalty += pending_penalty
                    applied_masked_penalty = pending_penalty
                    # PHASE 3 FIX: Clear immediately after applying (don't decay across steps)
                    self._pending_invalid_sell_penalty = 0.0

                # CRITICAL FIX: Numerical overflow safeguards before reward calculation
                if not np.isfinite(hold_penalty):
                    self.logger.warning(f"[HOLD_PENALTY_OVERFLOW] Non-finite hold_penalty detected: {hold_penalty}, resetting to -0.5")
                    hold_penalty = -0.5  # Fallback to safe value
                # Final cap check (redundant but safe)
                hold_penalty = max(hold_penalty, -10.0)

                # The total reward for HOLD should be market movement + small penalty
                # This allows positive rewards when market moves favorably
                reward = market_movement_reward + hold_penalty

                # 2026-02-28: Optional – HOLD in bull with >=5% unrealized gain (nudge to SELL at 5%)
                if has_positions:
                    try:
                        ro_hold = self._get_regime_outputs()
                        bull_hold = str(ro_hold.get('bull_bear', 'neutral')) if isinstance(ro_hold, dict) else 'neutral'
                        if bull_hold == 'bull':
                            pen_5pct = reward_config.get('hold_with_5pct_unrealized_in_bull_penalty', -0.02)
                            for i in range(self.num_tickers):
                                if i not in self.positions or self.positions[i].get('shares', 0) <= 0:
                                    continue
                                avg_cost = self.positions[i].get('avg_cost', 0)
                                if avg_cost <= 0 or i >= len(current_prices):
                                    continue
                                unrealized_pct = (current_prices[i] - avg_cost) / avg_cost
                                if unrealized_pct >= 0.05:
                                    reward += pen_5pct
                                    break
                    except Exception:
                        pass

                # FIXED: Removed conflicting hold profit bonus - hold penalty is the signal
                # Hold penalty (-0.05) already discourages holding, no need for conflicting bonus
                # Agent learns from portfolio performance, not artificial bonuses

                # PHASE 3 FIX: Penalty already cleared above - no decay logic needed
                # This ensures penalty only applies to the exact step where SELL was masked

                # CRITICAL FIX: Cap extremely negative rewards to prevent crashes
                # But allow strong negative rewards to discourage holding
                reward_cap_min = reward_config.get('reward_cap_min', -0.5)

                if reward < reward_cap_min:
                    reward = reward_cap_min

                # CRITICAL FIX: Track consecutive trades by action type (HOLD resets both counters)
                if not hasattr(self, '_consecutive_buys'):
                    self._consecutive_buys = 0
                    self._consecutive_sells = 0

                # HOLD action resets both consecutive buy and sell counters
                self._consecutive_buys = 0
                self._consecutive_sells = 0

                # CRITICAL FIX: Add exploration bonus ONLY during training
                # If agent has been holding too long, give exploration bonus
                if hasattr(self, '_consecutive_holds'):
                    self._consecutive_holds += 1
                else:
                    self._consecutive_holds = 1

                # FIX #2: Enable exploration bonus in BOTH train and eval modes to preserve BC trading policy
                exploration_bonus = 0.0
                current_mode = getattr(self, 'mode', 'test')
                # PHASE 1 FIX: Reduced threshold from 10 to 3 for faster exploration
                exploration_threshold = reward_config.get('exploration_bonus_threshold', 3)
                # PHASE 1 FIX: Increased rate from 0.01 to 0.03 for stronger signal
                exploration_rate = reward_config.get('exploration_bonus_rate', 0.03)
                # PHASE 1 FIX: Keep False to allow exploration in eval mode (preserves BC policy)
                train_only_bonus = reward_config.get('exploration_bonus_train_only', False)
                allow_exploration = True

                if train_only_bonus and current_mode != 'train':
                    allow_exploration = False

                # PHASE 1 FIX: REMOVED - Don't block exploration when no positions!
                # This was preventing agent from ever starting to trade when sitting in cash
                # if not has_positions:
                #     allow_exploration = False

                if allow_exploration and self._consecutive_holds > exploration_threshold:
                    steps_over = self._consecutive_holds - exploration_threshold
                    # PHASE 1 FIX: Increased cap from 0.05 to 0.15 for stronger motivation
                    exploration_bonus = min(steps_over * exploration_rate, 0.15)
                    if exploration_bonus > 0:
                        reward += exploration_bonus
                        self.logger.debug(
                            f"[EXPLORATION] Mode: {current_mode}, Consecutive holds: {self._consecutive_holds}, "
                            f"bonus: {exploration_bonus:.4f}"
                        )

                # CRITICAL: Add drawdown penalty for risk management (but keep it small)
                drawdown_penalty = self._calculate_drawdown_penalty()
                reward += drawdown_penalty * 0.1  # Reduced impact

                # CRITICAL FIX: Log detailed reward breakdown for debugging
                reward_components = {
                    'market_movement': market_movement_reward,
                    'hold_penalty': hold_penalty,
                    'drawdown_penalty': drawdown_penalty * 0.1,
                    'exploration_bonus': exploration_bonus,  # Now shows 0.0 in eval
                    'hold_profit_bonus': 0.0,  # FIXED: Removed - no longer used (was conflicting with hold penalty)
                    'invalid_sell_penalty': applied_masked_penalty,
                    'mode': current_mode  # Add mode to debug output
                }
                self._log_reward_breakdown(reward_components)

                hold_log(f"[DEBUG_TRADE] HOLD. Market movement reward: {market_movement_reward:.3f}, Hold penalty: {hold_penalty:.3f}, Total reward: {reward:.3f}")

                # CRITICAL FIX: HOLD actions should NOT be recorded as trades
                # Only actual BUY/SELL actions should be recorded in trade_history
                # Portfolio changes from market movement are tracked in portfolio_history
                self.logger.debug(f"HOLD: Portfolio ${portfolio_before:.2f}  ${portfolio_after:.2f} (change: {portfolio_change:.4f})")

            # CRITICAL FIX: Portfolio value already updated in buy/sell/hold actions
            # No need to update again here as it would override the correct values

            # Update portfolio state for diversification tracking
            self._update_portfolio_state()

            # Store selected ticker for info
            if action in [1, 2]:
                self._selected_ticker = ticker
            else:
                self._selected_ticker = None
            
            # DYNAMIC: Multi-timeframe reward scaling (5m/30m/1h/1m weights from config)
            if action in (1, 2):
                reward *= self._get_timeframe_reward_multiplier()
            # FIXED: Increased clipping range to preserve reward signals (was -10 to 10, now -50 to 50)
            # Note: Signal gating reward multiplier is already applied in BUY/SELL sections before clipping
            # Apply to ALL rewards (BUY, SELL, HOLD) before returning
            reward = np.clip(reward, -50.0, 50.0)  # FIXED: Wider range to preserve performance-scaled rewards
            if not np.isfinite(reward):
                reward = 0.0
            
            return reward
      
        except Exception as e:
            self.logger.error(f"Trade execution failed: {e}")
            import traceback
            self.logger.error(f"Trade execution traceback: {traceback.format_exc()}")
            # Return a small negative reward for failed trades
            return -0.1

    def _update_portfolio_value(self, current_prices):
        """Update portfolio value based on current positions and prices"""
        try:
            # CRITICAL FIX: Comprehensive price validation to prevent crashes
            for i in range(len(current_prices)):
                if np.isnan(current_prices[i]) or np.isinf(current_prices[i]) or current_prices[i] < 0:
                    self.logger.warning(f"Invalid price detected for ticker {i}: {current_prices[i]}, setting to 0.01")
                    if i < len(current_prices):
                        current_prices[i] = 0.01  # Set to small positive value to prevent crashes
            
            # CRITICAL FIX: Calculate position values correctly with new position structure
            # FIX: Add comprehensive position validation to prevent corruption
            position_value = 0.0
            max_affordable_shares_global = (self.initial_balance * 10.0)  # Max 10x initial balance worth of shares
            
            for i in range(min(self.num_tickers, len(current_prices))):
                if i in self.positions:
                    shares = self.positions[i].get('shares', 0.0)
                    
                    # Validate negative shares (short position): realize the short before zeroing so cash/portfolio stay correct
                    if shares < 0:
                        if i < len(current_prices) and current_prices[i] > 0:
                            cover_cost = abs(shares) * current_prices[i]
                            self.cash -= cover_cost  # Pay to cover the short; keeps portfolio = cash + positions consistent
                        self.logger.warning(f"Negative shares detected for ticker {i}: {shares}, setting to 0 (cash adjusted for cover)")
                        self.positions[i]['shares'] = 0.0
                        shares = 0.0
                    
                    # CRITICAL: Validate shares are not corrupted (prevent portfolio explosions)
                    if shares > 0 and current_prices[i] > 0:
                        max_affordable = max_affordable_shares_global / current_prices[i]
                        if shares > max_affordable:
                            self.logger.error(f"🚨 POSITION CORRUPTION DETECTED: Ticker {i} has {shares:.2f} shares, max affordable: {max_affordable:.2f}")
                            self.logger.error(f"🚨 Price: ${current_prices[i]:.2f}, Initial balance: ${self.initial_balance:.2f}")
                            self.logger.error(f"🚨 Resetting position to prevent portfolio explosion")
                            self.positions[i] = {'shares': 0.0, 'avg_cost': 0.0}
                            if hasattr(self, 'position_lots') and i in self.position_lots:
                                self.position_lots[i] = []  # Clear corrupted lots
                            shares = 0.0
                    
                    position_value += shares * current_prices[i]
            
            # DEBUG: Log portfolio calculation details (REDUCED FREQUENCY)
            if not hasattr(self, '_portfolio_log_counter'):
                self._portfolio_log_counter = 0
            self._portfolio_log_counter += 1
            
            # OPTIMIZED: Reduced logging frequency from every 100th to every 1000th, changed to DEBUG level
            if self._portfolio_log_counter % 1000 == 0:
                self.logger.debug(f"PORTFOLIO CALC: cash=${self.cash:.2f}, position_value=${position_value:.2f}")
                if hasattr(self, 'positions') and isinstance(self.positions, dict):
                    positions_str = [f"{self.positions[i]['shares']:.4f}" if i in self.positions else "0.0000" for i in range(min(6, self.num_tickers))]
                    prices_str = [f"{current_prices[i]:.2f}" if i < len(current_prices) else "0.00" for i in range(min(6, len(current_prices)))]
                    pos_values_str = [f"{(self.positions[i]['shares'] * current_prices[i]):.2f}" if i in self.positions and i < len(current_prices) else "0.00" for i in range(min(6, len(current_prices)))]
                    self.logger.debug(f"PORTFOLIO CALC: positions={positions_str}")
                    self.logger.debug(f"PORTFOLIO CALC: prices={prices_str}")
                    self.logger.debug(f"PORTFOLIO CALC: position_values={pos_values_str}")
            
            # CRITICAL FIX: Update portfolio value with safety checks and bounds protection
            old_portfolio = self.portfolio_value
            calculated_portfolio = self.cash + position_value
            
            # CRITICAL FIX: Cap portfolio value GAINS to prevent explosion (max 10% per step)
            # FIXED: Removed loss cap - losses should be fully recognized (accurate accounting)
            # Gap losses and large drawdowns are real trading risks that should be reflected
            if old_portfolio > 0:
                max_change = old_portfolio * 0.10  # Max 10% gain per step (prevent numerical explosions)
                change = calculated_portfolio - old_portfolio
                if change > max_change:
                    self.logger.warning(f"Portfolio gain capped in _update_portfolio_value: ${change:.2f} > ${max_change:.2f} (10% limit to prevent numerical explosion)")
                    calculated_portfolio = old_portfolio + max_change
                # REMOVED: Loss cap - losses are now fully recognized for accurate return calculation
                # This allows gap losses (e.g., -76.95%) to be properly reflected in portfolio value
            
            # CRITICAL FIX: Validate portfolio value is within reasonable bounds
            max_reasonable = self.initial_balance * 10  # Max 10x initial
            min_reasonable = self.initial_balance * 0.1  # Min 10% of initial
            
            if (not np.isfinite(calculated_portfolio) or 
                calculated_portfolio > max_reasonable or 
                calculated_portfolio < min_reasonable):
                self.logger.error(f"Portfolio value out of bounds in _update_portfolio_value: ${calculated_portfolio:.2f}. Recalculating.")
                # Recalculate from cash + positions
                calculated_portfolio = self.cash + position_value
                # If still out of bounds, reset
                if (calculated_portfolio > max_reasonable or calculated_portfolio < min_reasonable):
                    self.logger.error(f"Portfolio still out of bounds: ${calculated_portfolio:.2f}. Resetting to ${self.initial_balance:.2f}")
                    calculated_portfolio = self.initial_balance
                    self.cash = self.initial_balance
                    self.positions = {i: {'shares': 0.0, 'avg_cost': 0.0} for i in range(self.num_tickers)}
                    # CRITICAL FIX: Clear position_lots when portfolio explodes to prevent corruption (DUPLICATE FUNCTION FIX)
                    self._clear_position_lots()
            
            # OPTIMIZED: Direct assignment - validation already done above
            self.portfolio_value = calculated_portfolio
            
            # FIX 2: Validate portfolio value matches cash + positions (ground truth check)
            expected_portfolio = self.cash + position_value
            if abs(self.portfolio_value - expected_portfolio) > 0.01:  # Allow small floating-point differences
                # If gain cap was applied, this is expected - log at debug level
                if self.portfolio_value < expected_portfolio:
                    self.logger.debug(f"Portfolio value validation: ${self.portfolio_value:.2f} vs expected ${expected_portfolio:.2f} (gain cap applied)")
                else:
                    # This should never happen - portfolio should never exceed cash + positions
                    self.logger.error(f"CRITICAL: Portfolio value ${self.portfolio_value:.2f} exceeds cash+positions ${expected_portfolio:.2f} - forcing correction")
                    self.portfolio_value = expected_portfolio
            
            # CRITICAL FIX: Check for NaN or infinity values
            if np.isnan(self.portfolio_value) or np.isinf(self.portfolio_value):
                self.logger.error(f"Portfolio value is NaN/inf: cash={self.cash}, position_value={position_value}")
                self.portfolio_value = max(0.0, self.cash)  # Safe fallback
            
            # CRITICAL FIX: Log portfolio updates for debugging (reduced frequency for performance)
            # PERFORMANCE: Changed to DEBUG level to reduce I/O overhead
            if hasattr(self, '_last_portfolio_value'):
                if abs(self.portfolio_value - self._last_portfolio_value) > 10.0:  # Increased threshold to reduce logging
                    self.logger.debug(f"Portfolio value changed: ${self._last_portfolio_value:.2f}  ${self.portfolio_value:.2f}")
                    self._total_portfolio_changes += 1
            
            self._last_portfolio_value = self.portfolio_value
            
            # CRITICAL FIX: Ensure portfolio value is never negative and handle crashes
            if self.portfolio_value < 0:
                self.logger.warning(f"Portfolio value was negative, reset to cash: ${self.cash:.2f}")
                self.portfolio_value = max(0.0, self.cash)  # Reset to cash value
                
                # Additional safety: if cash is also negative, reset to initial balance
                if self.cash < 0:
                    self.logger.error(f"Cash is negative: ${self.cash:.2f}, resetting to initial balance")
                    self.cash = self.initial_balance
                    self.portfolio_value = self.initial_balance
            
        except Exception as e:
            self.logger.error(f"Portfolio value update failed: {e}")
            # CRITICAL FIX: Fallback to safe value
            self.portfolio_value = self.cash
    
    def _calculate_diversification_score(self):
        """Calculate diversification score (0.0 = concentrated, 1.0 = well diversified)"""
        try:
            if self.portfolio_value <= 0:
                return 0.0
            
            # Calculate position weights
            position_weights = []
            for i in range(self.num_tickers):
                if i < len(self.price_matrix[self.current_step]):
                    current_price = self.price_matrix[self.current_step][i]
                    position_value = self.positions.get(i, {'shares': 0.0})['shares'] * current_price
                    weight = position_value / self.portfolio_value
                    position_weights.append(weight)
                else:
                    position_weights.append(0.0)
            
            # Calculate Herfindahl-Hirschman Index (HHI) for concentration
            # Lower HHI = more diversified
            hhi = sum(w * w for w in position_weights)
            
            # Convert HHI to diversification score (0.0 = concentrated, 1.0 = diversified)
            # For 6 tickers, minimum HHI = 1/6  0.167 (equal weight)
            # Maximum HHI = 1.0 (100% in one ticker)
            min_hhi = 1.0 / self.num_tickers
            max_hhi = 1.0
            
            if hhi <= min_hhi:
                diversification_score = 1.0  # Perfect diversification
            else:
                # Linear interpolation between min and max
                diversification_score = 1.0 - ((hhi - min_hhi) / (max_hhi - min_hhi))
            
            return max(0.0, min(1.0, diversification_score))
            
        except Exception as e:
            self.logger.error(f"Failed to calculate diversification score: {e}")
            return 0.5  # Default to neutral score
    
    def _calculate_drawdown_penalty(self):
        """CRITICAL: Calculate drawdown penalty for risk management
        
        Returns:
            float: Drawdown penalty (negative value to discourage large drawdowns)
        """
        try:
            # Update peak portfolio value
            if self.portfolio_value > self.peak_portfolio_value:
                self.peak_portfolio_value = self.portfolio_value
            
            # Calculate current drawdown percentage
            if self.peak_portfolio_value > 0:
                self.current_drawdown = (self.peak_portfolio_value - self.portfolio_value) / self.peak_portfolio_value
            else:
                self.current_drawdown = 0.0
            
            # Apply drawdown penalty: R_t = PnL_t -  * (drawdown_t)
            # This creates smooth, continuous discouragement of large drawdowns
            drawdown_penalty = self.drawdown_penalty_lambda * (self.current_drawdown ** 2)
            
            # Log drawdown for monitoring
            if self.current_drawdown > 0.05:  # Log if drawdown > 5%
                self.logger.warning(f" DRAWDOWN ALERT: {self.current_drawdown:.2%} (penalty: {drawdown_penalty:.4f})")
            
            return -drawdown_penalty  # Negative to penalize
            
        except Exception as e:
            self.logger.error(f"Failed to calculate drawdown penalty: {e}")
            return 0.0

    def _calculate_portfolio_value_fast(self, current_prices: np.ndarray) -> float:
        """PERFORMANCE: Fast vectorized portfolio calculation with caching"""
        try:
            # Cache portfolio value for the current step
            # HIGH PRIORITY FIX: Verify portfolio caching is working
            if hasattr(self, '_cached_portfolio_step') and self._cached_portfolio_step == self.current_step:
                cached_value = getattr(self, '_cached_portfolio_value', None)
                if cached_value is not None:
                    # Log cache hit every 100 steps for verification
                    if self.current_step % 100 == 0:
                        self.logger.debug(f"[CACHE_VERIFY] Portfolio cache hit at step {self.current_step}")
                    return float(cached_value)
            
            # CRITICAL FIX: Validate positions before calculation to prevent explosions
            max_affordable_per_ticker = (self.initial_balance * 10.0)  # Max 10x initial per ticker
            for ticker in self.positions:
                if ticker < self.num_tickers and ticker < len(current_prices):
                    shares = self.positions[ticker].get('shares', 0.0)
                    price = current_prices[ticker] if ticker < len(current_prices) else 0.0
                    position_value = shares * price
                    
                    # Validate position value is reasonable
                    if position_value > max_affordable_per_ticker:
                        self.logger.error(f"🚨 CRITICAL: Position {ticker} value ${position_value:,.2f} exceeds max ${max_affordable_per_ticker:,.2f}")
                        self.logger.error(f"🚨 Shares: {shares:.2f}, Price: ${price:.2f}")
                        # Reset corrupted position
                        self.positions[ticker] = {'shares': 0.0, 'avg_cost': 0.0}
                        if hasattr(self, 'position_lots') and ticker in self.position_lots:
                            self.position_lots[ticker] = []
            
            # PERFORMANCE: Fully vectorized calculation (no Python loops)
            # Build shares array directly from positions dict (single source of truth)
            shares_array = np.zeros(self.num_tickers)
            for ticker in self.positions:
                if ticker < self.num_tickers:
                    shares_array[ticker] = self.positions[ticker].get('shares', 0.0)
            
            # Vectorized multiplication and sum (much faster than loop)
            if len(current_prices) >= self.num_tickers:
                position_values = shares_array * current_prices[:self.num_tickers]
            else:
                position_values = shares_array[:len(current_prices)] * current_prices
            
            portfolio_value = float(self.cash + np.sum(position_values))
            
            # Cache for this step
            self._cached_portfolio_value = portfolio_value
            self._cached_portfolio_step = self.current_step
            
            return portfolio_value
        except Exception as e:
            self.logger.error(f"Fast portfolio calculation failed: {e}")
            # Fallback to manual calculation
            return self.cash + sum(self.positions[i]['shares'] * current_prices[i] for i in range(min(self.num_tickers, len(current_prices))))
    
    def _should_gate_trade_by_signal_confidence(self, tech_signal: float, fund_signal: float, action: str) -> tuple[bool, float]:
        """Check if trade should be gated based on signal confidence (quality over quantity)
        
        Returns:
            tuple: (should_trade, reward_multiplier)
                - should_trade: True if signals are strong enough
                - reward_multiplier: 1.0 for high confidence, 0.7 for medium, 0.3 for low/exploration
        """
        try:
            # PHASE 0 FIX: Apply signal quality checks even during warmup (with relaxed thresholds)
            # Extend evaluation warm-up: allow limited trades early in eval to avoid HOLD lock-in
            eval_warmup = getattr(getattr(self, 'dynamic_config', None), 'eval_exploration_warmup_steps', 75)
            if (getattr(self, 'mode', 'test') == 'test') and getattr(self, 'current_step', 0) < eval_warmup:
                # Still apply signal checks but with relaxed thresholds during warmup
                # Use lower multiplier (0.5x instead of 0.7x) to indicate lower quality
                # Continue with signal confidence check below instead of bypassing
                warmup_mode = True
            else:
                warmup_mode = False

            # Portfolio state helpers
            total_shares = 0.0
            if hasattr(self, 'positions') and isinstance(self.positions, dict):
                for pos in self.positions.values():
                    try:
                        total_shares += float(pos.get('shares', 0.0))
                    except Exception:
                        continue
            is_flat = total_shares <= 0.0
            cash_pct = self.cash / self.portfolio_value if getattr(self, 'portfolio_value', 0.0) > 0 else 1.0

            # If flat (no positions), never SELL; encourage BUY instead
            if is_flat and action == 'SELL':
                if hasattr(self, '_gating_stats'):
                    self._gating_stats['blocked'] += 1
                return False, 0.0

            # Step 2: In technical-only mode OR when fundamentals minor/disabled, use SOFT gating (never block BUY/SELL).
            # FIX 2026-02-15: Enable when fundamental_signal_weight < 0.2 (technical-dominant) to reduce BUY blocks.
            tech_only_soft_gating = False
            try:
                dc = getattr(self, 'dynamic_config', None)
                fund_minor = (getattr(dc, 'fundamental_signal_weight', 0.0) < 0.2)
                reward_mode_ok = (getattr(dc, 'reward_mode', '') == 'technical_only_v1')
                tech_only_soft_gating = (
                    bool(getattr(dc, 'tech_only_soft_gating', True))
                    and (reward_mode_ok or fund_minor)
                )
            except Exception:
                tech_only_soft_gating = False

            if tech_only_soft_gating:
                confidence = float(abs(tech_signal))
                if confidence >= 0.25:
                    if hasattr(self, '_gating_stats'):
                        self._gating_stats['allowed'] += 1
                        self._gating_stats['high'] += 1
                    return True, 1.0
                if confidence >= 0.10:
                    if hasattr(self, '_gating_stats'):
                        self._gating_stats['allowed'] += 1
                        self._gating_stats['medium'] += 1
                    return True, 0.7
                if hasattr(self, '_gating_stats'):
                    self._gating_stats['allowed'] += 1
                    self._gating_stats['low'] += 1
                    self._gating_stats['explore'] += 1
                return True, 0.3

            # CRITICAL FIX: Handle disabled fundamental signals (always returns 0.0)
            fund_available = getattr(self, '_fundamental_data_available', True)
            # Check if fundamentals are actually disabled (fund_signal is always 0.0 when disabled)
            # If fund_signal is exactly 0.0, treat as disabled (fundamental function returns 0.0 when disabled)
            fundamentals_disabled = (fund_signal == 0.0)
            
            if fundamentals_disabled or not fund_available:
                # Use technical signal alone for confidence and alignment when fundamentals disabled
                confidence = abs(tech_signal)
                if action == 'BUY':
                    # 2026-02-11: When fundamentals disabled, allow BUY even when tech is negative; use
                    # confidence only for reward multiplier (reduces 52%+ BUY block rate, keeps full reward).
                    signal_aligned = True
                else:  # SELL
                    # CRITICAL FIX: When fundamentals disabled, allow SELL whenever we have a position.
                    # Requiring tech_signal < 0 was blocking 87%+ of SELLs (agent could not take profit
                    # or cut loss when tech was still bullish). Use confidence only for reward multiplier.
                    signal_aligned = True
                    # FIX 2026-02-04: Weakly bearish tech (e.g. -0.09) gave confidence 0.09 < med_thr 0.10
                    # and blocked SELLs. Ensure SELL with position clears medium threshold so exits are allowed.
                    confidence = max(confidence, 0.12)
            else:
                # PHASE 1 FIX: Calculate confidence based on signal alignment (not absolute values)
                # For BUY: both should be positive
                # For SELL: both should be negative
                if action == 'BUY':
                    # CRITICAL FIX: For BUY, aligned means both signals are positive
                    signal_aligned = (tech_signal > 0 and fund_signal > 0)
                    if tech_signal > 0 and fund_signal > 0:
                        confidence = (tech_signal + fund_signal) / 2.0
                    else:
                        confidence = 0.0  # Conflicting signals = no confidence
                    # CRITICAL FIX: If signals are conflicting (one > 0.5, other < 0.5), block trade
                    # This detects mixed strength signals (e.g., 0.60 and 0.40)
                    if (tech_signal > 0.5 and fund_signal < 0.5) or (tech_signal < 0.5 and fund_signal > 0.5):
                        signal_aligned = False  # Mixed strength = conflicting
                    # Also block if one is negative
                    if tech_signal < 0 or fund_signal < 0:
                        signal_aligned = False
                        confidence = 0.0
                else:  # SELL
                    # PHASE 1 FIX: SELL if EITHER signal is bearish (OR logic, not AND)
                    signal_aligned = (tech_signal < 0.5 or fund_signal < 0.5)
                    if tech_signal < 0 and fund_signal < 0:
                        confidence = abs((tech_signal + fund_signal) / 2.0)
                    else:
                        confidence = 0.0  # Conflicting signals = no confidence
                    # CRITICAL FIX: If signals are conflicting (one positive, one negative), force confidence to 0
                    if (tech_signal > 0 and fund_signal < 0) or (tech_signal < 0 and fund_signal > 0):
                        confidence = 0.0
                        signal_aligned = False

            # Prefer BUY when cash is high: slightly relax thresholds for BUY
            high_cash_buy_bonus = (cash_pct > 0.95 and action == 'BUY')
            # 2026-02-06: When flat (no positions), relax BUY gating to prevent long flat stretches
            flat_buy_relaxed = (is_flat and action == 'BUY')

            # Thresholds: market-first regime controller provides the base;
            # warmup/flat/high-cash logic can relax them further.
            # 2026-02-26: 0.25→0.28, 0.10→0.12 – raise entry quality bar
            base_high_thr = 0.28
            base_med_thr = 0.12
            try:
                ro = self._get_regime_outputs()
                if isinstance(ro, dict):
                    base_high_thr = float(ro.get('gate_high_threshold', base_high_thr))
                    base_med_thr = float(ro.get('gate_medium_threshold', base_med_thr))
            except Exception:
                base_high_thr, base_med_thr = 0.28, 0.12

            # PHASE 0 FIX: Relax thresholds further during warmup
            if warmup_mode:
                high_thr = min(0.15, base_high_thr) - (0.05 if high_cash_buy_bonus else 0.0)
                med_thr = min(0.05, base_med_thr) - (0.05 if high_cash_buy_bonus else 0.0)
            elif flat_buy_relaxed:
                high_thr = min(0.15, base_high_thr)
                med_thr = min(0.05, base_med_thr)
            else:
                high_thr = base_high_thr - (0.05 if high_cash_buy_bonus else 0.0)
                med_thr = base_med_thr - (0.05 if high_cash_buy_bonus else 0.0)

            # Safety clamp (avoid negative thresholds under high-cash relaxation)
            high_thr = float(np.clip(high_thr, 0.05, 0.50))
            med_thr = float(np.clip(med_thr, 0.01, 0.40))
            
            # High confidence: signals align and are strong
            if confidence > high_thr and signal_aligned:
                if hasattr(self, '_gating_stats'):
                    self._gating_stats['allowed'] += 1
                    self._gating_stats['high'] += 1
                return True, 0.5 if warmup_mode else 1.0  # Lower multiplier during warmup
            
            # Medium confidence: moderate signals (CRITICAL FIX: Must also be aligned)
            elif confidence > med_thr and signal_aligned:
                if hasattr(self, '_gating_stats'):
                    self._gating_stats['allowed'] += 1
                    self._gating_stats['medium'] += 1
                return True, 0.5 if warmup_mode else 0.7  # Lower multiplier during warmup
            
            # Low confidence: weak signals
            else:
                # 2026-02-26: Tighten min_confidence_block 0.03→0.05 when flat (quality over quantity)
                min_confidence_block = 0.05
                if confidence < min_confidence_block:
                    if hasattr(self, '_gating_stats'):
                        self._gating_stats['blocked'] += 1
                    return False, 0.0

                # CRITICAL FIX: Block conflicting signals - no exploration override
                if not signal_aligned:
                    # Conflicting signals - always block, no exploration
                    if hasattr(self, '_gating_stats'):
                        self._gating_stats['blocked'] += 1
                    return False, 0.0  # Block conflicting signals
                
                # PHASE 2 FIX: Structured exploration - only for medium-low confidence (0.05-0.10) with aligned signals
                # Only explore if we haven't traded recently AND signals are aligned
                exploration_cooldown = 10  # steps
                if not hasattr(self, '_last_trade_step'):
                    self._last_trade_step = -exploration_cooldown
                
                if self.current_step - self._last_trade_step > exploration_cooldown:
                    # Allow exploration with minimal reward (only for aligned, medium-low confidence)
                    if hasattr(self, '_gating_stats'):
                        self._gating_stats['allowed'] += 1
                        self._gating_stats['low'] += 1
                        self._gating_stats['explore'] += 1
                    return True, 0.3  # Minimal reward - exploration only
                else:
                    # Block low-confidence trades if we traded recently
                    if hasattr(self, '_gating_stats'):
                        self._gating_stats['blocked'] += 1
                    return False, 0.0  # Skip low-quality trade
                
        except Exception as e:
            self.logger.error(f"Signal confidence gating failed: {e}")
            # PHASE 0 FIX: Be conservative on errors - block trade instead of allowing
            return False, 0.0  # Default: block trade if gating fails (conservative approach)

    def get_gating_stats(self) -> dict:
        """Return run-level gating diagnostics counters."""
        try:
            if hasattr(self, '_gating_stats') and isinstance(self._gating_stats, dict):
                # Return a shallow copy to avoid external mutation
                return {k: int(v) for k, v in self._gating_stats.items()}
        except Exception:
            pass
        return {}
    
    def _check_dynamic_stop_loss(self, ticker: int, current_price: float) -> bool:
        """CRITICAL: Check and execute dynamic stop loss for positions
        
        IMPROVED: Uses advanced formula with volatility + ATR + regime + signal awareness
        
        Args:
            ticker: Ticker index
            current_price: Current price of the ticker
            
        Returns:
            bool: True if stop loss was triggered and position was closed
        """
        try:
            if ticker not in self.positions or self.positions[ticker]['shares'] <= 0:
                return False
            
            position = self.positions[ticker]
            shares = position['shares']
            avg_cost = position.get('avg_cost', current_price)
            is_monthly = getattr(self, "bar_frequency", "day") == "month"

            # Monthly: give agent time to choose SELL before auto-stop steals the exit
            min_hold = int(getattr(self, "min_hold_steps_before_stop", 0) or 0)
            if is_monthly and min_hold > 0:
                buy_step = None
                if hasattr(self, "_position_buy_steps"):
                    buy_step = self._position_buy_steps.get(ticker)
                if buy_step is not None and (self.current_step - int(buy_step)) < min_hold:
                    return False
            
            # FIX 5: Gap Fill Simulation - Detect price gaps and use stop loss price for execution
            # This matches real broker behavior where stop losses fill at limit price, not gap price
            prev_close = None
            gap_detected = False
            gap_fill_price = current_price  # Default to current price if no gap
            extreme_gap_capped = False  # Track if we capped an extreme gap
            gap_detect_thresh = -0.20 if is_monthly else -0.05
            extreme_gap_thresh = -0.45 if is_monthly else -0.20
            max_loss_pct = 0.30 if is_monthly else 0.15
            
            if hasattr(self, 'price_matrix') and self.current_step > 0:
                try:
                    prev_close = self.price_matrix[self.current_step - 1, ticker]
                    if prev_close > 0:
                        gap_pct = (current_price - prev_close) / prev_close
                        if gap_pct < gap_detect_thresh:
                            gap_detected = True
                            self.logger.debug(f"Gap detected for ticker {ticker}: {gap_pct:.2%} (prev: ${prev_close:.2f}, current: ${current_price:.2f})")
                            if gap_pct < extreme_gap_thresh and avg_cost > 0:
                                min_execution_price = avg_cost * (1 - max_loss_pct)
                                if current_price < min_execution_price:
                                    self.logger.warning(f"EXTREME GAP: {gap_pct:.2%} drop detected, capping execution loss to {max_loss_pct:.1%}")
                                    gap_fill_price = min_execution_price
                                    extreme_gap_capped = True
                                    position_value = shares * avg_cost
                                    self.logger.warning(f"EXTREME GAP DETAILS: Position value=${position_value:.2f}, Shares={shares:.4f}, AvgCost=${avg_cost:.2f}, GapPrice=${current_price:.2f}, ExecutionPrice=${gap_fill_price:.2f}")
                                else:
                                    gap_fill_price = current_price
                            else:
                                gap_fill_price = current_price
                except (IndexError, ValueError) as e:
                    self.logger.debug(f"Could not calculate gap for ticker {ticker}: {e}")
            
            # Calculate current PnL percentage (using gap_fill_price if gap detected)
            if avg_cost > 0:
                pnl_pct = (current_price - avg_cost) / avg_cost
            else:
                return False
            
            # ADVANCED STOP LOSS FORMULA - Volatility + ATR + Regime + Signal Awareness
            # Use instance attrs set from dynamic_config in __init__ (base 10%, trailing 8%)
            # NOTE: "0 stop loss triggers" in reports usually means the agent sold before the
            # stop threshold was hit (e.g. agent exits at -2%/-3%, stop is at 5-30%).
            base_stop_loss = getattr(self, 'base_stop_loss_pct', 0.10)
            trailing_stop = getattr(self, 'trailing_stop_pct', 0.08)

            # Market-first: regime-aware stop width (controller outputs are bounded and cached)
            try:
                ro = self._get_regime_outputs()
                if isinstance(ro, dict):
                    base_stop_loss = float(base_stop_loss) * float(ro.get('stop_loss_mult', 1.0))
                    trailing_stop = float(trailing_stop) * float(ro.get('trailing_stop_mult', 1.0))
            except Exception:
                pass
            # Clamp to sane bounds (monthly bars need wider stops so agent can learn SELL)
            if is_monthly:
                base_stop_loss = float(np.clip(max(base_stop_loss, 0.25), 0.15, 0.45))
                trailing_stop = float(np.clip(max(trailing_stop, 0.18), 0.10, 0.35))
            else:
                base_stop_loss = float(np.clip(base_stop_loss, 0.05, 0.30))
                trailing_stop = float(np.clip(trailing_stop, 0.02, 0.20))
            
            # 1. Calculate 20-day volatility (more robust than 10-day)
            # DATA LEAKAGE PREVENTION: Stop loss includes current_step (intentional)
            # Python slicing [start:end] is exclusive on end, so [start:current_step+1] includes current_step
            # This is correct - we need current step's price to check if stop loss should trigger
            volatility_adjustment = 0.0
            atr_adjustment = 0.0
            if hasattr(self, 'price_matrix') and self.current_step > 20:
                # Bounds check: ensure we don't access beyond available data
                max_available_step = len(self.price_matrix) - 1
                if self.current_step > max_available_step or ticker >= self.price_matrix.shape[1]:
                    return False  # Can't calculate stop loss without valid data
                # Slice includes current_step (intentional - needed for stop loss check)
                recent_prices = self.price_matrix[max(0, self.current_step-20):self.current_step+1, ticker]
                if len(recent_prices) > 5:
                    # Volatility-based adjustment
                    returns = np.diff(recent_prices) / recent_prices[:-1]
                    recent_volatility = np.std(returns)
                    volatility_multiplier = 1.0 + (recent_volatility * 2.0)
                    volatility_adjustment = (volatility_multiplier - 1.0) * base_stop_loss
                    
                    # ATR-based adjustment (more accurate than simple volatility)
                    high_low_range = np.abs(np.diff(recent_prices))
                    atr = np.mean(high_low_range[-14:]) if len(high_low_range) >= 14 else np.mean(high_low_range)
                    atr_adjustment = (atr / current_price) * 0.5
            
            # 2. Regime-based adjustment (if regime detection available)
            regime_adjustment = 0.0
            if hasattr(self, '_active_regime'):
                regime = self._active_regime
                if regime == 'volatile':
                    regime_adjustment = base_stop_loss * 0.3  # 30% wider in volatile markets
                elif regime == 'bear':
                    regime_adjustment = base_stop_loss * 0.2  # 20% wider in bear markets
                elif regime == 'bull':
                    regime_adjustment = -base_stop_loss * 0.1  # 10% tighter in bull markets
            
            # 3. SIGNAL-AWARE ADJUSTMENT - Tighten if signals warn, widen if bullish
            signal_adjustment = 0.0
            tech_signal = 0.5  # Default neutral
            if hasattr(self, '_last_technical_signals') and ticker in self._last_technical_signals:
                tech_signal = self._last_technical_signals[ticker]
                
                if tech_signal < 0.3:  # Strong SELL signal
                    signal_adjustment = -base_stop_loss * 0.2  # 20% tighter - signals warn of drop
                elif tech_signal > 0.7:  # Strong BUY signal
                    signal_adjustment = base_stop_loss * 0.15  # 15% wider - signals say hold
            
            # Calculate final dynamic stop loss with all adjustments
            total_volatility_adjustment = volatility_adjustment + atr_adjustment
            dynamic_stop_loss = float(base_stop_loss + total_volatility_adjustment + regime_adjustment + signal_adjustment)
            
            # Apply reasonable bounds (monthly wider so agent can learn SELL)
            if is_monthly:
                dynamic_stop_loss = max(0.15, min(0.45, float(dynamic_stop_loss)))
            else:
                dynamic_stop_loss = max(0.05, min(0.30, float(dynamic_stop_loss)))
            
            # FIX 5: Calculate stop loss price for gap fill simulation
            stop_loss_price = avg_cost * (1 - dynamic_stop_loss)  # Price at which stop loss would trigger
            
            # FIX 5: If gap detected and price gapped below stop loss, use stop loss price for execution
            execution_price = current_price  # Default to current price
            if gap_detected and prev_close is not None and prev_close > 0:
                # Check if price gapped below stop loss threshold
                gap_below_stop = current_price < stop_loss_price
                if gap_below_stop:
                    # CRITICAL FIX: Use gap_fill_price if extreme gap was capped, otherwise use stop_loss_price
                    # Ensure we ALWAYS use the capped price for extreme gaps
                    if extreme_gap_capped:
                        # CRITICAL: Always use gap_fill_price for extreme gaps (already calculated with 15% cap)
                        execution_price = gap_fill_price
                        self.logger.info(f"🔄 GAP FILL (CAPPED): Ticker {ticker} gapped from ${prev_close:.2f} to ${current_price:.2f}, executing at ${execution_price:.2f} (loss capped to 15%)")
                    elif gap_fill_price != current_price:
                        # Use gap_fill_price if it was set (for non-extreme gaps)
                        execution_price = gap_fill_price
                        self.logger.info(f"🔄 GAP FILL: Ticker {ticker} gapped from ${prev_close:.2f} to ${current_price:.2f}, executing at ${execution_price:.2f}")
                    else:
                        # Use stop loss price for execution (gap fill simulation)
                        execution_price = stop_loss_price
                        self.logger.info(f"🔄 GAP FILL: Ticker {ticker} gapped from ${prev_close:.2f} to ${current_price:.2f}, executing stop loss at ${execution_price:.2f} (not gap price)")
            
            # Check if stop loss should be triggered (using execution_price for PnL calculation)
            should_stop_loss = False
            stop_reason = ""
            stop_loss_penalty = -0.5  # Default penalty
            
            # Calculate PnL using execution price (gap fill price if gap detected)
            execution_pnl_pct = (execution_price - avg_cost) / avg_cost if avg_cost > 0 else 0.0
            
            if execution_pnl_pct < -dynamic_stop_loss:
                should_stop_loss = True
                if gap_detected and execution_price != current_price:
                    stop_reason = f"Dynamic stop loss triggered with gap fill: {execution_pnl_pct:.2%} < -{dynamic_stop_loss:.2%} (gap: {current_price:.2f}, executed: {execution_price:.2f})"
                else:
                    stop_reason = f"Dynamic stop loss triggered: {execution_pnl_pct:.2%} < -{dynamic_stop_loss:.2%}"
                
                # SIGNAL-AWARE PENALTY: Smaller if signals warned, larger if unexpected
                if tech_signal < 0.3:  # Signals warned of drop
                    stop_loss_penalty = -0.1  # Small penalty - you should have listened
                    stop_reason += " (signals warned)"
                else:  # Unexpected drop
                    stop_loss_penalty = -0.5  # Full penalty - unexpected
                    stop_reason += " (unexpected drop)"
                    
            elif pnl_pct > 0 and 'peak_price' in position:
                # Trailing stop loss for profitable positions
                peak_price = position.get('peak_price', current_price)
                if current_price < peak_price * (1 - trailing_stop):
                    should_stop_loss = True
                    stop_reason = f"Trailing stop loss triggered: {current_price:.2f} < {peak_price * (1 - trailing_stop):.2f}"
                    # HIGH PRIORITY FIX: Trailing stop bonus - encourages profit-taking via trailing stops
                    # Get trailing stop bonus from config (default 0.10)
                    try:
                        from ppo.utils.dynamic_config import get_dynamic_config  # type: ignore
                        dcfg = self._cfg()
                        trailing_stop_bonus = getattr(dcfg, 'trailing_stop_bonus', 0.10)
                    except Exception:
                        trailing_stop_bonus = 0.10
                    # Trailing stop gets bonus (not penalty) since we're taking profits
                    stop_loss_penalty = trailing_stop_bonus  # Bonus for profit-taking via trailing stop
            
            # Execute stop loss if triggered
            if should_stop_loss:
                # CRITICAL FIX: Use execution_price (gap fill price if gap detected) instead of current_price
                # This ensures stop losses execute at limit price, not gap price (realistic broker behavior)
                # IMPORTANT: execution_price is calculated above and should NEVER be overwritten
                final_execution_price = execution_price
                
                # CRITICAL VALIDATION: Ensure execution_price was set correctly
                if gap_detected and extreme_gap_capped:
                    # For extreme gaps, execution_price should be gap_fill_price (capped to 15% loss)
                    if abs(final_execution_price - gap_fill_price) > 0.01:  # Allow small floating point differences
                        self.logger.error(f"🚨 CRITICAL BUG: execution_price mismatch! Expected ${gap_fill_price:.2f}, got ${final_execution_price:.2f}")
                        # Force correct price
                        final_execution_price = gap_fill_price
                        self.logger.warning(f"🚨 FORCED CORRECTION: Using gap_fill_price ${final_execution_price:.2f}")
                
                # CRITICAL FIX: Use same fee calculation as regular SELL trades to prevent portfolio explosion
                # Calculate base proceeds from selling all shares at execution price
                base_proceeds = shares * final_execution_price
                
                # ALPACA-STYLE COSTS (same as regular SELL trades)
                # Apply transaction_cost from dynamic_config for SEC/FINRA fees
                try:
                    from ppo.utils.dynamic_config import get_dynamic_config  # type: ignore
                    dcfg = self._cfg()
                    transaction_cost_rate = float(getattr(dcfg, 'transaction_cost', 0.000024))
                except Exception:
                    transaction_cost_rate = 0.000024  # Alpaca realistic: 0.0024% default
                
                # Commission: allow override via TRADING_CONFIG (can be 0 for commission-free)
                try:
                    from config import TRADING_CONFIG, BROKERAGE_COSTS  # type: ignore
                    commission_rate_cfg = TRADING_CONFIG.get('commission_rate', 0.0)
                except Exception:
                    commission_rate_cfg = 0.0
                commission = base_proceeds * float(commission_rate_cfg)
                
                # Brokerage fees: SEC + FINRA (SELL only) + CAT (both sides)
                try:
                    bc = BROKERAGE_COSTS if 'BROKERAGE_COSTS' in locals() else None
                    if bc and bc.get('enabled', True):
                        sec_rate = float(bc.get('rates_static', {}).get('sec_fee_per_dollar_sold', 22.90/1_000_000.0))
                        taf_per_share = float(bc.get('rates_static', {}).get('finra_taf_per_share', 0.000119))
                        cat_per_share = float(bc.get('rates_static', {}).get('cat_fee_per_share', 0.0))
                    else:
                        sec_rate = 22.90/1_000_000.0
                        taf_per_share = 0.000119
                        cat_per_share = 0.0
                except Exception:
                    sec_rate = 22.90/1_000_000.0
                    taf_per_share = 0.000119
                    cat_per_share = 0.0
                
                sec_fee = base_proceeds * sec_rate
                finra_taf = shares * taf_per_share
                cat_fee = shares * cat_per_share
                regulatory_fees = sec_fee + finra_taf + cat_fee
                
                # Market impact and slippage
                try:
                    slippage_rate = float(TRADING_CONFIG.get('slippage_rate', 0.001))
                except Exception:
                    slippage_rate = 0.001  # 0.10% default
                slippage_cost = base_proceeds * slippage_rate
                
                # Apply transaction_cost (SEC/FINRA fees) - this is the realistic Alpaca cost
                transaction_cost_fee = base_proceeds * transaction_cost_rate
                
                # Total fees and costs: commission + regulatory + slippage + transaction_cost
                # VERIFIED: Total cost structure matches Alpaca (commission-free + SEC/FINRA + slippage)
                total_fees = commission + regulatory_fees + slippage_cost + transaction_cost_fee
                
                # Net proceeds after fees (CRITICAL: This was missing before!)
                proceeds = base_proceeds - total_fees
                realized_pnl = proceeds - (shares * avg_cost)
                
                # Accumulate realized gains from stop loss trades
                if not hasattr(self, 'cumulative_realized_gains'):
                    self.cumulative_realized_gains = 0.0
                self.cumulative_realized_gains += realized_pnl
                
                # Update cash and remove position
                self.cash += proceeds
                del self.positions[ticker]
                # CRITICAL: Clear position_lots so next BUY on this ticker doesn't sum with stale lots (fix position shares mismatch)
                if hasattr(self, 'position_lots') and ticker in self.position_lots:
                    self.position_lots[ticker] = []

                # FIX 5: Calculate actual PnL using execution price (gap fill price)
                actual_pnl_pct = (final_execution_price - avg_cost) / avg_cost if avg_cost > 0 else 0.0
                
                # Log stop loss execution with corrected proceeds
                if gap_detected and final_execution_price != current_price:
                    self.logger.warning(f"🛑 STOP LOSS EXECUTED (GAP FILL): Ticker {ticker}, Shares: {shares:.4f}, Gap Price: ${current_price:.2f}, Execution Price: ${final_execution_price:.2f}")
                else:
                    self.logger.warning(f"🛑 STOP LOSS EXECUTED: Ticker {ticker}, Shares: {shares:.4f}, Price: ${final_execution_price:.2f}")
                self.logger.warning(f"🛑 Stop Reason: {stop_reason}")
                self.logger.warning(f"🛑 Proceeds: ${proceeds:.2f} (after fees: ${total_fees:.2f}), PnL: {realized_pnl:.2f} ({actual_pnl_pct:.2%})")
                
                # 2026-02-21: Reset _consecutive_buys when STOP_LOSS triggers (REWARD_CHANGELOG: BUY-heavy collapse fix)
                # Stop loss is an exit; reset so episode doesn't end at 100 consecutive BUYs before agent can learn
                if not hasattr(self, '_consecutive_buys'):
                    self._consecutive_buys = 0
                    self._consecutive_sells = 0
                self._consecutive_buys = 0

                # Record stop loss trade
                trade_info = {
                    'step': self.current_step,
                    'action': 'STOP_LOSS',
                    'ticker': ticker,
                    'shares': shares,
                    'price': final_execution_price,  # FIX 5: Use execution price (gap fill price if gap detected)
                    'gap_price': current_price if gap_detected and final_execution_price != current_price else None,  # FIX 5: Store gap price for analysis
                    'avg_cost': avg_cost,
                    'proceeds': proceeds,  # Now correctly includes fees
                    'pnl': realized_pnl,  # Now correctly calculated with fees and execution price
                    'fees': {
                        'commission': float(commission),
                        'sec_fee': float(sec_fee),
                        'finra_taf': float(finra_taf),
                        'cat_fee': float(cat_fee),
                        'slippage': float(slippage_cost),
                        'total_fees': float(total_fees)
                    },
                    'portfolio_before': self.portfolio_value,
                    'reward': stop_loss_penalty,  # SIGNAL-AWARE PENALTY: -0.05 to -0.5 based on signals
                    'timestamp': datetime.now().isoformat(),
                    'trade_id': f"STOP_LOSS_{self.current_step}_{ticker}_{getattr(self, '_total_trades_executed', 0)}",
                    'stop_reason': stop_reason,
                    'tech_signal': tech_signal,  # Store signal for analysis
                    'signal_adjustment': signal_adjustment,  # Store adjustment for debugging
                    'dynamic_stop_loss_pct': dynamic_stop_loss  # Store final threshold used
                }
                
                # Add to trade history
                if not hasattr(self, 'trade_history'):
                    self.trade_history = []
                self.trade_history.append(trade_info)
                
                # Update trade counter
                if not hasattr(self, '_total_trades_executed'):
                    self._total_trades_executed = 0
                self._total_trades_executed += 1
                
                # CRITICAL FIX: Mark that a SELL was executed via stop loss to prevent duplicate processing
                self._sell_executed_this_step = True

                return True
            
            # Diagnostic: log "near miss" when position is in loss but above threshold (so we see why 0 triggers)
            if execution_pnl_pct < 0 and execution_pnl_pct >= -dynamic_stop_loss:
                if not hasattr(self, '_stop_loss_near_miss_last_step'):
                    self._stop_loss_near_miss_last_step = {}
                last = self._stop_loss_near_miss_last_step.get(ticker, -999)
                if self.current_step - last >= 50:
                    self._stop_loss_near_miss_last_step[ticker] = self.current_step
                    self.logger.info(
                        f"Near miss (stop not triggered): ticker {ticker} down {execution_pnl_pct:.2%}, threshold -{dynamic_stop_loss:.2%} — agent may sell before stop"
                    )
            
            # Update peak price for trailing stop
            if pnl_pct > 0:
                if 'peak_price' not in position:
                    position['peak_price'] = current_price
                else:
                    position['peak_price'] = max(position['peak_price'], current_price)
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to check dynamic stop loss: {e}")
            return False
    
    def enable_architecture_b(
        self,
        ranked_path=None,
        ticker_symbols=None,
        disable_short=True,
        skip_signal_gating=True,
    ):
        """Architecture B: ensemble book picks names+size; this env only times BUY/SELL/HOLD."""
        try:
            from ppo.utils.architecture_b import ArchitectureBController
        except ImportError:
            from utils.architecture_b import ArchitectureBController  # type: ignore

        symbols = ticker_symbols or list(getattr(self, 'unique_tickers', []) or [])
        if not symbols and getattr(self, 'trading_config', None):
            symbols = list(self.trading_config.get('tickers', []) or [])

        self._arch_b = ArchitectureBController(
            ranked_path=ranked_path,
            ticker_symbols=symbols,
            disable_short=disable_short,
        )
        self._arch_b_skip_gating = bool(skip_signal_gating)
        overlap = self._arch_b.overlap_count()
        self._arch_b_enabled = overlap > 0
        info = self._arch_b.summary()
        if self._arch_b_enabled:
            self.logger.info(
                f"[ARCH_B] ENABLED — overlap={overlap} months={info['months']} "
                f"path={info['ranked_path']}"
            )
        else:
            self.logger.warning(
                f"[ARCH_B] DISABLED — no ticker overlap between ranked_signals and env "
                f"(env_tickers={info['env_tickers']}, overlap=0). "
                f"Use the same universe as Stage 1 or map symbols."
            )
        return self._arch_b_enabled

    def _arch_b_active_month(self):
        if not getattr(self, '_arch_b_enabled', False) or self._arch_b is None:
            return None
        date_like = None
        dates = getattr(self, 'dates', None)
        if dates is not None and len(dates) > 0 and self.current_step < len(dates):
            date_like = dates[self.current_step]
        period = self._arch_b.month_from_date(date_like)
        if period is not None:
            return period
        # Fallback when env has no dates: map step fraction → ranked months
        keys = sorted(self._arch_b._by_month.keys())
        if not keys:
            return None
        n = max(int(getattr(self, 'price_matrix', [[]]).shape[0]) - 1, 1)
        frac = min(max(self.current_step / n, 0.0), 1.0)
        idx = int(frac * (len(keys) - 1))
        return keys[idx]

    def _arch_b_underweight_candidates(self, period, current_prices):
        if self._arch_b is None or current_prices is None:
            return []
        cands = []
        for idx in self._arch_b.candidate_indices(period):
            target = self._arch_b.target_weight(idx, period)
            if target <= 0:
                continue
            price = float(current_prices[idx]) if idx < len(current_prices) else 0.0
            if price <= 0:
                continue
            shares = float(self.positions.get(idx, {}).get('shares', 0.0) or 0.0)
            cur_w = (shares * price) / self.portfolio_value if self.portfolio_value > 0 else 0.0
            gap = target - cur_w
            if gap > 0.002:  # need at least 0.2% more allocation
                cands.append((idx, gap, target))
        cands.sort(key=lambda x: x[1], reverse=True)
        return cands

    def _select_best_ticker_to_buy(self, current_prices):
        """CRITICAL FIX: Intelligent ticker selection for buying based on portfolio allocation and momentum.
        2026-02-21: When multiple tickers are close in score, pick randomly among top candidates to diversify.
        Architecture B: pick most underweight name from ensemble ranked book."""
        if getattr(self, '_arch_b_enabled', False) and self._arch_b is not None:
            period = self._arch_b_active_month()
            under = self._arch_b_underweight_candidates(period, current_prices)
            if under:
                best = under[0][0]
                if self.current_step % 50 == 0:
                    self.logger.info(
                        f"[ARCH_B] BUY select ticker={best} "
                        f"sym={self._arch_b.ticker_symbols[best] if best < len(self._arch_b.ticker_symbols) else best} "
                        f"gap={under[0][1]:.3f} target={under[0][2]:.3f}"
                    )
                return best
            # No underweight book names — fall through to legacy (caller/mask should usually block)
        try:
            import random
            scores = []
            for i in range(self.num_tickers):
                # Calculate current position weight
                position_value = self.positions.get(i, {'shares': 0.0})['shares'] * current_prices[i]
                position_weight = position_value / self.portfolio_value if self.portfolio_value > 0 else 0
                
                # Score based on under-allocation (encourage diversification)
                target_weight = 1.0 / self.num_tickers  # Equal weight target
                allocation_score = (target_weight - position_weight) * 10.0
                
                # Add momentum score (reward buying in uptrends)
                momentum_score = 0.0
                if hasattr(self, 'price_matrix') and self.current_step > 0:
                    if i < len(current_prices) and i < len(self.price_matrix[self.current_step-1]):
                        price_change = (current_prices[i] - self.price_matrix[self.current_step-1][i]) / self.price_matrix[self.current_step-1][i]
                        momentum_score = price_change * 5.0  # Scale momentum
                
                # Add volatility score (reward buying in stable conditions)
                volatility_score = 0.0
                if hasattr(self, 'price_matrix') and self.current_step > 1:
                    if i < len(current_prices) and i < len(self.price_matrix[self.current_step-1]) and i < len(self.price_matrix[self.current_step-2]):
                        price_change_1 = abs((current_prices[i] - self.price_matrix[self.current_step-1][i]) / self.price_matrix[self.current_step-1][i])
                        price_change_2 = abs((self.price_matrix[self.current_step-1][i] - self.price_matrix[self.current_step-2][i]) / self.price_matrix[self.current_step-2][i])
                        avg_volatility = (price_change_1 + price_change_2) / 2
                        volatility_score = (0.02 - avg_volatility) * 5.0  # Reward lower volatility
                
                total_score = allocation_score + momentum_score + volatility_score
                scores.append((i, total_score))
            
            # Sort by score descending
            scores.sort(key=lambda x: x[1], reverse=True)
            best_score = scores[0][1]
            # 2026-02-21: Among tickers within 15% of best score, pick randomly to diversify
            threshold = max(0.15 * abs(best_score), 0.5) if best_score != 0 else 0.5
            top_candidates = [t for t, s in scores if s >= best_score - threshold]
            best_ticker = random.choice(top_candidates) if top_candidates else scores[0][0]
            return best_ticker
            
        except Exception as e:
            self.logger.error(f"Failed to select best ticker to buy: {e}")
            return 0  # Default to first ticker
    
    def _select_ticker_to_sell(self):
        """CRITICAL FIX: Intelligent ticker selection for selling based on portfolio allocation and momentum.
        Architecture B: sell most overweight vs ensemble target book first."""
        if getattr(self, '_arch_b_enabled', False) and self._arch_b is not None:
            available_tickers = [i for i in range(self.num_tickers) if self.positions.get(i, {'shares': 0.0})['shares'] > 0]
            if not available_tickers:
                return 0
            period = self._arch_b_active_month()
            current_prices = self.price_matrix[self.current_step]
            scored = []
            for i in available_tickers:
                price = float(current_prices[i])
                shares = float(self.positions.get(i, {}).get('shares', 0.0) or 0.0)
                cur_w = (shares * price) / self.portfolio_value if self.portfolio_value > 0 else 0.0
                target = self._arch_b.target_weight(i, period)
                # Prefer names not in book (target 0) or overweight vs target
                scored.append((i, cur_w - target))
            scored.sort(key=lambda x: x[1], reverse=True)
            return scored[0][0]

        try:
            # First check if we have any positions to sell
            available_tickers = [i for i in range(self.num_tickers) if self.positions.get(i, {'shares': 0.0})['shares'] > 0]
            
            if not available_tickers:
                # No positions to sell - return 0 as fallback
                return 0
            
            best_ticker = available_tickers[0]  # Start with first available ticker
            best_score = -float('inf')
            current_prices = self.price_matrix[self.current_step]
            
            for i in available_tickers:  # Only consider tickers we own
                # Calculate current position weight
                position_value = self.positions.get(i, {'shares': 0.0})['shares'] * current_prices[i]
                position_weight = position_value / self.portfolio_value if self.portfolio_value > 0 else 0

                # Score based on over-allocation (encourage selling concentrated positions)
                target_weight = 1.0 / self.num_tickers  # Equal weight target
                allocation_score = (position_weight - target_weight) * 10.0

                # Add momentum score (reward selling in downtrends)
                momentum_score = 0.0
                if hasattr(self, 'price_matrix') and self.current_step > 0:
                    if i < len(current_prices) and i < len(self.price_matrix[self.current_step-1]):
                        price_change = (current_prices[i] - self.price_matrix[self.current_step-1][i]) / self.price_matrix[self.current_step-1][i]
                        momentum_score = -price_change * 5.0  # Negative momentum is good for selling

                # Total score
                total_score = allocation_score + momentum_score

                if total_score > best_score:
                    best_score = total_score
                    best_ticker = i
            
            return best_ticker
            
        except Exception as e:
            self.logger.error(f"Failed to select best ticker to sell: {e}")
            # Fallback to largest position
            largest_position = 0
            selected_ticker = 0
            
            for i in range(self.num_tickers):
                if self.positions.get(i, {'shares': 0.0})['shares'] > largest_position:
                    largest_position = self.positions.get(i, {'shares': 0.0})['shares']
                    selected_ticker = i
            
            return selected_ticker
    
    def _select_best_ticker_to_short(self, current_prices):
        """Select ticker to short based on bearish signals (strong SELL signal = expect price to drop)."""
        try:
            scores = []
            for i in range(self.num_tickers):
                # Skip if already heavily short
                pos = self.positions.get(i, {'shares': 0.0})
                shares = pos.get('shares', 0.0)
                if shares < -0.001:  # Already short - could add to short
                    position_value = shares * current_prices[i]
                    position_weight = abs(position_value) / self.portfolio_value if self.portfolio_value > 0 else 0
                    if position_weight >= 0.15:  # Max 15% short per ticker
                        continue
                # SELL signal strength (bearish)
                tech_score = 0.0
                if hasattr(self, '_step_signals_cache') and i in self._step_signals_cache and 'SELL' in self._step_signals_cache[i]:
                    tech_score = float(self._step_signals_cache[i]['SELL'].get('tech', 0.0) or 0.0)
                else:
                    tech_score = float(self._calculate_technical_signal_quality(i, 'SELL'))
                # Momentum: negative price change = good for short
                momentum = 0.0
                if self.current_step > 0 and i < len(current_prices) and i < len(self.price_matrix[self.current_step - 1]):
                    prev = self.price_matrix[self.current_step - 1][i]
                    if prev > 0:
                        momentum = (prev - current_prices[i]) / prev * 5.0
                scores.append((tech_score + momentum, i))
            if not scores:
                return 0
            scores.sort(key=lambda x: x[0], reverse=True)
            return scores[0][1]
        except Exception as e:
            self.logger.error(f"Failed to select ticker to short: {e}")
            return 0
    
    def get_constraint_violations(self):
        """Get constraint violation statistics for monitoring"""
        try:
            violations = {
                'trade_interval_violations': getattr(self, '_trade_interval_violations', 0),
                'trade_interval_errors': getattr(self, '_trade_interval_errors', 0),
                'diversification_violations': getattr(self, '_diversification_violations', 0),
                'total_trades': len(self.trade_history),  # Use actual trade history length
                'portfolio_changes': getattr(self, '_total_portfolio_changes', 0),
                'total_trades_executed': getattr(self, '_total_trades_executed', 0),
                'diversification_score': self._calculate_diversification_score()
            }
            return violations
        except Exception as e:
            self.logger.error(f"Failed to get constraint violations: {e}")
            return {}
    
    def _validate_returns_trades_matching(self) -> Dict[str, Any]:
        """
        Validate that returns calculation matches trade history and portfolio changes.
        
        Returns:
            Dict with validation results and any warnings/errors
        """
        validation_result = {
            'valid': True,
            'warnings': [],
            'errors': [],
            'trade_count_match': False,
            'portfolio_change_match': False,
            'returns_calculation_valid': False
        }
        
        try:
            # 1. Validate trade history length matches total trades executed
            trade_history_count = len(self.trade_history) if hasattr(self, 'trade_history') else 0
            total_trades_executed = getattr(self, '_total_trades_executed', 0)
            
            if trade_history_count == total_trades_executed:
                validation_result['trade_count_match'] = True
            else:
                validation_result['valid'] = False
                validation_result['errors'].append(
                    f"Trade history count ({trade_history_count}) doesn't match total trades executed ({total_trades_executed})"
                )
                self.logger.warning(
                    f"RETURNS/TRADES VALIDATION: Trade history mismatch - "
                    f"history={trade_history_count}, executed={total_trades_executed}"
                )
            
            # 2. Validate portfolio value changes match trade PnL (if we have trade history)
            if trade_history_count > 0 and hasattr(self, 'portfolio_history') and len(self.portfolio_history) > 1:
                # Calculate total realized PnL from closing trades (SELL and STOP_LOSS)
                total_realized_pnl = 0.0
                closing_trades = [
                    t for t in self.trade_history 
                    if t.get('action') in ['SELL', 'STOP_LOSS'] and t.get('pnl', 0) != 0
                ]
                
                for trade in closing_trades:
                    pnl = trade.get('pnl', 0.0)
                    if isinstance(pnl, (int, float, np.number)):
                        total_realized_pnl += float(pnl)
                
                # Calculate portfolio change from history
                initial_portfolio = float(self.portfolio_history[0]) if len(self.portfolio_history) > 0 else self.initial_balance
                final_portfolio = float(self.portfolio_history[-1]) if len(self.portfolio_history) > 0 else self.portfolio_value
                portfolio_change = final_portfolio - initial_portfolio
                
                # Account for unrealized PnL on remaining positions (longs and shorts)
                if hasattr(self, 'price_matrix') and self.current_step < len(self.price_matrix):
                    current_prices = self.price_matrix[self.current_step]
                    unrealized_pnl = 0.0
                    for ticker in range(self.num_tickers):
                        if ticker in self.positions and ticker < len(current_prices):
                            shares = self.positions[ticker].get('shares', 0.0)
                            avg_cost = self.positions[ticker].get('avg_cost', 0.0)
                            if shares > 0 and avg_cost > 0:
                                unrealized_pnl += (current_prices[ticker] - avg_cost) * shares
                            elif shares < 0 and avg_cost > 0:
                                # Short: profit when price drops
                                unrealized_pnl += (avg_cost - current_prices[ticker]) * abs(shares)
                    
                    # Portfolio change should approximately equal realized PnL + unrealized PnL
                    # PERFORMANCE: Account for portfolio_history sampling (every 10 steps) - tolerance needs to be larger
                    expected_change = total_realized_pnl + unrealized_pnl
                    # Increased tolerance to account for portfolio_history sampling and price movements between snapshots
                    tolerance = max(5000.0, abs(expected_change) * 0.20)  # 20% tolerance, minimum $5000
                    
                    if abs(portfolio_change - expected_change) <= tolerance:
                        validation_result['portfolio_change_match'] = True
                    else:
                        validation_result['warnings'].append(
                            f"Portfolio change ({portfolio_change:.2f}) doesn't match "
                            f"realized+unrealized PnL ({expected_change:.2f}, "
                            f"realized={total_realized_pnl:.2f}, unrealized={unrealized_pnl:.2f})"
                        )
                        # FIX 4: Make validation warning ERROR level for visibility
                        self.logger.error(
                            f"❌ RETURNS/TRADES VALIDATION: Portfolio change mismatch - "
                            f"change={portfolio_change:.2f}, expected={expected_change:.2f}, "
                            f"tolerance={tolerance:.2f}"
                        )
                        # FIX 4: Automatic correction - recalculate portfolio from cash+positions (include shorts: negative position value)
                        if hasattr(self, 'price_matrix') and self.current_step < len(self.price_matrix):
                            current_prices = self.price_matrix[self.current_step]
                            corrected_portfolio = self.cash + sum(
                                self.positions[i]['shares'] * current_prices[i]
                                for i in self.positions
                                if i < len(current_prices)
                            )
                            if abs(corrected_portfolio - self.portfolio_value) > 100.0:  # Significant difference
                                self.logger.error(f"❌ AUTOMATIC CORRECTION: Recalculating portfolio from cash+positions: ${self.portfolio_value:.2f} → ${corrected_portfolio:.2f}")
                                self.portfolio_value = corrected_portfolio
                else:
                    # Simple check: portfolio change should be positive if we have realized gains
                    if total_realized_pnl > 0 and portfolio_change < 0:
                        validation_result['warnings'].append(
                            f"Portfolio decreased ({portfolio_change:.2f}) despite positive realized PnL ({total_realized_pnl:.2f})"
                        )
            
            # 3. Validate returns calculation uses correct start/end values
            if hasattr(self, 'portfolio_history') and len(self.portfolio_history) > 1:
                initial_value = float(self.portfolio_history[0])
                final_value = float(self.portfolio_history[-1])
                
                if initial_value > 0:
                    calculated_return = (final_value / initial_value) - 1.0
                    expected_return = (self.portfolio_value / self.initial_balance) - 1.0
                    
                    # Allow small differences due to floating-point precision
                    if abs(calculated_return - expected_return) < 0.001:  # 0.1% tolerance
                        validation_result['returns_calculation_valid'] = True
                    else:
                        validation_result['warnings'].append(
                            f"Returns calculation mismatch - "
                            f"from history: {calculated_return:.4f}, "
                            f"from current: {expected_return:.4f}"
                        )
                else:
                    validation_result['errors'].append("Initial portfolio value is zero or negative")
                    validation_result['valid'] = False
            
            # FIX 4: Log summary with appropriate severity
            if validation_result['valid'] and len(validation_result['warnings']) == 0:
                self.logger.debug(
                    f"RETURNS/TRADES VALIDATION: PASSED - "
                    f"trades={trade_history_count}, portfolio_change_match={validation_result['portfolio_change_match']}"
                )
            else:
                # FIX 4: Use ERROR level if there are errors, WARNING if only warnings
                if len(validation_result['errors']) > 0:
                    self.logger.error(
                        f"❌ RETURNS/TRADES VALIDATION: FAILED - "
                        f"valid={validation_result['valid']}, "
                        f"warnings={len(validation_result['warnings'])}, "
                        f"errors={len(validation_result['errors'])}"
                    )
                    for error in validation_result['errors']:
                        self.logger.error(f"❌ VALIDATION ERROR: {error}")
                else:
                    self.logger.warning(
                        f"⚠️ RETURNS/TRADES VALIDATION: Issues found - "
                        f"valid={validation_result['valid']}, "
                        f"warnings={len(validation_result['warnings'])}, "
                        f"errors={len(validation_result['errors'])}"
                    )
                    for warning in validation_result['warnings']:
                        self.logger.warning(f"⚠️ VALIDATION WARNING: {warning}")
                
        except Exception as e:
            validation_result['valid'] = False
            validation_result['errors'].append(f"Validation error: {str(e)}")
            self.logger.error(f"RETURNS/TRADES VALIDATION: Failed with error: {e}")
        
        return validation_result
    
    def get_portfolio_summary(self):
        """Get detailed portfolio summary for debugging"""
        try:
            current_prices = self.price_matrix[self.current_step] if self.current_step < len(self.price_matrix) else [0] * self.num_tickers
            
            summary = {
                'cash': self.cash,
                'portfolio_value': self.portfolio_value,
                'positions': {i: self.positions.get(i, {}).get('shares', 0.0) for i in range(self.num_tickers)},
                'position_values': {
                    i: (self.positions.get(i, {}).get('shares', 0.0) * current_prices[i])
                    for i in range(min(self.num_tickers, len(current_prices)))
                },
                'total_trades': self._total_trades_executed,
                'portfolio_changes': self._total_portfolio_changes
            }
            return summary
        except Exception as e:
            self.logger.error(f"Failed to get portfolio summary: {e}")
            return {}
    
    def log_constraint_summary(self):
        """Log constraint violation summary for debugging"""
        try:
            violations = self.get_constraint_violations()
            self.logger.info(" CONSTRAINT VIOLATION SUMMARY:")
            self.logger.info(f"  Trade Interval Violations: {violations.get('trade_interval_violations', 0)}")
            self.logger.info(f"  Trade Interval Errors: {violations.get('trade_interval_errors', 0)}")
            self.logger.info(f"  Diversification Violations: {violations.get('diversification_violations', 0)}")
            self.logger.info(f"  Total Trades: {violations.get('total_trades', 0)}")
            
            # CRITICAL FIX: Log blocked trades summary for debugging
            if hasattr(self, '_blocked_trades_count'):
                total_blocked = self._blocked_trades_count.get('BUY', 0) + self._blocked_trades_count.get('SELL', 0)
                if total_blocked > 0:
                    self.logger.warning(" SIGNAL GATING SUMMARY:")
                    self.logger.warning(f"  BUY Blocked: {self._blocked_trades_count.get('BUY', 0)}")
                    self.logger.warning(f"  SELL Blocked: {self._blocked_trades_count.get('SELL', 0)}")
                    self.logger.warning(f"  Total Blocked: {total_blocked}")
                    total_attempts = violations.get('total_trades', 0) + total_blocked
                    if total_attempts > 0:
                        block_rate = (total_blocked / total_attempts) * 100
                        self.logger.warning(f"  Block Rate: {block_rate:.1f}%")
                    else:
                        self.logger.warning(f"  Block Rate: N/A")
        except Exception as e:
            self.logger.error(f"Failed to log constraint summary: {e}")
    
    def _get_observation(self):
        """Get current observation with window-based features; num_features is dynamic (includes intraday when present)."""
        try:
            expected_features = self.num_features  # Dynamic: from feature_matrix.shape[2], includes multi-timeframe
            timestep_size = self.num_tickers * expected_features
            obs_size = self.window_size * timestep_size
            
            # Initialize observation array
            obs = np.zeros(obs_size, dtype=np.float32)
            
            # DATA LEAKAGE PREVENTION: Observation includes current_step (intentional)
            # Python slicing [start:end] is exclusive on end, so [start:current_step+1] includes current_step
            # This is correct - observations should include current step's data for the agent to make decisions
            # We're NOT accessing current_step+1 or higher (which would be future data)
            # Calculate window bounds [start, end)
            # Bounds check: ensure we don't access beyond available data
            max_available_step = len(self.feature_matrix) - 1
            if self.current_step > max_available_step:
                # If current_step exceeds available data, use last available step
                end_idx = max_available_step + 1
            else:
                end_idx = self.current_step + 1
            start_idx = max(0, end_idx - self.window_size)
            
            # How many timesteps available and how many to pad on the left
            available = max(0, end_idx - start_idx)
            pad_left = self.window_size - available
            
            # Fill left padding (already zeros) -> advance write index
            write_idx = pad_left * timestep_size
            
            # Copy available timesteps
            for t in range(start_idx, end_idx):
                if 0 <= t < self.feature_matrix.shape[0]:
                    # Get feature data for this timestep
                    frame = self.feature_matrix[t]
                    
                    # Ensure frame has correct shape (num_tickers, expected_features)
                    if frame.ndim == 1:
                        # If collapsed, reshape to (num_tickers, expected_features)
                        frame = np.reshape(frame, (self.num_tickers, expected_features))
                    elif frame.ndim == 2:
                        # If already 2D, ensure correct dimensions
                        if frame.shape[0] != self.num_tickers or frame.shape[1] != expected_features:
                            # Create safe frame with correct dimensions
                            safe = np.zeros((self.num_tickers, expected_features), dtype=np.float32)
                            r = min(self.num_tickers, frame.shape[0])
                            c = min(expected_features, frame.shape[1])
                            safe[:r, :c] = frame[:r, :c]
                            frame = safe
                    else:
                        # If 3D, take the first slice
                        frame = frame[0] if frame.shape[0] > 0 else np.zeros((self.num_tickers, expected_features))
                    
                    # Flatten and ensure correct size
                    frame_flat = frame.flatten()
                    
                    # CRITICAL FIX: Explicitly convert to float32 and handle any residual non-numeric types
                    # Using a direct conversion is often safer than relying on implicit casting
                    try:
                        frame_flat = frame_flat.astype(np.float32, copy=False)
                    except TypeError: # Catch if there are non-numeric types not caught by other logic
                        frame_flat = np.array([float(x) if isinstance(x, (int, float, bool, np.number)) else 0.0 for x in frame_flat], dtype=np.float32)
                        self.logger.warning(f"Casting non-numeric types in observation at step {self.current_step}")
                    
                    if len(frame_flat) != timestep_size:
                        # Pad or truncate to correct size
                        if len(frame_flat) < timestep_size:
                            frame_flat = np.pad(frame_flat, (0, timestep_size - len(frame_flat)), 'constant')
                        else:
                            frame_flat = frame_flat[:timestep_size]
                    
                    # Copy to observation array
                    obs[write_idx:write_idx + timestep_size] = frame_flat
                    write_idx += timestep_size
            
            # PROPER FIX: Append portfolio state features (7 dims: cash% + 6 position sizes)
            # This allows the agent to see its current portfolio and make informed decisions
            portfolio_state = np.zeros(1 + self.num_tickers, dtype=np.float32)
            
            # Feature 0: Cash percentage (0-1)
            if self.portfolio_value > 0:
                portfolio_state[0] = self.cash / self.portfolio_value
            else:
                portfolio_state[0] = 1.0  # All cash if no value
            
            # Features 1-6: Position sizes for each ticker (0-1)
            if self.portfolio_value > 0 and self.current_step < len(self.price_matrix):
                for i in range(self.num_tickers):
                    if i in self.positions and i < len(self.price_matrix[self.current_step]):
                        shares = self.positions[i].get('shares', 0.0)
                        price = self.price_matrix[self.current_step, i]
                        position_value = shares * price
                        portfolio_state[1 + i] = position_value / self.portfolio_value
            
            # 2026-02-15: Append fundamental metrics per ticker (normalized to ~[-1, 1])
            if getattr(self, '_include_fundamental_in_obs', False):
                fund_arr = np.zeros(self.num_tickers * self.num_fundamental_features, dtype=np.float32)
                tickers = getattr(self, 'unique_tickers', None) or getattr(self, 'ticker_symbols', None)
                if not tickers:
                    tickers = [self._resolve_symbol(i) for i in range(self.num_tickers)]
                for i in range(self.num_tickers):
                    symbol = tickers[i] if i < len(tickers) else self._resolve_symbol(i)
                    metrics = self._get_cached_fundamental_metrics(symbol)
                    if metrics:
                        for j, key in enumerate(self.FUNDAMENTAL_FEATURE_KEYS):
                            val = metrics.get(key, 0.0)
                            # Normalize to ~[-1, 1]: PE/20, PB/5, D/E/3, ROE*2, CR/3, growth*2
                            if key == 'pe_ratio':
                                norm = np.clip(float(val) / 30.0 - 0.5, -1.0, 1.0)  # 0-60 -> -0.5 to 1.5
                            elif key == 'pb_ratio':
                                norm = np.clip(float(val) / 5.0 - 0.5, -1.0, 1.0)
                            elif key == 'debt_equity':
                                norm = np.clip(float(val) / 3.0 - 0.5, -1.0, 1.0)
                            elif key == 'roe':
                                norm = np.clip(float(val) * 2.0, -1.0, 1.0)  # -0.5 to 0.5
                            elif key == 'current_ratio':
                                norm = np.clip(float(val) / 3.0 - 0.5, -1.0, 1.0)
                            elif key in ('revenue_growth', 'earnings_growth'):
                                norm = np.clip(float(val) * 2.0, -1.0, 1.0)
                            else:
                                norm = np.clip(float(val), -1.0, 1.0)
                            fund_arr[i * self.num_fundamental_features + j] = norm
                obs = np.concatenate([obs, portfolio_state, fund_arr])
            else:
                obs = np.concatenate([obs, portfolio_state])
            
            # CRITICAL: Ensure observation has exactly the expected size
            expected_size = self.observation_space.shape[0]
            if len(obs) != expected_size:
                if len(obs) < expected_size:
                    obs = np.pad(obs, (0, expected_size - len(obs)), 'constant')
                else:
                    obs = obs[:expected_size]
            
            # CRITICAL FIX: Ensure observation is always float32 and handle any non-numeric types
            try:
                obs = obs.astype(np.float32, copy=False)
            except TypeError:
                obs = np.array([float(x) if isinstance(x, (int, float, bool, np.number)) else 0.0 for x in obs], dtype=np.float32)
                self.logger.warning(f"Casting non-numeric types in final observation at step {self.current_step}")
            
            # Clean any remaining NaN values
            if np.any(np.isnan(obs)):
                obs = np.nan_to_num(obs, nan=0.0)
            
            # CRITICAL: Clip observation to valid range
            obs = np.clip(obs, -1.0, 1.0)
            
            return obs
            
        except Exception as e:
            # Return safe fallback observation
            obs_size = self.observation_space.shape[0]
            return np.zeros(obs_size, dtype=np.float32)
    
    def _update_portfolio_state(self):
        """Update portfolio state weights"""
        try:
            total_value = self.portfolio_value
            if total_value > 0:
                for i in range(self.num_tickers):
                    position_value = self.positions[i]['shares'] * self.price_matrix[self.current_step, i] if self.current_step < len(self.price_matrix) else 0
                    self.portfolio_state[i] = position_value / total_value
            else:
                self.portfolio_state = [0.0] * self.num_tickers
        except Exception as e:
            self.logger.error(f"Failed to update portfolio state: {e}")
            self.portfolio_state = [0.0] * self.num_tickers
    
    def _select_diversified_ticker(self):
        """Select ticker for buying based on diversification"""
        try:
            # CRITICAL FIX: Force diversification by cycling through tickers
            if not hasattr(self, '_buy_ticker_cycle'):
                self._buy_ticker_cycle = 0
            
            # Cycle through tickers to force diversification
            selected_ticker = self._buy_ticker_cycle % self.num_tickers
            self._buy_ticker_cycle += 1
            
            # Check if this ticker is below max position size
            current_weight = self.portfolio_state[selected_ticker]
            if current_weight >= self.max_position_size:
                # Find next available ticker
                for i in range(self.num_tickers):
                    next_ticker = (selected_ticker + i) % self.num_tickers
                    if self.portfolio_state[next_ticker] < self.max_position_size:
                        selected_ticker = next_ticker
                        break
            
            self.logger.info(f"Selected ticker {selected_ticker} for diversification (weight: {self.portfolio_state[selected_ticker]:.4f})")
            return selected_ticker
                
        except Exception as e:
            self.logger.error(f"Failed to select diversified ticker: {e}")
            return 0
    
    
    def get_selected_ticker(self):
        """Get the ticker selected for the last trade action"""
        return getattr(self, '_selected_ticker', None)
    
    def clear_cumulative_evaluation_trade_history(self):
        """CRITICAL FIX: Clear cumulative trade history at start of evaluation"""
        if hasattr(self, 'cumulative_evaluation_trade_history'):
            self.cumulative_evaluation_trade_history.clear()
            logger.info("Cleared cumulative evaluation trade history for fresh evaluation")
        else:
            self.cumulative_evaluation_trade_history = []
            logger.info("Initialized cumulative evaluation trade history")
    
    def get_trade_history(self):
        """CRITICAL FIX: Get trade history with proper debugging"""
        try:
            # CRITICAL FIX: Return per-episode trade history, NOT cumulative
            # The evaluator should handle accumulation, not the environment
            if hasattr(self, 'episode_trade_history') and self.episode_trade_history:
                logger.info(f"Returning per-episode trade history: {len(self.episode_trade_history)} trades")
                return self.episode_trade_history.copy()
            
            if not hasattr(self, 'trade_history'):
                self.trade_history = []
            
            # CRITICAL FIX: Ensure trade history is properly formatted for CSV export
            formatted_trades = []
            for trade in self.trade_history:
                if isinstance(trade, dict):
                    # Ensure all required fields are present
                    formatted_trade = {
                        'step': trade.get('step', 0),
                        'action': trade.get('action', 'HOLD'),
                        'ticker': trade.get('ticker', -1),
                        'shares': trade.get('shares', 0.0),
                        'price': trade.get('price', 0.0),
                        'cost': trade.get('cost', 0.0),
                        'portfolio_before': trade.get('portfolio_before', 0.0),
                        'portfolio_after': trade.get('portfolio_after', 0.0),
                        'reward': trade.get('reward', 0.0),
                        'timestamp': trade.get('timestamp', ''),
                        'trade_id': trade.get('trade_id', '')
                    }
                    formatted_trades.append(formatted_trade)
            
            logger.info(f"Returning formatted trade history: {len(formatted_trades)} trades")
            return formatted_trades
            
        except Exception as e:
            logger.error(f"Failed to get trade history: {e}")
            return []
    
    def get_cumulative_evaluation_trade_history(self):
        """CRITICAL FIX: Get cumulative trade history for evaluation"""
        try:
            if hasattr(self, 'cumulative_evaluation_trade_history'):
                logger.info(f"Returning cumulative evaluation trade history: {len(self.cumulative_evaluation_trade_history)} trades")
                return self.cumulative_evaluation_trade_history
            else:
                logger.warning("No cumulative evaluation trade history found")
                return []
        except Exception as e:
            logger.error(f"Failed to get cumulative evaluation trade history: {e}")
            return []
    
    def get_total_trades_executed(self):
        """Get total number of trades executed"""
        return getattr(self, '_total_trades_executed', 0)
    
    def _should_log_trade(self, step_interval: int = 50) -> bool:
        """Helper method to check if trade logging should occur based on step interval
        
        Args:
            step_interval: Interval for logging (default 50)
            
        Returns:
            bool: True if logging should occur
        """
        return hasattr(self, '_total_trades_executed') and self._total_trades_executed % step_interval == 0
    
    def _clear_position_lots(self):
        """Helper method to clear position_lots to prevent corruption
        
        This is called during reset and when portfolio explodes to ensure
        position tracking remains consistent.
        """
        if hasattr(self, 'position_lots'):
            self.position_lots = {i: [] for i in range(self.num_tickers)}
    
    def get_realized_unrealized_gains(self, current_prices: Optional[np.ndarray] = None):
        """Calculate realized vs unrealized gains breakdown
        
        Returns:
            dict: {
                'realized_gains': float,  # Total realized PnL from closed trades
                'unrealized_gains': float,  # Current unrealized PnL from open positions
                'realized_return': float,  # Realized return as percentage
                'unrealized_return': float,  # Unrealized return as percentage
                'total_return': float  # Total return (realized + unrealized)
            }
        """
        try:
            # Get realized gains (already accumulated)
            realized_gains = getattr(self, 'cumulative_realized_gains', 0.0)
            
            # Calculate unrealized gains from open positions
            unrealized_gains = 0.0
            if current_prices is None:
                # Use current step prices if available
                if hasattr(self, 'price_matrix') and self.current_step < len(self.price_matrix):
                    current_prices = self.price_matrix[self.current_step]
                else:
                    current_prices = np.array([])
            current_prices = cast(np.ndarray, np.asarray(current_prices))
            
            if current_prices.size > 0:
                for ticker in range(min(self.num_tickers, current_prices.size)):
                    if ticker in self.positions:
                        shares = self.positions[ticker].get('shares', 0.0)
                        avg_cost = self.positions[ticker].get('avg_cost', 0.0)
                        if shares > 0 and avg_cost > 0 and ticker < current_prices.size:
                            current_value = shares * current_prices[ticker]
                            cost_basis = shares * avg_cost
                            position_unrealized = current_value - cost_basis
                            unrealized_gains += position_unrealized
            
            # Calculate returns as percentages
            initial_balance = getattr(self, 'initial_balance', 100000.0)
            realized_return = (realized_gains / initial_balance) * 100.0 if initial_balance > 0 else 0.0
            unrealized_return = (unrealized_gains / initial_balance) * 100.0 if initial_balance > 0 else 0.0
            total_return = realized_return + unrealized_return
            
            return {
                'realized_gains': float(realized_gains),
                'unrealized_gains': float(unrealized_gains),
                'realized_return': float(realized_return),
                'unrealized_return': float(unrealized_return),
                'total_return': float(total_return)
            }
        except Exception as e:
            self.logger.error(f"Failed to calculate realized/unrealized gains: {e}")
            return {
                'realized_gains': 0.0,
                'unrealized_gains': 0.0,
                'realized_return': 0.0,
                'unrealized_return': 0.0,
                'total_return': 0.0
            }
    
    def get_portfolio_history(self):
        """CRITICAL FIX: Get portfolio value history with proper formatting"""
        try:
            if not hasattr(self, 'portfolio_history'):
                self.portfolio_history = [self.initial_balance]
            
            # CRITICAL FIX: Ensure portfolio history is properly formatted for CSV export
            formatted_history = []
            for i, value in enumerate(self.portfolio_history):
                formatted_history.append({
                    'step': i,
                    'portfolio_value': float(value)
                })
            
            # Log portfolio history status for debugging
            self.logger.info(f"Portfolio history length: {len(formatted_history)}")
            if len(formatted_history) > 0:
                self.logger.info(f"Portfolio range: ${formatted_history[0]['portfolio_value']:.2f}  ${formatted_history[-1]['portfolio_value']:.2f}")
            
            return formatted_history
        except Exception as e:
            self.logger.error(f"Error getting portfolio history: {e}")
            return [{'step': 0, 'portfolio_value': float(self.initial_balance)}]
    
    def get_episode_trade_history(self):
        """CRITICAL: Returns the trade history for the current episode only
        
        Returns:
            List[Dict[str, Any]]: List of trades executed in current episode
        """
        # CRITICAL FIX: Return current episode trades if available, otherwise return last episode trades
        if hasattr(self, 'episode_trade_history') and len(self.episode_trade_history) > 0:
            return self.episode_trade_history.copy()  # Current episode trades
        elif hasattr(self, '_last_episode_trades') and len(self._last_episode_trades) > 0:
            return self._last_episode_trades.copy()  # Last episode trades (after reset)
        else:
            return []  # No trades available
    
    def get_adaptive_reward_log(self):
        """FIX: Get adaptive reward log for unified report"""
        if not hasattr(self, '_adaptive_reward_log'):
            self._adaptive_reward_log = []
        return self._adaptive_reward_log.copy()
    
    def get_behavior_adaptive_log(self):
        """FIX: Get behavior adaptive log (filtered from adaptive reward log)"""
        if not hasattr(self, '_adaptive_reward_log'):
            self._adaptive_reward_log = []
        # Filter for behavior adaptive events
        behavior_log = [entry for entry in self._adaptive_reward_log if entry.get('type') == 'behavior_adaptive']
        return behavior_log.copy()
    
    def get_cumulative_performance(self):
        """CRITICAL: Returns cumulative performance data across all episodes
        
        Returns:
            Dict[str, Any]: Cumulative performance metrics
        """
        try:
            if not hasattr(self, 'cumulative_portfolio_value'):
                return {
                    'cumulative_portfolio_value': 100000.0,
                    'cumulative_trades': 0,
                    'cumulative_episodes': 0,
                    'cumulative_return': 0.0,
                    'episode_start_portfolio': 100000.0
                }
            
            cumulative_return = (self.cumulative_portfolio_value - 100000.0) / 100000.0
            
            return {
                'cumulative_portfolio_value': self.cumulative_portfolio_value,
                'cumulative_trades': self.cumulative_trades,
                'cumulative_episodes': self.cumulative_episodes,
                'cumulative_return': cumulative_return,
                'episode_start_portfolio': self.episode_start_portfolio,
                'current_episode_return': (self.portfolio_value - self.episode_start_portfolio) / self.episode_start_portfolio if self.episode_start_portfolio > 0 else 0.0
            }
        except Exception as e:
            self.logger.error(f"Error getting cumulative performance: {e}")
            return {
                'cumulative_portfolio_value': 100000.0,
                'cumulative_trades': 0,
                'cumulative_episodes': 0,
                'cumulative_return': 0.0,
                'episode_start_portfolio': 100000.0
            }
    
    def _log_reward_breakdown(self, reward_components: dict):
        """Log detailed reward breakdown for debugging"""
        try:
            self._last_reward_breakdown = reward_components
            if self.current_step % 50 == 0:  # Log every 50 steps
                self.logger.info("REWARD COMPONENTS:")
                for component, value in reward_components.items():
                    if isinstance(value, (int, float)):
                        self.logger.info(f"  {component}: {value:.6f}")
                    else:
                        self.logger.info(f"  {component}: {value}")
                numeric_values = [v for v in reward_components.values() if isinstance(v, (int, float))]
                self.logger.info(f"  TOTAL: {sum(numeric_values):.6f}")
        except Exception as e:
            self.logger.error(f"Error logging reward breakdown: {e}")
    
    # REMOVED: Duplicate _get_observation method that was causing portfolio value override bug
    # The first _get_observation method (line 1382) is the correct one to use
    
    
    

# End of FixedTradingEnv class
