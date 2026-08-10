"""
Real-time News Stream Integration
Based on TG4.1 AlpacaNewsStream implementation
"""

import os
import time
import json
import threading
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
from collections import deque
import requests

logger = logging.getLogger(__name__)

class AlpacaNewsStream:
    """Real-time news stream integration like TG4.1"""
    
    def __init__(self, api_key: str, secret_key: str, base_url: str = "https://data.alpaca.markets"):
        self.api_key = api_key
        self.secret_key = secret_key
        # Use Alpaca Data API for news
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({
            'APCA-API-KEY-ID': api_key,
            'APCA-API-SECRET-KEY': secret_key
        })
        
        # News storage
        self.news_buffer = deque(maxlen=1000)
        self.sentiment_scores = {}
        self.ticker_news = {}
        
        # Stream control
        self.is_running = False
        self.stream_thread = None
        self.callbacks = []
        
        # Configuration
        self.update_interval = 60  # seconds
        self.max_news_age = 3600  # 1 hour
        self.sentiment_threshold = 0.1
        
        logger.info(" AlpacaNewsStream initialized")
    
    def start_stream(self):
        """Start real-time news stream"""
        if self.is_running:
            logger.warning("  News stream already running")
            return
        
        self.is_running = True
        self.stream_thread = threading.Thread(target=self._stream_loop, daemon=True)
        self.stream_thread.start()
        logger.info(" News stream started")
    
    def start_monitoring(self):
        """Alias for start_stream - for compatibility with existing code"""
        self.start_stream()
    
    def stop_stream(self):
        """Stop real-time news stream"""
        self.is_running = False
        if self.stream_thread:
            self.stream_thread.join(timeout=5)
        logger.info(" News stream stopped")
    
    def _stream_loop(self):
        """Main streaming loop"""
        while self.is_running:
            try:
                self._fetch_latest_news()
                self._process_news_sentiment()
                self._notify_callbacks()
                time.sleep(self.update_interval)
            except Exception as e:
                logger.error(f" News stream error: {e}")
                time.sleep(10)  # Wait before retry
    
    def _fetch_latest_news(self):
        """Fetch latest news from Alpaca"""
        try:
            # Get news for tracked tickers
            tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META']
            
            for ticker in tickers:
                # Alpaca Data API v1beta1 endpoint
                url = f"{self.base_url}/v1beta1/news"
                params = {
                    'symbols': ticker,
                    'limit': 10
                }

                response = self.session.get(url, params=params, timeout=15)
                if response.ok:
                    payload = response.json() or {}
                    items = payload.get('news') if isinstance(payload, dict) else payload
                    if items:
                        self._process_news_data(ticker, items)
                else:
                    logger.warning(f"  Failed to fetch news for {ticker}: {response.status_code}")
                    
        except Exception as e:
            logger.error(f" News fetch error: {e}")
    
    def _process_news_data(self, ticker: str, news_data: List[Dict]):
        """Process and store news data"""
        for news_item in news_data:
            news_entry = {
                'ticker': ticker,
                'headline': news_item.get('headline', news_item.get('title', '')),
                'summary': news_item.get('summary', news_item.get('description', '')),
                'author': news_item.get('author', ''),
                'published_at': news_item.get('published_at', news_item.get('created_at', '')),
                'url': news_item.get('url', news_item.get('link', '')),
                'timestamp': datetime.now().isoformat()
            }
            
            self.news_buffer.append(news_entry)
            
            # Store by ticker
            if ticker not in self.ticker_news:
                self.ticker_news[ticker] = []
            self.ticker_news[ticker].append(news_entry)
    
    def _process_news_sentiment(self):
        """Process sentiment analysis for news"""
        # CRITICAL FIX: Convert deque to list before slicing
        news_items = list(self.news_buffer)
        for news_item in news_items[-10:]:  # Process last 10 items
            sentiment = self._analyze_sentiment(news_item['headline'] + ' ' + news_item['summary'])
            ticker = news_item['ticker']
            
            if ticker not in self.sentiment_scores:
                self.sentiment_scores[ticker] = deque(maxlen=50)
            
            self.sentiment_scores[ticker].append({
                'sentiment': sentiment,
                'timestamp': news_item['timestamp'],
                'headline': news_item['headline']
            })
    
    def _analyze_sentiment(self, text: str) -> float:
        """Simple sentiment analysis (enhanced version would use NLP)"""
        text_lower = text.lower()
        
        # Positive keywords
        positive_words = ['up', 'rise', 'gain', 'positive', 'growth', 'profit', 'beat', 'exceed', 'strong']
        # Negative keywords  
        negative_words = ['down', 'fall', 'loss', 'negative', 'decline', 'miss', 'weak', 'drop', 'crash']
        
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count == 0 and negative_count == 0:
            return 0.0
        
        sentiment = (positive_count - negative_count) / (positive_count + negative_count)
        return max(-1.0, min(1.0, sentiment))  # Clamp to [-1, 1]
    
    def get_ticker_sentiment(self, ticker: str) -> float:
        """Get current sentiment for a ticker"""
        if ticker not in self.sentiment_scores or not self.sentiment_scores[ticker]:
            return 0.0
        
        # CRITICAL FIX: Convert deque to list before slicing
        ticker_sentiments = list(self.sentiment_scores[ticker])
        recent_sentiments = [item['sentiment'] for item in ticker_sentiments[-5:]]
        return sum(recent_sentiments) / len(recent_sentiments)
    
    def get_market_sentiment(self) -> float:
        """Get overall market sentiment"""
        if not self.sentiment_scores:
            return 0.0
        
        all_sentiments = []
        for ticker_sentiments in self.sentiment_scores.values():
            if ticker_sentiments:
                # CRITICAL FIX: Convert deque to list before slicing
                ticker_list = list(ticker_sentiments)
                recent = [item['sentiment'] for item in ticker_list[-3:]]
                all_sentiments.extend(recent)
        
        return sum(all_sentiments) / len(all_sentiments) if all_sentiments else 0.0
    
    def add_callback(self, callback: Callable):
        """Add callback for news updates"""
        self.callbacks.append(callback)
    
    def _notify_callbacks(self):
        """Notify all registered callbacks"""
        market_sentiment = self.get_market_sentiment()
        
        for callback in self.callbacks:
            try:
                # CRITICAL FIX: Convert deque to list before slicing
                recent_news = list(self.news_buffer)[-5:]  # Last 5 news items
                callback({
                    'market_sentiment': market_sentiment,
                    'ticker_sentiments': {ticker: self.get_ticker_sentiment(ticker) 
                                        for ticker in self.sentiment_scores.keys()},
                    'recent_news': recent_news
                })
            except Exception as e:
                logger.error(f" Callback error: {e}")
    
    def get_recent_news(self, ticker: str = None, limit: int = 10) -> List[Dict]:
        """Get recent news items"""
        if ticker:
            ticker_news = self.ticker_news.get(ticker, [])
            return ticker_news[-limit:] if ticker_news else []
        else:
            # CRITICAL FIX: Convert deque to list before slicing
            news_items = list(self.news_buffer)
            return news_items[-limit:] if news_items else []
    
    def cleanup_old_news(self):
        """Clean up old news items"""
        cutoff_time = datetime.now() - timedelta(hours=1)
        
        # Clean news buffer
        self.news_buffer = deque(
            [item for item in self.news_buffer 
             if datetime.fromisoformat(item['timestamp']) > cutoff_time],
            maxlen=1000
        )
        
        # Clean sentiment scores
        for ticker in list(self.sentiment_scores.keys()):
            self.sentiment_scores[ticker] = deque(
                [item for item in self.sentiment_scores[ticker]
                 if datetime.fromisoformat(item['timestamp']) > cutoff_time],
                maxlen=50
            )
        
        logger.info(" Cleaned up old news items")
