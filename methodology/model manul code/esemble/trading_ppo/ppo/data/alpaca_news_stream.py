"""
Alpaca News Stream Integration
Real-time news data processing and sentiment analysis
"""

import os
import time
import logging
import threading
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import requests
import json
from collections import deque

logger = logging.getLogger(__name__)

class AlpacaNewsStream:
    """
    Real-time news stream from Alpaca API
    Integrates with sentiment analysis for trading signals
    """
    
    def __init__(self, api_key: str = None, secret_key: str = None):
        self.api_key = api_key or os.environ.get('ALPACA_API_KEY', '')
        self.secret_key = secret_key or os.environ.get('ALPACA_SECRET_KEY', '')
        
        # News storage
        self.news_buffer = deque(maxlen=1000)
        self.sentiment_scores = deque(maxlen=1000)
        self.ticker_news = {}
        
        # Stream control
        self.is_running = False
        self.stream_thread = None
        self.last_news_time = datetime.now()
        
        # Configuration
        self.update_interval = 30  # seconds
        self.max_news_age = 3600  # 1 hour
        
        logger.info("AlpacaNewsStream initialized")
    
    def start_stream(self):
        """Start the news stream in background thread"""
        if self.is_running:
            logger.warning("News stream already running")
            return
        
        self.is_running = True
        self.stream_thread = threading.Thread(target=self._stream_news, daemon=True)
        self.stream_thread.start()
        logger.info("News stream started")
    
    def stop_stream(self):
        """Stop the news stream"""
        self.is_running = False
        if self.stream_thread:
            self.stream_thread.join(timeout=5)
        logger.info("News stream stopped")
    
    def _stream_news(self):
        """Background thread for news streaming"""
        while self.is_running:
            try:
                self._fetch_latest_news()
                time.sleep(self.update_interval)
            except Exception as e:
                logger.error(f"Error in news stream: {e}")
                time.sleep(60)  # Wait before retry
    
    def _fetch_latest_news(self):
        """Fetch latest news from Alpaca API"""
        if not self.api_key or not self.secret_key:
            logger.warning("Alpaca credentials not configured")
            return
        
        try:
            # Alpaca news API endpoint
            url = "https://data.alpaca.markets/v1beta1/news"
            headers = {
                "APCA-API-KEY-ID": self.api_key,
                "APCA-API-SECRET-KEY": self.secret_key
            }
            
            # Get news from last hour
            since = datetime.now() - timedelta(hours=1)
            params = {
                "start": since.isoformat(),
                "limit": 50
            }
            
            response = requests.get(url, headers=headers, params=params, timeout=10)
            
            if response.status_code == 200:
                news_data = response.json()
                self._process_news_data(news_data)
            else:
                logger.warning(f"News API error: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Error fetching news: {e}")
    
    def _process_news_data(self, news_data: Dict[str, Any]):
        """Process and store news data"""
        if 'news' not in news_data:
            return
        
        for news_item in news_data['news']:
            try:
                # Extract news information
                news_info = {
                    'id': news_item.get('id'),
                    'headline': news_item.get('headline', ''),
                    'summary': news_item.get('summary', ''),
                    'content': news_item.get('content', ''),
                    'author': news_item.get('author', ''),
                    'created_at': news_item.get('created_at'),
                    'updated_at': news_item.get('updated_at'),
                    'url': news_item.get('url', ''),
                    'symbols': news_item.get('symbols', []),
                    'source': news_item.get('source', '')
                }
                
                # Add to buffer
                self.news_buffer.append(news_info)
                
                # Update ticker-specific news
                for symbol in news_info['symbols']:
                    if symbol not in self.ticker_news:
                        self.ticker_news[symbol] = []
                    self.ticker_news[symbol].append(news_info)
                
                self.last_news_time = datetime.now()
                
            except Exception as e:
                logger.error(f"Error processing news item: {e}")
    
    def get_recent_news(self, symbol: str = None, hours: int = 1) -> List[Dict]:
        """Get recent news for a specific symbol or all news"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        if symbol:
            # Get symbol-specific news
            symbol_news = self.ticker_news.get(symbol, [])
            return [
                news for news in symbol_news 
                if self._parse_time(news.get('created_at', '')) > cutoff_time
            ]
        else:
            # Get all recent news
            return [
                news for news in self.news_buffer 
                if self._parse_time(news.get('created_at', '')) > cutoff_time
            ]
    
    def _parse_time(self, time_str: str) -> datetime:
        """Parse ISO time string to datetime"""
        try:
            return datetime.fromisoformat(time_str.replace('Z', '+00:00'))
        except:
            return datetime.now()
    
    def get_market_sentiment(self, symbol: str = None) -> Dict[str, float]:
        """Calculate market sentiment from recent news"""
        recent_news = self.get_recent_news(symbol, hours=24)
        
        if not recent_news:
            return {'sentiment': 0.0, 'confidence': 0.0}
        
        # Simple sentiment calculation (can be enhanced)
        positive_keywords = ['positive', 'growth', 'profit', 'gain', 'up', 'bullish', 'strong']
        negative_keywords = ['negative', 'loss', 'decline', 'down', 'bearish', 'weak', 'risk']
        
        positive_count = 0
        negative_count = 0
        total_words = 0
        
        for news in recent_news:
            text = f"{news.get('headline', '')} {news.get('summary', '')}".lower()
            words = text.split()
            total_words += len(words)
            
            for word in words:
                if word in positive_keywords:
                    positive_count += 1
                elif word in negative_keywords:
                    negative_count += 1
        
        if total_words == 0:
            return {'sentiment': 0.0, 'confidence': 0.0}
        
        sentiment = (positive_count - negative_count) / total_words
        confidence = min(1.0, len(recent_news) / 10.0)  # More news = higher confidence
        
        return {
            'sentiment': sentiment,
            'confidence': confidence,
            'news_count': len(recent_news)
        }
    
    def get_news_summary(self) -> Dict[str, Any]:
        """Get summary of current news state"""
        return {
            'total_news': len(self.news_buffer),
            'active_symbols': len(self.ticker_news),
            'last_update': self.last_news_time.isoformat(),
            'is_streaming': self.is_running
        }
    
    def cleanup_old_news(self):
        """Remove old news from buffers"""
        cutoff_time = datetime.now() - timedelta(seconds=self.max_news_age)
        
        # Clean main buffer
        self.news_buffer = deque(
            [news for news in self.news_buffer 
             if self._parse_time(news.get('created_at', '')) > cutoff_time],
            maxlen=1000
        )
        
        # Clean ticker-specific news
        for symbol in list(self.ticker_news.keys()):
            self.ticker_news[symbol] = [
                news for news in self.ticker_news[symbol]
                if self._parse_time(news.get('created_at', '')) > cutoff_time
            ]
            
            # Remove empty ticker entries
            if not self.ticker_news[symbol]:
                del self.ticker_news[symbol]
