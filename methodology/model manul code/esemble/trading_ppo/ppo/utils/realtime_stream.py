"""
Real-time data stream functionality for the PPO Trading Agent
Extracted from TG4.1.py - AlpacaNewsStream class and related utilities
"""

import os
import sys
import warnings
import logging
import pickle
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dotenv import load_dotenv
import numpy as np
import pandas as pd

# REAL-TIME DATA INTEGRATION
import threading
import json
import time
from collections import deque

# CRITICAL FIX: Add missing imports (gymnasium)
import gymnasium as gym
from gymnasium import spaces

# CRITICAL FIX: Set API keys for Alpaca paper trading
os.environ['ALPACA_API_KEY'] = 'PK3GN5XTAZEVXHLRO0TD'
os.environ['ALPACA_SECRET_KEY'] = 'UKsuudqVsbVkj6YhU6wFtcyZWh76voXXq401BwYY'
os.environ['ALPACA_BASE_URL'] = 'https://paper-api.alpaca.markets'

class AlpacaNewsStream:
    """Real-time news stream from Alpaca for sentiment analysis - PICKLE-SAFE VERSION"""
    
    def __init__(self, api_key, secret_key, symbols=None):
        self.api_key = api_key
        self.secret_key = secret_key
        self.symbols = symbols or []
        self.news_buffer = deque(maxlen=1000)  # Keep last 1000 news items
        self.sentiment_cache = {}
        self.is_running = False
        self.ws = None
        self.ws_thread = None
        
        # PICKLE-SAFE: Add __getstate__ and __setstate__ methods
        self._pickle_safe = True
    
    def __getstate__(self):
        """Make the object pickle-safe by excluding non-serializable components"""
        state = self.__dict__.copy()
        # Remove non-serializable components
        state['ws'] = None
        state['ws_thread'] = None
        state['is_running'] = False
        return state
    
    def __setstate__(self, state):
        """Restore state after unpickling"""
        self.__dict__.update(state)
        self.ws = None
        self.ws_thread = None
        self.is_running = False
    
    def start_stream(self):
        """Start the WebSocket stream"""
        try:
            import websocket
            
            ws_url = "wss://stream.data.alpaca.markets/v1beta1/news"
            
            def on_message(ws, message):
                try:
                    data = json.loads(message)
                    # FIXED: Handle different message types properly
                    if isinstance(data, dict):
                        if data.get('T') == 'n':  # News message
                            self._process_news(data)
                        elif data.get('T') == 'success':  # Subscription confirmation
                            print("[REALTIME] Successfully subscribed to news feed")
                        elif data.get('T') == 'error':  # Error message
                            print(f"[REALTIME] WebSocket error: {data.get('msg', 'Unknown error')}")
                    elif isinstance(data, list):
                        # Handle array messages
                        for item in data:
                            if isinstance(item, dict) and item.get('T') == 'n':
                                self._process_news(item)
                except Exception as e:
                    print(f"[REALTIME] Message parsing error: {e}")
                    # Don't stop the stream for parsing errors
            
            def on_error(ws, error):
                print(f"[REALTIME] WebSocket error: {error}")
            
            def on_close(ws, close_status_code, close_msg):
                print("[REALTIME] WebSocket connection closed")
                self.is_running = False
            
            def on_open(ws):
                print("[REALTIME] WebSocket connection opened")
                # Subscribe to news
                subscribe_msg = {"action": "subscribe", "news": ["*"]}
                ws.send(json.dumps(subscribe_msg))
                print("[REALTIME] Subscribed to news feed")
                self.is_running = True
            
            # Create WebSocket connection
            self.ws = websocket.WebSocketApp(
                ws_url,
                header={
                    "APCA-API-KEY-ID": self.api_key,
                    "APCA-API-SECRET-KEY": self.secret_key
                },
                on_open=on_open,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close
            )
            
            # Run WebSocket in a thread
            self.ws_thread = threading.Thread(target=self.ws.run_forever)
            self.ws_thread.daemon = True
            self.ws_thread.start()
            
            print("[REALTIME] News stream started successfully")
            return True
            
        except ImportError:
            print("[REALTIME] websocket-client not installed. Install with: pip install websocket-client")
            return False
        except Exception as e:
            print(f"[REALTIME] Failed to start stream: {e}")
            return False
    
    def _process_news(self, news_data):
        """Process incoming news data"""
        try:
            # Extract relevant information
            headline = news_data.get('headline', '')
            symbols = news_data.get('symbols', [])
            created_at = news_data.get('created_at', '')
            
            # Calculate sentiment
            sentiment = self._analyze_sentiment(headline)
            
            # Store in buffer
            news_item = {
                'headline': headline,
                'symbols': symbols,
                'sentiment': sentiment,
                'timestamp': created_at,
                'processed_at': datetime.now().isoformat()
            }
            
            self.news_buffer.append(news_item)
            
            # Update sentiment cache for relevant symbols
            for symbol in symbols:
                if symbol in self.symbols:
                    self.sentiment_cache[symbol] = sentiment
            
            print(f"[REALTIME] News: {headline[:50]}...  Sentiment: {sentiment:.2f}")
            
        except Exception as e:
            print(f"[REALTIME] Error processing news: {e}")
    
    def _analyze_sentiment(self, headline):
        """Simple sentiment analysis (can be replaced with more sophisticated analysis)"""
        try:
            # Simple keyword-based sentiment analysis
            positive_words = [
                'strong', 'growth', 'positive', 'up', 'gain', 'profit', 'beat', 'exceed',
                'rise', 'surge', 'jump', 'climb', 'rally', 'soar', 'boost', 'improve',
                'recovery', 'bounce', 'rebound', 'outperform', 'bullish', 'optimistic'
            ]
            
            negative_words = [
                'delay', 'fall', 'drop', 'loss', 'volatility', 'uncertainty', 'decline',
                'plunge', 'crash', 'dip', 'slump', 'weaken', 'concern', 'risk', 'bearish',
                'pessimistic', 'miss', 'disappoint', 'warn', 'cut', 'reduce', 'lower'
            ]
            
            headline_lower = headline.lower()
            positive_score = sum(1 for word in positive_words if word in headline_lower)
            negative_score = sum(1 for word in negative_words if word in headline_lower)
            
            if positive_score > negative_score:
                return min(0.8, 0.3 + (positive_score - negative_score) * 0.1)
            elif negative_score > positive_score:
                return max(-0.8, -0.3 - (negative_score - positive_score) * 0.1)
            else:
                return 0.0
                
        except Exception as e:
            print(f"[REALTIME] Sentiment analysis error: {e}")
            return 0.0
    
    def get_sentiment_for_symbol(self, symbol):
        """Get current sentiment for a specific symbol"""
        return self.sentiment_cache.get(symbol, 0.0)
    
    def get_recent_news(self, symbol=None, limit=10):
        """Get recent news items, optionally filtered by symbol"""
        if symbol:
            return [item for item in self.news_buffer if symbol in item['symbols']][-limit:]
        else:
            return list(self.news_buffer)[-limit:]
    
    def stop_stream(self):
        """Stop the WebSocket stream"""
        if self.ws:
            self.ws.close()
        self.is_running = False
        print("[REALTIME] News stream stopped")


# Backward-compatibility shim for older imports expecting RealtimeStream
class RealtimeStream:
    """Minimal compatibility wrapper mapping to AlpacaNewsStream."""
    def __init__(self, *args, **kwargs):
        self._impl = AlpacaNewsStream(
            api_key=kwargs.get('api_key', os.environ.get('ALPACA_API_KEY', '')),
            secret_key=kwargs.get('secret_key', os.environ.get('ALPACA_SECRET_KEY', '')),
            symbols=kwargs.get('symbols', None),
        )

    def start_stream(self):
        return self._impl.start_stream()

    def stop_stream(self):
        return self._impl.stop_stream()

    def get_recent_news(self, *args, **kwargs):
        return self._impl.get_recent_news(*args, **kwargs)
