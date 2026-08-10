"""
Sentiment Analyzer for News Analysis
Based on TG4.1 sentiment analysis implementation
"""

import logging
from typing import Dict, List, Optional
import numpy as np

logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """Advanced sentiment analysis like TG4.1"""
    
    def __init__(self):
        # Sentiment keywords
        self.positive_words = [
            'up', 'rise', 'gain', 'positive', 'growth', 'profit', 'beat', 'exceed', 'strong',
            'bullish', 'rally', 'surge', 'jump', 'climb', 'soar', 'boost', 'improve', 'recovery'
        ]
        self.negative_words = [
            'down', 'fall', 'loss', 'negative', 'decline', 'miss', 'weak', 'drop', 'crash',
            'bearish', 'plunge', 'tumble', 'slump', 'dip', 'slide', 'downgrade', 'worry', 'concern'
        ]
        
        logger.info(" SentimentAnalyzer initialized")
    
    def analyze_sentiment(self, text: str) -> float:
        """Analyze sentiment of text"""
        text_lower = text.lower()
        
        positive_count = sum(1 for word in self.positive_words if word in text_lower)
        negative_count = sum(1 for word in self.negative_words if word in text_lower)
        
        if positive_count == 0 and negative_count == 0:
            return 0.0
        
        sentiment = (positive_count - negative_count) / (positive_count + negative_count)
        return max(-1.0, min(1.0, sentiment))
    
    def analyze_sentiment_with_fundamentals(self, text: str, fundamental_data: Dict) -> Dict:
        """
        Analyze sentiment with fundamental data adjustment
        
        Args:
            text: News text to analyze
            fundamental_data: Dictionary containing fundamental metrics
            
        Returns:
            Dictionary with sentiment analysis results
        """
        # Base sentiment analysis
        base_sentiment = self.analyze_sentiment(text)
        
        # Fundamental adjustment factors
        pe_ratio = fundamental_data.get('pe_ratio', 0)
        revenue_growth = fundamental_data.get('revenue_growth', 0)
        profit_margins = fundamental_data.get('profit_margins', 0)
        return_on_equity = fundamental_data.get('return_on_equity', 0)
        
        # Calculate fundamental adjustment
        fundamental_adjustment = 0.0
        
        # P/E ratio adjustment (lower P/E = more positive)
        if pe_ratio > 0:
            pe_adjustment = max(-0.2, min(0.2, (25 - pe_ratio) / 25))  # Normalize around P/E of 25
            fundamental_adjustment += pe_adjustment * 0.1
        
        # Revenue growth adjustment
        if revenue_growth > 0:
            growth_adjustment = max(-0.2, min(0.2, revenue_growth))
            fundamental_adjustment += growth_adjustment * 0.2
        
        # Profit margin adjustment
        if profit_margins > 0:
            margin_adjustment = max(-0.2, min(0.2, (profit_margins - 0.1) / 0.1))  # Normalize around 10%
            fundamental_adjustment += margin_adjustment * 0.15
        
        # ROE adjustment
        if return_on_equity > 0:
            roe_adjustment = max(-0.2, min(0.2, (return_on_equity - 0.15) / 0.15))  # Normalize around 15%
            fundamental_adjustment += roe_adjustment * 0.1
        
        # Apply fundamental adjustment to base sentiment
        adjusted_sentiment = base_sentiment + fundamental_adjustment
        adjusted_sentiment = max(-1.0, min(1.0, adjusted_sentiment))
        
        return {
            'base_sentiment': base_sentiment,
            'fundamental_adjustment': fundamental_adjustment,
            'fundamental_adjusted_sentiment': adjusted_sentiment,
            'fundamental_metrics': {
                'pe_ratio': pe_ratio,
                'revenue_growth': revenue_growth,
                'profit_margins': profit_margins,
                'return_on_equity': return_on_equity
            }
        }
    
    def get_market_sentiment(self, news_items: List[Dict]) -> float:
        """Get overall market sentiment from news items"""
        if not news_items:
            return 0.0
        
        sentiments = []
        for item in news_items:
            text = f"{item.get('headline', '')} {item.get('summary', '')}"
            sentiment = self.analyze_sentiment(text)
            sentiments.append(sentiment)
        
        return np.mean(sentiments) if sentiments else 0.0
