"""
Performance Attribution Analysis
Detailed breakdown of performance drivers and factor analysis
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class AttributionConfig:
    """Performance attribution configuration"""
    risk_free_rate: float = 0.02  # 2% risk-free rate
    benchmark_symbol: str = "SPY"
    attribution_window: int = 252  # Trading days
    factor_analysis: bool = True
    sector_analysis: bool = True
    style_analysis: bool = True

class PerformanceAttribution:
    """
    Comprehensive performance attribution analysis
    """
    
    def __init__(self, initial_capital: float = 100000.0):
        self.initial_capital = initial_capital
        self.config = AttributionConfig()
        
        # Performance data
        self.portfolio_returns = []
        self.benchmark_returns = []
        self.risk_free_returns = []
        
        # Attribution components
        self.asset_allocation_effect = []
        self.stock_selection_effect = []
        self.interaction_effect = []
        
        # Factor analysis
        self.factor_exposures = {}
        self.factor_returns = {}
        self.factor_attribution = {}
        
        # Sector analysis
        self.sector_weights = {}
        self.sector_returns = {}
        self.sector_attribution = {}
        
        # Style analysis
        self.style_weights = {}
        self.style_returns = {}
        self.style_attribution = {}
        
        # Risk decomposition
        self.risk_decomposition = {}
        self.volatility_attribution = {}
        
        logger.info("PerformanceAttribution initialized")
    
    def calculate_brinson_attribution(self, portfolio_weights: Dict[str, float], 
                                    benchmark_weights: Dict[str, float],
                                    asset_returns: Dict[str, float]) -> Dict[str, Any]:
        """Calculate Brinson attribution analysis"""
        try:
            # Asset allocation effect
            allocation_effect = {}
            for asset in asset_returns:
                if asset in benchmark_weights:
                    allocation_effect[asset] = (
                        (portfolio_weights.get(asset, 0) - benchmark_weights[asset]) *
                        (asset_returns[asset] - self._calculate_benchmark_return(benchmark_weights, asset_returns))
                    )
            
            # Stock selection effect
            selection_effect = {}
            for asset in asset_returns:
                if asset in benchmark_weights:
                    selection_effect[asset] = (
                        benchmark_weights[asset] *
                        (asset_returns[asset] - self._calculate_benchmark_return(benchmark_weights, asset_returns))
                    )
            
            # Interaction effect
            interaction_effect = {}
            for asset in asset_returns:
                if asset in benchmark_weights:
                    interaction_effect[asset] = (
                        (portfolio_weights.get(asset, 0) - benchmark_weights[asset]) *
                        (asset_returns[asset] - self._calculate_benchmark_return(benchmark_weights, asset_returns))
                    )
            
            return {
                'allocation_effect': allocation_effect,
                'selection_effect': selection_effect,
                'interaction_effect': interaction_effect,
                'total_effect': {
                    'allocation': sum(allocation_effect.values()),
                    'selection': sum(selection_effect.values()),
                    'interaction': sum(interaction_effect.values())
                }
            }
            
        except Exception as e:
            logger.error(f"Error calculating Brinson attribution: {e}")
            return {}
    
    def _calculate_benchmark_return(self, weights: Dict[str, float], returns: Dict[str, float]) -> float:
        """Calculate weighted benchmark return"""
        try:
            total_return = 0.0
            total_weight = 0.0
            
            for asset, weight in weights.items():
                if asset in returns:
                    total_return += weight * returns[asset]
                    total_weight += weight
            
            return total_return / total_weight if total_weight > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating benchmark return: {e}")
            return 0.0
    
    def calculate_factor_attribution(self, portfolio_returns: pd.Series, 
                                   factor_returns: Dict[str, pd.Series],
                                   factor_exposures: Dict[str, float]) -> Dict[str, Any]:
        """Calculate factor attribution analysis"""
        try:
            attribution = {}
            
            # Calculate factor contributions
            factor_contributions = {}
            for factor, exposure in factor_exposures.items():
                if factor in factor_returns:
                    # Calculate factor contribution
                    factor_contribution = exposure * factor_returns[factor].mean()
                    factor_contributions[factor] = factor_contribution
            
            # Calculate residual return
            total_factor_contribution = sum(factor_contributions.values())
            residual_return = portfolio_returns.mean() - total_factor_contribution
            
            attribution = {
                'factor_contributions': factor_contributions,
                'total_factor_contribution': total_factor_contribution,
                'residual_return': residual_return,
                'factor_exposures': factor_exposures
            }
            
            return attribution
            
        except Exception as e:
            logger.error(f"Error calculating factor attribution: {e}")
            return {}
    
    def calculate_sector_attribution(self, portfolio_weights: Dict[str, float],
                                   sector_returns: Dict[str, float]) -> Dict[str, Any]:
        """Calculate sector attribution analysis"""
        try:
            sector_attribution = {}
            
            # Calculate sector contributions
            for sector, weight in portfolio_weights.items():
                if sector in sector_returns:
                    sector_contribution = weight * sector_returns[sector]
                    sector_attribution[sector] = {
                        'weight': weight,
                        'return': sector_returns[sector],
                        'contribution': sector_contribution
                    }
            
            # Calculate total sector contribution
            total_sector_contribution = sum(
                attr['contribution'] for attr in sector_attribution.values()
            )
            
            return {
                'sector_attribution': sector_attribution,
                'total_sector_contribution': total_sector_contribution
            }
            
        except Exception as e:
            logger.error(f"Error calculating sector attribution: {e}")
            return {}
    
    def calculate_style_attribution(self, portfolio_weights: Dict[str, float],
                                  style_returns: Dict[str, float]) -> Dict[str, Any]:
        """Calculate style attribution analysis"""
        try:
            style_attribution = {}
            
            # Calculate style contributions
            for style, weight in portfolio_weights.items():
                if style in style_returns:
                    style_contribution = weight * style_returns[style]
                    style_attribution[style] = {
                        'weight': weight,
                        'return': style_returns[style],
                        'contribution': style_contribution
                    }
            
            # Calculate total style contribution
            total_style_contribution = sum(
                attr['contribution'] for attr in style_attribution.values()
            )
            
            return {
                'style_attribution': style_attribution,
                'total_style_contribution': total_style_contribution
            }
            
        except Exception as e:
            logger.error(f"Error calculating style attribution: {e}")
            return {}
    
    def calculate_risk_decomposition(self, portfolio_returns: pd.Series,
                                   asset_weights: Dict[str, float],
                                   asset_returns: Dict[str, pd.Series]) -> Dict[str, Any]:
        """Calculate risk decomposition analysis"""
        try:
            # Calculate portfolio volatility
            portfolio_vol = portfolio_returns.std()
            
            # Calculate individual asset volatilities
            asset_vols = {}
            for asset, returns in asset_returns.items():
                asset_vols[asset] = returns.std()
            
            # Calculate risk contributions
            risk_contributions = {}
            for asset, weight in asset_weights.items():
                if asset in asset_vols:
                    # Marginal risk contribution
                    marginal_risk = weight * asset_vols[asset]
                    risk_contributions[asset] = {
                        'weight': weight,
                        'volatility': asset_vols[asset],
                        'marginal_risk': marginal_risk,
                        'risk_contribution': marginal_risk / portfolio_vol if portfolio_vol > 0 else 0
                    }
            
            return {
                'portfolio_volatility': portfolio_vol,
                'risk_contributions': risk_contributions,
                'concentration_risk': self._calculate_concentration_risk(asset_weights),
                'diversification_benefit': self._calculate_diversification_benefit(asset_weights, asset_vols)
            }
            
        except Exception as e:
            logger.error(f"Error calculating risk decomposition: {e}")
            return {}
    
    def _calculate_concentration_risk(self, weights: Dict[str, float]) -> float:
        """Calculate portfolio concentration risk (Herfindahl index)"""
        try:
            return sum(weight ** 2 for weight in weights.values())
        except Exception as e:
            logger.error(f"Error calculating concentration risk: {e}")
            return 0.0
    
    def _calculate_diversification_benefit(self, weights: Dict[str, float], 
                                         volatilities: Dict[str, float]) -> float:
        """Calculate diversification benefit"""
        try:
            weighted_vol = sum(weights[asset] * volatilities[asset] 
                             for asset in weights if asset in volatilities)
            portfolio_vol = sum(weights.values())  # Simplified
            
            if portfolio_vol > 0:
                return 1 - (portfolio_vol / weighted_vol)
            return 0.0
            
        except Exception as e:
            logger.error(f"Error calculating diversification benefit: {e}")
            return 0.0
    
    def calculate_volatility_attribution(self, portfolio_returns: pd.Series,
                                       factor_returns: Dict[str, pd.Series],
                                       factor_exposures: Dict[str, float]) -> Dict[str, Any]:
        """Calculate volatility attribution to factors"""
        try:
            volatility_attribution = {}
            
            # Calculate factor volatilities
            for factor, exposure in factor_exposures.items():
                if factor in factor_returns:
                    factor_vol = factor_returns[factor].std()
                    volatility_contribution = abs(exposure) * factor_vol
                    volatility_attribution[factor] = {
                        'exposure': exposure,
                        'factor_volatility': factor_vol,
                        'volatility_contribution': volatility_contribution
                    }
            
            # Calculate total factor volatility
            total_factor_vol = sum(
                attr['volatility_contribution'] for attr in volatility_attribution.values()
            )
            
            # Calculate residual volatility
            portfolio_vol = portfolio_returns.std()
            residual_vol = max(0, portfolio_vol - total_factor_vol)
            
            return {
                'factor_volatility_attribution': volatility_attribution,
                'total_factor_volatility': total_factor_vol,
                'residual_volatility': residual_vol,
                'portfolio_volatility': portfolio_vol
            }
            
        except Exception as e:
            logger.error(f"Error calculating volatility attribution: {e}")
            return {}
    
    def calculate_rolling_attribution(self, portfolio_returns: pd.Series,
                                    benchmark_returns: pd.Series,
                                    window: int = 60) -> Dict[str, pd.Series]:
        """Calculate rolling attribution analysis"""
        try:
            rolling_attribution = {}
            
            # Rolling excess return
            excess_returns = portfolio_returns - benchmark_returns
            rolling_attribution['excess_return'] = excess_returns.rolling(window).mean()
            
            # Rolling information ratio
            rolling_ir = (excess_returns.rolling(window).mean() / 
                         excess_returns.rolling(window).std())
            rolling_attribution['information_ratio'] = rolling_ir
            
            # Rolling tracking error
            tracking_error = excess_returns.rolling(window).std()
            rolling_attribution['tracking_error'] = tracking_error
            
            # Rolling beta
            rolling_beta = (portfolio_returns.rolling(window).cov(benchmark_returns) / 
                           benchmark_returns.rolling(window).var())
            rolling_attribution['beta'] = rolling_beta
            
            return rolling_attribution
            
        except Exception as e:
            logger.error(f"Error calculating rolling attribution: {e}")
            return {}
    
    def generate_attribution_report(self, portfolio_data: Dict[str, Any],
                                  benchmark_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive attribution report"""
        try:
            report = {
                'summary': {
                    'total_return': portfolio_data.get('total_return', 0),
                    'benchmark_return': benchmark_data.get('total_return', 0),
                    'excess_return': portfolio_data.get('total_return', 0) - benchmark_data.get('total_return', 0),
                    'information_ratio': self._calculate_information_ratio(portfolio_data, benchmark_data),
                    'tracking_error': self._calculate_tracking_error(portfolio_data, benchmark_data)
                },
                'attribution_analysis': {
                    'brinson_attribution': portfolio_data.get('brinson_attribution', {}),
                    'factor_attribution': portfolio_data.get('factor_attribution', {}),
                    'sector_attribution': portfolio_data.get('sector_attribution', {}),
                    'style_attribution': portfolio_data.get('style_attribution', {})
                },
                'risk_analysis': {
                    'risk_decomposition': portfolio_data.get('risk_decomposition', {}),
                    'volatility_attribution': portfolio_data.get('volatility_attribution', {})
                },
                'rolling_analysis': portfolio_data.get('rolling_attribution', {}),
                'timestamp': datetime.now().isoformat()
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating attribution report: {e}")
            return {}
    
    def _calculate_information_ratio(self, portfolio_data: Dict[str, Any],
                                   benchmark_data: Dict[str, Any]) -> float:
        """Calculate information ratio"""
        try:
            excess_return = portfolio_data.get('total_return', 0) - benchmark_data.get('total_return', 0)
            tracking_error = self._calculate_tracking_error(portfolio_data, benchmark_data)
            
            return excess_return / tracking_error if tracking_error > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating information ratio: {e}")
            return 0.0
    
    def _calculate_tracking_error(self, portfolio_data: Dict[str, Any],
                                benchmark_data: Dict[str, Any]) -> float:
        """Calculate tracking error"""
        try:
            # This would use actual return series
            # For now, return a default value
            return 0.05  # 5% default tracking error
            
        except Exception as e:
            logger.error(f"Error calculating tracking error: {e}")
            return 0.05
    
    def update_performance_data(self, portfolio_return: float, benchmark_return: float,
                              risk_free_return: Optional[float] = None):
        """Update performance data for attribution analysis"""
        try:
            self.portfolio_returns.append(portfolio_return)
            self.benchmark_returns.append(benchmark_return)
            
            if risk_free_return is not None:
                self.risk_free_returns.append(risk_free_return)
            else:
                self.risk_free_returns.append(self.config.risk_free_rate / 252)  # Daily risk-free rate
            
            # Keep only recent data
            max_length = self.config.attribution_window
            if len(self.portfolio_returns) > max_length:
                self.portfolio_returns = self.portfolio_returns[-max_length:]
                self.benchmark_returns = self.benchmark_returns[-max_length:]
                self.risk_free_returns = self.risk_free_returns[-max_length:]
                
        except Exception as e:
            logger.error(f"Error updating performance data: {e}")
    
    def get_attribution_summary(self) -> Dict[str, Any]:
        """Get summary of attribution analysis"""
        try:
            return {
                'data_points': len(self.portfolio_returns),
                'portfolio_volatility': np.std(self.portfolio_returns) if self.portfolio_returns else 0,
                'benchmark_volatility': np.std(self.benchmark_returns) if self.benchmark_returns else 0,
                'excess_return': np.mean(self.portfolio_returns) - np.mean(self.benchmark_returns) if self.portfolio_returns and self.benchmark_returns else 0,
                'information_ratio': self._calculate_information_ratio(
                    {'total_return': np.mean(self.portfolio_returns) if self.portfolio_returns else 0},
                    {'total_return': np.mean(self.benchmark_returns) if self.benchmark_returns else 0}
                )
            }
            
        except Exception as e:
            logger.error(f"Error getting attribution summary: {e}")
            return {}
