"""
DataFrame Optimization Utilities
Optimized pandas operations for large datasets
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union, Callable
import logging

logger = logging.getLogger(__name__)

class DataFrameOptimizer:
    """
    Optimized DataFrame operations for large financial datasets
    """
    
    def __init__(self):
        self.optimization_stats = {}
    
    def optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame dtypes to reduce memory usage"""
        try:
            original_memory = df.memory_usage(deep=True).sum()
            
            # Optimize numeric columns
            for col in df.select_dtypes(include=['int64']).columns:
                df[col] = pd.to_numeric(df[col], downcast='integer')
            
            for col in df.select_dtypes(include=['float64']).columns:
                df[col] = pd.to_numeric(df[col], downcast='float')
            
            # Optimize object columns
            for col in df.select_dtypes(include=['object']).columns:
                if df[col].nunique() / len(df) < 0.5:  # Low cardinality
                    df[col] = df[col].astype('category')
            
            optimized_memory = df.memory_usage(deep=True).sum()
            reduction = (original_memory - optimized_memory) / original_memory * 100
            
            self.optimization_stats['memory_reduction'] = reduction
            logger.info(f"Memory usage reduced by {reduction:.1f}%")
            
            return df
            
        except Exception as e:
            logger.error(f"Error optimizing dtypes: {e}")
            return df
    
    def fast_forward_fill(self, df: pd.DataFrame, limit: Optional[int] = None) -> pd.DataFrame:
        """Optimized forward fill operation"""
        try:
            # Use vectorized operations where possible
            if limit is None:
                return df.fillna(method='ffill')
            else:
                return df.fillna(method='ffill', limit=limit)
        except Exception as e:
            logger.error(f"Error in fast forward fill: {e}")
            return df
    
    def vectorized_pivot(self, df: pd.DataFrame, 
                        index: str, 
                        columns: str, 
                        values: str,
                        aggfunc: str = 'last') -> pd.DataFrame:
        """Optimized pivot table operation"""
        try:
            # Use pandas pivot_table with optimized settings
            pivot_df = df.pivot_table(
                index=index,
                columns=columns,
                values=values,
                aggfunc=aggfunc,
                fill_value=np.nan
            )
            
            # Optimize the result
            return self.optimize_dtypes(pivot_df)
            
        except Exception as e:
            logger.error(f"Error in vectorized pivot: {e}")
            return pd.DataFrame()
    
    def chunked_operation(self, df: pd.DataFrame, 
                         operation: Callable[[pd.DataFrame], pd.DataFrame], 
                         chunk_size: int = 10000) -> pd.DataFrame:
        """Perform operations on DataFrame in chunks to manage memory"""
        try:
            results = []
            total_chunks = len(df) // chunk_size + 1
            
            for i in range(total_chunks):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, len(df))
                
                chunk = df.iloc[start_idx:end_idx].copy()
                processed_chunk = operation(chunk)
                results.append(processed_chunk)
                
                logger.debug(f"Processed chunk {i+1}/{total_chunks}")
            
            return pd.concat(results, ignore_index=True)
            
        except Exception as e:
            logger.error(f"Error in chunked operation: {e}")
            return df
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization statistics"""
        return self.optimization_stats.copy()

# Utility functions for backward compatibility
def optimize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Quick optimization function"""
    optimizer = DataFrameOptimizer()
    return optimizer.optimize_dtypes(df)

def fast_pivot(df: pd.DataFrame, index: str, columns: str, values: str) -> pd.DataFrame:
    """Quick pivot function"""
    optimizer = DataFrameOptimizer()
    return optimizer.vectorized_pivot(df, index, columns, values)
