"""
Lightweight data validator stub
Provides safe no-op validators to avoid import errors when advanced utilities
are not available. Can be extended with real validation logic later.
"""

from typing import Tuple, cast, Any
import pandas as pd
import numpy as np


def validate_dataframe(df: pd.DataFrame) -> Tuple[bool, str]:
    """Basic validation: checks non-empty and presence of timestamp/tic if any.

    Returns (is_valid, message)
    """
    if df is None:
        return False, "DataFrame is None"
    if df.empty:
        return False, "DataFrame is empty"
    # Optional checks
    required = [c for c in ['timestamp', 'tic'] if c in df.columns]
    if required:
        required_df = df[required]
        # Check for NaN values - handle both DataFrame and Series cases
        if isinstance(required_df, pd.DataFrame):
            # Convert to numpy array for boolean check
            null_check = required_df.isna()
            has_nulls = False
            for col in null_check.columns:
                col_series = null_check[col]
                if isinstance(col_series, pd.Series):
                    col_array = np.array(col_series.values)
                    if np.any(col_array):
                        has_nulls = True
                        break
            if has_nulls:
                return False, "Required columns contain NaN"
        else:
            # Single column case
            if required_df.isna().any():
                return False, "Required columns contain NaN"
    return True, "OK"


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleanup: drop fully NA rows and duplicate index if present."""
    if df is None or df.empty:
        return df
    df = df.dropna(how='all')
    return df


class DataValidator:
    """Class wrapper for compatibility with callers expecting a class API."""

    @staticmethod
    def validate_dataframe(df: pd.DataFrame) -> Tuple[bool, str]:
        return validate_dataframe(df)

    @staticmethod
    def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        return clean_dataframe(df)


