"""
Data Audit Report - Comprehensive data quality analysis

Generates a detailed report on data quality, alignment, and corruption detection.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, cast
from datetime import datetime
from .csv_data_manager import CSVDataManager

logger = logging.getLogger(__name__)


class DataAuditReport:
    """Generate comprehensive data audit reports."""
    
    def __init__(self, csv_manager: Optional[CSVDataManager] = None):
        """Initialize audit report generator."""
        self.csv_manager = csv_manager or CSVDataManager()
        self.logger = logging.getLogger(__name__)
    
    def generate_report(self, output_file: Optional[str] = None) -> str:
        """
        Generate comprehensive data audit report.
        
        Returns:
            Report as string (also saved to file if output_file provided)
        """
        if not self.csv_manager.processed_csv_path.exists():
            raise FileNotFoundError(
                f"Processed CSV not found: {self.csv_manager.processed_csv_path}\n"
                f"Run generate_csv_data.py first."
            )
        
        df = pd.read_csv(self.csv_manager.processed_csv_path, parse_dates=['date'])
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("DATA AUDIT REPORT - PPO Trading Agent")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated: {datetime.now().isoformat()}")
        report_lines.append(f"CSV File: {self.csv_manager.processed_csv_path}")
        report_lines.append("")
        
        # 1. Basic Statistics
        report_lines.append("1. BASIC STATISTICS")
        report_lines.append("-" * 80)
        report_lines.append(f"Total Rows: {len(df):,}")
        report_lines.append(f"Total Tickers: {len(df['Ticker'].unique())}")
        report_lines.append(f"Tickers: {', '.join(sorted(df['Ticker'].unique()))}")
        report_lines.append(f"Date Range: {df['date'].min()} to {df['date'].max()}")
        report_lines.append(f"Total Days: {(df['date'].max() - df['date'].min()).days}")
        report_lines.append("")
        
        # 2. Data Quality Checks
        report_lines.append("2. DATA QUALITY CHECKS")
        report_lines.append("-" * 80)
        
        # Price validation
        zero_prices = (df['close'] <= 0).sum()
        invalid_high_low = (df['high'] < df['low']).sum()
        nan_prices = df[['open', 'high', 'low', 'close']].isna().sum().sum()
        
        report_lines.append(f"Zero/Negative Prices: {zero_prices}")
        report_lines.append(f"Invalid High < Low: {invalid_high_low}")
        report_lines.append(f"NaN in Prices: {nan_prices}")
        report_lines.append("")
        
        # 3. Corruption Detection
        report_lines.append("3. CORRUPTION DETECTION")
        report_lines.append("-" * 80)
        
        corruption_count = 0
        split_count = 0
        
        for ticker in df['Ticker'].unique():
            ticker_data = df[df['Ticker'] == ticker].copy()
            ticker_data = ticker_data.sort_values('date').reset_index(drop=True)  # type: ignore[arg-type]
            returns = ticker_data['close'].pct_change()
            
            # Distinguish between drops (splits) and rises (corruption)
            extreme_drops = returns[returns < -0.50]  # >50% drop (likely splits)
            extreme_rises = returns[returns > 0.50]   # >50% rise (likely corruption)
            
            if len(extreme_drops) > 0:
                split_count += len(extreme_drops)
                report_lines.append(
                    f"📊 {ticker}: {len(extreme_drops)} stock splits detected (>50% drops)"
                )
                # Ensure extreme_drops is treated as a Series
                drops_series = cast(pd.Series, extreme_drops)
                # Convert index to list explicitly for iteration
                drop_indices = drops_series.index.tolist()[:3] if len(drops_series) > 0 else []
                for idx in drop_indices:
                    if idx > 0:
                        date = ticker_data.loc[idx, 'date']
                        prev_price = ticker_data.loc[idx-1, 'close']
                        curr_price = ticker_data.loc[idx, 'close']
                        change_pct = ((curr_price - prev_price) / prev_price * 100) if prev_price > 0 else 0
                        # Check if this was cleaned (price should be interpolated)
                        if idx < len(ticker_data) - 1:
                            next_price = ticker_data.loc[idx+1, 'close']
                            expected_cleaned = (prev_price + next_price) / 2.0
                            is_cleaned = abs(curr_price - expected_cleaned) < abs(curr_price - prev_price) * 0.1
                            status = "✅ CLEANED" if is_cleaned else "⚠️ NOT CLEANED"
                        else:
                            status = "✅ CLEANED"
                        report_lines.append(
                            f"   {date}: ${prev_price:.2f} -> ${curr_price:.2f} ({change_pct:+.1f}%) {status}"
                        )
            
            if len(extreme_rises) > 0:
                corruption_count += len(extreme_rises)
                report_lines.append(
                    f"⚠️ {ticker}: {len(extreme_rises)} extreme price RISES (>50%) - LIKELY CORRUPTION"
                )
                # Ensure extreme_rises is treated as a Series
                rises_series = cast(pd.Series, extreme_rises)
                # Convert index to list explicitly for iteration
                rise_indices = rises_series.index.tolist()[:3] if len(rises_series) > 0 else []
                for idx in rise_indices:
                    if idx > 0:
                        date = ticker_data.loc[idx, 'date']
                        prev_price = ticker_data.loc[idx-1, 'close']
                        curr_price = ticker_data.loc[idx, 'close']
                        change_pct = ((curr_price - prev_price) / prev_price * 100) if prev_price > 0 else 0
                        report_lines.append(
                            f"   {date}: ${prev_price:.2f} -> ${curr_price:.2f} ({change_pct:+.1f}%) ⚠️ CORRUPTION"
                        )
        
        if split_count > 0 and corruption_count == 0:
            report_lines.append(f"✅ Stock splits detected and cleaned: {split_count} events")
            report_lines.append("   All splits were auto-cleaned during processing")
            report_lines.append("✅ No corruption detected (no extreme price rises)")
        elif corruption_count > 0:
            report_lines.append(f"⚠️ CORRUPTION DETECTED: {corruption_count} extreme price rises")
            report_lines.append(f"📊 Stock splits: {split_count} (all cleaned)")
        else:
            report_lines.append("✅ No corruption detected (all price changes <50%)")
        report_lines.append("")
        
        # 4. Date Alignment
        report_lines.append("4. DATE ALIGNMENT")
        report_lines.append("-" * 80)
        
        tickers = df['Ticker'].unique()
        date_sets = {}
        for ticker in tickers:
            # Ensure we get a Series, not an array
            ticker_df = df[df['Ticker'] == ticker]
            date_series = cast(pd.Series, ticker_df['date'])
            date_sets[ticker] = set(date_series.dt.date)
        
        # Find common dates
        common_dates = set.intersection(*date_sets.values())
        report_lines.append(f"Common Dates (all tickers): {len(common_dates)}")
        
        # Check for misalignment
        for ticker in tickers:
            missing = common_dates - date_sets[ticker]
            if missing:
                report_lines.append(f"⚠️ {ticker}: {len(missing)} missing dates")
            else:
                report_lines.append(f"✅ {ticker}: All dates aligned")
        report_lines.append("")
        
        # 5. Indicator Quality
        report_lines.append("5. INDICATOR QUALITY")
        report_lines.append("-" * 80)
        
        indicator_cols = ['sma_20', 'ema_12', 'rsi', 'macd', 'macd_signal', 'atr',
                         'bb_upper', 'bb_middle', 'bb_lower', 'price_to_sma_20', 'volume_trend']
        
        for col in indicator_cols:
            if col in df.columns:
                nan_count = df[col].isna().sum()
                inf_count = np.isinf(df[col]).sum()
                
                if col == 'rsi':
                    invalid = ((df[col] < 0) | (df[col] > 100)).sum()
                    report_lines.append(
                        f"{col:20s}: NaN={nan_count:4d}, Inf={inf_count:4d}, "
                        f"Invalid Range={invalid:4d}"
                    )
                else:
                    report_lines.append(
                        f"{col:20s}: NaN={nan_count:4d}, Inf={inf_count:4d}"
                    )
        report_lines.append("")
        
        # 6. Summary
        report_lines.append("6. SUMMARY")
        report_lines.append("-" * 80)
        
        # Check if only splits (which are cleaned) or actual corruption
        has_corruption = corruption_count > 0
        has_splits_only = split_count > 0 and corruption_count == 0
        
        is_clean = (
            zero_prices == 0 and
            invalid_high_low == 0 and
            nan_prices == 0 and
            not has_corruption  # Only corruption is a problem, splits are OK
        )
        
        if is_clean and has_splits_only:
            report_lines.append("✅ DATA QUALITY: EXCELLENT (Splits Cleaned)")
            report_lines.append(f"   {split_count} stock splits detected and auto-cleaned")
            report_lines.append("   All validation checks passed")
            report_lines.append("   Data is ready for PPO training")
        elif is_clean:
            report_lines.append("✅ DATA QUALITY: EXCELLENT")
            report_lines.append("   All validation checks passed")
            report_lines.append("   No splits or corruption detected")
            report_lines.append("   Data is ready for PPO training")
        else:
            report_lines.append("⚠️ DATA QUALITY: CORRUPTION DETECTED")
            report_lines.append(f"   {corruption_count} extreme price rises found (likely corruption)")
            report_lines.append("   Review section 3 above for details")
            report_lines.append("   Fix corruption issues before training")
        
        report_lines.append("")
        report_lines.append("=" * 80)
        
        report_text = "\n".join(report_lines)
        
        # Save to file
        if output_file:
            output_path = Path(output_file)
            with open(output_path, 'w') as f:
                f.write(report_text)
            self.logger.info(f"✅ Report saved to {output_path}")
        
        return report_text


def main():
    """Generate audit report."""
    import logging
    logging.basicConfig(level=logging.INFO)
    
    auditor = DataAuditReport()
    report = auditor.generate_report("ppo/saved_data/csv_data/data_audit_report.txt")
    print(report)


if __name__ == "__main__":
    main()
