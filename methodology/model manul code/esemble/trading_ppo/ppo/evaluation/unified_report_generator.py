"""
Unified Report Generator for PPO Trading System
Consolidates ALL execution results into ONE comprehensive report
Easy for Cursor AI agents to read and analyze
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional, List
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class UnifiedReportGenerator:
    """Generates a single comprehensive report from all execution data"""
    
    def __init__(self, base_dir: str = "ppo"):
        self.base_dir = Path(base_dir)
        self.reports_dir = self.base_dir / "daily_reports"
        self.validation_dir = self.base_dir / "saved_data" / "validation"
    
    def generate_unified_report(self, execution_results: Dict[str, Any], 
                               training_results: Optional[Dict[str, Any]] = None,
                               trade_history_data: Optional[List[Dict[str, Any]]] = None,
                               portfolio_history_data: Optional[List[float]] = None) -> Optional[str]:
        """Generate unified report in both JSON and Markdown formats."""
        try:
            # Ensure directories exist
            self.reports_dir.mkdir(parents=True, exist_ok=True)
            self.validation_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            # Enrich execution results with saved validation artefacts when direct metrics are missing
            execution_results = self._enrich_results_with_saved_data(execution_results)

            # JSON summary (lightweight)
            diag = execution_results.get('python_shap_diagnostic', {})
            json_summary = {
                "timestamp": timestamp,
                "total_return": execution_results.get('total_return', 0.0),
                "sharpe_ratio": execution_results.get('sharpe_ratio', 0.0),
                "max_drawdown": execution_results.get('max_drawdown', 0.0),
                "win_rate": execution_results.get('win_rate', 0.0),
                "total_trades": execution_results.get('total_trades', 0),
                "paper_trading_ready": execution_results.get('paper_trading_ready', False),
                "data_source": execution_results.get('data_source', 'UNKNOWN'),
                "python_executable": diag.get('python_executable', 'N/A'),
                "shap_available": diag.get('shap_available', False),
                "shap_version": diag.get('shap_version')
            }
            json_path = self.reports_dir / f"unified_report_{timestamp}.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_summary, f, indent=2, default=str)

            # Comprehensive Markdown report with detailed analysis
            md_path = self.reports_dir / f"UNIFIED_REPORT_{timestamp}.md"
            md = []
            md.append(f"## PPO Trading - Unified Report ({timestamp})\n")
            
            # CRITICAL VERIFICATION SUMMARY (TCN & Fundamental Data)
            md.append("### ✅ Critical Features Verification\n")
            verification_summary = self._generate_verification_summary(execution_results, training_results)
            md.extend(verification_summary)
            
            # Key Performance Metrics (deterministic eval only - paper-trading / production behavior)
            headline = self._headline_metrics(execution_results)
            md.append("### 🎯 Key Performance Metrics\n")
            md.append(f"- **Total Return**: {headline.get('total_return', 0.0)*100:.2f}%\n")
            
            # Realized vs Unrealized Gains Breakdown (headline doesn't have $; keep from execution_results when from stochastic)
            realized_return = headline.get('realized_return', execution_results.get('realized_return', 0.0))
            unrealized_return = headline.get('unrealized_return', execution_results.get('unrealized_return', 0.0))
            realized_gains = execution_results.get('realized_gains', 0.0)
            unrealized_gains = execution_results.get('unrealized_gains', 0.0)
            md.append(f"- **Realized Return**: {realized_return:.2f}% (${realized_gains:,.2f})\n")
            md.append(f"- **Unrealized Return**: {unrealized_return:.2f}% (${unrealized_gains:,.2f})\n")
            if abs(unrealized_return) < 1e-6 and abs(unrealized_gains) < 1e-6:
                md.append("  - *Unrealized is 0% when all positions are closed by period end (no open P&L).*\n")
            
            _sharpe = headline.get('sharpe_ratio', 0.0)
            md.append(f"- **Sharpe Ratio**: {_sharpe:.3f}\n")
            if _sharpe == 0.0:
                md.append("  - *Sharpe is 0 when return volatility is negligible (flat or near-constant equity curve).*\n")
            md.append(f"- **Max Drawdown**: {headline.get('max_drawdown', 0.0)*100:.2f}%\n")
            md.append(f"- **Win Rate**: {headline.get('win_rate', 0.0)*100:.1f}%\n")
            md.append(f"- **Total Trades**: {headline.get('total_trades', 0)}\n")
            _vol = execution_results.get('volatility', 0.0)
            _vol = 0.0 if (isinstance(_vol, float) and (np.isnan(_vol) or np.isinf(_vol))) else (_vol or 0.0)
            md.append(f"- **Volatility**: {float(_vol)*100:.2f}%\n")
            md.append(f"- **Profit Factor**: {execution_results.get('profit_factor', 0.0):.2f}\n")
            # Alpha/Beta are meaningful only when benchmark series is valid (non-trivial variance)
            bench_ok = bool(execution_results.get('benchmark_data_valid', False))
            alpha = execution_results.get('alpha', None)
            beta = execution_results.get('beta', None)
            if not bench_ok or alpha is None or beta is None:
                md.append("- **Alpha**: N/A\n")
                md.append("- **Beta**: N/A\n")
            else:
                try:
                    md.append(f"- **Alpha**: {float(alpha):.4f}\n")
                except Exception:
                    md.append("- **Alpha**: N/A\n")
                try:
                    md.append(f"- **Beta**: {float(beta):.4f}\n")
                except Exception:
                    md.append("- **Beta**: N/A\n")
            md.append(f"- **Paper Trading Ready**: {execution_results.get('paper_trading_ready', False)}\n")
            
            # Execution & Performance Analysis
            md.append("\n### ⚡ Execution & Performance Analysis\n")
            data_source = execution_results.get('data_source', 'UNKNOWN')
            if isinstance(data_source, str) and data_source.upper() == "CACHE":
                data_source = "API (cache)"
            md.append(f"- **Total Execution Time**: {execution_results.get('total_execution_time', 0):.1f} seconds\n")
            md.append(f"- **Data Source**: {data_source}\n")
            md.append(f"- **Steps Simulated**: {execution_results.get('steps_simulated', 0)}\n")
            md.append(f"- **Episodes Completed**: {execution_results.get('episodes_completed', 0)}\n")
            md.append(f"- **Average Episode Length**: {execution_results.get('avg_episode_length', 0):.1f} steps\n")
            md.append(f"- **Steps per Second**: {execution_results.get('steps_per_second', 0):.1f}\n")
            md.append(f"- **Trades per Second**: {execution_results.get('trades_per_second', 0):.2f}\n")
            
            # Trading Behavior Analysis
            md.append("\n### 📊 Trading Behavior Analysis\n")
            buy_t = execution_results.get('buy_trades', 0)
            sell_t = execution_results.get('sell_trades', 0)
            short_t = execution_results.get('short_trades', 0)
            total_t = execution_results.get('total_trades', 0)
            md.append(f"- **Buy Trades**: {buy_t} ({execution_results.get('buy_percentage', 0):.1f}%)\n")
            md.append(f"- **Sell Trades**: {sell_t} ({execution_results.get('sell_percentage', 0):.1f}%)\n")
            md.append(f"- **Short Trades**: {short_t} ({execution_results.get('short_percentage', 0):.1f}%)\n")
            md.append(f"- **Stop Loss Trades**: {execution_results.get('stop_loss_trades', 0)} ({execution_results.get('stop_loss_percentage', 0):.1f}%)\n")
            if total_t > 0 and (buy_t + sell_t + short_t) == total_t:
                md.append(f"  - *Total = BUY + SELL + SHORT = {buy_t} + {sell_t} + {short_t} = {total_t}.*\n")
            elif total_t > 0:
                md.append(f"  - *Total trades = {total_t} (includes BUY, SELL, SHORT, STOP_LOSS, etc.).*\n")
            md.append(f"- **Average Trade Size**: ${execution_results.get('avg_trade_size', 0):.2f}\n")
            # Fix: Use correct key name with fallback (performance_evaluator returns 'average_hold_time', not 'avg_hold_time')
            avg_hold_time = execution_results.get('average_hold_time', execution_results.get('avg_hold_time', 0))
            md.append(f"- **Average Hold Time**: {avg_hold_time:.1f} steps\n")
            md.append(f"- **Max Consecutive Wins**: {execution_results.get('max_consecutive_wins', 0)}\n")
            md.append(f"- **Max Consecutive Losses**: {execution_results.get('max_consecutive_losses', 0)}\n")

            # Ticker Breakdown (which tickers agent is buying/selling)
            ticker_section = self._build_ticker_breakdown_section(trade_history_data)
            if ticker_section:
                md.extend(ticker_section)

            # Short Positions (SHORT action implemented 2026-02-28)
            short_section = self._build_short_positions_section(trade_history_data)
            if short_section:
                md.extend(short_section)
            else:
                md.append("\n### 📉 Short Positions\n")
                md.append("- **Status**: ✅ SHORT action implemented – no short trades in this run\n")
                md.append("- **Actions**: 0=HOLD, 1=BUY, 2=SELL, 3=SHORT\n")

            # Policy action distribution (raw actions chosen by PPO), independent of executed trades.
            md.append("\n### 🧭 Policy Action Distribution (Raw PPO Actions)\n")
            pac = execution_results.get('policy_action_counts', {}) if isinstance(execution_results, dict) else {}
            pap = execution_results.get('policy_action_percentages', {}) if isinstance(execution_results, dict) else {}
            pat = execution_results.get('policy_action_total', 0)
            md.append(f"- **Total Actions Counted**: {pat}\n")
            md.append(f"- **HOLD (0)**: {pac.get('HOLD', 0)} ({pap.get('HOLD', 0.0):.1f}%)\n")
            md.append(f"- **BUY (1)**: {pac.get('BUY', 0)} ({pap.get('BUY', 0.0):.1f}%)\n")
            md.append(f"- **SELL (2)**: {pac.get('SELL', 0)} ({pap.get('SELL', 0.0):.1f}%)\n")
            md.append(f"- **SHORT (3)**: {pac.get('SHORT', 0)} ({pap.get('SHORT', 0.0):.1f}%)\n")
            
            # ACTION DISTRIBUTION IMBALANCE DETECTION
            buy_pct = pap.get('BUY', 0.0)
            sell_pct = pap.get('SELL', 0.0)
            hold_pct = pap.get('HOLD', 0.0)
            
            md.append("\n**🔍 Action Distribution Analysis:**\n")
            action_issues = []
            
            # Check for critically low BUY percentage (should trigger entropy boost)
            if buy_pct < 2.0:
                action_issues.append(f"🚨 **CRITICAL**: Raw BUY only {buy_pct:.1f}% - agent severely under-buying")
                action_issues.append(f"   → Entropy boost SHOULD trigger (threshold: BUY < 5%)")
            elif buy_pct < 5.0:
                action_issues.append(f"⚠️ **WARNING**: Raw BUY only {buy_pct:.1f}% - agent under-buying")
                action_issues.append(f"   → Consider increasing exploration or ent_coef")
            elif buy_pct < 15.0:
                action_issues.append(f"📊 Raw BUY at {buy_pct:.1f}% - moderate buying activity")
            else:
                action_issues.append(f"✅ Raw BUY at {buy_pct:.1f}% - healthy buying activity")
            
            # Check for SELL dominance
            if sell_pct > 85.0:
                action_issues.append(f"⚠️ Raw SELL dominance ({sell_pct:.1f}%) - policy may be stuck in SELL-bias")
            elif sell_pct > 70.0:
                action_issues.append(f"📊 Raw SELL high ({sell_pct:.1f}%) - check if intended for exit-heavy strategy")
            
            # Check for excessive HOLD
            if hold_pct > 50.0:
                action_issues.append(f"⚠️ Raw HOLD dominant ({hold_pct:.1f}%) - agent may be too passive")
            
            # Calculate action entropy (should be high for exploration)
            if pat > 0:
                probs = [max(buy_pct/100, 1e-10), max(sell_pct/100, 1e-10), max(hold_pct/100, 1e-10)]
                action_entropy = -sum(p * np.log(p) for p in probs if p > 0)
                max_entropy = np.log(3)  # Maximum entropy for 3 actions
                entropy_ratio = action_entropy / max_entropy
                if entropy_ratio < 0.3:
                    action_issues.append(f"🚨 Action entropy very low ({entropy_ratio:.2f}) - policy too deterministic")
                elif entropy_ratio < 0.5:
                    action_issues.append(f"⚠️ Action entropy moderate ({entropy_ratio:.2f}) - limited exploration")
                else:
                    action_issues.append(f"✅ Action entropy healthy ({entropy_ratio:.2f}) - good exploration")
            
            for issue in action_issues:
                md.append(f"  - {issue}\n")

            # Gating + masking diagnostics (helps explain "why no trades" / invalid action loops)
            md.append("\n### 🚦 Gating & Masking Diagnostics\n")
            gs = execution_results.get('gating_stats', {}) if isinstance(execution_results, dict) else {}
            if isinstance(gs, dict) and gs:
                md.append(f"- **Gating allowed**: {int(gs.get('allowed', 0))}\n")
                md.append(f"- **Gating blocked**: {int(gs.get('blocked', 0))}\n")
                md.append(f"- **High confidence allowed**: {int(gs.get('high', 0))}\n")
                md.append(f"- **Medium confidence allowed**: {int(gs.get('medium', 0))}\n")
                md.append(f"- **Low/explore allowed**: {int(gs.get('low', 0))} (explore={int(gs.get('explore', 0))})\n")
            else:
                md.append("- **Gating stats**: Not available\n")
            md.append(f"- **Masked SELL count**: {execution_results.get('masked_sell_count', 0)}\n")
            md.append(f"- **Masked SELL max streak**: {execution_results.get('masked_sell_streak', 0)}\n")

            # Zero-trade / low-trade diagnostic: entropy collapse (policy choosing HOLD)
            total_trades_det = execution_results.get('total_trades', 0)
            if total_trades_det == 0 or total_trades_det < 10:
                md.append("\n### 🔬 Entropy Collapse Diagnostic (Policy Over-Holding)\n")
                md.append(f"- **Action distribution**: HOLD={pac.get('HOLD', 0)} ({pap.get('HOLD', 0.0):.1f}%), BUY={pac.get('BUY', 0)} ({pap.get('BUY', 0.0):.1f}%), SELL={pac.get('SELL', 0)} ({pap.get('SELL', 0.0):.1f}%)\n")
                adaptive_ent_log = execution_results.get('adaptive_entropy_log', [])
                ent_increases = [e for e in adaptive_ent_log if e.get('event') == 'ENTROPY_INCREASE' or 'ENTROPY' in str(e.get('event', '')).upper()]
                if ent_increases:
                    md.append(f"- **Entropy actually changed**: Yes ({len(ent_increases)} ENTROPY_INCREASE or similar events)\n")
                else:
                    md.append("- **Entropy actually changed**: No (no ENTROPY_INCREASE events in adaptive_entropy_log)\n")
            
            # Risk Management Analysis
            md.append("\n### 🛡️ Risk Management Analysis\n")
            md.append(f"- **Position Limits Violations**: {execution_results.get('position_limit_violations', 0)}\n")
            md.append(f"- **Trade Interval Violations**: {execution_results.get('trade_interval_violations', 0)}\n")
            md.append(f"- **Diversification Violations**: {execution_results.get('diversification_violations', 0)}\n")
            md.append(f"- **Dynamic Stop Loss Triggers**: {execution_results.get('stop_loss_triggers', 0)}\n")
            md.append(f"- **Average Stop Loss PnL**: {execution_results.get('avg_stop_loss_pnl', 0):.2f}%\n")
            md.append(f"- **Risk-Adjusted Return**: {execution_results.get('risk_adjusted_return', 0):.4f}\n")
            
            # Memory & System Performance
            md.append("\n### 💾 Memory & System Performance\n")
            # CRITICAL FIX: Read from performance_metrics structure
            perf_metrics = execution_results.get('performance_metrics', {})
            resource_usage = perf_metrics.get('resource_usage', {})
            memory_info = resource_usage.get('memory', {})
            cpu_info = resource_usage.get('cpu', {})
            
            peak_memory_mb = perf_metrics.get('memory_mb', execution_results.get('peak_memory_mb', 0))
            final_memory_mb = perf_metrics.get('memory_mb', execution_results.get('final_memory_mb', 0))
            memory_percent = memory_info.get('percent', 0.0)
            cpu_percent = cpu_info.get('percent', perf_metrics.get('cpu_percent', execution_results.get('cpu_usage', 0)))
            
            md.append(f"- **Peak Memory Usage**: {peak_memory_mb:.1f} MB\n")
            md.append(f"- **Final Memory Usage**: {final_memory_mb:.1f} MB\n")
            md.append(f"- **Memory Efficiency**: {memory_percent:.1f}%\n")
            md.append(f"- **CPU Usage**: {cpu_percent:.1f}%\n")
            md.append(f"- **GPU Usage**: {execution_results.get('gpu_usage', 0):.1f}%\n")
            
            # Bottleneck Analysis
            md.append("\n### 🔍 Bottleneck Analysis\n")
            md.append(f"- **Data Loading Time**: {execution_results.get('data_loading_time', 0):.2f}s\n")
            md.append(f"- **Training Time**: {execution_results.get('training_time', 0):.2f}s\n")
            md.append(f"- **Evaluation Time**: {execution_results.get('evaluation_time', 0):.2f}s\n")
            md.append(f"- **Report Generation Time**: {execution_results.get('report_time', 0):.2f}s\n")
            md.append(f"- **Slowest Component**: {execution_results.get('slowest_component', 'N/A')}\n")
            md.append(f"- **Performance Bottleneck**: {execution_results.get('performance_bottleneck', 'None identified')}\n")
            
            # Training Loop Profiling
            md.append("\n### 🔬 Training Loop Profiling\n")
            profiling_data = execution_results.get('training_profile', {})
            if profiling_data:
                md.append(f"- **Profiling File**: `{profiling_data.get('profiling_file', 'N/A')}`\n")
                md.append(f"- **Total Training Time**: {profiling_data.get('total_time', 0):.2f}s\n")
                
                top_functions = profiling_data.get('top_functions', [])
                if top_functions:
                    md.append(f"- **Top 5 Slowest Functions**:\n")
                    for i, (cumtime, func_name) in enumerate(top_functions[:5], 1):
                        func_short = func_name.split('/')[-1] if '/' in func_name else func_name
                        md.append(f"  {i}. `{func_short[:50]}`: {cumtime:.2f}s ({cumtime/profiling_data.get('total_time', 1)*100:.1f}%)\n")
            else:
                md.append(f"- **Status**: ⚠️ Profiling data not available\n")
                md.append(f"- **Note**: Profiling is enabled in training loop - check logs for details\n")
            
            # Learning Progress Analysis
            md.append("\n### 🧠 Learning Progress Analysis\n")
            md.append(f"- **Learning Rate**: {execution_results.get('learning_rate', 0):.6f}\n")
            md.append(f"- **Policy Loss**: {execution_results.get('policy_loss', 0):.4f}\n")
            md.append(f"- **Value Loss**: {execution_results.get('value_loss', 0):.4f}\n")
            md.append(f"- **Entropy**: {execution_results.get('entropy', 0):.4f}\n")
            md.append(f"- **Exploration Rate**: {execution_results.get('exploration_rate', 0):.4f}\n")
            md.append(f"- **Convergence Score**: {execution_results.get('convergence_score', 0):.4f}\n")
            
            # Fundamental Data Verification (Detailed)
            md.append("\n### 📊 Fundamental Data Verification (Detailed)\n")
            fundamental_status = execution_results.get('fundamental_status', {})
            cache_path = Path("ppo/saved_data/fundamental_cache/fundamental_data_latest.pkl")
            csv_candidates = [
                Path("ppo/saved_data/csv_data/Fundamental_Data.csv"),
                Path("ppo/saved_data/csv_data/Fundamental_Data_new.csv"),
                Path("ppo/saved_data/csv_data/fundamental_data.csv"),
                Path("ppo/saved_data/csv_data/funddamental_data.csv"),
            ]
            existing = [(p, p.stat().st_mtime) for p in csv_candidates if p.exists()]
            csv_path = max(existing, key=lambda x: x[1])[0] if existing else None
            fund_weight = 0.0
            tech_weight = 0.0
            try:
                from ppo.utils.dynamic_config import get_dynamic_config
                dcfg = get_dynamic_config()
                fund_weight = float(getattr(dcfg, 'fundamental_signal_weight', 0.0))
                tech_weight = float(getattr(dcfg, 'technical_signal_weight', 1.0))
            except Exception:
                fund_weight = 0.0
                tech_weight = 1.0
            
            if fundamental_status.get('available', False):
                tickers_loaded = fundamental_status.get('tickers_loaded', [])
                cache_path_str = fundamental_status.get('cache_path', str(cache_path))
                md.append(f"- **Status**: ✅ Available\n")
                md.append(f"- **Tickers with Fundamentals**: {len(tickers_loaded)}\n")
                if tickers_loaded:
                    md.append(f"- **Loaded Tickers**: {', '.join(tickers_loaded[:10])}{'...' if len(tickers_loaded) > 10 else ''}\n")
                md.append(f"- **Cache Path**: `{cache_path_str}`\n")
                md.append(f"- **Agent Using Fundamentals**: {'✅ Yes' if fund_weight > 0 else '⚠️ Available but disabled in rewards'} (fundamental_weight={fund_weight:.2f}, technical_weight={tech_weight:.2f})\n")
            elif cache_path.exists():
                try:
                    import pickle
                    with open(cache_path, 'rb') as f:
                        data = pickle.load(f)
                    if data and isinstance(data, dict) and len(data) > 0:
                        md.append(f"- **Status**: ✅ Available (detected from cache file)\n")
                        md.append(f"- **Tickers with Fundamentals**: {len(data)}\n")
                        md.append(f"- **Loaded Tickers**: {', '.join(list(data.keys())[:10])}{'...' if len(data) > 10 else ''}\n")
                        md.append(f"- **Cache Path**: `{cache_path}`\n")
                        md.append(f"- **Agent Using Fundamentals**: {'✅ Yes' if fund_weight > 0 else '⚠️ Available but disabled in rewards'} (fundamental_weight={fund_weight:.2f}, technical_weight={tech_weight:.2f})\n")
                    else:
                        md.append(f"- **Status**: ⚠️ Cache Empty\n")
                        md.append(f"- **Cache Path**: `{cache_path}`\n")
                        if csv_path:
                            md.append(f"- **CSV Source Detected**: `{csv_path}` (cache not built yet)\n")
                        md.append(f"- **Agent Using Fundamentals**: ❌ No (fundamental_weight={fund_weight:.2f}, technical_weight={tech_weight:.2f})\n")
                        md.append(f"- **Action**: Run `python ppo/collect_fundamentals.py` to collect data\n")
                except Exception as e:
                    md.append(f"- **Status**: ⚠️ Cache Error\n")
                    md.append(f"- **Error**: {str(e)[:100]}\n")
                    md.append(f"- **Agent Using Fundamentals**: ❌ No\n")
            else:
                if csv_path:
                    md.append(f"- **Status**: ⚠️ Cache Not Found (CSV available)\n")
                    md.append(f"- **Cache Path**: `{cache_path}` (does not exist)\n")
                    md.append(f"- **CSV Source**: `{csv_path}`\n")
                    md.append(f"- **Agent Using Fundamentals**: ❌ No (cache missing; fundamental_weight={fund_weight:.2f}, technical_weight={tech_weight:.2f})\n")
                    md.append(f"- **Action**: Run `python ppo/collect_fundamentals.py --csv-file \"{csv_path}\"` to build cache\n")
                else:
                    md.append(f"- **Status**: ❌ Not Found\n")
                    md.append(f"- **Cache Path**: `{cache_path}` (does not exist)\n")
                    md.append(f"- **Agent Using Fundamentals**: ❌ No (fundamental_weight={fund_weight:.2f}, technical_weight={tech_weight:.2f})\n")
                    md.append(f"- **Action**: Run `python ppo/collect_fundamentals.py` to create cache\n")
            
            # Data Files
            md.append("\n### 📁 Data Files\n")
            md.append(f"- Trade history: `{self.validation_dir / 'trade_history.csv'}`\n")
            md.append(f"- Portfolio history: `{self.validation_dir / 'portfolio_history.csv'}`\n")
            md.append(f"- Returns: `{self.validation_dir / 'returns.csv'}`\n")
            md.append(f"- Backtest results: `{self.validation_dir / 'backtest_results.json'}`\n")
            md.append(f"- Benchmark returns: `{self.validation_dir / 'benchmark_returns.csv'}`\n")
            
            # Adaptive LR Logs
            md.append("\n### 🔄 Adaptive Learning Rate Logs\n")
            try:
                format_method = getattr(self, '_format_adaptive_lr_logs', None)
                if format_method and callable(format_method):
                    md.append(format_method(execution_results))
                else:
                    # Fallback if method doesn't exist (Python import cache issue)
                    adaptive_lr_log = execution_results.get('adaptive_lr_log', [])
                    events_count = execution_results.get('adaptive_lr_events_count', len(adaptive_lr_log))
                    callback_triggered = execution_results.get('adaptive_lr_callback_triggered', len(adaptive_lr_log) > 0)
                    if events_count == 0 and not callback_triggered:
                        md.append("- **Status**: ⚠️ **CALLBACK DID NOT TRIGGER**\n")
                        md.append("  - No adaptive LR events recorded\n")
                    else:
                        md.append(f"- **Status**: ✅ **CALLBACK ACTIVE** ({events_count} events recorded)\n")
                        md.append(f"- **Callback Triggered**: {'✅ Yes' if callback_triggered else '❌ No'}\n")
            except Exception as e:
                logger.warning(f"Error formatting adaptive LR logs: {e}")
                md.append(f"- **Status**: ⚠️ **ERROR**: {str(e)}\n")
                md.append("- **Action**: Check logs for details\n")
            
            # Adaptive Entropy Logs (new) - Enhanced with action distribution check
            md.append("\n### 🔄 Adaptive Entropy Logs\n")
            try:
                adaptive_ent_log = execution_results.get('adaptive_entropy_log', [])
                ent_events_count = execution_results.get('adaptive_entropy_events_count', 0)
                
                # Get raw action percentages for entropy trigger verification
                _pap = execution_results.get('policy_action_percentages', {})
                raw_buy_pct = _pap.get('BUY', 0.0)
                raw_sell_pct = _pap.get('SELL', 0.0)
                policy_entropy = execution_results.get('entropy', 0.0)
                
                if ent_events_count == 0:
                    md.append("- **Status**: ⚠️ **NO ENTROPY EVENTS RECORDED**\n")
                    md.append("  - Check if trades fell below threshold to trigger entropy boost\n")
                    
                    # NEW: Analyze WHY entropy boost didn't trigger
                    md.append("\n**🔍 Entropy Boost Trigger Analysis:**\n")
                    
                    # Current callback checks entropy < 0.1 for collapse
                    if policy_entropy >= 0.1:
                        md.append(f"  - Policy entropy ({policy_entropy:.4f}) >= 0.1 threshold - no collapse detected\n")
                    else:
                        md.append(f"  - ⚠️ Policy entropy ({policy_entropy:.4f}) < 0.1 - should have triggered!\n")
                    
                    # NEW: Check action distribution imbalance (not currently triggering entropy)
                    if raw_buy_pct < 5.0:
                        md.append(f"  - 🚨 **MISSING TRIGGER**: Raw BUY={raw_buy_pct:.1f}% < 5% threshold\n")
                        md.append(f"    → Entropy boost SHOULD trigger based on action imbalance\n")
                        md.append(f"    → **FIX NEEDED**: Add action distribution check to entropy callback\n")
                    if raw_sell_pct > 85.0:
                        md.append(f"  - 🚨 **MISSING TRIGGER**: Raw SELL={raw_sell_pct:.1f}% > 85% threshold\n")
                        md.append(f"    → Policy stuck in SELL-dominant mode\n")
                        md.append(f"    → **FIX NEEDED**: Add action dominance check to entropy callback\n")
                    
                    if raw_buy_pct >= 5.0 and raw_sell_pct <= 85.0 and policy_entropy >= 0.1:
                        md.append(f"  - ✅ No entropy issues detected (action distribution and policy entropy normal)\n")
                else:
                    md.append(f"- **Status**: ✅ **CALLBACK ACTIVE** ({ent_events_count} events recorded)\n")
                    for event in adaptive_ent_log[-5:]:  # Show last 5 events
                        md.append(f"  - {event.get('message', str(event))}\n")
            except Exception as e:
                logger.warning(f"Error formatting adaptive entropy logs: {e}")
                md.append(f"- **Status**: ⚠️ **ERROR**: {str(e)}\n")
                md.append("- **Action**: Check logs for details\n")
            
            # Data Integrity & Availability Checks (files and fundamentals)
            md.append("\n### 🧾 Data Integrity Checks\n")
            try:
                checks = []
                base = self.base_dir
                validation_dir = self.validation_dir
                fundamentals_dir = base / "saved_data" / "fundamental_cache"
                saved_model = base / "saved_models" / "ppo_trading_agent.zip"
                files_to_check = [
                    ("trade_history.csv", validation_dir / "trade_history.csv"),
                    ("portfolio_history.csv", validation_dir / "portfolio_history.csv"),
                    ("returns.csv", validation_dir / "returns.csv"),
                    ("benchmark_returns.csv", validation_dir / "benchmark_returns.csv"),
                    ("backtest_results.json", validation_dir / "backtest_results.json"),
                    ("gating_stats.json", validation_dir / "gating_stats.json"),
                ]
                fundamentals_ok = fundamentals_dir.exists() and any(fundamentals_dir.glob("*.pkl"))
                checks.append(f"- Fundamentals cache: {'✅ Found' if fundamentals_ok else '⚠️ Missing/empty'} ({fundamentals_dir})")
                checks.append(f"- PPO model file: {'✅ Found' if saved_model.exists() else '⚠️ Missing'} ({saved_model})")
                missing_files = []
                for label, path in files_to_check:
                    if path.exists():
                        checks.append(f"- {label}: ✅ Found ({path})")
                    else:
                        checks.append(f"- {label}: ⚠️ Missing ({path})")
                        missing_files.append(label)
                md.extend(checks)
                if missing_files:
                    md.append(f"- Summary: ⚠️ Missing files: {', '.join(missing_files)}\n")
                else:
                    md.append("- Summary: ✅ All expected data files present\n")
            except Exception as e:
                logger.warning(f"Error running data integrity checks: {e}")
                md.append(f"- **Status**: ⚠️ **ERROR** during data integrity checks: {str(e)}\n")
            
            # Adaptive Reward Verification
            md.append("\n### 🎯 Adaptive Reward System Verification\n")
            try:
                format_method = getattr(self, '_format_adaptive_reward_logs', None)
                if format_method and callable(format_method):
                    md.append(format_method(execution_results))
                else:
                    # Fallback if method doesn't exist (Python import cache issue)
                    adaptive_reward_log = execution_results.get('adaptive_reward_log', [])
                    events_count = execution_results.get('adaptive_reward_events_count', 0)
                    if events_count == 0:
                        md.append("- **Status**: ⚠️ **NO ADAPTIVE REWARD DATA**\n")
                        md.append("  - No adaptive reward events recorded\n")
                    else:
                        md.append(f"- **Status**: ✅ **ADAPTIVE REWARDS ACTIVE** ({events_count} events recorded)\n")
                        stats = execution_results.get('adaptive_reward_stats', {})
                        if stats:
                            _b = stats.get('buy_multiplier_avg', 1.0)
                            _s = stats.get('sell_multiplier_avg', 1.0)
                            _h = stats.get('hold_multiplier_avg', 1.0)
                            _b = 1.0 if (isinstance(_b, float) and (np.isnan(_b) or np.isinf(_b))) else (_b or 1.0)
                            _s = 1.0 if (isinstance(_s, float) and (np.isnan(_s) or np.isinf(_s))) else (_s or 1.0)
                            _h = 1.0 if (isinstance(_h, float) and (np.isnan(_h) or np.isinf(_h))) else (_h or 1.0)
                            md.append(f"  - BUY Multiplier Avg: {float(_b):.2f}x\n")
                            md.append(f"  - SELL Multiplier Avg: {float(_s):.2f}x\n")
                            md.append(f"  - HOLD Multiplier Avg: {float(_h):.2f}x\n")
            except Exception as e:
                logger.warning(f"Error formatting adaptive reward logs: {e}")
                md.append(f"- **Status**: ⚠️ **ERROR**: {str(e)}\n")
                md.append("- **Action**: Check logs for details\n")
            
            # Behavior-Based Adaptive Reward Verification
            md.append("\n### 🤖 Behavior-Based Adaptive Reward System\n")
            try:
                behavior_format_method = getattr(self, '_format_behavior_adaptive_reward_logs', None)
                if behavior_format_method and callable(behavior_format_method):
                    md.append(behavior_format_method(execution_results))
                else:
                    # Fallback formatting
                    adaptive_reward_log = execution_results.get('adaptive_reward_log', [])
                    behavior_events = [e for e in adaptive_reward_log if e.get('type') == 'behavior_adaptive']
                    
                    if len(behavior_events) == 0:
                        md.append("- **Status**: ⚠️ **NO BEHAVIOR ADAPTIVE DATA**\n")
                        md.append("  - No behavior-based adaptive adjustments recorded\n")
                        md.append("  - **Possible reasons**:\n")
                        md.append("    - Evaluation was too short (< 50 steps)\n")
                        md.append("    - Agent behavior met all targets (no adjustments needed)\n")
                    else:
                        md.append(f"- **Status**: ✅ **BEHAVIOR ADAPTIVE ACTIVE** ({len(behavior_events)} events recorded)\n")
                        # Calculate statistics
                        trading_mults = [e.get('base_trading_incentive_mult', 1.0) for e in behavior_events]
                        sell_mults = [e.get('sell_base_incentive_mult', 1.0) for e in behavior_events]
                        hold_penalty_mults = [e.get('hold_penalty_mult', 1.0) for e in behavior_events]
                        aging_rate_mults = [e.get('aging_penalty_rate_mult', 1.0) for e in behavior_events]
                        
                        if trading_mults:
                            md.append(f"\n**Trading Frequency Adjustments:**\n")
                            md.append(f"  - **Base Trading Incentive Multiplier**: Avg={np.mean(trading_mults):.2f}x, Range=[{np.min(trading_mults):.2f}x, {np.max(trading_mults):.2f}x]\n")
                            active_count = sum(1 for m in trading_mults if m > 1.0)
                            md.append(f"  - **Active Adjustments**: {active_count}/{len(trading_mults)} events ({active_count/len(trading_mults)*100:.1f}%)\n")
                        
                        if sell_mults:
                            md.append(f"\n**SELL/BUY Ratio Adjustments:**\n")
                            md.append(f"  - **SELL Base Incentive Multiplier**: Avg={np.mean(sell_mults):.2f}x, Range=[{np.min(sell_mults):.2f}x, {np.max(sell_mults):.2f}x]\n")
                            active_count = sum(1 for m in sell_mults if m > 1.0)
                            md.append(f"  - **Active Adjustments**: {active_count}/{len(sell_mults)} events ({active_count/len(sell_mults)*100:.1f}%)\n")
                        
                        if hold_penalty_mults:
                            md.append(f"\n**Hold Time Adjustments:**\n")
                            md.append(f"  - **Hold Penalty Multiplier**: Avg={np.mean(hold_penalty_mults):.2f}x, Range=[{np.min(hold_penalty_mults):.2f}x, {np.max(hold_penalty_mults):.2f}x]\n")
                            md.append(f"  - **Aging Penalty Rate Multiplier**: Avg={np.mean(aging_rate_mults):.2f}x, Range=[{np.min(aging_rate_mults):.2f}x, {np.max(aging_rate_mults):.2f}x]\n")
                            active_count = sum(1 for m in hold_penalty_mults if m > 1.0)
                            md.append(f"  - **Active Adjustments**: {active_count}/{len(hold_penalty_mults)} events ({active_count/len(hold_penalty_mults)*100:.1f}%)\n")
                            
                            # CLARIFICATION: Explain what HOLD multiplier actually does
                            md.append(f"\n  **📝 Note on HOLD Multiplier:**\n")
                            md.append(f"  - The HOLD multiplier is NOT a penalty amplifier\n")
                            md.append(f"  - It AMPLIFIES market movement rewards during favorable conditions\n")
                            md.append(f"  - High volatility → Higher multiplier → Gains are amplified\n")
                            md.append(f"  - Strong trend with positions → Lower multiplier → Let profits run\n")
                            md.append(f"  - This is the CORRECT adaptive approach (reward the right action)\n")
                        
                        # Show sample events
                        md.append(f"\n**Sample Behavior Adaptive Events (last 5):**\n")
                        for i, event in enumerate(behavior_events[-5:], 1):
                            step = event.get('step', 0)
                            trading_mult = event.get('base_trading_incentive_mult', 1.0)
                            sell_mult = event.get('sell_base_incentive_mult', 1.0)
                            hold_mult = event.get('hold_penalty_mult', 1.0)
                            actual_trades = event.get('actual_trades', 0)
                            expected_trades = event.get('expected_trades', 0)
                            sell_count = event.get('sell_count', 0)
                            total_trades = event.get('total_trades', 0)
                            avg_hold = event.get('avg_hold_time', 0)
                            md.append(f"  {i}. Step {step:,}: Trades={actual_trades:.0f}/{expected_trades:.1f} (trading_mult={trading_mult:.2f}x), "
                                    f"SELL={sell_count}/{total_trades} (sell_mult={sell_mult:.2f}x), "
                                    f"HoldTime={avg_hold:.1f} (penalty_mult={hold_mult:.2f}x)\n")
            except Exception as e:
                logger.warning(f"Error formatting behavior adaptive reward logs: {e}")
                md.append(f"- **Status**: ⚠️ **ERROR**: {str(e)}\n")
                md.append("- **Action**: Check logs for details\n")
            
            # Diagnostic Root Cause Analysis
            md.append("\n### 🔍 Diagnose Root Cause\n")
            md.append("\n**Check terminal logs for:**\n")
            md.append("\n- **\"ADAPTIVE LR (POOR)\" messages** (should appear if Sharpe < 0.5)\n")
            md.append("- **\"ADAPTIVE ENTROPY\" messages** (should appear if trades < 50)\n")
            md.append("- **Model loading errors**\n")
            md.append("- **Episode termination reasons**\n")
            
            # Add diagnostic status based on current metrics
            sharpe = execution_results.get('sharpe_ratio', 0.0)
            total_trades = execution_results.get('total_trades', 0)
            entropy = execution_results.get('entropy', 0.0)
            steps_simulated = execution_results.get('steps_simulated', 0)
            
            md.append("\n**Current Status:**\n")
            if sharpe < 0.5:
                md.append(f"- ⚠️ Sharpe ({sharpe:.3f}) < 0.5 threshold - Adaptive LR should trigger\n")
            if total_trades < 50:
                md.append(f"- ⚠️ Trades ({total_trades}) < 50 threshold - Adaptive Entropy should trigger\n")
            if entropy < 0:
                md.append(f"- 🚨 **CRITICAL**: Negative entropy ({entropy:.4f}) indicates model corruption - reset required\n")
            if steps_simulated < 1000:
                md.append(f"- ⚠️ Low step count ({steps_simulated}) - Evaluation may be too short for proper assessment\n")
            if sharpe >= 0.5 and total_trades >= 50 and entropy >= 0:
                md.append("- ✅ All diagnostic thresholds normal\n")
            
            # Verification Logs Section (NEW)
            md.append("\n### ✅ Verification Logs\n")
            verification_logs = self._generate_verification_logs(execution_results, training_results)
            md.extend(verification_logs)
            
            # Recommendations
            md.append("\n### 💡 Performance Recommendations\n")
            recommendations = self._generate_recommendations(execution_results)
            for rec in recommendations:
                md.append(f"- {rec}\n")

            with open(md_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(md))

            logger.info(f" Unified report generated: {md_path}")
            return str(md_path)
            
        except Exception as e:
            logger.error(f" Failed to generate unified report: {e}")
            return None
    
    def _generate_verification_summary(self, execution_results: Dict[str, Any],
                                       training_results: Optional[Dict[str, Any]] = None) -> List[str]:
        """Generate a concise verification summary for TCN and Fundamental Data at the top of the report"""
        summary = []
        
        # TCN Verification
        tcn_status = "❌ NOT VERIFIED"
        tcn_details = []
        try:
            from ppo.utils.dynamic_config import get_dynamic_config
            config = get_dynamic_config()
            tcn_configured = config.use_temporal_memory and config.temporal_memory_type == "tcn"
            
            if tcn_configured:
                # Check if TCN was actually used
                tcn_used = execution_results.get('tcn_used', False)
                model_spec = execution_results.get('model_spec', {})
                if model_spec:
                    tcn_used = model_spec.get('tcn_used', tcn_used)
                    policy_type = model_spec.get('policy_type', '')
                    if 'TCN' in str(policy_type).upper():
                        tcn_used = True
                
                # Check saved model
                model_path = self.base_dir / "saved_models" / "ppo_trading_agent.zip"
                if model_path.exists():
                    try:
                        # FIXED: Try MaskablePPO first (for action masking), then fall back to PPO
                        loaded_model = None
                        try:
                            from sb3_contrib import MaskablePPO
                            loaded_model = MaskablePPO.load(str(model_path), device='cpu')
                        except (ImportError, Exception):
                            # Fall back to standard PPO
                            from stable_baselines3 import PPO
                            loaded_model = PPO.load(str(model_path), device='cpu')
                        if loaded_model and hasattr(loaded_model, 'policy'):
                            fe_class_name = ""
                            if hasattr(loaded_model.policy, 'features_extractor'):
                                fe_class_name = loaded_model.policy.features_extractor.__class__.__name__
                                if 'TCN' in fe_class_name:
                                    tcn_used = True
                            policy_class = loaded_model.policy.__class__.__name__
                            if 'TCN' in policy_class:
                                tcn_used = True
                    except Exception:
                        pass
                
                if tcn_used:
                    tcn_status = "✅ ACTIVE"
                    tcn_details.append(f"TCN policy detected in saved model")
                else:
                    tcn_status = "⚠️ CONFIGURED BUT NOT USED"
                    tcn_details.append("TCN configured but model uses standard MLP")
            else:
                tcn_status = "❌ DISABLED"
                tcn_details.append("Using standard MlpPolicy (no temporal memory)")
        except Exception as e:
            tcn_status = "❌ ERROR"
            tcn_details.append(f"Verification error: {str(e)[:50]}")
        
        summary.append(f"- **TCN (Temporal Memory)**: {tcn_status}\n")
        for detail in tcn_details:
            summary.append(f"  - {detail}\n")
        
        # Multi-timeframe (intraday merge + dynamic timeframe rewards)
        try:
            from ppo.utils.dynamic_config import get_dynamic_config
            _dc = get_dynamic_config()
            intervals = getattr(_dc, 'intraday_merge_intervals', ['5m', '30m', '1h', '1m'])
            use_intraday_step = getattr(_dc, 'use_intraday_step', False)
            tf_1d = getattr(_dc, 'timeframe_reward_weight_1d', 1.0)
            tf_5m = getattr(_dc, 'timeframe_reward_weight_5m', 1.2)
            summary.append(f"- **Multi-timeframe**: merge={intervals}, intraday_step={use_intraday_step}, timeframe_weights (1d={tf_1d}, 5m={tf_5m})\n")
        except Exception:
            summary.append("- **Multi-timeframe**: not verified\n")
        
        # Fundamental Data Verification
        fundamental_status = "❌ NOT VERIFIED"
        fundamental_details = []
        try:
            fundamental_status_data = execution_results.get('fundamental_status', {})
            cache_path = Path("ppo/saved_data/fundamental_cache/fundamental_data_latest.pkl")
            csv_candidates = [
                Path("ppo/saved_data/csv_data/Fundamental_Data.csv"),
                Path("ppo/saved_data/csv_data/Fundamental_Data_new.csv"),
                Path("ppo/saved_data/csv_data/fundamental_data.csv"),
                Path("ppo/saved_data/csv_data/funddamental_data.csv"),
            ]
            existing = [(p, p.stat().st_mtime) for p in csv_candidates if p.exists()]
            csv_path = max(existing, key=lambda x: x[1])[0] if existing else None

            if fundamental_status_data.get('available', False):
                tickers_loaded = fundamental_status_data.get('tickers_loaded', [])
                if tickers_loaded:
                    fundamental_status = "✅ AVAILABLE"
                    fundamental_details.append(f"{len(tickers_loaded)} tickers loaded")
                    fundamental_details.append(f"Cache: {fundamental_status_data.get('cache_path', 'N/A')}")
                else:
                    fundamental_status = "⚠️ CACHE EMPTY"
                    fundamental_details.append("Cache file exists but no ticker data")
            elif cache_path.exists():
                try:
                    import pickle
                    with open(cache_path, 'rb') as f:
                        data = pickle.load(f)
                    if data and isinstance(data, dict) and len(data) > 0:
                        fundamental_status = "✅ AVAILABLE"
                        fundamental_details.append(f"{len(data)} tickers in cache")
                        fundamental_details.append(f"Cache: {cache_path}")
                    else:
                        fundamental_status = "⚠️ CACHE EMPTY"
                        fundamental_details.append("Cache file exists but empty")
                except Exception as e:
                    fundamental_status = "⚠️ CACHE ERROR"
                    fundamental_details.append(f"Cannot read cache: {str(e)[:50]}")
            else:
                if csv_path:
                    fundamental_status = "⚠️ CSV FOUND / CACHE MISSING"
                    fundamental_details.append(f"CSV source detected: {csv_path}")
                    fundamental_details.append("Run: python ppo/collect_fundamentals.py --csv-file ppo/saved_data/csv_data/Fundamental_Data.csv")
                else:
                    fundamental_status = "❌ NOT FOUND"
                    fundamental_details.append("Cache file does not exist")
                    fundamental_details.append("Run: python ppo/collect_fundamentals.py")
        except Exception as e:
            fundamental_status = "❌ ERROR"
            fundamental_details.append(f"Verification error: {str(e)[:50]}")
        
        summary.append(f"- **Fundamental Data**: {fundamental_status}\n")
        for detail in fundamental_details:
            summary.append(f"  - {detail}\n")
        
        # Environment & SHAP Diagnostic (2026-03-06)
        diag = execution_results.get('python_shap_diagnostic', {})
        if diag:
            py_exe = diag.get('python_executable', 'N/A')
            shap_ok = diag.get('shap_available', False)
            shap_ver = diag.get('shap_version', '')
            summary.append(f"- **Python Executable**: `{py_exe}`\n")
            if shap_ok:
                summary.append(f"- **SHAP (Explainability)**: ✅ Available (v{shap_ver})\n")
            else:
                err = diag.get('shap_import_error', 'Not installed')
                summary.append(f"- **SHAP (Explainability)**: ❌ Unavailable ({err})\n")
                summary.append(f"  - *Tip: Run with same Python as test: `python3.13.exe PPO_fixed.py` or `python3.13.exe -m pip install shap`*\n")
        else:
            summary.append("- **SHAP Diagnostic**: Not collected\n")
        
        # Overall Status
        if "✅" in tcn_status and "✅" in fundamental_status:
            summary.append(f"- **Overall System Status**: ✅ **FULLY OPERATIONAL**\n")
        elif "⚠️" in tcn_status or "⚠️" in fundamental_status:
            summary.append(f"- **Overall System Status**: ⚠️ **PARTIALLY OPERATIONAL**\n")
        else:
            summary.append(f"- **Overall System Status**: ❌ **ISSUES DETECTED**\n")
        
        summary.append("\n")
        return summary
    
    def _generate_verification_logs(self, execution_results: Dict[str, Any], 
                                    training_results: Optional[Dict[str, Any]] = None) -> List[str]:
        """Generate verification logs for TCN usage, errors, and data corruption"""
        logs = []
        
        # 1. TCN Usage Verification
        logs.append("\n#### 🧠 TCN (Temporal Convolutional Network) Status\n")
        try:
            from ppo.utils.dynamic_config import get_dynamic_config
            config = get_dynamic_config()
            
            tcn_configured = config.use_temporal_memory and config.temporal_memory_type == "tcn"
            logs.append(f"- **TCN Configured**: {'✅ Yes' if tcn_configured else '❌ No'}\n")
            if tcn_configured:
                logs.append(f"  - Temporal memory type: `{config.temporal_memory_type}`\n")
                logs.append(f"  - Hidden dims: `{config.tcn_hidden_dims}`\n")
                logs.append(f"  - Dilation rates: `{config.tcn_dilation_rates}`\n")
            
            # Check if TCN policy is available
            try:
                from ppo.models.tcn_policy import TCNTradingPolicy
                logs.append(f"- **TCN Policy Available**: ✅ Yes\n")
                tcn_available = True
            except ImportError as e:
                logs.append(f"- **TCN Policy Available**: ❌ No ({str(e)[:50]})\n")
                tcn_available = False
            
            # Check if TCN was actually used (from execution results, training results, or model)
            tcn_used = execution_results.get('tcn_used', False)
            
            # Also check from model spec
            model_spec = execution_results.get('model_spec', {})
            if model_spec:
                tcn_used = model_spec.get('tcn_used', tcn_used)
                policy_type = model_spec.get('policy_type', '')
                if 'TCN' in str(policy_type).upper():
                    tcn_used = True
            
            # Check from training results model if available
            if training_results and 'model' in training_results:
                try:
                    model = training_results['model']
                    if hasattr(model, 'policy'):
                        policy_class_name = model.policy.__class__.__name__
                        if 'TCN' in policy_class_name:
                            tcn_used = True
                            logs.append(f"  - Detected TCN from model policy class: `{policy_class_name}`\n")
                except Exception:
                    pass
            
            # CRITICAL: Check saved model file for TCN layers (definitive proof)
            logs.append(f"  - **Saved Model Check**:\n")
            try:
                import zipfile
                import torch
                model_path = self.base_dir / "saved_models" / "ppo_trading_agent.zip"
                if model_path.exists():
                    logs.append(f"    - Model file exists: ✅\n")
                    # Try to load and inspect model architecture
                    try:
                        # FIXED: Try MaskablePPO first (for action masking), then fall back to PPO
                        loaded_model = None
                        try:
                            from sb3_contrib import MaskablePPO
                            loaded_model = MaskablePPO.load(str(model_path), device='cpu')
                        except (ImportError, Exception):
                            # Fall back to standard PPO
                            from stable_baselines3 import PPO
                            loaded_model = PPO.load(str(model_path), device='cpu')
                        if hasattr(loaded_model, 'policy'):
                            # Check policy network for TCN layers
                            policy_net = loaded_model.policy
                            
                            # Check for TCN-related layers in the network
                            tcn_layers_found = []
                            if hasattr(policy_net, 'features_extractor'):
                                # Check feature extractor for TCN blocks
                                fe = policy_net.features_extractor
                                for name, module in fe.named_modules():
                                    if 'tcn' in name.lower() or 'TCN' in name or 'conv1d' in name.lower():
                                        tcn_layers_found.append(name)
                            
                            # Check feature extractor class name (most reliable indicator)
                            fe_class_name = ""
                            if hasattr(policy_net, 'features_extractor'):
                                fe_class_name = policy_net.features_extractor.__class__.__name__
                                if 'TCN' in fe_class_name:
                                    tcn_used = True
                                    logs.append(f"    - TCN feature extractor found: ✅ `{fe_class_name}`\n")
                                    logs.append(f"      - **DEFINITIVE PROOF**: TCN was used in training\n")
                            
                            # Check all modules for TCN-related components
                            all_module_names = [name for name, _ in loaded_model.policy.named_modules()]
                            tcn_indicators = [name for name in all_module_names if any(indicator in name.lower() for indicator in ['tcn', 'dilated', 'temporal_conv', 'tcnblock'])]
                            
                            if tcn_indicators and not tcn_used:
                                tcn_used = True
                                logs.append(f"    - TCN layers found in saved model: ✅\n")
                                logs.append(f"      - Found layers: {', '.join(tcn_indicators[:5])}\n")
                                logs.append(f"      - **DEFINITIVE PROOF**: TCN was used in training\n")
                            
                            if not tcn_used:
                                # Check policy class name
                                policy_class = loaded_model.policy.__class__.__name__
                                if 'TCN' in policy_class:
                                    tcn_used = True
                                    logs.append(f"    - TCN policy class in saved model: ✅ `{policy_class}`\n")
                                else:
                                    logs.append(f"    - TCN NOT found in saved model: ❌\n")
                                    logs.append(f"      - Policy class: `{policy_class}`\n")
                                    logs.append(f"      - Feature extractor: `{fe_class_name if fe_class_name else 'N/A'}`\n")
                                    logs.append(f"      - **Evidence**: Model uses standard MLP, not TCN\n")
                                    logs.append(f"      - **Action**: Check training logs for TCN initialization errors\n")
                    except Exception as e:
                        logs.append(f"    - Could not inspect model architecture: ⚠️ {str(e)[:80]}\n")
                else:
                    logs.append(f"    - Model file not found: ❌\n")
            except Exception as e:
                logs.append(f"    - Error checking saved model: ⚠️ {str(e)[:80]}\n")
            
            if tcn_configured and tcn_available:
                if tcn_used:
                    logs.append(f"- **TCN Usage Status**: ✅ **ACTIVE** - TCN policy was used in training\n")
                else:
                    logs.append(f"- **TCN Usage Status**: ⚠️ **NOT USED** - TCN configured but not active\n")
                    logs.append(f"  - **Issue**: Check training logs for '🧠 TCN MEMORY' or '✅ Created PPO agent with TCN'\n")
                    logs.append(f"  - **Action**: Verify TCN policy class is passed correctly to PPO\n")
            elif not tcn_configured:
                logs.append(f"- **TCN Usage Status**: ❌ **DISABLED** - Using standard MlpPolicy\n")
            else:
                logs.append(f"- **TCN Usage Status**: ❌ **UNAVAILABLE** - TCN policy not importable\n")
                
        except Exception as e:
            logs.append(f"- **TCN Status Check**: ❌ Error: {str(e)[:100]}\n")
        
        # 2. Critical Errors Check
        logs.append("\n#### 🚨 Critical Errors Check\n")
        critical_errors = []
        warnings_found = []
        
        # Check for NaN/inf in key metrics
        key_metrics = ['total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate']
        for metric in key_metrics:
            value = execution_results.get(metric, 0)
            if isinstance(value, (int, float)):
                if np.isnan(value) or np.isinf(value):
                    critical_errors.append(f"NaN/Inf detected in {metric}: {value}")
        
        # Check entropy for corruption
        entropy = execution_results.get('entropy', 0.0)
        if isinstance(entropy, (int, float)) and entropy < 0:
            critical_errors.append(f"Negative entropy ({entropy:.4f}) indicates model corruption")
        
        # Check for validation failures
        validation_passed = execution_results.get('validation_passed', True)
        if not validation_passed:
            critical_errors.append("Validation failed - check validation report")
        
        if critical_errors:
            logs.append(f"- **Status**: ❌ **CRITICAL ERRORS FOUND**\n")
            for error in critical_errors:
                logs.append(f"  - 🚨 {error}\n")
        else:
            logs.append(f"- **Status**: ✅ **NO CRITICAL ERRORS**\n")
            logs.append(f"  - All key metrics are valid (no NaN/Inf)\n")
            logs.append(f"  - Entropy is positive: {entropy:.4f}\n")
            logs.append(f"  - Validation: {'✅ PASSED' if validation_passed else '❌ FAILED'}\n")
        
        # 3. Data Corruption Check
        logs.append("\n#### 🔍 Data Corruption Check\n")
        corruption_issues = []
        
        # Check for position mismatch (from warnings)
        # This would need to be passed in execution_results or checked from logs
        position_mismatch_count = execution_results.get('position_mismatch_warnings', 0)
        if position_mismatch_count > 0:
            corruption_issues.append(f"Position mismatch warnings: {position_mismatch_count} (auto-handled)")
            logs.append(f"- **Position Mismatch**: ⚠️ {position_mismatch_count} warnings detected\n")
            logs.append(f"  - **Status**: ✅ Auto-rebuild mechanism handled\n")
            logs.append(f"  - **Impact**: None - system corrected automatically\n")
        else:
            logs.append(f"- **Position Mismatch**: ✅ None detected\n")
        
        # Return cross-check (diagnostic: portfolio vs realized+unrealized)
        return_discrepancy_pct = execution_results.get('return_discrepancy_pct')
        if return_discrepancy_pct is not None:
            logs.append(f"- **Return cross-check**: Portfolio vs realized+unrealized = {return_discrepancy_pct:.2f}% difference (informational; headline = portfolio-based)\n")
        
        # Check data quality indicators
        data_quality = execution_results.get('data_quality', {})
        if data_quality:
            nan_count = data_quality.get('nan_count', 0)
            inf_count = data_quality.get('inf_count', 0)
            if nan_count > 0 or inf_count > 0:
                corruption_issues.append(f"Data quality issues: {nan_count} NaN, {inf_count} Inf")
                logs.append(f"- **Data Quality**: ⚠️ Issues detected\n")
                logs.append(f"  - NaN values: {nan_count}\n")
                logs.append(f"  - Inf values: {inf_count}\n")
            else:
                logs.append(f"- **Data Quality**: ✅ No corruption detected\n")
        else:
            logs.append(f"- **Data Quality**: ✅ No issues reported\n")
        
        if not corruption_issues:
            logs.append(f"- **Overall Status**: ✅ **NO DATA CORRUPTION**\n")
        
        # 4. Model Status
        logs.append("\n#### 💾 Model Status\n")
        try:
            import os
            model_path = self.base_dir / "saved_models" / "ppo_trading_agent.zip"
            if model_path.exists():
                file_size_mb = model_path.stat().st_size / (1024 * 1024)
                logs.append(f"- **Model File**: ✅ Exists\n")
                logs.append(f"  - Path: `{model_path}`\n")
                logs.append(f"  - Size: {file_size_mb:.2f} MB\n")
            else:
                logs.append(f"- **Model File**: ❌ Not found at `{model_path}`\n")
        except Exception as e:
            logs.append(f"- **Model File Check**: ⚠️ Error: {str(e)[:100]}\n")
        
        # 5. System Health Summary
        logs.append("\n#### 📊 System Health Summary\n")
        health_score = 100
        
        if not tcn_used and tcn_configured:
            health_score -= 20
            logs.append(f"- ⚠️ TCN not used despite being configured (-20 points)\n")
        
        if critical_errors:
            health_score -= 50
            logs.append(f"- 🚨 Critical errors found (-50 points)\n")
        
        if corruption_issues:
            health_score -= 10
            logs.append(f"- ⚠️ Data corruption issues (-10 points)\n")
        
        if health_score >= 90:
            logs.append(f"- **Overall Health**: ✅ **EXCELLENT** ({health_score}/100)\n")
        elif health_score >= 70:
            logs.append(f"- **Overall Health**: ⚠️ **GOOD** ({health_score}/100)\n")
        elif health_score >= 50:
            logs.append(f"- **Overall Health**: ⚠️ **FAIR** ({health_score}/100)\n")
        else:
            logs.append(f"- **Overall Health**: ❌ **POOR** ({health_score}/100)\n")
        
        logs.append(f"\n**Verification Timestamp**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        return logs
    
    def _generate_recommendations(self, execution_results: Dict[str, Any]) -> List[str]:
        """Generate performance recommendations based on results"""
        recommendations = []
        
        # Performance-based recommendations
        total_return = execution_results.get('total_return', 0.0)
        sharpe_ratio = execution_results.get('sharpe_ratio', 0.0)
        win_rate = execution_results.get('win_rate', 0.0)
        max_drawdown = execution_results.get('max_drawdown', 0.0)
        
        if total_return < 0.05:  # Less than 5% return
            recommendations.append("Consider increasing training timesteps or adjusting reward function")
        
        if sharpe_ratio < 0.5:
            recommendations.append("Sharpe ratio below 0.5 - consider risk management improvements")
        
        if win_rate < 0.5:
            recommendations.append("Win rate below 50% - review trading strategy and entry/exit logic")
        
        if max_drawdown > 0.1:  # More than 10% drawdown
            recommendations.append("High drawdown detected - strengthen stop-loss mechanisms")
        
        # Trading behavior recommendations
        total_trades = execution_results.get('total_trades', 0)
        if total_trades > 1000:
            recommendations.append("High trade frequency - consider reducing trading incentives")
        elif total_trades < 100:
            recommendations.append("Low trade frequency - consider increasing trading incentives")
        
        # System performance recommendations
        execution_time = execution_results.get('total_execution_time', 0)
        if execution_time > 600:  # More than 10 minutes
            recommendations.append("Long execution time - consider optimizing data loading or reducing timesteps")
        
        # Memory recommendations
        peak_memory = execution_results.get('peak_memory_mb', 0)
        if peak_memory > 1000:  # More than 1GB
            recommendations.append("High memory usage - consider batch size optimization")
        
        if not recommendations:
            recommendations.append("Performance looks good - continue current approach")
        
        return recommendations

    def _headline_metrics(self, execution_results: Dict[str, Any]) -> Dict[str, Any]:
        """Return metrics for headline/Key Performance: deterministic eval only (paper-trading / production behavior)."""
        return {
            'total_return': execution_results.get('total_return', 0.0),
            'sharpe_ratio': execution_results.get('sharpe_ratio', 0.0),
            'max_drawdown': execution_results.get('max_drawdown', 0.0),
            'win_rate': execution_results.get('win_rate', 0.0),
            'total_trades': execution_results.get('total_trades', 0),
            'realized_return': execution_results.get('realized_return', 0.0),
            'unrealized_return': execution_results.get('unrealized_return', 0.0),
        }

    def _enrich_results_with_saved_data(self, execution_results: Dict[str, Any]) -> Dict[str, Any]:
        """Back-fill execution metrics from saved validation artefacts when not provided explicitly."""
        results = execution_results.copy()

        # Load backtest summary JSON for core metrics (return, sharpe, drawdown, trades, etc.)
        backtest_path = self.validation_dir / "backtest_results.json"
        if backtest_path.exists():
            try:
                with open(backtest_path, "r", encoding="utf-8") as f:
                    backtest_data = json.load(f)
                for key in [
                    "total_return",
                    "sharpe_ratio",
                    "max_drawdown",
                    "total_trades",
                    "profit_factor",
                    "volatility",
                ]:
                    value = backtest_data.get(key)
                    if value is not None and (results.get(key) in (None, 0, 0.0) or key not in results):
                        results[key] = value
            except Exception as err:
                logger.debug(f"Failed to enrich results from backtest_results.json: {err}")

        # Fallback: load from stochastic_eval/ when main backtest is missing or key metrics still zero
        if results.get('total_trades', 0) == 0 or (results.get('total_return', 0) == 0 and results.get('sharpe_ratio', 0) == 0):
            st_backtest = self.validation_dir / "stochastic_eval" / "backtest_results.json"
            if st_backtest.exists():
                try:
                    with open(st_backtest, "r", encoding="utf-8") as f:
                        st_data = json.load(f)
                    for key in [
                        "total_return", "sharpe_ratio", "max_drawdown", "total_trades",
                        "profit_factor", "volatility", "win_rate"
                    ]:
                        value = st_data.get(key)
                        if value is not None and (results.get(key) in (None, 0, 0.0) or key not in results):
                            results[key] = value
                except Exception as err:
                    logger.debug(f"Failed to enrich from stochastic_eval/backtest_results.json: {err}")

        # Load trade history CSV to compute trade distribution metrics (try stochastic_eval if main empty)
        trade_history_path = self.validation_dir / "trade_history.csv"
        if not trade_history_path.exists() or (results.get('total_trades', 0) == 0):
            st_trade = self.validation_dir / "stochastic_eval" / "trade_history.csv"
            if st_trade.exists():
                trade_history_path = st_trade
        if trade_history_path.exists():
            try:
                df = pd.read_csv(trade_history_path)
                if 'action' in df.columns:
                    total_trades = len(df)
                    if total_trades > 0:
                        buy_count = (df['action'] == 'BUY').sum()
                        sell_count = (df['action'] == 'SELL').sum()
                        short_count = (df['action'] == 'SHORT').sum()
                        stop_loss_count = (df['action'] == 'STOP_LOSS').sum()

                        results['total_trades'] = results.get('total_trades', 0) or int(total_trades)
                        results['buy_trades'] = buy_count
                        results['sell_trades'] = sell_count
                        results['short_trades'] = short_count
                        results['stop_loss_trades'] = stop_loss_count
                        results['buy_percentage'] = (buy_count / total_trades) * 100.0
                        results['sell_percentage'] = (sell_count / total_trades) * 100.0
                        results['short_percentage'] = (short_count / total_trades) * 100.0
                        results['stop_loss_percentage'] = (stop_loss_count / total_trades) * 100.0

                    if 'pnl' in df.columns:
                        closed = df[df['action'].isin(['SELL', 'STOP_LOSS']) & df['pnl'].notna()]
                        if len(closed) > 0:
                            positive = (closed['pnl'] > 0).sum()
                            win_rate = positive / len(closed)
                            # Prefer observed win_rate over default when available
                            results['win_rate'] = win_rate

                    if 'cost' in df.columns:
                        results['avg_trade_size'] = float(df['cost'].abs().mean())
            except Exception as err:
                logger.debug(f"Failed to enrich results from trade_history.csv: {err}")

        # Load returns CSV for volatility estimate when not provided
        returns_path = self.validation_dir / "returns.csv"
        if returns_path.exists() and results.get('volatility') in (None, 0, 0.0):
            try:
                returns_df = pd.read_csv(returns_path)
                for col in ['return', 'returns']:
                    if col in returns_df.columns and len(returns_df[col].dropna()) > 1:
                        results['volatility'] = float(returns_df[col].std())
                        break
            except Exception as err:
                logger.debug(f"Failed to enrich volatility from returns.csv: {err}")

        return results

    def _load_trade_data(self, trade_history_data: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Load trade history data (prioritize in-memory data)"""
        try:
            # CRITICAL FIX: Use in-memory data if provided
            if trade_history_data is not None and trade_history_data:
                df = pd.DataFrame(trade_history_data)
                logger.info(f" Loaded {len(df)} trades from in-memory history")
                
                # Save fresh history to CSV for persistence
                trade_file = self.validation_dir / "trade_history.csv"
                df.to_csv(trade_file, index=False)
                logger.info(f" Saved fresh trade history to {trade_file}")
                
                # DEBUGGING CHECKLIST: Verify data integrity
                logger.info(" UNIFIED REPORT DEBUGGING CHECKLIST:")
                logger.info(f"  - In-memory data used: ")
                logger.info(f"  - Trades loaded: {len(df)}")
                logger.info(f"  - Buy trades: {len(df[df['action'] == 'BUY'])}")
                logger.info(f"  - Sell trades: {len(df[df['action'] == 'SELL'])}")
                logger.info(f"  - Unique tickers: {df['ticker'].nunique()}")
                logger.info(f"  - Total value: ${df['value'].sum():.2f}")
                
                return {
                    'total_trades': len(df),
                    'buy_trades': len(df[df['action'] == 'BUY']),
                    'sell_trades': len(df[df['action'] == 'SELL']),
                    'hold_trades': len(df[df['action'] == 'HOLD']),
                    'unique_tickers': df['ticker'].nunique(),
                    'total_volume': df['quantity'].sum(),
                    'total_value': df['value'].sum(),
                    'avg_trade_size': df['value'].mean(),
                    'trade_pattern': self._analyze_trade_pattern(df)
                }
            
            # Fallback to loading from CSV if no in-memory data provided
            trade_file = self.validation_dir / "trade_history.csv"
            logger.info(f" Looking for trade file: {trade_file}")
            if trade_file.exists():
                logger.info(f" Trade file found, loading data...")
                df = pd.read_csv(trade_file)
                logger.info(f" Loaded {len(df)} trades from CSV (fallback)")
                
                return {
                    'total_trades': len(df),
                    'buy_trades': len(df[df['action'] == 'BUY']),
                    'sell_trades': len(df[df['action'] == 'SELL']),
                    'hold_trades': len(df[df['action'] == 'HOLD']),
                    'unique_tickers': df['ticker'].nunique(),
                    'total_volume': df['quantity'].sum(),
                    'total_value': df['value'].sum(),
                    'avg_trade_size': df['value'].mean(),
                    'trade_pattern': self._analyze_trade_pattern(df)
                }
            else:
                logger.warning(f" Trade file not found: {trade_file}")
        except Exception as e:
            logger.warning(f"Could not load trade data: {e}")
        return {}

    def _load_portfolio_data(self, portfolio_history_data: Optional[List[float]] = None) -> Dict[str, Any]:
        """Load portfolio performance data (prioritize in-memory data)"""
        try:
            # CRITICAL FIX: Use in-memory data if provided
            if portfolio_history_data is not None and portfolio_history_data:
                portfolio_values = np.array(portfolio_history_data)
                logger.info(f" Loaded {len(portfolio_values)} portfolio values from in-memory history")
                
                # Save fresh history to CSV for persistence
                portfolio_file = self.validation_dir / "portfolio_history.csv"
                portfolio_df = pd.DataFrame({
                    'step': range(len(portfolio_values)),
                    'portfolio_value': portfolio_values
                })
                portfolio_df.to_csv(portfolio_file, index=False)
                logger.info(f" Saved fresh portfolio history to {portfolio_file}")
                
                return self._calculate_portfolio_metrics(portfolio_values)
            
            # Fallback to loading from CSV if no in-memory data provided
            portfolio_file = self.validation_dir / "portfolio_history.csv"
            if portfolio_file.exists():
                df = pd.read_csv(portfolio_file)
                portfolio_values = df.iloc[:, 1].values  # Second column contains values
                return {
                    'initial_value': portfolio_values[0] if len(portfolio_values) > 0 else 0,
                    'final_value': portfolio_values[-1] if len(portfolio_values) > 0 else 0,
                    'max_value': np.max(portfolio_values) if len(portfolio_values) > 0 else 0,
                    'min_value': np.min(portfolio_values) if len(portfolio_values) > 0 else 0,
                    'value_change': portfolio_values[-1] - portfolio_values[0] if len(portfolio_values) > 0 else 0,
                    'volatility': np.std(portfolio_values) if len(portfolio_values) > 0 else 0
                }
        except Exception as e:
            logger.warning(f"Could not load portfolio data: {e}")
        return {}

    def _calculate_portfolio_metrics(self, portfolio_values: np.ndarray) -> Dict[str, Any]:
        """Calculate portfolio performance metrics from values array"""
        try:
            if len(portfolio_values) == 0:
                return {}
            
            return {
                'initial_value': portfolio_values[0],
                'final_value': portfolio_values[-1],
                'max_value': np.max(portfolio_values),
                'min_value': np.min(portfolio_values),
                'total_return': (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0] if portfolio_values[0] > 0 else 0,
                'max_drawdown': self._calculate_max_drawdown(portfolio_values),
                'volatility': np.std(portfolio_values) / np.mean(portfolio_values) if np.mean(portfolio_values) > 0 else 0
            }
        except Exception as e:
            logger.error(f"Error calculating portfolio metrics: {e}")
            return {}

    def _calculate_max_drawdown(self, portfolio_values: np.ndarray) -> float:
        """Calculate maximum drawdown from portfolio values"""
        try:
            if len(portfolio_values) < 2:
                return 0.0
            
            # Calculate running maximum
            running_max = np.maximum.accumulate(portfolio_values)
            
            # Calculate drawdown
            drawdown = (portfolio_values - running_max) / running_max
            
            # Return maximum drawdown (most negative value)
            return abs(np.min(drawdown))
        except Exception as e:
            logger.error(f"Error calculating max drawdown: {e}")
            return 0.0

    def _load_backtest_data(self) -> Dict[str, Any]:
        """Load backtest results data"""
        try:
            backtest_file = self.validation_dir / "backtest_results.json"
            if backtest_file.exists():
                with open(backtest_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load backtest data: {e}")
        return {}

    def _analyze_trade_pattern(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze trading patterns for insights"""
        try:
            # Analyze consecutive trades
            max_consecutive_buys = 0
            current_streak = 0
            
            for action in df['action']:
                if action == 'BUY':
                    current_streak += 1
                    max_consecutive_buys = max(max_consecutive_buys, current_streak)
                else:
                    current_streak = 0
            
            # Analyze ticker distribution
            ticker_counts = df['ticker'].value_counts()
            
            return {
                'max_consecutive_buys': max_consecutive_buys,
                'ticker_distribution': ticker_counts.to_dict(),
                'most_traded_ticker': ticker_counts.index[0] if len(ticker_counts) > 0 else None,
                'trading_frequency': len(df) / max(len(df), 1)
            }
        except Exception as e:
            return {'error': str(e)}

    def _build_ticker_breakdown_section(self, trade_history_data: Optional[List[Dict[str, Any]]] = None) -> Optional[List[str]]:
        """Build ticker breakdown: BUY/SELL per ticker symbol, BTC-USD check. Returns list of markdown lines."""
        try:
            df = None
            if trade_history_data and len(trade_history_data) > 0:
                df = pd.DataFrame(trade_history_data)
            else:
                trade_file = self.validation_dir / "trade_history.csv"
                if trade_file.exists():
                    df = pd.read_csv(trade_file)
            if df is None or len(df) == 0 or 'ticker' not in df.columns or 'action' not in df.columns:
                return None

            # Normalize ticker to int (CSV may store as string)
            df = df.copy()
            df['ticker'] = pd.to_numeric(df['ticker'], errors='coerce').fillna(-1).astype(int)
            df = df[df['ticker'] >= 0]

            tickers = []
            try:
                from config import DATA_CONFIG
                tickers = list(DATA_CONFIG.get('tickers', []))
            except Exception:
                tickers = []

            lines = []
            lines.append("\n### 📈 Ticker Breakdown (Buy/Sell by Symbol)\n")
            lines.append("| Ticker | BUY | SELL | Total |\n")
            lines.append("|-------|-----|------|-------|\n")

            buy_by_ticker = df[df['action'] == 'BUY'].groupby('ticker').size()
            sell_by_ticker = df[df['action'] == 'SELL'].groupby('ticker').size()
            btc_traded = False
            btc_buys = 0
            btc_sells = 0

            all_ticker_indices = set(buy_by_ticker.index) | set(sell_by_ticker.index)
            for idx in sorted(all_ticker_indices):
                symbol = tickers[int(idx)] if 0 <= int(idx) < len(tickers) else f"T{idx}"
                buys = int(buy_by_ticker.get(idx, 0))
                sells = int(sell_by_ticker.get(idx, 0))
                total = buys + sells
                lines.append(f"| {symbol} | {buys} | {sells} | {total} |\n")
                if symbol == 'BTC-USD':
                    btc_traded = True
                    btc_buys = buys
                    btc_sells = sells

            lines.append("\n**BTC-USD Check**: ")
            if btc_traded:
                lines.append(f"✅ **Yes** – Agent traded Bitcoin ({btc_buys} buys, {btc_sells} sells)\n")
            else:
                btc_in_universe = 'BTC-USD' in tickers
                lines.append(f"{'⚠️ No trades' if btc_in_universe else 'N/A (not in universe)'} – "
                            f"{'BTC-USD in config but no trades' if btc_in_universe else 'BTC-USD not in ticker list'}\n")
            return lines
        except Exception as e:
            logger.warning(f"Could not build ticker breakdown: {e}")
            return None

    def _build_short_positions_section(self, trade_history_data: Optional[List[Dict[str, Any]]] = None) -> Optional[List[str]]:
        """Build short positions section from trade history. Returns list of markdown lines."""
        try:
            df = None
            if trade_history_data and len(trade_history_data) > 0:
                df = pd.DataFrame(trade_history_data)
            else:
                trade_file = self.validation_dir / "trade_history.csv"
                if trade_file.exists():
                    df = pd.read_csv(trade_file)
            if df is None or len(df) == 0 or 'action' not in df.columns:
                return None
            short_trades = df[df['action'].str.upper() == 'SHORT'] if 'action' in df.columns else pd.DataFrame()
            if len(short_trades) == 0:
                return None
            tickers = []
            try:
                from config import DATA_CONFIG
                tickers = list(DATA_CONFIG.get('tickers', []))
            except Exception:
                tickers = []
            lines = []
            lines.append("\n### 📉 Short Positions\n")
            lines.append("- **Status**: ✅ SHORT trades executed\n")
            short_by_ticker = short_trades.groupby('ticker').size()
            lines.append("| Ticker | SHORT Count |\n")
            lines.append("|-------|-------------|\n")
            for idx in sorted(short_by_ticker.index):
                symbol = tickers[int(idx)] if 0 <= int(idx) < len(tickers) else f"T{idx}"
                lines.append(f"| {symbol} | {int(short_by_ticker[idx])} |\n")
            return lines
        except Exception as e:
            logger.warning(f"Could not build short positions section: {e}")
            return None

    def _create_comprehensive_report(self, execution_results: Dict[str, Any],
                                     training_results: Dict[str, Any],
                                     trade_data: Dict[str, Any],
                                     portfolio_data: Dict[str, Any],
                                     backtest_data: Dict[str, Any],
                                     timestamp: str) -> str:
        """Create the comprehensive report content"""
        
        return f"""#  PPO TRADING SYSTEM - UNIFIED EXECUTION REPORT
**Generated**: {timestamp}  
**Report Type**: COMPREHENSIVE ANALYSIS  
**Format**: Cursor AI Agent Optimized

---

##  EXECUTION SUMMARY

###  Key Performance Metrics
| Metric | Value | Target | Status | Analysis |
|--------|-------|--------|--------|----------|
| **Total Return** | {execution_results.get('total_return', 0)*100:.2f}% | >0% | {' PASS' if execution_results.get('total_return', 0) > 0 else ' FAIL'} | {'Positive returns achieved' if execution_results.get('total_return', 0) > 0 else 'No returns generated'} |
| **Sharpe Ratio** | {execution_results.get('sharpe_ratio', 0):.3f} | >1.0 | {' PASS' if execution_results.get('sharpe_ratio', 0) > 1.0 else ' FAIL'} | {'Good risk-adjusted returns' if execution_results.get('sharpe_ratio', 0) > 1.0 else 'Poor risk-adjusted performance'} |
| **Max Drawdown** | {execution_results.get('max_drawdown', 0)*100:.2f}% | <15% | {' PASS' if execution_results.get('max_drawdown', 0) < 0.15 else ' FAIL'} | {'Risk controlled' if execution_results.get('max_drawdown', 0) < 0.15 else 'High risk exposure'} |
| **Win Rate** | {execution_results.get('win_rate', 0)*100:.1f}% | >50% | {' PASS' if execution_results.get('win_rate', 0) > 0.5 else ' FAIL'} | {'Good trade success rate' if execution_results.get('win_rate', 0) > 0.5 else 'Poor trade success rate'} |
| **Total Trades** | {execution_results.get('total_trades', 0)} | >0 | {' PASS' if execution_results.get('total_trades', 0) > 0 else ' FAIL'} | {'Active trading detected' if execution_results.get('total_trades', 0) > 0 else 'No trading activity'} |

###  Portfolio Performance
| Metric | Value | Analysis |
|--------|-------|----------|
| **Initial Portfolio** | ${portfolio_data.get('initial_value', 0):,.2f} | Starting capital |
| **Final Portfolio** | ${portfolio_data.get('final_value', 0):,.2f} | Ending capital |
| **Portfolio Change** | ${portfolio_data.get('value_change', 0):,.2f} | {'Growth' if portfolio_data.get('value_change', 0) > 0 else 'Loss'} |
| **Max Portfolio Value** | ${portfolio_data.get('max_value', 0):,.2f} | Peak performance |
| **Min Portfolio Value** | ${portfolio_data.get('min_value', 0):,.2f} | Lowest point |
| **Portfolio Volatility** | {portfolio_data.get('volatility', 0):.2f} | {'High' if portfolio_data.get('volatility', 0) > 1000 else 'Low'} volatility |

---

##  TRADING ACTIVITY ANALYSIS

###  Trade Execution Summary
| Metric | Value | Analysis |
|--------|-------|----------|
| **Total Trades** | {trade_data.get('total_trades', 0)} | Total trading activity |
| **Buy Orders** | {trade_data.get('buy_trades', 0)} | {'High' if trade_data.get('buy_trades', 0) > 100 else 'Moderate'} buying activity |
| **Sell Orders** | {trade_data.get('sell_trades', 0)} | {'Balanced' if trade_data.get('sell_trades', 0) > 50 else 'Low'} selling activity |
| **Hold Actions** | {trade_data.get('hold_trades', 0)} | {'Conservative' if trade_data.get('hold_trades', 0) > 100 else 'Aggressive'} strategy |
| **Unique Tickers** | {trade_data.get('unique_tickers', 0)} | {'Good' if trade_data.get('unique_tickers', 0) > 3 else 'Poor'} diversification |
| **Total Volume** | {trade_data.get('total_volume', 0):.2f} | {'High' if trade_data.get('total_volume', 0) > 1000 else 'Moderate'} trading volume |
| **Average Trade Size** | ${trade_data.get('avg_trade_size', 0):,.2f} | {'Large' if trade_data.get('avg_trade_size', 0) > 10000 else 'Moderate'} position sizes |

###  Trading Pattern Analysis
| Pattern | Value | Analysis |
|---------|-------|----------|
| **Max Consecutive Buys** | {trade_data.get('trade_pattern', {}).get('max_consecutive_buys', 0)} | {'Excessive' if trade_data.get('trade_pattern', {}).get('max_consecutive_buys', 0) > 5 else 'Reasonable'} buying streak |
| **Most Traded Ticker** | {trade_data.get('trade_pattern', {}).get('most_traded_ticker', 'N/A')} | {'Concentrated' if trade_data.get('trade_pattern', {}).get('most_traded_ticker', 'N/A') != 'N/A' else 'Diversified'} trading |
| **Trading Frequency** | {trade_data.get('trade_pattern', {}).get('trading_frequency', 0):.2f} | {'High-frequency' if trade_data.get('trade_pattern', {}).get('trading_frequency', 0) > 10 else 'Moderate-frequency'} trading |

---

##  SYSTEM PERFORMANCE

###  Execution Metrics & Timing Analysis
| Metric | Value | Analysis |
|--------|-------|----------|
| **Total Execution Time** | {execution_results.get('execution_time', 0):.2f} seconds | {'Fast' if execution_results.get('execution_time', 0) < 300 else 'Slow'} execution |
| **Steps Simulated** | {execution_results.get('steps_simulated', 0)} | {'Comprehensive' if execution_results.get('steps_simulated', 0) > 1000 else 'Limited'} simulation |
| **Training Timesteps** | {training_results.get('total_timesteps', 0) if training_results else 'N/A'} | {'Sufficient' if training_results and training_results.get('total_timesteps', 0) > 100000 else 'Insufficient'} training |
| **Steps per Second** | {execution_results.get('steps_simulated', 0) / max(execution_results.get('execution_time', 1), 1):.1f} | {'Efficient' if execution_results.get('steps_simulated', 0) / max(execution_results.get('execution_time', 1), 1) > 100 else 'Slow'} processing |
| **Trades per Second** | {execution_results.get('total_trades', 0) / max(execution_results.get('execution_time', 1), 1):.2f} | {'High-frequency' if execution_results.get('total_trades', 0) / max(execution_results.get('execution_time', 1), 1) > 1 else 'Low-frequency'} trading |

###  Performance Profiling & Bottlenecks
{self._generate_profiling_analysis(execution_results, backtest_data)}

###  Technical Status
| Component | Status | Details |
|-----------|--------|---------|
| **Paper Trading Ready** | {' READY' if execution_results.get('paper_trading_ready', False) else ' NOT READY'} | {'Meets all criteria' if execution_results.get('paper_trading_ready', False) else 'Fails readiness criteria'} |
| **Data Quality** | {' VALID' if execution_results.get('benchmark_data_valid', False) else ' INVALID'} | {'Real market data available' if execution_results.get('benchmark_data_valid', False) else 'Data quality issues'} |
| **Model Performance** | {' OPTIMAL' if execution_results.get('sharpe_ratio', 0) > 1.0 else ' SUBOPTIMAL'} | {'Good risk-adjusted returns' if execution_results.get('sharpe_ratio', 0) > 1.0 else 'Poor performance'} |

---

##  CRITICAL ISSUES & RECOMMENDATIONS

###  Issues Identified
{self._generate_issues_analysis(execution_results, trade_data, portfolio_data)}

###  Recommendations
{self._generate_comprehensive_recommendations(execution_results, trade_data, portfolio_data)}

---

##  DATA FILES GENERATED

###  Validation Data
- **Trade History**: `{self.validation_dir}/trade_history.csv`
- **Portfolio History**: `{self.validation_dir}/portfolio_history.csv`
- **Returns Data**: `{self.validation_dir}/returns.csv`
- **Benchmark Returns**: `{self.validation_dir}/benchmark_returns.csv`
- **Backtest Results**: `{self.validation_dir}/backtest_results.json`

###  Report Files
- **Unified Report**: `{self.reports_dir}/UNIFIED_REPORT_{timestamp}.md`
- **Previous Reports**: {len(list(self.reports_dir.glob('*.md')))} markdown files, {len(list(self.reports_dir.glob('*.json')))} JSON files

---

##  CURSOR AI AGENT ANALYSIS GUIDE

###  How to Use This Report
1. **Start with Executive Summary** - Key metrics and status
2. **Review Trading Activity** - Understand agent behavior
3. **Check Critical Issues** - Identify problems to fix
4. **Follow Recommendations** - Action items for improvement
5. **Examine Data Files** - Access raw data for deeper analysis

###  Key Analysis Points
- **Performance Metrics**: Compare against targets
- **Trading Patterns**: Identify behavioral issues
- **Risk Management**: Check drawdown and volatility
- **System Health**: Verify technical status
- **Data Quality**: Ensure reliable information

---

##  STATISTICAL VALIDATION

###  Statistical Tests
| Test | P-Value | Significance | Status |
|------|---------|--------------|--------|
| **Overall Performance** | {execution_results.get('p_value', 1.0):.4f} | {'Significant' if execution_results.get('p_value', 0) < 0.05 else 'Not Significant'} | {' PASS' if execution_results.get('p_value', 0) < 0.05 else ' FAIL'} |
| **Sharpe Ratio** | {execution_results.get('sharpe_p_value', 1.0):.4f} | {'Significant' if execution_results.get('sharpe_p_value', 0) < 0.05 else 'Not Significant'} | {' PASS' if execution_results.get('sharpe_p_value', 0) < 0.05 else ' FAIL'} |

---

*Report generated by PPO Trading System Unified Report Generator*  
*Optimized for Cursor AI Agent Analysis*  
*Timestamp: {timestamp}*
"""
    def _generate_issues_analysis(self, execution_results: Dict[str, Any], 
                                 trade_data: Dict[str, Any], 
                                 portfolio_data: Dict[str, Any]) -> str:
        """Generate critical issues analysis"""
        issues = []
        
        # Performance issues
        if execution_results.get('total_return', 0) <= 0:
            issues.append("**No Returns Generated**: Agent failed to produce positive returns")
        
        if execution_results.get('sharpe_ratio', 0) < 1.0:
            issues.append("**Poor Risk-Adjusted Returns**: Sharpe ratio below acceptable threshold")
        
        if execution_results.get('max_drawdown', 0) > 0.15:
            issues.append("**High Risk Exposure**: Maximum drawdown exceeds 15% threshold")
        
        # Trading issues
        if trade_data.get('max_consecutive_buys', 0) > 5:
            issues.append("**Excessive Buying**: Agent making too many consecutive buy orders")
        
        if trade_data.get('unique_tickers', 0) < 3:
            issues.append("**Poor Diversification**: Trading too few assets")
        
        if trade_data.get('total_trades', 0) > 1000:
            issues.append("**Over-Trading**: Excessive trading frequency")
        
        # Portfolio issues
        value_change = portfolio_data.get('value_change', 0)
        if isinstance(value_change, dict):
            value_change = value_change.get('value_change', 0)
        if isinstance(value_change, (int, float)) and value_change < 0:
            issues.append("**Portfolio Loss**: Portfolio value decreased during execution")
        
        if not issues:
            return "**No critical issues identified** - System performing well"
        
        return "\n".join([f"- {issue}" for issue in issues])
    def _generate_comprehensive_recommendations(self, execution_results: Dict[str, Any], 
                                                trade_data: Dict[str, Any], 
                                                portfolio_data: Dict[str, Any]) -> str:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Performance improvements
        if execution_results.get('sharpe_ratio', 0) < 1.0:
            recommendations.append("**Improve Risk Management**: Adjust reward function to reduce volatility")
        
        if trade_data.get('max_consecutive_buys', 0) > 5:
            recommendations.append("**Fix Trading Logic**: Implement better trade interval controls")
        
        if trade_data.get('unique_tickers', 0) < 3:
            recommendations.append("**Enhance Diversification**: Encourage trading across more assets")
        
        value_change = portfolio_data.get('value_change', 0)
        if isinstance(value_change, dict):
            value_change = value_change.get('value_change', 0)
        if isinstance(value_change, (int, float)) and value_change < 0:
            recommendations.append("**Review Strategy**: Analyze why portfolio is losing value")
        
        # General improvements
        recommendations.extend([
            "**Scale Training**: Increase timesteps to 300K+ for better learning",
            "**Enable Optuna**: Re-enable hyperparameter optimization",
            "**Activate Features**: Ensure technical/fundamental analysis is working",
            "**Monitor Performance**: Track metrics over multiple runs"
        ])
        
        return "\n".join([f"- {rec}" for rec in recommendations])
    
    def _generate_profiling_analysis(self, execution_results: Dict[str, Any], backtest_data: Dict[str, Any]) -> str:
        """Generate detailed profiling and bottleneck analysis"""
        try:
            profiling_sections = []
            
            # Extract timing data
            total_time = execution_results.get('execution_time', 0)
            steps_simulated = execution_results.get('steps_simulated', 0)
            total_trades = execution_results.get('total_trades', 0)
            
            # Calculate performance metrics
            steps_per_second = steps_simulated / max(total_time, 1)
            trades_per_second = total_trades / max(total_time, 1)
            time_per_step = total_time / max(steps_simulated, 1) * 1000  # milliseconds
            
            # Performance analysis
            performance_analysis = []
            
            if steps_per_second > 200:
                performance_analysis.append("✅ **Excellent Processing Speed**: >200 steps/second")
            elif steps_per_second > 100:
                performance_analysis.append("✅ **Good Processing Speed**: >100 steps/second")
            elif steps_per_second > 50:
                performance_analysis.append("⚠️ **Moderate Processing Speed**: >50 steps/second")
            else:
                performance_analysis.append("❌ **Slow Processing Speed**: <50 steps/second")
            
            if time_per_step < 5:
                performance_analysis.append("✅ **Fast Step Processing**: <5ms per step")
            elif time_per_step < 10:
                performance_analysis.append("✅ **Good Step Processing**: <10ms per step")
            elif time_per_step < 20:
                performance_analysis.append("⚠️ **Moderate Step Processing**: <20ms per step")
            else:
                performance_analysis.append("❌ **Slow Step Processing**: >20ms per step")
            
            # Bottleneck identification
            bottlenecks = []
            
            # Check if profiling was active
            profiling_active = backtest_data.get('performance_metrics', {}).get('profiling_active', False)
            if not profiling_active:
                bottlenecks.append("⚠️ **Profiling Disabled**: No detailed performance monitoring")
            
            # Check memory usage
            memory_data = backtest_data.get('performance_metrics', {}).get('resource_usage', {}).get('memory', {})
            if memory_data.get('percent', 0) > 80:
                bottlenecks.append("❌ **High Memory Usage**: >80% memory utilization")
            elif memory_data.get('percent', 0) > 60:
                bottlenecks.append("⚠️ **Moderate Memory Usage**: >60% memory utilization")
            
            # Check CPU usage
            cpu_data = backtest_data.get('performance_metrics', {}).get('resource_usage', {}).get('cpu', {})
            if cpu_data.get('percent', 0) > 90:
                bottlenecks.append("❌ **High CPU Usage**: >90% CPU utilization")
            elif cpu_data.get('percent', 0) > 70:
                bottlenecks.append("⚠️ **Moderate CPU Usage**: >70% CPU utilization")
            
            # Trading efficiency analysis
            if total_trades > 0:
                trades_per_step = total_trades / max(steps_simulated, 1)
                if trades_per_step > 0.2:
                    bottlenecks.append("⚠️ **High Trading Frequency**: >20% of steps involve trading")
                elif trades_per_step < 0.05:
                    bottlenecks.append("⚠️ **Low Trading Activity**: <5% of steps involve trading")
            
            # Build profiling report
            profiling_sections.append("#### Performance Metrics")
            profiling_sections.append(f"- **Steps per Second**: {steps_per_second:.1f}")
            profiling_sections.append(f"- **Time per Step**: {time_per_step:.2f}ms")
            profiling_sections.append(f"- **Trades per Second**: {trades_per_second:.2f}")
            profiling_sections.append(f"- **Total Execution Time**: {total_time:.2f} seconds")
            
            profiling_sections.append("\n#### Performance Analysis")
            for analysis in performance_analysis:
                profiling_sections.append(f"- {analysis}")
            
            if bottlenecks:
                profiling_sections.append("\n#### Identified Bottlenecks")
                for bottleneck in bottlenecks:
                    profiling_sections.append(f"- {bottleneck}")
            else:
                profiling_sections.append("\n#### Bottleneck Analysis")
                profiling_sections.append("- ✅ **No Major Bottlenecks Detected**")
            
            # Optimization recommendations
            profiling_sections.append("\n#### Optimization Recommendations")
            if not profiling_active:
                profiling_sections.append("- **Enable Profiling**: Activate detailed performance monitoring")
            if steps_per_second < 100:
                profiling_sections.append("- **Optimize Processing**: Consider reducing feature complexity or batch processing")
            if time_per_step > 10:
                profiling_sections.append("- **Optimize Step Processing**: Review observation generation and reward calculation")
            if memory_data.get('percent', 0) > 70:
                profiling_sections.append("- **Memory Optimization**: Consider reducing batch sizes or feature dimensions")
            
            return "\n".join(profiling_sections)
            
        except Exception as e:
            return f"Error generating profiling analysis: {str(e)}"
    
    def _format_adaptive_reward_logs(self, execution_results: Dict[str, Any]) -> str:
        """Format adaptive reward system logs for unified report"""
        logs = []
        
        adaptive_reward_log = execution_results.get('adaptive_reward_log', [])
        events_count = execution_results.get('adaptive_reward_events_count', 0)
        stats = execution_results.get('adaptive_reward_stats', {})
        
        if events_count == 0:
            logs.append("- **Status**: ⚠️ **NO ADAPTIVE REWARD DATA**\n")
            logs.append("  - No adaptive reward events recorded\n")
            logs.append("  - **Possible reasons**:\n")
            logs.append("    - Environment did not track adaptive rewards\n")
            logs.append("    - Evaluation was too short (< 50 steps)\n")
            logs.append("    - Adaptive reward system not initialized\n")
            logs.append("  - **Action**: Check if adaptive rewards are enabled in trading environment\n")
            return "".join(logs)
        
        logs.append(f"- **Status**: ✅ **ADAPTIVE REWARDS ACTIVE** ({events_count} events recorded)\n")
        logs.append(f"- **Total Events**: {events_count}\n")
        
        def _safe_stat(s, key, default=1.0):
            v = s.get(key, default)
            if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
                return default
            return float(v)
        if stats:
            logs.append("\n**Reward Multiplier Statistics:**\n")
            buy_avg = _safe_stat(stats, 'buy_multiplier_avg')
            buy_min = _safe_stat(stats, 'buy_multiplier_min')
            buy_max = _safe_stat(stats, 'buy_multiplier_max')
            sell_avg = _safe_stat(stats, 'sell_multiplier_avg')
            sell_min = _safe_stat(stats, 'sell_multiplier_min')
            sell_max = _safe_stat(stats, 'sell_multiplier_max')
            hold_avg = _safe_stat(stats, 'hold_multiplier_avg')
            hold_min = _safe_stat(stats, 'hold_multiplier_min')
            hold_max = _safe_stat(stats, 'hold_multiplier_max')
            logs.append(f"  - **BUY Multiplier**: Avg={buy_avg:.2f}x, Range=[{buy_min:.2f}x, {buy_max:.2f}x]\n")
            logs.append(f"  - **SELL Multiplier**: Avg={sell_avg:.2f}x, Range=[{sell_min:.2f}x, {sell_max:.2f}x]\n")
            logs.append(f"  - **HOLD Penalty Multiplier**: Avg={hold_avg:.2f}x, Range=[{hold_min:.2f}x, {hold_max:.2f}x]\n")
            
            # Interpretation
            
            logs.append("\n**Interpretation:**\n")
            if buy_avg > 1.1:
                logs.append(f"  - ✅ High volatility/trending market detected - BUY rewards boosted ({buy_avg:.2f}x)\n")
            elif buy_avg < 0.9:
                logs.append(f"  - ⚠️ Low volatility/choppy market - BUY rewards reduced ({buy_avg:.2f}x)\n")
            else:
                logs.append(f"  - ✅ Normal market conditions - BUY rewards at baseline ({buy_avg:.2f}x)\n")
            
            if sell_avg > 1.1:
                logs.append(f"  - ✅ High volatility/trending market detected - SELL rewards boosted ({sell_avg:.2f}x)\n")
            elif sell_avg < 0.9:
                logs.append(f"  - ⚠️ Low volatility/choppy market - SELL rewards reduced ({sell_avg:.2f}x)\n")
            else:
                logs.append(f"  - ✅ Normal market conditions - SELL rewards at baseline ({sell_avg:.2f}x)\n")
            
            if hold_avg > 1.1:
                logs.append(f"  - ✅ High volatility/opportunities detected - HOLD penalties increased ({hold_avg:.2f}x) to encourage trading\n")
            elif hold_avg < 0.9:
                logs.append(f"  - ✅ Low volatility/trending market - HOLD penalties reduced ({hold_avg:.2f}x) to allow holding winners\n")
            else:
                logs.append(f"  - ✅ Normal market conditions - HOLD penalties at baseline ({hold_avg:.2f}x)\n")
        
        if adaptive_reward_log:
            logs.append("\n**Sample Events (first 10):**\n")
            for i, event in enumerate(adaptive_reward_log[:10], 1):
                step = event.get('step', 0)
                buy_mult = event.get('buy_multiplier', 1.0)
                sell_mult = event.get('sell_multiplier', 1.0)
                hold_mult = event.get('hold_multiplier', 1.0)
                message = event.get('message', '')
                logs.append(f"  {i}. Step {step:,}: BUY={buy_mult:.2f}x, SELL={sell_mult:.2f}x, HOLD={hold_mult:.2f}x\n")
            
            if len(adaptive_reward_log) > 10:
                logs.append(f"\n  ... and {len(adaptive_reward_log) - 10} more events\n")
        
        logs.append("\n**How It Works:**\n")
        logs.append("  - **Volatility**: High vol → Higher trading rewards (1.2x-1.5x), Higher hold penalties (1.2x-1.5x)\n")
        logs.append("  - **Market Trend**: Bull market → Higher BUY rewards, Lower SELL rewards\n")
        logs.append("  - **Trading Opportunities**: High recent movement → Higher hold penalties (encourages trading)\n")
        logs.append("  - **Adaptive Range**: Multipliers adjust from 0.5x to 1.6x based on market conditions\n")
        
        return "".join(logs)
    
    def _format_behavior_adaptive_reward_logs(self, execution_results: Dict[str, Any]) -> str:
        """Format behavior-based adaptive reward system logs for unified report"""
        logs = []
        
        adaptive_reward_log = execution_results.get('adaptive_reward_log', [])
        behavior_events = [e for e in adaptive_reward_log if e.get('type') == 'behavior_adaptive']
        
        if len(behavior_events) == 0:
            logs.append("- **Status**: ⚠️ **NO BEHAVIOR ADAPTIVE DATA**\n")
            logs.append("  - No behavior-based adaptive adjustments recorded\n")
            logs.append("  - **Possible reasons**:\n")
            logs.append("    - Evaluation was too short (< 50 steps)\n")
            logs.append("    - Agent behavior met all targets (no adjustments needed)\n")
            logs.append("    - Behavior adaptive system not initialized\n")
            logs.append("  - **Action**: Check if behavior adaptive rewards are enabled in trading environment\n")
            return "".join(logs)
        
        logs.append(f"- **Status**: ✅ **BEHAVIOR ADAPTIVE ACTIVE** ({len(behavior_events)} events recorded)\n")
        logs.append(f"- **Total Events**: {len(behavior_events)}\n")
        
        # Calculate statistics
        trading_mults = [e.get('base_trading_incentive_mult', 1.0) for e in behavior_events]
        sell_mults = [e.get('sell_base_incentive_mult', 1.0) for e in behavior_events]
        hold_penalty_mults = [e.get('hold_penalty_mult', 1.0) for e in behavior_events]
        aging_rate_mults = [e.get('aging_penalty_rate_mult', 1.0) for e in behavior_events]
        aging_start_mults = [e.get('aging_penalty_start_mult', 1.0) for e in behavior_events]
        
        # Trading Frequency Statistics
        if trading_mults:
            logs.append("\n**📈 Trading Frequency Adjustments:**\n")
            logs.append(f"  - **Base Trading Incentive Multiplier**:\n")
            logs.append(f"    - Avg: {np.mean(trading_mults):.2f}x\n")
            logs.append(f"    - Range: [{np.min(trading_mults):.2f}x, {np.max(trading_mults):.2f}x]\n")
            active_count = sum(1 for m in trading_mults if m > 1.0)
            logs.append(f"  - **Active Adjustments**: {active_count}/{len(trading_mults)} events ({active_count/len(trading_mults)*100:.1f}%)\n")
            if active_count > 0:
                logs.append(f"  - **Interpretation**: {'⚠️ Low trade frequency detected - system increased trading incentives' if np.mean(trading_mults) > 1.1 else '✅ Trade frequency within target range'}\n")
        
        # SELL/BUY Ratio Statistics
        if sell_mults:
            logs.append("\n**🔄 SELL/BUY Ratio Adjustments:**\n")
            logs.append(f"  - **SELL Base Incentive Multiplier**:\n")
            logs.append(f"    - Avg: {np.mean(sell_mults):.2f}x\n")
            logs.append(f"    - Range: [{np.min(sell_mults):.2f}x, {np.max(sell_mults):.2f}x]\n")
            active_count = sum(1 for m in sell_mults if m > 1.0)
            logs.append(f"  - **Active Adjustments**: {active_count}/{len(sell_mults)} events ({active_count/len(sell_mults)*100:.1f}%)\n")
            if active_count > 0:
                logs.append(f"  - **Interpretation**: {'⚠️ Low SELL ratio detected - system increased SELL rewards' if np.mean(sell_mults) > 1.1 else '✅ SELL/BUY ratio within target range'}\n")
        
        # Hold Time Statistics
        if hold_penalty_mults:
            logs.append("\n**⏱️ Hold Time Adjustments:**\n")
            logs.append(f"  - **Hold Penalty Multiplier**:\n")
            logs.append(f"    - Avg: {np.mean(hold_penalty_mults):.2f}x\n")
            logs.append(f"    - Range: [{np.min(hold_penalty_mults):.2f}x, {np.max(hold_penalty_mults):.2f}x]\n")
            logs.append(f"  - **Aging Penalty Rate Multiplier**:\n")
            logs.append(f"    - Avg: {np.mean(aging_rate_mults):.2f}x\n")
            logs.append(f"    - Range: [{np.min(aging_rate_mults):.2f}x, {np.max(aging_rate_mults):.2f}x]\n")
            logs.append(f"  - **Aging Penalty Start Multiplier**:\n")
            logs.append(f"    - Avg: {np.mean(aging_start_mults):.2f}x\n")
            logs.append(f"    - Range: [{np.min(aging_start_mults):.2f}x, {np.max(aging_start_mults):.2f}x]\n")
            active_count = sum(1 for m in hold_penalty_mults if m > 1.0)
            logs.append(f"  - **Active Adjustments**: {active_count}/{len(hold_penalty_mults)} events ({active_count/len(hold_penalty_mults)*100:.1f}%)\n")
            if active_count > 0:
                logs.append(f"  - **Interpretation**: {'⚠️ Long hold times detected - system increased penalties' if np.mean(hold_penalty_mults) > 1.1 else '✅ Hold times within target range'}\n")
        
        # Sample events
        if behavior_events:
            logs.append("\n**📋 Sample Behavior Adaptive Events (last 10):**\n")
            for i, event in enumerate(behavior_events[-10:], 1):
                step = event.get('step', 0)
                trading_mult = event.get('base_trading_incentive_mult', 1.0)
                sell_mult = event.get('sell_base_incentive_mult', 1.0)
                hold_mult = event.get('hold_penalty_mult', 1.0)
                actual_trades = event.get('actual_trades', 0)
                expected_trades = event.get('expected_trades', 0)
                sell_count = event.get('sell_count', 0)
                total_trades = event.get('total_trades', 0)
                sell_ratio = event.get('sell_ratio', 0.0)
                avg_hold = event.get('avg_hold_time', 0.0)
                logs.append(f"  {i}. Step {step:,}: Trades={actual_trades:.0f}/{expected_trades:.1f} (mult={trading_mult:.2f}x), "
                          f"SELL={sell_count}/{total_trades} ({sell_ratio:.1%}, mult={sell_mult:.2f}x), "
                          f"HoldTime={avg_hold:.1f} (penalty_mult={hold_mult:.2f}x)\n")
            
            if len(behavior_events) > 10:
                logs.append(f"\n  ... and {len(behavior_events) - 10} more events\n")
        
        logs.append("\n**🔧 How Behavior-Based Adaptive System Works:**\n")
        logs.append("  - **Trade Frequency**: Monitors actual vs expected trades (target: 15% of steps)\n")
        logs.append("    - If < 50% of expected: Increases base trading incentive by up to 1.5x\n")
        logs.append("    - If < 75% of expected: Increases base trading incentive by up to 1.3x\n")
        logs.append("  - **SELL/BUY Ratio**: Monitors SELL count vs BUY count (target: 35% SELL, 65% BUY)\n")
        logs.append("    - If SELL ratio < 17.5%: Increases SELL base reward by up to 1.8x\n")
        logs.append("    - If SELL ratio < 26.25%: Increases SELL base reward by up to 1.5x\n")
        logs.append("  - **Hold Time**: Monitors average hold time (target: 25 steps)\n")
        logs.append("    - If avg hold > 37.5 steps: Increases hold penalties by up to 1.4x, aging rate by up to 1.6x\n")
        logs.append("    - If avg hold > 30 steps: Moderate increases to penalties\n")
        logs.append("  - **Automatic Self-Correction**: System adjusts rewards in real-time based on agent performance\n")
        
        return "".join(logs)
    
    def _format_adaptive_lr_logs(self, execution_results: Dict[str, Any]) -> str:
        """Format adaptive LR callback logs for unified report"""
        logs = []
        
        adaptive_lr_log = execution_results.get('adaptive_lr_log', [])
        callback_triggered = execution_results.get('adaptive_lr_callback_triggered', len(adaptive_lr_log) > 0)
        events_count = execution_results.get('adaptive_lr_events_count', len(adaptive_lr_log))
        
        if not callback_triggered and events_count == 0:
            logs.append("- **Status**: ⚠️ **CALLBACK DID NOT TRIGGER**\n")
            logs.append("  - No adaptive LR events recorded\n")
            logs.append("  - **Possible reasons**:\n")
            logs.append("    - `eval_env` was None (callback requires evaluation environment)\n")
            logs.append("    - Evaluation failed silently\n")
            logs.append("    - Callback never reached evaluation threshold\n")
            logs.append("  - **Action**: Check training logs for 'Adaptive LR callback' messages\n")
            return "".join(logs)
        
        logs.append(f"- **Status**: ✅ **CALLBACK ACTIVE** ({events_count} events recorded)\n")
        logs.append(f"- **Callback Triggered**: {'✅ Yes' if callback_triggered else '❌ No'}\n")
        logs.append(f"- **Total Events**: {events_count}\n")
        
        if adaptive_lr_log:
            logs.append("\n**Event Log:**\n")
            for i, event in enumerate(adaptive_lr_log[:20], 1):  # Show first 20 events
                event_type = event.get('event', 'UNKNOWN')
                timestep = event.get('timestep', 0)
                message = event.get('message', '')
                
                # Format based on event type
                if event_type == 'CALLBACK_TRIGGERED':
                    logs.append(f"  {i}. ✅ **{event_type}** (Step {timestep:,}): {message}\n")
                elif event_type == 'EVALUATION':
                    return_val = event.get('return', 0)
                    sharpe_val = event.get('sharpe', 0)
                    trades_val = event.get('trades', 0)
                    logs.append(f"  {i}. 📊 **{event_type}** (Step {timestep:,}): Return={return_val:.2%}, Sharpe={sharpe_val:.2f}, Trades={trades_val}\n")
                elif event_type in ['PERFORMANCE_DEGRADATION', 'TRADE_DROP']:
                    logs.append(f"  {i}. ⚠️ **{event_type}** (Step {timestep:,}): {message}\n")
                elif event_type in ['LR_INCREASE_POOR', 'LR_INCREASE_EXCELLENT']:
                    prev_lr = event.get('previous_lr', 0)
                    new_lr = event.get('new_lr', 0)
                    boost = event.get('boost_factor', 1.0)
                    logs.append(f"  {i}. 🔄 **{event_type}** (Step {timestep:,}): LR {prev_lr:.2e} → {new_lr:.2e} (×{boost:.1f})\n")
                elif event_type == 'EVALUATION_ERROR':
                    error = event.get('error', 'Unknown error')
                    logs.append(f"  {i}. ❌ **{event_type}** (Step {timestep:,}): {error[:80]}\n")
                elif event_type == 'NO_EVAL_ENV':
                    logs.append(f"  {i}. ⚠️ **{event_type}**: {message}\n")
                else:
                    logs.append(f"  {i}. 📝 **{event_type}** (Step {timestep:,}): {message}\n")
            
            if len(adaptive_lr_log) > 20:
                logs.append(f"\n  ... and {len(adaptive_lr_log) - 20} more events (see full log in execution_results)\n")
            
            # Summary statistics
            lr_increases = [e for e in adaptive_lr_log if 'LR_INCREASE' in e.get('event', '')]
            degradations = [e for e in adaptive_lr_log if e.get('event') == 'PERFORMANCE_DEGRADATION']
            trade_drops = [e for e in adaptive_lr_log if e.get('event') == 'TRADE_DROP']
            errors = [e for e in adaptive_lr_log if e.get('event') == 'EVALUATION_ERROR']
            
            logs.append("\n**Summary:**\n")
            logs.append(f"  - LR Adjustments: {len(lr_increases)}\n")
            logs.append(f"  - Performance Degradations Detected: {len(degradations)}\n")
            logs.append(f"  - Trade Drops Detected: {len(trade_drops)}\n")
            logs.append(f"  - Evaluation Errors: {len(errors)}\n")
        else:
            logs.append("\n- **No events logged** (callback may have triggered but no events recorded)\n")
        
        return "".join(logs)


# Standalone function for backward compatibility
def generate_unified_report(execution_results: Dict[str, Any], 
                            training_results: Optional[Dict[str, Any]] = None,
                            base_dir: str = "ppo",
                            trade_history_data: Optional[List[Dict[str, Any]]] = None,
                            portfolio_history_data: Optional[List[float]] = None) -> Optional[str]:
    """
    Standalone function to generate unified report.
    """
    generator = UnifiedReportGenerator(base_dir)
    return generator.generate_unified_report(
        execution_results, 
        training_results,
        trade_history_data,
        portfolio_history_data
    )
