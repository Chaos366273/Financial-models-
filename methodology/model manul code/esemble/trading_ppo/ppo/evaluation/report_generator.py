"""
Report Generator Module for PPO Trading System
Generates comprehensive performance reports in markdown and JSON formats
"""

import os
import json
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import Dict, Any, Optional

logger = logging.getLogger("PPOTrading.ReportGenerator")

def generate_performance_report(
    results: Dict[str, Any],
    report_path: str,
    include_json: bool = True
) -> None:
    """
    Generate comprehensive performance report in markdown format
    
    Args:
        results: Dictionary of evaluation results
        report_path: Path to save the markdown report
        include_json: Whether to also save a JSON version of the report
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        
        # Generate markdown report
        md_content = _generate_markdown_report(results)
        
        # Save markdown report with proper encoding
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        logger.info(f"Performance report saved to {report_path}")
        
        # Generate and save JSON report if requested
        if include_json:
            json_path = report_path.replace('.md', '.json')
            # Fix JSON serialization by converting non-serializable types
            from evaluation.performance_evaluator import _make_json_serializable
            serializable_results = _make_json_serializable(results)
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_results, f, indent=2)
            
            logger.info(f"JSON report saved to {json_path}")
            
            # Generate Cursor AI friendly report
            cursor_ai_path = os.path.join(
                os.path.dirname(report_path),
                f"cursor_ai_report_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
            )
            
            # Create a simplified version for Cursor AI
            cursor_ai_report = {
                "summary": {
                    "total_return": results.get("total_return", 0.0),
                    "sharpe_ratio": results.get("sharpe_ratio", 0.0),
                    "max_drawdown": results.get("max_drawdown", 0.0),
                    "win_rate": results.get("win_rate", 0.0),
                    "total_trades": results.get("total_trades", 0),
                    "final_portfolio": results.get("final_portfolio_value", 0.0),
                    "steps_simulated": results.get("steps_simulated", 0),
                    "benchmark_return": results.get("benchmark_return", 0.0),
                    "alpha": results.get("alpha", 0.0),
                    "beta": results.get("beta", 0.0),
                    "profit_factor": results.get("profit_factor", 0.0),
                    "volatility": results.get("volatility", 0.0),
                    "paper_trading_ready": results.get("paper_trading_ready", False)
                },
                "statistical_validation": {
                    "p_value": results.get("p_value", 1.0),
                    "sharpe_p_value": results.get("sharpe_p_value", 1.0),
                    "benchmark_data_valid": results.get("benchmark_data_valid", False),
                    "significance": results.get("p_value", 1.0) < 0.05 or results.get("sharpe_p_value", 1.0) < 0.05
                },
                "performance_metrics": results.get("performance_metrics", {}),
                "env_spec": results.get("env_spec", {}),
                "model_spec": results.get("model_spec", {}),
                "paths": results.get("paths", {}),
                "timestamp": datetime.now().isoformat(),
                "run_id": results.get("run_id", datetime.now().strftime("%Y%m%d_%H%M%S"))
            }
            
            # Fix JSON serialization by converting non-serializable types
            from evaluation.performance_evaluator import _make_json_serializable
            serializable_cursor_ai_report = _make_json_serializable(cursor_ai_report)
            
            with open(cursor_ai_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_cursor_ai_report, f, indent=2)
            
            logger.info(f"Cursor AI report saved to {cursor_ai_path}")
        
    except Exception as e:
        logger.error(f"Failed to generate performance report: {e}")

def _generate_markdown_report(results: Dict[str, Any]) -> str:
    """
    Generate markdown report content
    
    Args:
        results: Dictionary of evaluation results
    
    Returns:
        Markdown report content
    """
    # Extract key metrics
    total_return = results.get("total_return", 0.0)
    sharpe_ratio = results.get("sharpe_ratio", 0.0)
    max_drawdown = results.get("max_drawdown", 0.0)
    win_rate = results.get("win_rate", 0.0)
    total_trades = results.get("total_trades", 0)
    final_portfolio = results.get("final_portfolio_value", 0.0)
    initial_portfolio = results.get("initial_portfolio_value", 100000.0)
    steps_simulated = results.get("steps_simulated", 0)
    benchmark_return = results.get("benchmark_return", 0.0)
    alpha = results.get("alpha", 0.0)
    alpha_annual = results.get("alpha_annual", 0.0)
    beta = results.get("beta", 0.0)
    profit_factor = results.get("profit_factor", 0.0)
    volatility = results.get("volatility", 0.0)
    execution_time = results.get("execution_time", 0.0)
    p_value = results.get("p_value", 1.0)
    sharpe_p_value = results.get("sharpe_p_value", 1.0)
    benchmark_data_valid = results.get("benchmark_data_valid", False)
    paper_trading_ready = results.get("paper_trading_ready", False)
    
    # Format metrics for display
    total_return_str = f"{total_return * 100:.2f}%"
    sharpe_ratio_str = f"{sharpe_ratio:.3f}"
    max_drawdown_str = f"{max_drawdown * 100:.2f}%"
    win_rate_str = f"{win_rate * 100:.1f}%"
    benchmark_return_str = f"{benchmark_return * 100:.2f}%"
    alpha_str = f"{alpha * 100:.2f}%"
    alpha_annual_str = f"{alpha_annual * 100:.2f}%"
    beta_str = f"{beta:.3f}"
    profit_factor_str = f"{profit_factor:.3f}"
    volatility_str = f"{volatility * 100:.2f}%"
    
    # Determine significance status (ASCII only for Windows compatibility)
    significance_status = "PASS" if p_value < 0.05 or sharpe_p_value < 0.05 else "FAIL"
    
    # Determine paper trading readiness status (ASCII only for Windows compatibility)
    readiness_status = "READY" if paper_trading_ready else "NOT READY"
    
    # Build markdown report
    md_content = f"""# Daily Trading Report - {datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}

##  Performance Summary

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Total Return | {total_return_str} | >0% | {"PASS" if total_return > 0 else "FAIL"} |
| Sharpe Ratio | {sharpe_ratio_str} | >1.5 | {"PASS" if sharpe_ratio > 1.5 else "FAIL"} |
| Max Drawdown | {max_drawdown_str} | <5% | {"PASS" if max_drawdown < 0.05 else "FAIL"} |
| Win Rate | {win_rate_str} | >50% | {"PASS" if win_rate > 0.5 else "FAIL"} |
| Statistical Significance | p={p_value:.4f} | p<0.05 | {"PASS" if p_value < 0.05 else "FAIL"} |
| Sharpe Significance | p={sharpe_p_value:.4f} | p<0.05 | {"PASS" if sharpe_p_value < 0.05 else "FAIL"} |

##  Strategy Metrics

- Total Trades: {total_trades}
- Profit Factor: {profit_factor_str}
- Volatility (Ann.): {volatility_str}
- Alpha (Horizon): {alpha_str}
- Alpha (Annual): {alpha_annual_str}
- Beta: {beta_str}
- Benchmark Return: {benchmark_return_str}
- Final Portfolio: ${final_portfolio:.2f} (from ${initial_portfolio:.2f} initial)
- Steps Simulated: {steps_simulated}
- Execution Time: {execution_time:.2f}s

##  Statistical Analysis

- **Statistical Significance**: {significance_status}
- **Benchmark Data**: {"VALID" if benchmark_data_valid else "INVALID"}
- **Paper Trading Readiness**: {readiness_status}

##  Recommendations

"""
    
    # Add recommendations based on results
    recommendations = []
    
    if total_return <= 0:
        recommendations.append("FAIL: Strategy is not profitable - consider model retraining or parameter tuning")
    else:
        recommendations.append("PASS: Strategy is profitable")
    
    if sharpe_ratio < 1.0:
        recommendations.append("FAIL: Low Sharpe ratio - consider improving risk-adjusted returns")
    else:
        recommendations.append("PASS: Good Sharpe ratio - risk-adjusted returns are positive")
    
    if max_drawdown > 0.1:
        recommendations.append("FAIL: High drawdown - consider implementing stronger risk management")
    else:
        recommendations.append("PASS: Acceptable drawdown - risk management is working")
    
    if win_rate < 0.45:
        recommendations.append("FAIL: Low win rate - consider improving entry/exit signals")
    else:
        recommendations.append("PASS: Acceptable win rate")
    
    if total_trades < 10:
        recommendations.append("WARNING: Few trades - consider longer backtest period or more active strategy")
    
    if not benchmark_data_valid:
        recommendations.append("FAIL: Invalid benchmark data - ensure SPY data is available")
    
    if p_value >= 0.05 and sharpe_p_value >= 0.05:
        recommendations.append("FAIL: Results not statistically significant - consider longer backtest or model improvements")
    
    # Add recommendations to markdown
    for recommendation in recommendations:
        md_content += f"- {recommendation}\n"
    
    # Add execution details
    md_content += f"""
##  Execution Details

- **Run ID**: {results.get("run_id", "Unknown")}
- **Timestamp**: {results.get("timestamp", datetime.now().isoformat())}
- **Environment**: {results.get("env_spec", {}).get("tickers", 0)} tickers, {results.get("env_spec", {}).get("features", 0)} features, {results.get("env_spec", {}).get("window", 0)} window (Obs Dim: {results.get("env_spec", {}).get("obs_dim", "N/A")})
- **Model**: {results.get("model_spec", {}).get("device", "unknown")} device, seed {results.get("model_spec", {}).get("seed", 0)}

##  Output Files

- Trade History: {results.get("paths", {}).get("trade_history_csv", "N/A")}
- Portfolio History: {results.get("paths", {}).get("portfolio_history_csv", "N/A")}
- Returns: {results.get("paths", {}).get("returns_csv", "N/A")}
- Benchmark Returns: {results.get("paths", {}).get("benchmark_returns_csv", "N/A")}
"""
    
    return md_content
