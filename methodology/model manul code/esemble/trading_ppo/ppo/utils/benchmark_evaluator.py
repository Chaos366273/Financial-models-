"""
Lightweight benchmark evaluator stub.
Prevents import errors when the advanced utility is unavailable.
Provides alpha, beta, and benchmark return calculations compatible with the
backtesting pipeline. Safe no-op defaults on bad inputs.
"""

from typing import Dict
import numpy as np


def compute_metrics(agent_returns: np.ndarray, benchmark_returns: np.ndarray, risk_free_annual: float = 0.02) -> Dict[str, float]:
    try:
        if agent_returns is None or benchmark_returns is None:
            return {"alpha": 0.0, "beta": 1.0, "benchmark_return": 0.0}

        agent = np.nan_to_num(np.asarray(agent_returns, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        bench = np.nan_to_num(np.asarray(benchmark_returns, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)

        if agent.size == 0 or bench.size == 0:
            return {"alpha": 0.0, "beta": 1.0, "benchmark_return": 0.0}

        n = min(agent.size, bench.size)
        agent = agent[:n]
        bench = bench[:n]

        # Benchmark total return in percent
        benchmark_total_return = (float(np.prod(1.0 + bench)) - 1.0) * 100.0

        # Beta from covariance/variance with zero-variance guard
        if np.std(bench) > 1e-12:
            cov = float(np.cov(agent, bench)[0, 1])
            var = float(np.var(bench))
            beta = cov / var if var > 0 else 1.0
        else:
            beta = 1.0

        # Annualized alpha from excess returns
        mean_agent = float(np.mean(agent))
        mean_bench = float(np.mean(bench))
        risk_free_daily = (1.0 + float(risk_free_annual)) ** (1.0 / 252.0) - 1.0
        annual_agent = (1.0 + mean_agent) ** 252 - 1.0
        annual_bench = (1.0 + mean_bench) ** 252 - 1.0
        agent_excess = annual_agent - ((1.0 + risk_free_daily) ** 252 - 1.0)
        bench_excess = annual_bench - ((1.0 + risk_free_daily) ** 252 - 1.0)
        alpha_annual = (agent_excess - beta * bench_excess) * 100.0

        return {"alpha": alpha_annual, "beta": beta, "benchmark_return": benchmark_total_return}
    except Exception:
        return {"alpha": 0.0, "beta": 1.0, "benchmark_return": 0.0}


class BenchmarkEvaluator:
    @staticmethod
    def evaluate(agent_returns: np.ndarray, benchmark_returns: np.ndarray, risk_free_annual: float = 0.02) -> Dict[str, float]:
        return compute_metrics(agent_returns, benchmark_returns, risk_free_annual)


