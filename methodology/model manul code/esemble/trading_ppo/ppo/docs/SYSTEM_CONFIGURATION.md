# PPO Trading Agent - Complete System Configuration

**Last Updated**: 2025-11-30  
**Version**: 1.0  
**Status**: Production Ready (Seed 42 Validated)

---

## 📋 Table of Contents

1. [Features](#features)
2. [Rewards](#rewards)
3. [Constraints](#constraints)
4. [Timesteps](#timesteps)
5. [Model Configuration](#model-configuration)
6. [Data Configuration](#data-configuration)
7. [Trading Configuration](#trading-configuration)
8. [Risk Management](#risk-management)
9. [Performance Configuration](#performance-configuration)

---

## 🎯 Features

### Core Features (Always Implemented)

| Feature | Type | Description | Status |
|---------|------|-------------|--------|
| **Open Price** | Price Data | Opening price per bar | ✅ ENABLED |
| **High Price** | Price Data | Highest price per bar | ✅ ENABLED |
| **Low Price** | Price Data | Lowest price per bar | ✅ ENABLED |
| **Close Price** | Price Data | Closing price per bar | ✅ ENABLED |
| **Volume** | Volume Data | Trading volume per bar | ✅ ENABLED |

**Total Core Features**: 5

### Technical Indicators

| Feature | Type | Description | Status | Parameters |
|---------|------|-------------|--------|------------|
| **Returns** | Technical | Price returns calculation | ✅ ENABLED | Numba JIT optimized |
| **SMA** | Technical | Simple Moving Average | ✅ ENABLED | Periods: [10, 20, 50, 200] |
| **EMA** | Technical | Exponential Moving Average | ✅ ENABLED | Periods: [5, 10, 20, 50] |
| **RSI** | Technical | Relative Strength Index | ✅ ENABLED | Period: 14 |
| **MACD** | Technical | Moving Average Convergence Divergence | ✅ ENABLED | Fast: 12, Slow: 26, Signal: 9 |
| **ATR** | Technical | Average True Range | ✅ ENABLED | Period: 14 |
| **Bollinger Bands** | Technical | BB Upper/Middle/Lower | ✅ ENABLED | Period: 20, Std: 2.0 |
| **Price to SMA** | Technical | Price relative to SMA | ✅ ENABLED | SMA period: 20 |
| **Volume Trend** | Technical | Volume trend indicator | ✅ ENABLED | Lookback: 20 |

**Total Technical Features**: 9

### Advanced Features

| Feature | Type | Description | Status |
|---------|------|-------------|--------|
| **Fundamental Analysis** | Advanced | PE ratio, ROE, Debt-to-Equity, Growth | ⚠️ Configured but cache empty |
| **Sentiment Analysis** | Advanced | News sentiment scoring | ⚠️ Configured but cache empty |
| **Market Regime Detection** | Advanced | Volatility/trend regime classification | ✅ ENABLED |
| **Dynamic Stop Loss** | Advanced | Adaptive stop loss based on volatility | ✅ ENABLED |
| **Portfolio Hedging** | Advanced | Correlation-based hedging | ✅ ENABLED |
| **Trade Validation** | Advanced | Pre-trade validation checks | ✅ ENABLED |

**Total Advanced Features**: 6 (2 disabled due to cache)

### Feature Engineering Configuration

```python
# Feature Count
NUM_FEATURES = 15  # Total features (5 core + 9 technical + 1 advanced)
OBSERVATION_DIMENSIONS = 10 * 6 * 15 = 900  # window_size * num_tickers * num_features

# Feature Scaling
SCALER_TYPE = 'RobustScaler'  # Robust to outliers
PRICE_COLUMNS_PRESERVED = True  # Price data not scaled

# Feature Engineering
ENABLE_NUMBA_JIT = True  # JIT compilation for performance
VECTORIZED_CALCULATIONS = True  # NumPy vectorization
PARALLEL_PROCESSING = True  # Multi-threaded feature calculation
```

### Observation Space

- **Market Features**: 900 dimensions (10 window × 6 tickers × 15 features)
- **Portfolio State**: 7 dimensions (cash, portfolio_value, positions, etc.)
- **Total Observation Space**: 907 dimensions
- **Action Space**: Discrete(3) - [BUY, SELL, HOLD]

---

## 💰 Rewards

### Base Trading Rewards

| Reward Component | Value | Description |
|-----------------|-------|-------------|
| **Base Trading Incentive** | 0.40 | Base reward for executing trades (adaptive: 0.7x-1.6x multiplier) |
| **Sell Base Incentive** | 0.35 | Base reward for SELL actions (25% increase from 0.28) |
| **First Trade Bonus** | 0.15 | Bonus for first trade in episode (50% increase from 0.10) |
| **Trading Frequency Bonus** | 0.025 | Bonus for high trading frequency (67% increase from 0.015) |
| **Learning Bonus** | 0.03 | Exploration bonus for attempting trades |
| **Signal Alignment Bonus** | 0.10 | Bonus when signals align with action |
| **Profitable Trade Bonus** | 0.10 | Bonus for profitable trades |

### SELL-Specific Rewards

| Reward Component | Value | Description |
|-----------------|-------|-------------|
| **Sell Profit Taking Bonus** | 0.25 | Bonus for profitable SELL (39% increase from 0.18) |
| **Sell Bull Market Boost** | 0.08 | Extra reward in bull markets (60% increase from 0.05) |
| **Unrealized Gain Bonus** | 0.05 | Bonus for SELL when position has unrealized gains |
| **Trailing Stop Bonus** | 0.10 | Bonus for SELL via trailing stop |
| **Hold Time Bonus** | 0.08 | Bonus for holding profitable positions >10 steps (60% increase) |

### Hold Penalties

| Penalty Component | Value | Description |
|------------------|-------|-------------|
| **Hold Penalty (Base)** | -0.001 | Base hold penalty (10x lighter) |
| **Hold Penalty (With Positions)** | -0.012 | Hold penalty when holding positions (40% lighter) |
| **Hold Penalty (No Positions)** | -0.025 | Hold penalty when no positions (29% lighter) |
| **Eval Hold Penalty** | -0.015 | Evaluation mode hold penalty (25% lighter) |
| **Quick Flip Penalty** | -0.15 | Penalty for selling < 10 steps (increased from -0.1) |
| **Min Hold Steps** | 10 | Minimum steps before SELL allowed |

### PnL Rewards

| Reward Component | Value | Description |
|-----------------|-------|-------------|
| **PnL Reward Scale** | 5.0 | Scaling factor for realized PnL rewards |
| **Profit Bonus Scale** | 1.0 | Scale for additional profit bonuses |
| **Profit Bonus Cap** | 0.25 | Maximum profit bonus contribution |
| **Loss Penalty Scale** | 1.0 | Scale for losses |
| **Loss Penalty Cap** | 0.4 | Maximum loss penalty magnitude |
| **Sell Loss Penalty Cap** | -2.0 | Maximum penalty for unprofitable sells |
| **Buy Loss Penalty Cap** | -0.5 | Maximum penalty for losing BUY trades |

### Adaptive Rewards (Market Condition Multipliers)

| Market Condition | BUY Multiplier | SELL Multiplier | HOLD Multiplier |
|-----------------|----------------|-----------------|-----------------|
| **High Volatility** | 1.2x - 1.5x | 1.2x - 1.5x | 0.8x - 1.2x |
| **Strong Trend** | 1.3x - 1.6x | 1.3x - 1.6x | 0.9x - 1.3x |
| **Mean Reversion** | 0.8x - 1.1x | 0.8x - 1.1x | 0.7x - 1.0x |
| **Normal Conditions** | 1.0x | 1.0x | 1.0x |

**Multiplier Range**: 0.7x - 1.6x (clamped)

### Signal-Based Rewards

| Signal Type | Weight | Description |
|-------------|-------|-------------|
| **Technical Signal Weight** | 0.40 | Weight for technical indicator signals (40%) |
| **Fundamental Signal Weight** | 0.20 | Weight for fundamental analysis signals (20%) |
| **PnL Weight** | 0.60 | Weight for P&L signals (60% - primary) |
| **Signal Bonus Cap** | 0.3 | Maximum contribution from signal rewards |

### Reward Caps

| Cap Type | Min | Max |
|----------|-----|-----|
| **Reward Cap** | -0.5 | 1.0 |
| **Signal Bonus Cap** | - | 0.3 |
| **Profit Bonus Cap** | - | 0.25 |
| **Loss Penalty Cap** | -0.4 | - |

---

## 🚫 Constraints

### Trading Constraints

| Constraint | Value | Description |
|-----------|-------|-------------|
| **Min Trade Interval** | 1 step | Minimum steps between trades on same ticker |
| **Min Hold Steps** | 10 steps | Minimum steps before SELL allowed |
| **Max Trades Per Episode** | 500 | Maximum trades per episode |
| **Max Trades Per Step** | 1 | Maximum trades per step |
| **Daily Trade Limit** | 100 | Maximum trades per day |
| **Min Trade Size** | $20.00 | Minimum trade value |
| **Min Cash Balance** | $100.00 | Minimum cash required |
| **Max Position Size** | 2% | Maximum position size per ticker (4% in dynamic_config) |
| **Max Portfolio Concentration** | 50% | Maximum concentration in single asset |
| **Min Cash Reserve** | 5% | Minimum cash reserve percentage |

### Position Constraints

| Constraint | Value | Description |
|-----------|-------|-------------|
| **Max Single Position** | 20% | Maximum position size per ticker |
| **Max Total Exposure** | 90% | Maximum total portfolio exposure |
| **Max Sector Concentration** | 40% | Maximum concentration per sector |
| **Max Tech Weight** | 60% | Maximum weight in technology sector |
| **Correlation Limit** | 0.7 | Maximum correlation for new positions |

### Risk Constraints

| Constraint | Value | Description |
|-----------|-------|-------------|
| **Base Stop Loss** | 10% | Base stop loss percentage |
| **Trailing Stop** | 8% | Trailing stop loss percentage |
| **Max Drawdown Limit** | 25% | Maximum drawdown before circuit breaker |
| **Dynamic Stop Loss** | Enabled | Adaptive stop loss based on volatility |
| **Gap Fill Cap** | 15% | Maximum loss from gap fills (extreme gaps capped) |

### Trade Validation Constraints

| Constraint | Value | Description |
|-----------|-------|-------------|
| **Trade Cooldown** | 2 steps | Cooldown period after trade |
| **Min Trade Value** | $10.00 | Minimum trade value for validation |
| **Position Check** | Enabled | Validate position exists before SELL |
| **Cash Check** | Enabled | Validate sufficient cash before BUY |
| **Diversification Check** | Enabled | Enforce diversification limits |

---

## ⏱️ Timesteps

### Current Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Total Timesteps** | 300,000 | Current training timesteps |
| **Min Timesteps** | 100,000 | Minimum required timesteps |
| **Max Timesteps** | 1,000,000 | Maximum allowed timesteps |
| **Target Timesteps** | 600,000 | Target for super agent scaling |
| **Scaling Phase** | "300k" | Current scaling phase |

### Timestep Scaling Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Start Timesteps** | 5,000 | Initial timesteps for scaling |
| **Increment Size** | 50,000 | Increase by 50k each successful run |
| **Max Errors Per Run** | 2 | Maximum errors before reducing timesteps |
| **Reduction Factor** | 0.8 | Reduce by 20% if errors occur |
| **Success Runs Before Increase** | 2 | Need 2 clean runs to increase timesteps |

### Scaling Criteria

| Metric | Threshold | Description |
|--------|-----------|-------------|
| **Min Sharpe to Scale** | 0.2 | Minimum Sharpe ratio to increase timesteps |
| **Min Return to Scale** | 2% | Minimum return to increase timesteps |
| **Max Drawdown to Scale** | 15% | Maximum drawdown to allow scaling |
| **Min Trades Per Episode** | 20 | Minimum trades to ensure activity |
| **Max Trades Per Episode** | 60 | Maximum trades to prevent over-trading |

### Timestep History

| Phase | Timesteps | Status | Date |
|-------|-----------|--------|------|
| **Baseline** | 300,000 | ✅ Completed | 2025-11-30 |
| **Seed 42** | 300,000 | ✅ Completed | 2025-11-30 |
| **Seed 0** | 300,000 | ✅ Completed | 2025-11-30 |
| **Seed 123** | 300,000 | ✅ Completed | 2025-11-30 |

### Future Scaling Plan

1. **Phase 1**: 300k → 400k (if metrics stable)
2. **Phase 2**: 400k → 500k (if Sharpe ≥ 1.0, Return ≥ 10%)
3. **Phase 3**: 500k → 600k (if Sharpe ≥ 1.2, Return ≥ 12%)
4. **Phase 4**: 600k → 800k (if Sharpe ≥ 1.5, Return ≥ 15%)
5. **Phase 5**: 800k → 1M (if Sharpe ≥ 1.8, Return ≥ 18%)

---

## 🤖 Model Configuration

### PPO Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Learning Rate** | 3e-05 | Initial learning rate (3x increase from 1e-5) |
| **LR Schedule** | Linear | Learning rate decay type |
| **LR Initial** | 3e-5 | Starting learning rate |
| **LR Final** | 1.5e-5 | Final learning rate (2x increase) |
| **N Steps** | 2,048 | PPO rollout length |
| **Batch Size** | 512 | Training batch size |
| **N Epochs** | 4 | Training epochs per update |
| **Gamma** | 0.995 | Discount factor |
| **GAE Lambda** | 0.98 | GAE lambda parameter |
| **Clip Range** | 0.15 | PPO clipping range |
| **Entropy Coefficient** | 0.03 | Exploration bonus (50% increase) |
| **Value Function Coef** | 0.25 | Value function loss weight |
| **Max Grad Norm** | 0.3 | Gradient clipping threshold |
| **Target KL** | 0.01 | Target KL divergence |

### Adaptive Learning Rate

| Condition | Threshold | Action |
|-----------|-----------|--------|
| **High Performance** | Sharpe > 2.0 OR Return > 3% | Maintain higher LR (1.5x boost) |
| **Poor Performance** | Sharpe < 1.5 OR Return < 3% | Increase LR (2.0x boost, max 5e-5) |
| **Low Trading** | Trades < 50 | Increase entropy (2.0x boost, max 0.02) |

### Model Architecture

| Component | Configuration | Description |
|-----------|--------------|-------------|
| **Policy Type** | ActorCriticPolicy | Standard PPO policy |
| **Feature Extractor** | FlattenExtractor | Flatten feature matrix |
| **TCN Status** | ⚠️ Configured but NOT Active | TCN configured but using MLP |
| **Network Architecture** | [256, 128] | Policy network hidden layers |
| **Value Network** | [256, 128] | Value network hidden layers |
| **Activation** | ReLU | Activation function |
| **Device** | CPU/CUDA | Training device |

### TCN Configuration (Not Active)

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Temporal Memory Type** | "tcn" | Temporal Convolutional Network |
| **Hidden Dims** | [256, 256, 128, 128] | Hidden dimensions per TCN block |
| **Kernel Size** | 3 | Convolution kernel size |
| **Dilation Rates** | [1, 2, 4, 8] | Exponential dilation for long-range memory |
| **Dropout** | 0.1 | Dropout rate for TCN blocks |
| **Use Residual** | True | Residual connections in TCN blocks |

---

## 📊 Data Configuration

### Market Data

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Start Date** | 2020-01-01 | Data start date (4 years) |
| **End Date** | 2024-10-17 | Data end date |
| **Tickers** | 6 | AAPL, MSFT, NVDA, JPM, SPY, GLD |
| **Bar Size** | 1Day | Daily bars (swing trading) |
| **Timeframe** | 1Day | Daily timeframe |
| **Window Size** | 10 | Lookback window (10 bars) |
| **Initial Balance** | $100,000 | Starting portfolio value |

### Data Sources

| Source | Status | Rate Limit | Description |
|--------|--------|------------|-------------|
| **Alpaca API** | ✅ Enabled | 200 calls/min (Basic) | Primary data source |
| **Yahoo Finance** | ❌ Disabled | N/A | Fallback source |
| **FRED API** | ✅ Enabled | N/A | Economic data |
| **Cache Expiry** | 24 days | N/A | Cache retention period |

### Data Processing

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Float32** | True | Use float32 for memory efficiency |
| **Parquet Memory Map** | True | Memory-mapped parquet reads |
| **Parallel Downloads** | True | Parallel data downloads |
| **Chunk Processing** | True | Process data in chunks |
| **Lazy Loading** | True | Lazy load data when needed |
| **Precomputed Features** | True | Precompute technical indicators |
| **Vectorized Calculations** | True | Use vectorized calculations |

---

## 📈 Trading Configuration

### Position Sizing

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Max Position Size** | 2% (config) / 4% (dynamic) | Maximum position per ticker |
| **Min Position Size** | 5% | Minimum position for diversification |
| **Action Multiplier** | 0.5 | Position sizing multiplier |
| **Kelly Fraction** | 0.25 | Kelly Criterion fraction |
| **Max Portfolio Risk** | 70% | Maximum portfolio risk percentage |

### Transaction Costs

| Cost Type | Rate | Description |
|-----------|------|-------------|
| **Commission Rate** | 0.001 (0.1%) | Trading commission |
| **Slippage Rate** | 0.0002 (0.02%) | Market slippage |
| **SEC Fee** | $22.90 per $1M | SEC fee per dollar sold |
| **FINRA TAF** | $0.000119 per share | FINRA TAF fee (sells) |
| **CAT Fee** | $0.00010 per share | CAT fee (buys/sells) |
| **Real Alpaca Cost** | 0.000024 (0.0024%) | SEC + FINRA fees only |

### Trading Profiles

#### Train Profile
- **Slippage**: 0.0005 (0.05%)
- **Market Impact**: 0.0002 (0.02%)
- **Commission**: 0.0029 (0.29%)
- **Max Trades Per Step**: 3
- **Partial Fills**: Enabled (70% probability)
- **Order Queue**: Enabled

#### Paper/Backtest Profile
- **Slippage**: 0.0008 (0.08%)
- **Market Impact**: 0.0003 (0.03%)
- **Max Trades Per Step**: 75-200
- **Partial Fills**: Enabled/Disabled (profile dependent)
- **Order Queue**: Enabled/Disabled (profile dependent)

---

## 🛡️ Risk Management

### Stop Loss Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Base Stop Loss** | 10% | Base stop loss percentage (widened) |
| **Trailing Stop** | 8% | Trailing stop percentage (widened) |
| **Dynamic Stop Loss** | Enabled | Adaptive stop loss based on volatility |
| **Gap Fill Cap** | 15% | Maximum loss from extreme gaps |
| **Extreme Gap Threshold** | -20% | Gap threshold for capping |

### Drawdown Management

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Max Drawdown Limit** | 25% | Maximum drawdown before action |
| **Drawdown Penalty Lambda** | 0.5 | Drawdown penalty weight |
| **Drawdown Penalty** | 2.5 | Drawdown penalty in reward function |
| **Smooth Penalty Lambda** | 2.0 | Smooth penalty coefficient |

### Portfolio Hedging

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Portfolio Hedging** | Enabled | Correlation-based hedging |
| **Hedging Pairs** | AAPL→SPY, MSFT→SPY, etc. | Hedging pair mappings |
| **Hedging Lookback** | 30 days | Lookback for z-score calculation |
| **Hedging Z-Threshold** | 2.0 | Z-score threshold for hedge activation |
| **Hedge Ratio** | 0.5 | 50% hedge ratio |
| **Hedge Notional** | 5% | 5% of portfolio for hedging |

---

## ⚡ Performance Configuration

### Memory Optimization

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Max Memory Usage** | 24 GB | Maximum memory limit |
| **Garbage Collection Freq** | 5,000 steps | GC frequency |
| **Cache Clear Freq** | 2,000 steps | Cache clearing frequency |
| **Memory Optimization Interval** | 1,000 steps | Memory optimization frequency |
| **Warning Threshold** | 80% | Memory warning threshold |
| **Critical Threshold** | 90% | Memory critical threshold |

### GPU Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| **GPU Memory Fraction** | 95% | GPU memory usage |
| **GPU Memory Growth** | Enabled | Dynamic GPU memory allocation |
| **Mixed Precision** | Enabled | Mixed precision training |
| **Tensor Core Optimization** | Enabled | Tensor core optimization |
| **CUDA Optimizations** | Enabled | All CUDA optimizations |

### Threading Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| **OMP Num Threads** | 12 | OpenMP threads |
| **MKL Num Threads** | 12 | MKL threads |
| **NumExpr Num Threads** | 12 | NumExpr threads |
| **OpenBLAS Num Threads** | 12 | OpenBLAS threads |
| **Num Workers** | 4 | Parallel workers |

---

## 📝 Additional Configuration

### Logging Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Console Level** | WARNING | Console log level |
| **File Level** | WARNING | File log level |
| **Log Every N Steps** | 50 | Logging frequency |
| **Verbose Trade Logs** | True | Detailed trade logs |
| **Verbose Step Logs** | False | Step-by-step logs disabled |
| **Performance Metrics** | True | Performance profiling enabled |

### Evaluation Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Eval Frequency** | 60,000 steps | Evaluation frequency |
| **N Eval Episodes** | 10-15 | Evaluation episodes |
| **Test Days** | 90-120 | Test period (days) |
| **Eval Hold Penalty** | -0.015 | Evaluation hold penalty |
| **Eval Exploration Warmup** | 150 steps | Early exploration in eval |

### Seed Configuration

| Seed | Status | Performance | Recommendation |
|------|--------|-------------|----------------|
| **42** | ✅ Validated | 12.71% return, 1.222 Sharpe | 🥇 **RECOMMENDED** |
| **0** | ⚠️ Validated | 12.47% return, 1.022 Sharpe, 9.53% DD | Acceptable |
| **123** | ✅ Validated | 10.66% return, 1.086 Sharpe | Acceptable |

**Current Seed**: 123 (for testing)

---

## 🔄 Configuration Sources

### Primary Sources

1. **`config.py`**: Main configuration file
2. **`ppo/utils/dynamic_config.py`**: Dynamic configuration system (single source of truth)
3. **`ppo/models/trading_env.py`**: Environment-specific configuration

### Configuration Priority

1. **Dynamic Config** (highest priority)
2. **Config.py** (fallback)
3. **Hardcoded Defaults** (last resort)

### Configuration Sync

- **Auto-sync**: Dynamic config syncs with config.py on initialization
- **Listener System**: Environments notified when config changes
- **Thread-safe**: Configuration updates are thread-safe

---

## 📊 Performance Metrics

### Current Performance (Seed 42)

| Metric | Value | Status |
|--------|-------|--------|
| **Total Return** | 12.71% | ✅ Excellent |
| **Sharpe Ratio** | 1.222 | ✅ Good |
| **Max Drawdown** | 7.26% | ✅ Acceptable |
| **Win Rate** | 90.0% | ✅ Excellent |
| **Total Trades** | 64 | ✅ Active |
| **Volatility** | 9.34% | ✅ Stable |
| **Profit Factor** | 1.30 | ✅ Profitable |

### Validation Status

- ✅ **3-Seed Validation**: PASSED (3/3 seeds meet thresholds)
- ✅ **Paper Trading Ready**: YES
- ✅ **Model Robustness**: ACCEPTABLE
- ⚠️ **Seed Sensitivity**: Moderate (acceptable variance)

---

## 🎯 Recommendations

### Immediate Actions

1. ✅ **Use Seed 42** for paper trading (best risk-adjusted performance)
2. ⚠️ **Monitor Drawdown** closely (alert at 7%)
3. ✅ **Continue Training** at 300k timesteps (sufficient for validation)

### Future Improvements

1. **Enable TCN**: Fix TCN initialization to use temporal memory
2. **Scale Training**: Consider 400k-500k timesteps for production
3. **Reduce Seed Sensitivity**: Implement ensemble methods
4. **Improve Risk Management**: Tighten drawdown controls (target < 6%)

---

**Document Status**: ✅ Complete  
**Last Review**: 2025-11-30  
**Next Review**: After next training run

