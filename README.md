# PythonPine v2.0 - High-Performance Technical Indicators

**Vectorized Pine Script indicators** for machine learning and algorithmic trading. 10-100x faster than loop-based implementations.

## Features

- ⚡ **NumPy/Pandas Vectorization** - Millions of bars in seconds
- 🧠 **ML Alpha Module** - Z-score, percentile rank, fractional differentiation
- 📊 **50+ Indicators** - RSI, MACD, ATR, Bollinger, SuperTrend, Ichimoku...
- 🔄 **Pine Script Parity** - Same signatures, same math
- 📈 **Price Action** - Vectorized pattern detection

## Installation

```bash
pip install pythonpine
```

## Quick Start

```python
import numpy as np
from pythonpine import rsi, macd, atr, zscore

# Your price data
close = np.random.randn(10000).cumsum() + 100
high = close + np.abs(np.random.randn(10000)) * 0.5
low = close - np.abs(np.random.randn(10000)) * 0.5

# Calculate indicators (returns NumPy arrays)
rsi_values = rsi(close, 14)
macd_line, signal, hist = macd(close)
atr_values = atr(high, low, close, 14)

# ML preprocessing
rsi_zscore = zscore(rsi_values, 20)  # Normalized for ML
```

## ML Alpha Module

```python
from pythonpine import zscore, percentile_rank, log_returns, fractional_diff

# Stationarity transforms
returns = log_returns(close)
frac_diff = fractional_diff(close, d=0.4)  # López de Prado method

# Feature engineering
rsi_z = zscore(rsi(close), 50)  # Z-score normalized RSI
rsi_rank = percentile_rank(rsi(close), 100)  # Percentile rank
```

## Benchmarks

| Indicator | Legacy (loop) | V2 (vectorized) | Speedup |
|-----------|---------------|-----------------|---------|
| RSI | 1.2s | 0.01s | **120x** |
| MACD | 0.8s | 0.008s | **100x** |
| ATR | 0.5s | 0.005s | **100x** |
| Bollinger | 0.9s | 0.007s | **130x** |

*Tested on 1M bars*

## Indicator Reference

### Momentum
- `rsi(close, 14)` - Relative Strength Index
- `macd(close, 12, 26, 9)` - MACD line, signal, histogram
- `stochastic(close, high, low)` - Stochastic Oscillator
- `adx(high, low, close)` - Average Directional Index
- `cci(close, high, low)` - Commodity Channel Index

### Volatility
- `atr(high, low, close, 14)` - Average True Range
- `bollinger_bands(close, 20, 2)` - Upper, lower, middle
- `keltner_channel(high, low, close)` - Keltner bands
- `historical_volatility(close, 20)` - HV annualized

### Trend
- `sma(close, 20)` - Simple Moving Average
- `ema(close, 20)` - Exponential MA
- `supertrend(high, low, close)` - Trend direction + line
- `ichimoku(high, low, close)` - Full cloud

### Price Action
- `engulfing(o, h, l, c)` - Returns +1 (bullish), -1 (bearish), 0
- `doji(o, h, l, c)` - Doji detection
- `support_resistance_zones(h, l, c, v)` - Volume-based S/R

## License

MIT
