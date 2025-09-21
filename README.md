# PythonPine

## A Pine Script-style indicator library in Python using MetaTrader5 OHLCV data — 100+ real-time indicators for algorithmic trading.

### About
PythonPine is a comprehensive library that provides 100+ technical indicators implemented in Python, designed to work with MetaTrader5 OHLCV data. The library follows Pine Script conventions, making it familiar to TradingView users while leveraging Python's flexibility for algorithmic trading applications.

### Features
- 100+ technical indicators across multiple categories
- Compatible with MetaTrader5 OHLCV data
- Pine Script-style function signatures
- Real-time calculation capabilities
- Optimized for algorithmic trading
- Easy-to-use API

### Installation

Install PythonPine using pip:

```bash
pip install pythonpine
```

Or install from source:

```bash
git clone https://github.com/kshlgrg/pythonpine.git
cd pythonpine
pip install -e .
```

### Basic Usage

```python
import pythonpine as pp

# Example OHLCV data
open_prices = [100, 101, 102, 103, 104]
high_prices = [105, 106, 107, 108, 109]
low_prices = [99, 100, 101, 102, 103]
close_prices = [104, 105, 106, 107, 108]
volume = [1000, 1100, 1200, 1300, 1400]

# Calculate indicators
sma_20 = pp.sma(close_prices, 20)
rsi_14 = pp.rsi(close_prices, 14)
macd_line, signal_line, histogram = pp.macd(close_prices, 12, 26, 9)

# Use with MetaTrader5 data
import MetaTrader5 as mt5

# Initialize MT5 connection
if not mt5.initialize():
    print("Initialize() failed")
    quit()

# Get historical data
rates = mt5.copy_rates_from_pos("EURUSD", mt5.TIMEFRAME_M1, 0, 1000)
close_prices = [rate['close'] for rate in rates]

# Calculate indicators
rsi = pp.rsi(close_prices, 14)
print(f"Current RSI: {rsi[-1]}")
```

## Indicator Categories

### Trend Indicators

| Function | Signature | Description |
|----------|-----------|-------------|
| `sma` | `sma(priceList, length=14)` | Simple Moving Average |
| `dema` | `dema(priceList, length=14)` | Double Exponential Moving Average |
| `tema` | `tema(priceList, length=14)` | Triple Exponential Moving Average |
| `wma` | `wma(priceList, length=14)` | Weighted Moving Average |
| `hma` | `hma(priceList, length=14)` | Hull Moving Average |
| `vwma` | `vwma(priceList, volumeList, length=14)` | Volume Weighted Moving Average |
| `kama` | `kama(priceList, length=10, fastend=2, slowend=30)` | Kaufman Adaptive Moving Average |
| `supertrend` | `supertrend(highList, lowList, closeList, period=10, multiplier=3.0)` | SuperTrend indicator |
| `vortex_indicator` | `vortex_indicator(highList, lowList, closeList, length=14)` | Vortex Indicator |
| `aroon` | `aroon(highList, lowList, length=14)` | Aroon Up/Down indicator |
| `linear_regression` | `linear_regression(priceList, length=14)` | Linear Regression Line |
| `donchian_channel` | `donchian_channel(highList, lowList, length=20)` | Donchian Channel |
| `fama` | `fama(priceList, length=10)` | Fractal Adaptive Moving Average |
| `moving_average_envelope` | `moving_average_envelope(priceList, length=20, percent=2.5)` | Moving Average Envelope |
| `ichimoku_cloud` | `ichimoku_cloud(highList, lowList, closeList, tenkan=9, kijun=26, senkou=52)` | Ichimoku Cloud |
| `parabolic_sar` | `parabolic_sar(highList, lowList, step=0.02, max_af=0.2)` | Parabolic SAR |
| `guppy_mma` | `guppy_mma(priceList)` | Guppy Multiple Moving Averages |
| `ma_slope_angle` | `ma_slope_angle(maList)` | Moving Average Slope Angle |
| `zero_lag_ema` | `zero_lag_ema(priceList, length=14)` | Zero-Lag EMA |
| `median_price` | `median_price(highList, lowList)` | Median Price |
| `typical_price` | `typical_price(highList, lowList, closeList)` | Typical Price |

### Momentum Indicators

| Function | Signature | Description |
|----------|-----------|-------------|
| `williams_percent_r` | `williams_percent_r(closeList, highList, lowList, period=14)` | Williams %R |
| `dmi_adx` | `dmi_adx(highList, lowList, closeList, period=14)` | DMI/ADX indicator |
| `momentum` | `momentum(closeList, period=10)` | Momentum Indicator |
| `elder_impulse` | `elder_impulse(closeList, ema_period=13, macd_fast=12, macd_slow=26)` | Elder Impulse System |
| `schaff_trend_cycle` | `schaff_trend_cycle(closeList, fast=23, slow=50, cycle=10)` | Schaff Trend Cycle |
| `chande_momentum_oscillator` | `chande_momentum_oscillator(closeList, period=14)` | Chande Momentum Oscillator |
| `relative_vigor_index` | `relative_vigor_index(openList, highList, lowList, closeList, period=10)` | Relative Vigor Index |
| `rsi` | `rsi(closeList, period=14)` | Relative Strength Index |
| `stoch_rsi` | `stoch_rsi(closeList, rsi_period=14, stoch_period=14)` | Stochastic RSI |
| `stochastic_oscillator` | `stochastic_oscillator(closeList, highList, lowList, period=14)` | Stochastic Oscillator |
| `macd` | `macd(closeList, fast=12, slow=26, signal=9)` | MACD |
| `roc` | `roc(closeList, period=12)` | Rate of Change |
| `cci` | `cci(closeList, highList, lowList, period=20)` | Commodity Channel Index |
| `trix` | `trix(closeList, period=15)` | TRIX |
| `ultimate_oscillator` | `ultimate_oscillator(closeList, highList, lowList, s1=7, s2=14, s3=28)` | Ultimate Oscillator |
| `true_strength_index` | `true_strength_index(closeList, long=25, short=13)` | True Strength Index |
| `connors_rsi` | `connors_rsi(closeList, rsi_period=3, streak_rsi_period=2, rank_period=100)` | Connors RSI |
| `vortex_indicator` | `vortex_indicator(highList, lowList, closeList, period=14)` | Vortex Indicator |
| `rsx` | `rsx(closeList, period=14)` | Smoothed RSI |
| `slope_of_ema` | `slope_of_ema(closeList, period=14)` | Slope of EMA |
| `directional_trend_index` | `directional_trend_index(closeList, period=14)` | Directional Trend Index |

### Volatility Indicators

| Function | Signature | Description |
|----------|-----------|-------------|
| `atr` | `atr(highList, lowList, closeList, period=14)` | Average True Range |
| `bollinger_bands` | `bollinger_bands(closeList, period=20, std_dev=2)` | Bollinger Bands |
| `keltner_channel` | `keltner_channel(highList, lowList, closeList, period=20, multiplier=2)` | Keltner Channel |
| `true_range` | `true_range(highList, lowList, closeList)` | True Range |

### Price Action Indicators

| Function | Signature | Description |
|----------|-----------|-------------|
| `inside_bar` | `inside_bar(highList, lowList)` | Inside Bar pattern |
| `outside_bar` | `outside_bar(highList, lowList)` | Outside Bar pattern |
| `doji` | `doji(openList, closeList, threshold=0.1)` | Doji candlestick pattern |
| `hammer` | `hammer(openList, highList, lowList, closeList)` | Hammer candlestick pattern |
| `shooting_star` | `shooting_star(openList, highList, lowList, closeList)` | Shooting Star pattern |

### Cycles Indicators

| Function | Signature | Description |
|----------|-----------|-------------|
| `mesa_adaptive_filter` | `mesa_adaptive_filter(closeList, fast_limit=0.5, slow_limit=0.05)` | MESA Adaptive Filter |
| `hilbert_transform` | `hilbert_transform(closeList)` | Hilbert Transform |
| `cycle_period` | `cycle_period(closeList, period=20)` | Cycle Period indicator |

### Experimental Indicators

| Function | Signature | Description |
|----------|-----------|-------------|
| `fibonacci_retracement` | `fibonacci_retracement(highList, lowList)` | Fibonacci Retracement levels |
| `elliott_wave` | `elliott_wave(closeList, period=20)` | Elliott Wave analysis |
| `gann_angles` | `gann_angles(closeList, period=20)` | Gann Angle analysis |

### Volume Indicators

| Function | Signature | Description |
|----------|-----------|-------------|
| `obv` | `obv(closeList, volumeList)` | On-Balance Volume |
| `ad_line` | `ad_line(highList, lowList, closeList, volumeList)` | Accumulation/Distribution Line |
| `mfi` | `mfi(highList, lowList, closeList, volumeList, period=14)` | Money Flow Index |
| `ease_of_movement` | `ease_of_movement(highList, lowList, volumeList, period=14)` | Ease of Movement |
| `volume_price_trend` | `volume_price_trend(closeList, volumeList)` | Volume Price Trend |
| `chaikin_oscillator` | `chaikin_oscillator(highList, lowList, closeList, volumeList, fast=3, slow=10)` | Chaikin Oscillator |
| `klinger_oscillator` | `klinger_oscillator(highList, lowList, closeList, volumeList, fast=34, slow=55)` | Klinger Oscillator |

### Meta Indicators

| Function | Signature | Description |
|----------|-----------|-------------|
| `correlation` | `correlation(series1, series2, period=20)` | Correlation between two series |
| `covariance` | `covariance(series1, series2, period=20)` | Covariance between two series |
| `beta` | `beta(stockList, marketList, period=20)` | Beta coefficient |
| `alpha` | `alpha(stockList, marketList, riskFreeRate, period=20)` | Alpha coefficient |

### Time/Session Indicators

| Function | Signature | Description |
|----------|-----------|-------------|
| `session_high` | `session_high(highList, session_start, session_end)` | Session High |
| `session_low` | `session_low(lowList, session_start, session_end)` | Session Low |
| `session_volume` | `session_volume(volumeList, session_start, session_end)` | Session Volume |
| `time_weighted_average` | `time_weighted_average(priceList, timeList, period=20)` | Time Weighted Average Price |

### Statistical Indicators

| Function | Signature | Description |
|----------|-----------|-------------|
| `variance` | `variance(priceList, period=20)` | Price Variance |
| `standard_deviation` | `standard_deviation(priceList, period=20)` | Standard Deviation |
| `skewness` | `skewness(priceList, period=20)` | Price Skewness |
| `kurtosis` | `kurtosis(priceList, period=20)` | Price Kurtosis |
| `zscore` | `zscore(priceList, period=20)` | Z-Score |
| `percentile` | `percentile(priceList, period=20, percentile=50)` | Percentile |
| `entropy` | `entropy(priceList, period=20)` | Shannon Entropy |

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

If you find this library useful, please consider giving it a star ⭐ on GitHub.

## Disclaimer

This library is for educational and research purposes. Trading involves risk, and past performance does not guarantee future results. Always do your own research before making trading decisions.
