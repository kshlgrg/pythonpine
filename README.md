
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

## Detailed API Reference

### Trend Indicators (`trend.py`)

| Function | Signature | Inputs | Description |
|----------|-----------|--------|-------------|
| `sma` | `sma(priceList, length=14)` | `priceList`: List of prices<br>`length`: Period | Simple Moving Average. |
| `dema` | `dema(priceList, length=14)` | `priceList`: List of prices<br>`length`: Period | Double Exponential Moving Average. |
| `tema` | `tema(priceList, length=14)` | `priceList`: List of prices<br>`length`: Period | Triple Exponential Moving Average. |
| `wma` | `wma(priceList, length=14)` | `priceList`: List of prices<br>`length`: Period | Weighted Moving Average. |
| `hma` | `hma(priceList, length=14)` | `priceList`: List of prices<br>`length`: Period | Hull Moving Average. |
| `vwma` | `vwma(priceList, volumeList, length=14)` | `priceList`: List of prices<br>`volumeList`: List of volumes<br>`length`: Period | Volume Weighted Moving Average. |
| `kama` | `kama(priceList, length=10, fastend=2, slowend=30)` | `priceList`: List of prices<br>`length`: Period<br>`fastend`: Fast SC<br>`slowend`: Slow SC | Kaufman Adaptive Moving Average. |
| `supertrend` | `supertrend(highList, lowList, closeList, period=10, multiplier=3.0)` | `highList`, `lowList`, `closeList`: Price lists<br>`period`: ATR period<br>`multiplier`: ATR multiplier | SuperTrend indicator. Returns trend direction (1 or -1). |
| `vortex_indicator` | `vortex_indicator(highList, lowList, closeList, length=14)` | `highList`, `lowList`, `closeList`: Price lists<br>`length`: Period | Vortex Indicator. Returns `(vi_plus, vi_minus)`. |
| `aroon` | `aroon(highList, lowList, length=14)` | `highList`, `lowList`: Price lists<br>`length`: Period | Aroon Up/Down indicator. Returns `(aroon_up, aroon_down)`. |
| `linear_regression` | `linear_regression(priceList, length=14)` | `priceList`: List of prices<br>`length`: Period | Linear Regression Line (Endpoint Moving Average). |
| `donchian_channel` | `donchian_channel(highList, lowList, length=20)` | `highList`, `lowList`: Price lists<br>`length`: Period | Donchian Channel. Returns `(upper, lower, mid)`. |
| `fama` | `fama(priceList, length=10)` | `priceList`: List of prices<br>`length`: Period | Fractal Adaptive Moving Average (Wrapper for KAMA). |
| `moving_average_envelope` | `moving_average_envelope(priceList, length=20, percent=2.5)` | `priceList`: List of prices<br>`length`: Period<br>`percent`: Envelope percentage | Moving Average Envelope. Returns `(upper, lower, ma)`. |
| `ichimoku_cloud` | `ichimoku_cloud(highList, lowList, closeList, tenkan=9, kijun=26, senkou=52)` | `highList`, `lowList`, `closeList`: Price lists<br>`tenkan`, `kijun`, `senkou`: Periods | Ichimoku Cloud. Returns `(conversion, base, span_a, span_b, lagging)`. |
| `parabolic_sar` | `parabolic_sar(highList, lowList, step=0.02, max_af=0.2)` | `highList`, `lowList`: Price lists<br>`step`: Acceleration step<br>`max_af`: Max acceleration | Parabolic SAR. Returns list of SAR values. |
| `guppy_mma` | `guppy_mma(priceList)` | `priceList`: List of prices | Guppy Multiple Moving Averages. Returns `(short_emas, long_emas)`. |
| `ma_slope_angle` | `ma_slope_angle(maList)` | `maList`: List of moving average values | Moving Average Slope Angle in degrees. |
| `zero_lag_ema` | `zero_lag_ema(priceList, length=14)` | `priceList`: List of prices<br>`length`: Period | Zero-Lag EMA. |
| `median_price` | `median_price(highList, lowList)` | `highList`, `lowList`: Price lists | Median Price `(High + Low) / 2`. |
| `typical_price` | `typical_price(highList, lowList, closeList)` | `highList`, `lowList`, `closeList`: Price lists | Typical Price `(High + Low + Close) / 3`. |

### Momentum Indicators (`momentum.py`)

| Function | Signature | Inputs | Description |
|----------|-----------|--------|-------------|
| `williams_percent_r` | `williams_percent_r(closeList, highList, lowList, period=14)` | `closeList`, `highList`, `lowList`: Price lists<br>`period`: Lookback | Williams %R. |
| `dmi_adx` | `dmi_adx(highList, lowList, closeList, period=14)` | `highList`, `lowList`, `closeList`: Price lists<br>`period`: Lookback | DMI/ADX indicator. Returns `(plus_di, minus_di, adx)`. |
| `momentum` | `momentum(closeList, period=10)` | `closeList`: List of prices<br>`period`: Lookback | Momentum Indicator (Price difference). |
| `elder_impulse` | `elder_impulse(closeList, ema_period=13, macd_fast=12, macd_slow=26)` | `closeList`: Price list<br>`ema_period`: EMA Period<br>`macd_fast/slow`: MACD settings | Elder Impulse System. Returns list of 1 (Green), -1 (Red), 0 (Blue). |
| `schaff_trend_cycle` | `schaff_trend_cycle(closeList, fast=23, slow=50, cycle=10)` | `closeList`: Price list<br>`fast/slow`: MACD settings<br>`cycle`: Stoch cycle | Schaff Trend Cycle. |
| `chande_momentum_oscillator` | `chande_momentum_oscillator(closeList, period=14)` | `closeList`: Price list<br>`period`: Lookback | Chande Momentum Oscillator. |
| `relative_vigor_index` | `relative_vigor_index(openList, highList, lowList, closeList, period=10)` | `openList`, `highList`, `lowList`, `closeList`: Price lists<br>`period`: Lookback | Relative Vigor Index. Returns `(rvi, signal)`. |
| `rsi` | `rsi(closeList, period=14)` | `closeList`: Price list<br>`period`: Lookback | Relative Strength Index. |
| `stoch_rsi` | `stoch_rsi(closeList, rsi_period=14, stoch_period=14)` | `closeList`: Price list<br>`rsi_period`: RSI Period<br>`stoch_period`: Stoch Period | Stochastic RSI. |
| `stochastic_oscillator` | `stochastic_oscillator(closeList, highList, lowList, period=14)` | `closeList`, `highList`, `lowList`: Price lists<br>`period`: Lookback | Stochastic Oscillator %K. |
| `macd` | `macd(closeList, fast=12, slow=26, signal=9)` | `closeList`: Price list<br>`fast`: Fast EMA<br>`slow`: Slow EMA<br>`signal`: Signal EMA | MACD. Returns `(macd_line, signal_line, histogram)`. |
| `roc` | `roc(closeList, period=12)` | `closeList`: Price list<br>`period`: Lookback | Rate of Change (Percentage). |
| `cci` | `cci(closeList, highList, lowList, period=20)` | `closeList`, `highList`, `lowList`: Price lists<br>`period`: Lookback | Commodity Channel Index. |
| `trix` | `trix(closeList, period=15)` | `closeList`: Price list<br>`period`: Lookback | TRIX (Triple Exponential Average). |
| `ultimate_oscillator` | `ultimate_oscillator(closeList, highList, lowList, s1=7, s2=14, s3=28)` | `closeList`, `highList`, `lowList`: Price lists<br>`s1, s2, s3`: Periods | Ultimate Oscillator. |
| `true_strength_index` | `true_strength_index(closeList, long=25, short=13)` | `closeList`: Price list<br>`long`: Long smoothing<br>`short`: Short smoothing | True Strength Index (TSI). |
| `connors_rsi` | `connors_rsi(closeList, rsi_period=3, streak_rsi_period=2, rank_period=100)` | `closeList`: Price list<br>Periods for calculation | Connors RSI (Composite of RSI, Streak RSI, and Rank). |
| `rsx` | `rsx(closeList, period=14)` | `closeList`: Price list<br>`period`: Lookback | RSX (Smoothed RSI approximation). |
| `slope_of_ema` | `slope_of_ema(closeList, period=14)` | `closeList`: Price list<br>`period`: EMA Period | Slope of EMA. |
| `directional_trend_index` | `directional_trend_index(closeList, period=14)` | `closeList`: Price list<br>`period`: Lookback | Directional Trend Index. |

### Volatility Indicators (`volatility.py`)

| Function | Signature | Inputs | Description |
|----------|-----------|--------|-------------|
| `atr` | `atr(highList, lowList, closeList, period=14)` | `highList`, `lowList`, `closeList`: Price lists<br>`period`: Lookback | Average True Range. |
| `bollinger_bands` | `bollinger_bands(priceList, period=20, stddev=2)` | `priceList`: List of prices<br>`period`: SMA Period<br>`stddev`: Std Dev Multiplier | Bollinger Bands. Returns `(upper, lower, mid)`. |
| `keltner_channel` | `keltner_channel(highList, lowList, closeList, period=20, multiplier=2)` | `highList`, `lowList`, `closeList`: Price lists<br>`period`: EMA Period<br>`multiplier`: ATR Multiplier | Keltner Channel. Returns `(upper, lower, mid)`. |
| `donchian_channel_width` | `donchian_channel_width(highList, lowList, period=20)` | `highList`, `lowList`: Price lists<br>`period`: Lookback | Width of Donchian Channel. |
| `true_range` | `true_range(highList, lowList, closeList)` | `highList`, `lowList`, `closeList`: Price lists | True Range for each bar. |
| `standard_deviation` | `standard_deviation(priceList, period=20)` | `priceList`: List of prices<br>`period`: Lookback | Rolling Standard Deviation. |
| `chaikin_volatility` | `chaikin_volatility(highList, lowList, period=10)` | `highList`, `lowList`: Price lists<br>`period`: Lookback | Chaikin Volatility. |
| `bollinger_percent_b` | `bollinger_percent_b(priceList, period=20, stddev=2)` | `priceList`: List of prices<br>`period`: SMA Period<br>`stddev`: Multiplier | Bollinger %B (position within bands). |
| `historical_volatility` | `historical_volatility(priceList, period=20)` | `priceList`: List of prices<br>`period`: Lookback | Annualized Historical Volatility. |
| `relative_volatility_index` | `relative_volatility_index(stddevList, up_stddev, down_stddev, period=14)` | `stddevList`: List of std devs<br>`period`: Lookback | Relative Volatility Index. |
| `normalized_atr` | `normalized_atr(atrList, priceList)` | `atrList`: ATR values<br>`priceList`: Prices | ATR normalized by price. |
| `volatility_ratio` | `volatility_ratio(atrList, period=14)` | `atrList`: ATR values<br>`period`: SMA Period | Ratio of current ATR to its moving average. |
| `atr_percent` | `atr_percent(atrList, priceList)` | `atrList`: ATR values<br>`priceList`: Prices | ATR as a percentage of price. |
| `ulcer_index` | `ulcer_index(priceList, period=14)` | `priceList`: List of prices<br>`period`: Lookback | Ulcer Index (Risk measure). |
| `mass_index` | `mass_index(highList, lowList, ema_period=9)` | `highList`, `lowList`: Price lists<br>`ema_period`: Period | Mass Index. |
| `garman_klass_volatility` | `garman_klass_volatility(openList, highList, lowList, closeList, period=10)` | `openList`, `highList`, `lowList`, `closeList`: Price lists<br>`period`: Lookback | Garman-Klass Volatility estimator. |
| `parkinson_volatility` | `parkinson_volatility(highList, lowList, period=10)` | `highList`, `lowList`: Price lists<br>`period`: Lookback | Parkinson Volatility estimator. |
| `range_based_volatility` | `range_based_volatility(highList, lowList, closeList)` | `highList`, `lowList`, `closeList`: Price lists | Simple Range-Based Volatility. |

### Volume Indicators (`volume.py`)

| Function | Signature | Inputs | Description |
|----------|-----------|--------|-------------|
| `obv` | `obv(closeList, volumeList)` | `closeList`: Prices<br>`volumeList`: Volumes | On-Balance Volume. |
| `vwap` | `vwap(highList, lowList, closeList, volumeList)` | `highList`, `lowList`, `closeList`: Prices<br>`volumeList`: Volumes | Volume Weighted Average Price (Cumulative). |
| `adl` | `adl(highList, lowList, closeList, volumeList)` | `highList`, `lowList`, `closeList`: Prices<br>`volumeList`: Volumes | Accumulation/Distribution Line. |
| `cmf` | `cmf(highList, lowList, closeList, volumeList, period=20)` | `highList`, `lowList`, `closeList`: Prices<br>`volumeList`: Volumes<br>`period`: Lookback | Chaikin Money Flow. |
| `volume_oscillator` | `volume_oscillator(volumeList, short_period=14, long_period=28)` | `volumeList`: Volumes<br>`short_period`, `long_period`: Periods | Volume Oscillator (Percentage difference between MAs). |
| `force_index` | `force_index(closeList, volumeList)` | `closeList`: Prices<br>`volumeList`: Volumes | Force Index. |
| `mfi` | `mfi(highList, lowList, closeList, volumeList, period=14)` | `highList`, `lowList`, `closeList`: Prices<br>`volumeList`: Volumes<br>`period`: Lookback | Money Flow Index. |
| `ease_of_movement` | `ease_of_movement(highList, lowList, volumeList, period=14)` | `highList`, `lowList`: Prices<br>`volumeList`: Volumes<br>`period`: Lookback | Ease of Movement. |
| `volume_roc` | `volume_roc(volumeList, period=14)` | `volumeList`: Volumes<br>`period`: Lookback | Volume Rate of Change. |
| `volume_delta` | `volume_delta(buy_volume, sell_volume)` | `buy_volume`, `sell_volume`: Volume lists | Difference between buy and sell volume. |
| `intraday_intensity` | `intraday_intensity(closeList, highList, lowList, volumeList, period=14)` | `closeList`, `highList`, `lowList`: Prices<br>`volumeList`: Volumes<br>`period`: Lookback | Intraday Intensity Index. |
| `price_volume_trend` | `price_volume_trend(closeList, volumeList)` | `closeList`: Prices<br>`volumeList`: Volumes | Price Volume Trend (PVT). |
| `vw_macd` | `vw_macd(closeList, volumeList, short_period=12, long_period=26, signal_period=9)` | `closeList`: Prices<br>`volumeList`: Volumes<br>MACD settings | Volume-Weighted MACD. Returns `(macd, signal, hist)`. |
| `smoothed_obv` | `smoothed_obv(obvList, period=14)` | `obvList`: OBV values<br>`period`: EMA Period | Smoothed On-Balance Volume. |
| `klinger_oscillator` | `klinger_oscillator(highList, lowList, closeList, volumeList, fast=34, slow=55, signal=13)` | Prices, Volumes, Settings | Klinger Volume Oscillator. Returns `(kvo, signal)`. |
| `volume_flow_indicator` | `volume_flow_indicator(closeList, volumeList, period=130, coef=0.2)` | Prices, Volumes, Settings | Volume Flow Indicator (VFI). |
| `pvi` | `pvi(closeList, volumeList)` | Prices, Volumes | Positive Volume Index. |
| `nvi` | `nvi(closeList, volumeList)` | Prices, Volumes | Negative Volume Index. |

### Price Action Indicators (`price_action.py`)

| Function | Signature | Inputs | Description |
|----------|-----------|--------|-------------|
| `pivot_points` | `pivot_points(high, low, close, method='classic')` | Single high, low, close values<br>`method`: 'classic', 'fibonacci', 'camarilla' | Returns dictionary of pivot levels (pp, r1, s1, etc.). |
| `price_roc` | `price_roc(priceList, period=14, scale=True)` | `priceList`: Prices<br>`period`: Lookback | Price Rate of Change. |
| `zigzag` | `zigzag(priceList, threshold=5.0, use_percentage=True)` | `priceList`: Prices<br>`threshold`: Change threshold | ZigZag indicator. Returns filtered points. |
| `heikin_ashi` | `heikin_ashi(openList, highList, lowList, closeList)` | Prices | Heikin Ashi candles. Returns `(ha_open, ha_high, ha_low, ha_close)`. |
| `renko_boxes` | `renko_boxes(closeList, box_size=1.0, show_full_boxes=False)` | `closeList`: Prices<br>`box_size`: Box size | Renko Boxes. Returns list of box prices. |
| `detect_engulfing` | `detect_engulfing(openList, closeList, require_body_ratio=1.0)` | Prices | Returns 1 (Bullish), -1 (Bearish), 0 (None). |
| `detect_pin_bar` | `detect_pin_bar(openList, highList, lowList, closeList, wick_ratio=2.0)` | Prices | Returns 1 (Bullish Pin Bar), -1 (Bearish Pin Bar), 0 (None). |
| `detect_double_top_bottom` | `detect_double_top_bottom(highList, lowList, threshold=0.005, min_spacing=2)` | Prices | Returns `(tops, bottoms)` indices. |
| `support_resistance_zones` | `support_resistance_zones(highList, lowList, sensitivity=5, method='pivot')` | Prices | Returns `(support_indices, resistance_indices)`. |
| `detect_doji` | `detect_doji(openList, closeList, highList, lowList, body_threshold=0.1)` | Prices | Returns 1 if Doji detected, else 0. |
| `detect_inside_outside_bars` | `detect_inside_outside_bars(highList, lowList)` | Prices | Returns 1 (Inside), -1 (Outside), 0 (None). |
| `detect_marubozu` | `detect_marubozu(openList, closeList, highList, lowList, tolerance=0.1)` | Prices | Returns 1 (Bullish), -1 (Bearish), 0 (None). |
| `detect_three_bar_reversal` | `detect_three_bar_reversal(closeList)` | Prices | Returns 1 (Bullish), -1 (Bearish), 0 (None). |
| `detect_fractals` | `detect_fractals(highList, lowList, window=2)` | Prices | Returns `(fractal_high_indices, fractal_low_indices)`. |
| `bar_range_ratio` | `bar_range_ratio(openList, closeList, highList, lowList)` | Prices | Ratio of body to total range. |
| `wick_ratio` | `wick_ratio(openList, closeList, highList, lowList)` | Prices | Ratio of wicks to total range. |
| `high_low_breakout` | `high_low_breakout(priceList, lookback=20)` | Prices | Returns 1 (High Breakout), -1 (Low Breakout), 0 (None). |
| `trend_candle_strength` | `trend_candle_strength(openList, closeList, length=20)` | Prices | Percentage of bullish candles in lookback period. |
| `price_action_score` | `price_action_score(...)` | various pattern arrays | Composite score of price action signals. |

### Cycle Indicators (`cycles.py`)

| Function | Signature | Inputs | Description |
|----------|-----------|--------|-------------|
| `fisher_transform` | `fisher_transform(source_type="hl2", ...)` | Source type, settings | Fisher Transform values. |
| `hilbert_transform` | `hilbert_transform(priceList)` | Prices | Quadrature component of Hilbert Transform. |
| `ht_sine` | `ht_sine(priceList)` | Prices | Hilbert Transform SineWave. Returns `(sine, leadsine)`. |
| `ht_phase` | `ht_phase(priceList)` | Prices | Instantaneous Phase in degrees. |
| `ht_trendline` | `ht_trendline(priceList, alpha=0.07)` | Prices | Hilbert Transform Trendline. |
| `ht_dominant_cycle` | `ht_dominant_cycle(priceList, min_period=10, max_period=50)` | Prices | Dominant Cycle Period. |
| `ht_itrend` | `ht_itrend(priceList, alpha=0.07)` | Prices | Instantaneous Trendline. |
| `detrended_price_oscillator` | `detrended_price_oscillator(priceList, length=14)` | Prices | Detrended Price Oscillator. |
| `laguerre_rsi` | `laguerre_rsi(priceList, gamma=0.5)` | Prices | Laguerre RSI. |
| `qstick` | `qstick(openList, closeList, length=14)` | Prices | QStick Indicator. |
| `stochastic_momentum_index` | `stochastic_momentum_index(closeList, highList, lowList, length=14...)` | Prices | SMI. Returns `(smi_k, smi_d)`. |
| `adaptive_cycle_divergence` | `adaptive_cycle_divergence(priceList)` | Prices | Adaptive Cycle Divergence. |
| `phase_accumulation_cycle` | `phase_accumulation_cycle(priceList)` | Prices | Phase Accumulation Cycle Period. |
| `inverse_fisher_transform` | `inverse_fisher_transform(series)` | Series | Inverse Fisher Transform. |
| `mama_fama` | `mama_fama(priceList, ...)` | Prices | MESA Adaptive Moving Average. Returns `(mama, fama)`. |
| `super_smoother` | `super_smoother(priceList, period)` | Prices | Super Smoother Filter. |
| `roofing_filter` | `roofing_filter(priceList)` | Prices | Roofing Filter. |
| `center_of_gravity` | `center_of_gravity(priceList, length=10)` | Prices | Center of Gravity Oscillator. |
| `bandpass_filter` | `bandpass_filter(priceList, period, bandwidth=0.3)` | Prices | Bandpass Filter. |
| `dc_based_rsi` | `dc_based_rsi(priceList, cycleList)` | Prices, Cycle Lengths | RSI with adaptive cycle length. |
| `cyber_cycle` | `cyber_cycle(priceList, alpha=0.07)` | Prices | Cyber Cycle. |

### Statistical Indicators (`statistical.py`)

| Function | Signature | Inputs | Description |
|----------|-----------|--------|-------------|
| `z_score` | `z_score(priceList, period=20)` | `priceList`, `period` | Z-Score of price relative to moving average/std. |
| `rolling_mean_std` | `rolling_mean_std(priceList, period=20)` | `priceList`, `period` | Returns `(means, stds)`. |
| `skewness_kurtosis` | `skewness_kurtosis(priceList, period=20)` | `priceList`, `period` | Returns `(skewness, kurtosis)`. |
| `percentile_rank` | `percentile_rank(priceList, period=20)` | `priceList`, `period` | Percentile rank of current price in window. |
| `median_absolute_deviation` | `median_absolute_deviation(priceList, period=20)` | `priceList`, `period` | MAD (Median Absolute Deviation). |

### Experimental Indicators (`experimental.py`)

| Function | Signature | Inputs | Description |
|----------|-----------|--------|-------------|
| `fractal_dimension` | `fractal_dimension(priceList, period=10)` | `priceList`, `period` | Fractal Dimension Index (Katz). |
| `kalman_filter_slope` | `kalman_filter_slope(priceList, R=0.01, Q=0.001)` | `priceList`, settings | Slope of Kalman Filter estimate. |
| `hurst_exponent` | `hurst_exponent(priceList, max_lag=20)` | `priceList` | Hurst Exponent. |
| `shannon_entropy` | `shannon_entropy(priceList, period=20)` | `priceList`, `period` | Shannon Entropy of price distribution. |
| `kl_divergence` | `kl_divergence(priceList, period=20)` | `priceList`, `period` | KL Divergence against normal distribution. |
| `tsf` | `tsf(priceList, period=14)` | `priceList`, `period` | Time Series Forecast (Linear Regression value). |
| `roofing_filter` | `roofing_filter(priceList)` | `priceList` | Ehlers Roofing Filter (duplicate in cycles, verify usage). |
| `smoothed_heikin_ashi` | `smoothed_heikin_ashi(closeList, openList, period=10)` | Prices | Smoothed Heikin Ashi Oscillator. |

### Time/Session Indicators (`time_session.py`)

| Function | Signature | Inputs | Description |
|----------|-----------|--------|-------------|
| `normalized_time_of_day` | `normalized_time_of_day(timestamps)` | `timestamps` (datetime) | Time of day normalized 0-1. |
| `session_high_low` | `session_high_low(highList, lowList, start, end)` | Prices, indices | High/Low for specific session. |
| `session_overlay_flags` | `session_overlay_flags(timestamps)` | `timestamps` | Returns flags `(asia, london, ny)`. |
| `day_of_week` | `day_of_week(timestamps)` | `timestamps` | Returns day of week integer (0=Mon). |
| `time_since_last_extreme` | `time_since_last_extreme(priceList, extreme='high')` | Prices, type | Bars since last high/low. |

### Meta/Composite Indicators (`meta.py`)

| Function | Signature | Inputs | Description |
|----------|-----------|--------|-------------|
| `ma_crossover_count` | `ma_crossover_count(fast_ma, slow_ma)` | MA lists | Count of crossovers. |
| `indicator_consensus` | `indicator_consensus(*signals)` | Multiple signal lists | Average of signals. |
| `momentum_volatility_score` | `momentum_volatility_score(mom, vol)` | Indicator lists | Product of momentum and volatility. |
| `trend_strength_score` | `trend_strength_score(priceList, period=14)` | `priceList` | Average gain over period. |
| `macd_histogram_angle` | `macd_histogram_angle(macd_hist)` | MACD Hist list | Angle of histogram change. |
| `rsi_divergence_count` | `rsi_divergence_count(priceList, rsiList, lookback)` | Prices, RSI | Count of simple divergences. |
| `volume_spike` | `volume_spike(volumeList, multiplier=2.0)` | `volumeList` | Returns 1 if volume spike detected. |
| `mtf_ema_alignment` | `mtf_ema_alignment(ema_short, ema_mid, ema_long)` | MA lists | Returns 1 if Short > Mid > Long. |
| `trend_reversal_likelihood` | `trend_reversal_likelihood(priceList, rsiList)` | Prices, RSI | Simple reversal heuristic. |
| `consolidation_detector` | `consolidation_detector(priceList, period, threshold)` | Prices | Returns 1 if volatility (std dev) is low. |

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Disclaimer

This library is for educational and research purposes. Trading involves risk, and past performance does not guarantee future results. Always do your own research before making trading decisions.
