
🧠 PYTHONPINE

**An ultra-powerful Python library to compute over 100+ TradingView-style technical indicators** using real-time OHLCV data from MetaTrader5.

📦 Think of this as Pine Script in Python — built to help algorithmic traders, quants, and curious developers power their backtests, trading bots, and research with advanced indicator logic.

---

## 🔧 Features

✅ 100+ technical indicators categorized into:
- 🟩 Trend Indicators  
- 🟦 Momentum Indicators  
- 🟧 Volatility Indicators  
- 🟨 Volume-Based Indicators  
- 🟪 Price Action & Support/Resistance  
- 🟥 Oscillators & Cycles  
- ⚫ Custom Composites  
- 🟤 Time/Session-Based  
- 🧠 Statistical / Nonlinear  

✅ Real-time OHLCV updates from MetaTrader5  
✅ Designed for clean usage — `import pythonpine` and start  
✅ Fully extensible — easily add your own indicators  
✅ Minimal setup, maximum power  

---

## 🚀 Quick Start

### 1. Install MetaTrader5 module
```bash
pip install MetaTrader5 numpy pandas scipy scikit-learn
````

### 2. Clone this repo

```bash
git clone https://github.com/kshgrg/pythonpine
cd pythonpine
```

### 3. Add the folder to your project or install locally

```bash
pip install -e .
```

---

## 🛠️ How to Use

### ✅ Step 1: Connect to MetaTrader5

```python
from pythonpine import *

connect_to_mt5(
    login=123456,
    password="yourpassword",
    server="yourserver",
    path="C:\\Path\\To\\Terminal64.exe"
)
```

### ✅ Step 2: Get OHLCV arrays

```python
open, high, low, close, volume = get_ohlcv_arrays("EURUSD")
```

### ✅ Step 3: Run background price updater (Optional for real-time trading)

Paste this in your main code to auto-update price arrays every minute:

```python
import time

while True:
    open, high, low, close, volume = get_ohlcv_arrays("EURUSD")
    time.sleep(60)  # Updates every 60 seconds
```

---

## 📊 Example Usage (Indicator Calculations)

```python
# EMA and RSI
ema_20 = ema(close, length=20)
rsi_14 = rsi(close, length=14)

# Bollinger Bands
upper, middle, lower = bollinger_bands(close, length=20, std_dev=2)

# MACD
macd_line, signal_line, histogram = macd(close)

# SuperTrend
supertrend, direction = supertrend_calc(high, low, close, atr_period=10, multiplier=3)
```

---

## 🧪 Full Indicator List

Expand to see all indicators:

<details>
<summary>Click to expand</summary>

### 🟩 Trend Indicators

EMA, SMA, DEMA, TEMA, WMA, HMA, VWMA, KAMA, SuperTrend, Vortex, Aroon, Linear Regression, Donchian, FAMA, MA Envelope

### 🟦 Momentum Indicators

RSI, Stoch RSI, Stochastic, MACD, ROC, CCI, TRIX, Ultimate Osc, Williams %R, DMI/ADX, Elder Impulse, Schaff, CMO, RVI

### 🟧 Volatility Indicators

ATR, Bollinger Bands, Keltner Channel, Donchian Width, True Range, Std Dev, Chaikin Vol, Boll %B, Hist Volatility

### 🟨 Volume-Based Indicators

OBV, VWAP, Accum/Dist, CMF, Vol Osc, Force Index, MFI, Ease of Move, Vol ROC, Vol Delta, Intraday Intensity

### 🟪 Price Action / Support-Resistance

Pivot Points, Price ROC, ZigZag, Heikin Ashi, Renko, Engulfing, Pin Bar, Double Top, S/R Zones, Pattern Count

### 🟥 Oscillators & Cycles

Fisher Transform, Hilbert Transforms, Ehler Trendline, DPO, Laguerre RSI, QStick, SMI, Adaptive Cycle, Inverse Fisher

### ⚫ Meta-Indicators

MA Crossover Count, Consensus Score, Momentum-Vol Composite, Trend Strength, MACD Angle, RSI Divergence, MTF EMA

### 🟤 Time/Session Based

Time of Day, Session High/Low, Market Sessions Overlay, Day of Week, Time Since High/Low

### 🧠 Statistical/Experimental

Z-Score, Rolling Stats, Skewness, Percentile, MAD, Fractal Dim, Garman-Klass, Kalman, Hurst, Entropy, TSF, Neural Score

</details>

---

## 📘 Built-in Functions Reference

All functions are available directly after importing `pythonpine`. The functions are grouped by category and support standard Python lists or NumPy arrays. Make sure to call `get_ohlcv_arrays()` to retrieve updated price data.

---

### 🔧 Utility Functions

| **Function**             | **Inputs**                                                   | **Description**                                                                    |
| ------------------------ | ------------------------------------------------------------ | ---------------------------------------------------------------------------------- |
| `initialize_data()`      | `source: str`, `symbol: str`, `timeframe: str`, `count: int` | Initializes global OHLCV arrays from selected data source (MetaTrader or cTrader). |
| `open(i=0)`              | `i: int`                                                     | Returns the open price at index `i`.                                               |
| `high(i=0)`              | `i: int`                                                     | Returns the high price at index `i`.                                               |
| `low(i=0)`               | `i: int`                                                     | Returns the low price at index `i`.                                                |
| `close(i=0)`             | `i: int`                                                     | Returns the close price at index `i`.                                              |
| `volume(i=0)`            | `i: int`                                                     | Returns the volume at index `i`.                                                   |
| `openList(start, end)`   | `start: int`, `end: int`                                     | Returns open prices from index `start` to `end` (inclusive).                       |
| `highList(start, end)`   | `start: int`, `end: int`                                     | Returns high prices from index `start` to `end` (inclusive).                       |
| `lowList(start, end)`    | `start: int`, `end: int`                                     | Returns low prices from index `start` to `end` (inclusive).                        |
| `closeList(start, end)`  | `start: int`, `end: int`                                     | Returns close prices from index `start` to `end` (inclusive).                      |
| `volumeList(start, end)` | `start: int`, `end: int`                                     | Returns volume values from index `start` to `end` (inclusive).                     |
| `highest(end)`           | `end: int`                                                   | Returns highest high from index `0` to `end - 1`.                                  |
| `lowest(end)`            | `end: int`                                                   | Returns lowest low from index `0` to `end - 1`.                                    |


---

### 📈 Trend Indicators

| Function Name                    | Inputs           | Description                                   | Tier           |
| -------------------------------- | ---------------- | --------------------------------------------- | -------------- |
| `ema(length)`                    | `length: int`    | Exponential Moving Average.                   | ✅ Core         |
| `sma(length)`                    | `length: int`    | Simple Moving Average.                        | ✅ Core         |
| `dema(length)`                   | `length: int`    | Double Exponential MA.                        | ✅ Core         |
| `tema(length)`                   | `length: int`    | Triple Exponential MA.                        | ✅ Core         |
| `wma(length)`                    | `length: int`    | Weighted MA.                                  | ✅ Core         |
| `hma(length)`                    | `length: int`    | Hull Moving Average.                          | ✅ Core         |
| `vwma(length)`                   | `length: int`    | Volume Weighted MA.                           | ✅ Core         |
| `kama(length)`                   | `length: int`    | Kaufman Adaptive MA.                          | ✅ Core         |
| `supertrend(period, multiplier)` | `int`, `float`   | ATR-based trend-following indicator.          | ✅ Core         |
| `vortex(period)`                 | `period: int`    | Trend strength via VI+ / VI−.                 | ✅ Core         |
| `aroon(period)`                  | `period: int`    | Measures time since highs/lows.               | ✅ Core         |
| `linear_regression(length)`      | `length: int`    | Line of best fit slope.                       | ✅ Core         |
| `donchian_channel(length)`       | `length: int`    | Highest high, lowest low channel.             | ✅ Core         |
| `fractal_ma(length)`             | `length: int`    | Fractal Adaptive MA.                          | ✅ Core         |
| `ma_envelope(length, deviation)` | `int`, `float`   | MA bands with % deviation.                    | ✅ Core         |
| `tillson_t3(length)`             | `length: int`    | Smooth advanced moving average.               | 🟩 Recommended |
| `ichimoku_cloud()`               | *(none)*         | Full Ichimoku components.                     | 🟩 Recommended |
| `parabolic_sar(step, max_step)`  | `float`, `float` | Trend following system with trailing stops.   | 🟩 Recommended |
| `adaptive_moving_average()`      | *(none)*         | Dynamically adjusts MA based on volatility.   | 🟩 Recommended |
| `polarized_fractal_efficiency()` | *(none)*         | Measures trend vs. noise.                     | 🟨 Optional    |
| `trend_strength_indicator()`     | *(none)*         | Flags trend confidence using multiple inputs. | 🟨 Optional    |
| `dynamic_zone_ma()`              | *(none)*         | Adaptive band-based MA using zones.           | 🟨 Optional    |

---

### 🟦 Momentum Indicators

| Function Name                        | Inputs              | Description                                           | Tier           |
| ------------------------------------ | ------------------- | ----------------------------------------------------- | -------------- |
| `rsi(length)`                        | `length: int`       | Relative Strength Index measuring gain/loss momentum. | ✅ Core         |
| `stochastic_rsi(length)`             | `length: int`       | RSI applied to normalized RSI values.                 | ✅ Core         |
| `stochastic_oscillator(k, d)`        | `k: int`, `d: int`  | %K and %D momentum oscillator.                        | ✅ Core         |
| `macd(fast, slow, signal)`           | `int`, `int`, `int` | Moving Average Convergence Divergence.                | ✅ Core         |
| `roc(length)`                        | `length: int`       | Rate of Change = % price change over time.            | ✅ Core         |
| `cci(length)`                        | `length: int`       | Commodity Channel Index.                              | ✅ Core         |
| `trix(length)`                       | `length: int`       | Triple-smoothed EMA momentum.                         | ✅ Core         |
| `ultimate_oscillator()`              | *(none)*            | Combines 3 different timeframes of momentum.          | ✅ Core         |
| `williams_r(length)`                 | `length: int`       | Measures overbought/oversold based on high-low range. | 🟩 Recommended |
| `dmi_adx(length)`                    | `length: int`       | Directional Movement Index and ADX.                   | 🟩 Recommended |
| `momentum(length)`                   | `length: int`       | Raw momentum = current − previous price.              | 🟩 Recommended |
| `elder_impulse()`                    | *(none)*            | Combines EMA trend and MACD momentum for entry/exit.  | 🟩 Recommended |
| `schaff_trend_cycle()`               | *(none)*            | Combines MACD and cycles for fast momentum turns.     | 🟨 Optional    |
| `chande_momentum_oscillator(length)` | `length: int`       | CMO = (SumUp - SumDown) / (SumUp + SumDown)           | 🟨 Optional    |
| `rvi(length)`                        | `length: int`       | Relative Vigor Index: Close–Open vs High–Low.         | 🟨 Optional    |

### 🟧 Volatility Indicators

| Function Name                         | Inputs         | Description                                 | Tier           |
| ------------------------------------- | -------------- | ------------------------------------------- | -------------- |
| `atr(length)`                         | `length: int`  | Average True Range.                         | ✅ Core         |
| `bollinger_bands(length, stddev)`     | `int`, `float` | MA ± standard deviation bands.              | ✅ Core         |
| `keltner_channel(length, multiplier)` | `int`, `float` | EMA + ATR envelope.                         | ✅ Core         |
| `donchian_channel_width(length)`      | `length: int`  | Range width = High − Low.                   | ✅ Core         |
| `true_range()`                        | *(none)*       | Current bar's TR (High-Low vs. PrevClose).  | ✅ Core         |
| `std_dev(length)`                     | `length: int`  | Rolling standard deviation.                 | ✅ Core         |
| `chaikin_volatility(length)`          | `length: int`  | EMA of high-low range vs. longer EMA.       | ✅ Core         |
| `bollinger_percent_b()`               | `length: int`  | Where price is relative to Bollinger Bands. | ✅ Core         |
| `historical_volatility(length)`       | `length: int`  | Std deviation of log returns.               | ✅ Core         |
| `average_range_percent(length)`       | `length: int`  | ATR as % of close price.                    | 🟩 Recommended |
| `normalized_volatility(length)`       | `length: int`  | ATR normalized by price.                    | 🟩 Recommended |
| `range_ratio_indicator(length)`       | `length: int`  | Ratio of candle ranges.                     | 🟩 Recommended |
| `rolling_volatility_spikes()`         | *(none)*       | Flags bars with unusually high volatility.  | 🟩 Recommended |
| `ulcer_index(length)`                 | `length: int`  | Drawdown-based risk metric.                 | 🟨 Optional    |
| `volatility_stop()`                   | `length: int`  | Volatility-based trailing stop logic.       | 🟨 Optional    |
| `atr_trailing_band()`                 | `length: int`  | Custom trailing stop bands via ATR.         | 🟨 Optional    |


---

### 🟨 Volume-Based Indicators

| Function Name                    | Inputs        | Description                               | Tier           |
| -------------------------------- | ------------- | ----------------------------------------- | -------------- |
| `obv()`                          | *(none)*      | On Balance Volume.                        | ✅ Core         |
| `vwap()`                         | *(none)*      | Volume Weighted Average Price.            | ✅ Core         |
| `ad_line()`                      | *(none)*      | Accumulation/Distribution Line.           | ✅ Core         |
| `cmf(length)`                    | `length: int` | Chaikin Money Flow.                       | ✅ Core         |
| `volume_oscillator(short, long)` | `int`, `int`  | Volume short MA − long MA.                | ✅ Core         |
| `force_index(length)`            | `length: int` | Volume × Price Change.                    | ✅ Core         |
| `mfi(length)`                    | `length: int` | Money Flow Index.                         | ✅ Core         |
| `eom(length)`                    | `length: int` | Ease of Movement (Price-Volume flow).     | ✅ Core         |
| `vroc(length)`                   | `length: int` | Volume Rate of Change.                    | ✅ Core         |
| `volume_delta()`                 | *(none)*      | Buy volume − Sell volume estimate.        | 🟩 Recommended |
| `intraday_intensity(length)`     | `length: int` | Closing location + volume weighted.       | 🟩 Recommended |
| `balanced_volume_flow(length)`   | `length: int` | Refined version of OBV with thresholds.   | 🟩 Recommended |
| `volume_climax_spike()`          | *(none)*      | Flags high-volume candles.                | 🟩 Recommended |
| `relative_volume(length)`        | `length: int` | Volume vs. recent average.                | 🟨 Optional    |
| `smoothed_volume_ratio()`        | *(none)*      | Custom EMA of volume divided by baseline. | 🟨 Optional    |
| `vfi(length)`                    | `length: int` | Volume Flow Index.                        | 🟨 Optional    |

## 🟪 **Price Action / Support & Resistance**

| Function Name                           | Inputs                            | Description                                                   | Tier           |
| --------------------------------------- | --------------------------------- | ------------------------------------------------------------- | -------------- |
| `pivot_points(type, period)`            | `type: str`, `period: int`        | Classic, Fibonacci, Camarilla pivot levels.                   | ✅ Core         |
| `price_roc(period)`                     | `period: int`                     | Price Rate of Change (momentum).                              | ✅ Core         |
| `zigzag(threshold)`                     | `threshold: float`                | Detects swings above a % threshold.                           | ✅ Core         |
| `heikin_ashi()`                         | *(none)*                          | Converts raw OHLC to Heikin Ashi.                             | ✅ Core         |
| `renko_boxes(box_size)`                 | `box_size: float`                 | Renko-style brick chart generation.                           | ✅ Core         |
| `detect_engulfing()`                    | *(none)*                          | Bullish/bearish engulfing detection.                          | ✅ Core         |
| `detect_pinbar()`                       | *(none)*                          | Detects pin bars based on wick-body ratio.                    | ✅ Core         |
| `double_top_bottom(period, threshold)`  | `period: int`, `threshold: float` | Detects double top/bottom patterns.                           | ✅ Core         |
| `support_resistance_zones(sensitivity)` | `sensitivity: float`              | Finds key horizontal SR zones.                                | ✅ Core         |
| `candlestick_pattern_count(n, pattern)` | `n: int`, `pattern: str`          | Counts pattern occurrence over last `n` bars.                 | ✅ Core         |
| `three_bar_reversal()`                  | *(none)*                          | Identifies 3-bar reversals (e.g., swing setups).              | 🟩 Recommended |
| `doji_finder()`                         | *(none)*                          | Detects indecision candles (Doji).                            | 🟩 Recommended |
| `price_gap_detector(threshold)`         | `threshold: float`                | Flags gap ups/downs between candles.                          | 🟩 Recommended |
| `range_breakout_flags(window)`          | `window: int`                     | Flags when price breaks from tight range.                     | 🟩 Recommended |
| `range_contraction_detection(window)`   | `window: int`                     | Detects squeeze zones (narrow price ranges).                  | 🟨 Optional    |
| `candle_shadow_ratio()`                 | *(none)*                          | Calculates body-to-wick ratio. Useful for bar classification. | 🟨 Optional    |
| `bar_color_classifier()`                | *(none)*                          | Classifies candles as bullish, bearish, indecisive.           | 🟨 Optional    |



---

## 🟥 **Oscillators & Cycles**

| Function Name                                             | Inputs                                  | Description                                            | Tier           |
| --------------------------------------------------------- | --------------------------------------- | ------------------------------------------------------ | -------------- |
| `fisher_transform(source, length, smooth, smooth_period)` | `List[float]`, `int`, `bool`, `int`     | Normalized Fisher transform with smoothing.            | ✅ Core         |
| `hilbert_transform(series)`                               | `series: List[float]`                   | Computes in-phase and quadrature wave components.      | ✅ Core         |
| `ht_sine(series)`                                         | `series: List[float]`                   | Generates sine and leadsine for cycle phase.           | ✅ Core         |
| `ht_phase(series)`                                        | `series: List[float]`                   | Computes instantaneous phase shift.                    | ✅ Core         |
| `ht_trendline(series)`                                    | `series: List[float]`                   | Smooth filtered trendline from Hilbert Transform.      | ✅ Core         |
| `ht_dominant_cycle(series)`                               | `series: List[float]`                   | Detects dominant price cycle length.                   | ✅ Core         |
| `dpo(series, period)`                                     | `series: List[float]`, `period: int`    | Detrended Price Oscillator.                            | ✅ Core         |
| `ehlers_instantaneous_trendline(series)`                  | `series: List[float]`                   | Ehlers’ smoothed cycle tracking trendline.             | ✅ Core         |
| `phase_accumulation(series)`                              | `series: List[float]`                   | Tracks phase change to estimate cycle count.           | ✅ Core         |
| `inverse_fisher(series)`                                  | `series: List[float]`                   | Applies inverse Fisher transform (0–1 bounded).        | ✅ Core         |
| `roofing_filter(series, period)`                          | `series: List[float]`, `period: int`    | Removes low/high-frequency components (Ehlers filter). | 🟩 Recommended |
| `bandpass_filter(series, low, high)`                      | `series: List[float]`, `float`, `float` | Filters signal to a defined band of frequencies.       | 🟩 Recommended |
| `ehlers_deviation_cycle(series)`                          | `series: List[float]`                   | Experimental variation of DPO with deviation band.     | 🟩 Recommended |
| `cycle_trigger(series, length)`                           | `series: List[float]`, `length: int`    | Generates on/off signals based on phase alignment.     | 🟩 Recommended |
| `dominant_cycle_index(series)`                            | `series: List[float]`                   | Measures confidence in a dominant cycle.               | 🟨 Optional    |
| `adaptive_cycle_filter(series)`                           | `series: List[float]`                   | Auto-tunes filtering to current market rhythm.         | 🟨 Optional    |
| `wavelet_transform_swing(series)`                         | `series: List[float]`                   | Experimental: wavelet-based swing logic (placeholder). | 🟨 Optional    |


---

## ⚫ **Meta-Indicators / Custom Composites**

| Function Name                                 | Inputs               | Description                                         |
| --------------------------------------------- | -------------------- | --------------------------------------------------- |
| `ma_crossover_signal_count(fast_ma, slow_ma)` | `fast_ma`, `slow_ma` | Counts moving average crossovers.                   |
| `indicator_consensus(*indicators)`            | `*indicators`        | Aggregates multiple indicators for consensus score. |
| `momentum_volatility_composite(rsi, atr)`     | `rsi`, `atr`         | Combines RSI and ATR into a composite score.        |
| `trend_strength_score(adx, ma_slope)`         | `adx`, `ma_slope`    | Scores trend strength using ADX and MA slope.       |
| `macd_histogram_angle(macd_hist)`             | `macd_hist`          | Measures angle of change in MACD histogram.         |
| `rsi_divergence_count(rsi, price)`            | `rsi`, `price`       | Detects number of RSI divergences.                  |
| `volume_spike_flag(volume)`                   | `volume`             | Flags volume spikes relative to recent activity.    |
| `multi_timeframe_ema_alignment(*emas)`        | `*emas`              | Checks EMA alignment across timeframes.             |
| `trend_reversal_likelihood(rsi, macd)`        | `rsi`, `macd`        | Estimates chance of trend reversal.                 |
| `consolidation_detector(close)`               | `close`              | Detects price consolidation zones.                  |

---

## 🟤 **Time-Based & Session Indicators**

| Function Name                  | Inputs        | Description                                  | Tier           |
| ------------------------------ | ------------- | -------------------------------------------- | -------------- |
| `time_of_day_normalized()`     | *(none)*      | Converts time to normalized 0–1 range.       | ✅ Core         |
| `session_high_low()`           | *(none)*      | Tracks high/low of current session.          | ✅ Core         |
| `session_overlay(region)`      | `region: str` | Shows active hours (London, NY, Asia).       | ✅ Core         |
| `day_of_week_encoding()`       | *(none)*      | Encodes Monday–Friday numerically.           | ✅ Core         |
| `time_since_high(period)`      | `period: int` | Time since last n-bar high.                  | ✅ Core         |
| `time_since_low(period)`       | `period: int` | Time since last n-bar low.                   | ✅ Core         |
| `intraday_volatility_window()` | *(none)*      | Compares volatility in different hours.      | 🟩 Recommended |
| `session_range_width()`        | *(none)*      | Tracks open-to-close distance in session.    | 🟩 Recommended |
| `hourly_volume_profile()`      | *(none)*      | Aggregates volume per hour.                  | 🟩 Recommended |
| `active_hour_marker()`         | *(none)*      | Marks overlapping sessions or active ranges. | 🟩 Recommended |
| `time_gap_detector()`          | *(none)*      | Flags missing bars or large time gaps.       | 🟨 Optional    |
| `session_stdev_band()`         | *(none)*      | Rolling volatility band per session.         | 🟨 Optional    |
| `session_smooth_trend()`       | *(none)*      | MA only inside market hours.                 | 🟨 Optional    |

---

## 🧠 **Statistical & Derived Indicators**

| Function Name                        | Inputs            | Description                                  |
| ------------------------------------ | ----------------- | -------------------------------------------- |
| `z_score(close, window=20)`          | `close`, `window` | Calculates z-score of price over a window.   |
| `rolling_mean_std(close, window=20)` | `close`, `window` | Returns rolling mean and standard deviation. |
| `skew_kurt(close, window=20)`        | `close`, `window` | Calculates skewness and kurtosis.            |
| `percentile_rank(close, window=20)`  | `close`, `window` | Finds price percentile rank.                 |
| `mad(close, window=20)`              | `close`, `window` | Computes Median Absolute Deviation.          |

---

## 🧪 **Experimental / Nonlinear Indicators**

| Function Name                                      | Inputs                         | Description                                               |
| -------------------------------------------------- | ------------------------------ | --------------------------------------------------------- |
| `fractal_dimension_index(close)`                   | `close`                        | Estimates market roughness.                               |
| `garman_klass_volatility(open, high, low, close)`  | `open`, `high`, `low`, `close` | Volatility estimator using Garman-Klass model.            |
| `kalman_filter_slope(close)`                       | `close`                        | Smooths trend using Kalman filter.                        |
| `hurst_exponent(close)`                            | `close`                        | Calculates Hurst Exponent for fractal behavior.           |
| `shannon_entropy(close, bins=10)`                  | `close`, `bins`                | Measures entropy in price distribution.                   |
| `kld_price(close1, close2)`                        | `close1`, `close2`             | KL divergence between price series.                       |
| `tsf(close, length=14)`                            | `close`, `length`              | Time Series Forecast using linear regression.             |
| `roofing_filter(close)`                            | `close`                        | Ehler’s Roofing Filter to remove high-frequency noise.    |
| `smoothed_heikin_ashi_osc(open, high, low, close)` | `open`, `high`, `low`, `close` | Oscillator using smoothed Heikin Ashi.                    |
| `neural_indicator_score(features)`                 | `features`                     | Outputs a score from trained neural model using features. |

---

## 📚 Tutorials & Help

We recommend using a Jupyter notebook or Python script to:

* Import `pythonpine`
* Connect to MetaTrader5
* Pull the price arrays every minute
* Pass those arrays to your desired custom indicator functions

---

## 📢 Want to Share?

If you're using this library in trading, research, or just for learning — feel free to tag or DM me on Instagram:

📸 **@kushalgarggg**

---

## 🛡️ License

This library is licensed under **Creative Commons BY-NC-SA 4.0**.

* ✅ Free to use for personal and educational purposes
* ❌ Commercial or profit-based use requires permission
* 🔁 Attribution and same-license distribution required

More at: [LICENSE](./LICENSE)

---

## 💡 Contribution

Want to improve or expand this project? Feel free to fork, star 🌟, and submit PRs!

---

```



This project is licensed under the GNU Affero General Public License v3.0.  
See the [LICENSE](LICENSE) file for more information.

For commercial use, please contact the author.
