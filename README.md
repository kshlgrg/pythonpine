
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

| Function                                        | Inputs                    | Description                                     |
| ----------------------------------------------- | ------------------------- | ----------------------------------------------- |
| `connect_to_mt5(login, password, server, path)` | MT5 login credentials     | Connects to MetaTrader5 terminal                |
| `get_ohlcv_arrays(symbol)`                      | symbol (str)              | Returns updated OHLCV arrays for a given symbol |
| `get_price_at_index(price_array, index)`        | array (list), index (int) | Returns price at a specific bar index           |
| `get_close(symbol)`                             | symbol (str)              | Returns the close array                         |
| `get_open(symbol)`                              | symbol (str)              | Returns the open array                          |
| `get_high(symbol)`                              | symbol (str)              | Returns the high array                          |
| `get_low(symbol)`                               | symbol (str)              | Returns the low array                           |
| `get_volume(symbol)`                            | symbol (str)              | Returns the volume array                        |

---

### 📈 Trend Indicators

| Function                                           | Inputs                               | Description                |
| -------------------------------------------------- | ------------------------------------ | -------------------------- |
| `ema(data, period)`                                | data, period                         | Exponential Moving Average |
| `sma(data, period)`                                | data, period                         | Simple Moving Average      |
| `wma(data, period)`                                | data, period                         | Weighted Moving Average    |
| `hma(data, period)`                                | data, period                         | Hull Moving Average        |
| `vwma(data, volume, period)`                       | data, volume, period                 | Volume-Weighted MA         |
| `supertrend(high, low, close, period, multiplier)` | high, low, close, period, multiplier | Supertrend indicator       |
| `ichimoku(high, low)`                              | high, low                            | Ichimoku components        |
| `parabolic_sar(high, low)`                         | high, low                            | Parabolic SAR values       |
| `moving_average_ribbon(close, periods)`            | close, periods (list)                | Multiple MAs on one plot   |
| `trend_strength_indicator(close, period)`          | close, period                        | Measures trend consistency |
| `frama(close, period)`                             | close, period                        | Fractal Adaptive MA        |

---

### 🟦 Momentum Indicators

| Function                                                      | Inputs                    | Description                 |
| ------------------------------------------------------------- | ------------------------- | --------------------------- |
| `rsi(close, period)`                                          | close, period             | Relative Strength Index     |
| `stoch_rsi(close, period)`                                    | close, period             | Stochastic RSI              |
| `stochastic_oscillator(high, low, close, k_period, d_period)` | high, low, close, k, d    | %K and %D Oscillator        |
| `macd(close, fast, slow, signal)`                             | close, fast, slow, signal | MACD and histogram          |
| `roc(close, period)`                                          | close, period             | Rate of Change              |
| `cci(close, typical, period)`                                 | close, typical, period    | Commodity Channel Index     |
| `trix(close, period)`                                         | close, period             | Triple EMA Oscillator       |
| `ultimate_oscillator(high, low, close)`                       | high, low, close          | Momentum from 3 timeframes  |
| `williams_r(high, low, close, period)`                        | high, low, close, period  | Williams %R                 |
| `adx(high, low, close, period)`                               | high, low, close, period  | ADX with +DI and -DI        |
| `momentum(close, period)`                                     | close, period             | Simple momentum calculation |
| `elder_impulse(ema_period)`                                   | ema\_period               | Color-based trend/momentum  |
| `schaff_trend_cycle(close)`                                   | close                     | Smoothed MACD-based cycle   |
| `cmo(close, period)`                                          | close, period             | Chande Momentum Oscillator  |
| `rvi(close, period)`                                          | close, period             | Relative Vigor Index        |

---

### 🟧 Volatility Indicators

| Function                                    | Inputs                   | Description                 |
| ------------------------------------------- | ------------------------ | --------------------------- |
| `atr(high, low, close, period)`             | high, low, close, period | Average True Range          |
| `bollinger_bands(close, period, devs)`      | close, period, devs      | Bollinger Bands             |
| `keltner_channel(high, low, close, period)` | high, low, close, period | ATR-based envelope          |
| `donchian_channel(high, low, period)`       | high, low, period        | Channel of extremes         |
| `true_range(high, low, close)`              | high, low, close         | Daily true range            |
| `std_dev(close, period)`                    | close, period            | Standard deviation of price |
| `chaikin_volatility(high, low)`             | high, low                | Chaikin's volatility method |
| `bollinger_percent_b(close, period)`        | close, period            | %B inside Bollinger Bands   |
| `historical_volatility(close, period)`      | close, period            | StdDev log returns          |

---

### 🟨 Volume-Based Indicators

| Function                                       | Inputs                           | Description                    |
| ---------------------------------------------- | -------------------------------- | ------------------------------ |
| `obv(close, volume)`                           | close, volume                    | On Balance Volume              |
| `vwap(high, low, close, volume)`               | high, low, close, volume         | Volume-Weighted Avg Price      |
| `adl(high, low, close, volume)`                | high, low, close, volume         | Accumulation/Distribution Line |
| `cmf(high, low, close, volume, period)`        | high, low, close, volume, period | Chaikin Money Flow             |
| `volume_oscillator(volume, short, long)`       | volume, short, long              | Volume-based oscillator        |
| `force_index(close, volume)`                   | close, volume                    | Force Index                    |
| `mfi(high, low, close, volume, period)`        | high, low, close, volume, period | Money Flow Index               |
| `ease_of_movement(high, low, volume)`          | high, low, volume                | EMV Oscillator                 |
| `vroc(volume, period)`                         | volume, period                   | Volume Rate of Change          |
| `volume_delta(close, volume)`                  | close, volume                    | Buy-sell volume imbalance      |
| `intraday_intensity(close, high, low, volume)` | close, high, low, volume         | Intraday pressure indicator    |

---

## 🟪 **Price Action / Support & Resistance**

| Function Name                                                         | Inputs                                            | Description                                                                |
| --------------------------------------------------------------------- | ------------------------------------------------- | -------------------------------------------------------------------------- |
| `pivot_points(close, method='classic')`                               | `close`, `method`                                 | Calculates pivot points based on Classic, Fibonacci, or Camarilla methods. |
| `price_roc(close, period=14)`                                         | `close`, `period`                                 | Measures rate of change of price.                                          |
| `zigzag(close, deviation=5)`                                          | `close`, `deviation`                              | ZigZag pattern based on price deviation percentage.                        |
| `heikin_ashi(open, high, low, close)`                                 | `open`, `high`, `low`, `close`                    | Converts OHLC data to Heikin Ashi candles.                                 |
| `renko_boxes(close, brick_size)`                                      | `close`, `brick_size`                             | Renko box generation based on fixed brick size.                            |
| `engulfing_pattern(open, close)`                                      | `open`, `close`                                   | Detects bullish and bearish engulfing candlestick patterns.                |
| `pin_bar(open, high, low, close)`                                     | `open`, `high`, `low`, `close`                    | Detects pin bars based on wick sizes.                                      |
| `double_top_bottom(high, low, close)`                                 | `high`, `low`, `close`                            | Identifies potential double top/bottom patterns.                           |
| `support_resistance_zones(high, low, sensitivity=5)`                  | `high`, `low`, `sensitivity`                      | Detects SR zones based on swing highs and lows.                            |
| `candlestick_pattern_count(open, high, low, close, pattern, bars=50)` | `open`, `high`, `low`, `close`, `pattern`, `bars` | Counts the occurrences of a given candlestick pattern over recent bars.    |

---

## 🟥 **Oscillators & Cycles**

| Function Name                            | Inputs                           | Description                                                   |
| ---------------------------------------- | -------------------------------- | ------------------------------------------------------------- |
| `fisher_transform(close, length=10)`     | `close`, `length`                | Applies the Fisher Transform to normalize price oscillations. |
| `hilbert_transform(close, mode='cycle')` | `close`, `mode`                  | Computes Hilbert Transform in cycle or trend mode.            |
| `ehler_instant_trendline(close)`         | `close`                          | Ehler’s Instantaneous Trendline filter.                       |
| `dpo(close, length=20)`                  | `close`, `length`                | Detrended Price Oscillator to remove long-term trends.        |
| `laguerre_rsi(close, gamma=0.5)`         | `close`, `gamma`                 | Smoothed RSI using Laguerre filter.                           |
| `qstick(open, close, length=10)`         | `open`, `close`, `length`        | Measures average candlestick body size.                       |
| `smi(close, high, low, length=14)`       | `close`, `high`, `low`, `length` | Stochastic Momentum Index calculation.                        |
| `adaptive_cycle_divergence(close)`       | `close`                          | Detects adaptive cycle divergence.                            |
| `phase_accumulation(close)`              | `close`                          | Calculates phase accumulation cycle.                          |
| `inverse_fisher(rsi_values)`             | `rsi_values`                     | Applies inverse Fisher transform to RSI.                      |

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

| Function Name                                               | Inputs                                 | Description                                  |
| ----------------------------------------------------------- | -------------------------------------- | -------------------------------------------- |
| `time_of_day_normalized()`                                  | None                                   | Returns normalized UTC time of day.          |
| `session_high_low(high, low, timestamps, session='London')` | `high`, `low`, `timestamps`, `session` | Tracks high/low of specific sessions.        |
| `session_overlay(timestamps)`                               | `timestamps`                           | Creates a visual overlay of market sessions. |
| `day_of_week_encoding(timestamps)`                          | `timestamps`                           | Encodes day of week as numeric values.       |
| `time_since_last_high_low(close)`                           | `close`                                | Measures time since last high/low occurred.  |

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
