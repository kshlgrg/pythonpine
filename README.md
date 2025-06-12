
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
git clone https://github.com/kshgrg/pythonpine.git
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
