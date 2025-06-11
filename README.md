
---

````markdown
# 🧠 pythonpine

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

Let me know if you'd like:
- A Markdown badge version (e.g., for GitHub stars, issues, forks)
- Embedded charts/images/gifs of indicators
- To auto-generate the `setup.py` again
- Or help turning it into a GitHub Pages site or portfolio!
```


This project is licensed under the GNU Affero General Public License v3.0.  
See the [LICENSE](LICENSE) file for more information.

For commercial use, please contact the author.
