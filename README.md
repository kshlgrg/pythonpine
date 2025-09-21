🧠 PYTHONPINE
**
An ultra-powerful Python library to compute over 100+ TradingView-style technical indicators
** using real-time OHLCV data from MetaTrader5.
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
```
bash
pip install MetaTrader5 numpy pandas scipy scikit-learn
```
### 2. Clone this repo
```
bash
git clone https://github.com/kshgrg/pythonpine
cd pythonpine
```
### 3. Add the folder to your project or install locally
```
bash
pip install -e .
```
---
## 🛠️ How to Use
### ✅ Step 1: Connect to MetaTrader5
```
python
from pythonpine import *
connect_to_mt5(
    login=123456,
    password="yourpassword",
    server="yourserver",
    path="C:\\Path\\To\\Terminal64.exe"
)
```
### ✅ Step 2: Get OHLCV arrays
```
python
open, high, low, close, volume = get_ohlcv_arrays("EURUSD")
```
### ✅ Step 3: Run background price updater (Optional for real-time trading)
Paste this in your main code to auto-update price arrays every minute:
```
python
import time
while True:
    open, high, low, close, volume = get_ohlcv_arrays("EURUSD")
    time.sleep(60)  # Updates every 60 seconds
```
---
## 📊 Example Usage (Indicator Calculations)
```
python
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
## 🟥 Oscillators & Cycles (updated)
This section lists all oscillators and cycle-related indicators available in pythonpine as of the latest update to cycles.py. Defaults and signatures match the code. Where applicable, many algorithms are inspired by the work of John F. Ehlers.

### Core oscillators and Hilbert-based tools
- fisher_transform(source_type: str = "close", length: int = 10, start: int = 0, end: Optional[int] = None, smooth: bool = False, smooth_factor: float = 0.33, close_list: NumberArray) -> List[float]
  - Fisher transform of the selected source, optionally smoothed; maps values toward a Gaussianized range for sharper turning points.
- hilbert_transform(series: NumberArray) -> List[float]
  - Returns a transformed series intended for in-phase/quadrature style processing downstream.
- ht_sine(series: NumberArray) -> Tuple[List[float], List[float]]
  - Produces sine and leadsine sequences used to visualize and trade by cycle phase.
- ht_phase(series: NumberArray) -> List[float]
  - Estimates instantaneous phase (radians) from the signal.
- ht_trendline(series: NumberArray, alpha: float = 0.1) -> List[float]
  - Smooth trendline derived from HT processing; alpha controls responsiveness.
- ht_dominant_cycle(series: NumberArray) -> List[float]
  - Estimates dominant cycle length (period) bar-by-bar.
- ht_itrend(series: NumberArray) -> List[float]
  - Instantaneous trendline variant for cycle-aware smoothing.

### Cycles and related oscillators defined in cycles.py
- detrended_price_oscillator(price_list: NumberArray, length: int = 14) -> List[float]
  - Detrends price by removing an average over a window to highlight cyclical swings.
- laguerre_rsi(price_list: NumberArray, gamma: float = 0.5) -> List[float]
  - RSI computed via a 4-stage Laguerre filter; reacts faster with low lag; 0 < gamma < 1.
- qstick(open_list: NumberArray, close_list: NumberArray, length: int = 14) -> List[float]
  - Average candle body over N periods; positive when closes tend to exceed opens.
- stochastic_momentum_index(close_list: NumberArray, high_list: NumberArray, low_list: NumberArray, length: int = 14, smooth_k: int = 3, smooth_d: int = 3) -> Tuple[List[float], List[float]]
  - SMI with double smoothing. Returns K and D lines in [-100, 100].
- adaptive_cycle_divergence(price_list: NumberArray) -> List[float]
  - Divergence of price from its dominant-cycle mean; uses ht_dominant_cycle length and EMA-like smoothing.
- phase_accumulation_cycle(price_list: NumberArray) -> List[float]
  - Simplified phase-accumulation dominant cycle estimator; outputs estimated period from phase changes.
- inverse_fisher_transform(series: NumberArray) -> List[float]
  - Inverse Fisher transform; input clipped to [-1, 1] and mapped back to (-1,1) via tanh-like formula.
- mama_fama(price_list: NumberArray, fast_limit: float = 0.5, slow_limit: float = 0.05) -> Tuple[List[float], List[float]]
  - Ehlers’ adaptive moving averages (MAMA/FAMA) with per-bar variable alpha; returns two adaptive MAs.
- super_smoother(price_list: NumberArray, period: int) -> List[float]
  - Two-pole Super Smoother (Ehlers) for low-lag smoothing at a chosen period.
- roofing_filter(price_list: NumberArray) -> List[float]
  - High-pass followed by smoothing (roofing-like filter) to confine spectrum to tradable cycles.
- center_of_gravity(price_list: NumberArray, length: int = 10) -> List[float]
  - Center of Gravity oscillator; a weighted average ratio emphasizing recent prices.
- bandpass_filter(price_list: NumberArray, period: int, bandwidth: float = 0.3) -> List[float]
  - Simple recursive band-pass; passband centered on period with fractional bandwidth (0<bandwidth<1).
- dc_based_rsi(price_list: NumberArray, cycle_list: NumberArray) -> List[float]
  - RSI computed over a bar-by-bar variable lookback derived from a provided dominant cycle series.
- cyber_cycle(price_list: NumberArray, alpha: float = 0.07) -> List[float]
  - Ehlers’ Cyber Cycle filter emphasizing cyclic content with low lag; 0<alpha<1.

Notes for maintainers:
- All functions accept Python lists or NumPy arrays (NumberArray); internal helpers _as_np, _validate_length, and _validate_same_length handle coercion and checks.
- Defaults and edge-case behavior: Many functions output 0.0 until sufficient history exists; rounding is controlled via ROUND_DECIMALS within the module.
- Several functions are composed (e.g., adaptive_cycle_divergence uses ht_dominant_cycle); keep cross-module imports stable when reorganizing.
- Where specified, parameters must satisfy guards in code (e.g., 0<alpha<1; 0<slow_limit<=fast_limit<=1; bandwidth in (0,1)).

### Quick examples (cycles)
```
python
import numpy as np
from pythonpine import fisher_transform, ht_dominant_cycle, bandpass_filter

# Example price series
price = np.linspace(0, 4*np.pi, 200)
price = 100 + 2*np.sin(price) + 0.5*np.cos(3*price)

# 1) Fisher Transform of close
fisher = fisher_transform(
    source_type="close", length=10, start=0, end=len(price),
    smooth=True, smooth_factor=0.33, close_list=price
)

# 2) Another cycle detector: dominant cycle estimate
dc = ht_dominant_cycle(price)

# Optional: use band-pass filter around a target period
bp = bandpass_filter(price, period=30, bandwidth=0.3)
```
Credit: Several algorithms and filter structures are derived from or inspired by John F. Ehlers’ work on digital signal processing for markets.

---
## 🧪 Full Indicator List
Expand to see all indicators:<details>
<summary>Click to expand</summary>

### 🟩 Trend Indicators
EMA, SMA, DEMA, TEMA, WMA, HMA, VWMA, KAMA, SuperTrend, Vortex, Aroon, Linear Regression, Donchian, FAMA, MA Envelope

### 🟦 Momentum Indicators
RSI, Stoch RSI, Stochastic, MACD, ROC, CCI, TRIX, Ultimate Osc, Williams %R, DMI/ADX, Elder Impulse, Schaff, CMO, RVI

### 🟧 Volatility Indicators
ATR, Bollinger Bands, Keltner Channel, Donchian Width, True Range, Std Dev, Chaikin Vol, Boll %B, Hist Volatility
</details>
---
## 📚 Tutorials & Help
We recommend using a Jupyter notebook or Python script to:
- Import `pythonpine`
- Connect to MetaTrader5
- Pull the price arrays every minute
- Pass those arrays to your desired custom indicator functions
---
## 📢 Want to Share?
If you're using this library in trading, research, or just for learning — feel free to tag or DM me on Instagram:
📸 **@kushalgarggg**
---
## 🛡️ License
This library is licensed under **Creative Commons BY-NC-SA 4.0**.
- ✅ Free to use for personal and educational purposes
- ❌ Commercial or profit-based use requires permission
- 🔁 Attribution and same-license distribution required
More at: [LICENSE](./LICENSE)
---
## 💡 Contribution
Want to improve or expand this project? Feel free to fork, star 🌟, and submit PRs!
---
```
This project is licensed under the GNU Affero General Public License v3.0.  
See the [LICENSE](LICENSE) file for more information.
For commercial use, please contact the author.
