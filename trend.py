import numpy as np

# 1. Exponential Moving Average (EMA)
def ema(source, period):
    ema_values = []
    multiplier = 2 / (period + 1)
    ema_values.append(source[0])
    for i in range(1, len(source)):
        ema_values.append((source[i] - ema_values[-1]) * multiplier + ema_values[-1])
    return [float(x) for x in ema_values]

# 2. Simple Moving Average (SMA)
def sma(source, period):
    sma_values = np.convolve(source, np.ones(period)/period, mode='valid')
    return [float(x) for x in sma_values]

# 3. Double EMA (DEMA)
def dema(source, period):
    ema1 = ema(source, period)
    ema2 = ema(ema1, period)
    dema_values = [2 * e1 - e2 for e1, e2 in zip(ema1[-len(ema2):], ema2)]
    return [float(x) for x in dema_values]

# 4. Triple EMA (TEMA)
def tema(source, period):
    ema1 = ema(source, period)
    ema2 = ema(ema1, period)
    ema3 = ema(ema2, period)
    tema_values = [3 * e1 - 3 * e2 + e3 for e1, e2, e3 in zip(ema1[-len(ema3):], ema2[-len(ema3):], ema3)]
    return [float(x) for x in tema_values]

# 5. Weighted MA (WMA)
def wma(source, period):
    weights = np.arange(1, period + 1)
    wma_values = []
    for i in range(len(source) - period + 1):
        w = np.dot(source[i:i+period], weights) / weights.sum()
        wma_values.append(float(w))
    return wma_values

# 6. Hull MA (HMA)
def hma(source, period):
    half = period // 2
    sqrt_len = int(np.sqrt(period))
    wma_half = wma(source, half)
    wma_full = wma(source, period)
    diff = [2 * h - f for h, f in zip(wma_half[-len(wma_full):], wma_full)]
    return wma(diff, sqrt_len)

# 7. VWMA (Volume Weighted MA)
def vwma(close, volume, period):
    vwma_values = []
    for i in range(len(close) - period + 1):
        c = close[i:i+period]
        v = volume[i:i+period]
        vwap = np.sum(c * v) / np.sum(v)
        vwma_values.append(float(vwap))
    return vwma_values

# 8. KAMA (Kaufman’s Adaptive MA)
def kama(source, er_period=10, fast=2, slow=30):
    fast_sc = 2 / (fast + 1)
    slow_sc = 2 / (slow + 1)
    kama_vals = [source[0]]
    for i in range(1, len(source)):
        change = abs(source[i] - source[max(i - er_period, 0)])
        volatility = np.sum(np.abs(np.diff(source[max(0, i - er_period):i + 1])))
        er = change / (volatility + 1e-10)
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        kama_vals.append(kama_vals[-1] + sc * (source[i] - kama_vals[-1]))
    return [float(x) for x in kama_vals]

# 9. SuperTrend
def supertrend(high, low, close, atr_period=10, multiplier=3):
    def atr(h, l, c, p):
        tr = [max(h[i] - l[i], abs(h[i] - c[i-1]), abs(l[i] - c[i-1])) for i in range(1, len(h))]
        tr.insert(0, h[0] - l[0])
        atr = []
        atr.append(np.mean(tr[:p]))
        for i in range(p, len(tr)):
            atr.append((atr[-1] * (p - 1) + tr[i]) / p)
        return atr
    atr_vals = atr(high, low, close, atr_period)
    hl2 = [(h + l) / 2 for h, l in zip(high, low)]
    upper_band = [hl2[i] + multiplier * atr_vals[i] for i in range(len(atr_vals))]
    lower_band = [hl2[i] - multiplier * atr_vals[i] for i in range(len(atr_vals))]
    return [float(x) for x in upper_band], [float(x) for x in lower_band]

# 10. Vortex Indicator
def vortex(high, low, close, period=14):
    vi_plus, vi_minus = [], []
    for i in range(1, len(close)):
        tr = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        vp = abs(high[i] - low[i - 1])
        vm = abs(low[i] - high[i - 1])
        vi_plus.append(vp)
        vi_minus.append(vm)
    vi_plus_sums = [np.sum(vi_plus[i:i+period]) for i in range(len(vi_plus) - period + 1)]
    vi_minus_sums = [np.sum(vi_minus[i:i+period]) for i in range(len(vi_minus) - period + 1)]
    tr_sums = [np.sum([max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1])) for j in range(i, i+period)]) for i in range(len(close) - period)]
    vi_p = [vp / (tr + 1e-10) for vp, tr in zip(vi_plus_sums, tr_sums)]
    vi_m = [vm / (tr + 1e-10) for vm, tr in zip(vi_minus_sums, tr_sums)]
    return [float(x) for x in vi_p], [float(x) for x in vi_m]

# 11. Aroon Up/Down
def aroon(high, low, period=14):
    aroon_up, aroon_down = [], []
    for i in range(period, len(high)):
        highest = np.argmax(high[i - period:i])  # 0-based
        lowest = np.argmax(low[i - period:i])
        aroon_up.append(100 * (period - highest) / period)
        aroon_down.append(100 * (period - lowest) / period)
    return [float(x) for x in aroon_up], [float(x) for x in aroon_down]

# 12. Linear Regression Line (slope)
def linear_regression_slope(source, period=14):
    x = np.arange(period)
    slopes = []
    for i in range(len(source) - period + 1):
        y = source[i:i+period]
        slope, _ = np.polyfit(x, y, 1)
        slopes.append(float(slope))
    return slopes

# 13. Donchian Channel
def donchian_channel(high, low, period=20):
    upper = [max(high[i - period:i]) for i in range(period, len(high))]
    lower = [min(low[i - period:i]) for i in range(period, len(low))]
    return [float(x) for x in upper], [float(x) for x in lower]

# 14. Fractal Adaptive MA (approximation using mid-fractals)
def fractal_adaptive_ma(source, period=14):
    def fractal_index(prices):
        return np.std(prices) / (np.mean(prices) + 1e-10)
    fama = []
    for i in range(len(source) - period + 1):
        f = fractal_index(source[i:i+period])
        weighted = np.mean(source[i:i+period]) * (1 / (1 + f))
        fama.append(float(weighted))
    return fama

# 15. Moving Average Envelope
def ma_envelope(source, period=20, percent=2):
    ma = sma(source, period)
    upper = [x * (1 + percent / 100) for x in ma]
    lower = [x * (1 - percent / 100) for x in ma]
    return [float(x) for x in upper], [float(x) for x in lower]
