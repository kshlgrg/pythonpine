import numpy as np

# --- Trend Indicators ---

# --- 2. SMA (Simple Moving Average) ---
def sma(priceList, length=14):
    return [sum(priceList[i - length + 1:i + 1]) / length if i >= length - 1 else 0.0 for i in range(len(priceList))]

# --- 3. DEMA (Double EMA) ---
def dema(priceList, length=14):
    ema1 = ema(priceList, length)
    ema2 = ema(ema1, length)
    return [2 * e1 - e2 for e1, e2 in zip(ema1, ema2)]

# --- 4. TEMA (Triple EMA) ---
def tema(priceList, length=14):
    ema1 = ema(priceList, length)
    ema2 = ema(ema1, length)
    ema3 = ema(ema2, length)
    return [3 * e1 - 3 * e2 + e3 for e1, e2, e3 in zip(ema1, ema2, ema3)]

# --- 5. WMA (Weighted Moving Average) ---
def wma(priceList, length=14):
    result = []
    for i in range(len(priceList)):
        if i < length - 1:
            result.append(0.0)
        else:
            weights = list(range(1, length + 1))
            values = priceList[i - length + 1:i + 1]
            weighted_sum = sum(w * p for w, p in zip(weights, values))
            result.append(weighted_sum / sum(weights))
    return result

# --- 6. HMA (Hull Moving Average) ---
def hma(priceList, length=14):
    half = int(length / 2)
    sqrt_len = int(length ** 0.5)
    wma_half = wma(priceList, half)
    wma_full = wma(priceList, length)
    diff = [2 * wh - wf for wh, wf in zip(wma_half, wma_full)]
    return wma(diff, sqrt_len)

# --- 7. VWMA (Volume Weighted MA) ---
def vwma(priceList, volumeList, length=14):
    result = []
    for i in range(len(priceList)):
        if i < length - 1:
            result.append(0.0)
        else:
            sum_pv = sum(priceList[i - length + 1 + j] * volumeList[i - length + 1 + j] for j in range(length))
            sum_v = sum(volumeList[i - length + 1 + j] for j in range(length)) + 1e-10
            result.append(sum_pv / sum_v)
    return result

# --- 8. KAMA (Kaufman Adaptive MA) ---
def kama(priceList, length=10, fastend=2, slowend=30):
    kama_vals = [priceList[0]]
    fast_alpha = 2 / (fastend + 1)
    slow_alpha = 2 / (slowend + 1)
    for i in range(1, len(priceList)):
        if i < length:
            kama_vals.append(priceList[i])
        else:
            change = abs(priceList[i] - priceList[i - length])
            volatility = sum(abs(priceList[j] - priceList[j - 1]) for j in range(i - length + 1, i + 1)) + 1e-10
            er = change / volatility
            sc = (er * (fast_alpha - slow_alpha) + slow_alpha) ** 2
            kama_vals.append(kama_vals[-1] + sc * (priceList[i] - kama_vals[-1]))
    return kama_vals

# --- 9. SuperTrend ---
def supertrend(highList, lowList, closeList, period=10, multiplier=3.0):
    atr = [0.0]
    for i in range(1, len(closeList)):
        tr = max(highList[i] - lowList[i], abs(highList[i] - closeList[i - 1]), abs(lowList[i] - closeList[i - 1]))
        atr.append((atr[-1] * (period - 1) + tr) / period)

    upper_band = [(highList[i] + lowList[i]) / 2 + multiplier * atr[i] for i in range(len(atr))]
    lower_band = [(highList[i] + lowList[i]) / 2 - multiplier * atr[i] for i in range(len(atr))]

    trend = [1] * len(closeList)
    for i in range(1, len(closeList)):
        if closeList[i] > upper_band[i - 1]:
            trend[i] = 1
        elif closeList[i] < lower_band[i - 1]:
            trend[i] = -1
        else:
            trend[i] = trend[i - 1]
    return trend

# --- 10. Vortex Indicator ---
def vortex_indicator(highList, lowList, closeList, length=14):
    plus_vm = [0.0]
    minus_vm = [0.0]
    tr = [0.0]
    for i in range(1, len(closeList)):
        plus_vm.append(abs(highList[i] - lowList[i - 1]))
        minus_vm.append(abs(lowList[i] - highList[i - 1]))
        tr.append(max(highList[i] - lowList[i], abs(highList[i] - closeList[i - 1]), abs(lowList[i] - closeList[i - 1])))

    vi_plus = [sum(plus_vm[max(0, i - length + 1):i + 1]) / sum(tr[max(0, i - length + 1):i + 1] + [1e-10]) for i in range(len(closeList))]
    vi_minus = [sum(minus_vm[max(0, i - length + 1):i + 1]) / sum(tr[max(0, i - length + 1):i + 1] + [1e-10]) for i in range(len(closeList))]
    return vi_plus, vi_minus

# --- 11. Aroon Up/Down ---
def aroon(highList, lowList, length=14):
    aroon_up = []
    aroon_down = []
    for i in range(len(highList)):
        if i < length:
            aroon_up.append(0.0)
            aroon_down.append(0.0)
        else:
            max_idx = highList[i - length + 1:i + 1].index(max(highList[i - length + 1:i + 1]))
            min_idx = lowList[i - length + 1:i + 1].index(min(lowList[i - length + 1:i + 1]))
            aroon_up.append(((length - (length - 1 - max_idx)) / length) * 100)
            aroon_down.append(((length - (length - 1 - min_idx)) / length) * 100)
    return aroon_up, aroon_down

# --- 12. Linear Regression Line ---
def linear_regression(priceList, length=14):
    result = []
    for i in range(len(priceList)):
        if i < length - 1:
            result.append(0.0)
        else:
            x = list(range(length))
            y = priceList[i - length + 1:i + 1]
            avg_x = sum(x) / length
            avg_y = sum(y) / length
            num = sum((x[j] - avg_x) * (y[j] - avg_y) for j in range(length))
            den = sum((x[j] - avg_x) ** 2 for j in range(length)) + 1e-10
            slope = num / den
            intercept = avg_y - slope * avg_x
            result.append(slope * (length - 1) + intercept)
    return result

# --- 13. Donchian Channel ---
def donchian_channel(highList, lowList, length=20):
    upper = [0.0 if i < length else max(highList[i - length + 1:i + 1]) for i in range(len(highList))]
    lower = [0.0 if i < length else min(lowList[i - length + 1:i + 1]) for i in range(len(lowList))]
    mid = [(u + l) / 2 for u, l in zip(upper, lower)]
    return upper, lower, mid

# --- 14. Fractal Adaptive Moving Average ---
def fama(priceList, length=10):
    return kama(priceList, length)

# --- 15. Moving Average Envelope ---
def moving_average_envelope(priceList, length=20, percent=2.5):
    ma = sma(priceList, length)
    upper = [m * (1 + percent / 100) for m in ma]
    lower = [m * (1 - percent / 100) for m in ma]
    return upper, lower, ma

# --- Ichimoku Cloud ---
def ichimoku_cloud(highList, lowList, closeList, tenkan=9, kijun=26, senkou=52):
    conversion = [(max(highList[i - tenkan + 1:i + 1]) + min(lowList[i - tenkan + 1:i + 1])) / 2 if i >= tenkan - 1 else 0.0 for i in range(len(closeList))]
    base = [(max(highList[i - kijun + 1:i + 1]) + min(lowList[i - kijun + 1:i + 1])) / 2 if i >= kijun - 1 else 0.0 for i in range(len(closeList))]
    span_a = [(conversion[i] + base[i]) / 2 if i >= kijun - 1 else 0.0 for i in range(len(closeList))]
    span_b = [(max(highList[i - senkou + 1:i + 1]) + min(lowList[i - senkou + 1:i + 1])) / 2 if i >= senkou - 1 else 0.0 for i in range(len(closeList))]
    lagging = [closeList[i - kijun] if i >= kijun else 0.0 for i in range(len(closeList))]
    return conversion, base, span_a, span_b, lagging


# --- Parabolic SAR ---
def parabolic_sar(highList, lowList, step=0.02, max_af=0.2):
    sar = [lowList[0]]
    ep = highList[0]
    af = step
    trend = 1  # 1 for uptrend, -1 for downtrend

    for i in range(1, len(highList)):
        prev_sar = sar[-1]
        if trend == 1:
            sar_new = prev_sar + af * (ep - prev_sar)
            sar_new = min(sar_new, lowList[i - 1], lowList[i])
            if lowList[i] < sar_new:
                trend = -1
                sar_new = ep
                ep = lowList[i]
                af = step
            else:
                if highList[i] > ep:
                    ep = highList[i]
                    af = min(af + step, max_af)
        else:
            sar_new = prev_sar + af * (ep - prev_sar)
            sar_new = max(sar_new, highList[i - 1], highList[i])
            if highList[i] > sar_new:
                trend = 1
                sar_new = ep
                ep = highList[i]
                af = step
            else:
                if lowList[i] < ep:
                    ep = lowList[i]
                    af = min(af + step, max_af)
        sar.append(sar_new)
    return sar


# --- Guppy MMA (Guppy Multiple Moving Averages) ---
def guppy_mma(priceList):
    short_lengths = [3, 5, 8, 10, 12, 15]
    long_lengths = [30, 35, 40, 45, 50, 60]
    short_emas = [ema(priceList, length) for length in short_lengths]
    long_emas = [ema(priceList, length) for length in long_lengths]
    return short_emas, long_emas


# --- Trend Angle from MA Slope ---
def ma_slope_angle(maList):
    import math
    angles = [0.0]
    for i in range(1, len(maList)):
        delta = maList[i] - maList[i - 1]
        angle = math.degrees(math.atan(delta))
        angles.append(round(angle, 4))
    return angles


# --- Zero-Lag EMA ---
def zero_lag_ema(priceList, length=14):
    ema1 = ema(priceList, length)
    ema2 = ema(ema1, length)
    return [2 * e1 - e2 for e1, e2 in zip(ema1, ema2)]


# --- Median Price ---
def median_price(highList, lowList):
    return [(h + l) / 2 for h, l in zip(highList, lowList)]


# --- Typical Price ---
def typical_price(highList, lowList, closeList):
    return [(h + l + c) / 3 for h, l, c in zip(highList, lowList, closeList)]
