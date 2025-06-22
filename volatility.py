import numpy as np

# --- 31. ATR (Average True Range) ---
def atr(highList, lowList, closeList, period=14):
    tr_list = [max(highList[i] - lowList[i], abs(highList[i] - closeList[i - 1]), abs(lowList[i] - closeList[i - 1])) if i > 0 else highList[i] - lowList[i] for i in range(len(closeList))]
    atr_vals = [sum(tr_list[:period]) / period] if len(tr_list) >= period else [0.0]
    for i in range(period, len(tr_list)):
        atr_vals.append((atr_vals[-1] * (period - 1) + tr_list[i]) / period)
    return [0.0] * (period - 1) + atr_vals

# --- 32. Bollinger Bands ---
def bollinger_bands(priceList, period=20, stddev=2):
    sma_vals = [sum(priceList[i - period + 1:i + 1]) / period if i >= period - 1 else 0.0 for i in range(len(priceList))]
    upper = []
    lower = []
    for i in range(len(priceList)):
        if i >= period - 1:
            mean = sma_vals[i]
            std = (sum((priceList[j] - mean) ** 2 for j in range(i - period + 1, i + 1)) / period) ** 0.5
            upper.append(mean + stddev * std)
            lower.append(mean - stddev * std)
        else:
            upper.append(0.0)
            lower.append(0.0)
    return upper, lower, sma_vals

# --- 33. Keltner Channel ---
def keltner_channel(highList, lowList, closeList, period=20, multiplier=2):
    typical_price = [(h + l + c) / 3 for h, l, c in zip(highList, lowList, closeList)]
    ema_vals = ema(typical_price, period)
    atr_vals = atr(highList, lowList, closeList, period)
    upper = [e + multiplier * a for e, a in zip(ema_vals, atr_vals)]
    lower = [e - multiplier * a for e, a in zip(ema_vals, atr_vals)]
    return upper, lower, ema_vals

# --- 34. Donchian Channel Width ---
def donchian_channel_width(highList, lowList, period=20):
    width = [max(highList[i - period + 1:i + 1]) - min(lowList[i - period + 1:i + 1]) if i >= period - 1 else 0.0 for i in range(len(highList))]
    return width

# --- 35. True Range ---
def true_range(highList, lowList, closeList):
    return [max(highList[i] - lowList[i], abs(highList[i] - closeList[i - 1]), abs(lowList[i] - closeList[i - 1])) if i > 0 else highList[i] - lowList[i] for i in range(len(closeList))]

# --- 36. Standard Deviation ---
def standard_deviation(priceList, period=20):
    return [((sum((priceList[j] - sum(priceList[i - period + 1:i + 1]) / period) ** 2 for j in range(i - period + 1, i + 1)) / period) ** 0.5) if i >= period - 1 else 0.0 for i in range(len(priceList))]

# --- 37. Chaikin Volatility ---
def chaikin_volatility(highList, lowList, period=10):
    diff = [high - low for high, low in zip(highList, lowList)]
    ema_short = ema(diff, period)
    ema_long = ema(diff, 2 * period)
    return [(s - l) / (l + 1e-10) * 100 for s, l in zip(ema_short, ema_long)]

# --- 38. Bollinger %B ---
def bollinger_percent_b(priceList, period=20, stddev=2):
    upper, lower, middle = bollinger_bands(priceList, period, stddev)
    return [(p - l) / ((u - l) + 1e-10) if u != l else 0.0 for p, u, l in zip(priceList, upper, lower)]

# --- 39. Historical Volatility ---
def historical_volatility(priceList, period=20):
    import math
    returns = [math.log(priceList[i] / priceList[i - 1]) if i > 0 else 0.0 for i in range(len(priceList))]
    std_dev = standard_deviation(returns, period)
    return [sd * math.sqrt(252) for sd in std_dev]  # Annualized volatility

def relative_volatility_index(stddevList, up_stddev, down_stddev, period=14):
    up = [s if s > 0 else 0.0 for s in up_stddev]
    down = [abs(s) if s < 0 else 0.0 for s in down_stddev]
    avg_up = [sum(up[i - period + 1:i + 1]) / period if i >= period - 1 else 0.0 for i in range(len(up))]
    avg_down = [sum(down[i - period + 1:i + 1]) / period if i >= period - 1 else 0.0 for i in range(len(down))]
    rvi = [100 * (u / (u + d + 1e-10)) if (u + d) > 0 else 0.0 for u, d in zip(avg_up, avg_down)]
    return rvi

# --- Normalized ATR ---
def normalized_atr(atrList, priceList):
    return [atr / price if price != 0 else 0.0 for atr, price in zip(atrList, priceList)]

# --- Volatility Ratio ---
def volatility_ratio(atrList, period=14):
    sma_atr = [sum(atrList[i - period + 1:i + 1]) / period if i >= period - 1 else 0.0 for i in range(len(atrList))]
    return [atr / sma if sma != 0 else 0.0 for atr, sma in zip(atrList, sma_atr)]

# --- ATR Percent (ATRP) ---
def atr_percent(atrList, priceList):
    return [(atr / price) * 100 if price != 0 else 0.0 for atr, price in zip(atrList, priceList)]

# --- Ulcer Index ---
def ulcer_index(priceList, period=14):
    index = []
    for i in range(len(priceList)):
        if i < period - 1:
            index.append(0.0)
        else:
            max_close = max(priceList[i - period + 1:i + 1])
            sum_sq = sum([(100 * (priceList[j] - max_close) / max_close) ** 2 for j in range(i - period + 1, i + 1)])
            index.append((sum_sq / period) ** 0.5)
    return index

# --- Mass Index ---
def mass_index(highList, lowList, ema_period=9):
    hl_range = [h - l for h, l in zip(highList, lowList)]
    ema1 = ema(hl_range, ema_period)
    ema2 = ema(ema1, ema_period)
    mass = [e1 / (e2 + 1e-10) for e1, e2 in zip(ema1, ema2)]
    return [sum(mass[i - 25 + 1:i + 1]) if i >= 24 else 0.0 for i in range(len(mass))]

# --- Garman-Klass Volatility ---
def garman_klass_volatility(openList, highList, lowList, closeList, period=10):
    import math
    log_range = [0.5 * (math.log(h / l) ** 2) - (2 * math.log(2) - 1) * (math.log(c / o) ** 2) if o != 0 and l != 0 else 0.0 for h, l, c, o in zip(highList, lowList, closeList, openList)]
    return [math.sqrt(sum(log_range[i - period + 1:i + 1]) / period) if i >= period - 1 else 0.0 for i in range(len(log_range))]

# --- Parkinson Volatility ---
def parkinson_volatility(highList, lowList, period=10):
    import math
    log_hl = [(math.log(h / l)) ** 2 if l != 0 else 0.0 for h, l in zip(highList, lowList)]
    factor = 1 / (4 * math.log(2))
    return [math.sqrt(factor * sum(log_hl[i - period + 1:i + 1]) / period) if i >= period - 1 else 0.0 for i in range(len(log_hl))]

# --- Range-Based Volatility ---
def range_based_volatility(highList, lowList, closeList):
    return [(h - l) / c if c != 0 else 0.0 for h, l, c in zip(highList, lowList, closeList)]

