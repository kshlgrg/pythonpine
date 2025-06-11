import numpy as np

# 31. Average True Range (ATR)
def atr(high, low, close, period=14):
    tr = [max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1])) for i in range(1, len(close))]
    atr_vals = []
    atr_vals.append(np.mean(tr[:period]))
    for i in range(period, len(tr)):
        val = (atr_vals[-1] * (period - 1) + tr[i]) / period
        atr_vals.append(val)
    return [float(x) for x in atr_vals]

# 32. Bollinger Bands
def bollinger_bands(source, period=20, std_dev=2):
    mid = [np.mean(source[i - period:i]) for i in range(period, len(source))]
    stds = [np.std(source[i - period:i]) for i in range(period, len(source))]
    upper = [m + std_dev * s for m, s in zip(mid, stds)]
    lower = [m - std_dev * s for m, s in zip(mid, stds)]
    return (
        [float(x) for x in upper],
        [float(x) for x in mid],
        [float(x) for x in lower]
    )

# 33. Keltner Channel
def keltner_channel(high, low, close, period=20, multiplier=2):
    typical = [(h + l + c) / 3 for h, l, c in zip(high, low, close)]
    ema_typical = ema(typical, period)
    atr_vals = atr(high, low, close, period)
    min_len = min(len(ema_typical), len(atr_vals))
    middle = ema_typical[-min_len:]
    atr_trimmed = atr_vals[-min_len:]
    upper = [m + multiplier * a for m, a in zip(middle, atr_trimmed)]
    lower = [m - multiplier * a for m, a in zip(middle, atr_trimmed)]
    return (
        [float(x) for x in upper],
        [float(x) for x in middle],
        [float(x) for x in lower]
    )

# 34. Donchian Channel Width
def donchian_channel_width(high, low, period=20):
    return [float(max(high[i - period:i]) - min(low[i - period:i])) for i in range(period, len(high))]

# 35. True Range (TR)
def true_range(high, low, close):
    return [float(max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))) for i in range(1, len(close))]

# 36. Standard Deviation
def std_dev(source, period=20):
    return [float(np.std(source[i - period:i])) for i in range(period, len(source))]

# 37. Chaikin Volatility
def chaikin_volatility(high, low, period=10):
    hl_diff = [h - l for h, l in zip(high, low)]
    ema1 = ema(hl_diff, period)
    vol = [(ema1[i] - ema1[i - period]) / (ema1[i - period] + 1e-10) * 100 for i in range(period, len(ema1))]
    return [float(x) for x in vol]

# 38. Bollinger %B
def bollinger_percent_b(source, period=20, std_dev=2):
    upper, mid, lower = bollinger_bands(source, period, std_dev)
    percent_b = [(source[i] - l) / (u - l + 1e-10) * 100 for i, (u, l) in enumerate(zip(upper, lower), start=period)]
    return [float(x) for x in percent_b]

# 39. Historical Volatility (standard deviation of log returns)
def historical_volatility(close, period=14):
    log_returns = [np.log(close[i] / close[i - 1]) for i in range(1, len(close))]
    return [float(np.std(log_returns[i - period:i]) * np.sqrt(252)) for i in range(period, len(log_returns))]

# -- Reusable EMA --
def ema(source, period):
    ema_vals = [source[0]]
    multiplier = 2 / (period + 1)
    for i in range(1, len(source)):
        ema_vals.append((source[i] - ema_vals[-1]) * multiplier + ema_vals[-1])
    return [float(x) for x in ema_vals]
