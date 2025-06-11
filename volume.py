import numpy as np

# 40. On Balance Volume (OBV)
def obv(close, volume):
    obv_vals = [0]
    for i in range(1, len(close)):
        if close[i] > close[i - 1]:
            obv_vals.append(obv_vals[-1] + volume[i])
        elif close[i] < close[i - 1]:
            obv_vals.append(obv_vals[-1] - volume[i])
        else:
            obv_vals.append(obv_vals[-1])
    return [float(x) for x in obv_vals]

# 41. VWAP (Volume Weighted Average Price)
def vwap(high, low, close, volume):
    typical_price = [(h + l + c) / 3 for h, l, c in zip(high, low, close)]
    cum_vol_price = np.cumsum([tp * v for tp, v in zip(typical_price, volume)])
    cum_vol = np.cumsum(volume)
    vwap_vals = cum_vol_price / (cum_vol + 1e-10)
    return [float(x) for x in vwap_vals]

# 42. Accumulation/Distribution Line
def ad_line(high, low, close, volume):
    ad_vals = []
    for i in range(len(close)):
        clv = ((close[i] - low[i]) - (high[i] - close[i])) / (high[i] - low[i] + 1e-10)
        ad_vals.append(clv * volume[i])
    return list(np.cumsum(ad_vals).astype(float))

# 43. Chaikin Money Flow (CMF)
def cmf(high, low, close, volume, period=20):
    mfv = [((c - l) - (h - c)) / (h - l + 1e-10) * v for c, l, h, v in zip(close, low, high, volume)]
    cmf_vals = [
        float(sum(mfv[i - period:i]) / (sum(volume[i - period:i]) + 1e-10))
        for i in range(period, len(close))
    ]
    return cmf_vals

# 44. Volume Oscillator
def volume_oscillator(volume, short_period=14, long_period=28):
    short_ma = ema(volume, short_period)
    long_ma = ema(volume, long_period)
    min_len = min(len(short_ma), len(long_ma))
    return [float((short_ma[-min_len:][i] - long_ma[-min_len:][i]) / (long_ma[-min_len:][i] + 1e-10) * 100)
            for i in range(min_len)]

# 45. Force Index
def force_index(close, volume, period=13):
    fi = [float((close[i] - close[i - 1]) * volume[i]) for i in range(1, len(close))]
    return ema(fi, period)

# 46. Money Flow Index (MFI)
def mfi(high, low, close, volume, period=14):
    tp = [(h + l + c) / 3 for h, l, c in zip(high, low, close)]
    raw_money_flow = [tp[i] * volume[i] for i in range(len(tp))]
    positive_flow, negative_flow = [], []
    for i in range(1, len(tp)):
        if tp[i] > tp[i - 1]:
            positive_flow.append(raw_money_flow[i])
            negative_flow.append(0)
        elif tp[i] < tp[i - 1]:
            positive_flow.append(0)
            negative_flow.append(raw_money_flow[i])
        else:
            positive_flow.append(0)
            negative_flow.append(0)
    mfi_vals = []
    for i in range(period, len(tp)):
        pos = sum(positive_flow[i - period:i])
        neg = sum(negative_flow[i - period:i]) + 1e-10
        mfr = pos / neg
        mfi = 100 - (100 / (1 + mfr))
        mfi_vals.append(float(mfi))
    return mfi_vals

# 47. Ease of Movement
def ease_of_movement(high, low, volume, period=14):
    emv = []
    for i in range(1, len(high)):
        distance = ((high[i] + low[i]) / 2) - ((high[i - 1] + low[i - 1]) / 2)
        box_ratio = (volume[i] / 100000000) / (high[i] - low[i] + 1e-10)
        emv.append(distance / (box_ratio + 1e-10))
    return sma(emv, period)

# 48. Volume Rate of Change
def volume_roc(volume, period=12):
    return [float((volume[i] - volume[i - period]) / (volume[i - period] + 1e-10) * 100)
            for i in range(period, len(volume))]

# 49. Volume Delta (basic form: buy/sell imbalance)
def volume_delta(close, volume):
    delta = [volume[i] if close[i] > close[i - 1] else -volume[i] for i in range(1, len(close))]
    return [float(x) for x in delta]

# 50. Intraday Intensity
def intraday_intensity(high, low, close, volume, period=21):
    ii = [((2 * close[i] - high[i] - low[i]) / (high[i] - low[i] + 1e-10)) * volume[i]
          for i in range(len(close))]
    return [float(sum(ii[i - period:i]) / sum(volume[i - period:i] + 1e-10))
            for i in range(period, len(close))]

# --- Utility SMA and EMA for reuse ---
def sma(source, period):
    return [float(np.mean(source[i - period:i])) for i in range(period, len(source))]

def ema(source, period):
    ema_vals = [source[0]]
    k = 2 / (period + 1)
    for i in range(1, len(source)):
        ema_vals.append((source[i] - ema_vals[-1]) * k + ema_vals[-1])
    return [float(x) for x in ema_vals]
