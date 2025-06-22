import numpy as np

# --- 40. On Balance Volume (OBV) ---
def obv(closeList, volumeList):
    obv_values = [0.0]
    for i in range(1, len(closeList)):
        if closeList[i] > closeList[i - 1]:
            obv_values.append(obv_values[-1] + volumeList[i])
        elif closeList[i] < closeList[i - 1]:
            obv_values.append(obv_values[-1] - volumeList[i])
        else:
            obv_values.append(obv_values[-1])
    return obv_values

# --- 41. Volume Weighted Average Price (VWAP) ---
def vwap(highList, lowList, closeList, volumeList):
    cum_vol = 0.0
    cum_vp = 0.0
    result = []
    for h, l, c, v in zip(highList, lowList, closeList, volumeList):
        typical = (h + l + c) / 3
        cum_vol += v
        cum_vp += typical * v
        result.append(cum_vp / cum_vol if cum_vol != 0 else 0.0)
    return result

# --- 42. Accumulation/Distribution Line (ADL) ---
def adl(highList, lowList, closeList, volumeList):
    adl_vals = []
    running_sum = 0.0
    for h, l, c, v in zip(highList, lowList, closeList, volumeList):
        mfm = ((c - l) - (h - c)) / (h - l + 1e-10)
        mfv = mfm * v
        running_sum += mfv
        adl_vals.append(running_sum)
    return adl_vals

# --- 43. Chaikin Money Flow (CMF) ---
def cmf(highList, lowList, closeList, volumeList, period=20):
    mfv = [(((c - l) - (h - c)) / (h - l + 1e-10)) * v for h, l, c, v in zip(highList, lowList, closeList, volumeList)]
    return [sum(mfv[i - period + 1:i + 1]) / (sum(volumeList[i - period + 1:i + 1]) + 1e-10) if i >= period - 1 else 0.0 for i in range(len(volumeList))]

# --- 44. Volume Oscillator ---
def volume_oscillator(volumeList, short_period=14, long_period=28):
    short_ma = [sum(volumeList[i - short_period + 1:i + 1]) / short_period if i >= short_period - 1 else 0.0 for i in range(len(volumeList))]
    long_ma = [sum(volumeList[i - long_period + 1:i + 1]) / long_period if i >= long_period - 1 else 0.0 for i in range(len(volumeList))]
    return [(s - l) / l * 100 if l != 0 else 0.0 for s, l in zip(short_ma, long_ma)]

# --- 45. Force Index ---
def force_index(closeList, volumeList):
    return [(closeList[i] - closeList[i - 1]) * volumeList[i] if i > 0 else 0.0 for i in range(len(closeList))]

# --- 46. Money Flow Index (MFI) ---
def mfi(highList, lowList, closeList, volumeList, period=14):
    typical_price = [(h + l + c) / 3 for h, l, c in zip(highList, lowList, closeList)]
    raw_mf = [tp * v for tp, v in zip(typical_price, volumeList)]
    pos_flow = [raw_mf[i] if typical_price[i] > typical_price[i - 1] else 0.0 for i in range(1, len(typical_price))]
    neg_flow = [raw_mf[i] if typical_price[i] < typical_price[i - 1] else 0.0 for i in range(1, len(typical_price))]
    mfi_vals = [0.0] * len(closeList)
    for i in range(period, len(pos_flow)):
        pos_sum = sum(pos_flow[i - period:i])
        neg_sum = sum(neg_flow[i - period:i])
        if neg_sum == 0:
            mfi_vals[i + 1] = 100.0
        else:
            money_ratio = pos_sum / neg_sum
            mfi_vals[i + 1] = 100 - (100 / (1 + money_ratio))
    return mfi_vals

# --- 47. Ease of Movement (EOM) ---
def ease_of_movement(highList, lowList, volumeList, period=14):
    emv = [((highList[i] + lowList[i]) / 2 - (highList[i - 1] + lowList[i - 1]) / 2) * (highList[i] - lowList[i]) / (volumeList[i] + 1e-10) if i > 0 else 0.0 for i in range(len(highList))]
    return [sum(emv[i - period + 1:i + 1]) / period if i >= period - 1 else 0.0 for i in range(len(emv))]

# --- 48. Volume Rate of Change (VROC) ---
def volume_roc(volumeList, period=14):
    return [(volumeList[i] - volumeList[i - period]) / volumeList[i - period] * 100 if i >= period and volumeList[i - period] != 0 else 0.0 for i in range(len(volumeList))]

# --- 49. Volume Delta (buy/sell imbalance) ---
def volume_delta(buy_volume, sell_volume):
    return [b - s for b, s in zip(buy_volume, sell_volume)]

# --- 50. Intraday Intensity ---
def intraday_intensity(closeList, highList, lowList, volumeList, period=14):
    ii = [((2 * c - h - l) / (h - l + 1e-10)) * v for c, h, l, v in zip(closeList, highList, lowList, volumeList)]
    return [sum(ii[i - period + 1:i + 1]) / sum(volumeList[i - period + 1:i + 1]) if i >= period - 1 else 0.0 for i in range(len(volumeList))]

def price_volume_trend(closeList, volumeList):
    pvt = [0.0]
    for i in range(1, len(closeList)):
        change = (closeList[i] - closeList[i - 1]) / closeList[i - 1] if closeList[i - 1] != 0 else 0.0
        pvt.append(pvt[-1] + change * volumeList[i])
    return pvt

# --- Volume-Weighted MACD ---
def vw_macd(closeList, volumeList, short_period=12, long_period=26, signal_period=9):
    vwap = [(c * v) for c, v in zip(closeList, volumeList)]
    total_vol = [volumeList[i] if volumeList[i] != 0 else 1e-10 for i in range(len(volumeList))]
    weighted_price = [vwap[i] / total_vol[i] for i in range(len(vwap))]
    short_ema = ema(weighted_price, short_period)
    long_ema = ema(weighted_price, long_period)
    macd_line = [s - l for s, l in zip(short_ema, long_ema)]
    signal_line = ema(macd_line, signal_period)
    hist = [m - s for m, s in zip(macd_line, signal_line)]
    return macd_line, signal_line, hist

# --- Smoothed OBV ---
def smoothed_obv(obvList, period=14):
    return ema(obvList, period)

# --- Klinger Volume Oscillator ---
def klinger_oscillator(highList, lowList, closeList, volumeList, fast=34, slow=55, signal=13):
    trend = [1 if (h + l + c) / 3 > (h_ + l_ + c_) / 3 else -1 for h, l, c, h_, l_, c_ in zip(highList[1:], lowList[1:], closeList[1:], highList[:-1], lowList[:-1], closeList[:-1])]
    dm = [h - l for h, l in zip(highList, lowList)]
    cm = [abs(dm[i] - dm[i - 1]) if i > 0 else 0.0 for i in range(len(dm))]
    vf = [volumeList[i] * trend[i - 1] * 100 * (dm[i] / cm[i] if cm[i] != 0 else 0.0) if i > 0 else 0.0 for i in range(len(dm))]
    fast_ema = ema(vf, fast)
    slow_ema = ema(vf, slow)
    kvo = [f - s for f, s in zip(fast_ema, slow_ema)]
    signal_line = ema(kvo, signal)
    return kvo, signal_line

# --- Volume Flow Indicator (VFI) ---
def volume_flow_indicator(closeList, volumeList, period=130, coef=0.2):
    import math
    vfi = []
    avg_vol = [sum(volumeList[max(0, i - period + 1):i + 1]) / period for i in range(len(volumeList))]
    for i in range(1, len(closeList)):
        log_ret = math.log(closeList[i] / closeList[i - 1]) if closeList[i - 1] != 0 else 0.0
        cut_off = coef * avg_vol[i]
        v = volumeList[i] if volumeList[i] < 2 * cut_off else cut_off
        vfi.append(v * log_ret if log_ret > 0 else 0.0)
    vfi = [0.0] + vfi
    return ema(vfi, period)

# --- Positive Volume Index (PVI) ---
def pvi(closeList, volumeList):
    result = [1000.0]
    for i in range(1, len(closeList)):
        if volumeList[i] > volumeList[i - 1]:
            change = (closeList[i] - closeList[i - 1]) / closeList[i - 1] if closeList[i - 1] != 0 else 0.0
            result.append(result[-1] + result[-1] * change)
        else:
            result.append(result[-1])
    return result

# --- Negative Volume Index (NVI) ---
def nvi(closeList, volumeList):
    result = [1000.0]
    for i in range(1, len(closeList)):
        if volumeList[i] < volumeList[i - 1]:
            change = (closeList[i] - closeList[i - 1]) / closeList[i - 1] if closeList[i - 1] != 0 else 0.0
            result.append(result[-1] + result[-1] * change)
        else:
            result.append(result[-1])
    return result
