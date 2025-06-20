# --- 51. Pivot Points (Classic, Fibonacci, Camarilla) ---
def pivot_points(high, low, close, method='classic'):
    pp = (high + low + close) / 3
    range_ = high - low
    pivots = {
        'classic': {
            'pp': pp,
            'r1': 2 * pp - low,
            's1': 2 * pp - high,
            'r2': pp + range_,
            's2': pp - range_,
            'r3': high + 2 * (pp - low),
            's3': low - 2 * (high - pp)
        },
        'fibonacci': {
            'pp': pp,
            'r1': pp + 0.382 * range_,
            's1': pp - 0.382 * range_,
            'r2': pp + 0.618 * range_,
            's2': pp - 0.618 * range_,
            'r3': pp + 1.0 * range_,
            's3': pp - 1.0 * range_
        },
        'camarilla': {
            'pp': pp,
            'r1': close + range_ * 1.1 / 12,
            's1': close - range_ * 1.1 / 12,
            'r2': close + range_ * 1.1 / 6,
            's2': close - range_ * 1.1 / 6,
            'r3': close + range_ * 1.1 / 4,
            's3': close - range_ * 1.1 / 4
        }
    }
    return pivots.get(method.lower(), pivots['classic'])


# --- 52. Price Rate of Change ---
def price_roc(priceList, period=14, scale=True):
    return [0.0 if i < period else float(round((priceList[i] - priceList[i - period]) / (priceList[i - period] + 1e-10) * (100 if scale else 1), 6)) for i in range(len(priceList))]


# --- 53. ZigZag ---
def zigzag(priceList, threshold=5.0, use_percentage=True):
    zz = [priceList[0]]
    direction = None
    for i in range(1, len(priceList)):
        base = zz[-1]
        change = (priceList[i] - base) / (abs(base) + 1e-10) * 100 if use_percentage else priceList[i] - base
        if direction is None:
            direction = 'up' if change > 0 else 'down'
        if direction == 'up' and change <= -threshold:
            zz.append(priceList[i])
            direction = 'down'
        elif direction == 'down' and change >= threshold:
            zz.append(priceList[i])
            direction = 'up'
    zz += [priceList[-1]] if zz[-1] != priceList[-1] else []
    return zz


# --- 54. Heikin Ashi Candles ---
def heikin_ashi(openList, highList, lowList, closeList, round_decimals=6):
    ha_open = [(openList[0] + closeList[0]) / 2]
    ha_close = []
    ha_high = []
    ha_low = []

    for i in range(len(closeList)):
        ha_c = (openList[i] + highList[i] + lowList[i] + closeList[i]) / 4
        ha_close.append(round(ha_c, round_decimals))
        if i > 0:
            ha_o = (ha_open[-1] + ha_close[-2]) / 2
            ha_open.append(round(ha_o, round_decimals))
        ha_h = max(highList[i], ha_open[-1], ha_close[-1])
        ha_l = min(lowList[i], ha_open[-1], ha_close[-1])
        ha_high.append(round(ha_h, round_decimals))
        ha_low.append(round(ha_l, round_decimals))

    return ha_open, ha_high, ha_low, ha_close


# --- 55. Renko Boxes ---
def renko_boxes(closeList, box_size=1.0, show_full_boxes=False):
    renko = [closeList[0]]
    direction = None
    for price in closeList[1:]:
        last = renko[-1]
        if direction == 'up':
            while price >= last + box_size:
                last += box_size
                renko.append(last if show_full_boxes else price)
        elif direction == 'down':
            while price <= last - box_size:
                last -= box_size
                renko.append(last if show_full_boxes else price)
        else:
            if price > last + box_size:
                direction = 'up'
            elif price < last - box_size:
                direction = 'down'
    return renko


# --- 56. Engulfing Pattern ---
def detect_engulfing(openList, closeList, require_body_ratio=1.0):
    result = [0.0] * len(closeList)
    for i in range(1, len(closeList)):
        prev_body = abs(closeList[i - 1] - openList[i - 1])
        curr_body = abs(closeList[i] - openList[i])
        if curr_body < prev_body * require_body_ratio:
            continue
        if closeList[i - 1] > openList[i - 1] and closeList[i] < openList[i] and closeList[i] < openList[i - 1] and openList[i] > closeList[i - 1]:
            result[i] = -1
        elif closeList[i - 1] < openList[i - 1] and closeList[i] > openList[i] and closeList[i] > openList[i - 1] and openList[i] < closeList[i - 1]:
            result[i] = 1
    return result


# --- 57. Pin Bar Detection ---
def detect_pin_bar(openList, highList, lowList, closeList, wick_ratio=2.0):
    result = [0.0] * len(closeList)
    for i in range(len(closeList)):
        body = abs(closeList[i] - openList[i])
        upper = highList[i] - max(closeList[i], openList[i])
        lower = min(closeList[i], openList[i]) - lowList[i]
        if upper > wick_ratio * body and lower < body:
            result[i] = -1
        elif lower > wick_ratio * body and upper < body:
            result[i] = 1
    return result


# --- 58. Double Top/Bottom Detection ---
def detect_double_top_bottom(highList, lowList, threshold=0.005, min_spacing=2):
    tops = []
    bottoms = []
    for i in range(min_spacing, len(highList)):
        if abs(highList[i] - highList[i - min_spacing]) / (highList[i - min_spacing] + 1e-10) < threshold:
            tops.append(i)
        if abs(lowList[i] - lowList[i - min_spacing]) / (lowList[i - min_spacing] + 1e-10) < threshold:
            bottoms.append(i)
    return tops, bottoms


# --- 59. Support/Resistance Zones ---
def support_resistance_zones(highList, lowList, sensitivity=5, method='pivot'):
    support = []
    resistance = []
    for i in range(sensitivity, len(highList) - sensitivity):
        if method == 'pivot':
            is_support = all(lowList[i] < lowList[j] for j in range(i - sensitivity, i + sensitivity))
            is_resistance = all(highList[i] > highList[j] for j in range(i - sensitivity, i + sensitivity))
            if is_support:
                support.append(i)
            if is_resistance:
                resistance.append(i)
    return support, resistance


# --- 60. Candlestick Pattern Count ---
def count_candle_patterns(pattern_list, bars=20, pattern_filter=None):
    from collections import Counter
    filtered = pattern_list[-bars:]
    if pattern_filter:
        filtered = [x for x in filtered if x in pattern_filter]
    return Counter(filtered)

# --- 61. Doji / Spinning Top Detection ---
def detect_doji(openList, closeList, highList, lowList, body_threshold=0.1):
    result = [0.0] * len(closeList)
    for i in range(len(closeList)):
        body = abs(closeList[i] - openList[i])
        range_ = highList[i] - lowList[i]
        if range_ == 0:
            continue
        if body / range_ < body_threshold:
            result[i] = 1
    return result


# --- 62. Inside / Outside Bar Detection ---
def detect_inside_outside_bars(highList, lowList):
    result = [0.0] * len(highList)
    for i in range(1, len(highList)):
        if highList[i] < highList[i - 1] and lowList[i] > lowList[i - 1]:
            result[i] = 1  # Inside Bar
        elif highList[i] > highList[i - 1] and lowList[i] < lowList[i - 1]:
            result[i] = -1  # Outside Bar
    return result


# --- 63. Marubozu Detection ---
def detect_marubozu(openList, closeList, highList, lowList, tolerance=0.1):
    result = [0.0] * len(closeList)
    for i in range(len(closeList)):
        candle_high = highList[i]
        candle_low = lowList[i]
        candle_open = openList[i]
        candle_close = closeList[i]
        total_range = candle_high - candle_low
        if total_range == 0:
            continue
        upper_shadow = candle_high - max(candle_open, candle_close)
        lower_shadow = min(candle_open, candle_close) - candle_low
        if upper_shadow < tolerance * total_range and lower_shadow < tolerance * total_range:
            result[i] = 1 if candle_close > candle_open else -1
    return result


# --- 64. 3-Bar Reversal Pattern ---
def detect_three_bar_reversal(closeList):
    result = [0.0] * len(closeList)
    for i in range(2, len(closeList)):
        if closeList[i - 2] < closeList[i - 1] > closeList[i] and closeList[i] < closeList[i - 2]:
            result[i] = -1
        elif closeList[i - 2] > closeList[i - 1] < closeList[i] and closeList[i] > closeList[i - 2]:
            result[i] = 1
    return result


# --- 65. Fractal High/Low ---
def detect_fractals(highList, lowList, window=2):
    highs = []
    lows = []
    for i in range(window, len(highList) - window):
        if all(highList[i] > highList[i - j] and highList[i] > highList[i + j] for j in range(1, window + 1)):
            highs.append(i)
        if all(lowList[i] < lowList[i - j] and lowList[i] < lowList[i + j] for j in range(1, window + 1)):
            lows.append(i)
    return highs, lows


# --- 66. Bar Range Ratio ---
def bar_range_ratio(openList, closeList, highList, lowList):
    result = []
    for i in range(len(closeList)):
        body = abs(closeList[i] - openList[i])
        total_range = highList[i] - lowList[i]
        ratio = body / (total_range + 1e-10)
        result.append(round(ratio, 4))
    return result


# --- 67. Wick Ratio Indicator ---
def wick_ratio(openList, closeList, highList, lowList):
    result = []
    for i in range(len(closeList)):
        upper = highList[i] - max(openList[i], closeList[i])
        lower = min(openList[i], closeList[i]) - lowList[i]
        body = abs(closeList[i] - openList[i])
        total = upper + lower + body + 1e-10
        result.append(round((upper + lower) / total, 4))
    return result


# --- 68. High/Low Breakout Detector ---
def high_low_breakout(priceList, lookback=20):
    result = [0.0] * len(priceList)
    for i in range(lookback, len(priceList)):
        highest = max(priceList[i - lookback:i])
        lowest = min(priceList[i - lookback:i])
        if priceList[i] > highest:
            result[i] = 1
        elif priceList[i] < lowest:
            result[i] = -1
    return result


# --- 69. Trend Candle Strength Index ---
def trend_candle_strength(openList, closeList, length=20):
    result = []
    for i in range(len(closeList)):
        if i < length:
            result.append(0.0)
            continue
        count = sum(1 for j in range(i - length, i) if closeList[j] > openList[j])
        result.append(round(count / length, 4))
    return result


# --- 70. Price Action Score ---
def price_action_score(engulfing, pinbar, fractal_highs, fractal_lows, index):
    score = 0
    if index < len(engulfing) and engulfing[index] != 0:
        score += 1
    if index < len(pinbar) and pinbar[index] != 0:
        score += 1
    if index in fractal_highs:
        score += 1
    if index in fractal_lows:
        score += 1
    return score

