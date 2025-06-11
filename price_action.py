#51. Pivot Points (Classic, Fibonacci, Camarilla)

def pivot_points_classic(high, low, close):
    pivot = (high[-2] + low[-2] + close[-2]) / 3
    r1 = 2 * pivot - low[-2]
    s1 = 2 * pivot - high[-2]
    r2 = pivot + (high[-2] - low[-2])
    s2 = pivot - (high[-2] - low[-2])
    return pivot, r1, s1, r2, s2

def pivot_points_fibonacci(high, low, close):
    pivot = (high[-2] + low[-2] + close[-2]) / 3
    diff = high[-2] - low[-2]
    r1 = pivot + 0.382 * diff
    r2 = pivot + 0.618 * diff
    r3 = pivot + 1.000 * diff
    s1 = pivot - 0.382 * diff
    s2 = pivot - 0.618 * diff
    s3 = pivot - 1.000 * diff
    return pivot, r1, r2, r3, s1, s2, s3

def pivot_points_camarilla(high, low, close):
    h = high[-2]
    l = low[-2]
    c = close[-2]
    diff = h - l
    r4 = c + (diff * 1.5000) / 2
    r3 = c + (diff * 1.2500) / 2
    r2 = c + (diff * 1.1666) / 2
    r1 = c + (diff * 1.0833) / 2
    s1 = c - (diff * 1.0833) / 2
    s2 = c - (diff * 1.1666) / 2
    s3 = c - (diff * 1.2500) / 2
    s4 = c - (diff * 1.5000) / 2
    return r4, r3, r2, r1, s1, s2, s3, s4

#52. Price Rate of Change
def price_rate_of_change(close, period=12):
    return ((close[0] - close[period]) / close[period]) * 100

#53. ZigZag (simplified version using threshold)
def zigzag(close, threshold=5):
    result = [close[0]]
    for i in range(1, len(close)):
        change = abs((close[i] - result[-1]) / result[-1]) * 100
        if change >= threshold:
            result.append(close[i])
    return result

#54. Heikin Ashi Candles
def heikin_ashi(open_array, high_array, low_array, close_array):
    ha_close = [(o + h + l + c) / 4 for o, h, l, c in zip(open_array, high_array, low_array, close_array)]
    ha_open = [open_array[0]]
    for i in range(1, len(open_array)):
        ha_open.append((ha_open[i-1] + ha_close[i-1]) / 2)
    ha_high = [max(h, o, c) for h, o, c in zip(high_array, ha_open, ha_close)]
    ha_low = [min(l, o, c) for l, o, c in zip(low_array, ha_open, ha_close)]
    return ha_open, ha_high, ha_low, ha_close

#55. Renko Boxes (very simplified placeholder version)
def renko_boxes(close, box_size=10):
    boxes = [close[0]]
    for price in close[1:]:
        if abs(price - boxes[-1]) >= box_size:
            boxes.append(price)
    return boxes

#56. Engulfing Pattern
def is_bullish_engulfing(open_array, close_array):
    return open_array[1] > close_array[1] and close_array[0] > open_array[0] and open_array[0] < close_array[1]

def is_bearish_engulfing(open_array, close_array):
    return open_array[1] < close_array[1] and close_array[0] < open_array[0] and open_array[0] > close_array[1]

#57. Pin Bar Detection
def is_pin_bar(open_array, high_array, low_array, close_array):
    body = abs(open_array[0] - close_array[0])
    upper_wick = high_array[0] - max(open_array[0], close_array[0])
    lower_wick = min(open_array[0], close_array[0]) - low_array[0]
    return upper_wick > 2 * body or lower_wick > 2 * body

#58. Double Top / Bottom (very basic logic)
def is_double_top(close):
    return abs(close[0] - close[2]) < 0.005 and close[1] < close[0]

def is_double_bottom(close):
    return abs(close[0] - close[2]) < 0.005 and close[1] > close[0]

#59. Support / Resistance Zones (basic version)
def support_resistance_zones(high_array, low_array, lookback=20):
    resistance = max(high_array[:lookback])
    support = min(low_array[:lookback])
    return support, resistance

#60. Candlestick Pattern Count (Last N Bars)
def count_bullish_bars(open_array, close_array, n=20):
    return sum(1 for i in range(n) if close_array[i] > open_array[i])

def count_bearish_bars(open_array, close_array, n=20):
    return sum(1 for i in range(n) if close_array[i] < open_array[i])
