import numpy as np

# 16. RSI (Relative Strength Index)
def rsi(source, period=14):
    gains, losses = [], []
    for i in range(1, len(source)):
        change = source[i] - source[i-1]
        gains.append(max(change, 0))
        losses.append(abs(min(change, 0)))
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    rsis = []
    for i in range(period, len(source) - 1):
        gain = gains[i]
        loss = losses[i]
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
        rs = avg_gain / (avg_loss + 1e-10)
        rsis.append(100 - (100 / (1 + rs)))
    return [float(x) for x in rsis]

# 17. Stochastic RSI
def stochastic_rsi(source, rsi_period=14, stoch_period=14):
    rsi_vals = rsi(source, rsi_period)
    stoch = []
    for i in range(stoch_period, len(rsi_vals)):
        window = rsi_vals[i - stoch_period:i]
        stoch_val = (rsi_vals[i] - min(window)) / (max(window) - min(window) + 1e-10)
        stoch.append(stoch_val * 100)
    return [float(x) for x in stoch]

# 18. Stochastic Oscillator
def stochastic_oscillator(close, high, low, period=14):
    k_values = []
    for i in range(period, len(close)):
        highest = max(high[i - period:i])
        lowest = min(low[i - period:i])
        k = (close[i] - lowest) / (highest - lowest + 1e-10) * 100
        k_values.append(k)
    return [float(x) for x in k_values]

# 19. MACD
def macd(source, fast=12, slow=26, signal=9):
    fast_ema = ema(source, fast)
    slow_ema = ema(source, slow)
    macd_line = [f - s for f, s in zip(fast_ema[-len(slow_ema):], slow_ema)]
    signal_line = ema(macd_line, signal)
    histogram = [m - s for m, s in zip(macd_line[-len(signal_line):], signal_line)]
    return (
        [float(x) for x in macd_line[-len(signal_line):]],
        [float(x) for x in signal_line],
        [float(x) for x in histogram]
    )

# Reuse EMA
def ema(source, period):
    ema_values = []
    multiplier = 2 / (period + 1)
    ema_values.append(source[0])
    for i in range(1, len(source)):
        ema_values.append((source[i] - ema_values[-1]) * multiplier + ema_values[-1])
    return ema_values

# 20. Rate of Change (ROC)
def roc(source, period=14):
    roc_vals = []
    for i in range(period, len(source)):
        val = ((source[i] - source[i - period]) / source[i - period]) * 100
        roc_vals.append(val)
    return [float(x) for x in roc_vals]

# 21. Commodity Channel Index (CCI)
def cci(high, low, close, period=20):
    tp = [(h + l + c) / 3 for h, l, c in zip(high, low, close)]
    cci_vals = []
    for i in range(period, len(tp)):
        mean = np.mean(tp[i - period:i])
        dev = np.mean([abs(x - mean) for x in tp[i - period:i]])
        cci = (tp[i] - mean) / (0.015 * dev + 1e-10)
        cci_vals.append(cci)
    return [float(x) for x in cci_vals]

# 22. TRIX
def trix(source, period=15):
    ema1 = ema(source, period)
    ema2 = ema(ema1, period)
    ema3 = ema(ema2, period)
    trix_vals = []
    for i in range(1, len(ema3)):
        value = ((ema3[i] - ema3[i - 1]) / (ema3[i - 1] + 1e-10)) * 100
        trix_vals.append(value)
    return [float(x) for x in trix_vals]

# 23. Ultimate Oscillator
def ultimate_oscillator(high, low, close, s1=7, s2=14, s3=28):
    bp, tr = [], []
    for i in range(1, len(close)):
        bp.append(close[i] - min(low[i], close[i - 1]))
        tr.append(max(high[i], close[i - 1]) - min(low[i], close[i - 1]))
    def avg(p, t, s):
        return sum(p[-s:]) / (sum(t[-s:]) + 1e-10)
    uo = []
    for i in range(max(s1, s2, s3), len(bp)):
        avg1 = avg(bp[i-s1:i], tr[i-s1:i], s1)
        avg2 = avg(bp[i-s2:i], tr[i-s2:i], s2)
        avg3 = avg(bp[i-s3:i], tr[i-s3:i], s3)
        uo_val = 100 * (4 * avg1 + 2 * avg2 + avg3) / 7
        uo.append(uo_val)
    return [float(x) for x in uo]

# 24. Williams %R
def williams_r(high, low, close, period=14):
    wr = []
    for i in range(period, len(close)):
        highest = max(high[i - period:i])
        lowest = min(low[i - period:i])
        r = (highest - close[i]) / (highest - lowest + 1e-10) * -100
        wr.append(r)
    return [float(x) for x in wr]

# 25. DMI / ADX
def adx(high, low, close, period=14):
    plus_dm, minus_dm, tr = [], [], []
    for i in range(1, len(high)):
        up = high[i] - high[i - 1]
        dn = low[i - 1] - low[i]
        plus_dm.append(up if up > dn and up > 0 else 0)
        minus_dm.append(dn if dn > up and dn > 0 else 0)
        tr.append(max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1])))
    atr = ema(tr, period)
    plus_di = [100 * pdm / (a + 1e-10) for pdm, a in zip(ema(plus_dm, period), atr)]
    minus_di = [100 * mdm / (a + 1e-10) for mdm, a in zip(ema(minus_dm, period), atr)]
    dx = [100 * abs(p - m) / (p + m + 1e-10) for p, m in zip(plus_di, minus_di)]
    adx_vals = ema(dx, period)
    return [float(x) for x in adx_vals]

# 26. Momentum Indicator
def momentum(source, period=10):
    return [float(source[i] - source[i - period]) for i in range(period, len(source))]

# 27. Elder Impulse
def elder_impulse(close, ema_period=13, macd_fast=12, macd_slow=26):
    ema_vals = ema(close, ema_period)
    macd_line = macd(close, macd_fast, macd_slow)[0]
    min_len = min(len(ema_vals), len(macd_line))
    impulse = []
    for i in range(min_len):
        if ema_vals[i] > ema_vals[i - 1] and macd_line[i] > macd_line[i - 1]:
            impulse.append(1)  # green
        elif ema_vals[i] < ema_vals[i - 1] and macd_line[i] < macd_line[i - 1]:
            impulse.append(-1)  # red
        else:
            impulse.append(0)  # blue
    return impulse

# 28. Schaff Trend Cycle (Simplified)
def schaff_trend_cycle(source, short=23, long=50, cycle=10):
    macd_line = macd(source, short, long)[0]
    stc = stochastic_oscillator(macd_line, macd_line, macd_line, cycle)
    return stc

# 29. Chande Momentum Oscillator
def cmo(source, period=14):
    cmo_vals = []
    for i in range(period, len(source)):
        diff = np.diff(source[i - period:i])
        up = sum([x for x in diff if x > 0])
        down = sum([abs(x) for x in diff if x < 0])
        cmo = 100 * (up - down) / (up + down + 1e-10)
        cmo_vals.append(cmo)
    return [float(x) for x in cmo_vals]

# 30. Relative Vigor Index (RVI)
def rvi(close, open_, high, low, period=10):
    numerator = [(close[i] - open_[i]) for i in range(len(close))]
    denominator = [(high[i] - low[i]) + 1e-10 for i in range(len(high))]
    rvi_raw = [n / d for n, d in zip(numerator, denominator)]
    return sma(rvi_raw, period)

# SMA reused
def sma(source, period):
    return [float(np.mean(source[i - period:i])) for i in range(period, len(source))]
