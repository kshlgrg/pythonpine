import numpy as np

# --- Momentum Indicators ---

# --- 1. Williams %R ---
def williams_percent_r(closeList, highList, lowList, period=14):
    wr = []
    for i in range(len(closeList)):
        if i < period:
            wr.append(0.0)
            continue
        highest_high = max(highList[i - period + 1:i + 1])
        lowest_low = min(lowList[i - period + 1:i + 1])
        value = -100 * (highest_high - closeList[i]) / (highest_high - lowest_low + 1e-10)
        wr.append(float(round(value, 6)))
    return wr


# --- 2. DMI / ADX ---
def dmi_adx(highList, lowList, closeList, period=14):
    plus_di = []
    minus_di = []
    adx_vals = []
    tr_list = []
    plus_dm = []
    minus_dm = []

    for i in range(len(closeList)):
        if i == 0:
            tr_list.append(0.0)
            plus_dm.append(0.0)
            minus_dm.append(0.0)
            continue

        tr = max(highList[i] - lowList[i], abs(highList[i] - closeList[i - 1]), abs(lowList[i] - closeList[i - 1]))
        tr_list.append(tr)

        up_move = highList[i] - highList[i - 1]
        down_move = lowList[i - 1] - lowList[i]

        plus_dm_val = up_move if up_move > down_move and up_move > 0 else 0
        minus_dm_val = down_move if down_move > up_move and down_move > 0 else 0

        plus_dm.append(plus_dm_val)
        minus_dm.append(minus_dm_val)

    def wilder_smooth(data):
        smoothed = []
        for i in range(len(data)):
            if i < period:
                smoothed.append(sum(data[:i+1]) / max(i+1, 1))
            else:
                smoothed.append(smoothed[-1] - (smoothed[-1] / period) + data[i])
        return smoothed

    sm_plus_dm = wilder_smooth(plus_dm)
    sm_minus_dm = wilder_smooth(minus_dm)
    sm_tr = wilder_smooth(tr_list)

    for i in range(len(closeList)):
        if sm_tr[i] == 0:
            plus_di.append(0.0)
            minus_di.append(0.0)
            adx_vals.append(0.0)
        else:
            pdi = 100 * sm_plus_dm[i] / sm_tr[i]
            mdi = 100 * sm_minus_dm[i] / sm_tr[i]
            dx = 100 * abs(pdi - mdi) / (pdi + mdi + 1e-10)
            plus_di.append(float(round(pdi, 6)))
            minus_di.append(float(round(mdi, 6)))
            adx_vals.append(float(round(dx, 6)))

    adx_smoothed = wilder_smooth(adx_vals)
    return plus_di, minus_di, adx_smoothed


# --- 3. Momentum Indicator ---
def momentum(closeList, period=10):
    mom = []
    for i in range(len(closeList)):
        if i < period:
            mom.append(0.0)
        else:
            val = closeList[i] - closeList[i - period]
            mom.append(float(round(val, 6)))
    return mom


# --- 4. Elder Impulse ---
def elder_impulse(closeList, ema_period=13, macd_fast=12, macd_slow=26):
    ema_vals = ema(closeList, ema_period)
    macd_line = ema(closeList, macd_fast)
    macd_signal = ema(closeList, macd_slow)

    impulse = []
    for i in range(len(closeList)):
        up = ema_vals[i] > ema_vals[i - 1] if i > 0 else False
        macd_up = macd_line[i] > macd_signal[i]
        if up and macd_up:
            impulse.append(1)
        elif not up and not macd_up:
            impulse.append(-1)
        else:
            impulse.append(0)
    return impulse


# --- 5. Schaff Trend Cycle (STC) ---
def schaff_trend_cycle(closeList, fast=23, slow=50, cycle=10):
    macd = [ema(closeList, fast)[i] - ema(closeList, slow)[i] for i in range(len(closeList))]
    stoch_k = []

    for i in range(len(macd)):
        if i < cycle:
            stoch_k.append(0.0)
            continue
        hh = max(macd[i - cycle + 1:i + 1])
        ll = min(macd[i - cycle + 1:i + 1])
        val = 100 * (macd[i] - ll) / (hh - ll + 1e-10)
        stoch_k.append(val)

    stc = ema(stoch_k, cycle)
    return [float(round(val, 6)) for val in stc]


# --- 6. Chande Momentum Oscillator ---
def chande_momentum_oscillator(closeList, period=14):
    cmo = []
    for i in range(len(closeList)):
        if i < period:
            cmo.append(0.0)
            continue
        up = sum(max(closeList[j] - closeList[j - 1], 0) for j in range(i - period + 1, i + 1))
        down = sum(max(closeList[j - 1] - closeList[j], 0) for j in range(i - period + 1, i + 1))
        total = up + down
        cmo_val = 100 * (up - down) / (total + 1e-10)
        cmo.append(float(round(cmo_val, 6)))
    return cmo


# --- 7. Relative Vigor Index (RVI) ---
def relative_vigor_index(openList, highList, lowList, closeList, period=10):
    rvi = []
    signal = []
    for i in range(len(closeList)):
        if i < period:
            rvi.append(0.0)
            signal.append(0.0)
            continue
        num = sum((closeList[j] - openList[j]) for j in range(i - period + 1, i + 1))
        den = sum((highList[j] - lowList[j]) for j in range(i - period + 1, i + 1))
        val = num / den if den != 0 else 0.0
        rvi.append(float(round(val, 6)))

    # Signal = 4-bar SMA of RVI
    for i in range(len(rvi)):
        if i < 3:
            signal.append(0.0)
        else:
            s = sum(rvi[i - j] for j in range(4)) / 4
            signal.append(float(round(s, 6)))

    return rvi, signal

# --- Utility EMA ---
def ema(series, period):
    ema_vals = []
    multiplier = 2 / (period + 1)
    for i in range(len(series)):
        if i == 0:
            ema_vals.append(series[0])
        else:
            ema_val = (series[i] - ema_vals[-1]) * multiplier + ema_vals[-1]
            ema_vals.append(ema_val)
    return ema_vals

# --- 8. RSI ---
def rsi(closeList, period=14):
    rsi_vals = []
    gain, loss = 0, 0
    for i in range(1, period + 1):
        change = closeList[i] - closeList[i - 1]
        gain += max(change, 0)
        loss += max(-change, 0)
    avg_gain = gain / period
    avg_loss = loss / period
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_vals = [100 - (100 / (1 + rs))] * (period + 1)
    for i in range(period + 1, len(closeList)):
        change = closeList[i] - closeList[i - 1]
        gain = max(change, 0)
        loss = max(-change, 0)
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
        rs = avg_gain / (avg_loss + 1e-10)
        rsi_vals.append(100 - (100 / (1 + rs)))
    return [float(round(x, 6)) for x in rsi_vals]


# --- 9. Stochastic RSI ---
def stoch_rsi(closeList, rsi_period=14, stoch_period=14):
    rsi_vals = rsi(closeList, rsi_period)
    stoch_vals = []
    for i in range(len(rsi_vals)):
        if i < stoch_period:
            stoch_vals.append(0.0)
            continue
        lowest = min(rsi_vals[i - stoch_period + 1:i + 1])
        highest = max(rsi_vals[i - stoch_period + 1:i + 1])
        value = (rsi_vals[i] - lowest) / (highest - lowest + 1e-10)
        stoch_vals.append(float(round(value, 6)))
    return stoch_vals


# --- 10. Stochastic Oscillator ---
def stochastic_oscillator(closeList, highList, lowList, period=14):
    k_vals = []
    for i in range(len(closeList)):
        if i < period:
            k_vals.append(0.0)
            continue
        high = max(highList[i - period + 1:i + 1])
        low = min(lowList[i - period + 1:i + 1])
        value = 100 * (closeList[i] - low) / (high - low + 1e-10)
        k_vals.append(float(round(value, 6)))
    return k_vals


# --- 11. MACD ---
def macd(closeList, fast=12, slow=26, signal=9):
    ema_fast = ema(closeList, fast)
    ema_slow = ema(closeList, slow)
    macd_line = [f - s for f, s in zip(ema_fast, ema_slow)]
    signal_line = ema(macd_line, signal)
    histogram = [m - s for m, s in zip(macd_line, signal_line)]
    return (
        [float(round(x, 6)) for x in macd_line],
        [float(round(x, 6)) for x in signal_line],
        [float(round(x, 6)) for x in histogram]
    )


# --- 12. Rate of Change (ROC) ---
def roc(closeList, period=12):
    roc_vals = []
    for i in range(len(closeList)):
        if i < period:
            roc_vals.append(0.0)
            continue
        value = 100 * (closeList[i] - closeList[i - period]) / (closeList[i - period] + 1e-10)
        roc_vals.append(float(round(value, 6)))
    return roc_vals


# --- 13. CCI ---
def cci(closeList, highList, lowList, period=20):
    cci_vals = []
    for i in range(len(closeList)):
        if i < period:
            cci_vals.append(0.0)
            continue
        tp = [(highList[j] + lowList[j] + closeList[j]) / 3 for j in range(i - period + 1, i + 1)]
        tp_current = tp[-1]
        sma = sum(tp) / period
        mean_dev = sum([abs(x - sma) for x in tp]) / period
        cci = (tp_current - sma) / (0.015 * mean_dev + 1e-10)
        cci_vals.append(float(round(cci, 6)))
    return cci_vals


# --- 14. TRIX ---
def trix(closeList, period=15):
    ema1 = ema(closeList, period)
    ema2 = ema(ema1, period)
    ema3 = ema(ema2, period)
    trix_vals = [0.0]
    for i in range(1, len(ema3)):
        val = 100 * (ema3[i] - ema3[i - 1]) / (ema3[i - 1] + 1e-10)
        trix_vals.append(float(round(val, 6)))
    return trix_vals


# --- 15. Ultimate Oscillator ---
def ultimate_oscillator(closeList, highList, lowList, s1=7, s2=14, s3=28):
    bp = []
    tr = []
    for i in range(len(closeList)):
        prev_close = closeList[i - 1] if i > 0 else closeList[0]
        bp.append(closeList[i] - min(lowList[i], prev_close))
        tr.append(max(highList[i], prev_close) - min(lowList[i], prev_close))

    def avg(values, period):
        return [sum(values[i - period + 1:i + 1]) for i in range(len(values))]

    avg7 = [0.0]*s3
    avg14 = [0.0]*s3
    avg28 = [0.0]*s3

    for i in range(s3, len(closeList)):
        bps7 = sum(bp[i - s1 + 1:i + 1])
        trs7 = sum(tr[i - s1 + 1:i + 1]) + 1e-10
        bps14 = sum(bp[i - s2 + 1:i + 1])
        trs14 = sum(tr[i - s2 + 1:i + 1]) + 1e-10
        bps28 = sum(bp[i - s3 + 1:i + 1])
        trs28 = sum(tr[i - s3 + 1:i + 1]) + 1e-10
        uo = 100 * (4 * bps7 / trs7 + 2 * bps14 / trs14 + bps28 / trs28) / 7
        avg28.append(float(round(uo, 6)))

    return avg28

# --- 16. True Strength Index (TSI) ---
def true_strength_index(closeList, long=25, short=13):
    momentum = [0] + [closeList[i] - closeList[i - 1] for i in range(1, len(closeList))]
    abs_momentum = [abs(m) for m in momentum]

    ema1 = ema(momentum, short)
    ema2 = ema(ema1, long)
    ema_abs1 = ema(abs_momentum, short)
    ema_abs2 = ema(ema_abs1, long)

    tsi = [100 * (e / a) if a != 0 else 0.0 for e, a in zip(ema2, ema_abs2)]
    return [float(round(x, 6)) for x in tsi]


# --- 17. Kaufman Adaptive Moving Average (KAMA) ---
def kama(priceList, period=10, fast=2, slow=30):
    kama_vals = [priceList[0]]
    fast_sc = 2 / (fast + 1)
    slow_sc = 2 / (slow + 1)

    for i in range(1, len(priceList)):
        if i < period:
            kama_vals.append(priceList[i])
            continue

        change = abs(priceList[i] - priceList[i - period])
        volatility = sum(abs(priceList[j] - priceList[j - 1]) for j in range(i - period + 1, i + 1))
        er = change / (volatility + 1e-10)
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        kama_val = kama_vals[-1] + sc * (priceList[i] - kama_vals[-1])
        kama_vals.append(kama_val)
    return [float(round(x, 6)) for x in kama_vals]


# --- 18. Connors RSI ---
def connors_rsi(closeList, rsi_period=3, streak_rsi_period=2, rank_period=100):
    def streaks(prices):
        streak = [0]
        for i in range(1, len(prices)):
            if prices[i] > prices[i - 1]:
                streak.append(streak[-1] + 1 if streak[-1] >= 0 else 1)
            elif prices[i] < prices[i - 1]:
                streak.append(streak[-1] - 1 if streak[-1] <= 0 else -1)
            else:
                streak.append(0)
        return streak

    def rsi_like(series, period):
        up, down = 0, 0
        for i in range(1, period + 1):
            diff = series[i] - series[i - 1]
            up += max(diff, 0)
            down += max(-diff, 0)
        avg_up = up / period
        avg_down = down / period
        rs = avg_up / (avg_down + 1e-10)
        rsi_vals = [100 - 100 / (1 + rs)]
        for i in range(period + 1, len(series)):
            diff = series[i] - series[i - 1]
            up = max(diff, 0)
            down = max(-diff, 0)
            avg_up = (avg_up * (period - 1) + up) / period
            avg_down = (avg_down * (period - 1) + down) / period
            rs = avg_up / (avg_down + 1e-10)
            rsi_vals.append(100 - 100 / (1 + rs))
        return [0.0] * (len(series) - len(rsi_vals)) + rsi_vals

    streak = streaks(closeList)
    rsi_close = rsi_like(closeList, rsi_period)
    rsi_streak = rsi_like(streak, streak_rsi_period)

    rank_vals = []
    for i in range(len(closeList)):
        if i < rank_period:
            rank_vals.append(0.0)
            continue
        count = sum(1 for j in range(i - rank_period + 1, i + 1) if closeList[j] < closeList[i])
        rank = 100 * count / rank_period
        rank_vals.append(rank)

    crsi = [(a + b + c) / 3 for a, b, c in zip(rsi_close, rsi_streak, rank_vals)]
    return [float(round(x, 6)) for x in crsi]


# --- 19. Vortex Indicator (VI+ / VI-) ---
def vortex_indicator(highList, lowList, closeList, period=14):
    vi_plus = []
    vi_minus = []
    tr_list = [0.0]
    vm_plus = [0.0]
    vm_minus = [0.0]

    for i in range(1, len(closeList)):
        tr = max(highList[i] - lowList[i], abs(highList[i] - closeList[i - 1]), abs(lowList[i] - closeList[i - 1]))
        tr_list.append(tr)
        vm_plus.append(abs(highList[i] - lowList[i - 1]))
        vm_minus.append(abs(lowList[i] - highList[i - 1]))

    for i in range(len(closeList)):
        if i < period:
            vi_plus.append(0.0)
            vi_minus.append(0.0)
            continue
        sum_tr = sum(tr_list[i - period + 1:i + 1]) + 1e-10
        sum_vm_plus = sum(vm_plus[i - period + 1:i + 1])
        sum_vm_minus = sum(vm_minus[i - period + 1:i + 1])
        vi_plus.append(float(round(sum_vm_plus / sum_tr, 6)))
        vi_minus.append(float(round(sum_vm_minus / sum_tr, 6)))

    return vi_plus, vi_minus


# --- 20. RSX (Smoothed RSI) ---
def rsx(closeList, period=14):
    # This is a simplified RSX approximation using double-smoothed RSI
    rsi_vals = rsi(closeList, period)
    smoothed = ema(rsi_vals, period)
    return [float(round(x, 6)) for x in smoothed]


# --- 21. Slope of EMA ---
def slope_of_ema(closeList, period=14):
    ema_vals = ema(closeList, period)
    slope_vals = [0.0]
    for i in range(1, len(ema_vals)):
        slope = ema_vals[i] - ema_vals[i - 1]
        slope_vals.append(float(round(slope, 6)))
    return slope_vals


# --- 22. Directional Trend Index (DTI) ---
def directional_trend_index(closeList, period=14):
    dti = []
    for i in range(len(closeList)):
        if i < period:
            dti.append(0.0)
            continue
        trend_strength = abs(closeList[i] - closeList[i - period])
        price_range = max(closeList[i - period:i + 1]) - min(closeList[i - period:i + 1]) + 1e-10
        dti_val = 100 * trend_strength / price_range
        dti.append(float(round(dti_val, 6)))
    return dti
