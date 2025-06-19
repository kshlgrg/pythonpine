import numpy as np
from collections import deque
import pandas as pd
import math
# 61. Fisher Transform
def fisher_transform(source_type="hl2", length=10, start=0, end=500, smooth=True, smooth_factor=0.33):
    """
    Fisher Transform with high/low normalization, smoothing, and clean output.

    Parameters:
        source_type (str): One of "close", "hl2", "ohlc4", "typical".
        length (int): Lookback period.
        start (int): Start index.
        end (int): End index.
        smooth (bool): Whether to apply recursive smoothing.
        smooth_factor (float): EMA smoothing factor (e.g., 0.33 for fast, 0.1 for slow).

    Returns:
        list: Fisher Transform values as clean floats.
    """
    if source_type == "close":
        source = closeList(start, end)
    elif source_type == "hl2":
        source = [(h + l) / 2 for h, l in zip(highList(start, end), lowList(start, end))]
    elif source_type == "ohlc4":
        source = [(o + h + l + c) / 4 for o, h, l, c in zip(
            openList(start, end), highList(start, end), lowList(start, end), closeList(start, end))]
    elif source_type == "typical":
        source = [(h + l + c) / 3 for h, l, c in zip(highList(start, end), lowList(start, end), closeList(start, end))]
    else:
        raise ValueError("Invalid source_type. Use 'close', 'hl2', 'ohlc4', or 'typical'.")

    fish = []
    smoothed_val = 0.0

    for i in range(len(source)):
        idx = i + start

        if idx < length:
            fish.append(0.0)
            continue

        high_val = highest(idx + 1)
        low_val = lowest(idx + 1)
        range_val = high_val - low_val if high_val != low_val else 1e-10

        normalized = 2 * (source[i] - low_val) / range_val - 1
        normalized = max(min(normalized, 0.999), -0.999)

        if smooth:
            smoothed_val = smooth_factor * normalized + (1 - smooth_factor) * smoothed_val
            input_val = smoothed_val
        else:
            input_val = normalized

        fisher = 0.5 * np.log((1 + input_val) / (1 - input_val))
        fish.append(float(round(fisher, 6)))

    return fish


# 62. Hilbert Transform (Cycle and Trend components, simplified)
def hilbert_transform(priceList):
    """
    Discrete Hilbert Transform using a 4-coefficient FIR filter.
    Returns 90° phase-shifted (quadrature) component.
    """
    coeffs = [0.0962, 0.5769, 0.5769, 0.0962]
    buffer = [0.0] * len(priceList)

    for i in range(6, len(priceList)):
        buffer[i] = (
            coeffs[0] * priceList[i]
            - coeffs[1] * priceList[i - 2]
            + coeffs[2] * priceList[i - 4]
            - coeffs[3] * priceList[i - 6]
        )

    return [float(round(val, 6)) for val in buffer]


def ht_sine(priceList):
    """
    Returns sine and leadsine lines using Hilbert components.
    Useful for identifying cyclic turning points.
    """
    q = hilbert_transform(priceList)
    sine = []
    leadsine = []

    for i in range(len(q)):
        in_phase = priceList[i]
        quadrature = q[i]

        phase = math.atan2(quadrature, in_phase)
        sine_val = math.sin(phase)
        leadsine_val = math.sin(phase + math.pi / 4)  # 45° lead

        sine.append(float(round(sine_val, 6)))
        leadsine.append(float(round(leadsine_val, 6)))

    return sine, leadsine


def ht_phase(priceList):
    """
    Returns the instantaneous phase in degrees (0–360°).
    """
    q = hilbert_transform(priceList)
    phase_deg = []

    for i in range(len(q)):
        in_phase = priceList[i]
        quadrature = q[i]
        phase = math.atan2(quadrature, in_phase)
        deg = math.degrees(phase)
        if deg < 0:
            deg += 360
        phase_deg.append(float(round(deg, 2)))

    return phase_deg


def ht_trendline(priceList, alpha=0.07):
    """
    Smoothed price using Hilbert-based phase relationship (trend approximation).
    """
    smooth = [0.0] * len(priceList)
    for i in range(1, len(priceList)):
        smooth[i] = alpha * priceList[i] + (1 - alpha) * smooth[i - 1]
    return [float(round(val, 6)) for val in smooth]


def ht_dominant_cycle(priceList, min_period=10, max_period=50):
    """
    Estimates the dominant cycle length using phase change rate.
    Based on Ehlers' method.
    """
    phase = ht_phase(priceList)
    period = [0.0] * len(priceList)
    delta_phase = 0.0

    for i in range(1, len(priceList)):
        delta = phase[i - 1] - phase[i]
        if delta < 1:
            delta += 360
        if delta > 1:
            delta_phase = delta

        cycle = 0.0 if delta_phase == 0 else 360 / delta_phase
        cycle = min(max(cycle, min_period), max_period)
        period[i] = float(round(cycle, 2))

    return period
# 63. Ehler’s Instantaneous Trendline
def ht_itrend(priceList, alpha=0.07):
    """
    Ehlers' Instantaneous Trendline (ITrend).

    A zero-lag smoothed version of price, using a one-pole recursive filter.
    Tracks price with minimal lag.

    Parameters:
        priceList (list): The input price series (e.g., hl2, closeList, etc.).
        alpha (float): Smoothing constant, usually between 0.05 and 0.2.

    Returns:
        list: Instantaneous trendline as clean floats.
    """
    itrend = [float(priceList[0])]  # Initialize with first price

    for i in range(1, len(priceList)):
        new_val = alpha * priceList[i] + (1 - alpha) * itrend[-1]
        itrend.append(float(round(new_val, 6)))

    return itrend


# 64. Detrended Price Oscillator
def detrended_price_oscillator(priceList, length=14):
    """
    Detrended Price Oscillator (DPO).

    DPO removes long-term trend by subtracting a centered SMA.
    Helps identify short-term cycles and highs/lows.

    Parameters:
        priceList (list): Price data (e.g., closeList, hl2, etc.).
        length (int): Period for SMA and offset calculation.

    Returns:
        list: DPO values as clean floats.
    """
    dpo = []
    offset = int(length / 2) + 1

    for i in range(len(priceList)):
        if i < length or i < offset:
            dpo.append(0.0)
            continue

        price_offset = priceList[i - offset]
        sma = sum(priceList[i - length + 1: i + 1]) / length
        value = price_offset - sma
        dpo.append(float(round(value, 6)))

    return dpo

# 65. Laguerre RSI
def laguerre_rsi(priceList, gamma=0.5):
    """
    Laguerre RSI by John Ehlers.
    A fast-smoothing RSI using Laguerre filter.

    Parameters:
        priceList (list): List of prices (e.g., closeList).
        gamma (float): Smoothing factor (0.2 to 0.8). Default is 0.5.

    Returns:
        list: Laguerre RSI values (0–1 float).
    """
    l0, l1, l2, l3 = 0.0, 0.0, 0.0, 0.0
    rsi_vals = []

    for price in priceList:
        prev_l0, prev_l1, prev_l2, prev_l3 = l0, l1, l2, l3

        l0 = (1 - gamma) * price + gamma * prev_l0
        l1 = - (1 - gamma) * l0 + prev_l0 + gamma * prev_l1
        l2 = - (1 - gamma) * l1 + prev_l1 + gamma * prev_l2
        l3 = - (1 - gamma) * l2 + prev_l2 + gamma * prev_l3

        cu = cd = 0.0
        if l0 >= l1:
            cu += l0 - l1
        else:
            cd += l1 - l0

        if l1 >= l2:
            cu += l1 - l2
        else:
            cd += l2 - l1

        if l2 >= l3:
            cu += l2 - l3
        else:
            cd += l3 - l2

        rsi = cu / (cu + cd) if (cu + cd) != 0 else 0.0
        rsi_vals.append(float(round(rsi, 6)))

    return rsi_vals


# 66. Qstick
def qstick(openList, closeList, length=14):
    """
    QStick Indicator.

    Measures the average candle body over N periods.

    Parameters:
        openList (list): Open prices.
        closeList (list): Close prices.
        length (int): Period for SMA calculation.

    Returns:
        list: QStick values as simple floats.
    """
    qstick_vals = []

    for i in range(len(closeList)):
        if i < length - 1:
            qstick_vals.append(0.0)
            continue

        body_sum = 0.0
        for j in range(i - length + 1, i + 1):
            body_sum += closeList[j] - openList[j]

        avg_body = body_sum / length
        qstick_vals.append(float(round(avg_body, 6)))

    return qstick_vals


# 67. Stochastic Momentum Index (SMI)
def stochastic_momentum_index(closeList, highList, lowList, length=14, smoothK=3, smoothD=3):
    """
    Stochastic Momentum Index (SMI).

    A refined version of Stochastic Oscillator with smoothing.

    Parameters:
        closeList (list): Close prices.
        highList (list): High prices.
        lowList (list): Low prices.
        length (int): Lookback for high/low range.
        smoothK (int): Smoothing for the numerator and denominator (K line).
        smoothD (int): Smoothing for the SMI itself (D line).

    Returns:
        tuple: (smi_k_line, smi_d_line) as lists of floats.
    """

    smi_k = []
    smi_d = []

    price_range = deque(maxlen=length)

    def ema(series, period):
        ema_vals = []
        k = 2 / (period + 1)
        for i, val in enumerate(series):
            if i == 0:
                ema_vals.append(val)
            else:
                ema_vals.append(k * val + (1 - k) * ema_vals[-1])
        return ema_vals

    for i in range(len(closeList)):
        if i < length - 1:
            smi_k.append(0.0)
            smi_d.append(0.0)
            continue

        hh = max(highList[i - length + 1: i + 1])
        ll = min(lowList[i - length + 1: i + 1])
        mid = (hh + ll) / 2
        range_ = (hh - ll) / 2

        num = closeList[i] - mid
        den = range_ if range_ != 0 else 1e-10  # avoid div by zero

        smi_raw = 100 * (num / den)
        smi_k.append(smi_raw)

    # Double smoothing
    smi_k_smooth = ema(ema(smi_k, smoothK), smoothK)
    smi_d_smooth = ema(smi_k_smooth, smoothD)

    # Round to clean floats
    smi_k_smooth = [float(round(x, 6)) for x in smi_k_smooth]
    smi_d_smooth = [float(round(x, 6)) for x in smi_d_smooth]

    return smi_k_smooth, smi_d_smooth


# 68. Adaptive Cycle Divergence (EMA Difference)
def adaptive_cycle_divergence(priceList):
    """
    Adaptive Cycle Divergence (ACD) by Ehlers.

    Detects divergence between price and its dominant cycle component.
    Requires `ht_dominant_cycle()` to be implemented.

    Parameters:
        priceList (list): Price series (e.g., closeList, hl2).

    Returns:
        list: Adaptive Cycle Divergence values as floats.
    """

    # Step 1: Get dominant cycle length
    dcList = ht_dominant_cycle(priceList)
    
    acd = []
    smooth = 0.0

    for i in range(len(priceList)):
        length = int(round(dcList[i]))
        if length < 4 or i < length:
            acd.append(0.0)
            continue

        # Step 2: Get cyclic mean value over dominant cycle
        cycle_component = sum(priceList[i - length + 1:i + 1]) / length

        # Step 3: Measure divergence (price - cycle component)
        div = priceList[i] - cycle_component

        # Step 4: Smooth result (EMA-style smoothing)
        smooth = 0.2 * div + 0.8 * smooth
        acd.append(float(round(smooth, 6)))

    return acd


# 69. Phase Accumulation Cycle (simplified using FFT phase angle)
def phase_accumulation_cycle(priceList):
    """
    Ehlers' Phase Accumulation Dominant Cycle Estimator.

    Tracks phase change to determine the cycle length.
    More responsive than Hilbert transform in some cases.

    Parameters:
        priceList (list): Price input (e.g., hl2).

    Returns:
        list: Estimated dominant cycle lengths as floats.
    """
    cycle_lengths = []
    prev_phase = 0.0
    prev_i = prev_q = 0.0
    smooth_price = 0.0

    for i in range(len(priceList)):
        # Smooth input using 3-bar WMA
        if i < 2:
            cycle_lengths.append(0.0)
            continue

        smooth_price = (priceList[i] + 2 * priceList[i - 1] + priceList[i - 2]) / 4

        # In-phase & Quadrature approximations
        i_part = smooth_price - priceList[i - 2]
        q_part = 2 * (priceList[i - 1] - priceList[i - 2])

        # Avoid division by 0
        if abs(q_part) < 1e-10:
            q_part = 1e-10

        # Instantaneous phase angle (in radians)
        phase = np.arctan(i_part / q_part)

        # Unwrap phase
        if phase < prev_phase:
            delta = 2 * np.pi + phase - prev_phase
        else:
            delta = phase - prev_phase

        prev_phase = phase

        # Convert to bars per cycle
        if delta != 0:
            cycle_length = (2 * np.pi) / delta
        else:
            cycle_length = 0.0

        cycle_lengths.append(float(round(cycle_length, 2)))

    return cycle_lengths


# 70. Inverse Fisher Transform
def inverse_fisher_transform(series):
    """
    Inverse Fisher Transform.

    Sharpens turning points of a bounded oscillator.
    Usually used with normalized indicators like RSI or momentum.

    Parameters:
        series (list): Input values between -1 and +1.

    Returns:
        list: Transformed output in range -1 to +1.
    """

    result = []
    for val in series:
        # Clip to avoid instability
        x = max(min(val, 0.999), -0.999)
        inv = (np.exp(2 * x) - 1) / (np.exp(2 * x) + 1)
        result.append(float(round(inv, 6)))

    return result

# These are some additional indcators there by dont follow the number order 
# --- Expert-Level Cycle Indicators ---

# --- 1. MAMA & FAMA ---
def mama_fama(priceList, fast_limit=0.5, slow_limit=0.05):
    mama = []
    fama = []
    phase_prev = 0.0
    mama_val = fama_val = 0.0

    for i in range(len(priceList)):
        if i < 6:
            mama.append(0.0)
            fama.append(0.0)
            continue

        # Estimate phase via arctangent of in-phase and quadrature components
        delta_phase = 0.0
        if i >= 6:
            detrender = (0.0962 * priceList[i] + 0.5769 * priceList[i-2] - 0.5769 * priceList[i-4] - 0.0962 * priceList[i-6])
            i_part = detrender
            q_part = priceList[i-3] - priceList[i-6]
            if abs(q_part) < 1e-10: q_part = 1e-10
            phase = np.arctan(i_part / q_part)
            delta_phase = phase - phase_prev
            phase_prev = phase

        alpha = fast_limit / abs(delta_phase) if abs(delta_phase) > 1e-5 else slow_limit
        alpha = max(min(alpha, fast_limit), slow_limit)

        mama_val = alpha * priceList[i] + (1 - alpha) * mama_val
        fama_val = 0.5 * alpha * mama_val + (1 - 0.5 * alpha) * fama_val

        mama.append(float(round(mama_val, 6)))
        fama.append(float(round(fama_val, 6)))

    return mama, fama


# --- 2. Super Smoother Filter ---
def super_smoother(priceList, period):
    a1 = np.exp(-1.414 * np.pi / period)
    b1 = 2 * a1 * np.cos(1.414 * np.pi / period)
    c2 = b1
    c3 = -a1 ** 2
    c1 = 1 - c2 - c3
    filt = []
    for i in range(len(priceList)):
        if i < 2:
            filt.append(priceList[i])
            continue
        val = c1 * (priceList[i] + priceList[i - 1]) / 2 + c2 * filt[i - 1] + c3 * filt[i - 2]
        filt.append(float(round(val, 6)))
    return filt


# --- 3. Roofing Filter (Bandpass-like) ---
def roofing_filter(priceList):
    hp = []
    filt = []
    for i in range(len(priceList)):
        if i < 2:
            hp.append(0.0)
            filt.append(priceList[i])
            continue
        hp_val = 0.5 * (priceList[i] - priceList[i - 2]) + 0.995 * hp[i - 1]
        hp.append(hp_val)

        filt_val = 0.5 * (hp[i] + hp[i - 1])
        filt.append(float(round(filt_val, 6)))
    return filt


# --- 4. Center of Gravity (COG) ---
def center_of_gravity(priceList, length=10):
    cog_vals = []
    for i in range(len(priceList)):
        if i < length:
            cog_vals.append(0.0)
            continue
        num = sum(j * priceList[i - j] for j in range(length))
        denom = sum(priceList[i - j] for j in range(length))
        cog = num / denom if denom != 0 else 0.0
        cog_vals.append(float(round(cog, 6)))
    return cog_vals


# --- 5. Bandpass Filter ---
def bandpass_filter(priceList, period, bandwidth=0.3):
    alpha = (np.cos(2 * np.pi / period) + np.sin(2 * np.pi / period) - 1) / np.cos(2 * np.pi * bandwidth / period)
    bp = []
    for i in range(len(priceList)):
        if i < 2:
            bp.append(0.0)
            continue
        val = 0.5 * (1 - alpha) * (priceList[i] - priceList[i - 2]) + alpha * bp[i - 1]
        bp.append(float(round(val, 6)))
    return bp


# --- 6. DC-Based RSI ---
def dc_based_rsi(priceList, cycleList):
    rsi_vals = []
    for i in range(len(priceList)):
        cycle_len = int(round(cycleList[i]))
        if cycle_len < 2 or i < cycle_len:
            rsi_vals.append(0.0)
            continue
        gains = [max(priceList[j] - priceList[j - 1], 0) for j in range(i - cycle_len + 1, i + 1)]
        losses = [max(priceList[j - 1] - priceList[j], 0) for j in range(i - cycle_len + 1, i + 1)]
        avg_gain = sum(gains) / cycle_len
        avg_loss = sum(losses) / cycle_len
        rs = avg_gain / avg_loss if avg_loss != 0 else 0.0
        rsi = 100 - (100 / (1 + rs))
        rsi_vals.append(float(round(rsi, 6)))
    return rsi_vals


# --- 7. Cyber Cycle ---
def cyber_cycle(priceList, alpha=0.07):
    cycle = []
    for i in range(len(priceList)):
        if i < 2:
            cycle.append(priceList[i])
            continue
        val = (1 - 0.5 * alpha) ** 2 * (priceList[i] - 2 * priceList[i - 1] + priceList[i - 2]) + \
              2 * (1 - alpha) * cycle[i - 1] - (1 - alpha) ** 2 * cycle[i - 2]
        cycle.append(float(round(val, 6)))
    return cycle
