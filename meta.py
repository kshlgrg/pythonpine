# --- Meta/Composite Indicators ---

# --- 71. MA Crossover Signal Count ---
def ma_crossover_count(fast_ma, slow_ma):
    count = 0
    for i in range(1, len(fast_ma)):
        if fast_ma[i] > slow_ma[i] and fast_ma[i - 1] <= slow_ma[i - 1]:
            count += 1
    return count

# --- 72. Indicator Consensus Score ---
def indicator_consensus(*signals):
    return [sum(sig[i] for sig in signals) / len(signals) for i in range(len(signals[0]))]

# --- 73. Momentum + Volatility Composite ---
def momentum_volatility_score(momentum, volatility):
    return [m * v for m, v in zip(momentum, volatility)]

# --- 74. Trend Strength Score ---
def trend_strength_score(priceList, period=14):
    ts = []
    for i in range(len(priceList)):
        if i < period:
            ts.append(0.0)
        else:
            gains = [priceList[j] - priceList[j - 1] for j in range(i - period + 1, i + 1) if priceList[j] > priceList[j - 1]]
            ts.append(sum(gains) / period if gains else 0.0)
    return ts

# --- 75. MACD Histogram Angle ---
def macd_histogram_angle(macd_hist):
    angles = [0.0]
    for i in range(1, len(macd_hist)):
        angles.append(math.degrees(math.atan(macd_hist[i] - macd_hist[i - 1])))
    return angles

# --- 76. RSI Divergence Count (Simplified) ---
def rsi_divergence_count(priceList, rsiList, lookback=14):
    count = 0
    for i in range(lookback, len(priceList)):
        if priceList[i] > priceList[i - lookback] and rsiList[i] < rsiList[i - lookback]:
            count += 1
    return count

# --- 77. Volume Spike Flag ---
def volume_spike(volumeList, multiplier=2.0):
    avg_vol = np.mean(volumeList)
    return [1 if v > multiplier * avg_vol else 0 for v in volumeList]

# --- 78. Multi-Timeframe EMA Alignment (Stub) ---
def mtf_ema_alignment(ema_short, ema_mid, ema_long):
    return [1 if s > m > l else 0 for s, m, l in zip(ema_short, ema_mid, ema_long)]

# --- 79. Trend Reversal Likelihood ---
def trend_reversal_likelihood(priceList, rsiList):
    return [1 if rsi > 70 and priceList[i] < priceList[i - 1] else 0 for i, rsi in enumerate(rsiList) if i > 0]

# --- 80. Consolidation Detector (Std Dev Drop) ---
def consolidation_detector(priceList, period=20, threshold=0.01):
    stds = rolling_mean_std(priceList, period)[1]
    return [1 if std < threshold else 0 for std in stds]
