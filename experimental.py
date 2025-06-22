import numpy as np
from numpy.linalg import lstsq
from scipy.signal import lfilter
from math import log, sqrt

# --- 91. Fractal Dimension Index (Katz Method) ---
def fractal_dimension(priceList, period=10):
    fdi = []
    for i in range(len(priceList)):
        if i < period:
            fdi.append(0.0)
        else:
            window = priceList[i - period + 1:i + 1]
            L = sum(abs(window[j] - window[j - 1]) for j in range(1, period))
            d = max(abs(window[-1] - window[0]), 1e-10)
            n = period
            fdi.append(math.log(n) / (math.log(n) + math.log(d / L)))
    return fdi

# --- 92. Kalman Filter Slope ---
def kalman_filter_slope(priceList, R=0.01, Q=0.001):
    estimate = priceList[0]
    error = 1.0
    result = [estimate]
    for price in priceList[1:]:
        error += Q
        K = error / (error + R)
        estimate += K * (price - estimate)
        error *= (1 - K)
        result.append(estimate)
    slope = [0.0] + [result[i] - result[i - 1] for i in range(1, len(result))]
    return slope

# --- 93. Hurst Exponent ---
def hurst_exponent(priceList, max_lag=20):
    lags = range(2, max_lag)
    tau = [np.std(np.subtract(priceList[lag:], priceList[:-lag])) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return [poly[0]] * len(priceList)

# --- 94. Shannon Entropy ---
def shannon_entropy(priceList, period=20):
    entropy = []
    for i in range(len(priceList)):
        if i < period - 1:
            entropy.append(0.0)
        else:
            window = priceList[i - period + 1:i + 1]
            prob = np.histogram(window, bins='auto', density=True)[0]
            prob = prob[prob > 0]
            entropy.append(-np.sum(prob * np.log2(prob)))
    return entropy

# --- 95. KL Divergence (Against Normal Distribution) ---
def kl_divergence(priceList, period=20):
    kl = []
    for i in range(len(priceList)):
        if i < period:
            kl.append(0.0)
        else:
            window = priceList[i - period + 1:i + 1]
            p_hist, _ = np.histogram(window, bins=20, density=True)
            q = stats.norm.pdf(np.linspace(min(window), max(window), len(p_hist)))
            p_hist = np.where(p_hist == 0, 1e-10, p_hist)
            q = np.where(q == 0, 1e-10, q)
            kl.append(np.sum(p_hist * np.log(p_hist / q)))
    return kl

# --- 96. Time Series Forecast (TSF) ---
def tsf(priceList, period=14):
    forecast = []
    for i in range(len(priceList)):
        if i < period:
            forecast.append(0.0)
        else:
            y = priceList[i - period + 1:i + 1]
            x = np.arange(period)
            slope, intercept = np.polyfit(x, y, 1)
            forecast.append(intercept + slope * (period - 1))
    return forecast

# --- 97. Ehlers Roofing Filter ---
def roofing_filter(priceList):
    hp = [0.0] * len(priceList)
    roof = [0.0] * len(priceList)
    for i in range(6, len(priceList)):
        hp[i] = (0.5 * (priceList[i] - priceList[i - 1]) +
                 0.99 * hp[i - 1])
        roof[i] = (0.5 * hp[i] + 0.5 * roof[i - 1])
    return roof

# --- 98. Smoothed Heikin Ashi Oscillator ---
def smoothed_heikin_ashi(closeList, openList, period=10):
    ha_close = [(o + h + l + c) / 4 for o, h, l, c in zip(openList, closeList, closeList, closeList)]
    ha_smoothed = np.convolve(ha_close, np.ones(period)/period, mode='same')
    return ha_smoothed.tolist()

# --- 99. Neural Indicator Score Placeholder ---
def neural_indicator_score(input_vector):
    # This would be computed by your AI model externally.
    return [0.5] * len(input_vector)
