import numpy as np
from numpy.linalg import lstsq
from scipy.signal import lfilter
from math import log, sqrt

# 91. Fractal Dimension Index (simplified)
def fractal_dimension(price_array, window=100):
    L = np.sum(np.abs(np.diff(price_array[-window:])))
    N = window
    return 1 + (np.log(L) / np.log(N)) if N > 1 else 1.0

# 92. Garman-Klass Volatility Estimator
def garman_klass_volatility(high, low, open_, close, window=20):
    log_hl = np.log(high[-window:] / low[-window:])
    log_co = np.log(close[-window:] / open_[-window:])
    return np.sqrt(0.5 * np.mean(log_hl ** 2) - (2 * log(2) - 1) * np.mean(log_co ** 2))

# 93. Kalman Filter Slope
def kalman_filter_slope(price_array, q=1e-5, r=0.01):
    n = len(price_array)
    xhat = np.zeros(n)
    P = np.zeros(n)
    xhat[0] = price_array[0]
    P[0] = 1.0
    for k in range(1, n):
        xhat_minus = xhat[k - 1]
        P_minus = P[k - 1] + q
        K = P_minus / (P_minus + r)
        xhat[k] = xhat_minus + K * (price_array[k] - xhat_minus)
        P[k] = (1 - K) * P_minus
    return (xhat[-1] - xhat[-2])

# 94. Hurst Exponent
def hurst_exponent(price_array, max_lag=20):
    lags = range(2, max_lag)
    tau = [np.std(np.subtract(price_array[lag:], price_array[:-lag])) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0] * 2.0

# 95. Shannon Entropy of Price
def shannon_entropy(price_array, window=20, bins=10):
    hist, _ = np.histogram(price_array[-window:], bins=bins, density=True)
    hist = hist[hist > 0]
    return -np.sum(hist * np.log2(hist))

# 96. KL Divergence (Price diff from mean)
def kl_divergence(price_array, window=20):
    p = np.histogram(price_array[-window:], bins=10, density=True)[0] + 1e-8
    q = np.ones_like(p) / len(p)
    return np.sum(p * np.log(p / q))

# 97. Time Series Forecast (Linear Regression)
def tsf(price_array, window=20):
    y = price_array[-window:]
    x = np.arange(window)
    A = np.vstack([x, np.ones(window)]).T
    m, c = lstsq(A, y, rcond=None)[0]
    return m * window + c

# 98. Ehler’s Roofing Filter (Bandpass filter)
def ehlers_roofing(price_array):
    a1 = np.exp(-1.414 * np.pi / 10)
    b1 = 2 * a1 * np.cos(1.414 * np.pi / 10)
    c2 = b1
    c3 = -a1 ** 2
    c1 = 1 - c2 - c3
    filt = [0] * len(price_array)
    for i in range(2, len(price_array)):
        filt[i] = c1 * (price_array[i] - price_array[i - 2]) + c2 * filt[i - 1] + c3 * filt[i - 2]
    return filt[-1]

# 99. Smoothed Heikin Ashi Oscillator
def smoothed_heikin_ashi_osc(close, open_, high, low, alpha=0.2):
    ha_close = (open_ + high + low + close) / 4
    ha_open = [open_[0]]
    for i in range(1, len(close)):
        ha_open.append((ha_open[i-1] + ha_close[i-1]) / 2)
    osc = [ha_close[i] - ha_open[i] for i in range(len(close))]
    return np.mean(osc[-5:])

# 100. Neural Indicator Score (placeholder)
def neural_indicator_score(features_array, model=None):
    # Use a trained ML model on features_array; return predicted signal/score
    if model:
        return float(model.predict([features_array[-1]]))  # Assuming 1 sample
    return 0.0  # Default if model is None
