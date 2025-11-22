import numpy as np
from scipy.stats import skew, kurtosis, entropy
from numpy.lib.stride_tricks import sliding_window_view

# --- 86. Z-score of Price ---
def z_score(priceList, period=20):
    z = []
    for i in range(len(priceList)):
        if i < period - 1:
            z.append(0.0)
        else:
            window = priceList[i - period + 1:i + 1]
            mean = np.mean(window)
            std = np.std(window)
            z.append((priceList[i] - mean) / std if std != 0 else 0.0)
    return z

# --- 87. Rolling Mean and Std ---
def rolling_mean_std(priceList, period=20):
    means = []
    stds = []
    for i in range(len(priceList)):
        if i < period - 1:
            means.append(0.0)
            stds.append(0.0)
        else:
            window = priceList[i - period + 1:i + 1]
            means.append(np.mean(window))
            stds.append(np.std(window))
    return means, stds

# --- 88. Skewness and Kurtosis ---
def skewness_kurtosis(priceList, period=20):
    skew_vals = []
    kurt_vals = []
    for i in range(len(priceList)):
        if i < period - 1:
            skew_vals.append(0.0)
            kurt_vals.append(0.0)
        else:
            window = priceList[i - period + 1:i + 1]
            skew_vals.append(skew(window))
            kurt_vals.append(kurtosis(window))
    return skew_vals, kurt_vals

# --- 89. Price Percentile Rank ---
def percentile_rank(priceList, period=20):
    ranks = []
    for i in range(len(priceList)):
        if i < period - 1:
            ranks.append(0.0)
        else:
            window = priceList[i - period + 1:i + 1]
            # Using simple rank calculation
            rank = sum(1 for v in window if v < priceList[i]) / period
            ranks.append(rank)
    return ranks

# --- 90. Median Absolute Deviation (MAD) ---
def median_absolute_deviation(priceList, period=20):
    mad = []
    for i in range(len(priceList)):
        if i < period - 1:
            mad.append(0.0)
        else:
            window = priceList[i - period + 1:i + 1]
            median = np.median(window)
            mad.append(np.median([abs(x - median) for x in window]))
    return mad
