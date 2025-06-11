import numpy as np
from scipy.stats import skew, kurtosis, entropy, rankdata
from numpy.lib.stride_tricks import sliding_window_view

# 86. Z-score of Price
def z_score(price_array, window=20):
    mean = np.mean(price_array[-window:])
    std = np.std(price_array[-window:])
    return (price_array[-1] - mean) / std if std != 0 else 0.0

# 87. Rolling Mean/Std of Close
def rolling_mean_std(price_array, window=20):
    return np.mean(price_array[-window:]), np.std(price_array[-window:])

# 88. Skewness/Kurtosis (last 20 bars)
def skewness_kurtosis(price_array, window=20):
    segment = price_array[-window:]
    return skew(segment), kurtosis(segment)

# 89. Price Percentile Rank
def percentile_rank(price_array, window=20):
    sub_array = price_array[-window:]
    return rankdata(sub_array)[-1] / window

# 90. Median Absolute Deviation
def median_absolute_deviation(price_array, window=20):
    sub_array = price_array[-window:]
    median = np.median(sub_array)
    return np.median(np.abs(sub_array - median))
