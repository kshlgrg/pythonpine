"""
PythonPine Normalize - ML Alpha Factor Preprocessing
Convert raw indicator values into ML-ready features
"""

import numpy as np
import pandas as pd
from typing import Union, Optional
from .utils import ensure_array, rolling_mean, rolling_std


ArrayLike = Union[np.ndarray, pd.Series, list]


def zscore(data: ArrayLike, window: int = 20) -> np.ndarray:
    """
    Rolling Z-Score Normalization.
    
    Formula: (value - rolling_mean) / rolling_std
    
    Returns values centered around 0 with unit variance.
    Critical for Neural Networks that expect standardized inputs.
    
    Args:
        data: Price or indicator series
        window: Lookback period for mean/std calculation
    
    Returns:
        Z-score normalized array
    """
    arr = ensure_array(data)
    mean = rolling_mean(arr, window)
    std = rolling_std(arr, window)
    
    # Avoid division by zero
    std = np.where(std == 0, 1e-10, std)
    
    zscore = (arr - mean) / std
    zscore[:window] = np.nan  # Warmup period
    
    return zscore


def percentile_rank(data: ArrayLike, window: int = 100) -> np.ndarray:
    """
    Rolling Percentile Rank.
    
    Returns where the current value ranks (0-100) relative to the last N bars.
    Useful for mean-reversion signals.
    
    Args:
        data: Price or indicator series
        window: Lookback period
    
    Returns:
        Percentile rank (0-100 scale)
    """
    arr = ensure_array(data)
    n = len(arr)
    result = np.full(n, np.nan)
    
    for i in range(window, n):
        window_data = arr[i - window:i + 1]
        rank = np.sum(window_data < arr[i]) / window * 100
        result[i] = rank
    
    return result


def percentile_rank_vectorized(data: ArrayLike, window: int = 100) -> np.ndarray:
    """
    Faster vectorized percentile rank using pandas.
    """
    s = pd.Series(ensure_array(data))
    return s.rolling(window).apply(
        lambda x: (x < x.iloc[-1]).sum() / len(x) * 100,
        raw=False
    ).values


def log_returns(data: ArrayLike) -> np.ndarray:
    """
    Log Returns - makes price data stationary.
    
    Formula: ln(price[t] / price[t-1])
    
    Essential for:
    - Neural networks (removes price level bias)
    - Time series models (ARIMA, GARCH)
    - Volatility modeling
    """
    arr = ensure_array(data)
    result = np.log(arr[1:] / arr[:-1])
    return np.concatenate([[np.nan], result])


def simple_returns(data: ArrayLike) -> np.ndarray:
    """
    Simple percentage returns.
    
    Formula: (price[t] - price[t-1]) / price[t-1]
    """
    arr = ensure_array(data)
    with np.errstate(divide='ignore', invalid='ignore'):
        returns = (arr[1:] - arr[:-1]) / arr[:-1]
    return np.concatenate([[np.nan], returns])


def fractional_diff(data: ArrayLike, d: float = 0.4, threshold: float = 1e-5) -> np.ndarray:
    """
    Fractional Differentiation.
    
    Balances stationarity vs memory preservation.
    d=0: Original series (non-stationary, full memory)
    d=1: First difference (stationary, no memory)
    d=0.3-0.5: Optimal for ML (mostly stationary, partial memory)
    
    Reference: López de Prado, "Advances in Financial Machine Learning"
    
    Args:
        data: Price series (typically log prices)
        d: Differentiation order (0 < d < 1)
        threshold: Weight cutoff for computational efficiency
    
    Returns:
        Fractionally differentiated series
    """
    arr = ensure_array(data)
    n = len(arr)
    
    # Calculate weights
    weights = [1.0]
    k = 1
    while abs(weights[-1]) > threshold:
        w = -weights[-1] * (d - k + 1) / k
        weights.append(w)
        k += 1
    
    weights = np.array(weights[::-1])  # Reverse for convolution
    w_len = len(weights)
    
    # Apply weights via convolution
    result = np.full(n, np.nan)
    for i in range(w_len - 1, n):
        result[i] = np.dot(weights, arr[i - w_len + 1:i + 1])
    
    return result


def minmax_scale(data: ArrayLike, window: int = 20) -> np.ndarray:
    """
    Rolling Min-Max Scaling to [0, 1] range.
    
    Formula: (value - min) / (max - min)
    
    Useful for indicators that need bounded outputs.
    """
    arr = ensure_array(data)
    s = pd.Series(arr)
    
    rolling_min = s.rolling(window, min_periods=window).min()
    rolling_max = s.rolling(window, min_periods=window).max()
    
    range_val = rolling_max - rolling_min
    range_val = range_val.replace(0, 1e-10)  # Avoid div by zero
    
    scaled = (s - rolling_min) / range_val
    return scaled.values


def robust_scale(data: ArrayLike, window: int = 20) -> np.ndarray:
    """
    Robust Scaling using median and IQR.
    
    Formula: (value - median) / IQR
    
    Less sensitive to outliers than z-score.
    """
    s = pd.Series(ensure_array(data))
    
    median = s.rolling(window, min_periods=window).median()
    q1 = s.rolling(window, min_periods=window).quantile(0.25)
    q3 = s.rolling(window, min_periods=window).quantile(0.75)
    iqr = q3 - q1
    iqr = iqr.replace(0, 1e-10)
    
    return ((s - median) / iqr).values


def cusum_filter(data: ArrayLike, threshold: float) -> np.ndarray:
    """
    CUSUM Filter - detects structural breaks.
    
    Returns 1 when cumulative sum exceeds threshold, -1 for negative threshold.
    Useful for event-driven ML labels.
    """
    arr = ensure_array(data)
    n = len(arr)
    
    s_pos = 0.0
    s_neg = 0.0
    result = np.zeros(n)
    
    for i in range(1, n):
        diff = arr[i] - arr[i-1]
        s_pos = max(0, s_pos + diff)
        s_neg = min(0, s_neg + diff)
        
        if s_pos > threshold:
            result[i] = 1
            s_pos = 0
        elif s_neg < -threshold:
            result[i] = -1
            s_neg = 0
    
    return result


def triple_barrier_labels(
    close: ArrayLike,
    take_profit: float,
    stop_loss: float,
    max_bars: int = 20
) -> np.ndarray:
    """
    Triple Barrier Labeling for ML.
    
    Labels each bar based on future outcome:
    +1: Hit take profit first
    -1: Hit stop loss first
     0: Neither hit within max_bars (hold)
    
    Args:
        close: Close prices
        take_profit: TP as fraction (e.g., 0.01 = 1%)
        stop_loss: SL as fraction
        max_bars: Maximum bars to look forward
    
    Returns:
        Labels array
    """
    arr = ensure_array(close)
    n = len(arr)
    labels = np.zeros(n)
    
    for i in range(n - max_bars):
        entry = arr[i]
        tp_level = entry * (1 + take_profit)
        sl_level = entry * (1 - stop_loss)
        
        for j in range(i + 1, min(i + max_bars + 1, n)):
            if arr[j] >= tp_level:
                labels[i] = 1
                break
            elif arr[j] <= sl_level:
                labels[i] = -1
                break
    
    # Mark last bars as unknown
    labels[-max_bars:] = np.nan
    
    return labels
