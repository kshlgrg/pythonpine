"""
PythonPine Utils - Core vectorized math helpers
Centralized utilities to avoid code duplication and maximize performance
"""

import numpy as np
import pandas as pd
from typing import Union, Optional

# Type alias for flexible input
ArrayLike = Union[np.ndarray, pd.Series, list]


def ensure_array(data: ArrayLike) -> np.ndarray:
    """Convert input to NumPy array if needed."""
    if isinstance(data, np.ndarray):
        return data.astype(np.float64)
    elif isinstance(data, pd.Series):
        return data.values.astype(np.float64)
    else:
        return np.array(data, dtype=np.float64)


def pad_nan(result: np.ndarray, warmup: int) -> np.ndarray:
    """Replace first `warmup` values with NaN for stability."""
    if warmup > 0 and len(result) > warmup:
        result[:warmup] = np.nan
    return result


def rolling_max(arr: np.ndarray, window: int) -> np.ndarray:
    """Vectorized rolling maximum."""
    df = pd.Series(arr)
    return df.rolling(window=window, min_periods=1).max().values


def rolling_min(arr: np.ndarray, window: int) -> np.ndarray:
    """Vectorized rolling minimum."""
    df = pd.Series(arr)
    return df.rolling(window=window, min_periods=1).min().values


def rolling_sum(arr: np.ndarray, window: int) -> np.ndarray:
    """Vectorized rolling sum."""
    df = pd.Series(arr)
    return df.rolling(window=window, min_periods=1).sum().values


def rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    """Vectorized rolling mean (SMA)."""
    df = pd.Series(arr)
    return df.rolling(window=window, min_periods=window).mean().values


def rolling_std(arr: np.ndarray, window: int, ddof: int = 0) -> np.ndarray:
    """Vectorized rolling standard deviation."""
    df = pd.Series(arr)
    return df.rolling(window=window, min_periods=window).std(ddof=ddof).values


def ewm_mean(arr: np.ndarray, span: int, adjust: bool = False) -> np.ndarray:
    """Vectorized Exponential Weighted Moving Average."""
    df = pd.Series(arr)
    return df.ewm(span=span, adjust=adjust).mean().values


def ewm_alpha(arr: np.ndarray, alpha: float, adjust: bool = False) -> np.ndarray:
    """EWM with explicit alpha."""
    df = pd.Series(arr)
    return df.ewm(alpha=alpha, adjust=adjust).mean().values


def wilder_smooth(arr: np.ndarray, period: int) -> np.ndarray:
    """
    Wilder's smoothing method (RMA) - used in RSI, ATR, ADX.
    Equivalent to: smoothed[i] = (smoothed[i-1] * (period-1) + arr[i]) / period
    """
    alpha = 1.0 / period
    return ewm_alpha(arr, alpha, adjust=False)


def true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    """Vectorized True Range calculation."""
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    
    tr1 = high - low
    tr2 = np.abs(high - prev_close)
    tr3 = np.abs(low - prev_close)
    
    return np.maximum(np.maximum(tr1, tr2), tr3)


def diff(arr: np.ndarray, periods: int = 1) -> np.ndarray:
    """Vectorized difference."""
    result = np.empty_like(arr)
    result[:periods] = np.nan
    result[periods:] = arr[periods:] - arr[:-periods]
    return result


def shift(arr: np.ndarray, periods: int = 1) -> np.ndarray:
    """Shift array by n periods."""
    result = np.empty_like(arr)
    if periods > 0:
        result[:periods] = np.nan
        result[periods:] = arr[:-periods]
    elif periods < 0:
        result[periods:] = np.nan
        result[:periods] = arr[-periods:]
    else:
        result = arr.copy()
    return result


def cross_above(arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
    """Detect when arr1 crosses above arr2."""
    above = arr1 > arr2
    below_prev = shift(arr1, 1) <= shift(arr2, 1)
    result = above & below_prev
    result[0] = False
    return result


def cross_below(arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
    """Detect when arr1 crosses below arr2."""
    below = arr1 < arr2
    above_prev = shift(arr1, 1) >= shift(arr2, 1)
    result = below & above_prev
    result[0] = False
    return result


def highest(arr: np.ndarray, length: int) -> np.ndarray:
    """Highest value over last `length` bars."""
    return rolling_max(arr, length)


def lowest(arr: np.ndarray, length: int) -> np.ndarray:
    """Lowest value over last `length` bars."""
    return rolling_min(arr, length)


# Warmup period constants for indicators
WARMUP_PERIODS = {
    'sma': lambda p: p,
    'ema': lambda p: p * 2,
    'rsi': lambda p: p + 1,
    'atr': lambda p: p,
    'macd': lambda fast, slow, signal: slow + signal,
    'bollinger': lambda p: p,
    'adx': lambda p: p * 2,
}


def get_warmup(indicator: str, *args) -> int:
    """Get recommended warmup period for an indicator."""
    if indicator in WARMUP_PERIODS:
        return WARMUP_PERIODS[indicator](*args)
    return 14  # Default
