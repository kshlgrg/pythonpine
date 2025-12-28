"""
PythonPine Trend V2 - High-Performance Vectorized Indicators
"""

import numpy as np
import pandas as pd
from typing import Union, Tuple, List
from .utils import (
    ensure_array, rolling_mean, rolling_max, rolling_min,
    ewm_mean, pad_nan, shift
)


ArrayLike = Union[np.ndarray, pd.Series, list]


def sma(source: ArrayLike, length: int = 14) -> np.ndarray:
    """Simple Moving Average - Vectorized."""
    close = ensure_array(source)
    return rolling_mean(close, length)


def ema(source: ArrayLike, length: int = 14) -> np.ndarray:
    """Exponential Moving Average - Vectorized."""
    close = ensure_array(source)
    return ewm_mean(close, length)


def wma(source: ArrayLike, length: int = 14) -> np.ndarray:
    """Weighted Moving Average - Vectorized."""
    close = ensure_array(source)
    weights = np.arange(1, length + 1)
    
    s = pd.Series(close)
    wma_vals = s.rolling(length).apply(
        lambda x: np.dot(x, weights) / weights.sum(),
        raw=True
    ).values
    
    return wma_vals


def dema(source: ArrayLike, length: int = 14) -> np.ndarray:
    """Double EMA - Vectorized."""
    close = ensure_array(source)
    ema1 = ewm_mean(close, length)
    ema2 = ewm_mean(ema1, length)
    return 2 * ema1 - ema2


def tema(source: ArrayLike, length: int = 14) -> np.ndarray:
    """Triple EMA - Vectorized."""
    close = ensure_array(source)
    ema1 = ewm_mean(close, length)
    ema2 = ewm_mean(ema1, length)
    ema3 = ewm_mean(ema2, length)
    return 3 * ema1 - 3 * ema2 + ema3


def hma(source: ArrayLike, length: int = 14) -> np.ndarray:
    """Hull Moving Average - Vectorized."""
    close = ensure_array(source)
    half_length = max(1, length // 2)
    sqrt_length = max(1, int(np.sqrt(length)))
    
    wma_half = wma(close, half_length)
    wma_full = wma(close, length)
    
    diff = 2 * wma_half - wma_full
    return wma(diff, sqrt_length)


def vwma(source: ArrayLike, volume: ArrayLike, length: int = 14) -> np.ndarray:
    """Volume Weighted Moving Average - Vectorized."""
    close = ensure_array(source)
    vol = ensure_array(volume)
    
    pv = close * vol
    
    sum_pv = rolling_mean(pv, length) * length
    sum_vol = rolling_mean(vol, length) * length
    
    sum_vol = np.where(sum_vol == 0, 1e-10, sum_vol)
    return sum_pv / sum_vol


def kama(source: ArrayLike, length: int = 10, fast: int = 2, slow: int = 30
         ) -> np.ndarray:
    """Kaufman Adaptive Moving Average - Vectorized."""
    close = ensure_array(source)
    n = len(close)
    
    fast_sc = 2 / (fast + 1)
    slow_sc = 2 / (slow + 1)
    
    kama_vals = np.empty(n)
    kama_vals[0] = close[0]
    
    for i in range(1, n):
        if i < length:
            kama_vals[i] = close[i]
        else:
            change = abs(close[i] - close[i - length])
            volatility = np.sum(np.abs(np.diff(close[i-length:i+1])))
            volatility = max(volatility, 1e-10)
            
            er = change / volatility
            sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
            kama_vals[i] = kama_vals[i-1] + sc * (close[i] - kama_vals[i-1])
    
    return pad_nan(kama_vals, length)


def supertrend(high: ArrayLike, low: ArrayLike, close: ArrayLike,
               length: int = 10, mult: float = 3.0
               ) -> Tuple[np.ndarray, np.ndarray]:
    """
    SuperTrend - Vectorized.
    
    Returns:
        (trend_direction, supertrend_line)
        trend_direction: 1 = uptrend, -1 = downtrend
    """
    from .volatility_v2 import atr
    
    h = ensure_array(high)
    l = ensure_array(low)
    c = ensure_array(close)
    
    atr_vals = atr(h, l, c, length)
    hl2 = (h + l) / 2
    
    upper = hl2 + mult * atr_vals
    lower = hl2 - mult * atr_vals
    
    n = len(c)
    trend = np.ones(n)
    st_line = np.empty(n)
    st_line[0] = lower[0]
    
    for i in range(1, n):
        if c[i] > upper[i-1]:
            trend[i] = 1
        elif c[i] < lower[i-1]:
            trend[i] = -1
        else:
            trend[i] = trend[i-1]
        
        if trend[i] == 1:
            st_line[i] = max(lower[i], st_line[i-1]) if trend[i-1] == 1 else lower[i]
        else:
            st_line[i] = min(upper[i], st_line[i-1]) if trend[i-1] == -1 else upper[i]
    
    return trend.astype(int), pad_nan(st_line, length)


def aroon(high: ArrayLike, low: ArrayLike, length: int = 14
          ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Aroon Indicator - Vectorized.
    
    Returns:
        (aroon_up, aroon_down)
    """
    h = ensure_array(high)
    l = ensure_array(low)
    n = len(h)
    
    aroon_up = np.full(n, np.nan)
    aroon_down = np.full(n, np.nan)
    
    for i in range(length, n):
        window_high = h[i-length+1:i+1]
        window_low = l[i-length+1:i+1]
        
        bars_since_high = length - 1 - np.argmax(window_high)
        bars_since_low = length - 1 - np.argmin(window_low)
        
        aroon_up[i] = 100 * (length - bars_since_high) / length
        aroon_down[i] = 100 * (length - bars_since_low) / length
    
    return aroon_up, aroon_down


def ichimoku(high: ArrayLike, low: ArrayLike, close: ArrayLike,
             tenkan: int = 9, kijun: int = 26, senkou: int = 52
             ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Ichimoku Cloud - Vectorized.
    
    Returns:
        (conversion, base, span_a, span_b, lagging)
    """
    h = ensure_array(high)
    l = ensure_array(low)
    c = ensure_array(close)
    
    conversion = (rolling_max(h, tenkan) + rolling_min(l, tenkan)) / 2
    base = (rolling_max(h, kijun) + rolling_min(l, kijun)) / 2
    span_a = (conversion + base) / 2
    span_b = (rolling_max(h, senkou) + rolling_min(l, senkou)) / 2
    lagging = shift(c, -kijun)  # Shifted forward
    
    return conversion, base, span_a, span_b, lagging


def linear_regression(source: ArrayLike, length: int = 14) -> np.ndarray:
    """Linear Regression Line - Vectorized."""
    close = ensure_array(source)
    
    s = pd.Series(close)
    x = np.arange(length)
    
    def lr_value(y):
        n = len(y)
        x_local = np.arange(n)
        slope = (n * np.sum(x_local * y) - np.sum(x_local) * np.sum(y)) / \
                (n * np.sum(x_local**2) - np.sum(x_local)**2 + 1e-10)
        intercept = (np.sum(y) - slope * np.sum(x_local)) / n
        return slope * (n - 1) + intercept
    
    lr = s.rolling(length).apply(lr_value, raw=True).values
    return lr


def parabolic_sar(high: ArrayLike, low: ArrayLike,
                  af_step: float = 0.02, af_max: float = 0.2
                  ) -> np.ndarray:
    """Parabolic SAR - Optimized."""
    h = ensure_array(high)
    l = ensure_array(low)
    n = len(h)
    
    sar = np.empty(n)
    trend = 1
    af = af_step
    ep = h[0]
    sar[0] = l[0]
    
    for i in range(1, n):
        prev_sar = sar[i-1]
        
        if trend == 1:
            sar[i] = prev_sar + af * (ep - prev_sar)
            sar[i] = min(sar[i], l[i-1])
            if i > 1:
                sar[i] = min(sar[i], l[i-2])
            
            if l[i] < sar[i]:
                trend = -1
                sar[i] = ep
                ep = l[i]
                af = af_step
            else:
                if h[i] > ep:
                    ep = h[i]
                    af = min(af + af_step, af_max)
        else:
            sar[i] = prev_sar + af * (ep - prev_sar)
            sar[i] = max(sar[i], h[i-1])
            if i > 1:
                sar[i] = max(sar[i], h[i-2])
            
            if h[i] > sar[i]:
                trend = 1
                sar[i] = ep
                ep = h[i]
                af = af_step
            else:
                if l[i] < ep:
                    ep = l[i]
                    af = min(af + af_step, af_max)
    
    return sar


def zero_lag_ema(source: ArrayLike, length: int = 14) -> np.ndarray:
    """Zero-Lag EMA (ZLEMA) - Vectorized."""
    close = ensure_array(source)
    ema1 = ewm_mean(close, length)
    ema2 = ewm_mean(ema1, length)
    return 2 * ema1 - ema2


def mcginley_dynamic(source: ArrayLike, length: int = 14) -> np.ndarray:
    """McGinley Dynamic - Optimized."""
    close = ensure_array(source)
    n = len(close)
    
    md = np.empty(n)
    md[0] = close[0]
    
    for i in range(1, n):
        ratio = close[i] / md[i-1] if md[i-1] != 0 else 1
        md[i] = md[i-1] + (close[i] - md[i-1]) / (length * ratio**4)
    
    return md
