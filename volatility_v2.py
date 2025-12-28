"""
PythonPine Volatility V2 - High-Performance Vectorized Indicators
"""

import numpy as np
import pandas as pd
from typing import Union, Tuple
from .utils import (
    ensure_array, rolling_mean, rolling_std, rolling_max, rolling_min,
    ewm_mean, wilder_smooth, true_range, pad_nan, shift
)


ArrayLike = Union[np.ndarray, pd.Series, list]


def atr(high: ArrayLike, low: ArrayLike, close: ArrayLike,
        length: int = 14) -> np.ndarray:
    """
    Average True Range (ATR) - Vectorized with Wilder's smoothing.
    
    Uses RMA (Wilder's smoothing) for Pine Script parity.
    
    Warmup: length bars
    """
    h = ensure_array(high)
    l = ensure_array(low)
    c = ensure_array(close)
    
    tr = true_range(h, l, c)
    atr_vals = wilder_smooth(tr, length)
    
    return pad_nan(atr_vals, length)


def bollinger_bands(source: ArrayLike, length: int = 20, mult: float = 2.0
                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Bollinger Bands - Vectorized.
    
    Returns:
        (upper, lower, middle)
    """
    close = ensure_array(source)
    
    middle = rolling_mean(close, length)
    std = rolling_std(close, length, ddof=0)
    
    upper = middle + mult * std
    lower = middle - mult * std
    
    return (
        pad_nan(upper, length),
        pad_nan(lower, length),
        pad_nan(middle, length)
    )


def keltner_channel(high: ArrayLike, low: ArrayLike, close: ArrayLike,
                    length: int = 20, mult: float = 2.0
                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Keltner Channel - Vectorized.
    
    Uses EMA for middle band and ATR for bands.
    
    Returns:
        (upper, lower, middle)
    """
    h = ensure_array(high)
    l = ensure_array(low)
    c = ensure_array(close)
    
    tp = (h + l + c) / 3
    middle = ewm_mean(tp, length)
    atr_vals = atr(h, l, c, length)
    
    upper = middle + mult * atr_vals
    lower = middle - mult * atr_vals
    
    return upper, lower, middle


def donchian_channel(high: ArrayLike, low: ArrayLike, length: int = 20
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Donchian Channel - Vectorized.
    
    Returns:
        (upper, lower, middle)
    """
    h = ensure_array(high)
    l = ensure_array(low)
    
    upper = rolling_max(h, length)
    lower = rolling_min(l, length)
    middle = (upper + lower) / 2
    
    return (
        pad_nan(upper, length),
        pad_nan(lower, length),
        pad_nan(middle, length)
    )


def bollinger_bandwidth(source: ArrayLike, length: int = 20, mult: float = 2.0
                        ) -> np.ndarray:
    """Bollinger Bandwidth - Vectorized."""
    upper, lower, middle = bollinger_bands(source, length, mult)
    middle = np.where(middle == 0, 1e-10, middle)
    return (upper - lower) / middle


def bollinger_percent_b(source: ArrayLike, length: int = 20, mult: float = 2.0
                        ) -> np.ndarray:
    """Bollinger %B - Vectorized."""
    close = ensure_array(source)
    upper, lower, _ = bollinger_bands(close, length, mult)
    
    denom = upper - lower
    denom = np.where(denom == 0, 1e-10, denom)
    
    return (close - lower) / denom


def historical_volatility(source: ArrayLike, length: int = 20,
                          annualize: bool = True) -> np.ndarray:
    """
    Historical Volatility (HV) - Vectorized.
    
    Returns annualized volatility by default.
    """
    close = ensure_array(source)
    
    log_returns = np.log(close[1:] / close[:-1])
    log_returns = np.concatenate([[np.nan], log_returns])
    
    hv = rolling_std(log_returns, length, ddof=1)
    
    if annualize:
        hv = hv * np.sqrt(252)
    
    return pad_nan(hv, length)


def normalized_atr(high: ArrayLike, low: ArrayLike, close: ArrayLike,
                   length: int = 14) -> np.ndarray:
    """NATR (Normalized ATR) - Vectorized."""
    c = ensure_array(close)
    atr_vals = atr(high, low, close, length)
    
    c = np.where(c == 0, 1e-10, c)
    return 100 * atr_vals / c


def ulcer_index(source: ArrayLike, length: int = 14) -> np.ndarray:
    """Ulcer Index - Vectorized."""
    close = ensure_array(source)
    
    max_close = rolling_max(close, length)
    max_close = np.where(max_close == 0, 1e-10, max_close)
    
    drawdown_pct = 100 * (close - max_close) / max_close
    squared_dd = drawdown_pct ** 2
    
    mean_sq = rolling_mean(squared_dd, length)
    ui = np.sqrt(mean_sq)
    
    return pad_nan(ui, length)


def chaikin_volatility(high: ArrayLike, low: ArrayLike, length: int = 10
                       ) -> np.ndarray:
    """Chaikin Volatility - Vectorized."""
    h = ensure_array(high)
    l = ensure_array(low)
    
    hl_diff = h - l
    ema_short = ewm_mean(hl_diff, length)
    shifted = shift(ema_short, length)
    
    shifted = np.where(shifted == 0, 1e-10, shifted)
    
    cv = 100 * (ema_short - shifted) / shifted
    return pad_nan(cv, length * 2)


def mass_index(high: ArrayLike, low: ArrayLike, ema_length: int = 9,
               sum_length: int = 25) -> np.ndarray:
    """Mass Index - Vectorized."""
    h = ensure_array(high)
    l = ensure_array(low)
    
    hl_range = h - l
    ema1 = ewm_mean(hl_range, ema_length)
    ema2 = ewm_mean(ema1, ema_length)
    
    ema2 = np.where(ema2 == 0, 1e-10, ema2)
    ratio = ema1 / ema2
    
    s = pd.Series(ratio)
    mass = s.rolling(sum_length, min_periods=sum_length).sum().values
    
    return pad_nan(mass, sum_length + ema_length)


def parkinson_volatility(high: ArrayLike, low: ArrayLike, length: int = 10
                         ) -> np.ndarray:
    """Parkinson Volatility Estimator - Vectorized."""
    h = ensure_array(high)
    l = ensure_array(low)
    
    log_hl = np.log(h / l) ** 2
    factor = 1 / (4 * np.log(2))
    
    mean_log = rolling_mean(log_hl, length)
    pv = np.sqrt(factor * mean_log)
    
    return pad_nan(pv, length)


def garman_klass_volatility(open_: ArrayLike, high: ArrayLike, low: ArrayLike,
                            close: ArrayLike, length: int = 10) -> np.ndarray:
    """Garman-Klass Volatility Estimator - Vectorized."""
    o = ensure_array(open_)
    h = ensure_array(high)
    l = ensure_array(low)
    c = ensure_array(close)
    
    # Handle zeros
    o = np.where(o == 0, 1e-10, o)
    l = np.where(l == 0, 1e-10, l)
    
    term1 = 0.5 * np.log(h / l) ** 2
    term2 = (2 * np.log(2) - 1) * np.log(c / o) ** 2
    
    gk = term1 - term2
    mean_gk = rolling_mean(gk, length)
    
    return pad_nan(np.sqrt(mean_gk), length)
