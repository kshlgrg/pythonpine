"""
PythonPine Momentum V2 - High-Performance Vectorized Indicators
All indicators use NumPy/Pandas for 100x+ speedup over loop-based versions
"""

import numpy as np
import pandas as pd
from typing import Union, Tuple
from .utils import (
    ensure_array, rolling_max, rolling_min, rolling_mean, rolling_sum,
    ewm_mean, wilder_smooth, pad_nan, highest, lowest, diff, shift
)


ArrayLike = Union[np.ndarray, pd.Series, list]


# ========================== CORE INDICATORS ==========================

def rsi(source: ArrayLike, length: int = 14) -> np.ndarray:
    """
    Relative Strength Index (RSI) - Vectorized.
    
    Uses Wilder's smoothing (RMA) for Pine Script parity.
    
    Warmup: length + 1 bars
    """
    close = ensure_array(source)
    delta = diff(close, 1)
    
    gains = np.where(delta > 0, delta, 0)
    losses = np.where(delta < 0, -delta, 0)
    
    # Handle NaN in first position
    gains[0] = 0
    losses[0] = 0
    
    # Wilder's smoothing (RMA)
    avg_gain = wilder_smooth(gains, length)
    avg_loss = wilder_smooth(losses, length)
    
    # Avoid division by zero
    avg_loss = np.where(avg_loss == 0, 1e-10, avg_loss)
    
    rs = avg_gain / avg_loss
    rsi_values = 100 - (100 / (1 + rs))
    
    return pad_nan(rsi_values, length)


def macd(source: ArrayLike, fast: int = 12, slow: int = 26, signal: int = 9
         ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    MACD (Moving Average Convergence Divergence) - Vectorized.
    
    Returns:
        (macd_line, signal_line, histogram)
    
    Warmup: slow + signal bars
    """
    close = ensure_array(source)
    
    fast_ema = ewm_mean(close, fast)
    slow_ema = ewm_mean(close, slow)
    
    macd_line = fast_ema - slow_ema
    signal_line = ewm_mean(macd_line, signal)
    histogram = macd_line - signal_line
    
    warmup = slow + signal
    return (
        pad_nan(macd_line, slow),
        pad_nan(signal_line, warmup),
        pad_nan(histogram, warmup)
    )


def stochastic(close: ArrayLike, high: ArrayLike, low: ArrayLike,
               length: int = 14, smooth_k: int = 3, smooth_d: int = 3
               ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Stochastic Oscillator - Vectorized.
    
    Returns:
        (k_line, d_line)
    """
    c = ensure_array(close)
    h = ensure_array(high)
    l = ensure_array(low)
    
    hh = rolling_max(h, length)
    ll = rolling_min(l, length)
    
    denom = hh - ll
    denom = np.where(denom == 0, 1e-10, denom)
    
    raw_k = 100 * (c - ll) / denom
    k = rolling_mean(raw_k, smooth_k)
    d = rolling_mean(k, smooth_d)
    
    return pad_nan(k, length), pad_nan(d, length + smooth_d)


def williams_r(close: ArrayLike, high: ArrayLike, low: ArrayLike,
               length: int = 14) -> np.ndarray:
    """Williams %R - Vectorized."""
    c = ensure_array(close)
    h = ensure_array(high)
    l = ensure_array(low)
    
    hh = rolling_max(h, length)
    ll = rolling_min(l, length)
    
    denom = hh - ll
    denom = np.where(denom == 0, 1e-10, denom)
    
    wr = -100 * (hh - c) / denom
    return pad_nan(wr, length)


def cci(close: ArrayLike, high: ArrayLike, low: ArrayLike,
        length: int = 20) -> np.ndarray:
    """
    Commodity Channel Index (CCI) - Vectorized.
    """
    c = ensure_array(close)
    h = ensure_array(high)
    l = ensure_array(low)
    
    tp = (h + l + c) / 3
    sma_tp = rolling_mean(tp, length)
    
    # Mean deviation
    s = pd.Series(tp)
    mean_dev = s.rolling(length).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True).values
    mean_dev = np.where(mean_dev == 0, 1e-10, mean_dev)
    
    cci_vals = (tp - sma_tp) / (0.015 * mean_dev)
    return pad_nan(cci_vals, length)


def roc(source: ArrayLike, length: int = 12) -> np.ndarray:
    """Rate of Change (ROC) - Vectorized."""
    close = ensure_array(source)
    prev = shift(close, length)
    prev = np.where(prev == 0, 1e-10, prev)
    
    roc_vals = 100 * (close - prev) / prev
    return pad_nan(roc_vals, length)


def momentum(source: ArrayLike, length: int = 10) -> np.ndarray:
    """Momentum Indicator - Vectorized."""
    close = ensure_array(source)
    prev = shift(close, length)
    return close - prev


def adx(high: ArrayLike, low: ArrayLike, close: ArrayLike,
        length: int = 14) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Average Directional Index (ADX) - Vectorized.
    
    Returns:
        (adx, plus_di, minus_di)
    
    Warmup: length * 2 bars
    """
    h = ensure_array(high)
    l = ensure_array(low)
    c = ensure_array(close)
    
    # True Range
    from .utils import true_range
    tr = true_range(h, l, c)
    
    # Directional Movement
    prev_h = shift(h, 1)
    prev_l = shift(l, 1)
    
    up_move = h - prev_h
    down_move = prev_l - l
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Wilder smoothing
    atr = wilder_smooth(tr, length)
    smooth_plus_dm = wilder_smooth(plus_dm, length)
    smooth_minus_dm = wilder_smooth(minus_dm, length)
    
    # DI calculations
    atr = np.where(atr == 0, 1e-10, atr)
    plus_di = 100 * smooth_plus_dm / atr
    minus_di = 100 * smooth_minus_dm / atr
    
    # DX and ADX
    di_sum = plus_di + minus_di
    di_sum = np.where(di_sum == 0, 1e-10, di_sum)
    dx = 100 * np.abs(plus_di - minus_di) / di_sum
    
    adx_val = wilder_smooth(dx, length)
    
    warmup = length * 2
    return (
        pad_nan(adx_val, warmup),
        pad_nan(plus_di, length),
        pad_nan(minus_di, length)
    )


def stoch_rsi(source: ArrayLike, rsi_length: int = 14, stoch_length: int = 14,
              smooth_k: int = 3, smooth_d: int = 3
              ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Stochastic RSI - Vectorized.
    
    Returns:
        (k, d)
    """
    rsi_vals = rsi(source, rsi_length)
    
    hh = rolling_max(rsi_vals, stoch_length)
    ll = rolling_min(rsi_vals, stoch_length)
    
    denom = hh - ll
    denom = np.where(denom == 0, 1e-10, denom)
    
    stoch = (rsi_vals - ll) / denom
    k = rolling_mean(stoch, smooth_k)
    d = rolling_mean(k, smooth_d)
    
    warmup = rsi_length + stoch_length
    return pad_nan(k, warmup), pad_nan(d, warmup + smooth_d)


def trix(source: ArrayLike, length: int = 15) -> np.ndarray:
    """TRIX Indicator - Vectorized."""
    close = ensure_array(source)
    
    ema1 = ewm_mean(close, length)
    ema2 = ewm_mean(ema1, length)
    ema3 = ewm_mean(ema2, length)
    
    prev_ema3 = shift(ema3, 1)
    prev_ema3 = np.where(prev_ema3 == 0, 1e-10, prev_ema3)
    
    trix_vals = 100 * (ema3 - prev_ema3) / prev_ema3
    return pad_nan(trix_vals, length * 3)


def ultimate_oscillator(high: ArrayLike, low: ArrayLike, close: ArrayLike,
                        s1: int = 7, s2: int = 14, s3: int = 28) -> np.ndarray:
    """Ultimate Oscillator - Vectorized."""
    h = ensure_array(high)
    l = ensure_array(low)
    c = ensure_array(close)
    
    prev_c = shift(c, 1)
    
    # Buying Pressure and True Range
    bp = c - np.minimum(l, prev_c)
    tr = np.maximum(h, prev_c) - np.minimum(l, prev_c)
    
    # Rolling sums
    sum_bp_s1 = rolling_sum(bp, s1)
    sum_tr_s1 = rolling_sum(tr, s1)
    sum_bp_s2 = rolling_sum(bp, s2)
    sum_tr_s2 = rolling_sum(tr, s2)
    sum_bp_s3 = rolling_sum(bp, s3)
    sum_tr_s3 = rolling_sum(tr, s3)
    
    # Avoid division by zero
    sum_tr_s1 = np.where(sum_tr_s1 == 0, 1e-10, sum_tr_s1)
    sum_tr_s2 = np.where(sum_tr_s2 == 0, 1e-10, sum_tr_s2)
    sum_tr_s3 = np.where(sum_tr_s3 == 0, 1e-10, sum_tr_s3)
    
    uo = 100 * (
        4 * sum_bp_s1 / sum_tr_s1 +
        2 * sum_bp_s2 / sum_tr_s2 +
        sum_bp_s3 / sum_tr_s3
    ) / 7
    
    return pad_nan(uo, s3)


def cmo(source: ArrayLike, length: int = 14) -> np.ndarray:
    """Chande Momentum Oscillator - Vectorized."""
    close = ensure_array(source)
    delta = diff(close, 1)
    delta = np.nan_to_num(delta)
    
    up = np.where(delta > 0, delta, 0)
    down = np.where(delta < 0, -delta, 0)
    
    sum_up = rolling_sum(up, length)
    sum_down = rolling_sum(down, length)
    
    total = sum_up + sum_down
    total = np.where(total == 0, 1e-10, total)
    
    cmo_vals = 100 * (sum_up - sum_down) / total
    return pad_nan(cmo_vals, length)


def tsi(source: ArrayLike, long_length: int = 25, short_length: int = 13
        ) -> np.ndarray:
    """True Strength Index - Vectorized."""
    close = ensure_array(source)
    mom = diff(close, 1)
    mom = np.nan_to_num(mom)
    abs_mom = np.abs(mom)
    
    ema1 = ewm_mean(mom, short_length)
    ema2 = ewm_mean(ema1, long_length)
    
    abs_ema1 = ewm_mean(abs_mom, short_length)
    abs_ema2 = ewm_mean(abs_ema1, long_length)
    
    abs_ema2 = np.where(abs_ema2 == 0, 1e-10, abs_ema2)
    
    tsi_vals = 100 * ema2 / abs_ema2
    return pad_nan(tsi_vals, long_length + short_length)
