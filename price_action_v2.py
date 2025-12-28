"""
PythonPine Price Action V2 - Vectorized Pattern Detection
Returns numerical flags instead of bar-by-bar detection
"""

import numpy as np
import pandas as pd
from typing import Union, Tuple
from .utils import ensure_array, shift, rolling_max, rolling_min


ArrayLike = Union[np.ndarray, pd.Series, list]


def engulfing(open_: ArrayLike, high: ArrayLike, low: ArrayLike, close: ArrayLike
              ) -> np.ndarray:
    """
    Engulfing Pattern Detection - Vectorized.
    
    Returns:
        +1: Bullish engulfing
        -1: Bearish engulfing
         0: No pattern
    """
    o = ensure_array(open_)
    h = ensure_array(high)
    l = ensure_array(low)
    c = ensure_array(close)
    
    prev_o = shift(o, 1)
    prev_c = shift(c, 1)
    
    # Bullish: Previous bar was bearish, current engulfs it
    bullish = (
        (prev_c < prev_o) &  # Previous was bearish
        (c > o) &             # Current is bullish
        (o <= prev_c) &       # Current open <= prev close
        (c >= prev_o)         # Current close >= prev open
    )
    
    # Bearish: Previous bar was bullish, current engulfs it
    bearish = (
        (prev_c > prev_o) &   # Previous was bullish
        (c < o) &             # Current is bearish
        (o >= prev_c) &       # Current open >= prev close
        (c <= prev_o)         # Current close <= prev open
    )
    
    result = np.zeros(len(c))
    result[bullish] = 1
    result[bearish] = -1
    result[0] = 0  # First bar has no previous
    
    return result.astype(int)


def doji(open_: ArrayLike, high: ArrayLike, low: ArrayLike, close: ArrayLike,
         threshold: float = 0.1) -> np.ndarray:
    """
    Doji Detection - Vectorized.
    
    Returns:
        1: Doji detected
        0: No doji
    
    Args:
        threshold: Max body/range ratio to qualify as doji (default 10%)
    """
    o = ensure_array(open_)
    h = ensure_array(high)
    l = ensure_array(low)
    c = ensure_array(close)
    
    body = np.abs(c - o)
    range_ = h - l
    range_ = np.where(range_ == 0, 1e-10, range_)
    
    ratio = body / range_
    doji_bars = (ratio < threshold).astype(int)
    
    return doji_bars


def hammer(open_: ArrayLike, high: ArrayLike, low: ArrayLike, close: ArrayLike,
           body_ratio: float = 0.3, wick_ratio: float = 2.0) -> np.ndarray:
    """
    Hammer Pattern (Bullish) - Vectorized.
    
    Returns:
        1: Hammer detected
        0: No pattern
    """
    o = ensure_array(open_)
    h = ensure_array(high)
    l = ensure_array(low)
    c = ensure_array(close)
    
    body = np.abs(c - o)
    range_ = h - l
    range_ = np.where(range_ == 0, 1e-10, range_)
    
    # Body in upper third
    body_top = np.maximum(c, o)
    upper_wick = h - body_top
    lower_wick = np.minimum(c, o) - l
    
    is_hammer = (
        (body / range_ < body_ratio) &  # Small body
        (lower_wick > wick_ratio * body) &  # Long lower wick
        (upper_wick < body)  # Small upper wick
    )
    
    return is_hammer.astype(int)


def shooting_star(open_: ArrayLike, high: ArrayLike, low: ArrayLike, close: ArrayLike,
                  body_ratio: float = 0.3, wick_ratio: float = 2.0) -> np.ndarray:
    """
    Shooting Star Pattern (Bearish) - Vectorized.
    
    Returns:
        1: Shooting star detected
        0: No pattern
    """
    o = ensure_array(open_)
    h = ensure_array(high)
    l = ensure_array(low)
    c = ensure_array(close)
    
    body = np.abs(c - o)
    range_ = h - l
    range_ = np.where(range_ == 0, 1e-10, range_)
    
    body_top = np.maximum(c, o)
    upper_wick = h - body_top
    lower_wick = np.minimum(c, o) - l
    
    is_star = (
        (body / range_ < body_ratio) &
        (upper_wick > wick_ratio * body) &
        (lower_wick < body)
    )
    
    return is_star.astype(int)


def morning_star(open_: ArrayLike, high: ArrayLike, low: ArrayLike, close: ArrayLike
                 ) -> np.ndarray:
    """
    Morning Star (3-bar bullish reversal) - Vectorized.
    
    Returns:
        1: Pattern detected on 3rd bar
        0: No pattern
    """
    o = ensure_array(open_)
    c = ensure_array(close)
    
    # First bar: strong bearish
    prev2_bearish = (shift(c, 2) < shift(o, 2)) & \
                    (np.abs(shift(c, 2) - shift(o, 2)) > 0)
    
    # Second bar: small body (indecision)
    prev1_body = np.abs(shift(c, 1) - shift(o, 1))
    prev2_body = np.abs(shift(c, 2) - shift(o, 2))
    small_body = prev1_body < prev2_body * 0.5
    
    # Third bar: strong bullish closing above first bar's midpoint
    current_bullish = (c > o)
    closes_above = c > (shift(o, 2) + shift(c, 2)) / 2
    
    is_morning_star = prev2_bearish & small_body & current_bullish & closes_above
    
    result = np.zeros(len(c))
    result[is_morning_star] = 1
    result[:2] = 0
    
    return result.astype(int)


def evening_star(open_: ArrayLike, high: ArrayLike, low: ArrayLike, close: ArrayLike
                 ) -> np.ndarray:
    """
    Evening Star (3-bar bearish reversal) - Vectorized.
    
    Returns:
        1: Pattern detected on 3rd bar
        0: No pattern
    """
    o = ensure_array(open_)
    c = ensure_array(close)
    
    # First bar: strong bullish
    prev2_bullish = (shift(c, 2) > shift(o, 2))
    
    # Second bar: small body
    prev1_body = np.abs(shift(c, 1) - shift(o, 1))
    prev2_body = np.abs(shift(c, 2) - shift(o, 2))
    small_body = prev1_body < prev2_body * 0.5
    
    # Third bar: strong bearish
    current_bearish = (c < o)
    closes_below = c < (shift(o, 2) + shift(c, 2)) / 2
    
    is_evening_star = prev2_bullish & small_body & current_bearish & closes_below
    
    result = np.zeros(len(c))
    result[is_evening_star] = 1
    result[:2] = 0
    
    return result.astype(int)


def inside_bar(high: ArrayLike, low: ArrayLike) -> np.ndarray:
    """
    Inside Bar Detection - Vectorized.
    
    Returns:
        1: Inside bar detected
        0: No pattern
    """
    h = ensure_array(high)
    l = ensure_array(low)
    
    prev_h = shift(h, 1)
    prev_l = shift(l, 1)
    
    inside = (h < prev_h) & (l > prev_l)
    
    result = inside.astype(int)
    result[0] = 0
    
    return result


def outside_bar(high: ArrayLike, low: ArrayLike) -> np.ndarray:
    """
    Outside Bar Detection - Vectorized.
    
    Returns:
        1: Outside bar detected
        0: No pattern
    """
    h = ensure_array(high)
    l = ensure_array(low)
    
    prev_h = shift(h, 1)
    prev_l = shift(l, 1)
    
    outside = (h > prev_h) & (l < prev_l)
    
    result = outside.astype(int)
    result[0] = 0
    
    return result


def support_resistance_zones(
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
    volume: ArrayLike,
    lookback: int = 50,
    num_zones: int = 3
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Support/Resistance Zone Detection based on volume profile.
    
    Identifies price levels where significant volume occurred.
    
    Returns:
        (support_levels, resistance_levels) - arrays of price levels
    """
    h = ensure_array(high)
    l = ensure_array(low)
    c = ensure_array(close)
    v = ensure_array(volume)
    
    n = len(c)
    
    # Rolling volume profile
    support = np.full(n, np.nan)
    resistance = np.full(n, np.nan)
    
    for i in range(lookback, n):
        window_h = h[i-lookback:i]
        window_l = l[i-lookback:i]
        window_c = c[i-lookback:i]
        window_v = v[i-lookback:i]
        
        # Price bins
        price_min = np.min(window_l)
        price_max = np.max(window_h)
        
        if price_max == price_min:
            continue
        
        bins = np.linspace(price_min, price_max, 20)
        
        # Volume by price level
        volume_profile = np.zeros(len(bins) - 1)
        for j in range(len(window_c)):
            bin_idx = np.digitize(window_c[j], bins) - 1
            bin_idx = np.clip(bin_idx, 0, len(volume_profile) - 1)
            volume_profile[bin_idx] += window_v[j]
        
        # Find high volume nodes
        sorted_idx = np.argsort(volume_profile)[::-1]
        
        if len(sorted_idx) >= num_zones:
            # Current price relative to zones
            current_price = c[i]
            zone_prices = [(bins[idx] + bins[idx+1]) / 2 for idx in sorted_idx[:num_zones*2]]
            
            below = [p for p in zone_prices if p < current_price]
            above = [p for p in zone_prices if p > current_price]
            
            if below:
                support[i] = max(below)
            if above:
                resistance[i] = min(above)
    
    return support, resistance


def higher_highs_lower_lows(high: ArrayLike, low: ArrayLike, close: ArrayLike,
                            lookback: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Higher Highs / Lower Lows Detection - Vectorized.
    
    Returns:
        (higher_high_flag, lower_low_flag)
        Each is 1 when detected, 0 otherwise
    """
    h = ensure_array(high)
    l = ensure_array(low)
    
    prev_high = rolling_max(shift(h, 1), lookback)
    prev_low = rolling_min(shift(l, 1), lookback)
    
    hh = (h > prev_high).astype(int)
    ll = (l < prev_low).astype(int)
    
    hh[:lookback] = 0
    ll[:lookback] = 0
    
    return hh, ll
