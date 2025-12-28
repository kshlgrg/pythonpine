"""
PythonPine v2.0.0 - High-Performance Vectorized Technical Indicators

Pine Script-compatible indicators optimized for machine learning and backtesting.
All v2 modules use NumPy/Pandas vectorization for 10-100x speedup.
"""

__version__ = "2.0.0"
__author__ = "Kushal Garg"

# ============== V2 VECTORIZED MODULES (RECOMMENDED) ==============

# Core utilities
from .utils import (
    ensure_array, rolling_max, rolling_min, rolling_mean, rolling_sum,
    rolling_std, ewm_mean, wilder_smooth, true_range, diff, shift,
    cross_above, cross_below, highest, lowest, get_warmup
)

# ML Normalization
from .normalize import (
    zscore, percentile_rank, log_returns, simple_returns,
    fractional_diff, minmax_scale, robust_scale,
    cusum_filter, triple_barrier_labels
)

# Momentum (Vectorized)
from .momentum_v2 import (
    rsi, macd, stochastic, williams_r, cci, roc, momentum,
    adx, stoch_rsi, trix, ultimate_oscillator, cmo, tsi
)

# Volatility (Vectorized)
from .volatility_v2 import (
    atr, bollinger_bands, keltner_channel, donchian_channel,
    bollinger_bandwidth, bollinger_percent_b, historical_volatility,
    normalized_atr, ulcer_index, chaikin_volatility, mass_index,
    parkinson_volatility, garman_klass_volatility
)

# Trend (Vectorized)
from .trend_v2 import (
    sma, ema, wma, dema, tema, hma, vwma, kama,
    supertrend, aroon, ichimoku, linear_regression,
    parabolic_sar, zero_lag_ema, mcginley_dynamic
)

# Price Action (Vectorized)
from .price_action_v2 import (
    engulfing, doji, hammer, shooting_star,
    morning_star, evening_star, inside_bar, outside_bar,
    support_resistance_zones, higher_highs_lower_lows
)

# ============== LEGACY MODULES (BACKWARD COMPATIBILITY) ==============

# Legacy modules preserved - import as needed
# from .momentum import rsi as rsi_legacy
# from .volatility import atr as atr_legacy
# from .trend import sma as sma_legacy

# Cycles (complex number indicators)
from .cycles import (
    hilbert_transform, fisher_transform, ehlers_smooth,
    stochastic_rsi_fisher
)

# Meta indicators
from .meta import (
    trend_strength_index, trending_or_ranging
)

# Price fetcher
from .price_fetcher import (
    fetch_price_data, get_candles
)


# ============== CONVENIENCE CLASSES ==============

class Indicators:
    """
    Convenience class grouping all vectorized indicators.
    
    Example:
        from pythonpine import Indicators
        ind = Indicators()
        rsi_values = ind.rsi(close, 14)
    """
    
    # Momentum
    rsi = staticmethod(rsi)
    macd = staticmethod(macd)
    stochastic = staticmethod(stochastic)
    adx = staticmethod(adx)
    cci = staticmethod(cci)
    roc = staticmethod(roc)
    tsi = staticmethod(tsi)
    
    # Volatility
    atr = staticmethod(atr)
    bollinger_bands = staticmethod(bollinger_bands)
    keltner_channel = staticmethod(keltner_channel)
    historical_volatility = staticmethod(historical_volatility)
    
    # Trend
    sma = staticmethod(sma)
    ema = staticmethod(ema)
    wma = staticmethod(wma)
    hma = staticmethod(hma)
    supertrend = staticmethod(supertrend)
    ichimoku = staticmethod(ichimoku)
    
    # Normalization
    zscore = staticmethod(zscore)
    percentile_rank = staticmethod(percentile_rank)
    log_returns = staticmethod(log_returns)


__all__ = [
    # Version
    "__version__",
    
    # Convenience class
    "Indicators",
    
    # Utils
    "ensure_array", "rolling_max", "rolling_min", "rolling_mean",
    "ewm_mean", "wilder_smooth", "true_range",
    
    # Normalize
    "zscore", "percentile_rank", "log_returns", "fractional_diff",
    "triple_barrier_labels",
    
    # Momentum V2
    "rsi", "macd", "stochastic", "adx", "cci", "roc", "tsi",
    
    # Volatility V2
    "atr", "bollinger_bands", "keltner_channel", "historical_volatility",
    
    # Trend V2
    "sma", "ema", "wma", "hma", "supertrend", "ichimoku",
    
    # Price Action V2
    "engulfing", "doji", "hammer", "support_resistance_zones",
]
