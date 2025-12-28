"""
PythonPine Benchmark - Performance Comparison
Compare vectorized vs loop-based implementations
"""

import time
import numpy as np
from typing import Callable, Dict
import sys


def benchmark(
    func_v2: Callable,
    func_legacy: Callable,
    args: tuple,
    iterations: int = 10,
    name: str = "Indicator"
) -> Dict:
    """
    Benchmark vectorized vs legacy function.
    
    Returns:
        {
            'name': str,
            'v2_time': float (seconds),
            'legacy_time': float (seconds),
            'speedup': float,
            'v2_result_sample': first 5 values,
            'legacy_result_sample': first 5 values
        }
    """
    # Warmup
    _ = func_v2(*args)
    _ = func_legacy(*args)
    
    # V2 timing
    start = time.perf_counter()
    for _ in range(iterations):
        result_v2 = func_v2(*args)
    v2_time = (time.perf_counter() - start) / iterations
    
    # Legacy timing
    start = time.perf_counter()
    for _ in range(iterations):
        result_legacy = func_legacy(*args)
    legacy_time = (time.perf_counter() - start) / iterations
    
    # Handle tuple results
    if isinstance(result_v2, tuple):
        v2_sample = result_v2[0][:5] if hasattr(result_v2[0], '__getitem__') else result_v2[:5]
        legacy_sample = result_legacy[0][:5] if hasattr(result_legacy[0], '__getitem__') else result_legacy[:5]
    else:
        v2_sample = result_v2[:5] if hasattr(result_v2, '__getitem__') else result_v2
        legacy_sample = result_legacy[:5] if hasattr(result_legacy, '__getitem__') else result_legacy
    
    return {
        'name': name,
        'v2_time': v2_time,
        'legacy_time': legacy_time,
        'speedup': legacy_time / v2_time if v2_time > 0 else float('inf'),
        'v2_result_sample': list(v2_sample) if hasattr(v2_sample, '__iter__') else v2_sample,
        'legacy_result_sample': list(legacy_sample) if hasattr(legacy_sample, '__iter__') else legacy_sample
    }


def run_all_benchmarks(n_bars: int = 100000):
    """
    Run benchmarks on all major indicators.
    
    Args:
        n_bars: Number of bars to test with (default 100,000)
    """
    print(f"\n{'='*60}")
    print(f"PythonPine Performance Benchmark")
    print(f"Testing with {n_bars:,} bars")
    print(f"{'='*60}\n")
    
    # Generate test data
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(n_bars) * 0.5)
    high = close + np.abs(np.random.randn(n_bars)) * 0.3
    low = close - np.abs(np.random.randn(n_bars)) * 0.3
    open_ = close + np.random.randn(n_bars) * 0.1
    volume = np.random.randint(1000, 10000, n_bars)
    
    close_list = close.tolist()
    high_list = high.tolist()
    low_list = low.tolist()
    
    results = []
    
    # RSI
    try:
        from .momentum_v2 import rsi as rsi_v2
        from .momentum import rsi as rsi_legacy
        r = benchmark(
            lambda: rsi_v2(close, 14),
            lambda: rsi_legacy(close_list, 14),
            (),
            name="RSI"
        )
        results.append(r)
    except Exception as e:
        print(f"RSI benchmark failed: {e}")
    
    # MACD
    try:
        from .momentum_v2 import macd as macd_v2
        from .momentum import macd as macd_legacy
        r = benchmark(
            lambda: macd_v2(close),
            lambda: macd_legacy(close_list),
            (),
            name="MACD"
        )
        results.append(r)
    except Exception as e:
        print(f"MACD benchmark failed: {e}")
    
    # ATR
    try:
        from .volatility_v2 import atr as atr_v2
        from .volatility import atr as atr_legacy
        r = benchmark(
            lambda: atr_v2(high, low, close),
            lambda: atr_legacy(high_list, low_list, close_list),
            (),
            name="ATR"
        )
        results.append(r)
    except Exception as e:
        print(f"ATR benchmark failed: {e}")
    
    # Bollinger Bands
    try:
        from .volatility_v2 import bollinger_bands as bb_v2
        from .volatility import bollinger_bands as bb_legacy
        r = benchmark(
            lambda: bb_v2(close),
            lambda: bb_legacy(close_list),
            (),
            name="Bollinger Bands"
        )
        results.append(r)
    except Exception as e:
        print(f"Bollinger benchmark failed: {e}")
    
    # SMA
    try:
        from .trend_v2 import sma as sma_v2
        from .trend import sma as sma_legacy
        r = benchmark(
            lambda: sma_v2(close, 20),
            lambda: sma_legacy(close_list, 20),
            (),
            name="SMA"
        )
        results.append(r)
    except Exception as e:
        print(f"SMA benchmark failed: {e}")
    
    # EMA
    try:
        from .trend_v2 import ema as ema_v2
        from .momentum import ema as ema_legacy
        r = benchmark(
            lambda: ema_v2(close, 20),
            lambda: ema_legacy(close_list, 20),
            (),
            name="EMA"
        )
        results.append(r)
    except Exception as e:
        print(f"EMA benchmark failed: {e}")
    
    # Print results
    print(f"{'Indicator':<20} {'Legacy (s)':<15} {'V2 (s)':<15} {'Speedup':<15}")
    print("-" * 65)
    
    total_speedup = []
    for r in results:
        print(f"{r['name']:<20} {r['legacy_time']:.6f}s      {r['v2_time']:.6f}s      {r['speedup']:.1f}x")
        total_speedup.append(r['speedup'])
    
    print("-" * 65)
    avg_speedup = np.mean(total_speedup) if total_speedup else 0
    print(f"{'AVERAGE':<20} {'':<15} {'':<15} {avg_speedup:.1f}x")
    print(f"\n✓ Vectorized implementation is {avg_speedup:.0f}x faster on average\n")
    
    return results


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100000
    run_all_benchmarks(n)
