def ma_crossover_count(short_ma, long_ma):
    count = 0
    for i in range(1, len(short_ma)):
        if short_ma[i-1] < long_ma[i-1] and short_ma[i] > long_ma[i]:
            count += 1
    return count

def indicator_consensus_score(*indicators):
    scores = [1 if ind > 0 else -1 if ind < 0 else 0 for ind in indicators]
    return sum(scores)

def momentum_volatility_composite(momentum_values, volatility_values):
    return sum(momentum_values[-5:]) * sum(volatility_values[-5:])

def trend_strength_score(trend_indicator):
    mean = sum(trend_indicator[-10:]) / 10
    return abs(trend_indicator[-1] - mean)

def macd_histogram_angle(macd_hist):
    if len(macd_hist) < 3:
        return 0.0
    return (macd_hist[-1] - macd_hist[-3]) / 2.0

def rsi_divergence_count(price, rsi):
    count = 0
    for i in range(2, len(price)):
        if price[i] > price[i-1] > price[i-2] and rsi[i] < rsi[i-1] < rsi[i-2]:
            count += 1
    return count

def volume_spike_flag(volume_array, threshold_multiplier=2.0):
    avg_volume = sum(volume_array[-20:]) / 20
    return 1 if volume_array[-1] > threshold_multiplier * avg_volume else 0

def multi_timeframe_ema_alignment(*emas):
    return 1 if all(emas[i] > emas[i+1] for i in range(len(emas)-1)) else -1

def trend_reversal_likelihood(price_array):
    recent = price_array[-5:]
    return 1 if recent == sorted(recent, reverse=True) else -1 if recent == sorted(recent) else 0

def consolidation_detector(price_array, threshold=0.002):
    max_p = max(price_array[-10:])
    min_p = min(price_array[-10:])
    return 1 if (max_p - min_p) / min_p < threshold else 0
