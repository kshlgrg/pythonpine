import numpy as np
import pandas as pd

# 61. Fisher Transform
def fisher_transform(source: pd.Series, length=10):
    val = 0.33 * 2 * ((source - source.rolling(length).min()) / (source.rolling(length).max() - source.rolling(length).min()) - 0.5)
    val = val.clip(-0.999, 0.999)
    fish = 0.5 * np.log((1 + val) / (1 - val))
    return fish.fillna(0).astype(float)

# 62. Hilbert Transform (Cycle and Trend components, simplified)
def hilbert_transform(price: pd.Series):
    imag = price - price.shift(2)
    real = price.shift(1) - price.shift(3)
    return imag.fillna(0).astype(float), real.fillna(0).astype(float)

# 63. Ehler’s Instantaneous Trendline
def ehlers_instantaneous_trendline(close: pd.Series):
    it = (close + 2 * close.shift(1) + 2 * close.shift(2) + close.shift(3)) / 6
    return it.fillna(0).astype(float)

# 64. Detrended Price Oscillator
def detrended_price_oscillator(close: pd.Series, period=20):
    dpo = close.shift(int(period/2 + 1)) - close.rolling(window=period).mean()
    return dpo.fillna(0).astype(float)

# 65. Laguerre RSI
def laguerre_rsi(price: pd.Series, gamma=0.5):
    L = [0]*4
    lrsi = []
    for i in range(len(price)):
        L[0] = (1 - gamma) * price.iloc[i] + gamma * L[0]
        L[1] = -gamma * L[0] + L[0] + gamma * L[1]
        L[2] = -gamma * L[1] + L[1] + gamma * L[2]
        L[3] = -gamma * L[2] + L[2] + gamma * L[3]
        CU = sum([max(L[j] - L[j + 1], 0) for j in range(3)])
        CD = sum([max(L[j + 1] - L[j], 0) for j in range(3)])
        lrsi.append(CU / (CU + CD) if CU + CD != 0 else 0)
    return pd.Series(lrsi, index=price.index).astype(float)

# 66. Qstick
def qstick(open_: pd.Series, close: pd.Series, length=10):
    return (close - open_).rolling(length).mean().fillna(0).astype(float)

# 67. Stochastic Momentum Index (SMI)
def stochastic_momentum_index(close: pd.Series, high: pd.Series, low: pd.Series, length=14):
    hl_avg = (high.rolling(length).max() + low.rolling(length).min()) / 2
    diff = close - hl_avg
    range_ = high.rolling(length).max() - low.rolling(length).min()
    smi = 100 * diff / (range_ / 2)
    return smi.fillna(0).astype(float)

# 68. Adaptive Cycle Divergence (EMA Difference)
def adaptive_cycle_divergence(close: pd.Series, length=14):
    ema1 = close.ewm(span=length, adjust=False).mean()
    ema2 = ema1.ewm(span=length, adjust=False).mean()
    return (ema1 - ema2).fillna(0).astype(float)

# 69. Phase Accumulation Cycle (simplified using FFT phase angle)
def phase_accumulation_cycle(close: pd.Series, length=10):
    close_filled = close.fillna(method='bfill')
    fft_phase = np.angle(np.fft.fft(close_filled))
    phase_series = pd.Series(fft_phase[:len(close)], index=close.index)
    return phase_series.astype(float)

# 70. Inverse Fisher Transform
def inverse_fisher_transform(data: pd.Series):
    return ((np.exp(2 * data) - 1) / (np.exp(2 * data) + 1)).fillna(0).astype(float)
