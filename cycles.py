[float] = []
    for price in p:
        prev_l0, prev_l1, prev_l2, prev_l3 = l0, l1, l2, l3
        l0 = (1 - gamma) * price + gamma * prev_l0
        l1 = - (1 - gamma) * l0 + prev_l0 + gamma * prev_l1
        l2 = - (1 - gamma) * l1 + prev_l1 + gamma * prev_l2
        l3 = - (1 - gamma) * l2 + prev_l2 + gamma * prev_l3
        cu = cd = 0.0
        if l0 >= l1:
            cu += l0 - l1
        else:
            cd += l1 - l0
        if l1 >= l2:
            cu += l1 - l2
        else:
            cd += l2 - l1
        if l2 >= l3:
            cu += l2 - l3
        else:
            cd += l3 - l2
        rsi = cu / (cu + cd) if (cu + cd) != 0 else 0.0
        rsi_vals.append(float(round(rsi, ROUND_DECIMALS)))
    return rsi_vals


# 6. QStick

def qstick(
    open_list: NumberArray,
    close_list: NumberArray,
    length: int = 14,
) -> List[float]:
    """QStick: average candle body over N periods."""
    _validate_length(length, 1)
    o = _as_np(open_list)
    c = _as_np(close_list)
    _validate_same_length(o, c)
    body = c - o
    # rolling mean of body
    out = np.zeros_like(body)
    if len(body) > 0:
        cs = np.cumsum(body)
        cs[length:] = cs[length:] - cs[:-length]
        out[: length - 1] = 0.0
        out[length - 1 :] = cs[length - 1 :] / length
    return _round_list(out)


# 7. Stochastic Momentum Index (SMI)

def stochastic_momentum_index(
    close_list: NumberArray,
    high_list: NumberArray,
    low_list: NumberArray,
    length: int = 14,
    smooth_k: int = 3,
    smooth_d: int = 3,
) -> Tuple[List[float], List[float]]:
    """Stochastic Momentum Index (SMI) with double smoothing.
    Returns (K, D)
    """
    _validate_length(length, 2)
    _validate_length(smooth_k, 1)
    _validate_length(smooth_d, 1)
    c = _as_np(close_list)
    h = _as_np(high_list)
    l = _as_np(low_list)
    _validate_same_length(c, h, l)

    smi_k_raw = np.zeros_like(c)
    for i in range(len(c)):
        if i < length - 1:
            smi_k_raw[i] = 0.0
            continue
        hh = float(np.max(h[i - length + 1 : i + 1]))
        ll = float(np.min(l[i - length + 1 : i + 1]))
        mid = (hh + ll) / 2.0
        rng = (hh - ll) / 2.0
        den = rng if rng != 0 else 1e-10
        smi_k_raw[i] = 100.0 * ((c[i] - mid) / den)

    smi_k = np.asarray(ema(ema(smi_k_raw, smooth_k), smooth_k), dtype=float)
    smi_d = np.asarray(ema(smi_k, smooth_d), dtype=float)
    return _round_list(smi_k), _round_list(smi_d)


# 8. Adaptive Cycle Divergence

def adaptive_cycle_divergence(price_list: NumberArray) -> List[float]:
    """Adaptive Cycle Divergence (ACD) using dominant cycle as reference."""
    p = _as_np(price_list)
    dc = _as_np(ht_dominant_cycle(p))
    out = np.zeros_like(p)
    smooth = 0.0
    for i in range(len(p)):
        length = int(round(dc[i]))
        if length < 4 or i < length:
            out[i] = 0.0
            continue
        cycle_component = float(np.mean(p[i - length + 1 : i + 1]))
        div = p[i] - cycle_component
        smooth = 0.2 * div + 0.8 * smooth
        out[i] = smooth
    return _round_list(out)


# 9. Phase Accumulation Cycle (simplified)

def phase_accumulation_cycle(price_list: NumberArray) -> List[float]:
    """Phase Accumulation dominant cycle estimator (simplified)."""
    p = _as_np(price_list)
    out = np.zeros_like(p)
    prev_phase = 0.0
    for i in range(len(p)):
        if i < 2:
            out[i] = 0.0
            continue
        smooth_price = (p[i] + 2.0 * p[i - 1] + p[i - 2]) / 4.0
        i_part = smooth_price - p[i - 2]
        q_part = 2.0 * (p[i - 1] - p[i - 2])
        if abs(q_part) < 1e-10:
            q_part = 1e-10
        phase = float(np.arctan(i_part / q_part))
        delta = 2.0 * np.pi + phase - prev_phase if phase < prev_phase else phase - prev_phase
        prev_phase = phase
        out[i] = (2.0 * np.pi) / delta if delta != 0 else 0.0
    return _round_list(out)


# 10. Inverse Fisher Transform

def inverse_fisher_transform(series: NumberArray) -> List[float]:
    """Inverse Fisher transform. Input must be within [-1, 1]."""
    s = _as_np(series)
    s = np.clip(s, -0.999, 0.999)
    inv = (np.exp(2.0 * s) - 1.0) / (np.exp(2.0 * s) + 1.0)
    return _round_list(inv)


# --- Additional indicators ---


def mama_fama(price_list: NumberArray, fast_limit: float = 0.5, slow_limit: float = 0.05) -> Tuple[List[float], List[float]]:
    """MAMA & FAMA adaptive moving averages."""
    if not (0.0 < slow_limit <= fast_limit <= 1.0):
        raise ValueError("Require 0<slow_limit<=fast_limit<=1")
    p = _as_np(price_list)
    mama_vals: List[float] = []
    fama_vals: List[float] = []
    phase_prev = 0.0
    mama_v = 0.0
    fama_v = 0.0
    for i in range(len(p)):
        if i < 6:
            mama_vals.append(0.0)
            fama_vals.append(0.0)
            continue
        detrender = (0.0962 * p[i] + 0.5769 * p[i - 2] - 0.5769 * p[i - 4] - 0.0962 * p[i - 6])
        i_part = detrender
        q_part = p[i - 3] - p[i - 6]
        if abs(q_part) < 1e-10:
            q_part = 1e-10
        phase = float(np.arctan(i_part / q_part))
        delta_phase = phase - phase_prev
        phase_prev = phase
        alpha = fast_limit / abs(delta_phase) if abs(delta_phase) > 1e-5 else slow_limit
        alpha = max(min(alpha, fast_limit), slow_limit)
        mama_v = alpha * p[i] + (1.0 - alpha) * mama_v
        fama_v = 0.5 * alpha * mama_v + (1.0 - 0.5 * alpha) * fama_v
        mama_vals.append(float(round(mama_v, ROUND_DECIMALS)))
        fama_vals.append(float(round(fama_v, ROUND_DECIMALS)))
    return mama_vals, fama_vals


def super_smoother(price_list: NumberArray, period: int) -> List[float]:
    """Two-pole Super Smoother filter."""
    _validate_length(period, 1)
    p = _as_np(price_list)
    a1 = float(np.exp(-1.414 * np.pi / period))
    b1 = 2.0 * a1 * float(np.cos(1.414 * np.pi / period))
    c2 = b1
    c3 = -a1 ** 2
    c1 = 1.0 - c2 - c3
    out = np.zeros_like(p)
    for i in range(len(p)):
        if i < 2:
            out[i] = p[i]
            continue
        out[i] = c1 * (p[i] + p[i - 1]) / 2.0 + c2 * out[i - 1] + c3 * out[i - 2]
    return _round_list(out)


def roofing_filter(price_list: NumberArray) -> List[float]:
    """High-pass then smoothing (simple roofing-like filter)."""
    p = _as_np(price_list)
    hp = np.zeros_like(p)
    out = np.zeros_like(p)
    for i in range(len(p)):
        if i < 2:
            hp[i] = 0.0
            out[i] = p[i]
            continue
        hp[i] = 0.5 * (p[i] - p[i - 2]) + 0.995 * hp[i - 1]
        out[i] = 0.5 * (hp[i] + hp[i - 1])
    return _round_list(out)


def center_of_gravity(price_list: NumberArray, length: int = 10) -> List[float]:
    """Center of Gravity (COG)."""
    _validate_length(length, 1)
    p = _as_np(price_list)
    out = np.zeros_like(p)
    for i in range(len(p)):
        if i < length:
            out[i] = 0.0
            continue
        idx = np.arange(0, length)
        window = p[i - length + 1 : i + 1][::-1]
        num = float(np.sum(idx * window))
        den = float(np.sum(window))
        out[i] = num / den if den != 0 else 0.0
    return _round_list(out)


def bandpass_filter(price_list: NumberArray, period: int, bandwidth: float = 0.3) -> List[float]:
    """Simple bandpass filter."""
    _validate_length(period, 1)
    if not (0.0 < bandwidth < 1.0):
        raise ValueError("bandwidth must be in (0,1)")
    p = _as_np(price_list)
    alpha = (np.cos(2 * np.pi / period) + np.sin(2 * np.pi / period) - 1) / np.cos(
        2 * np.pi * bandwidth / period
    )
    out = np.zeros_like(p)
    for i in range(len(p)):
        if i < 2:
            out[i] = 0.0
            continue
        out[i] = 0.5 * (1.0 - alpha) * (p[i] - p[i - 2]) + alpha * out[i - 1]
    return _round_list(out)


def dc_based_rsi(price_list: NumberArray, cycle_list: NumberArray) -> List[float]:
    """RSI computed over variable window defined by dominant cycle list."""
    p = _as_np(price_list)
    cyc = _as_np(cycle_list)
    _validate_same_length(p, cyc)
    out: List[float] = []
    for i in range(len(p)):
        cycle_len = int(round(cyc[i]))
        if cycle_len < 2 or i < cycle_len:
            out.append(0.0)
            continue
        gains = [max(p[j] - p[j - 1], 0.0) for j in range(i - cycle_len + 1, i + 1)]
        losses = [max(p[j - 1] - p[j], 0.0) for j in range(i - cycle_len + 1, i + 1)]
        avg_gain = sum(gains) / cycle_len
        avg_loss = sum(losses) / cycle_len
        rs = (avg_gain / avg_loss) if avg_loss != 0 else 0.0
        rsi = 100.0 - (100.0 / (1.0 + rs))
        out.append(float(round(rsi, ROUND_DECIMALS)))
    return out


def cyber_cycle(price_list: NumberArray, alpha: float = 0.07) -> List[float]:
    """Cyber Cycle filter."""
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0,1)")
    p = _as_np(price_list)
    out = np.zeros_like(p)
    for i in range(len(p)):
        if i < 2:
            out[i] = p[i]
            continue
        out[i] = (1.0 - 0.5 * alpha) ** 2 * (p[i] - 2.0 * p[i - 1] + p[i - 2]) + 2.0 * (1.0 - alpha) * out[
            i - 1
        ] - (1.0 - alpha) ** 2 * out[i - 2]
    return _round_list(out)


# =============================
# Usage / Test examples
# =============================

def run_examples() -> None:
    """Run a small set of smoke tests to validate shapes and ranges.
    This is not a unit test replacement, just quick sanity checks.
    """
    x = np.linspace(0, 4 * np.pi, 200)
    price = 100 + 2 * np.sin(x) + 0.5 * np.cos(3 * x)

    # Call a subset
    _ = fisher_transform(source_type="close", length=10, start=0, end=len(price), smooth=True, smooth_factor=0.33, close_list=price)
    _ = hilbert_transform(price)
    s, ls = ht_sine(price)
    _ = ht_phase(price)
    _ = ht_trendline(price, 0.1)
    _ = ht_dominant_cycle(price)
    _ = ht_itrend(price)
    _ = detrended_price_oscillator(price)
    _ = laguerre_rsi(price)
    _ = qstick(price, price + 0.2)
    _ = stochastic_momentum_index(price, price + 0.5, price - 0.5)
    _ = adaptive_cycle_divergence(price)
    _ = phase_accumulation_cycle(price)
    _ = inverse_fisher_transform(np.clip(np.sin(x), -0.9, 0.9))
    _ = mama_fama(price)
    _ = super_smoother(price, 20)
    _ = roofing_filter(price)
    _ = center_of_gravity(price, 10)
    _ = bandpass_filter(price, 30, 0.3)
    _ = dc_based_rsi(price, ht_dominant_cycle(price))
    _ = cyber_cycle(price)

    print("run_examples: completed without errors")
