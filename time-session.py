from datetime import datetime, timedelta

# 81. Normalized Time of Day [0.0 - 1.0]
def time_of_day_normalized_utc():
    now = datetime.utcnow()
    seconds = now.hour * 3600 + now.minute * 60 + now.second
    return seconds / 86400  # Total seconds in a day

# 82. Session High/Low
def session_high_low_utc(highs, lows, session_start_utc, session_end_utc, timestamps_utc):
    selected_highs = []
    selected_lows = []
    for i in range(len(timestamps_utc)):
        if session_start_utc.time() <= timestamps_utc[i].time() <= session_end_utc.time():
            selected_highs.append(highs[i])
            selected_lows.append(lows[i])
    return max(selected_highs) if selected_highs else None, min(selected_lows) if selected_lows else None

# 83. Session Overlay (returns session name)
def session_overlay_index_utc():
    hour = datetime.utcnow().hour
    if 0 <= hour < 8:
        return "Asia"
    elif 8 <= hour < 16:
        return "London"
    else:
        return "New York"

# 84. Day of Week Encoding (0 = Monday, 6 = Sunday)
def day_of_week_encoding_utc():
    return datetime.utcnow().weekday()

# 85. Time Since Last High/Low (in bars ago)
def time_since_last_high_low(price_array):
    last_high = max(price_array)
    last_low = min(price_array)
    high_index = len(price_array) - 1 - price_array[::-1].index(last_high)
    low_index = len(price_array) - 1 - price_array[::-1].index(last_low)
    return len(price_array) - high_index - 1, len(price_array) - low_index - 1
