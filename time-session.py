from datetime import datetime, timedelta

# --- 81. Time of Day (normalized 0 to 1) ---
def normalized_time_of_day(timestamps):
    utc_times = [ts.astimezone(timezone.utc) for ts in timestamps]
    return [((ts.hour * 60 + ts.minute) / 1440) for ts in utc_times]

# --- 82. Session High/Low ---
def session_high_low(highList, lowList, session_start_index, session_end_index):
    high = max(highList[session_start_index:session_end_index+1])
    low = min(lowList[session_start_index:session_end_index+1])
    return high, low

# --- 83. Session Overlay Flags (London/NY/Asia) ---
def session_overlay_flags(timestamps):
    utc_times = [ts.astimezone(timezone.utc) for ts in timestamps]
    asia = [1 if 0 <= ts.hour < 9 else 0 for ts in utc_times]
    london = [1 if 8 <= ts.hour < 17 else 0 for ts in utc_times]
    ny = [1 if 13 <= ts.hour < 22 else 0 for ts in utc_times]
    return asia, london, ny

# --- 84. Day of Week Encoding (0=Mon, ..., 6=Sun) ---
def day_of_week(timestamps):
    utc_times = [ts.astimezone(timezone.utc) for ts in timestamps]
    return [ts.weekday() for ts in utc_times]

# --- 85. Time Since Last High/Low ---
def time_since_last_extreme(priceList, extreme='high'):
    last_time = 0
    time_since = []
    for i in range(len(priceList)):
        if i == 0 or (extreme == 'high' and priceList[i] >= max(priceList[:i+1])) or (extreme == 'low' and priceList[i] <= min(priceList[:i+1])):
            last_time = 0
        else:
            last_time += 1
        time_since.append(last_time)
    return time_since
