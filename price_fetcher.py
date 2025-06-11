import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Global OHLCV arrays
open_array, high_array, low_array, close_array, volume_array = [], [], [], [], []

# Connect to MetaTrader 5
def connect_to_mt5(login: int, password: str, server: str, path: str = None):
    if path:
        mt5.initialize(path)
    else:
        mt5.initialize()

    authorized = mt5.login(login=login, password=password, server=server)
    if not authorized:
        raise Exception(f"Failed to connect to MT5. Error code: {mt5.last_error()}")

# Get OHLCV arrays
def get_ohlcv_arrays(symbol: str, timeframe=mt5.TIMEFRAME_M1, bars: int = 1000):
    global open_array, high_array, low_array, close_array, volume_array

    utc_from = datetime.utcnow() - timedelta(minutes=bars)
    rates = mt5.copy_rates_from(symbol, timeframe, utc_from, bars)

    if rates is None or len(rates) == 0:
        raise Exception("No data retrieved. Check symbol or connection.")

    df = pd.DataFrame(rates)

    open_array = df['open'].round(5).astype(float).tolist()
    high_array = df['high'].round(5).astype(float).tolist()
    low_array = df['low'].round(5).astype(float).tolist()
    close_array = df['close'].round(5).astype(float).tolist()
    volume_array = df['tick_volume'].round().astype(int).tolist()

    return open_array, high_array, low_array, close_array, volume_array

# Custom getters by index
def get_open_at(index): return float(open_array[index]) if index < len(open_array) else None
def get_high_at(index): return float(high_array[index]) if index < len(high_array) else None
def get_low_at(index): return float(low_array[index]) if index < len(low_array) else None
def get_close_at(index): return float(close_array[index]) if index < len(close_array) else None
def get_volume_at(index): return int(volume_array[index]) if index < len(volume_array) else None
