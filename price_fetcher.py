import MetaTrader5 as mt5

# Global variables to store settings and data
SYMBOL = None
TIMEFRAME = None
DATA_COUNT = None
_rates = []

def initialize_mt5(login, password, server, symbol, timeframe, data_count):
    global SYMBOL, TIMEFRAME, DATA_COUNT, _rates

    # Initialize connection
    if not mt5.initialize():
        raise RuntimeError(f"initialize() failed, error code: {mt5.last_error()}")

    authorized = mt5.login(login=login, password=password, server=server)
    if not authorized:
        raise RuntimeError(f"MT5 login failed, error code: {mt5.last_error()}")

    # Set global config
    SYMBOL = symbol
    TIMEFRAME = timeframe
    DATA_COUNT = data_count

    # Fetch OHLCV data
    _rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, DATA_COUNT)
    if _rates is None or len(_rates) == 0:
        raise RuntimeError("Failed to fetch market data. Check symbol/timeframe.")

# Internal helpers
def _get_value(field, index):
    if index < 0 or index >= len(_rates):
        return None
    return float(_rates[index][field])

def _get_list(field, start, end):
    if start < 0 or end >= len(_rates) or end < start:
        return []
    return [float(_rates[i][field]) for i in range(start, end + 1)]

# Single value functions
def close(index=0):
    return _get_value('close', index)

def open(index=0):
    return _get_value('open', index)

def high(index=0):
    return _get_value('high', index)

def low(index=0):
    return _get_value('low', index)

def volume(index=0):
    return _get_value('tick_volume', index)

# List functions
def closeList(start, end):
    return _get_list('close', start, end)

def openList(start, end):
    return _get_list('open', start, end)

def highList(start, end):
    return _get_list('high', start, end)

def lowList(start, end):
    return _get_list('low', start, end)

def volumeList(start, end):
    return _get_list('tick_volume', start, end)

def highest(end) :
    return max(highList(0,end - 1))

def lowest(end) :
    return min(lowList(0,end - 1))
