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

def get_position_count(symbol=None):
    positions = mt5.positions_get(symbol=symbol) if symbol else mt5.positions_get()
    if positions is None:
        return 0
    return len(positions)

def get_price(symbol):
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        raise RuntimeError(f"Failed to fetch price for {symbol}")
    return tick.bid, tick.ask

def get_account_info():
    account_info = mt5.account_info()
    if account_info is None:
        raise RuntimeError("Failed to fetch account info")
    return account_info.balance, account_info.equity, account_info.margin_free


# 🔥 Calculate Lot Size
def calculate_lot(symbol, risk_percent=None, fixed_amount=None):
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        raise RuntimeError(f"Symbol {symbol} not found")

    balance, _, _ = get_account_info()

    contract_size = symbol_info.trade_contract_size  # Usually 100,000 for forex
    tick_value = symbol_info.trade_tick_value        # Value per tick
    point = symbol_info.point

    if risk_percent:
        risk_amount = balance * risk_percent / 100
        lot = risk_amount / contract_size
    elif fixed_amount:
        lot = fixed_amount / contract_size
    else:
        lot = 0.01  # Default 0.01 lot if nothing specified

    lot = max(symbol_info.volume_min, min(lot, symbol_info.volume_max))
    lot = round(lot, symbol_info.volume_step)  # Adjust to allowed step

    return lot


# 🔥 Place Order with Risk Management
def place_order(
    symbol,
    action,
    lot=None,
    risk_percent=None,
    fixed_amount=None,
    sl_pips=None,
    tp_pips=None,
    sl_usd=None,
    tp_usd=None,
    deviation=20,
    magic=1000,
    comment="AlgoTrade"
):
    bid, ask = get_price(symbol)
    price = ask if action == "buy" else bid
    point = mt5.symbol_info(symbol).point
    contract_size = mt5.symbol_info(symbol).trade_contract_size

    # ✔️ Calculate Lot
    if lot is None:
        lot = calculate_lot(symbol, risk_percent, fixed_amount)

    # ✔️ Calculate SL and TP in price
    sl = None
    tp = None

    # If SL/TP in pips (points)
    if sl_pips:
        sl = price - sl_pips * point if action == "buy" else price + sl_pips * point
    if tp_pips:
        tp = price + tp_pips * point if action == "buy" else price - tp_pips * point

    # If SL/TP in USD terms
    if sl_usd:
        sl_distance = sl_usd / (contract_size * lot)
        sl = price - sl_distance if action == "buy" else price + sl_distance
    if tp_usd:
        tp_distance = tp_usd / (contract_size * lot)
        tp = price + tp_distance if action == "buy" else price - tp_distance

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": mt5.ORDER_TYPE_BUY if action == "buy" else mt5.ORDER_TYPE_SELL,
        "price": price,
        "sl": sl,
        "tp": tp,
        "deviation": deviation,
        "magic": magic,
        "comment": comment,
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_RETURN,
    }

    result = mt5.order_send(request)

    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"❌ Order failed. Retcode: {result.retcode}")
    else:
        print(f"✅ Order successful: {action.upper()} {symbol}, Lot: {lot}")

    return result

