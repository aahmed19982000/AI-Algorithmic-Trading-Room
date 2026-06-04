"""
MT5 Order Execution Module
===========================
Handles sending buy/sell orders, detecting filling modes, checking open positions,
and closing active trades.
"""

import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import MetaTrader5 as mt5

# Symbol filling flags (not exposed by MetaTrader5 module)
SYMBOL_FILLING_FOK = 1
SYMBOL_FILLING_IOC = 2

def get_filling_mode(symbol):
    """
    Detect the supported order filling mode for the broker symbol.
    Especially important for Exness and ECN accounts.
    """
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        print(f"[WARNING] Could not get symbol info for {symbol} to detect filling mode. Defaulting to FOK.")
        return mt5.ORDER_FILLING_FOK

    filling_mode = symbol_info.filling_mode
    
    # Check flags
    if filling_mode & SYMBOL_FILLING_FOK:
        return mt5.ORDER_FILLING_FOK
    elif filling_mode & SYMBOL_FILLING_IOC:
        return mt5.ORDER_FILLING_IOC
    
    # Default fallback
    return mt5.ORDER_FILLING_FOK


def get_open_positions(symbol=None):
    """
    Get list of currently open positions. Optionally filtered by symbol.
    """
    if symbol:
        positions = mt5.positions_get(symbol=symbol)
    else:
        positions = mt5.positions_get()
        
    if positions is None:
        error = mt5.last_error()
        print(f"[ERROR] Failed to get open positions. Error code: {error}")
        return []
    
    return list(positions)


def open_trade(action, symbol, volume, sl=0.0, tp=0.0, magic=123456, comment=""):
    """
    Open a new trading position.

    Args:
        action: 'BUY' or 'SELL'
        symbol: Symbol to trade (e.g. 'EURUSDm')
        volume: Volume in lots (e.g. 0.01)
        sl: Stop Loss price level
        tp: Take Profit price level
        magic: Magic number to identify bot orders
        comment: Order comment

    Returns:
        dict: Result details or None on failure
    """
    action = action.upper()
    if action not in ['BUY', 'SELL']:
        print(f"[ERROR] Invalid trade action: {action}")
        return None

    # Get current price
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        print(f"[ERROR] Failed to get tick price for {symbol} to open trade")
        return None

    price = tick.ask if action == 'BUY' else tick.bid
    order_type = mt5.ORDER_TYPE_BUY if action == 'BUY' else mt5.ORDER_TYPE_SELL
    filling = get_filling_mode(symbol)

    # Format requests dict
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": float(volume),
        "type": order_type,
        "price": float(price),
        "sl": float(sl) if sl > 0 else 0.0,
        "tp": float(tp) if tp > 0 else 0.0,
        "deviation": 20,
        "magic": int(magic),
        "comment": comment,
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": filling,
    }

    print(f"\n[TRADE] Sending order request:")
    print(f"  Action:       {action}")
    print(f"  Symbol:       {symbol}")
    print(f"  Volume:       {volume} lots")
    print(f"  Entry Price:  {price}")
    print(f"  Stop Loss:    {sl}")
    print(f"  Take Profit:  {tp}")
    print(f"  Filling Mode: {filling} (FOK=1, IOC=2, RETURN=0)")

    # Send order to MT5
    result = mt5.order_send(request)
    
    if result is None:
        error = mt5.last_error()
        print(f"[ERROR] Order send failed completely (returned None). MT5 error: {error}")
        return None

    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"[FAIL] Order execution failed!")
        print(f"  Return code:  {result.retcode}")
        print(f"  Error description: {result.comment}")
        return {
            "success": False,
            "retcode": result.retcode,
            "comment": result.comment,
            "ticket": None
        }

    print(f"  [SUCCESS] Trade executed successfully!")
    print(f"  Ticket:       {result.order}")
    print(f"  Volume:       {result.volume} lots")
    print(f"  Executed Price: {result.price}")
    
    return {
        "success": True,
        "retcode": result.retcode,
        "comment": result.comment,
        "ticket": result.order,
        "price": result.price
    }


def close_position(ticket, magic=123456, comment="Close position"):
    """
    Close an active position by its ticket.
    """
    # Fetch position details
    positions = mt5.positions_get(ticket=ticket)
    if not positions or len(positions) == 0:
        print(f"[ERROR] Position ticket {ticket} not found to close")
        return False
        
    position = positions[0]
    symbol = position.symbol
    volume = position.volume
    pos_type = position.type
    
    # Opposite action
    order_type = mt5.ORDER_TYPE_SELL if pos_type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
    
    # Get current price
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        print(f"[ERROR] Failed to get tick price for {symbol} to close trade")
        return False
        
    price = tick.bid if pos_type == mt5.POSITION_TYPE_BUY else tick.ask
    filling = get_filling_mode(symbol)
    
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": float(volume),
        "type": order_type,
        "position": int(ticket),
        "price": float(price),
        "deviation": 20,
        "magic": int(magic),
        "comment": comment,
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": filling,
    }
    
    print(f"\n[TRADE] Closing position ticket {ticket}:")
    print(f"  Symbol:       {symbol}")
    print(f"  Volume:       {volume} lots")
    print(f"  Close Price:  {price}")
    
    result = mt5.order_send(request)
    
    if result is None:
        print(f"[ERROR] Order send failed completely when closing position")
        return False
        
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"[FAIL] Close execution failed!")
        print(f"  Return code:  {result.retcode}")
        print(f"  Error description: {result.comment}")
        return False
        
    print(f"  [SUCCESS] Position closed successfully!")
    return True


def modify_position_sl_tp(ticket, sl, tp):
    """
    Modify Stop Loss and Take Profit levels of an active position.
    """
    # Fetch position details
    positions = mt5.positions_get(ticket=ticket)
    if not positions or len(positions) == 0:
        print(f"[ERROR] Position ticket {ticket} not found to modify")
        return False
        
    position = positions[0]
    symbol = position.symbol
    
    request = {
        "action": mt5.TRADE_ACTION_SLTP,
        "position": int(ticket),
        "symbol": symbol,
        "sl": float(sl),
        "tp": float(tp),
    }
    
    result = mt5.order_send(request)
    
    if result is None:
        print(f"[ERROR] Modify SL/TP send failed completely for ticket {ticket}")
        return False
        
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"[FAIL] Modify SL/TP failed for ticket {ticket}!")
        print(f"  Return code:  {result.retcode}")
        print(f"  Error description: {result.comment}")
        return False
        
    print(f"  [SUCCESS] Position {ticket} SL/TP modified successfully: SL={sl}, TP={tp}")
    return True
