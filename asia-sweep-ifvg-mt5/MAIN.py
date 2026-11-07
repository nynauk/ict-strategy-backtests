import MetaTrader5 as mt5
import pandas as pd
import time
import sys

# === CONFIG ===
# Make sure these symbols exist & are visible in Market Watch.
SYMBOLS = ["UK100.cash"]#"EURUSD", "GBPUSD", "GBPJPY", "GER40.cash", "UK100.cash"]
TIMEFRAME = mt5.TIMEFRAME_M5
MAGIC = 20250806
DEVIATION = 20
RISK_PERCENT = 1.0      # % of balance per trade
RR_TARGET = 2.0         # <-- enforce 2R
SLEEP_SEC = 60
MIN_GAP_TICKS = 2       # ignore micro-gaps
MIN_LOOKBACK = 7        # look back 7 candles
REQUIRED_BARS = MIN_LOOKBACK + 3  # enough for [-7], [-5], [-2]

# === SYMBOL / PRICING HELPERS ===
def symbol_meta(symbol):
    info = mt5.symbol_info(symbol)
    if not info:
        return None

    # tolerant getters for older MT5 builds/brokers
    def g(obj, name, default=0.0):
        return getattr(obj, name, default)

    tick_size = g(info, "trade_tick_size", 0.0) or g(info, "point", 0.0)
    tick_value = (
        g(info, "trade_tick_value", 0.0)
        or g(info, "trade_tick_value_profit", 0.0)
        or g(info, "trade_tick_value_loss", 0.0)
    )
    freeze_level_pts = int(g(info, "freeze_level", 0) or g(info, "trade_freeze_level", 0))

    return {
        "digits": int(g(info, "digits", 5)),
        "point": float(g(info, "point", 0.0001)),
        "tick_size": float(tick_size if tick_size > 0 else g(info, "point", 0.0001)),
        "tick_value": float(tick_value),  # account-currency per tick (1.00 lot)
        "vol_min": float(g(info, "volume_min", 0.01)),
        "vol_max": float(g(info, "volume_max", 100.0)),
        "vol_step": float(g(info, "volume_step", 0.01)),
        "stops_level": int(g(info, "trade_stops_level", 0) or g(info, "stops_level", 0)),
        "freeze_level": freeze_level_pts,  # may be 0 if unsupported
        "contract_size": float(g(info, "trade_contract_size", g(info, "contract_size", 1.0))),
    }

def round_price_to_tick(symbol, price):
    m = symbol_meta(symbol)
    if not m:
        return price
    ts, d = m["tick_size"], m["digits"]
    if ts <= 0:
        return round(price, d)
    return round(round(price / ts) * ts, d)

def round_volume(symbol, lots):
    m = symbol_meta(symbol)
    if not m:
        return 0.0
    step = m["vol_step"] or 0.01
    steps = int(lots / step)
    lots_f = steps * step
    return max(m["vol_min"], min(m["vol_max"], lots_f))

def min_stop_distance_price(symbol):
    m = symbol_meta(symbol)
    if not m:
        return 0.0
    return m["stops_level"] * m["point"]

# === DATA ===
def get_candles(symbol, count=REQUIRED_BARS+3):
    data = mt5.copy_rates_from_pos(symbol, TIMEFRAME, 0, count)
    return pd.DataFrame(data) if data is not None else pd.DataFrame()

def has_open_position(symbol):
    poss = mt5.positions_get(symbol=symbol) or []
    return len(poss) > 0

# === RISK / SIZING ===
def calculate_dynamic_lot(symbol, entry, sl, risk_percent=RISK_PERCENT):
    """
    Risk per trade = lots * ticks_to_sl * tick_value
    ticks_to_sl    = abs(entry - sl) / tick_size
    tick_value     = account-currency value per tick for 1.00 lot (broker)
    NOTE: Uses the final (spread-adjusted & rounded) SL.
    """
    acc = mt5.account_info()
    m = symbol_meta(symbol)
    if not acc or not m:
        print(f"[{symbol}] Could not retrieve account/symbol info.")
        return 0.0

    entry = round_price_to_tick(symbol, entry)
    sl    = round_price_to_tick(symbol, sl)

    distance = abs(entry - sl)
    ts, tv = m["tick_size"], m["tick_value"]
    if distance <= 0 or ts <= 0 or tv <= 0:
        print(f"[{symbol}] Missing tick meta or zero distance (ts={ts}, tv={tv}, dist={distance}).")
        return 0.0

    ticks = distance / ts
    risk_amount = acc.balance * (risk_percent / 100.0)
    cost_per_lot = ticks * tv
    if cost_per_lot <= 0:
        print(f"[{symbol}] Invalid cost_per_lot.")
        return 0.0

    lots = risk_amount / cost_per_lot
    lots = round_volume(symbol, lots)

    expected_risk = lots * ticks * tv
    print(f"[{symbol}] Target {risk_percent:.2f}%=${risk_amount:.2f} | ticks={ticks:.1f} | tick=${tv:.5f} | lot={lots} | ≈${expected_risk:.2f}")
    return lots

# === ORDER CONSTRAINTS ===
def check_order_constraints(symbol, order_type, entry, sl, tp):
    m = symbol_meta(symbol)
    if not m:
        return False, "no symbol meta"

    entry = round_price_to_tick(symbol, entry)
    sl    = round_price_to_tick(symbol, sl)
    tp    = round_price_to_tick(symbol, tp)
    msd   = min_stop_distance_price(symbol)

    # SL/TP side correctness
    if order_type == mt5.ORDER_TYPE_BUY_LIMIT:
        if not (sl < entry < tp):
            return False, "buy SL/TP side"
    elif order_type == mt5.ORDER_TYPE_SELL_LIMIT:
        if not (tp < entry < sl):
            return False, "sell SL/TP side"

    # SL/TP distance from entry
    if msd > 0 and (abs(entry - sl) < msd or abs(entry - tp) < msd):
        return False, "SL/TP too close to entry"

    tick = mt5.symbol_info_tick(symbol)
    if not tick:
        return False, "no tick"

    # pending price distance from market considering stops level
    if order_type == mt5.ORDER_TYPE_BUY_LIMIT:
        if msd > 0 and not (entry < tick.ask - msd):
            return False, "entry too close/above ask"
    else:  # SELL_LIMIT
        if msd > 0 and not (entry > tick.bid + msd):
            return False, "entry too close/below bid"

    # optional freeze level (only if broker reports it)
    fl = m["freeze_level"] * m["point"]
    if fl > 0:
        if order_type == mt5.ORDER_TYPE_BUY_LIMIT and (tick.ask - entry) < fl:
            return False, "freeze level (buy)"
        if order_type == mt5.ORDER_TYPE_SELL_LIMIT and (entry - tick.bid) < fl:
            return False, "freeze level (sell)"

    return True, ""

# === HOUSEKEEPING ===
def cancel_unfilled_orders_on_tp(symbol, current_tick):
    orders = mt5.orders_get(symbol=symbol)
    if not orders:
        return

    for order in orders:
        tp = getattr(order, "tp", 0.0)
        if not tp or tp == 0.0:
            continue

        price_now = current_tick.ask if order.type == mt5.ORDER_TYPE_BUY_LIMIT else current_tick.bid
        if order.type == mt5.ORDER_TYPE_BUY_LIMIT and price_now >= tp:
            req = {"action": mt5.TRADE_ACTION_REMOVE, "order": order.ticket}
            res = mt5.order_send(req)
            if res.retcode == mt5.TRADE_RETCODE_DONE:
                print(f"[{symbol}] ❌ Cancelled unfilled BUY_LIMIT — price hit TP without entry")
        elif order.type == mt5.ORDER_TYPE_SELL_LIMIT and price_now <= tp:
            req = {"action": mt5.TRADE_ACTION_REMOVE, "order": order.ticket}
            res = mt5.order_send(req)
            if res.retcode == mt5.TRADE_RETCODE_DONE:
                print(f"[{symbol}] ❌ Cancelled unfilled SELL_LIMIT — price hit TP without entry")

# === STRATEGY: IFVG DETECTION (7-candle lookback) ===
def is_valid_gap(symbol, high, low):
    m = symbol_meta(symbol)
    if not m:
        return False
    ts = m["tick_size"]
    return (high - low) >= (MIN_GAP_TICKS * ts)

def detect_ifvg(symbol, df):
    """
    Uses lookback window of 7 candles:
      c1 = df.iloc[-7]
      c3 = df.iloc[-5]  (two candles after c1)
      c4 = df.iloc[-2]  (latest closed candle)
    - Bullish FVG: c1.high < c3.low
        If c4.close < fvg_low => disrespect => SELL setup at fvg_low
    - Bearish FVG: c1.low  > c3.high
        If c4.close > fvg_high => disrespect => BUY setup at fvg_high
    """
    setups = []
    if len(df) < REQUIRED_BARS:
        return setups

    c1 = df.iloc[-MIN_LOOKBACK]        # -7
    c3 = df.iloc[-(MIN_LOOKBACK - 2)]  # -5
    c4 = df.iloc[-2]                    # last closed bar

    # Bullish FVG
    if c1['high'] < c3['low']:
        fvg_low = c1['high']
        fvg_high = c3['low']
        if not is_valid_gap(symbol, fvg_high, fvg_low):
            return setups
        print(f"[{symbol}] Bullish FVG (7-back): {fvg_low} → {fvg_high}")
        if c4['close'] < fvg_low:
            entry = fvg_low
            swing_high = max(df.iloc[-MIN_LOOKBACK:-1]['high'])  # window from c1..latest-1
            sl = swing_high
            # tp is provisional; final TP will be recalculated after spread-adjust + rounding
            tp = entry - RR_TARGET * (sl - entry)
            setups.append(("sell", symbol, entry, sl, tp))
        else:
            print(f"[{symbol}] Waiting for bullish FVG disrespect...")

    # Bearish FVG
    elif c1['low'] > c3['high']:
        fvg_high = c1['low']
        fvg_low = c3['high']
        if not is_valid_gap(symbol, fvg_high, fvg_low):
            return setups
        print(f"[{symbol}] Bearish FVG (7-back): {fvg_high} → {fvg_low}")
        if c4['close'] > fvg_high:
            entry = fvg_high
            swing_low = min(df.iloc[-MIN_LOOKBACK:-1]['low'])
            sl = swing_low
            # provisional TP; will be recomputed after spread-adjust + rounding
            tp = entry + RR_TARGET * (entry - sl)
            setups.append(("buy", symbol, entry, sl, tp))
        else:
            print(f"[{symbol}] Waiting for bearish FVG disrespect...")

    return setups

# === ORDER PLACEMENT ===
def place_limit_order(direction, symbol, entry, sl, tp):
    if not mt5.symbol_select(symbol, True):
        print(f"[{symbol}] Not available or hidden in Market Watch. Skipping.")
        return

    # Skip if a position already exists on this symbol
    if has_open_position(symbol):
        print(f"[{symbol}] Position already open. Skipping new order.")
        return

    # --- SPREAD-ADJUST SL (so risk accounts for Bid/Ask hit logic) ---
    tick = mt5.symbol_info_tick(symbol)
    if tick:
        spread = tick.ask - tick.bid
        if direction == "buy":
            sl -= spread   # SL further down for BUYs
        else:  # "sell"
            sl += spread   # SL further up for SELLs

    # Normalize entry & SL after spread adjustment
    entry = round_price_to_tick(symbol, entry)
    sl    = round_price_to_tick(symbol, sl)

    # === RECOMPUTE TP TO ENFORCE EXACT 2R (after SL adjustment) ===
    if direction == "buy":
        tp = entry + RR_TARGET * (entry - sl)
    else:  # sell
        tp = entry - RR_TARGET * (sl - entry)
    tp = round_price_to_tick(symbol, tp)

    # Side sanity
    if direction == "buy" and not (sl < entry < tp):
        print(f"[{symbol}] ❌ Invalid BUY sides (sl<{sl} entry<{entry} tp>{tp}).")
        return
    if direction == "sell" and not (tp < entry < sl):
        print(f"[{symbol}] ❌ Invalid SELL sides.")
        return

    # Remove existing limits (replace) with robust attribute access
    existing = mt5.orders_get(symbol=symbol) or []
    EPS = 1e-10

    def _ord_price(o):
        return getattr(o, "price_open", getattr(o, "price", None))

    for o in existing:
        if o.type in (mt5.ORDER_TYPE_BUY_LIMIT, mt5.ORDER_TYPE_SELL_LIMIT):
            op = _ord_price(o)
            osl = getattr(o, "sl", None)
            otp = getattr(o, "tp", None)

            if op is not None and osl is not None and otp is not None:
                if abs(op - entry) < EPS and abs(osl - sl) < EPS and abs(otp - tp) < EPS:
                    print(f"[{symbol}] ⏸️ Identical pending exists (ticket {o.ticket}). Skip.")
                    return

            print(f"[{symbol}] 🔁 Removing existing limit order {o.ticket}...")
            mt5.order_send({"action": mt5.TRADE_ACTION_REMOVE, "order": o.ticket})

    order_type = mt5.ORDER_TYPE_BUY_LIMIT if direction == "buy" else mt5.ORDER_TYPE_SELL_LIMIT

    # Constraints (min distances, market relation)
    ok, why = check_order_constraints(symbol, order_type, entry, sl, tp)
    if not ok:
        print(f"[{symbol}] ❌ Constraints fail ({why}). Skipping.")
        return

    # Calculate lot size **after** spread-adjusted SL (and final rounding)
    lot_size = calculate_dynamic_lot(symbol, entry, sl)
    if lot_size <= 0:
        print(f"[{symbol}] ❌ Invalid lot size. Skipping.")
        return

    req = {
        "action": mt5.TRADE_ACTION_PENDING,
        "symbol": symbol,
        "volume": lot_size,
        "type": order_type,
        "price": entry,
        "sl": sl,
        "tp": tp,
        "deviation": DEVIATION,
        "magic": MAGIC,
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_RETURN,
        "comment": f"{direction.upper()} IFVG",
    }
    res = mt5.order_send(req)
    if res.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"[{symbol}] ❌ Order FAILED: {res.retcode} ({getattr(res, 'comment', '')})")
    else:
        # Quick RR check for your logs
        rr = (abs(tp - entry) / max(1e-12, abs(entry - sl)))
        print(f"[{symbol}] ✅ {direction.upper()} LIMIT @ {entry} | {lot_size} lots (SL spread-adjusted) | RR≈{rr:.2f}")

# === BOT LOOP ===
def run_bot(trade_filter="both"):
    """
    trade_filter: 'sell' | 'buy' | 'both'
    """
    print("🚀 IFVG bot running — tick-based sizing, spread-adjusted SL, exact 2R TP, stop checks, order replacement")
    print(f"🎛️ Trade filter: {trade_filter.upper()}")
    try:
        while True:
            for symbol in SYMBOLS:
                try:
                    if not mt5.symbol_select(symbol, True):
                        print(f"[{symbol}] Not available or hidden in Market Watch. Skipping.")
                        continue

                    df = get_candles(symbol)
                    if df.empty or len(df) < REQUIRED_BARS:
                        continue

                    tick = mt5.symbol_info_tick(symbol)
                    if tick:
                        cancel_unfilled_orders_on_tp(symbol, tick)

                    setups = detect_ifvg(symbol, df)
                    for direction, sym, entry, sl, _tp in setups:
                        if trade_filter == "both" or trade_filter == direction:
                            place_limit_order(direction, sym, entry, sl, _tp)
                        else:
                            print(f"[{sym}] Skipped {direction} (filter={trade_filter})")
                except Exception as e:
                    print(f"[{symbol}] ERROR: {e}")
            time.sleep(SLEEP_SEC)
    except KeyboardInterrupt:
        print("Bot manually stopped.")

# === MENU / MAIN ===
def main():
    while True:
        print("\n=== IFVG Bot ===")
        print("1) Run")
        print("2) Exit")
        choice = input("Select an option: ").strip()

        if choice == "2":
            print("Exiting.")
            sys.exit(0)
        elif choice == "1":
            if not mt5.initialize():
                raise RuntimeError(f"MT5 init failed: {mt5.last_error()}")

            print("\nTrade Direction:")
            print("1) Only SELLs")
            print("2) Only BUYs")
            print("3) Both")
            tchoice = input("Select trade direction: ").strip()

            trade_filter = "both"
            if tchoice == "1":
                trade_filter = "sell"
            elif tchoice == "2":
                trade_filter = "buy"
            elif tchoice != "3":
                print("Invalid choice. Defaulting to BOTH.")

            try:
                run_bot(trade_filter=trade_filter)
            finally:
                mt5.shutdown()
        else:
            print("Invalid option. Please choose 1 or 2.")

if __name__ == "__main__":
    main()
