import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import csv
import os
import atexit
import time
from collections import defaultdict

# ===================== CONFIG =====================
SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY", "GBPJPY", "GER40.cash", "USDCAD", "USDCHF"]
TIMEFRAME = mt5.TIMEFRAME_M5
MAGIC = 20250806
DEVIATION = 20
RISK_PERCENT = 1.0      # % of balance per trade
RR_TARGET = 2.0         # enforce exact 2R
SLEEP_SEC = 30
MIN_GAP_TICKS = 2       # ignore micro-gaps
MIN_LOOKBACK = 7        # look back 7 candles
REQUIRED_BARS = MIN_LOOKBACK + 3  # enough for [-7], [-5], [-2]
HTF_FVG_TIMEFRAMES = [
    mt5.TIMEFRAME_D1, mt5.TIMEFRAME_H4, mt5.TIMEFRAME_H1,
    mt5.TIMEFRAME_M30, mt5.TIMEFRAME_M15
]
HTF_FVG_LOOKBACK = 500  # candles per HTF to scan for PD arrays
ORDERBLOCK_BODY_FACTOR = 0.5  # use mid of OB body to define zone depth

# === SESSIONS (SERVER TIME) ===
ASIA_START_H = 3
ASIA_END_H   = 9
LDN_START_H  = 9
LDN_END_H    = 12

# ============== LOGGING (per-symbol CSV) ==============
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

class CsvLogger:
    def __init__(self, symbol):
        date_tag = pd.Timestamp.utcnow().strftime("%Y%m%d")
        self.path = os.path.join(LOG_DIR, f"{date_tag}_{symbol}.csv")
        self.file = open(self.path, "a", newline="", encoding="utf-8")
        self.writer = csv.writer(self.file)
        if self.file.tell() == 0:
            self.writer.writerow([
                "ts_utc","symbol","event","direction","server_time",
                "price_bid","price_ask","asia_high","asia_low","detail"
            ])
            self.file.flush()
    def log(self, symbol, event, direction, now_dt, tick, asia_high, asia_low, detail):
        self.writer.writerow([
            pd.Timestamp.utcnow().isoformat(),
            symbol,
            event,
            direction or "",
            now_dt.isoformat() if now_dt is not None else "",
            f"{getattr(tick,'bid',np.nan):.6f}" if tick else "",
            f"{getattr(tick,'ask',np.nan):.6f}" if tick else "",
            f"{asia_high:.6f}" if asia_high else "",
            f"{asia_low:.6f}" if asia_low else "",
            detail
        ])
        self.file.flush()  # immediate durability
    def close(self):
        try:
            self.file.flush()
            self.file.close()
        except Exception:
            pass

_LOGGERS = {}
def get_logger(symbol):
    if symbol not in _LOGGERS:
        _LOGGERS[symbol] = CsvLogger(symbol)
    return _LOGGERS[symbol]

def _close_all_logs():
    for lg in list(_LOGGERS.values()):
        lg.close()
atexit.register(_close_all_logs)

# ===================== SYMBOL / PRICING HELPERS =====================
def symbol_meta(symbol):
    info = mt5.symbol_info(symbol)
    if not info:
        return None
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
        "tick_value": float(tick_value),
        "vol_min": float(g(info, "volume_min", 0.01)),
        "vol_max": float(g(info, "volume_max", 100.0)),
        "vol_step": float(g(info, "volume_step", 0.01)),
        "stops_level": int(g(info, "trade_stops_level", 0) or g(info, "stops_level", 0)),
        "freeze_level": freeze_level_pts,
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

def effective_tick_value(symbol):
    m = symbol_meta(symbol)
    if not m:
        return 0.0
    tv = m["tick_value"]
    if tv and tv > 0:
        return tv
    cs = m["contract_size"] or 1.0
    ts = m["tick_size"] or m["point"] or 0.0
    return cs * ts if ts > 0 else 0.0

# ============================== DATA ===============================
def get_candles(symbol, count=REQUIRED_BARS+5, timeframe=TIMEFRAME):
    data = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
    return pd.DataFrame(data) if data is not None else pd.DataFrame()

def get_candles_range(symbol, timeframe, time_from, time_to):
    data = mt5.copy_rates_range(symbol, timeframe, time_from, time_to)
    return pd.DataFrame(data) if data is not None else pd.DataFrame()

def has_open_position(symbol):
    poss = mt5.positions_get(symbol=symbol) or []
    return len(poss) > 0

# ========================= RISK / SIZING ===========================
def calculate_dynamic_lot(symbol, entry, sl, risk_percent=RISK_PERCENT):
    acc = mt5.account_info()
    m = symbol_meta(symbol)
    if not acc or not m:
        print(f"[{symbol}] Could not retrieve account/symbol info.")
        return 0.0
    entry = round_price_to_tick(symbol, entry)
    sl    = round_price_to_tick(symbol, sl)
    distance = abs(entry - sl)
    ts, tv = m["tick_size"], effective_tick_value(symbol)
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
    print(f"[{symbol}] SIZING  target {risk_percent:.2f}%=${risk_amount:.2f} | ticks={ticks:.1f} | tick=${tv:.5f} | lot={lots} | ≈${expected_risk:.2f}")
    return lots

# ======================= SAFE SEND & CONSTRAINTS ===================
def check_order_constraints(symbol, order_type, entry, sl, tp):
    m = symbol_meta(symbol)
    if not m:
        return False, "no symbol meta"

    entry = round_price_to_tick(symbol, entry)
    sl    = round_price_to_tick(symbol, sl)
    tp    = round_price_to_tick(symbol, tp)

    tick = mt5.symbol_info_tick(symbol)
    if not tick:
        return False, "no tick"

    msd_pts = min_stop_distance_price(symbol)  # price distance

    # Sides sanity
    if order_type == mt5.ORDER_TYPE_BUY_LIMIT and not (sl < entry < tp):
        return False, f"buy sides invalid (sl={sl} entry={entry} tp={tp})"
    if order_type == mt5.ORDER_TYPE_SELL_LIMIT and not (tp < entry < sl):
        return False, f"sell sides invalid (tp={tp} entry={entry} sl={sl})"

    # Distances to entry for SL and TP
    if msd_pts > 0:
        if abs(entry - sl) < msd_pts:
            return False, f"SL too close: |entry-sl|={abs(entry-sl):.6f} < msd={msd_pts:.6f}"
        if abs(entry - tp) < msd_pts:
            return False, f"TP too close: |entry-tp|={abs(entry-tp):.6f} < msd={msd_pts:.6f}"

    # Pending price must be on correct side of book (even if msd==0)
    if order_type == mt5.ORDER_TYPE_BUY_LIMIT:
        if not (entry < tick.ask):
            return False, f"BUY_LIMIT must be below Ask (entry={entry} ask={tick.ask})"
    else:
        if not (entry > tick.bid):
            return False, f"SELL_LIMIT must be above Bid (entry={entry} bid={tick.bid})"

    # Freeze level guard (if broker uses it)
    fl = m["freeze_level"] * m["point"]
    if fl > 0:
        if order_type == mt5.ORDER_TYPE_BUY_LIMIT and (tick.ask - entry) < fl:
            return False, f"freeze(buy): ask-entry={tick.ask-entry:.6f} < {fl:.6f}"
        if order_type == mt5.ORDER_TYPE_SELL_LIMIT and (entry - tick.bid) < fl:
            return False, f"freeze(sell): entry-bid={entry-tick.bid:.6f} < {fl:.6f}"

    return True, ""

def safe_order_send(symbol, req, logger, now_dt, tick, direction):
    # Round prices to tick before check/send
    for k in ("price","sl","tp"):
        if k in req and req[k] is not None:
            req[k] = round_price_to_tick(symbol, req[k])

    check = mt5.order_check(req)
    if check is None:
        code, detail = mt5.last_error()
        msg = f"order_check=None last_error={code}:{detail}"
        print(f"[{symbol}] ❌ {msg}")
        logger.log(symbol, "order_check_none", direction, now_dt, tick, None, None, msg)
        return None

    if check.retcode != mt5.TRADE_RETCODE_DONE:
        msg = f"CHECK_FAIL {check.retcode} {getattr(check,'comment','')}".strip()
        print(f"[{symbol}] ❌ {msg}")
        logger.log(symbol, "order_check_fail", direction, now_dt, tick, None, None, msg)
        return None

    res = mt5.order_send(req)
    if res is None:
        code, detail = mt5.last_error()
        msg = f"order_send=None last_error={code}:{detail}"
        print(f"[{symbol}] ❌ {msg}")
        logger.log(symbol, "order_send_none", direction, now_dt, tick, None, None, msg)
        return None

    return res

# =========================== HOUSEKEEPING ==========================
def cancel_unfilled_orders_on_invalid_rr(symbol, current_tick):
    """
    Optional invalidation: if price moves >1R beyond SL side before fill, cancel.
    """
    orders = mt5.orders_get(symbol=symbol) or []
    for o in orders:
        if o.type not in (mt5.ORDER_TYPE_BUY_LIMIT, mt5.ORDER_TYPE_SELL_LIMIT):
            continue
        entry = getattr(o, "price_open", getattr(o, "price", 0.0))
        sl, tp = getattr(o, "sl", 0.0), getattr(o, "tp", 0.0)
        if not (entry and sl and tp and current_tick):
            continue
        price_now = current_tick.bid if o.type == mt5.ORDER_TYPE_SELL_LIMIT else current_tick.ask
        r = abs(tp - entry)
        if o.type == mt5.ORDER_TYPE_BUY_LIMIT and price_now <= sl - r:
            mt5.order_send({"action": mt5.TRADE_ACTION_REMOVE, "order": o.ticket})
            print(f"[{symbol}] ❌ Cancel BUY_LIMIT — invalidated (>1R past SL)")
        if o.type == mt5.ORDER_TYPE_SELL_LIMIT and price_now >= sl + r:
            mt5.order_send({"action": mt5.TRADE_ACTION_REMOVE, "order": o.ticket})
            print(f"[{symbol}] ❌ Cancel SELL_LIMIT — invalidated (>1R past SL)")

# ====================== IFVG DETECTION (7-back) ====================
def is_valid_gap(symbol, high, low):
    m = symbol_meta(symbol)
    if not m:
        return False
    ts = m["tick_size"]
    return (high - low) >= (MIN_GAP_TICKS * ts)

def detect_ifvg(symbol, df):
    setups = []
    if len(df) < REQUIRED_BARS:
        return setups
    c1 = df.iloc[-MIN_LOOKBACK]        # -7
    c3 = df.iloc[-(MIN_LOOKBACK - 2)]  # -5
    c4 = df.iloc[-2]                    # last closed bar

    # Bullish FVG (potential SELL on disrespect)
    if c1['high'] < c3['low']:
        fvg_low = c1['high']
        fvg_high = c3['low']
        if not is_valid_gap(symbol, fvg_high, fvg_low):
            return setups
        if c4['close'] < fvg_low:
            entry = fvg_low
            swing_high = max(df.iloc[-MIN_LOOKBACK:-1]['high'])
            sl = swing_high
            tp = entry - RR_TARGET * (sl - entry)
            setups.append(("sell", symbol, entry, sl, tp))

    # Bearish FVG (potential BUY on disrespect)
    elif c1['low'] > c3['high']:
        fvg_high = c1['low']
        fvg_low = c3['high']
        if not is_valid_gap(symbol, fvg_high, fvg_low):
            return setups
        if c4['close'] > fvg_high:
            entry = fvg_high
            swing_low = min(df.iloc[-MIN_LOOKBACK:-1]['low'])
            sl = swing_low
            tp = entry + RR_TARGET * (entry - sl)
            setups.append(("buy", symbol, entry, sl, tp))

    return setups

# =================== HTF PD ARRAYS (FVGs & Order Blocks) ===================
def find_htf_fvgs(symbol):
    """
    Returns list of dicts: {"tf": timeframe, "dir": "bull"/"bear", "low": x, "high": y}
    """
    zones = []
    for tf in HTF_FVG_TIMEFRAMES:
        df = get_candles(symbol, count=HTF_FVG_LOOKBACK, timeframe=tf)
        if df.empty or len(df) < 10:
            continue
        highs = df['high'].values
        lows  = df['low'].values
        for i in range(len(df) - 2):
            h0, l0 = highs[i], lows[i]
            h2, l2 = highs[i+2], lows[i+2]
            # Bullish FVG if candle0.high < candle2.low (gap between them)
            if h0 < l2:
                low, high = h0, l2
                if is_valid_gap(symbol, high, low):
                    zones.append({"tf": tf, "dir": "bull", "low": low, "high": high})
            # Bearish FVG if candle0.low > candle2.high
            if l0 > h2:
                low, high = h2, l0
                if is_valid_gap(symbol, high, low):
                    zones.append({"tf": tf, "dir": "bear", "low": low, "high": high})
    return zones

def find_simple_orderblocks(symbol):
    """
    Very lightweight OB heuristic.
    Returns list of dicts: {"tf": timeframe, "dir":"bull"/"bear","low":x,"high":y}
    """
    zones = []
    for tf in HTF_FVG_TIMEFRAMES:
        df = get_candles(symbol, count=HTF_FVG_LOOKBACK, timeframe=tf)
        if df.empty or len(df) < 30:
            continue
        body = (df['close'] - df['open']).abs().rolling(20).median().fillna(method='bfill')
        for i in range(1, len(df)):
            o, c, h, l = df['open'].iloc[i], df['close'].iloc[i], df['high'].iloc[i], df['low'].iloc[i]
            if body.iloc[i] == 0:
                continue
            # Large up displacement
            if (c - o) > body.iloc[i]:
                j = i - 1
                while j >= 0 and df['close'].iloc[j] >= df['open'].iloc[j]:
                    j -= 1
                if j >= 0:
                    ob_low = min(df['low'].iloc[j], df['open'].iloc[j])
                    ob_high = max(df['open'].iloc[j], df['close'].iloc[j])
                    mid = ob_low + ORDERBLOCK_BODY_FACTOR * (ob_high - ob_low)
                    zones.append({"tf": tf, "dir":"bull", "low": ob_low, "high": mid})
            # Large down displacement
            if (o - c) > body.iloc[i]:
                j = i - 1
                while j >= 0 and df['close'].iloc[j] <= df['open'].iloc[j]:
                    j -= 1
                if j >= 0:
                    ob_low = min(df['open'].iloc[j], df['close'].iloc[j])
                    ob_high = max(df['high'].iloc[j], df['open'].iloc[j])
                    mid = ob_high - ORDERBLOCK_BODY_FACTOR * (ob_high - ob_low)
                    zones.append({"tf": tf, "dir":"bear", "low": mid, "high": ob_high})
    return zones

def price_touches_zone(direction, price, zone):
    """direction filters: need 'bear' zone for sells, 'bull' for buys"""
    if direction == "sell" and zone["dir"] != "bear":
        return False
    if direction == "buy" and zone["dir"] != "bull":
        return False
    return zone["low"] <= price <= zone["high"]

# ======================= PREVIOUS H/L LEVELS =======================
def get_prev_day_week_month_levels(symbol, now_dt):
    """
    Returns dict with PDH/PDL, PWH/PWL, PMH/PML.
    """
    res = {}
    # Helper to get OHLC in time window
    def _hl_for_range(start_ts, end_ts, tf=mt5.TIMEFRAME_H1):
        df = get_candles_range(symbol, tf, int(start_ts.timestamp()), int(end_ts.timestamp()))
        if df.empty:
            return None, None
        return float(df['high'].max()), float(df['low'].min())

    # Previous Day
    day = now_dt.normalize()
    prev_day_start = day - pd.Timedelta(days=1)
    prev_day_end = day
    res["PDH"], res["PDL"] = _hl_for_range(prev_day_start, prev_day_end)

    # Previous Week (Mon-Sun assumption based on server)
    week_start = (day - pd.Timedelta(days=day.weekday()))  # this Monday 00:00
    prev_week_start = week_start - pd.Timedelta(weeks=1)
    prev_week_end = week_start
    res["PWH"], res["PWL"] = _hl_for_range(prev_week_start, prev_week_end, tf=mt5.TIMEFRAME_H4)

    # Previous Month
    first_of_month = pd.Timestamp(year=day.year, month=day.month, day=1)
    prev_month_end = first_of_month
    prev_month_start = (first_of_month - pd.Timedelta(days=1)).replace(day=1)
    res["PMH"], res["PML"] = _hl_for_range(prev_month_start, prev_month_end, tf=mt5.TIMEFRAME_H4)

    return res

def swept_prev_levels(direction, tick, prev_levels, buffer_pts):
    """
    After Asia sweep, allow one more 'sweep' of PMH/PDH/PWH (for sells) or PML/PDL/PWL (for buys).
    Using 1-tick buffer.
    """
    if not tick or not prev_levels:
        return False, ""
    if direction == "sell":
        highs = [prev_levels.get(k) for k in ["PMH","PWH","PDH"] if prev_levels.get(k)]
        for lvl in highs:
            if tick.ask > lvl + buffer_pts:
                return True, f"swept_prev_high@{lvl:.5f}"
    else:
        lows = [prev_levels.get(k) for k in ["PML","PWL","PDL"] if prev_levels.get(k)]
        for lvl in lows:
            if tick.bid < lvl - buffer_pts:
                return True, f"swept_prev_low@{lvl:.5f}"
    return False, ""

# ======================= ENTRY PLACEMENT ===========================
def place_limit_order(direction, symbol, entry, sl, tp, logger, now_dt, tick):
    if not mt5.symbol_select(symbol, True):
        print(f"[{symbol}] Not available or hidden. Skipping.")
        logger.log(symbol, "order_skip_symbol_hidden", direction, now_dt, tick, None, None, "")
        return

    if has_open_position(symbol):
        print(f"[{symbol}] Position already open. Skipping.")
        logger.log(symbol, "order_skip_pos_open", direction, now_dt, tick, None, None, "")
        return

    # Spread-adjust SL slightly (keeps broker-distance safer)
    if tick:
        spread = tick.ask - tick.bid
        if direction == "buy":
            sl -= spread
        else:
            sl += spread

    entry = round_price_to_tick(symbol, entry)
    sl    = round_price_to_tick(symbol, sl)

    # Recompute exact 2R TP from (entry, sl)
    if direction == "buy":
        tp = entry + RR_TARGET * (entry - sl)
    else:
        tp = entry - RR_TARGET * (sl - entry)
    tp = round_price_to_tick(symbol, tp)

    # Early sanity
    msd = min_stop_distance_price(symbol)
    if msd > 0 and abs(entry - sl) < msd:
        print(f"[{symbol}] ❌ SL too close after spread ({msd}).")
        logger.log(symbol, "order_fail_sl_distance", direction, now_dt, tick, None, None, f"msd={msd}")
        return

    if direction == "buy" and not (sl < entry < tp):
        print(f"[{symbol}] ❌ Invalid BUY sides.")
        logger.log(symbol, "order_fail_side", direction, now_dt, tick, None, None, "")
        return
    if direction == "sell" and not (tp < entry < sl):
        print(f"[{symbol}] ❌ Invalid SELL sides.")
        logger.log(symbol, "order_fail_side", direction, now_dt, tick, None, None, "")
        return

    # Remove existing limits
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
                    print(f"[{symbol}] ⏸️ Identical pending exists (ticket {o.ticket}).")
                    logger.log(symbol, "order_skip_identical_pending", direction, now_dt, tick, None, None, f"ticket={o.ticket}")
                    return
            mt5.order_send({"action": mt5.TRADE_ACTION_REMOVE, "order": o.ticket})

    order_type = mt5.ORDER_TYPE_BUY_LIMIT if direction == "buy" else mt5.ORDER_TYPE_SELL_LIMIT

    ok, why = check_order_constraints(symbol, order_type, entry, sl, tp)
    if not ok:
        print(f"[{symbol}] ❌ Constraints fail ({why}).")
        logger.log(symbol, "order_fail_constraints", direction, now_dt, tick, None, None, why)
        return

    lot_size = calculate_dynamic_lot(symbol, entry, sl)
    if lot_size <= 0:
        print(f"[{symbol}] ❌ Invalid lot size.")
        logger.log(symbol, "order_fail_lot", direction, now_dt, tick, None, None, "")
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
        "type_filling": mt5.ORDER_FILLING_RETURN,  # fine for pendings; many brokers ignore for pendings anyway
        "comment": f"{direction.upper()} IFVG (Asia sweep + HTF PD/Sweep)",
    }

    res = safe_order_send(symbol, req, logger, now_dt, tick, direction)
    if not res:
        # safe_order_send already logged the precise reason
        return

    if res.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"[{symbol}] ❌ Order FAILED: {res.retcode} ({getattr(res, 'comment','')})")
        logger.log(symbol, "order_fail_send", direction, now_dt, tick, None, None, f"{res.retcode}:{getattr(res,'comment','')}")
        return

    rr = (abs(tp - entry) / max(1e-12, abs(entry - sl)))
    print(f"[{symbol}] ✅ {direction.upper()} LIMIT @ {entry} | {lot_size} lots | RR≈{rr:.2f}")
    logger.log(symbol, "order_sent", direction, now_dt, tick, None, None, f"entry={entry},sl={sl},tp={tp},lot={lot_size},RR≈{rr:.2f}")

# ===================== TIME / SESSION HELPERS ======================
def to_ts(sec):
    return pd.to_datetime(sec, unit='s')

def current_server_dt(symbol="EURUSD"):
    tick = mt5.symbol_info_tick(symbol)
    if not tick:
        return None
    return to_ts(tick.time)

def same_day(a, b):
    return (a.year==b.year) and (a.month==b.month) and (a.day==b.day)

def in_london_session(dt):
    h = dt.hour
    return (h >= LDN_START_H) and (h < LDN_END_H)

def asia_window_for_day(day_dt):
    start = pd.Timestamp(year=day_dt.year, month=day_dt.month, day=day_dt.day, hour=ASIA_START_H)
    end   = pd.Timestamp(year=day_dt.year, month=day_dt.month, day=day_dt.day, hour=ASIA_END_H)
    return int(start.timestamp()), int(end.timestamp())

# ===================== ASIA RANGE / SWEEP LOGIC ====================
daily_state = defaultdict(lambda: {
    "date": None,
    "asia_high": None,
    "asia_low": None,
    "direction_committed": None,   # "buy" or "sell"
    "manipulation_confirmed": False,  # gate after Asia sweep
    "manip_reason": "",               # why manipulation confirmed
    "htf_pd_zones": [],               # cached zones for quick checking
    "prev_levels": {},                # cached prev H/L levels
})

def refresh_daily_asia_range(symbol, now_dt):
    st = daily_state[symbol]
    # Reset if date changed
    if st["date"] is None or not same_day(st["date"], now_dt):
        st.update({
            "date": now_dt.normalize(),
            "asia_high": None,
            "asia_low": None,
            "direction_committed": None,
            "manipulation_confirmed": False,
            "manip_reason": "",
            "htf_pd_zones": [],
            "prev_levels": {},
        })
    # Only compute once per day after Asia session complete
    if now_dt.hour < ASIA_END_H:
        return
    if st["asia_high"] is None or st["asia_low"] is None:
        t_from, t_to = asia_window_for_day(now_dt)
        asia_df = get_candles_range(symbol, TIMEFRAME, t_from, t_to)
        if asia_df.empty:
            return
        st["asia_high"] = float(asia_df["high"].max())
        st["asia_low"]  = float(asia_df["low"].min())
        print(f"[{symbol}] Asia range set: HIGH={st['asia_high']} LOW={st['asia_low']}")

def detect_sweep_commit_direction(symbol, now_dt, tick=None):
    if not in_london_session(now_dt):
        return
    st = daily_state[symbol]
    if st["asia_high"] is None or st["asia_low"] is None:
        return
    if st["direction_committed"] is not None:
        return
    if tick is None:
        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            return
    m = symbol_meta(symbol)
    ts = (m["tick_size"] if m else 0.0) or (m["point"] if m else 0.0)
    if tick.bid < st["asia_low"] - ts:
        st["direction_committed"] = "buy"
        get_logger(symbol).log(symbol, "sweep_commit", "buy", now_dt, tick, st["asia_high"], st["asia_low"], "AsiaLow swept")
        print(f"[{symbol}] ✅ LIVE SWEEP  Bid {tick.bid:.5f} < Asia LOW {st['asia_low']:.5f} → BUY bias.")
    elif tick.ask > st["asia_high"] + ts:
        st["direction_committed"] = "sell"
        get_logger(symbol).log(symbol, "sweep_commit", "sell", now_dt, tick, st["asia_high"], st["asia_low"], "AsiaHigh swept")
        print(f"[{symbol}] ✅ LIVE SWEEP  Ask {tick.ask:.5f} > Asia HIGH {st['asia_high']:.5f} → SELL bias.")

# ======================= MANIPULATION GATE =========================
def ensure_pd_arrays_and_prev_levels(symbol, now_dt):
    st = daily_state[symbol]
    if not st["htf_pd_zones"]:
        fvgs = find_htf_fvgs(symbol)
        obs  = find_simple_orderblocks(symbol)
        st["htf_pd_zones"] = fvgs + obs
    if not st["prev_levels"]:
        st["prev_levels"] = get_prev_day_week_month_levels(symbol, now_dt)

def check_manipulation_gate(symbol, now_dt, tick):
    """
    After Asia sweep and direction commitment:
      - Confirm manipulation if price touches a matching HTF PD array zone, OR
      - Confirm if price sweeps a prev TF high/low on the correct side.
    """
    st = daily_state[symbol]
    if st["manipulation_confirmed"] or st["direction_committed"] is None:
        return
    ensure_pd_arrays_and_prev_levels(symbol, now_dt)

    m = symbol_meta(symbol)
    ts = (m["tick_size"] if m else 0.0) or (m["point"] if m else 0.0)

    # 1) Touch PD array zone (bearish for sells with Ask; bullish for buys with Bid)
    px = tick.ask if st["direction_committed"] == "sell" else tick.bid
    for z in st["htf_pd_zones"]:
        if price_touches_zone(st["direction_committed"], px, z):
            st["manipulation_confirmed"] = True
            tf_label = {mt5.TIMEFRAME_D1:"D1", mt5.TIMEFRAME_H4:"H4", mt5.TIMEFRAME_H1:"H1",
                        mt5.TIMEFRAME_M30:"M30", mt5.TIMEFRAME_M15:"M15"}.get(z["tf"], str(z["tf"]))
            reason = f"touch_{z['dir']}_PD({tf_label})[{z['low']:.5f},{z['high']:.5f}]"
            st["manip_reason"] = reason
            get_logger(symbol).log(symbol, "manip_confirmed_pd", st["direction_committed"], now_dt, tick, st["asia_high"], st["asia_low"], reason)
            print(f"[{symbol}] 🎯 Manipulation confirmed via PD array: {reason}")
            return

    # 2) Sweep a previous TF H/L
    ok, why = swept_prev_levels(st["direction_committed"], tick, st["prev_levels"], ts)
    if ok:
        st["manipulation_confirmed"] = True
        st["manip_reason"] = why
        get_logger(symbol).log(symbol, "manip_confirmed_prev_sweep", st["direction_committed"], now_dt, tick, st["asia_high"], st["asia_low"], why)
        print(f"[{symbol}] 🎯 Manipulation confirmed via prev sweep: {why}")

# =========================== BOT LOOP ==============================
def run_bot():
    print("🚀 Asia Sweep → (HTF PD touch OR Prev H/L sweep) → IFVG entries — London 09:00–12:00")
    try:
        while True:
            for symbol in SYMBOLS:
                logger = get_logger(symbol)
                try:
                    if not mt5.symbol_select(symbol, True):
                        print(f"[{symbol}] Not available or hidden in Market Watch. Skipping.")
                        logger.log(symbol, "symbol_unavailable", None, pd.Timestamp.utcnow(), None, None, None, "")
                        continue

                    now_dt = current_server_dt(symbol)
                    if now_dt is None:
                        continue
                    tick = mt5.symbol_info_tick(symbol)
                    if not tick:
                        continue

                    print(f"[{symbol}] ⏱  Server {now_dt.strftime('%Y-%m-%d %H:%M:%S')} | Bid {tick.bid:.6f} | Ask {tick.ask:.6f}")

                    # Refresh/set Asia range (after 09:00 server)
                    refresh_daily_asia_range(symbol, now_dt)

                    st = daily_state[symbol]
                    if st["asia_high"] is None or st["asia_low"] is None:
                        print(f"[{symbol}] ℹ️  Asia range not ready (waiting for 09:00 server).")
                        logger.log(symbol, "asia_not_ready", None, now_dt, tick, None, None, "")
                    else:
                        print(f"[{symbol}] 📏 Asia H={st['asia_high']:.6f}  L={st['asia_low']:.6f}")

                    # Cancel invalidated unfilled orders
                    cancel_unfilled_orders_on_invalid_rr(symbol, tick)

                    # Session gate
                    if not in_london_session(now_dt):
                        print(f"[{symbol}] 💤 Outside London session (09:00–12:00).")
                        logger.log(symbol, "outside_london", None, now_dt, tick, st["asia_high"], st["asia_low"], "")
                        continue

                    # Commit direction on LIVE sweep
                    if st["direction_committed"] is None:
                        detect_sweep_commit_direction(symbol, now_dt, tick)

                    direction_committed = st["direction_committed"]
                    if direction_committed is None:
                        print(f"[{symbol}] 👀 Waiting for Asia sweep during London…")
                        logger.log(symbol, "waiting_sweep", None, now_dt, tick, st["asia_high"], st["asia_low"], "")
                        continue
                    else:
                        print(f"[{symbol}] ✅ Sweep committed → {direction_committed.upper()}-only today.")
                        logger.log(symbol, "sweep_committed", direction_committed, now_dt, tick, st["asia_high"], st["asia_low"], "")

                    # NEW Gate: wait for manipulation into HTF PD array OR sweep previous TF H/L
                    if not st["manipulation_confirmed"]:
                        check_manipulation_gate(symbol, now_dt, tick)
                        if not st["manipulation_confirmed"]:
                            print(f"[{symbol}] ⛓ Waiting for manipulation: PD touch or Prev H/L sweep…")
                            logger.log(symbol, "waiting_manipulation", direction_committed, now_dt, tick, st["asia_high"], st["asia_low"], "")
                            continue
                        else:
                            print(f"[{symbol}] 🔒 Manipulation confirmed: {st['manip_reason']}")

                    # Detect IFVG setups and take only the committed side
                    df_m5 = get_candles(symbol, count=REQUIRED_BARS+5, timeframe=TIMEFRAME)
                    if df_m5.empty or len(df_m5) < REQUIRED_BARS:
                        print(f"[{symbol}] ⚠️ Not enough M5 bars yet for IFVG detection.")
                        logger.log(symbol, "not_enough_bars", direction_committed, now_dt, tick, st["asia_high"], st["asia_low"], "")
                        continue

                    setups = detect_ifvg(symbol, df_m5)
                    took = False
                    for direction, sym, entry, sl, _tp in setups:
                        if direction != direction_committed:
                            print(f"[{sym}] ↩️ Skip {direction} (committed {direction_committed}).")
                            logger.log(symbol, "ifvg_skip_wrong_dir", direction_committed, now_dt, tick, st["asia_high"], st["asia_low"], f"found={direction}")
                            continue
                        print(f"[{sym}] 🎯 Candidate {direction.upper()} LIMIT @ {entry} | SL {sl}")
                        logger.log(symbol, "ifvg_candidate", direction_committed, now_dt, tick, st["asia_high"], st["asia_low"], f"entry={entry},sl={sl}")
                        place_limit_order(direction, sym, entry, sl, _tp, logger, now_dt, tick)
                        took = True

                    if not took:
                        print(f"[{symbol}] 🔎 No IFVG entry yet (dir={direction_committed}).")
                        logger.log(symbol, "ifvg_none", direction_committed, now_dt, tick, st["asia_high"], st["asia_low"], "")

                except Exception as e:
                    print(f"[{symbol}] ERROR: {e}")
                    get_logger(symbol).log(symbol, "error", None, pd.Timestamp.utcnow(), None, None, None, str(e))

            time.sleep(SLEEP_SEC)

    except KeyboardInterrupt:
        print("Bot manually stopped.")

# =========================== MENU / MAIN ===========================
def main():
    if not mt5.initialize():
        raise RuntimeError(f"MT5 init failed: {mt5.last_error()}")
    try:
        run_bot()
    finally:
        _close_all_logs()
        mt5.shutdown()

if __name__ == "__main__":
    main()
