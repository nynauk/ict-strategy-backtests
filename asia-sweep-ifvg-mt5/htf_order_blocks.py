#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Higher-Timeframe Order Blocks (outside Asia session)

- Timeframes: M15, M30, H1, H4, D1
- Detects bullish/bearish order blocks using a practical BOS + displacement heuristic.
- Skips candles that fall within the Asia session window (configurable).
- Prints recent detected blocks per symbol & timeframe; can also export to CSV.

Dependencies:
    pip install MetaTrader5 pandas numpy pytz

Tip: Run with a logged-in MT5 terminal. No explicit login here.
"""
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz

# -------------------------- CONFIG --------------------------
SYMBOLS = [
    # Update to your full list to mirror your Asia Sweep setup
    "EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "XAGUSD",
    "US500", "US100", "UK100", "DE40"
]
TIMEFRAMES = {
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30,
    "H1":  mt5.TIMEFRAME_H1,
    "H4":  mt5.TIMEFRAME_H4,
    "D1":  mt5.TIMEFRAME_D1,
}
# Broker/server timezone (adjust to your broker). Example: 'Etc/GMT-2' for GMT+2 style servers
ASIA_TZ = "UTC"
ASIA_START_HOUR = 1   # 01:00 local broker time
ASIA_END_HOUR = 7     # 07:00 local broker time

BARS = 5000
SWING_WINDOW = 3       # swing detection window (left/right)
MIN_DISPLACEMENT_ATR = 0.8  # fraction of ATR to call a displacement body
LOOKAHEAD_BARS = 3     # bars after the candidate OB to confirm BOS
EXPORT_TO_CSV = False
CSV_PATH = "htf_order_blocks.csv"

# -------------------------- HELPERS --------------------------
def init_mt5():
    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialize() failed: {mt5.last_error()}")

def copy_rates(symbol, timeframe, count=BARS):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
    if rates is None:
        return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df.rename(columns={'open':'o','high':'h','low':'l','close':'c','tick_volume':'v'}, inplace=True)
    return df[['time','o','h','l','c','v']]

def in_asia_session(ts_utc):
    # Convert to broker/server TZ then check hour
    tz = pytz.timezone(ASIA_TZ)
    local = ts_utc.tz_convert(tz)
    h = local.hour
    if ASIA_START_HOUR <= ASIA_END_HOUR:
        return ASIA_START_HOUR <= h < ASIA_END_HOUR
    else:
        # handle wrap past midnight
        return h >= ASIA_START_HOUR or h < ASIA_END_HOUR

def mark_asia(df):
    # boolean col 'in_asia'
    tz = pytz.timezone(ASIA_TZ)
    local_times = df['time'].dt.tz_convert(tz)
    hours = local_times.dt.hour
    if ASIA_START_HOUR <= ASIA_END_HOUR:
        df['in_asia'] = (hours >= ASIA_START_HOUR) & (hours < ASIA_END_HOUR)
    else:
        df['in_asia'] = (hours >= ASIA_START_HOUR) | (hours < ASIA_END_HOUR)
    return df

def atr(df, period=14):
    tr = np.maximum(df['h']-df['l'], np.maximum((df['h']-df['c'].shift()).abs(), (df['l']-df['c'].shift()).abs()))
    return tr.rolling(period).mean()

def is_swing_high(df, i, w=SWING_WINDOW):
    if i-w < 0 or i+w >= len(df): return False
    return df['h'].iloc[i] == df['h'].iloc[i-w:i+w+1].max()

def is_swing_low(df, i, w=SWING_WINDOW):
    if i-w < 0 or i+w >= len(df): return False
    return df['l'].iloc[i] == df['l'].iloc[i-w:i+w+1].min()

def detect_order_blocks(df):
    """
    Heuristic:
    - Bullish OB: find a bearish candle; within the next LOOKAHEAD_BARS, if a candle closes above a recent swing high
      with displacement (body >= MIN_DISPLACEMENT_ATR * ATR), then the bearish candle's body defines the bullish OB zone.
    - Bearish OB: symmetric logic for bullish candle and break below swing low.
    Skips candles that are in Asia session.
    """
    out = []
    df = df.copy()
    df = mark_asia(df)
    df['atr'] = atr(df, 14)
    # Precompute swing highs/lows
    swing_highs = df.index[df.apply(lambda r: False, axis=1)].tolist()
    swing_lows  = df.index[df.apply(lambda r: False, axis=1)].tolist()
    swing_highs = [i for i in range(len(df)) if is_swing_high(df, i)]
    swing_lows  = [i for i in range(len(df)) if is_swing_low(df, i)]

    last_swing_high = pd.Series(False, index=df.index)
    last_swing_low  = pd.Series(False, index=df.index)
    last_swing_high.loc[swing_highs] = True
    last_swing_low.loc[swing_lows] = True

    # We also track the most recent swing high/low values as of each index
    recent_swing_high = df['h'].where(last_swing_high).ffill()
    recent_swing_low  = df['l'].where(last_swing_low).ffill()

    for i in range(len(df)-LOOKAHEAD_BARS-1):
        if df['in_asia'].iloc[i]:
            continue
        o, h, l, c = df.loc[df.index[i], ['o','h','l','c']]
        body = abs(c - o)
        # Candidate bearish candle for bullish OB
        if c < o:
            # look ahead for BOS up
            for j in range(1, LOOKAHEAD_BARS+1):
                k = i + j
                if k >= len(df): break
                if df['in_asia'].iloc[k]:
                    continue
                ck_o, ck_c, ck_h, ck_l = df['o'].iloc[k], df['c'].iloc[k], df['h'].iloc[k], df['l'].iloc[k]
                displacement = abs(ck_c - ck_o) >= (df['atr'].iloc[k] * MIN_DISPLACEMENT_ATR if pd.notna(df['atr'].iloc[k]) else 0)
                # Break above the most recent swing high seen up to i
                bos_level = recent_swing_high.iloc[i] if pd.notna(recent_swing_high.iloc[i]) else None
                if bos_level is not None and ck_c > bos_level and displacement:
                    ob_low  = min(o, c)  # bearish candle body low
                    ob_high = max(o, c)  # bearish candle body high
                    out.append({
                        'time': df['time'].iloc[i],
                        'type': 'bullish',
                        'ob_low': float(ob_low),
                        'ob_high': float(ob_high),
                        'confirm_time': df['time'].iloc[k],
                        'bos_level': float(bos_level),
                    })
                    break

        # Candidate bullish candle for bearish OB
        if c > o:
            for j in range(1, LOOKAHEAD_BARS+1):
                k = i + j
                if k >= len(df): break
                if df['in_asia'].iloc[k]:
                    continue
                ck_o, ck_c, ck_h, ck_l = df['o'].iloc[k], df['c'].iloc[k], df['h'].iloc[k], df['l'].iloc[k]
                displacement = abs(ck_c - ck_o) >= (df['atr'].iloc[k] * MIN_DISPLACEMENT_ATR if pd.notna(df['atr'].iloc[k]) else 0)
                bos_level = recent_swing_low.iloc[i] if pd.notna(recent_swing_low.iloc[i]) else None
                if bos_level is not None and ck_c < bos_level and displacement:
                    ob_low  = min(o, c)  # bullish candle body low
                    ob_high = max(o, c)  # bullish candle body high
                    out.append({
                        'time': df['time'].iloc[i],
                        'type': 'bearish',
                        'ob_low': float(ob_low),
                        'ob_high': float(ob_high),
                        'confirm_time': df['time'].iloc[k],
                        'bos_level': float(bos_level),
                    })
                    break
    return pd.DataFrame(out)

def run():
    init_mt5()
    rows = []
    for sym in SYMBOLS:
        for tf_name, tf in TIMEFRAMES.items():
            df = copy_rates(sym, tf, BARS)
            if df is None or df.empty:
                print(f"[WARN] No data for {sym} {tf_name}")
                continue
            obs = detect_order_blocks(df)
            if obs.empty:
                print(f"[INFO] No OBs found for {sym} {tf_name}")
                continue
            obs.insert(0, 'symbol', sym)
            obs.insert(1, 'timeframe', tf_name)
            rows.append(obs)
    if rows:
        result = pd.concat(rows, ignore_index=True)
        print(result.tail(50).to_string(index=False))
        if EXPORT_TO_CSV:
            result.to_csv(CSV_PATH, index=False)
            print(f"[OK] Saved: {CSV_PATH}")
    else:
        print("[INFO] Nothing detected.")

if __name__ == "__main__":
    run()
