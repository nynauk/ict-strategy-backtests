#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Higher-Timeframe Fair Value Gaps (bullish & bearish)

- Timeframes: M15, M30, H1, H4, D1
- Detects classic 3-candle FVGs:
    * Bullish FVG: low[n] > high[n-2]
    * Bearish FVG: high[n] < low[n-2]
- Returns gap bounds and type.
- Optional filter to skip gaps formed during Asia session (toggle EXCLUDE_ASIA).

Dependencies:
    pip install MetaTrader5 pandas numpy pytz
"""
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime
import pytz

# -------------------------- CONFIG --------------------------
SYMBOLS = [
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
BARS = 5000

# Asia filter (if you want to exclude FVGs formed within Asia)
EXCLUDE_ASIA = True
ASIA_TZ = "UTC"
ASIA_START_HOUR = 1
ASIA_END_HOUR = 7

EXPORT_TO_CSV = False
CSV_PATH = "htf_fvgs.csv"

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

def mark_asia(df):
    tz = pytz.timezone(ASIA_TZ)
    local_times = df['time'].dt.tz_convert(tz)
    hours = local_times.dt.hour
    if ASIA_START_HOUR <= ASIA_END_HOUR:
        df['in_asia'] = (hours >= ASIA_START_HOUR) & (hours < ASIA_END_HOUR)
    else:
        df['in_asia'] = (hours >= ASIA_START_HOUR) | (hours < ASIA_END_HOUR)
    return df

def detect_fvgs(df):
    df = df.copy()
    if EXCLUDE_ASIA:
        df = mark_asia(df)
    out = []
    for i in range(2, len(df)):
        # Bullish FVG: current low > high two bars ago
        if df['l'].iloc[i] > df['h'].iloc[i-2]:
            if not EXCLUDE_ASIA or not df['in_asia'].iloc[i-1]:
                out.append({
                    'time': df['time'].iloc[i],
                    'type': 'bullish',
                    'gap_low': float(df['h'].iloc[i-2]),
                    'gap_high': float(df['l'].iloc[i]),
                })
        # Bearish FVG: current high < low two bars ago
        if df['h'].iloc[i] < df['l'].iloc[i-2]:
            if not EXCLUDE_ASIA or not df['in_asia'].iloc[i-1]:
                out.append({
                    'time': df['time'].iloc[i],
                    'type': 'bearish',
                    'gap_low': float(df['h'].iloc[i]),
                    'gap_high': float(df['l'].iloc[i-2]),
                })
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
            fvgs = detect_fvgs(df)
            if fvgs.empty:
                print(f"[INFO] No FVGs found for {sym} {tf_name}")
                continue
            fvgs.insert(0, 'symbol', sym)
            fvgs.insert(1, 'timeframe', tf_name)
            rows.append(fvgs)
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
