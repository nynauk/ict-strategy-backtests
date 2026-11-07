#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
D1/H4/H1 Bias via 20-bar Fractal Market Structure (BOS/CHoCH)

Logic:
- Create swing highs/lows using a 20-bar period (10 left + 10 right bars).
- Track last swing high/low while iterating forward.
- BOS_UP  when close > last swing high (break of structure up)
- BOS_DOWN when close < last swing low  (break of structure down)
- Bias per timeframe:
    * If last CONTINUITY events are all BOS_UP  -> Bullish
    * If last CONTINUITY events are all BOS_DOWN -> Bearish
    * Else -> Neutral
- CHoCH: last event direction != previous event direction.

Console output example:
EURUSD  D1:Bullish(CHOCH:No)  H4:Bullish(CHOCH:No)  H1:Bearish(CHOCH:Yes)  STATUS: INVALID (diff: H1)
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np

# ---------- CONFIG ----------
SYMBOLS = [
    "EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "XAGUSD",
    "US500", "US100", "UK100", "DE40"
]

TF_MAP = {
    "D1": mt5.TIMEFRAME_D1,
    "H4": mt5.TIMEFRAME_H4,
    "H1": mt5.TIMEFRAME_H1,
}

BARS = 3000
SWING_LEFT = 10      # for 20-bar fractal (10 left, 10 right)
SWING_RIGHT = 10
CONTINUITY = 2       # how many last BOS events must align for a valid trend
EPS = 0.0            # buffer for BOS comparisons


# ---------- HELPERS ----------
def init_mt5():
    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialize() failed: {mt5.last_error()}")


def copy_rates(symbol, timeframe, count=BARS):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
    if rates is None:
        return None
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df.rename(columns={"open": "o", "high": "h", "low": "l", "close": "c", "tick_volume": "v"}, inplace=True)
    return df[["time", "o", "h", "l", "c", "v"]]


def find_fractals(df, L=SWING_LEFT, R=SWING_RIGHT):
    n = len(df)
    swing_high = pd.Series(False, index=df.index)
    swing_low = pd.Series(False, index=df.index)
    for i in range(L, n - R):
        window = slice(i - L, i + R + 1)
        hi = df["h"].iloc[window].max()
        lo = df["l"].iloc[window].min()
        if df["h"].iloc[i] == hi:
            swing_high.iloc[i] = True
        if df["l"].iloc[i] == lo:
            swing_low.iloc[i] = True
    return swing_high, swing_low


def bos_events(df):
    if len(df) < (SWING_LEFT + SWING_RIGHT + 10):
        return []

    swing_high, swing_low = find_fractals(df)
    df = df.copy()

    # exclude last bar (in progress)
    df = df.iloc[:-1].reset_index(drop=True)
    swing_high = swing_high.iloc[:-1].reset_index(drop=True)
    swing_low = swing_low.iloc[:-1].reset_index(drop=True)

    last_high = np.nan
    last_low = np.nan
    events = []

    for i in range(len(df)):
        if swing_high.iloc[i]:
            last_high = df["h"].iloc[i]
        if swing_low.iloc[i]:
            last_low = df["l"].iloc[i]

        c = df["c"].iloc[i]
        if not np.isnan(last_high) and c > last_high + EPS:
            events.append({"time": df["time"].iloc[i], "dir": "UP", "level": float(last_high)})
        if not np.isnan(last_low) and c < last_low - EPS:
            events.append({"time": df["time"].iloc[i], "dir": "DOWN", "level": float(last_low)})

    return events


def bias_from_events(events, continuity=CONTINUITY):
    if len(events) == 0:
        return "NoData", None, None
    if len(events) < continuity:
        choch = (len(events) >= 2 and events[-1]["dir"] != events[-2]["dir"])
        return "Neutral", events[-1]["dir"], choch

    lastN = [e["dir"] for e in events[-continuity:]]
    if all(d == "UP" for d in lastN):
        bias = "Bullish"
    elif all(d == "DOWN" for d in lastN):
        bias = "Bearish"
    else:
        bias = "Neutral"

    choch = (len(events) >= 2 and events[-1]["dir"] != events[-2]["dir"])
    return bias, events[-1]["dir"], choch


def timeframe_bias(symbol, tf_name, tf_const):
    df = copy_rates(symbol, tf_const, BARS)
    if df is None or df.empty:
        return ("NoData", None, None)
    events = bos_events(df)
    return bias_from_events(events)


def run():
    init_mt5()
    print("=== D1/H4/H1 20-bar Fractal Structure Bias ===")
    print("Symbol  D1                H4                H1                STATUS")

    for sym in SYMBOLS:
        b_d1, d_d1, ch_d1 = timeframe_bias(sym, "D1", TF_MAP["D1"])
        b_h4, d_h4, ch_h4 = timeframe_bias(sym, "H4", TF_MAP["H4"])
        b_h1, d_h1, ch_h1 = timeframe_bias(sym, "H1", TF_MAP["H1"])

        d1_txt = f"{b_d1}(CHOCH:{'Yes' if ch_d1 else 'No'})"
        h4_txt = f"{b_h4}(CHOCH:{'Yes' if ch_h4 else 'No'})"
        h1_txt = f"{b_h1}(CHOCH:{'Yes' if ch_h1 else 'No'})"

        biases = {"D1": b_d1, "H4": b_h4, "H1": b_h1}
        unique = set(biases.values())

        if len(unique) == 1 and "Neutral" not in unique and "NoData" not in unique:
            status = f"VALID {next(iter(unique))}"
            diff_txt = ""
        else:
            counts = {}
            for v in biases.values():
                counts[v] = counts.get(v, 0) + 1
            max_count = max(counts.values())
            modes = {k for k, v in counts.items() if v == max_count}
            if len(modes) == 1:
                mode_val = next(iter(modes))
                diffs = [tf for tf, val in biases.items() if val != mode_val]
            else:
                diffs = list(biases.keys())
            status = "INVALID"
            diff_txt = f" (diff: {', '.join(diffs)})"

        print(f"{sym:<7} D1:{d1_txt:<16} H4:{h4_txt:<16} H1:{h1_txt:<16} STATUS: {status}{diff_txt}")


if __name__ == "__main__":
    run()
