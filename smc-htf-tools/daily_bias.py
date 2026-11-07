#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Daily Bias Printer

- Computes a simple daily bias for each symbol based on D1 50 EMA.
- Bias rule (configurable):
    * close > EMA50 * (1 + NEUTRAL_BAND) => Bullish
    * close < EMA50 * (1 - NEUTRAL_BAND) => Bearish
    * else => Neutral
- Prints the bias to console for today's last completed D1 candle.

Dependencies:
    pip install MetaTrader5 pandas numpy
"""
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime

SYMBOLS = [
    "EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "XAGUSD",
    "US500", "US100", "UK100", "DE40"
]

BARS = 400
EMA_LEN = 50
NEUTRAL_BAND = 0.001  # 0.1% band around EMA for "Neutral"

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

def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def daily_bias_for_symbol(sym):
    df = copy_rates(sym, mt5.TIMEFRAME_D1, BARS)
    if df is None or len(df) < EMA_LEN + 5:
        return (sym, "NoData", None, None)
    df['ema50'] = ema(df['c'], EMA_LEN)
    # Use the last fully completed daily candle (second-to-last row)
    last = df.iloc[-2]
    c = last['c']
    e = last['ema50']
    if pd.isna(e):
        return (sym, "NoData", float(c), float(e))
    upper = e * (1 + NEUTRAL_BAND)
    lower = e * (1 - NEUTRAL_BAND)
    if c > upper:
        bias = "Bullish"
    elif c < lower:
        bias = "Bearish"
    else:
        bias = "Neutral"
    return (sym, bias, float(c), float(e))

def run():
    init_mt5()
    print("=== Daily Bias (EMA50) ===")
    print("Symbol   Bias     Close        EMA50")
    for sym in SYMBOLS:
        sym, bias, close, ema50 = daily_bias_for_symbol(sym)
        if bias == "NoData":
            print(f"{sym:<7} NoData   -            -")
        else:
            print(f"{sym:<7} {bias:<8} {close:<12.5f} {ema50:<12.5f}")

if __name__ == "__main__":
    run()
