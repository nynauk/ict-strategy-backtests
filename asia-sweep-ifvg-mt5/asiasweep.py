import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta, timezone
import time

# ====== CONFIG ======
SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY", "GBPJPY", "GER40.cash", "USDCAD", "USDCHF"]
ASIA_START_H = 3   # server time
ASIA_END_H   = 9   # server time
POLL_SECONDS = 2   # how often to check price (secs)

# ====== INIT ======
if not mt5.initialize():
    raise RuntimeError(f"MT5 init failed: {mt5.last_error()}")

def get_server_utc_offset_hours(reference_symbol="EURUSD"):
    rates = mt5.copy_rates_from_pos(reference_symbol, mt5.TIMEFRAME_D1, 0, 1)
    if rates is None or len(rates) == 0:
        raise RuntimeError("Could not fetch D1 candle to infer server timezone.")
    ts = int(rates[0]["time"])
    dt_utc = datetime.fromtimestamp(ts, tz=timezone.utc)
    # server midnight should map to some UTC hour; infer offset
    offset = (24 - dt_utc.hour) % 24
    if offset > 14:
        offset -= 24
    return offset

def to_utc_from_server(dt_server_naive, offset_hours):
    # treat input as server-time aware, then convert to UTC
    dt_server = dt_server_naive.replace(tzinfo=timezone(timedelta(hours=offset_hours)))
    return dt_server.astimezone(timezone.utc)

def asia_bounds_utc_for_server_today(offset_hours):
    now_utc = datetime.now(timezone.utc)
    now_server = now_utc.astimezone(timezone(timedelta(hours=offset_hours)))
    server_today = now_server.date()

    start_server = datetime(server_today.year, server_today.month, server_today.day, ASIA_START_H, 0, 0)
    end_server   = datetime(server_today.year, server_today.month, server_today.day, ASIA_END_H,   0, 0)

    start_utc = to_utc_from_server(start_server, offset_hours)
    end_utc   = to_utc_from_server(end_server, offset_hours)
    return start_utc, end_utc, server_today

def fetch_m5_range(symbol, start_utc, end_utc):
    rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M5, start_utc, end_utc)
    if rates is None or len(rates) == 0:
        return pd.DataFrame()
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    return df

def get_asia_levels(symbol, start_utc, end_utc):
    df = fetch_m5_range(symbol, start_utc, end_utc)
    if df.empty:
        return None, None, 0
    return float(df["high"].max()), float(df["low"].min()), int(len(df))

def mid_price(tick):
    if tick is None:
        return None
    ask = getattr(tick, "ask", None)
    bid = getattr(tick, "bid", None)
    if ask is None or bid is None:
        return None
    if ask <= 0 or bid <= 0:
        return None
    return (ask + bid) / 2.0

def in_utc_window(now_utc, start_hour, end_hour):
    start = now_utc.replace(hour=start_hour, minute=0, second=0, microsecond=0)
    end   = now_utc.replace(hour=end_hour,   minute=0, second=0, microsecond=0)
    return start <= now_utc < end

def sleep_until_utc(hour):
    now = datetime.now(timezone.utc)
    target = now.replace(hour=hour, minute=0, second=0, microsecond=0)
    if now >= target:
        target += timedelta(days=1)
    while True:
        now = datetime.now(timezone.utc)
        if now >= target:
            return
        time.sleep(min(30, max(1, (target - now).total_seconds())))

def main():
    try:
        # ensure symbols are selected
        for sym in SYMBOLS:
            mt5.symbol_select(sym, True)

        offset_hours = get_server_utc_offset_hours(reference_symbol=SYMBOLS[0])
        print(f"Server UTC offset: {offset_hours:+d}h")
        print("Asia window (server): 03:00 → 09:00")
        print("Valid sweep window (UTC): 09:00 → 12:00\n")

        current_server_date = None
        asia_levels = {}              # symbol -> {"high": x, "low": y, "bars": n}
        swept = {}                    # symbol -> {"high": False, "low": False}

        while True:
            now_utc = datetime.now(timezone.utc)

            # Recompute Asia levels at each new server date
            start_utc, end_utc, server_today = asia_bounds_utc_for_server_today(offset_hours)
            if current_server_date != server_today:
                print(f"\n=== New server day: {server_today} — computing Asia levels ===")
                current_server_date = server_today
                asia_levels.clear()
                swept = {sym: {"high": False, "low": False} for sym in SYMBOLS}

                for sym in SYMBOLS:
                    hi, lo, bars = get_asia_levels(sym, start_utc, end_utc)
                    if hi is None:
                        print(f"[{sym}] No M5 data in Asia window.")
                        continue
                    asia_levels[sym] = {"high": hi, "low": lo, "bars": bars}
                    print(f"[{sym}] Asia bars: {bars} | High: {hi} | Low: {lo}")
                print("=== Levels ready ===\n")

            # Only monitor during 09:00–12:00 UTC
            if not in_utc_window(now_utc, 9, 12):
                if now_utc.hour < 9:
                    sleep_until_utc(9)
                else:
                    print("Window closed (after 12:00 UTC). Sleeping until next 09:00 UTC...")
                    sleep_until_utc(9)
                continue

            # Monitor each symbol for sweeps (first occurrence only)
            for sym in SYMBOLS:
                if sym not in asia_levels:
                    continue
                levels = asia_levels[sym]
                tick = mt5.symbol_info_tick(sym)
                price = mid_price(tick)
                if price is None:
                    continue

                # High sweep -> SELL (trigger once)
                if not swept[sym]["high"] and price > levels["high"]:
                    swept[sym]["high"] = True
                    print(f"[{sym}] 🔺 High swept at {price:.5f} (Asia High {levels['high']:.5f}) → SELL")

                # Low sweep -> BUY (trigger once)
                if not swept[sym]["low"] and price < levels["low"]:
                    swept[sym]["low"] = True
                    print(f"[{sym}] 🔻 Low swept at {price:.5f} (Asia Low {levels['low']:.5f}) → BUY")

            time.sleep(POLL_SECONDS)

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        mt5.shutdown()

if __name__ == "__main__":
    main()
