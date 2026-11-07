# backtest_m3_asia_htfconfluence_choch_fvg_pairing.py
# ---------------------------------------------------------------
# London (09:00–12:00) — Asia sweep ➜ HTF (OB/FVG/IFVG) confluence ➜ CHoCH + leg’s highest FVG pullback:
#   - If Asia HIGH is swept → require price inside bearish HTF zone (D1/H4/H1)
#     → bearish CHoCH (down) → wait for pullback to the HIGHEST FVG created by the displacement leg
#     → SELL @ touch/close. SL = swing high; TP = opposing Asia liquidity (Asia Low) with 2R floor.
#   - If Asia LOW  is swept → require price inside bullish HTF zone (D1/H4/H1)
#     → bullish CHoCH (up) → wait for pullback to the HIGHEST FVG created by the displacement leg
#     → BUY  @ touch/close.  SL = swing low;  TP = opposing Asia liquidity (Asia High) with 2R floor.
#
# Correlation (EURUSD + GBPUSD + DXY):
#   - EURUSD & GBPUSD must sweep the SAME Asia side in London; DXY must sweep the OPPOSITE side.
#   - Optional: DXY must also show opposite CHoCH + FVG pullback near its sweep (no HTF confluence required for DXY).
#
# SMT (logged):
#   - EURUSD vs GBPUSD SMT: bearish if only one sweeps Asia High; bullish if only one sweeps Asia Low.
#
# Notes:
#   - No higher-timeframe bias filter.
#   - PDH/PWH/PMH logged only (optional confluence), not required.
#   - One trade per symbol per day (configurable). SL/TP tie on same bar favors SL (conservative).
#
# Outputs:
#   - CSVs: trades, equity, daily logs, per-symbol summary, overall summary.
#   - Plots: example + latest per symbol (dark theme) with Asia/London, PD levels, CHoCH leg FVG zone, entry/SL/TP.
# ---------------------------------------------------------------

import os
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import MetaTrader5 as mt5
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ====================== USER INPUTS ======================

SYMBOLS = [
    # === Correlation trio ===
    "EURUSD", "GBPUSD", "DXY.cash",   # set DXY symbol to your broker's code

    # === Majors & minors ===
    "USDJPY", "GBPJPY", "AUDUSD", "NZDUSD", "USDCAD", "USDCHF",
    "EURJPY", "AUDJPY", "EURGBP",

    # === Indices (broker-style; tweak to your broker names) ===
    "UK100.cash", "GER40.cash",
    "US30.cash", "NAS100", "SPX500", "JP225.cash", "EU50.cash", "FRA40.cash",
]

# Execution timeframe — M3 preferred; switch to M5 if unavailable
TIMEFRAME = mt5.TIMEFRAME_M3  # mt5.TIMEFRAME_M5 for 5-minute

# --- Data Range (choose ONE mode) ---
RANGE_MODE = "months"   # "months" | "days" | "dates"
RANGE_MONTHS = 6
RANGE_DAYS = 120
DATE_FROM = "2025-04-01"
DATE_TO   = "2025-08-12"

# Sessions (server time)
ASIA_START_H = 3
ASIA_END_H   = 9    # exclusive
LDN_START_H  = 9
LDN_END_H    = 12   # exclusive

# ----- CHoCH + FVG pullback params -----
SWING_LOOKBACK = 5          # bars left/right to define swing highs/lows
MIN_DISPLACEMENT_MULT = 1.5 # body size >= X * average body (of last N bars)
AVG_BODY_N = 20             # average body window for displacement threshold
LEG_MAX_BARS = 30           # max bars to consider when walking back to the leg origin
PULLBACK_WAIT_BARS = 60     # how long after CHoCH we wait for highest FVG touch
FVG_MIN_TICKS = 2           # min FVG size (ticks) for leg FVGs
ENTRY_ON_TOUCH = True       # True: execute at touch-close; False: execute at FVG boundary

SPREAD_PIPS = 0.0           # optional SL pad in pips per symbol

# TP / RR rules (use Asia opposing liquidity with a 2R floor)
TP_MODE   = "asia_min2R"    # "asia_min2R" | "fixed_rr"
RR_TARGET = 2.0             # min-2R floor and fixed_rr target if selected

# Risk & accounting
RISK_PER_TRADE = 1.0
START_BALANCE  = 10000.0

# Management
ALLOW_OVERLAP = False         # defined (not used), ONE_TRADE_PER_DAY gates trades
ONE_TRADE_PER_DAY = True

# ---- HTF confluence (REQUIRED for entry) ----
REQUIRE_HTF_CONFLUENCE = True
HTF_TTLS_DAYS = 20  # lifetime of zones from creation

USE_D1_OB   = True
USE_H4_OB   = True
USE_H1_OB   = True

USE_D1_FVG  = True
USE_D1_IFVG = True
USE_H4_FVG  = True
USE_H4_IFVG = True
USE_H1_FVG  = True
USE_H1_IFVG = True

# ---- PD levels (for LOGGING/optional confluence ONLY; not required) ----
LOG_PMH_PML = True
LOG_PWH_PWL = True
LOG_PDH_PDL = True

# ===== Correlation & confirmation (EU/GU paired with DXY) =====
ENABLE_DXY_CONFIRM = True
DXY_SYMBOL = "DXY.cash"           # must match a symbol in SYMBOLS
REQUIRE_DXY_CHOCH_FVG = False     # if True, DXY must also show opposite CHoCH+FVG near its sweep
DXY_CONFIRM_TOL_BARS = 20         # bars window after DXY sweep to find CHoCH+FVG

# SMT detection between EURUSD and GBPUSD (logged, not required)
LOG_SMT_EU_GU = True

# Visualization
OUT_DIR = "bt_out"
PLOT_SYMBOL = "EURUSD"
PLOT_TRADE_INDEX = 0
PLOT_LOOKBACK_BARS = 120
PLOT_LOOKAHEAD_BARS = 180

PRINT_EVERY_N_DAYS = 5
ANNUALIZE_SHARPE = False

# ==================== UTILS & META =======================

@dataclass
class SymbolMeta:
    digits: int
    point: float
    tick_size: float
    tick_value: float
    vol_min: float
    vol_max: float
    vol_step: float

def server_now() -> datetime:
    for s in SYMBOLS:
        tick = mt5.symbol_info_tick(s)
        if tick and getattr(tick, "time", None):
            return datetime.fromtimestamp(tick.time, tz=timezone.utc).replace(tzinfo=None)
    return datetime.utcnow()

def compute_date_range() -> Tuple[datetime, datetime]:
    mode = (RANGE_MODE or "").lower()
    now = server_now().replace(second=0, microsecond=0)
    if mode == "months":
        start = (pd.Timestamp(now) - pd.DateOffset(months=int(RANGE_MONTHS))).to_pydatetime()
        return start, now
    if mode == "days":
        start = now - timedelta(days=int(RANGE_DAYS))
        return start, now
    if mode == "dates":
        start = datetime.strptime(DATE_FROM, "%Y-%m-%d").replace(hour=0, minute=0, second=0, microsecond=0)
        end   = datetime.strptime(DATE_TO,   "%Y-%m-%d").replace(hour=23, minute=59, second=59, microsecond=0)
        return start, end
    start = (pd.Timestamp(now) - pd.DateOffset(months=4)).to_pydatetime()
    return start, now

def get_meta(symbol: str) -> Optional[SymbolMeta]:
    info = mt5.symbol_info(symbol)
    if not info:
        return None
    def g(o, n, d=0.0): return getattr(o, n, d)
    tick_size = g(info, "trade_tick_size", 0.0) or g(info, "point", 0.0)
    tick_value = g(info, "trade_tick_value", 0.0) or g(info, "trade_tick_value_profit", 0.0) or g(info, "trade_tick_value_loss", 0.0)
    return SymbolMeta(
        digits=int(g(info, "digits", 5)),
        point=float(g(info, "point", 0.0001)),
        tick_size=float(tick_size if tick_size > 0 else g(info, "point", 0.0001)),
        tick_value=float(tick_value),
        vol_min=float(g(info, "volume_min", 0.01)),
        vol_max=float(g(info, "volume_max", 100.0)),
        vol_step=float(g(info, "volume_step", 0.01)),
    )

def pip_value(symbol: str, meta: SymbolMeta) -> float:
    if symbol.endswith("JPY"):
        return 0.01
    if ".cash" in symbol or symbol.upper().startswith(("GER", "DE", "US30", "NAS", "ES", "SPX", "UK", "JP", "EU", "FR", "GERMAN", "UK100")):
        return meta.point
    return 0.0001

def round_to_tick(price: float, meta: SymbolMeta) -> float:
    if meta.tick_size <= 0:
        return round(price, meta.digits)
    steps = round(price / meta.tick_size)
    return round(steps * meta.tick_size, meta.digits)

def load_bars(symbol: str, timeframe, start: datetime, end: datetime) -> pd.DataFrame:
    rates = mt5.copy_rates_range(symbol, timeframe, start, end)
    df = pd.DataFrame(rates) if rates is not None else pd.DataFrame()
    if df.empty:
        return df
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df.sort_values("time").reset_index(drop=True)
    return df

# =============== HTF ZONES (OBs & FVGs) ==================

@dataclass
class Zone:
    tf: str          # 'D1','H4','H1'
    kind: str        # 'OB','FVG','IFVG'
    direction: str   # 'bull','bear'
    low: float
    high: float
    start: pd.Timestamp
    expiry: pd.Timestamp

def detect_htf_zones(symbol: str, meta: SymbolMeta, start: datetime, end: datetime) -> List[Zone]:
    zones: List[Zone] = []
    ttl = pd.Timedelta(days=HTF_TTLS_DAYS)

    tf_map = [
        ("D1", mt5.TIMEFRAME_D1,  USE_D1_OB,  USE_D1_FVG,  USE_D1_IFVG),
        ("H4", mt5.TIMEFRAME_H4,  USE_H4_OB,  USE_H4_FVG,  USE_H4_IFVG),
        ("H1", mt5.TIMEFRAME_H1,  USE_H1_OB,  USE_H1_FVG,  USE_H1_IFVG),
    ]

    def add(kind, tf, direction, low, high, tstart):
        zones.append(Zone(tf=tf, kind=kind, direction=direction,
                          low=float(low), high=float(high),
                          start=tstart, expiry=(tstart + ttl)))

    for tf_name, tf_const, use_ob, use_fvg, use_ifvg in tf_map:
        if not any([use_ob, use_fvg, use_ifvg]):
            continue
        df = load_bars(symbol, tf_const, start - timedelta(days=90), end + timedelta(days=2))
        if df.empty:
            continue

        # FVGs / IFVGs
        for i in range(2, len(df)):
            c1, c3 = df.iloc[i-2], df.iloc[i]
            tstart = df.iloc[i].time
            # Bullish gap
            if (c1.high < c3.low) and ((c3.low - c1.high) >= FVG_MIN_TICKS * meta.tick_size):
                if use_fvg:  add("FVG",  tf_name, "bull", c1.high, c3.low, tstart)
                if use_ifvg: add("IFVG", tf_name, "bull", c1.high, c3.low, tstart)
            # Bearish gap
            if (c1.low > c3.high) and ((c1.low - c3.high) >= FVG_MIN_TICKS * meta.tick_size):
                if use_fvg:  add("FVG",  tf_name, "bear", c3.high, c1.low, tstart)
                if use_ifvg: add("IFVG", tf_name, "bear", c3.high, c1.low, tstart)

        # OBs (simple heuristic)
        if use_ob:
            for i in range(0, len(df)-1):
                c, n = df.iloc[i], df.iloc[i+1]
                tstart = df.iloc[i].time
                # bullish OB: down candle then next closes above its high
                if c.close < c.open and n.close > c.high:
                    low, high = min(c.open, c.high), max(c.open, c.high)
                    add("OB", tf_name, "bull", low, high, tstart)
                # bearish OB: up candle then next closes below its low
                if c.close > c.open and n.close < c.low:
                    low, high = min(c.low, c.open), max(c.low, c.open)
                    add("OB", tf_name, "bear", low, high, tstart)

    return zones

def bar_touches_zone(bar_low: float, bar_high: float, z: Zone) -> bool:
    return not (bar_high < z.low or bar_low > z.high)

def confluence_ok(t: pd.Timestamp, bar_low: float, bar_high: float, zones: List[Zone], want_dir: str) -> bool:
    if not REQUIRE_HTF_CONFLUENCE:
        return True
    want = "bull" if want_dir == "buy" else "bear"
    for z in zones:
        if z.direction == want and z.start <= t <= z.expiry:
            if bar_touches_zone(bar_low, bar_high, z):
                return True
    return False

# ==================== PD ARRAYS (LOGGING ONLY) ============

def day_bounds(day: pd.Timestamp) -> Tuple[pd.Timestamp, pd.Timestamp]:
    d0 = day.replace(hour=0, minute=0, second=0, microsecond=0)
    return d0, d0 + pd.Timedelta(days=1)

def week_bounds(day: pd.Timestamp) -> Tuple[pd.Timestamp, pd.Timestamp]:
    wd = day.weekday()
    start = (day - pd.Timedelta(days=wd)).replace(hour=0, minute=0, second=0, microsecond=0)
    return start, start + pd.Timedelta(days=7)

def month_bounds(day: pd.Timestamp) -> Tuple[pd.Timestamp, pd.Timestamp]:
    start = day.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    end = (start + pd.DateOffset(months=1)).to_pydatetime()
    return start, end

# ==================== CHoCH + FVG ENTRY (M3) =====================

def is_swing_high(df: pd.DataFrame, i: int, k: int) -> bool:
    lo = max(0, i - k); hi = min(len(df)-1, i + k)
    ph = df.iloc[i].high
    return all(df.iloc[j].high <= ph for j in range(lo, hi+1) if j != i)

def is_swing_low(df: pd.DataFrame, i: int, k: int) -> bool:
    lo = max(0, i - k); hi = min(len(df)-1, i + k)
    pl = df.iloc[i].low
    return all(df.iloc[j].low >= pl for j in range(lo, hi+1) if j != i)

def avg_body(df: pd.DataFrame, i: int, n: int) -> float:
    if i <= 0: return float((df["close"] - df["open"]).abs().iloc[:max(1, n)].mean())
    lo = max(0, i-n+1); sl = df.iloc[lo:i+1]
    return float((sl["close"] - sl["open"]).abs().mean())

def detect_choch(df: pd.DataFrame, i: int, k: int) -> Optional[dict]:
    """
    Returns dict with:
      {"direction": "down"/"up", "break_idx": i, "broken_swing_idx": j}
    when close breaks the most recent opposite swing.
    """
    if i <= k+1: return None
    # find most recent swing high / low before i
    j_hi = None; j_lo = None
    for j in range(i-1, k, -1):
        if j_hi is None and is_swing_high(df, j, k): j_hi = j
        if j_lo is None and is_swing_low(df, j, k):  j_lo = j
        if j_hi is not None and j_lo is not None: break
    if j_hi is None or j_lo is None: return None

    c = df.iloc[i]
    if c.close < df.iloc[j_lo].low:   # break of swing low => bearish CHoCH
        return {"direction": "down", "break_idx": i, "broken_swing_idx": j_lo}
    if c.close > df.iloc[j_hi].high:  # break of swing high => bullish CHoCH
        return {"direction": "up", "break_idx": i, "broken_swing_idx": j_hi}
    return None

def displacement_ok(df: pd.DataFrame, i: int, n_avg: int, mult: float) -> bool:
    body = abs(df.iloc[i].close - df.iloc[i].open)
    return body >= mult * avg_body(df, i-1, n_avg)

def leg_indices(df: pd.DataFrame, break_idx: int, direction: str, k: int, max_bars: int) -> Tuple[int,int]:
    """
    From break_idx go backward to the pivot that likely started the leg.
    Stop at opposite swing or max_bars.
    """
    start = max(0, break_idx - max_bars)
    if direction == "down":
        # bearish leg: from a recent swing high to break_idx
        for j in range(break_idx-1, start, -1):
            if is_swing_high(df, j, k): return j, break_idx
    else:
        # bullish leg: from a recent swing low to break_idx
        for j in range(break_idx-1, start, -1):
            if is_swing_low(df, j, k):  return j, break_idx
    return max(0, break_idx-5), break_idx

def leg_fvgs(df: pd.DataFrame, meta: SymbolMeta, i0: int, i1: int, direction: str, min_ticks: int):
    """
    Collect FVGs created within [i0..i1] by the displacement leg.
    For bearish leg (down): bearish FVGs (c1.low > c3.high) -> band [high,low] = [c1.low, c3.high]
    For bullish leg (up):   bullish FVGs (c1.high < c3.low) -> band [low,high] = [c1.high, c3.low]
    Returns list of dicts: {"low":float,"high":float,"i":int}
    """
    out = []
    for i in range(i0+2, i1+1):
        c1 = df.iloc[i-2]; c3 = df.iloc[i]
        if direction == "down":
            if c1.low > c3.high and (c1.low - c3.high) >= FVG_MIN_TICKS * meta.tick_size:
                out.append({"low": float(c3.high), "high": float(c1.low), "i": i})
        else:
            if c1.high < c3.low and (c3.low - c1.high) >= FVG_MIN_TICKS * meta.tick_size:
                out.append({"low": float(c1.high), "high": float(c3.low), "i": i})
    return out

def pick_highest_fvg(fvgs: List[dict]) -> Optional[dict]:
    if not fvgs: return None
    return max(fvgs, key=lambda z: z["high"])

def detect_choch_fvg_entry(symbol: str,
                           df: pd.DataFrame,
                           meta: SymbolMeta,
                           idx_now: int,
                           asia_path: str) -> Optional[Tuple[str, float, float, dict]]:
    """
    After Asia sweep (known path: "high" or "low"), watch for:
      - CHoCH with displacement in the expected direction
      - Then a pullback that touches the HIGHEST FVG created by that displacement leg
    Returns (direction, entry, sl, debug) exactly at the bar that touches FVG and executes.
    """
    choch = detect_choch(df, idx_now, SWING_LOOKBACK)
    if not choch: return None

    desired = "down" if asia_path == "high" else "up"
    if choch["direction"] != desired: return None
    if not displacement_ok(df, choch["break_idx"], AVG_BODY_N, MIN_DISPLACEMENT_MULT):
        return None

    # define displacement leg
    i0, i1 = leg_indices(df, choch["break_idx"], choch["direction"], SWING_LOOKBACK, LEG_MAX_BARS)

    # collect leg FVGs and pick highest
    fvgs = leg_fvgs(df, meta, i0, i1, choch["direction"], FVG_MIN_TICKS)
    top = pick_highest_fvg(fvgs)
    if not top: return None

    # wait for pullback into that top FVG within PULLBACK_WAIT_BARS
    end_wait = min(len(df)-1, choch["break_idx"] + PULLBACK_WAIT_BARS)
    for i in range(choch["break_idx"]+1, end_wait+1):
        bar = df.iloc[i]
        if bar.low <= top["high"] and bar.high >= top["low"]:  # touched band
            if choch["direction"] == "down":
                entry = round_to_tick(float(bar.close if ENTRY_ON_TOUCH else top["high"]), meta)
                sl = float(df.iloc[max(i0, i-SWING_LOOKBACK):i]["high"].max())
                sl = round_to_tick(sl + SPREAD_PIPS * pip_value(symbol, meta), meta)
                if entry < sl:
                    dbg = {"choch_break_idx": choch["break_idx"], "leg": (i0, i1), "fvg": top, "fill_idx": i}
                    return ("sell", entry, sl, dbg)
            else:
                entry = round_to_tick(float(bar.close if ENTRY_ON_TOUCH else top["low"]), meta)
                sl = float(df.iloc[max(i0, i-SWING_LOOKBACK):i]["low"].min())
                sl = round_to_tick(sl - SPREAD_PIPS * pip_value(symbol, meta), meta)
                if sl < entry:
                    dbg = {"choch_break_idx": choch["break_idx"], "leg": (i0, i1), "fvg": top, "fill_idx": i}
                    return ("buy", entry, sl, dbg)
    return None

# ============== Correlation, SMT, and DXY confirm ==============

def asia_sweep_side(day_df: pd.DataFrame, asia_high: float, asia_low: float, start_h: int, end_h: int) -> Tuple[Optional[str], Optional[pd.Timestamp]]:
    """Return ('high' or 'low', time) of first Asia sweep in [start_h,end_h)."""
    ldn_slice = day_df[(day_df["time"].dt.hour >= start_h) & (day_df["time"].dt.hour < end_h)]
    for _, bar in ldn_slice.iterrows():
        if bar.high > asia_high:
            return "high", pd.Timestamp(bar.time)
        if bar.low  < asia_low:
            return "low",  pd.Timestamp(bar.time)
    return None, None

def get_day_asia_levels(m3: pd.DataFrame, day_ts: pd.Timestamp) -> Tuple[float, float]:
    asia_s = day_ts.replace(hour=ASIA_START_H, minute=0, second=0, microsecond=0)
    asia_e = day_ts.replace(hour=ASIA_END_H,   minute=0, second=0, microsecond=0)
    asia_slice = m3[(m3["time"] >= asia_s) & (m3["time"] < asia_e)]
    return float(asia_slice["high"].max()), float(asia_slice["low"].min())

def compute_smt_eu_gu(day_df_eu: pd.DataFrame, day_df_gu: pd.DataFrame, asia_high_eu: float, asia_low_eu: float,
                      asia_high_gu: float, asia_low_gu: float) -> str:
    """
    SMT-bearish: only one sweeps Asia High during London.
    SMT-bullish: only one sweeps Asia Low during London.
    """
    l_eu = day_df_eu[(day_df_eu["time"].dt.hour >= LDN_START_H) & (day_df_eu["time"].dt.hour < LDN_END_H)]
    l_gu = day_df_gu[(day_df_gu["time"].dt.hour >= LDN_START_H) & (day_df_gu["time"].dt.hour < LDN_END_H)]

    eu_hi = (l_eu["high"].max() > asia_high_eu); gu_hi = (l_gu["high"].max() > asia_high_gu)
    eu_lo = (l_eu["low"].min()  < asia_low_eu);  gu_lo = (l_gu["low"].min()  < asia_low_gu)

    if (eu_hi and not gu_hi) or (gu_hi and not eu_hi):
        return "bearish"
    if (eu_lo and not gu_lo) or (gu_lo and not eu_lo):
        return "bullish"
    return "none"

def dxy_confirm_ok(m3_all: Dict[str, pd.DataFrame],
                   meta_map: Dict[str, SymbolMeta],
                   day_ts: pd.Timestamp,
                   desired_dir: str) -> Tuple[bool, dict]:
    """
    desired_dir is EU/GU direction ('buy' when Asia Low swept, 'sell' when Asia High).
    DXY must sweep the OPPOSITE Asia side inside London.
    If REQUIRE_DXY_CHOCH_FVG: DXY must also show opposite CHoCH+FVG within DXY_CONFIRM_TOL_BARS (no HTF confluence).
    """
    info = {"dxy_asia_path": None, "dxy_sweep_time": None, "dxy_detail_ok": None}
    if not ENABLE_DXY_CONFIRM or DXY_SYMBOL not in m3_all:
        info["note"] = "dxy_disabled_or_missing"
        return True, info

    dxy_df = m3_all[DXY_SYMBOL]
    dxy_day = dxy_df[dxy_df["time"].dt.floor("D") == day_ts].copy()
    if dxy_day.empty:
        info["note"] = "dxy_day_empty"
        return False, info

    asia_high_dxy, asia_low_dxy = get_day_asia_levels(dxy_df, day_ts)
    dxy_path, dxy_t = asia_sweep_side(dxy_day, asia_high_dxy, asia_low_dxy, LDN_START_H, LDN_END_H)
    info.update({"dxy_asia_path": dxy_path, "dxy_sweep_time": dxy_t})

    want_dxy = "high" if desired_dir == "buy" else "low"
    if dxy_path != want_dxy:
        info["note"] = "dxy_wrong_side"
        return False, info

    if not REQUIRE_DXY_CHOCH_FVG:
        info["dxy_detail_ok"] = True
        return True, info

    # Optional detailed CHoCH+FVG requirement near sweep (no HTF confluence here)
    meta = meta_map.get(DXY_SYMBOL)
    if meta is None or dxy_t is None:
        info["note"] = "dxy_meta_or_time_missing"
        return False, info

    # Find index of sweep time in full DXY df
    i0 = dxy_df["time"].searchsorted(pd.to_datetime(dxy_t))
    i1 = min(i0 + DXY_CONFIRM_TOL_BARS, len(dxy_df) - 1)
    desired_for_dxy = "high" if want_dxy == "high" else "low"  # same string; detect_choch_fvg_entry expects "high"/"low"

    # Scan forward for any CHoCH+FVG entry in the opposite direction to EU/GU
    opp_asia_path = desired_for_dxy  # we pass the actual DXY swept side
    opp_dir = "buy" if desired_dir == "sell" else "sell"

    ok = False
    for idx in range(i0, i1 + 1):
        got = detect_choch_fvg_entry(DXY_SYMBOL, dxy_df, meta, idx, opp_asia_path)
        if got:
            got_dir, *_ = got
            if (opp_dir == got_dir):
                ok = True
                break
    info["dxy_detail_ok"] = ok
    return ok, info

# ==================== TRADE SIMULATION ====================

@dataclass
class Trade:
    symbol: str
    direction: str
    placed_at: pd.Timestamp
    entry: float
    sl: float
    tp: float
    exit_at: Optional[pd.Timestamp] = None
    outcome: Optional[str] = None  # 'TP'/'SL'/'Open'
    R_at_fill: Optional[float] = None
    duration_min: Optional[float] = None
    notes: Optional[str] = None
    fvg_low: Optional[float] = None
    fvg_high: Optional[float] = None
    choch_break_time: Optional[pd.Timestamp] = None

def bar_hit(low, high, level) -> bool:
    return (low <= level <= high)

def simulate_exit_on_future_bars(df: pd.DataFrame, start_idx: int, tr: Trade) -> Trade:
    for j in range(start_idx + 1, len(df)):
        bar = df.iloc[j]
        lo, hi = bar.low, bar.high
        sl_hit = bar_hit(lo, hi, tr.sl)
        tp_hit = bar_hit(lo, hi, tr.tp)
        if sl_hit and tp_hit:
            tr.exit_at, tr.outcome = bar.time, "SL"
            break
        if tp_hit:
            tr.exit_at, tr.outcome = bar.time, "TP"
            break
        if sl_hit:
            tr.exit_at, tr.outcome = bar.time, "SL"
            break
    if tr.exit_at is None:
        tr.exit_at, tr.outcome = df.iloc[-1].time, "Open"
    try:
        tr.duration_min = float((pd.to_datetime(tr.exit_at) - pd.to_datetime(tr.placed_at)).total_seconds() / 60.0)
    except Exception:
        tr.duration_min = None
    return tr

# ==================== STATS HELPERS ======================

def longest_streak(series_outcomes: List[str], win=True) -> int:
    target = "TP" if win else "SL"
    streak = best = 0
    for o in series_outcomes:
        if o == target:
            streak += 1
            best = max(best, streak)
        elif o in ("TP", "SL"):
            streak = 0
    return best

def profit_factor(pnls: np.ndarray) -> float:
    gp = pnls[pnls > 0].sum()
    gl = -pnls[pnls < 0].sum()
    if gl > 0:
        return float(gp / gl)
    return math.inf if gp > 0 else 0.0

def sharpe_ratio(R: np.ndarray, eps: float = 1e-9, annualize: bool = False) -> float:
    if R.size == 0:
        return 0.0
    mu = float(R.mean())
    sd = float(R.std(ddof=1)) if R.size > 1 else 0.0
    sr = mu / (sd + eps)
    if annualize:
        sr *= math.sqrt(252)
    return sr

def max_drawdown(equity: np.ndarray) -> Tuple[float, int, int]:
    if equity.size == 0:
        return 0.0, -1, -1
    peaks = np.maximum.accumulate(equity)
    dd = peaks - equity
    i = int(np.argmax(dd))
    peak_idx = int(np.argmax(equity[:i+1])) if i >= 0 else -1
    return float(dd[i]), peak_idx, i

# ==================== BACKTEST CORE =======================

def backtest_symbol(symbol: str, start: datetime, end: datetime,
                    m3_cache_all: Dict[str, pd.DataFrame],
                    meta_map: Dict[str, SymbolMeta]) -> Tuple[pd.DataFrame, dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print(f"\n=== Backtesting {symbol} (M3) — Asia sweep ➜ HTF confluence ➜ CHoCH + FVG (NO HTF BIAS) ===")
    meta = meta_map.get(symbol)
    if meta is None:
        print(f"[{symbol}] No symbol_info; skipping.")
        return pd.DataFrame(), {}, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    m3 = m3_cache_all.get(symbol)
    if m3 is None or m3.empty or len(m3) < 500:
        print(f"[{symbol}] Not enough M3 data.")
        return pd.DataFrame(), {}, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # Precompute HTF zones
    zones = detect_htf_zones(symbol, meta, start, end)

    m3["day"] = m3["time"].dt.floor("D")
    trades: List[Trade] = []
    daily_log_rows: List[dict] = []

    all_days = m3["day"].drop_duplicates().tolist()
    for di, d in enumerate(all_days, start=1):
        if PRINT_EVERY_N_DAYS and di % PRINT_EVERY_N_DAYS == 0:
            print(f"[{symbol}] Processing day {di}/{len(all_days)} …")

        day_df = m3[m3["day"] == d].copy()
        if day_df.empty:
            continue

        day_ts = pd.Timestamp(d)
        prev_day = day_ts - pd.Timedelta(days=1)

        # ---- PD levels for logging only ----
        pdh = pdl = pwh = pwl = pmh = pml = np.nan
        if LOG_PDH_PDL:
            prev_from, prev_to = day_bounds(prev_day)
            prev_slice = m3[(m3["time"] >= prev_from) & (m3["time"] < prev_to)]
            if not prev_slice.empty:
                pdh = float(prev_slice["high"].max()); pdl = float(prev_slice["low"].min())
        if LOG_PWH_PWL:
            wk_from, wk_to = week_bounds(prev_day)
            wk_slice = m3[(m3["time"] >= wk_from) & (m3["time"] < wk_to)]
            if not wk_slice.empty:
                pwh = float(wk_slice["high"].max()); pwl = float(wk_slice["low"].min())
        if LOG_PMH_PML:
            mo_from, mo_to = month_bounds(prev_day)
            mo_slice = m3[(m3["time"] >= mo_from) & (m3["time"] < mo_to)]
            if mo_slice.empty:
                mo_slice = m3[(m3["time"] >= (prev_day - pd.Timedelta(days=30))) & (m3["time"] < prev_day)]
            if not mo_slice.empty:
                pmh = float(mo_slice["high"].max()); pml = float(mo_slice["low"].min())

        # Asia session range
        asia_high, asia_low = get_day_asia_levels(m3, day_ts)
        if not np.isfinite(asia_high) or not np.isfinite(asia_low):
            daily_log_rows.append({"symbol": symbol, "day": day_ts})
            continue

        # STEP 1: detect first Asia sweep inside London for this symbol
        asia_path, sweep_time = asia_sweep_side(day_df, asia_high, asia_low, LDN_START_H, LDN_END_H)

        # ---------- Pairing & SMT (only enforced for EU/GU) ----------
        smt_signal = "none"
        if symbol in ("EURUSD", "GBPUSD") and ENABLE_DXY_CONFIRM:
            if ("EURUSD" in m3_cache_all) and ("GBPUSD" in m3_cache_all):
                eu_df = m3_cache_all["EURUSD"]; gu_df = m3_cache_all["GBPUSD"]
                eu_day = eu_df[eu_df["time"].dt.floor("D") == day_ts]
                gu_day = gu_df[gu_df["time"].dt.floor("D") == day_ts]
                if not eu_day.empty and not gu_day.empty:
                    eu_ah, eu_al = get_day_asia_levels(eu_df, day_ts)
                    gu_ah, gu_al = get_day_asia_levels(gu_df, day_ts)
                    eu_path, eu_t = asia_sweep_side(eu_day, eu_ah, eu_al, LDN_START_H, LDN_END_H)
                    gu_path, gu_t = asia_sweep_side(gu_day, gu_ah, gu_al, LDN_START_H, LDN_END_H)

                    # SMT logging
                    if LOG_SMT_EU_GU:
                        smt_signal = compute_smt_eu_gu(eu_day, gu_day, eu_ah, eu_al, gu_ah, gu_al)

                    pair_ok = (eu_path is not None) and (gu_path is not None) and (eu_path == gu_path)
                    pair_dir = eu_path if symbol == "EURUSD" else gu_path

                    # DXY confirmation (opposite side + optional CHoCH+FVG)
                    if pair_ok:
                        dxy_ok, dxy_info = dxy_confirm_ok(m3_cache_all, meta_map, day_ts,
                                                          desired_dir=("buy" if pair_dir == "low" else "sell"))
                    else:
                        dxy_ok, dxy_info = False, {"note": "pair_not_aligned"}

                    if not (pair_ok and dxy_ok):
                        daily_log_rows.append({
                            "symbol": symbol, "day": day_ts,
                            "asia_high": asia_high, "asia_low": asia_low,
                            "pdh": pdh, "pdl": pdl, "pwh": pwh, "pwl": pwl, "pmh": pmh, "pml": pml,
                            "asia_path": asia_path, "asia_sweep_time": sweep_time,
                            "smt_signal": smt_signal,
                            "pair_eu_path": eu_path, "pair_gu_path": gu_path, "pair_ok": pair_ok,
                            **{f"dxy_{k}": v for k, v in dxy_info.items()}
                        })
                        continue
                    else:
                        # lock direction for this day from the pair
                        asia_path = pair_dir
                        sweep_time = eu_t if symbol == "EURUSD" else gu_t
                        daily_log_rows.append({
                            "symbol": symbol, "day": day_ts,
                            "asia_high": asia_high, "asia_low": asia_low,
                            "pdh": pdh, "pdl": pdl, "pwh": pwh, "pwl": pwl, "pmh": pmh, "pml": pml,
                            "asia_path": asia_path, "asia_sweep_time": sweep_time,
                            "smt_signal": smt_signal, "pair_ok": True,
                            "pair_eu_path": eu_path, "pair_gu_path": gu_path,
                            "dxy_ok": True
                        })
                else:
                    daily_log_rows.append({
                        "symbol": symbol, "day": day_ts,
                        "asia_high": asia_high, "asia_low": asia_low,
                        "note": "pair_day_missing", "smt_signal": "none"
                    })
                    continue
            else:
                daily_log_rows.append({
                    "symbol": symbol, "day": day_ts,
                    "asia_high": asia_high, "asia_low": asia_low,
                    "note": "pair_cache_missing", "smt_signal": "none"
                })
                continue
        else:
            # normal logging when not enforcing pair
            daily_log_rows.append({
                "symbol": symbol, "day": day_ts,
                "asia_high": asia_high, "asia_low": asia_low,
                "pdh": pdh, "pdl": pdl, "pwh": pwh, "pwl": pwl, "pmh": pmh, "pml": pml,
                "asia_path": asia_path, "asia_sweep_time": sweep_time,
                "smt_signal": smt_signal
            })

        if asia_path is None:
            continue

        desired_dir = 'sell' if asia_path == 'high' else 'buy'

        # STEP 2: after the Asia sweep, require HTF confluence and CHoCH+FVG for entry
        took_trade = False
        for _, bar in day_df.reset_index(drop=True).iterrows():
            t = bar["time"]
            if not (LDN_START_H <= t.hour < LDN_END_H) or (sweep_time is not None and t < sweep_time):
                continue

            # Required HTF confluence (for the trading symbol only)
            if not confluence_ok(t, bar.low, bar.high, zones, desired_dir):
                continue

            # Find index in full DF
            idx_arr = m3.index[(m3["time"] == t)]
            if len(idx_arr) == 0:
                continue
            idx = int(idx_arr[0])

            setup = detect_choch_fvg_entry(symbol, m3, meta, idx, asia_path)
            if not setup:
                continue
            sdir, entry, sl, dbg = setup
            if sdir != desired_dir:
                continue

            # ----- TP selection with "min 2R to Asia" logic -----
            if TP_MODE == "asia_min2R":
                asia_target = asia_high if sdir == "buy" else asia_low
                asia_target = round_to_tick(asia_target, meta)

                risk   = abs(entry - sl)
                reward = abs(asia_target - entry)
                rr = (reward / risk) if risk > 0 else 0.0

                if rr < 2.0:
                    # extend to exactly 2R
                    if sdir == "buy":
                        tp = round_to_tick(entry + 2.0 * risk, meta)
                    else:
                        tp = round_to_tick(entry - 2.0 * risk, meta)
                else:
                    tp = asia_target
            else:
                # plain fixed RR
                if sdir == "buy":
                    tp = round_to_tick(entry + RR_TARGET * (entry - sl), meta)
                else:
                    tp = round_to_tick(entry - RR_TARGET * (sl - entry), meta)

            # Geometry sanity
            if sdir == "buy" and not (sl < entry < tp):
                continue
            if sdir == "sell" and not (tp < entry < sl):
                continue

            tr = Trade(
                symbol=symbol, direction=sdir, placed_at=t,
                entry=entry, sl=sl, tp=tp,
                notes=f"Asia {asia_path} sweep; HTF confluence; CHoCH+FVG pullback",
                fvg_low=float(dbg["fvg"]["low"]), fvg_high=float(dbg["fvg"]["high"]),
                choch_break_time=pd.to_datetime(m3.iloc[dbg["choch_break_idx"]].time)
            )
            tr = simulate_exit_on_future_bars(m3, idx, tr)
            if tr.outcome == "TP":
                risk = abs(tr.entry - tr.sl)
                reward = abs(tr.tp - tr.entry)
                tr.R_at_fill = reward / risk if risk > 0 else np.nan

            trades.append(tr)
            took_trade = True
            break  # first valid CHoCH+FVG after sweep

        if ONE_TRADE_PER_DAY and took_trade:
            continue

    # ==== Realize PnL & outputs ====  (robust even if no trades)
    rows = []
    equity = START_BALANCE
    equity_rows = []
    for tr in trades:
        if tr.outcome == "TP":
            risk = abs(tr.entry - tr.sl)
            reward = abs(tr.tp - tr.entry)
            R = (reward / risk) if risk > 0 else 0.0
        elif tr.outcome == "SL":
            R = -1.0
        else:
            R = 0.0
        risk_money = equity * (RISK_PER_TRADE / 100.0)
        pnl = R * risk_money
        equity += pnl

        rows.append({
            "symbol": tr.symbol, "direction": tr.direction, "placed_at": tr.placed_at,
            "entry": tr.entry, "sl": tr.sl, "tp": tr.tp, "exit_at": tr.exit_at, "outcome": tr.outcome,
            "R": R, "pnl": pnl, "equity_after": equity, "duration_min": tr.duration_min, "notes": tr.notes,
            "fvg_low": tr.fvg_low, "fvg_high": tr.fvg_high, "choch_break_time": tr.choch_break_time
        })
        equity_rows.append({"time": tr.exit_at if tr.exit_at is not None else tr.placed_at, "equity": equity})

    df_trades = pd.DataFrame(rows)
    if not df_trades.empty:
        df_trades = df_trades.sort_values(["symbol", "placed_at"]).reset_index(drop=True)
    else:
        df_trades = pd.DataFrame(columns=[
            "symbol","direction","placed_at","entry","sl","tp","exit_at","outcome","R","pnl",
            "equity_after","duration_min","notes","fvg_low","fvg_high","choch_break_time"
        ])

    df_equity = pd.DataFrame(equity_rows)
    if not df_equity.empty:
        df_equity = df_equity.sort_values("time").reset_index(drop=True)
    else:
        df_equity = pd.DataFrame(columns=["time","equity"])

    df_daily  = pd.DataFrame(daily_log_rows)

    # Per-symbol stats
    summary: Dict[str, object] = {}
    if not df_trades.empty:
        sym_trs = df_trades
        wins = int((sym_trs["outcome"] == "TP").sum())
        losses = int((sym_trs["outcome"] == "SL").sum())
        opens = int((sym_trs["outcome"] == "Open").sum())
        wr = round(100.0 * wins / max(1, len(sym_trs)), 2)

        streak_win  = longest_streak(sym_trs["outcome"].tolist(), win=True)
        streak_lose = longest_streak(sym_trs["outcome"].tolist(), win=False)

        maxR   = float(sym_trs["R"].max()) if len(sym_trs) else 0.0
        maxPnL = float(sym_trs["pnl"].max()) if len(sym_trs) else 0.0

        pf = profit_factor(sym_trs["pnl"].values)
        sr = sharpe_ratio(sym_trs["R"].values, annualize=ANNUALIZE_SHARPE)

        durs = sym_trs["duration_min"].dropna().values
        dur_mean = float(np.mean(durs)) if durs.size else np.nan
        dur_med  = float(np.median(durs)) if durs.size else np.nan
        dur_max  = float(np.max(durs)) if durs.size else np.nan
        dur_min  = float(np.min(durs)) if durs.size else np.nan

        eq_series = sym_trs["equity_after"].values
        mdd, _, _ = max_drawdown(eq_series)

        avg_R_all  = float(sym_trs["R"].mean()) if len(sym_trs) else np.nan
        wins_only  = sym_trs[sym_trs["outcome"] == "TP"]["R"]
        avg_R_wins = float(wins_only.mean()) if len(wins_only) else np.nan

        summary = {
            "symbol": symbol,
            "trades": int(len(sym_trs)),
            "wins": wins, "losses": losses, "open": opens,
            "winrate%": wr,
            "sum_R": round(float(sym_trs["R"].sum()), 2),
            "pnl": round(float(sym_trs["pnl"].sum()), 2),
            "equity_end": round(float(sym_trs["equity_after"].iloc[-1]), 2),
            "longest_win_streak": int(streak_win),
            "longest_lose_streak": int(streak_lose),
            "highest_win_R": round(maxR, 2),
            "highest_win_pnl": round(maxPnL, 2),
            "profit_factor": round(float(pf), 3) if math.isfinite(pf) else float('inf'),
            "sharpe_per_trade": round(float(sr), 3),
            "avg_duration_min": round(dur_mean, 1) if not math.isnan(dur_mean) else np.nan,
            "med_duration_min": round(dur_med, 1) if not math.isnan(dur_med) else np.nan,
            "max_duration_min": round(dur_max, 1) if not math.isnan(dur_max) else np.nan,
            "min_duration_min": round(dur_min, 1) if not math.isnan(dur_min) else np.nan,
            "max_drawdown_money": round(float(mdd), 2),
            "avg_R_all": round(avg_R_all, 3) if not math.isnan(avg_R_all) else np.nan,
            "avg_R_wins": round(avg_R_wins, 3) if not math.isnan(avg_R_wins) else np.nan,
        }

    return df_trades, summary, df_equity, df_daily, m3

# ==================== PLOTTING (TradingView-style) ==================

# --- colors & styling (dark theme) ---
_BG      = "#0f1115"
_GRID    = "#23262d"
_WICK    = "#b7bcc5"
_BODY_UP = "#e6e8eb"
_BODY_DN = "#aeb3b8"
_TXT     = "#cfd3da"

_ASIA    = "#7fd8ff"
_LDNBOX  = "#133b4b"   # London session shading
_TRADEBOX= "#114153"   # trade duration box
_HTF_BULL= "#2a7f63"
_HTF_BEAR= "#5b3b8d"
_FVG     = "#c24b4b"

plt.rcParams["axes.facecolor"]   = _BG
plt.rcParams["figure.facecolor"] = _BG
plt.rcParams["savefig.facecolor"]= _BG
plt.rcParams["axes.edgecolor"]   = _GRID
plt.rcParams["xtick.color"]      = _TXT
plt.rcParams["ytick.color"]      = _TXT
plt.rcParams["text.color"]       = _TXT
plt.rcParams["axes.labelcolor"]  = _TXT

def _tv_candles(ax, df_window: pd.DataFrame):
    xs = np.arange(len(df_window))
    for i, r in enumerate(df_window.itertuples(index=False)):
        o, h, l, c = r.open, r.high, r.low, r.close
        # wick
        ax.vlines(xs[i], l, h, linewidth=1, color=_WICK)
        # body
        lower = min(o, c)
        height = max(abs(c - o), 1e-12)
        face = _BODY_UP if c >= o else _BODY_DN
        rect = Rectangle((xs[i]-0.35, lower), 0.7, height,
                         linewidth=0, facecolor=face, alpha=1.0)
        ax.add_patch(rect)
    return xs

def _x_at_time(view_times: pd.Series, xs: np.ndarray, ts: pd.Timestamp) -> int:
    j = view_times.searchsorted(pd.to_datetime(ts))
    j = max(0, min(j, len(view_times)-1))
    return int(xs[j])

def _shade_session(ax, xs, view_times, day_ts, start_h, end_h, color, alpha=0.25, label=None):
    s = day_ts.replace(hour=start_h, minute=0, second=0, microsecond=0)
    e = day_ts.replace(hour=end_h,   minute=0, second=0, microsecond=0)
    x0 = _x_at_time(view_times, xs, s)
    x1 = _x_at_time(view_times, xs, e)
    ax.axvspan(x0, x1, color=color, alpha=alpha)
    if label:
        ax.text((x0+x1)/2, ax.get_ylim()[1], label, ha="center", va="top", alpha=0.8)

def _label_hline(ax, y, text, ls="--", lw=1, color=_TXT, ypad=0.0):
    ax.axhline(y, linestyle=ls, linewidth=lw, color=color, alpha=0.7)
    ax.text(ax.get_xlim()[0], y+ypad, f" {text}", va="bottom", color=color, alpha=0.9)

def plot_trade_candles(symbol: str,
                       m3_df: pd.DataFrame,
                       daily_df: pd.DataFrame,
                       trade_row: pd.Series,
                       lookback_bars: int,
                       lookahead_bars: int,
                       out_path: Optional[str] = None):
    placed = pd.to_datetime(trade_row["placed_at"])
    exit_at = pd.to_datetime(trade_row["exit_at"]) if pd.notnull(trade_row["exit_at"]) else placed

    # daily row for annotations
    day = placed.floor("D")
    drow = None
    if not daily_df.empty:
        ddf = daily_df[(daily_df["symbol"] == symbol) & (daily_df["day"] == day)]
        if not ddf.empty:
            drow = ddf.iloc[0]

    idx = m3_df["time"].searchsorted(placed)
    i0 = max(0, idx - lookback_bars)
    i1 = min(len(m3_df) - 1, idx + lookahead_bars)
    view = m3_df.iloc[i0:i1 + 1].reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(15, 7.5))
    xs = _tv_candles(ax, view)

    # X ticks
    step = max(1, len(xs) // 10)
    ax.set_xticks(xs[::step])
    ax.set_xticklabels(view["time"].dt.strftime("%H:%M")[::step], rotation=0, ha="center")
    ax.grid(True, linewidth=0.6, color=_GRID, alpha=0.6)

    # Session shading (London)
    if drow is not None:
        _shade_session(ax, xs, view["time"], day, LDN_START_H, LDN_END_H, _LDNBOX, 0.25, label="London")

    # Asia range lines + label
    if drow is not None and pd.notnull(drow.get("asia_high", np.nan)) and pd.notnull(drow.get("asia_low", np.nan)):
        _label_hline(ax, drow["asia_high"], "Asian High", ls="-", lw=1, color=_ASIA, ypad=0.0)
        _label_hline(ax, drow["asia_low"],  "Asian Low",  ls="-", lw=1, color=_ASIA, ypad=0.0)
        y_mid = 0.5*(drow["asia_high"]+drow["asia_low"])
        ax.text(ax.get_xlim()[0]+1, y_mid, "Asian", color=_ASIA, alpha=0.7)

    # PD (optional confluence) thin dashed lines
    if drow is not None:
        for key in ("pdh","pdl","pwh","pwl","pmh","pml"):
            val = drow.get(key, np.nan)
            if pd.notnull(val):
                _label_hline(ax, val, key.upper(), ls=":", lw=0.9, color="#6e7380")

    # CHoCH leg's top FVG band saved in trade_row
    if "fvg_low" in trade_row and "fvg_high" in trade_row and pd.notnull(trade_row["fvg_low"]) and pd.notnull(trade_row["fvg_high"]):
        lo, hi = float(trade_row["fvg_low"]), float(trade_row["fvg_high"])
        ax.axhspan(lo, hi, color=_FVG, alpha=0.15)
        ax.text(xs[0]+2, hi, f"leg FVG", color=_FVG, alpha=0.9, va="bottom")

    # Entry/SL/TP + trade duration box
    x_entry = _x_at_time(view["time"], xs, placed)
    x_exit  = _x_at_time(view["time"], xs, exit_at)
    y_entry = float(trade_row["entry"])
    y_sl    = float(trade_row["sl"])
    y_tp    = float(trade_row["tp"])

    y0 = min(y_sl, y_tp)
    y1 = max(y_sl, y_tp)
    rect = Rectangle((x_entry, y0), max(1, x_exit - x_entry), y1 - y0,
                     facecolor=_TRADEBOX, edgecolor="none", alpha=0.25)
    ax.add_patch(rect)

    _label_hline(ax, y_entry, "entry", ls="--", lw=1.0, color="#d9dbe0")
    _label_hline(ax, y_sl,    "SL",    ls="--", lw=1.0, color="#ff7a7a")
    _label_hline(ax, y_tp,    "TP",    ls="--", lw=1.0, color="#76e3a1")

    ax.axvline(x_entry, linestyle=":", linewidth=1, color="#c7cbd3")
    ax.text(x_entry, ax.get_ylim()[1], " placed", va="top", rotation=90, alpha=0.9)
    if pd.notnull(exit_at):
        ax.axvline(x_exit, linestyle=":", linewidth=1, color="#c7cbd3")
        ax.text(x_exit, ax.get_ylim()[1], f" exit ({trade_row['outcome']})", va="top", rotation=90, alpha=0.9)

    # optional: mark Asia sweep time & CHoCH break time
    if drow is not None and pd.notnull(drow.get("asia_sweep_time", None)):
        x_sw = _x_at_time(view["time"], xs, pd.to_datetime(drow["asia_sweep_time"]))
        ax.axvline(x_sw, linestyle="--", linewidth=1, color="#9fb3c8", alpha=0.9)
        path = drow.get("asia_path","")
        ax.text(x_sw, ax.get_ylim()[1], f" Asia sweep ({path})", va="top", rotation=90, alpha=0.9)

    if "choch_break_time" in trade_row and pd.notnull(trade_row["choch_break_time"]):
        try:
            x_ch = _x_at_time(view["time"], xs, pd.to_datetime(trade_row["choch_break_time"]))
            ax.axvline(x_ch, linestyle="--", linewidth=1, color="#b992e8", alpha=0.9)
            ax.text(x_ch, ax.get_ylim()[1], f" CHoCH break", va="top", rotation=90, alpha=0.9)
        except Exception:
            pass

    ax.set_title(f"{symbol} — {trade_row['direction'].upper()} | {placed:%Y-%m-%d %H:%M}  |  {trade_row.get('notes','')}",
                 fontsize=12, color=_TXT)
    ax.set_ylabel("Price")
    ax.margins(x=0.01)
    fig.tight_layout()

    out_path = out_path or os.path.join(OUT_DIR, f"{symbol}_trade_plot.png")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"[{symbol}] Trade plot saved → {out_path}")

# ==================== MAIN RUNNER ========================

def main():
    if not mt5.initialize():
        raise RuntimeError(f"MT5 init failed: {mt5.last_error()}")

    try:
        os.makedirs(OUT_DIR, exist_ok=True)
        start, end = compute_date_range()
        print(f"Testing range (server): {start} → {end} (Asia ➜ HTF confluence ➜ CHoCH+FVG, min-2R Asia, EU/GU+DXY pairing)")

        # -------- First pass: ensure symbols visible, build meta map, load M3 for ALL symbols --------
        meta_map: Dict[str, SymbolMeta] = {}
        m3_cache: Dict[str, pd.DataFrame] = {}

        for sym in SYMBOLS:
            if not mt5.symbol_select(sym, True):
                print(f"[{sym}] not visible in Market Watch; skipping load.")
                continue
            meta = get_meta(sym)
            if meta: meta_map[sym] = meta
            df = load_bars(sym, TIMEFRAME, start, end)
            if df is None or df.empty:
                print(f"[{sym}] no data in range.")
                continue
            m3_cache[sym] = df

        # -------- Second pass: run backtests using the shared caches --------
        all_trades: List[pd.DataFrame] = []
        all_summ: List[dict] = []
        eq_curves: Dict[str, pd.DataFrame] = {}
        daily_logs: Dict[str, pd.DataFrame] = {}

        for sym in SYMBOLS:
            if sym not in m3_cache or sym not in meta_map:
                continue
            trades, summ, eq, daily, _m3 = backtest_symbol(sym, start, end, m3_cache, meta_map)

            if not trades.empty:
                trades_path = os.path.join(OUT_DIR, f"{sym}_M3_trades.csv")
                trades.to_csv(trades_path, index=False)
                print(f"[{sym}] trades: {len(trades)}  → {trades_path}")
                all_trades.append(trades)

            if summ:
                all_summ.append(summ)

            if not eq.empty:
                eq_curves[sym] = eq.assign(symbol=sym)

            if not daily.empty:
                daily_logs[sym] = daily

        # Combined trades CSV
        if all_trades:
            merged = pd.concat(all_trades, ignore_index=True).sort_values("placed_at")
            merged_path = os.path.join(OUT_DIR, "ALL_trades_M3.csv")
            merged.to_csv(merged_path, index=False)
            print(f"ALL trades → {merged_path}")

        # Per-symbol summaries
        if all_summ:
            df_summ = pd.DataFrame(all_summ)
            summ_path = os.path.join(OUT_DIR, "SUMMARY_M3_EXTENDED.csv")
            df_summ.to_csv(summ_path, index=False)
            print("\nSUMMARY (per symbol, extended):")
            print(df_summ.to_string(index=False))
        else:
            print("No trades generated. Check sessions/correlation/data availability.")

        # Overall stats across all symbols
        if all_trades:
            all_df = pd.concat(all_trades, ignore_index=True)
            wins = int((all_df["outcome"] == "TP").sum())
            losses = int((all_df["outcome"] == "SL").sum())
            opens = int((all_df["outcome"] == "Open").sum())
            wr = round(100.0 * wins / max(1, len(all_df)), 2)
            pf = profit_factor(all_df["pnl"].values)
            sr = sharpe_ratio(all_df["R"].values, annualize=ANNUALIZE_SHARPE)
            streak_win  = longest_streak(all_df["outcome"].tolist(), win=True)
            streak_lose = longest_streak(all_df["outcome"].tolist(), win=False)
            maxR   = float(all_df["R"].max()) if len(all_df) else 0.0
            maxPnL = float(all_df["pnl"].max()) if len(all_df) else 0.0
            durs = all_df["duration_min"].dropna().values
            dur_mean = float(np.mean(durs)) if durs.size else np.nan
            dur_med  = float(np.median(durs)) if durs.size else np.nan
            d_max    = float(np.max(durs)) if durs.size else np.nan
            d_min    = float(np.min(durs)) if durs.size else np.nan
            portfolio_eq = all_df.sort_values("exit_at")["pnl"].cumsum() + START_BALANCE
            mdd, _, _ = max_drawdown(portfolio_eq.values)
            avg_R_all  = float(all_df["R"].mean()) if len(all_df) else np.nan
            avg_R_wins = float(all_df[all_df["outcome"] == "TP"]["R"].mean()) if wins > 0 else np.nan

            overall = {
                "symbols": int(len(set(all_df['symbol']))),
                "total_trades": int(len(all_df)),
                "wins": wins, "losses": losses, "open": opens,
                "winrate%": wr,
                "overall_sum_R": round(float(all_df["R"].sum()), 2),
                "overall_pnl": round(float(all_df["pnl"].sum()), 2),
                "longest_win_streak": streak_win,
                "longest_lose_streak": streak_lose,
                "highest_win_R": round(maxR, 2),
                "highest_win_pnl": round(maxPnL, 2),
                "profit_factor": round(float(pf), 3) if math.isfinite(pf) else float('inf'),
                "sharpe_per_trade": round(float(sr), 3),
                "avg_duration_min": round(dur_mean, 1) if not math.isnan(dur_mean) else np.nan,
                "med_duration_min": round(dur_med, 1) if not math.isnan(dur_med) else np.nan,
                "max_duration_min": round(d_max, 1) if not math.isnan(d_max) else np.nan,
                "min_duration_min": round(d_min, 1) if not math.isnan(d_min) else np.nan,
                "max_drawdown_money": round(float(mdd), 2),
                "avg_R_all": round(avg_R_all, 3) if not math.isnan(avg_R_all) else np.nan,
                "avg_R_wins": round(avg_R_wins, 3) if not math.isnan(avg_R_wins) else np.nan,
            }
            overall_path = os.path.join(OUT_DIR, "SUMMARY_OVERALL_M3_EXTENDED.csv")
            pd.DataFrame([overall]).to_csv(overall_path, index=False)
            print("\nOVERALL (extended):")
            print(pd.DataFrame([overall]).to_string(index=False))
        else:
            print("No trades generated overall.")

        # Equity curves CSV
        if eq_curves:
            all_eq = pd.concat(eq_curves.values(), ignore_index=True).sort_values(["symbol", "time"])
            eq_path = os.path.join(OUT_DIR, "EQUITY_CURVES_M3.csv")
            all_eq.to_csv(eq_path, index=False)
            print(f"Equity curves → {eq_path}")

        # Candlestick visualizations
        if all_trades:
            merged_all = pd.concat(all_trades, ignore_index=True).sort_values("placed_at")
            # Plot specific request
            if (PLOT_SYMBOL in m3_cache) and (PLOT_SYMBOL in daily_logs) and ((merged_all["symbol"] == PLOT_SYMBOL).any()):
                sym_trades = merged_all[merged_all["symbol"] == PLOT_SYMBOL].sort_values("placed_at")
                if len(sym_trades) > PLOT_TRADE_INDEX:
                    tr_row = sym_trades.iloc[PLOT_TRADE_INDEX]
                    plot_trade_candles(
                        symbol=PLOT_SYMBOL,
                        m3_df=m3_cache[PLOT_SYMBOL],
                        daily_df=daily_logs.get(PLOT_SYMBOL, pd.DataFrame()),
                        trade_row=tr_row,
                        lookback_bars=PLOT_LOOKBACK_BARS,
                        lookahead_bars=PLOT_LOOKAHEAD_BARS,
                        out_path=os.path.join(OUT_DIR, f"{PLOT_SYMBOL}_trade_plot.png")
                    )
            # Plot latest per symbol
            for sym in SYMBOLS:
                if sym not in m3_cache or sym not in daily_logs:
                    continue
                sym_trades = merged_all[merged_all["symbol"] == sym].sort_values("placed_at")
                if sym_trades.empty:
                    continue
                last_row = sym_trades.iloc[-1]
                plot_trade_candles(
                    symbol=sym,
                    m3_df=m3_cache[sym],
                    daily_df=daily_logs.get(sym, pd.DataFrame()),
                    trade_row=last_row,
                    lookback_bars=PLOT_LOOKBACK_BARS,
                    lookahead_bars=PLOT_LOOKAHEAD_BARS,
                    out_path=os.path.join(OUT_DIR, f"{sym}_trade_plot_latest.png")
                )

    finally:
        mt5.shutdown()

if __name__ == "__main__":
    main()
