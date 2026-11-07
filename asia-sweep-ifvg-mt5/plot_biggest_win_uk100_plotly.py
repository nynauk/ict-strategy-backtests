# backtest_uk100_asia_ifvg_retest_POI_FILTER.py (UPDATED WITH IFVG PROFILE STRATEGY)
# ===============================================================
# Sessions & Logic
#   Profiles
#     • ASIA profile:
#         sweep: 03:00–10:00, execution: 10:00–14:00
#         symbols: EURUSD, GBPUSD, UK100.cash, GER40.cash, DXY.cash, GBPJPY
#     • LONDON profile (USD majors):
#         sweep: 09:00–14:00, execution (NY KZ): 14:00–17:00
#         symbols: USDCAD, US30.cash, US100.cash, USDJPY
#
# Rules (high level)
#   • Sweep = bias. High swept → SELL bias; Low swept → BUY bias. Only trade that direction.
#   • Take only the FIRST IFVG disrespect after the sweep window ends (on M5).
#   • HTF POI confluence required: clean unmitigated FVG/OB outside session range.
#       - If FVG: the disrespect/tap must occur in the golden pocket (0.382–0.618) of that FVG.
#       - Fallback: previous period level (PDH/PDL/PWH/PWL/PMH/PML) outside session range and untouched during session.
#   • EURUSD/GBPUSD must pass DXY gate:
#       - DXY bias must be OPPOSITE.
#       - DXY must print its FIRST IFVG in the direction of its own bias before/at entry.
#   • Entry type:
#       - Compute gap = |FVG_hi - FVG_lo|, body = |close - open| of disrespect candle,
#         distance = close-to-gap-edge distance in breakout direction.
#       - If body ≥ 2×gap OR distance > gap*(1+0.618): use LIMIT at 50% of the disrespected gap.
#         Else: MARKET at disrespect close.
#   • SL: beyond relevant swing (above for sells; below for buys). TP = 2R fixed.
#   • Risk: 1% dynamic. One trade per symbol per day. Kill switch after 3 SLs in a day.
#   • Logging includes profile, sweep, IFVG metrics, entry type, POI details, DXY status.
# ===============================================================

import os, math, calendar, sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import MetaTrader5 as mt5
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ====================== USER INPUTS ======================

# Default symbols (you can change)
SYMBOLS = ["EURUSD", "GER40.cash", "GBPUSD", "UK100.cash", "DXY.cash", "US100.cash", "USDCAD", "USDJPY", "GBPJPY", "US30.cash"]

# Use M5 per strategy
TIMEFRAME = mt5.TIMEFRAME_M5

# --- Data Range (choose ONE mode) ---
RANGE_MODE = "months"   # "months" | "days" | "dates"
RANGE_MONTHS = 6
RANGE_DAYS = 120
DATE_FROM = "2024-07-01"
DATE_TO   = "2025-08-22"

# Profile definitions (server time)
PROFILES = {
    "ASIA": {
        "sweep_start": 3, "sweep_end": 10,
        "exec_start": 10, "exec_end": 14,
        "symbols": {"EURUSD","GBPUSD","UK100.cash","GER40.cash","DXY.cash","GBPJPY"}
    },
    "LONDON": {
        "sweep_start": 9, "sweep_end": 14,
        "exec_start": 14, "exec_end": 17,  # NY KZ
        "symbols": {"USDCAD","US30.cash","US100.cash","USDJPY"}
    }
}

# IFVG detection & entry
MIN_GAP_TICKS = 5
GOLDEN_LOW = 0.382
GOLDEN_HIGH = 0.618

# DXY confluence (EURUSD & GBPUSD)
USE_DXY_FILTER = True
DXY_CANDIDATES = ["DXY", "USDX", "USDOLLAR", "DX", "DOLLAR", "DXY.cash"]

# Manipulation/POI stage required
USE_MANIPULATION_POI_STAGE = True

# ================== TAKE-PROFIT MODE =====================
TP_MODE = "FIXED_RR"
RR_TARGET = 2.0

# Risk & accounting
RISK_PER_TRADE = 1.0
START_BALANCE  = 100000.0  # start 100k
DAILY_KILL_LOSSES = 3      # daily kill switch

ONE_TRADE_PER_DAY = True
OUT_DIR = "bt_out"
PLOTS_DIR = os.path.join(OUT_DIR, "plots")
EQUITY_DIR = os.path.join(OUT_DIR, "equity")

PRINT_EVERY_N_DAYS = 5
ANNUALIZE_SHARPE = False

# --- HTF Bias Filter (toggleable) ---
USE_HTF_BIAS = False          # turn ON/OFF the bias filter
BIAS_METHOD = "EMA"
BIAS_EMA_PERIOD = 50
BIAS_TFS = ("H4",)
BIAS_REQUIRE_ALL = False

# --- HTF POI toggles ---
FVG_TFS_ENABLED = {"D1": True, "H4": True, "H1": True}
OB_TFS_ENABLED  = {"D1": False, "H4": False, "H1": False}

# --- Prev-period highs/lows toggles ---
PREV_LEVELS_ENABLED = {
    "PDH": True, "PDL": True,
    "PWH": True, "PWL": True,
    "PMH": True, "PML": True,
}

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

# ===================== DXY HANDLING ======================

def resolve_dxy_symbol() -> Tuple[Optional[str], bool]:
    if not USE_DXY_FILTER:
        return None, False
    for cand in DXY_CANDIDATES:
        info = mt5.symbol_info(cand)
        if info:
            if mt5.symbol_select(cand, True):
                return cand, True
    print("[WARN] DXY symbol not found; disabling DXY filter for EURUSD/GBPUSD.")
    return None, False

DXY_SYMBOL: Optional[str] = None
DXY_ENABLED: bool = False
DXY_DF: Optional[pd.DataFrame] = None

# ==================== SESSION HELPERS ====================

def get_session_levels(df: pd.DataFrame, day_ts: pd.Timestamp, h_start: int, h_end: int) -> Tuple[float, float]:
    s = day_ts.replace(hour=h_start, minute=0, second=0, microsecond=0)
    e = day_ts.replace(hour=h_end,   minute=0, second=0, microsecond=0)
    slc = df[(df["time"] >= s) & (df["time"] < e)]
    return float(slc["high"].max()), float(slc["low"].min())

def first_sweep_of_range(day_df: pd.DataFrame, high_level: float, low_level: float,
                         start_h: int, end_h: int, after_ts: Optional[pd.Timestamp]=None) -> Tuple[Optional[str], Optional[pd.Timestamp]]:
    w = day_df[(day_df["time"].dt.hour >= start_h) & (day_df["time"].dt.hour < end_h)].copy()
    if after_ts is not None:
        w = w[w["time"] >= after_ts]
    for _, bar in w.iterrows():
        if bar.high > high_level:
            return "high", pd.Timestamp(bar.time)
        if bar.low < low_level:
            return "low", pd.Timestamp(bar.time)
    return None, None

# ==================== PREV PERIOD LEVELS =================

def day_bounds(ts: pd.Timestamp) -> Tuple[pd.Timestamp, pd.Timestamp]:
    s = ts.replace(hour=0, minute=0, second=0, microsecond=0)
    e = s + pd.Timedelta(days=1)
    return s, e

def week_bounds(ts: pd.Timestamp) -> Tuple[pd.Timestamp, pd.Timestamp]:
    weekday = ts.weekday()
    monday = (ts - pd.Timedelta(days=weekday)).replace(hour=0, minute=0, second=0, microsecond=0)
    next_monday = monday + pd.Timedelta(days=7)
    return monday, next_monday

def month_bounds(ts: pd.Timestamp) -> Tuple[pd.Timestamp, pd.Timestamp]:
    start = ts.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    last_day = calendar.monthrange(ts.year, ts.month)[1]
    end = start.replace(day=last_day) + pd.Timedelta(days=1)
    return start, end

def prev_period_hilo(df_d1: pd.DataFrame, day_ts: pd.Timestamp) -> Dict[str, Optional[float]]:
    pd_s, _ = day_bounds(day_ts - pd.Timedelta(days=1))
    pw_s, pw_e = week_bounds(day_ts - pd.Timedelta(days=7))
    pm_ref = (day_ts - pd.DateOffset(months=1))
    pm_s = pm_ref.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    pm_e = day_ts.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

    def hl(slice_df):
        if slice_df.empty:
            return None, None
        return float(slice_df["high"].max()), float(slice_df["low"].min())

    pday = df_d1[(df_d1["time"] >= pd_s) & (df_d1["time"] < pd_s + pd.Timedelta(days=1))]
    pdh, pdl = hl(pday)

    pweek = df_d1[(df_d1["time"] >= pw_s) & (df_d1["time"] < pw_e)]
    pwh, pwl = hl(pweek)

    pmonth = df_d1[(df_d1["time"] >= pm_s) & (df_d1["time"] < pm_e)]
    pmh, pml = hl(pmonth)

    return {"PDH": pdh, "PDL": pdl, "PWH": pwh, "PWL": pwl, "PMH": pmh, "PML": pml}

# ==================== HTF STRUCTURES ======================

def is_valid_gap(gap_low: float, gap_high: float, meta: SymbolMeta) -> bool:
    return (float(gap_high) - float(gap_low)) >= MIN_GAP_TICKS * meta.tick_size

@dataclass
class Zone:
    tf: str               # "D1" or "H4" or "H1"
    kind: str             # "bullish_fvg" | "bearish_fvg" | "bullish_ob" | "bearish_ob"
    formed_at: pd.Timestamp
    lo: float
    hi: float

def detect_fvgs(htf: pd.DataFrame, meta: SymbolMeta, tf_name: str) -> List[Zone]:
    zones: List[Zone] = []
    for i in range(3, len(htf)):
        c1 = htf.iloc[i-3]
        c3 = htf.iloc[i-1]
        c4 = htf.iloc[i]
        # Bullish FVG: gap up
        if (c1.high < c3.low) and is_valid_gap(c1.high, c3.low, meta):
            zones.append(Zone(tf_name, "bullish_fvg", pd.Timestamp(c4.time), float(c1.high), float(c3.low)))
        # Bearish FVG: gap down
        if (c1.low > c3.high) and is_valid_gap(c3.high, c1.low, meta):
            zones.append(Zone(tf_name, "bearish_fvg", pd.Timestamp(c4.time), float(c3.high), float(c1.low)))
    return zones

# === Order Blocks (swing-based) ===
OB_SWING_LEFT = 2
OB_SWING_RIGHT = 2
OB_CONFIRM_MIN_CANDLES = 2

def _is_swing_high(df: pd.DataFrame, i: int, left: int = OB_SWING_LEFT, right: int = OB_SWING_RIGHT) -> bool:
    hi = df.iloc[i]["high"]
    for k in range(i - left, i + right + 1):
        if k < 0 or k >= len(df) or k == i:
            continue
        if df.iloc[k]["high"] > hi:
            return False
    return True

def _is_swing_low(df: pd.DataFrame, i: int, left: int = OB_SWING_LEFT, right: int = OB_SWING_RIGHT) -> bool:
    lo = df.iloc[i]["low"]
    for k in range(i - left, i + right + 1):
        if k < 0 or k >= len(df) or k == i:
            continue
        if df.iloc[k]["low"] < lo:
            return False
    return True

def detect_order_blocks(htf: pd.DataFrame, tf_name: str) -> List[Zone]:
    zones: List[Zone] = []
    n = len(htf)
    if n < (OB_SWING_LEFT + OB_SWING_RIGHT + OB_CONFIRM_MIN_CANDLES + 3):
        return zones

    for i in range(OB_SWING_LEFT, n - max(OB_SWING_RIGHT, OB_CONFIRM_MIN_CANDLES) - 1):
        c = htf.iloc[i]
        o, h, l, cl = float(c["open"]), float(c["high"]), float(c["low"]), float(c["close"])

        c1 = htf.iloc[i+1]
        c2 = htf.iloc[i+2]

        # Bearish OB
        if cl > o and _is_swing_high(htf, i):
            if (c1["close"] < c1["open"]) and (c2["close"] < c2["open"]):
                if (float(c1["close"]) < l) or (float(c2["close"]) < l):
                    zones.append(Zone(tf_name, "bearish_ob", pd.Timestamp(htf.iloc[i+2]["time"]), l, h))

        # Bullish OB
        if cl < o and _is_swing_low(htf, i):
            if (c1["close"] > c1["open"]) and (c2["close"] > c2["open"]):
                if (float(c1["close"]) > h) or (float(c2["close"]) > h):
                    zones.append(Zone(tf_name, "bullish_ob", pd.Timestamp(htf.iloc[i+2]["time"]), l, h))
    return zones

def load_htf_structures(symbol: str, start: datetime, end: datetime, meta: SymbolMeta):
    pad_days = 120
    s = start - timedelta(days=pad_days)

    d1 = load_bars(symbol, mt5.TIMEFRAME_D1, s, end + timedelta(days=2))
    h4 = load_bars(symbol, mt5.TIMEFRAME_H4, s, end + timedelta(days=2))
    h1 = load_bars(symbol, mt5.TIMEFRAME_H1, s, end + timedelta(days=2))

    if d1 is None: d1 = pd.DataFrame()
    if h4 is None: h4 = pd.DataFrame()
    if h1 is None: h1 = pd.DataFrame()

    d1 = d1.sort_values("time").reset_index(drop=True)
    h4 = h4.sort_values("time").reset_index(drop=True)
    h1 = h1.sort_values("time").reset_index(drop=True)

    fvg_d1 = detect_fvgs(d1, meta, "D1") if FVG_TFS_ENABLED.get("D1", False) else []
    fvg_h4 = detect_fvgs(h4, meta, "H4") if FVG_TFS_ENABLED.get("H4", False) else []
    fvg_h1 = detect_fvgs(h1, meta, "H1") if FVG_TFS_ENABLED.get("H1", False) else []

    ob_d1  = detect_order_blocks(d1, "D1") if OB_TFS_ENABLED.get("D1", False) else []
    ob_h4  = detect_order_blocks(h4, "H4") if OB_TFS_ENABLED.get("H4", False) else []
    ob_h1  = detect_order_blocks(h1, "H1") if OB_TFS_ENABLED.get("H1", False) else []

    return d1, h4, fvg_d1 + fvg_h4 + fvg_h1, ob_d1 + ob_h4 + ob_h1

# ==================== HTF BIAS (EMA) ======================

def _add_ema(df: pd.DataFrame, period: int) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    out["ema"] = out["close"].ewm(span=period, adjust=False).mean()
    return out

def make_htf_bias_checker(d1_df: pd.DataFrame,
                          h4_df: pd.DataFrame,
                          period: int = 50,
                          require_all: bool = True):
    d1 = _add_ema(d1_df, period) if d1_df is not None else pd.DataFrame()
    h4 = _add_ema(h4_df, period) if h4_df is not None else pd.DataFrame()

    def _tf_vote(tf_df: pd.DataFrame, t: pd.Timestamp) -> Optional[str]:
        if tf_df is None or tf_df.empty:
            return None
        row = tf_df[tf_df["time"] <= t].tail(1)
        if row.empty or pd.isna(row["ema"].iloc[0]):
            return None
        c = float(row["close"].iloc[0]); e = float(row["ema"].iloc[0])
        if c > e:  return "buy"
        if c < e:  return "sell"
        return None

    def bias_at(t: pd.Timestamp) -> Optional[str]:
        votes = []
        if "D1" in BIAS_TFS: votes.append(_tf_vote(d1, t))
        if "H4" in BIAS_TFS: votes.append(_tf_vote(h4, t))
        votes = [v for v in votes if v is not None]
        if not votes:
            return None
        if require_all:
            if all(v == "buy" for v in votes):  return "buy"
            if all(v == "sell" for v in votes): return "sell"
            return None
        return "buy" if votes.count("buy") >= votes.count("sell") else "sell"

    return bias_at

# ==================== POI / GOLDEN-POCKET =================

def bar_intersects_zone(bar_low: float, bar_high: float, z: Zone) -> bool:
    return not (bar_high < z.lo or bar_low > z.hi)

def zone_strict_outside(z: Zone, mark_high: float, mark_low: float, polarity: str) -> bool:
    if polarity == "sell":
        return z.lo > mark_high
    else:
        return z.hi < mark_low

def zone_unmitigated(z: Zone, ltf: pd.DataFrame, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> bool:
    w = ltf[(ltf["time"] >= start_ts) & (ltf["time"] <= end_ts)]
    if w.empty: return True
    touched = ((w["high"] >= z.lo) & (w["low"] <= z.hi)).any()
    return not bool(touched)

def fvg_imbalance_intact(z: Zone, ltf: pd.DataFrame, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> bool:
    if "fvg" not in z.kind:
        return True
    w = ltf[(ltf["time"] >= start_ts) & (ltf["time"] <= end_ts)]
    if w.empty: return True
    gap_lo, gap_hi = z.lo, z.hi
    spanned = ((w["low"] <= gap_lo) & (w["high"] >= gap_hi)).any()
    mid = 0.5 * (gap_lo + gap_hi)
    mid_cross = ((w["low"] <= mid) & (w["high"] >= mid)).any()
    return not (spanned or mid_cross)

def in_golden_pocket(z: Zone, bar_low: float, bar_high: float) -> bool:
    if "fvg" not in z.kind:
        return True
    lo, hi = z.lo, z.hi
    g_lo = lo + (hi - lo) * GOLDEN_LOW
    g_hi = lo + (hi - lo) * GOLDEN_HIGH
    # Any part of bar range intersects golden pocket
    return not (bar_high < g_lo or bar_low > g_hi)

def prev_level_untouched_during_session(level: float, df: pd.DataFrame, s: pd.Timestamp, e: pd.Timestamp) -> bool:
    w = df[(df["time"] >= s) & (df["time"] < e)]
    if w.empty: return True
    # Consider a "touch" if any bar trades through level
    touched = ((w["low"] <= level) & (w["high"] >= level)).any()
    return not bool(touched)

def poi_filter(direction: str, bar_low: float, bar_high: float, bar_time: pd.Timestamp,
               mark_high: float, mark_low: float,
               fvg_zones: List[Zone], ob_zones: List[Zone],
               ltf_df: pd.DataFrame,
               session_start_ts: pd.Timestamp, session_end_ts: pd.Timestamp,
               prevs: Dict[str, Optional[float]]) -> Tuple[bool, str, Dict[str, bool]]:
    need = ("bearish_ob", "bearish_fvg") if direction == "sell" else ("bullish_ob", "bullish_fvg")
    candidates = [z for z in (ob_zones + fvg_zones) if z.kind in need and z.formed_at <= bar_time]
    candidates = [z for z in candidates if zone_strict_outside(z, mark_high, mark_low, direction)]
    candidates = [z for z in candidates if bar_intersects_zone(bar_low, bar_high, z)]
    # Golden pocket requirement for FVGs
    gp_candidates = []
    for z in candidates:
        if "fvg" in z.kind:
            if in_golden_pocket(z, bar_low, bar_high):
                gp_candidates.append(z)
        else:
            gp_candidates.append(z)
    candidates = gp_candidates
    # Nearest
    best = None; best_dist = None
    mid_price = 0.5 * (bar_low + bar_high)
    for z in candidates:
        z_mid = 0.5 * (z.lo + z.hi)
        d = abs(z_mid - mid_price)
        if best is None or d < best_dist:
            best, best_dist = z, d
    if best is None:
        # Fallback: previous levels untouched during session
        sweep_keys = ["PDH","PWH","PMH"] if direction=="sell" else ["PDL","PWL","PML"]
        for k in sweep_keys:
            if not PREV_LEVELS_ENABLED.get(k, True):
                continue
            v = prevs.get(k)
            if v is None:
                continue
            if direction=="sell" and v <= mark_high:
                continue
            if direction=="buy"  and v >= mark_low:
                continue
            if not prev_level_untouched_during_session(v, ltf_df, session_start_ts, session_end_ts):
                continue
            # Pass using prev level
            tag = f"prev_level:{k}"
            flags = {"poi_prev_level": True}
            return True, tag, flags
        return False, "", {}

    unmit = zone_unmitigated(best, ltf_df, best.formed_at, bar_time)
    imb  = fvg_imbalance_intact(best, ltf_df, best.formed_at, bar_time)
    if not unmit:
        return False, "", {}
    if ("fvg" in best.kind) and (not imb):
        return False, "", {}
    tag = f"tap:{best.tf}:{best.kind}:unmitigated"
    if "fvg" in best.kind:
        tag += ":imbalance_intact:golden_pocket"
    flags = {
        "poi_strict_outside_range": True,
        "poi_unmitigated": unmit,
        "poi_imbalance_intact": imb,
        "poi_tf_"+best.tf: True,
        "poi_kind": best.kind
    }
    return True, tag, flags

# ==================== IFVG & DISRESPECT ===================

def detect_ifvg_disrespect(df: pd.DataFrame, meta: SymbolMeta, i_now: int) -> Optional[Tuple[str, float, Tuple[float, float], float, float, float]]:
    """Return (dir, close_px, (fvg_lo,fvg_hi), gap, body, distance_to_edge)."""
    if i_now < 3: return None
    c0 = df.iloc[i_now]      # disrespect candle
    c1 = df.iloc[i_now - 3]
    c3 = df.iloc[i_now - 1]
    # Bullish FVG then disrespect down → SELL
    if (c1.high < c3.low) and is_valid_gap(c1.high, c3.low, meta):
        gap_lo  = float(c1.high)
        gap_hi  = float(c3.low)
        gap = gap_hi - gap_lo
        if float(c0.close) < gap_lo:
            body = abs(float(c0.close) - float(c0.open))
            distance = gap_lo - float(c0.close)  # how far below gap low the close is
            lo = min(float(c0.open), float(c0.close))
            hi = max(float(c0.open), float(c0.close))
            if not is_valid_gap(lo, hi, meta):
                eps = meta.tick_size * max(2, MIN_GAP_TICKS)
                lo, hi = float(c0.close) - eps, float(c0.close) + eps
            return ("sell", float(c0.close), (gap_lo, gap_hi), gap, body, distance)
    # Bearish FVG then disrespect up → BUY
    if (c1.low > c3.high) and is_valid_gap(c3.high, c1.low, meta):
        gap_lo  = float(c3.high)
        gap_hi  = float(c1.low)
        gap = gap_hi - gap_lo
        if float(c0.close) > gap_hi:
            body = abs(float(c0.close) - float(c0.open))
            distance = float(c0.close) - gap_hi  # how far above gap high the close is
            lo = min(float(c0.open), float(c0.close))
            hi = max(float(c0.open), float(c0.close))
            if not is_valid_gap(lo, hi, meta):
                eps = meta.tick_size * max(2, MIN_GAP_TICKS)
                lo, hi = float(c0.close) - eps, float(c0.close) + eps
            return ("buy", float(c0.close), (gap_lo, gap_hi), gap, body, distance)
    return None

def fvg_mid(fvg_lo: float, fvg_hi: float) -> float:
    return 0.5 * (float(fvg_lo) + float(fvg_hi))

# ==================== DXY CONFLUENCE =====================

@dataclass
class DxyConfluence:
    bias: Optional[str]  # 'buy'/'sell' from its own sweep
    first_buy_disrespect_time: Optional[pd.Timestamp]
    first_sell_disrespect_time: Optional[pd.Timestamp]

def compute_bias_from_sweep(sweep_side: Optional[str]) -> Optional[str]:
    if sweep_side == "high": return "sell"
    if sweep_side == "low":  return "buy"
    return None

def precompute_dxy_day_confluence(dxy_day_df: pd.DataFrame, dxy_meta: SymbolMeta,
                                  sweep_start_h: int, sweep_end_h: int) -> DxyConfluence:
    day_ts = dxy_day_df["time"].iloc[0].floor("D") if not dxy_day_df.empty else None
    sweep_side, _sweep_time = None, None
    if day_ts is not None:
        dxy_high, dxy_low = get_session_levels(dxy_day_df, day_ts, sweep_start_h, sweep_end_h)
        sweep_side, _sweep_time = first_sweep_of_range(dxy_day_df, dxy_high, dxy_low, sweep_start_h, sweep_end_h)
    # first IFVGs
    first_buy_t, first_sell_t = None, None
    for i in range(len(dxy_day_df)):
        det = detect_ifvg_disrespect(dxy_day_df, dxy_meta, i)
        if not det: continue
        sdir, _, _, _, _, _ = det
        t = pd.Timestamp(dxy_day_df.iloc[i]["time"])
        if sdir == "buy" and first_buy_t is None:
            first_buy_t = t
        if sdir == "sell" and first_sell_t is None:
            first_sell_t = t
        if first_buy_t and first_sell_t:
            break
    return DxyConfluence(compute_bias_from_sweep(sweep_side), first_buy_t, first_sell_t)

def dxy_gate_ok(symbol_dir: str, candidate_time: pd.Timestamp,
                dxy_conf: DxyConfluence) -> bool:
    """Require DXY bias opposite, and its first IFVG in bias direction already printed."""
    if dxy_conf is None or dxy_conf.bias is None:
        return False
    # opposite biases
    if symbol_dir == "buy" and dxy_conf.bias != "sell":
        return False
    if symbol_dir == "sell" and dxy_conf.bias != "buy":
        return False
    # require first IFVG in DXY bias direction
    if dxy_conf.bias == "buy":
        return (dxy_conf.first_buy_disrespect_time is not None and
                dxy_conf.first_buy_disrespect_time <= candidate_time)
    if dxy_conf.bias == "sell":
        return (dxy_conf.first_sell_disrespect_time is not None and
                dxy_conf.first_sell_disrespect_time <= candidate_time)
    return False

# ==================== TP / R computation =================

def compute_tp_and_R(entry: float, sl: float, direction: str,
                     meta: SymbolMeta) -> Tuple[float, float, str]:
    risk = abs(entry - sl)
    if risk <= 0:
        return entry, 0.0, "invalid"
    tp = entry + RR_TARGET * risk if direction == "buy" else entry - RR_TARGET * risk
    tp = round_to_tick(float(tp), meta)
    r_target = abs(tp - entry) / risk
    return tp, r_target, f"{RR_TARGET:.2f}R"

# ==================== TRADE SIMULATION ====================

@dataclass
class Trade:
    symbol: str
    direction: str
    profile: str
    placed_at: pd.Timestamp
    entry: float
    sl: float
    tp: float
    r_target: float
    exit_at: Optional[pd.Timestamp] = None
    outcome: Optional[str] = None
    duration_min: Optional[float] = None
    notes: Optional[str] = None
    bias_source: Optional[str] = None
    sweep_side: Optional[str] = None
    session_sweep_high: Optional[float] = None
    session_sweep_low: Optional[float] = None
    ifvg_gap: Optional[float] = None
    ifvg_body: Optional[float] = None
    ifvg_distance: Optional[float] = None
    entry_type: Optional[str] = None  # "market" or "limit50"
    poi_tag: Optional[str] = None
    dxy_ok: Optional[bool] = None

def bar_hit(low, high, level) -> bool:
    return (low <= level <= high)

def simulate_exit_on_future_bars(df: pd.DataFrame, start_idx: int, tr: Trade) -> Trade:
    for j in range(start_idx + 1, len(df)):
        bar = df.iloc[j]
        lo, hi = bar.low, bar.high
        sl_hit = bar_hit(lo, hi, tr.sl)
        tp_hit = bar_hit(lo, hi, tr.tp)
        if sl_hit and tp_hit:
            tr.exit_at, tr.outcome = bar.time, "SL"  # worst-case
            break
        if tp_hit:
            tr.exit_at, tr.outcome = bar.time, "TP"
            break
        if sl_hit:
            tr.exit_at, tr.outcome = bar.time, "SL"
            break
    if tr.exit_at is None:
        tr.exit_at, tr.outcome = df.iloc[-1].time, "Open"
    tr.duration_min = float((pd.to_datetime(tr.exit_at) - pd.to_datetime(tr.placed_at)).total_seconds() / 60.0)
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

def swing_sl(df: pd.DataFrame, i: int, direction: str) -> float:
    """Simple swing-based SL using last 5 bars' extremes before/including i."""
    start_i = max(0, i - 4)
    window = df.iloc[start_i:i+1]
    return float(window["low"].min()) if direction == "buy" else float(window["high"].max())

def backtest_symbol(symbol: str, start: datetime, end: datetime) -> Tuple[pd.DataFrame, dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    meta = get_meta(symbol)
    if meta is None:
        print(f"[{symbol}] No symbol_info; skipping.")
        return pd.DataFrame(), {}, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    df = load_bars(symbol, TIMEFRAME, start, end)
    if df is None or df.empty or len(df) < 500:
        print(f"[{symbol}] Not enough data.")
        return pd.DataFrame(), {}, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    df["day"] = df["time"].dt.floor("D")

    # HTF data + structures
    d1_df, h4_df, fvg_zones_all, ob_zones_all = load_htf_structures(symbol, start, end, meta)

    # HTF bias checker (optional)
    htf_bias_at = None
    if USE_HTF_BIAS and BIAS_METHOD.upper() == "EMA":
        htf_bias_at = make_htf_bias_checker(d1_df, h4_df, BIAS_EMA_PERIOD, BIAS_REQUIRE_ALL)

    # For DXY confluence (preload full)
    dxy_meta = None
    if DXY_ENABLED and symbol in ("EURUSD", "GBPUSD") and DXY_DF is not None:
        dxy_meta = get_meta(DXY_SYMBOL)

    trades: List[Trade] = []
    equity_rows: List[dict] = []

    # Determine profile per symbol
    prof = "ASIA" if symbol in PROFILES["ASIA"]["symbols"] else ("LONDON" if symbol in PROFILES["LONDON"]["symbols"] else "ASIA")
    sweep_start_h = PROFILES[prof]["sweep_start"]; sweep_end_h = PROFILES[prof]["sweep_end"]
    exec_start_h  = PROFILES[prof]["exec_start"];  exec_end_h  = PROFILES[prof]["exec_end"]

    # Iterate days
    all_days = df["day"].drop_duplicates().tolist()
    for di, d in enumerate(all_days, start=1):
        if PRINT_EVERY_N_DAYS and di % PRINT_EVERY_N_DAYS == 0:
            print(f"[{symbol}] Day {di}/{len(all_days)} …")

        day_df = df[df["day"] == d].copy()
        if day_df.empty:
            continue
        day_ts = pd.Timestamp(d)
        sweep_s_ts = day_ts.replace(hour=sweep_start_h, minute=0, second=0, microsecond=0)
        sweep_e_ts = day_ts.replace(hour=sweep_end_h,   minute=0, second=0, microsecond=0)
        exec_s_ts  = day_ts.replace(hour=exec_start_h,  minute=0, second=0, microsecond=0)
        exec_e_ts  = day_ts.replace(hour=exec_end_h,    minute=0, second=0, microsecond=0)

        # Session levels for sweep window
        try:
            sess_high, sess_low = get_session_levels(df, day_ts, sweep_start_h, sweep_end_h)
            if not np.isfinite(sess_high) or not np.isfinite(sess_low):
                continue
        except Exception:
            continue

        # DXY day precomp (for EURUSD/GBPUSD)
        dxy_conf = None
        if DXY_ENABLED and symbol in ("EURUSD", "GBPUSD") and DXY_DF is not None and dxy_meta is not None:
            dxy_day = DXY_DF[DXY_DF["time"].dt.floor("D") == day_ts]
            if not dxy_day.empty:
                dxy_conf = precompute_dxy_day_confluence(dxy_day, dxy_meta, sweep_start_h, sweep_end_h)

        # First sweep inside sweep window
        sweep_side, sweep_time = first_sweep_of_range(day_df, sess_high, sess_low, sweep_start_h, sweep_end_h)
        if sweep_side is None or sweep_time is None:
            continue
        bias = "sell" if sweep_side == "high" else "buy"

        # Daily kill switch counter
        daily_losses = 0

        # Search FIRST IFVG disrespect in execution window AFTER sweep window end
        exec_slice = day_df[(day_df["time"] >= sweep_e_ts) & (day_df["time"] >= exec_s_ts) & (day_df["time"] < exec_e_ts)].copy()
        took_trade = False
        for _, bar in exec_slice.iterrows():
            t = pd.to_datetime(bar.time)
            i = int(df.index[df["time"] == t][0])

            det = detect_ifvg_disrespect(df, meta, i)
            if not det:
                continue
            sdir, close_px, (fvg_lo, fvg_hi), gap, body, dist = det
            if sdir != bias:
                continue

            # HTF bias (optional)
            if htf_bias_at is not None:
                _bias = htf_bias_at(t)
                if _bias is None or _bias != sdir:
                    continue

            # POI / prev-level gate
            gate_ok = True
            poi_tag = ""
            if USE_MANIPULATION_POI_STAGE:
                gate_ok = False
                ok_poi, poi_tag, _flags = poi_filter(
                    sdir, float(bar.low), float(bar.high), t,
                    sess_high, sess_low, fvg_zones_all, ob_zones_all, df,
                    session_start_ts=sweep_s_ts, session_end_ts=exec_e_ts, prevs=prev_period_hilo(d1_df, day_ts)
                )
                if ok_poi:
                    gate_ok = True
            if not gate_ok:
                continue

            # DXY gate for EURUSD/GBPUSD
            dxy_okay = True
            if DXY_ENABLED and symbol in ("EURUSD","GBPUSD"):
                dxy_okay = dxy_gate_ok(sdir, t, dxy_conf) if dxy_conf else False
                if not dxy_okay:
                    continue

            # Entry type rule
            use_limit = (body >= 2.0*gap) or (dist > gap*(1.0+0.618))
            entry_type = "limit50" if use_limit else "market"

            # Compute entry/SL/TP
            entry_idx = i
            if entry_type == "market":
                entry_price = round_to_tick(close_px, meta)
                sl_raw = swing_sl(df, i, sdir)
                sl = round_to_tick(sl_raw, meta)
            else:
                entry_level = round_to_tick(fvg_mid(fvg_lo, fvg_hi), meta)
                # find the fill within execution window
                search = exec_slice[exec_slice["time"] >= t]
                filled = None; fill_idx = None
                for _, fbar in search.iterrows():
                    if (fbar.low <= entry_level <= fbar.high):
                        filled = pd.to_datetime(fbar.time)
                        fill_idx = int(df.index[df["time"] == filled][0])
                        break
                if filled is None:
                    continue  # limit not filled
                entry_price = entry_level
                entry_idx = fill_idx
                sl_raw = swing_sl(df, i, sdir)  # SL from disrespect swing
                sl = round_to_tick(sl_raw, meta)

            # validate geometry
            if sdir == "sell" and not (entry_price < sl): continue
            if sdir == "buy"  and not (sl < entry_price): continue

            tp, rT, tp_label = compute_tp_and_R(entry_price, sl, sdir, meta)
            if rT <= 0: continue

            tr = Trade(
                symbol=symbol, direction=sdir, profile=prof, placed_at=df.iloc[entry_idx]["time"],
                entry=entry_price, sl=sl, tp=tp, r_target=rT,
                bias_source="sweep", sweep_side=sweep_side,
                session_sweep_high=sess_high, session_sweep_low=sess_low,
                ifvg_gap=gap, ifvg_body=body, ifvg_distance=dist,
                entry_type=entry_type, poi_tag=poi_tag, dxy_ok=dxy_okay
            )
            tr = simulate_exit_on_future_bars(df, entry_idx, tr)
            trades.append(tr)
            # daily kill switch update
            if tr.outcome == "SL":
                daily_losses += 1
            took_trade = True
            break  # first IFVG only

        if ONE_TRADE_PER_DAY and took_trade:
            continue
        if daily_losses >= DAILY_KILL_LOSSES:
            continue

    # ==== Realize PnL & outputs ====
    rows = []; equity = START_BALANCE
    for tr in trades:
        if tr.outcome == "TP": R = tr.r_target
        elif tr.outcome == "SL": R = -1.0
        else: R = 0.0
        risk_money = equity * (RISK_PER_TRADE / 100.0)
        pnl = R * risk_money; equity += pnl
        rows.append({
            "symbol": tr.symbol, "profile": tr.profile, "direction": tr.direction, "placed_at": tr.placed_at,
            "entry": tr.entry, "sl": tr.sl, "tp": tr.tp, "exit_at": tr.exit_at, "outcome": tr.outcome,
            "R": R, "R_target": tr.r_target, "pnl": pnl, "equity_after": equity, "duration_min": tr.duration_min,
            "bias_source": tr.bias_source, "sweep_side": tr.sweep_side,
            "session_high": tr.session_sweep_high, "session_low": tr.session_sweep_low,
            "ifvg_gap": tr.ifvg_gap, "ifvg_body": tr.ifvg_body, "ifvg_distance": tr.ifvg_distance,
            "entry_type": tr.entry_type, "poi_tag": tr.poi_tag, "dxy_ok": tr.dxy_ok
        })

    df_trades = pd.DataFrame(rows)
    if not df_trades.empty:
        df_trades = df_trades.sort_values(["symbol", "placed_at"]).reset_index(drop=True)
    else:
        df_trades = pd.DataFrame(columns=[
            "symbol","profile","direction","placed_at","entry","sl","tp","exit_at","outcome","R","R_target","pnl",
            "equity_after","duration_min","bias_source","sweep_side","session_high","session_low",
            "ifvg_gap","ifvg_body","ifvg_distance","entry_type","poi_tag","dxy_ok"
        ])

    # Per-symbol summary
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
        eq_series = sym_trs["equity_after"].values
        mdd, _, _ = max_drawdown(eq_series)
        avg_R_all  = float(sym_trs["R"].mean()) if len(sym_trs) else np.nan
        avg_R_wins = float(sym_trs[sym_trs["outcome"] == "TP"]["R"].mean()) if wins > 0 else np.nan

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
            "max_drawdown_money": round(float(mdd), 2),
            "avg_R_all": round(avg_R_all, 3) if not math.isnan(avg_R_all) else np.nan,
            "avg_R_wins": round(avg_R_wins, 3) if not math.isnan(avg_R_wins) else np.nan,
        }

    return df_trades, summary, pd.DataFrame(equity_rows), pd.DataFrame(), df

# ==================== MAIN ===============================

def plot_equity(equity_df: pd.DataFrame, title: str, out_path: str):
    if equity_df.empty:
        return
    plt.figure(figsize=(12,5))
    plt.plot(equity_df["time"], equity_df["equity"])
    plt.title(title)
    plt.xlabel("Time"); plt.ylabel("Equity")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()

def compute_equity_from_trades(trades_df: pd.DataFrame, start_balance: float, risk_pct: float) -> pd.DataFrame:
    equity = start_balance
    rows = []
    if trades_df.empty:
        return pd.DataFrame(columns=["time", "equity"])
    for _, r in trades_df.sort_values("exit_at").iterrows():
        R = r["R"]
        risk_money = equity * (risk_pct / 100.0)
        pnl = R * risk_money
        equity += pnl
        rows.append({"time": r["exit_at"], "equity": equity})
    return pd.DataFrame(rows)

def weekday_name(idx: int) -> str:
    return ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"][idx]

def main():
    if not mt5.initialize():
        raise RuntimeError(f"MT5 init failed: {mt5.last_error()}")
    try:
        os.makedirs(OUT_DIR, exist_ok=True)
        os.makedirs(PLOTS_DIR, exist_ok=True)
        os.makedirs(EQUITY_DIR, exist_ok=True)
        start, end = compute_date_range()
        print(f"Range (server): {start} → {end} | Profiles IFVG backtest (M5), TP={RR_TARGET}R, risk={RISK_PER_TRADE}%")

        # Try resolve/load DXY if needed
        global DXY_SYMBOL, DXY_ENABLED, DXY_DF
        DXY_SYMBOL, DXY_ENABLED = resolve_dxy_symbol()
        if DXY_ENABLED and DXY_SYMBOL:
            DXY_DF = load_bars(DXY_SYMBOL, TIMEFRAME, start, end)
            if DXY_DF is None or DXY_DF.empty:
                print("[WARN] DXY data empty; disabling DXY filter.")
                DXY_ENABLED = False

        all_trades: List[pd.DataFrame] = []
        all_summ: List[dict] = []
        eq_curves: Dict[str, pd.DataFrame] = {}

        # Backtest each symbol
        for sym in SYMBOLS:
            if not mt5.symbol_select(sym, True):
                print(f"[{sym}] not visible in Market Watch; skipping.")
                continue

            df_tr, summ, df_eq, _df_daily, full_df = backtest_symbol(sym, start, end)

            # Save trades CSV per symbol
            if not df_tr.empty:
                path = os.path.join(OUT_DIR, f"{sym}_trades.csv")
                df_tr.to_csv(path, index=False)
                print(f"[{sym}] trades: {len(df_tr)} → {path}")
                all_trades.append(df_tr)

                # Equity curve per symbol
                eq_full = compute_equity_from_trades(df_tr, 100000.0, RISK_PER_TRADE)
                eq_curves[sym] = eq_full.assign(symbol=sym)
                eq_path = os.path.join(EQUITY_DIR, f"{sym}_EQUITY_FULL.png")
                plot_equity(eq_full, f"{sym} Equity (ALL) — start 100k", eq_path)

                # Weekday curves per symbol
                df_tr["weekday"] = pd.to_datetime(df_tr["placed_at"]).dt.weekday
                for wd in range(5):  # Mon..Fri
                    sub = df_tr[df_tr["weekday"] == wd]
                    eq_sub = compute_equity_from_trades(sub, 100000.0, RISK_PER_TRADE)
                    outp = os.path.join(EQUITY_DIR, f"{sym}_EQUITY_{weekday_name(wd).upper()}.png")
                    plot_equity(eq_sub, f"{sym} Equity ({weekday_name(wd)}) — start 100k", outp)

            # Save per-symbol summary
            if summ:
                all_summ.append(summ)

        # Combined trades CSV
        if all_trades:
            merged = pd.concat(all_trades, ignore_index=True).sort_values("placed_at")
            merged_path = os.path.join(OUT_DIR, "ALL_trades.csv")
            merged.to_csv(merged_path, index=False)
            print(f"ALL trades → {merged_path}")

        # Per-symbol summary CSV
        if all_summ:
            df_summ = pd.DataFrame(all_summ)
            summ_path = os.path.join(OUT_DIR, "SUMMARY_PER_SYMBOL.csv")
            df_summ.to_csv(summ_path, index=False)
            print("\nSUMMARY (per symbol):")
            print(df_summ.to_string(index=False))
        else:
            print("No trades generated. Check sessions/data.")

        # Overall portfolio metrics & equity
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
            portfolio_eq = compute_equity_from_trades(all_df, 100000.0, RISK_PER_TRADE)
            mdd, _, _ = max_drawdown(portfolio_eq["equity"].values) if not portfolio_eq.empty else (0.0, -1, -1)
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
            overall_path = os.path.join(OUT_DIR, "SUMMARY_OVERALL.csv")
            pd.DataFrame([overall]).to_csv(overall_path, index=False)

            print("\nOVERALL:")
            print(pd.DataFrame([overall]).to_string(index=False))

            # Combined equity curve
            eq_path = os.path.join(EQUITY_DIR, f"ALL_SYMBOLS_EQUITY.png")
            plot_equity(portfolio_eq, "Combined Equity (ALL symbols) — start 100k", eq_path)

        print(f"\nPNGs → {PLOTS_DIR}")
        print(f"Equity PNGs → {EQUITY_DIR}")

    finally:
        mt5.shutdown()

if __name__ == "__main__":
    main()
