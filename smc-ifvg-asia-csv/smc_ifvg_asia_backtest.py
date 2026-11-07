# smc_ifvg_asia_backtest.py
# ---------------------------------------------------------------
# London (10:00–18:00 server) — Asia sweep ➜ IFVG disrespect
# + CSV MODE (toggle DATA_SOURCE = "csv")
#   - Reads intraday OHLC from CSV(s), builds H4/D1 by resampling.
#   - CSV columns supported:
#       • Datetime, Open, High, Low, Close
#       • OR Date, Time, Open, High, Low, Close
#   - Time is treated as GMT/UTC (tz-naive).
# ---------------------------------------------------------------

import os, math, calendar
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ===== Optional MT5 import (only used when DATA_SOURCE == "mt5") =====
try:
    import MetaTrader5 as mt5
except Exception:
    mt5 = None  # fine in CSV mode

# ====================== USER INPUTS ======================

# MT5 or CSV
DATA_SOURCE = "csv"     # "csv" | "mt5"

# Map symbols -> CSV paths (if not provided, CSV_DEFAULT is used)
CSV_FILES = {
    # "UK100.cash": "UK100_M5.csv",
     "GER40.cash": "DEUIDXEUR_M5_cleaned.csv",
    # "EURUSD":     "EURUSD_M5.csv",
    # "GBPUSD":     "GBPUSD_M5.csv",
}
# Fallback CSV used for any symbol not explicitly mapped above:
CSV_DEFAULT = "DEUIDXEUR_M5_cleaned.csv"   # change this if you like

SYMBOLS = ['GER40.cash']

# MT5 timeframe (ignored in CSV mode, CSV should already be M5 or intraday)
TIMEFRAME = getattr(mt5, "TIMEFRAME_M5", None)

# --- Data Range (choose ONE mode) ---
RANGE_MODE = "months"   # "months" | "days" | "dates"
RANGE_MONTHS = 8
RANGE_DAYS = 120
DATE_FROM = "2025-07-01"
DATE_TO   = "2025-08-12"

# Sessions (server/GMT time)
ASIA_START_H = 1
ASIA_END_H   = 8    # exclusive
LDN_START_H  = 10
LDN_END_H    = 13    # exclusive

# IFVG detection & entry
LOOKBACK = 7                # must be 3 for 3-candle FVG logic here
MIN_GAP_TICKS = 5
ENTRY_ON_RETEST_50 = False
RETTEST_WITHIN_LONDON_ONLY = True

# ================== TAKE-PROFIT MODE =====================
TP_MODE = "ASIA_RANGE"  # "FIXED_RR" | "ASIA_RANGE" | "PREV_SESSION"
RR_TARGET = 2.3

# Risk & accounting
RISK_PER_TRADE = 1.0
START_BALANCE  = 10000.0

ONE_TRADE_PER_DAY = False
OUT_DIR = "bt_out"
PLOTS_DIR = os.path.join(OUT_DIR, "plots")

PRINT_EVERY_N_DAYS = 5
ANNUALIZE_SHARPE = False
VERBOSE = True  # print what the bot is doing (per-day logs)

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
    # UTC-safe (avoids deprecation of datetime.utcnow)
    return datetime.now(timezone.utc).replace(tzinfo=None)

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

def _infer_meta_from_prices(df: pd.DataFrame) -> SymbolMeta:
    # Infer tick size from price diffs (best-effort)
    closes = df["close"].dropna()
    diffs = np.abs(np.diff(closes.values)) if len(closes) > 1 else np.array([0.0001])
    diffs = diffs[diffs > 0]
    tick = float(np.round(np.min(diffs), 10)) if diffs.size else 0.0001
    # Get decimals from tick, default 5
    digits = max(0, min(10, str(tick)[::-1].find('.')))
    return SymbolMeta(
        digits=digits or 5,
        point=10**(-digits or 5),
        tick_size=tick if tick > 0 else 0.0001,
        tick_value=1.0,
        vol_min=0.01, vol_max=100.0, vol_step=0.01
    )

def get_meta(symbol: str, df_hint: Optional[pd.DataFrame]=None) -> Optional[SymbolMeta]:
    if DATA_SOURCE == "mt5" and mt5 is not None:
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
    # CSV: infer from data
    return _infer_meta_from_prices(df_hint) if df_hint is not None else SymbolMeta(
        digits=5, point=0.00001, tick_size=0.0001, tick_value=1.0, vol_min=0.01, vol_max=100.0, vol_step=0.01
    )

def round_to_tick(price: float, meta: SymbolMeta) -> float:
    if meta.tick_size <= 0:
        return round(price, meta.digits)
    steps = round(price / meta.tick_size)
    return round(steps * meta.tick_size, meta.digits)

# ==================== DATA LOADERS =======================

def _normalize_ohlc_cols(df: pd.DataFrame) -> pd.DataFrame:
    # Standardize to lowercase: open,high,low,close
    ren = {}
    for need in ["open","high","low","close"]:
        if need not in df.columns:
            for c in df.columns:
                if c.lower() == need:
                    ren[c] = need
                    break
    if ren:
        df = df.rename(columns=ren)
    return df[["open","high","low","close"]]

def load_bars_csv(csv_path: str, start: datetime, end: datetime) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    cols = {c.lower(): c for c in df.columns}
    if "datetime" in cols or "timestamp" in cols:
        dtcol = cols.get("datetime", cols.get("timestamp"))
        df["time"] = pd.to_datetime(df[dtcol])
    else:
        d = cols.get("date"); t = cols.get("time")
        if d is None or t is None:
            raise ValueError(f"CSV must have Datetime or Date+Time: {csv_path}")
        df["time"] = pd.to_datetime(df[d] + " " + df[t])
    df = df.sort_values("time").reset_index(drop=True)
    df = df[(df["time"] >= start) & (df["time"] <= end)].copy()
    df = pd.concat([df[["time"]], _normalize_ohlc_cols(df)], axis=1)
    # ensure numeric
    for c in ["open","high","low","close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna()

def resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    g = df.set_index("time").sort_index()
    o = g["open"].resample(rule).first()
    h = g["high"].resample(rule).max()
    l = g["low"].resample(rule).min()
    c = g["close"].resample(rule).last()
    out = pd.concat({"open":o,"high":h,"low":l,"close":c}, axis=1).dropna().reset_index()
    return out

def load_bars_mt5(symbol: str, timeframe, start: datetime, end: datetime) -> pd.DataFrame:
    rates = mt5.copy_rates_range(symbol, timeframe, start, end)
    df = pd.DataFrame(rates) if rates is not None else pd.DataFrame()
    if df.empty:
        return df
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df.sort_values("time").reset_index(drop=True)
    df = df.rename(columns={"open":"open","high":"high","low":"low","close":"close"})
    return df[["time","open","high","low","close"]]

def load_bars(symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
    if DATA_SOURCE == "mt5":
        return load_bars_mt5(symbol, TIMEFRAME, start, end)
    # CSV
    path = CSV_FILES.get(symbol, CSV_DEFAULT)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"CSV for {symbol} not found: {path}")
    return load_bars_csv(path, start, end)

# ==================== ASIA / LONDON HELPERS ===============

def get_day_asia_levels(df: pd.DataFrame, day_ts: pd.Timestamp) -> Tuple[float, float]:
    asia_s = day_ts.replace(hour=ASIA_START_H, minute=0, second=0, microsecond=0)
    asia_e = day_ts.replace(hour=ASIA_END_H,   minute=0, second=0, microsecond=0)
    asia_slice = df[(df["time"] >= asia_s) & (df["time"] < asia_e)]
    return float(asia_slice["high"].max()), float(asia_slice["low"].min())

def asia_sweep_side(day_df: pd.DataFrame, asia_high: float, asia_low: float) -> Tuple[Optional[str], Optional[pd.Timestamp]]:
    ldn = day_df[(day_df["time"].dt.hour >= LDN_START_H) & (day_df["time"].dt.hour < LDN_END_H)]
    for _, bar in ldn.iterrows():
        if bar.high > asia_high:
            return "high", pd.Timestamp(bar.time)
        if bar.low  < asia_low:
            return "low",  pd.Timestamp(bar.time)
    return None, None

# ==================== PREV D/W/M LEVELS ===================

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

def prev_period_hilo(d1: pd.DataFrame, day_ts: pd.Timestamp) -> Dict[str, Optional[float]]:
    def hl(slice_df):
        if slice_df.empty:
            return None, None
        return float(slice_df["high"].max()), float(slice_df["low"].min())

    pd_s, _ = day_bounds(day_ts - pd.Timedelta(days=1))
    pw_s, pw_e = week_bounds(day_ts - pd.Timedelta(days=7))
    pm_ref = (day_ts - pd.DateOffset(months=1))
    pm_s = pm_ref.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    pm_e = day_ts.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

    pday  = d1[(d1["time"] >= pd_s) & (d1["time"] < pd_s + pd.Timedelta(days=1))]
    pweek = d1[(d1["time"] >= pw_s) & (d1["time"] < pw_e)]
    pmonth= d1[(d1["time"] >= pm_s) & (d1["time"] < pm_e)]

    pdh, pdl = hl(pday)
    pwh, pwl = hl(pweek)
    pmh, pml = hl(pmonth)
    return {"PDH": pdh, "PDL": pdl, "PWH": pwh, "PWL": pwl, "PMH": pmh, "PML": pml}

# ==================== HTF STRUCTURES ======================

@dataclass
class Zone:
    tf: str               # "D1" or "H4"
    kind: str             # "bullish_fvg" | "bearish_fvg" | "bullish_ob" | "bearish_ob"
    formed_at: pd.Timestamp
    lo: float
    hi: float

def is_valid_gap(gap_low: float, gap_high: float, meta: SymbolMeta) -> bool:
    return (float(gap_high) - float(gap_low)) >= max(meta.tick_size, 1e-12) * max(2, MIN_GAP_TICKS)

def detect_fvgs(htf: pd.DataFrame, meta: SymbolMeta, tf_name: str) -> List[Zone]:
    zones: List[Zone] = []
    for i in range(3, len(htf)):
        c1 = htf.iloc[i-3]
        c3 = htf.iloc[i-1]
        c4 = htf.iloc[i]
        if (c1.high < c3.low) and is_valid_gap(c1.high, c3.low, meta):
            zones.append(Zone(tf_name, "bullish_fvg", pd.Timestamp(c4.time), float(c1.high), float(c3.low)))
        if (c1.low > c3.high) and is_valid_gap(c3.high, c1.low, meta):
            zones.append(Zone(tf_name, "bearish_fvg", pd.Timestamp(c4.time), float(c3.high), float(c1.low)))
    return zones

def detect_order_blocks(htf: pd.DataFrame, tf_name: str) -> List[Zone]:
    zones: List[Zone] = []
    for i in range(2, len(htf)):
        c0 = htf.iloc[i-2]
        c1 = htf.iloc[i-1]
        c2 = htf.iloc[i]
        if c1.open < c1.close and (c2.low < min(c0.low, c1.low)):
            lo = float(min(c1.open, c1.close))
            hi = float(max(c1.open, c1.close))
            zones.append(Zone(tf_name, "bearish_ob", pd.Timestamp(c2.time), lo, hi))
        if c1.open > c1.close and (c2.high > max(c0.high, c1.high)):
            lo = float(min(c1.open, c1.close))
            hi = float(max(c1.open, c1.close))
            zones.append(Zone(tf_name, "bullish_ob", pd.Timestamp(c2.time), lo, hi))
    return zones

def load_htf_structures(symbol: str, start: datetime, end: datetime, meta: SymbolMeta, intraday_df: pd.DataFrame):
    if DATA_SOURCE == "mt5":
        # Load with padding for context
        pad_days = 120
        s = start - timedelta(days=pad_days)
        d1 = load_bars_mt5(symbol, mt5.TIMEFRAME_D1, s, end + timedelta(days=2))
        h4 = load_bars_mt5(symbol, mt5.TIMEFRAME_H4, s, end + timedelta(days=2))
        d1 = d1.sort_values("time").reset_index(drop=True)
        h4 = h4.sort_values("time").reset_index(drop=True)
    else:
        # CSV: resample intraday (lowercase rules to avoid FutureWarnings)
        g = intraday_df.copy()
        d1 = resample_ohlc(g, "1d")
        h4 = resample_ohlc(g, "4h")

    fvg_d1 = detect_fvgs(d1, meta, "D1")
    fvg_h4 = detect_fvgs(h4, meta, "H4")
    ob_d1  = detect_order_blocks(d1, "D1")
    ob_h4  = detect_order_blocks(h4, "H4")
    return d1, h4, fvg_d1 + fvg_h4, ob_d1 + ob_h4

# ============== POI CONDITION (MANDATORY FILTER) ==========

def bar_intersects_zone(bar_low: float, bar_high: float, z: Zone) -> bool:
    return not (bar_high < z.lo or bar_low > z.hi)

def zone_outside_asia(z: Zone, asia_high: float, asia_low: float) -> bool:
    return (z.hi <= asia_low) or (z.lo >= asia_high)

def poi_condition_met(direction: str,
                      bar_low: float, bar_high: float, bar_time: pd.Timestamp,
                      asia_high: float, asia_low: float,
                      prev_levels: Dict[str, Optional[float]],
                      fvg_zones: List[Zone], ob_zones: List[Zone]) -> Tuple[bool, str]:
    tag = ""
    if direction == "sell":
        for k in ["PDH", "PWH", "PMH"]:
            lvl = prev_levels.get(k)
            if lvl is not None and lvl > asia_high and bar_high > lvl:
                return True, f"sweep:{k}"
        for z in ob_zones + fvg_zones:
            if z.formed_at > bar_time: continue
            if z.kind in ("bearish_ob", "bearish_fvg") and zone_outside_asia(z, asia_high, asia_low) and bar_intersects_zone(bar_low, bar_high, z):
                return True, f"tap:{z.tf}:{z.kind}"
        return False, tag
    else:
        for k in ["PDL", "PWL", "PML"]:
            lvl = prev_levels.get(k)
            if lvl is not None and lvl < asia_low and bar_low < lvl:
                return True, f"sweep:{k}"
        for z in ob_zones + fvg_zones:
            if z.formed_at > bar_time: continue
            if z.kind in ("bullish_ob", "bullish_fvg") and zone_outside_asia(z, asia_high, asia_low) and bar_intersects_zone(bar_low, bar_high, z):
                return True, f"tap:{z.tf}:{z.kind}"
        return False, tag

# ==================== IFVG (3-candle) & DISRESPECT =======

def is_valid_gap(gap_low: float, gap_high: float, meta: SymbolMeta) -> bool:
    return (float(gap_high) - float(gap_low)) >= max(meta.tick_size, 1e-12) * max(2, MIN_GAP_TICKS)

def detect_ifvg_disrespect(df: pd.DataFrame, meta: SymbolMeta, i_now: int) -> Optional[Tuple[str, float, Tuple[float, float]]]:
    """
    Detects the IFVG disrespect at position i_now using a 3-candle pattern:
      - Bullish FVG: c1.high < c3.low, disrespected when c4.close < gap_low  -> SELL
      - Bearish FVG: c1.low  > c3.high, disrespected when c4.close > gap_high -> BUY

    Returns:
      ("sell" or "buy", disrespect_close, (fvg_lo, fvg_hi))
    """
    # Need at least 4 bars up to i_now (c1, c2, c3, c4)
    if i_now < 3 or len(df) < 4:
        return None

    # Local 4-bar window to avoid positional OOB
    w = df.iloc[i_now-3:i_now+1]
    if len(w) < 4:
        return None

    c1 = w.iloc[0]
    c3 = w.iloc[2]
    c4 = w.iloc[3]

    # Prior Bullish FVG (gap up): disrespect if c4.close < gap_low
    if (c1.high < c3.low) and is_valid_gap(c1.high, c3.low, meta):
        gap_low, gap_high = float(c1.high), float(c3.low)
        if float(c4.close) < gap_low:
            lo = min(float(c4.open), float(c4.close))
            hi = max(float(c4.open), float(c4.close))
            if not is_valid_gap(lo, hi, meta):
                eps = meta.tick_size * max(2, MIN_GAP_TICKS)
                lo, hi = float(c4.close) - eps, float(c4.close) + eps
            return ("sell", float(c4.close), (lo, hi))

    # Prior Bearish FVG (gap down): disrespect if c4.close > gap_high
    if (c1.low > c3.high) and is_valid_gap(c3.high, c1.low, meta):
        gap_low, gap_high = float(c3.high), float(c1.low)
        if float(c4.close) > gap_high:
            lo = min(float(c4.open), float(c4.close))
            hi = max(float(c4.open), float(c4.close))
            if not is_valid_gap(lo, hi, meta):
                eps = meta.tick_size * max(2, MIN_GAP_TICKS)
                lo, hi = float(c4.close) - eps, float(c4.close) + eps
            return ("buy", float(c4.close), (lo, hi))

    return None

def fvg_mid(fvg_lo: float, fvg_hi: float) -> float:
    return 0.5 * (float(fvg_lo) + float(fvg_hi))

# ==================== TP / R computation =================

def compute_tp_and_R(entry: float, sl: float, direction: str,
                     meta: SymbolMeta,
                     asia_high: Optional[float], asia_low: Optional[float],
                     prev_high: Optional[float], prev_low: Optional[float]) -> Tuple[float, float, str]:
    risk = abs(entry - sl)
    if risk <= 0:
        return entry, 0.0, "invalid"
    mode = (TP_MODE or "ASIA_RANGE").upper()
    label = mode
    tp = None
    if mode == "FIXED_RR":
        tp = entry + RR_TARGET * risk if direction == "buy" else entry - RR_TARGET * risk
        label = f"{RR_TARGET:.2f}R"
    elif mode == "ASIA_RANGE":
        if direction == "buy" and asia_high is not None:
            tp = asia_high; label = "Asia High"
        elif direction == "sell" and asia_low is not None:
            tp = asia_low;  label = "Asia Low"
    elif mode == "PREV_SESSION":
        if direction == "buy" and prev_high is not None:
            tp = prev_high; label = "Prev High"
        elif direction == "sell" and prev_low is not None:
            tp = prev_low;  label = "Prev Low"
    if tp is None or (direction == "buy" and tp <= entry) or (direction == "sell" and tp >= entry):
        tp = entry + RR_TARGET * risk if direction == "buy" else entry - RR_TARGET * risk
        label = f"{RR_TARGET:.2f}R*"
    tp = round_to_tick(float(tp), meta)
    r_target = abs(tp - entry) / risk
    return tp, r_target, label

# ==================== TRADE SIMULATION ====================

@dataclass
class Trade:
    symbol: str
    direction: str
    placed_at: pd.Timestamp
    entry: float
    sl: float
    tp: float
    r_target: float
    exit_at: Optional[pd.Timestamp] = None
    outcome: Optional[str] = None
    duration_min: Optional[float] = None
    notes: Optional[str] = None
    asia_path: Optional[str] = None
    asia_high: Optional[float] = None
    asia_low: Optional[float] = None
    disrespect_time: Optional[pd.Timestamp] = None
    retest_zone: Optional[Tuple[float,float]] = None
    prev_high: Optional[float] = None
    prev_low: Optional[float] = None
    tp_label: Optional[str] = None
    poi_tag: Optional[str] = None

def bar_hit(low, high, level) -> bool:
    return (low <= level <= high)

def simulate_exit_on_future_bars(df: pd.DataFrame, start_idx: int, tr: Trade) -> Trade:
    for j in range(start_idx + 1, len(df)):
        bar = df.iloc[j]
        lo, hi = bar.low, bar.high
        sl_hit = bar_hit(lo, hi, tr.sl)
        tp_hit = bar_hit(lo, hi, tr.tp)
        if sl_hit and tp_hit:
            tr.exit_at, tr.outcome = bar.time, "SL"; break
        if tp_hit:
            tr.exit_at, tr.outcome = bar.time, "TP"; break
        if sl_hit:
            tr.exit_at, tr.outcome = bar.time, "SL"; break
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
            streak += 1; best = max(best, streak)
        elif o in ("TP", "SL"):
            streak = 0
    return best

def profit_factor(pnls: np.ndarray) -> float:
    gp = pnls[pnls > 0].sum()
    gl = -pnls[pnls < 0].sum()
    if gl > 0: return float(gp / gl)
    return math.inf if gp > 0 else 0.0

def sharpe_ratio(R: np.ndarray, eps: float = 1e-9, annualize: bool = False) -> float:
    if R.size == 0: return 0.0
    mu = float(R.mean())
    sd = float(R.std(ddof=1)) if R.size > 1 else 0.0
    sr = mu / (sd + eps)
    if annualize: sr *= math.sqrt(252)
    return sr

def max_drawdown(equity: np.ndarray) -> Tuple[float, int, int]:
    if equity.size == 0: return 0.0, -1, -1
    peaks = np.maximum.accumulate(equity)
    dd = peaks - equity
    i = int(np.argmax(dd))
    peak_idx = int(np.argmax(equity[:i+1])) if i >= 0 else -1
    return float(dd[i]), peak_idx, i

# ==================== PLOTTING (TV-ish) ==================

_BG="#0f1115"; _GRID="#23262d"; _WICK="#b7bcc5"
_BODY_UP="#e6e8eb"; _BODY_DN="#aeb3b8"; _TXT="#cfd3da"
_ASIA="#7fd8ff"; _LDNBOX="#133b4b"; _IFVG="#c24b4b"
_SLBOX="#ff7a7a"; _TPBOX="#76e3a1"

plt.rcParams["axes.facecolor"]=_BG; plt.rcParams["figure.facecolor"]=_BG; plt.rcParams["savefig.facecolor"]=_BG
plt.rcParams["axes.edgecolor"]=_GRID; plt.rcParams["xtick.color"]=_TXT; plt.rcParams["ytick.color"]=_TXT
plt.rcParams["text.color"]=_TXT; plt.rcParams["axes.labelcolor"]=_TXT

def _tv_candles(ax, df_window: pd.DataFrame):
    xs = np.arange(len(df_window))
    for i, r in enumerate(df_window.itertuples(index=False)):
        o, h, l, c = r.open, r.high, r.low, r.close
        ax.vlines(xs[i], l, h, linewidth=1, color=_WICK)
        lower = min(o, c); height = max(abs(c - o), 1e-12)
        face = _BODY_UP if c >= o else _BODY_DN
        rect = Rectangle((xs[i]-0.35, lower), 0.7, height, linewidth=0, facecolor=face, alpha=1.0)
        ax.add_patch(rect)
    return xs

def _x_at_time(view_times: pd.Series, xs: np.ndarray, ts: pd.Timestamp) -> int:
    j = view_times.searchsorted(pd.to_datetime(ts))
    j = max(0, min(j, len(view_times)-1))
    return int(xs[j])

def _segment_hline(ax, y, x0, x1, label=None, color=_ASIA, lw=1.0):
    ax.hlines(y, x0, x1, colors=color, linewidth=lw)
    if label:
        ax.text((x0+x1)/2, y, f" {label} ", va="bottom", ha="center", color=color, alpha=0.9)

def plot_trade_tv(symbol: str, full_df: pd.DataFrame, trade_row: pd.Series, out_path: str):
    placed = pd.to_datetime(trade_row["placed_at"])
    exit_at = pd.to_datetime(trade_row["exit_at"]) if pd.notnull(trade_row["exit_at"]) else full_df["time"].iloc[-1]
    day = placed.floor("D")

    asia_s = day.replace(hour=ASIA_START_H, minute=0, second=0, microsecond=0)
    asia_e = day.replace(hour=ASIA_END_H,   minute=0, second=0, microsecond=0)
    start_time = min(asia_s, placed) - timedelta(minutes=20)
    end_time   = max(exit_at, asia_e) + timedelta(minutes=20)

    view = full_df[(full_df["time"] >= start_time) & (full_df["time"] <= end_time)].reset_index(drop=True)
    if view.empty:
        idx = full_df["time"].searchsorted(placed)
        i0 = max(0, idx - 120); i1 = min(len(full_df) - 1, idx + 220)
        view = full_df.iloc[i0:i1+1].reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(14, 7.5))
    xs = _tv_candles(ax, view)

    step = max(1, len(xs) // 10)
    ax.set_xticks(xs[::step])
    ax.set_xticklabels(view["time"].dt.strftime("%H:%M")[::step], rotation=0, ha="center")
    ax.grid(True, linewidth=0.6, color=_GRID, alpha=0.6)

    ldn_s = day.replace(hour=LDN_START_H, minute=0, second=0, microsecond=0)
    ldn_e = day.replace(hour=LDN_END_H,   minute=0, second=0, microsecond=0)
    x_ldn0 = _x_at_time(view["time"], xs, ldn_s)
    x_ldn1 = _x_at_time(view["time"], xs, ldn_e)
    ax.axvspan(x_ldn0, x_ldn1, color=_LDNBOX, alpha=0.25)

    asia_high, asia_low = trade_row["asia_high"], trade_row["asia_low"]
    x_asia0 = _x_at_time(view["time"], xs, asia_s)
    x_asia1 = _x_at_time(view["time"], xs, asia_e)
    if pd.notnull(asia_high) and pd.notnull(asia_low):
        _segment_hline(ax, asia_high, x_asia0, x_asia1, "Asian High")
        _segment_hline(ax, asia_low,  x_asia0, x_asia1, "Asian Low")

    if pd.notnull(trade_row.get("disrespect_time", None)) and pd.notnull(trade_row.get("retest_lo", None)):
        t_dis = pd.to_datetime(trade_row["disrespect_time"])
        i_dis = full_df["time"].searchsorted(t_dis)
        start_idx = max(0, i_dis - 4)
        end_idx   = i_dis
        v0 = view["time"].searchsorted(full_df["time"].iloc[start_idx])
        v1 = view["time"].searchsorted(full_df["time"].iloc[end_idx])
        lo = float(trade_row["retest_lo"]); hi = float(trade_row["retest_hi"])
        rect = Rectangle((v0, lo), max(1, v1 - v0), hi - lo, linewidth=1, edgecolor=_IFVG, facecolor=_IFVG, alpha=0.12)
        ax.add_patch(rect)
        ax.text(v0+1, hi, "IFVG (5 bars)", color=_IFVG, alpha=0.9, va="bottom")

    x_entry = _x_at_time(view["time"], xs, placed); x_exit  = _x_at_time(view["time"], xs, exit_at)
    y_entry = float(trade_row["entry"]); y_sl = float(trade_row["sl"]); y_tp = float(trade_row["tp"])

    sl_lo, sl_hi = (min(y_entry, y_sl), max(y_entry, y_sl))
    ax.add_patch(Rectangle((x_entry, sl_lo), max(1, x_exit - x_entry), sl_hi - sl_lo,
                           linewidth=1, edgecolor=_SLBOX, facecolor=_SLBOX, alpha=0.16))
    tp_lo, tp_hi = (min(y_entry, y_tp), max(y_entry, y_tp))
    ax.add_patch(Rectangle((x_entry, tp_lo), max(1, x_exit - x_entry), tp_hi - tp_lo,
                           linewidth=1, edgecolor=_TPBOX, facecolor=_TPBOX, alpha=0.16))

    ax.axvline(x_entry, linestyle=":", linewidth=1, color="#c7cbd3")
    ax.text(x_entry, ax.get_ylim()[1], " placed", va="top", rotation=90, alpha=0.9)
    if pd.notnull(trade_row["exit_at"]):
        ax.axvline(x_exit, linestyle=":", linewidth=1, color="#c7cbd3")
        ax.text(x_exit, ax.get_ylim()[1], f" exit ({trade_row['outcome']})", va="top", rotation=90, alpha=0.9)

    ttl = (f"{symbol} — {trade_row['direction'].upper()} | {placed:%Y-%m-%d %H:%M} | "
           f"Asia {trade_row['asia_path']} sweep → {'close-entry' if not ENTRY_ON_RETEST_50 else '50% retest'} "
           f"({trade_row.get('tp_label','TP')}) • POI:{trade_row.get('poi_tag','-')}")
    ax.set_title(ttl, fontsize=12, color=_TXT)
    ax.set_ylabel("Price")
    ax.margins(x=0.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

# ==================== BACKTEST CORE =======================

def backtest_symbol(symbol: str, start: datetime, end: datetime) -> Tuple[pd.DataFrame, dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print(f"\n=== Backtesting {symbol} — Asia sweep ➜ IFVG disrespect → POI filter (D1/H4 + PD/W/M) → TP:{TP_MODE} ===")

    # Load intraday (CSV or MT5)
    df = load_bars(symbol, start, end)
    if df is None or df.empty or len(df) < 500:
        print(f"[{symbol}] Not enough data.")
        return pd.DataFrame(), {}, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # Infer meta in CSV mode or fetch MT5 meta
    meta = get_meta(symbol, df if DATA_SOURCE == "csv" else None)
    if meta is None:
        print(f"[{symbol}] No meta; skipping.")
        return pd.DataFrame(), {}, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    df["day"] = df["time"].dt.floor("D")

    # HTF data + structures from the same data source
    d1_df, h4_df, fvg_zones_all, ob_zones_all = load_htf_structures(symbol, start, end, meta, df)

    trades: List[Trade] = []
    daily_logs: List[dict] = []

    all_days = df["day"].drop_duplicates().tolist()
    for di, d in enumerate(all_days, start=1):
        day_df = df[df["day"] == d].copy()
        if day_df.empty:
            continue
        day_ts = pd.Timestamp(d)

        # Asia range
        try:
            asia_high, asia_low = get_day_asia_levels(df, day_ts)
            if not np.isfinite(asia_high) or not np.isfinite(asia_low):
                daily_logs.append({"symbol": symbol, "day": day_ts, "note": "asia_nan"})
                continue
        except Exception:
            daily_logs.append({"symbol": symbol, "day": day_ts, "note": "asia_err"})
            continue

        if VERBOSE:
            print(f"\n=== {symbol} {day_ts.date()} ===  Asia High={asia_high:.2f}  Asia Low={asia_low:.2f}")

        # Previous period levels from D1 series (resampled CSV or MT5)
        prevs = prev_period_hilo(d1_df, day_ts)
        prev_high = max([x for x in [prevs.get("PDH"), prevs.get("PWH"), prevs.get("PMH")] if x is not None], default=None)
        prev_low  = min([x for x in [prevs.get("PDL"), prevs.get("PWL"), prevs.get("PML")] if x is not None], default=None)

        # First Asia sweep inside London
        asia_path, sweep_time = asia_sweep_side(day_df, asia_high, asia_low)
        if asia_path is None or sweep_time is None:
            if VERBOSE: print("No sweep found in London window → skip.")
            daily_logs.append({"symbol": symbol, "day": day_ts, "asia_high": asia_high, "asia_low": asia_low,
                               "note": "no_sweep"})
            continue

        want_dir = "sell" if asia_path == "high" else "buy"
        if VERBOSE: print(f"Sweep: {asia_path.upper()} at {sweep_time} → Bias = {want_dir.upper()}")

        # Iterate London bars AFTER the sweep, looking for first disrespect + POI condition
        ldn_slice = day_df[(day_df["time"] >= sweep_time) &
                           (day_df["time"].dt.hour >= LDN_START_H) &
                           (day_df["time"].dt.hour <  LDN_END_H)]
        took_trade = False

        for _, bar in ldn_slice.iterrows():
            t = pd.to_datetime(bar.time)
            # robust index lookup + guard for short history
            idxs = np.flatnonzero(df["time"].values == np.datetime64(t))
            if idxs.size == 0:
                continue
            i = int(idxs[0])
            if i < 3:
                continue

            det = detect_ifvg_disrespect(df, meta, i)
            if not det:
                continue
            sdir, close_px, (fvg_lo, fvg_hi) = det
            if sdir != want_dir:
                continue

            # Side constraint at DISRESPECT bar
            if sdir == "sell" and (bar.low <= asia_low):
                continue
            if sdir == "buy"  and (bar.high >= asia_high):
                continue

            # POI FILTER (outside Asia)
            ok, poi_tag = poi_condition_met(
                sdir, float(bar.low), float(bar.high), t,
                asia_high, asia_low, prevs,
                fvg_zones_all, ob_zones_all
            )
            if not ok:
                continue

            if VERBOSE:
                print(f"IFVG {sdir.UPPER() if hasattr(str,'UPPER') else sdir.upper()} disrespect @ {t}  "
                      f"FVG[{min(fvg_lo,fvg_hi):.2f},{max(fvg_lo,fvg_hi):.2f}]  POI={poi_tag}")

            # === ENTRY LOGIC ===
            if not ENTRY_ON_RETEST_50:
                entry_time = t
                entry_idx  = i
                entry = round_to_tick(close_px, meta)

                start_i = max(0, i - 3)
                window = df.iloc[start_i:i+1]
                sl_raw = float(window["low"].min()) if sdir == "buy" else float(window["high"].max())
                sl = round_to_tick(sl_raw, meta)

                if sdir == "sell":
                    if not (entry < sl): continue
                    if entry <= asia_low: continue
                else:
                    if not (sl < entry): continue
                    if entry >= asia_high: continue

                tp, rT, tp_label = compute_tp_and_R(entry, sl, sdir, meta, asia_high, asia_low, prev_high, prev_low)
                if rT <= 0: continue

                tr = Trade(
                    symbol=symbol, direction=sdir, placed_at=entry_time,
                    entry=entry, sl=sl, tp=tp, r_target=rT,
                    asia_path=asia_path, asia_high=asia_high, asia_low=asia_low,
                    disrespect_time=t, retest_zone=(min(fvg_lo, fvg_hi), max(fvg_lo, fvg_hi)),
                    prev_high=prev_high, prev_low=prev_low, tp_label=tp_label,
                    poi_tag=poi_tag,
                    notes=f"Asia {asia_path} sweep → {sdir.upper()} on DISRESPECT CLOSE ({tp_label}) • {poi_tag}"
                )
                tr = simulate_exit_on_future_bars(df, entry_idx, tr)
                if VERBOSE:
                    print(f"ENTRY {sdir.upper()} @ {entry:.2f}  SL={sl:.2f}  TP={tp:.2f}  → {tr.outcome}")
                trades.append(tr); took_trade = True; break

            else:
                entry_level = round_to_tick(0.5*(min(fvg_lo,fvg_hi)+max(fvg_lo,fvg_hi)), meta)
                start_i = max(0, i - 3)
                window = df.iloc[start_i:i+1]
                sl_raw = float(window["low"].min()) if sdir == "buy" else float(window["high"].max())
                sl = round_to_tick(sl_raw, meta)

                if sdir == "sell" and not (entry_level < sl): continue
                if sdir == "buy"  and not (sl < entry_level): continue

                search = day_df[(day_df["time"] >= t)]
                if RETTEST_WITHIN_LONDON_ONLY:
                    search = search[(search["time"].dt.hour >= LDN_START_H) & (search["time"].dt.hour < LDN_END_H)]
                filled = None; fill_idx = None
                for _, fbar in search.iterrows():
                    if (fbar.low <= entry_level <= fbar.high):
                        if sdir == "sell" and (fbar.low <= asia_low): break
                        if sdir == "buy"  and (fbar.high >= asia_high): break
                        filled = pd.to_datetime(fbar.time)
                        fill_idx = int(df.index[df["time"] == filled][0])
                        break
                if filled is None:
                    continue

                tp, rT, tp_label = compute_tp_and_R(entry_level, sl, sdir, meta, asia_high, asia_low, prev_high, prev_low)
                if rT <= 0: continue

                tr = Trade(
                    symbol=symbol, direction=sdir, placed_at=filled,
                    entry=entry_level, sl=sl, tp=tp, r_target=rT,
                    asia_path=asia_path, asia_high=asia_high, asia_low=asia_low,
                    disrespect_time=t, retest_zone=(min(fvg_lo, fvg_hi), max(fvg_lo, fvg_hi)),
                    prev_high=prev_high, prev_low=prev_low, tp_label=tp_label,
                    poi_tag=poi_tag,
                    notes=f"Asia {asia_path} sweep → {sdir.upper()} on 50% RETEST ({tp_label}) • {poi_tag}"
                )
                tr = simulate_exit_on_future_bars(df, fill_idx, tr)
                if VERBOSE:
                    print(f"ENTRY {sdir.upper()} @ {entry_level:.2f}  SL={sl:.2f}  TP={tp:.2f}  → {tr.outcome}")
                trades.append(tr); took_trade = True; break

        daily_logs.append({"symbol": symbol, "day": day_ts, "asia_high": asia_high, "asia_low": asia_low,
                           "asia_path": asia_path, "asia_sweep_time": sweep_time, "took_trade": took_trade})

        if ONE_TRADE_PER_DAY and took_trade:
            continue

    # ==== Realize PnL & outputs ====
    rows = []; equity = START_BALANCE; equity_rows = []
    for tr in trades:
        if tr.outcome == "TP": R = tr.r_target
        elif tr.outcome == "SL": R = -1.0
        else: R = 0.0
        risk_money = equity * (RISK_PER_TRADE / 100.0)
        pnl = R * risk_money; equity += pnl

        rows.append({
            "symbol": tr.symbol, "direction": tr.direction, "placed_at": tr.placed_at,
            "entry": tr.entry, "sl": tr.sl, "tp": tr.tp, "exit_at": tr.exit_at, "outcome": tr.outcome,
            "R": R, "R_target": tr.r_target, "pnl": pnl, "equity_after": equity, "duration_min": tr.duration_min,
            "notes": tr.notes, "poi_tag": tr.poi_tag,
            "asia_path": tr.asia_path, "asia_high": tr.asia_high, "asia_low": tr.asia_low,
            "disrespect_time": tr.disrespect_time, "retest_lo": tr.retest_zone[0] if tr.retest_zone else np.nan,
            "retest_hi": tr.retest_zone[1] if tr.retest_zone else np.nan,
            "prev_high": tr.prev_high, "prev_low": tr.prev_low, "tp_label": tr.tp_label
        })
        equity_rows.append({"time": tr.exit_at if tr.exit_at is not None else tr.placed_at, "equity": equity})

    df_trades = pd.DataFrame(rows)
    if not df_trades.empty:
        df_trades = df_trades.sort_values(["symbol", "placed_at"]).reset_index(drop=True)
    else:
        df_trades = pd.DataFrame(columns=[
            "symbol","direction","placed_at","entry","sl","tp","exit_at","outcome","R","R_target","pnl",
            "equity_after","duration_min","notes","poi_tag","asia_path","asia_high","asia_low",
            "disrespect_time","retest_lo","retest_hi","prev_high","prev_low","tp_label"
        ])

    df_equity = pd.DataFrame(equity_rows)
    if not df_equity.empty:
        df_equity = df_equity.sort_values("time").reset_index(drop=True)
    else:
        df_equity = pd.DataFrame(columns=["time","equity"])

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

    return df_trades, summary, df_equity, pd.DataFrame(), df

# ==================== MAIN ===============================

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    start, end = compute_date_range()
    print(f"Range: {start} → {end} | Asia sweep → IFVG disrespect → POI filter → TP:{TP_MODE} | Source={DATA_SOURCE.upper()}")

    if DATA_SOURCE == "mt5":
        if mt5 is None:
            raise RuntimeError("MetaTrader5 package not available. Switch DATA_SOURCE to 'csv' or install MT5.")
        if not mt5.initialize():
            raise RuntimeError(f"MT5 init failed: {mt5.last_error()}")

    try:
        all_trades: List[pd.DataFrame] = []
        all_summ: List[dict] = []
        eq_curves: Dict[str, pd.DataFrame] = {}

        for sym in SYMBOLS:
            if DATA_SOURCE == "mt5":
                if not mt5.symbol_select(sym, True):
                    print(f"[{sym}] not visible in Market Watch; skipping.")
                    continue

            df_tr, summ, df_eq, _df_daily, full_df = backtest_symbol(sym, start, end)

            if not df_tr.empty:
                path = os.path.join(OUT_DIR, f"{sym}_trades.csv")
                df_tr.to_csv(path, index=False)
                print(f"[{sym}] trades: {len(df_tr)} → {path}")
                all_trades.append(df_tr)

                # Earliest 10 plots
                for i, row in df_tr.sort_values("placed_at").tail(10).reset_index(drop=True).iterrows():
                    fname = f"{sym}_EARLY{i+1:02d}_{pd.to_datetime(row['placed_at']).strftime('%Y%m%d_%H%M')}.png"
                    outp = os.path.join(PLOTS_DIR, fname)
                    plot_trade_tv(sym, full_df, row, outp)

            if summ:
                all_summ.append(summ)

            if not df_eq.empty:
                eq_curves[sym] = df_eq.assign(symbol=sym)

        if all_trades:
            merged = pd.concat(all_trades, ignore_index=True).sort_values("placed_at")
            merged_path = os.path.join(OUT_DIR, "ALL_trades.csv")
            merged.to_csv(merged_path, index=False)
            print(f"ALL trades → {merged_path}")

        if all_summ:
            df_summ = pd.DataFrame(all_summ)
            summ_path = os.path.join(OUT_DIR, "SUMMARY_PER_SYMBOL.csv")
            df_summ.to_csv(summ_path, index=False)
            print("\nSUMMARY (per symbol):")
            print(df_summ.to_string(index=False))
        else:
            print("No trades generated. Check sessions/data.")

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
            overall_path = os.path.join(OUT_DIR, "SUMMARY_OVERALL.csv")
            pd.DataFrame([overall]).to_csv(overall_path, index=False)

            print("\nOVERALL:")
            print(pd.DataFrame([overall]).to_string(index=False))
        else:
            print("No trades generated overall.")

        if eq_curves:
            all_eq = pd.concat(eq_curves.values(), ignore_index=True).sort_values(["symbol", "time"])
            eq_path = os.path.join(OUT_DIR, "EQUITY_CURVES.csv")
            all_eq.to_csv(eq_path, index=False)
            print(f"Equity curves → {eq_path}")

        print(f"\nEarliest trade plots → {PLOTS_DIR}")

    finally:
        if DATA_SOURCE == "mt5" and mt5 is not None:
            mt5.shutdown()

if __name__ == "__main__":
    main()
