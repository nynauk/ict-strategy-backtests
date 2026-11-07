# main.py
import os
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.dates as mdates

# =========================
# CONFIG
# =========================
SYMBOL = "DEUIDXEUR"

M5_FILE = f"{SYMBOL}_M5.csv"
H1_FILE = f"{SYMBOL}_H1.csv"
H4_FILE = f"{SYMBOL}_H4.csv"
D1_FILE = f"{SYMBOL}_D1.csv"

# Asia window
ASIA_START_HHMM = (0, 0)
ASIA_END_HHMM   = (7, 0)

# London window
LONDON_START_HHMM = (8, 0)
LONDON_END_HHMM   = (10, 0)

# Inversion FVG requirements
MAX_INV_CANDLES = 5

# HTF FVG strictness
STRICT_UNMITIGATED = True   # if True, require HTF FVG never touched since creation

# --- IFVG selection rules ---
# Only the DISRESPECT must be after the tap now; IFVG may have formed earlier.
IFVG_MUST_FORM_AFTER_TAP   = False          # changed from True
# Enforce the IFVG zone is inside the HTF FVG POI (when POI is an FVG); ignored for prev high/low
IFVG_MUST_BE_INSIDE_POI    = True           # changed from False
# Require fully inside (not just overlap) for HTF FVG POIs
IFVG_ALLOW_PARTIAL_OVERLAP = False          # changed from True

# --- Asia mid side rule ---
# Sell setup → IFVG fully ABOVE Asia mid; Buy setup → IFVG fully BELOW Asia mid
IFVG_MUST_BE_ON_CORRECT_SIDE_OF_ASIA_MID = True

# --- TP CONFIG ---
# We run BOTH "asia_opposite" and "fixed_rr" in one process().
FIXED_R_MULT = 2.3           # used for fixed_rr mode

# --- EQUITY SETTINGS ---
RISK_PER_TRADE = 0.01        # 1% per trade equity curve (for normalized equity)

# Risk guards to prevent absurd R from near-zero risk
MIN_RISK_ABS = 1.0            # absolute points (e.g., 1 index point)
MIN_RISK_BPS = 5              # 5 bps = 0.05% of entry price

# --- FILTERS ---
# For fixed-RR: enforce TP must be on the allowed side of Asia.
#   - BUY (swept_side == "low"): TP must be strictly BELOW Asia High (tp < asia_high)
#   - SELL (swept_side == "high"): TP must be strictly ABOVE Asia Low  (tp > asia_low)
FILTER_FIXED_RR_TP_MUST_RESIDE_ON_ALLOWED_SIDE_OF_ASIA = True

# For asia_opposite mode: require the Asia-opposite TP distance to be at least this many R, else skip trade
FILTER_ASIA_OPPOSITE_MIN_R = True
ASIA_OPPOSITE_MIN_R = 1.5

# Output
PLOT_DIR = "plots"
OUTPUT_DIR = "output"
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Colors
COLORS = {
    "candle_up":   "#26a69a",
    "candle_down": "#ef5350",
    "asia":        "#42a5f5",
    "london":      "#ab47bc",
    "poi":         "#ffca28",
    "ifvg":        "#66bb6a",
    "entry":       "#1e88e5",
    "exit":        "#d81b60",
    "tap_v":       "#616161",
    "fvg_v":       "#5c6bc0",
    "dis_v":       "#00796b",
    "sl":          "#d32f2f",
    "tp":          "#2e7d32",
}

# Logging
VERBOSE = True
PER_DAY_LOG = True

def log(msg: str):
    if VERBOSE:
        print(f"[{pd.Timestamp.now()}] {msg}", flush=True)

# =========================
# UTILITIES
# =========================

def _auto_parse_datetime(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    looks_iso = s.str.match(r"^\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}(?::\d{2})?)?$").mean() >= 0.8
    uses_slashes = s.str.contains(r"/").mean() >= 0.5
    if looks_iso:
        log("Datetime parse: detected ISO style → dayfirst=False")
        return pd.to_datetime(s, errors="coerce", dayfirst=False)
    elif uses_slashes:
        log("Datetime parse: detected slash style → dayfirst=True")
        return pd.to_datetime(s, errors="coerce", dayfirst=True)
    else:
        log("Datetime parse: ambiguous → trying both dayfirst True/False and merging")
        dt_a = pd.to_datetime(s, errors="coerce", dayfirst=True)
        dt_b = pd.to_datetime(s, errors="coerce", dayfirst=False)
        return dt_a.where(dt_a.notna(), dt_b)


def read_csv_any(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    log(f"Loading CSV: {path}")
    df = pd.read_csv(path)

    def _norm(c): return c.lower().strip() if isinstance(c, str) else str(c)
    norm_cols = [_norm(c) for c in df.columns]
    known = {"time","datetime","timestamp","date","open","high","low","close","volume","vol"}
    has_known = any(c in known for c in norm_cols)

    if not has_known:
        log("Headerless CSV detected → reloading with header=None and assigning names by position")
        df = pd.read_csv(path, header=None)
        if df.shape[1] < 5:
            raise ValueError(f"CSV {path} has {df.shape[1]} columns; expected at least 5.")
        names = ["time","open","high","low","close"]
        if df.shape[1] >= 6: names.append("volume")
        df = df.iloc[:, :len(names)]
        df.columns = names
    else:
        log("Named columns detected → normalizing column names")
        cols = {c.lower().strip(): c for c in df.columns if isinstance(c, str)}
        rename = {}
        for k in ("time","datetime","timestamp","date"):
            if k in cols: rename[cols[k]] = "time"; break
        for k in ("open","high","low","close","volume","vol"):
            if k in cols: rename[cols[k]] = "volume" if k in ("volume","vol") else k
        df = df.rename(columns=rename)

    if "time" not in df.columns or not all(c in df.columns for c in ("open","high","low","close")):
        raise ValueError(f"Missing required columns in {path}. Found columns: {list(df.columns)}")

    dt = _auto_parse_datetime(df["time"])
    df = df.assign(
        time=dt,
        open=pd.to_numeric(df["open"],  errors="coerce"),
        high=pd.to_numeric(df["high"],  errors="coerce"),
        low =pd.to_numeric(df["low"],   errors="coerce"),
        close=pd.to_numeric(df["close"],errors="coerce"),
        volume=pd.to_numeric(df["volume"],errors="coerce") if "volume" in df.columns else np.nan
    )
    before = len(df)
    df = df.dropna(subset=["time","open","high","low","close"]).copy()
    after = len(df)
    log(f"Parsed {after}/{before} rows → index by time and sort")
    df = df.set_index("time").sort_index()
    return df[["open","high","low","close","volume"]]


def day_bounds(ts: pd.Timestamp, hhmm_start, hhmm_end) -> Tuple[pd.Timestamp, pd.Timestamp]:
    y,m,d = ts.year, ts.month, ts.day
    st = pd.Timestamp(year=y, month=m, day=d, hour=hhmm_start[0], minute=hhmm_start[1])
    en = pd.Timestamp(year=y, month=m, day=d, hour=hhmm_end[0], minute=hhmm_end[1])
    return st, en


def window(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    return df.loc[(df.index >= start) & (df.index <= end)]

# =========================
# CORE STRUCTURES
# =========================

@dataclass
class FVG:
    tf: str
    created_at: pd.Timestamp
    is_bearish: bool        # True = bearish gap (supply), False = bullish (demand)
    zone_low: float
    zone_high: float
    first_touch_time: Optional[pd.Timestamp] = None
    _created_pos: Optional[int] = None


def detect_fvgs(df: pd.DataFrame, tf_label: str) -> List[FVG]:
    fvgs: List[FVG] = []
    if len(df) < 3: return fvgs
    hs = df["high"].values
    ls = df["low"].values
    idx = df.index.to_list()
    for i in range(1, len(df)-1):
        a_high = hs[i-1]; a_low = ls[i-1]
        c_high = hs[i+1]; c_low = ls[i+1]
        if a_high < c_low:  # bullish
            fvgs.append(FVG(tf=tf_label, created_at=idx[i], is_bearish=False, zone_low=a_high, zone_high=c_low))
        if a_low > c_high:  # bearish
            fvgs.append(FVG(tf=tf_label, created_at=idx[i], is_bearish=True, zone_low=c_high, zone_high=a_low))
    return fvgs


def precompute_first_touches(tf_df: pd.DataFrame, fvgs: List[FVG]) -> None:
    if not fvgs: return
    t0 = time.time()
    log(f"Precomputing first-touch for {fvgs[0].tf} FVGs ({len(fvgs)} items)")
    highs = tf_df["high"].values
    lows  = tf_df["low"].values
    idx   = tf_df.index.to_numpy()
    pos_map = {ts: i for i, ts in enumerate(idx)}
    for f in fvgs:
        pos = pos_map.get(f.created_at)
        f._created_pos = pos
        if pos is None: continue
        zL, zH = f.zone_low, f.zone_high
        j = pos + 1
        while j < len(highs):
            if highs[j] >= zL and lows[j] <= zH:
                f.first_touch_time = idx[j]
                break
            j += 1
    log(f"  Done precompute for {fvgs[0].tf}: {time.time()-t0:.2f}s")


def is_unmitigated_before_fast(fvg: FVG, cutoff: pd.Timestamp) -> bool:
    if fvg.first_touch_time is None: return True
    return fvg.first_touch_time >= cutoff


def poi_fully_outside_asia(fvg: FVG, asia_low: float, asia_high: float) -> bool:
    return (fvg.zone_high < asia_low) or (fvg.zone_low > asia_high)


def poi_side_matches_sweep(fvg: FVG, swept_side: str, asia_low: float, asia_high: float) -> bool:
    if swept_side == "high":
        return fvg.is_bearish and (fvg.zone_low > asia_high)   # supply above Asia High
    else:
        return (not fvg.is_bearish) and (fvg.zone_high < asia_low)  # demand below Asia Low


def first_tap_in_zone_time(m5: pd.DataFrame, z_low: float, z_high: float,
                           start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> Optional[pd.Timestamp]:
    sub = window(m5, start_ts, end_ts)
    if sub.empty: return None
    touch = (sub["high"] >= z_low) & (sub["low"] <= z_high)
    if not touch.any(): return None
    return sub.index[np.argmax(touch.values)]


def asia_range(m5_day: pd.DataFrame, asia_start: pd.Timestamp, asia_end: pd.Timestamp) -> Optional[Tuple[float,float]]:
    asia = window(m5_day, asia_start, asia_end)
    if len(asia) == 0: return None
    return float(asia["low"].min()), float(asia["high"].max())

# =========================
# STRATEGY STEPS
# =========================

def first_asia_edge_swept_after_asia(
    m5: pd.DataFrame,
    asia_low: float,
    asia_high: float,
    asia_end: pd.Timestamp,
    day_end: pd.Timestamp
) -> Optional[Tuple[str, pd.Timestamp]]:
    """
    From Asia end to day end, find the *first* sweep of the Asia range.
    Returns:
      - ("high", t) if Asia High is swept first at time t  → SELL bias
      - ("low",  t)  if Asia Low  is swept first at time t  → BUY bias
    """
    rest = window(m5, asia_end, day_end)
    if rest.empty:
        return None

    broke_high = rest["high"] >= asia_high
    broke_low  = rest["low"]  <= asia_low

    t_high = rest.index[np.argmax(broke_high.values)] if broke_high.any() else None
    t_low  = rest.index[np.argmax(broke_low.values)]  if broke_low.any()  else None

    if t_high is None and t_low is None:
        return None
    if t_high is not None and t_low is not None:
        return ("high", t_high) if t_high <= t_low else ("low", t_low)
    if t_high is not None:
        return ("high", t_high)
    return ("low", t_low)

def previous_day_hilo(m5: pd.DataFrame, day_ts: pd.Timestamp) -> Tuple[Optional[float], Optional[float]]:
    """
    Return (prev_day_high, prev_day_low) for the calendar day BEFORE `day_ts`.
    If there is no data for that day, returns (None, None).
    """
    prev_date = (day_ts - pd.Timedelta(days=1)).date()
    day_start = pd.Timestamp(prev_date)
    day_end   = day_start + pd.Timedelta(days=1) - pd.Timedelta(minutes=1)
    prev = window(m5, day_start, day_end)
    if prev.empty:
        return None, None
    return float(prev["high"].max()), float(prev["low"].min())

def find_first_valid_poi_tap(
    m5: pd.DataFrame,
    asia_low: float,
    asia_high: float,
    swept_side: str,
    london_start: pd.Timestamp,
    london_end: pd.Timestamp,
    htf_fvgs_sorted: List[FVG],
    day_start: pd.Timestamp,
    prev_high: Optional[float],
    prev_low: Optional[float],
    london_low: float,
    london_high: float,
    require_unmitigated: bool = STRICT_UNMITIGATED,
) -> Optional[Tuple[str, object, pd.Timestamp, Tuple[float, float]]]:
    candidates: List[Tuple[pd.Timestamp, str, object, Tuple[float,float]]] = []
    cnt_prior = cnt_outside = cnt_side = cnt_unmit_ok = cnt_reach = 0

    for f in htf_fvgs_sorted:
        if f.created_at >= day_start: break
        cnt_prior += 1
        if not poi_fully_outside_asia(f, asia_low, asia_high): continue
        cnt_outside += 1
        if not poi_side_matches_sweep(f, swept_side, asia_low, asia_high): continue
        cnt_side += 1
        if require_unmitigated and (not is_unmitigated_before_fast(f, london_start)): continue
        cnt_unmit_ok += 1
        if swept_side == "high" and f.zone_low > london_high: continue
        if swept_side == "low" and f.zone_high < london_low: continue
        cnt_reach += 1
        tap_ts = first_tap_in_zone_time(m5, f.zone_low, f.zone_high, london_start, london_end)
        if tap_ts is not None:
            candidates.append((tap_ts, f"FVG_{f.tf}", f, (f.zone_low, f.zone_high)))

    if PER_DAY_LOG:
        log(
            f"  HTF FVG filter counts → prior:{cnt_prior}, outsideAsia:{cnt_outside}, "
            f"sideOK:{cnt_side}, "
            f"{'unmitOK:'+str(cnt_unmit_ok) if require_unmitigated else 'unmitOK:SKIPPED'}, "
            f"reachOK:{cnt_reach}, tapped:{len(candidates)}"
        )

    # Prev High/Low as POIs
    eps = 1e-9
    if prev_high is not None and swept_side == "high" and prev_high > asia_high and prev_high <= london_high:
        t = first_tap_in_zone_time(m5, prev_high-eps, prev_high+eps, london_start, london_end)
        if t is not None:
            candidates.append((t, "PrevHigh", prev_high, (prev_high, prev_high)))

    if prev_low is not None and swept_side == "low" and prev_low < asia_low and prev_low >= london_low:
        t = first_tap_in_zone_time(m5, prev_low-eps, prev_low+eps, london_start, london_end)
        if t is not None:
            candidates.append((t, "PrevLow", prev_low, (prev_low, prev_low)))

    if not candidates: return None
    candidates.sort(key=lambda x: x[0])
    tap_time, poi_type, obj, zone = candidates[0]
    return poi_type, obj, tap_time, zone


def disrespect_within_n(
    m5: pd.DataFrame,
    fvg: FVG,
    max_candles: int,
    london_end: pd.Timestamp
) -> Optional[pd.Timestamp]:
    """
    Returns the timestamp of the first candle that 'disrespects' the FVG within max_candles,
    capped at London end. None if not found.
    """
    base = m5.loc[m5.index >= fvg.created_at]
    if base.empty: return None
    base = base.iloc[:max_candles]
    base = base.loc[base.index <= london_end]
    if base.empty: return None
    cond = (base["close"] > fvg.zone_high) if fvg.is_bearish else (base["close"] < fvg.zone_low)
    if not cond.any(): return None
    return base.index[np.argmax(cond.values)]


def first_ifvg_disrespected_after_tap(
    m5: pd.DataFrame,
    tap_time: pd.Timestamp,
    swept_side: str,
    london_start: pd.Timestamp,
    london_end: pd.Timestamp,
    max_inv_candles: int,
    poi_zone: Tuple[float, float],
    enforce_inside: bool,
    asia_mid: float,
    poi_type: str,                            # NEW
) -> Optional[Tuple[FVG, pd.Timestamp, pd.Timestamp, int]]:
    """
    Scan opposite-direction M5 IFVGs inside London.
    Choose the one whose *disrespect* occurs earliest AFTER tap_time (within max_inv_candles).
    """
    m5_london = window(m5, london_start, london_end)
    if len(m5_london) < 3:
        return None

    fvgs = detect_fvgs(m5_london, "M5")
    if swept_side == "high":
        fvgs = [f for f in fvgs if not f.is_bearish]     # bullish IFVG for sell setup
    else:
        fvgs = [f for f in fvgs if f.is_bearish]         # bearish IFVG for buy setup

    fvgs.sort(key=lambda f: f.created_at)

    zL, zH = poi_zone
    skipped = 0
    candidates: List[Tuple[pd.Timestamp, FVG]] = []

    # Only enforce "inside" when POI is an HTF FVG box (not PrevHigh/PrevLow lines)
    enforce_inside_local = enforce_inside and poi_type.startswith("FVG_")   # NEW

    for f in fvgs:
        # IFVG may have formed before the tap now (global flag controls this)
        if IFVG_MUST_FORM_AFTER_TAP and f.created_at < tap_time:
            skipped += 1
            if PER_DAY_LOG:
                log(f"    Skip IFVG @ {f.created_at}: formed before tap")
            continue

        # For HTF FVG POIs: require the IFVG zone to be inside the POI box
        if enforce_inside_local and np.isfinite(zL) and np.isfinite(zH) and zH > zL:
            if IFVG_ALLOW_PARTIAL_OVERLAP:
                inside_ok = not (f.zone_high < zL or f.zone_low > zH)
            else:
                inside_ok = (f.zone_low >= zL) and (f.zone_high <= zH)
            if not inside_ok:
                skipped += 1
                if PER_DAY_LOG:
                    log(f"    Skip IFVG @ {f.created_at}: IFVG not inside HTF FVG POI")
                continue

            # Optional: ensure the forming candle traded inside the POI
            try:
                bar = m5.loc[f.created_at]
                formed_inside_range = (bar["high"] >= zL) and (bar["low"] <= zH)
            except KeyError:
                formed_inside_range = True  # if missing due to index granularity
            if not formed_inside_range:
                skipped += 1
                if PER_DAY_LOG:
                    log(f"    Skip IFVG @ {f.created_at}: forming candle not trading inside POI")
                continue

        # Side of Asia mid
        if IFVG_MUST_BE_ON_CORRECT_SIDE_OF_ASIA_MID:
            if swept_side == "high":  # SELL → above mid
                if not (f.zone_low >= asia_mid):
                    skipped += 1
                    if PER_DAY_LOG: log(f"    Skip IFVG @ {f.created_at}: not ABOVE Asia mid ({asia_mid:.2f})")
                    continue
            else:  # BUY → below mid
                if not (f.zone_high <= asia_mid):
                    skipped += 1
                    if PER_DAY_LOG: log(f"    Skip IFVG @ {f.created_at}: not BELOW Asia mid ({asia_mid:.2f})")
                    continue

        # Must get disrespected within N candles (and before London end)
        dis_t = disrespect_within_n(m5, f, max_inv_candles, london_end)
        if dis_t is None:
            skipped += 1
            if PER_DAY_LOG: log(f"    Skip IFVG @ {f.created_at}: no disrespect within {max_inv_candles}")
            continue

        # Disrespect must be AFTER tap
        if dis_t < tap_time:
            skipped += 1
            if PER_DAY_LOG: log(f"    Skip IFVG @ {f.created_at}: disrespect {dis_t} before tap {tap_time}")
            continue

        candidates.append((dis_t, f))

    if not candidates:
        return None

    candidates.sort(key=lambda t: t[0])
    chosen_dis_t, chosen_f = candidates[0]
    return chosen_f, chosen_f.created_at, chosen_dis_t, skipped

# --- TP helper ---
def compute_tp_price(tp_mode: str, swept_side: str, entry_price: float,
                     asia_low: float, asia_high: float, risk_used: float,
                     fixed_r_mult: float) -> float:
    if tp_mode == "asia_opposite":
        return asia_low if swept_side == "high" else asia_high
    elif tp_mode == "asia_mid":
        return 0.5 * (asia_low + asia_high)
    elif tp_mode == "fixed_rr":
        return entry_price - risk_used * fixed_r_mult if swept_side == "high" else entry_price + risk_used * fixed_r_mult
    else:
        return asia_low if swept_side == "high" else asia_high


def simulate_trade(
    m5: pd.DataFrame,
    swept_side: str,
    asia_low: float,
    asia_high: float,
    entry_time: pd.Timestamp,
    entry_price: float,
    fvg_time: pd.Timestamp,
    disrespect_time: pd.Timestamp,
    tp_mode: str,
    fixed_r_mult: float
) -> Tuple[bool, float, float, float, pd.Timestamp, Optional[pd.Timestamp], Optional[pd.Timestamp], float, Optional[float], float, float, float, str]:
    """
    Return:
      (is_win, sl, tp, entry_price, entry_time, sl_hit_time, tp_hit_time, realized_R, exit_price,
       risk_raw, risk_floor, risk_used, exit_reason)
    """
    sub = window(m5, fvg_time, disrespect_time)
    if sub.empty:
        return False, np.nan, np.nan, entry_price, entry_time, None, None, float('nan'), None, float('nan'), float('nan'), float('nan'), "NoSubWindowForSL"

    hh = float(sub["high"].max()); ll = float(sub["low"].min())

    # SL from swing extremes between FVG and disrespect
    if swept_side == "high":   # SELL
        sl = hh + 1e-6
    else:                      # BUY
        sl = ll - 1e-6

    # --- Risk floor to prevent absurd R ---
    risk_raw = abs(sl - entry_price)
    risk_floor = max(MIN_RISK_ABS, entry_price * (MIN_RISK_BPS / 10000.0))
    risk_used = max(risk_raw, risk_floor)

    # TP per mode
    tp = compute_tp_price(tp_mode, swept_side, entry_price, asia_low, asia_high, risk_used, fixed_r_mult)

    # --- Filter: fixed-RR TP must be on the allowed side of Asia range ---
    if tp_mode == "fixed_rr" and FILTER_FIXED_RR_TP_MUST_RESIDE_ON_ALLOWED_SIDE_OF_ASIA:
        if swept_side == "low" and not (tp < asia_high):   # BUY must be below Asia High
            return False, sl, tp, entry_price, entry_time, None, None, float('nan'), None, risk_raw, risk_floor, risk_used, "FilteredFixedRR_BuyTPNotBelowAsiaHigh"
        if swept_side == "high" and not (tp > asia_low):   # SELL must be above Asia Low
            return False, sl, tp, entry_price, entry_time, None, None, float('nan'), None, risk_raw, risk_floor, risk_used, "FilteredFixedRR_SellTPNotAboveAsiaLow"

    # --- Filter: Asia-opposite TP must be at least ASIA_OPPOSITE_MIN_R away (in R using risk_used) ---
    if tp_mode == "asia_opposite" and FILTER_ASIA_OPPOSITE_MIN_R:
        asia_required_R = abs(entry_price - tp) / (risk_used if risk_used != 0 else np.inf)
        if not (asia_required_R >= ASIA_OPPOSITE_MIN_R):
            return False, sl, tp, entry_price, entry_time, None, None, float('nan'), None, risk_raw, risk_floor, risk_used, "FilteredAsiaOppositeBelowMinR"

    # Evaluate forward path AFTER entry
    fut = m5.loc[m5.index > entry_time]
    if fut.empty:
        return False, sl, tp, entry_price, entry_time, None, None, -1.0, None, risk_raw, risk_floor, risk_used, "NoFutureData"

    if swept_side == "high":  # SELL
        hit_sl = fut["high"] >= sl
        hit_tp = fut["low"]  <= tp
    else:                      # BUY
        hit_sl = fut["low"]  <= sl
        hit_tp = fut["high"] >= tp

    sl_t = fut.index[np.argmax(hit_sl.values)] if hit_sl.any() else None
    tp_t = fut.index[np.argmax(hit_tp.values)] if hit_tp.any() else None

    if sl_t is not None and tp_t is not None:
        if tp_t <= sl_t:
            rr = abs(entry_price - tp) / risk_used
            return True, sl, tp, entry_price, entry_time, None, tp_t, rr, tp, risk_raw, risk_floor, risk_used, "TP"
        else:
            return False, sl, tp, entry_price, entry_time, sl_t, None, -1.0, sl, risk_raw, risk_floor, risk_used, "SL"
    elif sl_t is not None:
        return False, sl, tp, entry_price, entry_time, sl_t, None, -1.0, sl, risk_raw, risk_floor, risk_used, "SL"
    elif tp_t is not None:
        rr = abs(entry_price - tp) / risk_used
        return True, sl, tp, entry_price, entry_time, None, tp_t, rr, tp, risk_raw, risk_floor, risk_used, "TP"
    else:
        return False, sl, tp, entry_price, entry_time, None, None, -1.0, None, risk_raw, risk_floor, risk_used, "NoExit"

# =========================
# RESULTS / PLOTTING
# =========================

@dataclass
class DayResult:
    date: pd.Timestamp
    swept_side: str
    asia_low: float
    asia_high: float
    poi_type: str
    poi_zone: Tuple[float,float]
    tap_time: pd.Timestamp
    ifvg_zone: Tuple[float,float]
    fvg_time: pd.Timestamp
    disrespect_time: pd.Timestamp
    entry_time: pd.Timestamp
    is_win: bool
    sl: float
    tp: float
    entry_price: float
    exit_time: Optional[pd.Timestamp]
    exit_price: Optional[float]
    rr: float  # realized R

def to_num(ts: pd.Timestamp) -> float:
    return mdates.date2num(pd.to_datetime(ts))

def plot_box(ax, x0: pd.Timestamp, x1: pd.Timestamp, y0: float, y1: float, color: str, alpha: float, label: str, z=0.5):
    if x0 >= x1 or y0 >= y1: return
    x0n = to_num(x0); x1n = to_num(x1)
    rect = Rectangle((x0n, y0), x1n - x0n, y1 - y0, facecolor=color, alpha=alpha, linewidth=0.8, edgecolor=color, zorder=z)
    ax.add_patch(rect)
    ax.text(x0n, y1, f" {label}", va="bottom", color=color, zorder=z+0.1)

def _candles(ax, df: pd.DataFrame, up="#26a69a", down="#ef5350", z=1.5):
    if df.empty: return
    xs = mdates.date2num(df.index.to_pydatetime())
    opens = df["open"].values; highs = df["high"].values
    lows  = df["low"].values;  closes= df["close"].values
    if len(xs) >= 2: spacing = float(np.median(np.diff(xs)))
    else: spacing = 5.0 / (24*60)
    bw = spacing * 0.6; half = bw / 2.0
    for x, o, h, l, c in zip(xs, opens, highs, lows, closes):
        color = up if c >= o else down
        ax.plot([x, x], [l, h], color=color, linewidth=1.0, zorder=z)  # wick
        y_bottom = min(o, c)
        height = max(abs(c - o), 0.5)  # doji visibility
        rect = Rectangle((x - half, y_bottom), bw, height, facecolor=color, edgecolor=color, zorder=z)
        ax.add_patch(rect)

def plot_trade(m5: pd.DataFrame, res: DayResult, out_path: str):
    day = res.date.normalize()
    a_st, a_en = day_bounds(day, ASIA_START_HHMM, ASIA_END_HHMM)
    l_st, l_en = day_bounds(day, LONDON_START_HHMM, LONDON_END_HHMM)
    day_end = day + pd.Timedelta(days=1) - pd.Timedelta(minutes=1)
    sub = window(m5, day, day_end)
    if sub.empty: return

    lo = window(m5, l_st, l_en)
    if not lo.empty:
        london_low = float(lo["low"].min())
        london_high = float(lo["high"].max())
    else:
        london_low = float(sub["low"].min())
        london_high = float(sub["high"].max())

    fig, ax = plt.subplots(figsize=(14,8))

    # Boxes (under candles)
    plot_box(ax, a_st, a_en, res.asia_low, res.asia_high, COLORS["asia"], 0.10, "Asia", z=0.2)
    plot_box(ax, l_st, l_en, london_low, london_high, COLORS["london"], 0.10, "London", z=0.25)

    # HTF POI zone across day
    zL, zH = res.poi_zone
    if abs(zH - zL) < 1e-7:
        ax.axhline(zL, linewidth=1.4, linestyle="--", color=COLORS["poi"], label=f"POI {res.poi_type}", zorder=0.3)
    else:
        plot_box(ax, day, day_end, zL, zH, COLORS["poi"], 0.12, f"{res.poi_type}", z=0.3)

    # IFVG zone
    fL, fH = res.ifvg_zone
    start_x = max(res.fvg_time, l_st); end_x = l_en
    if fH > fL:
        plot_box(ax, start_x, end_x, fL, fH, COLORS["ifvg"], 0.14, "IFVG", z=0.35)

    # Candlesticks
    _candles(ax, sub, up=COLORS["candle_up"], down=COLORS["candle_down"], z=1.5)

    # Vertical markers
    ax.axvline(res.tap_time, linestyle=":", linewidth=1.2, color=COLORS["tap_v"], label="POI tap", zorder=2.0)
    ax.axvline(res.fvg_time, linestyle=":", linewidth=1.2, color=COLORS["fvg_v"], label="IFVG formed", zorder=2.0)
    ax.axvline(res.disrespect_time, linestyle=":", linewidth=1.2, color=COLORS["dis_v"], label="Disrespect", zorder=2.0)

    # SL/TP
    ax.axhline(res.sl, linewidth=1.3, color=COLORS["sl"], label="SL", zorder=2.0)
    ax.axhline(res.tp, linewidth=1.3, color=COLORS["tp"], label="TP", zorder=2.0)

    # Entry / Exit
    ax.plot(res.entry_time, res.entry_price, marker="*", markersize=16, color=COLORS["entry"], label="Entry (limit)", zorder=3.0)
    ax.text(res.entry_time, res.entry_price, " Entry (limit)", va="bottom", color=COLORS["entry"])
    if res.exit_time is not None and res.exit_price is not None:
        ax.plot(res.exit_time, res.exit_price, marker="*", markersize=16, color=COLORS["exit"], label="Exit", zorder=3.0)
        ax.text(res.exit_time, res.exit_price, f" {'Exit TP' if res.is_win else 'Exit SL'}", va="bottom", color=COLORS["exit"])

    title_outcome = 'WIN' if res.is_win else 'LOSS'
    ax.set_title(
        f"{SYMBOL} {res.date.date()} | {res.swept_side.upper()} sweep | {res.poi_type} | "
        f"{title_outcome} | R={res.rr:.2f}"
        f"{' | TP=fixed_rr('+str(FIXED_R_MULT)+'R)' if 'fixedrr' in out_path else ' | TP=asia_opposite'}"
    )
    ax.legend(loc="best")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def plot_equity(dates: List[pd.Timestamp], cum_r: np.ndarray, equity: np.ndarray, out_path: str, title: str):
    fig, ax1 = plt.subplots(figsize=(12,6))
    x = pd.to_datetime(dates)
    ax1.plot(x, equity, linewidth=1.6)
    ax1.set_ylabel("Equity (1.0 start, {:.2%} risk/trade)".format(RISK_PER_TRADE))
    ax1.set_title(title)
    ax1.grid(True, alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(x, cum_r, linewidth=1.2, linestyle="--")
    ax2.set_ylabel("Cumulative R")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def plot_equity_overlay(dates1: List[pd.Timestamp], eq1: np.ndarray, label1: str,
                        dates2: List[pd.Timestamp], eq2: np.ndarray, label2: str,
                        out_path: str, title: str):
    fig, ax = plt.subplots(figsize=(12,6))
    x1 = pd.to_datetime(dates1); x2 = pd.to_datetime(dates2)
    ax.plot(x1, eq1, linewidth=1.6, label=label1)
    ax.plot(x2, eq2, linewidth=1.6, label=label2)
    ax.set_ylabel("Equity (1.0 start, {:.2%} risk/trade)".format(RISK_PER_TRADE))
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

# =========================
# BACKTEST RUNNER (per mode)
# =========================

def run_backtest_for_mode(
    tp_mode: str,
    m5: pd.DataFrame,
    htf_fvgs_sorted: List[FVG],
    fixed_r_mult: float
) -> Dict:
    """
    Returns a dict with:
      - 'trades': List[DayResult]
      - 'stats': dict
      - 'equity_dates': list of timestamps (entry_time order)
      - 'cum_r': np.array
      - 'equity': np.array
      - 'trades_csv_path': str
      - 'recent_plots': List[str]
      - 'best_plot': Optional[str]
    """
    log(f"=== Start backtest mode: {tp_mode} ===")
    dates = pd.to_datetime(m5.index.date)
    unique_days = sorted(pd.unique(dates))
    log(f"M5 covers {len(unique_days)} trading day(s)")

    # Stats counters
    poi_touches = trades_taken = wins = losses = 0
    sweep_high_cnt = sweep_low_cnt = 0
    cur_win = cur_los = max_win = max_los = 0
    rr_values: List[float] = []
    all_trades: List[DayResult] = []
    trade_rows: List[dict] = []
    best_trade: Optional[DayResult] = None
    best_rr_val = -1.0

    for i, day in enumerate(unique_days, start=1):
        day_ts = pd.Timestamp(day)
        if PER_DAY_LOG: log(f"[{tp_mode}] Day {i}/{len(unique_days)} {day_ts.date()} → start")

        asia_start, asia_end = day_bounds(day_ts, ASIA_START_HHMM, ASIA_END_HHMM)
        london_start, london_end = day_bounds(day_ts, LONDON_START_HHMM, LONDON_END_HHMM)

        day_start = day_ts
        day_end = day_ts + pd.Timedelta(days=1) - pd.Timedelta(minutes=1)
        day_m5 = window(m5, day_start, day_end)
        if len(day_m5) == 0:
            if PER_DAY_LOG: log("  No M5 data for day → skip"); continue

        ar = asia_range(day_m5, asia_start, asia_end)
        if ar is None:
            if PER_DAY_LOG: log("  No Asia data in day window → skip"); continue
        a_low, a_high = ar
        a_mid = 0.5 * (a_low + a_high)
        if PER_DAY_LOG: log(f"  Asia range: low={a_low:.2f} high={a_high:.2f} mid={a_mid:.2f}")

        lo = window(m5, london_start, london_end)
        if lo.empty:
            if PER_DAY_LOG: log("  No London data → skip"); continue
        london_low = float(lo["low"].min())
        london_high = float(lo["high"].max())

        # Bias from first Asia sweep AFTER Asia ends; must occur during London
        asia_sweep = first_asia_edge_swept_after_asia(m5, a_low, a_high, asia_end, day_end)
        if asia_sweep is None:
            if PER_DAY_LOG: log("  No Asia edge swept after Asia session → skip (no bias)")
            continue
        swept_side, sweep_time = asia_sweep
        if sweep_time < london_start or sweep_time > london_end:
            if PER_DAY_LOG: log(f"  Asia first sweep @ {sweep_time} outside London → skip")
            continue

        if swept_side == "high": sweep_high_cnt += 1
        else: sweep_low_cnt += 1
        if PER_DAY_LOG:
            bias_str = "SELL" if swept_side == "high" else "BUY"
            log(f"  Day bias: {bias_str} (swept {swept_side} @ {sweep_time})")

        prev_hi, prev_lo = previous_day_hilo(m5, day_ts)

        strike = find_first_valid_poi_tap(
            m5=m5, asia_low=a_low, asia_high=a_high, swept_side=swept_side,
            london_start=london_start, london_end=london_end,
            htf_fvgs_sorted=htf_fvgs_sorted, day_start=day_start,
            prev_high=prev_hi, prev_low=prev_lo,
            london_low=london_low, london_high=london_high,
            require_unmitigated=STRICT_UNMITIGATED,
        )
        if strike is None:
            if PER_DAY_LOG: log("  No qualifying POI tap inside London → skip"); continue

        poi_type, poi_obj, tap_time, poi_zone = strike
        poi_touches += 1
        if PER_DAY_LOG:
            if abs(poi_zone[1]-poi_zone[0]) < 1e-7:
                log(f"  POI tap: {poi_type} level={poi_zone[0]:.2f} @ {tap_time}")
            else:
                log(f"  POI tap: {poi_type} zone=({poi_zone[0]:.2f},{poi_zone[1]:.2f}) @ {tap_time}")

        # IFVG selection
        # Enforce "inside" only when the POI is an HTF FVG; ignore for PrevHigh/PrevLow (line POIs)
        enforce_inside = poi_type.startswith("FVG_") and IFVG_MUST_BE_INSIDE_POI

        found = first_ifvg_disrespected_after_tap(
            m5=m5,
            tap_time=tap_time,
            swept_side=swept_side,
            london_start=london_start,
            london_end=london_end,
            max_inv_candles=MAX_INV_CANDLES,
            poi_zone=poi_zone,
            enforce_inside=enforce_inside,
            asia_mid=a_mid,
            poi_type=poi_type,                        # pass POI type
        )
        if found is None:
            if PER_DAY_LOG: log("  No opposite-direction M5 IFVG meets rules → no-trade")
            continue

        ifvg, ifvg_time, dis_time, skipped_count = found
        if PER_DAY_LOG:
            direction = "bullish" if not ifvg.is_bearish else "bearish"
            if skipped_count > 0:
                log(f"  Skipped {skipped_count} IFVG(s) that failed filters")
            log(f"  Chosen M5 IFVG ({direction}) @ {ifvg_time}; disrespect @ {dis_time}")

        # === LIMIT ENTRY after disrespect ===
        # bullish IFVG → limit at zone_low; bearish → limit at zone_high
        limit_price = ifvg.zone_low if (not ifvg.is_bearish) else ifvg.zone_high
        fut = m5.loc[m5.index > dis_time]
        if fut.empty:
            if PER_DAY_LOG: log("  No future bars after disrespect → skip")
            continue
        touched = (fut["low"] <= limit_price) & (fut["high"] >= limit_price)
        if not touched.any():
            if PER_DAY_LOG: log("  Limit not filled after disrespect → skip trade")
            continue
        entry_time = fut.index[np.argmax(touched.values)]
        entry_price = float(limit_price)

        if PER_DAY_LOG:
            log(f"  Limit entry planned @ {entry_price:.2f} (time {entry_time})")

        (is_win, sl, tp, entry_price, et, sl_t, tp_t, rr, exit_price,
         risk_raw, risk_floor, risk_used, exit_reason) = simulate_trade(
            m5, swept_side, a_low, a_high, entry_time, entry_price, ifvg_time, dis_time, tp_mode, fixed_r_mult
        )

        # Skip filtered trades
        if exit_reason in (
            "FilteredFixedRR_BuyTPNotBelowAsiaHigh",
            "FilteredFixedRR_SellTPNotAboveAsiaLow",
            "FilteredAsiaOppositeBelowMinR",
            "NoSubWindowForSL",
        ):
            if PER_DAY_LOG:
                log(f"  Filter: {exit_reason} → trade ignored")
            continue

        # Record stats
        trades_taken += 1
        if rr == rr:
            rr_values.append(rr)

        if is_win:
            wins += 1; cur_win += 1; max_win = max(max_win, cur_win); cur_los = 0
            exit_time = tp_t
            if PER_DAY_LOG: log(f"  RESULT: WIN | entry={entry_price:.2f} SL={sl:.2f} TP={tp:.2f} TP @ {tp_t} | R={rr:.2f}")
            if rr == rr and rr > best_rr_val:
                best_rr_val = rr
                best_trade = DayResult(
                    date=day_ts, swept_side=swept_side, asia_low=a_low, asia_high=a_high,
                    poi_type=poi_type, poi_zone=poi_zone, tap_time=tap_time,
                    ifvg_zone=(ifvg.zone_low, ifvg.zone_high), fvg_time=ifvg_time,
                    disrespect_time=dis_time, entry_time=entry_time, is_win=is_win,
                    sl=sl, tp=tp, entry_price=entry_price, exit_time=tp_t, exit_price=exit_price, rr=rr
                )
        else:
            losses += 1; cur_los += 1; max_los = max(max_los, cur_los); cur_win = 0
            exit_time = sl_t
            if PER_DAY_LOG: log(f"  RESULT: LOSS | entry={entry_price:.2f} SL={sl:.2f} TP={tp:.2f} SL @ {sl_t} | R={rr:.2f}")

        trade = DayResult(
            date=day_ts, swept_side=swept_side, asia_low=a_low, asia_high=a_high,
            poi_type=poi_type, poi_zone=poi_zone, tap_time=tap_time,
            ifvg_zone=(ifvg.zone_low, ifvg.zone_high), fvg_time=ifvg_time,
            disrespect_time=dis_time, entry_time=entry_time, is_win=is_win,
            sl=sl, tp=tp, entry_price=entry_price, exit_time=exit_time, exit_price=exit_price, rr=rr
        )
        all_trades.append(trade)

        # Row for CSV
        trade_rows.append({
            "trade_index": len(trade_rows)+1,
            "symbol": SYMBOL,
            "tp_mode": tp_mode,
            "fixed_r_mult": fixed_r_mult if tp_mode == "fixed_rr" else np.nan,
            "date": str(day_ts.date()),
            "asia_low": a_low, "asia_high": a_high, "asia_mid": a_mid,
            "london_start": str(london_start), "london_end": str(london_end),
            "sweep_side": swept_side, "sweep_time": str(sweep_time),
            "poi_type": poi_type,
            "poi_zone_low": poi_zone[0], "poi_zone_high": poi_zone[1],
            "tap_time": str(tap_time),
            "ifvg_tf": "M5",
            "ifvg_is_bearish": bool(ifvg.is_bearish),
            "ifvg_zone_low": ifvg.zone_low, "ifvg_zone_high": ifvg.zone_high,
            "ifvg_time": str(ifvg_time),
            "ifvg_disrespect_time": str(dis_time),
            "entry_time": str(entry_time), "entry_price": entry_price,
            "entry_type": "limit",
            "limit_price": limit_price,
            "sl": sl, "tp": tp,
            "risk_raw": risk_raw, "risk_floor": risk_floor, "risk_used": risk_used,
            "exit_time": str(exit_time) if exit_time is not None else "",
            "exit_price": exit_price if exit_price is not None else np.nan,
            "is_win": bool(is_win),
            "rr": rr,
            "prev_day_high": prev_hi if prev_hi is not None else np.nan,
            "prev_day_low":  prev_lo if prev_lo is not None else np.nan,
        })

    # --- Stats summary ---
    stats = {}
    stats["trades"] = trades_taken
    stats["wins"] = wins
    stats["losses"] = losses
    stats["winrate_pct"] = (100.0 * wins / trades_taken) if trades_taken > 0 else float('nan')
    rr_clean = [r for r in rr_values if r == r]
    stats["avg_R"] = float(np.mean(rr_clean)) if rr_clean else float('nan')
    stats["best_R"] = float(np.max(rr_clean)) if rr_clean else float('nan')
    stats["worst_R"] = float(np.min(rr_clean)) if rr_clean else float('nan')
    stats["cum_R"] = float(np.sum(rr_clean)) if rr_clean else 0.0
    pos_sum = float(np.sum([r for r in rr_clean if r > 0])) if rr_clean else 0.0
    neg_sum = float(np.sum([-r for r in rr_clean if r < 0])) if rr_clean else 0.0
    stats["profit_factor"] = (pos_sum / neg_sum) if neg_sum > 0 else float('inf')

    # --- Equity curves (by entry_time order) ---
    all_trades_sorted = sorted(all_trades, key=lambda t: t.entry_time)
    eq_dates = [t.entry_time for t in all_trades_sorted]
    rrs = np.array([t.rr for t in all_trades_sorted], dtype=float)
    cum_r = np.cumsum(rrs) if len(rrs) else np.array([])
    equity = np.cumprod(1.0 + RISK_PER_TRADE * rrs) if len(rrs) else np.array([])

    # --- Export CSV ---
    mode_tag = tp_mode.replace("_", "")
    trades_csv_path = os.path.join(OUTPUT_DIR, f"{SYMBOL}_trades_{mode_tag}.csv")
    if trade_rows:
        pd.DataFrame(trade_rows).to_csv(trades_csv_path, index=False)
        print(f"[{tp_mode}] Saved trades CSV → {trades_csv_path}", flush=True)
    else:
        print(f"[{tp_mode}] No trades to export.", flush=True)

    # --- Plots: recent 7 + biggest win ---
    recent_plots = []
    if all_trades_sorted:
        recent = all_trades_sorted[-7:]
        print(f"[{tp_mode}] Plotting {len(recent)} most recent trades...", flush=True)
        for idx, tr in enumerate(recent, 1):
            fname = f"{SYMBOL}_{tr.date.date()}_{tr.poi_type}_{'WIN' if tr.is_win else 'LOSS'}_R{tr.rr:.2f}_recent{idx}_{mode_tag}.png"
            out_png = os.path.join(PLOT_DIR, fname)
            try:
                plot_trade(m5, tr, out_png)
                recent_plots.append(out_png)
                print(f"  Saved → {out_png}", flush=True)
            except Exception as e:
                print(f"  Plot error for recent trade #{idx}: {e}", flush=True)

    best_plot = None
    if best_trade is not None:
        out_best = os.path.join(PLOT_DIR, f"{SYMBOL}_{best_trade.date.date()}_BIGGEST_WIN_R{best_trade.rr:.2f}_{mode_tag}.png")
        try:
            plot_trade(m5, best_trade, out_best)
            best_plot = out_best
            print(f"[{tp_mode}] Saved biggest win plot → {out_best}", flush=True)
        except Exception as e:
            print(f"[{tp_mode}] Plot error (best): {e}", flush=True)

    # --- Equity plot ---
    if len(eq_dates) > 0:
        eq_title = f"{SYMBOL} Equity - {tp_mode} ({len(rrs)} trades, full sample)"
        eq_path = os.path.join(PLOT_DIR, f"{SYMBOL}_equity_{mode_tag}.png")
        try:
            plot_equity(eq_dates, cum_r, equity, eq_path, eq_title)
            print(f"[{tp_mode}] Saved equity curve → {eq_path}", flush=True)
        except Exception as e:
            print(f"[{tp_mode}] Equity plot error: {e}", flush=True)

    log(f"=== End backtest mode: {tp_mode} ===")
    return {
        "trades": all_trades_sorted,
        "stats": stats,
        "equity_dates": eq_dates,
        "cum_r": cum_r,
        "equity": equity,
        "trades_csv_path": trades_csv_path,
        "recent_plots": recent_plots,
        "best_plot": best_plot,
    }

# =========================
# MAIN
# =========================

def process():
    log("=== Start ===")
    log(f"IFVG rules → must_form_after_tap={IFVG_MUST_FORM_AFTER_TAP}, inside_POI={IFVG_MUST_BE_INSIDE_POI}, allow_partial_overlap={IFVG_ALLOW_PARTIAL_OVERLAP}, side_of_mid={IFVG_MUST_BE_ON_CORRECT_SIDE_OF_ASIA_MID}")
    log(f"Filters → fixedRR_side={FILTER_FIXED_RR_TP_MUST_RESIDE_ON_ALLOWED_SIDE_OF_ASIA}, asiaOpp_minR={'ON' if FILTER_ASIA_OPPOSITE_MIN_R else 'OFF'}({ASIA_OPPOSITE_MIN_R if FILTER_ASIA_OPPOSITE_MIN_R else ''})")
    log(f"Fixed RR mult: {FIXED_R_MULT}R | Risk/trade for equity: {RISK_PER_TRADE:.2%}")
    log("Sample = FULL M5 CSV")

    m5 = read_csv_any(M5_FILE)
    h1 = read_csv_any(H1_FILE)
    h4 = read_csv_any(H4_FILE)
    d1 = read_csv_any(D1_FILE)

    log("Detecting HTF FVGs (H1/H4/D1)")
    t0 = time.time()
    fvgs_h1 = detect_fvgs(h1, "H1")
    fvgs_h4 = detect_fvgs(h4, "H4")
    fvgs_d1 = detect_fvgs(d1, "D1")
    all_fvgs = fvgs_h1 + fvgs_h4 + fvgs_d1
    log(f"FVG counts → H1: {len(fvgs_h1)}, H4: {len(fvgs_h4)}, D1: {len(fvgs_d1)} (total {len(all_fvgs)}) in {time.time()-t0:.2f}s")

    precompute_first_touches(h1, fvgs_h1)
    precompute_first_touches(h4, fvgs_h4)
    precompute_first_touches(d1, fvgs_d1)

    htf_fvgs_sorted = sorted(all_fvgs, key=lambda f: f.created_at)

    # Run BOTH modes
    result_asia = run_backtest_for_mode("asia_opposite", m5, htf_fvgs_sorted, fixed_r_mult=FIXED_R_MULT)
    result_fixed = run_backtest_for_mode("fixed_rr",     m5, htf_fvgs_sorted, fixed_r_mult=FIXED_R_MULT)

    # Print separate stats
    def print_stats(tag: str, stats: Dict):
        print(f"\n=== {SYMBOL} Stats — {tag} (Full sample) ===", flush=True)
        print(f"Trades: {stats['trades']}", flush=True)
        print(f"Wins: {stats['wins']} | Losses: {stats['losses']}", flush=True)
        print(f"Win rate: {stats['winrate_pct']:.2f}%"
              if stats['trades']>0 else "Win rate: n/a", flush=True)
        print(f"Average R: {stats['avg_R']:.2f} | Best R: {stats['best_R']:.2f} | Worst R: {stats['worst_R']:.2f}", flush=True)
        pf = stats['profit_factor']
        print(f"Cumulative R: {stats['cum_R']:.2f} | Profit Factor: {pf:.2f}" if np.isfinite(pf) else f"Cumulative R: {stats['cum_R']:.2f} | Profit Factor: inf", flush=True)

    print_stats("Asia-Opposite (min 1.5R to edge)", result_asia["stats"])
    print_stats(f"Fixed-RR ({FIXED_R_MULT}R, TP side constraint)", result_fixed["stats"])

    # Combined equity overlay
    if result_asia["equity_dates"] and result_fixed["equity_dates"]:
        overlay_path = os.path.join(PLOT_DIR, f"{SYMBOL}_equity_overlay.png")
        try:
            plot_equity_overlay(
                result_asia["equity_dates"], result_asia["equity"], "Asia-Opposite",
                result_fixed["equity_dates"], result_fixed["equity"], f"Fixed-RR ({FIXED_R_MULT}R)",
                overlay_path, f"{SYMBOL} Equity Overlay (Full sample)"
            )
            print(f"Saved combined equity overlay → {overlay_path}", flush=True)
        except Exception as e:
            print(f"Overlay equity plot error: {e}", flush=True)

    log("=== Done ===")


if __name__ == "__main__":
    try:
        process()
    except Exception as e:
        log(f"FATAL: {type(e).__name__}: {e}")
        raise
