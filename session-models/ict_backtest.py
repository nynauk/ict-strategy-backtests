"""
ICT Session Sweep Backtest — EURUSD M5
========================================
Strategy:
  London Kill Zone (07:00-10:00 BST):
    1. Mark Asia range (01:00-07:00 BST).
    2. During the kill zone, wait for Asia High or Low to be swept.
    3. After the sweep, price must tap a higher-timeframe PD array (1H/4H/1D FVG
       OR Previous Day/Week/Month High or Low) that lies OUTSIDE the Asia range
       (above Asia High for bearish, below Asia Low for bullish).
    4. A M5 FVG formed after the sweep gets fully passed through → creates an iFVG.
    5. Enter on the retest of the iFVG (price touches bottom of bearish iFVG from
       below, or top of bullish iFVG from above).
       SL = highest High (sell) or lowest Low (buy) of candles between the
            disrespect candle and the entry candle, + a 1-pip buffer.
       TP = opposite session extreme (Asia Low for sells, Asia High for buys).

  New York Kill Zone (12:00-15:00 BST):
    Same logic, but sweeps the pre-NY London range (08:00-12:00 BST).
    PD array must lie OUTSIDE the London range.
    TP = London range Low (sells) / High (buys).

P&L: Win = +2%, Loss = -1%, compounded from $10,000.
"""

import pandas as pd
import pytz
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates  as mdates
from datetime import time, timedelta
import bisect

# ── Configuration ─────────────────────────────────────────────────────────────
DATA_PATH        = r"C:\Users\Ayaan\Downloads\EURUSD_M5 (1).csv"
INSTRUMENT       = "EURUSD"
STARTING_BALANCE = 10_000
WIN_PCT          = 0.02
LOSS_PCT         = 0.01
SL_BUFFER        = 0.00010      # 1 pip for EURUSD (0.0001)

# Set to any trade index (0-based) to plot that trade after the backtest.
# Set to None to skip plotting.
PLOT_TRADE_IDX   = 0

UK_TZ = pytz.timezone("Europe/London")

# Session/KZ windows (BST)
ASIA_START       = time(1,  0)
ASIA_END         = time(7,  0)
LONDON_RANGE_END = time(12, 0)   # end of London range used for NY setup
LONDON_KZ_START  = time(7,  0)
LONDON_KZ_END    = time(10, 0)
NY_KZ_START      = time(12, 0)
NY_KZ_END        = time(15, 0)
# ──────────────────────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(
        path, sep="\t", header=0,
        names=["Time", "Open", "High", "Low", "Close", "Volume", "Ticks"],
        parse_dates=["Time"],
    )
    df["Time"] = df["Time"].dt.tz_localize("UTC").dt.tz_convert(UK_TZ)
    df.set_index("Time", inplace=True)
    df.sort_index(inplace=True)
    return df


def resample_htf(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Resample M5 OHLC to a higher timeframe."""
    df_utc = df.tz_convert("UTC")
    htf = df_utc.resample(freq, label="left", closed="left").agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
    ).dropna()
    htf.index = htf.index.tz_convert(UK_TZ)
    return htf


# ══════════════════════════════════════════════════════════════════════════════
# PRE-COMPUTATION: HTF FVGs
# ══════════════════════════════════════════════════════════════════════════════

def precompute_fvgs(htf: pd.DataFrame) -> list:
    """
    Detect every FVG in a HTF DataFrame and record when (if ever) it was filled.
    A bullish FVG  [lo=c0.High, hi=c2.Low]  is filled when Close < lo.
    A bearish FVG  [lo=c2.High, hi=c0.Low]  is filled when Close > hi.
    Returns a list sorted by formed_at.
    """
    rows  = htf.reset_index().to_dict("records")
    n     = len(rows)
    fvgs  = []

    for i in range(n - 2):
        c0, c2 = rows[i], rows[i + 2]
        lo = hi = ftype = None

        if c2["Low"] > c0["High"]:
            lo, hi, ftype = float(c0["High"]), float(c2["Low"]), "bullish"
        elif c0["Low"] > c2["High"]:
            lo, hi, ftype = float(c2["High"]), float(c0["Low"]), "bearish"
        else:
            continue

        formed_at = c2["Time"]
        fill_time = pd.NaT

        for j in range(i + 3, n):
            cj = rows[j]
            if ftype == "bullish" and cj["Close"] < lo:
                fill_time = cj["Time"]; break
            if ftype == "bearish" and cj["Close"] > hi:
                fill_time = cj["Time"]; break

        fvgs.append({"type": ftype, "lo": lo, "hi": hi,
                     "formed_at": formed_at, "fill_time": fill_time})
    return fvgs


def active_fvgs_at(precomputed: list, query_ts, fvg_type: str,
                   outside_above: float = None, outside_below: float = None) -> list:
    """
    Return FVGs of fvg_type active at query_ts (formed before, not yet filled).

    outside_above: if set, bearish FVG's lo must be >= this value (zone is above session high).
    outside_below: if set, bullish FVG's hi must be <= this value (zone is below session low).
    """
    result = []
    for f in precomputed:
        if f["type"] != fvg_type:
            continue
        if f["formed_at"] >= query_ts:
            continue
        if not (pd.isna(f["fill_time"]) or f["fill_time"] > query_ts):
            continue
        # PD array must lie OUTSIDE the session range
        if fvg_type == "bearish" and outside_above is not None:
            if f["lo"] < outside_above:          # zone is inside/below session high
                continue
        if fvg_type == "bullish" and outside_below is not None:
            if f["hi"] > outside_below:          # zone is inside/above session low
                continue
        result.append(f)
    return result


def candle_taps_fvg(candle, fvg: dict, bias: str) -> bool:
    """
    True if the candle taps into the FVG zone from the correct direction.
    Bearish FVG (resistance above): price's High enters the zone from below.
    Bullish FVG (support below):    price's Low  enters the zone from above.
    """
    if bias == "bearish":
        return candle["High"] >= fvg["lo"] and candle["Low"] <= fvg["hi"]
    else:
        return candle["Low"] <= fvg["hi"] and candle["High"] >= fvg["lo"]


# ══════════════════════════════════════════════════════════════════════════════
# PRE-COMPUTATION: PREVIOUS DAY / WEEK / MONTH LEVELS
# ══════════════════════════════════════════════════════════════════════════════

def precompute_daily_levels(df: pd.DataFrame) -> dict:
    daily = df.resample("1D").agg({"High": "max", "Low": "min"}).dropna()
    dates, highs, lows = list(daily.index.date), list(daily["High"]), list(daily["Low"])
    return {dates[i]: (highs[i - 1], lows[i - 1]) for i in range(1, len(dates))}


def precompute_weekly_levels(df: pd.DataFrame) -> dict:
    daily  = df.resample("1D").agg({"High": "max", "Low": "min"}).dropna()
    weekly = df.resample("W-MON", label="left", closed="left").agg(
        {"High": "max", "Low": "min"}
    ).dropna()
    wdates, wh, wl = list(weekly.index.date), list(weekly["High"]), list(weekly["Low"])
    result = {}
    for date in daily.index.date:
        idx = bisect.bisect_right(wdates, date) - 1
        result[date] = (wh[idx - 1], wl[idx - 1]) if idx > 0 else (None, None)
    return result


def precompute_monthly_levels(df: pd.DataFrame) -> dict:
    daily   = df.resample("1D").agg({"High": "max", "Low": "min"}).dropna()
    monthly = df.resample("MS").agg({"High": "max", "Low": "min"}).dropna()
    mdates, mh, ml = list(monthly.index.date), list(monthly["High"]), list(monthly["Low"])
    result = {}
    for date in daily.index.date:
        idx = bisect.bisect_right(mdates, date) - 1
        result[date] = (mh[idx - 1], ml[idx - 1]) if idx > 0 else (None, None)
    return result


# ══════════════════════════════════════════════════════════════════════════════
# M5 iFVG TRACKER  (per kill zone instance)
# ══════════════════════════════════════════════════════════════════════════════

IFVG_MAX_CANDLES = 5   # disrespect must occur within this many candles of FVG formation


class FvgTracker:
    """
    Incrementally tracks M5 FVGs formed after a session sweep.
    When a FVG is fully passed through it becomes an iFVG.

    Rule: the disrespect (fill) must occur within IFVG_MAX_CANDLES candles of
    the FVG being formed; otherwise the FVG is discarded as invalid.

    Records the disrespect candle (the one that created the iFVG) for SL calculation.
    """
    def __init__(self, bias: str):
        self.bias    = bias     # "bearish" or "bullish"
        self.buf     = []       # (ts, candle) rolling window of last 3
        self.pending = []       # FVGs not yet filled, each has a candles_left counter
        self.ifvgs   = []       # completed iFVGs with disrespect_ts recorded

    def feed(self, ts, candle):
        self.buf.append((ts, candle))

        # Detect new FVG from last 3 candles
        if len(self.buf) >= 3:
            _, c0 = self.buf[-3]
            _, c2 = self.buf[-1]
            if self.bias == "bearish" and c2["Low"] > c0["High"]:
                # Bullish FVG forms during the upward sweep — will flip to bearish iFVG
                self.pending.append({
                    "type":         "bullish",
                    "lo":           float(c0["High"]),
                    "hi":           float(c2["Low"]),
                    "candles_left": IFVG_MAX_CANDLES,   # countdown to expiry
                })
            elif self.bias == "bullish" and c0["Low"] > c2["High"]:
                # Bearish FVG forms during the downward sweep — will flip to bullish iFVG
                self.pending.append({
                    "type":         "bearish",
                    "lo":           float(c2["High"]),
                    "hi":           float(c0["Low"]),
                    "candles_left": IFVG_MAX_CANDLES,
                })

        # Check if any pending FVG was just fully passed through (disrespected)
        still_pending = []
        for fvg in self.pending:
            if fvg["type"] == "bullish" and candle["Close"] < fvg["lo"]:
                # Bullish FVG fully filled within window → bearish iFVG
                self.ifvgs.append({
                    "lo":              fvg["lo"],
                    "hi":              fvg["hi"],
                    "pol":             "bearish",
                    "disrespect_ts":   ts,
                    "disrespect_high": candle["High"],
                })
            elif fvg["type"] == "bearish" and candle["Close"] > fvg["hi"]:
                # Bearish FVG fully filled within window → bullish iFVG
                self.ifvgs.append({
                    "lo":             fvg["lo"],
                    "hi":             fvg["hi"],
                    "pol":            "bullish",
                    "disrespect_ts":  ts,
                    "disrespect_low": candle["Low"],
                })
            else:
                # Not yet disrespected — decrement countdown and keep if still valid
                fvg["candles_left"] -= 1
                if fvg["candles_left"] >= 0:
                    still_pending.append(fvg)
                # else: window expired → silently drop this FVG
        self.pending = still_pending

    def get_retested_ifvg(self, candle):
        """
        Return the first iFVG retested by this candle, or None.
        Bearish iFVG retest: price comes UP and touches iFVG_lo from below
          (candle High enters zone, Close stays below zone top).
        Bullish iFVG retest: price comes DOWN and touches iFVG_hi from above
          (candle Low enters zone, Close stays above zone bottom).
        """
        for ifvg in self.ifvgs:
            if ifvg["pol"] == "bearish":
                if candle["High"] >= ifvg["lo"] and candle["Close"] < ifvg["hi"]:
                    return ifvg
            else:
                if candle["Low"] <= ifvg["hi"] and candle["Close"] > ifvg["lo"]:
                    return ifvg
        return None


# ══════════════════════════════════════════════════════════════════════════════
# TRADE RESOLUTION
# ══════════════════════════════════════════════════════════════════════════════

def resolve_trade(df_all, entry_ts, entry, sl, tp, direction):
    """
    Scan M5 candles from entry_ts onward (up to 5 calendar days) for TP or SL.
    Conservative: if both hit on same candle, SL wins.
    Returns 'W' or 'L'.
    """
    after   = df_all[df_all.index >= entry_ts]
    if after.empty:
        return "L"
    cutoff  = after.index[0] + pd.Timedelta(days=5)

    for ts, row in after.iterrows():
        if ts > cutoff:
            break
        if direction == "SELL":
            sl_hit = row["High"] >= sl
            tp_hit = row["Low"]  <= tp
        else:
            sl_hit = row["Low"]  <= sl
            tp_hit = row["High"] >= tp

        if sl_hit:
            return "L"       # SL first (conservative)
        if tp_hit:
            return "W"

    return "L"               # no resolution → conservative loss


# ══════════════════════════════════════════════════════════════════════════════
# KILL ZONE LOGIC
# ══════════════════════════════════════════════════════════════════════════════

def run_kill_zone(
    df_all, day_candles,
    kz_label, range_high, range_low,
    kz_start, kz_end,
    fvgs_1h, fvgs_4h, fvgs_1d,
    pdh, pdl, pwh, pwl, pmh, pml,
):
    kz = day_candles[
        (day_candles.index.time >= kz_start) &
        (day_candles.index.time <  kz_end)
    ]
    if kz.empty:
        return None

    # ── 1. Sweep detection ────────────────────────────────────────────────
    bias = sweep_idx = None
    for ts, row in kz.iterrows():
        bh = row["High"] > range_high
        bl = row["Low"]  < range_low
        if bh and bl:
            return None        # both swept on same candle → skip
        if bh:
            bias, sweep_idx = "bearish", ts; break
        if bl:
            bias, sweep_idx = "bullish", ts; break

    if bias is None:
        return None

    post_sweep = kz[kz.index > sweep_idx]
    if post_sweep.empty:
        return None

    # ── 2. HTF PD array tap (must be OUTSIDE the session range) ──────────
    # For bearish: PD array must be above range_high (outside_above=range_high)
    # For bullish: PD array must be below range_low  (outside_below=range_low)
    if bias == "bearish":
        htf_fvgs = (
            active_fvgs_at(fvgs_1h, sweep_idx, "bearish", outside_above=range_high) +
            active_fvgs_at(fvgs_4h, sweep_idx, "bearish", outside_above=range_high) +
            active_fvgs_at(fvgs_1d, sweep_idx, "bearish", outside_above=range_high)
        )
        # Previous highs that are at or above range_high
        prev_levels = [l for l in [pdh, pwh, pmh]
                       if l is not None and l >= range_high]
    else:
        htf_fvgs = (
            active_fvgs_at(fvgs_1h, sweep_idx, "bullish", outside_below=range_low) +
            active_fvgs_at(fvgs_4h, sweep_idx, "bullish", outside_below=range_low) +
            active_fvgs_at(fvgs_1d, sweep_idx, "bullish", outside_below=range_low)
        )
        prev_levels = [l for l in [pdl, pwl, pml]
                       if l is not None and l <= range_low]

    # Tag each HTF FVG with its timeframe so the chart can label it
    htf_fvgs_tagged = (
        [(f, "1H") for f in active_fvgs_at(fvgs_1h, sweep_idx,
            "bearish" if bias == "bearish" else "bullish",
            outside_above=range_high if bias == "bearish" else None,
            outside_below=range_low  if bias == "bullish" else None)] +
        [(f, "4H") for f in active_fvgs_at(fvgs_4h, sweep_idx,
            "bearish" if bias == "bearish" else "bullish",
            outside_above=range_high if bias == "bearish" else None,
            outside_below=range_low  if bias == "bullish" else None)] +
        [(f, "1D") for f in active_fvgs_at(fvgs_1d, sweep_idx,
            "bearish" if bias == "bearish" else "bullish",
            outside_above=range_high if bias == "bearish" else None,
            outside_below=range_low  if bias == "bullish" else None)]
    )

    pd_tapped = pd_tap_idx = None
    matched_pd_fvg = None
    matched_pd_tf  = None
    matched_pd_lvl = None

    for ts, row in post_sweep.iterrows():
        for fvg, tf in htf_fvgs_tagged:
            if candle_taps_fvg(row, fvg, bias):
                pd_tapped, pd_tap_idx = True, ts
                matched_pd_fvg = fvg
                matched_pd_tf  = tf
                break
        if pd_tapped:
            break
        for lv in prev_levels:
            if (bias == "bearish" and row["High"] >= lv) or \
               (bias == "bullish" and row["Low"]  <= lv):
                pd_tapped, pd_tap_idx = True, ts
                matched_pd_lvl = lv
                break
        if pd_tapped:
            break

    if not pd_tapped:
        return None

    # ── 3 & 4. iFVG formation + retest ────────────────────────────────────
    tracker = FvgTracker(bias=bias)

    # Pre-feed candles from sweep to PD tap to build FVG context
    pre_feed = day_candles[
        (day_candles.index >= sweep_idx) &
        (day_candles.index <= pd_tap_idx)
    ]
    for ts, row in pre_feed.iterrows():
        tracker.feed(ts, row)

    # Look for iFVG retest within the remaining kill zone window
    post_tap       = kz[kz.index > pd_tap_idx]
    active_ifvg    = None
    entry_ts       = None

    for ts, row in post_tap.iterrows():
        tracker.feed(ts, row)
        retested = tracker.get_retested_ifvg(row)
        if retested is not None:
            active_ifvg = retested
            entry_ts    = ts
            break

    if active_ifvg is None:
        return None

    # ── 5. Build trade levels ─────────────────────────────────────────────
    # Entry: bottom of bearish iFVG (for sell), top of bullish iFVG (for buy)
    # SL:    highest High  (sell) or lowest Low (buy) of ALL candles between
    #        the disrespect candle and the entry candle, + SL_BUFFER
    # TP:    opposite session extreme

    disrespect_ts = active_ifvg["disrespect_ts"]
    candles_in_range = df_all[
        (df_all.index >= disrespect_ts) &
        (df_all.index <= entry_ts)
    ]

    if bias == "bearish":
        entry     = active_ifvg["lo"]               # price retests from below → enter at zone bottom
        sl        = candles_in_range["High"].max() + SL_BUFFER   # highest high between disrespect & entry
        tp        = range_low
        direction = "SELL"
    else:
        entry     = active_ifvg["hi"]               # price retests from above → enter at zone top
        sl        = candles_in_range["Low"].min()  - SL_BUFFER   # lowest low between disrespect & entry
        tp        = range_high
        direction = "BUY"

    # Sanity checks
    if direction == "SELL":
        if tp >= entry:
            return None         # TP above entry for a sell → invalid
        if sl <= entry:
            return None         # SL must be above entry for a sell
    else:
        if tp <= entry:
            return None
        if sl >= entry:
            return None

    # ── 6. Resolve trade ─────────────────────────────────────────────────
    result = resolve_trade(df_all, entry_ts, entry, sl, tp, direction)

    return {
        "kz":            kz_label,
        "direction":     direction,
        "entry":         round(entry, 5),
        "sl":            round(sl,    5),
        "tp":            round(tp,    5),
        "result":        result,
        # ── metadata for charting ──
        "_range_high":    range_high,
        "_range_low":     range_low,
        "_kz_start":      kz_start,
        "_kz_end":        kz_end,
        "_sweep_ts":      sweep_idx,
        "_pd_tap_ts":     pd_tap_idx,
        "_disrespect_ts": active_ifvg["disrespect_ts"],
        "_entry_ts":      entry_ts,
        "_ifvg_lo":       active_ifvg["lo"],
        "_ifvg_hi":       active_ifvg["hi"],
        # HTF PD array that was tapped
        "_pd_fvg_lo":     matched_pd_fvg["lo"]        if matched_pd_fvg else None,
        "_pd_fvg_hi":     matched_pd_fvg["hi"]        if matched_pd_fvg else None,
        "_pd_fvg_tf":     matched_pd_tf,
        "_pd_fvg_formed": matched_pd_fvg["formed_at"] if matched_pd_fvg else None,
        "_pd_level":      matched_pd_lvl,
    }


# ══════════════════════════════════════════════════════════════════════════════
# MAIN BACKTEST LOOP
# ══════════════════════════════════════════════════════════════════════════════

def run_backtest(df: pd.DataFrame):
    print("  Pre-computing HTF FVGs (1H / 4H / 1D)...")
    fvgs_1h = precompute_fvgs(resample_htf(df, "1h"))
    fvgs_4h = precompute_fvgs(resample_htf(df, "4h"))
    fvgs_1d = precompute_fvgs(resample_htf(df, "1D"))
    print(f"  FVGs found -- 1H: {len(fvgs_1h)}, 4H: {len(fvgs_4h)}, 1D: {len(fvgs_1d)}")

    print("  Pre-computing daily / weekly / monthly levels...")
    daily_lvls   = precompute_daily_levels(df)
    weekly_lvls  = precompute_weekly_levels(df)
    monthly_lvls = precompute_monthly_levels(df)

    trades  = []
    balance = STARTING_BALANCE
    dates   = sorted(set(df.index.date))
    total   = len(dates)

    for i, date in enumerate(dates):
        if i % 200 == 0:
            print(f"  Processing day {i}/{total}...")

        day_df = df[df.index.date == date]
        pdh, pdl = daily_lvls.get(date,   (None, None))
        pwh, pwl = weekly_lvls.get(date,  (None, None))
        pmh, pml = monthly_lvls.get(date, (None, None))

        kwargs = dict(
            df_all=df, day_candles=day_df,
            fvgs_1h=fvgs_1h, fvgs_4h=fvgs_4h, fvgs_1d=fvgs_1d,
            pdh=pdh, pdl=pdl, pwh=pwh, pwl=pwl, pmh=pmh, pml=pml,
        )

        # ── London Kill Zone ──────────────────────────────────────────────
        asia = day_df[
            (day_df.index.time >= ASIA_START) &
            (day_df.index.time <  ASIA_END)
        ]
        if not asia.empty:
            ah, al = asia["High"].max(), asia["Low"].min()
            if ah != al:
                trade = run_kill_zone(
                    **kwargs,
                    kz_label="London", range_high=ah, range_low=al,
                    kz_start=LONDON_KZ_START, kz_end=LONDON_KZ_END,
                )
                if trade:
                    balance = _log(trades, trade, date, balance)

        # ── New York Kill Zone ────────────────────────────────────────────
        lon_range = day_df[
            (day_df.index.time >= ASIA_END) &
            (day_df.index.time <  LONDON_RANGE_END)
        ]
        if not lon_range.empty:
            lh, ll = lon_range["High"].max(), lon_range["Low"].min()
            if lh != ll:
                trade = run_kill_zone(
                    **kwargs,
                    kz_label="NY", range_high=lh, range_low=ll,
                    kz_start=NY_KZ_START, kz_end=NY_KZ_END,
                )
                if trade:
                    balance = _log(trades, trade, date, balance)

    return trades, balance


def _log(trades, trade, date, balance):
    pnl_pct = WIN_PCT if trade["result"] == "W" else -LOSS_PCT
    new_bal = balance + balance * pnl_pct
    trades.append({
        "Date":      str(date),
        "KZ":        trade["kz"],
        "Direction": trade["direction"],
        "Entry":     trade["entry"],
        "SL":        trade["sl"],
        "TP":        trade["tp"],
        "Result":    trade["result"],
        "P&L %":     f"{pnl_pct * 100:+.1f}%",
        "Balance":   round(new_bal, 2),
        # charting metadata (underscore-prefixed, not printed in table)
        "_range_high":    trade["_range_high"],
        "_range_low":     trade["_range_low"],
        "_kz_start":      trade["_kz_start"],
        "_kz_end":        trade["_kz_end"],
        "_sweep_ts":      trade["_sweep_ts"],
        "_pd_tap_ts":     trade["_pd_tap_ts"],
        "_disrespect_ts": trade["_disrespect_ts"],
        "_entry_ts":      trade["_entry_ts"],
        "_ifvg_lo":       trade["_ifvg_lo"],
        "_ifvg_hi":       trade["_ifvg_hi"],
        "_pd_fvg_lo":     trade["_pd_fvg_lo"],
        "_pd_fvg_hi":     trade["_pd_fvg_hi"],
        "_pd_fvg_tf":     trade["_pd_fvg_tf"],
        "_pd_fvg_formed": trade["_pd_fvg_formed"],
        "_pd_level":      trade["_pd_level"],
    })
    return new_bal


# ══════════════════════════════════════════════════════════════════════════════
# STATISTICS & OUTPUT
# ══════════════════════════════════════════════════════════════════════════════

def compute_stats(trades, final_balance):
    total  = len(trades)
    wins   = sum(1 for t in trades if t["Result"] == "W")
    losses = total - wins
    win_rate = (wins / total * 100) if total else 0

    gw = gl = 0.0
    for t in trades:
        pct = float(t["P&L %"].replace("%", "")) / 100
        bal = t["Balance"] / (1 + pct)
        if pct > 0: gw += bal * WIN_PCT
        else:       gl += bal * LOSS_PCT

    pf        = (gw / gl) if gl else float("inf")
    total_pct = (final_balance / STARTING_BALANCE - 1) * 100

    rb = [1 if t["Result"] == "W" else 0 for t in trades]
    mw = ml = cw = cl = 0
    for r in rb:
        if r:  cw += 1; cl = 0
        else:  cl += 1; cw = 0
        mw = max(mw, cw); ml = max(ml, cl)

    return {
        "Total Trades":        total,
        "Wins":                wins,
        "Losses":              losses,
        "Win Rate":            f"{win_rate:.1f}%",
        "Profit Factor":       f"{pf:.2f}",
        "Total % Gained":      f"{total_pct:+.2f}%",
        "Final Balance":       f"${final_balance:,.2f}",
        "Longest Win Streak":  mw,
        "Longest Loss Streak": ml,
    }


def print_results(stats, trades):
    print("\n" + "=" * 70)
    print(f"  ICT SESSION SWEEP BACKTEST — {INSTRUMENT} M5")
    print("=" * 70)
    for k, v in stats.items():
        print(f"  {k:<26} {v}")
    print("=" * 70)
    hdr = (f"{'Date':<12} {'KZ':<7} {'Dir':<5} "
           f"{'Entry':>8} {'SL':>8} {'TP':>8} "
           f"{'Res':<4} {'P&L':>6}  {'Balance':>10}")
    print(f"\n{hdr}")
    print("-" * 78)
    for t in trades:
        print(
            f"{t['Date']:<12} {t['KZ']:<7} {t['Direction']:<5} "
            f"{t['Entry']:>8.5f} {t['SL']:>8.5f} {t['TP']:>8.5f} "
            f"{t['Result']:<4} {t['P&L %']:>6}  ${t['Balance']:>9,.2f}"
        )
    print("-" * 78)


def plot_equity_curve(trades):
    if not trades:
        print("No trades to plot.")
        return
    x = list(range(len(trades)))
    y = [t["Balance"] for t in trades]

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(x, y, color="#1f77b4", linewidth=1.8, label="Account Balance")
    ax.axhline(STARTING_BALANCE, color="grey", linestyle="--", linewidth=0.8,
               label=f"Start ${STARTING_BALANCE:,}")
    for i, t in enumerate(trades):
        c = "#2ca02c" if t["Result"] == "W" else "#d62728"
        ax.axvline(i, color=c, alpha=0.18, linewidth=1)

    ax.set_title(f"ICT Session Sweep — Equity Curve ({INSTRUMENT} M5)",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Trade #")
    ax.set_ylabel("Account Balance (USD)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# TRADE CHART
# ══════════════════════════════════════════════════════════════════════════════

def _draw_candles_bw(ax, df, bar_minutes=5):
    """ICT-style black/white candlesticks: bullish = white body, bearish = black body."""
    bar_w = timedelta(minutes=bar_minutes)
    for ts, row in df.iterrows():
        bull    = row["Close"] >= row["Open"]
        fc      = "white"   if bull else "#1a1a1a"
        ec      = "#1a1a1a"
        # Wick
        ax.plot([ts, ts], [row["Low"], row["High"]],
                color="#1a1a1a", linewidth=0.6, zorder=7)
        # Body
        body_lo = min(row["Open"], row["Close"])
        body_hi = max(row["Open"], row["Close"])
        ax.bar(ts, max(body_hi - body_lo, 0.000005),
               bottom=body_lo, width=bar_w,
               color=fc, edgecolor=ec, linewidth=0.5, zorder=8)


def plot_trade(df, trades, trade_idx=0):
    """
    Plot one trade in ICT reference style:
      - Black/white M5 candlesticks
      - Session vertical bands: NY PM (prev) / Asia / London / NY AM
      - HTF PD array FVG drawn as a labelled horizontal box
      - Session range (Asia or London) as dotted horizontal lines
      - iFVG zone as a shaded horizontal box
      - Entry, SL, TP horizontal lines
      - Labelled event markers: sweep, PD tap, disrespect candle, entry
    """
    if not trades or trade_idx >= len(trades):
        print("Trade index out of range.")
        return

    t = trades[trade_idx]
    kz_label      = t["KZ"]
    direction     = t["Direction"]
    entry_price   = t["Entry"]
    sl_price      = t["SL"]
    tp_price      = t["TP"]
    result        = t["Result"]
    range_high    = t["_range_high"]
    range_low     = t["_range_low"]
    kz_start_t    = t["_kz_start"]
    kz_end_t      = t["_kz_end"]
    sweep_ts      = t["_sweep_ts"]
    pd_tap_ts     = t["_pd_tap_ts"]
    disrespect_ts = t["_disrespect_ts"]
    entry_ts      = t["_entry_ts"]
    ifvg_lo       = t["_ifvg_lo"]
    ifvg_hi       = t["_ifvg_hi"]
    pd_fvg_lo     = t.get("_pd_fvg_lo")
    pd_fvg_hi     = t.get("_pd_fvg_hi")
    pd_fvg_tf     = t.get("_pd_fvg_tf") or "HTF"
    pd_fvg_formed = t.get("_pd_fvg_formed")
    pd_level      = t.get("_pd_level")

    day = entry_ts.date()

    # ── Session anchor timestamps for this day ────────────────────────────
    def ts_at(h, m=0):
        return entry_ts.replace(hour=h, minute=m, second=0, microsecond=0)

    asia_open  = ts_at(ASIA_START.hour)
    asia_close = ts_at(ASIA_END.hour)          # 07:00 — also London KZ start
    lon_open   = ts_at(8)                       # London session 08:00
    lon_close  = ts_at(17)
    ny_open    = ts_at(NY_KZ_START.hour)        # 12:00
    kz_s       = ts_at(kz_start_t.hour, kz_start_t.minute)
    kz_e       = ts_at(kz_end_t.hour,   kz_end_t.minute)

    # ── View window: from HTF FVG formation (or midnight) to entry +2 h ──
    if pd_fvg_formed is not None:
        view_start = pd_fvg_formed - timedelta(minutes=15)
    else:
        view_start = asia_open - timedelta(hours=1)
    view_end = entry_ts + timedelta(hours=2, minutes=30)

    chart_df = df[(df.index >= view_start) & (df.index <= view_end)]
    if chart_df.empty:
        print("No candles in view window.")
        return

    # ── Figure setup ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(17, 7))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    price_range = chart_df["High"].max() - chart_df["Low"].min()
    pad         = price_range * 0.08

    # ── Session vertical bands (drawn first, lowest z-order) ─────────────
    # NY PM (whatever is left of chart before Asia opens)
    if view_start < asia_open:
        ax.axvspan(view_start, min(asia_open, view_end),
                   color="#f5f5f5", alpha=1.0, zorder=0)

    # Asia session
    if asia_open < view_end:
        ax.axvspan(max(asia_open, view_start), min(asia_close, view_end),
                   color="#ebebeb", alpha=1.0, zorder=0)

    # London session
    if lon_open < view_end:
        ax.axvspan(max(lon_open, view_start), min(lon_close, view_end),
                   color="#e2e2e2", alpha=1.0, zorder=0)

    # NY AM
    if ny_open < view_end:
        ax.axvspan(max(ny_open, view_start), view_end,
                   color="#ebebeb", alpha=1.0, zorder=0)

    # ── HTF PD array box ──────────────────────────────────────────────────
    if pd_fvg_lo is not None and pd_fvg_hi is not None:
        fvg_x0 = max(view_start, pd_fvg_formed) if pd_fvg_formed else view_start
        fvg_x1 = min(pd_tap_ts + timedelta(minutes=5), view_end) if pd_tap_ts else view_end
        ax.fill_between([fvg_x0, fvg_x1], pd_fvg_lo, pd_fvg_hi,
                        color="#d0d0d0", alpha=0.55, zorder=1)
        ax.hlines([pd_fvg_lo, pd_fvg_hi], fvg_x0, fvg_x1,
                  colors="#888888", linewidths=0.8, zorder=2)
        # Label at top-right of box
        ax.text(fvg_x1, pd_fvg_hi, f" {pd_fvg_tf}",
                va="bottom", ha="left", fontsize=9,
                color="#555555", fontweight="bold", zorder=9)
    elif pd_level is not None:
        ax.axhline(pd_level, color="#888888", linewidth=0.9,
                   linestyle="-", alpha=0.7, zorder=2)
        ax.text(view_start, pd_level, " Level  ",
                va="bottom", ha="left", fontsize=8, color="#888888", zorder=9)

    # ── Session range dotted lines ────────────────────────────────────────
    sess_label = "Asian" if kz_label == "London" else "London"
    ax.axhline(range_high, color="#aaaaaa", linewidth=1.0,
               linestyle=":", zorder=3)
    ax.axhline(range_low,  color="#aaaaaa", linewidth=1.0,
               linestyle=":", zorder=3)
    # Label in centre of the range, roughly at the Asia midpoint x
    mid_x = asia_open + (asia_close - asia_open) * 0.5
    ax.text(mid_x, range_high, f" {sess_label}",
            va="bottom", ha="left", fontsize=8, color="#aaaaaa", zorder=9)

    # ── iFVG box ──────────────────────────────────────────────────────────
    ifvg_color = "#c0c0c0"
    ax.axhspan(ifvg_lo, ifvg_hi, color=ifvg_color, alpha=0.65, zorder=4)
    ax.axhline(ifvg_lo, color="#666666", linewidth=0.7, linestyle="-", zorder=5)
    ax.axhline(ifvg_hi, color="#666666", linewidth=0.7, linestyle="-", zorder=5)
    ax.text(entry_ts, ifvg_hi, " iFVG",
            va="bottom", ha="left", fontsize=7, color="#555555", zorder=9)

    # ── Entry / SL / TP ───────────────────────────────────────────────────
    ax.axhline(entry_price, color="#333333", linewidth=1.3,
               linestyle="-",  zorder=6, label=f"Entry  {entry_price:.5f}")
    ax.axhline(sl_price,    color="#777777", linewidth=0.9,
               linestyle="--", zorder=6, label=f"SL     {sl_price:.5f}")
    ax.axhline(tp_price,    color="#777777", linewidth=0.9,
               linestyle="--", zorder=6, label=f"TP     {tp_price:.5f}")

    # ── Candlesticks (drawn on top of all zones) ──────────────────────────
    _draw_candles_bw(ax, chart_df)

    # ── Session labels at top of each band ───────────────────────────────
    y_label = chart_df["High"].max() + pad * 0.3
    band_labels = []
    if view_start < asia_open:
        band_labels.append((view_start + (asia_open - view_start) / 2, "New York PM"))
    if asia_open < view_end:
        band_labels.append((asia_open  + (min(asia_close, view_end) - asia_open) / 2, "Asian"))
    if lon_open < view_end:
        band_labels.append((max(lon_open, view_start) +
                            (min(lon_close, view_end) - max(lon_open, view_start)) / 2, "London"))
    if ny_open < view_end:
        band_labels.append((ny_open + (view_end - ny_open) / 2, "New York AM"))

    for lx, lname in band_labels:
        if view_start <= lx <= view_end:
            ax.text(lx, y_label, lname, ha="center", va="bottom",
                    fontsize=8, color="#aaaaaa", zorder=9)

    # ── Event markers ─────────────────────────────────────────────────────
    y_off = price_range * 0.012
    events = [
        (sweep_ts,      "Sweep",       "^", "#999999"),
        (pd_tap_ts,     "PD Tap",      "D", "#888888"),
        (disrespect_ts, "Disrespect",  "v", "#777777"),
        (entry_ts,      "Entry",       "o", "#333333"),
    ]
    for ev_ts, ev_label, marker, colour in events:
        if ev_ts is not None and view_start <= ev_ts <= view_end:
            if ev_ts in df.index:
                y = df.loc[ev_ts, "High"] + y_off
                ax.plot(ev_ts, y, marker=marker, color=colour,
                        markersize=8, zorder=10, markeredgewidth=0.8,
                        markeredgecolor="#333333")
                ax.text(ev_ts, y + y_off * 0.6, ev_label,
                        ha="center", va="bottom", fontsize=7,
                        color="#555555", zorder=10)

    # ── Formatting ────────────────────────────────────────────────────────
    outcome_str = "WIN  ✓" if result == "W" else "LOSS  ✗"
    ax.set_title(
        f"{INSTRUMENT}  M5   |   {t['Date']}   |   {kz_label} KZ   |   "
        f"{direction}   |   {outcome_str}",
        fontsize=11, fontweight="bold", loc="left", pad=8, color="#222222"
    )

    ax.set_xlim(view_start, view_end)
    ax.set_ylim(chart_df["Low"].min() - pad, chart_df["High"].max() + pad * 1.5)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=UK_TZ))
    ax.xaxis.set_major_locator(mdates.MinuteLocator(byminute=range(0, 60, 30)))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha="center", fontsize=8)
    ax.set_xlabel(f"BST  —  {day}", fontsize=9, color="#666666")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.4f}"))
    ax.tick_params(axis="both", labelsize=8, color="#aaaaaa")

    ax.grid(True, axis="y", color="#e8e8e8", linewidth=0.5, linestyle=":")
    ax.grid(False, axis="x")
    for spine in ax.spines.values():
        spine.set_edgecolor("#cccccc")

    ax.legend(loc="lower left", fontsize=8, framealpha=0.9,
              edgecolor="#cccccc", facecolor="white")
    plt.tight_layout()
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print(f"Loading {INSTRUMENT} M5 data...")
    df = load_data(DATA_PATH)
    print(f"  {len(df):,} candles loaded  ({df.index[0].date()} to {df.index[-1].date()})")

    print("Running ICT session sweep backtest...")
    trades, final_balance = run_backtest(df)

    stats = compute_stats(trades, final_balance)
    print_results(stats, trades)
    plot_equity_curve(trades)

    # ── Optional: plot a single trade in detail ───────────────────────────
    if PLOT_TRADE_IDX is not None and trades:
        idx = PLOT_TRADE_IDX
        t   = trades[idx]
        print(f"\nPlotting trade #{idx + 1}: {t['Date']} | {t['KZ']} KZ | "
              f"{t['Direction']} | Result: {t['Result']}")
        plot_trade(df, trades, trade_idx=idx)
