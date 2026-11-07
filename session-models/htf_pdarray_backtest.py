"""
HTF Bias + PD Array Backtest — EURUSD M5
=========================================
Strategy:
  1. HTF Bias (1H): Determine directional bias using 1H market structure
     (swing highs/lows via 2-bar pivot detection). Only trade with the bias.

  2. PD Array Entry (5m chart):
     - Bullish FVG : low[i] > high[i-2]  →  zone = [high[i-2], low[i]]
     - Bearish FVG : high[i] < low[i-2]  →  zone = [high[i], low[i-2]]
     - Bullish OB  : last bearish candle before a bullish FVG impulse
     - Bearish OB  : last bullish candle before a bearish FVG impulse

  3. Trade Setup:
     BUY  – HTF bullish + price taps BULLISH PD array (discount zone below price)
             + next 5m candle closes bullish (confirmation)
     SELL – HTF bearish + price taps BEARISH PD array (premium zone above price)
             + next 5m candle closes bearish (confirmation)

  4. Risk/Reward: Fixed 2:1
     - SL  = bottom of PD zone (buys) / top of PD zone (sells)
     - TP  = entry ± 2 × |entry − SL|
     - Only trades where SL distance is between MIN_SL_PIPS and MAX_SL_PIPS

  5. Kill Zone filter: entries are only allowed during two UTC windows each day.
       London Kill Zone  : 07:00 – 10:00 UTC
       New York Kill Zone: 12:00 – 15:00 UTC
     At most ONE trade may be entered per kill zone session. Once a trade is
     taken (win or loss), that session is locked for the rest of the day.
     Trades opened inside a kill zone may run past the window to their SL/TP.

  6. No opposite-liquidity filter (removed from previous strategy).
     Trade direction is determined solely by HTF bias.

Outputs:
  - Console : summary statistics
  - htf_pdarray_equity.png : equity curve
  - htf_pdarray_trades.csv : full trade log
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional, List, Tuple

# ── Configuration ─────────────────────────────────────────────────────────────
DATA_PATH        = r"C:\Users\Ayaan\Downloads\EURUSD_M5.csv"
OUTPUT_DIR       = r"C:\Users\Ayaan\PycharmProjects\9AMBACKTEST"

HTF_TIMEFRAME    = "1h"    # Resample period for bias (use "4h" to switch to 4H bias)
SWING_N          = 2       # Bars each side for 1H pivot confirmation (no look-ahead)

FVG_MAX_AGE      = 100     # Max 5m bars a FVG remains valid before auto-expiry
OB_MAX_AGE       = 100     # Max 5m bars an OB remains valid
OB_LOOKBACK      = 8       # How many bars back to search for the OB candle from FVG

REWARD_RATIO     = 2.0     # Fixed risk-to-reward ratio
PIP              = 0.0001  # 1 pip for EURUSD (0.0001)
MIN_SL_PIPS      = 3       # Skip trades with SL tighter than this
MAX_SL_PIPS      = 50      # Skip trades with SL wider than this

STARTING_BALANCE = 10_000  # Initial equity for simulation ($)
RISK_PCT         = 0.01    # Fraction of balance risked per trade (1%)

# Kill zone windows (UTC hours, inclusive start / exclusive end)
KILL_ZONES = {
    'london'  : (7,  10),   # 07:00 – 10:00 UTC
    'newyork' : (12, 15),   # 12:00 – 15:00 UTC
}
# ──────────────────────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PDArray:
    """Represents a Fair Value Gap or Order Block on the 5m chart."""
    formed_at: int    # Bar index when this PD array was identified
    kind: str         # 'FVG' or 'OB'
    direction: str    # 'bull' (discount support) or 'bear' (premium resistance)
    bottom: float     # Lower boundary of the zone
    top: float        # Upper boundary of the zone
    mitigated: bool = False  # True when price fully trades through the zone


@dataclass
class Trade:
    entry_time: pd.Timestamp
    direction: str            # 'buy' or 'sell'
    entry: float
    sl: float
    tp: float
    pd_kind: str              # 'FVG' or 'OB'
    pd_bottom: float
    pd_top: float
    htf_bias: str             # 'bull' or 'bear' at entry time
    kill_zone: str = ''       # 'london' or 'newyork'
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    result: Optional[str] = None  # 'win', 'loss', or 'open' (forced close)
    pips: Optional[float] = None


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_data(path: str) -> pd.DataFrame:
    """
    Load M5 EURUSD CSV. Handles both comma-separated and tab-separated files,
    with or without a header row.
    """
    if not os.path.exists(path):
        print("\n" + "=" * 62)
        print("  DATA FILE NOT FOUND")
        print("=" * 62)
        print(f"  Expected path : {path}")
        print()
        print("  Please place your EURUSD M5 CSV file at the path above.")
        print("  The file should be comma-separated (or tab-separated) with")
        print("  these columns in order:")
        print("    Time, Open, High, Low, Close, Volume")
        print()
        print("  Example row:")
        print("    2024-01-02 00:00,1.09500,1.09550,1.09480,1.09510,1234")
        print()
        print("  Alternatively, update DATA_PATH at the top of this script.")
        print("=" * 62 + "\n")
        sys.exit(1)

    # Auto-detect separator from first line
    with open(path, 'r') as f:
        first_line = f.readline()
    sep = '\t' if '\t' in first_line else ','

    # Detect whether there's a header (header line won't parse as a date)
    col_names = ['Time', 'Open', 'High', 'Low', 'Close', 'Volume']
    first_token = first_line.strip().split(sep)[0]
    has_header = not _looks_like_date(first_token)
    header_arg = 0 if has_header else None

    df = pd.read_csv(
        path,
        sep=sep,
        header=header_arg,
        names=col_names if not has_header else None,
        parse_dates=['Time'] if has_header else False,
    )

    # Normalise column names
    df.columns = [str(c).strip() for c in df.columns]
    if has_header:
        df.columns = col_names[:len(df.columns)]

    df['Time'] = pd.to_datetime(df['Time'])
    df.set_index('Time', inplace=True)
    df.sort_index(inplace=True)

    for col in ['Open', 'High', 'Low', 'Close']:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' missing from data file.")

    return df[['Open', 'High', 'Low', 'Close']].astype(float)


def _looks_like_date(s: str) -> bool:
    """Heuristic: return True if the string looks like a datetime."""
    try:
        pd.to_datetime(s)
        return True
    except Exception:
        return False


# ══════════════════════════════════════════════════════════════════════════════
# HTF BIAS
# ══════════════════════════════════════════════════════════════════════════════

def compute_htf_bias(m5: pd.DataFrame,
                     timeframe: str = '1h',
                     n_swing: int = 2) -> pd.Series:
    """
    Resample M5 → HTF candles, then classify each bar as 'bull', 'bear', or
    'neutral' using swing-high/swing-low market structure:

      bull  →  last confirmed pivot low  > prior pivot low   (higher lows)
               AND last confirmed pivot high > prior pivot high (higher highs)
      bear  →  last confirmed pivot high < prior pivot high  (lower highs)
               AND last confirmed pivot low  < prior pivot low  (lower lows)
      neutral → mixed or not enough structure yet

    Look-ahead prevention: a pivot at bar i is only *confirmed* once bar i+n_swing
    has formed, so the information is never used before it's available.

    Returns a pd.Series aligned to the M5 index via forward-fill (each 5m bar
    inherits the bias of the most recently completed HTF candle).
    """
    htf = m5.resample(timeframe, label='left', closed='left').agg(
        {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'}
    ).dropna()

    n = len(htf)
    highs = htf['High'].values
    lows  = htf['Low'].values

    # Collect (confirmation_bar_index, pivot_value) pairs
    pivot_highs: List[Tuple[int, float]] = []
    pivot_lows:  List[Tuple[int, float]] = []

    for i in range(n_swing, n - n_swing):
        # Strict pivot: strictly highest / lowest in the window
        if highs[i] == highs[i - n_swing: i + n_swing + 1].max():
            pivot_highs.append((i + n_swing, highs[i]))
        if lows[i] == lows[i - n_swing: i + n_swing + 1].min():
            pivot_lows.append((i + n_swing, lows[i]))

    # Build bias array, revealing pivots only when confirmed
    htf_bias   = np.full(n, 'neutral', dtype=object)
    ph_ptr = pl_ptr = 0
    known_phs: List[float] = []
    known_pls: List[float] = []
    current_bias = 'neutral'

    for i in range(n):
        while ph_ptr < len(pivot_highs) and pivot_highs[ph_ptr][0] <= i:
            known_phs.append(pivot_highs[ph_ptr][1])
            ph_ptr += 1
        while pl_ptr < len(pivot_lows) and pivot_lows[pl_ptr][0] <= i:
            known_pls.append(pivot_lows[pl_ptr][1])
            pl_ptr += 1

        if len(known_phs) >= 2 and len(known_pls) >= 2:
            lph, pph = known_phs[-1], known_phs[-2]  # last / prev pivot high
            lpl, ppl = known_pls[-1], known_pls[-2]  # last / prev pivot low

            if lph > pph and lpl > ppl:
                current_bias = 'bull'
            elif lph < pph and lpl < ppl:
                current_bias = 'bear'
            # Mixed structure → keep previous bias (trending until broken)

        htf_bias[i] = current_bias

    htf_bias_series = pd.Series(htf_bias, index=htf.index)

    # Forward-fill onto M5 index: each 5m bar gets the bias of the last
    # completed (closed) HTF candle
    combined = (
        htf_bias_series
        .reindex(htf_bias_series.index.union(m5.index))
        .ffill()
        .reindex(m5.index)
    )
    return combined.fillna('neutral')


# ══════════════════════════════════════════════════════════════════════════════
# PD ARRAY DETECTION  (Fair Value Gaps + Order Blocks)
# ══════════════════════════════════════════════════════════════════════════════

def detect_fvg(df: pd.DataFrame, i: int) -> Optional[PDArray]:
    """
    Check whether a Fair Value Gap completed at bar i (requires bars i-2, i-1, i).

    Bullish FVG : low[i]  > high[i-2]  →  gap = [high[i-2], low[i]]
    Bearish FVG : high[i] < low[i-2]   →  gap = [high[i], low[i-2]]
    """
    if i < 2:
        return None

    hi_prev2 = df['High'].iat[i - 2]
    lo_prev2 = df['Low'].iat[i - 2]
    hi_curr  = df['High'].iat[i]
    lo_curr  = df['Low'].iat[i]

    if lo_curr > hi_prev2:                 # Bullish FVG
        return PDArray(i, 'FVG', 'bull', bottom=hi_prev2, top=lo_curr)
    if hi_curr < lo_prev2:                 # Bearish FVG
        return PDArray(i, 'FVG', 'bear', bottom=hi_curr, top=lo_prev2)
    return None


def detect_ob_from_fvg(df: pd.DataFrame,
                       fvg: PDArray,
                       lookback: int = OB_LOOKBACK) -> Optional[PDArray]:
    """
    When a FVG forms, the associated Order Block is the last *opposite-coloured*
    candle immediately before the impulsive move that created the FVG.

    Bullish FVG  →  bullish OB = last bearish candle in [fvg.formed_at-2-lookback,
                                                          fvg.formed_at-2]
    Bearish FVG  →  bearish OB = last bullish candle in the same window

    The OB zone is the full candle range [low, high] of that candle.
    Returns None if no qualifying candle is found.
    """
    anchor = fvg.formed_at - 2   # last candle before the gap
    start  = max(0, anchor - lookback)

    if fvg.direction == 'bull':
        # Search backwards for last bearish candle (close < open)
        for j in range(anchor, start - 1, -1):
            if df['Close'].iat[j] < df['Open'].iat[j]:
                return PDArray(fvg.formed_at, 'OB', 'bull',
                               bottom=df['Low'].iat[j],
                               top=df['High'].iat[j])
    else:
        # Search backwards for last bullish candle (close > open)
        for j in range(anchor, start - 1, -1):
            if df['Close'].iat[j] > df['Open'].iat[j]:
                return PDArray(fvg.formed_at, 'OB', 'bear',
                               bottom=df['Low'].iat[j],
                               top=df['High'].iat[j])
    return None


def zone_touched(candle_high: float, candle_low: float,
                 zone_top: float, zone_bottom: float) -> bool:
    """True if the candle's range overlaps with the zone."""
    return candle_low <= zone_top and candle_high >= zone_bottom


# ══════════════════════════════════════════════════════════════════════════════
# KILL ZONE HELPER
# ══════════════════════════════════════════════════════════════════════════════

def get_kill_zone(ts: pd.Timestamp) -> Optional[str]:
    """
    Return the kill zone name ('london' or 'newyork') if the timestamp falls
    inside one, otherwise return None. Timestamps are assumed to be UTC.
    """
    h = ts.hour
    for name, (start, end) in KILL_ZONES.items():
        if start <= h < end:
            return name
    return None


# ══════════════════════════════════════════════════════════════════════════════
# BACKTEST ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def run_backtest(df: pd.DataFrame, bias: pd.Series) -> List[Trade]:
    """
    Bar-by-bar simulation.

    Loop structure per bar i:
      1. Update open trade (check SL / TP hit).
      2. Check pending entry: if confirmation candle arrived → open trade.
         Only finalises if the signal was set inside a valid kill zone.
      3. Detect new FVG (and its associated OB) at bar i.
      4. Expire / mitigate stale PD arrays.
      5. Scan active PD arrays for a new tap signal (sets pending entry).
         Only runs if the current bar is inside a kill zone AND that session
         has not already been traded today.

    Kill zone state:
      traded_sessions is a set of (date, kz_name) pairs. Once an entry is
      opened for a given session, that pair is added and blocks further
      entries for the rest of that session.

    Intra-bar SL/TP resolution:
      If both SL and TP are hit within the same candle, the SL is assumed to
      have been hit first (conservative — avoids overstating performance).
    """
    active_pds: List[PDArray] = []  # All live FVGs and OBs
    open_trade: Optional[Trade]    = None
    pending: Optional[dict]        = None  # Awaiting confirmation candle
    trades: List[Trade]            = []
    traded_sessions: set           = set()  # (date, kz_name) already used

    n = len(df)
    opens  = df['Open'].values
    highs  = df['High'].values
    lows   = df['Low'].values
    closes = df['Close'].values

    for i in range(3, n):
        cur_open  = opens[i]
        cur_high  = highs[i]
        cur_low   = lows[i]
        cur_close = closes[i]
        cur_time  = df.index[i]
        cur_bias  = bias.iat[i]

        # ── 1. Update open trade ──────────────────────────────────────────────
        if open_trade is not None:
            d = open_trade.direction
            sl_hit = (d == 'buy'  and cur_low  <= open_trade.sl) or \
                     (d == 'sell' and cur_high >= open_trade.sl)
            tp_hit = (d == 'buy'  and cur_high >= open_trade.tp) or \
                     (d == 'sell' and cur_low  <= open_trade.tp)

            if sl_hit and tp_hit:
                sl_hit, tp_hit = True, False   # conservative: SL first

            if sl_hit:
                open_trade.exit_time  = cur_time
                open_trade.exit_price = open_trade.sl
                open_trade.result     = 'loss'
                open_trade.pips       = -abs(open_trade.entry - open_trade.sl) / PIP
                trades.append(open_trade)
                open_trade = None
            elif tp_hit:
                open_trade.exit_time  = cur_time
                open_trade.exit_price = open_trade.tp
                open_trade.result     = 'win'
                open_trade.pips       = abs(open_trade.tp - open_trade.entry) / PIP
                trades.append(open_trade)
                open_trade = None

            if open_trade is not None:
                # Still in trade — skip entry logic for this bar
                continue

        # ── 2. Check pending entry (confirmation candle) ──────────────────────
        if pending is not None and open_trade is None:
            direction  = pending['direction']
            pd_bottom  = pending['pd_bottom']
            pd_top     = pending['pd_top']
            pd_kind    = pending['pd_kind']
            kz_name    = pending['kill_zone']
            kz_date    = pending['kz_date']

            # Confirmation: close in direction of bias
            confirmed = (direction == 'buy'  and cur_close > cur_open) or \
                        (direction == 'sell' and cur_close < cur_open)

            # Cancel if price has fully left the PD zone (no longer relevant)
            still_near = zone_touched(cur_high, cur_low, pd_top, pd_bottom)

            # Cancel if this session slot was already consumed by another trade
            # that opened between the signal bar and this bar
            session_free = (kz_date, kz_name) not in traded_sessions

            if confirmed and still_near and session_free:
                entry_price = cur_close

                if direction == 'buy':
                    sl       = pd_bottom
                    sl_dist  = entry_price - sl
                    tp       = entry_price + REWARD_RATIO * sl_dist
                else:
                    sl       = pd_top
                    sl_dist  = sl - entry_price
                    tp       = entry_price - REWARD_RATIO * sl_dist

                sl_pips = sl_dist / PIP
                if MIN_SL_PIPS <= sl_pips <= MAX_SL_PIPS:
                    open_trade = Trade(
                        entry_time = cur_time,
                        direction  = direction,
                        entry      = entry_price,
                        sl         = sl,
                        tp         = tp,
                        pd_kind    = pd_kind,
                        pd_bottom  = pd_bottom,
                        pd_top     = pd_top,
                        htf_bias   = cur_bias,
                        kill_zone  = kz_name,
                    )
                    # Lock this kill zone session for the rest of the day
                    traded_sessions.add((kz_date, kz_name))

            pending = None  # Clear regardless of outcome

        # ── 3. Detect new FVG (and OB) at bar i ─────────────────────────────
        new_fvg = detect_fvg(df, i)
        if new_fvg is not None:
            active_pds.append(new_fvg)
            # Associated Order Block from the impulse that made this FVG
            ob = detect_ob_from_fvg(df, new_fvg, OB_LOOKBACK)
            if ob is not None:
                # Only add if we don't already have this exact OB zone
                duplicate = any(
                    p.kind == 'OB' and p.direction == ob.direction
                    and abs(p.bottom - ob.bottom) < PIP
                    for p in active_pds
                )
                if not duplicate:
                    active_pds.append(ob)

        # ── 4. Expire / mitigate stale PD arrays ─────────────────────────────
        for pd in active_pds:
            if pd.mitigated:
                continue
            age = i - pd.formed_at
            max_age = FVG_MAX_AGE if pd.kind == 'FVG' else OB_MAX_AGE

            if age > max_age:
                pd.mitigated = True
                continue

            # A bullish PD is invalidated if price trades *below* its bottom
            # A bearish PD is invalidated if price trades *above* its top
            if pd.direction == 'bull' and cur_low < pd.bottom:
                pd.mitigated = True
            elif pd.direction == 'bear' and cur_high > pd.top:
                pd.mitigated = True

        active_pds = [p for p in active_pds if not p.mitigated]

        # ── 5. Scan for PD array tap → set pending entry ─────────────────────
        # Gate: must be inside a kill zone, and that session must be unused.
        cur_kz   = get_kill_zone(cur_time)
        cur_date = cur_time.date()
        kz_free  = cur_kz is not None and (cur_date, cur_kz) not in traded_sessions

        if open_trade is None and pending is None \
                and cur_bias in ('bull', 'bear') and kz_free:
            signal = _find_tap_signal(cur_high, cur_low, cur_bias, active_pds)
            if signal is not None:
                pending = {
                    'direction' : 'buy' if cur_bias == 'bull' else 'sell',
                    'pd_bottom' : signal.bottom,
                    'pd_top'    : signal.top,
                    'pd_kind'   : signal.kind,
                    'kill_zone' : cur_kz,
                    'kz_date'   : cur_date,
                    'signal_bar': i,
                }

    # Force-close any trade still open at end of data
    if open_trade is not None:
        last_close = closes[-1]
        open_trade.exit_time  = df.index[-1]
        open_trade.exit_price = last_close
        open_trade.result     = 'open'
        d = open_trade.direction
        open_trade.pips = ((last_close - open_trade.entry) if d == 'buy'
                           else (open_trade.entry - last_close)) / PIP
        trades.append(open_trade)

    return trades


def _find_tap_signal(high: float, low: float,
                     bias: str,
                     active_pds: List[PDArray]) -> Optional[PDArray]:
    """
    Scan active PD arrays for the most recent one that is:
    - Aligned with the current bias (bull PD for bullish bias, bear PD for bearish)
    - Being touched by the current candle (zone overlap)
    - In the correct relative position (discount = below price for buys;
      premium = above price for sells)

    FVGs take priority over OBs (checked first). Returns the most recent match.
    """
    fvg_match = None
    ob_match  = None

    for pd in reversed(active_pds):  # most-recent first
        if bias == 'bull' and pd.direction == 'bull':
            # Discount: zone must be at or below the current mid-price
            if pd.top < high and zone_touched(high, low, pd.top, pd.bottom):
                if pd.kind == 'FVG' and fvg_match is None:
                    fvg_match = pd
                elif pd.kind == 'OB' and ob_match is None:
                    ob_match = pd

        elif bias == 'bear' and pd.direction == 'bear':
            # Premium: zone must be at or above the current mid-price
            if pd.bottom > low and zone_touched(high, low, pd.top, pd.bottom):
                if pd.kind == 'FVG' and fvg_match is None:
                    fvg_match = pd
                elif pd.kind == 'OB' and ob_match is None:
                    ob_match = pd

    return fvg_match if fvg_match is not None else ob_match


# ══════════════════════════════════════════════════════════════════════════════
# METRICS + OUTPUTS
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(trades: List[Trade]) -> dict:
    """Compute all performance statistics and the equity curve."""
    completed = [t for t in trades if t.result in ('win', 'loss')]
    if not completed:
        return {'total': 0, 'wins': 0, 'losses': 0,
                'win_rate': 0.0, 'net_pips': 0.0,
                'profit_factor': 0.0, 'max_drawdown': 0.0,
                'equity': [STARTING_BALANCE]}

    wins   = [t for t in completed if t.result == 'win']
    losses = [t for t in completed if t.result == 'loss']

    gross_profit_pips = sum(t.pips for t in wins)
    gross_loss_pips   = abs(sum(t.pips for t in losses))
    profit_factor     = (gross_profit_pips / gross_loss_pips
                         if gross_loss_pips > 0 else float('inf'))

    # Equity curve: compound 1% risk per trade, 2:1 RR → +2% win / -1% loss
    equity = [STARTING_BALANCE]
    peak   = STARTING_BALANCE
    max_dd = 0.0

    for t in completed:
        bal = equity[-1]
        bal = bal * (1 + RISK_PCT * REWARD_RATIO) if t.result == 'win' \
              else bal * (1 - RISK_PCT)
        equity.append(bal)
        peak  = max(peak, bal)
        dd    = (peak - bal) / peak * 100
        max_dd = max(max_dd, dd)

    return {
        'total'               : len(completed),
        'wins'                : len(wins),
        'losses'              : len(losses),
        'win_rate'            : len(wins) / len(completed) * 100,
        'net_pips'            : sum(t.pips for t in completed),
        'gross_profit_pips'   : gross_profit_pips,
        'gross_loss_pips'     : gross_loss_pips,
        'profit_factor'       : profit_factor,
        'max_drawdown'        : max_dd,
        'equity'              : equity,
        'fvg_trades'          : sum(1 for t in completed if t.pd_kind == 'FVG'),
        'ob_trades'           : sum(1 for t in completed if t.pd_kind == 'OB'),
        'london_trades'       : sum(1 for t in completed if t.kill_zone == 'london'),
        'newyork_trades'      : sum(1 for t in completed if t.kill_zone == 'newyork'),
        'london_wins'         : sum(1 for t in completed if t.kill_zone == 'london'  and t.result == 'win'),
        'newyork_wins'        : sum(1 for t in completed if t.kill_zone == 'newyork' and t.result == 'win'),
    }


def print_summary(metrics: dict, df: pd.DataFrame) -> None:
    """Print a formatted results summary to the console."""
    sep = "=" * 54
    print()
    print(sep)
    print("  HTF Bias + PD Array Backtest  --  EURUSD M5")
    print(sep)
    print(f"  Data range : {df.index[0].date()}  to  {df.index[-1].date()}")
    print(f"  Bars       : {len(df):,}")
    print(sep)
    lnd = metrics['london_trades']
    lnd_wr = (metrics['london_wins'] / lnd * 100) if lnd else 0.0
    ny  = metrics['newyork_trades']
    ny_wr  = (metrics['newyork_wins']  / ny  * 100) if ny  else 0.0
    print(f"  Total Trades    : {metrics['total']}")
    print(f"    London KZ     : {lnd} trades  (WR {lnd_wr:.1f}%)")
    print(f"    New York KZ   : {ny} trades  (WR {ny_wr:.1f}%)")
    print(f"    FVG entries   : {metrics['fvg_trades']}")
    print(f"    OB  entries   : {metrics['ob_trades']}")
    print(f"  Wins / Losses   : {metrics['wins']} / {metrics['losses']}")
    print(f"  Win Rate        : {metrics['win_rate']:.1f}%")
    print(f"  Net Pips        : {metrics['net_pips']:+.1f}")
    print(f"  Profit Factor   : {metrics['profit_factor']:.2f}")
    print(f"  Max Drawdown    : {metrics['max_drawdown']:.1f}%")
    if metrics['total'] > 0:
        final   = metrics['equity'][-1]
        ret_pct = (final - STARTING_BALANCE) / STARTING_BALANCE * 100
        print(f"  Final Balance   : ${final:,.2f}  ({ret_pct:+.1f}%)")
    print(sep)
    print()


def save_equity_curve(metrics: dict, path: str) -> None:
    """Plot the equity curve and save as PNG."""
    equity = metrics['equity']
    if len(equity) < 2:
        print("Not enough completed trades to plot equity curve.")
        return

    fig, ax = plt.subplots(figsize=(13, 6))

    x = range(len(equity))
    ax.plot(equity, color='#1565C0', linewidth=1.6, zorder=3, label='Account equity')
    ax.fill_between(x, STARTING_BALANCE, equity,
                    where=[e >= STARTING_BALANCE for e in equity],
                    alpha=0.18, color='#43A047', label='Above start')
    ax.fill_between(x, STARTING_BALANCE, equity,
                    where=[e < STARTING_BALANCE for e in equity],
                    alpha=0.25, color='#E53935', label='Below start')
    ax.axhline(STARTING_BALANCE, color='gray', linestyle='--',
               linewidth=0.9, alpha=0.6)

    ax.set_title('HTF Bias + PD Array Strategy — Equity Curve', fontsize=14, pad=12)
    ax.set_xlabel('Trade #', fontsize=11)
    ax.set_ylabel('Account Balance ($)', fontsize=11)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'${v:,.0f}'))
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9)

    final   = equity[-1]
    ret_pct = (final - STARTING_BALANCE) / STARTING_BALANCE * 100
    info    = (f"Trades : {metrics['total']}\n"
               f"Win rate: {metrics['win_rate']:.1f}%\n"
               f"PF: {metrics['profit_factor']:.2f}\n"
               f"Max DD: {metrics['max_drawdown']:.1f}%\n"
               f"Return: {ret_pct:+.1f}%")
    ax.text(0.02, 0.97, info, transform=ax.transAxes,
            va='top', ha='left', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      edgecolor='#BDBDBD', alpha=0.9))

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Equity curve  -> {path}")


def save_trade_log(trades: List[Trade], path: str) -> None:
    """Write the full trade log to CSV."""
    rows = []
    for t in trades:
        rows.append({
            'Entry Time'  : t.entry_time,
            'Exit Time'   : t.exit_time,
            'Direction'   : t.direction,
            'Kill Zone'   : t.kill_zone,
            'PD Type'     : t.pd_kind,
            'HTF Bias'    : t.htf_bias,
            'Entry Price' : round(t.entry,     5),
            'SL'          : round(t.sl,        5),
            'TP'          : round(t.tp,        5),
            'Exit Price'  : round(t.exit_price, 5) if t.exit_price is not None else '',
            'Result'      : t.result,
            'Pips'        : round(t.pips, 1)   if t.pips is not None else '',
            'PD Zone Low' : round(t.pd_bottom, 5),
            'PD Zone High': round(t.pd_top,    5),
        })
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"Trade log     -> {path}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print(f"\nLoading data from:\n  {DATA_PATH}")
    df = load_data(DATA_PATH)
    print(f"Loaded {len(df):,} bars  |  {df.index[0]}  to  {df.index[-1]}\n")

    print(f"Computing HTF bias ({HTF_TIMEFRAME.upper()}, {SWING_N}-bar pivots)...")
    bias = compute_htf_bias(df, timeframe=HTF_TIMEFRAME, n_swing=SWING_N)
    bc   = bias.value_counts().to_dict()
    print(f"  bull: {bc.get('bull', 0):,} bars  |  "
          f"bear: {bc.get('bear', 0):,} bars  |  "
          f"neutral: {bc.get('neutral', 0):,} bars\n")

    print("Running backtest...")
    trades  = run_backtest(df, bias)
    metrics = compute_metrics(trades)

    print_summary(metrics, df)

    equity_path = os.path.join(OUTPUT_DIR, 'htf_pdarray_equity.png')
    log_path    = os.path.join(OUTPUT_DIR, 'htf_pdarray_trades.csv')

    save_equity_curve(metrics, equity_path)
    save_trade_log(trades, log_path)
    print("\nDone.")
