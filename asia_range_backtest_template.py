
"""
Asia Range Backtest Template
----------------------------
Computes the per-day Asia session range from M5 (or any intraday) OHLCV CSVs.
Defaults to 01:00–08:00 GMT. Designed to be easy to extend with your own rules.

Expected CSV columns (case-sensitive by default):
- Either:
    Date, Time, Open, High, Low, Close
  or:
    Datetime, Open, High, Low, Close

Times are treated as GMT/UTC unless you specify otherwise upstream.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional, Tuple

import pandas as pd


# ---------- Configuration Dataclass (easy to extend later) ----------

@dataclass
class SessionConfig:
    session_start: str = "01:00"  # GMT
    session_end: str = "08:00"    # GMT
    date_col: str = "Date"
    time_col: str = "Time"
    datetime_col: str = "Datetime"
    open_col: str = "Open"
    high_col: str = "High"
    low_col: str = "Low"
    close_col: str = "Close"
    inclusive_end: bool = True  # include bars at session_end timestamp


# ---------- IO & Parsing ----------

def load_intraday_csv(path: str, cfg: SessionConfig) -> pd.DataFrame:
    """
    Loads a CSV and returns a tz-naive DataFrame indexed by Datetime (assumed GMT).
    Accepts either separate Date+Time or a single Datetime column.
    """
    df = pd.read_csv(path)

    # Harmonize column names: keep user CSV case by default, but allow common variants
    cols = {c.lower(): c for c in df.columns}

    # If Datetime exists (any case), use it
    dt_col = None
    for k, v in cols.items():
        if k in ("datetime", "timestamp"):
            dt_col = v
            break

    if dt_col is not None:
        df[cfg.datetime_col] = pd.to_datetime(df[dt_col])
    else:
        # Expect separate Date + Time (any case variants)
        date_cand = cols.get(cfg.date_col.lower(), cols.get("date"))
        time_cand = cols.get(cfg.time_col.lower(), cols.get("time"))
        if date_cand is None or time_cand is None:
            raise ValueError(
                "CSV must contain either a 'Datetime' column or both 'Date' and 'Time' columns."
            )
        df[cfg.datetime_col] = pd.to_datetime(df[date_cand] + " " + df[time_cand])

    # Ensure OHLC columns exist (support common case-insensitive variants)
    for need in [cfg.open_col, cfg.high_col, cfg.low_col, cfg.close_col]:
        if need not in df.columns:
            # try case-insensitive fallback
            low = need.lower()
            if low in cols:
                df.rename(columns={cols[low]: need}, inplace=True)
            else:
                raise ValueError(f"Missing required column: {need}")

    # Set index to datetime (assumed GMT)
    df = df.set_index(cfg.datetime_col).sort_index()

    # Keep only essential columns; add others later as needed
    keep = [cfg.open_col, cfg.high_col, cfg.low_col, cfg.close_col]
    df = df[keep]

    # Ensure numeric
    df = df.apply(pd.to_numeric, errors="coerce").dropna()

    return df


# ---------- Core Logic ----------

def slice_session(df: pd.DataFrame, cfg: SessionConfig) -> pd.DataFrame:
    """
    Return rows between session_start and session_end (inclusive end optional).
    Uses DataFrame.between_time. Assumes the index is Datetime and tz-naive GMT.
    """
    # between_time is inclusive on both ends; emulate exclusive end if needed
    session = df.between_time(cfg.session_start, cfg.session_end)
    if not cfg.inclusive_end:
        # Drop rows exactly at session_end (e.g., 08:00:00) if present
        end_time = pd.to_datetime(cfg.session_end).time()
        session = session[session.index.time != end_time]
    return session


def compute_asia_range(df: pd.DataFrame, cfg: SessionConfig) -> pd.DataFrame:
    """
    For each calendar day, compute:
      - Asia_High, Asia_High_Time
      - Asia_Low,  Asia_Low_Time
      - Asia_Open (first session bar's Open)
      - Asia_Close (last session bar's Close)
      - Asia_Range (High - Low)
    Returns a DataFrame indexed by date (YYYY-MM-DD).
    """
    session = slice_session(df, cfg)

    if session.empty:
        return pd.DataFrame(columns=[
            "Asia_High", "Asia_High_Time", "Asia_Low", "Asia_Low_Time",
            "Asia_Open", "Asia_Close", "Asia_Range"
        ])

    # Helper to get first/last and idxmax/idxmin per day
    def _first_open(g: pd.DataFrame) -> float:
        return g.iloc[0][cfg.open_col]

    def _last_close(g: pd.DataFrame) -> float:
        return g.iloc[-1][cfg.close_col]

    # Group by date part of the index
    gb = session.groupby(session.index.date, sort=True)

    # High/Low values and timestamps
    asia_high = gb[cfg.high_col].max()
    asia_low = gb[cfg.low_col].min()

    # Timestamps where those extremes occurred
    asia_high_time = gb[cfg.high_col].idxmax()
    asia_low_time = gb[cfg.low_col].idxmin()

    # Open/Close of the session
    asia_open = gb.apply(_first_open)
    asia_close = gb.apply(_last_close)

    out = pd.DataFrame({
        "Asia_High": asia_high,
        "Asia_High_Time": asia_high_time.values,
        "Asia_Low": asia_low,
        "Asia_Low_Time": asia_low_time.values,
        "Asia_Open": asia_open,
        "Asia_Close": asia_close,
    })

    out["Asia_Range"] = out["Asia_High"] - out["Asia_Low"]

    # Make the index a proper DatetimeIndex of midnight for readability
    out.index = pd.to_datetime(out.index)
    out.index.name = "Date"

    return out


# ---------- Hooks for Your Strategy Rules ----------

def apply_your_rules(price: pd.DataFrame, asia: pd.DataFrame) -> pd.DataFrame:
    """
    Skeleton where you can implement signals/entries based on the Asia range.

    Example: add a simple breakout marker for later backtests.
    (This is just a placeholder; adjust/remove freely.)
    """
    asia = asia.copy()

    # Example placeholders for future columns
    asia["London_Break_Up"] = False
    asia["London_Break_Down"] = False

    # TODO: implement your logic here, possibly by merging back to intraday data
    # and scanning specific time windows for breaks of Asia_High/Asia_Low, etc.

    return asia


# ---------- CLI ----------

def main():
    p = argparse.ArgumentParser(description="Compute Asia session ranges (01:00–08:00 GMT by default).")
    p.add_argument("-i", "--input", required=True, help="Path to input CSV (intraday OHLC).")
    p.add_argument("-o", "--output", default="asia_ranges.csv", help="Path to write output CSV.")
    p.add_argument("--start", default="01:00", help="Session start time in HH:MM (GMT). Default: 01:00")
    p.add_argument("--end", default="08:00", help="Session end time in HH:MM (GMT). Default: 08:00")
    p.add_argument("--exclusive-end", action="store_true", help="Treat session_end as exclusive (drop bars exactly at end).")
    args = p.parse_args()

    cfg = SessionConfig(session_start=args.start, session_end=args.end, inclusive_end=not args.exclusive_end)

    df = load_intraday_csv(args.input, cfg)
    asia = compute_asia_range(df, cfg)

    # Place to call your strategy rules (kept separate for clarity)
    asia = apply_your_rules(df, asia)

    asia.to_csv(args.output, index=True)
    print(f"Wrote {len(asia)} rows to {args.output}")


if __name__ == "__main__":
    main()
