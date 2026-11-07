"""
IFVG Backtest — EURUSD M5
London Kill Zone ONLY (08:00–11:00 UTC)
Asia session: 01:00–07:59 UTC
Fully corrected, candle-by-candle, no lookahead.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Constants ────────────────────────────────────────────────────────────────
PIP        = 0.0001
BUFFER     = 5 * PIP      # ±5 pip buffer for previous level taps
BALANCE    = 10_000.0
RISK_PCT   = 0.01
DATA_PATH  = r"C:\Users\Ayaan\Downloads\EURUSD_M5.csv"

# ── Load M5 data ─────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv(DATA_PATH, header=None,
                 names=["datetime", "open", "high", "low", "close", "volume"])
df["datetime"] = pd.to_datetime(df["datetime"])
df = df.sort_values("datetime").reset_index(drop=True)
print(f"  {len(df):,} candles  |  {df['datetime'].iloc[0]}  to  {df['datetime'].iloc[-1]}")

# ── Resample to HTF bars ─────────────────────────────────────────────────────
def _resample(df, freq):
    return (df.set_index("datetime")
              .resample(freq)
              .agg(open=("open","first"), high=("high","max"),
                   low=("low","min"),    close=("close","last"))
              .dropna()
              .reset_index())

df_1h  = _resample(df, "1h")
df_4h  = _resample(df, "4h")
df_1d  = _resample(df, "1D")
df_1w  = _resample(df, "W-MON")   # week labelled by its Monday open
df_1mo = _resample(df, "MS")      # month labelled by first day of month

# Pre-convert datetime columns to numpy for fast binary search
_dts = {
    "1h":  df_1h["datetime"].values.astype("datetime64[ns]"),
    "4h":  df_4h["datetime"].values.astype("datetime64[ns]"),
    "1d":  df_1d["datetime"].values.astype("datetime64[ns]"),
    "1w":  df_1w["datetime"].values.astype("datetime64[ns]"),
    "1mo": df_1mo["datetime"].values.astype("datetime64[ns]"),
}

def _bars_before(htf_df, key, before_dt):
    """Return sub-DataFrame of HTF bars whose open time < before_dt."""
    ts  = np.datetime64(before_dt, "ns")
    idx = np.searchsorted(_dts[key], ts, side="left")   # first bar >= before_dt
    return htf_df.iloc[:idx]

# ── HTF PD array builders ─────────────────────────────────────────────────────
# Each function returns a list of (bot, top, label) tuples.
# For SELL zones: tap when 5m candle HIGH >= bot.
# For BUY  zones: tap when 5m candle LOW  <= top.

def _bearish_fvgs(htf_df, key, before_dt, asia_high, label):
    """Bearish HTF FVG: gap[i] = (bars[i].high, bars[i-2].low) where bars[i].high < bars[i-2].low.
    Zone entirely above Asia high means: bars[i].high > asia_high."""
    bars = _bars_before(htf_df, key, before_dt)
    zones = []
    h = bars["high"].values
    l = bars["low"].values
    for i in range(2, len(bars)):
        bot = h[i]          # top of candle i  (bottom of the gap)
        top = l[i-2]        # bottom of candle i-2 (top of the gap)
        if top > bot and bot > asia_high:   # gap exists AND entirely above Asia high
            zones.append((bot, top, label))
    return zones

def _bullish_fvgs(htf_df, key, before_dt, asia_low, label):
    """Bullish HTF FVG: gap[i] = (bars[i-2].high, bars[i].low) where bars[i-2].high < bars[i].low.
    Entirely below Asia low means: bars[i].low < asia_low."""
    bars = _bars_before(htf_df, key, before_dt)
    zones = []
    h = bars["high"].values
    l = bars["low"].values
    for i in range(2, len(bars)):
        bot = h[i-2]        # high of candle i-2 (bottom of gap)
        top = l[i]          # low of candle i    (top of gap)
        if top > bot and top < asia_low:    # gap exists AND entirely below Asia low
            zones.append((bot, top, label))
    return zones

def _bearish_obs(htf_df, key, before_dt, asia_high, label):
    """Bearish OB: last bullish candle before a bearish impulse (next close < current low).
    Zone = [OB.low, OB.high] entirely above Asia high."""
    bars = _bars_before(htf_df, key, before_dt)
    zones = []
    o = bars["open"].values
    h = bars["high"].values
    l = bars["low"].values
    c = bars["close"].values
    for i in range(1, len(bars) - 1):
        if c[i] > o[i] and c[i+1] < l[i]:     # bullish candle + next closes below its low
            if l[i] > asia_high:                # entirely above Asia high
                zones.append((l[i], h[i], label))
    return zones

def _bullish_obs(htf_df, key, before_dt, asia_low, label):
    """Bullish OB: last bearish candle before a bullish impulse (next close > current high).
    Zone = [OB.low, OB.high] entirely below Asia low."""
    bars = _bars_before(htf_df, key, before_dt)
    zones = []
    o = bars["open"].values
    h = bars["high"].values
    l = bars["low"].values
    c = bars["close"].values
    for i in range(1, len(bars) - 1):
        if c[i] < o[i] and c[i+1] > h[i]:     # bearish candle + next closes above its high
            if h[i] < asia_low:                 # entirely below Asia low
                zones.append((l[i], h[i], label))
    return zones

def _sell_prev_levels(before_dt, asia_high):
    """PDH / PWH / PMH that are above Asia high → treated as thin zones."""
    zones = []
    cd = pd.Timestamp(before_dt)

    prev_d = _bars_before(df_1d, "1d", cd.normalize())
    if not prev_d.empty:
        pdh = prev_d["high"].iloc[-1]
        if pdh > asia_high:
            zones.append((pdh - BUFFER, pdh + BUFFER, "PDH"))

    week_start = (cd - pd.Timedelta(days=cd.dayofweek)).normalize()
    prev_w = _bars_before(df_1w, "1w", week_start)
    if not prev_w.empty:
        pwh = prev_w["high"].iloc[-1]
        if pwh > asia_high:
            zones.append((pwh - BUFFER, pwh + BUFFER, "PWH"))

    mo_start = cd.replace(day=1).normalize()
    prev_m = _bars_before(df_1mo, "1mo", mo_start)
    if not prev_m.empty:
        pmh = prev_m["high"].iloc[-1]
        if pmh > asia_high:
            zones.append((pmh - BUFFER, pmh + BUFFER, "PMH"))

    return zones

def _buy_prev_levels(before_dt, asia_low):
    """PDL / PWL / PML that are below Asia low."""
    zones = []
    cd = pd.Timestamp(before_dt)

    prev_d = _bars_before(df_1d, "1d", cd.normalize())
    if not prev_d.empty:
        pdl = prev_d["low"].iloc[-1]
        if pdl < asia_low:
            zones.append((pdl - BUFFER, pdl + BUFFER, "PDL"))

    week_start = (cd - pd.Timedelta(days=cd.dayofweek)).normalize()
    prev_w = _bars_before(df_1w, "1w", week_start)
    if not prev_w.empty:
        pwl = prev_w["low"].iloc[-1]
        if pwl < asia_low:
            zones.append((pwl - BUFFER, pwl + BUFFER, "PWL"))

    mo_start = cd.replace(day=1).normalize()
    prev_m = _bars_before(df_1mo, "1mo", mo_start)
    if not prev_m.empty:
        pml = prev_m["low"].iloc[-1]
        if pml < asia_low:
            zones.append((pml - BUFFER, pml + BUFFER, "PML"))

    return zones

def get_sell_zones(before_dt, asia_high):
    z  = _bearish_fvgs(df_1h, "1h", before_dt, asia_high, "1H_FVG")
    z += _bearish_fvgs(df_4h, "4h", before_dt, asia_high, "4H_FVG")
    z += _bearish_fvgs(df_1d, "1d", before_dt, asia_high, "1D_FVG")
    z += _bearish_obs( df_1h, "1h", before_dt, asia_high, "1H_OB")
    z += _bearish_obs( df_4h, "4h", before_dt, asia_high, "4H_OB")
    z += _sell_prev_levels(before_dt, asia_high)
    return z

def get_buy_zones(before_dt, asia_low):
    z  = _bullish_fvgs(df_1h, "1h", before_dt, asia_low, "1H_FVG")
    z += _bullish_fvgs(df_4h, "4h", before_dt, asia_low, "4H_FVG")
    z += _bullish_fvgs(df_1d, "1d", before_dt, asia_low, "1D_FVG")
    z += _bullish_obs( df_1h, "1h", before_dt, asia_low, "1H_OB")
    z += _bullish_obs( df_4h, "4h", before_dt, asia_low, "4H_OB")
    z += _buy_prev_levels(before_dt, asia_low)
    return z

# ── London Kill Zone helper ───────────────────────────────────────────────────
def in_lkz(dt):
    tm = dt.hour * 60 + dt.minute
    return 8 * 60 <= tm <= 11 * 60     # 08:00–11:00 inclusive

# ── Pre-compute Asia session H/L per calendar day ─────────────────────────────
print("Computing Asia session ranges...")
asia_mask = (df["datetime"].dt.hour >= 1) & (df["datetime"].dt.hour < 8)
asia = (df[asia_mask]
        .groupby(df["datetime"].dt.date)
        .agg(asia_high=("high", "max"), asia_low=("low", "min")))
# Only keep days that have a full-ish Asia session (at least 12 candles = 1 hour of data)
asia_counts = (df[asia_mask]
               .groupby(df["datetime"].dt.date)
               .size())
asia = asia[asia_counts >= 12]

# ── Main backtest loop ────────────────────────────────────────────────────────
print("Running backtest...")

balance       = BALANCE
equity        = [balance]
trades        = []

# Pre-extract numpy arrays for speed
hi_arr  = df["high"].values
lo_arr  = df["low"].values
op_arr  = df["open"].values
cl_arr  = df["close"].values
dt_arr  = df["datetime"].values          # numpy datetime64

# State variables
active_trade  = None     # dict with trade info, or None
kz_done_days  = set()    # dates where KZ trade already taken

# Per-day setup state  (reset each morning)
setup = {
    "date":       None,
    "state":      "idle",    # idle | swept_sell | swept_buy | tapped_sell | tapped_buy
    "direction":  None,
    "asia_high":  None,
    "asia_low":   None,
    "pd_zones":   None,
    "pd_label":   None,
}

N = len(df)

for i in range(2, N):
    dt  = pd.Timestamp(dt_arr[i])
    day = dt.date()

    # ── Manage active trade (strict candle-by-candle, SL checked first) ───────
    if active_trade is not None:
        if active_trade["direction"] == "SELL":
            hit_sl = hi_arr[i] >= active_trade["sl"]
            hit_tp = lo_arr[i] <= active_trade["tp"]
        else:
            hit_sl = lo_arr[i] <= active_trade["sl"]
            hit_tp = hi_arr[i] >= active_trade["tp"]

        if hit_sl or hit_tp:
            result     = "LOSS" if hit_sl else "WIN"
            exit_price = active_trade["sl"] if hit_sl else active_trade["tp"]
            entry      = active_trade["entry"]

            if active_trade["direction"] == "SELL":
                pnl_pips = (entry - exit_price) / PIP
            else:
                pnl_pips = (exit_price - entry) / PIP

            risk_pips = abs(entry - active_trade["sl"]) / PIP
            pnl_usd   = (pnl_pips / risk_pips) * active_trade["risk_amt"]
            balance  += pnl_usd
            equity.append(balance)

            duration = int((dt - active_trade["entry_dt"]).total_seconds() / 60)
            trades.append({
                "date":           active_trade["entry_dt"].date(),
                "time":           active_trade["entry_dt"].time(),
                "direction":      active_trade["direction"],
                "pd_array_type":  active_trade["pd_label"],
                "entry":          round(entry, 5),
                "sl":             round(active_trade["sl"], 5),
                "tp":             round(active_trade["tp"], 5),
                "result":         result,
                "pips":           round(pnl_pips, 1),
                "duration_mins":  duration,
                "pnl_usd":        round(pnl_usd, 2),
            })
            active_trade = None
        # Always skip setup logic while in trade
        continue

    # ── Reset setup state at start of a new day ───────────────────────────────
    if setup["date"] != day:
        setup["date"]  = day
        setup["state"] = "idle"

    # ── Only act inside London Kill Zone ─────────────────────────────────────
    if not in_lkz(dt):
        # Once past 11:00, invalidate any partial setup
        if dt.hour > 11 or (dt.hour == 11 and dt.minute > 0):
            setup["state"] = "idle"
        continue

    # ── Skip if already took a trade in this KZ ──────────────────────────────
    if day in kz_done_days:
        continue

    # ── Require valid Asia session range ──────────────────────────────────────
    if day not in asia.index:
        continue
    asia_high = asia.loc[day, "asia_high"]
    asia_low  = asia.loc[day, "asia_low"]

    # ── STATE: idle → look for sweep ─────────────────────────────────────────
    if setup["state"] == "idle":
        # SELL sweep: 5m HIGH exceeds Asia high but CLOSE is back below it
        if hi_arr[i] > asia_high and cl_arr[i] < asia_high:
            pd_zones = get_sell_zones(dt, asia_high)
            if pd_zones:        # only proceed if any valid PD zones exist above Asia high
                setup.update(state="swept_sell", direction="SELL",
                             asia_high=asia_high, asia_low=asia_low,
                             pd_zones=pd_zones, pd_label=None)
        # BUY sweep: 5m LOW goes below Asia low but CLOSE is back above it
        elif lo_arr[i] < asia_low and cl_arr[i] > asia_low:
            pd_zones = get_buy_zones(dt, asia_low)
            if pd_zones:
                setup.update(state="swept_buy", direction="BUY",
                             asia_high=asia_high, asia_low=asia_low,
                             pd_zones=pd_zones, pd_label=None)

    # ── STATE: swept → look for HTF PD tap ───────────────────────────────────
    elif setup["state"] in ("swept_sell", "swept_buy"):
        if setup["state"] == "swept_sell":
            # Tap: 5m candle HIGH reaches into any bearish zone above Asia high
            for bot, top, lbl in setup["pd_zones"]:
                if hi_arr[i] >= bot:
                    setup["state"]    = "tapped_sell"
                    setup["pd_label"] = lbl
                    break
        else:
            # Tap: 5m candle LOW reaches into any bullish zone below Asia low
            for bot, top, lbl in setup["pd_zones"]:
                if lo_arr[i] <= top:
                    setup["state"]    = "tapped_buy"
                    setup["pd_label"] = lbl
                    break

    # ── STATE: tapped → look for 5m FVG + disrespect ─────────────────────────
    elif setup["state"] in ("tapped_sell", "tapped_buy"):
        direction = setup["direction"]

        # Find the MOST RECENTLY FORMED 5m FVG within the last 5 candles
        # (FVG at index fi means candle fi is the rightmost candle of the 3-bar pattern)
        fvg_fi  = None
        fvg_bot = fvg_top = None

        for fi in range(i - 1, max(i - 6, 1), -1):   # fi in [i-5 … i-1]
            if direction == "SELL":
                # Bullish 5m FVG: candle[fi].low > candle[fi-2].high
                if lo_arr[fi] > hi_arr[fi - 2]:
                    fvg_fi  = fi
                    fvg_bot = hi_arr[fi - 2]    # bottom of FVG zone
                    fvg_top = lo_arr[fi]         # top of FVG zone
                    break
            else:
                # Bearish 5m FVG: candle[fi].high < candle[fi-2].low
                if hi_arr[fi] < lo_arr[fi - 2]:
                    fvg_fi  = fi
                    fvg_top = lo_arr[fi - 2]    # top of FVG zone
                    fvg_bot = hi_arr[fi]         # bottom of FVG zone
                    break

        if fvg_fi is None:
            continue    # no FVG in window, keep waiting

        # Disrespect window: candles fi+1 through fi+5; current candle i must be in that range
        if not (fvg_fi < i <= fvg_fi + 5):
            continue

        # Check disrespect candle conditions
        if direction == "SELL":
            # Bearish candle (close < open) closing BELOW the bottom of the bullish FVG
            if cl_arr[i] < op_arr[i] and cl_arr[i] < fvg_bot:
                entry = cl_arr[i]
                # SL = highest high from candle[fvg_fi-2] to disrespect candle (inclusive)
                sl = float(np.max(hi_arr[fvg_fi - 2 : i + 1]))
                risk = sl - entry
                if risk <= 0:
                    continue
                tp       = entry - 2.0 * risk
                risk_amt = balance * RISK_PCT
                active_trade = dict(
                    direction="SELL", entry=entry, sl=sl, tp=tp,
                    entry_dt=dt, pd_label=setup["pd_label"],
                    risk_amt=risk_amt
                )
                kz_done_days.add(day)
                setup["state"] = "idle"

        else:   # BUY
            # Bullish candle (close > open) closing ABOVE the top of the bearish FVG
            if cl_arr[i] > op_arr[i] and cl_arr[i] > fvg_top:
                entry = cl_arr[i]
                # SL = lowest low from candle[fvg_fi-2] to disrespect candle (inclusive)
                sl = float(np.min(lo_arr[fvg_fi - 2 : i + 1]))
                risk = entry - sl
                if risk <= 0:
                    continue
                tp       = entry + 2.0 * risk
                risk_amt = balance * RISK_PCT
                active_trade = dict(
                    direction="BUY", entry=entry, sl=sl, tp=tp,
                    entry_dt=dt, pd_label=setup["pd_label"],
                    risk_amt=risk_amt
                )
                kz_done_days.add(day)
                setup["state"] = "idle"

# ── Handle any trade still open at end of data ───────────────────────────────
if active_trade is not None:
    last = df.iloc[-1]
    exit_price = last["close"]
    entry = active_trade["entry"]
    if active_trade["direction"] == "SELL":
        pnl_pips = (entry - exit_price) / PIP
    else:
        pnl_pips = (exit_price - entry) / PIP
    risk_pips = abs(entry - active_trade["sl"]) / PIP
    pnl_usd   = (pnl_pips / risk_pips) * active_trade["risk_amt"]
    balance  += pnl_usd
    equity.append(balance)
    trades.append({
        "date": active_trade["entry_dt"].date(),
        "time": active_trade["entry_dt"].time(),
        "direction": active_trade["direction"],
        "pd_array_type": active_trade["pd_label"],
        "entry": round(entry, 5),
        "sl":    round(active_trade["sl"], 5),
        "tp":    round(active_trade["tp"], 5),
        "result":        "OPEN",
        "pips":          round(pnl_pips, 1),
        "duration_mins": int((pd.Timestamp(dt_arr[-1]) - active_trade["entry_dt"]).total_seconds() / 60),
        "pnl_usd":       round(pnl_usd, 2),
    })

# ── Results ───────────────────────────────────────────────────────────────────
print()
if not trades:
    print("No trades found.")
else:
    tdf = pd.DataFrame(trades)
    tdf.to_csv("ifvg_trades.csv", index=False)

    closed = tdf[tdf["result"] != "OPEN"]
    total  = len(tdf)
    wins   = (tdf["result"] == "WIN").sum()
    losses = (tdf["result"] == "LOSS").sum()
    opens  = (tdf["result"] == "OPEN").sum()
    wr     = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0

    gross_p = tdf[tdf["pnl_usd"] > 0]["pnl_usd"].sum()
    gross_l = tdf[tdf["pnl_usd"] < 0]["pnl_usd"].abs().sum()
    pf      = gross_p / gross_l if gross_l > 0 else float("inf")
    net_pip = tdf["pips"].sum()
    net_usd = tdf["pnl_usd"].sum()

    eq_arr   = np.array(equity)
    roll_max = np.maximum.accumulate(eq_arr)
    dd_pct   = (eq_arr - roll_max) / roll_max * 100
    max_dd   = dd_pct.min()

    sep = "=" * 56
    print(sep)
    print("    IFVG BACKTEST RESULTS  --  EURUSD M5")
    print("    London Kill Zone ONLY  (08:00-11:00 UTC)  |  Asia 01:00-07:59 UTC")
    print(sep)
    print(f"  Total Trades   : {total}  ({wins}W / {losses}L / {opens} open)")
    print(f"  Win Rate       : {wr:.1f}%  (closed trades only)")
    print(f"  Profit Factor  : {pf:.2f}")
    print(f"  Net Pips       : {net_pip:+.1f}")
    print(f"  Max Drawdown   : {max_dd:.2f}%")
    print(f"  Starting Bal   : $10,000.00")
    print(f"  Final Balance  : ${balance:,.2f}  (net {net_usd:+,.2f})")
    print(f"  Return         : {(balance / BALANCE - 1) * 100:+.2f}%")
    print()

    print("  -- By PD Array Type --")
    for pd_type, g in tdf.groupby("pd_array_type"):
        w  = (g["result"] == "WIN").sum()
        l  = (g["result"] == "LOSS").sum()
        wr_g = w / (w + l) * 100 if (w + l) > 0 else 0
        print(f"    {pd_type:<10}  {len(g):3d} trades  "
              f"{w}W/{l}L  WR={wr_g:.0f}%  "
              f"net={g['pips'].sum():+.1f} pips")

    # Equity curve
    plt.figure(figsize=(13, 5), facecolor="#0d1117")
    ax = plt.gca()
    ax.set_facecolor("#0d1117")
    ax.plot(eq_arr, color="#2196F3", linewidth=1.2, zorder=3)
    ax.fill_between(range(len(eq_arr)), BALANCE, eq_arr,
                    where=eq_arr >= BALANCE, alpha=0.15, color="#26a69a")
    ax.fill_between(range(len(eq_arr)), BALANCE, eq_arr,
                    where=eq_arr <  BALANCE, alpha=0.15, color="#ef5350")
    ax.axhline(BALANCE, color="#555566", linewidth=0.8, linestyle="--")
    ax.set_title("IFVG Strategy -- Equity Curve (EURUSD M5, London KZ Only)",
                 color="white", fontsize=11)
    ax.set_xlabel("Trade #", color="#aaaaaa")
    ax.set_ylabel("Balance ($)", color="#aaaaaa")
    ax.tick_params(colors="#aaaaaa")
    ax.grid(True, color="#222233", linewidth=0.5)
    for spine in ax.spines.values():
        spine.set_color("#333344")
    plt.tight_layout()
    plt.savefig("ifvg_equity.png", dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close()

    print()
    print(f"  Saved: ifvg_trades.csv  ({len(tdf)} rows)")
    print("  Saved: ifvg_equity.png")
    print(sep)
