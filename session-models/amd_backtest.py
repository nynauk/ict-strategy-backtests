"""
AMD ICT / TJR Strategy Backtest
Instruments: USA500 (S&P 500), USATECH (NASDAQ), DEUIDX (DAX)
Timeframe: M5

Logic (SELL):
  1. Kill zone opens (London or NY for US; Frankfurt/London for DAX)
  2. 5m candle sweeps a liquidity level:
       - Overnight session high (prev 21:00/18:00 UTC to kill zone)
       - Previous Day High (PDH) or Previous Week High (PWH)
  3. Sweep candle HIGH taps into a 1H or 4H bearish OB or FVG
       (price manipulated INTO institutional supply = HTF PD array)
  4. After sweep: 5m close below pre-sweep swing low  -> bearish MSS
  5. Bearish FVG (imbalance) forms on 5m after the MSS
  6. Price retraces up into the FVG -> short entry at close
  SL: sweep candle high + buffer
  TP: 2:1 RR

Logic (BUY): mirror image.
  Sweep of overnight low / PDL / PWL -> tap into bullish 1H/4H OB or FVG
  -> bullish MSS -> bullish FVG -> long entry. SL at sweep low - buffer. TP 2:1.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Global constants ──────────────────────────────────────────────────────────
BALANCE       = 10_000.0
RISK_PCT      = 0.01
RR            = 2.0
MSS_LOOKBACK  = 20     # candles back to find MSS reference swing
MSS_TIMEOUT   = 200    # max candles after sweep to find MSS
FVG_TIMEOUT   = 100    # max candles after MSS to find FVG
ENTRY_TIMEOUT = 100    # max candles after FVG to find retrace

# ── Instrument configs ────────────────────────────────────────────────────────
INSTRUMENTS = [
    {
        "name":    "USA500 (S&P 500)",
        "short":   "USA500",
        "path":    r"C:\Users\Ayaan\Downloads\USA500IDXUSD_M5.csv",
        "sl_buf":  20.0,    # index points above/below sweep candle for SL
        # UTC kill zone windows (start_hour inclusive, end_hour exclusive)
        "kill_zones": [(8, 10), (13, 16)],
        # Overnight session = from session_open_h UTC (prev day) to session_close_h UTC
        "session_open_h":  21,
        "session_close_h": 13,
    },
    {
        "name":    "USATECH (NASDAQ)",
        "short":   "USATECH",
        "path":    r"C:\Users\Ayaan\Downloads\USATECHIDXUSD_M5.csv",
        "sl_buf":  50.0,
        "kill_zones": [(8, 10), (13, 16)],
        "session_open_h":  21,
        "session_close_h": 13,
    },
    {
        "name":    "DEUIDX (DAX)",
        "short":   "DEUIDX",
        "path":    r"C:\Users\Ayaan\Downloads\DEUIDXEUR_M5 (3).csv",
        "sl_buf":  50.0,
        "kill_zones": [(7, 10)],
        "session_open_h":  18,
        "session_close_h": 7,
    },
]

# ── HTF PD array helpers ──────────────────────────────────────────────────────
def _slice_before(htf_arr, htf_dts, before_dt, recency):
    """Return the last `recency` HTF bars whose open time < before_dt."""
    ts  = np.datetime64(pd.Timestamp(before_dt), "ns")
    idx = int(np.searchsorted(htf_dts, ts, side="left"))
    start = max(0, idx - recency)
    return {k: htf_arr[k][start:idx] for k in htf_arr}

def bearish_zones_htf(htf_arrays, htf_dts_1h, htf_dts_4h, before_dt,
                      sweep_high, recency_1h=300, recency_4h=150):
    """
    Return list of (bot, top, label) bearish OBs and FVGs from 1H and 4H
    where bot <= sweep_high (price tapped into the zone from below).
    """
    zones = []
    configs = [
        (htf_arrays["1h_o"], htf_arrays["1h_h"],
         htf_arrays["1h_l"], htf_arrays["1h_c"], htf_dts_1h, recency_1h, "1H"),
        (htf_arrays["4h_o"], htf_arrays["4h_h"],
         htf_arrays["4h_l"], htf_arrays["4h_c"], htf_dts_4h, recency_4h, "4H"),
    ]
    for o_all, h_all, l_all, c_all, dts_all, rec, lbl in configs:
        ts  = np.datetime64(pd.Timestamp(before_dt), "ns")
        idx = int(np.searchsorted(dts_all, ts, side="left"))
        s   = max(0, idx - rec)
        o = o_all[s:idx]; h = h_all[s:idx]
        l = l_all[s:idx]; c = c_all[s:idx]
        n = len(h)
        if n < 3:
            continue

        # Bearish OBs: last bullish candle where next candle closes below its low
        for j in range(1, n - 1):
            if c[j] > o[j] and c[j+1] < l[j]:
                bot, top = l[j], h[j]
                if bot <= sweep_high:          # price reached zone bottom
                    zones.append((bot, top, f"{lbl}_OB"))

        # Bearish FVGs: gap where h[j] < l[j-2]
        for j in range(2, n):
            bot, top = h[j], l[j-2]
            if top > bot and bot <= sweep_high:
                zones.append((bot, top, f"{lbl}_FVG"))

    return zones

def bullish_zones_htf(htf_arrays, htf_dts_1h, htf_dts_4h, before_dt,
                      sweep_low, recency_1h=300, recency_4h=150):
    """
    Return list of (bot, top, label) bullish OBs and FVGs from 1H and 4H
    where top >= sweep_low (price tapped into the zone from above).
    """
    zones = []
    configs = [
        (htf_arrays["1h_o"], htf_arrays["1h_h"],
         htf_arrays["1h_l"], htf_arrays["1h_c"], htf_dts_1h, recency_1h, "1H"),
        (htf_arrays["4h_o"], htf_arrays["4h_h"],
         htf_arrays["4h_l"], htf_arrays["4h_c"], htf_dts_4h, recency_4h, "4H"),
    ]
    for o_all, h_all, l_all, c_all, dts_all, rec, lbl in configs:
        ts  = np.datetime64(pd.Timestamp(before_dt), "ns")
        idx = int(np.searchsorted(dts_all, ts, side="left"))
        s   = max(0, idx - rec)
        o = o_all[s:idx]; h = h_all[s:idx]
        l = l_all[s:idx]; c = c_all[s:idx]
        n = len(h)
        if n < 3:
            continue

        # Bullish OBs: last bearish candle where next closes above its high
        for j in range(1, n - 1):
            if c[j] < o[j] and c[j+1] > h[j]:
                bot, top = l[j], h[j]
                if top >= sweep_low:
                    zones.append((bot, top, f"{lbl}_OB"))

        # Bullish FVGs: gap where l[j] > h[j-2]
        for j in range(2, n):
            bot, top = h[j-2], l[j]
            if top > bot and top >= sweep_low:
                zones.append((bot, top, f"{lbl}_FVG"))

    return zones

# ── Per-instrument runner ─────────────────────────────────────────────────────
def run_instrument(cfg):
    name    = cfg["name"]
    short   = cfg["short"]
    SL_BUF  = cfg["sl_buf"]
    KZ      = cfg["kill_zones"]
    SESS_OH = cfg["session_open_h"]
    SESS_CH = cfg["session_close_h"]

    print()
    print("=" * 64)
    print(f"  {name}")
    print("=" * 64)

    # Load
    try:
        df = pd.read_csv(cfg["path"], header=None,
                         names=["datetime","open","high","low","close","volume"])
    except FileNotFoundError:
        print(f"  File not found: {cfg['path']}")
        return None

    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    print(f"  {len(df):,} candles | {df['datetime'].iloc[0].date()} to {df['datetime'].iloc[-1].date()}")
    print(f"  Price range: {df['low'].min():.1f} - {df['high'].max():.1f}")

    hi  = df["high"].values
    lo  = df["low"].values
    op  = df["open"].values
    cl  = df["close"].values
    dts = df["datetime"].values
    N   = len(df)

    # HTF resample — store raw numpy arrays for speed
    def _resamp(freq):
        r = (df.set_index("datetime").resample(freq)
               .agg(open=("open","first"), high=("high","max"),
                    low=("low","min"),     close=("close","last"))
               .dropna())
        return r

    r1h = _resamp("1h"); r4h = _resamp("4h")
    htf_arrays = {
        "1h_o": r1h["open"].values,  "1h_h": r1h["high"].values,
        "1h_l": r1h["low"].values,   "1h_c": r1h["close"].values,
        "4h_o": r4h["open"].values,  "4h_h": r4h["high"].values,
        "4h_l": r4h["low"].values,   "4h_c": r4h["close"].values,
    }
    htf_dts_1h = r1h.index.values.astype("datetime64[ns]")
    htf_dts_4h = r4h.index.values.astype("datetime64[ns]")

    # Previous daily / weekly H/L dicts
    def _hl_dict(df, freq):
        r = (df.set_index("datetime").resample(freq)
               .agg(H=("high","max"), L=("low","min")).dropna())
        return {t.date(): (row.H, row.L) for t, row in r.iterrows()}

    daily_hl  = _hl_dict(df, "1D")
    weekly_hl = _hl_dict(df, "W-MON")

    def _prev(hl):
        dates = sorted(hl.keys())
        return {dates[i]: hl[dates[i-1]] for i in range(1, len(dates))}

    prev_day  = _prev(daily_hl)
    prev_week = _prev(weekly_hl)

    def week_start(dt):
        return (dt - pd.Timedelta(days=dt.dayofweek)).date()

    # Overnight session H/L per calendar date
    print(f"  Computing overnight session ranges ({SESS_OH}:00 UTC prev day to {SESS_CH}:00 UTC) ...")
    all_dates = sorted(set(pd.Timestamp(d).date() for d in dts))
    session_hl = {}
    for d in all_dates:
        d_ts   = pd.Timestamp(d)
        prev_d = d_ts - pd.Timedelta(days=1)
        mask = (
            (df["datetime"] >= prev_d + pd.Timedelta(hours=SESS_OH)) &
            (df["datetime"] <  d_ts   + pd.Timedelta(hours=SESS_CH))
        )
        sess = df[mask]
        if len(sess) >= 6:   # need at least 30 min of overnight data
            session_hl[d] = (sess["high"].max(), sess["low"].min())
    print(f"  Overnight ranges available: {len(session_hl)} days")

    def in_kz(dt):
        h = dt.hour
        return any(s <= h < e for s, e in KZ)

    # ── State machine ─────────────────────────────────────────────────────────
    STATE_TO = {
        "manipulated": MSS_TIMEOUT,
        "mss_sell": FVG_TIMEOUT,   "mss_buy": FVG_TIMEOUT,
        "fvg_sell": ENTRY_TIMEOUT, "fvg_buy": ENTRY_TIMEOUT,
    }

    state        = "idle"
    direction    = None
    sweep_lbl    = None    # what level was swept
    htf_lbl      = None    # which HTF zone was tapped
    sweep_hi     = None    # sweep candle high (SL anchor for sells)
    sweep_lo     = None    # sweep candle low  (SL anchor for buys)
    mss_ref      = None
    sweep_i      = None
    fvg_bot      = fvg_top = None
    state_since  = None

    active_trade = None
    trades       = []
    balance      = BALANCE
    equity       = [BALANCE]

    for i in range(2, N):
        dt  = pd.Timestamp(dts[i])
        day = dt.date()

        # ── Manage active trade ───────────────────────────────────────────────
        if active_trade is not None:
            d = active_trade["direction"]
            hit_sl = (hi[i] >= active_trade["sl"]) if d == "SELL" else (lo[i] <= active_trade["sl"])
            hit_tp = (lo[i] <= active_trade["tp"]) if d == "SELL" else (hi[i] >= active_trade["tp"])

            if hit_sl or hit_tp:
                result = "LOSS" if hit_sl else "WIN"
                xp     = active_trade["sl"] if hit_sl else active_trade["tp"]
                ep     = active_trade["entry"]
                pnl_pt = (ep - xp) if d == "SELL" else (xp - ep)
                rsk_pt = abs(ep - active_trade["sl"])
                pnl_u  = (pnl_pt / rsk_pt) * active_trade["risk_amt"]
                balance += pnl_u
                equity.append(balance)
                trades.append({
                    "entry_dt":  active_trade["entry_dt"],
                    "direction": d,
                    "sweep_lvl": active_trade["sweep_lbl"],
                    "htf_zone":  active_trade["htf_lbl"],
                    "entry":     round(ep, 2),
                    "sl":        round(active_trade["sl"], 2),
                    "tp":        round(active_trade["tp"], 2),
                    "result":    result,
                    "pts":       round(pnl_pt, 2),
                    "sl_pts":    round(rsk_pt, 2),
                    "pnl_usd":   round(pnl_u, 2),
                    "dur_m":     int((dt - active_trade["entry_dt"]).total_seconds() / 60),
                })
                active_trade = None
            continue

        # ── Period levels ─────────────────────────────────────────────────────
        pdh, pdl = prev_day.get(day, (np.nan, np.nan))
        pwh, pwl = prev_week.get(week_start(dt), (np.nan, np.nan))
        sh, sl_s = session_hl.get(day, (np.nan, np.nan))

        # ── Timeout ───────────────────────────────────────────────────────────
        if state != "idle" and state_since is not None:
            if (i - state_since) > STATE_TO.get(state, 9999):
                state = "idle"

        # ════════════════════════════════════════════════════════════════════
        # IDLE: look for manipulation during kill zone only
        # ════════════════════════════════════════════════════════════════════
        if state == "idle":
            if not in_kz(dt):
                continue

            # ── SELL: sweep of session high / PDH / PWH ───────────────────
            sell_found = False
            for lvl, lbl in [(sh, "SessH"), (pdh, "PDH"), (pwh, "PWH")]:
                if np.isnan(lvl) or hi[i] <= lvl:
                    continue
                # Sweep confirmed — check HTF PD array tap
                bzones = bearish_zones_htf(
                    htf_arrays, htf_dts_1h, htf_dts_4h, dt, hi[i])
                if not bzones:
                    continue
                # Use the highest-bot zone tapped (most precise supply)
                hit = max(bzones, key=lambda z: z[0])
                state       = "manipulated"
                direction   = "SELL"
                sweep_lbl   = lbl
                htf_lbl     = hit[2]
                sweep_hi    = hi[i]
                mss_ref     = float(np.min(lo[max(0, i - MSS_LOOKBACK):i + 1]))
                sweep_i     = i
                state_since = i
                sell_found  = True
                break

            if sell_found:
                continue

            # ── BUY: sweep of session low / PDL / PWL ────────────────────
            for lvl, lbl in [(sl_s, "SessL"), (pdl, "PDL"), (pwl, "PWL")]:
                if np.isnan(lvl) or lo[i] >= lvl:
                    continue
                bzones = bullish_zones_htf(
                    htf_arrays, htf_dts_1h, htf_dts_4h, dt, lo[i])
                if not bzones:
                    continue
                hit = min(bzones, key=lambda z: z[1])   # lowest-top zone
                state       = "manipulated"
                direction   = "BUY"
                sweep_lbl   = lbl
                htf_lbl     = hit[2]
                sweep_lo    = lo[i]
                mss_ref     = float(np.max(hi[max(0, i - MSS_LOOKBACK):i + 1]))
                sweep_i     = i
                state_since = i
                break

        # ════════════════════════════════════════════════════════════════════
        # MANIPULATED: wait for bearish / bullish MSS on 5m
        # ════════════════════════════════════════════════════════════════════
        elif state == "manipulated":
            if direction == "SELL":
                if cl[i] < mss_ref:
                    state       = "mss_sell"
                    state_since = i
            else:
                if cl[i] > mss_ref:
                    state       = "mss_buy"
                    state_since = i

        # ════════════════════════════════════════════════════════════════════
        # MSS: look for 5m imbalance (FVG) in the direction of the move
        # ════════════════════════════════════════════════════════════════════
        elif state == "mss_sell":
            # Bearish FVG: candle[i].high < candle[i-2].low
            if hi[i] < lo[i - 2]:
                fvg_bot     = hi[i]
                fvg_top     = lo[i - 2]
                state       = "fvg_sell"
                state_since = i

        elif state == "mss_buy":
            # Bullish FVG: candle[i].low > candle[i-2].high
            if lo[i] > hi[i - 2]:
                fvg_top     = lo[i]
                fvg_bot     = hi[i - 2]
                state       = "fvg_buy"
                state_since = i

        # ════════════════════════════════════════════════════════════════════
        # FVG: wait for price to retrace into the imbalance
        # ════════════════════════════════════════════════════════════════════
        elif state == "fvg_sell":
            # Price rallies back up into bearish FVG -> sell
            if hi[i] >= fvg_bot:
                ep   = cl[i]
                sl   = sweep_hi + SL_BUF
                risk = sl - ep
                if risk > 0:
                    active_trade = dict(
                        direction="SELL", entry=ep,
                        sl=sl, tp=ep - RR * risk,
                        entry_dt=dt, sweep_lbl=sweep_lbl, htf_lbl=htf_lbl,
                        risk_amt=balance * RISK_PCT,
                    )
                    state = "idle"

        elif state == "fvg_buy":
            # Price pulls back down into bullish FVG -> buy
            if lo[i] <= fvg_top:
                ep   = cl[i]
                sl   = sweep_lo - SL_BUF
                risk = ep - sl
                if risk > 0:
                    active_trade = dict(
                        direction="BUY", entry=ep,
                        sl=sl, tp=ep + RR * risk,
                        entry_dt=dt, sweep_lbl=sweep_lbl, htf_lbl=htf_lbl,
                        risk_amt=balance * RISK_PCT,
                    )
                    state = "idle"

    # ── Handle trade open at end of data ─────────────────────────────────────
    if active_trade is not None:
        ep  = active_trade["entry"]
        d   = active_trade["direction"]
        xp  = cl[-1]
        pnl_pt = (ep - xp) if d == "SELL" else (xp - ep)
        rsk_pt = abs(ep - active_trade["sl"])
        pnl_u  = (pnl_pt / rsk_pt) * active_trade["risk_amt"]
        balance += pnl_u
        equity.append(balance)
        trades.append({
            "entry_dt": active_trade["entry_dt"], "direction": d,
            "sweep_lvl": active_trade["sweep_lbl"], "htf_zone": active_trade["htf_lbl"],
            "entry": round(ep, 2), "sl": round(active_trade["sl"], 2),
            "tp": round(active_trade["tp"], 2), "result": "OPEN",
            "pts": round(pnl_pt, 2), "sl_pts": round(rsk_pt, 2),
            "pnl_usd": round(pnl_u, 2),
            "dur_m": int((pd.Timestamp(dts[-1]) - active_trade["entry_dt"]).total_seconds() / 60),
        })

    # ── Results ───────────────────────────────────────────────────────────────
    if not trades:
        print("  No trades generated.")
        return None

    tdf  = pd.DataFrame(trades)
    csv  = f"amd_{short}_trades.csv"
    tdf.to_csv(csv, index=False)

    wins   = (tdf["result"] == "WIN").sum()
    losses = (tdf["result"] == "LOSS").sum()
    opens  = (tdf["result"] == "OPEN").sum()
    total  = len(tdf)
    wr     = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0.0
    gp     = tdf[tdf["pnl_usd"] > 0]["pnl_usd"].sum()
    gl     = tdf[tdf["pnl_usd"] < 0]["pnl_usd"].abs().sum()
    pf     = gp / gl if gl > 0 else float("inf")
    net_pt = tdf["pts"].sum()
    net_u  = tdf["pnl_usd"].sum()
    w_pts  = tdf[tdf["result"] == "WIN"]["pts"]
    l_pts  = tdf[tdf["result"] == "LOSS"]["pts"]

    eq_arr   = np.array(equity)
    roll_max = np.maximum.accumulate(eq_arr)
    max_dd   = ((eq_arr - roll_max) / roll_max * 100).min()

    S2 = "-" * 64
    print(f"  Total Trades    : {total}  ({wins}W / {losses}L / {opens} open)")
    print(f"  Win Rate        : {wr:.1f}%")
    print(f"  Profit Factor   : {pf:.2f}")
    print(f"  Net Points      : {net_pt:+.1f}")
    print(f"  Max Drawdown    : {max_dd:.2f}%")
    print(f"  Avg Win         : +{w_pts.mean():.1f} pts" if len(w_pts) > 0 else "  Avg Win         : n/a")
    print(f"  Avg Loss        : {l_pts.mean():.1f} pts" if len(l_pts) > 0 else "  Avg Loss        : n/a")
    print(f"  RR              : {RR:.1f}:1 (fixed)")
    print(f"  Starting Bal    : ${BALANCE:,.2f}")
    print(f"  Final Balance   : ${balance:,.2f}  (net {net_u:+,.2f})")
    print(f"  Return          : {(balance / BALANCE - 1) * 100:+.2f}%")

    # Breakdowns
    print()
    print(f"  {S2}")
    print("  -- By Sweep Level --")
    for lbl, g in tdf.groupby("sweep_lvl"):
        w = (g["result"] == "WIN").sum(); l = (g["result"] == "LOSS").sum()
        wg = w / (w + l) * 100 if (w + l) > 0 else 0.0
        print(f"    {lbl:<10} {len(g):3d} trades  {w}W/{l}L  WR={wg:.0f}%  net={g['pts'].sum():+.1f} pts")

    print()
    print("  -- By HTF PD Array Zone --")
    for lbl, g in tdf.groupby("htf_zone"):
        w = (g["result"] == "WIN").sum(); l = (g["result"] == "LOSS").sum()
        wg = w / (w + l) * 100 if (w + l) > 0 else 0.0
        print(f"    {lbl:<10} {len(g):3d} trades  {w}W/{l}L  WR={wg:.0f}%  net={g['pts'].sum():+.1f} pts")

    print()
    print("  -- By Direction --")
    for d, g in tdf.groupby("direction"):
        w = (g["result"] == "WIN").sum(); l = (g["result"] == "LOSS").sum()
        wg = w / (w + l) * 100 if (w + l) > 0 else 0.0
        print(f"    {d:<5} {len(g):3d} trades  {w}W/{l}L  WR={wg:.0f}%  net={g['pts'].sum():+.1f} pts")

    print()
    print("  -- SL Size Distribution --")
    bins = [0, 20, 50, 100, 200, 500, 9999]
    for bucket, g in tdf.groupby(pd.cut(tdf["sl_pts"], bins=bins)):
        if len(g) == 0:
            continue
        w = (g["result"] == "WIN").sum(); l = (g["result"] == "LOSS").sum()
        wg = w / (w + l) * 100 if (w + l) > 0 else 0.0
        print(f"    SL {str(bucket):<20} {len(g):3d} trades  WR={wg:.0f}%  net={g['pts'].sum():+.1f} pts")

    # Equity curve
    plt.figure(figsize=(13, 5), facecolor="#0d1117")
    ax = plt.gca()
    ax.set_facecolor("#0d1117")
    ax.plot(eq_arr, color="#2196F3", linewidth=1.3, zorder=3)
    ax.fill_between(range(len(eq_arr)), BALANCE, eq_arr,
                    where=eq_arr >= BALANCE, alpha=0.15, color="#26a69a")
    ax.fill_between(range(len(eq_arr)), BALANCE, eq_arr,
                    where=eq_arr <  BALANCE, alpha=0.15, color="#ef5350")
    ax.axhline(BALANCE, color="#555566", linewidth=0.8, linestyle="--")
    ax.set_title(f"AMD ICT / TJR -- {name} -- Equity Curve (2:1 RR)",
                 color="white", fontsize=11)
    ax.set_xlabel("Trade #", color="#aaaaaa")
    ax.set_ylabel("Balance ($)", color="#aaaaaa")
    ax.tick_params(colors="#aaaaaa")
    ax.grid(True, color="#222233", linewidth=0.5)
    for sp in ax.spines.values():
        sp.set_color("#333344")
    plt.tight_layout()
    img = f"amd_{short}_equity.png"
    plt.savefig(img, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print()
    print(f"  Saved: {csv}, {img}")

    return {
        "name": name, "short": short, "total": total,
        "wins": wins, "losses": losses, "wr": wr, "pf": pf,
        "net_pt": net_pt, "net_u": net_u, "max_dd": max_dd,
        "balance": balance,
    }

# ── Run all instruments ───────────────────────────────────────────────────────
print("AMD ICT / TJR Strategy Backtest")
print("Logic: Kill zone sweep of session H/L or PDH/PDL/PWH/PWL")
print("       + tap into 1H/4H OB or FVG (manipulation into PD array)")
print("       + 5m bearish/bullish MSS + 5m FVG retrace entry")
print("       SL at sweep extreme | TP 2:1 RR")
print()

summary = []
for cfg in INSTRUMENTS:
    r = run_instrument(cfg)
    if r:
        summary.append(r)

# ── Summary table ─────────────────────────────────────────────────────────────
if len(summary) > 1:
    SEP = "=" * 75
    print()
    print(SEP)
    print("  SUMMARY -- All Instruments")
    print(SEP)
    print(f"  {'Instrument':<22} {'Trd':>5} {'WR%':>6} {'PF':>6} "
          f"{'Net Pts':>10} {'MaxDD%':>8} {'Return%':>8}")
    print(f"  {'-'*71}")
    for r in summary:
        ret = (r["balance"] / BALANCE - 1) * 100
        print(f"  {r['name']:<22} {r['total']:>5} {r['wr']:>5.1f}% "
              f"{r['pf']:>6.2f} {r['net_pt']:>+10.1f} "
              f"{r['max_dd']:>7.2f}% {ret:>+7.2f}%")
    print(SEP)
