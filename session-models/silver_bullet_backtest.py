"""
ICT Silver Bullet Strategy Backtest
Instruments : USA500 (S&P 500), USATECH (NASDAQ), DEUIDX (DAX)
Timeframe   : M5 entries, 4H bias

Rules (verbatim from the PDF):
  1. Only trade during one of the three 1-hour Silver Bullet windows (NY time):
       - London Open : 03:00-04:00 AM ET
       - NY AM       : 10:00-11:00 AM ET
       - NY PM       : 02:00-03:00 PM ET
  2. Determine 4H directional bias BEFORE the window opens:
       - Bullish if 4H close > 4H close 2 bars ago
       - Bearish if 4H close < 4H close 2 bars ago
       - No trade if flat
  3. Mark liquidity (PDH / PDL / session H/L) as TP targets.
  4. Within the window, find the FIRST M5 FVG in the direction of the 4H bias.
       - Bullish FVG : lo[i] > hi[i-2]  (upward gap, FVG zone = [hi[i-2], lo[i]])
       - Bearish FVG : hi[i] < lo[i-2]  (downward gap, FVG zone = [hi[i], lo[i-2]])
  5. Enter at the close of the first retrace candle that trades back INTO the FVG.
       - Long  : future candle LOW  <= lo[fvg_i]  (top of bullish FVG)
       - Short : future candle HIGH >= hi[fvg_i]  (bottom of bearish FVG)
       Retrace window : from FVG formation up to 2 hours (24 M5 candles) after window close.
  6. Stop Loss:
       - Long  : below LOW  of left candle of FVG (lo[fvg_i-2]) minus buffer
       - Short : above HIGH of left candle of FVG (hi[fvg_i-2]) plus buffer
  7. Take Profit : 2:1 RR (fixed).
  8. Silver Bullet framework : skip if TP distance < 10 pts (indices).
  9. One trade per window. A new window resets state even if previous trade is still open.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Constants ────────────────────────────────────────────────────────────────
BALANCE        = 10_000.0
RISK_PCT       = 0.01
RR             = 2.0
MIN_TP_PTS     = 10.0      # Silver Bullet minimum framework (points)
RETRACE_CANDLES = 24       # max candles after window close to wait for retrace

# Silver Bullet windows — (start_hour, end_hour) in New York time
SB_WINDOWS = [
    (3,  4,  "London"),
    (10, 11, "NY_AM"),
    (14, 15, "NY_PM"),
]

# ── Instrument configs ────────────────────────────────────────────────────────
INSTRUMENTS = [
    {
        "name":  "USA500 (S&P 500)",
        "short": "USA500",
        "path":  r"C:\Users\Ayaan\Downloads\USA500IDXUSD_M5.csv",
        "sl_buf": 5.0,
    },
    {
        "name":  "USATECH (NASDAQ)",
        "short": "USATECH",
        "path":  r"C:\Users\Ayaan\Downloads\USATECHIDXUSD_M5.csv",
        "sl_buf": 15.0,
    },
    {
        "name":  "DEUIDX (DAX)",
        "short": "DEUIDX",
        "path":  r"C:\Users\Ayaan\Downloads\DEUIDXEUR_M5 (3).csv",
        "sl_buf": 15.0,
    },
]

# ── Per-instrument runner ─────────────────────────────────────────────────────
def run_instrument(cfg):
    name   = cfg["name"]
    short  = cfg["short"]
    SL_BUF = cfg["sl_buf"]

    print()
    print("=" * 66)
    print(f"  {name}")
    print("=" * 66)

    try:
        df = pd.read_csv(cfg["path"], header=None,
                         names=["datetime","open","high","low","close","volume"])
    except FileNotFoundError:
        print(f"  File not found: {cfg['path']}")
        return None

    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    print(f"  {len(df):,} candles | {df['datetime'].iloc[0].date()} "
          f"to {df['datetime'].iloc[-1].date()}")

    hi  = df["high"].values
    lo  = df["low"].values
    op  = df["open"].values
    cl  = df["close"].values
    dts = df["datetime"].values
    N   = len(df)

    # ── Convert timestamps to New York time (handles DST) ────────────────────
    print("  Converting timestamps to New York time...")
    dt_ny   = df["datetime"].dt.tz_localize("UTC").dt.tz_convert("America/New_York")
    ny_hour = dt_ny.dt.hour.values
    ny_min  = dt_ny.dt.minute.values
    ny_date = dt_ny.dt.date.values          # numpy array of date objects

    # ── Resample to 4H for bias ──────────────────────────────────────────────
    r4h = (df.set_index("datetime").resample("4h")
             .agg(open=("open","first"), close=("close","last"))
             .dropna().reset_index())
    dts_4h = r4h["datetime"].values.astype("datetime64[ns]")
    cl_4h  = r4h["close"].values
    op_4h  = r4h["open"].values

    def get_4h_bias(before_dt):
        """
        Bias of the last COMPLETED 4H bar before before_dt.
        Returns 1 (bullish), -1 (bearish), or 0 (flat/skip).
        """
        ts  = np.datetime64(pd.Timestamp(before_dt), "ns")
        idx = int(np.searchsorted(dts_4h, ts, side="left")) - 1
        if idx < 2:
            return 0
        # 2-bar 4H momentum: compare most recent close to close 2 bars ago
        if cl_4h[idx] > cl_4h[idx - 2]:
            return 1
        elif cl_4h[idx] < cl_4h[idx - 2]:
            return -1
        return 0

    # ── Pre-compute previous day H/L for reference ──────────────────────────
    daily_hl = (df.set_index("datetime").resample("1D")
                  .agg(H=("high","max"), L=("low","min")).dropna())
    day_dict = {t.date(): (row.H, row.L) for t, row in daily_hl.iterrows()}
    prev_day_hl = {}
    dates = sorted(day_dict.keys())
    for i, d in enumerate(dates[1:], 1):
        prev_day_hl[d] = day_dict[dates[i-1]]

    # ── Determine which Silver Bullet window each candle belongs to ──────────
    # window_id[i] = 0, 1, 2 or -1 (not in any window)
    window_id = np.full(N, -1, dtype=np.int8)
    for wid, (wh_start, wh_end, _) in enumerate(SB_WINDOWS):
        mask = (ny_hour >= wh_start) & (ny_hour < wh_end)
        window_id[mask] = wid

    # ── Main backtest ─────────────────────────────────────────────────────────
    # State per (date, window_id):
    #   "idle"        - no FVG yet found
    #   "fvg_found"   - FVG formed, waiting for retrace
    # We track these per window slot to allow 3 setups/day.

    window_state  = {}    # (ny_date, wid) -> state dict
    active_trade  = None
    trades        = []
    balance       = BALANCE
    equity        = [BALANCE]

    for i in range(2, N):
        dt_ts = pd.Timestamp(dts[i])
        day   = ny_date[i]
        wid   = int(window_id[i])

        # ── Manage active trade ───────────────────────────────────────────────
        if active_trade is not None:
            d      = active_trade["direction"]
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
                dur = int((dt_ts - active_trade["entry_dt"]).total_seconds() / 60)
                trades.append({
                    "entry_dt":  active_trade["entry_dt"],
                    "direction": d,
                    "window":    active_trade["window"],
                    "bias_src":  active_trade["bias_src"],
                    "entry":     round(ep, 2),
                    "sl":        round(active_trade["sl"], 2),
                    "tp":        round(active_trade["tp"], 2),
                    "result":    result,
                    "pts":       round(pnl_pt, 2),
                    "sl_pts":    round(rsk_pt, 2),
                    "pnl_usd":   round(pnl_u, 2),
                    "dur_m":     dur,
                })
                active_trade = None
            # NOTE: we don't skip setup logic while in trade —
            # windows are independent and can set up the next trade

        key = (day, wid)

        # ── Candle is INSIDE a Silver Bullet window ───────────────────────────
        if wid >= 0:
            wh_start, wh_end, wname = SB_WINDOWS[wid]
            st = window_state.get(key)

            # ── Initialise state for this window on first candle ──────────────
            if st is None:
                bias = get_4h_bias(dt_ts)
                if bias == 0:
                    window_state[key] = {"status": "skip"}
                    continue
                window_state[key] = {
                    "status":    "searching",
                    "bias":      bias,        # 1=bull, -1=bear
                    "wname":     wname,
                    "window_end_i": None,     # set when window closes
                }
                st = window_state[key]

            if st["status"] == "skip":
                continue

            # ── SEARCHING: look for first FVG in bias direction ───────────────
            if st["status"] == "searching":
                bias = st["bias"]
                if bias == 1:      # bullish FVG: lo[i] > hi[i-2]
                    if lo[i] > hi[i - 2]:
                        fvg_top = lo[i]        # top of gap (retrace target from above)
                        fvg_bot = hi[i - 2]    # bottom of gap
                        sl      = lo[i - 2] - SL_BUF
                        st.update({
                            "status":  "fvg_found",
                            "fvg_i":   i,
                            "fvg_top": fvg_top,
                            "fvg_bot": fvg_bot,
                            "sl":      sl,
                            "direction": "BUY",
                        })
                else:              # bearish FVG: hi[i] < lo[i-2]
                    if hi[i] < lo[i - 2]:
                        fvg_bot = hi[i]        # bottom of gap (retrace target from below)
                        fvg_top = lo[i - 2]    # top of gap
                        sl      = hi[i - 2] + SL_BUF
                        st.update({
                            "status":  "fvg_found",
                            "fvg_i":   i,
                            "fvg_top": fvg_top,
                            "fvg_bot": fvg_bot,
                            "sl":      sl,
                            "direction": "SELL",
                        })

            # ── FVG FOUND: wait for retrace into FVG ─────────────────────────
            if st["status"] == "fvg_found" and active_trade is None:
                direction = st["direction"]
                fvg_top   = st["fvg_top"]
                fvg_bot   = st["fvg_bot"]
                sl        = st["sl"]

                triggered = False
                if direction == "BUY"  and lo[i] <= fvg_top:
                    triggered = True
                elif direction == "SELL" and hi[i] >= fvg_bot:
                    triggered = True

                if triggered:
                    ep   = cl[i]
                    risk = abs(ep - sl)
                    if risk > 0:
                        tp      = (ep + RR * risk) if direction == "BUY" else (ep - RR * risk)
                        tp_dist = abs(tp - ep)
                        if tp_dist >= MIN_TP_PTS:       # Silver Bullet framework check
                            active_trade = dict(
                                direction=direction, entry=ep, sl=sl, tp=tp,
                                entry_dt=dt_ts, window=wname,
                                bias_src="4H_momentum",
                                risk_amt=balance * RISK_PCT,
                            )
                            st["status"] = "done"

        # ── Candle is OUTSIDE windows: extend retrace wait after window close ─
        else:
            # For any window that had an FVG found but not yet entered/expired,
            # keep waiting for retrace for up to RETRACE_CANDLES after window close
            for k, st in list(window_state.items()):
                if st.get("status") != "fvg_found":
                    continue
                if active_trade is not None:
                    continue
                # Check if within retrace window
                fvg_i = st["fvg_i"]
                if (i - fvg_i) > RETRACE_CANDLES:
                    st["status"] = "expired"
                    continue

                direction = st["direction"]
                fvg_top   = st["fvg_top"]
                fvg_bot   = st["fvg_bot"]
                sl        = st["sl"]

                triggered = False
                if direction == "BUY"  and lo[i] <= fvg_top:
                    triggered = True
                elif direction == "SELL" and hi[i] >= fvg_bot:
                    triggered = True

                if triggered:
                    ep   = cl[i]
                    risk = abs(ep - sl)
                    if risk > 0:
                        tp      = (ep + RR * risk) if direction == "BUY" else (ep - RR * risk)
                        tp_dist = abs(tp - ep)
                        if tp_dist >= MIN_TP_PTS:
                            active_trade = dict(
                                direction=direction, entry=ep, sl=sl, tp=tp,
                                entry_dt=dt_ts, window=st["wname"],
                                bias_src="4H_momentum",
                                risk_amt=balance * RISK_PCT,
                            )
                            st["status"] = "done"

    # ── Handle open trade at end of data ─────────────────────────────────────
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
            "window": active_trade["window"], "bias_src": active_trade["bias_src"],
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
    csv  = f"sb_{short}_trades.csv"
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

    S2 = "-" * 66
    print(f"  Total Trades    : {total}  ({wins}W / {losses}L / {opens} open)")
    print(f"  Win Rate        : {wr:.1f}%")
    print(f"  Profit Factor   : {pf:.2f}")
    print(f"  Net Points      : {net_pt:+.1f}")
    print(f"  Max Drawdown    : {max_dd:.2f}%")
    print(f"  Avg Win         : +{w_pts.mean():.1f} pts" if len(w_pts) > 0 else "  Avg Win         : n/a")
    print(f"  Avg Loss        : {l_pts.mean():.1f} pts" if len(l_pts) > 0 else "  Avg Loss        : n/a")
    print(f"  RR              : {RR:.1f}:1  |  Min TP dist: {MIN_TP_PTS} pts")
    print(f"  Starting Bal    : ${BALANCE:,.2f}")
    print(f"  Final Balance   : ${balance:,.2f}  (net {net_u:+,.2f})")
    print(f"  Return          : {(balance / BALANCE - 1) * 100:+.2f}%")

    print()
    print(f"  {S2}")
    print("  -- By Silver Bullet Window --")
    for wlbl, g in tdf.groupby("window"):
        w  = (g["result"] == "WIN").sum()
        l  = (g["result"] == "LOSS").sum()
        wg = w / (w + l) * 100 if (w + l) > 0 else 0.0
        wp = g[g["result"] == "WIN"]["pts"].mean() if w > 0 else 0.0
        lp = g[g["result"] == "LOSS"]["pts"].mean() if l > 0 else 0.0
        print(f"    {wlbl:<10}  {len(g):3d} trades  {w}W/{l}L  "
              f"WR={wg:.0f}%  net={g['pts'].sum():+.1f} pts  "
              f"avg_win=+{wp:.1f}  avg_loss={lp:.1f}")

    print()
    print("  -- By Direction --")
    for d, g in tdf.groupby("direction"):
        w  = (g["result"] == "WIN").sum()
        l  = (g["result"] == "LOSS").sum()
        wg = w / (w + l) * 100 if (w + l) > 0 else 0.0
        print(f"    {d:<5}  {len(g):3d} trades  {w}W/{l}L  "
              f"WR={wg:.0f}%  net={g['pts'].sum():+.1f} pts")

    print()
    print("  -- Monthly Breakdown --")
    tdf["month"] = pd.to_datetime(tdf["entry_dt"]).dt.to_period("M")
    for m, g in tdf.groupby("month"):
        w  = (g["result"] == "WIN").sum()
        l  = (g["result"] == "LOSS").sum()
        wg = w / (w + l) * 100 if (w + l) > 0 else 0.0
        net = g["pts"].sum()
        bar = "+" * w + "-" * l
        print(f"    {str(m):<8}  {w}W/{l}L  WR={wg:.0f}%  net={net:+.1f}  [{bar}]")

    print()
    print("  -- SL Size Distribution --")
    bins = [0, 10, 25, 50, 100, 250, 9999]
    for bucket, g in tdf.groupby(pd.cut(tdf["sl_pts"], bins=bins)):
        if len(g) == 0:
            continue
        w  = (g["result"] == "WIN").sum()
        l  = (g["result"] == "LOSS").sum()
        wg = w / (w + l) * 100 if (w + l) > 0 else 0.0
        print(f"    SL {str(bucket):<18}  {len(g):3d} trades  "
              f"WR={wg:.0f}%  net={g['pts'].sum():+.1f} pts")

    # Equity curve
    plt.figure(figsize=(13, 5), facecolor="#0d1117")
    ax = plt.gca()
    ax.set_facecolor("#0d1117")
    ax.plot(eq_arr, color="#FFD700", linewidth=1.3, zorder=3)
    ax.fill_between(range(len(eq_arr)), BALANCE, eq_arr,
                    where=eq_arr >= BALANCE, alpha=0.18, color="#26a69a")
    ax.fill_between(range(len(eq_arr)), BALANCE, eq_arr,
                    where=eq_arr <  BALANCE, alpha=0.18, color="#ef5350")
    ax.axhline(BALANCE, color="#555566", linewidth=0.8, linestyle="--")
    ax.set_title(f"ICT Silver Bullet -- {name} -- Equity Curve (2:1 RR)",
                 color="white", fontsize=11)
    ax.set_xlabel("Trade #", color="#aaaaaa")
    ax.set_ylabel("Balance ($)", color="#aaaaaa")
    ax.tick_params(colors="#aaaaaa")
    ax.grid(True, color="#222233", linewidth=0.5)
    for sp in ax.spines.values():
        sp.set_color("#333344")
    plt.tight_layout()
    img = f"sb_{short}_equity.png"
    plt.savefig(img, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print()
    print(f"  Saved: {csv}, {img}")

    return {
        "name": name, "total": total, "wins": wins, "losses": losses,
        "wr": wr, "pf": pf, "net_pt": net_pt, "net_u": net_u,
        "max_dd": max_dd, "balance": balance,
    }

# ── Run all instruments ───────────────────────────────────────────────────────
print("ICT Silver Bullet Strategy Backtest")
print("Windows (NY time):  London 03-04  |  NY AM 10-11  |  NY PM 14-15")
print("Bias filter: 4H momentum (close vs close 2 bars ago)")
print("Entry: first M5 FVG in bias direction -> retrace into zone")
print("SL: below/above left candle of FVG  |  TP: 2:1 RR  |  Min TP: 10 pts")
print()

summary = []
for cfg in INSTRUMENTS:
    r = run_instrument(cfg)
    if r:
        summary.append(r)

if len(summary) > 1:
    SEP = "=" * 72
    print()
    print(SEP)
    print("  SUMMARY -- ICT Silver Bullet -- All Instruments")
    print(SEP)
    print(f"  {'Instrument':<22} {'Trd':>5} {'WR%':>6} {'PF':>6} "
          f"{'Net Pts':>10} {'MaxDD%':>8} {'Return%':>8}")
    print(f"  {'-'*68}")
    for r in summary:
        ret = (r["balance"] / BALANCE - 1) * 100
        print(f"  {r['name']:<22} {r['total']:>5} {r['wr']:>5.1f}% "
              f"{r['pf']:>6.2f} {r['net_pt']:>+10.1f} "
              f"{r['max_dd']:>7.2f}% {ret:>+7.2f}%")
    print(SEP)
