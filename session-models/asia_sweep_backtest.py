"""
Asia Sweep Strategy Backtest  (ClearLQ)
Instruments : DEUIDX (DAX), EURUSD, GBPUSD  -- M5

Rules (from PDF):
  STEP 1 - Asia Range
    - Asia session  : 22:00 - 04:30 UTC  (= 18:00-00:30 UTC-4 / EDT)
    - London Killzone : 06:00 - 09:00 UTC (= 02:00-05:00 UTC-4)

  STEP 2 - Sweep confirmation
    - During London KZ, price spikes ABOVE Asia high but CLOSES back below it -> bearish sweep -> SHORT
    - During London KZ, price spikes BELOW Asia low  but CLOSES back above it -> bullish sweep -> LONG
    - "Fast move out AND back in" -> captured by close being back inside range

  STEP 3 - Entry (Unicorn / FVG simplified)
    - After the sweep candle, find the FIRST M5 FVG in the reversal direction (still within London KZ)
      Bullish FVG  : lo[i] > hi[i-2]   (for long)
      Bearish FVG  : hi[i] < lo[i-2]   (for short)
    - Enter when price retraces INTO the FVG within London KZ
    - SL  : beyond the sweep extreme + buffer

  STEP 4 - TP & Invalidation
    - Primary TP : opposite Asia level (Asia high for longs swept below low; Asia low for shorts swept above high)
      - If the Asia target gives < 1.0× risk distance -> skip (not enough room)
      - If > 4× risk -> cap to 3:1 RR (don't get greedy on tiny SLs)
    - One trade per day.  No new entries after KZ closes.

Practical parameters:
    - RISK_PCT   = 1% of balance per trade (compounding)
    - MIN_TP_PTS = instrument-specific minimum TP distance
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Constants ────────────────────────────────────────────────────────────────
BALANCE  = 10_000.0
RISK_PCT = 0.01
MAX_RR   = 3.0      # cap TP if Asia level is very far

INSTRUMENTS = [
    {
        "name":    "DEUIDX (DAX)",
        "short":   "DEUIDX",
        "path":    r"C:\Users\Ayaan\Downloads\DEUIDXEUR_M5 (3).csv",
        "sl_buf":  10.0,
        "min_tp":  10.0,   # pts
        "pip_fmt": ".1f",
    },
    {
        "name":    "EURUSD",
        "short":   "EURUSD",
        "path":    r"C:\Users\Ayaan\Downloads\EURUSD_M5.csv",
        "sl_buf":  0.0005,
        "min_tp":  0.0010,
        "pip_fmt": ".5f",
    },
    {
        "name":    "GBPUSD",
        "short":   "GBPUSD",
        "path":    r"C:\Users\Ayaan\Downloads\GBPUSD_M5.csv",
        "sl_buf":  0.0005,
        "min_tp":  0.0010,
        "pip_fmt": ".5f",
    },
]

# UTC hours for the sessions
ASIA_START_H  = 22   # 22:00 UTC prev day
ASIA_END_H    = 4    # 04:30 UTC
ASIA_END_M    = 30
LKZ_START_H   = 6    # 06:00 UTC  London KZ open
LKZ_END_H     = 9    # 09:00 UTC  London KZ close


def run(cfg):
    name   = cfg["name"]
    short  = cfg["short"]
    SL_BUF = cfg["sl_buf"]
    MIN_TP = cfg["min_tp"]

    print()
    print("=" * 66)
    print(f"  {name}  -- Asia Sweep Strategy")
    print("=" * 66)

    try:
        df = pd.read_csv(cfg["path"], header=None,
                         names=["datetime", "open", "high", "low", "close", "volume"])
    except FileNotFoundError:
        print(f"  File not found: {cfg['path']}")
        return None

    df["datetime"] = pd.to_datetime(df["datetime"])
    # Assume data is UTC
    df = df.sort_values("datetime").reset_index(drop=True)
    print(f"  {len(df):,} candles | {df['datetime'].iloc[0].date()} -> {df['datetime'].iloc[-1].date()}")

    hi  = df["high"].values
    lo  = df["low"].values
    cl  = df["close"].values
    op  = df["open"].values
    dts = df["datetime"].values
    N   = len(df)

    hours   = df["datetime"].dt.hour.values
    minutes = df["datetime"].dt.minute.values
    dates   = df["datetime"].dt.date.values

    # ── Build Asia range per trading day ─────────────────────────────────────
    # For a given London KZ day D (UTC date), Asia range = candles where:
    #   - date == D-1  AND  hour >= 22     (22:00-23:55 prev day)
    #   - date == D    AND  (hour < 4 OR (hour == 4 AND min <= 30))
    all_dates = sorted(set(dates))
    asia_ranges = {}     # date -> (asia_high, asia_low, asia_high_time, asia_low_time)

    for d in all_dates:
        prev_d = pd.Timestamp(d) - pd.Timedelta(days=1)
        prev_d = prev_d.date()
        # prev day 22:00+
        mask_prev = (dates == prev_d) & (hours >= ASIA_START_H)
        # current day up to 04:30
        mask_curr = (dates == d) & (
            (hours < ASIA_END_H) | ((hours == ASIA_END_H) & (minutes <= ASIA_END_M))
        )
        mask = mask_prev | mask_curr
        if mask.sum() < 3:
            continue
        asia_hi = hi[mask].max()
        asia_lo = lo[mask].min()
        asia_ranges[d] = (asia_hi, asia_lo)

    print(f"  Asia ranges computed for {len(asia_ranges)} days")

    # ── Main backtest ─────────────────────────────────────────────────────────
    balance       = BALANCE
    equity        = [BALANCE]
    trades        = []
    active_trade  = None

    # Per-day state
    day_state = {}   # date -> dict

    for i in range(2, N):
        dt  = pd.Timestamp(dts[i])
        day = dates[i]
        h   = hours[i]
        m   = minutes[i]

        # ── Check active trade ────────────────────────────────────────────────
        if active_trade is not None:
            d     = active_trade["direction"]
            sl    = active_trade["sl"]
            tp    = active_trade["tp"]
            hit_sl = (hi[i] >= sl) if d == "SELL" else (lo[i] <= sl)
            hit_tp = (lo[i] <= tp) if d == "SELL" else (hi[i] >= tp)

            if hit_sl or hit_tp:
                result  = "LOSS" if hit_sl else "WIN"
                xp      = sl if hit_sl else tp
                ep      = active_trade["entry"]
                pnl_pt  = (ep - xp) if d == "SELL" else (xp - ep)
                rsk_pt  = abs(ep - sl)
                pnl_u   = (pnl_pt / rsk_pt) * active_trade["risk_amt"]
                balance += pnl_u
                equity.append(balance)
                dur = int((dt - active_trade["entry_dt"]).total_seconds() / 60)
                trades.append({
                    "entry_dt":    active_trade["entry_dt"],
                    "direction":   d,
                    "entry":       round(ep, 6),
                    "sl":          round(sl, 6),
                    "tp":          round(tp, 6),
                    "sweep_side":  active_trade["sweep_side"],
                    "result":      result,
                    "pts":         round(pnl_pt, 6),
                    "sl_pts":      round(rsk_pt, 6),
                    "pnl_usd":     round(pnl_u, 2),
                    "dur_m":       dur,
                    "asia_hi":     round(active_trade["asia_hi"], 6),
                    "asia_lo":     round(active_trade["asia_lo"], 6),
                })
                active_trade = None

        # ── Only act within London KZ ─────────────────────────────────────────
        in_lkz = (h >= LKZ_START_H) and (h < LKZ_END_H)
        if not in_lkz:
            continue

        if day not in asia_ranges:
            continue

        asia_hi, asia_lo = asia_ranges[day]
        asia_range_size  = asia_hi - asia_lo
        if asia_range_size <= 0:
            continue

        # Init day state
        if day not in day_state:
            day_state[day] = {
                "sweep_dir":    None,    # "BUY" or "SELL" or None
                "sweep_candle": None,    # index of sweep candle
                "sweep_ext":    None,    # extreme price of the sweep
                "fvg_found":    False,
                "fvg_top":      None,
                "fvg_bot":      None,
                "fvg_i":        None,
                "traded":       False,
            }
        st = day_state[day]

        if st["traded"]:
            continue

        # ── STEP 2: Detect sweep ──────────────────────────────────────────────
        if st["sweep_dir"] is None:
            # Bearish sweep: spike above Asia high but close back below it
            if hi[i] > asia_hi and cl[i] < asia_hi:
                st["sweep_dir"]    = "SELL"
                st["sweep_candle"] = i
                st["sweep_ext"]    = hi[i]   # spike high above Asia high
                continue  # wait for FVG on NEXT candles

            # Bullish sweep: spike below Asia low but close back above it
            elif lo[i] < asia_lo and cl[i] > asia_lo:
                st["sweep_dir"]    = "BUY"
                st["sweep_candle"] = i
                st["sweep_ext"]    = lo[i]   # spike low below Asia low
                continue

        # ── STEP 3: After sweep, look for FVG ────────────────────────────────
        if st["sweep_dir"] is not None and not st["fvg_found"]:
            bias = st["sweep_dir"]
            if bias == "BUY":   # bullish FVG
                if lo[i] > hi[i - 2]:
                    fvg_top = lo[i]
                    fvg_bot = hi[i - 2]
                    st.update({"fvg_found": True, "fvg_top": fvg_top,
                               "fvg_bot": fvg_bot, "fvg_i": i})
            else:               # bearish FVG
                if hi[i] < lo[i - 2]:
                    fvg_bot = hi[i]
                    fvg_top = lo[i - 2]
                    st.update({"fvg_found": True, "fvg_top": fvg_top,
                               "fvg_bot": fvg_bot, "fvg_i": i})

        # ── STEP 3: FVG found, wait for retrace entry ─────────────────────────
        if st["fvg_found"] and active_trade is None and not st["traded"]:
            direction = st["sweep_dir"]
            fvg_top   = st["fvg_top"]
            fvg_bot   = st["fvg_bot"]

            triggered = False
            if direction == "BUY"  and lo[i] <= fvg_top:
                triggered = True
            elif direction == "SELL" and hi[i] >= fvg_bot:
                triggered = True

            if triggered:
                ep  = cl[i]
                sl  = (st["sweep_ext"] - SL_BUF) if direction == "BUY" \
                      else (st["sweep_ext"] + SL_BUF)
                risk = abs(ep - sl)

                if risk <= 0:
                    continue

                # TP = opposite Asia level; cap at MAX_RR × risk; min = 1R
                if direction == "BUY":
                    asia_tp  = asia_hi
                    tp_dist  = asia_tp - ep
                else:
                    asia_tp  = asia_lo
                    tp_dist  = ep - asia_tp

                if tp_dist < risk:          # less than 1:1 - skip
                    st["traded"] = True
                    continue

                # Cap
                actual_tp_dist = min(tp_dist, MAX_RR * risk)
                tp = (ep + actual_tp_dist) if direction == "BUY" \
                     else (ep - actual_tp_dist)

                if actual_tp_dist < MIN_TP:
                    st["traded"] = True
                    continue

                active_trade = dict(
                    direction=direction, entry=ep, sl=sl, tp=tp,
                    entry_dt=dt, sweep_side=st["sweep_dir"],
                    risk_amt=balance * RISK_PCT,
                    asia_hi=asia_hi, asia_lo=asia_lo,
                )
                st["traded"] = True

    # ── Handle open trade at EOD ──────────────────────────────────────────────
    if active_trade is not None:
        ep     = active_trade["entry"]
        d      = active_trade["direction"]
        xp     = cl[-1]
        pnl_pt = (ep - xp) if d == "SELL" else (xp - ep)
        rsk_pt = abs(ep - active_trade["sl"])
        pnl_u  = (pnl_pt / rsk_pt) * active_trade["risk_amt"]
        balance += pnl_u
        equity.append(balance)
        trades.append({
            "entry_dt": active_trade["entry_dt"], "direction": d,
            "entry": round(ep, 6), "sl": round(active_trade["sl"], 6),
            "tp": round(active_trade["tp"], 6), "sweep_side": active_trade["sweep_side"],
            "result": "OPEN", "pts": round(pnl_pt, 6), "sl_pts": round(rsk_pt, 6),
            "pnl_usd": round(pnl_u, 2),
            "dur_m": int((pd.Timestamp(dts[-1]) - active_trade["entry_dt"]).total_seconds() / 60),
            "asia_hi": round(active_trade["asia_hi"], 6),
            "asia_lo": round(active_trade["asia_lo"], 6),
        })

    # ── Results ───────────────────────────────────────────────────────────────
    if not trades:
        print("  No trades generated.")
        return None

    tdf   = pd.DataFrame(trades)
    csv_f = f"asia_sweep_{short}_trades.csv"
    tdf.to_csv(csv_f, index=False)

    wins   = (tdf["result"] == "WIN").sum()
    losses = (tdf["result"] == "LOSS").sum()
    opens  = (tdf["result"] == "OPEN").sum()
    total  = len(tdf)
    wr     = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0.0
    gp     = tdf[tdf["pnl_usd"] > 0]["pnl_usd"].sum()
    gl     = tdf[tdf["pnl_usd"] < 0]["pnl_usd"].abs().sum()
    pf     = gp / gl if gl > 0 else float("inf")
    net_u  = tdf["pnl_usd"].sum()
    w_pts  = tdf[tdf["result"] == "WIN"]["pts"]
    l_pts  = tdf[tdf["result"] == "LOSS"]["pts"]

    eq_arr   = np.array(equity)
    roll_max = np.maximum.accumulate(eq_arr)
    max_dd   = ((eq_arr - roll_max) / roll_max * 100).min()

    print(f"  Total Trades    : {total}  ({wins}W / {losses}L / {opens} open)")
    print(f"  Win Rate        : {wr:.1f}%")
    print(f"  Profit Factor   : {pf:.2f}")
    print(f"  Max Drawdown    : {max_dd:.2f}%")
    print(f"  Starting Bal    : ${BALANCE:,.2f}")
    print(f"  Final Balance   : ${balance:,.2f}  (net {net_u:+,.2f})")
    print(f"  Return          : {(balance / BALANCE - 1) * 100:+.2f}%")
    if len(w_pts) > 0:
        fmt = cfg["pip_fmt"]
        print(f"  Avg Win         : +{w_pts.mean():{fmt}}")
        print(f"  Avg Loss        : {l_pts.mean():{fmt}}" if len(l_pts) > 0 else "  Avg Loss        : n/a")

    print()
    print("  -- By Sweep Direction --")
    for side, g in tdf.groupby("sweep_side"):
        w  = (g["result"] == "WIN").sum()
        l  = (g["result"] == "LOSS").sum()
        wg = w / (w + l) * 100 if (w + l) > 0 else 0.0
        print(f"    {side:<5}  {len(g):3d} trades  {w}W/{l}L  WR={wg:.0f}%  net=${g['pnl_usd'].sum():+,.2f}")

    print()
    print("  -- Monthly Breakdown --")
    tdf["month"] = pd.to_datetime(tdf["entry_dt"]).dt.to_period("M")
    for m, g in tdf.groupby("month"):
        w   = (g["result"] == "WIN").sum()
        l   = (g["result"] == "LOSS").sum()
        wg  = w / (w + l) * 100 if (w + l) > 0 else 0.0
        net = g["pnl_usd"].sum()
        bar = "+" * w + "-" * l
        print(f"    {str(m):<8}  {w}W/{l}L  WR={wg:.0f}%  net=${net:+,.2f}  [{bar}]")

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
    ax.set_title(f"Asia Sweep -- {name} -- Equity Curve", color="white", fontsize=11)
    ax.set_xlabel("Trade #", color="#aaaaaa")
    ax.set_ylabel("Balance ($)", color="#aaaaaa")
    ax.tick_params(colors="#aaaaaa")
    ax.grid(True, color="#222233", linewidth=0.5)
    for sp in ax.spines.values():
        sp.set_color("#333344")
    plt.tight_layout()
    img = f"asia_sweep_{short}_equity.png"
    plt.savefig(img, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print()
    print(f"  Saved: {csv_f}, {img}")

    return {
        "name": name, "short": short, "total": total, "wins": int(wins),
        "losses": int(losses), "wr": wr, "pf": pf, "max_dd": max_dd,
        "balance": balance, "net_u": net_u,
    }


# ── Run ───────────────────────────────────────────────────────────────────────
print("Asia Sweep Strategy Backtest  (ClearLQ)")
print("Asia Range  : 22:00-04:30 UTC  |  London KZ: 06:00-09:00 UTC")
print("Sweep: price spikes beyond Asia H/L and CLOSES back inside range")
print("Entry: first FVG in reversal direction -> retrace into FVG")
print("SL: beyond sweep extreme  |  TP: opposite Asia level (capped at 3:1 RR)")
print()

summary = []
for cfg in INSTRUMENTS:
    r = run(cfg)
    if r:
        summary.append(r)

if len(summary) > 1:
    SEP = "=" * 72
    print()
    print(SEP)
    print("  SUMMARY -- Asia Sweep -- All Instruments")
    print(SEP)
    print(f"  {'Instrument':<20} {'Trd':>5} {'WR%':>6} {'PF':>6} "
          f"{'MaxDD%':>8} {'Net $':>10} {'Return%':>8}")
    print(f"  {'-'*68}")
    for r in summary:
        ret = (r["balance"] / BALANCE - 1) * 100
        print(f"  {r['name']:<20} {r['total']:>5} {r['wr']:>5.1f}% "
              f"{r['pf']:>6.2f} {r['max_dd']:>7.2f}% "
              f"{r['net_u']:>+10.2f} {ret:>+7.2f}%")
    print(SEP)
