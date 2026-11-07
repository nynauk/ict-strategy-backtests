"""
Sweep -> MSS -> FVG Backtest -- EURUSD M5
No session filter. Runs all sessions 24/5.

SELL: 5m candle HIGH sweeps PDH / PWH / PMH
      -> 5m close below pre-sweep swing low  (bearish MSS)
      -> bearish FVG forms (hi[j] < lo[j-2]) after MSS
      -> price retraces UP into FVG          (entry short at close)
      SL = sweep candle high + 3 pip buffer
      TP = opposing liquidity (PDL / PWL / PML)

BUY:  5m candle LOW  sweeps PDL / PWL / PML
      -> 5m close above pre-sweep swing high (bullish MSS)
      -> bullish FVG forms (lo[j] > hi[j-2]) after MSS
      -> price retraces DOWN into FVG        (entry long at close)
      SL = sweep candle low - 3 pip buffer
      TP = opposing liquidity (PDH / PWH / PMH)
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Constants ────────────────────────────────────────────────────────────────
PIP           = 0.0001
SL_BUFFER     = 3 * PIP    # 3 pips buffer above/below sweep candle for SL
BALANCE       = 10_000.0
RISK_PCT      = 0.01
DATA_PATH     = r"C:\Users\Ayaan\Downloads\EURUSD_M5.csv"
MSS_LOOKBACK  = 20    # candles before sweep to define MSS reference level
MSS_TIMEOUT   = 150   # max candles to wait for MSS after sweep
FVG_TIMEOUT   = 80    # max candles to wait for first FVG after MSS
ENTRY_TIMEOUT = 80    # max candles to wait for FVG retrace entry

# ── Load data ────────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv(DATA_PATH, header=None,
                 names=["datetime","open","high","low","close","volume"])
df["datetime"] = pd.to_datetime(df["datetime"])
df = df.sort_values("datetime").reset_index(drop=True)
print(f"  {len(df):,} candles | {df['datetime'].iloc[0]} to {df['datetime'].iloc[-1]}")

hi  = df["high"].values
lo  = df["low"].values
op  = df["open"].values
cl  = df["close"].values
dts = df["datetime"].values
N   = len(df)

# ── Pre-compute previous period H/L ─────────────────────────────────────────
print("Pre-computing period levels...")

def resample_hl(df, freq):
    return (df.set_index("datetime")
              .resample(freq)
              .agg(H=("high","max"), L=("low","min"))
              .dropna())

daily_raw   = resample_hl(df, "1D")
weekly_raw  = resample_hl(df, "W-MON")
monthly_raw = resample_hl(df, "MS")

def hl_dict(raw):
    """period_start.date() -> (H, L)"""
    return {t.date(): (row.H, row.L) for t, row in raw.iterrows()}

daily_hl   = hl_dict(daily_raw)
weekly_hl  = hl_dict(weekly_raw)
monthly_hl = hl_dict(monthly_raw)

def prev_hl_dict(hl):
    """current_period_date -> (prev_period_H, prev_period_L)"""
    dates = sorted(hl.keys())
    return {dates[i]: hl[dates[i-1]] for i in range(1, len(dates))}

prev_daily   = prev_hl_dict(daily_hl)
prev_weekly  = prev_hl_dict(weekly_hl)
prev_monthly = prev_hl_dict(monthly_hl)

def week_start(dt):
    return (dt - pd.Timedelta(days=dt.dayofweek)).date()

def month_start(dt):
    return dt.replace(day=1).date()

# ── Main backtest loop ───────────────────────────────────────────────────────
print("Running backtest...")

state         = "idle"
direction     = None
swept_label   = None
swept_level   = None   # PDH/PWH/PMH value that was swept
sweep_hi      = None   # actual high of sweep candle  (SL reference for sells)
sweep_lo      = None   # actual low  of sweep candle  (SL reference for buys)
opposing_tp   = None   # TP level
mss_ref       = None   # price level to break for MSS confirmation
sweep_i       = None
mss_i         = None
fvg_j         = None
fvg_bot       = None
fvg_top       = None
state_since   = None   # candle index when current state was entered

active_trade  = None
trades        = []
balance       = BALANCE
equity        = [BALANCE]

STATE_TIMEOUTS = {
    "swept_sell": MSS_TIMEOUT,   "swept_buy": MSS_TIMEOUT,
    "mss_sell":   FVG_TIMEOUT,   "mss_buy":   FVG_TIMEOUT,
    "fvg_sell":   ENTRY_TIMEOUT, "fvg_buy":   ENTRY_TIMEOUT,
}

for i in range(2, N):
    dt  = pd.Timestamp(dts[i])
    day = dt.date()

    # ── Manage active trade (SL checked before TP) ───────────────────────────
    if active_trade is not None:
        if active_trade["direction"] == "SELL":
            hit_sl = hi[i] >= active_trade["sl"]
            hit_tp = lo[i] <= active_trade["tp"]
        else:
            hit_sl = lo[i] <= active_trade["sl"]
            hit_tp = hi[i] >= active_trade["tp"]

        if hit_sl or hit_tp:
            result     = "LOSS" if hit_sl else "WIN"
            exit_price = active_trade["sl"] if hit_sl else active_trade["tp"]
            entry_p    = active_trade["entry"]

            if active_trade["direction"] == "SELL":
                pnl_pips = (entry_p - exit_price) / PIP
            else:
                pnl_pips = (exit_price - entry_p) / PIP

            risk_pips = abs(entry_p - active_trade["sl"]) / PIP
            rr_actual = abs((active_trade["tp"] - entry_p) /
                            (active_trade["sl"] - entry_p))
            pnl_usd   = (pnl_pips / risk_pips) * active_trade["risk_amt"]
            balance  += pnl_usd
            equity.append(balance)

            duration = int((dt - active_trade["entry_dt"]).total_seconds() / 60)
            trades.append({
                "entry_dt":    active_trade["entry_dt"],
                "direction":   active_trade["direction"],
                "level":       active_trade["label"],
                "entry":       round(entry_p, 5),
                "sl":          round(active_trade["sl"], 5),
                "tp":          round(active_trade["tp"], 5),
                "opposing_liq":round(active_trade.get("opposing_liq", float("nan")), 5),
                "result":      result,
                "pips":        round(pnl_pips, 1),
                "sl_pips":     round(risk_pips, 1),
                "rr":          2.0,
                "pnl_usd":     round(pnl_usd, 2),
                "duration_m":  duration,
            })
            active_trade = None
        continue   # skip setup logic while in trade

    # ── Get current period levels ────────────────────────────────────────────
    pdh, pdl = prev_daily.get(day, (np.nan, np.nan))
    ws = week_start(dt)
    pwh, pwl = prev_weekly.get(ws, (np.nan, np.nan))
    ms = month_start(dt)
    pmh, pml = prev_monthly.get(ms, (np.nan, np.nan))

    # ── Check state timeout ──────────────────────────────────────────────────
    if state != "idle" and state_since is not None:
        if (i - state_since) > STATE_TIMEOUTS.get(state, 9999):
            state = "idle"

    # ════════════════════════════════════════════════════════════════════════
    # STATE: idle — scan for sweep
    # ════════════════════════════════════════════════════════════════════════
    if state == "idle":
        # SELL sweep — priority: monthly > weekly > daily
        sell_found = False
        for lvl, tp, lbl in [(pmh, pml, "PMH"),
                              (pwh, pwl, "PWH"),
                              (pdh, pdl, "PDH")]:
            if not np.isnan(lvl) and not np.isnan(tp) and hi[i] > lvl:
                state       = "swept_sell"
                direction   = "SELL"
                swept_label = lbl
                swept_level = lvl
                sweep_hi    = hi[i]
                opposing_tp = tp
                mss_ref     = float(np.min(lo[max(0, i - MSS_LOOKBACK):i + 1]))
                sweep_i     = i
                state_since = i
                sell_found  = True
                break

        if not sell_found:
            # BUY sweep — priority: monthly > weekly > daily
            for lvl, tp, lbl in [(pml, pmh, "PML"),
                                  (pwl, pwh, "PWL"),
                                  (pdl, pdh, "PDL")]:
                if not np.isnan(lvl) and not np.isnan(tp) and lo[i] < lvl:
                    state       = "swept_buy"
                    direction   = "BUY"
                    swept_label = lbl
                    swept_level = lvl
                    sweep_lo    = lo[i]
                    opposing_tp = tp
                    mss_ref     = float(np.max(hi[max(0, i - MSS_LOOKBACK):i + 1]))
                    sweep_i     = i
                    state_since = i
                    break

    # ════════════════════════════════════════════════════════════════════════
    # STATE: swept — wait for MSS
    # ════════════════════════════════════════════════════════════════════════
    elif state == "swept_sell":
        # MSS bearish: 5m close below pre-sweep swing low
        if cl[i] < mss_ref:
            state       = "mss_sell"
            mss_i       = i
            state_since = i

    elif state == "swept_buy":
        # MSS bullish: 5m close above pre-sweep swing high
        if cl[i] > mss_ref:
            state       = "mss_buy"
            mss_i       = i
            state_since = i

    # ════════════════════════════════════════════════════════════════════════
    # STATE: mss_confirmed — look for FVG
    # ════════════════════════════════════════════════════════════════════════
    elif state == "mss_sell":
        # Bearish FVG: gap between candle[i-2].low and candle[i].high
        # (candle[i].high < candle[i-2].low — downward gap)
        if hi[i] < lo[i - 2]:
            fvg_j       = i
            fvg_bot     = hi[i]       # lower edge of gap (sell entry zone bottom)
            fvg_top     = lo[i - 2]   # upper edge of gap
            state       = "fvg_sell"
            state_since = i

    elif state == "mss_buy":
        # Bullish FVG: gap between candle[i-2].high and candle[i].low
        # (candle[i].low > candle[i-2].high — upward gap)
        if lo[i] > hi[i - 2]:
            fvg_j       = i
            fvg_top     = lo[i]       # upper edge of gap (buy entry zone top)
            fvg_bot     = hi[i - 2]   # lower edge of gap
            state       = "fvg_buy"
            state_since = i

    # ════════════════════════════════════════════════════════════════════════
    # STATE: fvg_formed — wait for retrace into FVG
    # ════════════════════════════════════════════════════════════════════════
    elif state == "fvg_sell":
        # Retrace: price rallies back up, candle HIGH reaches FVG bottom
        if hi[i] >= fvg_bot:
            entry_p  = cl[i]
            sl       = sweep_hi + SL_BUFFER
            risk     = sl - entry_p
            tp       = entry_p - 2.0 * risk   # fixed 2:1 RR
            if risk > 0:
                active_trade = dict(
                    direction="SELL", entry=entry_p, sl=sl, tp=tp,
                    entry_dt=dt, label=swept_label,
                    opposing_liq=opposing_tp,
                    risk_amt=balance * RISK_PCT,
                )
                state = "idle"

    elif state == "fvg_buy":
        # Retrace: price pulls back down, candle LOW reaches FVG top
        if lo[i] <= fvg_top:
            entry_p  = cl[i]
            sl       = sweep_lo - SL_BUFFER
            risk     = entry_p - sl
            tp       = entry_p + 2.0 * risk   # fixed 2:1 RR
            if risk > 0:
                active_trade = dict(
                    direction="BUY", entry=entry_p, sl=sl, tp=tp,
                    entry_dt=dt, label=swept_label,
                    opposing_liq=opposing_tp,
                    risk_amt=balance * RISK_PCT,
                )
                state = "idle"

# ── Handle trade still open at end of data ───────────────────────────────────
if active_trade is not None:
    exit_price = cl[-1]
    entry_p = active_trade["entry"]
    if active_trade["direction"] == "SELL":
        pnl_pips = (entry_p - exit_price) / PIP
    else:
        pnl_pips = (exit_price - entry_p) / PIP
    risk_pips = abs(entry_p - active_trade["sl"]) / PIP
    pnl_usd   = (pnl_pips / risk_pips) * active_trade["risk_amt"]
    balance  += pnl_usd
    equity.append(balance)
    trades.append({
        "entry_dt":    active_trade["entry_dt"],
        "direction":   active_trade["direction"],
        "level":       active_trade["label"],
        "entry":       round(entry_p, 5),
        "sl":          round(active_trade["sl"], 5),
        "tp":          round(active_trade["tp"], 5),
        "opposing_liq":round(active_trade.get("opposing_liq", float("nan")), 5),
        "result":      "OPEN",
        "pips":        round(pnl_pips, 1),
        "sl_pips":     round(risk_pips, 1),
        "rr":          2.0,
        "pnl_usd":     round(pnl_usd, 2),
        "duration_m":  int((pd.Timestamp(dts[-1]) -
                            active_trade["entry_dt"]).total_seconds() / 60),
    })

# ── Results ───────────────────────────────────────────────────────────────────
print()
SEP  = "=" * 62
SEP2 = "-" * 62

if not trades:
    print("No trades found. Check data path and period-level computation.")
else:
    tdf = pd.DataFrame(trades)
    tdf.to_csv("sweep_mss_fvg_trades.csv", index=False)

    wins   = (tdf["result"] == "WIN").sum()
    losses = (tdf["result"] == "LOSS").sum()
    opens  = (tdf["result"] == "OPEN").sum()
    total  = len(tdf)
    wr     = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0.0

    gross_p = tdf[tdf["pnl_usd"] > 0]["pnl_usd"].sum()
    gross_l = tdf[tdf["pnl_usd"] < 0]["pnl_usd"].abs().sum()
    pf      = gross_p / gross_l if gross_l > 0 else float("inf")
    net_pip = tdf["pips"].sum()
    net_usd = tdf["pnl_usd"].sum()

    win_pips  = tdf[tdf["result"] == "WIN"]["pips"]
    loss_pips = tdf[tdf["result"] == "LOSS"]["pips"]

    eq_arr   = np.array(equity)
    roll_max = np.maximum.accumulate(eq_arr)
    dd_pct   = (eq_arr - roll_max) / roll_max * 100
    max_dd   = dd_pct.min()

    print(SEP)
    print("  SWEEP -> MSS -> FVG  --  EURUSD M5  (all sessions, 2:1 RR)")
    print(SEP)
    print(f"  Total Trades    : {total}  ({wins}W / {losses}L / {opens} open)")
    print(f"  Win Rate        : {wr:.1f}%")
    print(f"  Profit Factor   : {pf:.2f}")
    print(f"  Net Pips        : {net_pip:+.1f}")
    print(f"  Max Drawdown    : {max_dd:.2f}%")
    print(f"  Avg Win Pips    : {win_pips.mean():.1f}" if len(win_pips) > 0 else "  Avg Win Pips    : n/a")
    print(f"  Avg Loss Pips   : {loss_pips.mean():.1f}" if len(loss_pips) > 0 else "  Avg Loss Pips   : n/a")
    print(f"  RR              : 2.0:1 (fixed)")
    print(f"  Starting Bal    : ${BALANCE:,.2f}")
    print(f"  Final Balance   : ${balance:,.2f}  (net {net_usd:+,.2f})")
    print(f"  Return          : {(balance / BALANCE - 1) * 100:+.2f}%")
    print()

    print("  -- By Swept Level --")
    for lbl, g in tdf.groupby("level"):
        w    = (g["result"] == "WIN").sum()
        l    = (g["result"] == "LOSS").sum()
        wr_g = w / (w + l) * 100 if (w + l) > 0 else 0.0
        print(f"    {lbl:<6}  {len(g):3d} trades  {w}W/{l}L  "
              f"WR={wr_g:.0f}%  net={g['pips'].sum():+.1f} pip  "
              f"avg_win={g[g['result']=='WIN']['pips'].mean():.1f} pip")
    print()

    print("  -- By Direction --")
    for d, g in tdf.groupby("direction"):
        w    = (g["result"] == "WIN").sum()
        l    = (g["result"] == "LOSS").sum()
        wr_g = w / (w + l) * 100 if (w + l) > 0 else 0.0
        print(f"    {d:<5}  {len(g):3d} trades  {w}W/{l}L  "
              f"WR={wr_g:.0f}%  net={g['pips'].sum():+.1f} pip")
    print()

    print("  -- SL / RR Distribution --")
    for bucket, g in tdf.groupby(pd.cut(tdf["sl_pips"], bins=[0,5,10,15,20,30,50,999])):
        if len(g) == 0:
            continue
        w    = (g["result"] == "WIN").sum()
        l    = (g["result"] == "LOSS").sum()
        wr_g = w / (w + l) * 100 if (w + l) > 0 else 0.0
        print(f"    SL {str(bucket):<18}  {len(g):3d} trades  WR={wr_g:.0f}%  "
              f"net={g['pips'].sum():+.1f} pip")
    print()

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
    ax.set_title("Sweep -> MSS -> FVG  --  Equity Curve (EURUSD M5, all sessions)",
                 color="white", fontsize=11)
    ax.set_xlabel("Trade #", color="#aaaaaa")
    ax.set_ylabel("Balance ($)", color="#aaaaaa")
    ax.tick_params(colors="#aaaaaa")
    ax.grid(True, color="#222233", linewidth=0.5)
    for spine in ax.spines.values():
        spine.set_color("#333344")
    plt.tight_layout()
    plt.savefig("sweep_mss_fvg_equity.png", dpi=150,
                bbox_inches="tight", facecolor="#0d1117")
    plt.close()

    print(f"  Saved: sweep_mss_fvg_trades.csv  ({total} rows)")
    print("  Saved: sweep_mss_fvg_equity.png")
    print(SEP)
