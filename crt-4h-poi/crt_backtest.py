"""
CRT (Candle Range Theory) backtest - 4H range/manipulation/distribution model
gated by HTF (daily/weekly) points of interest.

Single file. Edit CONFIG, run: python crt_backtest.py

Data expected: raw M5 OHLC CSV, no header, cols = datetime,open,high,low,close,volume
(Dukascopy export format). M5 is used for intrabar fill/exit resolution; 4H and
daily/weekly series are built from it, anchored to the 18:00 ET futures session.
"""

import os
import sys
import itertools
import numpy as np
import pandas as pd

# ============================== CONFIG =====================================
CFG = dict(
    CSV            = "/mnt/user-data/uploads/Downloads/USATECHIDXUSD_M5.csv",
    CSV_TZ         = "UTC",                 # timezone of the raw timestamps
    SESSION_TZ     = "America/New_York",    # anchor tz for 4H / daily bars
    DAY_OPEN_HOUR  = 18,                    # 18:00 ET = CME session open
    OUT_DIR        = "/home/claude/out",

    DATE_FROM      = None,                  # e.g. "2023-06-01"
    DATE_TO        = None,

    # --- CRT definition -------------------------------------------------
    REQUIRE_C2_BODY_BACK   = True,   # C2 must CLOSE back inside the C1 range
    REQUIRE_C2_CLOSE_HALF  = False,  # stricter: C2 close in the far half of C1
    MIN_C1_RANGE_ATR       = 0.35,   # ignore dead C1 ranges (x ATR14 of 4H)
    MAX_C2_SWEEP_ATR       = 1.50,   # ignore blowout manipulation candles

    # --- HTF POI gate ---------------------------------------------------
    POI_TOL_ATR    = 0.15,   # how far outside a POI the sweep wick may sit
    FVG_MIN_ATR    = 0.10,   # minimum daily FVG height to be tradeable
    OB_BODY_ONLY   = False,  # daily OB zone = body only vs full candle range
    POI_LOOKBACK_D = 60,     # daily POIs older than this are dropped

    # --- Execution ------------------------------------------------------
    ENTRY_VALID_BARS = 2,     # 4H bars a 50% limit stays live (model B)
    SL_BUFFER_ATR    = 0.05,  # stop beyond the sweep wick
    MIN_STOP_PTS     = 10.0,  # skip setups with an absurdly tight stop
    MIN_RR_MODEL_A   = 0.50,  # skip classic-CRT setups whose range target < 0.5R
    FIXED_RR         = 2.0,   # model B target
    MAX_HOLD_BARS_4H = 12,    # time stop
    COST_PTS         = 1.0,   # round-trip spread + commission, index points
    ALLOW_OVERLAP    = False, # one position at a time
    RISK_USD         = 500.0, # $ risked per trade (1R) for the $ column
    SWING_K          = 2,     # daily fractal half-width (confirmation lag)

    # --- Sweeps to run --------------------------------------------------
    POI_MODES   = ["NONE", "PDH_PDL", "PWH_PWL", "DAILY_FVG", "DAILY_IFVG", "DAILY_OB", "ANY_POI"],
    ENTRY_MODES = ["A_CLOSE_RANGE", "B_MID_2R"],
)
# ===========================================================================


# ------------------------------- data --------------------------------------
def load_m5(cfg):
    df = pd.read_csv(cfg["CSV"], header=None,
                     names=["dt", "open", "high", "low", "close", "volume"],
                     parse_dates=["dt"])
    df = df.dropna().drop_duplicates("dt").sort_values("dt")
    idx = pd.DatetimeIndex(df["dt"]).tz_localize(cfg["CSV_TZ"]).tz_convert(cfg["SESSION_TZ"])
    df = df.drop(columns=["dt"]).set_index(idx.tz_localize(None))  # ET wall clock
    df.index.name = "et"
    if cfg["DATE_FROM"]:
        df = df[df.index >= cfg["DATE_FROM"]]
    if cfg["DATE_TO"]:
        df = df[df.index <= cfg["DATE_TO"]]
    return df


def session_date(index, day_open_hour):
    """CME-style session label: 18:00 ET Sun -> Mon session."""
    shift = 24 - day_open_hour
    return (index + pd.Timedelta(hours=shift)).floor("D")


OHLC = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}


def to_4h(m5, cfg):
    origin = m5.index[0].normalize() + pd.Timedelta(hours=cfg["DAY_OPEN_HOUR"])
    while origin > m5.index[0]:
        origin -= pd.Timedelta(days=1)
    b = m5.resample("4h", origin=origin, label="left", closed="left").agg(OHLC)
    n = m5.resample("4h", origin=origin, label="left", closed="left").size()
    b = b[n >= 6].dropna()                       # drop stubs (<30 min of data)
    b["sess"] = session_date(b.index, cfg["DAY_OPEN_HOUR"])
    return b


def to_daily(m5, cfg):
    sd = session_date(m5.index, cfg["DAY_OPEN_HOUR"])
    d = m5.groupby(sd).agg(OHLC)
    d.index.name = "sess"
    d["start"] = m5.groupby(sd).apply(lambda x: x.index[0], include_groups=False)
    d["end"] = m5.groupby(sd).apply(lambda x: x.index[-1], include_groups=False)
    return d


def to_weekly(daily):
    wk = daily.index.to_period("W-FRI")
    w = daily.groupby(wk).agg({"open": "first", "high": "max", "low": "min", "close": "last"})
    w["end_sess"] = daily.groupby(wk).apply(lambda x: x.index[-1], include_groups=False)
    return w


def atr(bars, n=14):
    pc = bars["close"].shift(1)
    tr = pd.concat([bars["high"] - bars["low"],
                    (bars["high"] - pc).abs(),
                    (bars["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=3).mean()


# ------------------------------- HTF POIs ----------------------------------
# A POI = dict(kind, side, lo, hi, valid_from(ts), valid_to(ts or None))
# side: +1 = support (bullish setups), -1 = resistance (bearish setups)

def build_level_pois(daily, weekly, cfg):
    """PDH/PDL and PWH/PWL as zero-width levels, live for the next session/week."""
    out = []
    sess = list(daily.index)
    for i in range(1, len(sess)):
        prev, cur = daily.loc[sess[i - 1]], daily.loc[sess[i]]
        vf, vt = cur["start"], cur["end"]
        out.append(dict(kind="PDH_PDL", side=-1, lo=prev["high"], hi=prev["high"],
                        valid_from=vf, valid_to=vt, label=f"PDH {sess[i-1].date()}"))
        out.append(dict(kind="PDH_PDL", side=+1, lo=prev["low"], hi=prev["low"],
                        valid_from=vf, valid_to=vt, label=f"PDL {sess[i-1].date()}"))

    wks = list(weekly.index)
    for i in range(1, len(wks)):
        prev = weekly.loc[wks[i - 1]]
        cur_sessions = daily[daily.index.to_period("W-FRI") == wks[i]]
        if cur_sessions.empty:
            continue
        vf, vt = cur_sessions["start"].iloc[0], cur_sessions["end"].iloc[-1]
        out.append(dict(kind="PWH_PWL", side=-1, lo=prev["high"], hi=prev["high"],
                        valid_from=vf, valid_to=vt, label=f"PWH {wks[i-1]}"))
        out.append(dict(kind="PWH_PWL", side=+1, lo=prev["low"], hi=prev["low"],
                        valid_from=vf, valid_to=vt, label=f"PWL {wks[i-1]}"))
    return out


def build_daily_fvg_pois(daily, datr, cfg):
    """Daily FVGs + their inversions (IFVG). Mitigation = full fill by a later daily bar."""
    fvgs, ifvgs = [], []
    sess = list(daily.index)
    for i in range(2, len(sess)):
        a, c = daily.loc[sess[i - 2]], daily.loc[sess[i]]
        a_ref = datr.get(sess[i], np.nan)
        if not np.isfinite(a_ref):
            continue
        cand = []
        if c["low"] > a["high"]:
            cand.append((+1, a["high"], c["low"]))        # bullish FVG = support
        if c["high"] < a["low"]:
            cand.append((-1, c["high"], a["low"]))        # bearish FVG = resistance
        for side, lo, hi in cand:
            if (hi - lo) < cfg["FVG_MIN_ATR"] * a_ref:
                continue
            vf = daily.loc[sess[i], "end"]                 # tradeable once the gap closes
            # find mitigation: first later session that fully fills the gap
            vt, inv_from = None, None
            for j in range(i + 1, min(len(sess), i + 1 + cfg["POI_LOOKBACK_D"])):
                d = daily.loc[sess[j]]
                filled = d["low"] <= lo if side == +1 else d["high"] >= hi
                if filled:
                    vt = d["end"]
                    inv_from = d["end"]
                    break
            if vt is None and i + 1 + cfg["POI_LOOKBACK_D"] < len(sess):
                vt = daily.loc[sess[min(len(sess) - 1, i + cfg["POI_LOOKBACK_D"])], "end"]
            fvgs.append(dict(kind="DAILY_FVG", side=side, lo=lo, hi=hi,
                             valid_from=vf, valid_to=vt, label=f"DFVG {sess[i].date()}"))
            if inv_from is not None:
                iv_to = None
                for j in range(len(sess)):
                    if daily.loc[sess[j], "end"] <= inv_from:
                        continue
                    d = daily.loc[sess[j]]
                    # inverted zone dies when price closes through it again
                    dead = d["close"] < lo if side == -1 else d["close"] > hi
                    if dead:
                        iv_to = d["end"]
                        break
                ifvgs.append(dict(kind="DAILY_IFVG", side=-side, lo=lo, hi=hi,
                                  valid_from=inv_from, valid_to=iv_to,
                                  label=f"DIFVG {sess[i].date()}"))
    return fvgs + ifvgs


def swings(bars, k=2):
    h, l = bars["high"].values, bars["low"].values
    sh = np.zeros(len(bars), bool)
    sl = np.zeros(len(bars), bool)
    for i in range(k, len(bars) - k):
        w = slice(i - k, i + k + 1)
        if h[i] == h[w].max() and (h[w] == h[i]).sum() == 1:
            sh[i] = True
        if l[i] == l[w].min() and (l[w] == l[i]).sum() == 1:
            sl[i] = True
    return sh, sl


def build_daily_ob_pois(daily, cfg):
    """Last opposing daily candle before a daily BOS.

    A fractal swing at bar j is only *confirmed* at bar j+k, so a swing is not
    eligible as a BOS reference until k bars after it printed - otherwise the
    backtest peeks forward.
    """
    out = []
    k = cfg["SWING_K"]
    sh, sl = swings(daily, k)
    sess = list(daily.index)
    o, c = daily["open"].values, daily["close"].values
    hi, lo = daily["high"].values, daily["low"].values
    last_sh = last_sl = None
    for i in range(len(sess)):
        jc = i - k                                  # swing confirmed this bar
        if jc >= 0:
            if sh[jc]:
                last_sh = jc
            if sl[jc]:
                last_sl = jc
        if last_sh is not None and c[i] > hi[last_sh]:          # bullish daily BOS
            j = next((x for x in range(i, max(-1, last_sh - 1), -1) if c[x] < o[x]), None)
            if j is not None:
                z_lo = min(o[j], c[j]) if cfg["OB_BODY_ONLY"] else lo[j]
                z_hi = max(o[j], c[j]) if cfg["OB_BODY_ONLY"] else hi[j]
                vt = None
                for m in range(i + 1, min(len(sess), i + 1 + cfg["POI_LOOKBACK_D"])):
                    if c[m] < z_lo:
                        vt = daily.loc[sess[m], "end"]
                        break
                out.append(dict(kind="DAILY_OB", side=+1, lo=z_lo, hi=z_hi,
                                valid_from=daily.loc[sess[i], "end"], valid_to=vt,
                                label=f"D+OB {sess[j].date()}"))
            last_sh = None
        if last_sl is not None and c[i] < lo[last_sl]:          # bearish daily BOS
            j = next((x for x in range(i, max(-1, last_sl - 1), -1) if c[x] > o[x]), None)
            if j is not None:
                z_lo = min(o[j], c[j]) if cfg["OB_BODY_ONLY"] else lo[j]
                z_hi = max(o[j], c[j]) if cfg["OB_BODY_ONLY"] else hi[j]
                vt = None
                for m in range(i + 1, min(len(sess), i + 1 + cfg["POI_LOOKBACK_D"])):
                    if c[m] > z_hi:
                        vt = daily.loc[sess[m], "end"]
                        break
                out.append(dict(kind="DAILY_OB", side=-1, lo=z_lo, hi=z_hi,
                                valid_from=daily.loc[sess[i], "end"], valid_to=vt,
                                label=f"D-OB {sess[j].date()}"))
            last_sl = None
    return out


def poi_hit(pois, kinds, side, sweep_px, ts, tol):
    """Does the sweep wick tag a live POI of the right kind and side?"""
    for p in pois:
        if p["kind"] not in kinds or p["side"] != side:
            continue
        if ts < p["valid_from"]:
            continue
        if p["valid_to"] is not None and ts > p["valid_to"]:
            continue
        if (p["lo"] - tol) <= sweep_px <= (p["hi"] + tol):
            return p
    return None


# ------------------------------ setups -------------------------------------
def find_setups(b4, a4, cfg):
    """C1 = range candle, C2 = manipulation candle. Returns raw (ungated) setups."""
    out = []
    h, l, o, c = b4["high"].values, b4["low"].values, b4["open"].values, b4["close"].values
    idx = b4.index
    for i in range(1, len(b4)):
        a = a4.iloc[i]
        if not np.isfinite(a) or a <= 0:
            continue
        c1_rng = h[i - 1] - l[i - 1]
        if c1_rng < cfg["MIN_C1_RANGE_ATR"] * a:
            continue

        # bullish: sweep C1 low, close back inside
        if l[i] < l[i - 1]:
            depth = l[i - 1] - l[i]
            ok = (not cfg["REQUIRE_C2_BODY_BACK"]) or (c[i] > l[i - 1])
            if cfg["REQUIRE_C2_CLOSE_HALF"]:
                ok = ok and c[i] > (l[i - 1] + h[i - 1]) / 2
            if ok and depth <= cfg["MAX_C2_SWEEP_ATR"] * a and c[i] < h[i - 1]:
                out.append(dict(i=i, dir=+1, t2=idx[i], sweep=l[i], c1_lo=l[i - 1],
                                c1_hi=h[i - 1], c2_hi=h[i], c2_lo=l[i], c2_c=c[i], atr=a))

        # bearish: sweep C1 high, close back inside
        if h[i] > h[i - 1]:
            depth = h[i] - h[i - 1]
            ok = (not cfg["REQUIRE_C2_BODY_BACK"]) or (c[i] < h[i - 1])
            if cfg["REQUIRE_C2_CLOSE_HALF"]:
                ok = ok and c[i] < (l[i - 1] + h[i - 1]) / 2
            if ok and depth <= cfg["MAX_C2_SWEEP_ATR"] * a and c[i] > l[i - 1]:
                out.append(dict(i=i, dir=-1, t2=idx[i], sweep=h[i], c1_lo=l[i - 1],
                                c1_hi=h[i - 1], c2_hi=h[i], c2_lo=l[i], c2_c=c[i], atr=a))
    return out


# ---------------------------- simulation -----------------------------------
def simulate(m5, entry_t, expiry_t, hard_t, direction, entry_px, sl, tp, limit_order, cost):
    """Walk M5 from entry_t. Conservative: SL wins a bar that touches both."""
    win = m5.loc[entry_t:hard_t]
    if win.empty:
        return None
    filled = not limit_order
    fill_t = entry_t if filled else None
    hi, lo = win["high"].values, win["low"].values
    ts = win.index
    for k in range(len(win)):
        if not filled:
            if ts[k] > expiry_t:
                return None
            touched = lo[k] <= entry_px if direction == +1 else hi[k] >= entry_px
            if touched:
                filled, fill_t = True, ts[k]
            else:
                continue
        hit_sl = lo[k] <= sl if direction == +1 else hi[k] >= sl
        hit_tp = hi[k] >= tp if direction == +1 else lo[k] <= tp
        if hit_sl and hit_tp:
            return dict(exit_t=ts[k], exit_px=sl, reason="SL", fill_t=fill_t)
        if hit_sl:
            return dict(exit_t=ts[k], exit_px=sl, reason="SL", fill_t=fill_t)
        if hit_tp:
            return dict(exit_t=ts[k], exit_px=tp, reason="TP", fill_t=fill_t)
    if not filled:
        return None
    return dict(exit_t=win.index[-1], exit_px=win["close"].iloc[-1], reason="TIME", fill_t=fill_t)


POI_KINDS = {
    "NONE":       None,
    "PDH_PDL":    {"PDH_PDL"},
    "PWH_PWL":    {"PWH_PWL"},
    "DAILY_FVG":  {"DAILY_FVG"},
    "DAILY_IFVG": {"DAILY_IFVG"},
    "DAILY_OB":   {"DAILY_OB"},
    "ANY_POI":    {"PDH_PDL", "PWH_PWL", "DAILY_FVG", "DAILY_IFVG", "DAILY_OB"},
}


def run_variant(setups, pois, b4, m5, poi_mode, entry_mode, cfg):
    kinds = POI_KINDS[poi_mode]
    idx = b4.index
    trades, busy_until = [], None

    for s in setups:
        i, d = s["i"], s["dir"]
        if i + 1 >= len(b4):
            continue
        t_entry_bar = idx[i + 1]                      # C3 opens here
        if not cfg["ALLOW_OVERLAP"] and busy_until is not None and t_entry_bar <= busy_until:
            continue

        tol = cfg["POI_TOL_ATR"] * s["atr"]
        poi_label = "none"
        if kinds is not None:
            p = poi_hit(pois, kinds, d, s["sweep"], s["t2"], tol)
            if p is None:
                continue
            poi_label = p["label"]

        buf = cfg["SL_BUFFER_ATR"] * s["atr"]
        if d == +1:
            sl = s["c2_lo"] - buf
            entry_px = s["c2_c"] if entry_mode == "A_CLOSE_RANGE" else (s["c2_hi"] + s["c2_lo"]) / 2
            risk = entry_px - sl
            tp = s["c1_hi"] if entry_mode == "A_CLOSE_RANGE" else entry_px + cfg["FIXED_RR"] * risk
        else:
            sl = s["c2_hi"] + buf
            entry_px = s["c2_c"] if entry_mode == "A_CLOSE_RANGE" else (s["c2_hi"] + s["c2_lo"]) / 2
            risk = sl - entry_px
            tp = s["c1_lo"] if entry_mode == "A_CLOSE_RANGE" else entry_px - cfg["FIXED_RR"] * risk

        if risk < cfg["MIN_STOP_PTS"]:
            continue
        rr_target = abs(tp - entry_px) / risk
        if entry_mode == "A_CLOSE_RANGE" and rr_target < cfg["MIN_RR_MODEL_A"]:
            continue

        limit = entry_mode != "A_CLOSE_RANGE"
        expiry = idx[min(len(idx) - 1, i + cfg["ENTRY_VALID_BARS"])]
        hard = idx[min(len(idx) - 1, i + cfg["MAX_HOLD_BARS_4H"])] + pd.Timedelta(hours=4)

        res = simulate(m5, t_entry_bar, expiry, hard, d, entry_px, sl, tp, limit, cfg["COST_PTS"])
        if res is None:
            continue
        gross = (res["exit_px"] - entry_px) * d
        net = gross - cfg["COST_PTS"]
        trades.append(dict(
            setup_time=s["t2"], entry_time=res["fill_t"], exit_time=res["exit_t"],
            dir="long" if d == +1 else "short", poi=poi_label,
            entry=round(entry_px, 2), sl=round(sl, 2), tp=round(tp, 2),
            risk_pts=round(risk, 2), target_R=round(rr_target, 2),
            exit=round(res["exit_px"], 2), reason=res["reason"],
            R=round(net / risk, 3), pts=round(net, 2),
            usd=round((net / risk) * cfg["RISK_USD"], 2),
        ))
        busy_until = res["exit_t"]

    return pd.DataFrame(trades)


# ------------------------------- stats -------------------------------------
def stats(tr, label):
    if tr.empty:
        return dict(variant=label, n=0)
    R = tr["R"]
    eq = R.cumsum()
    dd = (eq.cummax() - eq).max()
    wins, losses = R[R > 0], R[R <= 0]
    pf = wins.sum() / abs(losses.sum()) if losses.sum() != 0 else np.inf
    streak = mx = 0
    for r in R:
        streak = streak + 1 if r <= 0 else 0
        mx = max(mx, streak)
    se = R.std(ddof=1) / np.sqrt(len(R)) if len(R) > 1 else np.nan
    t = R.mean() / se if se and np.isfinite(se) and se > 0 else np.nan
    return dict(
        variant=label, n=len(tr),
        win_pct=round(100 * (R > 0).mean(), 1),
        exp_R=round(R.mean(), 3),
        SE=round(se, 3) if np.isfinite(se) else np.nan,
        t=round(t, 2) if np.isfinite(t) else np.nan,
        total_R=round(R.sum(), 1),
        PF=round(pf, 2) if np.isfinite(pf) else 99.0,
        maxDD_R=round(dd, 1),
        max_losses=mx,
        avg_win=round(wins.mean(), 2) if len(wins) else 0.0,
        avg_loss=round(losses.mean(), 2) if len(losses) else 0.0,
        long_n=int((tr["dir"] == "long").sum()),
        long_R=round(R[tr["dir"] == "long"].sum(), 1),
        short_R=round(R[tr["dir"] == "short"].sum(), 1),
        tp_pct=round(100 * (tr["reason"] == "TP").mean(), 1),
        usd=round(tr["usd"].sum(), 0),
    )


def yearly(tr):
    if tr.empty:
        return {}
    y = pd.to_datetime(tr["exit_time"]).dt.year
    g = tr.groupby(y)["R"]
    return {int(k): f"{v.sum():+.1f}R/{len(tr[y == k])}" for k, v in g}


# -------------------------------- main -------------------------------------
def main():
    cfg = CFG
    os.makedirs(cfg["OUT_DIR"], exist_ok=True)

    m5 = load_m5(cfg)
    b4 = to_4h(m5, cfg)
    daily = to_daily(m5, cfg)
    weekly = to_weekly(daily)
    a4 = atr(b4, 14)
    d_atr = atr(daily, 14)

    print(f"M5 bars    : {len(m5):,}   {m5.index[0]} -> {m5.index[-1]}  (ET)")
    print(f"4H bars    : {len(b4):,}")
    print(f"Daily bars : {len(daily):,}    Weeks: {len(weekly)}")

    pois = (build_level_pois(daily, weekly, cfg)
            + build_daily_fvg_pois(daily, d_atr, cfg)
            + build_daily_ob_pois(daily, cfg))
    kc = pd.Series([p["kind"] for p in pois]).value_counts()
    print("POIs built :", dict(kc))

    setups = find_setups(b4, a4, cfg)
    nb = sum(1 for s in setups if s["dir"] == +1)
    print(f"Raw CRT setups: {len(setups)}  ({nb} bullish / {len(setups)-nb} bearish)\n")

    rows, all_tr = [], {}
    for pm, em in itertools.product(cfg["POI_MODES"], cfg["ENTRY_MODES"]):
        label = f"{pm} | {em}"
        tr = run_variant(setups, pois, b4, m5, pm, em, cfg)
        all_tr[label] = tr
        rows.append(stats(tr, label))
        if not tr.empty:
            tr.to_csv(f"{cfg['OUT_DIR']}/trades_{pm}_{em}.csv", index=False)

    res = pd.DataFrame(rows).fillna(0)
    res = res.sort_values("exp_R", ascending=False)
    pd.set_option("display.width", 250, "display.max_columns", 50)
    print(res.to_string(index=False))
    res.to_csv(f"{cfg['OUT_DIR']}/summary.csv", index=False)

    print("\nYear-by-year (R / trades):")
    for label in res["variant"].head(6):
        print(f"  {label:34s} {yearly(all_tr[label])}")

    # equity curves
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
        for ax, em in zip(axes, cfg["ENTRY_MODES"]):
            for pm in cfg["POI_MODES"]:
                tr = all_tr[f"{pm} | {em}"]
                if len(tr) < 5:
                    continue
                ax.plot(pd.to_datetime(tr["exit_time"]), tr["R"].cumsum(),
                        label=f"{pm} (n={len(tr)})", lw=1.4)
            ax.axhline(0, color="k", lw=0.8)
            ax.set_title(em)
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8)
        axes[0].set_ylabel("Cumulative R")
        fig.suptitle("CRT 4H + HTF POI - NAS100, cumulative R by POI gate")
        fig.tight_layout()
        fig.savefig(f"{cfg['OUT_DIR']}/equity.png", dpi=130)
        print(f"\nsaved {cfg['OUT_DIR']}/equity.png")
    except Exception as e:
        print("plot skipped:", e)

    return res, all_tr


if __name__ == "__main__":
    main()
