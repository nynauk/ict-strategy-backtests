"""
Plot the best IFVG trade: 2024-11-13 SELL NY_KZ 1H_FVG  +46 pips WIN
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import matplotlib.dates as mdates

PIP = 0.0001
TRADE_DT   = pd.Timestamp("2024-11-13 13:45:00")   # disrespect/entry candle
DIRECTION  = "SELL"
ENTRY      = 1.06287
SL         = 1.06517
TP         = 1.05827
RESULT     = "WIN"
PIPS       = 46.0
KZ         = "NY Kill Zone  12:00–15:00 UTC"
PD_LABEL   = "1H FVG"
BIAS       = "BEARISH"

# ── Load M5 data ─────────────────────────────────────────────────────────────
df = pd.read_csv(
    r"C:\Users\Ayaan\Downloads\EURUSD_M5.csv",
    header=None, names=["datetime","open","high","low","close","volume"]
)
df["datetime"] = pd.to_datetime(df["datetime"])
df = df.sort_values("datetime").reset_index(drop=True)

entry_idx = df.index[df["datetime"] == TRADE_DT][0]

# Window: 80 before entry, find exit candle then 60 after
def find_exit_idx(start, tp, sl, direction):
    for i in range(start + 1, min(start + 300, len(df))):
        r = df.iloc[i]
        if direction == "SELL":
            if r["high"] >= sl: return i
            if r["low"]  <= tp: return i
        else:
            if r["low"]  <= sl: return i
            if r["high"] >= tp: return i
    return start + 100

exit_idx  = find_exit_idx(entry_idx, TP, SL, DIRECTION)
plot_start = max(0, entry_idx - 80)
plot_end   = min(len(df) - 1, exit_idx + 60)

sub = df.iloc[plot_start:plot_end + 1].reset_index(drop=True)
entry_pos = entry_idx - plot_start   # position within sub

# ── Session reference (2024-11-13) ───────────────────────────────────────────
trade_date = TRADE_DT.date()
day_df     = df[df["datetime"].dt.date == trade_date].reset_index(drop=True)

london_bars = day_df[
    (day_df["datetime"].dt.hour >= 7) & (day_df["datetime"].dt.hour < 12)
]
london_high = london_bars["high"].max()
london_low  = london_bars["low"].min()

# ── Find sweep candle (wick > london_high, close < london_high, in NY KZ) ───
sweep_idx_global = None
for i in day_df.index:
    r = day_df.iloc[i]
    tm = r["datetime"].hour * 60 + r["datetime"].minute
    if 12*60 <= tm <= 15*60:
        if r["high"] > london_high and r["close"] < london_high:
            sweep_idx_global = df.index[df["datetime"] == r["datetime"]][0]
            sweep_dt = r["datetime"]
            break

# ── Find 5m bullish FVG formed after sweep and before entry ──────────────────
fvg_top = fvg_bot = fvg_dt = None
fvg_pos = None
if sweep_idx_global is not None:
    for fi in range(sweep_idx_global + 2, entry_idx + 1):
        c0 = df.iloc[fi - 2]
        c2 = df.iloc[fi]
        if c0["high"] < c2["low"]:
            # bullish FVG
            fvg_top = c2["low"]
            fvg_bot = c0["high"]
            fvg_dt  = df.iloc[fi]["datetime"]
            fvg_pos = fi - plot_start
            break

# ── Find 1H FVG zone (bearish) active at trade time ─────────────────────────
df_1h = df.set_index("datetime").resample("1h").agg(
    open=("open","first"), high=("high","max"),
    low=("low","min"), close=("close","last")
).dropna().reset_index()

htf_fvg_top = htf_fvg_bot = None
for hi in range(2, len(df_1h)):
    bar = df_1h.iloc[hi]
    if bar["datetime"] >= TRADE_DT:
        break
    # bearish FVG: c[hi-2].high < c[hi].low  (gap above — premium)
    c0h = df_1h.iloc[hi - 2]
    c2h = bar
    if c0h["high"] < c2h["low"]:
        candidate_bot = c2h["low"]
        candidate_top = c0h["high"]
        # check if ENTRY is within or below this gap (price tapped into it)
        if ENTRY <= candidate_bot + 0.005 and ENTRY >= candidate_top - 0.005:
            htf_fvg_top = candidate_bot
            htf_fvg_bot = candidate_top

# Fallback: use a window around SL if not found
if htf_fvg_top is None:
    htf_fvg_top = SL + 0.0010
    htf_fvg_bot = SL - 0.0010

# ── Numeric x-axis (candle index) ────────────────────────────────────────────
xs = np.arange(len(sub))

# ── Build datetime ticks ─────────────────────────────────────────────────────
def fmt_tick(i):
    return sub["datetime"].iloc[i].strftime("%H:%M\n%b %d")

# ── Identify kill zone x-range in sub ────────────────────────────────────────
kz_mask = (sub["datetime"].dt.hour >= 12) & (sub["datetime"].dt.hour < 15) & \
          (sub["datetime"].dt.date == trade_date)
kz_xs   = xs[kz_mask.values]

# ── Plot ──────────────────────────────────────────────────────────────────────
plt.style.use("dark_background")
fig, ax = plt.subplots(figsize=(20, 10), facecolor="#0d1117")
ax.set_facecolor("#0d1117")

# KZ background shading
if len(kz_xs) > 0:
    ax.axvspan(kz_xs[0] - 0.5, kz_xs[-1] + 0.5, alpha=0.07,
               color="#FFD700", zorder=0, label="_kz")
    ax.text(kz_xs[0], ax.get_ylim()[1] if False else TP - 0.0008,
            "NY Kill Zone", color="#FFD700", fontsize=7.5, alpha=0.55,
            va="bottom", zorder=1)

# Draw candles
W = 0.4
for i, row in sub.iterrows():
    o, h, l, c = row["open"], row["high"], row["low"], row["close"]
    color = "#26a69a" if c >= o else "#ef5350"
    # body
    ax.add_patch(mpatches.FancyBboxPatch(
        (i - W/2, min(o, c)), W, abs(c - o),
        boxstyle="square,pad=0", linewidth=0,
        facecolor=color, zorder=3
    ))
    # wick
    ax.plot([i, i], [l, min(o, c)], color=color, linewidth=0.8, zorder=3)
    ax.plot([i, i], [max(o, c), h], color=color, linewidth=0.8, zorder=3)

# HTF 1H FVG zone (shaded rectangle)
htf_fvg_color = "#FF6B35"
ax.axhspan(htf_fvg_bot, htf_fvg_top, alpha=0.18, color=htf_fvg_color, zorder=1)
ax.axhline(htf_fvg_top, color=htf_fvg_color, linewidth=0.8, linestyle="--", alpha=0.7, zorder=2)
ax.axhline(htf_fvg_bot, color=htf_fvg_color, linewidth=0.8, linestyle="--", alpha=0.7, zorder=2)
ax.text(len(sub) - 2, (htf_fvg_top + htf_fvg_bot) / 2,
        f" {PD_LABEL} (Bearish)", color=htf_fvg_color,
        fontsize=8.5, va="center", ha="right", fontweight="bold", zorder=5)

# Session high swept line
if not np.isnan(london_high):
    ax.axhline(london_high, color="#E040FB", linewidth=1.1,
               linestyle=(0, (5, 3)), zorder=2)
    ax.text(1, london_high + PIP * 0.5, f"London Session High (Swept)  {london_high:.5f}",
            color="#E040FB", fontsize=8, va="bottom", zorder=5)

# Sweep candle highlight
if sweep_idx_global is not None:
    sp = sweep_idx_global - plot_start
    if 0 <= sp < len(sub):
        sc = sub.iloc[sp]
        ax.add_patch(mpatches.FancyBboxPatch(
            (sp - W/2 - 0.15, sc["low"] - PIP * 2), W + 0.3,
            sc["high"] - sc["low"] + PIP * 4,
            boxstyle="square,pad=0", linewidth=1.5,
            edgecolor="#E040FB", facecolor="none", zorder=4
        ))
        ax.text(sp, sc["high"] + PIP * 3, "SWEEP", color="#E040FB",
                fontsize=7.5, ha="center", va="bottom", fontweight="bold", zorder=5)

# 5m Bullish FVG zone
if fvg_top is not None and fvg_pos is not None and 0 <= fvg_pos < len(sub):
    fvg_color = "#64B5F6"
    ax.axhspan(fvg_bot, fvg_top, xmin=(fvg_pos - 3) / len(sub),
               xmax=min(1.0, (entry_pos + 2) / len(sub)),
               alpha=0.22, color=fvg_color, zorder=2)
    ax.plot([fvg_pos - 2, entry_pos + 1], [fvg_top, fvg_top],
            color=fvg_color, linewidth=0.9, linestyle="--", alpha=0.8, zorder=2)
    ax.plot([fvg_pos - 2, entry_pos + 1], [fvg_bot, fvg_bot],
            color=fvg_color, linewidth=0.9, linestyle="--", alpha=0.8, zorder=2)
    ax.text(fvg_pos - 1, (fvg_top + fvg_bot) / 2,
            f"5m Bull FVG\n{fvg_bot:.5f}–{fvg_top:.5f}",
            color=fvg_color, fontsize=7.5, va="center", ha="left", zorder=5)

# Highlight disrespect (entry) candle with box
ec = sub.iloc[entry_pos]
ax.add_patch(mpatches.FancyBboxPatch(
    (entry_pos - W/2 - 0.2, ec["low"] - PIP * 2), W + 0.4,
    ec["high"] - ec["low"] + PIP * 4,
    boxstyle="square,pad=0", linewidth=2,
    edgecolor="#FFD700", facecolor="none", zorder=6
))
ax.text(entry_pos, ec["low"] - PIP * 5, "IFVG\nDISRESPECT",
        color="#FFD700", fontsize=7.5, ha="center", va="top",
        fontweight="bold", zorder=7)

# Entry / SL / TP lines
ax.axhline(ENTRY, color="#FFD700", linewidth=1.4, linestyle="--", zorder=5)
ax.axhline(SL,    color="#ef5350", linewidth=1.4, linestyle="--", zorder=5)
ax.axhline(TP,    color="#26a69a", linewidth=1.4, linestyle="--", zorder=5)

right = len(sub) - 1
ax.text(right, ENTRY + PIP * 0.3, f" Entry  {ENTRY:.5f}", color="#FFD700",
        fontsize=8.5, va="bottom", ha="right", fontweight="bold", zorder=7)
ax.text(right, SL    + PIP * 0.3, f" SL     {SL:.5f}",    color="#ef5350",
        fontsize=8.5, va="bottom", ha="right", fontweight="bold", zorder=7)
ax.text(right, TP    - PIP * 0.3, f" TP     {TP:.5f}",    color="#26a69a",
        fontsize=8.5, va="top",    ha="right", fontweight="bold", zorder=7)

# Down arrow at entry candle (SELL)
ax.annotate("", xy=(entry_pos, ec["high"] + PIP * 6),
            xytext=(entry_pos, ec["high"] + PIP * 22),
            arrowprops=dict(arrowstyle="-|>", color="#FFD700",
                            lw=2.2, mutation_scale=18), zorder=8)

# HTF Bias label
ax.text(0.01, 0.97, f"HTF Bias: {BIAS}", transform=ax.transAxes,
        color="#ef5350", fontsize=11, fontweight="bold",
        va="top", ha="left", zorder=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#1a1a2e", edgecolor="#ef5350", alpha=0.85))

# KZ label (top-right corner)
ax.text(0.99, 0.97, KZ, transform=ax.transAxes,
        color="#FFD700", fontsize=9, fontweight="bold",
        va="top", ha="right", zorder=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#1a1a2e", edgecolor="#FFD700", alpha=0.85))

# X-axis ticks — every ~20 candles
tick_step = 20
tick_pos   = list(range(0, len(sub), tick_step))
tick_lbls  = [sub["datetime"].iloc[i].strftime("%H:%M\n%b %d") for i in tick_pos]
ax.set_xticks(tick_pos)
ax.set_xticklabels(tick_lbls, fontsize=7.5, color="#aaaaaa")
ax.set_xlim(-1, len(sub))

# Y-axis
price_range = sub["high"].max() - sub["low"].min()
ax.set_ylim(sub["low"].min() - price_range * 0.05,
            sub["high"].max() + price_range * 0.08)
ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.5f"))
ax.tick_params(axis="y", colors="#aaaaaa", labelsize=8)
ax.tick_params(axis="x", colors="#aaaaaa")

# Grid
ax.grid(True, color="#333344", linewidth=0.4, alpha=0.6, zorder=0)

# Legend patches
legend_items = [
    mpatches.Patch(color="#E040FB",  alpha=0.8,  label="Session High (Swept)"),
    mpatches.Patch(color="#FF6B35",  alpha=0.5,  label="1H FVG (HTF PD Array)"),
    mpatches.Patch(color="#64B5F6",  alpha=0.5,  label="5m Bullish FVG"),
    mpatches.Patch(color="#FFD700",  alpha=0.7,  label="IFVG Disrespect Candle"),
    mpatches.Patch(color="#FFD700",  alpha=0.25, label="NY Kill Zone"),
    mpatches.Patch(color="#26a69a",  label="Bullish candle"),
    mpatches.Patch(color="#ef5350",  label="Bearish candle"),
]
ax.legend(handles=legend_items, loc="lower left", fontsize=7.5,
          facecolor="#1a1a2e", edgecolor="#444455", labelcolor="white",
          framealpha=0.85, ncol=4)

# Title
rr = round(abs(ENTRY - TP) / abs(SL - ENTRY), 1)
ax.set_title(
    f"IFVG SELL  |  EURUSD M5  |  2024-11-13 13:45 UTC  |  NY Kill Zone  |  PD Array: {PD_LABEL}\n"
    f"Result: {RESULT}   Pips: +{PIPS:.1f}   RR: {rr:.1f}:1   Entry: {ENTRY:.5f}   SL: {SL:.5f}   TP: {TP:.5f}",
    color="white", fontsize=11, fontweight="bold", pad=12
)

ax.spines["bottom"].set_color("#333344")
ax.spines["left"].set_color("#333344")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig("ifvg_trade_chart.png", dpi=180, bbox_inches="tight",
            facecolor="#0d1117")
plt.close()
print("Saved: ifvg_trade_chart.png")
