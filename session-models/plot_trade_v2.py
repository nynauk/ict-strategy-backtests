"""
Plot best trade from corrected backtest: 2024-12-27 BUY 4H_FVG +23.4 pips WIN
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe

# ── Trade constants ───────────────────────────────────────────────────────────
TRADE_DT   = pd.Timestamp("2024-12-27 07:50:00")
DIRECTION  = "BUY"
ENTRY      = 1.04166
SL         = 1.04049
TP         = 1.04400
RESULT     = "WIN"
PIPS       = 23.4
PD_LABEL   = "4H FVG"
BIAS       = "BULLISH"
PIP        = 0.0001

# Reconstructed setup values
ASIA_HIGH      = 1.04237
ASIA_LOW       = 1.04065
SWEEP_DT       = pd.Timestamp("2024-12-27 07:40:00")
# 4H FVG zone (bullish, entirely below Asia low)
HTF_FVG_BOT    = 1.04030
HTF_FVG_TOP    = 1.04063
# 5m bearish FVG (got disrespected on entry candle)
FVG_5M_BOT     = 1.04089
FVG_5M_TOP     = 1.04091   # tiny gap — will render with min visual height

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv(
    r"C:\Users\Ayaan\Downloads\EURUSD_M5.csv",
    header=None, names=["datetime","open","high","low","close","volume"]
)
df["datetime"] = pd.to_datetime(df["datetime"])
df = df.sort_values("datetime").reset_index(drop=True)

entry_idx = df.index[df["datetime"] == TRADE_DT][0]
exit_idx  = df.index[df["datetime"] == pd.Timestamp("2024-12-27 11:55:00")][0]

plot_start = max(0, entry_idx - 80)
plot_end   = min(len(df) - 1, exit_idx + 60)

sub = df.iloc[plot_start : plot_end + 1].reset_index(drop=True)
sub_dts = sub["datetime"]

# Local index helpers
def local(global_idx):
    return global_idx - plot_start

entry_pos = local(entry_idx)
sweep_pos = local(df.index[df["datetime"] == SWEEP_DT][0])
exit_pos  = local(exit_idx)

N = len(sub)
xs = np.arange(N)

# ── Compute x-ranges for session shading ─────────────────────────────────────
trade_date = TRADE_DT.date()

def x_range_for(mask_fn):
    """Return (x_start, x_end) fractional for axvspan, and integer list for axvspan."""
    idxs = [i for i, dt in enumerate(sub_dts) if mask_fn(dt)]
    return idxs

asia_xs = x_range_for(lambda dt: dt.date() == trade_date and dt.hour < 7)
lkz_xs  = x_range_for(lambda dt: dt.date() == trade_date and
                        7*60 <= dt.hour*60 + dt.minute <= 10*60)

# ── Colours ───────────────────────────────────────────────────────────────────
C_BG      = "#0d1117"
C_UP      = "#26a69a"
C_DOWN    = "#ef5350"
C_ASIA    = "#1e3a5f"       # dark blue for Asia session bg
C_LKZ     = "#1e2d1e"       # dark green for LKZ bg
C_SWEEP   = "#CE93D8"       # purple
C_HTF_FVG = "#FF8F00"       # amber for 4H FVG
C_5M_FVG  = "#64B5F6"       # light blue for 5m FVG
C_ENTRY   = "#FFD700"       # gold
C_SL      = "#ef5350"
C_TP      = "#26a69a"
C_ASIA_H  = "#90CAF9"       # light blue for Asia High line
C_ASIA_L  = "#90CAF9"

# ── Plot ──────────────────────────────────────────────────────────────────────
plt.style.use("dark_background")
fig, ax = plt.subplots(figsize=(22, 10), facecolor=C_BG)
ax.set_facecolor(C_BG)

# ── 1. Asia session shading ──────────────────────────────────────────────────
if asia_xs:
    ax.axvspan(asia_xs[0] - 0.5, asia_xs[-1] + 0.5,
               facecolor=C_ASIA, alpha=0.35, zorder=0)
    # Label at top of Asia band
    mid_asia = (asia_xs[0] + asia_xs[-1]) / 2
    ax.text(mid_asia, TP + 0.0006, "Asia Session\n00:00–07:00 UTC",
            color="#90CAF9", fontsize=7.5, ha="center", va="bottom",
            alpha=0.75, zorder=1)

# ── 2. London Kill Zone shading ───────────────────────────────────────────────
if lkz_xs:
    ax.axvspan(lkz_xs[0] - 0.5, lkz_xs[-1] + 0.5,
               facecolor="#22381e", alpha=0.40, zorder=0)
    ax.text(lkz_xs[0] + 1, SL - PIP * 3,
            "London Kill Zone  07:00–10:00", color="#66BB6A",
            fontsize=7.5, va="top", alpha=0.65, zorder=1)

# ── 3. Draw candlesticks ──────────────────────────────────────────────────────
W = 0.42
for i in range(N):
    o = sub["open"].iloc[i]
    h = sub["high"].iloc[i]
    l = sub["low"].iloc[i]
    c = sub["close"].iloc[i]
    col = C_UP if c >= o else C_DOWN
    # body
    body_bot = min(o, c)
    body_h   = max(abs(c - o), PIP * 0.3)     # minimum 0.3 pip visible body
    ax.add_patch(mpatches.Rectangle(
        (i - W/2, body_bot), W, body_h,
        linewidth=0, facecolor=col, zorder=3
    ))
    # wicks
    ax.plot([i, i], [l, body_bot],        color=col, linewidth=0.8, zorder=3)
    ax.plot([i, i], [body_bot + body_h, h], color=col, linewidth=0.8, zorder=3)

# ── 4. Asia High / Asia Low dashed lines ──────────────────────────────────────
ax.axhline(ASIA_HIGH, color=C_ASIA_H, linewidth=1.1, linestyle=(0, (6, 4)),
           alpha=0.85, zorder=4)
ax.axhline(ASIA_LOW,  color=C_ASIA_L, linewidth=1.1, linestyle=(0, (6, 4)),
           alpha=0.85, zorder=4)

# Labels at the Asia border (end of Asia session)
asia_end_x = asia_xs[-1] if asia_xs else 0
ax.text(asia_end_x + 1, ASIA_HIGH + PIP * 0.5, f"Asia High  {ASIA_HIGH:.5f}",
        color=C_ASIA_H, fontsize=8, va="bottom", zorder=5,
        fontweight="bold")
ax.text(asia_end_x + 1, ASIA_LOW - PIP * 0.5, f"Asia Low  {ASIA_LOW:.5f}",
        color=C_ASIA_L, fontsize=8, va="top", zorder=5,
        fontweight="bold")

# ── 5. 4H FVG zone (HTF PD array) ────────────────────────────────────────────
# Draw from start of chart to entry candle
ax.add_patch(mpatches.Rectangle(
    (-0.5, HTF_FVG_BOT), entry_pos + 2, HTF_FVG_TOP - HTF_FVG_BOT,
    facecolor=C_HTF_FVG, alpha=0.18, zorder=1
))
ax.plot([-0.5, entry_pos + 2], [HTF_FVG_BOT, HTF_FVG_BOT],
        color=C_HTF_FVG, linewidth=0.9, linestyle="--", alpha=0.7, zorder=2)
ax.plot([-0.5, entry_pos + 2], [HTF_FVG_TOP, HTF_FVG_TOP],
        color=C_HTF_FVG, linewidth=0.9, linestyle="--", alpha=0.7, zorder=2)
# Label
ax.text(entry_pos + 3, (HTF_FVG_BOT + HTF_FVG_TOP) / 2,
        f"4H FVG (Bullish)\n{HTF_FVG_BOT:.5f} - {HTF_FVG_TOP:.5f}",
        color=C_HTF_FVG, fontsize=8, va="center", fontweight="bold", zorder=5)

# ── 6. 5m Bearish FVG zone (IFVG) ────────────────────────────────────────────
# Give it minimum visual height (5 pips so it's visible)
fvg_display_bot = FVG_5M_BOT - PIP * 2
fvg_display_top = FVG_5M_TOP + PIP * 2
fvg_x_start = sweep_pos - 2
fvg_x_end   = entry_pos + 1
ax.add_patch(mpatches.Rectangle(
    (fvg_x_start, fvg_display_bot), fvg_x_end - fvg_x_start,
    fvg_display_top - fvg_display_bot,
    facecolor=C_5M_FVG, alpha=0.25, zorder=2
))
ax.plot([fvg_x_start, fvg_x_end], [fvg_display_top, fvg_display_top],
        color=C_5M_FVG, linewidth=1.0, linestyle="--", alpha=0.7, zorder=2)
ax.plot([fvg_x_start, fvg_x_end], [fvg_display_bot, fvg_display_bot],
        color=C_5M_FVG, linewidth=1.0, linestyle="--", alpha=0.7, zorder=2)
ax.text(fvg_x_start - 1, (fvg_display_bot + fvg_display_top) / 2,
        f"5m Bear FVG\n(IFVG zone)",
        color=C_5M_FVG, fontsize=7.5, va="center", ha="right", zorder=5)

# ── 7. Sweep candle highlight ─────────────────────────────────────────────────
sc = sub.iloc[sweep_pos]
sc_low  = sc["low"]
sc_high = sc["high"]
ax.add_patch(mpatches.FancyBboxPatch(
    (sweep_pos - W/2 - 0.2, sc_low - PIP * 2), W + 0.4,
    sc_high - sc_low + PIP * 4,
    boxstyle="square,pad=0", linewidth=1.8,
    edgecolor=C_SWEEP, facecolor="none", zorder=6
))
ax.text(sweep_pos, sc_low - PIP * 5, "SWEEP",
        color=C_SWEEP, fontsize=8, ha="center", va="top",
        fontweight="bold", zorder=7)
# Arrow pointing at the wick that swept below Asia Low
ax.annotate("",
    xy=(sweep_pos, sc_low - PIP * 0.5),
    xytext=(sweep_pos - 3, sc_low - PIP * 12),
    arrowprops=dict(arrowstyle="-|>", color=C_SWEEP, lw=1.4,
                    mutation_scale=10),
    zorder=8
)

# ── 8. Disrespect / entry candle highlight ────────────────────────────────────
ec = sub.iloc[entry_pos]
ax.add_patch(mpatches.FancyBboxPatch(
    (entry_pos - W/2 - 0.2, ec["low"] - PIP * 2), W + 0.4,
    ec["high"] - ec["low"] + PIP * 4,
    boxstyle="square,pad=0", linewidth=2.2,
    edgecolor=C_ENTRY, facecolor="none", zorder=6
))
ax.text(entry_pos, ec["low"] - PIP * 6, "IFVG\nDISRESPECT",
        color=C_ENTRY, fontsize=7.5, ha="center", va="top",
        fontweight="bold", zorder=7)

# ── 9. Entry / SL / TP horizontal lines ──────────────────────────────────────
for price, color, lbl in [
    (ENTRY, C_ENTRY, f"Entry  {ENTRY:.5f}"),
    (SL,    C_SL,    f"SL     {SL:.5f}"),
    (TP,    C_TP,    f"TP     {TP:.5f}"),
]:
    ax.axhline(price, color=color, linewidth=1.4, linestyle="--", zorder=5)

# Right-side labels
right = N - 1
ax.text(right, ENTRY + PIP * 0.4, f" Entry  {ENTRY:.5f}",
        color=C_ENTRY, fontsize=9, va="bottom", ha="right", fontweight="bold", zorder=7)
ax.text(right, SL - PIP * 0.4, f" SL     {SL:.5f}",
        color=C_SL, fontsize=9, va="top", ha="right", fontweight="bold", zorder=7)
ax.text(right, TP + PIP * 0.4, f" TP     {TP:.5f}",
        color=C_TP, fontsize=9, va="bottom", ha="right", fontweight="bold", zorder=7)

# ── 10. UP arrow at entry candle (BUY) ───────────────────────────────────────
ax.annotate("",
    xy=(entry_pos, ec["low"] - PIP * 6),
    xytext=(entry_pos, ec["low"] - PIP * 20),
    arrowprops=dict(arrowstyle="-|>", color=C_ENTRY, lw=2.5,
                    mutation_scale=20),
    zorder=8
)

# ── 11. Badges: HTF Bias (top-left) and KZ (top-right) ───────────────────────
badge_kw = dict(transform=ax.transAxes, va="top", fontsize=10,
                fontweight="bold", zorder=10)
ax.text(0.01, 0.97, f"HTF Bias: {BIAS}", color=C_TP, **badge_kw,
        ha="left",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#0d2010",
                  edgecolor=C_TP, alpha=0.9))
ax.text(0.99, 0.97, "London Kill Zone  07:00-10:00 UTC", color="#66BB6A",
        **badge_kw, ha="right",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#0d2010",
                  edgecolor="#66BB6A", alpha=0.9))

# ── 12. TP hit marker ─────────────────────────────────────────────────────────
ax.plot(exit_pos, TP, marker="*", markersize=16,
        color=C_TP, zorder=8, markeredgewidth=0)
ax.text(exit_pos + 1, TP + PIP * 2, "TP HIT", color=C_TP,
        fontsize=8, fontweight="bold", va="bottom", zorder=8)

# ── 13. X-axis ticks ─────────────────────────────────────────────────────────
step = 20
tick_pos  = list(range(0, N, step))
tick_lbls = [sub_dts.iloc[i].strftime("%H:%M\n%b %d") for i in tick_pos]
ax.set_xticks(tick_pos)
ax.set_xticklabels(tick_lbls, fontsize=7.5, color="#aaaaaa")
ax.set_xlim(-1, N)

# ── 14. Y-axis + limits ───────────────────────────────────────────────────────
pr = sub["high"].max() - sub["low"].min()
ax.set_ylim(sub["low"].min()  - pr * 0.06,
            sub["high"].max() + pr * 0.10)
ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.5f"))
ax.tick_params(axis="y", colors="#aaaaaa", labelsize=8)
ax.tick_params(axis="x", colors="#aaaaaa")

# ── 15. Grid & spines ─────────────────────────────────────────────────────────
ax.grid(True, color="#1e2030", linewidth=0.5, alpha=0.8, zorder=0)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
for spine in ["bottom", "left"]:
    ax.spines[spine].set_color("#333344")

# ── 16. Legend ────────────────────────────────────────────────────────────────
legend_items = [
    mpatches.Patch(color=C_ASIA,    alpha=0.6,  label="Asia Session (00:00-07:00)"),
    mpatches.Patch(color="#22381e", alpha=0.8,  label="London Kill Zone (07:00-10:00)"),
    mpatches.Patch(color=C_HTF_FVG, alpha=0.5,  label="4H FVG (HTF PD Array)"),
    mpatches.Patch(color=C_5M_FVG,  alpha=0.5,  label="5m Bearish FVG (IFVG)"),
    mpatches.Patch(color=C_SWEEP,   alpha=0.8,  label="Sweep Candle"),
    mpatches.Patch(color=C_ENTRY,   alpha=0.7,  label="IFVG Disrespect / Entry"),
    mpatches.Patch(color=C_UP,      label="Bullish candle"),
    mpatches.Patch(color=C_DOWN,    label="Bearish candle"),
]
ax.legend(handles=legend_items, loc="lower left", fontsize=8,
          facecolor="#1a1a2e", edgecolor="#444455", labelcolor="white",
          framealpha=0.9, ncol=4, borderpad=0.6)

# ── 17. Title ─────────────────────────────────────────────────────────────────
rr = round(abs(TP - ENTRY) / abs(SL - ENTRY), 1)
ax.set_title(
    f"IFVG {DIRECTION}  |  EURUSD M5  |  2024-12-27 07:50 UTC  |  "
    f"PD Array: {PD_LABEL}  |  Result: {RESULT}  |  +{PIPS} pips  |  {rr}:1 RR",
    color="white", fontsize=11, fontweight="bold", pad=14
)

plt.tight_layout()
plt.savefig("ifvg_trade_chart_v2.png", dpi=180, bbox_inches="tight",
            facecolor=C_BG)
plt.close()
print("Saved: ifvg_trade_chart_v2.png")
