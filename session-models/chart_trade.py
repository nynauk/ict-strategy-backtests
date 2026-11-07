"""
Trade Chart — HTF Bias + PD Array (FVG) Setup
==============================================
Visualises the selected trade with:
  - 5m candlestick price action (window around the trade)
  - Shaded FVG / OB zone
  - Entry, SL, TP horizontal lines
  - Entry candle marker
  - HTF bias label
  - Title with outcome, pips, and RR
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

# ── Trade to visualise ────────────────────────────────────────────────────────
TRADE = {
    'entry_time'  : '2025-04-03 09:05:00',
    'exit_time'   : '2025-04-03 10:10:00',
    'direction'   : 'buy',
    'kill_zone'   : 'London',
    'pd_type'     : 'FVG',
    'htf_bias'    : 'Bullish (1H)',
    'entry'       : 1.10307,
    'sl'          : 1.09943,
    'tp'          : 1.11035,
    'exit_price'  : 1.11035,
    'result'      : 'WIN',
    'pips'        : 72.8,
    'pd_low'      : 1.09943,   # FVG zone bottom
    'pd_high'     : 1.10053,   # FVG zone top
}

# ── Data ──────────────────────────────────────────────────────────────────────
DATA_PATH  = r"C:\Users\Ayaan\Downloads\EURUSD_M5.csv"
OUTPUT_DIR = r"C:\Users\Ayaan\PycharmProjects\9AMBACKTEST"

BARS_BEFORE = 80    # 5m candles to show before entry
BARS_AFTER  = 60    # 5m candles to show after exit

# ── Colour palette ────────────────────────────────────────────────────────────
COL_BULL_BODY   = '#26a69a'   # teal  — bullish candle body
COL_BEAR_BODY   = '#ef5350'   # red   — bearish candle body
COL_WICK        = '#888888'
COL_ENTRY       = '#FFD700'   # gold
COL_SL          = '#ef5350'   # red
COL_TP          = '#26a69a'   # green
COL_FVG         = '#42a5f5'   # blue fill for FVG zone
COL_BG          = '#131722'   # dark background
COL_GRID        = '#2a2e39'
COL_TEXT        = '#d1d4dc'


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def load_data(path):
    with open(path, 'r') as f:
        first = f.readline()
    sep = '\t' if '\t' in first else ','
    try:
        pd.to_datetime(first.split(sep)[0])
        has_header = False
    except Exception:
        has_header = True

    cols = ['Time', 'Open', 'High', 'Low', 'Close', 'Volume']
    df = pd.read_csv(path, sep=sep,
                     header=0 if has_header else None,
                     names=cols if not has_header else None)
    df.columns = cols[:len(df.columns)]
    df['Time'] = pd.to_datetime(df['Time'])
    df.set_index('Time', inplace=True)
    df.sort_index(inplace=True)
    return df[['Open', 'High', 'Low', 'Close']].astype(float)


def draw_candles(ax, df_slice):
    """Draw OHLC candlesticks as thin wicks + body rectangles."""
    for x, (ts, row) in enumerate(df_slice.iterrows()):
        o, h, l, c = row['Open'], row['High'], row['Low'], row['Close']
        bull = c >= o
        body_col = COL_BULL_BODY if bull else COL_BEAR_BODY

        # Wick
        ax.plot([x, x], [l, h], color=COL_WICK, linewidth=0.7, zorder=2)

        # Body
        body_bot = min(o, c)
        body_h   = abs(c - o) if abs(c - o) > 0 else 0.00001
        rect = mpatches.FancyBboxPatch(
            (x - 0.35, body_bot), 0.70, body_h,
            boxstyle='square,pad=0',
            linewidth=0,
            facecolor=body_col,
            zorder=3
        )
        ax.add_patch(rect)


def price_to_label(price, pip=0.0001):
    return f"{price:.5f}"


# ══════════════════════════════════════════════════════════════════════════════
# MAIN CHART
# ══════════════════════════════════════════════════════════════════════════════

def make_chart():
    t = TRADE
    entry_ts = pd.Timestamp(t['entry_time'])
    exit_ts  = pd.Timestamp(t['exit_time'])

    # Load and slice data
    df = load_data(DATA_PATH)

    if entry_ts not in df.index:
        # Find nearest bar
        entry_ts = df.index[df.index.get_indexer([entry_ts], method='nearest')[0]]

    entry_pos = df.index.get_loc(entry_ts)
    exit_pos  = df.index.get_loc(
        df.index[df.index.get_indexer([exit_ts], method='nearest')[0]]
    )

    start = max(0, entry_pos - BARS_BEFORE)
    end   = min(len(df), exit_pos  + BARS_AFTER + 1)
    df_w  = df.iloc[start:end].copy()

    # x-axis positions relative to the slice
    entry_x = entry_pos - start
    exit_x  = exit_pos  - start

    # ── Figure setup ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(18, 9), facecolor=COL_BG)
    ax.set_facecolor(COL_BG)

    # ── Candlesticks ─────────────────────────────────────────────────────────
    draw_candles(ax, df_w)

    n = len(df_w)
    ax.set_xlim(-1, n)

    # ── FVG shaded zone ───────────────────────────────────────────────────────
    # Extend from a few bars before entry to exit_x
    fvg_x_start = max(0, entry_x - 40)
    fvg_width   = exit_x - fvg_x_start + 5
    fvg_rect = mpatches.FancyBboxPatch(
        (fvg_x_start, t['pd_low']),
        fvg_width,
        t['pd_high'] - t['pd_low'],
        boxstyle='square,pad=0',
        facecolor=COL_FVG,
        edgecolor=COL_FVG,
        alpha=0.20,
        zorder=1,
        label=f"FVG Zone  [{price_to_label(t['pd_low'])} – {price_to_label(t['pd_high'])}]"
    )
    ax.add_patch(fvg_rect)
    # FVG border lines (dashed)
    ax.hlines([t['pd_low'], t['pd_high']],
              xmin=fvg_x_start, xmax=fvg_x_start + fvg_width,
              colors=COL_FVG, linewidths=0.8, linestyles='--', alpha=0.60, zorder=2)

    # ── Kill zone background shading ──────────────────────────────────────────
    # Shade the London kill zone bars (07:00-10:00 UTC)
    kz_xs = [x for x, ts in enumerate(df_w.index) if 7 <= ts.hour < 10]
    if kz_xs:
        kz_start = min(kz_xs)
        kz_end   = max(kz_xs)
        y_min, y_max = ax.get_ylim() if ax.get_ylim() != (0, 1) else \
                       (df_w['Low'].min() * 0.9998, df_w['High'].max() * 1.0002)
        ax.axvspan(kz_start - 0.5, kz_end + 0.5,
                   color='#FFD700', alpha=0.04, zorder=0,
                   label='London Kill Zone (07–10 UTC)')

    # ── Horizontal level lines ────────────────────────────────────────────────
    line_start = max(0, entry_x - 5)
    line_end   = n - 1

    # TP line
    ax.hlines(t['tp'], xmin=line_start, xmax=line_end,
              colors=COL_TP, linewidths=1.4, linestyles='-', zorder=4,
              label=f"TP  {price_to_label(t['tp'])}  (+{t['pips']:.1f} pips)")
    ax.text(n - 0.5, t['tp'], f"TP  {price_to_label(t['tp'])}",
            color=COL_TP, fontsize=8.5, va='center', ha='right',
            bbox=dict(facecolor=COL_BG, edgecolor='none', alpha=0.75, pad=1.5))

    # Entry line
    ax.hlines(t['entry'], xmin=line_start, xmax=line_end,
              colors=COL_ENTRY, linewidths=1.2, linestyles='--', zorder=4,
              label=f"Entry  {price_to_label(t['entry'])}")
    ax.text(n - 0.5, t['entry'], f"Entry  {price_to_label(t['entry'])}",
            color=COL_ENTRY, fontsize=8.5, va='center', ha='right',
            bbox=dict(facecolor=COL_BG, edgecolor='none', alpha=0.75, pad=1.5))

    # SL line
    sl_pips = abs(t['entry'] - t['sl']) / 0.0001
    ax.hlines(t['sl'], xmin=line_start, xmax=line_end,
              colors=COL_SL, linewidths=1.4, linestyles='-', zorder=4,
              label=f"SL  {price_to_label(t['sl'])}  (-{sl_pips:.1f} pips)")
    ax.text(n - 0.5, t['sl'], f"SL  {price_to_label(t['sl'])}",
            color=COL_SL, fontsize=8.5, va='center', ha='right',
            bbox=dict(facecolor=COL_BG, edgecolor='none', alpha=0.75, pad=1.5))

    # ── RR shading between entry and TP / SL ─────────────────────────────────
    ax.fill_between(range(line_start, line_end + 1),
                    t['entry'], t['tp'],
                    color=COL_TP,  alpha=0.05, zorder=1)
    ax.fill_between(range(line_start, line_end + 1),
                    t['sl'], t['entry'],
                    color=COL_SL, alpha=0.07, zorder=1)

    # ── Entry candle arrow ────────────────────────────────────────────────────
    entry_candle_low = df_w['Low'].iloc[entry_x]
    arrow_tip_y  = entry_candle_low - (t['entry'] - t['sl']) * 0.35
    arrow_base_y = entry_candle_low - (t['entry'] - t['sl']) * 0.05

    ax.annotate(
        '',
        xy=(entry_x, arrow_base_y),
        xytext=(entry_x, arrow_tip_y),
        arrowprops=dict(
            arrowstyle='->', color=COL_ENTRY,
            lw=2.0,
            mutation_scale=18,
        ),
        zorder=6
    )
    ax.text(entry_x, arrow_tip_y - (t['entry'] - t['sl']) * 0.12,
            'ENTRY', color=COL_ENTRY, fontsize=8, ha='center', va='top',
            fontweight='bold', zorder=7)

    # ── Exit marker ───────────────────────────────────────────────────────────
    exit_candle_high = df_w['High'].iloc[exit_x]
    ax.scatter([exit_x], [exit_candle_high + (t['tp'] - t['entry']) * 0.08],
               marker='*', s=160, color=COL_TP, zorder=6)
    ax.text(exit_x, exit_candle_high + (t['tp'] - t['entry']) * 0.18,
            'TP HIT', color=COL_TP, fontsize=8, ha='center', va='bottom',
            fontweight='bold', zorder=7)

    # ── X-axis tick labels (time) ─────────────────────────────────────────────
    step = max(1, n // 12)
    tick_xs    = list(range(0, n, step))
    tick_lbls  = [df_w.index[x].strftime('%d %b\n%H:%M') for x in tick_xs]
    ax.set_xticks(tick_xs)
    ax.set_xticklabels(tick_lbls, color=COL_TEXT, fontsize=7.5)

    # ── Y-axis ────────────────────────────────────────────────────────────────
    price_range = df_w['High'].max() - df_w['Low'].min()
    ax.set_ylim(df_w['Low'].min()  - price_range * 0.08,
                df_w['High'].max() + price_range * 0.12)
    ax.yaxis.set_tick_params(labelcolor=COL_TEXT, labelsize=8)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.4f}'))

    # ── Grid ──────────────────────────────────────────────────────────────────
    ax.grid(axis='y', color=COL_GRID, linewidth=0.5, zorder=0)
    ax.grid(axis='x', color=COL_GRID, linewidth=0.3, alpha=0.5, zorder=0)
    for spine in ax.spines.values():
        spine.set_edgecolor(COL_GRID)

    # ── HTF bias annotation (top-left) ────────────────────────────────────────
    ax.text(0.01, 0.97,
            f"HTF Bias: {t['htf_bias'].upper()}",
            transform=ax.transAxes,
            color=COL_BULL_BODY, fontsize=10, fontweight='bold',
            va='top', ha='left',
            bbox=dict(facecolor='#1a2332', edgecolor=COL_BULL_BODY,
                      linewidth=1, alpha=0.9, pad=4))

    # ── Kill zone label ───────────────────────────────────────────────────────
    ax.text(0.01, 0.89,
            f"Session: {t['kill_zone'].upper()} KILL ZONE  (07:00-10:00 UTC)",
            transform=ax.transAxes,
            color='#FFD700', fontsize=8.5,
            va='top', ha='left',
            bbox=dict(facecolor='#1a2332', edgecolor='#FFD700',
                      linewidth=0.8, alpha=0.85, pad=3))

    # ── Legend ────────────────────────────────────────────────────────────────
    legend = ax.legend(
        loc='upper right', fontsize=8.5,
        facecolor='#1a2332', edgecolor=COL_GRID,
        labelcolor=COL_TEXT, framealpha=0.92,
        borderpad=0.8, handlelength=1.8
    )

    # ── Title ─────────────────────────────────────────────────────────────────
    rr  = t['pips'] / (abs(t['entry'] - t['sl']) / 0.0001)
    title = (f"EURUSD M5  |  {t['direction'].upper()}  |  {t['pd_type']} Entry  |  "
             f"{t['result']}  +{t['pips']:.1f} pips  ({rr:.1f}R)  |  "
             f"{pd.Timestamp(t['entry_time']).strftime('%d %b %Y  %H:%M')} UTC")
    ax.set_title(title, color=COL_TEXT, fontsize=12, fontweight='bold', pad=10)

    ax.set_xlabel('Time (UTC)', color=COL_TEXT, fontsize=9, labelpad=6)
    ax.set_ylabel('Price (EURUSD)', color=COL_TEXT, fontsize=9, labelpad=6)

    # ── Save ─────────────────────────────────────────────────────────────────
    out_path = os.path.join(OUTPUT_DIR, 'trade_chart.png')
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches='tight', facecolor=COL_BG)
    plt.close()
    print(f"Chart saved -> {out_path}")
    return out_path


if __name__ == '__main__':
    make_chart()
