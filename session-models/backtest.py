"""
9AM UK Candle Breakout/Retest Backtest — XAUUSD M15
====================================================
Strategy:
  1. Find the 9:00 AM UK time candle each day.
  2. After it closes, watch for price to break above its High or below its Low.
  3. On a break, wait for price to retest the broken level.
  4. Enter the trade (BUY on high-break retest, SELL on low-break retest).
  5. SL on opposite side of the 9AM range; TP at 2× risk distance.

P&L: Win = +2% of balance at entry; Loss = −1% of balance at entry.
"""

import pandas as pd
import pytz
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import time

# ── Configuration ─────────────────────────────────────────────────────────────
DATA_PATH       = r"C:\Users\Ayaan\Downloads\XAUUSD_M15.csv"
STARTING_BALANCE = 10_000          # USD
WIN_PCT          = 0.02            # +2% per win
LOSS_PCT         = 0.01            # −1% per loss
EOD_UK_HOUR      = 16              # stop looking for/managing trades after 16:00 UK
UK_TZ            = pytz.timezone("Europe/London")
# ──────────────────────────────────────────────────────────────────────────────


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep="\t",
        header=0,
        names=["Time", "Open", "High", "Low", "Close", "Volume", "Ticks"],
        parse_dates=["Time"],
    )
    # Localize GMT → Europe/London (handles BST automatically)
    df["Time"] = df["Time"].dt.tz_localize("UTC").dt.tz_convert(UK_TZ)
    df.set_index("Time", inplace=True)
    df.sort_index(inplace=True)
    return df


def get_trading_days(df: pd.DataFrame):
    """Return a list of unique UK-local dates that have a 9 AM candle."""
    days = []
    for date, group in df.groupby(df.index.date):
        nine_am = group[group.index.time == time(9, 0)]
        if not nine_am.empty:
            days.append((date, group))
    return days


def resolve_trade(candles, entry, sl, tp, direction):
    """
    Scan candles after entry to find TP/SL hit or EOD close.
    Returns: 'W', 'L', 'EOD-W', or 'EOD-L'
    """
    for _, row in candles.iterrows():
        at_eod = row.name.time() >= time(EOD_UK_HOUR, 0)

        if direction == "BUY":
            sl_hit = row["Low"] <= sl
            tp_hit = row["High"] >= tp

            if sl_hit and tp_hit:
                return "L"          # conservative: SL first
            if sl_hit:
                return "L"
            if tp_hit:
                return "W"
            if at_eod:
                close = row["Close"]
                return "EOD-W" if close >= tp else "EOD-L"

        else:  # SELL
            sl_hit = row["High"] >= sl
            tp_hit = row["Low"] <= tp

            if sl_hit and tp_hit:
                return "L"
            if sl_hit:
                return "L"
            if tp_hit:
                return "W"
            if at_eod:
                close = row["Close"]
                return "EOD-W" if close <= tp else "EOD-L"

    return "EOD-L"                  # ran out of candles without resolution


def run_backtest(df: pd.DataFrame):
    days = get_trading_days(df)
    balance  = STARTING_BALANCE
    trades   = []

    for date, day_df in days:
        nine_candle = day_df[day_df.index.time == time(9, 0)].iloc[0]
        nine_high   = nine_candle["High"]
        nine_low    = nine_candle["Low"]

        if nine_high == nine_low:       # zero-range candle — skip
            continue

        # Candles from 09:15 up to (but not including) 16:00
        post_nine = day_df[
            (day_df.index.time > time(9, 0)) &
            (day_df.index.time < time(EOD_UK_HOUR, 0))
        ]
        if post_nine.empty:
            continue

        # ── Phase 1: find first break ──────────────────────────────────────
        direction     = None
        break_candle_idx = None

        for idx, row in post_nine.iterrows():
            broke_high = row["High"] > nine_high
            broke_low  = row["Low"]  < nine_low

            if broke_high and broke_low:
                direction = None        # both on same candle → skip day
                break
            if broke_high:
                direction = "BUY"
                break_candle_idx = idx
                break
            if broke_low:
                direction = "SELL"
                break_candle_idx = idx
                break

        if direction is None:
            continue

        # Candles strictly after the break candle
        after_break = post_nine[post_nine.index > break_candle_idx]
        if after_break.empty:
            continue

        # ── Phase 2: wait for retest (with invalidation) ──────────────────
        entry_candle_idx = None

        for idx, row in after_break.iterrows():
            if direction == "BUY":
                # Invalidated: price drops back below 9AM low before retesting high
                if row["Low"] < nine_low:
                    break
                # Retest: price returns to the broken high
                if row["Low"] <= nine_high:
                    entry_candle_idx = idx
                    break
            else:  # SELL
                # Invalidated: price rises back above 9AM high before retesting low
                if row["High"] > nine_high:
                    break
                # Retest: price returns to the broken low
                if row["High"] >= nine_low:
                    entry_candle_idx = idx
                    break

        if entry_candle_idx is None:
            continue

        # ── Phase 3: set entry, SL, TP ────────────────────────────────────
        if direction == "BUY":
            entry = nine_high
            sl    = nine_low
            risk  = entry - sl
            tp    = entry + 2 * risk
        else:
            entry = nine_low
            sl    = nine_high
            risk  = sl - entry
            tp    = entry - 2 * risk

        if risk <= 0:
            continue

        # ── Phase 4: resolve trade ────────────────────────────────────────
        # Include the entry candle itself (price can hit TP/SL on entry bar)
        resolution_candles = day_df[day_df.index >= entry_candle_idx]
        # Restrict to within trading day
        resolution_candles = resolution_candles[
            resolution_candles.index.time <= time(EOD_UK_HOUR, 0)
        ]

        result = resolve_trade(resolution_candles, entry, sl, tp, direction)

        # ── Phase 5: update balance ───────────────────────────────────────
        balance_at_entry = balance
        if result in ("W", "EOD-W"):
            pnl_pct = WIN_PCT
        else:
            pnl_pct = -LOSS_PCT

        pnl_usd  = balance_at_entry * pnl_pct
        balance += pnl_usd

        trades.append({
            "Date":      str(date),
            "Direction": direction,
            "Entry":     round(entry, 3),
            "SL":        round(sl, 3),
            "TP":        round(tp, 3),
            "Result":    result,
            "P&L %":     f"{pnl_pct*100:+.1f}%",
            "Balance":   round(balance, 2),
        })

    return trades, balance


def compute_stats(trades, final_balance):
    total   = len(trades)
    wins    = sum(1 for t in trades if t["Result"] in ("W", "EOD-W"))
    losses  = total - wins
    win_rate = (wins / total * 100) if total else 0

    gross_wins   = wins   * WIN_PCT  * STARTING_BALANCE   # approximation using fixed %
    gross_losses = losses * LOSS_PCT * STARTING_BALANCE

    # Proper compounded gross figures
    gross_wins_usd   = 0.0
    gross_losses_usd = 0.0
    for t in trades:
        pnl = float(t["P&L %"].replace("%", "")) / 100
        # Recover balance before trade
        bal_before = t["Balance"] / (1 + pnl) if pnl != 0 else t["Balance"]
        if pnl > 0:
            gross_wins_usd   += bal_before * WIN_PCT
        else:
            gross_losses_usd += bal_before * LOSS_PCT

    profit_factor = (gross_wins_usd / gross_losses_usd) if gross_losses_usd else float("inf")
    total_pct     = (final_balance / STARTING_BALANCE - 1) * 100

    # Streaks
    results_bin  = [1 if t["Result"] in ("W", "EOD-W") else 0 for t in trades]
    max_win_streak = max_loss_streak = 0
    cur_win = cur_loss = 0
    for r in results_bin:
        if r == 1:
            cur_win += 1
            cur_loss = 0
        else:
            cur_loss += 1
            cur_win  = 0
        max_win_streak  = max(max_win_streak,  cur_win)
        max_loss_streak = max(max_loss_streak, cur_loss)

    return {
        "Total Trades":          total,
        "Wins":                  wins,
        "Losses":                losses,
        "Win Rate":              f"{win_rate:.1f}%",
        "Profit Factor":         f"{profit_factor:.2f}",
        "Total % Gained":        f"{total_pct:+.2f}%",
        "Final Balance":         f"${final_balance:,.2f}",
        "Longest Win Streak":    max_win_streak,
        "Longest Loss Streak":   max_loss_streak,
    }


def print_results(stats, trades):
    print("\n" + "=" * 55)
    print("  9AM XAUUSD BREAKOUT/RETEST BACKTEST RESULTS")
    print("=" * 55)
    for k, v in stats.items():
        print(f"  {k:<25} {v}")
    print("=" * 55)

    print(f"\n{'Date':<12} {'Dir':<5} {'Entry':>8} {'SL':>8} {'TP':>8} {'Result':<8} {'P&L':>6}  {'Balance':>10}")
    print("-" * 72)
    for t in trades:
        print(
            f"{t['Date']:<12} {t['Direction']:<5} {t['Entry']:>8.3f} "
            f"{t['SL']:>8.3f} {t['TP']:>8.3f} {t['Result']:<8} "
            f"{t['P&L %']:>6}  ${t['Balance']:>9,.2f}"
        )
    print("-" * 72)


def plot_equity_curve(trades):
    if not trades:
        print("No trades to plot.")
        return

    dates    = [t["Date"] for t in trades]
    balances = [t["Balance"] for t in trades]
    x        = list(range(len(trades)))

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(x, balances, color="#1f77b4", linewidth=1.5, label="Account Balance")
    ax.axhline(STARTING_BALANCE, color="grey", linestyle="--", linewidth=0.8, label=f"Start ${STARTING_BALANCE:,}")

    # Colour individual bars
    for i, t in enumerate(trades):
        colour = "#2ca02c" if t["Result"] in ("W", "EOD-W") else "#d62728"
        ax.axvline(i, color=colour, alpha=0.15, linewidth=1)

    ax.set_title("9AM XAUUSD — Equity Curve", fontsize=14, fontweight="bold")
    ax.set_xlabel("Trade #")
    ax.set_ylabel("Account Balance (USD)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("Loading data…")
    df = load_data(DATA_PATH)
    print(f"  {len(df):,} candles loaded  ({df.index[0].date()} to {df.index[-1].date()})")

    print("Running backtest…")
    trades, final_balance = run_backtest(df)

    stats = compute_stats(trades, final_balance)
    print_results(stats, trades)
    plot_equity_curve(trades)
