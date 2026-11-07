from asiasweephighlow import find_asia_session_sweeps
from inverse_fair_value_gap import find_ifvg_after_sweep
import pandas as pd
import plotly.graph_objects as go

# Load and format the data
df = pd.read_csv("Formatted_GBPUSD_M5_Data.csv")
df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
df.set_index('datetime', inplace=True)
df = df[['Open', 'High', 'Low', 'Close']].sort_index()

# Extract sweep information from pre-session analysis
sweep_results = find_asia_session_sweeps("Formatted_GBPUSD_M5_Data.csv")

# Initialize performance tracking variables
trades = []
wins = losses = 0
total_rr = 0
max_rr = 0
win_streak = loss_streak = max_win_streak = max_loss_streak = 0
example_trade = None

# Evaluate each sweep result to determine a valid trade
for result in sweep_results:
    trade = None

    if result['asia_high_swept']:
        # If Asia High was swept, look for sell opportunity after the sweep
        sweep_time = pd.to_datetime(f"{result['date']} {result['high_sweep_time']}")
        trade = find_ifvg_after_sweep(
            df, 'high', sweep_time,
            asia_high=result['asia_high'], asia_low=result['asia_low']
        )

    elif result['asia_low_swept']:
        # If Asia Low was swept, look for buy opportunity after the sweep
        sweep_time = pd.to_datetime(f"{result['date']} {result['low_sweep_time']}")
        trade = find_ifvg_after_sweep(
            df, 'low', sweep_time,
            asia_high=result['asia_high'], asia_low=result['asia_low']
        )

    # If a trade is found and actually triggered
    if trade and trade['entry_time'] and trade['outcome']:
        trade['date'] = result['date']
        trade['asia_high'] = result['asia_high']
        trade['asia_low'] = result['asia_low']
        trades.append(trade)

        if not example_trade:
            example_trade = trade  # Capture first trade for visualization

        if trade['outcome'] == 'win':
            total_rr += trade['rr']
            max_rr = max(max_rr, trade['rr'])
            wins += 1
            win_streak += 1
            loss_streak = 0
        elif trade['outcome'] == 'loss':
            total_rr -= 1
            losses += 1
            loss_streak += 1
            win_streak = 0

        max_win_streak = max(max_win_streak, win_streak)
        max_loss_streak = max(max_loss_streak, loss_streak)

        # Print individual trade summary
        print(f"\n=== {trade['date']} ===")
        print(f"Direction: {trade['direction'].upper()}")
        print(f"Entry: {trade['entry_price']} at {trade['entry_time']}")
        print(f"SL: {trade['stop_loss']}, TP: {trade['take_profit']}")
        print(f"FVG: {trade['fvg_low']} → {trade['fvg_high']}")
        print(f"RR: {trade['rr']}, Outcome: {trade['outcome'].upper()}")

# Output all trades
print("\n=== All Trades ===")
for t in trades:
    print(f"{t['date']} | {t['direction'].upper()} | Entry: {t['entry_price']} | SL: {t['stop_loss']} | "
          f"TP: {t['take_profit']} | RR: {t['rr']} | Outcome: {t['outcome'].upper()}")

# Performance summary calculations
total_trades = wins + losses
win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
avg_rr = round(sum(t['rr'] for t in trades if t['outcome'] == 'win') / max(wins, 1), 2)

# Summary output
print("\n=== Performance Summary ===")
print(f"Total Trades: {total_trades}")
print(f"Wins: {wins}, Losses: {losses}")
print(f"Win Rate: {win_rate:.2f}%")
print(f"Average RR on Wins: {avg_rr}")
print(f"Max Win Streak: {max_win_streak}")
print(f"Max Loss Streak: {max_loss_streak}")
print(f"Max Single Trade RR: {max_rr}")
print(f"Total Profit (in RR): {round(total_rr, 2)} R")

# Visualization of a single trade
target_date = '2025-05-30'

for trade in trades:
    if str(trade['date']) == target_date:
        day_data = df[df.index.date == pd.to_datetime(target_date).date()]
        entry_time = trade['entry_time']

        # Confirm entry candle exists
        entry_idx = day_data.index.get_indexer([entry_time])[0]
        if entry_idx == -1:
            print(f"Entry time {entry_time} not found for {trade['date']}. Skipping visualization.")
            continue

        # Determine start of FVG zone (up to 7 candles before entry)
        fvg_start_idx = max(0, entry_idx - 7)
        fvg_start_time = day_data.index[fvg_start_idx]
        fvg_end_time = entry_time

        # Search for candle where TP or SL is hit
        trade_exit_time = None
        for idx in day_data.index[entry_idx:]:
            candle = day_data.loc[idx]
            if trade['direction'] == 'buy' and (candle['High'] >= trade['take_profit'] or candle['Low'] <= trade['stop_loss']):
                trade_exit_time = idx
                break
            elif trade['direction'] == 'sell' and (candle['Low'] <= trade['take_profit'] or candle['High'] >= trade['stop_loss']):
                trade_exit_time = idx
                break

        # Plot the chart
        fig = go.Figure()

        # Price candles
        fig.add_trace(go.Candlestick(
            x=day_data.index,
            open=day_data['Open'],
            high=day_data['High'],
            low=day_data['Low'],
            close=day_data['Close'],
            name="Price"
        ))

        # Asia high/low horizontal lines
        fig.add_trace(go.Scatter(
            x=[day_data.index[0], day_data.index[-1]],
            y=[trade['asia_high'], trade['asia_high']],
            mode="lines",
            line=dict(color="blue", dash="dot"),
            name="Asia High"
        ))
        fig.add_trace(go.Scatter(
            x=[day_data.index[0], day_data.index[-1]],
            y=[trade['asia_low'], trade['asia_low']],
            mode="lines",
            line=dict(color="orange", dash="dot"),
            name="Asia Low"
        ))

        # FVG candle marked as vertical dashed line
        fig.add_shape(type="line",
                      x0=fvg_start_time, x1=fvg_start_time,
                      y0=day_data['Low'].min(), y1=day_data['High'].max(),
                      line=dict(color='green' if trade['direction'] == 'buy' else 'red', width=2, dash='dot'),
                      name="FVG Candle")
        fig.add_annotation(
            x=fvg_start_time,
            y=day_data['High'].max(),
            text="IFVG",
            showarrow=False,
            font=dict(size=10, color='white'),
            align='center',
            yshift=10
        )

        # Mark entry/SL/TP levels
        fig.add_hline(y=trade['entry_price'], line_color="green", annotation_text="Entry")
        fig.add_hline(y=trade['stop_loss'], line_color="red", annotation_text="Stop Loss")
        fig.add_hline(y=trade['take_profit'], line_color="purple", annotation_text="Take Profit")

        # Entry star
        if entry_time in day_data.index:
            entry_candle = day_data.loc[entry_time]
            fig.add_trace(go.Scatter(
                x=[entry_time],
                y=[entry_candle['High'] + 0.0005],
                mode='markers+text',
                marker=dict(symbol='star', size=12, color='gold'),
                text=["Entry"],
                textposition="top center",
                name="Entry Marker"
            ))

        # Exit marker
        if trade_exit_time:
            exit_price = day_data.loc[trade_exit_time]['Close']
            fig.add_trace(go.Scatter(
                x=[trade_exit_time],
                y=[exit_price],
                mode='markers+text',
                marker=dict(symbol='x', size=10, color='red'),
                text=["Exit"],
                textposition="bottom center",
                name="Exit Marker"
            ))

        fig.update_layout(title=f"Full Day View: {trade['date']} ({trade['direction'].upper()})",
                          xaxis_rangeslider_visible=False,
                          template="plotly_dark")
        fig.show()
        break
else:
    print(f"\nNo trade found for date: {target_date}")
