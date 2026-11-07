import pandas as pd
import plotly.graph_objects as go
from datetime import time

df = pd.read_csv(
    r'C:\Users\Ayaan\PycharmProjects\csv back testing\5min1000Cnas100.csv',
    header=None,
    names=['timestamp','open','high','low','close','volume'],
    parse_dates=['timestamp']
)

df['time'] = df['timestamp'].dt.time

killzones = {
    'Asia':   (time(1, 0),  time(7, 0)),
    'London': (time(7, 0),  time(10,0)),
    'NewYork':(time(12,0),  time(15,0))
}

results = []
for date, day in df.groupby(df['timestamp'].dt.date):
    day = day.sort_values('timestamp')
    for name, (start, end) in killzones.items():
        segment = day[(day['time'] >= start) & (day['time'] < end)]
        for i in range(1, len(segment)):
            prev, curr = segment.iloc[i-1], segment.iloc[i]
            if curr['low'] > prev['high']:
                low, high, fvg_type = prev['high'], curr['low'], 'bullish'
            elif curr['high'] < prev['low']:
                low, high, fvg_type = curr['low'], prev['low'], 'bearish'
            else:
                continue
            mid = (low + high) / 2
            results.append({
                'prev_timestamp': prev['timestamp'],
                'timestamp':      curr['timestamp'],
                'fvg_low':        low,
                'fvg_high':       high,
                'fvg_mid':        mid,
                'fvg_type':       fvg_type
            })

gaps_df = pd.DataFrame(results)
if gaps_df.empty:
    raise ValueError("No fair value gaps found.")
latest = gaps_df.iloc[-1]
prev_time, curr_time = latest['prev_timestamp'], latest['timestamp']
plot_df = df[(df['timestamp'] >= prev_time) & (df['timestamp'] <= curr_time)]

fig = go.Figure(data=[go.Candlestick(
    x=plot_df['timestamp'],
    open=plot_df['open'],
    high=plot_df['high'],
    low=plot_df['low'],
    close=plot_df['close']
)])
fig.add_shape(
    type="rect",
    x0=prev_time,
    x1=curr_time,
    y0=latest['fvg_low'],
    y1=latest['fvg_high'],
    xref="x", yref="y",
    opacity=0.3
)
fig.update_layout(title="Most Recent Fair Value Gap", xaxis_title="Timestamp", yaxis_title="Price")
fig.show()
