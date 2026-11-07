import pandas as pd
import plotly.graph_objects as go

# Load CSV
df = pd.read_csv("EURUSD_M5_.csv")
df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
df = df.sort_values('Datetime')

# Keep weekdays only
df = df[df['Datetime'].dt.weekday < 5]

# Get last 60 calendar days (2 months)
last_day = df['Datetime'].max()
first_day = last_day - pd.Timedelta(days=60)
df_recent = df[df['Datetime'] >= first_day].copy().reset_index(drop=True)
df_recent['DateOnly'] = df_recent['Datetime'].dt.date
unique_days = df_recent['DateOnly'].unique()

def find_first_ifvg_after_sweep(day_data, max_candles=7):
    s2 = day_data[(day_data['Datetime'].dt.hour >= 1) & (day_data['Datetime'].dt.hour < 7)]
    if s2.empty:
        return None, None, None, None

    asia_high = s2['High'].max()
    asia_low = s2['Low'].min()

    s1 = day_data[(day_data['Datetime'].dt.hour >= 7) & (day_data['Datetime'].dt.hour < 10)]
    if s1.empty:
        return None, asia_high, asia_low, None

    # Detect sweep
    sweep_type, sweep_idx = None, None
    for i, row in s1.iterrows():
        if row['High'] > asia_high:
            sweep_type, sweep_idx = "high", i
            break
    for i, row in s1.iterrows():
        if row['Low'] < asia_low:
            sweep_type, sweep_idx = "low", i
            break

    if sweep_type is None:
        return None, asia_high, asia_low, None

    # Get candles 1 before sweep
    data_for_ifvg = s1.loc[s1.index <= sweep_idx].tail(3)
    data_for_ifvg = pd.concat([data_for_ifvg, s1.loc[s1.index >= sweep_idx]])

    found_ifvgs = []
    for i in range(2, len(data_for_ifvg)):
        prev2 = data_for_ifvg.iloc[i-2]
        curr = data_for_ifvg.iloc[i]

        # Bullish FVG (Sell Setup)
        if curr['Low'] > prev2['High']:
            fvg_start, fvg_end, fvg_time = prev2['High'], curr['Low'], curr['Datetime']
            for j in range(i+1, min(i+1+max_candles, len(data_for_ifvg))):
                future = data_for_ifvg.iloc[j]
                if future['Close'] < fvg_start and future.name >= sweep_idx:
                    if sweep_type == "high":
                        segment = data_for_ifvg.iloc[i:j+1]
                        high_idx = segment['High'].idxmax()
                        found_ifvgs.append({
                            "start_time": fvg_time,
                            "end_time": future['Datetime'],
                            "start": fvg_start,
                            "end": fvg_end,
                            "type": "bearish",
                            "extreme_price": segment.loc[high_idx, 'High'],
                            "extreme_time": segment.loc[high_idx, 'Datetime']
                        })
                    break

        # Bearish FVG (Buy Setup)
        if curr['High'] < prev2['Low']:
            fvg_start, fvg_end, fvg_time = curr['High'], prev2['Low'], curr['Datetime']
            for j in range(i+1, min(i+1+max_candles, len(data_for_ifvg))):
                future = data_for_ifvg.iloc[j]
                if future['Close'] > fvg_start and future.name >= sweep_idx:
                    if sweep_type == "low":
                        segment = data_for_ifvg.iloc[i:j+1]
                        low_idx = segment['Low'].idxmin()
                        found_ifvgs.append({
                            "start_time": fvg_time,
                            "end_time": future['Datetime'],
                            "start": fvg_start,
                            "end": fvg_end,
                            "type": "bullish",
                            "extreme_price": segment.loc[low_idx, 'Low'],
                            "extreme_time": segment.loc[low_idx, 'Datetime']
                        })
                    break

    if not found_ifvgs:
        return None, asia_high, asia_low, sweep_type

    # Return the first IFVG based on disrespect candle time
    found_ifvgs.sort(key=lambda x: x["end_time"])
    return found_ifvgs[0], asia_high, asia_low, sweep_type


# Create candlestick chart
fig = go.Figure(data=[go.Candlestick(
    x=df_recent['Datetime'],
    open=df_recent['Open'],
    high=df_recent['High'],
    low=df_recent['Low'],
    close=df_recent['Close'],
    name="EURUSD"
)])

# Loop through each day and plot only first disrespected IFVG after sweep
for d in unique_days:
    day_data = df_recent[df_recent['DateOnly'] == d]
    ifvg, asia_high, asia_low, sweep_type = find_first_ifvg_after_sweep(day_data)

    if ifvg is None:
        continue  # Skip days without sweeps or IFVGs

    # Highlight 7AM–10AM box
    s1 = day_data[(day_data['Datetime'].dt.hour >= 7) & (day_data['Datetime'].dt.hour < 10)]
    fig.add_shape(type="rect",
                  x0=s1['Datetime'].min(), x1=s1['Datetime'].max(),
                  y0=s1['Low'].min(), y1=s1['High'].max(),
                  line=dict(color="yellow", width=1),
                  fillcolor="rgba(255,255,0,0.1)", layer="below")

    # Highlight Asia box
    s2 = day_data[(day_data['Datetime'].dt.hour >= 1) & (day_data['Datetime'].dt.hour < 7)]
    if not s2.empty:
        fig.add_shape(type="rect",
                      x0=s2['Datetime'].min(), x1=s2['Datetime'].max(),
                      y0=s2['Low'].min(), y1=s2['High'].max(),
                      line=dict(color="cyan", width=1),
                      fillcolor="rgba(0,255,255,0.1)", layer="below")

        extend_until = s2['Datetime'].min().replace(hour=10, minute=0)
        fig.add_shape(type="line", x0=s2['Datetime'].min(), x1=extend_until,
                      y0=asia_high, y1=asia_high, line=dict(color="cyan", width=1, dash="dot"))
        fig.add_shape(type="line", x0=s2['Datetime'].min(), x1=extend_until,
                      y0=asia_low, y1=asia_low, line=dict(color="cyan", width=1, dash="dot"))

    # Plot IFVG rectangle
    fig.add_shape(
        type="rect",
        x0=ifvg["start_time"],
        x1=ifvg["end_time"],
        y0=ifvg["start"],
        y1=ifvg["end"],
        line=dict(width=0),
        fillcolor="rgba(255,0,0,0.4)" if ifvg["type"] == "bearish" else "rgba(0,255,0,0.4)",
        layer="below"
    )

    # Mark highest or lowest point
    fig.add_trace(go.Scatter(
        x=[ifvg["extreme_time"]],
        y=[ifvg["extreme_price"]],
        mode="markers+text",
        text=["Extreme"],
        textposition="top center",
        marker=dict(color="white", size=8, symbol="x")
    ))

# Final layout
fig.update_layout(
    title="EURUSD M5 – Past 2 Months (First Disrespected IFVG After Sweep + Extreme Point)",
    template="plotly_dark",
    xaxis_rangeslider_visible=False,
    height=950
)

fig.update_xaxes(
    rangebreaks=[dict(bounds=["sat", "mon"])]  # remove weekend gaps
)

fig.show()
