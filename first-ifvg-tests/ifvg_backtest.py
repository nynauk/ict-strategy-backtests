import pandas as pd
import ta
import plotly.graph_objects as go

CSV_FILE = "EURUSD_M5_.csv"
MAX_CANDLES_DISRESPECT = 7

# Load and preprocess data
df = pd.read_csv(CSV_FILE)
df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
df = df.sort_values('Datetime')
df = df[df['Datetime'].dt.weekday < 5]
df['DateOnly'] = df['Datetime'].dt.date

# Resample to HTF
df_h1 = df.resample("1h", on="Datetime").agg({"Open":"first","High":"max","Low":"min","Close":"last"}).dropna()
df_h4 = df.resample("4h", on="Datetime").agg({"Open":"first","High":"max","Low":"min","Close":"last"}).dropna()
df_d1 = df.resample("1d", on="Datetime").agg({"Open":"first","High":"max","Low":"min","Close":"last"}).dropna()

df_h4["EMA50"] = ta.trend.ema_indicator(df_h4["Close"], window=50)
df_d1["EMA50"] = ta.trend.ema_indicator(df_d1["Close"], window=50)

def get_htf_bias(time):
    try:
        h4_row = df_h4.loc[:time].iloc[-1]
        d1_row = df_d1.loc[:time].iloc[-1]
    except:
        return None
    if h4_row["Close"] > h4_row["EMA50"] and d1_row["Close"] > d1_row["EMA50"]:
        return "bullish"
    if h4_row["Close"] < h4_row["EMA50"] and d1_row["Close"] < d1_row["EMA50"]:
        return "bearish"
    return None

def find_fvgs(df_htf):
    fvgs = []
    for i in range(2, len(df_htf)):
        prev2 = df_htf.iloc[i-2]
        curr = df_htf.iloc[i]
        if curr["Low"] > prev2["High"]:
            fvgs.append({"type":"bullish","low":prev2["High"],"high":curr["Low"]})
        if curr["High"] < prev2["Low"]:
            fvgs.append({"type":"bearish","low":curr["High"],"high":prev2["Low"]})
    return fvgs

def find_obs(df_htf):
    obs = []
    for i in range(1, len(df_htf)):
        prev = df_htf.iloc[i-1]
        curr = df_htf.iloc[i]
        if prev["Close"] < prev["Open"] and curr["Close"] > prev["High"]:
            obs.append({"type":"bullish","low":prev["Low"],"high":prev["High"]})
        if prev["Close"] > prev["Open"] and curr["Close"] < prev["Low"]:
            obs.append({"type":"bearish","low":prev["Low"],"high":prev["High"]})
    return obs

fvgs_h1, fvgs_h4 = find_fvgs(df_h1), find_fvgs(df_h4)
obs_h1, obs_h4 = find_obs(df_h1), find_obs(df_h4)

def sweep_tapped_poi(sweep_price, sweep_type):
    pois = fvgs_h1 + fvgs_h4 + obs_h1 + obs_h4
    for poi in pois:
        if poi["type"] == "bullish" and sweep_type == "low":
            if poi["low"] <= sweep_price <= poi["high"]:
                return True
        if poi["type"] == "bearish" and sweep_type == "high":
            if poi["low"] <= sweep_price <= poi["high"]:
                return True
    return False

def find_swing_high(data, idx, lookback=3):
    if idx < lookback or idx+lookback >= len(data): return None
    high = data.iloc[idx]['High']
    if all(high > data.iloc[idx-i]['High'] for i in range(1, lookback+1)) and \
       all(high >= data.iloc[idx+i]['High'] for i in range(1, lookback+1)):
        return high
    return None

def find_swing_low(data, idx, lookback=3):
    if idx < lookback or idx+lookback >= len(data): return None
    low = data.iloc[idx]['Low']
    if all(low < data.iloc[idx-i]['Low'] for i in range(1, lookback+1)) and \
       all(low <= data.iloc[idx+i]['Low'] for i in range(1, lookback+1)):
        return low
    return None

def simulate_trade(entry, sl, tp, data):
    for _, row in data.iterrows():
        if row["Low"] <= sl and row["High"] >= tp:
            return "LOSS"
        if row["Low"] <= sl:
            return "LOSS"
        if row["High"] >= tp:
            return "WIN"
    return "OPEN"

trades = []

for date in df["DateOnly"].unique():
    day_data = df[df["DateOnly"] == date]
    asia = day_data[(day_data["Datetime"].dt.hour >= 1) & (day_data["Datetime"].dt.hour < 7)]
    if asia.empty: continue

    asia_high, asia_low = asia["High"].max(), asia["Low"].min()
    killzone = day_data[(day_data["Datetime"].dt.hour >= 7) & (day_data["Datetime"].dt.hour < 10)]
    if killzone.empty: continue

    sweep_type, sweep_idx = None, None
    for i, row in killzone.iterrows():
        if row["High"] > asia_high:
            sweep_type, sweep_idx = "high", i
            break
    for i, row in killzone.iterrows():
        if row["Low"] < asia_low:
            sweep_type, sweep_idx = "low", i
            break
    if sweep_type is None: continue

    sweep_price = killzone.loc[sweep_idx]["High"] if sweep_type == "high" else killzone.loc[sweep_idx]["Low"]
    if not sweep_tapped_poi(sweep_price, sweep_type):
        continue

    # CHoCH confirmation: break of opposite swing
    post_sweep = killzone.loc[killzone.index >= sweep_idx]
    choch = False
    for j in range(1, len(post_sweep)):
        if sweep_type == "high" and post_sweep.iloc[j]["Close"] < post_sweep.iloc[:j]["Low"].min():
            choch = True
            break
        if sweep_type == "low" and post_sweep.iloc[j]["Close"] > post_sweep.iloc[:j]["High"].max():
            choch = True
            break
    if not choch:
        continue

    search_data = killzone.loc[killzone.index <= sweep_idx].tail(3)
    search_data = pd.concat([search_data, killzone.loc[killzone.index >= sweep_idx]])

    found_trade = None

    for i in range(2, len(search_data)):
        prev2, curr = search_data.iloc[i-2], search_data.iloc[i]

        # Bearish setup
        if curr["Low"] > prev2["High"] and sweep_type == "high":
            for j in range(i+1, min(i+1+MAX_CANDLES_DISRESPECT, len(search_data))):
                future = search_data.iloc[j]
                if future["Close"] < prev2["High"] and future.name >= sweep_idx:
                    bias = get_htf_bias(future["Datetime"])
                    if bias != "bearish": continue
                    swings = [find_swing_high(search_data, k) for k in range(i,j)]
                    sl = max([s for s in swings if s]) if any(swings) else search_data.iloc[i:j]["High"].max()
                    entry = future["Close"]
                    tp = asia_low  # Target opposite liquidity
                    trade_data = search_data.iloc[j+1:]
                    result = simulate_trade(entry, sl, tp, trade_data)
                    found_trade = dict(Date=date, Type="SELL", Entry=entry, SL=sl, TP=tp, Result=result)
                    break

        # Bullish setup
        if curr["High"] < prev2["Low"] and sweep_type == "low":
            for j in range(i+1, min(i+1+MAX_CANDLES_DISRESPECT, len(search_data))):
                future = search_data.iloc[j]
                if future["Close"] > prev2["Low"] and future.name >= sweep_idx:
                    bias = get_htf_bias(future["Datetime"])
                    if bias != "bullish": continue
                    swings = [find_swing_low(search_data, k) for k in range(i,j)]
                    sl = min([s for s in swings if s]) if any(swings) else search_data.iloc[i:j]["Low"].min()
                    entry = future["Close"]
                    tp = asia_high
                    trade_data = search_data.iloc[j+1:]
                    result = simulate_trade(entry, sl, tp, trade_data)
                    found_trade = dict(Date=date, Type="BUY", Entry=entry, SL=sl, TP=tp, Result=result)
                    break

        if found_trade:
            trades.append(found_trade)
            break

results = pd.DataFrame(trades)
if not results.empty:
    wins = (results["Result"] == "WIN").sum()
    losses = (results["Result"] == "LOSS").sum()
    total = len(results)
    win_rate = wins/total*100
    net_rr = wins*2 - losses
    print("===== BACKTEST RESULTS =====")
    print(f"Total Trades: {total}")
    print(f"Wins: {wins}, Losses: {losses}")
    print(f"Win Rate: {win_rate:.1f}%")
    print(f"Net R Gain: {net_rr:.2f}R")
    results.to_csv("IFVG_CHoCH_Liquidity.csv", index=False)
else:
    print("No trades found.")

# === Plot one trade ===
def plot_trade(trade_idx=0):
    if results.empty or trade_idx >= len(results):
        print("No trades to plot.")
        return
    trade = results.iloc[trade_idx]
    day_data = df[df["DateOnly"] == trade["Date"]]
    plot_data = day_data[(day_data["Datetime"].dt.hour >= 1) & (day_data["Datetime"].dt.hour < 10)]

    fig = go.Figure(data=[go.Candlestick(
        x=plot_data["Datetime"], open=plot_data["Open"], high=plot_data["High"],
        low=plot_data["Low"], close=plot_data["Close"]
    )])
    fig.add_hline(y=trade["Entry"], line=dict(color="white", dash="dash"), annotation_text="Entry")
    fig.add_hline(y=trade["SL"], line=dict(color="red", dash="dash"), annotation_text="SL")
    fig.add_hline(y=trade["TP"], line=dict(color="green", dash="dash"), annotation_text="TP")
    fig.update_layout(template="plotly_dark", title=f"Trade {trade_idx} on {trade['Date']} ({trade['Type']})")
    fig.show()

# To plot: call plot_trade(0)
