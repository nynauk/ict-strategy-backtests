import pandas as pd

def find_ifvg_after_sweep(df, sweep_type, sweep_time, asia_high, asia_low):
    df = df[df.index > sweep_time]
    df = df.between_time("07:00", "10:00")

    for i in range(2, len(df) - 7):
        c1 = df.iloc[i - 2]
        c2 = df.iloc[i - 1]
        c3 = df.iloc[i]

        fvg_start_time = c3.name
        next_7 = df.iloc[i + 1:i + 8]

        if sweep_type == 'high':  # looking for bearish IFVG after bullish FVG is disrespected
            if c3['Low'] > c1['High']:  # Bullish FVG
                fvg_low = c1['High']
                fvg_high = c3['Low']
                for j in range(len(next_7)):
                    disrespect = next_7.iloc[j]
                    if disrespect['Close'] < fvg_low:
                        entry_time = next_7.index[j]
                        entry_price = disrespect['Close']
                        swing_high = df.loc[fvg_start_time:entry_time]['High'].max()
                        stop_loss = swing_high
                        take_profit = asia_low
                        risk = abs(stop_loss - entry_price)
                        rr = abs(take_profit - entry_price) / risk if risk != 0 else 0
                        outcome = "win" if df.loc[entry_time:]['Low'].min() <= take_profit else "loss"
                        return {
                            "direction": "sell",
                            "entry_time": entry_time,
                            "entry_price": entry_price,
                            "stop_loss": stop_loss,
                            "take_profit": take_profit,
                            "fvg_low": fvg_low,
                            "fvg_high": fvg_high,
                            "rr": round(rr, 2),
                            "outcome": outcome
                        }

        elif sweep_type == 'low':  # looking for bullish IFVG after bearish FVG is disrespected
            if c3['High'] < c1['Low']:  # Bearish FVG
                fvg_high = c1['Low']
                fvg_low = c3['High']
                for j in range(len(next_7)):
                    disrespect = next_7.iloc[j]
                    if disrespect['Close'] > fvg_high:
                        entry_time = next_7.index[j]
                        entry_price = disrespect['Close']
                        swing_low = df.loc[fvg_start_time:entry_time]['Low'].min()
                        stop_loss = swing_low
                        take_profit = asia_high
                        risk = abs(entry_price - stop_loss)
                        rr = abs(take_profit - entry_price) / risk if risk != 0 else 0
                        outcome = "win" if df.loc[entry_time:]['High'].max() >= take_profit else "loss"
                        return {
                            "direction": "buy",
                            "entry_time": entry_time,
                            "entry_price": entry_price,
                            "stop_loss": stop_loss,
                            "take_profit": take_profit,
                            "fvg_low": fvg_low,
                            "fvg_high": fvg_high,
                            "rr": round(rr, 2),
                            "outcome": outcome
                        }

    return None
