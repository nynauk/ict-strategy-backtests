#CISD
import pandas as pd

# Load data
df = pd.read_csv('5min1000Cnas100.csv', parse_dates=True, index_col=0)

# Normalize column names
df.columns = [c.lower() for c in df.columns]

def identify_cisd(df):
    signals = []
    for i in range(1, len(df)):
        prev = df.iloc[i-1]
        curr = df.iloc[i]
        # Bullish CISD: price dips below previous low then closes above it
        if curr['low'] < prev['low'] and curr['close'] > prev['low']:
            signals.append({
                'timestamp': curr.name,
                'type': 'bullish',
                'level': prev['low']
            })
        # Bearish CISD: price spikes above previous high then closes below it
        if curr['high'] > prev['high'] and curr['close'] < prev['high']:
            signals.append({
                'timestamp': curr.name,
                'type': 'bearish',
                'level': prev['high']
            })
    return pd.DataFrame(signals)

signals = identify_cisd(df)
print(signals)
