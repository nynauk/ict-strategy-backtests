#CHoCH
import pandas as pd
from datetime import time

# load CSV
df = pd.read_csv(
    '5min1000Cnas100.csv',
    header=None,
    names=['timestamp','open','high','low','close','volume'],
    parse_dates=['timestamp'],
    skip_blank_lines=True
)

# prepare date and time columns
df['date'] = df['timestamp'].dt.date
df['t'] = df['timestamp'].dt.time

# killzone ranges (London timezone) from Fair Value gap.py :contentReference[oaicite:0]{index=0}
killzones = {
    'Asia':   (time(1, 0),  time(7, 0)),
    'London': (time(7, 0),  time(10,0)),
    'NewYork':(time(12,0),  time(15,0))
}

results = []

for date, day in df.groupby('date'):
    day = day.sort_values('timestamp')
    for name, (start, end) in killzones.items():
        segment = day[(day['t'] >= start) & (day['t'] < end)]
        if len(segment) < 2:
            continue

        # initial trend based on first two bars
        first, second = segment.iloc[0], segment.iloc[1]
        if second['high'] > first['high'] and second['low'] > first['low']:
            trend = 'up'
        elif second['high'] < first['high'] and second['low'] < first['low']:
            trend = 'down'
        else:
            trend = None

        if not trend:
            continue

        for i in range(1, len(segment)):
            prev = segment.iloc[i-1]
            curr = segment.iloc[i]

            # bullish change of character (downtrend → up)
            if trend == 'down' and curr['high'] > prev['high']:
                low = prev['high']
                high = curr['high']
                mid = (low + high) / 2
                results.append({
                    'date': date,
                    'killzone': name,
                    'type': 'bullish',
                    'ch_low': low,
                    'ch_high': high,
                    'ch_mid': mid
                })
                trend = 'up'

            # bearish change of character (uptrend → down)
            elif trend == 'up' and curr['low'] < prev['low']:
                high = prev['low']
                low = curr['low']
                mid = (low + high) / 2
                results.append({
                    'date': date,
                    'killzone': name,
                    'type': 'bearish',
                    'ch_low': low,
                    'ch_high': high,
                    'ch_mid': mid
                })
                trend = 'down'

# output results
if results:
    out = pd.DataFrame(results)
    print(out[['date','killzone','type','ch_low','ch_high','ch_mid']])
else:
    print('No change-of-character events found in specified killzones.')
