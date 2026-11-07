import os
import pandas as pd

base = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(base, '5min1000Cnas100.csv')

df = pd.read_csv(
    csv_path,
    header=None,
    names=['datetime','open','high','low','close','volume'],
    parse_dates=['datetime']
)
df.set_index('datetime', inplace=True)

daily = df.groupby(df.index.date).agg(
    daily_high=('high', 'max'),
    daily_low =('low',  'min')
)
daily.index.name = 'date'
daily = daily.reset_index()

daily['PDHLdate'] = daily['date'].shift(1)
daily['PDH']       = daily['daily_high'].shift(1)
daily['PDL']       = daily['daily_low'].shift(1)

print(daily[['date','PDHLdate','PDH','PDL']])
