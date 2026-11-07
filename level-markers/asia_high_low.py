# Asia_High_Low_History.py
import os
import pandas as pd
from datetime import time

base = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(base, '5min1000Cnas100.csv')

df = pd.read_csv(
    csv_path,
    header=None,
    names=['datetime','open','high','low','close','volume'],
    parse_dates=['datetime']
)
df.set_index('datetime', inplace=True)

asia_mask = (df.index.time >= time(0, 0)) & (df.index.time <= time(9, 0))
asia = df.loc[asia_mask]

asia_daily = asia.groupby(asia.index.date).agg(
    asia_high=('high', 'max'),
    asia_low =('low',  'min')
)
print(asia_daily)
