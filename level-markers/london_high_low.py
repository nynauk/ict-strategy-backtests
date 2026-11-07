# London_High_Low_History.py
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

london_mask = (df.index.time >= time(7, 0)) & (df.index.time <= time(16, 0))
london = df.loc[london_mask]

london_daily = london.groupby(london.index.date).agg(
    london_high=('high', 'max'),
    london_low =('low',  'min')
)
print(london_daily)
