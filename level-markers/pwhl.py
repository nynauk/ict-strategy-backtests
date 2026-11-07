# PrevWeekHighLow.py
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
    daily_high=('high','max'),
    daily_low =('low','min')
).rename_axis('date').reset_index()
daily['date'] = pd.to_datetime(daily['date'])
daily['week'] = daily['date'].dt.to_period('W-MON')

weekly = daily.groupby('week').agg(
    weekly_high=('daily_high','max'),
    weekly_low =('daily_low','min')
).reset_index()
weekly['PrevWeekStart'] = weekly['week'].dt.to_timestamp()
weekly['PWH'] = weekly['weekly_high'].shift(1)
weekly['PWL'] = weekly['weekly_low'].shift(1)
weekly['PrevWeekStart'] = weekly['PrevWeekStart'].shift(1)
weekly = weekly[['week','PrevWeekStart','PWH','PWL']]

daily = daily.merge(weekly, on='week', how='left')
print(daily[['date','PrevWeekStart','PWH','PWL']])
