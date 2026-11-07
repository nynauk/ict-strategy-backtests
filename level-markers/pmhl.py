# PrevMonthHighLow.py
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
daily['month'] = daily['date'].dt.to_period('M')

monthly = daily.groupby('month').agg(
    monthly_high=('daily_high','max'),
    monthly_low =('daily_low','min')
).reset_index()
monthly['PrevMonthStart'] = monthly['month'].dt.to_timestamp()
monthly['PMH'] = monthly['monthly_high'].shift(1)
monthly['PML'] = monthly['monthly_low'].shift(1)
monthly['PrevMonthStart'] = monthly['PrevMonthStart'].shift(1)
monthly = monthly[['month','PrevMonthStart','PMH','PML']]

daily = daily.merge(monthly, on='month', how='left')
print(daily[['date','PrevMonthStart','PMH','PML']])
