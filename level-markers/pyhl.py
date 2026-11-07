# PrevYearHighLow.py
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
daily['year'] = daily['date'].dt.year

yearly = daily.groupby('year').agg(
    yearly_high=('daily_high','max'),
    yearly_low =('daily_low','min')
).reset_index()
yearly['PrevYearStart'] = pd.to_datetime(yearly['year'].astype(str) + '-01-01')
yearly['PYH'] = yearly['yearly_high'].shift(1)
yearly['PYL'] = yearly['yearly_low'].shift(1)
yearly['PrevYearStart'] = yearly['PrevYearStart'].shift(1)
yearly = yearly[['year','PrevYearStart','PYH','PYL']]

daily = daily.merge(yearly, on='year', how='left')
print(daily[['date','PrevYearStart','PYH','PYL']])
