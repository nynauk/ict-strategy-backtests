#4HRHL

# Prev4HourHighLow.py
import os
import pandas as pd

# locate CSV in same folder as script
base = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(base, '5min1000Cnas100.csv')

# load and parse data
df = pd.read_csv(
    csv_path,
    header=None,
    names=['datetime','open','high','low','close','volume'],
    parse_dates=['datetime']
)
df.set_index('datetime', inplace=True)

# resample into 4-hour periods (00:00-04:00, 04:00-08:00, …) using left‐label
four_hour = df.resample('4H', label='left', closed='left').agg(
    period_high=('high','max'),
    period_low =('low', 'min')
).reset_index().rename(columns={'datetime':'PeriodStart'})

# compute previous 4-hour high/low and its start time
four_hour['Prev4HStart'] = four_hour['PeriodStart'].shift(1)
four_hour['P4H'] = four_hour['period_high'].shift(1)
four_hour['P4L'] = four_hour['period_low'].shift(1)

print(four_hour[['PeriodStart','Prev4HStart','P4H','P4L']])
