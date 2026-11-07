import pandas as pd
from datetime import datetime, timedelta

def find_asia_session_sweeps(csv_path):
    df = pd.read_csv(csv_path)
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    df.set_index('datetime', inplace=True)
    df = df.sort_index()

    results = []

    grouped = df.groupby(df.index.date)
    for date, group in grouped:
        try:
            asia_start = datetime.combine(date, datetime.min.time()) + timedelta(hours=1)
            asia_end = asia_start + timedelta(hours=6)
            sweep_start = asia_end
            sweep_end = sweep_start + timedelta(hours=3)

            asia_range = group.between_time(asia_start.time(), asia_end.time())
            sweep_range = group.between_time(sweep_start.time(), sweep_end.time())

            if asia_range.empty or sweep_range.empty:
                continue

            asia_high = asia_range['High'].max()
            asia_low = asia_range['Low'].min()

            high_sweep_time = None
            low_sweep_time = None

            for time, row in sweep_range.iterrows():
                if row['High'] > asia_high and high_sweep_time is None:
                    high_sweep_time = time
                if row['Low'] < asia_low and low_sweep_time is None:
                    low_sweep_time = time
                if high_sweep_time and low_sweep_time:
                    break

            results.append({
                'date': date,
                'asia_high': round(asia_high, 5),
                'asia_low': round(asia_low, 5),
                'asia_high_swept': high_sweep_time is not None,
                'asia_low_swept': low_sweep_time is not None,
                'high_sweep_time': high_sweep_time.strftime("%H:%M:%S") if high_sweep_time else None,
                'low_sweep_time': low_sweep_time.strftime("%H:%M:%S") if low_sweep_time else None
            })
        except Exception as e:
            print(f"Error on {date}: {e}")
            continue

    return results
