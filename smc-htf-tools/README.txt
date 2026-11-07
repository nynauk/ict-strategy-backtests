Scripts created for higher-timeframe SMC utilities.

Files:
- htf_order_blocks.py : Detects bullish/bearish order blocks on M15/M30/H1/H4/D1, skipping Asia-session candles.
- htf_fvgs.py         : Detects bullish/bearish fair value gaps on the same timeframes.
- daily_bias.py       : Prints a simple Daily bias for your symbols using a 50 EMA heuristic.

Common notes:
- Update SYMBOLS to match your existing Asia Sweep scripts.
- Adjust ASIA_TZ and ASIA_* hours to match your broker server time (order blocks script excludes Asia by default; FVG script can exclude via EXCLUDE_ASIA).
- All scripts expect a logged-in MetaTrader 5 terminal on the same machine.
- CSV export can be toggled in each script.
- Heuristics are intentionally simple and fast; adapt thresholds to your style.
