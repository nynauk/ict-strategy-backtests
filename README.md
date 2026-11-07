# ICT and SMC strategy backtests (2025 to 2026)

The backtests I wrote while learning to test discretionary trading ideas mechanically, before and alongside
[SwiftPip](https://github.com/nynauk/swiftpip-research). Each folder takes a set of rules taught in ICT /
smart-money trading material (liquidity sweeps, fair value gaps, order blocks, market structure shifts),
writes them down precisely enough for a computer to follow, and replays them over historical bars from
MetaTrader 5 or Dukascopy exports.

The folders are in roughly the order I wrote them, and the code gets noticeably better along the way: from
ten-line level markers, to single-strategy scripts, to parameterised backtests with trade charts,
robustness tables and t-statistics. Read the results with the caveats at the end, because most of these
scripts do not charge any trading costs.

## What is here

| Folder | Period written | What it tests |
|---|---|---|
| `level-markers/` | May 2025 | First scripts: mark Asia, London, previous day, week, month and year highs and lows, CHoCH, CISD and fair value gaps on a CSV of bars |
| `first-ifvg-tests/` | Jul 2025 | First inverse-FVG backtests on EURUSD and GBPUSD M5, with trade logs |
| `smc-htf-tools/`, `asia-sweep-ifvg-mt5/` | Aug to Sep 2025 | Order block, FVG and daily-bias detectors on M15 to D1, and an Asia-sweep + IFVG strategy on EURUSD, GBPUSD, GER40 and UK100 through MetaTrader 5 |
| `smc-ifvg-asia-csv/` | Sep 2025 | The same strategy rebuilt to run from CSV files, so it can be tested without a broker connection |
| `dax-asia-sweep/` | Sep to Oct 2025 | Asia-sweep model on the DAX, comparing a fixed 2.3R target with a target at the opposite side of the Asia range |
| `session-models/` | Mar to Apr 2026 | Eight session models on FX, gold and US and German indices: Silver Bullet, AMD, Asia sweep, 9 AM breakout, sweep + MSS + FVG, HTF PD arrays, and IFVG with a filter search; with interactive HTML dashboards |
| `crt-4h-poi/` | Sep 2026 | Candle range theory on the 4-hour chart of the Nasdaq 100, gated by daily and weekly points of interest, with a robustness table |
| `asia_range_backtest_template.py` | Sep 2025 | A small template that computes daily Asia ranges from any intraday CSV |

## Results

Taken from each folder's own output files. "PF" is profit factor (gross profit over gross loss).

**`session-models/`** (1% of balance risked per trade, compounding, no spread or commission):

| Model | Market | Trades | Win rate | PF |
|---|---|---:|---:|---:|
| Silver Bullet | DAX | 1,210 | 42.5% | 1.44 |
| Silver Bullet | S&P 500 | 1,103 | 42.1% | 1.41 |
| Silver Bullet | Nasdaq 100 | 1,218 | 39.2% | 1.31 |
| Asia sweep | EURUSD | 145 | 36.6% | 1.45 |
| Asia sweep | DAX | 363 | 29.5% | 1.18 |
| Asia sweep | GBPUSD | 161 | 30.4% | 1.00 |
| IFVG, London kill zone | EURUSD | 131 | 38.9% | 1.26 |
| AMD | Nasdaq 100 | 414 | 36.5% | 1.14 |
| AMD | S&P 500 | 315 | 33.3% | 0.98 |
| AMD | DAX | 301 | 30.2% | 0.84 |
| HTF bias + PD array | EURUSD | 592 | 33.8% | 1.01 |
| Sweep + MSS + FVG | EURUSD | 332 | 30.7% | 0.88 |

**`asia-sweep-ifvg-mt5/`** (winners average 2.3R): 84 trades across four markets, 32.1% win rate, PF 1.08. EURUSD
(PF 2.00 on 15 trades) and GBPUSD (1.33 on 19) were positive; GER40 (0.86) and UK100 (0.70) were not.
**`smc-ifvg-asia-csv/`** on GER40 alone: 38 trades, 3 wins, PF 0.51.

**`crt-4h-poi/`** (1 point of round-trip cost per trade) is the most careful of these, because it reports the
standard error and t-statistic of the mean R for every variant:

| Point of interest, entry model | Trades | Win rate | Mean R | t | PF |
|---|---:|---:|---:|---:|---:|
| Previous week high/low, close-of-range entry | 26 | 34.6% | +0.51 | 0.68 | 1.77 |
| Previous week high/low, midpoint entry at 2R | 29 | 41.4% | +0.22 | 0.79 | 1.37 |
| Daily IFVG, close-of-range | 164 | 37.8% | +0.03 | 0.17 | 1.04 |
| Any point of interest, close-of-range | 479 | 38.8% | -0.01 | -0.12 | 0.99 |
| No filter, close-of-range | 856 | 37.0% | -0.06 | -1.10 | 0.91 |
| No filter, midpoint at 2R | 816 | 27.6% | -0.20 | -4.17 | 0.74 |

The best-looking variant made 1.77 times as much as it lost, but on 26 trades its t-statistic was 0.68,
which is well within what chance produces. With no point-of-interest filter the model loses money with a
t-statistic of -4.2 on the 2R entry, so the filters are selecting a handful of trades out of noise rather
than revealing an edge. I wrote it off.

## Caveats

- **No costs in most of these.** Everything in `session-models/`, `smc-ifvg-asia-csv/` and the early folders
  fills at the exact price with no spread, commission or slippage. On index CFDs and M5 entries that matters:
  the Silver Bullet results above would shrink, and could disappear, once a spread is charged.
- **No look-ahead audit.** These scripts were not put through the causal check that later sank the best
  SwiftPip model (its profit factor fell from 2.95 to 0.51 once swings were only used after they were
  confirmed). Several of these strategies also use swing points, so I treat every positive result here as
  unverified.
- **One sample each.** None of them has a held-out period chosen in advance; parameters were tuned and
  reported on the same data.
- **Data is not included.** The scripts read MetaTrader 5 directly or Dukascopy CSV exports; set the paths in
  each script's config block.

That combination is why this work led to the stricter testing in SwiftPip and later to my
[C++ backtesting engine](https://github.com/nynauk/cpp-backtesting-engine), where costs, causality and
independent cross-checks are built in rather than added afterwards.

## Run

```bash
python -m pip install pandas numpy matplotlib plotly MetaTrader5
python crt-4h-poi/crt_backtest.py            # edit CFG at the top: CSV path and output folder
python session-models/silver_bullet_backtest.py
```

Scripts that import `MetaTrader5` need the MetaTrader 5 terminal open and logged in (Windows). None of them
contain account details.
