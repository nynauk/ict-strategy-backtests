"""
ICT Silver Bullet — Interactive HTML Chart Generator
Reads the trade CSVs produced by silver_bullet_backtest.py
Outputs:  silver_bullet_chart.html
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json, os, sys

# ── Config ────────────────────────────────────────────────────────────────────
INSTRUMENTS = [
    {"name": "USATECH (NASDAQ)", "short": "USATECH",
     "path": r"C:\Users\Ayaan\Downloads\USATECHIDXUSD_M5.csv",
     "trades_csv": "sb_USATECH_trades.csv", "sl_buf": 15.0},
    {"name": "USA500 (S&P 500)", "short": "USA500",
     "path": r"C:\Users\Ayaan\Downloads\USA500IDXUSD_M5.csv",
     "trades_csv": "sb_USA500_trades.csv", "sl_buf": 5.0},
    {"name": "DEUIDX (DAX)", "short": "DEUIDX",
     "path": r"C:\Users\Ayaan\Downloads\DEUIDXEUR_M5 (3).csv",
     "trades_csv": "sb_DEUIDX_trades.csv", "sl_buf": 15.0},
]
WINDOW_CFG = [
    (3,  4,  "London", "100,149,237"),
    (10, 11, "NY AM",  "255,165,0"),
    (14, 15, "NY PM",  "147,112,219"),
]
DARK  = "#0d1117"
PANEL = "#161b22"
TEXT  = "#e6edf3"
MUTED = "#8b949e"
GREEN = "#26a69a"
RED   = "#ef5350"
GOLD  = "#ffd700"

# ── Helpers ───────────────────────────────────────────────────────────────────
def resample_m15(df):
    return (df.set_index("datetime").resample("15min")
              .agg(open=("open","first"), high=("high","max"),
                   low=("low","min"),    close=("close","last"))
              .dropna().reset_index())

def to_et(dt_series):
    return dt_series.dt.tz_localize("UTC").dt.tz_convert("America/New_York")

def calc_stats(t):
    wins   = (t["result"] == "WIN").sum()
    losses = (t["result"] == "LOSS").sum()
    total  = len(t)
    wr     = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0
    gp     = t[t["pnl_usd"] > 0]["pnl_usd"].sum()
    gl     = t[t["pnl_usd"] < 0]["pnl_usd"].abs().sum()
    pf     = gp / gl if gl > 0 else float("inf")
    net_pt = t["pts"].sum()
    w_pts  = t[t["result"] == "WIN"]["pts"]
    l_pts  = t[t["result"] == "LOSS"]["pts"]
    avg_w  = w_pts.mean() if len(w_pts) > 0 else 0
    avg_l  = l_pts.mean() if len(l_pts) > 0 else 0
    rr     = abs(avg_w / avg_l) if avg_l != 0 else 0
    net_u  = t["pnl_usd"].sum()
    # drawdown on cumulative pnl_usd
    eq = 10000 + t["pnl_usd"].cumsum()
    eq = pd.concat([pd.Series([10000.0]), eq]).reset_index(drop=True)
    roll_max = eq.cummax()
    dd = ((eq - roll_max) / roll_max * 100).min()
    ret = (eq.iloc[-1] / 10000 - 1) * 100
    return dict(total=total, wins=wins, losses=losses, wr=wr, pf=pf,
                net_pt=net_pt, avg_w=avg_w, avg_l=avg_l, rr=rr,
                net_u=net_u, dd=dd, ret=ret)

def window_shade_trace(m15, window_et_start, window_et_end, rgb, name):
    """Single scatter trace that fills all daily occurrences of a SB window."""
    et    = to_et(m15["datetime"])
    mask  = (et.dt.hour >= window_et_start) & (et.dt.hour < window_et_end)
    idx   = np.where(mask.values)[0]
    if len(idx) == 0:
        return None
    groups = np.split(idx, np.where(np.diff(idx) > 3)[0] + 1)
    y_lo   = m15["low"].min()
    y_hi   = m15["high"].max()
    xs, ys = [], []
    for grp in groups:
        if len(grp) == 0:
            continue
        t0 = m15["datetime"].iloc[grp[0]]
        t1 = m15["datetime"].iloc[grp[-1]] + pd.Timedelta(minutes=15)
        xs += [t0, t0, t1, t1, None]
        ys += [y_lo, y_hi, y_hi, y_lo, None]
    return go.Scatter(x=xs, y=ys, fill="toself",
                      fillcolor=f"rgba({rgb},0.12)", line=dict(width=0),
                      name=name, legendgroup=name, showlegend=True,
                      hoverinfo="skip", mode="lines")

def find_fvg(m5_df, trade):
    """Re-detect the exact FVG zone that triggered a trade."""
    ent_dt = pd.Timestamp(trade["entry_dt"])
    window_h = {"London": 3, "NY AM": 10, "NY_AM": 10, "NY PM": 14, "NY_PM": 14}.get(trade["window"], 10)
    ent_et   = ent_dt.tz_localize("UTC").tz_convert("America/New_York")
    ws_et    = pd.Timestamp(f"{ent_et.date()} {window_h:02d}:00:00", tz="America/New_York")
    ws_utc   = ws_et.tz_convert("UTC").tz_localize(None)
    sub      = m5_df[(m5_df["datetime"] >= ws_utc) &
                     (m5_df["datetime"] <= ent_dt)].reset_index(drop=True)
    hi, lo   = sub["high"].values, sub["low"].values
    for j in range(2, len(sub)):
        if trade["direction"] == "BUY" and lo[j] > hi[j-2]:
            return dict(bot=hi[j-2], top=lo[j],
                        t0=sub["datetime"].iloc[j-2],
                        t1=sub["datetime"].iloc[j] + pd.Timedelta(minutes=5))
        if trade["direction"] == "SELL" and hi[j] < lo[j-2]:
            return dict(bot=hi[j], top=lo[j-2],
                        t0=sub["datetime"].iloc[j-2],
                        t1=sub["datetime"].iloc[j] + pd.Timedelta(minutes=5))
    return None

# ── Build per-instrument chart data ──────────────────────────────────────────
print("Building charts...")
inst_data = []

for cfg in INSTRUMENTS:
    short = cfg["short"]
    if not os.path.exists(cfg["trades_csv"]):
        print(f"  WARNING: {cfg['trades_csv']} not found — run silver_bullet_backtest.py first")
        continue
    print(f"  Loading {short}...")

    df = pd.read_csv(cfg["path"], header=None,
                     names=["datetime","open","high","low","close","volume"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    m15 = resample_m15(df)
    trades = pd.read_csv(cfg["trades_csv"])
    trades["entry_dt"] = pd.to_datetime(trades["entry_dt"])
    st = calc_stats(trades)

    # Monthly P&L
    trades["month"] = trades["entry_dt"].dt.to_period("M")
    monthly = trades.groupby("month")["pts"].sum()
    monthly_wr = trades.groupby("month").apply(
        lambda g: (g["result"]=="WIN").sum() / max(len(g),1)*100).rename("wr")

    # Find a good example trade for the detail panel
    # BUY WIN, NY_AM, mid-sized SL — pick first clean one
    cands = trades[(trades["result"]=="WIN") &
                   (trades["window"].isin(["NY_AM","NY AM"])) &
                   (trades["direction"]=="BUY") &
                   (trades["sl_pts"] > 30) & (trades["sl_pts"] < 200)].head(10)
    example_trade = None
    example_fvg   = None
    for _, row in cands.iterrows():
        fvg = find_fvg(df, row)
        if fvg:
            example_trade = row
            example_fvg   = fvg
            break

    inst_data.append(dict(
        cfg=cfg, short=short, m15=m15, df_m5=df,
        trades=trades, stats=st,
        monthly=monthly, monthly_wr=monthly_wr,
        example_trade=example_trade, example_fvg=example_fvg,
    ))

if not inst_data:
    sys.exit("No trade CSVs found. Run silver_bullet_backtest.py first.")

# ── Build Plotly figures ──────────────────────────────────────────────────────
def build_main_fig(d):
    m15  = d["m15"]
    t    = d["trades"]
    mon  = d["monthly"]
    mwr  = d["monthly_wr"]
    name = d["cfg"]["name"]

    fig = make_subplots(
        rows=2, cols=1, row_heights=[0.72, 0.28], shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=["Price (M15) + Trades", "Monthly P&L (points)"],
    )

    # Window shading (add before candles so it's behind)
    for wh0, wh1, wname, rgb in WINDOW_CFG:
        tr = window_shade_trace(m15, wh0, wh1, rgb, wname)
        if tr:
            fig.add_trace(tr, row=1, col=1)

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=m15["datetime"], open=m15["open"], high=m15["high"],
        low=m15["low"],    close=m15["close"],
        increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
        name="Price", showlegend=False,
        increasing_fillcolor="#26a69a", decreasing_fillcolor="#ef5350",
    ), row=1, col=1)

    # Trade markers
    for direction, result in [("BUY","WIN"),("BUY","LOSS"),("SELL","WIN"),("SELL","LOSS")]:
        sub = t[(t["direction"]==direction) & (t["result"]==result)]
        sym = "triangle-up" if direction=="BUY" else "triangle-down"
        col = GREEN if result=="WIN" else RED
        fig.add_trace(go.Scatter(
            x=sub["entry_dt"], y=sub["entry"],
            mode="markers",
            marker=dict(symbol=sym, size=9, color=col,
                        line=dict(width=1, color="white")),
            name=f"{direction} {result}",
            customdata=np.column_stack([
                sub["window"].values, sub["pts"].round(1).values,
                sub["sl_pts"].round(1).values, sub["sl"].values, sub["tp"].values,
            ]),
            hovertemplate=(
                f"<b>{direction} {result}</b><br>"
                "Window: %{customdata[0]}<br>"
                "Entry: %{y:.2f}<br>"
                "P&L: %{customdata[1]:+.1f} pts<br>"
                "Risk: %{customdata[2]:.1f} pts<br>"
                "SL: %{customdata[3]:.2f}  TP: %{customdata[4]:.2f}<br>"
                "<extra></extra>"
            ),
        ), row=1, col=1)

    # Monthly P&L bars
    fig.add_trace(go.Bar(
        x=mon.index.to_timestamp().tolist(),
        y=mon.values,
        marker_color=[GREEN if v >= 0 else RED for v in mon.values],
        name="Monthly pts",
        customdata=np.column_stack([mwr.reindex(mon.index).fillna(0).round(1).values]),
        hovertemplate="<b>%{x|%b %Y}</b><br>P&L: %{y:+.1f} pts<br>WR: %{customdata[0]:.0f}%<extra></extra>",
    ), row=2, col=1)

    st = d["stats"]
    fig.update_layout(
        title=dict(text=f"<b>ICT Silver Bullet — {name}</b>  |  "
                        f"WR {st['wr']:.1f}%  |  PF {st['pf']:.2f}  |  "
                        f"Avg RR {st['rr']:.2f}:1  |  Return {st['ret']:+.1f}%",
                   font=dict(color=TEXT, size=14)),
        paper_bgcolor=DARK, plot_bgcolor=PANEL,
        font=dict(color=MUTED, size=11),
        legend=dict(bgcolor=PANEL, bordercolor="#30363d", borderwidth=1,
                    font=dict(color=TEXT, size=10)),
        xaxis=dict(gridcolor="#21262d", rangeslider=dict(visible=False),
                   rangeselector=dict(
                       buttons=[
                           dict(count=1, label="1M", step="month", stepmode="backward"),
                           dict(count=3, label="3M", step="month", stepmode="backward"),
                           dict(count=6, label="6M", step="month", stepmode="backward"),
                           dict(count=1, label="YTD", step="year", stepmode="todate"),
                           dict(step="all", label="ALL"),
                       ],
                       bgcolor=PANEL, activecolor="#30363d",
                       font=dict(color=TEXT),
                   )),
        xaxis2=dict(gridcolor="#21262d"),
        yaxis=dict(gridcolor="#21262d", title="Price"),
        yaxis2=dict(gridcolor="#21262d", title="Points"),
        hovermode="x unified",
        margin=dict(l=60, r=20, t=70, b=40),
        height=680,
    )
    # Default view: last 3 months
    end   = m15["datetime"].iloc[-1]
    start = end - pd.DateOffset(months=3)
    fig.update_xaxes(range=[start, end])
    return fig

def build_example_fig(d):
    """M5 chart zoomed into one complete Silver Bullet trade."""
    ex  = d["example_trade"]
    fvg = d["example_fvg"]
    if ex is None or fvg is None:
        return None

    ent_dt = pd.Timestamp(ex["entry_dt"])
    m5     = d["df_m5"]
    t0     = ent_dt - pd.Timedelta(hours=2, minutes=30)
    t1     = ent_dt + pd.Timedelta(hours=3)
    sub    = m5[(m5["datetime"] >= t0) & (m5["datetime"] <= t1)].copy()
    if len(sub) < 10:
        return None

    # Determine window start for shading
    window_h = {"London": 3, "NY AM": 10, "NY_AM": 10, "NY PM": 14, "NY_PM": 14}.get(ex["window"], 10)
    ent_et   = ent_dt.tz_localize("UTC").tz_convert("America/New_York")
    ws_et    = pd.Timestamp(f"{ent_et.date()} {window_h:02d}:00:00", tz="America/New_York")
    ws_utc   = ws_et.tz_convert("UTC").tz_localize(None)
    we_utc   = ws_et.replace(hour=window_h+1).tz_convert("UTC").tz_localize(None)

    direction = ex["direction"]
    entry_p   = ex["entry"]
    sl_p      = ex["sl"]
    tp_p      = ex["tp"]
    result    = ex["result"]

    fig = go.Figure()

    # Window shading
    y_lo = sub["low"].min() - abs(entry_p - sl_p) * 0.5
    y_hi = sub["high"].max() + abs(entry_p - sl_p) * 0.5
    ws_end = min(we_utc, sub["datetime"].iloc[-1])
    for wh0, wh1, wname, rgb in WINDOW_CFG:
        et_h = to_et(sub["datetime"])
        in_w = (et_h.dt.hour >= wh0) & (et_h.dt.hour < wh1)
        idx  = np.where(in_w.values)[0]
        if len(idx) == 0:
            continue
        t_s = sub["datetime"].iloc[idx[0]]
        t_e = sub["datetime"].iloc[idx[-1]] + pd.Timedelta(minutes=5)
        fig.add_shape(type="rect", x0=t_s, x1=t_e, y0=y_lo, y1=y_hi,
                      fillcolor=f"rgba({rgb},0.13)", line_width=0, layer="below")
        fig.add_annotation(x=t_s, y=y_hi, text=f"<b>{wname} window</b>",
                           showarrow=False, yanchor="bottom",
                           font=dict(color=f"rgb({rgb})", size=10))

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=sub["datetime"], open=sub["open"], high=sub["high"],
        low=sub["low"],    close=sub["close"],
        increasing_line_color=GREEN, decreasing_line_color=RED,
        increasing_fillcolor=GREEN, decreasing_fillcolor=RED,
        name="M5 Price", showlegend=False,
    ))

    # FVG zone
    fvg_col = "rgba(38,166,154,0.25)" if direction == "BUY" else "rgba(239,83,80,0.25)"
    fvg_brd = GREEN if direction == "BUY" else RED
    fig.add_shape(type="rect",
                  x0=fvg["t0"], x1=max(fvg["t1"], ent_dt + pd.Timedelta(hours=2)),
                  y0=fvg["bot"], y1=fvg["top"],
                  fillcolor=fvg_col, line=dict(color=fvg_brd, width=1.5, dash="dot"))
    mid_fvg = fvg["t0"] + (fvg["t1"] - fvg["t0"]) / 2
    fig.add_annotation(x=mid_fvg, y=(fvg["bot"]+fvg["top"])/2,
                       text=f"<b>FVG Zone</b><br>{fvg['bot']:.1f} - {fvg['top']:.1f}",
                       showarrow=False, font=dict(color=fvg_brd, size=10),
                       bgcolor=DARK, bordercolor=fvg_brd, borderwidth=1)

    # Entry, SL, TP horizontal lines extending to the right
    end_x = sub["datetime"].iloc[-1]
    for level, label, col in [
        (entry_p, f"Entry  {entry_p:.2f}", GOLD),
        (sl_p,    f"SL  {sl_p:.2f}",      RED),
        (tp_p,    f"TP  {tp_p:.2f}",      GREEN),
    ]:
        fig.add_shape(type="line", x0=ent_dt, x1=end_x, y0=level, y1=level,
                      line=dict(color=col, width=1.5,
                                dash="solid" if level==entry_p else "dash"))
        fig.add_annotation(x=end_x, y=level, text=f"<b>{label}</b>",
                           xanchor="left", showarrow=False,
                           font=dict(color=col, size=10))

    # Entry marker
    sym = "triangle-up" if direction == "BUY" else "triangle-down"
    fig.add_trace(go.Scatter(
        x=[ent_dt], y=[entry_p], mode="markers",
        marker=dict(symbol=sym, size=14, color=GOLD,
                    line=dict(width=1.5, color="white")),
        name=f"Entry ({direction})",
    ))

    # Result annotation
    res_col = GREEN if result == "WIN" else RED
    fig.add_annotation(
        x=ent_dt, y=tp_p if result == "WIN" else sl_p,
        text=f"<b>{'WIN' if result=='WIN' else 'LOSS'}</b>",
        showarrow=True, arrowhead=2, arrowcolor=res_col,
        font=dict(color=res_col, size=13), bgcolor=DARK,
        bordercolor=res_col, borderwidth=1,
    )

    dt_str = pd.Timestamp(ex["entry_dt"]).strftime("%d %b %Y")
    fig.update_layout(
        title=dict(
            text=f"<b>Example Trade — {d['cfg']['name']}  |  {ex['window']} window  |  "
                 f"{direction} {result}  |  {dt_str}  |  "
                 f"SL: {ex['sl_pts']:.0f} pts  TP: {ex['pts']:.0f} pts</b>",
            font=dict(color=TEXT, size=13)),
        paper_bgcolor=DARK, plot_bgcolor=PANEL,
        font=dict(color=MUTED, size=11),
        legend=dict(bgcolor=PANEL, bordercolor="#30363d", borderwidth=1),
        xaxis=dict(gridcolor="#21262d", rangeslider=dict(visible=False)),
        yaxis=dict(gridcolor="#21262d"),
        hovermode="x unified",
        margin=dict(l=60, r=120, t=60, b=40),
        height=480,
    )
    return fig

# ── Render figures to JSON ────────────────────────────────────────────────────
charts = []
for d in inst_data:
    print(f"  Rendering {d['short']}...")
    main_fig = build_main_fig(d)
    ex_fig   = build_example_fig(d)
    charts.append(dict(
        short=d["short"], name=d["cfg"]["name"],
        stats=d["stats"],
        main_json=main_fig.to_json(),
        ex_json=ex_fig.to_json() if ex_fig else None,
        trades=d["trades"],
    ))

# ── HTML template ─────────────────────────────────────────────────────────────
STRATEGY_STEPS = [
    ("1", "Wait for a Silver Bullet Window",
     "Three 1-hour windows open each trading day when institutional algorithms are most active. "
     "These are: <b>London Open 3-4 AM</b>, <b>NY Morning 10-11 AM</b>, and <b>NY Afternoon 2-3 PM</b> (all New York time). "
     "Outside these windows, no trades are taken — the algorithm is blind."),

    ("2", "Read the 4H Bias",
     "Before the window opens, check the 4-hour chart to determine which direction the market wants to move. "
     "Compare the most recent completed 4H close to the close two bars ago (8 hours back). "
     "If price is <b style='color:#26a69a'>higher</b> → bullish bias (look for buys only). "
     "If price is <b style='color:#ef5350'>lower</b> → bearish bias (look for sells only). "
     "This keeps you on the right side of institutional order flow."),

    ("3", "Identify the First Fair Value Gap (FVG)",
     "A <b>Fair Value Gap</b> is a 3-candle price imbalance — a gap between candle 1's extreme and candle 3's extreme. "
     "<br><b style='color:#26a69a'>Bullish FVG</b>: candle[3].low &gt; candle[1].high — price gapped upward leaving an unfilled zone below. "
     "This zone acts as support when price retraces. "
     "<br><b style='color:#ef5350'>Bearish FVG</b>: candle[3].high &lt; candle[1].low — price gapped downward. "
     "This zone acts as resistance on a rally. "
     "ICT calls this an <i>imbalance</i> that the market is magnetically drawn back to fill."),

    ("4", "Enter on the Retrace into the FVG",
     "Once the FVG forms in the direction of your 4H bias, wait for price to retrace INTO the FVG zone. "
     "<br>For a <b>buy</b>: price dips back down to the top edge of the bullish FVG (lo of candle 3) → enter long at close. "
     "<br>For a <b>sell</b>: price rallies back up to the bottom edge of the bearish FVG (hi of candle 3) → enter short at close. "
     "The entry must occur within the Silver Bullet window or within 2 hours after it closes."),

    ("5", "Stop Loss and Take Profit",
     "<b>Stop Loss</b>: placed below the low of the <i>first candle</i> of the FVG pattern (bullish) "
     "or above the high of the first candle (bearish). This puts the SL behind the structure that created the FVG. "
     "<br><b>Take Profit</b>: set at <b>2:1 risk-to-reward</b> — if you risk 50 points, target 100 points. "
     "Alternatively, target the next liquidity level (PDH/PDL, session high/low). "
     "<br><b>Framework filter</b>: skip any setup where the TP is less than 10 points from entry (indices)."),
]

def stat_box(label, value, sub="", color=TEXT):
    return f"""<div class="stat-box">
      <div class="stat-label">{label}</div>
      <div class="stat-value" style="color:{color}">{value}</div>
      {'<div class="stat-sub">'+sub+'</div>' if sub else ''}
    </div>"""

def render_stats(s):
    boxes = [
        stat_box("Total Trades", s["total"]),
        stat_box("Win Rate", f"{s['wr']:.1f}%",
                 f"{s['wins']}W / {s['losses']}L",
                 GREEN if s["wr"] >= 40 else RED),
        stat_box("Profit Factor", f"{s['pf']:.2f}", "",
                 GREEN if s["pf"] > 1 else RED),
        stat_box("Avg Realized RR", f"{s['rr']:.2f}:1",
                 f"+{s['avg_w']:.1f} / {s['avg_l']:.1f} pts",
                 GREEN if s["rr"] >= 1.8 else GOLD),
        stat_box("Net Points", f"{s['net_pt']:+,.0f}"),
        stat_box("Max Drawdown", f"{s['dd']:.1f}%", "", RED),
        stat_box("Total Return", f"{s['ret']:+.0f}%", "$10k start",
                 GREEN if s["ret"] > 0 else RED),
    ]
    return '<div class="stats-row">' + "".join(boxes) + '</div>'

# Build HTML
print("Generating HTML...")

tabs_nav = ""
tabs_content = ""
for i, c in enumerate(charts):
    active = "active" if i == 0 else ""
    tabs_nav += f'<button class="tab-btn {active}" onclick="showTab(\'{c["short"]}\')">{c["name"]}</button>\n'

    main_id = f"chart_{c['short']}_main"
    ex_id   = f"chart_{c['short']}_ex"

    ex_section = ""
    if c["ex_json"]:
        ex_section = f"""
      <h3 style="color:{MUTED};margin:24px 0 8px;font-size:13px;letter-spacing:.08em;text-transform:uppercase">
        Example Trade (M5 Detail)</h3>
      <div id="{ex_id}" style="border:1px solid #30363d;border-radius:8px;overflow:hidden"></div>
      <script>
        (function(){{
          var ex = {c['ex_json']};
          ex.layout.paper_bgcolor='{DARK}'; ex.layout.plot_bgcolor='{PANEL}';
          Plotly.newPlot('{ex_id}', ex.data, ex.layout, {{responsive:true,displayModeBar:false}});
        }})();
      </script>"""

    tabs_content += f"""
  <div id="tab_{c['short']}" class="tab-pane {('active' if i==0 else '')}">
    {render_stats(c["stats"])}
    <div id="{main_id}" style="border:1px solid #30363d;border-radius:8px;overflow:hidden;margin-top:16px"></div>
    <script>
      (function(){{
        var fig = {c['main_json']};
        fig.layout.paper_bgcolor='{DARK}'; fig.layout.plot_bgcolor='{PANEL}';
        Plotly.newPlot('{main_id}', fig.data, fig.layout, {{responsive:true,displayModeBar:true}});
      }})();
    </script>
    {ex_section}
  </div>
"""

steps_html = ""
for num, title, body in STRATEGY_STEPS:
    steps_html += f"""
  <div class="step">
    <div class="step-num">{num}</div>
    <div class="step-body">
      <div class="step-title">{title}</div>
      <div class="step-text">{body}</div>
    </div>
  </div>"""

HTML = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ICT Silver Bullet — Interactive Backtest</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{background:{DARK};color:{TEXT};font-family:'Segoe UI',system-ui,sans-serif;font-size:14px;line-height:1.6}}
  a{{color:{GOLD};text-decoration:none}}

  /* Header */
  .header{{background:linear-gradient(135deg,#161b22 0%,#0d1117 100%);
           border-bottom:1px solid #30363d;padding:24px 32px}}
  .header h1{{font-size:22px;font-weight:700;color:{TEXT};letter-spacing:-.3px}}
  .header h1 span{{color:{GOLD}}}
  .header p{{color:{MUTED};font-size:13px;margin-top:4px}}
  .badge{{display:inline-block;background:#21262d;border:1px solid #30363d;
          border-radius:20px;padding:3px 10px;font-size:11px;margin:2px;color:{TEXT}}}

  /* Layout */
  .container{{max-width:1400px;margin:0 auto;padding:24px 32px}}
  .section-title{{font-size:11px;font-weight:700;color:{MUTED};letter-spacing:.12em;
                  text-transform:uppercase;margin:28px 0 12px;
                  border-left:3px solid {GOLD};padding-left:10px}}

  /* Strategy steps */
  .steps{{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:8px}}
  @media(max-width:800px){{.steps{{grid-template-columns:1fr}}}}
  .step{{display:flex;gap:14px;background:{PANEL};border:1px solid #30363d;
         border-radius:10px;padding:16px;align-items:flex-start}}
  .step-num{{background:{GOLD};color:#000;font-weight:800;font-size:13px;
             border-radius:50%;min-width:26px;height:26px;display:flex;
             align-items:center;justify-content:center;flex-shrink:0}}
  .step-title{{font-weight:600;color:{TEXT};font-size:13px;margin-bottom:4px}}
  .step-text{{color:{MUTED};font-size:12px;line-height:1.55}}

  /* FVG explainer */
  .fvg-box{{background:{PANEL};border:1px solid #30363d;border-radius:10px;
            padding:16px 20px;margin-top:4px}}
  .fvg-box h4{{color:{GOLD};font-size:13px;margin-bottom:8px}}

  /* Tabs */
  .tab-nav{{display:flex;gap:0;border-bottom:1px solid #30363d;margin-bottom:20px;overflow-x:auto}}
  .tab-btn{{background:none;border:none;color:{MUTED};padding:10px 20px;font-size:13px;
            cursor:pointer;font-family:inherit;border-bottom:2px solid transparent;
            transition:all .15s;white-space:nowrap}}
  .tab-btn:hover{{color:{TEXT};background:#21262d}}
  .tab-btn.active{{color:{GOLD};border-bottom-color:{GOLD};font-weight:600}}
  .tab-pane{{display:none}}.tab-pane.active{{display:block}}

  /* Stats */
  .stats-row{{display:flex;gap:10px;flex-wrap:wrap}}
  .stat-box{{background:{PANEL};border:1px solid #30363d;border-radius:8px;
             padding:12px 16px;min-width:110px;flex:1}}
  .stat-label{{font-size:10px;color:{MUTED};text-transform:uppercase;letter-spacing:.08em;margin-bottom:4px}}
  .stat-value{{font-size:20px;font-weight:700;color:{TEXT}}}
  .stat-sub{{font-size:10px;color:{MUTED};margin-top:2px}}

  /* Legend */
  .legend{{display:flex;gap:16px;flex-wrap:wrap;margin:12px 0 4px;font-size:11px;color:{MUTED}}}
  .legend span{{display:flex;align-items:center;gap:5px}}
  .dot{{width:10px;height:10px;border-radius:50%;display:inline-block}}

  /* Window legend */
  .win-legend{{display:flex;gap:12px;margin:8px 0;font-size:11px}}
  .win-swatch{{width:14px;height:14px;border-radius:3px;display:inline-block;margin-right:4px;vertical-align:middle}}
</style>
</head>
<body>

<div class="header">
  <h1><span>ICT Silver Bullet</span> — Interactive Backtest</h1>
  <p>M5 entries &nbsp;|&nbsp; 4H bias filter &nbsp;|&nbsp; 2:1 RR &nbsp;|&nbsp;
     Data: Apr 2023 – Apr 2026 &nbsp;|&nbsp; 3 instruments</p>
  <div style="margin-top:10px">
    <span class="badge">London 03:00-04:00 ET</span>
    <span class="badge">NY AM 10:00-11:00 ET</span>
    <span class="badge">NY PM 14:00-15:00 ET</span>
    <span class="badge">First FVG in bias direction</span>
    <span class="badge">SL below FVG structure</span>
    <span class="badge">TP 2:1 RR</span>
  </div>
</div>

<div class="container">

  <!-- ── Strategy Walkthrough ───────────────────────────────────────── -->
  <div class="section-title">How the Strategy Works — Step by Step</div>

  <div class="fvg-box" style="margin-bottom:12px">
    <h4>What is a Fair Value Gap (FVG)?</h4>
    <p style="color:{MUTED};font-size:12px">
      An FVG is a 3-candle price imbalance. Institutions move the market so fast
      that they leave a <i>gap</i> — a price range that was never properly traded.
      The market has a magnetic tendency to return to these gaps, making them ideal entry zones.<br><br>
      <b style="color:{GREEN}">Bullish FVG</b> &nbsp;(buy setup):&nbsp;
      Candle 3's LOW &gt; Candle 1's HIGH &rarr; upward gap &rarr; buy when price dips back into it.<br>
      <b style="color:{RED}">Bearish FVG</b> &nbsp;(sell setup):&nbsp;
      Candle 3's HIGH &lt; Candle 1's LOW &rarr; downward gap &rarr; sell when price rallies into it.
    </p>
  </div>

  <div class="steps">
    {steps_html}
  </div>

  <!-- ── Charts ────────────────────────────────────────────────────── -->
  <div class="section-title" style="margin-top:32px">Backtest Results by Instrument</div>

  <div class="win-legend">
    {"".join(f'<span><span class="win-swatch" style="background:rgba({r},0.35);border:1px solid rgba({r},0.7)"></span>{n} window</span>'
             for _,_,n,r in WINDOW_CFG)}
    <span><span class="win-swatch" style="background:{GREEN}55;border:1px solid {GREEN}"></span>WIN trade</span>
    <span><span class="win-swatch" style="background:{RED}55;border:1px solid {RED}"></span>LOSS trade</span>
    <span style="color:{MUTED}">&#9650; = Buy &nbsp; &#9660; = Sell</span>
  </div>

  <div class="tab-nav">
    {tabs_nav}
  </div>

  {tabs_content}

</div><!-- /container -->

<script>
function showTab(id) {{
  document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.getElementById('tab_' + id).classList.add('active');
  document.querySelectorAll('.tab-btn').forEach(b => {{
    if (b.textContent.trim().includes(
      {json.dumps({c['short']: c['name'] for c in charts})}[id].split('(')[0].trim()
    )) b.classList.add('active');
  }});
  // Trigger resize so Plotly redraws correctly
  window.dispatchEvent(new Event('resize'));
}}
</script>
</body>
</html>
"""

out = "silver_bullet_chart.html"
with open(out, "w", encoding="utf-8") as f:
    f.write(HTML)

size_kb = os.path.getsize(out) / 1024
print(f"\nSaved: {out}  ({size_kb:.0f} KB)")
print("Open silver_bullet_chart.html in your browser.")
