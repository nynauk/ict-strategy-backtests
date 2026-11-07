"""Inject data into Asia Sweep dashboard template and save final HTML."""
import json, os

with open('asia_sweep_dashboard_data.json') as f:
    data = json.load(f)
json_str = json.dumps(data, separators=(',', ':'))

html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Asia Sweep Strategy -- Interactive Dashboard</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
:root{--bg:#0d1117;--surface:#161b22;--s2:#21262d;--border:#30363d;--text:#e6edf3;--muted:#8b949e;--gold:#FFD700;--green:#26a69a;--red:#ef5350;--blue:#58a6ff;}
*{box-sizing:border-box;margin:0;padding:0;}
body{background:var(--bg);color:var(--text);font-family:"Segoe UI",system-ui,sans-serif;}
nav{background:var(--surface);border-bottom:1px solid var(--border);padding:12px 24px;display:flex;align-items:center;gap:12px;flex-wrap:wrap;position:sticky;top:0;z-index:100;}
nav h1{color:var(--gold);font-size:1.05rem;letter-spacing:1px;}
.tb{background:none;border:1px solid var(--border);color:var(--muted);padding:6px 14px;border-radius:6px;cursor:pointer;font-size:.83rem;transition:all .2s;}
.tb:hover,.tb.active{background:var(--gold);color:#000;border-color:var(--gold);font-weight:700;}
.page{display:none;padding:20px 24px;max-width:1400px;margin:0 auto;}
.page.active{display:block;}
.sg{display:grid;grid-template-columns:repeat(auto-fill,minmax(155px,1fr));gap:10px;margin-bottom:16px;}
.sc{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:12px 14px;}
.sc .lb{font-size:.69rem;color:var(--muted);text-transform:uppercase;letter-spacing:.5px;margin-bottom:5px;}
.sc .vl{font-size:1.45rem;font-weight:700;}
.vg{color:var(--green);}.vr{color:var(--red);}.vo{color:var(--gold);}.vb{color:var(--blue);}
.cb{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:14px;margin-bottom:16px;}
.cb h3{color:var(--muted);font-size:.74rem;text-transform:uppercase;letter-spacing:1px;margin-bottom:10px;}
.pc{width:100%;height:340px;}.tall{height:440px;}
.c2{display:grid;grid-template-columns:1fr 1fr;gap:14px;}
@media(max-width:900px){.c2{grid-template-columns:1fr;}}
.gd{display:grid;grid-template-columns:1fr 1fr;gap:16px;}
@media(max-width:900px){.gd{grid-template-columns:1fr;}}
.gc{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:18px;}
.gc h2{color:var(--gold);font-size:.94rem;margin-bottom:12px;}
.gc ol,.gc ul{padding-left:18px;line-height:1.9;font-size:.87rem;color:var(--muted);}
.gc li{margin-bottom:3px;}
.gc li strong{color:var(--text);}
.tag{display:inline-block;background:var(--s2);border:1px solid var(--border);border-radius:4px;padding:2px 8px;font-size:.75rem;margin:2px;color:var(--gold);}
.mono{background:var(--s2);border-radius:6px;padding:12px;margin-top:10px;font-family:monospace;font-size:.74rem;color:var(--muted);line-height:1.65;white-space:pre;}
.fr{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:12px;align-items:center;}
.fr label{color:var(--muted);font-size:.82rem;}
.fr select,.fr input{background:var(--s2);border:1px solid var(--border);color:var(--text);padding:5px 10px;border-radius:6px;font-size:.82rem;}
.tw{overflow-x:auto;max-height:440px;overflow-y:auto;border-radius:8px;border:1px solid var(--border);}
table{width:100%;border-collapse:collapse;font-size:.79rem;}
thead{position:sticky;top:0;background:var(--s2);}
th{padding:9px 11px;color:var(--muted);font-weight:600;text-align:left;border-bottom:1px solid var(--border);}
td{padding:7px 11px;border-bottom:1px solid #1c2128;}
tr:hover td{background:rgba(255,255,255,.02);}
.wr td:first-child{border-left:3px solid var(--green);}
.lr td:first-child{border-left:3px solid var(--red);}
.il{color:var(--gold);font-weight:700;font-size:.87rem;padding:3px 0;grid-column:1/-1;}
::-webkit-scrollbar{width:5px;height:5px;}
::-webkit-scrollbar-track{background:var(--bg);}
::-webkit-scrollbar-thumb{background:var(--border);border-radius:3px;}
</style>
</head>
<body>
<nav>
  <h1>&#9889; ASIA SWEEP &mdash; ClearLQ</h1>
  <button class="tb active" onclick="showPage(event,'overview')">Overview</button>
  <button class="tb" onclick="showPage(event,'equity')">Equity</button>
  <button class="tb" onclick="showPage(event,'monthly')">Monthly</button>
  <button class="tb" onclick="showPage(event,'direction')">Direction</button>
  <button class="tb" onclick="showPage(event,'trades')">Trade Log</button>
  <button class="tb" onclick="showPage(event,'strategy')">Strategy Guide</button>
</nav>

<!-- OVERVIEW -->
<div id="page-overview" class="page active">
  <div id="ov-dax" class="sg"></div>
  <div id="ov-eur" class="sg"></div>
  <div id="ov-gbp" class="sg"></div>
  <div class="c2">
    <div class="cb"><h3>Win Rate %</h3><div id="ov-wr" class="pc"></div></div>
    <div class="cb"><h3>Profit Factor</h3><div id="ov-pf" class="pc"></div></div>
  </div>
  <div class="c2">
    <div class="cb"><h3>Return %</h3><div id="ov-ret" class="pc"></div></div>
    <div class="cb"><h3>Max Drawdown %</h3><div id="ov-dd" class="pc"></div></div>
  </div>
</div>

<!-- EQUITY -->
<div id="page-equity" class="page">
  <div class="fr">
    <label>Instrument:</label>
    <select id="eq-inst" onchange="renderEquity()">
      <option value="all">All overlaid</option>
      <option value="DEUIDX">DAX</option>
      <option value="EURUSD">EURUSD</option>
      <option value="GBPUSD">GBPUSD</option>
    </select>
  </div>
  <div class="cb"><h3>Equity Curve</h3><div id="eq-ch" class="pc tall"></div></div>
  <div class="cb"><h3>Drawdown %</h3><div id="dd-ch" class="pc"></div></div>
</div>

<!-- MONTHLY -->
<div id="page-monthly" class="page">
  <div class="fr">
    <label>Instrument:</label>
    <select id="mon-inst" onchange="renderMonthly()">
      <option value="DEUIDX">DAX</option>
      <option value="EURUSD">EURUSD</option>
      <option value="GBPUSD">GBPUSD</option>
    </select>
  </div>
  <div class="cb"><h3>Monthly Net P&amp;L ($)</h3><div id="mon-usd" class="pc tall"></div></div>
  <div class="cb"><h3>Monthly Win Rate %</h3><div id="mon-wr" class="pc"></div></div>
</div>

<!-- DIRECTION -->
<div id="page-direction" class="page">
  <div class="fr">
    <label>Instrument:</label>
    <select id="dir-inst" onchange="renderDirection()">
      <option value="DEUIDX">DAX</option>
      <option value="EURUSD">EURUSD</option>
      <option value="GBPUSD">GBPUSD</option>
    </select>
  </div>
  <div class="c2">
    <div class="cb"><h3>Trades by Sweep Side (W/L)</h3><div id="dir-cnt" class="pc"></div></div>
    <div class="cb"><h3>Net P&amp;L by Sweep Side</h3><div id="dir-pnl" class="pc"></div></div>
  </div>
  <div class="cb"><h3>RR Distribution (actual achieved)</h3><div id="dir-rr" class="pc"></div></div>
</div>

<!-- TRADES -->
<div id="page-trades" class="page">
  <div class="fr">
    <label>Instrument:</label>
    <select id="tr-inst" onchange="renderTrades()">
      <option value="DEUIDX">DAX</option>
      <option value="EURUSD">EURUSD</option>
      <option value="GBPUSD">GBPUSD</option>
    </select>
    <label>Sweep:</label>
    <select id="tr-side" onchange="renderTrades()">
      <option value="all">All</option>
      <option value="BUY">BUY sweep</option>
      <option value="SELL">SELL sweep</option>
    </select>
    <label>Result:</label>
    <select id="tr-res" onchange="renderTrades()">
      <option value="all">All</option>
      <option value="WIN">WIN</option>
      <option value="LOSS">LOSS</option>
    </select>
    <label>Search:</label>
    <input type="text" id="tr-q" placeholder="Date / price..." oninput="renderTrades()" style="width:140px;">
  </div>
  <div class="cb" style="margin-bottom:14px;"><h3>P&amp;L per Trade</h3><div id="tr-sc" class="pc tall"></div></div>
  <div class="cb">
    <h3>Trade Log <span id="tr-cnt" style="color:var(--muted);font-weight:400;font-size:.77rem;"></span></h3>
    <div class="tw">
      <table>
        <thead><tr>
          <th>Entry Time</th><th>Dir</th><th>Sweep</th>
          <th>Entry</th><th>SL</th><th>TP</th>
          <th>Result</th><th>Pts</th><th>SL pts</th><th>RR</th><th>P&amp;L $</th><th>Dur(m)</th>
          <th>Asia Hi</th><th>Asia Lo</th>
        </tr></thead>
        <tbody id="tr-body"></tbody>
      </table>
    </div>
  </div>
</div>

<!-- STRATEGY -->
<div id="page-strategy" class="page">
  <div class="gd">
    <div class="gc">
      <h2>What is the Asia Sweep?</h2>
      <p style="color:var(--muted);font-size:.87rem;line-height:1.7;">
        The Asia Sweep (by <strong style="color:var(--text);">ClearLQ</strong>) exploits stop-hunt manipulation
        of the Asian session range during the London Killzone. Smart money sweeps liquidity sitting above the
        Asia high or below the Asia low, then sharply reverses. You enter on that reversal using a
        Breaker Block / FVG entry model (the &ldquo;Unicorn&rdquo;).
      </p>
      <h2 style="margin-top:14px;">Sessions (UTC)</h2>
      <div>
        <span class="tag">Asia Range: 22:00 - 04:30 UTC</span><br>
        <span class="tag">London KZ: 06:00 - 09:00 UTC</span>
      </div>
    </div>
    <div class="gc">
      <h2>Step-by-Step Rules</h2>
      <ol>
        <li><strong>Mark Asia Range</strong> &mdash; High and Low from 22:00&ndash;04:30 UTC.</li>
        <li><strong>London KZ opens (06:00 UTC)</strong> &mdash; only act within this window.</li>
        <li><strong>Sweep confirmation</strong> &mdash; Candle HIGH > Asia High but CLOSES back below = bearish sweep (SHORT). Candle LOW &lt; Asia Low but CLOSES back above = bullish sweep (LONG).</li>
        <li><strong>Find entry (Unicorn/FVG)</strong> &mdash; After sweep, first M5 FVG in reversal direction. Retrace into FVG = entry.</li>
        <li><strong>Stop Loss</strong> &mdash; Beyond the sweep extreme + buffer.</li>
        <li><strong>Take Profit</strong> &mdash; Opposite Asia level. Low-hanging fruit along the way.</li>
        <li><strong>Invalidation</strong> &mdash; Price closes beyond the sweep extreme. One trade per day.</li>
      </ol>
    </div>
    <div class="gc">
      <h2>Bullish Setup (Long after sweep below Asia Low)</h2>
      <div class="mono"> Asia High ============================= (TP)
                               ^
 Asia Low  ======[====]======  |
                  |    |        |
              Sweep Low (wick)  |
              &lt; Asia Low         |
                  |             |
           Breaker Block + FVG  |
           Entry on retrace ----+</div>
    </div>
    <div class="gc">
      <h2>Bearish Setup (Short after sweep above Asia High)</h2>
      <div class="mono"> Asia High ======[====]====== (Sweep High &gt; Asia High)
                  |    |
              Sweep High (wick)
                  |
           Breaker Block + FVG
           Entry on retrace
                  |
                  v
 Asia Low  ===================== (TP)</div>
    </div>
    <div class="gc">
      <h2>Backtest Parameters</h2>
      <ul>
        <li><strong>Instruments:</strong> DAX, EURUSD, GBPUSD (M5 data)</li>
        <li><strong>Risk per trade:</strong> 1% of balance (compounding)</li>
        <li><strong>TP:</strong> Opposite Asia level (capped at 3:1 RR)</li>
        <li><strong>TP skipped if:</strong> Asia target gives &lt; 1:1 RR</li>
        <li><strong>Starting balance:</strong> $10,000</li>
        <li><strong>SL buffer:</strong> 10 pts (DAX), 5 pips (EURUSD/GBPUSD)</li>
        <li><strong>Entry:</strong> First M5 FVG in reversal direction after sweep</li>
        <li><strong>One trade per day maximum</strong></li>
      </ul>
    </div>
    <div class="gc">
      <h2>Key Results Summary</h2>
      <ul>
        <li><strong>DAX:</strong> 363 trades &bull; 29.5% WR &bull; PF 1.18 &bull; +48.5% return</li>
        <li><strong>EURUSD:</strong> 145 trades &bull; 36.6% WR &bull; PF 1.45 &bull; +42.8% return</li>
        <li><strong>GBPUSD:</strong> 161 trades &bull; 30.4% WR &bull; PF 1.00 &bull; flat (-0.01%)</li>
        <li><strong>Best window:</strong> EURUSD BUY sweeps (32W/48L but highest net $)</li>
        <li><strong>GBPUSD note:</strong> No detectable edge &mdash; needs additional confluence filters</li>
        <li><strong>Win rate is low</strong> because the TP is variable (Asia level) and often &gt;2R away &mdash; the edge comes from large wins vs small losses when it works</li>
      </ul>
    </div>
  </div>
</div>

<script>
const DATA = DATA_PLACEHOLDER;
const COLORS={DEUIDX:'#FFD700',EURUSD:'#26a69a',GBPUSD:'#58a6ff'};
const NAMES={DEUIDX:'DAX',EURUSD:'EURUSD',GBPUSD:'GBPUSD'};
const BASE={paper_bgcolor:'#161b22',plot_bgcolor:'#0d1117',font:{color:'#e6edf3',family:'Segoe UI,system-ui',size:12},margin:{t:20,b:50,l:60,r:20},xaxis:{gridcolor:'#21262d',linecolor:'#30363d',zerolinecolor:'#30363d'},yaxis:{gridcolor:'#21262d',linecolor:'#30363d',zerolinecolor:'#30363d'},legend:{bgcolor:'rgba(0,0,0,0)'},hovermode:'closest'};
const CFG={responsive:true};
function pl(id,tr,lay){Plotly.newPlot(id,tr,Object.assign({},BASE,lay),CFG);}
function card(lb,vl,cls){return '<div class="sc"><div class="lb">'+lb+'</div><div class="vl '+cls+'">'+vl+'</div></div>';}

function renderOverview(){
  [['DEUIDX','ov-dax'],['EURUSD','ov-eur'],['GBPUSD','ov-gbp']].forEach(([k,id])=>{
    const d=DATA[k],el=document.getElementById(id);
    const rc=d.ret_pct>0?'vg':'vr';
    el.innerHTML='<div class="il">'+d.name+'</div>'+
      card('Trades',d.total,'vb')+card('Win Rate',d.wr+'%',d.wr>=35?'vg':'vr')+
      card('Profit Factor',d.pf,d.pf>=1?'vg':'vr')+card('Return',d.ret_pct+'%',rc)+
      card('Max DD',d.max_dd+'%','vr')+card('Final Bal','$'+d.final_balance.toLocaleString('en-US',{maximumFractionDigits:0}),rc);
  });
  const ks=['DEUIDX','EURUSD','GBPUSD'],ns=ks.map(k=>NAMES[k]);
  pl('ov-wr',[{type:'bar',x:ns,y:ks.map(k=>DATA[k].wr),marker:{color:ks.map(k=>DATA[k].wr>=35?'#26a69a':'#ef5350')},text:ks.map(k=>DATA[k].wr+'%'),textposition:'outside'}],{yaxis:{title:'Win Rate %',range:[0,50]}});
  pl('ov-pf',[{type:'bar',x:ns,y:ks.map(k=>DATA[k].pf),marker:{color:ks.map(k=>DATA[k].pf>1?'#26a69a':'#ef5350')},text:ks.map(k=>DATA[k].pf.toFixed(2)),textposition:'outside'}],{yaxis:{title:'Profit Factor'},shapes:[{type:'line',x0:-0.5,x1:2.5,y0:1,y1:1,line:{color:'#ef5350',dash:'dash',width:1.5}}]});
  pl('ov-ret',[{type:'bar',x:ns,y:ks.map(k=>DATA[k].ret_pct),marker:{color:ks.map(k=>DATA[k].ret_pct>0?'#26a69a':'#ef5350')},text:ks.map(k=>(DATA[k].ret_pct>0?'+':'')+DATA[k].ret_pct+'%'),textposition:'outside'}],{yaxis:{title:'Return %'}});
  pl('ov-dd',[{type:'bar',x:ns,y:ks.map(k=>DATA[k].max_dd),marker:{color:'#ef5350'},text:ks.map(k=>DATA[k].max_dd+'%'),textposition:'outside'}],{yaxis:{title:'Max DD %'}});
}

function dd(eq){let pk=eq[0],r=[];for(const v of eq){if(v>pk)pk=v;r.push(((v-pk)/pk)*100);}return r;}

function renderEquity(){
  const inst=document.getElementById('eq-inst').value;
  const ks=inst==='all'?['DEUIDX','EURUSD','GBPUSD']:[inst];
  const tr=[],dr=[];
  ks.forEach(k=>{
    const eq=DATA[k].equity,x=Array.from({length:eq.length},(_,i)=>i);
    tr.push({x,y:eq,name:NAMES[k],line:{color:COLORS[k],width:1.8},mode:'lines'});
    dr.push({x,y:dd(eq),name:NAMES[k],line:{color:COLORS[k],width:1.3},mode:'lines',fill:'tozeroy',fillcolor:COLORS[k]+'33'});
  });
  pl('eq-ch',tr,{yaxis:{title:'Balance ($)'},xaxis:{title:'Trade #'},shapes:[{type:'line',x0:0,x1:1,xref:'paper',y0:10000,y1:10000,line:{color:'#444',dash:'dash',width:1}}]});
  pl('dd-ch',dr,{yaxis:{title:'Drawdown %'},xaxis:{title:'Trade #'}});
}

function renderMonthly(){
  const k=document.getElementById('mon-inst').value,d=DATA[k];
  const months=d.monthly.map(m=>m.month),usd=d.monthly.map(m=>m.net_usd);
  const wr=d.monthly.map(m=>{const t=m.wins+m.losses;return t>0?+(m.wins/t*100).toFixed(1):0;});
  pl('mon-usd',[{type:'bar',x:months,y:usd,marker:{color:usd.map(v=>v>0?'#26a69a':'#ef5350')},hovertemplate:'<b>%{x}</b><br>$%{y:,.2f}<extra></extra>'}],{yaxis:{title:'Net P&L ($)'},xaxis:{tickangle:-45}});
  pl('mon-wr',[{type:'bar',x:months,y:wr,marker:{color:wr.map(v=>v>=35?'#26a69a':'#ef5350')}}],{yaxis:{title:'Win Rate %',range:[0,100]},xaxis:{tickangle:-45},shapes:[{type:'line',x0:0,x1:1,xref:'paper',y0:33.3,y1:33.3,line:{color:'#ef5350',dash:'dash',width:1.5}}]});
}

function renderDirection(){
  const k=document.getElementById('dir-inst').value,d=DATA[k];
  const sides=d.by_side.map(s=>s.side);
  pl('dir-cnt',[{type:'bar',name:'Wins',x:sides,y:d.by_side.map(s=>s.wins),marker:{color:'#26a69a'}},{type:'bar',name:'Losses',x:sides,y:d.by_side.map(s=>s.losses),marker:{color:'#ef5350'}}],{barmode:'group',yaxis:{title:'Trades'}});
  pl('dir-pnl',[{type:'bar',x:sides,y:d.by_side.map(s=>s.net_usd),marker:{color:d.by_side.map(s=>s.net_usd>0?'#26a69a':'#ef5350')},text:d.by_side.map(s=>(s.net_usd>0?'+':'')+s.net_usd.toFixed(0)),textposition:'outside'}],{yaxis:{title:'Net P&L ($)'}});
  const wins=d.trades.filter(t=>t.result==='WIN'),loss=d.trades.filter(t=>t.result==='LOSS');
  pl('dir-rr',[{type:'histogram',name:'WIN',x:wins.map(t=>t.rr),marker:{color:'#26a69a',opacity:.7},nbinsx:25},{type:'histogram',name:'LOSS',x:loss.map(t=>t.rr),marker:{color:'#ef5350',opacity:.7},nbinsx:25}],{barmode:'overlay',xaxis:{title:'Achieved RR'},yaxis:{title:'Count'}});
}

function getFiltered(){
  const k=document.getElementById('tr-inst').value;
  const side=document.getElementById('tr-side').value;
  const res=document.getElementById('tr-res').value;
  const q=document.getElementById('tr-q').value.toLowerCase();
  return DATA[k].trades.filter(t=>{
    if(side!=='all'&&t.sweep_side!==side)return false;
    if(res!=='all'&&t.result!==res)return false;
    if(q&&!JSON.stringify(t).toLowerCase().includes(q))return false;
    return true;
  });
}

function fmt(v,dec){return typeof v==='number'?v.toFixed(dec):v;}

function renderTrades(){
  const ts=getFiltered();
  document.getElementById('tr-cnt').textContent='('+ts.length+' shown)';
  const wins=ts.filter(t=>t.result==='WIN'),loss=ts.filter(t=>t.result==='LOSS');
  const mk=(arr,name,col)=>({type:'scatter',mode:'markers',name,x:arr.map((_,i)=>i),y:arr.map(t=>t.pnl_usd),text:arr.map(t=>t.entry_dt+'<br>'+t.direction+' @ '+t.entry+'<br>Result: '+t.result+'<br>$'+t.pnl_usd.toFixed(2)+' | RR: '+t.rr+' | '+t.dur_m+'m'),hovertemplate:'%{text}<extra></extra>',marker:{color:col,size:7,opacity:.7}});
  pl('tr-sc',[mk(wins,'WIN','#26a69a'),mk(loss,'LOSS','#ef5350')],{yaxis:{title:'P&L ($)'},xaxis:{title:'Trade #'},shapes:[{type:'line',x0:0,x1:1,xref:'paper',y0:0,y1:0,line:{color:'#444',dash:'dash'}}]});
  const tbody=document.getElementById('tr-body');
  tbody.innerHTML=ts.slice(0,300).map(t=>{
    const gc=t.result==='WIN'?'#26a69a':'#ef5350';
    const dc=t.direction==='BUY'?'#26a69a':'#ef5350';
    const pc=t.pts>=0?'#26a69a':'#ef5350';
    const uc=t.pnl_usd>=0?'#26a69a':'#ef5350';
    return '<tr class="'+(t.result==='WIN'?'wr':'lr')+'">'+
      '<td>'+t.entry_dt+'</td>'+
      '<td style="color:'+dc+';font-weight:600">'+t.direction+'</td>'+
      '<td>'+t.sweep_side+'</td>'+
      '<td>'+t.entry+'</td><td>'+t.sl+'</td><td>'+t.tp+'</td>'+
      '<td style="color:'+gc+';font-weight:600">'+t.result+'</td>'+
      '<td style="color:'+pc+'">'+(t.pts>=0?'+':'')+fmt(t.pts,5)+'</td>'+
      '<td>'+fmt(t.sl_pts,5)+'</td>'+
      '<td style="color:'+(t.rr>=0?'#26a69a':'#ef5350')+'">'+t.rr+'</td>'+
      '<td style="color:'+uc+'">'+(t.pnl_usd>=0?'+':'')+'$'+t.pnl_usd.toFixed(2)+'</td>'+
      '<td>'+t.dur_m+'</td>'+
      '<td>'+t.asia_hi+'</td><td>'+t.asia_lo+'</td>'+
      '</tr>';
  }).join('');
}

function showPage(e,name){
  document.querySelectorAll('.page').forEach(p=>p.classList.remove('active'));
  document.querySelectorAll('.tb').forEach(b=>b.classList.remove('active'));
  document.getElementById('page-'+name).classList.add('active');
  e.target.classList.add('active');
  if(name==='overview')renderOverview();
  else if(name==='equity')renderEquity();
  else if(name==='monthly')renderMonthly();
  else if(name==='direction')renderDirection();
  else if(name==='trades')renderTrades();
}
renderOverview();
</script>
</body>
</html>"""

html = html.replace('DATA_PLACEHOLDER', json_str)
out = 'asia_sweep_dashboard.html'
with open(out, 'w', encoding='utf-8') as f:
    f.write(html)
print(f'Saved {out} ({os.path.getsize(out)//1024} KB)')
