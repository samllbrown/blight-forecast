"""Live dashboard for model runs. Stdlib only.
Usage: python3 scripts/dashboard.py [port]   then open http://localhost:8765
Reads results/progress.jsonl (per-season events from blightcast.evaluate), falls back to parsing
results/run_*.log for runs started before the event stream existed, and shows finished tables.
"""
import glob, json, os, re, sys, time, csv, subprocess
from http.server import HTTPServer, BaseHTTPRequestHandler
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, 'results')
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8765

def newest_log():
    logs = sorted(glob.glob(os.path.join(RES, 'run_*.log')), key=os.path.getmtime)
    return logs[-1] if logs else None

def parse_log(path):
    """Per-set AUC lines and outcome headers from a run log."""
    fits, outcomes, errors = [], [], []
    for line in open(path, errors='replace'):
        if line.startswith('panel ') or line.startswith('Part A'):   # a new run started: earlier errors are history
            errors = []
        m = re.match(r'\s+(\S+) (gbm|logistic): (\S+) auc ([\d.]+)', line)
        if m: fits.append(dict(outcome=m[1], model=m[2], set=m[3], auc=float(m[4])))
        m = re.match(r'Part B \[(\S+)\]: (\d+) test district-days, (\d+) positives, Hutton alert on ([\d.]+) of days with recall ([\d.]+)', line)
        if m: outcomes.append(dict(outcome=m[1], n=int(m[2]), positives=int(m[3]), hutton_rate=float(m[4]), hutton_recall=float(m[5])))
        if 'Killed' in line or 'Traceback' in line or 'MemoryError' in line: errors.append(line.strip())
    return fits, outcomes, errors

def events():
    p = os.path.join(RES, 'progress.jsonl')
    if not os.path.exists(p): return []
    out = []
    for line in open(p):
        try: out.append(json.loads(line))
        except Exception: pass
    return out

def proc_info():
    try:
        out = subprocess.run(['ps', '-eo', 'pid,rss,pcpu,etime,args'], capture_output=True, text=True).stdout.splitlines()
    except Exception: return None
    for l in out:
        parts = l.split(None, 4)
        if len(parts) == 5 and 'blightcast.evaluate' in parts[4] and parts[4].split()[0].endswith('python') and 'dashboard' not in l:
            return dict(pid=int(parts[0]), rss_gb=int(parts[1]) / 1e6, cpu=float(parts[2]), elapsed=parts[3], args=parts[4][-80:])
    return None

def memory():
    try:
        m = {l.split(':')[0]: int(l.split()[1]) for l in open('/proc/meminfo') if ':' in l}
        return dict(total_gb=m['MemTotal'] / 1e6, available_gb=m['MemAvailable'] / 1e6)
    except Exception: return None

def tables():
    out = {}
    for p in sorted(glob.glob(os.path.join(RES, 'tables', 'part_b.*.csv'))):
        name = os.path.basename(p)[7:-4]
        rows = list(csv.DictReader(open(p)))
        out[name] = [{k: (round(float(v), 3) if k in ('auc', 'recall_at_hutton_rate', 'rate_for_hutton_recall', 'within_auc') and v else v)
                      for k, v in r.items() if k != 'per_season'} for r in rows]
    return out

def status():
    log = newest_log()
    fits, outcomes, errors = parse_log(log) if log else ([], [], [])
    tail = open(log, errors='replace').read().splitlines()[-12:] if log else []
    ev = events()
    return dict(now=time.time(), log=os.path.basename(log) if log else None, log_mtime=os.path.getmtime(log) if log else None,
                fits=fits, outcomes=outcomes, errors=errors, tail=[t[:160] for t in tail], events=ev[-400:],
                proc=proc_info(), memory=memory(), tables=tables())

HTML = r'''<!doctype html><html><head><meta charset="utf-8"><title>blight runs</title>
<style>
body{font:14px/1.4 system-ui,sans-serif;background:#111;color:#ddd;margin:0;padding:18px 24px}
main{max-width:1100px;margin:0 auto}
h1{font-size:18px;margin:0 0 6px} h2{font-size:15px;margin:22px 0 6px;color:#c9a0ff}
.row{display:flex;gap:24px;flex-wrap:wrap} .card{background:#1b1b1f;border:1px solid #2a2a30;border-radius:8px;padding:10px 14px;min-width:220px}
.k{color:#888;font-size:12px} .v{font-size:20px} .ok{color:#6fd39a} .bad{color:#ff7b7b} .warn{color:#f0b35a}
table{border-collapse:collapse;font-size:13px} td,th{padding:3px 10px;border-bottom:1px solid #2a2a30;text-align:left} th{color:#aaa;font-weight:500}
.bar{display:inline-block;height:9px;background:#a93fe0;border-radius:2px;vertical-align:middle;margin-right:6px}
.bar2{background:#c97c12} .bar3{background:#35a066}
pre{background:#0b0b0d;color:#9a9;padding:8px;border-radius:6px;font-size:11px;max-height:220px;overflow:auto}
.prog{height:8px;background:#2a2a30;border-radius:4px;overflow:hidden;width:260px;display:inline-block;vertical-align:middle}
.prog i{display:block;height:100%;background:#35a066}
.season{display:inline-block;width:22px;text-align:center;font-size:11px;color:#bbb}
@keyframes pulse{0%{background:#4a2a70}100%{background:transparent}}
.new{animation:pulse 6s ease-out}
#toasts{position:fixed;right:18px;bottom:18px;display:flex;flex-direction:column;gap:8px;z-index:9}
.toast{background:#2b1a44;border:1px solid #a93fe0;color:#eee;padding:8px 12px;border-radius:8px;font-size:13px;box-shadow:0 4px 16px #0008;animation:fadein .3s}
@keyframes fadein{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:none}}
#feed{font-size:12px;color:#bbb;max-height:150px;overflow:auto} #feed div{padding:2px 0;border-bottom:1px solid #222}
#feed .t{color:#777;margin-right:8px} #dot{display:inline-block;width:9px;height:9px;border-radius:50%;background:#333;margin-left:8px;vertical-align:middle}
#dot.live{background:#6fd39a;box-shadow:0 0 8px #6fd39a}
</style></head><body><main>
<h1>Blight model runs <span id="stamp" class="k"></span><span id="dot" title="lights up when new data arrives"></span></h1>
<div class="row" id="cards"></div>
<div class="row"><div style="flex:1"><h2>Now fitting</h2><div id="now"></div></div>
<div style="flex:1"><h2>Latest results</h2><div id="feed"><span class="k">nothing new yet</span></div></div></div>
<h2>AUC by feature set (this run, pooled over held-out seasons)</h2><div id="fits"></div>
<h2>Per-season AUC of the set being fitted</h2><div id="seasons"></div>
<h2>Finished outcome tables</h2><div id="tables"></div>
<h2>Log tail</h2><pre id="tail"></pre>
</main><div id="toasts"></div>
<script>
let lastTables='', seenFits=new Set(), seenSeasons=new Set(), seenOutcomes=new Set(), firstTick=true, feed=[];
function toast(msg){const t=document.createElement('div');t.className='toast';t.textContent=msg;document.getElementById('toasts').appendChild(t);setTimeout(()=>t.remove(),7000)}
function arrived(msg,key){ if(firstTick){return} feed.unshift({t:new Date().toLocaleTimeString(),msg});feed=feed.slice(0,30);
  document.getElementById('feed').innerHTML=feed.map(f=>'<div><span class="t">'+f.t+'</span>'+f.msg+'</div>').join('');
  toast(msg); const d=document.getElementById('dot'); d.classList.add('live'); setTimeout(()=>d.classList.remove('live'),5000); }
const SETS=['weather_basic','weather','calendar+weather','calendar+reports','calendar+reports+prior','all'];
function fmt(x,d=3){return x==null?'':Number(x).toFixed(d)}
function bar(x,cls){return '<span class="bar '+(cls||'')+'" style="width:'+Math.max(0,(x-0.5)*400)+'px"></span>'+fmt(x)}
async function tick(){
  let s; try{s=await (await fetch('/status')).json()}catch(e){document.getElementById('stamp').textContent='(no server)';return}
  const age=s.log_mtime?Math.round(s.now-s.log_mtime):null;
  document.getElementById('stamp').textContent=new Date().toLocaleTimeString()+(s.log?'  '+s.log:'');
  const newFitKeys=new Set();
  for(const f of s.fits){const k=f.outcome+'|'+f.model+'|'+f.set; if(!seenFits.has(k)){seenFits.add(k);newFitKeys.add(k);arrived('result: '+f.outcome+' · '+f.model+' · '+f.set+' → AUC '+fmt(f.auc),k)}}
  for(const o of s.outcomes){ if(!seenOutcomes.has(o.outcome)){seenOutcomes.add(o.outcome);arrived('outcome finished: '+o.outcome+' (Hutton catch '+fmt(o.hutton_recall,2)+' on '+fmt(o.hutton_rate,2)+' of days)',o.outcome)}}
  let newSeason=null;
  for(const e of s.events){ if(e.event==='season'){const k=e.outcome+'|'+e.model+'|'+e.set+'|'+e.season; if(!seenSeasons.has(k)){seenSeasons.add(k); if(!firstTick){newSeason=k; arrived('season '+e.season+' done for '+e.outcome+' · '+e.model+' · '+e.set+(e.auc==null?'':' → AUC '+fmt(e.auc,2)),k)}}}}
  for(const e of s.errors){ if(!seenOutcomes.has('err|'+e)){seenOutcomes.add('err|'+e);arrived('error: '+e,e)}}
  let cards='';
  const p=s.proc; cards+='<div class="card"><div class="k">evaluate process</div><div class="v '+(p?'ok':'warn')+'">'+(p?'running':'not running')+'</div>'+(p?'<div class="k">rss '+fmt(p.rss_gb,2)+' GB, cpu '+p.cpu+'%, up '+p.elapsed+'</div>':'')+'</div>';
  const m=s.memory; cards+='<div class="card"><div class="k">system memory available</div><div class="v '+(m&&m.available_gb<1.5?'bad':'')+'">'+(m?fmt(m.available_gb,1)+' / '+fmt(m.total_gb,0)+' GB':'?')+'</div></div>';
  cards+='<div class="card"><div class="k">log last written</div><div class="v">'+(age==null?'':age+' s ago')+'</div></div>';
  if(s.errors.length) cards+='<div class="card"><div class="k">errors</div><div class="v bad">'+s.errors.slice(-2).join('<br>')+'</div></div>';
  for(const o of s.outcomes) cards+='<div class="card"><div class="k">'+o.outcome+' (done)</div><div class="k">'+o.positives.toLocaleString()+' positives of '+o.n.toLocaleString()+'</div><div class="k">Hutton on '+fmt(o.hutton_rate,2)+' of days, catch '+fmt(o.hutton_recall,2)+'</div></div>';
  document.getElementById('cards').innerHTML=cards;
  // now fitting: from events
  const ev=s.events; let last=null, seas=[], cur=null;
  for(const e of ev){ if(e.event==='start'){cur={...e,seasons:[]}} else if(e.event==='season'&&cur){cur.seasons.push(e)} else if(e.event==='done'&&cur){cur.done=true} }
  let now='';
  if(cur&&!cur.done){ now='<b>'+cur.outcome+'</b> · '+cur.model+' · '+cur.set+' &nbsp; <span class="prog"><i style="width:'+(100*cur.seasons.length/cur.n_seasons)+'%"></i></span> '+cur.seasons.length+'/'+cur.n_seasons+' seasons'; }
  else if(cur&&cur.done){ now='last finished: <b>'+cur.outcome+'</b> · '+cur.model+' · '+cur.set+' (waiting for next set to start, or run over)'; }
  else now='<span class="k">no event stream yet for this run (per-set results only)</span>';
  document.getElementById('now').innerHTML=now;
  if(cur){ document.getElementById('seasons').innerHTML=cur.seasons.map(e=>'<span class="season'+((cur.outcome+'|'+cur.model+'|'+cur.set+'|'+e.season)===newSeason?' new':'')+'">'+String(e.season).slice(2)+'<br>'+(e.auc==null?'·':fmt(e.auc,2))+'</span>').join(''); }
  // fits table: outcome x set, gbm and logistic
  const outs=[...new Set(s.fits.map(f=>f.outcome))]; let h='<table><tr><th>feature set</th>'+outs.map(o=>'<th>'+o+' gbm</th><th>'+o+' logistic</th>').join('')+'</tr>';
  for(const st of SETS){ h+='<tr><td>'+st+'</td>'; for(const o of outs){ for(const mo of ['gbm','logistic']){ const f=s.fits.find(x=>x.outcome===o&&x.set===st&&x.model===mo); const isNew=f&&newFitKeys.has(o+'|'+mo+'|'+st); h+='<td'+(isNew?' class="new"':'')+'>'+(f?bar(f.auc,mo==='logistic'?'bar2':''):'')+'</td>'; } } h+='</tr>'; }
  document.getElementById('fits').innerHTML=h+'</table>';
  // finished tables
  const tablesJson=JSON.stringify(s.tables);
  if(tablesJson!==lastTables){ lastTables=tablesJson;
  const open=new Set([...document.querySelectorAll('#tables details[open]')].map(d=>d.dataset.name));
  let t='';
  for(const [name,rows] of Object.entries(s.tables)){ t+='<details data-name="'+name+'"'+(open.has(name)?' open':'')+'><summary>'+name+' ('+rows.length+' rows)</summary><table><tr><th>model</th><th>AUC</th><th>catch at Hutton rate</th><th>days for Hutton catch</th><th>within-season AUC</th></tr>'+rows.map(r=>'<tr><td>'+r.model+'</td><td>'+bar(r.auc)+'</td><td>'+fmt(r.recall_at_hutton_rate)+'</td><td>'+fmt(r.rate_for_hutton_recall)+'</td><td>'+fmt(r.within_auc)+'</td></tr>').join('')+'</table></details>'; }
  document.getElementById('tables').innerHTML=t; }
  document.getElementById('tail').textContent=s.tail.join('\n');
  firstTick=false;
}
tick(); setInterval(tick,3000);
</script></body></html>'''

class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        if self.path.startswith('/status'):
            body = json.dumps(status()).encode(); ctype = 'application/json'
        else:
            body = HTML.encode(); ctype = 'text/html; charset=utf-8'
        self.send_response(200); self.send_header('Content-Type', ctype); self.send_header('Content-Length', str(len(body))); self.end_headers()
        self.wfile.write(body)

if __name__ == '__main__':
    print(f'dashboard on http://localhost:{PORT}', flush=True)
    HTTPServer(('0.0.0.0', PORT), H).serve_forever()
