"""
saroku benchmark dashboard.

Live web view for benchmark progress and results.

Usage:
    python3 -m saroku.benchmark_dashboard --port 7861
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from flask import Flask, Response, render_template_string

app = Flask(__name__)
LOG_FILE    = ""
RESULTS_FILE = ""

# ── Parser ────────────────────────────────────────────────────────────────────

def parse_state() -> dict:
    log_text = ""
    try:
        log_text = Path(app.config["LOG_FILE"]).read_text(errors="replace")
    except (FileNotFoundError, KeyError):
        pass

    # Progress lines: "[benchmark] Inference 150/500..."
    progress = {}
    for m in re.finditer(r"\[benchmark\] Inference (\d+)/(\d+)", log_text):
        current, total = int(m.group(1)), int(m.group(2))
        progress = {"current": current, "total": total, "pct": round(current / total * 100)}

    # Active dataset
    active_dataset = None
    if "[benchmark] Loading ATBench" in log_text:
        active_dataset = "ATBench"
    if "[benchmark] Loading Agent-SafetyBench" in log_text:
        # If ATBench inference is done, we're on ASB
        if re.search(r"Agent-SafetyBench.*prepared", log_text):
            active_dataset = "Agent-SafetyBench"

    # Dataset counts
    counts = {}
    for m in re.finditer(r"\[benchmark\] (ATBench|Agent-SafetyBench) prepared: (\d+) SAFE / (\d+) UNSAFE", log_text):
        counts[m.group(1)] = {"safe": int(m.group(2)), "unsafe": int(m.group(3))}

    done = "Results saved to" in log_text
    crashed = "Traceback" in log_text

    # Load results file if done
    results = {}
    try:
        results = json.loads(Path(app.config["RESULTS_FILE"]).read_text())
    except Exception:
        pass

    return {
        "progress":       progress,
        "active_dataset": active_dataset,
        "counts":         counts,
        "done":           done,
        "crashed":        crashed,
        "results":        results,
    }


# ── HTML ──────────────────────────────────────────────────────────────────────

_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>saroku benchmark dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  :root {
    --bg:#0f1117;--card:#1a1d27;--border:#2a2d3a;
    --text:#e2e8f0;--muted:#94a3b8;--accent:#6366f1;
    --green:#22c55e;--yellow:#eab308;--red:#ef4444;--blue:#38bdf8;--orange:#f97316;
  }
  *{box-sizing:border-box;margin:0;padding:0;}
  body{background:var(--bg);color:var(--text);font-family:'SF Mono','Fira Code',monospace;font-size:13px;padding:24px;}
  h1{font-size:18px;font-weight:600;margin-bottom:4px;}
  .subtitle{color:var(--muted);font-size:12px;margin-bottom:24px;}

  .progress-wrap{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:20px;margin-bottom:24px;}
  .progress-header{display:flex;justify-content:space-between;margin-bottom:12px;font-size:12px;color:var(--muted);}
  .progress-bar-bg{background:var(--border);border-radius:999px;height:10px;overflow:hidden;}
  .progress-bar-fg{background:linear-gradient(90deg,var(--accent),var(--blue));height:100%;border-radius:999px;transition:width 1s ease;}
  .progress-footer{display:flex;justify-content:space-between;margin-top:8px;font-size:11px;color:var(--muted);}

  .datasets{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:24px;}
  @media(max-width:700px){.datasets{grid-template-columns:1fr;}}
  .ds-card{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:16px;}
  .ds-title{font-size:13px;font-weight:600;margin-bottom:12px;}
  .metric-row{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-bottom:12px;}
  .metric-box{text-align:center;background:var(--bg);border-radius:6px;padding:10px 4px;}
  .metric-box .val{font-size:20px;font-weight:700;}
  .metric-box .lbl{font-size:10px;color:var(--muted);margin-top:2px;}
  .confusion{display:grid;grid-template-columns:1fr 1fr;gap:4px;font-size:11px;}
  .confusion div{background:var(--bg);border-radius:4px;padding:6px;text-align:center;}
  .confusion .tp{color:var(--green);}
  .confusion .tn{color:var(--blue);}
  .confusion .fp{color:var(--yellow);}
  .confusion .fn{color:var(--red);}

  .breakdown-wrap{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:16px;margin-bottom:24px;}
  .breakdown-wrap h2{font-size:13px;color:var(--muted);margin-bottom:14px;}
  .charts-row{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:24px;}
  @media(max-width:700px){.charts-row{grid-template-columns:1fr;}}
  .chart-card{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:16px;}
  .chart-card h2{font-size:13px;color:var(--muted);margin-bottom:12px;}

  .status-dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:6px;}
  .dot-live{background:var(--green);animation:pulse 1.5s infinite;}
  .dot-done{background:var(--accent);}
  .dot-error{background:var(--red);}
  @keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}

  .badge{display:inline-block;padding:2px 8px;border-radius:4px;font-size:11px;font-weight:600;}
  .badge-green{background:rgba(34,197,94,.15);color:var(--green);border:1px solid rgba(34,197,94,.3);}
  .badge-yellow{background:rgba(234,179,8,.15);color:var(--yellow);border:1px solid rgba(234,179,8,.3);}
  .badge-red{background:rgba(239,68,68,.15);color:var(--red);border:1px solid rgba(239,68,68,.3);}
  .badge-muted{background:rgba(148,163,184,.1);color:var(--muted);border:1px solid rgba(148,163,184,.2);}

  .bar-row{display:flex;align-items:center;gap:8px;margin-bottom:6px;}
  .bar-label{width:200px;font-size:11px;color:var(--muted);overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}
  .bar-bg{flex:1;background:var(--border);border-radius:4px;height:14px;overflow:hidden;}
  .bar-fg{height:100%;border-radius:4px;transition:width .5s ease;}
  .bar-val{width:50px;text-align:right;font-size:11px;}

  .waiting{color:var(--muted);font-size:12px;text-align:center;padding:32px;}
  footer{color:var(--muted);font-size:11px;text-align:center;margin-top:16px;}
</style>
</head>
<body>
<h1>saroku · benchmark dashboard</h1>
<p class="subtitle" id="ts">last updated —</p>

<div class="progress-wrap">
  <div class="progress-header">
    <span id="prog-label"><span class="status-dot dot-live"></span>loading…</span>
    <span id="prog-pct">0%</span>
  </div>
  <div class="progress-bar-bg"><div class="progress-bar-fg" id="prog-bar" style="width:0%"></div></div>
  <div class="progress-footer">
    <span id="prog-steps">—</span>
    <span id="prog-dataset">—</span>
  </div>
</div>

<div class="datasets" id="dataset-cards">
  <div class="ds-card" id="card-atbench">
    <div class="ds-title">ATBench <span class="badge badge-muted" id="badge-atbench">waiting</span></div>
    <div id="atbench-content"><p class="waiting">Awaiting results…</p></div>
  </div>
  <div class="ds-card" id="card-asb">
    <div class="ds-title">Agent-SafetyBench <span class="badge badge-muted" id="badge-asb">waiting</span></div>
    <div id="asb-content"><p class="waiting">Awaiting results…</p></div>
  </div>
</div>

<div id="breakdown-section" style="display:none">
  <div class="breakdown-wrap">
    <h2>ATBench — accuracy by risk category</h2>
    <div id="breakdown-bars"></div>
  </div>
  <div class="charts-row">
    <div class="chart-card"><h2>ATBench vs Agent-SafetyBench — key metrics</h2><canvas id="compareChart" style="max-height:220px"></canvas></div>
    <div class="chart-card"><h2>ATBench — confusion breakdown</h2><canvas id="confChart" style="max-height:220px"></canvas></div>
  </div>
</div>

<footer>saroku-safety-0.5b · auto-refreshes every 5s</footer>

<script>
let compareChart, confChart;

function initCharts() {
  compareChart = new Chart(document.getElementById('compareChart'), {
    type: 'bar',
    data: {
      labels: ['Accuracy','F1','Precision','Recall'],
      datasets: [
        { label: 'ATBench',            data: [], backgroundColor: 'rgba(99,102,241,.7)',  borderRadius: 4 },
        { label: 'Agent-SafetyBench',  data: [], backgroundColor: 'rgba(56,189,248,.7)',  borderRadius: 4 },
      ]
    },
    options: {
      responsive: true, animation: false,
      scales: {
        x: { ticks:{color:'#94a3b8'}, grid:{color:'#2a2d3a'} },
        y: { ticks:{color:'#94a3b8',callback:v=>v+'%'}, grid:{color:'#2a2d3a'}, min:0, max:100 },
      },
      plugins: { legend:{labels:{color:'#94a3b8',font:{size:11}}} },
    },
  });

  confChart = new Chart(document.getElementById('confChart'), {
    type: 'doughnut',
    data: {
      labels: ['TP (caught unsafe)','TN (allowed safe)','FP (false alarm)','FN (missed unsafe)'],
      datasets: [{ data:[0,0,0,0],
        backgroundColor:['#22c55e','#38bdf8','#eab308','#ef4444'],
        borderColor:'#1a1d27', borderWidth:2 }]
    },
    options: {
      responsive: true, animation: false, cutout:'60%',
      plugins: { legend:{ position:'right', labels:{color:'#94a3b8',font:{size:10}} } },
    },
  });
}

function pct(v){ return (v*100).toFixed(1)+'%'; }

function colorForAcc(v) {
  if(v>=0.90) return '#22c55e';
  if(v>=0.75) return '#eab308';
  return '#ef4444';
}

function metricsHTML(m) {
  return `
    <div class="metric-row">
      <div class="metric-box"><div class="val" style="color:#22c55e">${pct(m.accuracy)}</div><div class="lbl">accuracy</div></div>
      <div class="metric-box"><div class="val" style="color:#6366f1">${pct(m.f1)}</div><div class="lbl">F1</div></div>
      <div class="metric-box"><div class="val" style="color:#38bdf8">${pct(m.precision)}</div><div class="lbl">precision</div></div>
      <div class="metric-box"><div class="val" style="color:#f97316">${pct(m.recall)}</div><div class="lbl">recall</div></div>
    </div>
    <div class="confusion">
      <div class="tp">TP ${m.tp}</div><div class="fp">FP ${m.fp}</div>
      <div class="fn">FN ${m.fn}</div><div class="tn">TN ${m.tn}</div>
    </div>`;
}

async function refresh() {
  try {
    const r = await fetch('/api/status');
    const d = await r.json();
    document.getElementById('ts').textContent = 'last updated ' + new Date().toLocaleTimeString();

    // Progress bar
    const p = d.progress;
    const dotClass = d.done ? 'dot-done' : d.crashed ? 'dot-error' : 'dot-live';
    const statusLabel = d.done ? 'benchmark complete' : d.crashed ? 'error — check log' : `running inference on ${d.active_dataset || '…'}`;
    document.getElementById('prog-label').innerHTML = `<span class="status-dot ${dotClass}"></span>${statusLabel}`;

    if (p && p.total) {
      document.getElementById('prog-pct').textContent  = p.pct + '%';
      document.getElementById('prog-bar').style.width  = p.pct + '%';
      document.getElementById('prog-steps').textContent = `step ${p.current} / ${p.total}`;
    }

    const counts = d.counts || {};
    const parts = Object.entries(counts).map(([k,v]) => `${k}: ${v.safe+v.unsafe} (${v.safe}S/${v.unsafe}U)`);
    document.getElementById('prog-dataset').textContent = parts.join('  ·  ');

    const res = d.results || {};

    // ATBench
    if (res.atbench) {
      const m = res.atbench.metrics;
      document.getElementById('badge-atbench').className = 'badge badge-green';
      document.getElementById('badge-atbench').textContent = 'done';
      document.getElementById('atbench-content').innerHTML = metricsHTML(m);
    }

    // Agent-SafetyBench
    if (res.agent_safetybench) {
      const m = res.agent_safetybench.metrics;
      document.getElementById('badge-asb').className = 'badge badge-green';
      document.getElementById('badge-asb').textContent = 'done';
      document.getElementById('asb-content').innerHTML = metricsHTML(m);
    }

    // Charts + breakdown once both done
    if (res.atbench && res.agent_safetybench) {
      document.getElementById('breakdown-section').style.display = 'block';

      const ma = res.atbench.metrics;
      const mb = res.agent_safetybench.metrics;
      compareChart.data.datasets[0].data = [ma.accuracy,ma.f1,ma.precision,ma.recall].map(v=>+(v*100).toFixed(2));
      compareChart.data.datasets[1].data = [mb.accuracy,mb.f1,mb.precision,mb.recall].map(v=>+(v*100).toFixed(2));
      compareChart.update();

      confChart.data.datasets[0].data = [ma.tp, ma.tn, ma.fp, ma.fn];
      confChart.update();

      // Breakdown bars
      const bd = res.atbench.breakdown || {};
      const sorted = Object.entries(bd).sort((a,b) => b[1].accuracy - a[1].accuracy);
      document.getElementById('breakdown-bars').innerHTML = sorted.map(([cat, m]) => {
        const acc = (m.accuracy * 100).toFixed(1);
        const color = colorForAcc(m.accuracy);
        return `<div class="bar-row">
          <div class="bar-label" title="${cat}">${cat}</div>
          <div class="bar-bg"><div class="bar-fg" style="width:${acc}%;background:${color}"></div></div>
          <div class="bar-val" style="color:${color}">${acc}%</div>
        </div>`;
      }).join('');
    }

  } catch(e) { console.error(e); }
}

initCharts();
refresh();
setInterval(refresh, 5000);
</script>
</body>
</html>"""


@app.route("/")
def index():
    return render_template_string(_HTML)

@app.route("/api/status")
def api_status():
    return Response(json.dumps(parse_state()), mimetype="application/json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="saroku benchmark dashboard")
    parser.add_argument("--log", default="", help="Path to benchmark log file")
    parser.add_argument("--results", default="./models/saroku-safety-0.5b/model/benchmark_results.json")
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    app.config["LOG_FILE"]     = args.log
    app.config["RESULTS_FILE"] = args.results

    print(f"[saroku benchmark dashboard] Serving at http://localhost:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)
