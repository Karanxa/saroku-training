"""
saroku training dashboard.

A lightweight Flask web app that tails the training log and renders a
live-updating dashboard: progress bar, loss curves, dataset stats, ETA.

Usage:
    python3 -m saroku.dashboard --log <path-to-log-file> --port 7860
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from flask import Flask, Response, render_template_string

app = Flask(__name__)
LOG_FILE: str = ""

# ── Log parser ────────────────────────────────────────────────────────────────

_RE_PROGRESS = re.compile(
    r"(\d+)%\|.*?\|\s*(\d+)/(\d+)\s+\[(\d+:\d+)<(\S+),\s*([\d.]+)(\w+/it|it/s)"
)
_RE_EVAL     = re.compile(r"'eval_loss':\s*'?([\d.e+-]+)'?.*?'epoch':\s*'?([\d.]+)'?")
_RE_TRAIN_LOSS = re.compile(r"'train_loss':\s*'?([\d.e+-]+)'?.*?'epoch':\s*'?([\d.]+)'?")
_RE_SETUP    = re.compile(r"\[saroku trainer\] (.+?) : (.+)")
_RE_LOADER   = re.compile(r"\[saroku loader\] (.+)")
_RE_EVAL_METRICS = re.compile(
    r"Eval results.*?accuracy=([\d.]+).*?F1=([\d.]+).*?precision=([\d.]+).*?recall=([\d.]+)"
)


def parse_log(path: str) -> dict:
    try:
        text = Path(path).read_text(errors="replace")
    except FileNotFoundError:
        return {"error": f"Log file not found: {path}"}

    lines = text.splitlines()

    setup = {}
    eval_losses: list[dict] = []
    train_losses: list[dict] = []
    progress = {}
    loader_msgs: list[str] = []
    final_metrics: dict = {}
    done = False

    for line in lines:
        # Setup info
        m = _RE_SETUP.search(line)
        if m:
            setup[m.group(1).strip()] = m.group(2).strip()

        # Loader messages
        m = _RE_LOADER.search(line)
        if m:
            loader_msgs.append(m.group(1).strip())

        # Eval loss per epoch
        m = _RE_EVAL.search(line)
        if m and "eval_loss" in line:
            eval_losses.append({
                "epoch": float(m.group(2)),
                "loss":  float(m.group(1)),
            })

        # Train loss (end of training)
        m = _RE_TRAIN_LOSS.search(line)
        if m and "train_loss" in line:
            train_losses.append({
                "epoch": float(m.group(2)),
                "loss":  float(m.group(1)),
            })

        # Step progress bar  e.g.  3%|▎  | 134/4560 [05:13<2:30:24, 2.04s/it]
        m = _RE_PROGRESS.search(line)
        if m:
            elapsed_raw = m.group(4)        # "05:13"
            eta_raw     = m.group(5)        # "2:30:24" or "00:45"
            rate_val    = float(m.group(6))
            rate_unit   = m.group(7)

            def _to_secs(t: str) -> int:
                parts = t.split(":")
                parts = [int(p) for p in parts]
                if len(parts) == 3:
                    return parts[0]*3600 + parts[1]*60 + parts[2]
                return parts[0]*60 + parts[1]

            try:
                eta_secs = _to_secs(eta_raw) if eta_raw not in ("?", "00:00") else 0
            except Exception:
                eta_secs = 0

            progress = {
                "pct":     int(m.group(1)),
                "current": int(m.group(2)),
                "total":   int(m.group(3)),
                "elapsed": elapsed_raw,
                "eta_secs": eta_secs,
                "eta_str":  _fmt_eta(eta_secs),
                "rate":    f"{rate_val} {rate_unit}",
            }

        # Final eval metrics
        m = _RE_EVAL_METRICS.search(line)
        if m:
            final_metrics = {
                "accuracy":  float(m.group(1)),
                "f1":        float(m.group(2)),
                "precision": float(m.group(3)),
                "recall":    float(m.group(4)),
            }

        if "Done. Model saved" in line:
            done = True

    # Derive current epoch from latest eval entry
    current_epoch = eval_losses[-1]["epoch"] if eval_losses else 0
    max_epochs    = int(setup.get("epochs_requested", 5))

    return {
        "setup":         setup,
        "loader_msgs":   loader_msgs,
        "eval_losses":   eval_losses,
        "train_losses":  train_losses,
        "progress":      progress,
        "current_epoch": current_epoch,
        "max_epochs":    max_epochs,
        "final_metrics": final_metrics,
        "done":          done,
        "log_lines":     len(lines),
    }


def _fmt_eta(secs: int) -> str:
    if secs <= 0:
        return "—"
    h, rem = divmod(secs, 3600)
    m, s   = divmod(rem, 60)
    if h:
        return f"{h}h {m}m"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


# ── HTML template ─────────────────────────────────────────────────────────────

_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>saroku training dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  :root {
    --bg: #0f1117; --card: #1a1d27; --border: #2a2d3a;
    --text: #e2e8f0; --muted: #94a3b8; --accent: #6366f1;
    --green: #22c55e; --yellow: #eab308; --red: #ef4444; --blue: #38bdf8;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text); font-family: 'SF Mono', 'Fira Code', monospace; font-size: 13px; padding: 24px; }
  h1 { font-size: 18px; font-weight: 600; color: var(--text); margin-bottom: 4px; }
  .subtitle { color: var(--muted); font-size: 12px; margin-bottom: 24px; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin-bottom: 24px; }
  .card { background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 16px; }
  .card-label { color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 6px; }
  .card-value { font-size: 26px; font-weight: 700; }
  .card-sub { color: var(--muted); font-size: 11px; margin-top: 4px; }
  .green  { color: var(--green); }
  .yellow { color: var(--yellow); }
  .blue   { color: var(--blue); }
  .accent { color: var(--accent); }

  .progress-wrap { background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 20px; margin-bottom: 24px; }
  .progress-header { display: flex; justify-content: space-between; margin-bottom: 12px; font-size: 12px; color: var(--muted); }
  .progress-bar-bg { background: var(--border); border-radius: 999px; height: 10px; overflow: hidden; }
  .progress-bar-fg { background: linear-gradient(90deg, var(--accent), var(--blue)); height: 100%; border-radius: 999px; transition: width 1s ease; }
  .progress-footer { display: flex; justify-content: space-between; margin-top: 8px; font-size: 11px; color: var(--muted); }

  .charts { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 24px; }
  @media (max-width: 700px) { .charts { grid-template-columns: 1fr; } }
  .chart-card { background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 16px; }
  .chart-card h2 { font-size: 13px; margin-bottom: 12px; color: var(--muted); }
  .chart-card canvas { max-height: 200px; }

  .info-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 24px; }
  @media (max-width: 700px) { .info-grid { grid-template-columns: 1fr; } }
  .kv-list { list-style: none; }
  .kv-list li { display: flex; justify-content: space-between; padding: 5px 0; border-bottom: 1px solid var(--border); }
  .kv-list li:last-child { border-bottom: none; }
  .kv-key { color: var(--muted); }
  .kv-val { color: var(--text); text-align: right; max-width: 60%; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }

  .status-dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 6px; }
  .dot-live { background: var(--green); animation: pulse 1.5s infinite; }
  .dot-done { background: var(--accent); }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.3} }

  .badge { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; }
  .badge-green { background: rgba(34,197,94,.15); color: var(--green); border: 1px solid rgba(34,197,94,.3); }
  .badge-yellow { background: rgba(234,179,8,.15); color: var(--yellow); border: 1px solid rgba(234,179,8,.3); }

  .final-box { background: rgba(99,102,241,.08); border: 1px solid rgba(99,102,241,.3); border-radius: 8px; padding: 16px; margin-bottom: 24px; }
  .final-box h2 { color: var(--accent); margin-bottom: 12px; font-size: 14px; }
  .metrics-row { display: grid; grid-template-columns: repeat(4,1fr); gap: 12px; }
  .metric-box { text-align: center; }
  .metric-box .val { font-size: 22px; font-weight: 700; color: var(--green); }
  .metric-box .lbl { font-size: 11px; color: var(--muted); margin-top: 2px; }

  footer { color: var(--muted); font-size: 11px; text-align: center; margin-top: 8px; }
</style>
</head>
<body>
<h1>saroku · training dashboard</h1>
<p class="subtitle" id="ts">last updated —</p>

<div class="progress-wrap" id="prog-wrap">
  <div class="progress-header">
    <span id="prog-label">loading…</span>
    <span id="prog-pct">0%</span>
  </div>
  <div class="progress-bar-bg"><div class="progress-bar-fg" id="prog-bar" style="width:0%"></div></div>
  <div class="progress-footer">
    <span id="prog-steps">—</span>
    <span id="prog-eta">ETA —</span>
  </div>
</div>

<div class="grid" id="stat-cards">
  <div class="card"><div class="card-label">status</div><div class="card-value" id="s-status">—</div></div>
  <div class="card"><div class="card-label">epoch</div><div class="card-value blue" id="s-epoch">—</div><div class="card-sub" id="s-epoch-sub">—</div></div>
  <div class="card"><div class="card-label">latest eval loss</div><div class="card-value yellow" id="s-loss">—</div></div>
  <div class="card"><div class="card-label">train examples</div><div class="card-value accent" id="s-train">—</div><div class="card-sub" id="s-eval-n">—</div></div>
</div>

<div id="final-section" style="display:none">
  <div class="final-box">
    <h2>post-training evaluation results</h2>
    <div class="metrics-row">
      <div class="metric-box"><div class="val" id="m-acc">—</div><div class="lbl">accuracy</div></div>
      <div class="metric-box"><div class="val" id="m-f1">—</div><div class="lbl">F1</div></div>
      <div class="metric-box"><div class="val" id="m-prec">—</div><div class="lbl">precision</div></div>
      <div class="metric-box"><div class="val" id="m-rec">—</div><div class="lbl">recall</div></div>
    </div>
  </div>
</div>

<div class="charts">
  <div class="chart-card"><h2>eval loss per epoch</h2><canvas id="lossChart"></canvas></div>
  <div class="chart-card"><h2>dataset composition</h2><canvas id="dataChart"></canvas></div>
</div>

<div class="info-grid">
  <div class="card"><div class="card-label">training config</div><ul class="kv-list" id="cfg-list"></ul></div>
  <div class="card"><div class="card-label">data pipeline</div><ul class="kv-list" id="data-list"></ul></div>
</div>

<footer>saroku-safety-0.5b · auto-refreshes every 5s</footer>

<script>
let lossChart, dataChart;

function initCharts() {
  const chartDefaults = {
    responsive: true,
    animation: false,
    plugins: { legend: { labels: { color: '#94a3b8', font: { size: 11 } } } },
  };

  lossChart = new Chart(document.getElementById('lossChart'), {
    type: 'line',
    data: { labels: [], datasets: [{
      label: 'eval loss', data: [],
      borderColor: '#6366f1', backgroundColor: 'rgba(99,102,241,.1)',
      tension: 0.3, pointRadius: 4, fill: true,
    }]},
    options: {
      ...chartDefaults,
      scales: {
        x: { ticks: { color: '#94a3b8' }, grid: { color: '#2a2d3a' } },
        y: { ticks: { color: '#94a3b8' }, grid: { color: '#2a2d3a' }, beginAtZero: false },
      },
    },
  });

  dataChart = new Chart(document.getElementById('dataChart'), {
    type: 'doughnut',
    data: {
      labels: ['Train SAFE', 'Train UNSAFE', 'Eval SAFE', 'Eval UNSAFE'],
      datasets: [{
        data: [0, 0, 0, 0],
        backgroundColor: ['#22c55e','#ef4444','#86efac','#fca5a5'],
        borderColor: '#1a1d27', borderWidth: 2,
      }]
    },
    options: {
      ...chartDefaults,
      cutout: '65%',
      plugins: {
        ...chartDefaults.plugins,
        legend: { position: 'right', labels: { color: '#94a3b8', font: { size: 10 } } },
      },
    },
  });
}

function fmt(v) { return (v * 100).toFixed(1) + '%'; }

async function refresh() {
  try {
    const r = await fetch('/api/status');
    const d = await r.json();

    document.getElementById('ts').textContent =
      'last updated ' + new Date().toLocaleTimeString();

    // Progress bar
    const p = d.progress;
    if (p && p.total) {
      document.getElementById('prog-label').innerHTML =
        `<span class="status-dot ${d.done ? 'dot-done' : 'dot-live'}"></span>` +
        (d.done ? 'training complete' : `training — epoch ${d.current_epoch} / ${d.max_epochs}`);
      document.getElementById('prog-pct').textContent = p.pct + '%';
      document.getElementById('prog-bar').style.width = p.pct + '%';
      document.getElementById('prog-steps').textContent =
        `step ${p.current.toLocaleString()} / ${p.total.toLocaleString()} · ${p.rate}`;
      document.getElementById('prog-eta').textContent =
        d.done ? 'done' : `ETA ${p.eta_str} · elapsed ${p.elapsed}`;
    }

    // Stat cards
    const lastLoss = d.eval_losses.length ? d.eval_losses[d.eval_losses.length - 1].loss : null;
    document.getElementById('s-status').innerHTML = d.done
      ? '<span class="badge badge-green">complete</span>'
      : '<span class="badge badge-yellow">training</span>';
    document.getElementById('s-epoch').textContent =
      d.current_epoch ? `${d.current_epoch} / ${d.max_epochs}` : '—';
    document.getElementById('s-epoch-sub').textContent =
      d.eval_losses.length ? `${d.eval_losses.length} eval checkpoints` : 'awaiting first eval';
    document.getElementById('s-loss').textContent =
      lastLoss !== null ? lastLoss.toFixed(4) : '—';

    // Dataset info from setup
    const s = d.setup;
    document.getElementById('s-train').textContent =
      (s.Train || '').split(' ')[0] || '—';
    document.getElementById('s-eval-n').textContent =
      (s.Eval || '').split(' ')[0] ? `eval: ${(s.Eval||'').split(' ')[0]}` : '—';

    // Loss chart
    if (d.eval_losses.length) {
      lossChart.data.labels = d.eval_losses.map(e => `epoch ${e.epoch}`);
      lossChart.data.datasets[0].data = d.eval_losses.map(e => e.loss);
      lossChart.update();
    }

    // Data donut chart — parse from setup strings like "14589 (5838 SAFE / 8751 UNSAFE)"
    const trainMatch = (s.Train||'').match(/(\d+) SAFE \/ (\d+) UNSAFE/);
    const evalMatch  = (s.Eval ||'').match(/(\d+) SAFE \/ (\d+) UNSAFE/);
    if (trainMatch && evalMatch) {
      dataChart.data.datasets[0].data = [
        +trainMatch[1], +trainMatch[2], +evalMatch[1], +evalMatch[2]
      ];
      dataChart.update();
    }

    // Config list
    const cfgKeys = ['Base model', 'Output dir', 'Device'];
    const cfgList = document.getElementById('cfg-list');
    cfgList.innerHTML = cfgKeys.filter(k => s[k]).map(k =>
      `<li><span class="kv-key">${k}</span><span class="kv-val">${s[k]}</span></li>`
    ).join('') + (d.max_epochs ? `<li><span class="kv-key">max epochs</span><span class="kv-val">${d.max_epochs}</span></li>` : '');

    // Data list
    const dataList = document.getElementById('data-list');
    const loaderParts = d.loader_msgs.filter(m => m.startsWith('Prepared') || m.startsWith('Raw'));
    dataList.innerHTML = [
      ...loaderParts.slice(-3).map(m =>
        `<li><span class="kv-key">loader</span><span class="kv-val">${m}</span></li>`
      ),
      s['Total examples'] ? `<li><span class="kv-key">total</span><span class="kv-val">${s['Total examples']}</span></li>` : '',
    ].filter(Boolean).join('');

    // Final metrics
    if (d.final_metrics && d.final_metrics.accuracy !== undefined) {
      document.getElementById('final-section').style.display = 'block';
      document.getElementById('m-acc').textContent  = fmt(d.final_metrics.accuracy);
      document.getElementById('m-f1').textContent   = fmt(d.final_metrics.f1);
      document.getElementById('m-prec').textContent = fmt(d.final_metrics.precision);
      document.getElementById('m-rec').textContent  = fmt(d.final_metrics.recall);
    }

  } catch(e) { console.error(e); }
}

initCharts();
refresh();
setInterval(refresh, 5000);
</script>
</body>
</html>"""


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template_string(_HTML)


@app.route("/api/status")
def api_status():
    data = parse_log(LOG_FILE)
    return Response(json.dumps(data), mimetype="application/json")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="saroku training dashboard")
    parser.add_argument(
        "--log",
        default="/tmp/claude-1000/-home-karan-saroku/"
                "428c6d89-9314-449b-8c75-1dadf78d614c/tasks/bcfi7shzf.output",
        help="Path to training log file",
    )
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    LOG_FILE = args.log
    print(f"[saroku dashboard] Serving at http://localhost:{args.port}")
    print(f"[saroku dashboard] Reading log: {LOG_FILE}")
    app.run(host=args.host, port=args.port, debug=False)
