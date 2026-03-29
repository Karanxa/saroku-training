"""
saroku training dashboard — multi-run comparison.

Shows all 3 training runs side-by-side with full detail:
  v1 (saroku-safety-0.5b):    LoRA, 14,589 examples
  v2 (saroku-safety-0.5b-v2): Full fine-tune, 26K examples (label noise bug)
  v3 (saroku-safety-0.5b-v3): Full fine-tune, 22,500 balanced (fixed)

Usage:
    python dashboards/dashboard.py --port 7860
"""

from __future__ import annotations

import json
from pathlib import Path

from flask import Flask, Response, render_template_string

app = Flask(__name__)
BASE_DIR = Path(__file__).parent.parent  # repo root


# ── Data loaders ──────────────────────────────────────────────────────────────

def _load_trainer_state(checkpoints_dir: Path) -> dict:
    files = sorted(checkpoints_dir.glob("checkpoint-*/trainer_state.json"))
    if not files:
        return {}
    try:
        state = json.loads(files[-1].read_text())
    except Exception:
        return {}
    trains, evals, lrs = [], [], []
    for h in state.get("log_history", []):
        step  = h.get("step", 0)
        epoch = round(h.get("epoch", 0), 3)
        if "loss" in h and "eval_loss" not in h:
            trains.append({"step": step, "epoch": epoch, "loss": h["loss"]})
            if "learning_rate" in h:
                lrs.append({"step": step, "lr": h["learning_rate"]})
        if "eval_loss" in h:
            evals.append({"step": step, "epoch": epoch, "loss": h["eval_loss"]})
    return {
        "train_steps": trains,
        "eval_epochs": evals,
        "lr_history":  lrs,
        "global_step": state.get("global_step", 0),
        "max_steps":   state.get("max_steps", 0),
        "num_epochs":  state.get("num_train_epochs", 5),
        "best_metric": state.get("best_metric"),
        "checkpoint":  files[-1].parent.name,
    }


def _load_per_label_from_log(log_path: Path) -> dict:
    """Parse per-label F1/P/R from trainer_v3 style log."""
    if not log_path.exists():
        return {}
    per_label = {}
    in_section = False
    for line in log_path.read_text(errors="replace").splitlines():
        if "Per-label F1:" in line:
            in_section = True
            continue
        if in_section:
            # e.g. "  safe                 n=  62  F1=0.951  P=0.967  R=0.935"
            import re
            m = re.match(r"\s+(\S+)\s+n=\s*(\d+)\s+F1=([\d.]+)\s+P=([\d.]+)\s+R=([\d.]+)", line)
            if m:
                per_label[m.group(1)] = {
                    "n": int(m.group(2)),
                    "f1": float(m.group(3)),
                    "precision": float(m.group(4)),
                    "recall": float(m.group(5)),
                }
            elif line.strip() and not line.strip().startswith("["):
                pass
            else:
                in_section = False
    return per_label


def _load_binary_from_log(log_path: Path) -> dict:
    """Parse binary accuracy/F1 line from trainer_v3 style log."""
    if not log_path.exists():
        return {}
    import re
    for line in log_path.read_text(errors="replace").splitlines():
        m = re.search(r"Binary accuracy\s*:\s*([\d.]+)\s+F1=([\d.]+)\s+precision=([\d.]+)\s+recall=([\d.]+)", line)
        if m:
            return {
                "accuracy":  float(m.group(1)),
                "binary_f1": float(m.group(2)),
                "precision": float(m.group(3)),
                "recall":    float(m.group(4)),
            }
    return {}


def load_all_runs() -> dict:
    runs = {}

    # ── v1 ────────────────────────────────────────────────────────────────────
    v1_dir = BASE_DIR / "models" / "saroku-safety-0.5b"
    v1_state = _load_trainer_state(v1_dir / "checkpoints")
    v1_meta_path = v1_dir / "model" / "saroku_meta.json"
    v1_bench_path = v1_dir / "model" / "benchmark_results.json"
    v1_meta, v1_bench = {}, {}
    try:
        v1_meta  = json.loads(v1_meta_path.read_text())
    except Exception:
        pass
    try:
        v1_bench = json.loads(v1_bench_path.read_text())
    except Exception:
        pass

    runs["v1"] = {
        "label":       "v1 · LoRA",
        "version":     "saroku-safety-0.5b",
        "method":      "LoRA (rank 16)",
        "examples":    14589,
        "train_ex":    v1_meta.get("training_examples", 14589),
        "eval_ex":     v1_meta.get("eval_examples", 1620),
        "epochs":      v1_state.get("num_epochs", 5),
        "epochs_done": 3,
        "note":        "Early stopped at epoch 3. Overfit on 2-class data (safe/unsafe only).",
        "data_sources": v1_meta.get("data_sources", ["synthetic", "toolsafety"]),
        "trainer_state": v1_state,
        "binary": {
            "accuracy":  v1_meta.get("eval_metrics", {}).get("accuracy", 0.9967),
            "binary_f1": v1_meta.get("eval_metrics", {}).get("f1", 0.9967),
            "precision": v1_meta.get("eval_metrics", {}).get("precision", 0.9934),
            "recall":    v1_meta.get("eval_metrics", {}).get("recall", 1.0),
        },
        "per_label": {},
        "atbench":   v1_bench.get("atbench", {}).get("metrics", {}),
        "safetybench": v1_bench.get("agent_safetybench", {}).get("metrics", {}),
    }

    # ── v2 ────────────────────────────────────────────────────────────────────
    v2_dir = BASE_DIR / "models" / "saroku-safety-0.5b-v2"
    v2_state = _load_trainer_state(v2_dir / "checkpoints")
    runs["v2"] = {
        "label":       "v2 · Full fine-tune",
        "version":     "saroku-safety-0.5b-v2",
        "method":      "Full fine-tune",
        "examples":    26000,
        "train_ex":    23400,
        "eval_ex":     2600,
        "epochs":      5,
        "epochs_done": 5,
        "note":        "Label noise bug: ToolSafety unsafe examples had random labels (~87% noise). Model collapsed to predicting 'safe'.",
        "data_sources": ["toolsafety", "beavertails", "aegis2", "contrastive_synth"],
        "trainer_state": v2_state,
        "binary": {
            "accuracy":  0.684,
            "binary_f1": 0.962,
            "precision": 0.988,
            "recall":    0.938,
        },
        "per_label": {
            "safe":             {"n": 324, "f1": 0.980, "precision": 0.980, "recall": 0.980},
            "sycophancy":       {"n":  28, "f1": 0.213, "precision": 0.213, "recall": 0.213},
            "prompt_injection": {"n":  17, "f1": 0.000, "precision": 0.000, "recall": 0.000},
            "trust_hierarchy":  {"n":  20, "f1": 0.000, "precision": 0.000, "recall": 0.000},
            "corrigibility":    {"n":  22, "f1": 0.000, "precision": 0.000, "recall": 0.000},
            "minimal_footprint":{"n":  18, "f1": 0.000, "precision": 0.000, "recall": 0.000},
            "goal_drift":       {"n":  25, "f1": 0.000, "precision": 0.000, "recall": 0.000},
            "honesty":          {"n":  23, "f1": 0.000, "precision": 0.000, "recall": 0.000},
            "consistency":      {"n":  23, "f1": 0.000, "precision": 0.000, "recall": 0.000},
        },
        "atbench": {},
        "safetybench": {},
    }

    # ── v3 ────────────────────────────────────────────────────────────────────
    v3_dir   = BASE_DIR / "models" / "saroku-safety-0.5b-v3"
    v3_log   = v3_dir / "train.log"
    v3_state = _load_trainer_state(v3_dir / "checkpoints")
    v3_bin   = _load_binary_from_log(v3_log)
    v3_pl    = _load_per_label_from_log(v3_log)
    runs["v3"] = {
        "label":       "v3 · Fixed",
        "version":     "saroku-safety-0.5b-v3",
        "method":      "Full fine-tune + weighted loss",
        "examples":    22500,
        "train_ex":    20250,
        "eval_ex":     2250,
        "epochs":      5,
        "epochs_done": 5,
        "note":        "Fixed: dropped noisy ToolSafety unsafe labels, added real datasets (deepset, toxic-chat, sycophancy, truthfulqa), 1K synthetic per category via Gemini 2.5 Flash. Perfectly balanced at 2,500/label.",
        "data_sources": ["beavertails", "aegis2", "deepset_injections", "toxic_chat", "sycophancy_eval", "truthfulqa", "synthetic_gemini", "contrastive"],
        "trainer_state": v3_state,
        "binary": v3_bin or {
            "accuracy":  0.866,
            "binary_f1": 0.993,
            "precision": 0.991,
            "recall":    0.995,
        },
        "per_label": v3_pl or {
            "safe":             {"n":  62, "f1": 0.951, "precision": 0.967, "recall": 0.935},
            "sycophancy":       {"n":  53, "f1": 0.814, "precision": 0.739, "recall": 0.906},
            "prompt_injection": {"n":  51, "f1": 0.990, "precision": 1.000, "recall": 0.980},
            "trust_hierarchy":  {"n":  59, "f1": 0.781, "precision": 0.725, "recall": 0.848},
            "corrigibility":    {"n":  66, "f1": 0.800, "precision": 0.848, "recall": 0.758},
            "minimal_footprint":{"n":  53, "f1": 0.624, "precision": 0.725, "recall": 0.547},
            "goal_drift":       {"n":  52, "f1": 0.914, "precision": 0.906, "recall": 0.923},
            "honesty":          {"n":  53, "f1": 0.991, "precision": 1.000, "recall": 0.981},
            "consistency":      {"n":  51, "f1": 0.932, "precision": 0.923, "recall": 0.941},
        },
        "atbench": {},
        "safetybench": {},
    }

    return runs


# ── HTML ──────────────────────────────────────────────────────────────────────

_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>saroku · training runs</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
:root {
  --bg:#0f1117; --card:#1a1d27; --card2:#1e2133; --border:#2a2d3a;
  --text:#e2e8f0; --muted:#94a3b8; --accent:#6366f1;
  --green:#22c55e; --yellow:#eab308; --red:#ef4444; --blue:#38bdf8;
  --orange:#f97316; --pink:#ec4899; --purple:#a855f7;
  --v1:#38bdf8; --v2:#f97316; --v3:#22c55e;
}
*{box-sizing:border-box;margin:0;padding:0;}
body{background:var(--bg);color:var(--text);font-family:'SF Mono','Fira Code',monospace;font-size:13px;padding:28px 32px;}
h1{font-size:20px;font-weight:700;margin-bottom:4px;}
.subtitle{color:var(--muted);font-size:12px;margin-bottom:28px;}

/* ── Run cards ── */
.run-cards{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin-bottom:28px;}
.run-card{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:20px;position:relative;overflow:hidden;}
.run-card::before{content:'';position:absolute;top:0;left:0;right:0;height:3px;}
.run-card.v1::before{background:var(--v1);}
.run-card.v2::before{background:var(--v2);}
.run-card.v3::before{background:var(--v3);}
.run-title{font-size:14px;font-weight:700;margin-bottom:4px;}
.run-version{color:var(--muted);font-size:11px;margin-bottom:12px;}
.run-note{color:var(--muted);font-size:11px;line-height:1.5;margin-bottom:14px;padding:8px;background:rgba(255,255,255,.03);border-radius:6px;border-left:2px solid var(--border);}
.metric-row{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:8px;}
.metric-box{text-align:center;background:rgba(255,255,255,.03);border-radius:6px;padding:8px 4px;}
.metric-val{font-size:18px;font-weight:700;}
.metric-lbl{color:var(--muted);font-size:10px;margin-top:2px;}
.v1 .metric-val{color:var(--v1);}
.v2 .metric-val{color:var(--v2);}
.v3 .metric-val{color:var(--v3);}
.tag{display:inline-block;padding:2px 7px;border-radius:4px;font-size:10px;font-weight:600;margin-right:4px;margin-bottom:4px;}
.tag-blue{background:rgba(56,189,248,.12);color:var(--v1);border:1px solid rgba(56,189,248,.25);}
.tag-orange{background:rgba(249,115,22,.12);color:var(--v2);border:1px solid rgba(249,115,22,.25);}
.tag-green{background:rgba(34,197,94,.12);color:var(--v3);border:1px solid rgba(34,197,94,.25);}

/* ── Section headers ── */
.section{margin-bottom:28px;}
.section-title{font-size:13px;font-weight:600;color:var(--muted);text-transform:uppercase;letter-spacing:.06em;margin-bottom:14px;padding-bottom:8px;border-bottom:1px solid var(--border);}

/* ── Charts ── */
.chart-grid-2{display:grid;grid-template-columns:1fr 1fr;gap:16px;}
.chart-grid-3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px;}
@media(max-width:900px){.chart-grid-2,.chart-grid-3,.run-cards{grid-template-columns:1fr;}}
.chart-card{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:18px;}
.chart-card h3{font-size:12px;color:var(--muted);margin-bottom:14px;}
.chart-card canvas{max-height:200px;}

/* ── Per-label table ── */
.pl-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin-bottom:28px;}
@media(max-width:1100px){.pl-grid{grid-template-columns:1fr 1fr;}}
@media(max-width:700px){.pl-grid{grid-template-columns:1fr;}}
.pl-card{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:16px;}
.pl-card h3{font-size:12px;margin-bottom:12px;}
.pl-table{width:100%;border-collapse:collapse;font-size:11px;}
.pl-table th{color:var(--muted);text-align:left;padding:4px 6px;border-bottom:1px solid var(--border);}
.pl-table td{padding:5px 6px;border-bottom:1px solid rgba(42,45,58,.6);}
.pl-table tr:last-child td{border-bottom:none;}
.f1-cell{white-space:nowrap;}
.f1-bar{display:inline-block;height:5px;border-radius:3px;vertical-align:middle;margin-left:5px;}
.v1-bar{background:var(--v1);}
.v2-bar{background:var(--v2);}
.v3-bar{background:var(--v3);}
.zero{color:var(--red);font-weight:600;}

/* ── Config table ── */
.cfg-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin-bottom:28px;}
@media(max-width:900px){.cfg-grid{grid-template-columns:1fr;}}
.cfg-card{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:16px;}
.cfg-card h3{font-size:12px;margin-bottom:12px;}
.kv{list-style:none;}
.kv li{display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid rgba(42,45,58,.6);font-size:11px;}
.kv li:last-child{border-bottom:none;}
.kv .k{color:var(--muted);}
.kv .v{text-align:right;max-width:58%;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}

/* ── Benchmark ── */
.bench-grid{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:28px;}

/* ── Comparison highlight ── */
.compare-bar{height:100%;border-radius:3px;}
.delta-pos{color:var(--green);}
.delta-neg{color:var(--red);}
.delta-zero{color:var(--muted);}

footer{color:var(--muted);font-size:11px;text-align:center;margin-top:12px;}
</style>
</head>
<body>

<h1>saroku · training run comparison</h1>
<p class="subtitle">saroku-safety-0.5b · 3 runs · all details &nbsp;·&nbsp; <a href="/eval" style="color:var(--accent);">eval explorer →</a></p>

<!-- ── Run summary cards ── -->
<div class="section">
  <div class="section-title">run overview</div>
  <div class="run-cards" id="run-cards"><!-- filled by JS --></div>
</div>

<!-- ── Eval loss comparison chart ── -->
<div class="section">
  <div class="section-title">eval loss per epoch — all runs</div>
  <div class="chart-grid-2">
    <div class="chart-card"><h3>eval loss (all runs overlay)</h3><canvas id="evalCompChart"></canvas></div>
    <div class="chart-card"><h3>train loss (all runs overlay)</h3><canvas id="trainCompChart"></canvas></div>
  </div>
</div>

<!-- ── Per-label F1 comparison ── -->
<div class="section">
  <div class="section-title">per-label F1 — v2 vs v3</div>
  <div class="chart-grid-2">
    <div class="chart-card" style=""><h3>per-label F1 grouped bar</h3><canvas id="f1CompChart" style="max-height:260px;"></canvas></div>
    <div class="chart-card"><h3>binary metrics comparison</h3><canvas id="binaryChart"></canvas></div>
  </div>
</div>

<!-- ── Per-label detail tables ── -->
<div class="section">
  <div class="section-title">per-label detail (F1 / precision / recall)</div>
  <div class="pl-grid" id="pl-tables"><!-- filled by JS --></div>
</div>

<!-- ── Per-run train loss + LR ── -->
<div class="section">
  <div class="section-title">per-run training curves</div>
  <div class="chart-grid-3">
    <div class="chart-card"><h3>v1 · train loss</h3><canvas id="v1TrainChart"></canvas></div>
    <div class="chart-card"><h3>v2 · train loss</h3><canvas id="v2TrainChart"></canvas></div>
    <div class="chart-card"><h3>v3 · train loss</h3><canvas id="v3TrainChart"></canvas></div>
  </div>
  <div class="chart-grid-3" style="margin-top:16px;">
    <div class="chart-card"><h3>v1 · learning rate</h3><canvas id="v1LrChart"></canvas></div>
    <div class="chart-card"><h3>v2 · learning rate</h3><canvas id="v2LrChart"></canvas></div>
    <div class="chart-card"><h3>v3 · learning rate</h3><canvas id="v3LrChart"></canvas></div>
  </div>
</div>

<!-- ── v1 benchmark results ── -->
<div class="section" id="bench-section">
  <div class="section-title">v1 external benchmark results</div>
  <div class="bench-grid">
    <div class="chart-card"><h3>ATBench · per-category breakdown</h3><canvas id="atbenchChart" style="max-height:240px;"></canvas></div>
    <div class="chart-card"><h3>ATBench + AgentSafetyBench · summary</h3><canvas id="benchSummChart"></canvas></div>
  </div>
</div>

<!-- ── Config tables ── -->
<div class="section">
  <div class="section-title">training configuration</div>
  <div class="cfg-grid" id="cfg-tables"><!-- filled by JS --></div>
</div>

<footer>saroku-safety-0.5b · auto-refreshes every 30s · {{ ts }}</footer>

<script>
const DATA = {{ data_json }};

function pct(v){ return (v*100).toFixed(1)+'%'; }
function f1Color(f){ if(f>=0.85) return '#22c55e'; if(f>=0.65) return '#eab308'; return '#ef4444'; }
function thin(arr, n){ if(arr.length<=n) return arr; const s=Math.ceil(arr.length/n); return arr.filter((_,i)=>i%s===0); }

const baseOpts = (yFmt) => ({
  responsive:true, animation:false,
  plugins:{ legend:{ labels:{ color:'#94a3b8', font:{size:10} } } },
  scales:{
    x:{ ticks:{color:'#94a3b8',maxTicksLimit:8}, grid:{color:'#2a2d3a'} },
    y:{ ticks:{color:'#94a3b8', callback: yFmt||undefined}, grid:{color:'#2a2d3a'}, beginAtZero:false },
  },
});

// ── Run cards ─────────────────────────────────────────────────────────────────
const versionClass = {v1:'v1', v2:'v2', v3:'v3'};
const versionTagClass = {v1:'tag-blue', v2:'tag-orange', v3:'tag-green'};

document.getElementById('run-cards').innerHTML = ['v1','v2','v3'].map(k => {
  const r = DATA[k];
  const b = r.binary;
  return `
  <div class="run-card ${k}">
    <div class="run-title">${r.label}</div>
    <div class="run-version">${r.version}</div>
    <div class="run-note">${r.note}</div>
    <div class="metric-row">
      <div class="metric-box"><div class="metric-val">${pct(b.binary_f1||b.f1||0)}</div><div class="metric-lbl">binary F1</div></div>
      <div class="metric-box"><div class="metric-val">${pct(b.accuracy||0)}</div><div class="metric-lbl">accuracy</div></div>
    </div>
    <div class="metric-row">
      <div class="metric-box"><div class="metric-val">${pct(b.precision||0)}</div><div class="metric-lbl">precision</div></div>
      <div class="metric-box"><div class="metric-val">${pct(b.recall||0)}</div><div class="metric-lbl">recall</div></div>
    </div>
    <div style="margin-top:12px;">
      ${r.data_sources.map(s=>`<span class="tag ${versionTagClass[k]}">${s}</span>`).join('')}
    </div>
    <div style="margin-top:10px;font-size:11px;color:var(--muted);">
      <span>${r.train_ex.toLocaleString()} train</span> &nbsp;·&nbsp;
      <span>${r.eval_ex.toLocaleString()} eval</span> &nbsp;·&nbsp;
      <span>${r.epochs_done} epochs</span>
    </div>
  </div>`;
}).join('');

// ── Eval loss comparison ───────────────────────────────────────────────────────
const evalComp = new Chart(document.getElementById('evalCompChart'), {
  type:'line',
  data:{ labels:['epoch 1','epoch 2','epoch 3','epoch 4','epoch 5'],
    datasets:[
      { label:'v1', data: DATA.v1.trainer_state.eval_epochs.map(e=>e.loss), borderColor:'#38bdf8', backgroundColor:'rgba(56,189,248,.08)', tension:.3, pointRadius:4, fill:false },
      { label:'v2', data: DATA.v2.trainer_state.eval_epochs.map(e=>e.loss), borderColor:'#f97316', backgroundColor:'rgba(249,115,22,.08)', tension:.3, pointRadius:4, fill:false },
      { label:'v3', data: DATA.v3.trainer_state.eval_epochs.map(e=>e.loss), borderColor:'#22c55e', backgroundColor:'rgba(34,197,94,.08)', tension:.3, pointRadius:4, fill:false },
    ]
  },
  options: baseOpts(),
});

// ── Train loss comparison ──────────────────────────────────────────────────────
function trainDataset(key, color) {
  const pts = thin(DATA[key].trainer_state.train_steps, 80);
  return { label:key, data:pts.map(e=>e.loss), labels:pts.map(e=>e.epoch.toFixed(2)),
    borderColor:color, backgroundColor:'transparent', tension:.3, pointRadius:0, borderWidth:1.5, fill:false };
}
const allTrainLabels = thin(DATA.v3.trainer_state.train_steps, 80).map(e=>e.epoch.toFixed(2));
const trainComp = new Chart(document.getElementById('trainCompChart'), {
  type:'line',
  data:{ labels: allTrainLabels,
    datasets:[
      trainDataset('v1','#38bdf8'),
      trainDataset('v2','#f97316'),
      trainDataset('v3','#22c55e'),
    ]
  },
  options: baseOpts(),
});

// ── Per-label F1 grouped bar ───────────────────────────────────────────────────
const labels9 = Object.keys(DATA.v3.per_label);
const f1CompChart = new Chart(document.getElementById('f1CompChart'), {
  type:'bar',
  data:{
    labels: labels9,
    datasets:[
      { label:'v2 F1', data: labels9.map(l=>DATA.v2.per_label[l]?.f1||0), backgroundColor:'rgba(249,115,22,.7)', borderColor:'#f97316', borderWidth:1 },
      { label:'v3 F1', data: labels9.map(l=>DATA.v3.per_label[l]?.f1||0), backgroundColor:'rgba(34,197,94,.7)', borderColor:'#22c55e', borderWidth:1 },
    ]
  },
  options:{ ...baseOpts(), scales:{ x:{ticks:{color:'#94a3b8',maxTicksLimit:20,maxRotation:35},grid:{color:'#2a2d3a'}}, y:{ticks:{color:'#94a3b8'},grid:{color:'#2a2d3a'},min:0,max:1} } },
});

// ── Binary metrics comparison bar ─────────────────────────────────────────────
const binaryChart = new Chart(document.getElementById('binaryChart'), {
  type:'bar',
  data:{
    labels:['accuracy','binary F1','precision','recall'],
    datasets:[
      { label:'v1', data:[DATA.v1.binary.accuracy, DATA.v1.binary.binary_f1||DATA.v1.binary.f1, DATA.v1.binary.precision, DATA.v1.binary.recall], backgroundColor:'rgba(56,189,248,.7)', borderColor:'#38bdf8', borderWidth:1 },
      { label:'v2', data:[DATA.v2.binary.accuracy, DATA.v2.binary.binary_f1, DATA.v2.binary.precision, DATA.v2.binary.recall], backgroundColor:'rgba(249,115,22,.7)', borderColor:'#f97316', borderWidth:1 },
      { label:'v3', data:[DATA.v3.binary.accuracy, DATA.v3.binary.binary_f1, DATA.v3.binary.precision, DATA.v3.binary.recall], backgroundColor:'rgba(34,197,94,.7)', borderColor:'#22c55e', borderWidth:1 },
    ]
  },
  options:{ ...baseOpts(), scales:{ x:{ticks:{color:'#94a3b8'},grid:{color:'#2a2d3a'}}, y:{ticks:{color:'#94a3b8'},grid:{color:'#2a2d3a'},min:0,max:1} } },
});

// ── Per-label detail tables ────────────────────────────────────────────────────
const plHtml = ['v1','v2','v3'].map(k => {
  const r = DATA[k];
  const pl = r.per_label;
  const rows = Object.keys(pl).length
    ? Object.entries(pl).sort((a,b)=>b[1].f1-a[1].f1).map(([lbl,v])=>{
        const f = v.f1;
        const cls = f===0 ? 'zero' : '';
        return `<tr>
          <td>${lbl}</td>
          <td>${v.n}</td>
          <td class="f1-cell ${cls}">${f.toFixed(3)} <span class="f1-bar ${k}-bar" style="width:${Math.round(f*55)}px"></span></td>
          <td>${v.precision.toFixed(3)}</td>
          <td>${v.recall.toFixed(3)}</td>
        </tr>`;
      }).join('')
    : `<tr><td colspan="5" style="color:var(--muted);text-align:center;padding:12px;">no per-label data (binary classifier)</td></tr>`;
  return `
  <div class="pl-card">
    <h3 style="color:var(--${k})">${r.label}</h3>
    <table class="pl-table">
      <thead><tr><th>label</th><th>n</th><th>F1</th><th>P</th><th>R</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>
  </div>`;
}).join('');
document.getElementById('pl-tables').innerHTML = plHtml;

// ── Per-run train loss charts ──────────────────────────────────────────────────
function makeTrainChart(canvasId, key, color) {
  const pts = thin(DATA[key].trainer_state.train_steps, 100);
  return new Chart(document.getElementById(canvasId), {
    type:'line',
    data:{ labels:pts.map(e=>`${e.epoch.toFixed(1)}`), datasets:[{ label:'train loss', data:pts.map(e=>e.loss), borderColor:color, backgroundColor:`${color}12`, tension:.3, pointRadius:0, fill:true }] },
    options: baseOpts(),
  });
}
function makeLrChart(canvasId, key, color) {
  const pts = thin(DATA[key].trainer_state.lr_history, 100);
  return new Chart(document.getElementById(canvasId), {
    type:'line',
    data:{ labels:pts.map(e=>`${e.step}`), datasets:[{ label:'learning rate', data:pts.map(e=>e.lr), borderColor:color, backgroundColor:`${color}12`, tension:.3, pointRadius:0, fill:true }] },
    options: { ...baseOpts(v=>v.toExponential(1)) },
  });
}
makeTrainChart('v1TrainChart','v1','#38bdf8');
makeTrainChart('v2TrainChart','v2','#f97316');
makeTrainChart('v3TrainChart','v3','#22c55e');
makeLrChart('v1LrChart','v1','#38bdf8');
makeLrChart('v2LrChart','v2','#f97316');
makeLrChart('v3LrChart','v3','#22c55e');

// ── v1 ATBench ────────────────────────────────────────────────────────────────
const atb = DATA.v1.atbench_breakdown;
if(atb && Object.keys(atb).length) {
  const cats = Object.keys(atb);
  new Chart(document.getElementById('atbenchChart'), {
    type:'bar',
    data:{
      labels: cats.map(c=>c.replace(/_/g,' ')),
      datasets:[
        { label:'F1', data:cats.map(c=>atb[c].f1), backgroundColor:'rgba(56,189,248,.7)', borderColor:'#38bdf8', borderWidth:1 },
        { label:'accuracy', data:cats.map(c=>atb[c].accuracy), backgroundColor:'rgba(99,102,241,.5)', borderColor:'#6366f1', borderWidth:1 },
      ]
    },
    options:{ ...baseOpts(), scales:{ x:{ticks:{color:'#94a3b8',maxRotation:35,font:{size:9}},grid:{color:'#2a2d3a'}}, y:{ticks:{color:'#94a3b8'},grid:{color:'#2a2d3a'},min:0,max:1} } },
  });
}
const atbMeta = DATA.v1.atbench_meta;
const sbMeta  = DATA.v1.safetybench_meta;
if(atbMeta||sbMeta){
  new Chart(document.getElementById('benchSummChart'), {
    type:'bar',
    data:{
      labels:['accuracy','precision','recall','F1'],
      datasets:[
        { label:'ATBench',         data:atbMeta  ? [atbMeta.accuracy,atbMeta.precision,atbMeta.recall,atbMeta.f1]  : [], backgroundColor:'rgba(56,189,248,.7)',  borderColor:'#38bdf8', borderWidth:1 },
        { label:'AgentSafetyBench',data:sbMeta   ? [sbMeta.accuracy, sbMeta.precision, sbMeta.recall, sbMeta.f1]  : [], backgroundColor:'rgba(99,102,241,.6)',  borderColor:'#6366f1', borderWidth:1 },
      ]
    },
    options:{ ...baseOpts(), scales:{ x:{ticks:{color:'#94a3b8'},grid:{color:'#2a2d3a'}}, y:{ticks:{color:'#94a3b8'},grid:{color:'#2a2d3a'},min:0,max:1} } },
  });
}

// ── Config tables ─────────────────────────────────────────────────────────────
const cfgData = {
  v1: [
    ['method','LoRA rank 16'],
    ['base model','Qwen2.5-0.5B-Instruct'],
    ['examples',DATA.v1.examples.toLocaleString()],
    ['train / eval',`${DATA.v1.train_ex.toLocaleString()} / ${DATA.v1.eval_ex.toLocaleString()}`],
    ['epochs done',DATA.v1.epochs_done],
    ['early stopping','patience 2'],
    ['data sources','synthetic + toolsafety'],
    ['labels','2 (safe/unsafe)'],
    ['issue','overfit, no category labels'],
  ],
  v2: [
    ['method','full fine-tune'],
    ['base model','Qwen2.5-0.5B-Instruct'],
    ['examples',DATA.v2.examples.toLocaleString()],
    ['train / eval',`${DATA.v2.train_ex.toLocaleString()} / ${DATA.v2.eval_ex.toLocaleString()}`],
    ['epochs done',DATA.v2.epochs_done],
    ['LR','1e-5 cosine'],
    ['label smoothing','0.05'],
    ['labels','9 categories'],
    ['issue','~87% of unsafe labels were random (ToolSafety bug)'],
  ],
  v3: [
    ['method','full fine-tune + weighted loss'],
    ['base model','Qwen2.5-0.5B-Instruct'],
    ['examples',DATA.v3.examples.toLocaleString()],
    ['train / eval',`${DATA.v3.train_ex.toLocaleString()} / ${DATA.v3.eval_ex.toLocaleString()}`],
    ['epochs done',DATA.v3.epochs_done],
    ['LR','1e-5 cosine'],
    ['label smoothing','0.05'],
    ['labels','9 · 2,500 per label'],
    ['synthetic data','1,000/cat via Gemini 2.5 Flash'],
  ],
};
document.getElementById('cfg-tables').innerHTML = ['v1','v2','v3'].map(k=>`
  <div class="cfg-card">
    <h3 style="color:var(--${k});margin-bottom:12px;">${DATA[k].label}</h3>
    <ul class="kv">
      ${cfgData[k].map(([key,val])=>`<li><span class="k">${key}</span><span class="v">${val}</span></li>`).join('')}
    </ul>
  </div>`).join('');

// ── Auto-refresh every 30s ─────────────────────────────────────────────────────
setTimeout(()=>location.reload(), 30000);
</script>
</body>
</html>"""


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    runs = load_all_runs()
    # Flatten for template
    data = {}
    for k, r in runs.items():
        data[k] = {
            "label":        r["label"],
            "version":      r["version"],
            "note":         r["note"],
            "examples":     r["examples"],
            "train_ex":     r["train_ex"],
            "eval_ex":      r["eval_ex"],
            "epochs":       r["epochs"],
            "epochs_done":  r["epochs_done"],
            "data_sources": r["data_sources"],
            "binary":       r["binary"],
            "per_label":    r["per_label"],
            "trainer_state": r["trainer_state"],
            "atbench_meta":      r["atbench"],
            "atbench_breakdown": r.get("atbench_breakdown", {}),
            "safetybench_meta":  r["safetybench"],
        }
    # Embed ATBench breakdown
    v1 = runs.get("v1", {})
    v1_bench_path = BASE_DIR / "models" / "saroku-safety-0.5b" / "model" / "benchmark_results.json"
    try:
        bench = json.loads(v1_bench_path.read_text())
        data["v1"]["atbench_breakdown"] = bench.get("atbench", {}).get("breakdown", {})
        data["v1"]["atbench_meta"]      = bench.get("atbench", {}).get("metrics", {})
        data["v1"]["safetybench_meta"]  = bench.get("agent_safetybench", {}).get("metrics", {})
    except Exception:
        pass

    import datetime
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
    html = _HTML.replace("{{ data_json }}", json.dumps(data)).replace("{{ ts }}", ts)
    return render_template_string(html)


@app.route("/api/data")
def api_data():
    runs = load_all_runs()
    return Response(json.dumps(runs, default=str), mimetype="application/json")


@app.route("/api/eval")
def api_eval():
    path = BASE_DIR / "data" / "eval_predictions.json"
    if not path.exists():
        return Response(json.dumps({"error": "eval_predictions.json not found"}), mimetype="application/json")
    return Response(path.read_text(), mimetype="application/json")


_EVAL_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>saroku · eval explorer</title>
<style>
:root{
  --bg:#0f1117;--card:#1a1d27;--card2:#1e2133;--border:#2a2d3a;
  --text:#e2e8f0;--muted:#94a3b8;--accent:#6366f1;
  --green:#22c55e;--yellow:#eab308;--red:#ef4444;--blue:#38bdf8;
  --v1:#38bdf8;--v2:#f97316;--v3:#22c55e;
}
*{box-sizing:border-box;margin:0;padding:0;}
body{background:var(--bg);color:var(--text);font-family:'SF Mono','Fira Code',monospace;font-size:13px;padding:28px 32px;}
h1{font-size:20px;font-weight:700;margin-bottom:4px;}
.subtitle{color:var(--muted);font-size:12px;margin-bottom:20px;}
a{color:var(--accent);text-decoration:none;}

/* ── Controls ── */
.controls{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:20px;align-items:center;}
select,input{background:var(--card);border:1px solid var(--border);color:var(--text);padding:6px 10px;border-radius:6px;font-family:inherit;font-size:12px;}
select:focus,input:focus{outline:none;border-color:var(--accent);}
.btn{background:var(--accent);color:#fff;border:none;padding:6px 14px;border-radius:6px;cursor:pointer;font-size:12px;}
.btn:hover{opacity:.85;}
.count{color:var(--muted);font-size:12px;margin-left:auto;}

/* ── Stats bar ── */
.stats{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:20px;}
.stat-card{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:14px;}
.stat-card h3{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.06em;margin-bottom:8px;}
.acc{font-size:28px;font-weight:700;}
.acc.v1{color:var(--v1);}
.acc.v2{color:var(--v2);}
.acc.v3{color:var(--v3);}
.mini-bar{display:grid;grid-template-columns:1fr;gap:3px;margin-top:8px;font-size:10px;}
.mini-row{display:flex;align-items:center;gap:6px;}
.mini-lbl{width:120px;color:var(--muted);overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}
.mini-bg{flex:1;background:var(--border);border-radius:2px;height:5px;}
.mini-fg{height:5px;border-radius:2px;}
.fg-v1{background:var(--v1);}
.fg-v2{background:var(--v2);}
.fg-v3{background:var(--v3);}
.mini-val{width:36px;text-align:right;color:var(--muted);}

/* ── Table ── */
.table-wrap{overflow-x:auto;}
table{width:100%;border-collapse:collapse;font-size:12px;}
th{color:var(--muted);text-align:left;padding:8px 10px;border-bottom:2px solid var(--border);white-space:nowrap;cursor:pointer;user-select:none;}
th:hover{color:var(--text);}
td{padding:8px 10px;border-bottom:1px solid rgba(42,45,58,.5);vertical-align:top;}
tr:hover td{background:rgba(255,255,255,.02);}
.correct{color:var(--green);}
.wrong{color:var(--red);}
.tag{display:inline-block;padding:1px 7px;border-radius:4px;font-size:10px;font-weight:600;white-space:nowrap;}
.tag-safe{background:rgba(34,197,94,.12);color:var(--green);border:1px solid rgba(34,197,94,.2);}
.tag-unsafe{background:rgba(239,68,68,.12);color:var(--red);border:1px solid rgba(239,68,68,.2);}
.tag-match{background:rgba(34,197,94,.12);color:var(--green);border:1px solid rgba(34,197,94,.2);}
.tag-miss{background:rgba(239,68,68,.12);color:var(--red);border:1px solid rgba(239,68,68,.2);}
.tag-src{background:rgba(99,102,241,.12);color:var(--accent);border:1px solid rgba(99,102,241,.2);}
.prompt-cell{max-width:380px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:var(--muted);}
.prompt-cell.expanded{white-space:normal;color:var(--text);}
.expand-btn{background:none;border:none;color:var(--accent);cursor:pointer;font-size:10px;padding:0 4px;}

/* ── Pagination ── */
.pagination{display:flex;gap:6px;justify-content:center;margin-top:16px;flex-wrap:wrap;}
.page-btn{background:var(--card);border:1px solid var(--border);color:var(--text);padding:4px 10px;border-radius:4px;cursor:pointer;font-size:11px;}
.page-btn.active{background:var(--accent);border-color:var(--accent);}
.page-btn:hover:not(.active){border-color:var(--accent);}

/* ── Detail modal ── */
.modal-bg{display:none;position:fixed;inset:0;background:rgba(0,0,0,.7);z-index:100;padding:40px;overflow-y:auto;}
.modal-bg.open{display:flex;align-items:flex-start;justify-content:center;}
.modal{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:24px;max-width:760px;width:100%;position:relative;}
.modal-close{position:absolute;top:14px;right:16px;background:none;border:none;color:var(--muted);font-size:18px;cursor:pointer;}
.modal-close:hover{color:var(--text);}
.modal h2{font-size:14px;margin-bottom:16px;}
.prompt-block{background:var(--bg);border:1px solid var(--border);border-radius:6px;padding:12px;font-size:11px;line-height:1.6;white-space:pre-wrap;margin-bottom:16px;max-height:300px;overflow-y:auto;}
.pred-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;}
.pred-box{background:var(--bg);border:1px solid var(--border);border-radius:6px;padding:10px;text-align:center;}
.pred-box .ver{font-size:10px;color:var(--muted);margin-bottom:4px;}
.pred-box .val{font-size:15px;font-weight:700;}
.pred-box.ok{border-color:rgba(34,197,94,.4);}
.pred-box.fail{border-color:rgba(239,68,68,.4);}
</style>
</head>
<body>

<h1>saroku · eval explorer</h1>
<p class="subtitle"><a href="/">← back to overview</a> &nbsp;·&nbsp; 50 examples per label · 450 total · all 3 runs</p>

<div class="stats" id="stats-bar"><!-- filled --></div>

<div class="controls">
  <select id="filter-version"><option value="">all versions</option><option>v1</option><option>v2</option><option>v3</option></select>
  <select id="filter-label"><option value="">all labels</option></select>
  <select id="filter-source"><option value="">all sources</option></select>
  <select id="filter-result">
    <option value="">all results</option>
    <option value="all_correct">all 3 correct</option>
    <option value="v3_correct">v3 correct</option>
    <option value="v3_wrong">v3 wrong</option>
    <option value="all_wrong">all 3 wrong</option>
    <option value="regression">v3 worse than v2</option>
  </select>
  <input id="search" placeholder="search prompt…" style="min-width:200px;">
  <span class="count" id="count-label">— examples</span>
</div>

<div class="table-wrap">
<table id="main-table">
  <thead>
    <tr>
      <th>#</th>
      <th>source</th>
      <th>true label</th>
      <th>v1 pred</th>
      <th>v2 pred</th>
      <th>v3 pred</th>
      <th>prompt</th>
      <th></th>
    </tr>
  </thead>
  <tbody id="table-body"></tbody>
</table>
</div>
<div class="pagination" id="pagination"></div>

<!-- Modal -->
<div class="modal-bg" id="modal">
  <div class="modal">
    <button class="modal-close" onclick="closeModal()">✕</button>
    <h2>example detail</h2>
    <div class="prompt-block" id="modal-prompt"></div>
    <div style="margin-bottom:12px;font-size:12px;">
      True label: <strong id="modal-true"></strong> &nbsp;·&nbsp;
      Source: <strong id="modal-source"></strong> &nbsp;·&nbsp;
      Safe: <strong id="modal-safe"></strong>
    </div>
    <div class="pred-grid" id="modal-preds"></div>
  </div>
</div>

<script>
let ALL_DATA = null;
let ROWS = [];      // joined rows: {id, prompt, label, source, is_safe, v1:{pred,correct,raw}, v2:…, v3:…}
let FILTERED = [];
let PAGE = 0;
const PAGE_SIZE = 50;

async function init() {
  const r = await fetch('/api/eval');
  ALL_DATA = await r.json();

  // Join all versions into per-prompt rows
  const byId = {};
  for (const ver of ['v1','v2','v3']) {
    for (const p of (ALL_DATA[ver]||[])) {
      if (!byId[p.id]) byId[p.id] = { id:p.id, prompt:p.prompt, label:p.label, source:p.source, is_safe:p.is_safe };
      byId[p.id][ver] = { pred:p.pred, correct:p.correct, raw:p.raw_output };
    }
  }
  ROWS = Object.values(byId).sort((a,b)=>a.id-b.id);

  // Populate filter selects
  const labels  = [...new Set(ROWS.map(r=>r.label))].sort();
  const sources = [...new Set(ROWS.map(r=>r.source))].sort();
  document.getElementById('filter-label').innerHTML +=
    labels.map(l=>`<option>${l}</option>`).join('');
  document.getElementById('filter-source').innerHTML +=
    sources.map(s=>`<option>${s}</option>`).join('');

  renderStats();
  applyFilters();
}

function renderStats() {
  const html = ['v1','v2','v3'].map(ver => {
    const rows = ROWS.filter(r=>r[ver]);
    const total = rows.length;
    const correct = rows.filter(r=>r[ver].correct).length;
    const acc = total ? (correct/total*100).toFixed(1) : '—';
    // per-label
    const byLbl = {};
    for (const r of rows) {
      if (!byLbl[r.label]) byLbl[r.label] = {c:0,n:0};
      byLbl[r.label].n++;
      if (r[ver].correct) byLbl[r.label].c++;
    }
    const barRows = Object.entries(byLbl).sort((a,b)=>b[1].c/b[1].n - a[1].c/a[1].n).map(([lbl,v])=>{
      const f = v.n ? v.c/v.n : 0;
      return `<div class="mini-row"><span class="mini-lbl">${lbl}</span>
        <div class="mini-bg"><div class="mini-fg fg-${ver}" style="width:${(f*100).toFixed(0)}%"></div></div>
        <span class="mini-val">${(f*100).toFixed(0)}%</span></div>`;
    }).join('');
    return `<div class="stat-card">
      <h3>${ver}</h3>
      <div class="acc ${ver}">${acc}%</div>
      <div style="color:var(--muted);font-size:10px;margin-top:2px;">${correct}/${total} correct</div>
      <div class="mini-bar" style="margin-top:10px;">${barRows}</div>
    </div>`;
  }).join('');
  document.getElementById('stats-bar').innerHTML = html;
}

function applyFilters() {
  const ver    = document.getElementById('filter-version').value;
  const label  = document.getElementById('filter-label').value;
  const source = document.getElementById('filter-source').value;
  const result = document.getElementById('filter-result').value;
  const search = document.getElementById('search').value.toLowerCase();

  FILTERED = ROWS.filter(r => {
    if (label  && r.label  !== label)  return false;
    if (source && r.source !== source) return false;
    if (search && !r.prompt.toLowerCase().includes(search)) return false;
    if (result === 'all_correct')  return (r.v1?.correct && r.v2?.correct && r.v3?.correct);
    if (result === 'v3_correct')   return r.v3?.correct;
    if (result === 'v3_wrong')     return r.v3 && !r.v3.correct;
    if (result === 'all_wrong')    return (!r.v1?.correct && !r.v2?.correct && !r.v3?.correct);
    if (result === 'regression')   return (r.v2?.correct && r.v3 && !r.v3.correct);
    return true;
  });

  PAGE = 0;
  document.getElementById('count-label').textContent = `${FILTERED.length} examples`;
  renderTable();
  renderPagination();
}

function predTag(p, label, ver) {
  if (!p) return '<span style="color:var(--muted)">—</span>';
  const ok = p.pred === label;
  const cls = ok ? 'tag-match' : 'tag-miss';
  return `<span class="tag ${cls}">${p.pred}</span>`;
}

function renderTable() {
  const start = PAGE * PAGE_SIZE;
  const slice = FILTERED.slice(start, start + PAGE_SIZE);
  const html = slice.map((r, idx) => `
    <tr>
      <td style="color:var(--muted)">${start+idx+1}</td>
      <td><span class="tag tag-src">${r.source}</span></td>
      <td><span class="tag ${r.is_safe?'tag-safe':'tag-unsafe'}">${r.label}</span></td>
      <td>${predTag(r.v1, r.label, 'v1')}</td>
      <td>${predTag(r.v2, r.label, 'v2')}</td>
      <td>${predTag(r.v3, r.label, 'v3')}</td>
      <td class="prompt-cell">${r.prompt.replace(/</g,'&lt;').replace(/>/g,'&gt;').substring(0,120)}…</td>
      <td><button class="expand-btn" onclick="openModal(${r.id})">view</button></td>
    </tr>`).join('');
  document.getElementById('table-body').innerHTML = html;
}

function renderPagination() {
  const total = Math.ceil(FILTERED.length / PAGE_SIZE);
  if (total <= 1) { document.getElementById('pagination').innerHTML=''; return; }
  const pages = [];
  for (let i=0;i<total;i++) {
    pages.push(`<button class="page-btn ${i===PAGE?'active':''}" onclick="goPage(${i})">${i+1}</button>`);
  }
  document.getElementById('pagination').innerHTML = pages.join('');
}

function goPage(p) { PAGE=p; renderTable(); renderPagination(); window.scrollTo(0,0); }

function openModal(id) {
  const r = ROWS.find(x=>x.id===id);
  if (!r) return;
  document.getElementById('modal-prompt').textContent = r.prompt;
  document.getElementById('modal-true').textContent   = r.label;
  document.getElementById('modal-source').textContent = r.source;
  document.getElementById('modal-safe').textContent   = r.is_safe ? 'yes' : 'no';
  document.getElementById('modal-preds').innerHTML = ['v1','v2','v3'].map(ver => {
    const p = r[ver];
    if (!p) return `<div class="pred-box"><div class="ver">${ver}</div><div class="val" style="color:var(--muted)">N/A</div></div>`;
    const ok = p.pred === r.label;
    return `<div class="pred-box ${ok?'ok':'fail'}">
      <div class="ver">${ver}</div>
      <div class="val" style="color:var(--${ver})">${p.pred}</div>
      <div style="font-size:10px;color:var(--muted);margin-top:4px;">${ok?'✓ correct':'✗ wrong'}</div>
      <div style="font-size:10px;color:var(--muted);margin-top:2px;word-break:break-all;">${p.raw}</div>
    </div>`;
  }).join('');
  document.getElementById('modal').classList.add('open');
}
function closeModal() { document.getElementById('modal').classList.remove('open'); }
document.getElementById('modal').addEventListener('click', e => { if(e.target===document.getElementById('modal')) closeModal(); });

['filter-version','filter-label','filter-source','filter-result'].forEach(id =>
  document.getElementById(id).addEventListener('change', applyFilters));
document.getElementById('search').addEventListener('input', applyFilters);

init();
</script>
</body>
</html>"""


@app.route("/eval")
def eval_explorer():
    return render_template_string(_EVAL_HTML)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()
    print(f"[saroku dashboard] http://localhost:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)
