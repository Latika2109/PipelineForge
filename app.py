"""FastAPI server for PipelineForge — serves REST API + interactive web UI."""
from __future__ import annotations
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn

from env import PipelineForgeEnv
from models import Action

app = FastAPI(title="PipelineForge", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Single global environment (suitable for local testing; use sessions for multi-user)
_env = PipelineForgeEnv()


class ResetRequest(BaseModel):
    task_id: str = "easy"
    seed: int = 42


# ─────────────────────────────────────────────────────────────────────────────
# REST API
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "version": "1.0.0"}


@app.post("/reset")
def reset(req: Optional[ResetRequest] = None):
    if req is None:
        req = ResetRequest()
    try:
        obs = _env.reset(req.task_id, seed=req.seed)
        return obs.model_dump()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step")
def step(action: Action):
    try:
        obs, reward, done, info = _env.step(action)
        return {**obs.model_dump(), "info": info}
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/state")
def state():
    return _env.get_state()


# ─────────────────────────────────────────────────────────────────────────────
# Web UI
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def ui():
    return HTMLResponse(content=_HTML)


_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1.0"/>
<title>PipelineForge — CI/CD Optimizer</title>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet"/>
<style>
*{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#0a0c10;--surface:#12151c;--surface2:#1a1e28;--surface3:#222736;
  --border:#2a2f3f;--text:#e2e8f0;--text2:#94a3b8;--text3:#64748b;
  --indigo:#6366f1;--indigo2:#818cf8;--green:#22c55e;--red:#ef4444;
  --yellow:#eab308;--blue:#3b82f6;--orange:#f97316;--purple:#a855f7;
  --r:8px;
}
body{font-family:'Inter',sans-serif;background:var(--bg);color:var(--text);min-height:100vh;display:flex;flex-direction:column}
/* HEADER */
header{padding:14px 24px;background:var(--surface);border-bottom:1px solid var(--border);display:flex;align-items:center;gap:16px;flex-wrap:wrap}
.logo{font-size:1.2rem;font-weight:700;letter-spacing:-0.5px}
.logo span{color:var(--indigo2)}
.badge{font-size:.7rem;font-family:'JetBrains Mono',monospace;background:var(--surface3);border:1px solid var(--border);padding:3px 8px;border-radius:20px;color:var(--text2)}
.header-stats{margin-left:auto;display:flex;gap:16px;align-items:center;flex-wrap:wrap}
.stat{text-align:center}
.stat-val{font-size:1.1rem;font-weight:600;font-family:'JetBrains Mono',monospace}
.stat-label{font-size:.65rem;color:var(--text3);text-transform:uppercase;letter-spacing:.5px}
.stat-val.green{color:var(--green)} .stat-val.red{color:var(--red)} .stat-val.yellow{color:var(--yellow)} .stat-val.blue{color:var(--blue)}
/* LAYOUT */
.layout{display:grid;grid-template-columns:260px 1fr 300px;flex:1;overflow:hidden;height:calc(100vh - 57px)}
/* SIDEBAR */
.sidebar{background:var(--surface);border-right:1px solid var(--border);padding:16px;overflow-y:auto;display:flex;flex-direction:column;gap:12px}
.section-title{font-size:.7rem;font-weight:600;text-transform:uppercase;letter-spacing:.8px;color:var(--text3);margin-bottom:6px}
.task-btn{width:100%;padding:10px 14px;border-radius:var(--r);border:1px solid var(--border);background:var(--surface2);color:var(--text);font-family:'Inter',sans-serif;font-size:.85rem;cursor:pointer;text-align:left;transition:all .15s;line-height:1.4}
.task-btn:hover{border-color:var(--indigo);background:var(--surface3)}
.task-btn.active{border-color:var(--indigo);background:rgba(99,102,241,.12);color:var(--indigo2)}
.task-btn small{display:block;color:var(--text3);font-size:.72rem;margin-top:2px}
.seed-row{display:flex;gap:8px;align-items:center}
.seed-row input{flex:1;background:var(--surface2);border:1px solid var(--border);border-radius:6px;color:var(--text);padding:6px 10px;font-family:'JetBrains Mono',monospace;font-size:.8rem;outline:none}
.seed-row input:focus{border-color:var(--indigo)}
.start-btn{width:100%;padding:10px;background:var(--indigo);color:#fff;border:none;border-radius:var(--r);font-family:'Inter',sans-serif;font-size:.9rem;font-weight:600;cursor:pointer;transition:all .15s}
.start-btn:hover{background:var(--indigo2)}
.task-desc{background:var(--surface2);border-radius:var(--r);padding:10px;font-size:.78rem;color:var(--text2);line-height:1.5}
/* MAIN — DAG */
.main{overflow-y:auto;padding:16px;display:flex;flex-direction:column;gap:12px}
.dag-scroll{overflow-x:auto;padding-bottom:8px}
.dag{display:flex;gap:16px;align-items:flex-start;min-width:max-content}
.dag-column{display:flex;flex-direction:column;gap:8px}
.dag-col-label{font-size:.65rem;text-transform:uppercase;color:var(--text3);text-align:center;margin-bottom:4px;letter-spacing:.5px}
/* Stage cards */
.stage-card{width:150px;border-radius:var(--r);border:2px solid var(--border);background:var(--surface2);padding:10px;cursor:pointer;transition:all .2s;position:relative;user-select:none}
.stage-card:hover{border-color:var(--indigo);transform:translateY(-1px)}
.stage-card.selected{border-color:var(--indigo);background:rgba(99,102,241,.12);box-shadow:0 0 0 3px rgba(99,102,241,.2)}
.stage-card.passed{border-color:var(--green);background:rgba(34,197,94,.08)}
.stage-card.failed{border-color:var(--red);background:rgba(239,68,68,.08);animation:pulse-red .6s}
.stage-card.skipped{border-color:var(--yellow);background:rgba(234,179,8,.08);opacity:.7}
.stage-card.running{border-color:var(--blue);background:rgba(59,130,246,.08);animation:pulse-blue 1s infinite}
.stage-name{font-size:.78rem;font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.stage-meta{font-size:.67rem;color:var(--text3);margin-top:3px}
.stage-status{position:absolute;top:7px;right:6px;font-size:.6rem;font-family:'JetBrains Mono',monospace;font-weight:600;color:var(--text3)}
.critical-dot{display:inline-block;width:5px;height:5px;border-radius:50%;background:var(--orange);margin-left:4px;vertical-align:middle}
.deps-badge{font-size:.6rem;color:var(--text3);margin-top:3px;line-height:1.3}
@keyframes pulse-red{0%,100%{box-shadow:0 0 0 0 rgba(239,68,68,0)}50%{box-shadow:0 0 0 6px rgba(239,68,68,.2)}}
@keyframes pulse-blue{0%,100%{box-shadow:0 0 0 0 rgba(59,130,246,0)}50%{box-shadow:0 0 0 6px rgba(59,130,246,.15)}}
/* Legend */
.legend{display:flex;gap:12px;flex-wrap:wrap}
.leg-item{display:flex;align-items:center;gap:5px;font-size:.72rem;color:var(--text2)}
.leg-dot{width:9px;height:9px;border-radius:2px;border:1.5px solid}
/* RIGHT PANEL */
.right{background:var(--surface);border-left:1px solid var(--border);display:flex;flex-direction:column;overflow:hidden}
.action-panel{padding:14px;border-bottom:1px solid var(--border)}
.selected-stages{min-height:34px;background:var(--surface2);border-radius:6px;padding:6px 10px;font-size:.78rem;color:var(--text2);font-family:'JetBrains Mono',monospace;margin-bottom:10px;word-break:break-all}
.action-grid{display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-bottom:8px}
.act-btn{padding:8px 6px;border-radius:6px;border:1px solid var(--border);background:var(--surface2);color:var(--text);font-size:.78rem;font-weight:500;cursor:pointer;transition:all .15s;font-family:'Inter',sans-serif}
.act-btn:hover:not(:disabled){border-color:var(--indigo);background:var(--surface3);color:var(--indigo2)}
.act-btn:disabled{opacity:.35;cursor:not-allowed}
.act-btn.danger:hover:not(:disabled){border-color:var(--red);color:var(--red)}
.act-btn.warn:hover:not(:disabled){border-color:var(--yellow);color:var(--yellow)}
.reason-input{width:100%;background:var(--surface2);border:1px solid var(--border);border-radius:6px;color:var(--text);padding:6px 10px;font-size:.78rem;font-family:'Inter',sans-serif;outline:none;resize:none}
.reason-input:focus{border-color:var(--indigo)}
.retry-row{display:flex;align-items:center;gap:8px;margin-top:6px;font-size:.78rem;color:var(--text2)}
.retry-row select{background:var(--surface2);border:1px solid var(--border);color:var(--text);padding:4px 8px;border-radius:6px;font-size:.78rem;outline:none}
/* Last result */
.last-result{padding:10px 14px;border-bottom:1px solid var(--border);font-size:.75rem;font-family:'JetBrains Mono',monospace;color:var(--text2);line-height:1.6;min-height:60px;max-height:100px;overflow-y:auto}
/* Log */
.log-header{padding:10px 14px 6px;font-size:.7rem;font-weight:600;text-transform:uppercase;letter-spacing:.8px;color:var(--text3);display:flex;align-items:center;justify-content:space-between}
.log-clear{font-size:.7rem;background:none;border:none;color:var(--text3);cursor:pointer;font-family:'Inter',sans-serif}
.log-clear:hover{color:var(--text)}
.log{flex:1;overflow-y:auto;padding:0 14px 14px;display:flex;flex-direction:column;gap:5px}
.log-entry{background:var(--surface2);border-radius:6px;padding:7px 9px;font-size:.72rem;line-height:1.5;border-left:3px solid var(--border)}
.log-entry.reward-pos{border-color:var(--green)}
.log-entry.reward-neg{border-color:var(--red)}
.log-entry.reward-zero{border-color:var(--border)}
.log-step{font-family:'JetBrains Mono',monospace;color:var(--text3);font-size:.65rem}
.log-msg{color:var(--text2)}
.log-reward{font-weight:600;font-family:'JetBrains Mono',monospace}
.log-reward.pos{color:var(--green)} .log-reward.neg{color:var(--red)} .log-reward.zero{color:var(--text3)}
/* Score modal */
.score-banner{display:none;padding:14px;background:linear-gradient(135deg,rgba(99,102,241,.2),rgba(168,85,247,.15));border:1px solid rgba(99,102,241,.3);border-radius:var(--r);margin:16px;text-align:center}
.score-banner.show{display:block}
.score-big{font-size:2.5rem;font-weight:700;font-family:'JetBrains Mono',monospace;color:var(--indigo2)}
.score-label{font-size:.8rem;color:var(--text2);margin-top:4px}
/* Empty state */
.empty{text-align:center;padding:40px 20px;color:var(--text3)}
.empty h3{font-size:1rem;margin-bottom:8px;color:var(--text2)}
.empty p{font-size:.8rem;line-height:1.6}
</style>
</head>
<body>
<header>
  <div class="logo">Pipeline<span>Forge</span></div>
  <div class="badge">CI/CD Optimizer RL Environment</div>
  <div class="header-stats">
    <div class="stat"><div class="stat-val blue" id="h-steps">—</div><div class="stat-label">Steps Left</div></div>
    <div class="stat"><div class="stat-val" id="h-time">0s</div><div class="stat-label">Elapsed</div></div>
    <div class="stat"><div class="stat-val yellow" id="h-reward">0.00</div><div class="stat-label">Reward</div></div>
  </div>
</header>

<div class="layout">
  <!-- Sidebar -->
  <div class="sidebar">
    <div>
      <div class="section-title">Select Task</div>
      <div style="display:flex;flex-direction:column;gap:6px">
        <button class="task-btn" id="btn-easy" onclick="selectTask('easy')">
          Easy — Linear Pipeline<small>6 stages · Budget: 10 steps</small>
        </button>
        <button class="task-btn" id="btn-medium" onclick="selectTask('medium')">
          Medium — Parallel DAG<small>12 stages · Budget: 18 steps</small>
        </button>
        <button class="task-btn" id="btn-hard" onclick="selectTask('hard')">
          Hard — Enterprise DAG<small>20 stages · Budget: 25 steps</small>
        </button>
        <button class="task-btn" id="btn-expert" onclick="selectTask('expert')">
          Expert — Hotfix Pipeline<small>15 stages · Budget: 12 steps</small>
        </button>
      </div>
    </div>

    <div>
      <div class="section-title">Seed</div>
      <div class="seed-row">
        <input type="number" id="seed-input" value="42" min="0"/>
      </div>
    </div>

    <button class="start-btn" onclick="startEpisode()">Start Episode</button>

    <div>
      <div class="section-title">Task Info</div>
      <div class="task-desc" id="task-desc">Select a task and press Start.</div>
    </div>

    <div>
      <div class="section-title">Legend</div>
      <div class="legend">
        <div class="leg-item"><div class="leg-dot" style="border-color:#6b7280;background:rgba(107,114,128,.2)"></div>Pending</div>
        <div class="leg-item"><div class="leg-dot" style="border-color:#3b82f6;background:rgba(59,130,246,.2)"></div>Running</div>
        <div class="leg-item"><div class="leg-dot" style="border-color:#22c55e;background:rgba(34,197,94,.2)"></div>Passed</div>
        <div class="leg-item"><div class="leg-dot" style="border-color:#ef4444;background:rgba(239,68,68,.2)"></div>Failed</div>
        <div class="leg-item"><div class="leg-dot" style="border-color:#eab308;background:rgba(234,179,8,.2)"></div>Skipped</div>
        <div class="leg-item"><span style="color:#f97316;font-size:.8rem;font-weight:700">!</span> Critical</div>
      </div>
    </div>
  </div>

  <!-- Main DAG area -->
  <div class="main" id="main">
    <div class="empty">
      <h3>Welcome to PipelineForge</h3>
      <p>Select a task difficulty on the left,<br/>then press <strong>Start Episode</strong> to begin.<br/><br/>Click stage cards to select them,<br/>then use the action panel on the right.</p>
    </div>
  </div>

  <!-- Right panel -->
  <div class="right">
    <div class="action-panel">
      <div class="section-title" style="margin-bottom:8px">Actions</div>
      <div class="selected-stages" id="sel-display">No stage selected</div>
      <div class="action-grid">
        <button class="act-btn" id="btn-run" onclick="doAction('run')" disabled>Run</button>
        <button class="act-btn" id="btn-parallel" onclick="doAction('run_parallel')" disabled>Parallel</button>
        <button class="act-btn warn" id="btn-skip" onclick="doAction('skip')" disabled>Skip</button>
        <button class="act-btn" id="btn-inspect" onclick="doAction('inspect')" disabled>Inspect</button>
        <button class="act-btn" id="btn-retry" onclick="doAction('retry')" disabled>Retry</button>
        <button class="act-btn danger" id="btn-abort" onclick="doAction('abort')" disabled>Abort</button>
      </div>
      <div class="retry-row">
        <label>Retries:</label>
        <select id="retry-count"><option value="1">1</option><option value="2">2</option><option value="3">3</option></select>
      </div>
      <textarea class="reason-input" id="reason-input" rows="2" placeholder="Reason (optional)..." style="margin-top:8px"></textarea>
    </div>
    <div class="last-result" id="last-result" style="color:var(--text3);font-style:italic">No actions yet.</div>
    <div class="log-header">
      Action Log
      <button class="log-clear" onclick="clearLog()">Clear</button>
    </div>
    <div class="log" id="log"></div>
  </div>
</div>

<script>
let selectedStages = [];
let currentObs = null;
let selectedTask = null;
let episodeStarted = false;

const TASK_DESCS = {
  easy: "6-stage linear Python pipeline. unit-tests sometimes flaps (retry it!). integration-tests has a real failure this run — identify it and abort.",
  medium: "12-stage microservices DAG. Lint and build stages can run in parallel. lint-service-b is a heavy flapper. dependency-audit has a real CVE — real failure!",
  hard: "20-stage enterprise pipeline. 3 stages look like real failures but are flaps. 2 are genuine failures. There is a hidden cascade dependency to discover!",
  expert: "15-stage HOTFIX pipeline. Only 12 steps budget. 3 real failures, 4 flapping stages, 2 hidden dependencies. You MUST parallelize and skip non-critical stages to fit the budget."
};

function selectTask(id) {
  selectedTask = id;
  ['easy','medium','hard','expert'].forEach(t => {
    document.getElementById('btn-'+t).classList.toggle('active', t===id);
  });
  document.getElementById('task-desc').textContent = TASK_DESCS[id] || '';
}

async function startEpisode() {
  if (!selectedTask) { alert('Select a task first!'); return; }
  const seed = parseInt(document.getElementById('seed-input').value) || 42;
  selectedStages = [];
  episodeStarted = true;

  const res = await fetch('/reset', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({task_id: selectedTask, seed})
  });
  currentObs = await res.json();
  renderAll();
  document.getElementById('last-result').textContent = `Episode started. Pipeline: ${currentObs.pipeline_id}`;
  document.getElementById('log').innerHTML = '';
  updateActionButtons();
}

function renderAll() {
  if (!currentObs) return;
  renderStats();
  renderDAG();
  renderSelectionDisplay();
  if (currentObs.score !== null && currentObs.score !== undefined && currentObs.done) {
    showScore(currentObs.score);
  }
}

function renderStats() {
  const o = currentObs;
  document.getElementById('h-steps').textContent = o.steps_remaining ?? '—';
  document.getElementById('h-time').textContent = o.elapsed_time_seconds + 's';
  document.getElementById('h-reward').textContent = (o.cumulative_reward || 0).toFixed(3);
}

function renderDAG() {
  const o = currentObs;
  const stages = o.stages;
  
  // Group by level
  const byLevel = {};
  stages.forEach(s => {
    const lvl = s.level || 0;
    if (!byLevel[lvl]) byLevel[lvl] = [];
    byLevel[lvl].push(s);
  });
  
  const maxLevel = Math.max(...Object.keys(byLevel).map(Number));
  
  let html = `<div class="dag-scroll"><div class="dag">`;
  for (let l = 0; l <= maxLevel; l++) {
    const cols = byLevel[l] || [];
    html += `<div class="dag-column"><div class="dag-col-label">Wave ${l + 1}</div>`;
    cols.forEach(s => {
      const status = getStatus(s, o);
      const isSelected = selectedStages.includes(s.stage_id);
      const emoji = statusEmoji(status);
      const critDot = s.is_critical ? '<span class="critical-dot"></span>' : '';
      const deps = s.dependencies.length > 0 ? `<div class="deps-badge">← ${s.dependencies.join(', ')}</div>` : '';
      html += `
        <div class="stage-card ${status} ${isSelected ? 'selected' : ''}"
             onclick="toggleStage('${s.stage_id}', '${status}')"
             title="${s.name} | ~${s.estimated_duration}s | ${s.is_critical ? 'CRITICAL' : 'non-critical'}">
          <div class="stage-name">${s.name}${critDot}</div>
          <div class="stage-meta">~${s.estimated_duration}s</div>
          ${deps}
          <div class="stage-status">${emoji}</div>
        </div>`;
    });
    html += `</div>`;
  }
  html += `</div></div>`;
  
  // Score banner
  if (currentObs.done && currentObs.score !== null && currentObs.score !== undefined) {
    const sc = currentObs.score;
    const color = sc >= 0.75 ? '#22c55e' : sc >= 0.5 ? '#eab308' : '#ef4444';
    html += `<div class="score-banner show">
      <div class="score-big" style="color:${color}">${(sc * 100).toFixed(1)}%</div>
      <div class="score-label">Final Score · ${currentObs.message}</div>
    </div>`;
  }
  
  document.getElementById('main').innerHTML = html;
}

function getStatus(stage, obs) {
  if (obs.completed_stages.includes(stage.stage_id)) return 'passed';
  if (obs.failed_stages.includes(stage.stage_id)) return 'failed';
  if (obs.skipped_stages.includes(stage.stage_id)) return 'skipped';
  if (obs.running_stages && obs.running_stages.includes(stage.stage_id)) return 'running';
  return '';
}

function statusEmoji(status) {
  return {passed:'[OK]', failed:'[FAIL]', skipped:'[SKIP]', running:'[RUN]'}[status] || '';
}

function toggleStage(stageId, status) {
  if (!episodeStarted || (currentObs && currentObs.done)) return;
  const i = selectedStages.indexOf(stageId);
  if (i >= 0) selectedStages.splice(i, 1);
  else selectedStages.push(stageId);
  renderDAG();
  renderSelectionDisplay();
  updateActionButtons();
}

function renderSelectionDisplay() {
  const el = document.getElementById('sel-display');
  el.textContent = selectedStages.length > 0 ? selectedStages.join(', ') : 'No stage selected';
}

function updateActionButtons() {
  const active = episodeStarted && !(currentObs && currentObs.done);
  const n = selectedStages.length;
  document.getElementById('btn-run').disabled = !active || n !== 1;
  document.getElementById('btn-parallel').disabled = !active || n < 2;
  document.getElementById('btn-skip').disabled = !active || n !== 1;
  document.getElementById('btn-inspect').disabled = !active || n !== 1;
  document.getElementById('btn-retry').disabled = !active || n !== 1;
  document.getElementById('btn-abort').disabled = !active;
}

function showScore(score) {/* inline in renderDAG */}

async function doAction(type) {
  if (!episodeStarted) return;
  const reason = document.getElementById('reason-input').value || type;
  const retryCount = parseInt(document.getElementById('retry-count').value) || 1;

  let body = { action_type: type, reason };

  if (type === 'run' || type === 'skip' || type === 'inspect' || type === 'retry') {
    if (selectedStages.length !== 1) { alert('Select exactly 1 stage.'); return; }
    body.stage_id = selectedStages[0];
  }
  if (type === 'run_parallel') {
    if (selectedStages.length < 2) { alert('Select 2+ stages for parallel run.'); return; }
    body.parallel_stages = [...selectedStages];
  }
  if (type === 'retry') {
    body.retry_count = retryCount;
  }
  if (type === 'abort') {
    if (!confirm('Abort the pipeline?')) return;
  }

  const res = await fetch('/step', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify(body)
  });
  const data = await res.json();
  currentObs = data;

  // Update last result
  const lr = document.getElementById('last-result');
  lr.textContent = data.message || '(no message)';
  if (data.last_stage_result) {
    const r = data.last_stage_result;
    lr.textContent += ` | exit=${r.exit_code} | ${r.duration_seconds}s | ${r.stdout_tail}`;
  }

  // Add to log
  addLog(data.steps_taken, type, selectedStages, data.reward, data.message);

  // Clear selection
  selectedStages = [];
  renderAll();
  updateActionButtons();
  document.getElementById('reason-input').value = '';

  if (data.done) {
    document.getElementById('btn-abort').disabled = true;
    ['btn-run','btn-parallel','btn-skip','btn-inspect','btn-retry'].forEach(id => {
      document.getElementById(id).disabled = true;
    });
  }
}

function addLog(step, type, stages, reward, message) {
  const log = document.getElementById('log');
  const cls = reward > 0 ? 'reward-pos' : reward < 0 ? 'reward-neg' : 'reward-zero';
  const rwCls = reward > 0 ? 'pos' : reward < 0 ? 'neg' : 'zero';
  const entry = document.createElement('div');
  entry.className = `log-entry ${cls}`;
  entry.innerHTML = `
    <div class="log-step">Step ${step} · ${type.toUpperCase()}</div>
    <div class="log-msg">${message || stages.join(', ')}</div>
    <div class="log-reward ${rwCls}">${reward >= 0 ? '+' : ''}${reward.toFixed(3)}</div>`;
  log.prepend(entry);
}

function clearLog() {
  document.getElementById('log').innerHTML = '';
}
</script>
</body>
</html>"""

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
