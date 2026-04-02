"""
Macro Signal Engine: FastAPI Server
======================================
WebSocket (/ws) is the primary transport for persistent sessions.
HTTP endpoints (/reset, /step, /state) are available for debugging only.
Port is read from os.environ["PORT"], default 7860 for HF Spaces.
"""

from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from ..models import MacroSignalAction, MacroSignalObservation, MacroSignalState, StepResult
from .environment import MacroSignalEnvironment, _load_scenario_bank

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Lifespan: validate scenario bank at startup


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Macro Signal Engine...")
    try:
        bank = _load_scenario_bank()
        logger.info("Scenario bank validated: %d scenarios loaded", len(bank))
    except Exception as e:
        logger.error("FATAL: Scenario bank validation failed: %s", e)
        raise
    yield
    logger.info("Macro Signal Engine shutting down.")


app = FastAPI(
    title="Macro Signal Engine",
    description=(
        "OpenEnv environment: LLM agent plays macro quant analyst, "
        "managing a 4-asset portfolio in response to typed financial signal events."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# Health check


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "healthy", "version": "1.0.0", "environment": "macro-signal-env"}


# WebSocket endpoint (primary transport, used by MacroSignalEnv client)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """
    Persistent WebSocket session. Each connection gets its own environment instance.
    Message protocol:
      Client → Server: {"type": "reset", "task_type": "...", "scenario_id": "..."}
                       {"type": "step", "action": {...}}
                       {"type": "state"}
      Server → Client: {"type": "observation", "data": {...}}
                       {"type": "state", "data": {...}}
                       {"type": "error", "message": "..."}
    """
    await websocket.accept()
    env = MacroSignalEnvironment()
    logger.info("WebSocket session opened")

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError as e:
                await websocket.send_text(
                    json.dumps({"type": "error", "message": f"Invalid JSON: {e}"})
                )
                continue

            msg_type = msg.get("type")

            if msg_type == "reset":
                try:
                    obs = env.reset(
                        task_type=msg.get("task_type"),
                        scenario_id=msg.get("scenario_id"),
                    )
                    result = StepResult(observation=obs, reward=0.0, done=False, info={})
                    await websocket.send_text(
                        json.dumps({"type": "observation", "data": result.model_dump()})
                    )
                except Exception as e:
                    logger.exception("Error during reset")
                    await websocket.send_text(
                        json.dumps({"type": "error", "message": str(e)})
                    )

            elif msg_type == "step":
                try:
                    action = MacroSignalAction.model_validate(msg.get("action", {}))
                    result = env.step(action)
                    await websocket.send_text(
                        json.dumps({"type": "observation", "data": result.model_dump()})
                    )
                except Exception as e:
                    logger.exception("Error during step")
                    await websocket.send_text(
                        json.dumps({"type": "error", "message": str(e)})
                    )

            elif msg_type == "state":
                try:
                    state = env.state()
                    await websocket.send_text(
                        json.dumps({"type": "state", "data": state.model_dump()})
                    )
                except Exception as e:
                    logger.exception("Error during state")
                    await websocket.send_text(
                        json.dumps({"type": "error", "message": str(e)})
                    )

            else:
                await websocket.send_text(
                    json.dumps({"type": "error", "message": f"Unknown message type: {msg_type}"})
                )

    except WebSocketDisconnect:
        logger.info("WebSocket session closed")
    except Exception as e:
        logger.exception("Unexpected WebSocket error: %s", e)
        try:
            await websocket.send_text(
                json.dumps({"type": "error", "message": f"Server error: {e}"})
            )
        except Exception:
            pass


# HTTP endpoints (debug only, stateless, not for production HF Spaces use)

# Stateless HTTP sessions share a single environment instance, for debug only
_http_env = MacroSignalEnvironment()


@app.post("/reset", response_model=StepResult)
async def http_reset(task_type: Optional[str] = None, scenario_id: Optional[str] = None) -> StepResult:
    """
    HTTP reset endpoint. Stateless, resets the shared debug environment.
    WARNING: Not suitable for concurrent sessions. Use /ws for production.
    """
    obs = _http_env.reset(task_type=task_type, scenario_id=scenario_id)
    return StepResult(observation=obs, reward=0.0, done=False, info={})


@app.post("/step", response_model=StepResult)
async def http_step(action: MacroSignalAction) -> StepResult:
    """
    HTTP step endpoint. Stateless, operates on shared debug environment.
    WARNING: Not suitable for concurrent sessions. Use /ws for production.
    """
    try:
        return _http_env.step(action)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/state", response_model=MacroSignalState)
async def http_state() -> MacroSignalState:
    return _http_env.state()


# Web UI (interactive terminal for demos)


WEB_UI_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Macro Signal Engine | OpenEnv</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { background: #0a0e1a; color: #00ff88; font-family: 'Courier New', monospace; padding: 20px; }
    h1 { color: #00d4ff; font-size: 1.4em; margin-bottom: 10px; }
    .subtitle { color: #888; font-size: 0.85em; margin-bottom: 20px; }
    .terminal { background: #0d1117; border: 1px solid #1e3a5f; border-radius: 4px; padding: 16px; min-height: 400px; max-height: 600px; overflow-y: auto; font-size: 0.85em; line-height: 1.6; }
    .terminal .ts { color: #555; }
    .terminal .signal { color: #ffd700; }
    .terminal .reward { color: #00ff88; }
    .terminal .error { color: #ff4444; }
    .terminal .system { color: #00d4ff; }
    .controls { margin-top: 16px; display: flex; gap: 10px; flex-wrap: wrap; }
    button { background: #1e3a5f; color: #00d4ff; border: 1px solid #00d4ff; padding: 8px 16px; cursor: pointer; border-radius: 4px; font-family: monospace; }
    button:hover { background: #00d4ff; color: #0a0e1a; }
    .stats { margin-top: 16px; display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; }
    .stat { background: #0d1117; border: 1px solid #1e3a5f; padding: 10px; border-radius: 4px; text-align: center; }
    .stat .label { color: #888; font-size: 0.75em; }
    .stat .value { color: #00ff88; font-size: 1.1em; font-weight: bold; }
    select { background: #0d1117; color: #00d4ff; border: 1px solid #00d4ff; padding: 8px; border-radius: 4px; font-family: monospace; }
  </style>
</head>
<body>
  <h1>Macro Signal Engine</h1>
  <p class="subtitle">OpenEnv: LLM Agent as Macro Quant Analyst | Tasks: single_event · regime_shift · causal_chain</p>

  <div class="stats">
    <div class="stat"><div class="label">STEP</div><div class="value" id="stat-step">—</div></div>
    <div class="stat"><div class="label">NAV</div><div class="value" id="stat-nav">—</div></div>
    <div class="stat"><div class="label">REWARD</div><div class="value" id="stat-reward">—</div></div>
    <div class="stat"><div class="label">STATUS</div><div class="value" id="stat-status">IDLE</div></div>
  </div>

  <div class="terminal" id="terminal">
    <span class="system">[MACRO SIGNAL ENGINE] Ready. Select a task and press RESET to begin.</span><br>
  </div>

  <div class="controls">
    <select id="task-select">
      <option value="single_event">single_event (Easy)</option>
      <option value="regime_shift">regime_shift (Medium)</option>
      <option value="causal_chain">causal_chain (Hard)</option>
    </select>
    <button onclick="doReset()">RESET</button>
    <button onclick="doHold()">HOLD</button>
    <button onclick="doRiskOn()">RISK ON (long SPY+GLD)</button>
    <button onclick="doRiskOff()">RISK OFF (short SPY, long TLT)</button>
    <button onclick="doOilLong()">LONG OIL (USO)</button>
  </div>

  <script>
    let ws = null;
    const term = document.getElementById('terminal');

    function log(msg, cls='') {
      const ts = new Date().toLocaleTimeString();
      term.innerHTML += `<span class="ts">[${ts}]</span> <span class="${cls}">${msg}</span><br>`;
      term.scrollTop = term.scrollHeight;
    }

    function connect() {
      const proto = location.protocol === 'https:' ? 'wss' : 'ws';
      ws = new WebSocket(`${proto}://${location.host}/ws`);
      ws.onopen = () => log('WebSocket connected', 'system');
      ws.onclose = () => { log('WebSocket disconnected', 'error'); ws = null; };
      ws.onmessage = (e) => {
        const msg = JSON.parse(e.data);
        if (msg.type === 'observation') {
          const d = msg.data;
          const obs = d.observation;
          document.getElementById('stat-step').textContent = `${obs.step}/${obs.max_steps}`;
          document.getElementById('stat-nav').textContent = obs.portfolio_nav.toFixed(4);
          document.getElementById('stat-reward').textContent = obs.reward.toFixed(4);
          document.getElementById('stat-status').textContent = obs.done ? 'DONE' : 'ACTIVE';

          if (obs.signal_events && obs.signal_events.length > 0) {
            obs.signal_events.forEach(ev => {
              const dir = ev.magnitude > 0 ? 'BULLISH' : 'BEARISH';
              log(`SIGNAL: ${ev.event_type} | ${ev.asset} | magnitude=${ev.magnitude.toFixed(2)} (${dir})`, 'signal');
            });
            // Hint: derive the correct action from signals
            const hints = obs.signal_events.map(ev => {
              if (ev.magnitude > 0) return `→ Consider LONG ${ev.asset}`;
              else return `→ Consider SHORT ${ev.asset}`;
            });
            log(`HINT: ${hints.join(' | ')}`, 'ts');
          }
          if (obs.step_reward > 0) log(`Step reward: +${obs.step_reward.toFixed(4)}`, 'reward');
          if (obs.done) log(`EPISODE DONE: Final reward: ${obs.reward.toFixed(4)}`, 'reward');
        } else if (msg.type === 'error') {
          log(`ERROR: ${msg.message}`, 'error');
        }
      };
    }

    function send(payload) {
      if (!ws || ws.readyState !== 1) { connect(); setTimeout(() => send(payload), 500); return; }
      ws.send(JSON.stringify(payload));
    }

    function doReset() {
      const task = document.getElementById('task-select').value;
      log(`Resetting with task_type=${task}`, 'system');
      send({ type: 'reset', task_type: task });
    }

    function doHold() { send({ type: 'step', action: { trade_instructions: [], reasoning: 'Hold all positions.' } }); }

    function doRiskOn() {
      send({ type: 'step', action: {
        trade_instructions: [
          { asset: 'SPY', target_weight: 0.5, urgency: 'immediate' },
          { asset: 'GLD', target_weight: 0.3, urgency: 'immediate' }
        ],
        reasoning: 'Risk-on: long equity and gold.'
      }});
    }

    function doRiskOff() {
      send({ type: 'step', action: {
        trade_instructions: [
          { asset: 'SPY', target_weight: -0.3, urgency: 'immediate' },
          { asset: 'TLT', target_weight: 0.5, urgency: 'immediate' }
        ],
        reasoning: 'Risk-off: short equity, long bonds.'
      }});
    }

    function doOilLong() {
      send({ type: 'step', action: {
        trade_instructions: [
          { asset: 'USO', target_weight: 0.6, urgency: 'immediate' }
        ],
        reasoning: 'Long oil: energy supply disruption signal.'
      }});
    }

    connect();
  </script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
async def root() -> HTMLResponse:
    if os.environ.get("ENABLE_WEB_INTERFACE", "true").lower() == "true":
        return HTMLResponse(content=WEB_UI_HTML)
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/docs")


@app.get("/web", response_class=HTMLResponse)
async def web_ui() -> HTMLResponse:
    if os.environ.get("ENABLE_WEB_INTERFACE", "true").lower() != "true":
        raise HTTPException(status_code=404, detail="Web interface is disabled.")
    return HTMLResponse(content=WEB_UI_HTML)


# Entry point


def run() -> None:
    port = int(os.environ.get("PORT", 7860))
    host = os.environ.get("HOST", "0.0.0.0")
    workers = int(os.environ.get("WORKERS", 1))
    uvicorn.run("macro_signal.server.app:app", host=host, port=port, workers=workers)


if __name__ == "__main__":
    run()
