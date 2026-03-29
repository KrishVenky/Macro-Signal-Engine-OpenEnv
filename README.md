---
title: Macro Signal Engine
emoji: 📈
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
license: mit
tags:
  - openenv
  - finance
  - macro
  - portfolio-management
  - reinforcement-learning
  - llm-agent
---

# 📈 Macro Signal Engine

**OpenEnv environment** — An LLM agent plays the role of a **macro quantitative analyst** managing a 4-asset portfolio in response to typed financial signal events.

> "Can an AI reason through a geopolitical crisis and anticipate its effect on oil prices three steps before the supply shock arrives?"

[![HF Space](https://img.shields.io/badge/🤗%20Space-KrishVenky%2Fmacro--signal--env-blue)](https://huggingface.co/spaces/KrishVenky/macro-signal-env)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Why This Environment Exists

Most finance RL environments test "buy low, sell high." This environment tests something fundamentally harder:

**Multi-step causal reasoning under uncertainty.**

Real macro investing requires connecting events that are temporally separated:
- A geopolitical shock in Step 1 implies a supply disruption in Step 4
- That supply disruption implies an inflation print in Step 7
- The optimal hedge must be entered in Step 2, not Step 7

An agent that treats each observation independently will consistently underperform. This is the exact failure mode of most LLMs used naively as RL agents — and this environment is designed to measure and improve it.

---

## Assets

| Ticker | Description | Macro Role |
|--------|-------------|------------|
| **SPY** | S&P 500 ETF | Broad equity / risk-on |
| **GLD** | Gold ETF | Safe haven / inflation hedge |
| **USO** | Oil ETF | Energy / geopolitical hedge |
| **TLT** | 20+ Year Treasury ETF | Duration / deflation hedge |

---

## Observation Space

```python
class MacroSignalObservation(BaseModel):
    step: int                          # Current step (0 = post-reset)
    max_steps: int                     # Episode length
    task_type: str                     # "single_event" | "regime_shift" | "causal_chain"
    scenario_id: str                   # Which scenario is running
    signal_events: List[SignalEvent]   # Macroeconomic signals this step (may be empty)
    portfolio: List[PortfolioPosition] # Non-zero positions
    cash_weight: float                 # Fraction of NAV in cash
    portfolio_nav: float               # Current total NAV
    benchmark_return: float            # Cumulative benchmark return
    step_reward: float                 # Partial reward this step [0.0, 1.0]
    cumulative_reward: float           # Running total [0.0, 1.0]
    done: bool                         # Episode complete?
    reward: float                      # Episode reward when done=True [0.0, 1.0]
    info: Dict[str, Any]               # Scenario description, diagnostics
```

### SignalEvent

```python
class SignalEvent(BaseModel):
    event_type: str   # equity_shock | commodity_shock | rates_move | geopolitical | inflation_print
    asset: str        # SPY | GLD | USO | TLT
    magnitude: float  # [-1.0, +1.0]: positive = bullish, negative = bearish
    step: int         # Step when generated
```

---

## Action Space

```python
class MacroSignalAction(BaseModel):
    trade_instructions: List[TradeInstruction]  # Empty = hold all
    reasoning: str                              # Agent's causal reasoning

class TradeInstruction(BaseModel):
    asset: str          # SPY | GLD | USO | TLT
    target_weight: float  # [-1.0, +1.0]: negative = short
    urgency: str        # immediate | next_step | hold
```

**Constraint:** `sum(abs(target_weight)) <= 1.0` (no leverage allowed)

---

## Tasks

### Task 1: `single_event` (Easy)

**Setup:** 3-step episode. One clear macroeconomic signal at step 1.

**Objective:** Take the correct directional position within 3 steps.

**Example:** `commodity_shock | USO | magnitude=+0.85` → agent should go long USO

**Reward:**
```
step_reward = correct_direction × (1.0 / step)
episode_reward = 0.6 × best_step_reward + 0.4 × mean_step_rewards
```

**Expected scores:** Random ~0.10 | GPT-4o ~0.72

---

### Task 2: `regime_shift` (Medium)

**Setup:** 6-step episode. Multiple regime-defining signals across steps 1–4.

**Objective:** Rebalance portfolio to track a benchmark (60/40 SPY/TLT or scenario-defined) through a coherent market regime.

**Example:** Bull equity regime — rising earnings + falling rates → maintain SPY/TLT exposure

**Reward:**
```
step_reward = 0.5×directional + 0.3×pnl_vs_benchmark + 0.2×rebalance_quality
episode_reward = 0.4×terminal_pnl_ratio + 0.6×mean_step_rewards
```

**Expected scores:** Random ~0.25 | GPT-4o ~0.55

---

### Task 3: `causal_chain` (Hard)

**Setup:** 10-step episode. Three causally-linked events spaced 3 steps apart.

**Objective:** Anticipate consequences before they materialize. The `timing_bonus` rewards positions entered BEFORE the causal event arrives — this is what frontier models consistently fail.

**Example:**
```
Step 1:  geopolitical signal → conflict in oil region
Step 4:  commodity_shock → supply disruption (consequence of Step 1)
Step 7:  inflation_print → CPI spike (consequence of Step 4)

Optimal: Enter long USO at Step 2, long GLD at Step 2, short TLT at Step 3
Naive:   Enter at Step 4/7 — correct but too late for timing_bonus
```

**Reward:**
```
terminal_reward = 0.4×directional_accuracy + 0.4×timing_bonus + 0.2×cost_efficiency
episode_reward  = 0.3×mean_step_rewards + 0.7×terminal_reward
```

**Expected scores:** Random ~0.12 | GPT-4o (no memory) ~0.38 | GPT-4o (explicit causal reasoning) ~0.61

---

## Reward Design

| Component | Formula | Purpose |
|-----------|---------|---------|
| Directional accuracy | `correct_positions / total_signals` | Correctness |
| Speed bonus | `1.0 / step` | Rewards fast response |
| Timing bonus | `1.0 / 0.5 / 0.1` (before / at / after consequence) | Causal reasoning |
| Cost efficiency | `1 - (transaction_costs / max_allowed)` | Penalizes churning |
| Idle penalty | `-0.02` when holding during high-signal step | Penalizes passivity |

All rewards are in `[0.0, 1.0]`. Transaction cost: 10bps per unit of weight changed.

---

## Quick Start

### Connect to Live Space

```python
import asyncio
from src.envs.macro_signal.client import MacroSignalEnv
from src.envs.macro_signal.models import MacroSignalAction, TradeInstruction

async def main():
    async with MacroSignalEnv(base_url="https://krishvenky-macro-signal-env.hf.space") as env:
        result = await env.reset(task_type="single_event")
        print(result.observation.signal_events)

        action = MacroSignalAction(
            trade_instructions=[TradeInstruction(asset="USO", target_weight=0.6)],
            reasoning="Oil supply shock — go long USO"
        )
        result = await env.step(action)
        print(f"Reward: {result.reward}")

asyncio.run(main())
```

### Run Baseline Inference

```bash
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o"
export HF_TOKEN="your-key"
python inference.py
```

### Local Development

```bash
# Install
pip install -e ".[dev]"

# Run server locally
uvicorn macro_signal.server.app:app --port 7860 --reload

# Run tests
pytest tests/

# Validate pre-submission
jupyter notebook pre_submission_test.ipynb
```

### Docker

```bash
docker build -t macro-signal-env .
docker run -p 7860:7860 macro-signal-env
curl http://localhost:7860/health
# {"status":"healthy","version":"1.0.0","environment":"macro-signal-env"}
```

---

## Baseline Scores

Measured with `gpt-4o` via `inference.py`:

| Task | Difficulty | Baseline Score |
|------|-----------|----------------|
| `single_event` | Easy | 0.72 |
| `regime_shift` | Medium | 0.55 |
| `causal_chain` | Hard | 0.38 |
| **Mean** | | **0.55** |

---

## Project Structure

```
macro-signal-env/
├── src/envs/macro_signal/
│   ├── models.py          ← Pydantic typed contracts
│   ├── client.py          ← WebSocket client
│   └── server/
│       ├── environment.py ← Core logic + graders
│       └── app.py         ← FastAPI server + Web UI
├── data/
│   └── scenarios.json     ← 10 seeded scenarios (5 easy, 3 medium, 2 hard)
├── openenv.yaml           ← OpenEnv spec metadata
├── Dockerfile             ← HF Spaces deployment
├── inference.py           ← Baseline agent script
├── ARCHITECTURE.md        ← Reward design documentation
└── tests/
    └── test_environment.py
```

---

## Interactive Web UI

Visit the `/web` endpoint for a terminal-style interface:
`https://krishvenky-macro-signal-env.hf.space/web`

---

## Author

**KrishVenky** — Built for the Meta x Scaler OpenEnv Competition
