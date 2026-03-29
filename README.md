---
title: Macro Signal Engine
emoji: chart_with_upwards_trend
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

# Macro Signal Engine

An [OpenEnv](https://github.com/meta-pytorch/OpenEnv) environment where an LLM agent plays the role of a macro quantitative analyst. The agent manages a 4-asset portfolio in response to typed financial signal events, and must reason causally across time to score well.

[![HF Space](https://img.shields.io/badge/Space-KrishVenky%2Fmacro--signal--env-blue)](https://huggingface.co/spaces/KrishVenky/macro-signal-env)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> Can an AI reason through a geopolitical crisis and anticipate its effect on oil prices three steps before the supply shock actually arrives?

Most finance RL environments test "buy low, sell high." This one tests something harder: multi-step causal reasoning under uncertainty. A geopolitical shock at step 1 implies a supply disruption at step 4, which implies an inflation print at step 7. The optimal hedge needs to be in place at step 2. An agent that treats each observation independently will consistently underperform — that failure mode is exactly what this environment is designed to measure.

## For Judges

The fastest way to see the environment working is the live web terminal:

```
https://krishvenky-macro-signal-env.hf.space/web
```

Select a task, press Reset, and follow the hints. The `/docs` endpoint has the full OpenAPI spec. To run the baseline agent against all three tasks, set your API credentials and run `python inference.py` from the repo root.

## Assets

| Ticker | Description | Macro Role |
|--------|-------------|------------|
| SPY | S&P 500 ETF | Broad equity exposure |
| GLD | Gold ETF | Safe haven and inflation hedge |
| USO | Oil ETF | Energy and geopolitical hedge |
| TLT | 20+ Year Treasury ETF | Duration and deflation hedge |

## Observation Space

```python
class MacroSignalObservation(BaseModel):
    step: int                          # Current step (0 = post-reset)
    max_steps: int                     # Episode length
    task_type: str                     # single_event | regime_shift | causal_chain
    scenario_id: str                   # Active scenario identifier
    signal_events: List[SignalEvent]   # Macroeconomic signals this step (may be empty)
    portfolio: List[PortfolioPosition] # Current non-zero positions
    cash_weight: float                 # Fraction of NAV held as cash
    portfolio_nav: float               # Current total NAV
    benchmark_return: float            # Cumulative benchmark return for comparison
    step_reward: float                 # Partial reward earned this step [0.0, 1.0]
    cumulative_reward: float           # Running episode total [0.0, 1.0]
    done: bool                         # Whether the episode has ended
    reward: float                      # Final episode reward, populated when done=True
    info: Dict[str, Any]               # Scenario description and diagnostics
```

### SignalEvent

```python
class SignalEvent(BaseModel):
    event_type: str   # equity_shock | commodity_shock | rates_move | geopolitical | inflation_print
    asset: str        # SPY | GLD | USO | TLT
    magnitude: float  # Signed strength: positive = bullish, negative = bearish [-1.0, 1.0]
    step: int
```

## Action Space

```python
class MacroSignalAction(BaseModel):
    trade_instructions: List[TradeInstruction]  # Empty list = hold all positions
    reasoning: str                              # Agent's causal reasoning (used by hard task grader)

class TradeInstruction(BaseModel):
    asset: str            # SPY | GLD | USO | TLT
    target_weight: float  # Desired portfolio weight [-1.0, 1.0], negative = short
    urgency: str          # immediate | next_step | hold
```

Constraint: `sum(abs(target_weight))` across all instructions must be at or below 1.0. No leverage.

## Tasks

### single_event (Easy)

A 3-step episode with one clear macroeconomic signal at step 1. The agent needs to take the correct directional position before the episode ends. Speed matters — acting at step 1 scores higher than acting at step 3.

Example: `commodity_shock | USO | magnitude=+0.85` means oil supply shock, go long USO.

```
step_reward    = correct_direction * (1.0 / step)
episode_reward = 0.6 * best_step_reward + 0.4 * mean_step_rewards
```

Expected scores: random ~0.10, GPT-4o ~0.72

### regime_shift (Medium)

A 6-step episode with multiple regime-defining signals across steps 1 to 4. The agent needs to rebalance its portfolio to track a benchmark through a coherent market regime (bull equity, rising rates, or commodity supercycle).

```
step_reward    = 0.5 * directional + 0.3 * pnl_vs_benchmark + 0.2 * rebalance_quality
episode_reward = 0.4 * terminal_pnl_ratio + 0.6 * mean_step_rewards
```

Expected scores: random ~0.25, GPT-4o ~0.55

### causal_chain (Hard)

A 10-step episode with three causally linked events spaced 3 steps apart. This is where most LLM agents fail. The `timing_bonus` specifically rewards positions entered before the consequence materialises — agents that react rather than anticipate get partial credit at best.

```
Step 1:  geopolitical signal (conflict in oil-producing region)
Step 4:  commodity_shock (supply disruption — consequence of step 1)
Step 7:  inflation_print (CPI spike — consequence of step 4)

Optimal: long USO and GLD entered at step 2, short TLT at step 3
Reactive: entering at step 4 or 7 is correct direction but loses timing_bonus
```

```
terminal_reward = 0.4 * directional_accuracy + 0.4 * timing_bonus + 0.2 * cost_efficiency
episode_reward  = 0.3 * mean_step_rewards + 0.7 * terminal_reward
```

Expected scores: random ~0.12, GPT-4o without memory prompting ~0.38, GPT-4o with explicit causal reasoning ~0.61

## Reward Design Summary

| Component | How it works |
|-----------|-------------|
| Directional accuracy | Fraction of positions where the sign matches the signal |
| Speed bonus | `1.0 / step` — earlier correct action scores higher |
| Timing bonus | 1.0 if positioned before the causal consequence arrives, 0.5 if at the same step, 0.1 if after |
| Cost efficiency | `1 - (transaction_costs / max_allowed)` — penalises excessive churning |
| Idle penalty | Small deduction for holding when actionable signals are present |

All rewards are in `[0.0, 1.0]`. Transaction cost is 10 basis points per unit of weight changed, deducted from NAV.

## Quickstart

Connect to the live Space:

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

Run the baseline inference script:

```bash
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o"
export HF_TOKEN="your-key-here"
python inference.py
```

Local development:

```bash
pip install -r requirements.txt
pip install -e ".[dev]"
uvicorn macro_signal.server.app:app --port 7860 --reload
pytest tests/
```

Docker:

```bash
docker build -t macro-signal-env .
docker run -p 7860:7860 macro-signal-env
curl http://localhost:7860/health
```

## Baseline Scores

Measured with `gpt-4o` via `inference.py`:

| Task | Difficulty | Score |
|------|-----------|-------|
| single_event | Easy | 0.72 |
| regime_shift | Medium | 0.55 |
| causal_chain | Hard | 0.38 |
| Mean | | 0.55 |

## Project Structure

```
macro-signal-env/
├── src/envs/macro_signal/
│   ├── models.py          Pydantic typed contracts (single source of truth)
│   ├── client.py          WebSocket client for training code
│   └── server/
│       ├── environment.py Core logic and graders
│       └── app.py         FastAPI server and web UI
├── data/
│   └── scenarios.json     10 seeded scenarios (5 easy, 3 medium, 2 hard)
├── openenv.yaml           OpenEnv spec metadata
├── Dockerfile             HF Spaces deployment
├── inference.py           Baseline agent script
├── ARCHITECTURE.md        Reward design and session lifecycle docs
└── tests/
    └── test_environment.py
```

## Web Terminal

The `/web` endpoint provides an interactive terminal for exploring the environment manually. Useful for demos and sanity-checking grader behaviour before running inference.

`https://krishvenky-macro-signal-env.hf.space/web`

## Author

Built by KrishVenky for the Meta x Scaler OpenEnv Competition.
