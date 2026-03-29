# Macro Signal Engine — Architecture & Reward Design

## Overview

The Macro Signal Engine is an OpenEnv environment where an LLM agent acts as a macro quantitative analyst. The agent manages a 4-asset portfolio (SPY, GLD, USO, TLT) in response to typed financial signal events over multi-step episodes.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│  inference.py / training code                           │
│                                                         │
│  client = MacroSignalEnv(base_url=HF_SPACE_URL)        │
│  result  = client.reset(task_type="causal_chain")      │
│  result  = client.step(MacroSignalAction(...))         │
└───────────────────┬─────────────────────────────────────┘
                    │  WebSocket /ws (primary)
                    │  HTTP /reset /step (debug only)
┌───────────────────▼─────────────────────────────────────┐
│  FastAPI Server (app.py)                                │
│                                                         │
│  Per WebSocket connection:                             │
│    env = MacroSignalEnvironment()  ← fresh instance    │
│    handles reset / step / state messages              │
│                                                         │
│  ┌──────────────────────────────────────────────────┐  │
│  │  MacroSignalEnvironment (environment.py)         │  │
│  │                                                  │  │
│  │  _portfolio  _prices  _step  _history           │  │
│  │  _scenario ← loaded from data/scenarios.json   │  │
│  │                                                  │  │
│  │  reset() → MacroSignalObservation               │  │
│  │  step()  → MacroSignalObservation + reward     │  │
│  │  state() → MacroSignalState                    │  │
│  └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

---

## Session Lifecycle

```
WebSocket Connect
      │
      ▼
reset(task_type, scenario_id?)
  → loads scenario from data/scenarios.json
  → initializes portfolio: 100% cash
  → returns step=0 observation with first signals
      │
      ▼ (repeat until done=True)
step(MacroSignalAction)
  → validates trade instructions
  → executes trades (apply transaction costs)
  → advances prices via scenario.price_path[step]
  → computes step reward (grader dispatch)
  → checks terminal condition (step >= max_steps)
  → returns observation + reward + done
      │
      ▼
WebSocket Disconnect
  → session state destroyed
  → no persistence between sessions
```

---

## Reward Design

### Guiding Principles

1. **Non-sparse**: every step produces a reward signal, not just the terminal step
2. **Partial progress**: correct directional exposure earns reward even before episode ends
3. **Penalizes bad behavior**: idling during high-signal steps, excessive transaction costs
4. **Deterministic**: all rewards computable from seeded scenario data — no randomness at grader call time

### Rubric Mapping

| Reward Component | Maps To |
|---|---|
| Step-level directional reward | Environment Design (20%) — "useful varying signal" |
| Terminal bonus on causal chain | Task Quality (25%) — "genuinely challenges frontier models" |
| Transaction cost penalty | Novelty (10%) — "clever reward design" |
| Idle penalty | Real-world utility (30%) — "penalizes clearly undesirable behavior" |

---

## Task 1: single_event (Easy)

### Episode Structure
- 3 steps maximum
- 1 signal event at step 1 (clear, unambiguous directional signal)
- Correct response: take long exposure to the signaled asset within 2 steps

### Reward Formula

```
step_reward = correct_direction × speed_bonus × (1 - idle_penalty)

where:
  correct_direction = 1.0 if sign(agent_weight[signal.asset]) == sign(signal.magnitude) else 0.0
  speed_bonus       = 1.0 / step  (step 1 = 1.0, step 2 = 0.5, step 3 = 0.33)
  idle_penalty      = 0.1 × (number of hold actions with non-empty signal feed)

terminal_reward     = max(step_rewards_across_episode)
episode_reward      = 0.6 × terminal_reward + 0.4 × mean(step_rewards)
```

### Why not just binary?
An agent that correctly buys USO at step 1 should score higher than one that buys at step 3. The `speed_bonus` captures this. An agent that buys the WRONG asset (SPY when the signal is about USO) gets 0.0 regardless of speed.

### Baseline expected score
- Random agent: ~0.10 (occasionally correct by chance, rarely on step 1)
- GPT-4o baseline: ~0.72 (usually correct direction, sometimes delayed)

---

## Task 2: regime_shift (Medium)

### Episode Structure
- 6 steps
- 3 regime-defining signals spread across steps 1–4
- Agent must rebalance portfolio to track a benchmark (60/40 SPY/TLT or defined per scenario)

### Reward Formula

```
step_reward = 0.5 × directional_correct + 0.3 × pnl_vs_benchmark + 0.2 × rebalance_quality

where:
  directional_correct  = fraction of current positions aligned with prevailing regime signal
  pnl_vs_benchmark     = clamp(agent_NAV_change / benchmark_NAV_change, 0.0, 1.0)
                         (if benchmark_NAV_change <= 0 and agent > 0: return 1.0)
                         (if both <= 0: return 0.5)
  rebalance_quality    = 1.0 - (actual_tracking_error / max_allowed_tracking_error)

terminal_reward        = pnl_vs_benchmark at step 6
episode_reward         = 0.4 × terminal_reward + 0.6 × mean(step_rewards)
```

### Baseline expected score
- Random agent: ~0.25
- GPT-4o baseline: ~0.55

---

## Task 3: causal_chain (Hard)

### Episode Structure
- 10 steps
- 3 causally-linked events:
  - Step 1: "geopolitical" shock (e.g., conflict in oil-producing region)
  - Step 4: "commodity_shock" (supply disruption — consequence of Step 1)
  - Step 7: "inflation_print" (inflationary consequence of Step 4)
- Optimal hedge for Step 9 requires connecting Step 1 → Step 4 → Step 7
- An agent that treats each event independently will underperform

### Reward Formula

```
step_reward(t) = 0.3 × directional_accuracy(t)
               + 0.2 × position_building_quality(t)
               - 0.05 × excess_turnover(t)

terminal_components:
  directional_accuracy = fraction of steps where net exposure sign matched causal outcome
  timing_bonus         = 1.0 if USO/GLD long entered BEFORE step 4 (anticipating supply shock)
                         0.5 if entered at step 4
                         0.0 if entered after step 4
  cost_efficiency      = 1.0 - (total_transaction_costs / scenario.max_transaction_cost)

terminal_reward = 0.4 × directional_accuracy
                + 0.4 × timing_bonus
                + 0.2 × cost_efficiency

episode_reward  = 0.3 × mean(step_rewards) + 0.7 × terminal_reward
```

### Why this breaks most LLM agents
The `timing_bonus` requires the agent to anticipate the supply shock at step 4 BASED ON the geopolitical signal at step 1. Agents that don't maintain cross-step causal reasoning will wait until the supply shock is explicit (step 4) to act — earning `timing_bonus = 0.5` instead of `1.0`. This is a 20-point terminal reward difference.

### Baseline expected score
- Random agent: ~0.12
- GPT-4o (no memory prompt): ~0.38
- GPT-4o (with explicit causal tracking in system prompt): ~0.61

---

## Scenario Bank Structure

See `data/scenarios.json` for full spec. Key fields per scenario:

```
scenario_id         — globally unique string
task_type           — single_event | regime_shift | causal_chain
max_steps           — 3 | 6 | 10
signal_schedule     — list of {step, events[]} — what the agent sees per step
price_path          — {asset: [price_t0, ..., price_tN]} — N = max_steps
benchmark_weights   — {asset: weight} — for regime_shift grading
max_transaction_cost — float — for cost_efficiency grading
causal_chain        — null | [{step, linked_step, asset, expected_direction}]
```

---

## Transaction Cost Model

Each `TradeInstruction` incurs a cost:

```
transaction_cost = abs(new_weight - old_weight) × COST_RATE
COST_RATE = 0.001  (10 bps — realistic for liquid ETFs)
```

This is deducted from portfolio NAV at each step, not from the reward directly. The reward grader sees the post-cost NAV. This means excessive rebalancing naturally hurts the PnL-based reward components without requiring an explicit penalty term.

---

## Portfolio State

```
Initial state: 100% cash
Assets: SPY, GLD, USO, TLT
Weights: float in [-1.0, +1.0] (negative = short)
Constraint: sum(abs(weights)) <= 1.0 (no leverage)
Cash = 1.0 - sum(weights)  (cash earns 0)
```

Weight validation happens in `environment.py` before any trade is executed. Invalid instructions (sum > 1.0) are clipped proportionally with a small penalty logged to `info`.

---

## Adding New Scenarios

No code changes required. Edit `data/scenarios.json`:

1. Add a new scenario object following the schema above
2. Restart the server (or it will be picked up on next `reset()` call)
3. Test with `pre_submission_test.ipynb` cell "Scenario bank validation"

## Adding New Task Types

Requires code changes in `environment.py`:

1. Add grader method `_grade_<task_type>(action, step) -> float`
2. Add dispatch case in `_compute_reward()`
3. Add task metadata to `openenv.yaml` tasks list
4. Add at least 2 scenarios to `data/scenarios.json`
5. Add test cases to `tests/test_environment.py`
