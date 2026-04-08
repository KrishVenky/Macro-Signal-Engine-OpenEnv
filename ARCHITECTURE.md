# Macro Signal Engine: Architecture and Reward Design

## Overview

The Macro Signal Engine is an OpenEnv environment where an LLM agent acts as a macro quantitative analyst. The agent manages a 4-asset portfolio (SPY, GLD, USO, TLT) in response to typed financial signal events over multi-step episodes.

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│  inference.py / training code                           │
│                                                         │
│  client = MacroSignalEnv(base_url=HF_SPACE_URL)         │
│  result  = client.reset(task_type="causal_chain")       │
│  result  = client.step(MacroSignalAction(...))          │ 
└───────────────────┬─────────────────────────────────────┘
                    │  WebSocket /ws (primary transport)
                    │  HTTP /reset /step (debug only)
┌───────────────────▼─────────────────────────────────────┐
│  FastAPI Server (app.py)                                │
│                                                         │
│  Each WebSocket connection gets a fresh env instance.   │
│                                                         │
│  ┌──────────────────────────────────────────────────┐   │
│  │  MacroSignalEnvironment (environment.py)         │   │
│  │                                                  │   │
│  │  _portfolio  _prices  _step  _history            │   │
│  │  _scenario  loaded from data/scenarios.json      │   │
│  │                                                  │   │
│  │  reset()  MacroSignalObservation                 │   │
│  │  step()   MacroSignalObservation + reward        │   │
│  │  state()  MacroSignalState                       │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

## Session Lifecycle

```
WebSocket Connect
      │
      ▼
reset(task_type, scenario_id?)
  loads scenario from data/scenarios.json
  initializes portfolio to 100% cash
  returns step=0 observation with first signals
      │
      ▼  (repeat until done=True)
step(MacroSignalAction)
  validates trade instructions
  executes trades and deducts transaction costs
  advances prices via scenario.price_path[step]
  computes step reward via grader dispatch
  checks terminal condition (step >= max_steps)
  returns observation + reward + done
      │
      ▼
WebSocket Disconnect
  session state is destroyed
  no persistence between sessions
```

## Reward Design

### Guiding Principles

Every step produces a signal. The reward is never purely sparse. Correct directional exposure earns partial credit even mid-episode, and clearly bad behaviour (idling during high-signal steps, excessive churning) is penalised. All rewards are deterministic given the seeded scenario — no randomness at grader call time.

### Rubric Mapping

| Reward Component | Competition Rubric |
|---|---|
| Step-level directional reward | Environment Design (20%) — useful varying signal |
| Terminal bonus on causal chain | Task Quality (25%) — genuinely challenges frontier models |
| Transaction cost penalty | Novelty (10%) — clever reward design |
| Idle penalty | Real-world utility (30%) — penalises undesirable behaviour |

## Task 1: single_event (Easy)

3-step episode with one unambiguous directional signal at step 1. The agent should take long exposure to the signaled asset as early as possible.

```
step_reward    = correct_direction * speed_bonus
correct_direction = 1.0 if sign(agent_weight[asset]) == sign(signal.magnitude) else 0.0
speed_bonus       = 1.0 / step  (1.0 at step 1, 0.5 at step 2, 0.33 at step 3)

episode_reward = 0.6 * best_step_reward + 0.4 * mean(step_rewards)
```

An agent that gets the direction right at step 1 outscores one that gets it right at step 3. An agent that buys the wrong asset gets 0.0 regardless of speed.

Baseline: random ~0.10, GPT-4o ~0.72

## Task 2: regime_shift (Medium)

6-step episode with multiple regime-defining signals across steps 1 to 4. The agent needs to rebalance to track a benchmark through a coherent market regime.

```
step_reward = 0.5 * directional_correct + 0.3 * pnl_vs_benchmark + 0.2 * rebalance_quality

pnl_vs_benchmark = clamp(agent_NAV_change / benchmark_NAV_change, 0.0, 1.0)
                   returns 1.0 if benchmark is flat or negative and agent is positive
                   returns 0.5 if both are non-positive

episode_reward = 0.4 * terminal_pnl_ratio + 0.6 * mean(step_rewards)
```

Baseline: random ~0.25, GPT-4o ~0.55

## Task 3: causal_chain (Hard)

10-step episode with three causally linked events spaced 3 steps apart. This is where most LLM agents break down. The `timing_bonus` specifically rewards agents that position themselves before the consequence arrives, not after.

```
Step 1: geopolitical shock (conflict in oil-producing region)
Step 4: commodity_shock (supply disruption, consequence of step 1)
Step 7: inflation_print (CPI spike, consequence of step 4)

Optimal: long USO and GLD at step 2, short TLT at step 3
Reactive: entering at step 4 or 7 gets partial credit only
```

```
terminal_reward = 0.4 * directional_accuracy + 0.4 * timing_bonus + 0.2 * cost_efficiency

timing_bonus: 1.0 if position entered before the causal consequence
              0.5 if entered at the same step as consequence
              0.1 if entered after

episode_reward = 0.3 * mean(step_rewards) + 0.7 * terminal_reward
```

The timing_bonus creates a 20-point terminal reward gap between an agent that reasons causally and one that simply reacts. Most LLMs without explicit chain-of-thought prompting fall into the reactive pattern.

Baseline: random ~0.12, GPT-4o without memory prompting ~0.38, GPT-4o with explicit causal reasoning ~0.61

## Scenario Bank

Each scenario in `data/scenarios.json` defines a complete episode. Key fields:

```
scenario_id          globally unique identifier
task_type            single_event | regime_shift | causal_chain
max_steps            3 | 6 | 10
signal_schedule      list of {step, events[]} defining what the agent sees each step
price_path           {asset: [price_t0, ..., price_tN]} where N = max_steps
benchmark_weights    {asset: weight} used for regime_shift grading
max_transaction_cost float cap used for cost_efficiency grading
causal_chain         null for easy/medium, or [{step, linked_step, asset, expected_direction}] for hard
```

Scenarios are validated at server startup. The price path must have exactly `max_steps + 1` entries per asset — the server will refuse to start if this is violated.

## Transaction Cost Model

```
transaction_cost = abs(new_weight - old_weight) * 0.001  (10 basis points)
```

Costs are deducted from NAV directly, not from reward. This means excessive rebalancing naturally depresses PnL-based reward components without needing a separate penalty term in the grader.

## Portfolio State

```
Initial state: 100% cash
Assets:        SPY, GLD, USO, TLT
Weights:       float in [-1.0, 1.0], negative = short
Constraint:    sum(abs(weights)) <= 1.0, no leverage
Cash:          1.0 - sum(weights), earns zero return
```

Instructions that would push the total above 1.0 are scaled down proportionally before execution. The scaling is logged to `info` in the observation.

## Adding New Scenarios

No code changes needed. Add a new object to `data/scenarios.json` following the schema above, restart the server, and validate with the scenario bank cell in `pre_submission_test.ipynb`.

## Adding New Task Types

Requires changes in `environment.py`:

1. Add a `_grade_<task_type>(action, step) -> float` method
2. Add the dispatch case in `_compute_reward()`
3. Add task metadata to `openenv.yaml`
4. Add at least 2 scenarios to `data/scenarios.json`
5. Add test cases to `tests/test_environment.py`
