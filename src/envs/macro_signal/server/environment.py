"""
Macro Signal Engine — Core Environment Logic
=============================================
One instance per WebSocket session. All state is per-instance (never shared).
Imports only from models.py and stdlib/json.
"""

from __future__ import annotations

import json
import logging
import math
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..models import (
    ASSETS,
    MacroSignalAction,
    MacroSignalObservation,
    MacroSignalState,
    PortfolioPosition,
    SignalEvent,
    StepResult,
    TradeInstruction,
)

logger = logging.getLogger(__name__)

COST_RATE = 0.001  # 10 bps per unit of weight changed — realistic for liquid ETFs
INITIAL_NAV = 1.0
IDLE_PENALTY = 0.02  # deducted from step reward when holding during high-signal step


# ---------------------------------------------------------------------------
# Scenario loader (module-level cache — loaded once at first import)
# ---------------------------------------------------------------------------

def _find_scenarios_path() -> Path:
    """Locate scenarios.json relative to this file or the repo root."""
    # Try: repo_root/data/scenarios.json (works in Docker and local dev)
    candidates = [
        Path(__file__).parent.parent.parent.parent.parent / "data" / "scenarios.json",  # /app/data/
        Path(__file__).parent.parent.parent.parent / "data" / "scenarios.json",
        Path(__file__).parent.parent.parent / "data" / "scenarios.json",
        Path.cwd() / "data" / "scenarios.json",
    ]
    for p in candidates:
        if p.exists():
            return p
    # Return the most likely path for a clear error message
    return candidates[0]


_SCENARIOS_PATH = _find_scenarios_path()
_SCENARIO_CACHE: Optional[Dict[str, Any]] = None


def _load_scenario_bank() -> Dict[str, Any]:
    global _SCENARIO_CACHE
    if _SCENARIO_CACHE is not None:
        return _SCENARIO_CACHE

    if not _SCENARIOS_PATH.exists():
        raise FileNotFoundError(f"Scenario bank not found at {_SCENARIOS_PATH}")

    with open(_SCENARIOS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Validate all scenarios at startup — fail fast
    scenario_ids = set()
    for s in data["scenarios"]:
        sid = s["scenario_id"]
        if sid in scenario_ids:
            raise ValueError(f"Duplicate scenario_id: {sid}")
        scenario_ids.add(sid)

        expected_len = s["max_steps"] + 1
        for asset in ["SPY", "GLD", "USO", "TLT"]:
            actual_len = len(s["price_path"][asset])
            if actual_len != expected_len:
                raise ValueError(
                    f"Scenario {sid}: price_path[{asset}] has {actual_len} entries, "
                    f"expected {expected_len} (max_steps={s['max_steps']} + 1)"
                )

    _SCENARIO_CACHE = {s["scenario_id"]: s for s in data["scenarios"]}
    logger.info("Loaded %d scenarios from scenario bank", len(_SCENARIO_CACHE))
    return _SCENARIO_CACHE


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class MacroSignalEnvironment:
    """
    Stateful environment for one WebSocket session.
    All fields are reset by calling reset().
    """

    def __init__(self) -> None:
        self._episode_id: str = ""
        self._scenario: Dict[str, Any] = {}
        self._step: int = 0
        self._max_steps: int = 0
        self._done: bool = False

        # Portfolio state: asset -> current weight
        self._weights: Dict[str, float] = {a: 0.0 for a in ["SPY", "GLD", "USO", "TLT"]}
        self._entry_prices: Dict[str, float] = {a: 0.0 for a in ["SPY", "GLD", "USO", "TLT"]}
        self._nav: float = INITIAL_NAV
        self._cash: float = 1.0

        # Episode tracking
        self._cumulative_reward: float = 0.0
        self._step_rewards: List[float] = []
        self._total_transaction_cost: float = 0.0
        self._history: List[Dict[str, Any]] = []  # for causal_chain grader

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(
        self,
        task_type: Optional[str] = None,
        scenario_id: Optional[str] = None,
    ) -> MacroSignalObservation:
        """Start a new episode. Reinitializes all state."""
        bank = _load_scenario_bank()

        if scenario_id is not None:
            if scenario_id not in bank:
                raise ValueError(f"Unknown scenario_id: {scenario_id}")
            self._scenario = bank[scenario_id]
        elif task_type is not None:
            candidates = [s for s in bank.values() if s["task_type"] == task_type]
            if not candidates:
                raise ValueError(f"No scenarios found for task_type: {task_type}")
            # Deterministic selection: pick first alphabetically for reproducibility
            self._scenario = sorted(candidates, key=lambda s: s["scenario_id"])[0]
        else:
            # Default: single_event easy task
            candidates = [s for s in bank.values() if s["task_type"] == "single_event"]
            self._scenario = sorted(candidates, key=lambda s: s["scenario_id"])[0]

        self._episode_id = str(uuid.uuid4())
        self._step = 0
        self._max_steps = self._scenario["max_steps"]
        self._done = False

        # Reset portfolio
        self._weights = {a: 0.0 for a in ["SPY", "GLD", "USO", "TLT"]}
        self._entry_prices = {a: self._scenario["price_path"][a][0] for a in ["SPY", "GLD", "USO", "TLT"]}
        self._nav = INITIAL_NAV
        self._cash = 1.0

        # Reset tracking
        self._cumulative_reward = 0.0
        self._step_rewards = []
        self._total_transaction_cost = 0.0
        self._history = []

        logger.info(
            "Episode %s started: task=%s scenario=%s max_steps=%d",
            self._episode_id[:8],
            self._scenario["task_type"],
            self._scenario["scenario_id"],
            self._max_steps,
        )

        return self._build_observation(step_reward=0.0)

    def step(self, action: MacroSignalAction) -> StepResult:
        """Execute one environment step."""
        if self._done:
            raise RuntimeError("Cannot call step() after episode is done. Call reset() first.")

        self._step += 1

        # 1. Execute trades
        transaction_cost = self._execute_trades(action.trade_instructions)
        self._total_transaction_cost += transaction_cost

        # 2. Advance prices
        self._advance_prices()

        # 3. Record history for causal_chain grader
        self._history.append({
            "step": self._step,
            "weights": dict(self._weights),
            "reasoning": action.reasoning,
            "signals": self._get_signals_at_step(self._step),
            "nav": self._nav,
        })

        # 4. Compute step reward
        step_reward = self._compute_reward(action)
        self._step_rewards.append(step_reward)
        self._cumulative_reward = min(1.0, self._cumulative_reward + step_reward * 0.1)

        # 5. Check terminal condition
        self._done = self._step >= self._max_steps

        # 6. Compute final episode reward if done
        episode_reward = 0.0
        if self._done:
            episode_reward = self._compute_episode_reward()

        obs = self._build_observation(step_reward=step_reward, episode_reward=episode_reward if self._done else 0.0)

        return StepResult(
            observation=obs,
            reward=episode_reward if self._done else step_reward,
            done=self._done,
            info={"episode_id": self._episode_id, "transaction_cost": transaction_cost},
        )

    def state(self) -> MacroSignalState:
        """Return current episode state metadata."""
        return MacroSignalState(
            episode_id=self._episode_id,
            task_type=self._scenario.get("task_type", "single_event"),
            scenario_id=self._scenario.get("scenario_id", ""),
            step_count=self._step,
            max_steps=self._max_steps,
            portfolio_nav=self._nav,
            cumulative_reward=self._cumulative_reward,
            done=self._done,
        )

    # ------------------------------------------------------------------
    # Trade execution
    # ------------------------------------------------------------------

    def _execute_trades(self, instructions: List[TradeInstruction]) -> float:
        """Apply trade instructions. Returns total transaction cost incurred."""
        # Build target weight map
        target_weights: Dict[str, float] = dict(self._weights)
        for instr in instructions:
            target_weights[instr.asset] = instr.target_weight

        # Validate total weight (clamp if over limit)
        total_abs = sum(abs(w) for w in target_weights.values())
        if total_abs > 1.0 + 1e-6:
            scale = 1.0 / total_abs
            target_weights = {a: w * scale for a, w in target_weights.items()}
            logger.warning("Weight sum exceeded 1.0, scaled down by %.4f", scale)

        # Compute transaction costs and update entry prices
        total_cost = 0.0
        for asset in ["SPY", "GLD", "USO", "TLT"]:
            old_weight = self._weights[asset]
            new_weight = target_weights[asset]
            change = abs(new_weight - old_weight)
            cost = change * COST_RATE
            total_cost += cost

            # Update entry price when opening/increasing a position
            if abs(new_weight) > abs(old_weight):
                current_price = self._current_price(asset)
                self._entry_prices[asset] = current_price

            self._weights[asset] = new_weight

        # Deduct cost from NAV
        self._nav *= (1.0 - total_cost)
        self._cash = max(0.0, 1.0 - sum(abs(w) for w in self._weights.values()))

        return total_cost

    # ------------------------------------------------------------------
    # Price simulation
    # ------------------------------------------------------------------

    def _advance_prices(self) -> None:
        """Move to the next price step using the scenario price path."""
        # Price path indexed 0..max_steps — step N uses price_path[N]
        # After executing trades at step N, prices move to price_path[N]
        for asset in ["SPY", "GLD", "USO", "TLT"]:
            new_price = self._scenario["price_path"][asset][self._step]
            old_price = self._scenario["price_path"][asset][self._step - 1]
            price_return = (new_price - old_price) / old_price
            # Apply price return to NAV proportional to weight
            self._nav += self._nav * self._weights[asset] * price_return

        self._nav = max(0.01, self._nav)  # floor NAV to avoid zero/negative

    def _current_price(self, asset: str) -> float:
        idx = min(self._step, self._max_steps)
        return self._scenario["price_path"][asset][idx]

    # ------------------------------------------------------------------
    # Signal helpers
    # ------------------------------------------------------------------

    def _get_signals_at_step(self, step: int) -> List[Dict[str, Any]]:
        for entry in self._scenario["signal_schedule"]:
            if entry["step"] == step:
                return entry["events"]
        return []

    def _has_actionable_signals(self) -> bool:
        """True if current step has non-empty signal events."""
        return len(self._get_signals_at_step(self._step)) > 0

    # ------------------------------------------------------------------
    # Reward computation
    # ------------------------------------------------------------------

    def _compute_reward(self, action: MacroSignalAction) -> float:
        task = self._scenario["task_type"]

        if task == "single_event":
            reward = self._grade_single_event_step(action)
        elif task == "regime_shift":
            reward = self._grade_regime_shift_step()
        elif task == "causal_chain":
            reward = self._grade_causal_chain_step(action)
        else:
            reward = 0.0

        # Idle penalty: agent holds when there are actionable signals
        is_idle = len(action.trade_instructions) == 0 or all(
            abs(t.target_weight - self._weights.get(t.asset, 0.0)) < 0.01
            for t in action.trade_instructions
        )
        if is_idle and self._has_actionable_signals():
            reward = max(0.0, reward - IDLE_PENALTY)

        assert 0.0 <= reward <= 1.0, f"Step reward {reward} out of range [0.0, 1.0]"
        return reward

    def _grade_single_event_step(self, action: MacroSignalAction) -> float:
        """
        Rewards taking correct directional position relative to signal.
        speed_bonus: faster response = higher reward.
        """
        correct_direction = self._scenario.get("correct_direction", {})
        if not correct_direction:
            return 0.0

        correct_count = 0
        total_signals = len(correct_direction)

        for asset, expected_dir in correct_direction.items():
            agent_weight = self._weights.get(asset, 0.0)
            if expected_dir > 0 and agent_weight > 0.05:
                correct_count += 1
            elif expected_dir < 0 and agent_weight < -0.05:
                correct_count += 1

        if total_signals == 0:
            return 0.0

        directional_score = correct_count / total_signals
        speed_bonus = 1.0 / self._step  # step 1=1.0, step 2=0.5, step 3=0.33

        reward = directional_score * speed_bonus
        return float(min(1.0, max(0.0, reward)))

    def _grade_regime_shift_step(self) -> float:
        """
        Rewards tracking the benchmark return.
        Partial reward at each step for staying aligned with the regime.
        """
        if self._step < 2:
            return 0.0

        # Compute agent return since episode start
        agent_return = (self._nav - INITIAL_NAV) / INITIAL_NAV

        # Compute benchmark return
        benchmark_weights = self._scenario.get("benchmark_weights", {"SPY": 0.6, "TLT": 0.4})
        benchmark_return = 0.0
        for asset, bw in benchmark_weights.items():
            price_start = self._scenario["price_path"][asset][0]
            price_now = self._scenario["price_path"][asset][self._step]
            benchmark_return += bw * (price_now - price_start) / price_start

        # Guard: zero/negative benchmark
        if benchmark_return <= 0:
            if agent_return > 0:
                return 0.8  # agent positive while benchmark flat/negative = good
            return 0.4  # both non-positive = neutral

        ratio = agent_return / benchmark_return
        reward = min(1.0, max(0.0, ratio))
        return float(reward)

    def _grade_causal_chain_step(self, action: MacroSignalAction) -> float:
        """
        Step-level: rewards building positions in causally-linked assets.
        Terminal reward is computed separately in _compute_episode_reward.
        """
        causal_links = self._scenario.get("causal_chain") or []
        if not causal_links:
            return 0.0

        # Find which causal links are "active" — their trigger step <= current step
        # and their consequence step > current step (i.e., agent should be positioning now)
        reward_components = []
        for link in causal_links:
            trigger_step = link["step"]
            consequence_step = link["linked_step"]
            asset = link["asset"]
            expected_dir = link["expected_direction"]

            if trigger_step <= self._step < consequence_step:
                # Agent should be building position now
                agent_weight = self._weights.get(asset, 0.0)
                if expected_dir > 0 and agent_weight > 0.05:
                    reward_components.append(1.0)
                elif expected_dir < 0 and agent_weight < -0.05:
                    reward_components.append(1.0)
                else:
                    reward_components.append(0.0)

        if not reward_components:
            return 0.2  # neutral when no active causal window

        step_score = sum(reward_components) / len(reward_components)
        return float(min(1.0, max(0.0, step_score * 0.4)))  # scale down — terminal reward is primary

    # ------------------------------------------------------------------
    # Terminal / episode reward
    # ------------------------------------------------------------------

    def _compute_episode_reward(self) -> float:
        task = self._scenario["task_type"]

        if task == "single_event":
            reward = self._terminal_single_event()
        elif task == "regime_shift":
            reward = self._terminal_regime_shift()
        elif task == "causal_chain":
            reward = self._terminal_causal_chain()
        else:
            reward = 0.0

        assert 0.0 <= reward <= 1.0, f"Episode reward {reward} out of range [0.0, 1.0]"
        logger.info(
            "Episode %s done: task=%s reward=%.4f",
            self._episode_id[:8],
            task,
            reward,
        )
        return reward

    def _terminal_single_event(self) -> float:
        """
        episode_reward = 0.6 × best_step_reward + 0.4 × mean_step_rewards
        """
        if not self._step_rewards:
            return 0.0
        best = max(self._step_rewards)
        mean = sum(self._step_rewards) / len(self._step_rewards)
        reward = 0.6 * best + 0.4 * mean
        return float(min(1.0, max(0.0, reward)))

    def _terminal_regime_shift(self) -> float:
        """
        episode_reward = 0.4 × terminal_pnl_ratio + 0.6 × mean(step_rewards)
        """
        # Terminal PnL ratio
        agent_return = (self._nav - INITIAL_NAV) / INITIAL_NAV
        benchmark_weights = self._scenario.get("benchmark_weights", {"SPY": 0.6, "TLT": 0.4})
        benchmark_return = 0.0
        for asset, bw in benchmark_weights.items():
            price_start = self._scenario["price_path"][asset][0]
            price_end = self._scenario["price_path"][asset][self._max_steps]
            benchmark_return += bw * (price_end - price_start) / price_start

        if benchmark_return <= 0:
            pnl_ratio = 1.0 if agent_return > 0 else 0.4
        else:
            pnl_ratio = min(1.0, max(0.0, agent_return / benchmark_return))

        mean_step = sum(self._step_rewards) / len(self._step_rewards) if self._step_rewards else 0.0
        reward = 0.4 * pnl_ratio + 0.6 * mean_step
        return float(min(1.0, max(0.0, reward)))

    def _terminal_causal_chain(self) -> float:
        """
        terminal_reward = 0.4 × directional_accuracy
                        + 0.4 × timing_bonus
                        + 0.2 × cost_efficiency
        episode_reward  = 0.3 × mean(step_rewards) + 0.7 × terminal_reward
        """
        correct_direction = self._scenario.get("correct_direction", {})
        causal_links = self._scenario.get("causal_chain") or []
        timing_thresholds = self._scenario.get("timing_thresholds", {})

        # 1. Directional accuracy: fraction of steps where positions were correct
        directional_correct = 0
        directional_total = 0
        for record in self._history:
            for asset, expected_dir in correct_direction.items():
                weight = record["weights"].get(asset, 0.0)
                directional_total += 1
                if expected_dir > 0 and weight > 0.05:
                    directional_correct += 1
                elif expected_dir < 0 and weight < -0.05:
                    directional_correct += 1

        directional_accuracy = directional_correct / directional_total if directional_total > 0 else 0.0

        # 2. Timing bonus: was the position entered before the causal consequence?
        timing_scores = []
        for key, threshold_step in timing_thresholds.items():
            # Parse key: "ASSET_entry_before_step" or "ASSET_short_before_step"
            parts = key.split("_")
            asset = parts[0]
            is_short = "short" in key

            # Find first step where agent had meaningful exposure
            first_entry_step = None
            for record in self._history:
                weight = record["weights"].get(asset, 0.0)
                if is_short and weight < -0.05:
                    first_entry_step = record["step"]
                    break
                elif not is_short and weight > 0.05:
                    first_entry_step = record["step"]
                    break

            if first_entry_step is None:
                timing_scores.append(0.0)  # never entered
            elif first_entry_step < threshold_step:
                timing_scores.append(1.0)  # entered before consequence — full credit
            elif first_entry_step == threshold_step:
                timing_scores.append(0.5)  # entered at consequence — partial credit
            else:
                timing_scores.append(0.1)  # entered after — minimal credit

        timing_bonus = sum(timing_scores) / len(timing_scores) if timing_scores else 0.5

        # 3. Cost efficiency
        max_cost = self._scenario.get("max_transaction_cost", 0.06)
        if max_cost <= 0:
            cost_efficiency = 1.0
        else:
            cost_efficiency = max(0.0, 1.0 - (self._total_transaction_cost / max_cost))

        terminal_reward = (
            0.4 * directional_accuracy
            + 0.4 * timing_bonus
            + 0.2 * cost_efficiency
        )
        mean_step = sum(self._step_rewards) / len(self._step_rewards) if self._step_rewards else 0.0
        episode_reward = 0.3 * mean_step + 0.7 * terminal_reward

        assert 0.0 <= episode_reward <= 1.0, f"Causal chain reward {episode_reward} out of range"
        return float(min(1.0, max(0.0, episode_reward)))

    # ------------------------------------------------------------------
    # Observation builder
    # ------------------------------------------------------------------

    def _build_observation(
        self,
        step_reward: float = 0.0,
        episode_reward: float = 0.0,
    ) -> MacroSignalObservation:
        signals = self._get_signals_at_step(self._step + 1 if not self._done else self._step)
        signal_events = [
            SignalEvent(
                event_type=e["event_type"],
                asset=e["asset"],
                magnitude=e["magnitude"],
                step=e["step"],
            )
            for e in signals
        ]

        positions = []
        for asset in ["SPY", "GLD", "USO", "TLT"]:
            w = self._weights[asset]
            if abs(w) > 1e-4:
                ep = self._entry_prices[asset]
                cp = self._current_price(asset)
                upnl = w * (cp - ep) / ep if ep > 0 else 0.0
                positions.append(
                    PortfolioPosition(
                        asset=asset,
                        weight=w,
                        entry_price=ep,
                        current_price=cp,
                        unrealized_pnl=upnl,
                    )
                )

        cash = max(0.0, 1.0 - sum(abs(self._weights[a]) for a in ["SPY", "GLD", "USO", "TLT"]))

        # Benchmark return for context
        bw = self._scenario.get("benchmark_weights", {"SPY": 0.6, "TLT": 0.4})
        benchmark_return = 0.0
        for asset, weight in bw.items():
            price_start = self._scenario["price_path"][asset][0]
            price_now = self._scenario["price_path"][asset][min(self._step, self._max_steps)]
            benchmark_return += weight * (price_now - price_start) / price_start

        info: Dict[str, Any] = {
            "task_type": self._scenario["task_type"],
            "description": self._scenario.get("description", ""),
        }
        if self._done:
            info["episode_summary"] = {
                "total_transaction_cost": round(self._total_transaction_cost, 6),
                "final_nav": round(self._nav, 6),
                "step_rewards": [round(r, 4) for r in self._step_rewards],
            }

        return MacroSignalObservation(
            step=self._step,
            max_steps=self._max_steps,
            task_type=self._scenario["task_type"],
            scenario_id=self._scenario["scenario_id"],
            signal_events=signal_events,
            portfolio=positions,
            cash_weight=cash,
            portfolio_nav=round(self._nav, 6),
            benchmark_return=round(benchmark_return, 6),
            step_reward=round(step_reward, 6),
            cumulative_reward=round(self._cumulative_reward, 6),
            done=self._done,
            reward=round(episode_reward if self._done else step_reward, 6),
            info=info,
        )
