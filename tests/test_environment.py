"""
Tests for MacroSignalEnvironment — graders, portfolio logic, reward bounds.
Run with: pytest tests/
"""

from __future__ import annotations

import pytest

from src.envs.macro_signal.models import MacroSignalAction, TradeInstruction
from src.envs.macro_signal.server.environment import MacroSignalEnvironment


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def env() -> MacroSignalEnvironment:
    return MacroSignalEnvironment()


# ---------------------------------------------------------------------------
# reset()
# ---------------------------------------------------------------------------


def test_reset_returns_observation(env):
    obs = env.reset(task_type="single_event")
    assert obs.step == 0
    assert obs.portfolio_nav > 0
    assert obs.cash_weight == pytest.approx(1.0, abs=0.01)
    assert not obs.done


def test_reset_twice_reinitializes(env):
    env.reset(task_type="single_event")
    env.reset(task_type="regime_shift")
    assert env._step == 0
    assert env._task_type_for_test() == "regime_shift"


def test_reset_unknown_scenario_raises(env):
    with pytest.raises(ValueError, match="Unknown scenario_id"):
        env.reset(scenario_id="nonexistent_scenario_xyz")


def test_reset_unknown_task_type_raises(env):
    with pytest.raises(ValueError, match="No scenarios found"):
        env.reset(task_type="invalid_task_type_xyz")


# ---------------------------------------------------------------------------
# step()
# ---------------------------------------------------------------------------


def test_step_after_done_raises(env):
    env.reset(task_type="single_event")
    # Exhaust the episode
    action = MacroSignalAction(trade_instructions=[], reasoning="hold")
    result = None
    for _ in range(20):
        result = env.step(action)
        if result.done:
            break
    assert result.done
    with pytest.raises(RuntimeError, match="done"):
        env.step(action)


def test_step_increments_step_counter(env):
    env.reset(task_type="single_event")
    env.step(MacroSignalAction(trade_instructions=[], reasoning=""))
    assert env._step == 1


def test_step_reward_in_range(env):
    env.reset(task_type="single_event")
    action = MacroSignalAction(
        trade_instructions=[TradeInstruction(asset="USO", target_weight=0.5)],
        reasoning="test",
    )
    result = env.step(action)
    assert 0.0 <= result.reward <= 1.0


def test_episode_reward_in_range_all_tasks(env):
    for task in ["single_event", "regime_shift", "causal_chain"]:
        env.reset(task_type=task)
        action = MacroSignalAction(trade_instructions=[], reasoning="hold")
        result = None
        for _ in range(15):
            result = env.step(action)
            if result.done:
                break
        assert result is not None
        assert result.done, f"Task {task} did not complete"
        assert 0.0 <= result.reward <= 1.0, f"Reward {result.reward} out of range for task {task}"


# ---------------------------------------------------------------------------
# Grader: single_event
# ---------------------------------------------------------------------------


def test_single_event_correct_direction_gives_reward(env):
    """Long USO when the signal is bullish USO should score > 0."""
    env.reset(scenario_id="single_event_001")  # USO commodity shock +0.85
    action = MacroSignalAction(
        trade_instructions=[TradeInstruction(asset="USO", target_weight=0.6)],
        reasoning="Going long USO due to commodity shock",
    )
    result = env.step(action)
    assert result.reward > 0.0


def test_single_event_wrong_direction_gives_zero(env):
    """Short USO when the signal is bullish USO should score 0.0."""
    env.reset(scenario_id="single_event_001")
    action = MacroSignalAction(
        trade_instructions=[TradeInstruction(asset="USO", target_weight=-0.5)],
        reasoning="Wrong: shorting USO",
    )
    result = env.step(action)
    assert result.reward == pytest.approx(0.0, abs=0.01)


def test_single_event_speed_bonus_step1_gt_step2(env):
    """Action at step 1 should score higher than same action at step 2."""
    env.reset(scenario_id="single_event_001")
    action = MacroSignalAction(
        trade_instructions=[TradeInstruction(asset="USO", target_weight=0.6)],
        reasoning="Long USO",
    )
    result1 = env.step(action)
    reward_step1 = result1.observation.step_reward

    env.reset(scenario_id="single_event_001")
    env.step(MacroSignalAction(trade_instructions=[], reasoning="hold step 1"))
    result2 = env.step(action)
    reward_step2 = result2.observation.step_reward

    assert reward_step1 > reward_step2


# ---------------------------------------------------------------------------
# Grader: regime_shift
# ---------------------------------------------------------------------------


def test_regime_shift_completes(env):
    env.reset(task_type="regime_shift")
    action = MacroSignalAction(
        trade_instructions=[
            TradeInstruction(asset="SPY", target_weight=0.5),
            TradeInstruction(asset="TLT", target_weight=0.3),
        ],
        reasoning="Bull regime: long SPY and TLT",
    )
    result = None
    for _ in range(10):
        result = env.step(action)
        if result.done:
            break
    assert result.done
    assert 0.0 <= result.reward <= 1.0


# ---------------------------------------------------------------------------
# Portfolio constraints
# ---------------------------------------------------------------------------


def test_weight_sum_exceeds_limit_is_clamped(env):
    """Instructions summing to >1.0 should be scaled down, not rejected."""
    env.reset(task_type="single_event")
    action = MacroSignalAction(
        trade_instructions=[
            TradeInstruction(asset="SPY", target_weight=0.5),
            TradeInstruction(asset="GLD", target_weight=0.5),
            TradeInstruction(asset="USO", target_weight=0.4),  # sum = 1.4
        ],
        reasoning="Overweight test",
    )
    # Should not raise — environment clamps internally
    result = env.step(action)
    total = sum(abs(env._weights[a]) for a in ["SPY", "GLD", "USO", "TLT"])
    assert total <= 1.0 + 1e-4


def test_nav_never_goes_negative(env):
    """NAV should always be positive even after bad trades."""
    env.reset(task_type="causal_chain")
    action = MacroSignalAction(
        trade_instructions=[
            TradeInstruction(asset="USO", target_weight=-0.5),
            TradeInstruction(asset="SPY", target_weight=-0.4),
        ],
        reasoning="Short everything (adversarial)",
    )
    for _ in range(12):
        result = env.step(action)
        assert result.observation.portfolio_nav > 0
        if result.done:
            break


# ---------------------------------------------------------------------------
# Scenario bank
# ---------------------------------------------------------------------------


def test_all_scenarios_load_without_error():
    from src.envs.macro_signal.server.environment import _load_scenario_bank, _SCENARIO_CACHE
    # Clear cache to force reload
    import src.envs.macro_signal.server.environment as env_module
    env_module._SCENARIO_CACHE = None
    bank = _load_scenario_bank()
    assert len(bank) >= 10
    for sid, scenario in bank.items():
        assert "task_type" in scenario
        assert "max_steps" in scenario
        assert "price_path" in scenario
        for asset in ["SPY", "GLD", "USO", "TLT"]:
            assert len(scenario["price_path"][asset]) == scenario["max_steps"] + 1


def test_all_three_task_types_present():
    import src.envs.macro_signal.server.environment as env_module
    env_module._SCENARIO_CACHE = None
    bank = env_module._load_scenario_bank()
    task_types = {s["task_type"] for s in bank.values()}
    assert "single_event" in task_types
    assert "regime_shift" in task_types
    assert "causal_chain" in task_types


# ---------------------------------------------------------------------------
# Helper (accesses internal for testing)
# ---------------------------------------------------------------------------


def _task_type_for_test(self) -> str:
    return self._scenario.get("task_type", "")


MacroSignalEnvironment._task_type_for_test = _task_type_for_test
