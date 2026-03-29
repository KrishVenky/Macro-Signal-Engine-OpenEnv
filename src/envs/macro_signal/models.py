"""
Macro Signal Engine — Pydantic Models
======================================
Single source of truth for all typed interfaces.
Imported by client.py, server/environment.py, server/app.py, and inference.py.
DO NOT import from any local module here.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


ASSETS = Literal["SPY", "GLD", "USO", "TLT"]
EVENT_TYPES = Literal[
    "equity_shock",
    "commodity_shock",
    "rates_move",
    "geopolitical",
    "inflation_print",
]
TASK_TYPES = Literal["single_event", "regime_shift", "causal_chain"]
URGENCY = Literal["immediate", "next_step", "hold"]


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------


class SignalEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    event_type: EVENT_TYPES = Field(description="Category of macroeconomic signal")
    asset: ASSETS = Field(description="Primary asset affected by this signal")
    magnitude: float = Field(
        ge=-1.0,
        le=1.0,
        description="Signed directional strength: positive = bullish, negative = bearish",
    )
    step: int = Field(ge=1, description="Environment step when this event was generated")


class PortfolioPosition(BaseModel):
    model_config = ConfigDict(frozen=True)

    asset: ASSETS = Field(description="Asset ticker")
    weight: float = Field(
        ge=-1.0,
        le=1.0,
        description="Portfolio weight as fraction of NAV. Negative = short.",
    )
    entry_price: float = Field(gt=0.0, description="Price at which position was entered")
    current_price: float = Field(gt=0.0, description="Current market price")
    unrealized_pnl: float = Field(description="Unrealized PnL as fraction of NAV")


class TradeInstruction(BaseModel):
    model_config = ConfigDict(frozen=True)

    asset: ASSETS = Field(description="Asset to trade")
    target_weight: float = Field(
        ge=-1.0,
        le=1.0,
        description="Desired portfolio weight after trade. Negative = short.",
    )
    urgency: URGENCY = Field(
        default="immediate",
        description="Execution timing preference",
    )


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------


class MacroSignalAction(BaseModel):
    model_config = ConfigDict(frozen=True)

    trade_instructions: List[TradeInstruction] = Field(
        default_factory=list,
        description="List of target weight changes. Empty = hold all positions.",
    )
    reasoning: str = Field(
        default="",
        max_length=2000,
        description="Agent reasoning (used by causal_chain grader for timing analysis)",
    )

    @field_validator("trade_instructions")
    @classmethod
    def validate_weight_sum(cls, instructions: List[TradeInstruction]) -> List[TradeInstruction]:
        total = sum(abs(t.target_weight) for t in instructions)
        if total > 1.0 + 1e-6:
            raise ValueError(
                f"Sum of absolute target weights {total:.4f} exceeds 1.0 (no leverage allowed)"
            )
        return instructions


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------


class MacroSignalObservation(BaseModel):
    model_config = ConfigDict(frozen=True)

    # Episode context
    step: int = Field(ge=0, description="Current step index (0 = post-reset)")
    max_steps: int = Field(ge=1, description="Maximum steps in this episode")
    task_type: TASK_TYPES = Field(description="Task difficulty tier")
    scenario_id: str = Field(description="Scenario identifier from scenario bank")

    # What the agent sees
    signal_events: List[SignalEvent] = Field(
        default_factory=list,
        description="Signal events visible at this step (may be empty)",
    )
    portfolio: List[PortfolioPosition] = Field(
        default_factory=list,
        description="Current non-zero positions (cash not included)",
    )
    cash_weight: float = Field(
        ge=0.0,
        le=1.0,
        description="Fraction of NAV held as cash",
    )
    portfolio_nav: float = Field(gt=0.0, description="Current total portfolio NAV")

    # Benchmark context (for regime_shift grading)
    benchmark_return: float = Field(
        default=0.0,
        description="Cumulative benchmark return since episode start",
    )

    # Reward signals
    step_reward: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Partial reward earned at this step",
    )
    cumulative_reward: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Total reward accumulated so far this episode",
    )

    # Terminal signals
    done: bool = Field(default=False, description="Whether this is the final step")
    reward: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Final episode reward (meaningful only when done=True)",
    )

    # Metadata for agents
    info: Dict[str, Any] = Field(
        default_factory=dict,
        description="Grader metadata, hints, and diagnostics",
    )


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


class MacroSignalState(BaseModel):
    model_config = ConfigDict(frozen=True)

    episode_id: str = Field(description="Unique identifier for this episode session")
    task_type: TASK_TYPES
    scenario_id: str
    step_count: int = Field(ge=0)
    max_steps: int = Field(ge=1)
    portfolio_nav: float = Field(gt=0.0)
    cumulative_reward: float = Field(ge=0.0, le=1.0)
    done: bool


# ---------------------------------------------------------------------------
# StepResult (returned by client.step() and client.reset())
# ---------------------------------------------------------------------------


class StepResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    observation: MacroSignalObservation
    reward: float = Field(ge=0.0, le=1.0)
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)
