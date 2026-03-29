"""
Macro Signal Engine — Inference Script
=======================================
MANDATORY:
- API_BASE_URL   The API endpoint for the LLM (e.g. https://api.openai.com/v1)
- MODEL_NAME     The model identifier (e.g. gpt-4o, meta-llama/Llama-3-8b-instruct)
- HF_TOKEN       Your Hugging Face / API key

Usage:
    export API_BASE_URL="https://api.openai.com/v1"
    export MODEL_NAME="gpt-4o"
    export HF_TOKEN="your-key-here"
    python inference.py

Optional:
    export ENV_URL="https://krishvenky-macro-signal-env.hf.space"  # override default
    export MAX_STEPS=10  # override per-task step limit
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import textwrap
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI

from src.envs.macro_signal.client import MacroSignalEnv
from src.envs.macro_signal.models import MacroSignalAction, MacroSignalObservation, TradeInstruction

# ---------------------------------------------------------------------------
# Configuration — read from environment variables (NEVER hardcode)
# ---------------------------------------------------------------------------

API_BASE_URL: str = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
API_KEY: str = os.environ.get("HF_TOKEN") or os.environ.get("OPENAI_API_KEY") or "none"
MODEL_NAME: str = os.environ.get("MODEL_NAME", "gpt-4o")
ENV_URL: str = os.environ.get("ENV_URL", "https://krishvenky-macro-signal-env.hf.space")

MAX_STEPS: int = int(os.environ.get("MAX_STEPS", "12"))
TEMPERATURE: float = 0.1
MAX_TOKENS: int = 600
FALLBACK_ACTION = MacroSignalAction(trade_instructions=[], reasoning="Fallback: hold all positions.")

DEBUG: bool = os.environ.get("DEBUG", "0") == "1"

logging.basicConfig(level=logging.DEBUG if DEBUG else logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""
You are a macro quantitative analyst managing a portfolio of 4 assets:
  - SPY  (S&P 500 ETF — broad equity exposure)
  - GLD  (Gold ETF — safe haven / inflation hedge)
  - USO  (Oil ETF — energy / geopolitical hedge)
  - TLT  (Long-duration Treasury ETF — rates / deflation hedge)

At each step you receive:
- signal_events: typed macroeconomic signals with event_type, asset, and magnitude (signed)
- portfolio: your current positions (weight, entry_price, unrealized_pnl)
- cash_weight: fraction not invested
- step_reward: reward earned this step
- benchmark_return: cumulative benchmark return for comparison

CRITICAL RULES:
1. Respond ONLY with valid JSON matching the action schema below.
2. Sum of abs(target_weight) across ALL trade_instructions must be <= 1.0 (no leverage).
3. target_weight is in [-1.0, +1.0]: positive = long, negative = short.
4. Track causal chains: a geopolitical signal at step 1 may imply a supply shock at step 4.
   Enter positions BEFORE consequences materialize for maximum timing_bonus reward.
5. Minimize unnecessary trading — each trade incurs 10bps cost.

Action JSON schema:
{
  "trade_instructions": [
    {"asset": "SPY|GLD|USO|TLT", "target_weight": <float -1.0 to 1.0>, "urgency": "immediate|next_step|hold"}
  ],
  "reasoning": "<your causal reasoning in 1-2 sentences>"
}

If you want to hold all positions: {"trade_instructions": [], "reasoning": "Hold — no new signal."}
""").strip()

# ---------------------------------------------------------------------------
# Observation formatter
# ---------------------------------------------------------------------------


def format_observation(obs: MacroSignalObservation, step: int) -> str:
    lines = [
        f"=== STEP {obs.step}/{obs.max_steps} | Task: {obs.task_type} | Scenario: {obs.scenario_id} ===",
        f"Portfolio NAV: {obs.portfolio_nav:.4f} | Benchmark return: {obs.benchmark_return:+.4f}",
        f"Cash: {obs.cash_weight:.2%} | Step reward so far: {obs.step_reward:.4f}",
    ]

    if obs.signal_events:
        lines.append("\nINCOMING SIGNALS:")
        for ev in obs.signal_events:
            direction = "BULLISH" if ev.magnitude > 0 else "BEARISH"
            lines.append(f"  [{ev.event_type.upper()}] {ev.asset}: magnitude={ev.magnitude:+.2f} ({direction})")
    else:
        lines.append("\nNo new signals this step.")

    if obs.portfolio:
        lines.append("\nCURRENT POSITIONS:")
        for pos in obs.portfolio:
            lines.append(
                f"  {pos.asset}: weight={pos.weight:+.2f} | "
                f"entry={pos.entry_price:.2f} | current={pos.current_price:.2f} | "
                f"PnL={pos.unrealized_pnl:+.4f}"
            )
    else:
        lines.append("\nNo open positions (fully in cash).")

    if obs.info.get("description"):
        lines.append(f"\nScenario: {obs.info['description']}")

    lines.append("\nRespond with your trade action as JSON.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Action parser (with graceful fallback)
# ---------------------------------------------------------------------------


def parse_action(response_text: str) -> MacroSignalAction:
    """Parse LLM response into MacroSignalAction. Falls back to hold on any error."""
    if not response_text:
        return FALLBACK_ACTION

    # Extract JSON block
    json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
    if not json_match:
        logger.warning("No JSON found in response: %s", response_text[:200])
        return FALLBACK_ACTION

    try:
        data = json.loads(json_match.group(0))
        instructions = []
        for instr in data.get("trade_instructions", []):
            asset = instr.get("asset", "")
            if asset not in ("SPY", "GLD", "USO", "TLT"):
                continue
            weight = float(instr.get("target_weight", 0.0))
            weight = max(-1.0, min(1.0, weight))
            urgency = instr.get("urgency", "immediate")
            if urgency not in ("immediate", "next_step", "hold"):
                urgency = "immediate"
            instructions.append(TradeInstruction(asset=asset, target_weight=weight, urgency=urgency))

        # Validate weight sum
        total = sum(abs(i.target_weight) for i in instructions)
        if total > 1.0 + 1e-6:
            scale = 1.0 / total
            instructions = [
                TradeInstruction(asset=i.asset, target_weight=i.target_weight * scale, urgency=i.urgency)
                for i in instructions
            ]
            logger.warning("Scaled down weights by %.4f to satisfy leverage constraint", scale)

        reasoning = str(data.get("reasoning", ""))[:2000]
        return MacroSignalAction(trade_instructions=instructions, reasoning=reasoning)

    except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
        logger.warning("Failed to parse action: %s | Response: %s", e, response_text[:200])
        return FALLBACK_ACTION


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------


async def run_episode(
    client: AsyncOpenAI,
    task_type: str,
    env_url: str,
) -> float:
    """Run one full episode. Returns final episode reward."""
    async with MacroSignalEnv(base_url=env_url) as env:
        result = await env.reset(task_type=task_type)
        obs = result.observation

        logger.info(
            "Episode started: task=%s scenario=%s max_steps=%d",
            obs.task_type,
            obs.scenario_id,
            obs.max_steps,
        )

        conversation: List[Dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]

        step = 0
        final_reward = 0.0

        while not obs.done and step < MAX_STEPS:
            step += 1
            user_content = format_observation(obs, step)
            conversation.append({"role": "user", "content": user_content})

            # LLM call with retry on invalid JSON
            action = FALLBACK_ACTION
            last_error: Optional[str] = None
            for attempt in range(3):
                try:
                    prompt_messages = conversation.copy()
                    if last_error:
                        prompt_messages.append({
                            "role": "user",
                            "content": f"Your previous response was invalid JSON: {last_error}. "
                                       "Please respond with valid JSON only."
                        })

                    completion = await client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=prompt_messages,
                        temperature=TEMPERATURE,
                        max_tokens=MAX_TOKENS,
                        stream=False,
                    )
                    response_text = completion.choices[0].message.content or ""
                    parsed = parse_action(response_text)

                    if parsed is FALLBACK_ACTION and attempt < 2:
                        last_error = "no valid trade_instructions found"
                        continue

                    action = parsed
                    conversation.append({"role": "assistant", "content": response_text})
                    break

                except Exception as e:
                    logger.warning("LLM call failed (attempt %d): %s", attempt + 1, e)
                    last_error = str(e)

            logger.info(
                "Step %d: %d trades | reasoning=%s",
                step,
                len(action.trade_instructions),
                action.reasoning[:80],
            )

            try:
                result = await asyncio.wait_for(env.step(action), timeout=30.0)
            except asyncio.TimeoutError:
                logger.warning("Step timed out, ending episode early")
                break
            obs = result.observation
            final_reward = result.reward

            logger.info(
                "  step_reward=%.4f | cumulative=%.4f | nav=%.4f | done=%s",
                obs.step_reward,
                obs.cumulative_reward,
                obs.portfolio_nav,
                obs.done,
            )

        if obs.done:
            final_reward = obs.reward
            logger.info("Episode complete. Final reward: %.4f", final_reward)
        else:
            logger.warning("Max steps reached without done=True. Using cumulative_reward.")
            final_reward = obs.cumulative_reward

        return final_reward


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    client = AsyncOpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    tasks = [
        ("single_event", "Easy"),
        ("regime_shift", "Medium"),
        ("causal_chain", "Hard"),
    ]

    results: Dict[str, float] = {}

    print("\n" + "=" * 60)
    print("  MACRO SIGNAL ENGINE — BASELINE INFERENCE RUN")
    print("=" * 60)
    print(f"  Model:   {MODEL_NAME}")
    print(f"  Env URL: {ENV_URL}")
    print("=" * 60 + "\n")

    for task_type, difficulty in tasks:
        print(f"Running {difficulty} task: {task_type}...")
        try:
            reward = await run_episode(client, task_type, ENV_URL)
            results[task_type] = reward
            print(f"  {task_type}: reward = {reward:.4f}")
        except Exception as e:
            logger.error("Task %s failed: %s", task_type, e)
            results[task_type] = 0.0
            print(f"  {task_type}: FAILED ({e})")

    print("\n" + "=" * 60)
    print("  FINAL SCORES")
    print("=" * 60)
    for task_type, difficulty in tasks:
        reward = results.get(task_type, 0.0)
        bar = "█" * int(reward * 20)
        print(f"  {difficulty:8s} ({task_type:15s}): {reward:.4f}  |{bar:<20}|")

    mean_reward = sum(results.values()) / len(results) if results else 0.0
    print(f"\n  Mean reward: {mean_reward:.4f}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
