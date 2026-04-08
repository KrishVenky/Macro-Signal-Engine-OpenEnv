"""
Macro Signal Engine: Inference Script

Required environment variables:
    API_BASE_URL   LLM API endpoint (default: https://api.openai.com/v1)
    MODEL_NAME     Model identifier  (default: gpt-4o)
    HF_TOKEN       API key — mandatory, no default

Usage:
    export API_BASE_URL="https://api.groq.com/openai/v1"
    export MODEL_NAME="llama-3.3-70b-versatile"
    export HF_TOKEN="your-key-here"
    python inference.py
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import textwrap
from typing import Any, Dict, List, Optional

from openai import OpenAI

from src.envs.macro_signal.client import MacroSignalEnv
from src.envs.macro_signal.models import MacroSignalAction, MacroSignalObservation, TradeInstruction

# Environment variables — API_BASE_URL and MODEL_NAME have defaults, HF_TOKEN does not
API_BASE_URL: str = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME: str = os.getenv("MODEL_NAME", "gpt-4o")
HF_TOKEN: Optional[str] = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

ENV_URL: str = os.getenv("ENV_URL", "https://krishvenky-macro-signal-env.hf.space")
MAX_STEPS: int = int(os.getenv("MAX_STEPS", "12"))

# Sync OpenAI client as required by competition spec
client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

TEMPERATURE: float = 0.1
MAX_TOKENS: int = 600
FALLBACK_ACTION = MacroSignalAction(trade_instructions=[], reasoning="Hold, no new signal.")

SYSTEM_PROMPT = textwrap.dedent("""
You are a macro quantitative analyst managing a portfolio of 4 assets:
  - SPY  (S&P 500 ETF, broad equity exposure)
  - GLD  (Gold ETF, safe haven / inflation hedge)
  - USO  (Oil ETF, energy / geopolitical hedge)
  - TLT  (Long-duration Treasury ETF, rates / deflation hedge)

At each step you receive signal_events, portfolio positions, cash_weight, and step_reward.

RULES:
1. Respond ONLY with valid JSON matching the action schema.
2. Sum of abs(target_weight) across all instructions must be <= 1.0.
3. target_weight in [-1.0, +1.0]: positive = long, negative = short.
4. Track causal chains: a geopolitical signal at step 1 may cause a supply shock at step 4.
   Position BEFORE consequences arrive for maximum timing_bonus reward.
5. Minimize trading — each trade costs 10bps.

Action schema:
{
  "trade_instructions": [
    {"asset": "SPY|GLD|USO|TLT", "target_weight": <float>, "urgency": "immediate|next_step|hold"}
  ],
  "reasoning": "<causal reasoning in 1-2 sentences>"
}

To hold: {"trade_instructions": [], "reasoning": "Hold, no new signal."}
""").strip()


def format_observation(obs: MacroSignalObservation, step: int) -> str:
    lines = [
        f"STEP {obs.step}/{obs.max_steps} | Task: {obs.task_type} | Scenario: {obs.scenario_id}",
        f"NAV: {obs.portfolio_nav:.4f} | Benchmark: {obs.benchmark_return:+.4f} | Cash: {obs.cash_weight:.2%}",
    ]
    if obs.signal_events:
        lines.append("SIGNALS:")
        for ev in obs.signal_events:
            direction = "BULLISH" if ev.magnitude > 0 else "BEARISH"
            lines.append(f"  [{ev.event_type.upper()}] {ev.asset}: {ev.magnitude:+.2f} ({direction})")
    else:
        lines.append("No new signals.")
    if obs.portfolio:
        lines.append("POSITIONS:")
        for pos in obs.portfolio:
            lines.append(f"  {pos.asset}: weight={pos.weight:+.2f} pnl={pos.unrealized_pnl:+.4f}")
    lines.append("Respond with JSON action.")
    return "\n".join(lines)


def parse_action(response_text: str) -> MacroSignalAction:
    if not response_text:
        return FALLBACK_ACTION
    json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
    if not json_match:
        return FALLBACK_ACTION
    try:
        data = json.loads(json_match.group(0))
        instructions = []
        for instr in data.get("trade_instructions", []):
            asset = instr.get("asset", "")
            if asset not in ("SPY", "GLD", "USO", "TLT"):
                continue
            weight = max(-1.0, min(1.0, float(instr.get("target_weight", 0.0))))
            urgency = instr.get("urgency", "immediate")
            if urgency not in ("immediate", "next_step", "hold"):
                urgency = "immediate"
            instructions.append(TradeInstruction(asset=asset, target_weight=weight, urgency=urgency))
        total = sum(abs(i.target_weight) for i in instructions)
        if total > 1.0 + 1e-6:
            scale = 1.0 / total
            instructions = [
                TradeInstruction(asset=i.asset, target_weight=i.target_weight * scale, urgency=i.urgency)
                for i in instructions
            ]
        reasoning = str(data.get("reasoning", ""))[:500]
        return MacroSignalAction(trade_instructions=instructions, reasoning=reasoning)
    except (json.JSONDecodeError, KeyError, ValueError, TypeError):
        return FALLBACK_ACTION


def action_to_str(action: MacroSignalAction) -> str:
    if not action.trade_instructions:
        return "hold()"
    parts = [f"{i.asset}:{i.target_weight:+.2f}" for i in action.trade_instructions]
    return "trade(" + ",".join(parts) + ")"


async def run_episode(task_type: str, env_url: str) -> None:
    step_rewards: List[float] = []
    steps_taken = 0
    success = False

    print(f"[START] task={task_type} env=macro-signal-env model={MODEL_NAME}", flush=True)

    try:
        async with MacroSignalEnv(base_url=env_url) as env:
            result = await env.reset(task_type=task_type)
            obs = result.observation

            conversation: List[Dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]

            while not obs.done and steps_taken < MAX_STEPS:
                steps_taken += 1
                user_content = format_observation(obs, steps_taken)
                conversation.append({"role": "user", "content": user_content})

                action = FALLBACK_ACTION
                for attempt in range(3):
                    try:
                        completion = client.chat.completions.create(
                            model=MODEL_NAME,
                            messages=conversation,
                            temperature=TEMPERATURE,
                            max_tokens=MAX_TOKENS,
                        )
                        response_text = completion.choices[0].message.content or ""
                        parsed = parse_action(response_text)
                        if parsed is not FALLBACK_ACTION or attempt == 2:
                            action = parsed
                            conversation.append({"role": "assistant", "content": response_text})
                            break
                    except Exception as e:
                        if attempt == 2:
                            break

                error_str = "null"
                try:
                    result = await asyncio.wait_for(env.step(action), timeout=30.0)
                except asyncio.TimeoutError:
                    error_str = "timeout"
                    print(f"[STEP] step={steps_taken} action={action_to_str(action)} reward=0.01 done=false error={error_str}", flush=True)
                    step_rewards.append(0.01)
                    break

                obs = result.observation
                # Use the final episode reward when done, step reward otherwise
                reward = result.reward if obs.done else obs.step_reward
                step_rewards.append(reward)
                done_str = "true" if obs.done else "false"

                print(f"[STEP] step={steps_taken} action={action_to_str(action)} reward={reward:.2f} done={done_str} error={error_str}", flush=True)

                if obs.done:
                    success = reward > 0.0
                    break

    except Exception as e:
        error_msg = str(e).replace("\n", " ")[:100]
        print(f"[STEP] step={steps_taken} action=error reward=0.01 done=false error={error_msg}", flush=True)

    finally:
        rewards_str = ",".join(f"{r:.2f}" for r in step_rewards) if step_rewards else "0.01"
        print(f"[END] success={str(success).lower()} steps={steps_taken} rewards={rewards_str}", flush=True)


async def main() -> None:
    tasks = [
        ("single_event", "Easy"),
        ("regime_shift", "Medium"),
        ("causal_chain", "Hard"),
    ]
    for task_type, difficulty in tasks:
        await run_episode(task_type, ENV_URL)


if __name__ == "__main__":
    asyncio.run(main())
