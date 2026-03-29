"""
Macro Signal Engine — Python Client
=====================================
Import MacroSignalEnv in your training code. Never import from server/.
Supports both async (recommended) and sync usage.

Usage (async):
    async with MacroSignalEnv(base_url="https://krishvenky-macro-signal-env.hf.space") as env:
        result = await env.reset(task_type="single_event")
        result = await env.step(MacroSignalAction(...))

Usage (sync):
    with MacroSignalEnv(base_url="...").sync() as env:
        result = env.reset(task_type="single_event")
        result = env.step(MacroSignalAction(...))
"""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

import websockets
from websockets.exceptions import ConnectionClosed

from .models import MacroSignalAction, MacroSignalObservation, MacroSignalState, StepResult

logger = logging.getLogger(__name__)


class MacroSignalEnv:
    """
    Async WebSocket client for the Macro Signal Engine.
    One instance = one persistent session = one environment instance server-side.
    """

    def __init__(self, base_url: str = "https://krishvenky-macro-signal-env.hf.space") -> None:
        self._base_url = base_url.rstrip("/")
        # Convert http(s) to ws(s)
        self._ws_url = (
            self._base_url
            .replace("https://", "wss://")
            .replace("http://", "ws://")
        ) + "/ws"
        self._ws: Optional[Any] = None

    # ------------------------------------------------------------------
    # Async context manager
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "MacroSignalEnv":
        await self._connect()
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self._disconnect()

    async def _connect(self) -> None:
        logger.info("Connecting to %s", self._ws_url)
        self._ws = await websockets.connect(self._ws_url, ping_interval=20, ping_timeout=60)
        logger.info("Connected")

    async def _disconnect(self) -> None:
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None
            logger.info("Disconnected")

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    async def reset(
        self,
        task_type: Optional[str] = None,
        scenario_id: Optional[str] = None,
    ) -> StepResult:
        """Start a new episode. Returns initial observation."""
        msg: Dict[str, Any] = {"type": "reset"}
        if task_type is not None:
            msg["task_type"] = task_type
        if scenario_id is not None:
            msg["scenario_id"] = scenario_id

        response = await self._send_and_receive(msg)
        return self._parse_step_result(response)

    async def step(self, action: MacroSignalAction) -> StepResult:
        """Execute one step. Returns observation, reward, done."""
        msg = {"type": "step", "action": action.model_dump()}
        response = await self._send_and_receive(msg)
        return self._parse_step_result(response)

    async def state(self) -> MacroSignalState:
        """Get current episode state metadata."""
        response = await self._send_and_receive({"type": "state"})
        if response.get("type") == "error":
            raise RuntimeError(response.get("message", "Unknown server error"))
        return MacroSignalState.model_validate(response["data"])

    # ------------------------------------------------------------------
    # Transport
    # ------------------------------------------------------------------

    async def _send_and_receive(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        if self._ws is None:
            await self._connect()

        try:
            await self._ws.send(json.dumps(msg))
            raw = await self._ws.recv()
            return json.loads(raw)
        except ConnectionClosed as e:
            logger.warning("WebSocket connection closed: %s. Reconnecting...", e)
            await self._connect()
            await self._ws.send(json.dumps(msg))
            raw = await self._ws.recv()
            return json.loads(raw)

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_step_result(response: Dict[str, Any]) -> StepResult:
        if response.get("type") == "error":
            raise RuntimeError(response.get("message", "Unknown server error"))
        data = response.get("data", {})
        return StepResult.model_validate(data)

    # ------------------------------------------------------------------
    # Sync wrapper
    # ------------------------------------------------------------------

    def sync(self) -> "SyncMacroSignalEnv":
        """Return a synchronous wrapper for use without async/await."""
        return SyncMacroSignalEnv(self)


class SyncMacroSignalEnv:
    """
    Synchronous wrapper around MacroSignalEnv for non-async training loops.
    """

    def __init__(self, async_env: MacroSignalEnv) -> None:
        self._async_env = async_env
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def __enter__(self) -> "SyncMacroSignalEnv":
        self._loop = asyncio.new_event_loop()
        self._loop.run_until_complete(self._async_env._connect())
        return self

    def __exit__(self, *args: Any) -> None:
        if self._loop is not None:
            self._loop.run_until_complete(self._async_env._disconnect())
            self._loop.close()
            self._loop = None

    def _run(self, coro: Any) -> Any:
        if self._loop is None:
            raise RuntimeError("SyncMacroSignalEnv must be used as a context manager")
        return self._loop.run_until_complete(coro)

    def reset(self, task_type: Optional[str] = None, scenario_id: Optional[str] = None) -> StepResult:
        return self._run(self._async_env.reset(task_type=task_type, scenario_id=scenario_id))

    def step(self, action: MacroSignalAction) -> StepResult:
        return self._run(self._async_env.step(action))

    def state(self) -> MacroSignalState:
        return self._run(self._async_env.state())
