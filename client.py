"""
LegalMind-RL typed client.
Connects to the environment server via HTTP or WebSocket.

Usage (async):
    async with LegalMindEnv(base_url="http://localhost:7860") as env:
        obs = await env.reset(task="medium", role="prosecution")
        result = await env.step(LegalAction(action="argue", content="..."))

Usage (sync):
    with LegalMindEnv(base_url="http://localhost:7860").sync() as env:
        obs = env.reset(task="easy")
        result = env.step(LegalAction(action="argue", content="..."))

From Docker image (official OpenEnv pattern):
    env = await LegalMindEnv.from_docker_image(os.getenv("LOCAL_IMAGE_NAME"))
    async with env:
        obs = await env.reset()
"""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import time
from typing import Optional

import httpx

try:
    import websockets
    _HAS_WS = True
except ImportError:
    _HAS_WS = False

try:
    from .models import LegalAction, LegalObservation, LegalState, StepResult
except ImportError:
    from models import LegalAction, LegalObservation, LegalState, StepResult


def _obs_from_dict(d: dict) -> LegalObservation:
    return LegalObservation(
        case_summary=d["case_summary"],
        current_phase=d["current_phase"],
        round_number=d["round_number"],
        max_rounds=d["max_rounds"],
        available_evidence=d["available_evidence"],
        witnesses=d["witnesses"],
        conversation_history=d["conversation_history"],
        opponent_last_action=d.get("opponent_last_action"),
        valid_actions=d["valid_actions"],
        score_so_far=d["score_so_far"],
        agent_role=d["agent_role"],
        done=d.get("done", False),
        reward=d.get("reward", 0.0),
    )


class LegalMindEnv:
    """Async HTTP client for LegalMind-RL (OpenEnv pattern)."""

    DEFAULT_SESSION = "default"

    def __init__(self, base_url: str = "http://localhost:7860"):
        self.base_url = base_url.rstrip("/")
        self._client: Optional[httpx.AsyncClient] = None
        self.session_id = self.DEFAULT_SESSION
        self._container = None   # for from_docker_image

    async def __aenter__(self):
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=60.0)
        return self

    async def __aexit__(self, *_):
        await self.close()
        if self._client:
            await self._client.aclose()
        if self._container:
            self._container.terminate()

    # ── OpenEnv interface ─────────────────────────────────────────────

    async def reset(self, task: str = "easy", role: str = "prosecution") -> LegalObservation:
        r = await self._client.post("/reset", json={"task": task, "role": role, "session_id": self.session_id})
        r.raise_for_status()
        return _obs_from_dict(r.json()["observation"])

    async def step(self, action: LegalAction) -> StepResult:
        r = await self._client.post("/step", json={
            "session_id": self.session_id,
            "action": action.action,
            "target": action.target,
            "content": action.content,
        })
        r.raise_for_status()
        d = r.json()
        return StepResult(
            observation=_obs_from_dict(d["observation"]),
            reward=d["reward"], done=d["done"],
            info=d.get("info", {}), error=d.get("error"),
        )

    async def state(self) -> LegalState:
        r = await self._client.get(f"/state/{self.session_id}")
        r.raise_for_status()
        d = r.json()
        return LegalState(**d)

    async def score(self) -> float:
        r = await self._client.post("/score", json={"session_id": self.session_id})
        r.raise_for_status()
        return r.json()["score"]

    async def close(self):
        try:
            if self._client:
                await self._client.post(f"/close/{self.session_id}")
        except Exception:
            pass

    # ── Docker factory (official OpenEnv pattern) ─────────────────────

    @classmethod
    async def from_docker_image(
        cls,
        image_name: str,
        port: int = 7860,
        startup_wait: int = 8,
    ) -> "LegalMindEnv":
        """
        Start a Docker container from a local image and return a connected client.
        Mirrors the official OpenEnv from_docker_image() pattern.
        """
        container = subprocess.Popen(
            ["docker", "run", "--rm", "-p", f"{port}:7860",
             "-e", f"PORT=7860", image_name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        # Wait for container to be ready
        url = f"http://localhost:{port}"
        deadline = time.time() + startup_wait + 30
        while time.time() < deadline:
            try:
                r = httpx.get(f"{url}/health", timeout=2)
                if r.status_code == 200:
                    break
            except Exception:
                pass
            await asyncio.sleep(1)

        env = cls(base_url=url)
        env._container = container
        env._client = httpx.AsyncClient(base_url=url, timeout=60.0)
        return env

    # ── Sync wrapper ──────────────────────────────────────────────────

    def sync(self) -> "_SyncLegalMindEnv":
        return _SyncLegalMindEnv(self.base_url)


class _SyncLegalMindEnv:
    """Synchronous wrapper (for inference.py convenience)."""

    def __init__(self, base_url: str):
        self._http = httpx.Client(base_url=base_url, timeout=60.0)
        self.session_id = LegalMindEnv.DEFAULT_SESSION

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
        self._http.close()

    def reset(self, task: str = "easy", role: str = "prosecution") -> LegalObservation:
        r = self._http.post("/reset", json={"task": task, "role": role, "session_id": self.session_id})
        r.raise_for_status()
        return _obs_from_dict(r.json()["observation"])

    def step(self, action: LegalAction) -> StepResult:
        r = self._http.post("/step", json={"session_id": self.session_id, "action": action.action, "target": action.target, "content": action.content})
        r.raise_for_status()
        d = r.json()
        return StepResult(observation=_obs_from_dict(d["observation"]), reward=d["reward"], done=d["done"], info=d.get("info", {}), error=d.get("error"))

    def score(self) -> float:
        r = self._http.post("/score", json={"session_id": self.session_id})
        r.raise_for_status()
        return r.json()["score"]

    def close(self):
        try:
            self._http.post(f"/close/{self.session_id}")
        except Exception:
            pass
