"""
LegalMind-RL FastAPI server.
Implements OpenEnv HTTP + WebSocket endpoints.
WebSocket at /ws is used by the EnvClient for persistent sessions.
HTTP endpoints (/reset, /step, /state, /score) are for stateless / debugging use.
"""
from __future__ import annotations
import json
import os
import sys
import uuid
from typing import Any, Dict, Optional

# Dual-import pattern
try:
    from ..models import LegalAction, LegalObservation, LegalState
    from ..tasks import get_task
    from ..graders import get_grader
    from .legal_environment import LegalEnvironment
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from models import LegalAction, LegalObservation, LegalState
    from tasks import get_task
    from graders import get_grader
    from server.legal_environment import LegalEnvironment

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

# ── Pydantic request bodies for HTTP endpoints ────────────────────────────

class ResetRequest(BaseModel):
    task: str = "easy"
    role: str = "prosecution"
    session_id: str = "default"

class StepRequest(BaseModel):
    session_id: str = "default"
    action: str
    target: Optional[str] = None
    content: str

class ScoreRequest(BaseModel):
    session_id: str = "default"


# ── App factory ───────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    app = FastAPI(
        title="LegalMind-RL",
        description="Courtroom simulation environment for RL agents (OpenEnv compatible).",
        version="1.0.0",
    )
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    # session_id → LegalEnvironment
    _sessions: Dict[str, LegalEnvironment] = {}

    # ── Static / meta ─────────────────────────────────────────────────

    @app.get("/health")
    def health():
        return {"status": "healthy", "env": "LegalMind-RL", "version": "1.0.0"}

    @app.get("/openenv.yaml")
    def openenv_yaml():
        p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "openenv.yaml")
        return FileResponse(p, media_type="text/yaml")

    @app.get("/tasks")
    def list_tasks():
        return {"tasks": [
            {"id": "task_shoplifting_v1", "name": "The Riverside Market Theft",  "difficulty": "easy",   "max_rounds": 8},
            {"id": "task_fraud_v1",       "name": "The Harmon Capital Fraud",    "difficulty": "medium", "max_rounds": 12},
            {"id": "task_homicide_v1",    "name": "The Vantage Hotel Homicide",  "difficulty": "hard",   "max_rounds": 16},
        ]}

    # ── HTTP endpoints (stateless / debugging) ─────────────────────────

    @app.post("/reset")
    def reset(req: ResetRequest):
        try:
            case = get_task(req.task)
        except ValueError as e:
            raise HTTPException(400, str(e))
        env = LegalEnvironment(case=case, agent_role=req.role)
        obs = env.reset()
        _sessions[req.session_id] = env
        return {"session_id": req.session_id, "task": req.task,
                "observation": obs.to_dict(), "state": env.state.to_dict()}

    @app.post("/step")
    def step(req: StepRequest):
        env = _sessions.get(req.session_id)
        if not env:
            raise HTTPException(404, f"Session '{req.session_id}' not found. Call /reset first.")
        action = LegalAction(action=req.action, target=req.target, content=req.content)
        result = env.step(action)
        return {"observation": result.observation.to_dict(), "reward": result.reward,
                "done": result.done, "info": result.info, "error": result.error,
                "state": env.state.to_dict()}

    @app.get("/state/{session_id}")
    def state(session_id: str):
        env = _sessions.get(session_id)
        if not env:
            raise HTTPException(404, f"Session '{session_id}' not found.")
        return env.state.to_dict()

    @app.post("/score")
    def score(req: ScoreRequest):
        env = _sessions.get(req.session_id)
        if not env:
            raise HTTPException(404, f"Session '{req.session_id}' not found.")
        grader = get_grader(env.case.id)
        s = grader(env.state.to_dict())
        return {"session_id": req.session_id, "task": env.case.id, "score": s, "state": env.state.to_dict()}

    @app.post("/close/{session_id}")
    def close(session_id: str):
        env = _sessions.pop(session_id, None)
        return {"session_id": session_id, "closed": env is not None}

    # ── WebSocket endpoint (used by EnvClient for persistent sessions) ─

    @app.websocket("/ws")
    async def websocket_endpoint(ws: WebSocket):
        await ws.accept()
        env: Optional[LegalEnvironment] = None
        session_id = str(uuid.uuid4())

        try:
            while True:
                raw = await ws.receive_text()
                msg = json.loads(raw)
                cmd = msg.get("command", "")

                if cmd == "reset":
                    task_name = msg.get("task", "easy")
                    role      = msg.get("role", "prosecution")
                    try:
                        case = get_task(task_name)
                    except ValueError as e:
                        await ws.send_text(json.dumps({"error": str(e)}))
                        continue
                    env = LegalEnvironment(case=case, agent_role=role)
                    _sessions[session_id] = env
                    obs = env.reset()
                    await ws.send_text(json.dumps({
                        "type": "reset", "session_id": session_id,
                        "observation": obs.to_dict(), "state": env.state.to_dict()
                    }))

                elif cmd == "step":
                    if env is None:
                        await ws.send_text(json.dumps({"error": "Call reset first."}))
                        continue
                    action = LegalAction(
                        action=msg.get("action", "argue"),
                        target=msg.get("target"),
                        content=msg.get("content", ""),
                    )
                    result = env.step(action)
                    await ws.send_text(json.dumps({
                        "type": "step",
                        "observation": result.observation.to_dict(),
                        "reward": result.reward,
                        "done": result.done,
                        "info": result.info,
                        "error": result.error,
                        "state": env.state.to_dict(),
                    }))

                elif cmd == "state":
                    if env is None:
                        await ws.send_text(json.dumps({"error": "Call reset first."}))
                        continue
                    await ws.send_text(json.dumps({"type": "state", "state": env.state.to_dict()}))

                elif cmd == "score":
                    if env is None:
                        await ws.send_text(json.dumps({"error": "Call reset first."}))
                        continue
                    grader = get_grader(env.case.id)
                    s = grader(env.state.to_dict())
                    await ws.send_text(json.dumps({"type": "score", "score": s, "state": env.state.to_dict()}))

                elif cmd == "close":
                    _sessions.pop(session_id, None)
                    await ws.send_text(json.dumps({"type": "close", "session_id": session_id}))
                    break

                else:
                    await ws.send_text(json.dumps({"error": f"Unknown command '{cmd}'"}))

        except WebSocketDisconnect:
            _sessions.pop(session_id, None)

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
