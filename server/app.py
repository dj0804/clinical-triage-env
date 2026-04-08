"""
Clinical Triage AI — FastAPI Server

Implements the OpenEnv HTTP+WebSocket protocol:
  POST   /reset       → initialize episode
  POST   /step        → execute action
  GET    /state       → internal state
  GET    /schema      → action/observation JSON schemas
  GET    /health      → health check
  WS     /ws          → WebSocket real-time interaction
"""

import json
import sys
import os
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError

# Adjust import path for Docker/HF Space execution
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import (
    TriageAction,
    TriageObservation,
    AskQuestionAction,
    RequestTestAction,
    MakeDiagnosisAction,
    AssignRiskAction,
    RoutePatientAction,
    EscalateToHumanAction,
    FinalizeAction,
)
from server.environment import TriageEnvironment


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Clinical Triage AI — OpenEnv Environment",
    description=(
        "A multi-step clinical triage simulation environment for AI agent evaluation. "
        "Implements OpenEnv HTTP+WebSocket protocol with deterministic rewards."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Single shared environment instance (per-process, per-session via WebSocket)
_env = TriageEnvironment()


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    seed: Optional[int] = None
    episode_id: Optional[str] = None
    task_name: str = "easy_triage"


class StepRequest(BaseModel):
    action: Dict[str, Any]
    timeout_s: Optional[float] = None


class ResetResponse(BaseModel):
    observation: Dict[str, Any]
    reward: Optional[float] = None
    done: bool = False
    info: Dict[str, Any] = {}


class StepResponse(BaseModel):
    observation: Dict[str, Any]
    reward: float
    done: bool
    info: Dict[str, Any]


# ---------------------------------------------------------------------------
# Helper: build action from raw dict
# ---------------------------------------------------------------------------

def _parse_action(raw: Dict[str, Any]) -> TriageAction:
    """Parse raw dict into a typed TriageAction using discriminator."""
    action_type = raw.get("action_type")
    if not action_type:
        raise ValueError("Missing 'action_type' field in action payload.")

    dispatch = {
        "ask_question": AskQuestionAction,
        "request_test": RequestTestAction,
        "make_diagnosis": MakeDiagnosisAction,
        "assign_risk": AssignRiskAction,
        "route_patient": RoutePatientAction,
        "escalate_to_human": EscalateToHumanAction,
        "finalize": FinalizeAction,
    }

    cls = dispatch.get(action_type)
    if cls is None:
        raise ValueError(f"Unknown action_type: '{action_type}'. Valid: {list(dispatch.keys())}")

    return cls(**raw)


# ---------------------------------------------------------------------------
# HTTP endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> JSONResponse:
    """Health check endpoint."""
    return JSONResponse({"status": "healthy", "env": "clinical_triage_env"})


@app.get("/schema")
async def schema() -> JSONResponse:
    """Return JSON schemas for Action and Observation."""
    # Build union schema manually for clarity
    action_schema = {
        "oneOf": [
            AskQuestionAction.model_json_schema(),
            RequestTestAction.model_json_schema(),
            MakeDiagnosisAction.model_json_schema(),
            AssignRiskAction.model_json_schema(),
            RoutePatientAction.model_json_schema(),
            EscalateToHumanAction.model_json_schema(),
            FinalizeAction.model_json_schema(),
        ],
        "discriminator": "action_type",
    }
    return JSONResponse({
        "action": action_schema,
        "observation": TriageObservation.model_json_schema(),
    })


@app.post("/reset", response_model=ResetResponse)
async def reset(req: ResetRequest) -> ResetResponse:
    """Initialize or restart the episode."""
    obs = _env.reset(
        seed=req.seed,
        episode_id=req.episode_id,
        task_name=req.task_name,
    )
    info = {"status": "reset_ok"}
    if obs.done:
        info = {"error": obs.metadata.get("error", "Unknown error")}

    return ResetResponse(
        observation=obs.model_dump(),
        reward=None,
        done=obs.done,
        info=info,
    )


@app.post("/step", response_model=StepResponse)
async def step(req: StepRequest) -> StepResponse:
    """Execute an action and return the result."""
    try:
        action = _parse_action(req.action)
    except (ValueError, ValidationError) as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    obs, reward, done, info = _env.step(action)
    return StepResponse(
        observation=obs.model_dump(),
        reward=reward,
        done=done,
        info=info,
    )


@app.get("/state")
async def get_state() -> JSONResponse:
    """Return full internal episode state (for debugging)."""
    state = _env.state
    return JSONResponse(state.model_dump())


@app.get("/")
async def root() -> JSONResponse:
    return JSONResponse({
        "name": "clinical_triage_env",
        "version": "1.0.0",
        "endpoints": ["/reset", "/step", "/state", "/schema", "/health", "/ws"],
        "tasks": ["easy_triage", "medium_triage", "hard_triage"],
    })


# ---------------------------------------------------------------------------
# WebSocket endpoint — per-session isolated environments
# ---------------------------------------------------------------------------

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """
    WebSocket handler for real-time interaction.

    Each WebSocket connection gets its OWN environment instance.

    Message format (JSON):
        Reset:  {"type": "reset", "data": {"task_name": "easy_triage", "seed": 0}}
        Step:   {"type": "step",  "data": {"action_type": "ask_question", "question": "..."}}
        State:  {"type": "state"}
        Close:  {"type": "close"}

    Response format:
        Observation: {"type": "observation", "data": {...}}
        State:       {"type": "state",       "data": {...}}
        Error:       {"type": "error",       "data": {"message": "..."}}
    """
    await websocket.accept()
    # Per-session environment
    session_env = TriageEnvironment()

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_json(
                    {"type": "error", "data": {"message": "Invalid JSON"}}
                )
                continue

            msg_type = msg.get("type")

            if msg_type == "reset":
                data = msg.get("data", {})
                obs = session_env.reset(
                    seed=data.get("seed"),
                    episode_id=data.get("episode_id"),
                    task_name=data.get("task_name", "easy_triage"),
                )
                info = {"status": "reset_ok"}
                if obs.done:
                    info = {"error": obs.metadata.get("error", "Unknown error")}
                await websocket.send_json({
                    "type": "observation", 
                    "data": {
                        **obs.model_dump(),
                        "reward": None,
                        "done": obs.done,
                        "info": info
                    }
                })

            elif msg_type == "step":
                data = msg.get("data", {})
                try:
                    action = _parse_action(data)
                except (ValueError, ValidationError) as exc:
                    await websocket.send_json(
                        {"type": "error", "data": {"message": str(exc)}}
                    )
                    continue

                obs, reward, done, info = session_env.step(action)
                await websocket.send_json({
                    "type": "observation",
                    "data": {
                        **obs.model_dump(),
                        "reward": reward,
                        "done": done,
                        "info": info,
                    },
                })

            elif msg_type == "state":
                state = session_env.state
                await websocket.send_json(
                    {"type": "state", "data": state.model_dump()}
                )

            elif msg_type == "close":
                await websocket.close()
                break

            else:
                await websocket.send_json(
                    {"type": "error", "data": {"message": f"Unknown message type: {msg_type}"}}
                )

    except WebSocketDisconnect:
        pass  # Client disconnected cleanly


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
