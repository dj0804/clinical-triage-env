"""
Clinical Triage AI — Inference Script

Evaluates an AI agent on all 3 triage tasks via the OpenEnv HTTP API.

Environment variables:
    API_BASE_URL   — Base URL of the OpenAI-compatible endpoint (default: huggingface router)
    MODEL_NAME     — LLM model identifier (e.g., "gpt-4o-mini")
    HF_TOKEN       — Hugging Face token required for authentication
    ENV_BASE_URL   — Base URL of the clinical triage env server (default: http://localhost:7860)

Output format (strict):
    [START] task=<task_name> env=clinical_triage model=<model_name>
    [STEP] step=<n> action=<action_type> reward=<float> done=<bool> error=<msg|null>
    [END] success=<bool> steps=<n> rewards=<r1,r2,...>
"""

import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import requests

try:
    from openai import OpenAI
except ImportError:
    print("openai package not installed. Run: pip install openai", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE_URL: str = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME: str = os.environ.get("MODEL_NAME", "meta-llama/Meta-Llama-3-8B-Instruct")
HF_TOKEN: str = os.environ.get("HF_TOKEN", "")
ENV_BASE_URL: str = os.environ.get("ENV_BASE_URL", "http://localhost:7860")

MAX_STEPS_PER_TASK: int = 15
MAX_TASK_RUNTIME_SECONDS: int = 360  # 6 minutes per task = 18 min total < 20 min

TASKS: List[str] = ["easy_triage", "medium_triage", "hard_triage"]


# ---------------------------------------------------------------------------
# Environment Client
# ---------------------------------------------------------------------------

class TriageEnvClient:
    """HTTP client for the Clinical Triage environment."""

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")
        self._session = requests.Session()

    def reset(self, task_name: str, seed: int = 0) -> Dict[str, Any]:
        resp = self._session.post(
            f"{self.base_url}/reset",
            json={"task_name": task_name, "seed": seed},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        resp = self._session.post(
            f"{self.base_url}/step",
            json={"action": action},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    def get_state(self) -> Dict[str, Any]:
        resp = self._session.get(f"{self.base_url}/state", timeout=30)
        resp.raise_for_status()
        return resp.json()

    def health(self) -> bool:
        try:
            resp = self._session.get(f"{self.base_url}/health", timeout=10)
            return resp.status_code == 200
        except Exception:
            return False


# ---------------------------------------------------------------------------
# Agent (OpenAI-based)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a clinical triage AI agent. You receive partial patient data and must:
1. Gather information via questions and tests
2. Make a diagnosis with a confidence score (0.0-1.0)
3. Assign a risk level: Critical, Monitor, or Routine
4. Route the patient to the appropriate department
5. Optionally escalate to a human clinician if uncertain
6. Finalize the triage when complete

Always respond with EXACTLY ONE JSON action from the allowed action types:

{"action_type": "ask_question", "question": "<your question>"}
{"action_type": "request_test", "test_name": "<test name, e.g. troponin, ECG, CBC>"}
{"action_type": "make_diagnosis", "diagnosis": "<diagnosis>", "confidence": <0.0-1.0>}
{"action_type": "assign_risk", "risk_level": "<Critical|Monitor|Routine>"}
{"action_type": "route_patient", "department": "<department name>"}
{"action_type": "escalate_to_human", "reason": "<reason for escalation>"}
{"action_type": "finalize"}

Rules:
- Gather at least 1-2 pieces of information before making a diagnosis
- Only route AFTER making a diagnosis
- Use finalize only after diagnosis + risk are set
- If the case is ambiguous or you are uncertain, escalate_to_human is appropriate
- Do not repeat the same action
"""


def build_user_message(observation: Dict[str, Any], step: int) -> str:
    """Construct agent prompt from current observation."""
    obs = observation.get("observation", observation)
    lines = [f"=== STEP {step} ==="]

    symptoms = obs.get("symptoms", [])
    if symptoms:
        lines.append(f"SYMPTOMS: {', '.join(symptoms)}")

    history = obs.get("patient_history", {})
    if history:
        lines.append(f"PATIENT HISTORY: {json.dumps(history)}")

    tests = obs.get("revealed_tests", {})
    if tests:
        lines.append("REVEALED TESTS:")
        for name, result in tests.items():
            lines.append(f"  {name}: {result}")

    prev = obs.get("previous_actions", [])
    if prev:
        lines.append(f"PREVIOUS ACTIONS: {', '.join(prev[-3:])}")  # last 3 only

    meta = obs.get("metadata", {})
    lines.append(f"Has diagnosis: {meta.get('has_diagnosis', False)}")
    lines.append(f"Has risk: {meta.get('has_risk', False)}")
    lines.append(f"Max steps: {meta.get('max_steps', 15)}")

    lines.append("\nRespond with exactly one JSON action.")
    return "\n".join(lines)


def extract_action(raw_response: str) -> Optional[Dict[str, Any]]:
    """Extract JSON action from model response."""
    raw = raw_response.strip()
    # Try direct JSON parse
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Try extracting first JSON object from the text
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(raw[start:end + 1])
        except json.JSONDecodeError:
            pass

    return None


def run_agent_step(
    client: OpenAI,
    messages: List[Dict[str, str]],
    observation: Dict[str, Any],
    step: int,
) -> Tuple[Optional[Dict[str, Any]], str]:
    """Get next action from the LLM agent."""
    user_msg = build_user_message(observation, step)
    messages.append({"role": "user", "content": user_msg})

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.0,  # deterministic
        max_tokens=256,
    )

    raw = response.choices[0].message.content or ""
    messages.append({"role": "assistant", "content": raw})

    action = extract_action(raw)
    return action, raw


# ---------------------------------------------------------------------------
# Task runner
# ---------------------------------------------------------------------------

def run_task(
    agent: OpenAI,
    env_client: TriageEnvClient,
    task_name: str,
    seed: int = 0,
) -> Dict[str, Any]:
    """
    Run one task episode.

    Returns:
        {
            "task": str,
            "success": bool,
            "steps": int,
            "rewards": List[float],
            "final_score": float,
            "error": Optional[str],
        }
    """
    print(f"[START] task={task_name} env=clinical_triage model={MODEL_NAME}", flush=True)

    rewards: List[float] = []
    step = 0
    error_msg: Optional[str] = None
    final_score: float = 0.0
    success = False
    task_start = time.time()

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    try:
        # Reset
        obs = env_client.reset(task_name=task_name, seed=seed)
        done = obs.get("done", False)

        while not done and step < MAX_STEPS_PER_TASK:
            # Runtime guard
            elapsed = time.time() - task_start
            if elapsed > MAX_TASK_RUNTIME_SECONDS:
                error_msg = f"Task runtime exceeded {MAX_TASK_RUNTIME_SECONDS}s"
                break

            step += 1

            # Agent decides action
            action, _raw = run_agent_step(agent, messages, obs, step)

            if action is None:
                # Model returned unparseable output — use finalize as fallback
                action = {"action_type": "finalize"}
                error_msg = f"Could not parse model output at step {step}"

            # Execute action
            try:
                result = env_client.step(action)
            except requests.HTTPError as exc:
                error_msg = f"HTTP error at step {step}: {exc}"
                break

            reward = result.get("reward", 0.0)
            done = result.get("done", False)
            info = result.get("info", {})
            rewards.append(round(reward, 4))

            step_error = info.get("error")
            print(
                f"[STEP] step={step} action={action.get('action_type','unknown')} "
                f"reward={reward:.4f} done={str(done).lower()} "
                f"error={step_error if step_error else 'null'}",
                flush=True,
            )

            obs = result

            # Extract final score on done
            if done and "final_grade" in info:
                final_score = info["final_grade"].get("score", 0.0)
                success = True

    except Exception as exc:
        error_msg = str(exc)

    rewards_str = ",".join(str(r) for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={step} rewards={rewards_str}",
        flush=True,
    )

    return {
        "task": task_name,
        "success": success,
        "steps": step,
        "rewards": rewards,
        "final_score": final_score,
        "error": error_msg,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Validate environment setup
    hf_token = os.getenv("HF_TOKEN", HF_TOKEN)
    if not hf_token:
        print("ERROR: HF_TOKEN environment variable not set.", file=sys.stderr)
        sys.exit(1)

    # Configure OpenAI client pointing to HF Router
    agent = OpenAI(
        base_url=os.getenv("API_BASE_URL", API_BASE_URL),
        api_key=hf_token
    )

    # Verify environment server is running
    env_client = TriageEnvClient(ENV_BASE_URL)
    if not env_client.health():
        print(f"ERROR: Environment server at {ENV_BASE_URL} is not reachable.", file=sys.stderr)
        sys.exit(1)

    print(f"Connected to environment at {ENV_BASE_URL}", flush=True)
    print(f"Model: {MODEL_NAME}", flush=True)

    results = []
    for task in TASKS:
        result = run_task(agent, env_client, task, seed=0)
        results.append(result)

    # Summary
    print("\n=== EVALUATION SUMMARY ===", flush=True)
    total_score = 0.0
    for r in results:
        score = r["final_score"]
        total_score += score
        print(
            f"  {r['task']}: score={score:.4f} steps={r['steps']} "
            f"success={r['success']} error={r['error'] or 'null'}",
            flush=True,
        )
    print(f"  MEAN SCORE: {total_score / len(results):.4f}", flush=True)


if __name__ == "__main__":
    main()
