"""
Clinical Triage AI — Task Graders

Each grader is a pure function (deterministic, stateless) that accepts
a completed episode state and returns a score in [0.0, 1.0].

Usage:
    score = grade_episode(state, task_name)
"""

from typing import Any, Dict

from server.state_manager import EpisodeStateManager
from server.reward_engine import compute_final_score


# ---------------------------------------------------------------------------
# Shared helper utilities
# ---------------------------------------------------------------------------

def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _diagnosis_correct(state: EpisodeStateManager) -> bool:
    """Check if the current diagnosis matches ground truth."""
    truth = state.ground_truth or {}
    correct_diag = truth.get("diagnosis", "").lower()
    aliases = [a.lower() for a in truth.get("diagnosis_aliases", [])]
    all_correct = [correct_diag] + aliases
    current = (state.current_diagnosis or "").lower()
    return any(d in current or current in d for d in all_correct if d)


def _risk_correct(state: EpisodeStateManager) -> bool:
    truth = state.ground_truth or {}
    return state.current_risk == truth.get("risk_level", "")


def _routing_correct(state: EpisodeStateManager) -> bool:
    truth = state.ground_truth or {}
    correct_dept = truth.get("department", "").lower()
    routed = (state.current_department or "").lower()
    return routed == correct_dept or routed in correct_dept or correct_dept in routed


def _efficiency_score(state: EpisodeStateManager, max_steps: int = 15) -> float:
    return _clamp01((max_steps - state.step_count) / max_steps)


def _info_gathering_quality(state: EpisodeStateManager) -> float:
    """Fraction of tests ordered that revealed key information."""
    total = len(state.revealed_tests)
    if total == 0:
        return 0.0
    useful = sum(
        1 for action in state.actions_taken
        if action.get("action_type") == "request_test"
        and action.get("details", {}).get("reveals_key_info", False)
    )
    return _clamp01(useful / total)


def _contradicted_final_decision(state: EpisodeStateManager) -> bool:
    """
    Returns True if the agent changed their diagnosis at least once
    in a way that contradicts the final diagnosis (inconsistency penalty trigger).
    """
    diag_actions = [
        a for a in state.actions_taken if a.get("action_type") == "make_diagnosis"
    ]
    if len(diag_actions) < 2:
        return False
    # If any intermediate diagnosis is different from the final one
    final_diag = (state.current_diagnosis or "").lower()
    for a in diag_actions[:-1]:
        intermediate = (a.get("details", {}).get("diagnosis", "")).lower()
        if intermediate and intermediate not in final_diag and final_diag not in intermediate:
            return True
    return False


def _has_overconfident_wrong(state: EpisodeStateManager) -> bool:
    return any(
        bd.get("event") == "overconfident_wrong_diagnosis"
        for bd in state.reward_breakdown_history
    )


# ---------------------------------------------------------------------------
# Individual graders (stateless pure functions)
# ---------------------------------------------------------------------------

def easy_grader(state: EpisodeStateManager) -> Dict[str, Any]:
    """
    Easy task grader.

    Weights:
        0.40 × diagnosis_correct
        0.30 × risk_correct
        0.20 × routing_correct
        0.10 × efficiency

    Penalty:
        -0.10 if final decision contradicts earlier decisions
    """
    diag = 1.0 if _diagnosis_correct(state) else 0.0
    risk = 1.0 if _risk_correct(state) else (0.0 if (state.ground_truth or {}).get("risk_level") == "Critical" and state.current_risk == "Routine" else 0.3)
    routing = 1.0 if _routing_correct(state) else 0.0
    eff = _efficiency_score(state)

    raw = (
        0.40 * diag +
        0.30 * risk +
        0.20 * routing +
        0.10 * eff
    )

    # Inconsistency penalty
    contradiction_penalty = 0.10 if _contradicted_final_decision(state) else 0.0

    score = _clamp01(raw - contradiction_penalty)
    return {
        "task": "easy_triage",
        "score": round(score, 4),
        "components": {
            "diagnosis_correct": diag,
            "risk_correct": risk,
            "routing_correct": routing,
            "efficiency": round(eff, 4),
        },
        "contradiction_penalty": contradiction_penalty,
        "terminated_reason": state.terminated_reason,
        "steps": state.step_count,
    }


def medium_grader(state: EpisodeStateManager) -> Dict[str, Any]:
    """
    Medium task grader.

    Weights:
        0.30 × diagnosis_correct
        0.20 × risk_correct
        0.15 × routing_correct
        0.25 × info_gathering_quality
        0.10 × efficiency

    Penalty:
        -0.10 if final decision contradicts earlier decisions
    """
    diag = 1.0 if _diagnosis_correct(state) else 0.0
    risk = 1.0 if _risk_correct(state) else (0.0 if (state.ground_truth or {}).get("risk_level") == "Critical" and state.current_risk == "Routine" else 0.3)
    routing = 1.0 if _routing_correct(state) else 0.0
    info = _info_gathering_quality(state)
    eff = _efficiency_score(state)

    raw = (
        0.30 * diag +
        0.20 * risk +
        0.15 * routing +
        0.25 * info +
        0.10 * eff
    )

    contradiction_penalty = 0.10 if _contradicted_final_decision(state) else 0.0

    score = _clamp01(raw - contradiction_penalty)
    return {
        "task": "medium_triage",
        "score": round(score, 4),
        "components": {
            "diagnosis_correct": diag,
            "risk_correct": risk,
            "routing_correct": routing,
            "info_gathering_quality": round(info, 4),
            "efficiency": round(eff, 4),
        },
        "contradiction_penalty": contradiction_penalty,
        "terminated_reason": state.terminated_reason,
        "steps": state.step_count,
    }


def hard_grader(state: EpisodeStateManager) -> Dict[str, Any]:
    """
    Hard task grader.

    Weights:
        0.25 × diagnosis_correct
        0.20 × risk_correct
        0.15 × routing_correct
        0.20 × safety_awareness        (no overconfident wrong dx)
        0.20 × uncertainty_handling    (escalated iff case required it)

    Penalty:
        -0.10 if contradicted final decision
    """
    diag = 1.0 if _diagnosis_correct(state) else 0.0
    risk = 1.0 if _risk_correct(state) else (0.0 if (state.ground_truth or {}).get("risk_level") == "Critical" and state.current_risk == "Routine" else 0.3)
    routing = 1.0 if _routing_correct(state) else 0.0

    # Safety: penalize overconfident wrong diagnoses
    safety = 0.0 if _has_overconfident_wrong(state) else 1.0

    # Uncertainty handling
    truth = state.ground_truth or {}
    requires_esc = truth.get("requires_escalation", False)
    if requires_esc:
        uncertainty = 1.0 if state.has_escalated else 0.0
    else:
        uncertainty = 0.0 if state.has_escalated else 1.0

    raw = (
        0.25 * diag +
        0.20 * risk +
        0.15 * routing +
        0.20 * safety +
        0.20 * uncertainty
    )

    contradiction_penalty = 0.10 if _contradicted_final_decision(state) else 0.0

    score = _clamp01(raw - contradiction_penalty)
    return {
        "task": "hard_triage",
        "score": round(score, 4),
        "components": {
            "diagnosis_correct": diag,
            "risk_correct": risk,
            "routing_correct": routing,
            "safety_awareness": safety,
            "uncertainty_handling": uncertainty,
        },
        "contradiction_penalty": contradiction_penalty,
        "terminated_reason": state.terminated_reason,
        "steps": state.step_count,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

GRADERS = {
    "easy_triage": easy_grader,
    "medium_triage": medium_grader,
    "hard_triage": hard_grader,
}


def grade_episode(state: EpisodeStateManager, task_name: str) -> Dict[str, Any]:
    """
    Grade a completed episode using the appropriate task grader.

    Args:
        state: Completed episode state manager.
        task_name: One of "easy_triage", "medium_triage", "hard_triage".

    Returns:
        Dict with 'score' (float in [0.0, 1.0]) and component breakdown.
    """
    grader = GRADERS.get(task_name)
    if grader is None:
        return {
            "task": task_name,
            "score": 0.0,
            "error": f"Unknown task_name: {task_name}",
        }
    result = grader(state)
    assert 0.0 <= result["score"] <= 1.0, f"Grader produced out-of-range score: {result['score']}"
    return result
