"""
Clinical Triage AI — Reward Engine

Fully deterministic reward computation. All rewards are per-component,
then combined into a normalized [0.0, 1.0] final score.

Each call returns a (reward, breakdown) tuple where breakdown is a dict
of named components for logging.
"""

from typing import Any, Dict, Optional, Tuple

from server.state_manager import EpisodeStateManager


# ---------------------------------------------------------------------------
# Per-event reward constants
# ---------------------------------------------------------------------------

class RewardConstants:
    # Diagnosis
    DIAGNOSIS_CORRECT_HIGH_CONF = 0.30    # confidence >= 0.7
    DIAGNOSIS_CORRECT_LOW_CONF = 0.20     # confidence < 0.7
    DIAGNOSIS_WRONG = -0.25
    DIAGNOSIS_OVERCONFIDENT_WRONG = -0.35  # base; scaled by step index if early
    DIAGNOSIS_INSUFFICIENT_EVIDENCE = -0.15  # diagnosis before any info gathering

    # Re-diagnosis inconsistency
    REDIAGNOSIS_PENALTY = -0.10

    # Risk classification
    RISK_CORRECT = 0.20
    RISK_CRITICAL_AS_ROUTINE = -0.40      # worst mismatch (safety-critical)
    RISK_OTHER_WRONG = -0.10

    # Routing
    ROUTING_CORRECT = 0.15
    ROUTING_WRONG = -0.10
    ROUTING_WITHOUT_DIAGNOSIS = -0.15

    # Info gathering — tests
    TEST_KEY_INFO = 0.05
    TEST_USELESS = -0.02
    TEST_REDUNDANT = -0.02               # already ordered

    # Info gathering — questions
    QUESTION_HIGH_RELEVANCE = 0.03
    QUESTION_MEDIUM_RELEVANCE = 0.01
    QUESTION_LOW_RELEVANCE = -0.01
    QUESTION_ALREADY_ASKED = -0.02       # redundant

    # Escalation
    ESCALATION_APPROPRIATE = 0.15        # ambiguity/hard case present
    ESCALATION_UNJUSTIFIED = -0.05       # confident + no ambiguity

    # Per-step housekeeping
    STEP_PENALTY = -0.01
    MAX_STEPS_TIMEOUT = -0.20

    # Redundancy penalty multiplier applied per repetition
    REDUNDANCY_DECAY = 0.5               # reward × 0.5^(repeat_count)


R = RewardConstants


def _normalize(x: float) -> float:
    """Clamp to [-1.0, 1.0] for per-step rewards."""
    return max(-1.0, min(1.0, x))


# ---------------------------------------------------------------------------
# Reward functions per action type
# ---------------------------------------------------------------------------

def reward_ask_question(
    state: EpisodeStateManager,
    question_result: Dict[str, Any],
) -> Tuple[float, Dict[str, Any]]:
    """
    Compute reward for ask_question.

    Args:
        state: Current episode state manager.
        question_result: Dict returned by state.ask_question().

    Returns:
        (reward, breakdown)
    """
    breakdown: Dict[str, Any] = {"action": "ask_question"}

    if not question_result["valid"]:
        # Unrecognized question — small penalty
        r = R.QUESTION_LOW_RELEVANCE
        breakdown["event"] = "invalid_question"
        breakdown["value"] = r
        return _normalize(r + R.STEP_PENALTY), {**breakdown, "step_penalty": R.STEP_PENALTY}

    if question_result["already_asked"]:
        r = R.QUESTION_ALREADY_ASKED
        breakdown["event"] = "repeated_question"
        breakdown["value"] = r
    else:
        relevance = question_result["relevance"]
        if relevance == "high":
            r = R.QUESTION_HIGH_RELEVANCE
        elif relevance == "medium":
            r = R.QUESTION_MEDIUM_RELEVANCE
        else:
            r = R.QUESTION_LOW_RELEVANCE
        breakdown["event"] = f"question_{relevance}_relevance"
        breakdown["value"] = r

    # Apply redundancy decay
    sig = f"ask_{question_result.get('matched_question', question_result.get('answer', ''))}"
    repeat_count = state.get_action_repetition_count(sig)
    if repeat_count > 0:
        r = r * (R.REDUNDANCY_DECAY ** repeat_count)
        breakdown["redundancy_decayed"] = True
        breakdown["repeat_count"] = repeat_count

    total = _normalize(r + R.STEP_PENALTY)
    breakdown["step_penalty"] = R.STEP_PENALTY
    breakdown["total"] = total
    return total, breakdown


def reward_request_test(
    state: EpisodeStateManager,
    test_result: Dict[str, Any],
) -> Tuple[float, Dict[str, Any]]:
    """Compute reward for request_test."""
    breakdown: Dict[str, Any] = {"action": "request_test"}

    if not test_result["valid"]:
        r = -0.03  # unknown test
        breakdown["event"] = "invalid_test"
        breakdown["value"] = r
        total = _normalize(r + R.STEP_PENALTY)
        breakdown["step_penalty"] = R.STEP_PENALTY
        breakdown["total"] = total
        return total, breakdown

    if test_result["already_ordered"]:
        r = R.TEST_REDUNDANT
        breakdown["event"] = "redundant_test"
        breakdown["value"] = r
    elif test_result["reveals_key_info"]:
        r = R.TEST_KEY_INFO
        breakdown["event"] = "useful_test"
        breakdown["value"] = r
    else:
        r = R.TEST_USELESS
        breakdown["event"] = "useless_test"
        breakdown["value"] = r

    # Redundancy decay
    sig = f"test_{test_result['test_name']}"
    repeat_count = state.get_action_repetition_count(sig)
    if repeat_count > 0:
        r = r * (R.REDUNDANCY_DECAY ** repeat_count)
        breakdown["redundancy_decayed"] = True

    total = _normalize(r + R.STEP_PENALTY)
    breakdown["step_penalty"] = R.STEP_PENALTY
    breakdown["total"] = total
    return total, breakdown


def reward_make_diagnosis(
    state: EpisodeStateManager,
    diagnosis: str,
    confidence: float,
    is_re_diagnosis: bool,
) -> Tuple[float, Dict[str, Any]]:
    """Compute reward for make_diagnosis."""
    breakdown: Dict[str, Any] = {"action": "make_diagnosis", "diagnosis": diagnosis, "confidence": confidence}

    truth = state.ground_truth or {}
    correct_diag = truth.get("diagnosis", "").lower()
    aliases = [a.lower() for a in truth.get("diagnosis_aliases", [])]
    all_correct = [correct_diag] + aliases

    is_correct = any(
        d in diagnosis.lower() or diagnosis.lower() in d
        for d in all_correct
        if d
    )

    r = 0.0

    # Evidence gate: penalize diagnosis without any info gathering
    if state.evidence_actions_count < 1:
        r += R.DIAGNOSIS_INSUFFICIENT_EVIDENCE
        breakdown["insufficient_evidence_penalty"] = R.DIAGNOSIS_INSUFFICIENT_EVIDENCE

    if is_correct:
        if confidence >= 0.7:
            r += R.DIAGNOSIS_CORRECT_HIGH_CONF
            breakdown["event"] = "correct_diagnosis_high_conf"
        else:
            r += R.DIAGNOSIS_CORRECT_LOW_CONF
            breakdown["event"] = "correct_diagnosis_low_conf"
        breakdown["is_correct"] = True
    else:
        # Wrong diagnosis
        if confidence >= 0.8:
            # Overconfident wrong — scale penalty by how early the mistake was
            step_idx = state.step_count
            early_penalty_scaling = max(0.0, (5 - step_idx) * 0.02)  # up to +0.08 at step 0
            r += R.DIAGNOSIS_OVERCONFIDENT_WRONG - early_penalty_scaling
            breakdown["event"] = "overconfident_wrong_diagnosis"
            breakdown["early_penalty_scaling"] = early_penalty_scaling
        else:
            r += R.DIAGNOSIS_WRONG
            breakdown["event"] = "wrong_diagnosis"
        breakdown["is_correct"] = False

    # Inconsistency penalty for re-diagnosis that changes the answer
    if is_re_diagnosis:
        prev_diag = state.current_diagnosis or ""
        prev_was_correct = any(
            d in prev_diag.lower() or prev_diag.lower() in d
            for d in all_correct if d
        )
        if not prev_was_correct and is_correct:
            # Correcting → no extra penalty (late correction)
            breakdown["rediagnosis_type"] = "late_correction"
        elif prev_was_correct and not is_correct:
            # Was correct, changed to wrong → heavy inconsistency
            r += R.REDIAGNOSIS_PENALTY * 2
            breakdown["rediagnosis_penalty"] = R.REDIAGNOSIS_PENALTY * 2
        else:
            r += R.REDIAGNOSIS_PENALTY
            breakdown["rediagnosis_penalty"] = R.REDIAGNOSIS_PENALTY
        breakdown["is_re_diagnosis"] = True

    total = _normalize(r + R.STEP_PENALTY)
    breakdown["step_penalty"] = R.STEP_PENALTY
    breakdown["total"] = total
    return total, breakdown


def reward_assign_risk(
    state: EpisodeStateManager,
    risk_level: str,
) -> Tuple[float, Dict[str, Any]]:
    """Compute reward for assign_risk."""
    breakdown: Dict[str, Any] = {"action": "assign_risk", "assigned_risk": risk_level}

    truth = state.ground_truth or {}
    correct_risk = truth.get("risk_level", "")

    if risk_level == correct_risk:
        r = R.RISK_CORRECT
        breakdown["event"] = "correct_risk"
    elif correct_risk == "Critical" and risk_level == "Routine":
        r = R.RISK_CRITICAL_AS_ROUTINE
        breakdown["event"] = "critical_misclassified_as_routine"
    else:
        r = R.RISK_OTHER_WRONG
        breakdown["event"] = "wrong_risk"

    breakdown["correct_risk"] = correct_risk
    total = _normalize(r + R.STEP_PENALTY)
    breakdown["step_penalty"] = R.STEP_PENALTY
    breakdown["total"] = total
    return total, breakdown


def reward_route_patient(
    state: EpisodeStateManager,
    department: str,
) -> Tuple[float, Dict[str, Any]]:
    """Compute reward for route_patient."""
    breakdown: Dict[str, Any] = {"action": "route_patient", "routed_to": department}

    if not state.has_diagnosis:
        r = R.ROUTING_WITHOUT_DIAGNOSIS
        breakdown["event"] = "routing_without_diagnosis"
    else:
        truth = state.ground_truth or {}
        correct_dept = truth.get("department", "").lower()
        routed = department.lower()

        if routed == correct_dept or routed in correct_dept or correct_dept in routed:
            r = R.ROUTING_CORRECT
            breakdown["event"] = "correct_routing"
        else:
            r = R.ROUTING_WRONG
            breakdown["event"] = "wrong_routing"

        breakdown["correct_department"] = truth.get("department", "")

    total = _normalize(r + R.STEP_PENALTY)
    breakdown["step_penalty"] = R.STEP_PENALTY
    breakdown["total"] = total
    return total, breakdown


def reward_escalate_to_human(
    state: EpisodeStateManager,
    confidence: float,
) -> Tuple[float, Dict[str, Any]]:
    """
    Compute reward for escalate_to_human.

    Escalation rules:
    - If confidence >= 0.7 AND case does not require escalation → penalty -0.05
    - If ambiguity/hard case AND (confidence < 0.7 OR case requires escalation) → reward +0.15
    """
    breakdown: Dict[str, Any] = {"action": "escalate_to_human"}

    case_requires = state.requires_escalation
    is_ambiguous = state.has_conflicting_evidence or state.is_adversarial

    if confidence >= 0.7 and not case_requires:
        # Unjustified: high confidence, case doesn't need escalation
        r = R.ESCALATION_UNJUSTIFIED
        breakdown["event"] = "unjustified_escalation"
        breakdown["reason"] = "high_confidence_no_ambiguity"
    elif is_ambiguous or case_requires:
        # Appropriate: case is ambiguous or explicitly requires human
        r = R.ESCALATION_APPROPRIATE
        breakdown["event"] = "appropriate_escalation"
    else:
        # Gray area: ambiguity unclear, small penalty
        r = R.ESCALATION_UNJUSTIFIED
        breakdown["event"] = "uncertain_escalation"

    total = _normalize(r + R.STEP_PENALTY)
    breakdown["step_penalty"] = R.STEP_PENALTY
    breakdown["total"] = total
    return total, breakdown


def reward_finalize(
    state: EpisodeStateManager,
) -> Tuple[float, Dict[str, Any]]:
    """
    Compute terminal reward for finalize action.
    Provides a small positive reward if all required decisions are made.
    """
    breakdown: Dict[str, Any] = {"action": "finalize"}

    r = 0.0
    if not state.has_diagnosis:
        r -= 0.10
        breakdown["missing_diagnosis"] = True
    if not state.has_risk:
        r -= 0.05
        breakdown["missing_risk"] = True
    if r == 0.0:
        r = 0.02  # small bonus for clean finalization
        breakdown["event"] = "clean_finalize"

    total = _normalize(r + R.STEP_PENALTY)
    breakdown["step_penalty"] = R.STEP_PENALTY
    breakdown["total"] = total
    return total, breakdown


def reward_timeout(state: EpisodeStateManager) -> Tuple[float, Dict[str, Any]]:
    """Reward for max-step timeout — always a penalty."""
    breakdown = {"action": "timeout", "event": "max_steps_exceeded", "total": R.MAX_STEPS_TIMEOUT}
    return R.MAX_STEPS_TIMEOUT, breakdown


def reward_invalid_action(reason: str) -> Tuple[float, Dict[str, Any]]:
    """Reward for invalid/rejected action."""
    penalty = -0.05
    breakdown = {"action": "invalid", "reason": reason, "total": penalty}
    return penalty, breakdown


# ---------------------------------------------------------------------------
# Final episode score (per-component)
# ---------------------------------------------------------------------------

def compute_final_score(
    state: EpisodeStateManager,
    task_name: str,
) -> float:
    """
    Compute normalized final episode score in [0.0, 1.0].

    Uses per-component weighting, each clamped to [0.0, 1.0].
    NOT a simple clamp of cumulative reward.
    """
    truth = state.ground_truth or {}

    # --- Diagnosis component ---
    diag_score = 0.0
    if state.has_diagnosis:
        correct_diag = truth.get("diagnosis", "").lower()
        aliases = [a.lower() for a in truth.get("diagnosis_aliases", [])]
        all_correct = [correct_diag] + aliases
        current = (state.current_diagnosis or "").lower()
        is_correct = any(
            d in current or current in d
            for d in all_correct if d
        )
        if is_correct:
            diag_score = state.current_confidence if state.current_confidence >= 0.7 else 0.75
        else:
            if state.current_confidence >= 0.8:
                diag_score = 0.0  # overconfident wrong
            else:
                diag_score = 0.1  # wrong but humble

    # --- Risk component ---
    risk_score = 0.0
    if state.has_risk:
        correct_risk = truth.get("risk_level", "")
        if state.current_risk == correct_risk:
            risk_score = 1.0
        elif correct_risk == "Critical" and state.current_risk == "Routine":
            risk_score = 0.0  # worst mismatch
        else:
            risk_score = 0.3  # partial credit for adjacent error

    # --- Routing component ---
    routing_score = 0.0
    if state.attempted_routing:
        correct_dept = truth.get("department", "").lower()
        routed = (state.current_department or "").lower()
        if routed == correct_dept or routed in correct_dept or correct_dept in routed:
            routing_score = 1.0
        else:
            routing_score = 0.0

    # --- Efficiency component ---
    max_steps = 15
    steps = state.step_count
    efficiency_score = max(0.0, (max_steps - steps) / max_steps)

    # --- Weights by task ---
    if task_name == "easy_triage":
        weights = {"diag": 0.40, "risk": 0.30, "routing": 0.20, "eff": 0.10}
    elif task_name == "medium_triage":
        weights = {"diag": 0.35, "risk": 0.25, "routing": 0.15, "eff": 0.10, "info": 0.15}
    else:  # hard_triage
        weights = {"diag": 0.25, "risk": 0.20, "routing": 0.15, "eff": 0.05, "safety": 0.20, "uncertainty": 0.15}

    # --- Info gathering quality (medium task) ---
    info_score = 0.0
    if "info" in weights:
        total_tests = len(state.revealed_tests)
        useful = sum(
            1 for _ in state.actions_taken
            if _.get("action_type") == "request_test" and _.get("details", {}).get("reveals_key_info", False)
        )
        info_score = (useful / max(1, total_tests)) if total_tests > 0 else 0.0

    # --- Safety awareness (hard task) ---
    safety_score = 0.0
    if "safety" in weights:
        made_overconfident_wrong = any(
            (bd.get("event") == "overconfident_wrong_diagnosis")
            for bd in state.reward_breakdown_history
        )
        safety_score = 0.0 if made_overconfident_wrong else 1.0

    # --- Uncertainty handling (hard task) ---
    uncertainty_score = 0.0
    if "uncertainty" in weights:
        if truth.get("requires_escalation", False):
            uncertainty_score = 1.0 if state.has_escalated else 0.0
        else:
            uncertainty_score = 0.0 if state.has_escalated else 1.0

    # --- Aggregate ---
    components = {
        "diag": _clamp01(diag_score),
        "risk": _clamp01(risk_score),
        "routing": _clamp01(routing_score),
        "eff": _clamp01(efficiency_score),
        "info": _clamp01(info_score),
        "safety": _clamp01(safety_score),
        "uncertainty": _clamp01(uncertainty_score),
    }

    score = sum(weights.get(k, 0.0) * v for k, v in components.items())
    return round(_clamp01(score), 4)


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))
