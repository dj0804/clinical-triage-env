"""
Clinical Triage AI — Core Environment

Implements reset(), step(), and state property.
step() returns (Observation, reward, done, info) per OpenEnv spec.
"""

import uuid
from typing import Any, Dict, Optional, Tuple

from models import (
    AskQuestionAction,
    AssignRiskAction,
    EscalateToHumanAction,
    FinalizeAction,
    MakeDiagnosisAction,
    RequestTestAction,
    RoutePatientAction,
    TriageAction,
    TriageObservation,
    TriageState,
)
from server.case_generator import get_case
from server.grader import grade_episode
from server.reward_engine import (
    reward_ask_question,
    reward_assign_risk,
    reward_escalate_to_human,
    reward_finalize,
    reward_invalid_action,
    reward_make_diagnosis,
    reward_request_test,
    reward_route_patient,
    reward_timeout,
)
from server.state_manager import EpisodeStateManager, MAX_STEPS


class TriageEnvironment:
    """
    Clinical Triage AI Environment.

    Implements the OpenEnv API contract:
        reset(seed, episode_id, task_name) -> TriageObservation
        step(action) -> (TriageObservation, float, bool, dict)
        state -> TriageState
    """

    def __init__(self) -> None:
        self._mgr = EpisodeStateManager()
        self._initialized = False

    # -----------------------------------------------------------------------
    # reset()
    # -----------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_name: str = "easy_triage",
        **kwargs: Any,
    ) -> TriageObservation:
        """
        Initialize a new episode.

        Args:
            seed: Deterministic case selection (default 0).
            episode_id: Optional UUID string.
            task_name: "easy_triage" | "medium_triage" | "hard_triage"

        Returns:
            Initial TriageObservation.
        """
        _seed = seed if seed is not None else 0
        _episode_id = episode_id or str(uuid.uuid4())

        try:
            case = get_case(task_name, _seed)
        except ValueError as exc:
            # Return an error observation if task is unknown
            return TriageObservation(
                done=True,
                metadata={"status": "error", "error": str(exc)},
            )

        self._mgr.reset(case, task_name, episode_id=_episode_id)
        self._initialized = True

        return self._build_observation(done=False)

    # -----------------------------------------------------------------------
    # step()
    # -----------------------------------------------------------------------

    def step(
        self,
        action: TriageAction,
    ) -> Tuple[TriageObservation, float, bool, Dict[str, Any]]:
        """
        Execute one action.

        Returns:
            (observation, reward, done, info)
        """
        if not self._initialized:
            info = {"error": "Environment not initialized. Call reset() first."}
            obs = TriageObservation(
                done=True,
                metadata={"error": info["error"]},
            )
            return obs, -0.1, True, info

        mgr = self._mgr

        # Check terminal before acting
        if mgr.terminal:
            info = {"error": "Episode already terminated.", "reason": mgr.terminated_reason}
            obs = self._build_observation(
                done=True,
            )
            return obs, 0.0, True, info

        # Increment step count
        mgr.step_count += 1

        # Dispatch action
        reward, breakdown, error_msg = self._execute_action(action)

        # Log action
        action_type = action.action_type
        details = action.model_dump(exclude={"action_type", "metadata"})
        mgr.log_action(action_type, details, reward, breakdown)

        # Update evidence count
        if action_type in ("ask_question", "request_test"):
            mgr.increment_evidence_count()

        # Check termination after action
        done = mgr.terminal
        if not done and mgr.is_at_max_steps:
            _r, _bd = reward_timeout(mgr)
            mgr.log_action("timeout", {}, _r, _bd)
            mgr.terminate("max_steps")
            reward += _r  # add timeout penalty to last reward
            done = True

        info: Dict[str, Any] = {**breakdown}
        if error_msg:
            info["error"] = error_msg

        # Add final score on termination
        if done and not error_msg:
            grade = grade_episode(mgr, mgr.task_name)
            info["final_grade"] = grade

        obs = self._build_observation(done=done)
        return obs, reward, done, info

    def _execute_action(
        self, action: TriageAction
    ) -> Tuple[float, Dict[str, Any], Optional[str]]:
        """
        Route action to the appropriate reward function.

        Returns:
            (reward, breakdown_dict, error_message_or_None)
        """
        mgr = self._mgr
        at = action.action_type

        if at == "ask_question":
            q_result = mgr.ask_question(action.question)
            r, bd = reward_ask_question(mgr, q_result)
            sig = f"ask_{q_result.get('matched_question', action.question)}"
            mgr.increment_action_count(sig)
            return r, bd, None

        elif at == "request_test":
            t_result = mgr.request_test(action.test_name)
            r, bd = reward_request_test(mgr, t_result)
            sig = f"test_{action.test_name}"
            mgr.increment_action_count(sig)
            return r, bd, None

        elif at == "make_diagnosis":
            is_re = mgr.record_diagnosis(action.diagnosis, action.confidence)
            r, bd = reward_make_diagnosis(mgr, action.diagnosis, action.confidence, is_re)
            return r, bd, None

        elif at == "assign_risk":
            mgr.record_risk(action.risk_level)
            r, bd = reward_assign_risk(mgr, action.risk_level)
            return r, bd, None

        elif at == "route_patient":
            if not mgr.has_diagnosis:
                # Logical consistency violation: routing without diagnosis
                r, bd = reward_invalid_action("routing_without_diagnosis")
                bd["event"] = "routing_without_diagnosis"
                # Still record routing attempt
                mgr.record_routing(action.department)
                return r, bd, "Cannot route patient before making a diagnosis."
            mgr.record_routing(action.department)
            r, bd = reward_route_patient(mgr, action.department)
            return r, bd, None

        elif at == "escalate_to_human":
            mgr.record_escalation()
            conf = mgr.current_confidence if mgr.has_diagnosis else 0.0
            r, bd = reward_escalate_to_human(mgr, conf)
            mgr.terminate("escalate")
            return r, bd, None

        elif at == "finalize":
            if not mgr.has_diagnosis or not mgr.has_risk:
                r, bd = reward_invalid_action("finalize_incomplete")
                error = (
                    "Cannot finalize: missing "
                    + (", ".join(
                        filter(None, [
                            "diagnosis" if not mgr.has_diagnosis else None,
                            "risk assessment" if not mgr.has_risk else None,
                        ])
                    ))
                )
                return r, bd, error
            r, bd = reward_finalize(mgr)
            mgr.terminate("finalize")
            return r, bd, None

        else:
            r, bd = reward_invalid_action(f"unknown_action_type_{at}")
            return r, bd, f"Unknown action type: {at}"

    # -----------------------------------------------------------------------
    # state property
    # -----------------------------------------------------------------------

    @property
    def state(self) -> TriageState:
        """Return full internal state for debugging."""
        d = self._mgr.to_dict()
        return TriageState(**d)

    # -----------------------------------------------------------------------
    # Observation builder
    # -----------------------------------------------------------------------

    def _build_observation(
        self,
        done: bool,
    ) -> TriageObservation:
        """Construct a TriageObservation from current state."""
        mgr = self._mgr

        previous_actions = [
            f"[step {a['step']}] {a['action_type']}: {a.get('details', {})}"
            for a in mgr.actions_taken
        ]

        return TriageObservation(
            symptoms=mgr.initial_symptoms,
            patient_history=mgr.initial_patient_history,
            revealed_tests=mgr.revealed_tests,
            previous_actions=previous_actions,
            step_number=mgr.step_count,
            done=done,
            metadata={
                "episode_id": mgr.episode_id,
                "task_name": mgr.task_name,
                "case_id": mgr.case_id,
                "has_diagnosis": mgr.has_diagnosis,
                "has_risk": mgr.has_risk,
                "has_escalated": mgr.has_escalated,
                "max_steps": MAX_STEPS,
            },
        )
