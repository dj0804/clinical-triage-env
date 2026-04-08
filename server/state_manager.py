"""
Clinical Triage AI — State Manager

Tracks all episode state: known data, action history, flags, and rewards.
Enforces logical consistency and immutability of revealed information.
"""

import uuid
from copy import deepcopy
from typing import Any, Dict, List, Optional

from server.case_generator import CaseDict


MAX_STEPS = 15


class EpisodeStateManager:
    """
    Manages the mutable state of a single triage episode.

    Key guarantees:
    - Observations are immutable once revealed (no retroactive changes)
    - Max steps enforced at MAX_STEPS (default 15)
    - All state is isolated per instance (no shared mutable globals)
    """

    def __init__(self) -> None:
        self._reset_internal()

    def _reset_internal(self) -> None:
        """Initialize all state fields to defaults."""
        self.episode_id: str = str(uuid.uuid4())
        self.step_count: int = 0
        self.task_name: str = ""
        self.case_id: str = ""

        # The full case dict (hidden truth, tests, questions)
        self._case: Optional[CaseDict] = None

        # Immutable snapshot of initial observation
        self._initial_symptoms: List[str] = []
        self._initial_patient_history: Dict[str, Any] = {}

        # Accumulated known data (grows only — never shrinks)
        self._revealed_tests: Dict[str, Any] = {}  # test_name → result string
        self._asked_questions: Dict[str, str] = {}  # question_text → answer

        # Action history: list of dicts {step, action_type, details, reward}
        self.actions_taken: List[Dict[str, Any]] = []

        # Diagnosis / decision tracking
        self.current_diagnosis: Optional[str] = None
        self.current_confidence: float = 0.0
        self.current_risk: Optional[str] = None
        self.current_department: Optional[str] = None

        # Logical consistency flags
        self.has_diagnosis: bool = False
        self.has_risk: bool = False
        self.attempted_routing: bool = False
        self.has_escalated: bool = False

        # Evidence counter: # of ask_question + request_test actions taken
        self.evidence_actions_count: int = 0

        # Terminal state
        self.terminal: bool = False
        self.terminated_reason: str = ""  # "finalize"|"escalate"|"max_steps"|"invalid"|""

        # Reward history
        self.score_trajectory: List[float] = []
        self.reward_breakdown_history: List[Dict[str, Any]] = []

        # Redundancy tracking: counts per action signature
        self._action_counts: Dict[str, int] = {}

        # Diagnosis history for inconsistency detection
        self._diagnosis_history: List[str] = []

    # -----------------------------------------------------------------------
    # Public reset
    # -----------------------------------------------------------------------

    def reset(self, case: CaseDict, task_name: str, episode_id: Optional[str] = None) -> None:
        """
        Reset to a new episode with the given case.

        Args:
            case: The patient case dict from case_generator.
            task_name: Task name (e.g., "easy_triage").
            episode_id: Optional custom UUID.
        """
        self._reset_internal()

        if episode_id:
            self.episode_id = episode_id

        self.task_name = task_name
        self.case_id = case["case_id"]
        self._case = deepcopy(case)  # deep copy to prevent mutation

        # Lock the initial observation
        init_obs = case["initial_observation"]
        self._initial_symptoms = list(init_obs.get("symptoms", []))
        self._initial_patient_history = deepcopy(init_obs.get("patient_history", {}))

    # -----------------------------------------------------------------------
    # Properties exposing immutable initial data
    # -----------------------------------------------------------------------

    @property
    def initial_symptoms(self) -> List[str]:
        return list(self._initial_symptoms)  # defensive copy

    @property
    def initial_patient_history(self) -> Dict[str, Any]:
        return deepcopy(self._initial_patient_history)

    @property
    def revealed_tests(self) -> Dict[str, Any]:
        return dict(self._revealed_tests)  # defensive copy

    @property
    def asked_questions(self) -> Dict[str, str]:
        return dict(self._asked_questions)

    # -----------------------------------------------------------------------
    # Test / question revealing
    # -----------------------------------------------------------------------

    def request_test(self, test_name: str) -> Dict[str, Any]:
        """
        Reveal a test result. Returns result info dict.

        Returns:
            {
                "test_name": str,
                "result": str | None,
                "reveals_key_info": bool,
                "already_ordered": bool,
                "valid": bool,
            }
        """
        if self._case is None:
            return {"valid": False, "already_ordered": False, "result": None, "reveals_key_info": False, "test_name": test_name}

        available = self._case.get("available_tests", {})

        # Check case-insensitive / normalized
        matched_key = self._find_test_key(test_name, available)

        if matched_key is None:
            return {
                "valid": False,
                "already_ordered": False,
                "result": None,
                "reveals_key_info": False,
                "test_name": test_name,
            }

        already_ordered = matched_key in self._revealed_tests

        if not already_ordered:
            test_data = available[matched_key]
            # Immutably add to revealed tests
            self._revealed_tests[matched_key] = test_data["result"]

        return {
            "valid": True,
            "already_ordered": already_ordered,
            "result": available[matched_key]["result"],
            "reveals_key_info": available[matched_key]["reveals_key_info"],
            "test_name": matched_key,
        }

    def ask_question(self, question: str) -> Dict[str, Any]:
        """
        Find the best matching question and reveal answer.

        Returns:
            {
                "valid": bool,
                "answer": str | None,
                "relevance": "high"|"medium"|"low"|"unknown",
                "already_asked": bool,
                "matched_question": str | None,
            }
        """
        if self._case is None:
            return {"valid": False, "answer": None, "relevance": "unknown", "already_asked": False, "matched_question": None}

        valid_questions = self._case.get("valid_questions", {})
        question_relevance = self._case.get("question_relevance", {})

        matched_key = self._find_question_key(question, valid_questions)

        if matched_key is None:
            return {
                "valid": False,
                "answer": None,
                "relevance": "unknown",
                "already_asked": False,
                "matched_question": None,
            }

        already_asked = matched_key in self._asked_questions
        if not already_asked:
            self._asked_questions[matched_key] = valid_questions[matched_key]

        return {
            "valid": True,
            "answer": valid_questions[matched_key],
            "relevance": question_relevance.get(matched_key, "low"),
            "already_asked": already_asked,
            "matched_question": matched_key,
        }

    def _find_test_key(self, test_name: str, available: Dict) -> Optional[str]:
        """Case-insensitive / underscore-normalized test key lookup."""
        normalized = test_name.lower().replace(" ", "_").replace("-", "_")
        for k in available:
            if k.lower().replace(" ", "_").replace("-", "_") == normalized:
                return k
            # Also allow partial match for common abbreviations
            if normalized in k.lower() or k.lower() in normalized:
                return k
        return None

    def _find_question_key(self, question: str, valid_questions: Dict) -> Optional[str]:
        """Fuzzy question matching: look for key word overlap."""
        q_lower = question.lower().strip().rstrip("?")
        q_words = set(q_lower.split())

        best_key = None
        best_score = 0

        for k in valid_questions:
            k_words = set(k.lower().strip().rstrip("?").split())
            overlap = len(q_words & k_words)
            # Require at least 2 words in common or the question contains the key
            if overlap >= 2 or q_lower in k.lower() or k.lower() in q_lower:
                if overlap > best_score:
                    best_score = overlap
                    best_key = k

        return best_key

    # -----------------------------------------------------------------------
    # Decision recording
    # -----------------------------------------------------------------------

    def record_diagnosis(self, diagnosis: str, confidence: float) -> bool:
        """
        Record a diagnosis. Returns True if this is a re-diagnosis (inconsistency).
        """
        is_re_diagnosis = self.has_diagnosis
        if is_re_diagnosis:
            self._diagnosis_history.append(self.current_diagnosis)

        self.current_diagnosis = diagnosis
        self.current_confidence = confidence
        self.has_diagnosis = True
        return is_re_diagnosis

    def record_risk(self, risk_level: str) -> None:
        self.current_risk = risk_level
        self.has_risk = True

    def record_routing(self, department: str) -> None:
        self.current_department = department
        self.attempted_routing = True

    def record_escalation(self) -> None:
        self.has_escalated = True

    # -----------------------------------------------------------------------
    # Action logging
    # -----------------------------------------------------------------------

    def log_action(self, action_type: str, details: Dict[str, Any], reward: float,
                   reward_breakdown: Dict[str, Any]) -> None:
        """Append an action to history with reward information."""
        record = {
            "step": self.step_count,
            "action_type": action_type,
            "details": details,
            "reward": reward,
        }
        self.actions_taken.append(record)
        self.score_trajectory.append(reward)
        self.reward_breakdown_history.append(reward_breakdown)

    # -----------------------------------------------------------------------
    # Redundancy tracking
    # -----------------------------------------------------------------------

    def get_action_repetition_count(self, action_signature: str) -> int:
        """Return how many times this exact action signature has been seen (before current)."""
        return self._action_counts.get(action_signature, 0)

    def increment_action_count(self, action_signature: str) -> None:
        self._action_counts[action_signature] = self._action_counts.get(action_signature, 0) + 1

    # -----------------------------------------------------------------------
    # Terminal state
    # -----------------------------------------------------------------------

    def terminate(self, reason: str) -> None:
        """Mark episode as terminal with reason."""
        self.terminal = True
        self.terminated_reason = reason

    @property
    def is_at_max_steps(self) -> bool:
        return self.step_count >= MAX_STEPS

    # -----------------------------------------------------------------------
    # Evidence counting
    # -----------------------------------------------------------------------

    def increment_evidence_count(self) -> None:
        """Call this for each ask_question or request_test action."""
        self.evidence_actions_count += 1

    # -----------------------------------------------------------------------
    # Hidden truth access (for reward engine / grader)
    # -----------------------------------------------------------------------

    @property
    def ground_truth(self) -> Optional[Dict[str, Any]]:
        if self._case is None:
            return None
        return self._case.get("hidden_truth", {})

    @property
    def is_adversarial(self) -> bool:
        """True if the case has adversarial notes (hard case type)."""
        if self._case is None:
            return False
        return "adversarial_notes" in self._case

    @property
    def requires_escalation(self) -> bool:
        """True if the case's hidden truth requires escalation."""
        if self._case is None:
            return False
        return self._case.get("hidden_truth", {}).get("requires_escalation", False)

    @property
    def has_conflicting_evidence(self) -> bool:
        """Heuristic: adversarial cases with >1 mimic are considered ambiguous."""
        if self._case is None:
            return False
        return len(self._case.get("mimics", [])) > 0

    @property
    def case_difficulty(self) -> str:
        if self._case is None:
            return "unknown"
        return self._case.get("difficulty", "unknown")

    # -----------------------------------------------------------------------
    # Serialization
    # -----------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialize full state for the state() endpoint."""
        return {
            "episode_id": self.episode_id,
            "step_count": self.step_count,
            "task_name": self.task_name,
            "case_id": self.case_id,
            "case_difficulty": self.case_difficulty,
            "known_data": {
                "symptoms": self.initial_symptoms,
                "patient_history": self.initial_patient_history,
                "revealed_tests": self.revealed_tests,
                "asked_questions": self.asked_questions,
            },
            "actions_taken": self.actions_taken,
            "terminal": self.terminal,
            "terminated_reason": self.terminated_reason,
            "has_diagnosis": self.has_diagnosis,
            "has_risk": self.has_risk,
            "attempted_routing": self.attempted_routing,
            "has_escalated": self.has_escalated,
            "current_diagnosis": self.current_diagnosis,
            "current_confidence": self.current_confidence,
            "current_risk": self.current_risk,
            "current_department": self.current_department,
            "evidence_actions_count": self.evidence_actions_count,
            "score_trajectory": self.score_trajectory,
            "reward_breakdown_history": self.reward_breakdown_history,
        }
