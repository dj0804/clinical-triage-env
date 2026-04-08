"""
Clinical Triage AI — OpenEnv Models
Action, Observation, and State Pydantic models.
"""

from typing import Annotated, Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, ConfigDict, Field, field_validator


# ---------------------------------------------------------------------------
# Action subtypes (discriminated union on action_type)
# ---------------------------------------------------------------------------

class AskQuestionAction(BaseModel):
    """Ask the patient or records a clarifying question."""
    action_type: Literal["ask_question"] = "ask_question"
    question: str = Field(..., min_length=1, max_length=500,
                          description="The clinical question to ask")
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")


class RequestTestAction(BaseModel):
    """Order a diagnostic test."""
    action_type: Literal["request_test"] = "request_test"
    test_name: str = Field(..., min_length=1, max_length=200,
                           description="Name of the test to order (e.g. 'troponin', 'ECG')")
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")


class MakeDiagnosisAction(BaseModel):
    """Propose a clinical diagnosis with a confidence score."""
    action_type: Literal["make_diagnosis"] = "make_diagnosis"
    diagnosis: str = Field(..., min_length=1, max_length=300,
                           description="The proposed diagnosis")
    confidence: float = Field(..., ge=0.0, le=1.0,
                              description="Confidence in the diagnosis [0.0, 1.0]")
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")


class AssignRiskAction(BaseModel):
    """Assign a triage risk level to the patient."""
    action_type: Literal["assign_risk"] = "assign_risk"
    risk_level: Literal["Critical", "Monitor", "Routine"] = Field(
        ..., description="Patient risk classification")
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")


class RoutePatientAction(BaseModel):
    """Route the patient to a hospital department."""
    action_type: Literal["route_patient"] = "route_patient"
    department: str = Field(..., min_length=1, max_length=200,
                            description="Target department (e.g. 'Cardiology', 'ICU')")
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")


class EscalateToHumanAction(BaseModel):
    """Escalate the case to a senior clinician when uncertain."""
    action_type: Literal["escalate_to_human"] = "escalate_to_human"
    reason: str = Field(..., min_length=1, max_length=500,
                        description="Reason for escalation")
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")


class FinalizeAction(BaseModel):
    """Finalize the triage episode."""
    action_type: Literal["finalize"] = "finalize"
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")


# Discriminated union — `action_type` is the discriminator field
TriageAction = Annotated[
    Union[
        AskQuestionAction,
        RequestTestAction,
        MakeDiagnosisAction,
        AssignRiskAction,
        RoutePatientAction,
        EscalateToHumanAction,
        FinalizeAction,
    ],
    Field(discriminator="action_type"),
]


# ---------------------------------------------------------------------------
# Observation model
# ---------------------------------------------------------------------------

class TriageObservation(BaseModel):
    """
    Observation returned to the agent after reset() or step().
    Follows OpenEnv contract.
    """
    model_config = ConfigDict(extra="forbid")

    # Patient data visible to the agent
    symptoms: List[str] = Field(default_factory=list,
                                description="Current observable symptoms (may be noisy)")
    patient_history: Dict[str, Any] = Field(default_factory=dict,
                                            description="Age, sex, comorbidities, medications")
    revealed_tests: Dict[str, Any] = Field(default_factory=dict,
                                           description="Test results revealed so far (immutable once set)")
    previous_actions: List[str] = Field(default_factory=list,
                                        description="Summary of actions taken this episode")
    available_actions: List[str] = Field(
        default=[
            "ask_question", "request_test", "make_diagnosis",
            "assign_risk", "route_patient", "escalate_to_human", "finalize"
        ],
        description="Action types currently allowed",
    )

    # Episode metadata
    step_number: int = Field(default=0, ge=0, description="Current step index")
    done: bool = Field(default=False, description="Whether the episode has terminated")
    # General metadata slot
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# State model (returned from GET /state)
# ---------------------------------------------------------------------------

class TriageState(BaseModel):
    """
    Full internal episode state (for debugging and graders).
    Returned from state() / GET /state.
    """
    model_config = ConfigDict(extra="allow")

    # Episode identifiers
    episode_id: str = Field(..., description="Unique episode UUID")
    step_count: int = Field(default=0, ge=0)
    task_name: str = Field(default="easy_triage")
    case_id: str = Field(default="")

    # Accumulated knowledge
    known_data: Dict[str, Any] = Field(default_factory=dict,
                                       description="All data known by the agent")
    actions_taken: List[Dict[str, Any]] = Field(default_factory=list,
                                                description="Full action history")

    # Terminal flags
    terminal: bool = Field(default=False)
    terminated_reason: str = Field(
        default="",
        description="How the episode ended: 'finalize'|'escalate'|'max_steps'|'invalid'|''",
    )

    # Decision tracking
    has_diagnosis: bool = Field(default=False)
    has_risk: bool = Field(default=False)
    attempted_routing: bool = Field(default=False)
    has_escalated: bool = Field(default=False)

    # Reward history
    score_trajectory: List[float] = Field(default_factory=list,
                                          description="Reward earned at each step")
    reward_breakdown_history: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Per-step reward component breakdowns",
    )
