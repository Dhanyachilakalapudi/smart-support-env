"""Typed models for the SupportOps OpenEnv environment."""

from typing import Dict, List, Literal, Optional

from openenv.core.env_server.types import Action, Observation, State
from pydantic import BaseModel, ConfigDict, Field, field_validator

TicketCategory = Literal[
    "authentication",
    "billing",
    "integration",
    "performance",
    "security",
    "other",
]
TicketPriority = Literal["low", "medium", "high", "urgent"]
OwnerTeam = Literal["support_l1", "billing_ops", "engineering", "security", "success"]
WorkflowStep = Literal[
    "triage",
    "ask_clarification",
    "draft_response",
    "resolve_ticket",
]


class SupportAction(Action):
    """Action emitted by the agent at each step."""

    # Keep a safe default so generic Swagger templates do not crash the endpoint.
    model_config = ConfigDict(extra="ignore")

    workflow_step: WorkflowStep = Field(
        default="triage",
        description="Current workflow operation.",
    )
    category: Optional[TicketCategory] = Field(
        default=None,
        description="Ticket category chosen during triage.",
    )
    priority: Optional[TicketPriority] = Field(
        default=None,
        description="Ticket urgency assigned during triage.",
    )
    owner_team: Optional[OwnerTeam] = Field(
        default=None,
        description="Team assigned to execute the resolution.",
    )
    message_to_customer: str = Field(
        default="",
        description="Customer-facing message.",
    )
    internal_note: str = Field(
        default="",
        description="Internal reasoning or handoff note.",
    )

    @field_validator("message_to_customer", "internal_note")
    @classmethod
    def _strip_text(cls, value: str) -> str:
        return value.strip()


class SupportObservation(Observation):
    """Observation returned by reset()/step()."""

    task_name: str = Field(..., description="Current task id.")
    difficulty: Literal["easy", "medium", "hard"] = Field(
        ...,
        description="Task difficulty level.",
    )
    objective: str = Field(..., description="What the agent must accomplish.")
    ticket_id: str = Field(..., description="Unique ticket identifier.")
    customer_tier: Literal["free", "pro", "enterprise"] = Field(
        ...,
        description="Customer subscription tier.",
    )
    customer_message: str = Field(..., description="Incoming ticket text.")
    known_facts: List[str] = Field(
        default_factory=list,
        description="Additional context known by the support platform.",
    )
    clarification_requirements: List[str] = Field(
        default_factory=list,
        description="Fields that must be requested from the customer when needed.",
    )
    checklist_scores: Dict[str, float] = Field(
        default_factory=dict,
        description="Per-component grader progress (0.0 to 1.0).",
    )
    recent_actions: List[str] = Field(
        default_factory=list,
        description="Compact action history trace for short-horizon memory.",
    )
    steps_remaining: int = Field(..., ge=0, description="How many steps are left.")
    last_action_error: Optional[str] = Field(
        default=None,
        description="Validation/runtime issue from the previous step.",
    )


class SupportState(State):
    """Internal environment state returned by /state."""

    task_name: str = Field(default="easy")
    difficulty: Literal["easy", "medium", "hard"] = Field(default="easy")
    max_steps: int = Field(default=5, ge=1)
    cumulative_score: float = Field(default=0.0, ge=0.0, le=1.0)
    total_reward: float = Field(default=0.0, ge=0.0)
    selected_category: Optional[TicketCategory] = None
    selected_priority: Optional[TicketPriority] = None
    selected_owner_team: Optional[OwnerTeam] = None
    clarification_asked: bool = False
    clarification_message: str = ""
    draft_response: str = ""
    final_response: str = ""
    resolved: bool = False
    invalid_actions: int = Field(default=0, ge=0)
    repeated_actions: int = Field(default=0, ge=0)
    is_done: bool = False
    last_action_error: Optional[str] = None
    checklist_scores: Dict[str, float] = Field(default_factory=dict)
    history: List[str] = Field(default_factory=list)


class RewardBreakdown(BaseModel):
    """Typed reward payload used in observation metadata and analysis."""

    step_reward: float = Field(..., ge=0.0, le=1.0)
    score_delta: float = Field(..., ge=-1.0, le=1.0)
    previous_score: float = Field(..., ge=0.0, le=1.0)
    current_score: float = Field(..., ge=0.0, le=1.0)
    behavior_penalty: float = Field(..., ge=0.0)
    component_scores: Dict[str, float] = Field(default_factory=dict)


class EpisodeGrade(BaseModel):
    """Deterministic task grade in the range [0.0, 1.0]."""

    total_score: float = Field(..., ge=0.0, le=1.0)
    behavior_penalty: float = Field(..., ge=0.0)
    component_scores: Dict[str, float] = Field(default_factory=dict)
