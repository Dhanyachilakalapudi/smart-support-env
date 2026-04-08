"""Task catalog for the SupportOps environment."""

from typing import Dict, List, Literal

from pydantic import BaseModel, Field

from support_env.models import OwnerTeam, TicketCategory, TicketPriority


class TaskSpec(BaseModel):
    """Deterministic configuration for one benchmark task."""

    name: str
    difficulty: Literal["easy", "medium", "hard"]
    objective: str
    ticket_id: str
    customer_tier: Literal["free", "pro", "enterprise"]
    customer_message: str
    known_facts: List[str] = Field(default_factory=list)
    expected_category: TicketCategory
    expected_priority: TicketPriority
    expected_owner_team: OwnerTeam
    clarification_required: bool = False
    required_clarification_fields: List[str] = Field(default_factory=list)
    required_response_phrases: List[str] = Field(default_factory=list)
    forbidden_response_phrases: List[str] = Field(default_factory=list)
    max_steps: int = Field(..., ge=3, le=12)
    target_steps: int = Field(..., ge=2, le=10)


TASK_LIBRARY: Dict[str, TaskSpec] = {
    "easy": TaskSpec(
        name="easy",
        difficulty="easy",
        objective=(
            "Triage a login lockout ticket and send a safe, actionable password-access"
            " response."
        ),
        ticket_id="SUP-1001",
        customer_tier="free",
        customer_message=(
            "Hi team, I enabled 2FA yesterday and now I cannot log in. "
            "OTP is not arriving and I am locked out."
        ),
        known_facts=[
            "User account is active.",
            "No payment issue detected.",
            "Recent 2FA setup detected.",
        ],
        expected_category="authentication",
        expected_priority="medium",
        expected_owner_team="support_l1",
        clarification_required=False,
        required_response_phrases=[
            "password reset link",
            "backup code",
            "24 hours",
        ],
        forbidden_response_phrases=[
            "not our issue",
            "wait indefinitely",
        ],
        max_steps=5,
        target_steps=3,
    ),
    "medium": TaskSpec(
        name="medium",
        difficulty="medium",
        objective=(
            "Handle a possible double charge by collecting billing identifiers,"
            " triaging correctly, and proposing next actions."
        ),
        ticket_id="SUP-2049",
        customer_tier="pro",
        customer_message=(
            "My card was charged twice for the Pro renewal this morning. "
            "I only have one account. Please fix this quickly."
        ),
        known_facts=[
            "Renewal webhook fired once from our side.",
            "Customer is in monthly billing cycle.",
            "No cancellation request recorded.",
        ],
        expected_category="billing",
        expected_priority="high",
        expected_owner_team="billing_ops",
        clarification_required=True,
        required_clarification_fields=[
            "transaction id",
            "last 4 digits",
        ],
        required_response_phrases=[
            "refund review",
            "billing specialist",
            "transaction id",
        ],
        forbidden_response_phrases=[
            "chargeback immediately",
            "cannot help",
        ],
        max_steps=6,
        target_steps=4,
    ),
    "hard": TaskSpec(
        name="hard",
        difficulty="hard",
        objective=(
            "Coordinate an enterprise webhook signature incident with security-grade"
            " handling, precise triage, and mitigation guidance."
        ),
        ticket_id="SUP-8890",
        customer_tier="enterprise",
        customer_message=(
            "After rotating webhook keys across regions, signature validation started"
            " failing for critical callbacks. This might be a security incident."
        ),
        known_facts=[
            "Impacts production callbacks in two regions.",
            "Customer uses HMAC signature verification.",
            "Last successful webhook was 37 minutes ago.",
        ],
        expected_category="security",
        expected_priority="urgent",
        expected_owner_team="security",
        clarification_required=True,
        required_clarification_fields=[
            "request id",
            "timestamp",
            "region",
            "x-signature",
        ],
        required_response_phrases=[
            "rotate webhook secret",
            "hmac sha256",
            "temporary allowlist",
            "incident bridge",
        ],
        forbidden_response_phrases=[
            "ignore the errors",
            "disable signature checks permanently",
        ],
        max_steps=7,
        target_steps=5,
    ),
}


def list_task_names() -> List[str]:
    return list(TASK_LIBRARY.keys())


def get_task(task_name: str) -> TaskSpec:
    if task_name not in TASK_LIBRARY:
        supported = ", ".join(list_task_names())
        raise ValueError(f"Unknown task '{task_name}'. Supported tasks: {supported}")
    return TASK_LIBRARY[task_name].model_copy(deep=True)
