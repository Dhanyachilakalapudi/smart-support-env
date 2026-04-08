"""Client for interacting with the SupportOps environment over OpenEnv transport."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

from support_env.models import SupportAction, SupportObservation, SupportState


class SupportOpsEnv(EnvClient[SupportAction, SupportObservation, SupportState]):
    """Async OpenEnv client for the support environment."""

    def _step_payload(self, action: SupportAction) -> Dict:
        return action.model_dump(exclude_none=True)

    def _parse_result(self, payload: Dict) -> StepResult[SupportObservation]:
        obs_data = payload.get("observation", {})
        observation = SupportObservation(
            task_name=obs_data.get("task_name", "easy"),
            difficulty=obs_data.get("difficulty", "easy"),
            objective=obs_data.get("objective", ""),
            ticket_id=obs_data.get("ticket_id", ""),
            customer_tier=obs_data.get("customer_tier", "free"),
            customer_message=obs_data.get("customer_message", ""),
            known_facts=obs_data.get("known_facts", []),
            clarification_requirements=obs_data.get("clarification_requirements", []),
            checklist_scores=obs_data.get("checklist_scores", {}),
            recent_actions=obs_data.get("recent_actions", []),
            steps_remaining=obs_data.get("steps_remaining", 0),
            last_action_error=obs_data.get("last_action_error"),
            done=payload.get("done", False),
            reward=payload.get("reward", 0.0),
            metadata=obs_data.get("metadata", {}),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> SupportState:
        return SupportState(**payload)
