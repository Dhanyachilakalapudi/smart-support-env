"""SupportOps OpenEnv environment.

This environment simulates real-world customer support operations:
triage, clarification, response drafting, and ticket resolution.
"""

from __future__ import annotations

import os
from typing import Any, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import EnvironmentMetadata

from support_env.graders import grade_episode, shaped_reward
from support_env.models import SupportAction, SupportObservation, SupportState
from support_env.tasks import get_task, list_task_names, TaskSpec


class SupportOpsEnvironment(Environment[SupportAction, SupportObservation, SupportState]):
    """Customer-support workflow environment with deterministic graders."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, task_name: Optional[str] = None):
        super().__init__()
        self._default_task_name = task_name or os.getenv("SUPPORT_ENV_TASK", "easy")
        self._task: TaskSpec = get_task(self._default_task_name)
        self._state: SupportState = self._initial_state(self._task, episode_id=str(uuid4()))
        self._last_action_fingerprint: Optional[str] = None

    def _initial_state(self, task: TaskSpec, episode_id: str) -> SupportState:
        state = SupportState(
            episode_id=episode_id,
            step_count=0,
            task_name=task.name,
            difficulty=task.difficulty,
            max_steps=task.max_steps,
        )
        baseline = grade_episode(task, state)
        state.cumulative_score = baseline.total_score
        state.checklist_scores = baseline.component_scores
        return state

    def _resolve_task(self, seed: Optional[int], task_name: Optional[str], kwargs: dict[str, Any]) -> TaskSpec:
        requested = task_name or kwargs.get("task_name") or kwargs.get("task")
        if requested:
            return get_task(str(requested))

        if seed is not None:
            names = list_task_names()
            index = int(seed) % len(names)
            return get_task(names[index])

        return get_task(self._default_task_name)

    def _build_observation(
        self,
        reward: float,
        done: bool,
        last_action_error: Optional[str],
    ) -> SupportObservation:
        recent_history = self._state.history[-5:]
        return SupportObservation(
            task_name=self._task.name,
            difficulty=self._task.difficulty,
            objective=self._task.objective,
            ticket_id=self._task.ticket_id,
            customer_tier=self._task.customer_tier,
            customer_message=self._task.customer_message,
            known_facts=self._task.known_facts,
            clarification_requirements=self._task.required_clarification_fields,
            checklist_scores=self._state.checklist_scores,
            recent_actions=recent_history,
            steps_remaining=max(0, self._task.max_steps - self._state.step_count),
            last_action_error=last_action_error,
            done=done,
            reward=reward,
            metadata={
                "task_name": self._task.name,
                "step_count": self._state.step_count,
                "cumulative_score": self._state.cumulative_score,
            },
        )

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_name: Optional[str] = None,
        **kwargs: Any,
    ) -> SupportObservation:
        self._task = self._resolve_task(seed=seed, task_name=task_name, kwargs=kwargs)
        self._state = self._initial_state(
            self._task,
            episode_id=episode_id or str(uuid4()),
        )
        self._last_action_fingerprint = None
        self._state.last_action_error = None
        self._state.is_done = False
        return self._build_observation(reward=0.0, done=False, last_action_error=None)

    def _action_fingerprint(self, action: SupportAction) -> str:
        return "|".join(
            [
                action.workflow_step,
                str(action.category or ""),
                str(action.priority or ""),
                str(action.owner_team or ""),
                action.message_to_customer.strip().lower(),
                action.internal_note.strip().lower(),
            ]
        )

    def _set_error(self, current_error: Optional[str], new_message: str) -> str:
        if not current_error:
            return new_message
        return f"{current_error}; {new_message}"

    def step(self, action: SupportAction, timeout_s: Optional[float] = None, **kwargs: Any) -> SupportObservation:  # type: ignore[override]
        _ = timeout_s
        _ = kwargs

        if self._state.is_done:
            self._state.last_action_error = "Episode already completed. Call reset() for a new episode."
            return self._build_observation(
                reward=0.0,
                done=True,
                last_action_error=self._state.last_action_error,
            )

        self._state.step_count += 1
        immediate_penalty = 0.0
        last_error: Optional[str] = None

        fingerprint = self._action_fingerprint(action)
        if self._last_action_fingerprint == fingerprint:
            self._state.repeated_actions += 1
            immediate_penalty += 0.05
            last_error = self._set_error(last_error, "Repeated action pattern detected")
        self._last_action_fingerprint = fingerprint

        if action.workflow_step == "triage":
            if action.category and action.priority and action.owner_team:
                self._state.selected_category = action.category
                self._state.selected_priority = action.priority
                self._state.selected_owner_team = action.owner_team
            else:
                self._state.invalid_actions += 1
                immediate_penalty += 0.08
                last_error = self._set_error(
                    last_error,
                    "Triage requires category, priority, and owner_team",
                )

        elif action.workflow_step == "ask_clarification":
            if len(action.message_to_customer) < 15:
                self._state.invalid_actions += 1
                immediate_penalty += 0.07
                last_error = self._set_error(
                    last_error,
                    "Clarification message is too short",
                )
            else:
                self._state.clarification_asked = True
                self._state.clarification_message = action.message_to_customer
                if not self._task.clarification_required:
                    immediate_penalty += 0.03

        elif action.workflow_step == "draft_response":
            if len(action.message_to_customer) < 20:
                self._state.invalid_actions += 1
                immediate_penalty += 0.08
                last_error = self._set_error(
                    last_error,
                    "Draft response is too short",
                )
            else:
                self._state.draft_response = action.message_to_customer

        elif action.workflow_step == "resolve_ticket":
            self._state.resolved = True
            final_message = action.message_to_customer or self._state.draft_response
            self._state.final_response = final_message

            if len(final_message) < 20:
                self._state.invalid_actions += 1
                immediate_penalty += 0.1
                last_error = self._set_error(
                    last_error,
                    "Resolution message is too short",
                )

            if not (
                self._state.selected_category
                and self._state.selected_priority
                and self._state.selected_owner_team
            ):
                self._state.invalid_actions += 1
                immediate_penalty += 0.12
                last_error = self._set_error(
                    last_error,
                    "Resolve attempted before complete triage",
                )

            if self._task.clarification_required and not self._state.clarification_asked:
                self._state.invalid_actions += 1
                immediate_penalty += 0.1
                last_error = self._set_error(
                    last_error,
                    "Resolve attempted before required clarification",
                )

        if action.internal_note:
            history_line = f"step={self._state.step_count} action={action.workflow_step} note={action.internal_note}"
        else:
            history_line = f"step={self._state.step_count} action={action.workflow_step}"
        self._state.history.append(history_line)

        done = action.workflow_step == "resolve_ticket" or self._state.step_count >= self._task.max_steps
        self._state.is_done = done

        current_grade = grade_episode(self._task, self._state)
        reward_info = shaped_reward(
            previous_score=self._state.cumulative_score,
            current_grade=current_grade,
            immediate_penalty=immediate_penalty,
        )

        self._state.cumulative_score = current_grade.total_score
        self._state.total_reward += reward_info.step_reward
        self._state.checklist_scores = current_grade.component_scores
        self._state.last_action_error = last_error

        observation = self._build_observation(
            reward=reward_info.step_reward,
            done=done,
            last_action_error=last_error,
        )
        observation.metadata["reward_breakdown"] = reward_info.model_dump()

        return observation

    @property
    def state(self) -> SupportState:
        return self._state

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(
            name="support_env",
            description=(
                "Real-world customer support operations benchmark with triage, "
                "clarification, and resolution workflows."
            ),
            version="1.0.0",
            author="Dhanya Chilakalapudi",
        )

    def close(self) -> None:
        # No external resources to clean up for this environment.
        return None
