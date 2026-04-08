"""Deterministic graders and reward shaping for SupportOps tasks."""

from __future__ import annotations

from typing import Dict, Tuple

from support_env.models import EpisodeGrade, RewardBreakdown, SupportState
from support_env.tasks import TaskSpec


def _normalize(text: str) -> str:
    return " ".join(text.lower().strip().split())


def _phrase_coverage(text: str, phrases: list[str]) -> float:
    if not phrases:
        return 1.0
    normalized_text = _normalize(text)
    matched = sum(1 for phrase in phrases if _normalize(phrase) in normalized_text)
    return matched / len(phrases)


def _clarification_score(task: TaskSpec, state: SupportState) -> float:
    if not task.clarification_required:
        return 1.0
    if not state.clarification_asked:
        return 0.0
    return _phrase_coverage(
        state.clarification_message,
        task.required_clarification_fields,
    )


def _response_score(task: TaskSpec, state: SupportState) -> float:
    final_message = state.final_response or state.draft_response
    if not final_message:
        return 0.0

    required_coverage = _phrase_coverage(final_message, task.required_response_phrases)
    forbidden_hits = sum(
        1
        for phrase in task.forbidden_response_phrases
        if _normalize(phrase) in _normalize(final_message)
    )
    forbidden_penalty = min(0.6, forbidden_hits * 0.3)
    return max(0.0, required_coverage - forbidden_penalty)


def _efficiency_score(task: TaskSpec, state: SupportState) -> float:
    if state.step_count == 0:
        return 0.0
    if state.step_count <= task.target_steps:
        return 1.0
    over_steps = state.step_count - task.target_steps
    decay_window = max(1, task.max_steps - task.target_steps)
    return max(0.0, 1.0 - (over_steps / decay_window))


def grade_episode(task: TaskSpec, state: SupportState) -> EpisodeGrade:
    """Compute a deterministic grade in [0.0, 1.0] for the full trajectory."""

    component_scores: Dict[str, float] = {
        "category": 1.0 if state.selected_category == task.expected_category else 0.0,
        "priority": 1.0 if state.selected_priority == task.expected_priority else 0.0,
        "owner_team": (
            1.0 if state.selected_owner_team == task.expected_owner_team else 0.0
        ),
        "clarification": _clarification_score(task, state),
        "response_quality": _response_score(task, state),
        "resolution": 1.0 if state.resolved else 0.0,
        "efficiency": _efficiency_score(task, state),
    }

    weights = {
        "category": 0.18,
        "priority": 0.14,
        "owner_team": 0.14,
        "clarification": 0.16,
        "response_quality": 0.24,
        "resolution": 0.08,
        "efficiency": 0.06,
    }

    weighted_score = sum(weights[key] * component_scores[key] for key in weights)
    behavior_penalty = min(
        0.35,
        state.invalid_actions * 0.06 + state.repeated_actions * 0.04,
    )
    final_score = min(1.0, max(0.0, weighted_score - behavior_penalty))

    return EpisodeGrade(
        total_score=final_score,
        behavior_penalty=behavior_penalty,
        component_scores=component_scores,
    )


def shaped_reward(
    previous_score: float,
    current_grade: EpisodeGrade,
    immediate_penalty: float = 0.0,
) -> RewardBreakdown:
    """
    Build a shaped reward signal with partial progress and action penalties.

    Reward is bounded in [0.0, 1.0].
    """
    score_delta = current_grade.total_score - previous_score
    raw_reward = score_delta - immediate_penalty
    step_reward = min(1.0, max(0.0, raw_reward))

    return RewardBreakdown(
        step_reward=step_reward,
        score_delta=score_delta,
        previous_score=previous_score,
        current_score=current_grade.total_score,
        behavior_penalty=current_grade.behavior_penalty + immediate_penalty,
        component_scores=current_grade.component_scores,
    )


def run_task_grader(task: TaskSpec, state: SupportState) -> Tuple[float, Dict[str, float]]:
    """
    Utility API for offline evaluation scripts.

    Returns:
        Tuple[total_score, component_scores] where total_score is in [0.0, 1.0].
    """
    result = grade_episode(task, state)
    return result.total_score, result.component_scores
