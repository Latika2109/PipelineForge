"""Reward computation for PipelineForge.

Fixes from README spec:
- state.last_failed_stage is now a proper field in EpisodeState
- estimate_time_saved() is implemented in simulator.py
- state.pipeline[stage_id] replaced with stages[stage_id]
- combinations import included
"""
from __future__ import annotations
from itertools import combinations
from typing import Dict

from models import Action, EpisodeState, StageMeta, StageResult


def compute_reward(
    action: Action,
    result: StageResult | None,
    state: EpisodeState,
    stages: Dict[str, StageMeta],
    time_saved: float = 0.0,
) -> float:
    reward = 0.0

    # === CORRECT ABORT ON REAL FAILURE ===
    if action.action_type == "abort":
        if state.last_failed_stage and state.last_failed_stage in state.real_failures:
            reward += 0.50   # correctly identified and aborted
        elif state.last_failed_stage and state.last_failed_stage in state.flap_stages:
            reward -= 0.40   # aborted on a flap — wasted the run
        else:
            reward -= 0.20   # aborted for unclear reason

    # === RETRY LOGIC ===
    if action.action_type == "retry" and action.stage_id and result:
        sid = action.stage_id
        if sid in state.flap_stages:
            reward += 0.20  # correct: retrying a known flap
            if result.exit_code == 0:
                reward += 0.10  # bonus: retry worked
        elif sid in state.real_failures:
            reward -= 0.25  # wrong: retrying a real failure

    # === SKIP LOGIC ===
    if action.action_type == "skip" and action.stage_id:
        sid = action.stage_id
        stage = stages.get(sid)
        if stage:
            if not stage.is_critical and sid in state.flap_stages:
                reward += 0.15   # smart skip of non-critical flapper
            elif stage.is_critical:
                reward -= 0.50   # NEVER skip a critical stage
            else:
                reward -= 0.05   # skipping non-flap non-critical: minor waste

    # === PARALLELISM REWARD ===
    if action.action_type == "run_parallel" and action.parallel_stages:
        all_independent = all(
            _stages_independent(s1, s2, stages)
            for s1, s2 in combinations(action.parallel_stages, 2)
        )
        if all_independent:
            reward += min(time_saved / 60.0, 0.20)   # up to 0.20 for time saved
        else:
            reward -= 0.20  # parallelised dependent stages — dangerous

    # === TIME EFFICIENCY ===
    if result and result.exit_code == 0:
        if state.optimal_execution_time > 0:
            time_ratio = state.total_elapsed_time / state.optimal_execution_time
            if time_ratio < 1.2:
                reward += 0.10   # within 20% of optimal
            elif time_ratio > 2.0:
                reward -= 0.10   # way over budget

    # === SHIPPED REAL FAILURE — CATASTROPHIC ===
    if state.done and not state.aborted and state.real_failure_shipped:
        reward -= 1.0

    # === INSPECT BONUS (small) ===
    if action.action_type == "inspect" and action.stage_id:
        if action.stage_id not in state.inspected_stages:
            reward += 0.02   # small bonus for first-time inspection

    return float(max(-1.0, min(1.0, reward)))


def _stages_independent(s1: str, s2: str, stages: Dict[str, StageMeta]) -> bool:
    def all_deps(sid: str) -> set:
        visited: set = set()
        stack = list(stages[sid].dependencies) if sid in stages else []
        while stack:
            dep = stack.pop()
            if dep not in visited:
                visited.add(dep)
                if dep in stages:
                    stack.extend(stages[dep].dependencies)
        return visited

    d1, d2 = all_deps(s1), all_deps(s2)
    return s1 not in d2 and s2 not in d1
