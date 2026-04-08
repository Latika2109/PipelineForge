"""Episode graders for each task difficulty. Returns a score in [0.0, 1.0]."""
from __future__ import annotations
from typing import Dict

from models import EpisodeState, StageMeta


def grade_episode(state: EpisodeState, stages: Dict[str, StageMeta]) -> float:
    """Dispatch to the correct grader based on task_id."""
    if state.task_id == "easy":
        return _grade_easy(state)
    elif state.task_id == "medium":
        return _grade_medium(state, stages)
    elif state.task_id == "hard":
        return _grade_hard(state, stages)
    return 0.0


# ---------------------------------------------------------------------------
# Task 1 — Easy grader
# score = 0.40 * correctly_aborted_on_real_failure
#       + 0.30 * did_not_retry_real_failure_excessively (≤1 retry)
#       + 0.20 * retried_unit_test_flap_correctly
#       + 0.10 * efficiency_bonus (total time < 300s)
# ---------------------------------------------------------------------------

def _grade_easy(state: EpisodeState) -> float:
    score = 0.0

    # 0.40 — correctly aborted when integration-tests failed
    if state.real_failure_caught and state.aborted:
        score += 0.40
    elif state.real_failure_shipped:
        score += 0.0  # catastrophic

    # 0.30 — did NOT retry the real failure excessively
    retry_count_real = state.stage_retry_counts.get("integration-tests", 0)
    if retry_count_real == 0:
        score += 0.30   # perfect: no retries on real failure
    elif retry_count_real == 1:
        score += 0.15   # one retry tolerated
    # else: 0 (retried 2+ times)

    # 0.20 — handled unit-tests flap correctly
    if "unit-tests" in state.correctly_retried_flaps:
        score += 0.20
    elif "unit-tests" not in state.incorrectly_retried_real_failures:
        # unit-tests didn't flap this run (or agent skipped it correctly)
        if "unit-tests" not in state.failed_stages:
            score += 0.20  # nothing to handle — full credit

    # 0.10 — efficiency bonus
    if state.total_elapsed_time < 300:
        score += 0.10
    elif state.total_elapsed_time < 400:
        score += 0.05

    return round(min(1.0, score), 4)


# ---------------------------------------------------------------------------
# Task 2 — Medium grader
# score = 0.30 * parallelism_exploited
#       + 0.30 * correctly_identified_real_failure (dependency-audit)
#       + 0.20 * smart_skip_or_retry_decisions
#       + 0.20 * total_time_efficiency (vs optimal baseline)
# ---------------------------------------------------------------------------

def _grade_medium(state: EpisodeState, stages: Dict[str, StageMeta]) -> float:
    score = 0.0

    # 0.30 — parallelism
    if state.parallelism_opportunities_total > 0:
        ratio = min(state.parallel_runs_made / max(state.parallelism_opportunities_total, 1), 1.0)
        score += 0.30 * ratio
    else:
        score += 0.30  # no opportunities = full credit

    # 0.30 — correctly identified dependency-audit as real failure
    if state.real_failure_caught and state.aborted:
        if state.last_failed_stage == "dependency-audit" or \
                "dependency-audit" in state.real_failures:
            score += 0.30
    if state.real_failure_shipped:
        score -= 0.30

    # 0.20 — smart skip / retry
    smart_decisions = len(state.correctly_retried_flaps) + len(state.correctly_skipped_flaps)
    bad_decisions = len(state.incorrectly_retried_real_failures) + len(state.skipped_critical)
    decision_score = max(0.0, smart_decisions - bad_decisions) / max(smart_decisions + bad_decisions + 1, 1)
    score += 0.20 * decision_score

    # 0.20 — time efficiency vs optimal (360s)
    if state.optimal_execution_time > 0:
        ratio = state.total_elapsed_time / state.optimal_execution_time
        if ratio < 1.2:
            score += 0.20
        elif ratio < 1.5:
            score += 0.10
        elif ratio < 2.0:
            score += 0.05

    return round(min(1.0, max(0.0, score)), 4)


# ---------------------------------------------------------------------------
# Task 3 — Hard grader
# score = 0.35 * failure_classification_accuracy (5 ambiguous cases)
#       + 0.25 * no_real_failure_shipped (binary)
#       + 0.20 * time_efficiency
#       + 0.10 * parallelism_score
#       + 0.10 * cascade_detection_bonus
# ---------------------------------------------------------------------------

def _grade_hard(state: EpisodeState, stages: Dict[str, StageMeta]) -> float:
    score = 0.0

    # 0.35 — failure classification accuracy (5 ambiguous stages)
    ambiguous_total = 5  # e2e-test-suite, dependency-audit, perf-benchmark (flaps) + db-migration-check, canary-deploy (real)
    classification_score = state.ambiguous_failures_classified_correctly / ambiguous_total
    score += 0.35 * classification_score

    # 0.25 — no real failure shipped (binary gate)
    if not state.real_failure_shipped:
        score += 0.25

    # 0.20 — time efficiency vs optimal (550s)
    if state.optimal_execution_time > 0 and state.total_elapsed_time > 0:
        ratio = state.total_elapsed_time / state.optimal_execution_time
        if ratio < 1.2:
            score += 0.20
        elif ratio < 1.5:
            score += 0.12
        elif ratio < 2.0:
            score += 0.06

    # 0.10 — parallelism
    if state.parallelism_opportunities_total > 0:
        ratio = min(state.parallel_runs_made / max(state.parallelism_opportunities_total, 1), 1.0)
        score += 0.10 * ratio
    else:
        score += 0.10

    # 0.10 — cascade detection bonus
    if state.cascade_detected:
        score += 0.10

    return round(min(1.0, max(0.0, score)), 4)
