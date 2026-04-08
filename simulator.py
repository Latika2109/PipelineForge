"""DAG execution engine and probabilistic stage simulator."""
from __future__ import annotations
import random
from itertools import combinations
from typing import Dict, List, Optional, Tuple

from models import Action, EpisodeState, StageMeta, StageResult


class PipelineSimulator:
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    # ------------------------------------------------------------------
    # Core stage execution
    # ------------------------------------------------------------------

    def run_stage(
        self,
        stage_id: str,
        stages: Dict[str, StageMeta],
        state: EpisodeState,
    ) -> StageResult:
        """Simulate running a single stage. Returns a StageResult."""
        stage = stages[stage_id]
        retry_count = state.stage_retry_counts.get(stage_id, 0)

        # Real failure — always fails regardless of retry
        if stage_id in state.real_failures:
            return StageResult(
                stage_id=stage_id,
                exit_code=1,
                duration_seconds=round(stage.estimated_duration * 0.7, 1),
                stdout_tail=f"[FAIL] {stage.name}: command exited with code 1 — genuine failure detected",
                retry_count_so_far=retry_count,
                passed_after_retry=False,
            )

        # Hidden cascade: smoke-tests fails if canary-deploy failed (not shown in DAG)
        hidden_deps = state.hidden_dependencies.get(stage_id, [])
        for hidden_dep in hidden_deps:
            if hidden_dep in state.failed_stages or hidden_dep in state.skipped_stages:
                return StageResult(
                    stage_id=stage_id,
                    exit_code=1,
                    duration_seconds=round(stage.estimated_duration * 0.3, 1),
                    stdout_tail=f"[FAIL] {stage.name}: upstream service unreachable — cascade failure",
                    retry_count_so_far=retry_count,
                    passed_after_retry=False,
                )

        # Flap stage — probabilistic failure; retrying usually resolves it
        if stage_id in state.flap_stages:
            flap_prob = state.flap_probabilities.get(stage_id, 0.3)
            # On retry, significantly reduce flap probability (retry usually fixes it)
            effective_prob = flap_prob * (0.1 ** retry_count)
            if self.rng.random() < effective_prob:
                return StageResult(
                    stage_id=stage_id,
                    exit_code=1,
                    duration_seconds=round(stage.estimated_duration, 1),
                    stdout_tail=f"[FLAP] {stage.name}: transient error — connection reset / resource contention",
                    retry_count_so_far=retry_count,
                    passed_after_retry=False,
                )

        # Success
        jitter = 0.85 + self.rng.random() * 0.30
        return StageResult(
            stage_id=stage_id,
            exit_code=0,
            duration_seconds=round(stage.estimated_duration * jitter, 1),
            stdout_tail=f"[OK] {stage.name}: all checks passed ✓",
            retry_count_so_far=retry_count,
            passed_after_retry=retry_count > 0,
        )

    # ------------------------------------------------------------------
    # DAG topology helpers
    # ------------------------------------------------------------------

    def get_runnable_stages(
        self, stages: Dict[str, StageMeta], state: EpisodeState
    ) -> List[str]:
        """Return stages whose dependencies are all satisfied and which haven't started."""
        done = set(state.completed_stages) | set(state.skipped_stages)
        not_started = [
            sid
            for sid in stages
            if sid not in state.completed_stages
            and sid not in state.failed_stages
            and sid not in state.skipped_stages
        ]
        return [sid for sid in not_started if all(d in done for d in stages[sid].dependencies)]

    def are_independent(
        self, sid1: str, sid2: str, stages: Dict[str, StageMeta]
    ) -> bool:
        """True if two stages have no transitive dependency on each other."""
        def all_deps(sid: str) -> set:
            visited: set = set()
            stack = list(stages[sid].dependencies)
            while stack:
                dep = stack.pop()
                if dep not in visited:
                    visited.add(dep)
                    stack.extend(stages[dep].dependencies)
            return visited

        deps1, deps2 = all_deps(sid1), all_deps(sid2)
        return sid1 not in deps2 and sid2 not in deps1

    def estimate_time_saved(
        self, parallel_stages: List[str], stages: Dict[str, StageMeta]
    ) -> float:
        """Time saved by running stages in parallel vs sequentially."""
        if len(parallel_stages) <= 1:
            return 0.0
        durations = [stages[s].estimated_duration for s in parallel_stages if s in stages]
        return sum(durations) - max(durations, default=0)

    def compute_topological_levels(self, stages: Dict[str, StageMeta]) -> Dict[str, int]:
        """Assign a topological level to each stage for DAG visualization."""
        level: Dict[str, int] = {}
        def get_level(sid: str) -> int:
            if sid in level:
                return level[sid]
            deps = stages[sid].dependencies
            lvl = (max(get_level(d) for d in deps) + 1) if deps else 0
            level[sid] = lvl
            return lvl
        for sid in stages:
            get_level(sid)
        return level

    def check_dependency_satisfied(
        self, stage_id: str, stages: Dict[str, StageMeta], state: EpisodeState
    ) -> Tuple[bool, str]:
        """Check whether a stage's dependencies are satisfied."""
        done = set(state.completed_stages) | set(state.skipped_stages)
        missing = [d for d in stages[stage_id].dependencies if d not in done]
        if missing:
            return False, f"Unsatisfied dependencies: {missing}"
        return True, "OK"

    def count_parallelism_opportunities(
        self, stages: Dict[str, StageMeta], state: EpisodeState
    ) -> int:
        """Count how many runnable stages could currently be parallelised."""
        runnable = self.get_runnable_stages(stages, state)
        count = 0
        for s1, s2 in combinations(runnable, 2):
            if self.are_independent(s1, s2, stages):
                count += 1
        return count
