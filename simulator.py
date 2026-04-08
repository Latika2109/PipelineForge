"""DAG execution engine and probabilistic stage simulator."""
from __future__ import annotations
import random
from itertools import combinations
from typing import Dict, List, Optional, Tuple

from models import Action, EpisodeState, StageMeta, StageResult


# ---------------------------------------------------------------------------
# Realistic CI-style stdout generators per stage type
# ---------------------------------------------------------------------------

def _realistic_stdout(stage_id: str, name: str, passed: bool, rng: random.Random) -> str:
    """Return realistic CI log output based on stage type."""
    sid = stage_id.lower()

    if "checkout" in sid:
        if passed:
            sha = "".join(rng.choices("0123456789abcdef", k=8))
            return f"HEAD is now at {sha}. Fetched 246 objects. Branch 'main' up to date."
        return "fatal: remote: Repository not found or access denied. exit code 128."

    if "install-deps" in sid or ("deps" in sid and "audit" not in sid):
        if passed:
            n = rng.randint(32, 85)
            t = rng.randint(12, 45)
            return f"Successfully installed {n} packages in {t}s. All requirements satisfied."
        return "ERROR: Could not find a version satisfying requirement numpy==1.99 (from versions: none). exit code 1."

    if "lint" in sid and "infra" not in sid:
        if passed:
            files = rng.randint(48, 130)
            return f"ESLint: checked {files} files. 0 errors, 0 warnings."
        errors = rng.randint(1, 6)
        return (
            f"ESLint: {errors} errors found.\n"
            f"  src/api/handler.js:47:5 no-unused-vars 'res' is defined but never used.\n"
            f"  error: Command failed with exit code 1."
        )

    if "lint-infra" in sid or "type-check" in sid:
        if passed:
            files = rng.randint(20, 60)
            return f"mypy: checked {files} source files. No issues found."
        return "mypy: error: Argument 1 to 'run' has incompatible type 'str | None'. exit code 1."

    if "unit-tests" in sid:
        if passed:
            n = rng.randint(60, 210)
            t = round(rng.uniform(12, 95), 1)
            return f"pytest: {n} passed, 0 failed in {t}s. Coverage: {rng.randint(72, 94)}%."
        n = rng.randint(60, 210)
        t = round(rng.uniform(12, 45), 1)
        return (
            f"FAILED tests/test_api.py::test_user_auth - AssertionError: expected 200, got 500.\n"
            f"pytest: 1 failed, {n - 1} passed in {t}s."
        )

    if "e2e" in sid or "integration" in sid:
        if passed:
            n = rng.randint(18, 55)
            t = round(rng.uniform(65, 155), 1)
            return f"Cypress: {n} passing ({t}s). All scenarios passed."
        return (
            "Cypress: 1 failing. Timeout waiting for element '.submit-btn' (exceeded 4000ms). "
            "Test: 'User can submit form'. exit code 1."
        )

    if "build" in sid and "docker" not in sid:
        if passed:
            t = round(rng.uniform(22, 82), 1)
            kb = rng.randint(280, 820)
            return f"Build successful. Compiled 312 modules in {t}s. Bundle size: {kb}KB (gzip: {kb // 3}KB)."
        return "ERROR in src/components/Dashboard.tsx: Cannot find module './missing'. Build failed. exit code 1."

    if "build-docker" in sid or "docker" in sid:
        if passed:
            img_id = "".join(rng.choices("0123456789abcdef", k=12))
            mb = rng.randint(180, 620)
            return f"Successfully built {img_id}. Image size: {mb}MB. Pushed to registry."
        return "ERROR [build 7/9] COPY ./dist /app/dist: file not found in build context. exit code 1."

    if "security-scan" in sid:
        if passed:
            low = rng.randint(2, 7)
            return f"Trivy scan complete. CRITICAL: 0, HIGH: 0, MEDIUM: 1, LOW: {low}. Threshold: CRITICAL+HIGH=0. PASS."
        return "Trivy: CRITICAL vulnerability CVE-2024-21626 in runc 1.1.11. Threshold exceeded. exit code 1."

    if "security-audit" in sid or "dependency-audit" in sid:
        if passed:
            return "npm audit: 0 critical, 0 high vulnerabilities. 3 low severity (suppressed). Audit passed."
        return (
            "npm audit: CRITICAL: CVE-2024-55565 in cross-spawn@7.0.3 (prototype pollution). "
            "Run 'npm audit fix' immediately. exit code 1."
        )

    if "db-migration" in sid:
        if passed:
            return "Alembic: applied 3 pending migrations. Schema version: 20240401_004. Database up to date."
        return (
            "ERROR: alembic.exc.OperationalError: column 'user_uuid' already exists "
            "(migration 20240401_003_add_uuid). exit code 1."
        )

    if "canary" in sid or "deploy" in sid:
        if passed:
            return "Deployment: canary=10%. Health checks: 3/3 OK. Response time: 142ms. Rollout: in progress."
        return "Deployment FAILED: health check /api/health returned 503 after 30s. Rolling back. Pods: 0/3 Running."

    if "smoke" in sid:
        if passed:
            return "Smoke tests: 12/12 endpoints passing. p99 latency: 118ms. Status: HEALTHY."
        return "SMOKE FAIL: GET /api/v1/users returned 502 Bad Gateway. Service unreachable after deploy."

    if "perf" in sid or "load" in sid:
        if passed:
            rps = rng.randint(850, 2100)
            p99 = rng.randint(45, 195)
            return f"k6: {rps} req/s. p99: {p99}ms. Error rate: 0.00%. All SLA thresholds met."
        return "k6: FAIL. p99 latency 863ms exceeds SLA (500ms). Error rate: 2.4%. Threshold breached."

    # default
    if passed:
        return f"{name}: completed successfully. exit code 0."
    return f"{name}: command exited with code 1. Check runner logs for full output."


def _realistic_flap_stdout(stage_id: str, name: str, rng: random.Random) -> str:
    """Return a realistic transient-error log message for a flapping stage."""
    sid = stage_id.lower()

    if "lint" in sid or "type-check" in sid:
        return f"{name}: linter process killed (OOM, exit 137). Transient — retry recommended."
    if "test" in sid or "e2e" in sid or "integration" in sid:
        return f"{name}: test database connection lost (ECONNRESET). Intermittent network issue."
    if "build" in sid:
        return f"{name}: build cache corrupted. Stale lock file detected. Transient — retry usually fixes this."
    if "security" in sid or "audit" in sid:
        return f"{name}: NVD database fetch timeout (30s). Vulnerability DB unreachable. Transient."
    if "perf" in sid or "load" in sid:
        return f"{name}: k6 agent lost connection to target. High load on shared runner. Transient."
    if "deploy" in sid or "canary" in sid:
        return f"{name}: kubectl apply timed out waiting for pod ready. Node briefly overloaded. Transient."
    return f"{name}: transient runner error — resource contention on shared CI node. exit code 1."


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

        # Real failure — always fails regardless of retry.
        if stage_id in state.real_failures:
            duration = round(stage.estimated_duration * 0.7, 1)
            return StageResult(
                stage_id=stage_id,
                exit_code=1,
                duration_seconds=duration,
                stdout_tail=_realistic_stdout(stage_id, stage.name, False, self.rng),
                retry_count_so_far=retry_count,
                passed_after_retry=False,
            )

        # Hidden cascade: stage fails if a hidden upstream also failed.
        hidden_deps = state.hidden_dependencies.get(stage_id, [])
        for hidden_dep in hidden_deps:
            if hidden_dep in state.failed_stages or hidden_dep in state.skipped_stages:
                return StageResult(
                    stage_id=stage_id,
                    exit_code=1,
                    duration_seconds=round(stage.estimated_duration * 0.3, 1),
                    stdout_tail=(
                        f"{stage.name}: upstream service '{hidden_dep}' is unhealthy. "
                        f"Cascade failure — cannot proceed."
                    ),
                    retry_count_so_far=retry_count,
                    passed_after_retry=False,
                )

        # Flap stage — probabilistic failure; retrying usually resolves it.
        if stage_id in state.flap_stages:
            flap_prob = state.flap_probabilities.get(stage_id, 0.3)
            effective_prob = flap_prob * (0.1 ** retry_count)
            if self.rng.random() < effective_prob:
                return StageResult(
                    stage_id=stage_id,
                    exit_code=1,
                    duration_seconds=round(stage.estimated_duration, 1),
                    stdout_tail=_realistic_flap_stdout(stage_id, stage.name, self.rng),
                    retry_count_so_far=retry_count,
                    passed_after_retry=False,
                )

        # Success
        jitter = 0.85 + self.rng.random() * 0.30
        duration = round(stage.estimated_duration * jitter, 1)
        return StageResult(
            stage_id=stage_id,
            exit_code=0,
            duration_seconds=duration,
            stdout_tail=_realistic_stdout(stage_id, stage.name, True, self.rng),
            retry_count_so_far=retry_count,
            passed_after_retry=retry_count > 0,
        )

    # ------------------------------------------------------------------
    # DAG topology helpers
    # ------------------------------------------------------------------

    def get_runnable_stages(
        self, stages: Dict[str, StageMeta], state: EpisodeState
    ) -> List[str]:
        """Return stages whose dependencies are all satisfied and which have not started."""
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
        """Count how many runnable stage pairs could currently be parallelised."""
        runnable = self.get_runnable_stages(stages, state)
        count = 0
        for s1, s2 in combinations(runnable, 2):
            if self.are_independent(s1, s2, stages):
                count += 1
        return count
