"""Core PipelineForge environment — reset / step / observe."""
from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from models import Action, EpisodeState, Observation, StageMeta, StageResult
from simulator import PipelineSimulator
from reward import compute_reward
from grader import grade_episode

PIPELINE_BANK = Path(__file__).parent / "pipeline_bank"

TASK_FILES = {
    "easy": PIPELINE_BANK / "task_easy.json",
    "medium": PIPELINE_BANK / "task_medium.json",
    "hard": PIPELINE_BANK / "task_hard.json",
}


class PipelineForgeEnv:
    """OpenEnv-compatible CI/CD pipeline optimisation environment."""

    def __init__(self):
        self.state: Optional[EpisodeState] = None
        self.stages: Dict[str, StageMeta] = {}
        self.sim: Optional[PipelineSimulator] = None
        self._task_config: Optional[dict] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, task_id: str, seed: int = 42) -> Observation:
        """Start a new episode. Returns the initial observation."""
        if task_id not in TASK_FILES:
            raise ValueError(f"Unknown task_id '{task_id}'. Choose from: {list(TASK_FILES)}")

        config = json.loads(TASK_FILES[task_id].read_text())
        self._task_config = config
        gt = config["ground_truth"]

        self.sim = PipelineSimulator(seed=seed)

        # Build stages dict
        raw_stages = {s["stage_id"]: StageMeta(**s) for s in config["stages"]}
        levels = self.sim.compute_topological_levels(raw_stages)
        for sid, lvl in levels.items():
            raw_stages[sid].level = lvl
        self.stages = raw_stages

        # Build episode state
        self.state = EpisodeState(
            pipeline_id=config["pipeline_id"],
            task_id=task_id,
            step_budget=config["step_budget"],
            real_failures=gt["real_failures"],
            flap_stages=gt["flap_stages"],
            flap_probabilities=gt["flap_probabilities"],
            optimal_execution_time=gt["optimal_execution_time"],
            hidden_dependencies=gt.get("hidden_dependencies", {}),
            stage_statuses={sid: "not_run" for sid in raw_stages},
        )

        # Count available parallelism opportunities up front
        self.state.parallelism_opportunities_total = (
            self.sim.count_parallelism_opportunities(self.stages, self.state)
        )

        return self._build_observation(reward=0.0)

    def step(self, action: Action) -> Tuple[Observation, float, bool, dict]:
        """Execute an action. Returns (observation, reward, done, info)."""
        if self.state is None or self.state.done:
            raise RuntimeError("Call reset() before step().")

        self.state.steps_taken += 1
        self.state.last_action_type = action.action_type
        reward = 0.0
        result: Optional[StageResult] = None
        message = ""

        # ---- Dispatch action ----
        if action.action_type == "run":
            reward, result, message = self._handle_run(action)

        elif action.action_type == "run_parallel":
            reward, result, message = self._handle_run_parallel(action)

        elif action.action_type == "skip":
            reward, message = self._handle_skip(action)

        elif action.action_type == "retry":
            reward, result, message = self._handle_retry(action)

        elif action.action_type == "abort":
            reward, message = self._handle_abort(action)

        elif action.action_type == "inspect":
            reward, message = self._handle_inspect(action)

        elif action.action_type == "wait":
            message = f"Waiting for stage {action.stage_id}."

        else:
            message = f"Unknown action type: {action.action_type}"

        # ---- Check done conditions ----
        self._check_done()
        self.state.cumulative_reward = round(self.state.cumulative_reward + reward, 4)

        # ---- Log action ----
        self.state.execution_log.append({
            "step": self.state.steps_taken,
            "action": action.model_dump(),
            "reward": reward,
            "message": message,
        })

        # ---- Score if done ----
        score = None
        if self.state.done:
            score = grade_episode(self.state, self.stages)

        obs = self._build_observation(reward=reward, score=score, message=message)
        info = {
            "score": score,
            "cumulative_reward": self.state.cumulative_reward,
            "real_failure_shipped": self.state.real_failure_shipped,
            "message": message,
        }
        return obs, reward, self.state.done, info

    def get_state(self) -> dict:
        """Return full internal state as dict (for /state endpoint)."""
        if self.state is None:
            return {"error": "No active episode. Call /reset first."}
        return self.state.model_dump()

    # ------------------------------------------------------------------
    # Action handlers
    # ------------------------------------------------------------------

    def _handle_run(self, action: Action) -> Tuple[float, Optional[StageResult], str]:
        sid = action.stage_id
        if not sid or sid not in self.stages:
            return -0.05, None, f"Invalid stage_id: {sid}"

        ok, reason = self.sim.check_dependency_satisfied(sid, self.stages, self.state)
        if not ok:
            return -0.05, None, reason

        if sid in self.state.completed_stages or sid in self.state.skipped_stages:
            return -0.05, None, f"Stage {sid} already finished."

        result = self.sim.run_stage(sid, self.stages, self.state)
        self.state.total_elapsed_time += result.duration_seconds

        if result.exit_code == 0:
            self.state.completed_stages.append(sid)
            self.state.stage_statuses[sid] = "passed"
            self.stages[sid].last_known_status = "passed"
            message = f"Stage '{sid}' passed in {result.duration_seconds:.1f}s."
        else:
            self.state.failed_stages.append(sid)
            self.state.stage_statuses[sid] = "failed"
            self.stages[sid].last_known_status = "failed"
            self.state.last_failed_stage = sid
            message = f"Stage '{sid}' FAILED — {result.stdout_tail}"

        reward = compute_reward(action, result, self.state, self.stages)
        return reward, result, message

    def _handle_run_parallel(self, action: Action) -> Tuple[float, Optional[StageResult], str]:
        sids = action.parallel_stages or []
        if len(sids) < 2:
            return -0.05, None, "run_parallel requires at least 2 stages."

        for sid in sids:
            if sid not in self.stages:
                return -0.05, None, f"Unknown stage: {sid}"
            ok, reason = self.sim.check_dependency_satisfied(sid, self.stages, self.state)
            if not ok:
                return -0.05, None, f"Stage {sid}: {reason}"

        time_saved = self.sim.estimate_time_saved(sids, self.stages)
        max_duration = 0.0
        messages = []
        last_result = None

        for sid in sids:
            result = self.sim.run_stage(sid, self.stages, self.state)
            max_duration = max(max_duration, result.duration_seconds)
            last_result = result
            if result.exit_code == 0:
                self.state.completed_stages.append(sid)
                self.state.stage_statuses[sid] = "passed"
                self.stages[sid].last_known_status = "passed"
                messages.append(f"{sid}✓")
            else:
                self.state.failed_stages.append(sid)
                self.state.stage_statuses[sid] = "failed"
                self.stages[sid].last_known_status = "failed"
                self.state.last_failed_stage = sid
                messages.append(f"{sid}✗")

        self.state.total_elapsed_time += max_duration
        self.state.parallel_runs_made += 1

        reward = compute_reward(action, last_result, self.state, self.stages, time_saved=time_saved)
        return reward, last_result, f"Parallel run: {', '.join(messages)}"

    def _handle_skip(self, action: Action) -> Tuple[float, str]:
        sid = action.stage_id
        if not sid or sid not in self.stages:
            return -0.05, f"Invalid stage_id: {sid}"

        stage = self.stages[sid]
        self.state.skipped_stages.append(sid)
        self.state.stage_statuses[sid] = "skipped"
        stage.last_known_status = "skipped"

        if stage.is_critical:
            self.state.skipped_critical.append(sid)
        elif sid in self.state.flap_stages:
            self.state.correctly_skipped_flaps.append(sid)

        reward = compute_reward(action, None, self.state, self.stages)
        return reward, f"Stage '{sid}' skipped. Reason: {action.reason}"

    def _handle_retry(self, action: Action) -> Tuple[float, Optional[StageResult], str]:
        sid = action.stage_id
        if not sid or sid not in self.stages:
            return -0.05, None, f"Invalid stage_id: {sid}"

        self.state.stage_retry_counts[sid] = self.state.stage_retry_counts.get(sid, 0) + 1

        result = self.sim.run_stage(sid, self.stages, self.state)
        self.state.total_elapsed_time += result.duration_seconds

        if result.exit_code == 0:
            if sid in self.state.failed_stages:
                self.state.failed_stages.remove(sid)
            self.state.completed_stages.append(sid)
            self.state.stage_statuses[sid] = "passed"
            self.stages[sid].last_known_status = "passed"
            result.passed_after_retry = True
            if sid in self.state.flap_stages:
                self.state.correctly_retried_flaps.append(sid)
            message = f"Retry succeeded! '{sid}' now passing."
        else:
            self.state.stage_statuses[sid] = "failed"
            self.state.last_failed_stage = sid
            if sid in self.state.real_failures:
                self.state.incorrectly_retried_real_failures.append(sid)
                self.state.unnecessary_retries += 1
            message = f"Retry failed again for '{sid}'."

        reward = compute_reward(action, result, self.state, self.stages)
        return reward, result, message

    def _handle_abort(self, action: Action) -> Tuple[float, str]:
        self.state.aborted = True
        self.state.done = True

        if self.state.last_failed_stage in self.state.real_failures:
            self.state.real_failure_caught = True
            # Hard task: detect cascade
            if self.state.task_id == "hard":
                lf = self.state.last_failed_stage
                for sid, hidden in self.state.hidden_dependencies.items():
                    if lf in hidden:
                        self.state.cascade_detected = True
                        break

        reward = compute_reward(action, None, self.state, self.stages)
        return reward, f"Pipeline ABORTED. Reason: {action.reason}"

    def _handle_inspect(self, action: Action) -> Tuple[float, str]:
        sid = action.stage_id
        if not sid or sid not in self.stages:
            return 0.0, f"Invalid stage_id: {sid}"

        flap_rate = self.state.flap_probabilities.get(sid, 0.0)
        if sid not in self.state.inspected_stages:
            self.state.inspected_stages.append(sid)
        self.state.stage_flap_history[sid] = flap_rate  # stored in state for observation

        reward = compute_reward(action, None, self.state, self.stages)
        is_critical = self.stages[sid].is_critical
        return reward, f"Inspect '{sid}': flap_rate={flap_rate:.0%}, is_critical={is_critical}"

    # ------------------------------------------------------------------
    # Done check & observation builder
    # ------------------------------------------------------------------

    def _check_done(self):
        state = self.state
        if state.done:
            return

        # Out of steps
        if state.steps_taken >= state.step_budget:
            state.done = True
            # Check if any real failure wasn't caught
            for rf in state.real_failures:
                if rf in state.skipped_stages or (
                    rf not in state.completed_stages and rf not in state.failed_stages
                ):
                    state.real_failure_shipped = True
            return

        # All stages resolved (done, skipped, or failed and not retried)
        pending = [
            sid for sid in self.stages
            if sid not in state.completed_stages
            and sid not in state.skipped_stages
            and sid not in state.failed_stages
        ]
        if not pending and not state.aborted:
            state.done = True
            # Check if any real failure slipped through
            for rf in state.real_failures:
                if rf in state.completed_stages or rf in state.skipped_stages:
                    state.real_failure_shipped = True
            if not state.real_failure_shipped and all(
                rf in state.failed_stages for rf in state.real_failures
            ):
                state.pipeline_passed_clean = False  # caught properly
            return

    def _build_observation(
        self,
        reward: float = 0.0,
        score: Optional[float] = None,
        message: str = "",
    ) -> Observation:
        state = self.state
        stages_list = list(self.stages.values())

        pending = [
            sid for sid in self.stages
            if sid not in state.completed_stages
            and sid not in state.failed_stages
            and sid not in state.skipped_stages
        ]
        runnable = self.sim.get_runnable_stages(self.stages, state) if self.sim else []

        # Estimated remaining time: sum of pending runnable stage durations
        est_remaining = sum(
            self.stages[sid].estimated_duration for sid in pending if sid in self.stages
        )

        return Observation(
            pipeline_id=state.pipeline_id,
            stages=stages_list,
            dag_structure={sid: s.dependencies for sid, s in self.stages.items()},
            completed_stages=list(state.completed_stages),
            failed_stages=list(state.failed_stages),
            skipped_stages=list(state.skipped_stages),
            running_stages=[],
            pending_stages=pending,
            runnable_stages=runnable,
            last_stage_result=None,
            elapsed_time_seconds=round(state.total_elapsed_time, 1),
            estimated_remaining=round(est_remaining, 1),
            stage_flap_history=dict(state.stage_flap_history),
            steps_taken=state.steps_taken,
            steps_remaining=state.step_budget - state.steps_taken,
            reward=round(reward, 4),
            cumulative_reward=round(state.cumulative_reward + reward, 4),
            done=state.done,
            score=score,
            message=message,
        )
