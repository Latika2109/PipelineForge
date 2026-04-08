from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


class Action(BaseModel):
    """Agent action for the pipeline environment."""
    action_type: str  # run | run_parallel | skip | retry | abort | wait | inspect
    stage_id: Optional[str] = None
    parallel_stages: Optional[List[str]] = None
    retry_count: Optional[int] = 1  # 1–3
    reason: str = ""


class StageMeta(BaseModel):
    """Stage metadata visible to the agent."""
    stage_id: str
    name: str
    estimated_duration: int  # seconds
    dependencies: List[str]
    is_critical: bool
    last_known_status: str = "not_run"  # not_run | running | passed | failed | skipped
    level: int = 0  # topological level for UI rendering


class StageResult(BaseModel):
    """Result of executing a stage (is_flap hidden from agent observations)."""
    stage_id: str
    exit_code: int  # 0 = success
    duration_seconds: float
    stdout_tail: str
    retry_count_so_far: int = 0
    passed_after_retry: bool = False


class Observation(BaseModel):
    """Full observation returned to agent after each action."""
    pipeline_id: str
    stages: List[StageMeta]
    dag_structure: Dict[str, List[str]]  # stage_id -> [depends_on, ...]

    # Execution state
    completed_stages: List[str] = Field(default_factory=list)
    failed_stages: List[str] = Field(default_factory=list)
    skipped_stages: List[str] = Field(default_factory=list)
    running_stages: List[str] = Field(default_factory=list)
    pending_stages: List[str] = Field(default_factory=list)
    runnable_stages: List[str] = Field(default_factory=list)

    # Last action result (is_flap is NOT exposed here)
    last_stage_result: Optional[Dict[str, Any]] = None

    # Time tracking
    elapsed_time_seconds: float = 0.0
    estimated_remaining: float = 0.0

    # Flap history (populated only after inspect action)
    stage_flap_history: Dict[str, float] = Field(default_factory=dict)

    # Episode progress
    steps_taken: int = 0
    steps_remaining: int = 0

    # Reward & episode status
    reward: float = 0.0
    cumulative_reward: float = 0.0
    done: bool = False
    score: Optional[float] = None
    message: str = ""


class EpisodeState(BaseModel):
    """Full internal state — contains ground truth hidden from the agent."""
    pipeline_id: str
    task_id: str
    step_budget: int

    # Ground truth (NEVER expose to agent)
    real_failures: List[str]
    flap_stages: List[str]
    flap_probabilities: Dict[str, float]
    optimal_execution_time: float
    hidden_dependencies: Dict[str, List[str]] = Field(default_factory=dict)

    # Execution tracking
    execution_log: List[Dict[str, Any]] = Field(default_factory=list)
    total_elapsed_time: float = 0.0
    cumulative_reward: float = 0.0

    # Stage status tracking
    completed_stages: List[str] = Field(default_factory=list)
    failed_stages: List[str] = Field(default_factory=list)
    skipped_stages: List[str] = Field(default_factory=list)
    stage_retry_counts: Dict[str, int] = Field(default_factory=dict)
    stage_statuses: Dict[str, str] = Field(default_factory=dict)
    inspected_stages: List[str] = Field(default_factory=list)
    stage_flap_history: Dict[str, float] = Field(default_factory=dict)

    # Grader decision quality tracking
    skipped_critical: List[str] = Field(default_factory=list)
    unnecessary_retries: int = 0
    correctly_retried_flaps: List[str] = Field(default_factory=list)
    incorrectly_retried_real_failures: List[str] = Field(default_factory=list)
    correctly_skipped_flaps: List[str] = Field(default_factory=list)
    parallel_runs_made: int = 0
    parallelism_opportunities_total: int = 0

    # Outcome flags
    real_failure_caught: bool = False
    real_failure_shipped: bool = False
    pipeline_passed_clean: bool = False
    aborted: bool = False
    abort_reason: str = ""

    # Last action tracking (needed by reward.py)
    last_failed_stage: Optional[str] = None
    last_action_type: Optional[str] = None
    last_retry_passed: bool = False

    # Episode counters
    steps_taken: int = 0
    done: bool = False

    # Hard task specific
    cascade_detected: bool = False
    ambiguous_failures_classified_correctly: int = 0
