"""
inference.py — PipelineForge baseline inference script.

Environment variables:
  API_BASE_URL   API endpoint for the LLM (default: https://api.openai.com/v1)
  MODEL_NAME     Model identifier (default: gpt-4o-mini)
  HF_TOKEN       Hugging Face / API key (REQUIRED — no default)
  ENV_URL        PipelineForge server URL (default: http://localhost:7860)

Output format (strict — evaluated by OpenEnv):
  [START] task=<task> env=<env> model=<model>
  [STEP]  step=<n> action=<str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> rewards=<r1,r2,...>
"""
from __future__ import annotations
import json
import os
import sys
import textwrap
from typing import List

import requests
from openai import OpenAI

# ─────────────────────────────────────────────────────────────────────────────
# Environment variables  (defaults required by spec for API_BASE_URL, MODEL_NAME)
# ─────────────────────────────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "gpt-4o-mini")
HF_TOKEN     = os.getenv("HF_TOKEN")
ENV_URL      = os.getenv("ENV_URL",      "http://localhost:7860")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

TEMPERATURE = 0.0
MAX_TOKENS  = 512
MAX_STEPS   = 25
TASKS       = ["easy", "medium", "hard", "expert"]
ENV_NAME    = "pipelineforge"

SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert CI/CD pipeline manager controlling a PipelineForge environment.
    You receive the current pipeline state as JSON and must return a single JSON action.

    RULES:
    - Prefer run_parallel when multiple independent stages are runnable.
    - If a stage fails and its flap_rate > 0.2, retry it (likely transient).
    - If a critical stage fails and flap_rate is 0 or unknown, abort immediately.
    - Never skip a critical stage.
    - Use inspect to learn flap rates before deciding on ambiguous failures.

    Return ONLY valid JSON:
    {
      "action_type": "run" | "run_parallel" | "skip" | "retry" | "abort" | "inspect",
      "stage_id": "<id or null>",
      "parallel_stages": ["<id>", ...] or null,
      "retry_count": 1 | 2 | 3 or null,
      "reason": "<short explanation>"
    }
""").strip()


# ─────────────────────────────────────────────────────────────────────────────
# Logging — exact format required by OpenEnv evaluator
# ─────────────────────────────────────────────────────────────────────────────

def log_start(task: str) -> None:
    """[START] task=<task> env=<env> model=<model>"""
    print(f"[START] task={task} env={ENV_NAME} model={MODEL_NAME}", flush=True)


def log_step(step: int, action: dict, reward: float, done: bool, error: str | None) -> None:
    """[STEP] step=<n> action=<str> reward=<0.00> done=<bool> error=<msg|null>"""
    action_str = action.get("action_type", "unknown")
    stage = action.get("stage_id") or action.get("parallel_stages")
    if stage:
        action_str = f"{action_str}({stage})"
    err_val = error if error is not None else "null"
    print(
        f"[STEP] step={step} action={action_str} "
        f"reward={reward:.2f} done={str(done).lower()} error={err_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, rewards: List[float]) -> None:
    """[END] success=<bool> steps=<n> rewards=<r1,r2,...>"""
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}",
        flush=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Environment HTTP helpers
# ─────────────────────────────────────────────────────────────────────────────

def env_reset(task_id: str, seed: int = 42) -> dict:
    r = requests.post(f"{ENV_URL}/reset", json={"task_id": task_id, "seed": seed}, timeout=30)
    r.raise_for_status()
    return r.json()


def env_step(action: dict) -> dict:
    r = requests.post(f"{ENV_URL}/step", json=action, timeout=30)
    r.raise_for_status()
    return r.json()


# ─────────────────────────────────────────────────────────────────────────────
# LLM agent
# ─────────────────────────────────────────────────────────────────────────────

def build_user_prompt(step: int, obs: dict, history: List[str]) -> str:
    history_block = "\n".join(history[-6:]) if history else "None"
    obs_summary = {
        "pipeline_id":        obs.get("pipeline_id"),
        "steps_remaining":    obs.get("steps_remaining"),
        "elapsed_time_s":     obs.get("elapsed_time_seconds"),
        "runnable_stages":    obs.get("runnable_stages", []),
        "completed_stages":   obs.get("completed_stages", []),
        "failed_stages":      obs.get("failed_stages", []),
        "skipped_stages":     obs.get("skipped_stages", []),
        "stage_flap_history": obs.get("stage_flap_history", {}),
        "last_message":       obs.get("message", ""),
        "cumulative_reward":  obs.get("cumulative_reward", 0),
    }
    return textwrap.dedent(f"""
        Step: {step}
        Observation:
        {json.dumps(obs_summary, indent=2)}
        Recent history:
        {history_block}
        Choose your next action (return JSON only).
    """).strip()


def get_model_action(client: OpenAI, step: int, obs: dict, history: List[str]) -> dict:
    user_prompt = build_user_prompt(step, obs, history)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        text = (completion.choices[0].message.content or "").strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        return json.loads(text.strip())
    except Exception as exc:
        print(f"[DEBUG] Model error at step {step}: {exc}", flush=True)
        return _rule_based_action(obs)


def _rule_based_action(obs: dict) -> dict:
    """Fallback: rule-based agent used when LLM call fails."""
    runnable  = obs.get("runnable_stages", [])
    failed    = obs.get("failed_stages", [])
    flap_hist = obs.get("stage_flap_history", {})

    if failed:
        last_failed = failed[-1]
        flap_rate   = flap_hist.get(last_failed)
        if flap_rate is None:
            return {"action_type": "inspect", "stage_id": last_failed,
                    "reason": "checking flap rate before deciding"}
        if flap_rate > 0.2:
            return {"action_type": "retry", "stage_id": last_failed,
                    "retry_count": 1, "reason": "high flap rate, retrying"}
        return {"action_type": "abort",
                "reason": f"{last_failed} appears to be a real failure"}

    if len(runnable) >= 2:
        return {"action_type": "run_parallel", "parallel_stages": runnable[:4],
                "reason": "exploit parallelism"}
    if len(runnable) == 1:
        return {"action_type": "run", "stage_id": runnable[0],
                "reason": "run next available stage"}
    return {"action_type": "abort", "reason": "no runnable stages remaining"}


# ─────────────────────────────────────────────────────────────────────────────
# Run one task episode
# ─────────────────────────────────────────────────────────────────────────────

def run_task(client: OpenAI, task_id: str, seed: int = 42) -> float:
    log_start(task=task_id)

    obs        = env_reset(task_id, seed=seed)
    history:   List[str]   = []
    rewards:   List[float] = []
    steps_taken = 0
    success     = False

    for step in range(1, MAX_STEPS + 1):
        if obs.get("done"):
            break

        action = get_model_action(client, step, obs, history)
        error  = None

        try:
            result = env_step(action)
        except Exception as exc:
            error = str(exc)
            log_step(step=step, action=action, reward=0.0, done=True, error=error)
            rewards.append(0.0)
            steps_taken = step
            break

        reward = float(result.get("reward", 0.0))
        done   = bool(result.get("done", False))
        obs    = result
        steps_taken = step
        rewards.append(reward)

        log_step(step=step, action=action, reward=reward, done=done, error=None)

        history.append(
            f"Step {step}: {action.get('action_type')} "
            f"stage={action.get('stage_id') or action.get('parallel_stages')} "
            f"reward={reward:+.2f}"
        )

        if done:
            success = not result.get("info", {}).get("real_failure_shipped", False)
            break

    log_end(success=success, steps=steps_taken, rewards=rewards)
    score = obs.get("score") or obs.get("info", {}).get("score") or 0.0
    return float(score)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    # Verify environment is reachable
    try:
        r = requests.get(f"{ENV_URL}/health", timeout=10)
        r.raise_for_status()
    except Exception as e:
        print(f"[ERROR] Environment not reachable at {ENV_URL}: {e}", flush=True)
        sys.exit(1)

    task_scores = {}
    for task in TASKS:
        print(f"\n{'='*60}", flush=True)
        print(f"Running task: {task}", flush=True)
        print("=" * 60, flush=True)
        try:
            task_scores[task] = run_task(client, task_id=task, seed=42)
        except Exception as exc:
            print(f"[ERROR] Task {task} failed: {exc}", flush=True)
            task_scores[task] = 0.0

    print(f"\n{'='*60}", flush=True)
    print("BASELINE RESULTS (seed=42)", flush=True)
    print("=" * 60, flush=True)
    for task, score in task_scores.items():
        bar = "█" * int(score * 20)
        print(f"  {task:8s}: {score:.2f}  {bar}", flush=True)
    avg = sum(task_scores.values()) / max(len(task_scores), 1)
    print(f"\n  Average : {avg:.2f}", flush=True)


if __name__ == "__main__":
    main()
