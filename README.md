---
title: PipelineForge
emoji: ⚙️
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
tags:
- openenv
- cicd
- reinforcement-learning
- devops
---

# PipelineForge — CI/CD Pipeline Optimizer Environment

[![OpenEnv](https://img.shields.io/badge/openenv-compatible-brightgreen)](https://openenv.dev)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Space-yellow)](https://huggingface.co/spaces)
[![Docker](https://img.shields.io/badge/Docker-ready-blue)](https://docker.com)
[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)

> An OpenEnv-compatible reinforcement learning environment where an AI agent acts as a CI/CD pipeline manager — deciding how to execute, retry, skip, and parallelize build stages to minimize total pipeline time while catching all real failures.

---

## Environment Description

Every software team runs CI/CD pipelines. And every team knows the pain: a test flaked so the whole build retried for 20 minutes. Or a slow integration test ran before fast unit tests. Or stages ran sequentially when they could have been parallelized. Or a real failure got retried 3 times instead of failing fast.

**PipelineForge** turns CI/CD pipeline execution into an RL environment. The agent is given a pipeline DAG — a directed acyclic graph of build stages with known execution times, flap rates, and dependencies. It must decide how to execute each stage: run it, skip it, retry it, parallelize it, or abort. The goal is to minimize total wall-clock time while never letting a real failure ship to production.

The **core challenge**: distinguishing transient *flap* failures (network blips, resource contention — retry and they pass) from *real* failures (broken code, schema violations — retrying wastes time and the pipeline must abort).

---

## Project Structure

```
PipelineForge/
env.py # Core OpenEnv class (step/reset/state)
grader.py # Task graders, 0.0–1.0 scoring
reward.py # Reward computation logic
models.py # Pydantic typed models
simulator.py # DAG execution engine + stage runner
pipeline_bank/
task_easy.json # Task 1: Linear 6-stage pipeline
task_medium.json # Task 2: Parallel DAG, 12 stages
task_hard.json # Task 3: Complex DAG, adversarial flaps
app.py # FastAPI server + interactive web UI
inference.py # OpenAI-client baseline inference script
openenv.yaml # OpenEnv manifest
requirements.txt
Dockerfile
README.md
```

---

## Action Space

| Action | Description |
|---|---|
| `run(stage_id)` | Execute a stage normally |
| `run_parallel([s1, s2, ...])` | Execute multiple independent stages simultaneously |
| `skip(stage_id, reason)` | Skip a stage (smart for known-flaky non-critical stages) |
| `retry(stage_id, n)` | Retry a failed stage N times (1–3) |
| `abort(reason)` | Halt entire pipeline — real failure detected |
| `wait(stage_id)` | Wait for a running stage |
| `inspect(stage_id)` | View stage flap history before deciding |

**Pydantic Action Model:**
```python
class Action(BaseModel):
action_type: str # run | run_parallel | skip | retry | abort | wait | inspect
stage_id: Optional[str]
parallel_stages: Optional[List[str]]
retry_count: Optional[int] # 1, 2, or 3
reason: str
```

---

## Observation Space

```python
class Observation(BaseModel):
pipeline_id: str
stages: List[StageMeta] # all stages with dependencies visible
dag_structure: Dict # adjacency list: stage → [depends_on]
completed_stages: List[str]
failed_stages: List[str]
skipped_stages: List[str]
running_stages: List[str]
pending_stages: List[str]
runnable_stages: List[str] # stages ready to execute now
last_stage_result: Optional[Dict]
elapsed_time_seconds: float
estimated_remaining: float
stage_flap_history: Dict[str, float] # revealed after inspect
steps_taken: int
steps_remaining: int
reward: float
cumulative_reward: float
done: bool
score: Optional[float]
message: str
```

**StageMeta (per stage):**
```python
class StageMeta(BaseModel):
stage_id: str
name: str
estimated_duration: int # seconds
dependencies: List[str]
is_critical: bool # if this fails, pipeline must not ship
last_known_status: str # not_run | passed | failed | skipped
level: int # topological level (for DAG rendering)
```

---

## Tasks

### Task 1 — Easy: Linear Pipeline, Predictable Failures

**Pipeline:** `install-deps → lint → unit-tests → build → integration-tests → deploy-staging`

| Stage | Duration | Flap Rate | Critical | Notes |
|---|---|---|---|---|
| install-deps | 45s | 0% | Yes | Always passes |
| lint | 12s | 0% | No | Always passes |
| unit-tests | 90s | 75% | Yes | Will flap — must retry |
| build | 60s | 0% | Yes | Always passes |
| integration-tests | 180s | 0% | Yes | **Real failure this run** |
| deploy-staging | 30s | 0% | Yes | Would pass if reached |

**Agent must:** Run stages in order. Retry unit-tests when it flaps. When integration-tests fails, identify it as a real failure and abort.

**Target score:** > 0.80 | **Step budget:** 10

---

### Task 2 — Medium: Parallel DAG, Mixed Signals

**Pipeline:** 12-stage microservices pipeline with parallel lint, parallel build, and security scanning.

**Key challenges:**
- 3 lint stages are fully independent → should be parallelized
- 3 build stages are fully independent → should be parallelized
- `lint-service-b` is a heavy flapper (80%) — worth skipping or retrying
- `dependency-audit` has a real failure (CVE found)

**Target score:** > 0.75 | **Step budget:** 18

---

### Task 3 — Hard: Enterprise DAG with Adversarial Patterns

**Pipeline:** 20-stage enterprise monorepo pipeline.

- **3 stages that look like real failures but are flaps** (retry to resolve)
- **2 stages that look like flaps but are real failures** (must abort)
- **Hidden cascade**: `canary-deploy` failure causes `smoke-tests` to fail even though the DAG doesn't show this dependency
- Time budget pressure (550s optimal)

**Target score:** > 0.65 | **Step budget:** 25

---

## Reward Function

Non-sparse rewards issued at every step:

| Situation | Reward |
|---|---|
| Correctly abort on real failure | +0.50 |
| Abort on a flap (wasted run) | -0.40 |
| Retry a flap stage (correct) | +0.20 |
| Retry succeeds | +0.10 bonus |
| Retry a real failure (wrong) | -0.25 |
| Smart skip of non-critical flapper | +0.15 |
| Skip a critical stage | -0.50 |
| Good parallelism (time saved) | up to +0.20 |
| Parallelised dependent stages | -0.20 |
| Within 20% of optimal time | +0.10 |
| Real failure shipped to prod | **-1.00** |

---

## Baseline Scores

Rule-based fallback agent (`seed=42`):

| Task | Rule-Based | Target |
|---|---|---|
| Task 1 — Easy | ~0.55 | 0.80 |
| Task 2 — Medium | ~0.40 | 0.75 |
| Task 3 — Hard | ~0.25 | 0.65 |

---

## Setup & Running Locally

### Prerequisites
- Python 3.11+
- Docker (for containerized run)

### Install

```bash
git clone https://huggingface.co/spaces/YOUR_USERNAME/pipelineforge-env
cd pipelineforge-env
pip install -r requirements.txt
```

### Run the server

```bash
python app.py
# Server starts at http://localhost:7860
# Open your browser — interactive UI is at the root URL
```

### Run the inference script

```bash
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o-mini"
export HF_TOKEN="your_key_here"
export ENV_URL="http://localhost:7860"

python inference.py
```

### Docker

```bash
docker build -t pipelineforge .
docker run -p 7860:7860 \
-e API_BASE_URL="https://api.openai.com/v1" \
-e MODEL_NAME="gpt-4o-mini" \
-e HF_TOKEN="your_key" \
pipelineforge
```

### Validate OpenEnv spec

```bash
pip install openenv-core
openenv validate
```

---

## API Reference

```
POST /reset → { "task_id": "easy" | "medium" | "hard", "seed": 42 }
POST /step → Action JSON body → Observation JSON
GET /state → Full internal state (includes ground truth for debugging)
GET /health → 200 OK
GET / → Interactive web UI
```

**Example session:**

```python
import requests

BASE = "http://localhost:7860"

obs = requests.post(f"{BASE}/reset", json={"task_id": "easy"}).json()

# Run first stage
obs = requests.post(f"{BASE}/step", json={
"action_type": "run",
"stage_id": "install-deps",
"reason": "Starting pipeline"
}).json()
print(obs["reward"]) # → 0.1
print(obs["message"]) # → "Stage 'install-deps' passed in 40.2s."

# If unit-tests flaps, retry it
obs = requests.post(f"{BASE}/step", json={
"action_type": "retry",
"stage_id": "unit-tests",
"retry_count": 1,
"reason": "Known flap rate, retrying once"
}).json()

# When integration-tests fails — abort
obs = requests.post(f"{BASE}/step", json={
"action_type": "abort",
"reason": "integration-tests: genuine failure, not a flap"
}).json()
print(obs["score"]) # → 0.88
print(obs["done"]) # → true
```

---

## Design Decisions

**Why simulate the pipeline instead of running real CI?** Real CI requires external infrastructure and takes minutes per run. A simulated DAG with probabilistic stage outcomes runs in milliseconds, is fully deterministic with a seed, and requires no external credentials — perfect for Docker and HF Spaces.

**Why separate flap vs. real failure?** This is the core decision the agent must learn. Treating every failure as a flap wastes compute; treating every failure as real causes false aborts. The environment forces genuine reasoning.

**Why does Task 3 have a hidden dependency?** Real pipelines often have undocumented dependencies discovered only through failures. The hidden cascade (canary-deploy → smoke-tests) tests whether the agent can infer relationships from failure patterns.

**Why is shipping a real failure worth -1.0?** Because it reflects real-world consequences — broken code in production is the worst possible pipeline outcome.

---

*Built for the OpenEnv Hackathon. PipelineForge — because your pipeline shouldn't be dumber than your CI config.*
