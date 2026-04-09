---
title: SupportOps OpenEnv
emoji: "🎫"
colorFrom: blue
colorTo: green
sdk: docker
tags:
  - openenv
  - reinforcement-learning
  - customer-support
---

# SupportOps OpenEnv (Real-World Customer Support Workflow)

SupportOps is a realistic OpenEnv benchmark that simulates day-to-day customer support operations:
1. ticket triage
2. clarification collection
3. response drafting
4. ticket resolution

It is designed for agent training/evaluation with deterministic graders and shaped rewards.

## Why this environment is useful

Most support automation benchmarks are too shallow (single-step classification only). This environment models multi-step workflows with real operational constraints:
- correct routing (`category`, `priority`, `owner_team`)
- safety/compliance behavior (required clarifications)
- customer communication quality (required/forbidden phrases)
- trajectory quality (penalizes repeated or invalid actions)

## OpenEnv Compliance

This repo implements:
- typed `SupportAction`, `SupportObservation`, `SupportState`, `RewardBreakdown`
- full API via OpenEnv server: `reset`, `step`, `state`
- `openenv.yaml`
- `server.app:app` FastAPI app via `openenv.core.env_server.http_server.create_app`

## Task Set (Easy -> Medium -> Hard)

### 1) `easy`
- Domain: login lockout + 2FA
- Goal: triage correctly and provide safe recovery response
- Expected: `authentication`, `medium`, `support_l1`

### 2) `medium`
- Domain: double card charge (billing)
- Goal: request missing billing identifiers, then route and respond
- Expected: `billing`, `high`, `billing_ops`

### 3) `hard`
- Domain: enterprise webhook signature incident
- Goal: security triage + clarification + mitigation response
- Expected: `security`, `urgent`, `security`

All three tasks have deterministic grader outputs in `[0.0, 1.0]`.

## Action Space

Defined in `support_env/models.py` as `SupportAction`.

Fields:
- `workflow_step`: `triage | ask_clarification | draft_response | resolve_ticket`
- `category`: optional enum
- `priority`: optional enum
- `owner_team`: optional enum
- `message_to_customer`: string
- `internal_note`: string

## Observation Space

Defined in `support_env/models.py` as `SupportObservation`.

Key fields:
- task context (`task_name`, `difficulty`, `objective`)
- ticket context (`ticket_id`, `customer_message`, `known_facts`)
- live progress (`checklist_scores`, `recent_actions`, `steps_remaining`)
- errors (`last_action_error`)
- RL fields (`reward`, `done`)

## Reward Design

Reward is shaped across the full trajectory:
- reward increases when grader score improves (`score_delta`)
- partial credit for each component (category, priority, team, clarification, response quality, resolution, efficiency)
- penalties for repeated actions and invalid workflow actions
- final score and step rewards remain bounded in `[0.0, 1.0]`

Core grader code: `support_env/graders.py`.

## Project Structure

- `support_env/models.py` -> typed models
- `support_env/tasks.py` -> task catalog
- `support_env/graders.py` -> deterministic grader + reward shaping
- `support_env/env.py` -> `SupportOpsEnvironment`
- `support_env/client.py` -> OpenEnv client wrapper
- `server/app.py` -> FastAPI/OpenEnv app
- `openenv.yaml` -> environment metadata
- `inference.py` -> mandatory baseline script with strict `[START]/[STEP]/[END]` logs
- `Dockerfile` -> containerized deployment

## Step-by-Step Setup (Windows PowerShell)

### 1) Open terminal in project root
```powershell
cd D:\DHANYA\smart_support_env
```

### 2) Create virtual environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3) Install dependencies
```powershell
pip install -r requirements.txt
```

### 4) Validate OpenEnv structure
```powershell
openenv validate
```

### 5) Run server locally
```powershell
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

### 6) Check endpoints
Open browser:
- `http://127.0.0.1:8000/health`
- `http://127.0.0.1:8000/schema`

### 7) Run baseline inference
Set env vars first:
```powershell
$env:API_BASE_URL="https://router.huggingface.co/v1"
$env:MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
$env:HF_TOKEN="<your_hf_or_api_token>"
python inference.py
```

## Mandatory Variables (Hackathon)

Configure these in local env and HF Space secrets:
- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN`

`inference.py` also supports fallback keys:
- `OPENAI_API_KEY`
- `API_KEY`

## Strict Inference Logging Format

`inference.py` prints exactly these line types:
- `[START] task=<task_name> env=<benchmark> model=<model_name>`
- `[STEP] step=<n> action=<action_json> reward=<0.00> done=<true|false> error=<msg|null>`
- `[END] success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>`

## Docker

### Build
```powershell
docker build -t support-ops-openenv .
```

### Run
```powershell
docker run --rm -p 8000:8000 support-ops-openenv
```

Then test:
```powershell
curl http://127.0.0.1:8000/health
```

## Hugging Face Space Deployment

1. Create a new **Docker Space** on Hugging Face.
2. Push this repository to that Space.
3. In Space Settings -> Variables/Secrets, add:
   - `API_BASE_URL`
   - `MODEL_NAME`
   - `HF_TOKEN`
4. Ensure Space has tag `openenv`.
5. Once live, verify URL returns healthy endpoints and supports `reset`.

## Baseline Scores (Local Repro Run)

Command used:
```powershell
python inference.py
```

Observed scores:
- `easy`: `1.000`
- `medium`: `1.000`
- `hard`: `1.000`

Why reproducible:
- temperature fixed at `0.0`
- deterministic fallback action planner
- deterministic environment grader