# LegalMind-RL

**A courtroom simulation environment for RL agents** — implements the OpenEnv spec with `reset()`, `step()`, and `state()`.

> 🏛️ You are NOT building the lawyer AI. You are building the **courtroom** — a structured environment where any AI agent can argue, question witnesses, present evidence, and be evaluated with continuous reward feedback.

---

## Overview

LegalMind-RL puts an agent in the role of prosecution or defense counsel. The environment progresses through realistic courtroom phases, evaluates every action with dense rewards, and produces a normalized score between 0.0 and 1.0 via deterministic graders.

### Why legal reasoning?

Legal argumentation requires:
- **Evidence synthesis** — connecting disparate facts into a coherent narrative
- **Contradiction detection** — spotting inconsistencies across witness testimony
- **Strategic action selection** — knowing when to argue vs. question vs. object
- **Phase awareness** — opening statements differ from cross-examination

This maps naturally onto RL: states are rich, the action space is structured, and rewards can be shaped step-by-step.

---

## Environment Design

### State (Observation)

At each step, the agent sees:

| Field | Type | Description |
|-------|------|-------------|
| `case_summary` | string | Plain-English case description |
| `current_phase` | enum | `opening / examination / cross_examination / closing / verdict` |
| `round_number` | int | Current step within episode |
| `available_evidence` | list | All evidence items with ID, description, type, used-flag |
| `witnesses` | list | Witness names and their testimonies |
| `conversation_history` | list | Last N turns from both sides |
| `opponent_last_action` | string | What the opposing side just said |
| `valid_actions` | list | Actions permitted in current phase |
| `score_so_far` | float | Running cumulative reward |

### Action Space

| Action | Valid Phase(s) | Description |
|--------|----------------|-------------|
| `argue` | opening, closing | Present a legal argument |
| `question` | examination, cross_examination | Question a witness |
| `present_evidence` | examination, cross_examination | Introduce evidence by ID |
| `object` | cross_examination | Object to opponent's move |

```json
{
  "action": "present_evidence",
  "target": "E001",
  "content": "The security footage at 14:32 directly shows the defendant concealing the item."
}
```

### Reward Function (Dense)

Rewards are issued **per step** in the range `[-0.5, +0.5]`:

| Event | Reward |
|-------|--------|
| New, relevant evidence presented | +0.20 |
| Contradiction identified and leveraged | +0.20 |
| Strong, evidence-backed argument | +0.10 |
| Correct action for current phase | +0.05 |
| Targeted witness by name in question | +0.05 |
| Invalid action for phase | −0.15 |
| Repeated content (≥60% word overlap) | −0.10 |
| Weak, uncertain argument | −0.05 |
| Repeated evidence already presented | −0.05 |

This is **dense reward shaping** — the agent receives informative feedback at every step, not just a binary win/loss at the end.

### Episode End

An episode ends when:
1. The maximum rounds for that task are exhausted, OR
2. All phases have been completed (verdict phase reached)

---

## Tasks

### Task 1 — Easy: The Riverside Market Theft (`task_shoplifting_v1`)

- **Setting**: Shoplifting at a retail store
- **Evidence**: Security camera footage, eyewitness, physical possession of stolen goods, receipt log
- **Witnesses**: Store employee (credible), loss prevention officer (credible)
- **Key challenge**: Present the right evidence in the right order
- **Max rounds**: 8
- **Baseline score**: ~0.65

### Task 2 — Medium: The Harmon Capital Fraud (`task_fraud_v1`)

- **Setting**: CFO embezzlement via shell company
- **Evidence**: Wire transfer records, shell company registration, self-dealing emails, no services on record
- **Witnesses**: CEO (contradicts signed documents), accounting manager (contradicts her own stated policy)
- **Key challenge**: Identify 2 witness contradictions AND build the financial paper trail
- **Max rounds**: 12
- **Baseline score**: ~0.45

### Task 3 — Hard: The Vantage Hotel Homicide (`task_homicide_v1`)

- **Setting**: Premeditated murder at a hotel; timeline disputed; alibi contested
- **Evidence**: 8 items including keycard logs, DNA, text messages, alibi receipt, toxicology with ±45 min uncertainty
- **Witnesses**: Forensic pathologist (narrowed TOD inconsistently), alibi witness (1h23m gap in card activity), lab technician (unrefrigerated DNA sample)
- **Key challenge**: Synthesize 8 evidence items, expose 3 contradictions, dismantle alibi
- **Max rounds**: 16
- **Baseline score**: ~0.30

---

## Graders

Each task has a deterministic grader outputting a score in `[0.0, 1.0]`.

### Easy grader
```
score = 0.40 × (key_evidence_used / 2)
      + 0.30 × (cumulative_reward / 2.0)
      + 0.30 × completion_bonus
```

### Medium grader
```
score = 0.30 × (key_evidence_used / 3)
      + 0.35 × (contradictions_found / 2)
      + 0.35 × (cumulative_reward / 3.5)
```

### Hard grader
```
score = 0.25 × (key_evidence_used / 4)
      + 0.40 × (contradictions_found / 3)
      + 0.20 × (cumulative_reward / 5.0)
      + 0.15 × (all_3_contradictions_bonus)
```

---

## Setup

### Local (Python)

```bash
git clone https://github.com/your-org/legalmind-rl
cd legalmind-rl
pip install -r requirements.txt

# Start environment server
python server.py
```

Server runs at `http://localhost:7860`.

### Docker

```bash
docker build -t legalmind-rl .
docker run -p 7860:7860 \
  -e HF_TOKEN=your_token \
  -e API_BASE_URL=https://api.openai.com/v1 \
  -e MODEL_NAME=gpt-4o-mini \
  legalmind-rl
```

### Hugging Face Spaces

Deploy to a Spaces instance with Docker runtime. Set secrets:
- `HF_TOKEN` — your API key
- `API_BASE_URL` — your LLM endpoint
- `MODEL_NAME` — model name

---

## Running Inference

```bash
export API_BASE_URL=https://api.openai.com/v1
export MODEL_NAME=gpt-4o-mini
export HF_TOKEN=sk-...
export TASK=medium        # easy | medium | hard
export ROLE=prosecution   # prosecution | defense
export ENV_BASE_URL=http://localhost:7860

python inference.py
```

**Expected output:**
```
[START] task=medium env=legalmind-rl model=gpt-4o-mini
[STEP] step=1 action='argue(...): Opening statement ...' reward=0.15 done=false error=null
[STEP] step=2 action='present_evidence(E002): The shell...' reward=0.20 done=false error=null
...
[END] success=true steps=10 score=0.52 rewards=0.15,0.20,0.10,...
```

---

## API Reference

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Liveness check |
| POST | `/reset` | Start episode `{task, role, session_id}` |
| POST | `/step` | Submit action `{session_id, action, target, content}` |
| GET | `/state/{session_id}` | Current state snapshot |
| POST | `/score` | Graded score `{session_id}` |
| POST | `/close/{session_id}` | End session |
| GET | `/tasks` | List all available tasks |

---

## Project Structure

```
legalmind-rl/
├── legalmind_env/
│   ├── __init__.py
│   ├── environment.py    # Core engine: reset/step/state
│   ├── models.py         # Pydantic typed models
│   └── reward.py         # Dense reward function
├── tasks/
│   ├── __init__.py
│   └── cases.py          # 3 task definitions
├── graders/
│   ├── __init__.py
│   └── graders.py        # Deterministic 0→1 graders
├── server.py             # FastAPI HTTP server
├── inference.py          # Baseline LLM agent (OpenAI client)
├── openenv.yaml          # OpenEnv spec
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Baseline Scores

| Task | Model | Role | Score | Steps |
|------|-------|------|-------|-------|
| easy | gpt-4o-mini | prosecution | ~0.65 | 8 |
| medium | gpt-4o-mini | prosecution | ~0.45 | 12 |
| hard | gpt-4o-mini | prosecution | ~0.30 | 16 |

Scores above 0.75 require deliberate contradiction identification and full evidence coverage.

---

## License

MIT
