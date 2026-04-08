---

title: LegalMind RL
emoji: ⚖️
colorFrom: blue
colorTo: purple
sdk: docker
app_file: server/app.py
pinned: false
-------------

# LegalMind-RL

**A courtroom simulation environment for RL agents** — implements the OpenEnv spec with `reset()`, `step()`, and `state()`.

> 🏛️ You are NOT building the lawyer AI. You are building the **courtroom** — a structured environment where any AI agent can argue, question witnesses, present evidence, and be evaluated with continuous reward feedback.

---

## 🚀 Overview

LegalMind-RL places an agent in the role of prosecution or defense counsel inside a structured courtroom simulation.

The environment:

* Progresses through **realistic legal phases**
* Provides **dense reward signals**
* Outputs a **deterministic score (0.0 → 1.0)**

---

## 🧠 Why This Matters

Legal reasoning requires:

* Evidence synthesis
* Contradiction detection
* Strategic decision-making
* Phase-aware argumentation

👉 This makes it a **perfect real-world RL environment**

---

## ⚙️ Environment Design

### 📥 Observation Space

| Field                | Description                             |
| -------------------- | --------------------------------------- |
| case_summary         | Case description                        |
| current_phase        | opening / examination / cross / closing |
| round_number         | Current step                            |
| available_evidence   | Evidence objects                        |
| witnesses            | Witness testimonies                     |
| conversation_history | Past dialogue                           |
| valid_actions        | Allowed actions                         |
| score_so_far         | Cumulative reward                       |

---

### 🎮 Action Space

| Action           | Description      |
| ---------------- | ---------------- |
| argue            | Present argument |
| question         | Ask witness      |
| present_evidence | Use evidence     |
| object           | Raise objection  |

Example:

```json
{
  "action": "present_evidence",
  "target": "E001",
  "content": "Security footage shows concealment."
}
```

---

### 🎯 Reward Function

Dense reward in range `[-0.5, +0.5]`

| Event                   | Reward |
| ----------------------- | ------ |
| Evidence used correctly | +0.20  |
| Contradiction found     | +0.20  |
| Strong argument         | +0.10  |
| Correct phase action    | +0.05  |
| Invalid action          | −0.15  |
| Repetition              | −0.10  |

👉 Provides **continuous learning signal**

---

### 🏁 Episode End

* Max rounds reached
* OR verdict phase completed

---

## 🧪 Tasks

### 🟢 Easy — Shoplifting Case

* Clear evidence
* Goal: basic reasoning

---

### 🟡 Medium — Fraud Case

* Conflicting testimony
* Goal: detect contradictions

---

### 🔴 Hard — Homicide Case

* Complex timeline
* Goal: multi-step reasoning

---

## ⚖️ Grading System

Each task returns score **0.0 → 1.0**

* Evidence usage
* Contradictions found
* Reward accumulated

👉 Fully deterministic (OpenEnv compliant)

---

## 🛠️ Setup

### ▶️ Run Locally

```bash
pip install -r server/requirements.txt
python -m uvicorn server.app:app --host 0.0.0.0 --port 7860
```

---

### 🐳 Docker

```bash
docker build -t legalmind-rl .
docker run -p 7860:7860 legalmind-rl
```

---

## 🤖 Inference

```bash
export API_BASE_URL=...
export MODEL_NAME=...
export HF_TOKEN=...

python inference.py
```

---

## 📡 API Endpoints

| Endpoint        | Description   |
| --------------- | ------------- |
| GET /health     | Health check  |
| POST /reset     | Start episode |
| POST /step      | Take action   |
| GET /state/{id} | Get state     |
| POST /score     | Get score     |

---

## 📁 Project Structure

```
legalmind-rl/
├── inference.py
├── models.py
├── reward.py
├── tasks.py
├── graders.py
├── client.py
├── openenv.yaml
├── pyproject.toml
└── server/
    ├── app.py
    ├── legal_environment.py
    ├── Dockerfile
    └── requirements.txt
```

---

## 🏆 Key Features

* Real-world task (legal reasoning)
* Multi-step decision environment
* Dense reward shaping
* Deterministic grading
* OpenEnv compliant
* Docker + HF deployable

---

## 📊 Baseline Performance

| Task   | Score |
| ------ | ----- |
| Easy   | ~0.65 |
| Medium | ~0.45 |
| Hard   | ~0.30 |

---

## 📜 License

MIT
