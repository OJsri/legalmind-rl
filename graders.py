"""Deterministic graders — score in [0.0, 1.0] per task."""
from __future__ import annotations
from typing import Any, Dict, List

def grade_easy(state: Dict[str, Any]) -> float:
    used = set(state.get("used_evidence", []))
    key  = {"E001", "E003"}
    ev   = len(key & used) / len(key)
    rw   = max(0.0, min(1.0, state.get("cumulative_reward", 0.0) / 2.0))
    comp = 1.0 if state.get("done", False) else 0.4
    return round(0.40 * ev + 0.30 * rw + 0.30 * comp, 4)

def grade_medium(state: Dict[str, Any]) -> float:
    used = set(state.get("used_evidence", []))
    key  = {"E002", "E004", "E005"}
    ev   = len(key & used) / len(key)
    con  = min(1.0, state.get("contradictions_found", 0) / 2)
    rw   = max(0.0, min(1.0, state.get("cumulative_reward", 0.0) / 3.5))
    return round(0.30 * ev + 0.35 * con + 0.35 * rw, 4)

def grade_hard(state: Dict[str, Any]) -> float:
    used = set(state.get("used_evidence", []))
    key  = {"E001", "E003", "E006", "E007"}
    ev   = len(key & used) / len(key)
    n    = state.get("contradictions_found", 0)
    con  = min(1.0, n / 3)
    rw   = max(0.0, min(1.0, state.get("cumulative_reward", 0.0) / 5.0))
    bon  = 0.15 if n >= 3 else 0.0
    return round(min(1.0, 0.25 * ev + 0.40 * con + 0.20 * rw + bon), 4)

GRADERS = {
    "easy":               grade_easy,   "task_shoplifting_v1": grade_easy,
    "medium":             grade_medium, "task_fraud_v1":        grade_medium,
    "hard":               grade_hard,   "task_homicide_v1":     grade_hard,
}

def get_grader(task_name: str):
    if task_name not in GRADERS:
        raise ValueError(f"No grader for '{task_name}'. Available: {list(GRADERS.keys())}")
    return GRADERS[task_name]


def list_tasks() -> List[str]:
    """Return list of available task names."""
    return list(GRADERS.keys())