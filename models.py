"""
LegalMind-RL — Typed models.
Action / Observation / State follow the official OpenEnv dataclass pattern.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


# ── Phase enum ────────────────────────────────────────────────────────────

class LegalPhase(str, Enum):
    OPENING           = "opening"
    EXAMINATION       = "examination"
    CROSS_EXAMINATION = "cross_examination"
    CLOSING           = "closing"
    VERDICT           = "verdict"


class ActionType(str, Enum):
    ARGUE            = "argue"
    QUESTION         = "question"
    PRESENT_EVIDENCE = "present_evidence"
    OBJECT           = "object"


# ── Case data (server-side) ───────────────────────────────────────────────

@dataclass
class Evidence:
    id: str
    description: str
    type: str                        # physical | testimony | document | circumstantial
    relevance_tags: List[str] = field(default_factory=list)
    used: bool = False


@dataclass
class Witness:
    name: str
    testimony: str
    credibility: float = 0.8
    has_contradiction: bool = False
    contradiction_hint: Optional[str] = None


@dataclass
class Case:
    id: str
    title: str
    description: str
    prosecution_goal: str
    defense_goal: str
    evidence: List[Evidence]
    witnesses: List[Witness]
    verdict_criteria: Dict[str, Any]
    difficulty: str                  # easy | medium | hard
    max_rounds: int = 10


# ── OpenEnv contract types ─────────────────────────────────────────────────

@dataclass
class LegalAction:
    """OpenEnv Action — what the agent submits each step."""
    action: str          # argue | question | present_evidence | object
    content: str         # the legal argument text
    target: Optional[str] = None    # evidence ID or witness name


@dataclass
class LegalObservation:
    """OpenEnv Observation — what the agent sees each step."""
    case_summary: str
    current_phase: str
    round_number: int
    max_rounds: int
    available_evidence: List[Dict[str, Any]]
    witnesses: List[Dict[str, Any]]
    conversation_history: List[Dict[str, str]]
    opponent_last_action: Optional[str]
    valid_actions: List[str]
    score_so_far: float
    agent_role: str
    done: bool = False
    reward: float = 0.0

    def to_prompt(self) -> str:
        ev   = "\n".join(f"  [{e['id']}] {e['description']} (used:{e.get('used',False)})" for e in self.available_evidence)
        wt   = "\n".join(f"  {w['name']}: \"{w['testimony']}\"" for w in self.witnesses)
        hist = "\n".join(f"  [{h['role'].upper()}]: {h['content']}" for h in self.conversation_history[-6:]) or "  (none)"
        return (
            f"=== COURTROOM STATE ===\n"
            f"Role: {self.agent_role.upper()}\n"
            f"Case: {self.case_summary}\n"
            f"Phase: {self.current_phase.replace('_',' ').title()}\n"
            f"Round: {self.round_number}/{self.max_rounds}\n"
            f"Score so far: {self.score_so_far:.2f}\n\n"
            f"EVIDENCE:\n{ev}\n\nWITNESSES:\n{wt}\n\n"
            f"HISTORY:\n{hist}\n\n"
            f"OPPONENT: {self.opponent_last_action or '(none)'}\n"
            f"VALID ACTIONS: {', '.join(self.valid_actions)}\n\n"
            f'Respond ONLY with JSON: {{"action":"...","target":"...","content":"..."}}'
        )

    def to_dict(self) -> dict:
        return {
            "case_summary": self.case_summary,
            "current_phase": self.current_phase,
            "round_number": self.round_number,
            "max_rounds": self.max_rounds,
            "available_evidence": self.available_evidence,
            "witnesses": self.witnesses,
            "conversation_history": self.conversation_history,
            "opponent_last_action": self.opponent_last_action,
            "valid_actions": self.valid_actions,
            "score_so_far": self.score_so_far,
            "agent_role": self.agent_role,
            "done": self.done,
            "reward": self.reward,
        }


@dataclass
class LegalState:
    """OpenEnv State — serialisable episode snapshot."""
    case_id: str
    phase: str
    round_number: int
    max_rounds: int
    done: bool
    cumulative_reward: float
    used_evidence: List[str]
    contradictions_found: int
    history_length: int

    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "phase": self.phase,
            "round_number": self.round_number,
            "max_rounds": self.max_rounds,
            "done": self.done,
            "cumulative_reward": self.cumulative_reward,
            "used_evidence": self.used_evidence,
            "contradictions_found": self.contradictions_found,
            "history_length": self.history_length,
        }


@dataclass
class StepResult:
    observation: LegalObservation
    reward: float
    done: bool
    info: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
