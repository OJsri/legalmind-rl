"""
Dense per-step reward for LegalMind-RL.
Range per step: [-0.5, +0.5]

  +0.20  new, relevant evidence presented
  +0.20  contradiction identified in content
  +0.10  strong evidence-backed argument (2+ strong phrases)
  +0.05  decent argument (1 strong phrase)
  +0.05  action correct for current phase
  +0.05  question targeted at a named witness
  -0.05  weak/hedging language (2+ weak phrases)
  -0.05  evidence already entered
  -0.10  repetitive content (≥60% word overlap with recent history)
  -0.15  action invalid for current phase
"""
from __future__ import annotations
import re
from typing import List, Set, Tuple

PHASE_VALID_ACTIONS: dict[str, set[str]] = {
    "opening":           {"argue"},
    "examination":       {"question", "present_evidence"},
    "cross_examination": {"question", "object", "present_evidence"},
    "closing":           {"argue"},
    "verdict":           set(),
}

WEAK_PHRASES = [
    "i think maybe", "perhaps possibly", "i'm not sure", "could be",
    "might be", "just guessing", "i suppose", "maybe perhaps",
]
STRONG_PHRASES = [
    "evidence shows", "as demonstrated by", "this proves", "the record establishes",
    "exhibit", "testimony confirms", "contradicts", "inconsistent with",
    "directly refutes", "the footage shows", "the document states", "according to",
    "establishes beyond", "clearly demonstrates",
]
CONTRADICTION_KEYWORDS = [
    "contradict", "inconsistent", "earlier stated", "previously said",
    "but now claims", "changes story", "refutes", "disproves",
    "cannot be true", "conflicts with", "prior statement",
]


def compute_reward(
    action,
    case,
    phase: str,
    conversation_history: List[dict],
    used_evidence_ids: Set[str],
) -> Tuple[float, dict]:
    breakdown: dict = {}
    reward = 0.0
    content_lower = action.content.lower()

    # Phase validity
    valid = PHASE_VALID_ACTIONS.get(phase, set())
    if action.action in valid:
        reward += 0.05
        breakdown["phase_appropriate"] = 0.05
    else:
        breakdown["invalid_for_phase"] = -0.15
        return -0.15, breakdown

    # New evidence
    if action.action == "present_evidence" and action.target:
        ids = {e.id for e in case.evidence}
        if action.target in ids:
            if action.target not in used_evidence_ids:
                reward += 0.20
                breakdown["new_evidence_used"] = 0.20
            else:
                reward -= 0.05
                breakdown["repeated_evidence"] = -0.05

    # Contradiction
    if any(kw in content_lower for kw in CONTRADICTION_KEYWORDS):
        if action.action in {"question", "argue"}:
            reward += 0.20
            breakdown["contradiction_leveraged"] = 0.20

    # Argument quality
    strong = sum(1 for p in STRONG_PHRASES if p in content_lower)
    weak   = sum(1 for p in WEAK_PHRASES   if p in content_lower)
    if strong >= 2:
        reward += 0.10; breakdown["strong_argument"] = 0.10
    elif strong == 1:
        reward += 0.05; breakdown["decent_argument"] = 0.05
    if weak >= 2:
        reward -= 0.05; breakdown["weak_argument"] = -0.05

    # Repetition penalty
    if conversation_history:
        recent = " ".join(h["content"] for h in conversation_history[-4:]).lower()
        cw = set(re.findall(r'\w{5,}', content_lower))
        rw = set(re.findall(r'\w{5,}', recent))
        if cw and len(cw & rw) / len(cw) > 0.6:
            reward -= 0.10; breakdown["repetition_penalty"] = -0.10

    # Targeted witness
    if action.action == "question" and action.target:
        if action.target.lower() in {w.name.lower() for w in case.witnesses}:
            reward += 0.05; breakdown["targeted_witness"] = 0.05

    return round(max(-0.5, min(0.5, reward)), 4), breakdown
