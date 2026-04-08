"""
LegalMind-RL server-side environment.
Extends openenv.core.env_server.Environment (the OpenEnv base class).
Dual-import pattern required so the file works both as part of a package
and as a standalone server module.
"""
from __future__ import annotations
import copy
import os
import sys

# Dual-import pattern (OpenEnv requirement)
try:
    from ..models import Case, Evidence, LegalAction, LegalObservation, LegalPhase, LegalState, StepResult
    from ..reward import compute_reward, PHASE_VALID_ACTIONS
    from ..tasks import get_task
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from models import Case, Evidence, LegalAction, LegalObservation, LegalPhase, LegalState, StepResult
    from reward import compute_reward, PHASE_VALID_ACTIONS
    from tasks import get_task

try:
    from openenv.core.env_server import Environment
except ImportError:
    class Environment:  # type: ignore[misc]
        """Fallback base — used when openenv-core is not installed locally."""
        pass

PHASE_ORDER = [
    LegalPhase.OPENING,
    LegalPhase.EXAMINATION,
    LegalPhase.CROSS_EXAMINATION,
    LegalPhase.CLOSING,
    LegalPhase.VERDICT,
]
PHASE_BUDGET = {
    LegalPhase.OPENING: 2,
    LegalPhase.EXAMINATION: 4,
    LegalPhase.CROSS_EXAMINATION: 4,
    LegalPhase.CLOSING: 2,
    LegalPhase.VERDICT: 0,
}


class LegalEnvironment(Environment):
    """
    OpenEnv-compliant courtroom simulation environment.
    One instance = one session (the FastAPI layer manages sessions).
    """

    def __init__(self, case: Case, agent_role: str = "prosecution"):
        super().__init__()
        assert agent_role in {"prosecution", "defense"}
        self.case = case
        self.agent_role = agent_role
        self._init_state()

    # ── OpenEnv interface ────────────────────────────────────────────────

    def reset(self) -> LegalObservation:
        self._init_state()
        return self._obs()

    def step(self, action: LegalAction) -> StepResult:
        if self._done:
            return StepResult(observation=self._obs(), reward=0.0, done=True, error="Episode already finished.")

        err = self._validate(action)
        if err:
            return StepResult(observation=self._obs(), reward=-0.15, done=False, error=err)

        # Record action in history
        self._history.append({"role": self.agent_role, "content": f"[{action.action.upper()}] {action.content}"})

        # Mark evidence used
        if action.action == "present_evidence" and action.target:
            for ev in self._evidence:
                if ev.id == action.target:
                    ev.used = True

        reward, breakdown = compute_reward(
            action=action,
            case=self.case,
            phase=self._phase.value,
            conversation_history=self._history[:-1],
            used_evidence_ids=self._used_ev,
        )

        if action.action == "present_evidence" and action.target:
            self._used_ev.add(action.target)

        if any(kw in action.content.lower() for kw in ["contradict","inconsistent","earlier stated","previously said"]):
            self._contradictions += 1

        self._cumr += reward
        self._round += 1
        self._phase_round += 1
        self._advance_phase_if_needed()

        obs = self._obs()
        obs.reward = reward
        obs.done = self._done

        return StepResult(
            observation=obs, reward=reward, done=self._done,
            info={"reward_breakdown": breakdown, "phase": self._phase.value,
                  "round": self._round, "cumulative_reward": round(self._cumr, 4),
                  "contradictions_found": self._contradictions, "evidence_used": list(self._used_ev)},
        )

    @property
    def state(self) -> LegalState:
        return LegalState(
            case_id=self.case.id, phase=self._phase.value,
            round_number=self._round, max_rounds=self.case.max_rounds,
            done=self._done, cumulative_reward=round(self._cumr, 4),
            used_evidence=list(self._used_ev),
            contradictions_found=self._contradictions,
            history_length=len(self._history),
        )

    # ── Internals ────────────────────────────────────────────────────────

    def _init_state(self):
        self._phase        = LegalPhase.OPENING
        self._round        = 0
        self._phase_round  = 0
        self._done         = False
        self._history: list = []
        self._evidence     = [copy.deepcopy(e) for e in self.case.evidence]
        self._used_ev: set = set()
        self._contradictions = 0
        self._cumr         = 0.0

    def _obs(self) -> LegalObservation:
        valid = list(PHASE_VALID_ACTIONS.get(self._phase.value, set()))
        opp   = next((h["content"] for h in reversed(self._history) if h["role"] != self.agent_role), None)
        return LegalObservation(
            case_summary=self.case.description,
            current_phase=self._phase.value,
            round_number=self._round,
            max_rounds=self.case.max_rounds,
            available_evidence=[{"id":e.id,"description":e.description,"type":e.type,"used":e.used} for e in self._evidence],
            witnesses=[{"name":w.name,"testimony":w.testimony} for w in self.case.witnesses],
            conversation_history=list(self._history),
            opponent_last_action=opp,
            valid_actions=valid,
            score_so_far=round(self._cumr, 4),
            agent_role=self.agent_role,
            done=self._done,
            reward=0.0,
        )

    def _validate(self, action: LegalAction):
        valid = PHASE_VALID_ACTIONS.get(self._phase.value, set())
        if action.action not in valid:
            return f"Action '{action.action}' invalid in phase '{self._phase.value}'. Valid: {valid}"
        return None

    def _advance_phase_if_needed(self):
        budget = PHASE_BUDGET.get(self._phase, 0)
        if self._phase_round >= budget or self._round >= self.case.max_rounds:
            idx = PHASE_ORDER.index(self._phase)
            if idx + 1 >= len(PHASE_ORDER):
                self._done = True
            else:
                self._phase = PHASE_ORDER[idx + 1]
                self._phase_round = 0
                if self._phase == LegalPhase.VERDICT:
                    self._done = True
