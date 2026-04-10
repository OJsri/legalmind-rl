"""
Inference Script — LegalMind-RL
================================
MANDATORY environment variables:
    API_BASE_URL      The API endpoint for the LLM.
    MODEL_NAME        The model identifier to use for inference.
    HF_TOKEN          Your Hugging Face / API key.
    LOCAL_IMAGE_NAME  Local Docker image name (used by from_docker_image()).
                      Optional — if not set, connects to ENV_BASE_URL directly.

Defaults (reflect active inference setup):
    API_BASE_URL  = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
    MODEL_NAME    = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

STDOUT FORMAT (strict):
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

Rules:
    - One [START] line at episode begin.
    - One [STEP] line per step, immediately after env.step() returns.
    - One [END] line after env.close(), always emitted (even on exception).
    - reward and rewards formatted to 2 decimal places.
    - done and success are lowercase booleans: true or false.
    - error is the raw error string, or null if none.
    - All fields on a single line, no newlines within a line.
    - Score in [0, 1].
"""

import asyncio
import json
import os
import sys
import textwrap
from typing import List, Optional

from openai import OpenAI

# ── Required env vars (hackathon checklist item) ─────────────────────────
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")        # optional Docker image
HF_TOKEN = os.getenv("HF_TOKEN")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
TASK_NAME    = os.getenv("TASK",         "easy")          # easy | medium | hard
ROLE         = os.getenv("ROLE",         "prosecution")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")
BENCHMARK    = "legalmind-rl"
MAX_STEPS    = 20
SUCCESS_SCORE_THRESHOLD = 0.5

# ── LLM client (all LLM calls use OpenAI client — hackathon requirement) ─
client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert courtroom litigator participating in a legal simulation.
    Each turn you receive the courtroom state and must respond with ONE legal action in JSON.

    Rules:
    1. Only use actions listed in VALID ACTIONS.
    2. For present_evidence: set "target" to the evidence ID (e.g. "E001").
    3. For question: set "target" to the exact witness name.
    4. Cite evidence IDs and witness names explicitly in your content.
    5. Actively identify contradictions — they are highly rewarded.
    6. Never repeat a prior argument verbatim.
    7. Be specific, detailed, and legally precise.

    Respond ONLY with valid JSON — no preamble, no markdown, no explanation:
    {"action": "<argue|question|present_evidence|object>", "target": "<id_or_name_or_null>", "content": "<your argument>"}
""").strip()


# ── Logging helpers (strict stdout format) ────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error or 'null'}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)


# ── LLM call ─────────────────────────────────────────────────────────────

def get_model_action(obs_prompt: str) -> dict:
    """Call LLM via OpenAI client, return parsed action dict."""
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": obs_prompt},
            ],
            temperature=0.7,
            max_tokens=512,
        )
        raw = (completion.choices[0].message.content or "").strip()
        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw.strip())
    except Exception as exc:
        print(f"[DEBUG] LLM call failed: {exc}", flush=True, file=sys.stderr)
        return {"action": "argue", "target": None, "content": "The evidence directly establishes the facts of this case."}


# ── Build prompt from observation dict ───────────────────────────────────

def build_prompt(obs: dict, step: int) -> str:
    ev   = "\n".join(f"  [{e['id']}] {e['description']} (used:{e.get('used',False)})" for e in obs.get("available_evidence",[]))
    wt   = "\n".join(f"  {w['name']}: \"{w['testimony']}\"" for w in obs.get("witnesses",[]))
    hist = "\n".join(f"  [{h['role'].upper()}]: {h['content']}" for h in obs.get("conversation_history",[])[-6:]) or "  (none)"
    return textwrap.dedent(f"""
        Step: {step}
        Role: {obs.get('agent_role','prosecution').upper()}
        Case: {obs.get('case_summary','')}
        Phase: {obs.get('current_phase','?').replace('_',' ').title()}
        Round: {obs.get('round_number',0)}/{obs.get('max_rounds',0)}
        Score so far: {obs.get('score_so_far',0.0):.2f}

        AVAILABLE EVIDENCE:
        {ev}

        WITNESSES:
        {wt}

        RECENT HISTORY:
        {hist}

        OPPONENT LAST ACTION:
          {obs.get('opponent_last_action') or '(none)'}

        VALID ACTIONS: {', '.join(obs.get('valid_actions',[]))}
    """).strip()


# ── Main async loop ───────────────────────────────────────────────────────

async def main() -> None:
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    # Import here so the file is importable even without the env installed
    from client import LegalMindEnv
    from models import LegalAction

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    # Choose connection method: Docker image or direct URL
    if LOCAL_IMAGE_NAME:
        env = await LegalMindEnv.from_docker_image(LOCAL_IMAGE_NAME)
        ctx = env
    else:
        env = LegalMindEnv(base_url=ENV_BASE_URL)
        ctx = env

    try:
        async with ctx:
            obs = await env.reset(task=TASK_NAME, role=ROLE)
            done = obs.done

            for step in range(1, MAX_STEPS + 1):
                if done:
                    break

                prompt = build_prompt(obs.to_dict(), step)
                act    = get_model_action(prompt)

                action = LegalAction(
                    action=act.get("action", "argue"),
                    target=act.get("target"),
                    content=act.get("content", "The evidence supports this position."),
                )

                result = await env.step(action)
                reward = result.reward
                done   = result.done
                error  = result.error
                obs    = result.observation

                rewards.append(reward)
                steps_taken = step

                action_str = f"{action.action}({action.target or ''})"
                log_step(step=step, action=action_str, reward=reward, done=done, error=error)

                if done:
                    break

            score   = await env.score()
            success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Exception: {exc}", flush=True, file=sys.stderr)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    asyncio.run(main())
