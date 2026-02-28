"""
actions.py — Defines the agent action space and classifies which action
was performed at the end of a dialogue using LLM-as-a-judge.
"""

import json
from openai import OpenAI

# ── Action space ──────────────────────────────────────────────────────────────

ACTIONS = {
    "resolved":             "The issue was fully resolved and the customer is satisfied.",
    "refund_initiated":     "The agent initiated a refund or confirmed it will be processed.",
    "account_unlocked":     "The agent unlocked or restored access to the customer's account.",
    "info_provided":        "The agent provided information (pricing, plans, policies) without a concrete resolution.",
    "follow_up_scheduled":  "The agent and customer agreed to follow up later or the agent promised a callback.",
    "escalated_to_human":   "The conversation was escalated to a supervisor or human specialist.",
    "unresolved":           "The conversation ended without resolving the issue (agent stalled, gave wrong info, or ignored the problem).",
    "customer_left":        "The customer ended the conversation in frustration or dissatisfaction without resolution.",
}

# Human-readable list for prompts
ACTIONS_LIST = "\n".join(f'- "{k}": {v}' for k, v in ACTIONS.items())

# ── Classifier prompt ─────────────────────────────────────────────────────────

CLASSIFIER_SYSTEM_PROMPT = f"""You are a support chat analyst. Your job is to read a completed support dialogue and identify which single action the agent ultimately performed.

Available actions:
{ACTIONS_LIST}

Rules:
- Choose exactly ONE action that best describes the final outcome of the conversation.
- Base your decision on what actually happened in the dialogue, not on what should have happened.
- If the customer formally thanks the agent but the issue was never truly solved, choose "unresolved" or "customer_left", not "resolved".
- Respond ONLY with a valid JSON object in this exact format:
{{"action": "<action_key>", "reason": "<one sentence explanation>"}}

Do not include any text outside of the JSON object.
"""


def build_classifier_prompt(messages: list[dict]) -> str:
    dialogue_text = "\n".join(
        f"{'Customer' if m['role'] == 'customer' else 'Agent'}: {m['text']}"
        for m in messages
    )
    return f"Classify the agent action for this dialogue:\n\n{dialogue_text}"


def classify_action(client: OpenAI, model: str, messages: list[dict]) -> str:
    """
    Read a completed dialogue and return the agent_action key.

    Parameters:
        client   — OpenAI client pointed at Ollama
        model    — model name to use
        messages — parsed dialogue messages [{"role": ..., "text": ...}]

    Returns:
        One of the keys from ACTIONS (falls back to "unresolved" on parse error).
    """
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": CLASSIFIER_SYSTEM_PROMPT},
            {"role": "user",   "content": build_classifier_prompt(messages)},
        ],
        temperature=0.0,
        max_tokens=128,
    )

    raw = response.choices[0].message.content.strip()

    # Strip markdown code fences if model wraps JSON in ```json ... ```
    if raw.startswith("```"):
        lines = raw.splitlines()
        raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    try:
        result = json.loads(raw)
        action = result.get("action", "unresolved")
        if action not in ACTIONS:
            return "unresolved"
        return action
    except json.JSONDecodeError:
        return "unresolved"
