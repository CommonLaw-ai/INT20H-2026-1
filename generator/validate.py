"""
validate.py — LLM-as-a-judge validator for generated support chat dialogues.

Checks each dialogue against 2 criteria (MIN_LENGTH is checked in code):
  1. Dialogue is complete (not cut off mid-conversation)
  2. case_type matches the scenario (hidden dissatisfaction is subtle, agent_error has a mistake, etc.)

Note: placeholder cleanup is handled separately by anonymize.py — not validated here.
"""

import json
from openai import OpenAI

VALIDATOR_SYSTEM_PROMPT = """You are a quality evaluator for support chat dialogues used in an AI training dataset.

Your job is to evaluate whether a generated dialogue meets BOTH of the following criteria:

1. COMPLETE — The dialogue has a logical ending. It does not cut off mid-sentence or mid-thought. The conversation reaches a natural conclusion.

2. SCENARIO_MATCH — The dialogue content matches the case_type. Use the definitions below.
   Each definition includes what PASSES and what FAILS.

   "successful":
     PASS: Issue is resolved by the end. Customer is satisfied.
     FAIL: Issue is not resolved, or customer is frustrated.

   "agent_error":
     PASS: Agent gives wrong information (wrong prices, wrong timelines, wrong policy), OR ignores a direct question, OR provides contradictory answers.
           The customer may or may not notice the mistake — both are fine.
           The conversation can still end politely — that does NOT disqualify it.
     FAIL: Agent is merely unhelpful or slow but does not make a factual mistake.
     IMPORTANT: An agent saying wrong refund timelines (e.g. "30 days" when it should be less), wrong prices, or incorrect policy IS a clear agent_error. Accept it.

   "problematic":
     PASS: Agent repeatedly asks for the same information, stalls, or makes no progress toward resolution.
     FAIL: Agent eventually resolves the issue.

   "conflictual":
     PASS: Customer shows clear anger, frustration, or hostility. Tension is visible in the dialogue.
     FAIL: Customer is politely dissatisfied but not emotionally escalated.

   "hidden dissatisfaction":
     PASS: Customer formally thanks the agent or says "okay" at the end, but the actual problem was never solved (agent gave wrong info, stalled, or avoided the real issue).
     FAIL: Issue was actually resolved, or customer openly complains at the end.

3. SECURITY — The agent must never request sensitive payment data.
   FAIL (security violation): agent asks for full card number, CVV/CVC, full PIN, or full password.
   PASS: agent asks for last 4 digits of card, transaction date, transaction amount, country, merchant name, or a screenshot — these are all allowed.

Be lenient on realistic dialogue elements: polite titles like "Sir" or "ma'am", real-sounding names, order numbers, and phone numbers are all fine — do NOT flag these.

Respond ONLY with a valid JSON object in this exact format:
{
  "valid": true or false,
  "reason": "short explanation if invalid, or 'ok' if valid"
}

Do not include any text outside of the JSON object.
"""


def build_validator_prompt(scenario: dict, messages: list[dict]) -> str:
    dialogue_text = "\n".join(
        f"{'Customer' if m['role'] == 'customer' else 'Agent'}: {m['text']}"
        for m in messages
    )
    return (
        f"Scenario:\n"
        f"- Topic: {scenario['topic']}\n"
        f"- Case type: {scenario['case_type']}\n"
        f"- Description: {scenario['description']}\n\n"
        f"Dialogue ({len(messages)} messages):\n"
        f"{dialogue_text}\n\n"
        f"Evaluate the dialogue against all criteria and respond with JSON."
    )


def validate_dialogue(client: OpenAI, model: str, scenario: dict, messages: list[dict]) -> tuple[bool, str]:
    """
    Validate a generated dialogue using the LLM-as-a-judge pattern.

    Parameters:
        client   — OpenAI client pointed at Ollama
        model    — model name to use for validation (same as generation)
        scenario — the scenario dict (topic, case_type, description)
        messages — parsed dialogue messages [{"role": ..., "text": ...}]

    Returns:
        (valid: bool, reason: str)
    """
    # Fast pre-check: minimum length before calling LLM
    if len(messages) < 6:
        return False, f"Too short: only {len(messages)} messages (minimum 6 required)"

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": VALIDATOR_SYSTEM_PROMPT},
            {"role": "user",   "content": build_validator_prompt(scenario, messages)},
        ],
        temperature=0.0,   # deterministic — we want consistent judgements
        max_tokens=128,
    )

    raw = response.choices[0].message.content.strip()

    # Strip markdown code fences if model wraps JSON in ```json ... ```
    if raw.startswith("```"):
        lines = raw.splitlines()
        raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    try:
        result = json.loads(raw)
        valid = bool(result.get("valid", False))
        reason = str(result.get("reason", "no reason provided"))
        return valid, reason
    except json.JSONDecodeError:
        # If model didn't return valid JSON, treat as failure
        return False, f"Validator returned invalid JSON: {raw[:100]}"
