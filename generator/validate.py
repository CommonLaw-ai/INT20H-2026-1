"""
validate.py — LLM-as-a-judge validator for generated support chat dialogues.

Checks each dialogue against 4 criteria:
  1. Dialogue is complete (not cut off mid-conversation)
  2. case_type matches the scenario (hidden dissatisfaction is subtle, agent_error has a mistake, etc.)
  3. Minimum 6 messages
  4. No placeholder text ([email], #12345, [name], etc.)
"""

import json
from openai import OpenAI

VALIDATOR_SYSTEM_PROMPT = """You are a strict quality evaluator for support chat dialogues used in an AI training dataset.

Your job is to evaluate whether a generated dialogue meets ALL of the following criteria:

1. COMPLETE — The dialogue has a logical ending. It does not cut off mid-sentence or mid-thought. Both the customer and agent have a closing exchange.
2. SCENARIO_MATCH — The dialogue accurately reflects the given case_type:
   - "successful": issue is fully resolved, customer is satisfied.
   - "agent_error": agent makes at least one clear mistake (wrong info, ignores a question, rude tone).
   - "problematic": agent fails to resolve the issue, keeps asking the same things or stalls.
   - "conflictual": genuine emotional tension and conflict between customer and agent.
   - "hidden dissatisfaction" (described in description): customer formally thanks the agent at the end, but the core issue was never actually solved.
3. MIN_LENGTH — The dialogue has at least 6 messages total.
4. NO_PLACEHOLDERS — The dialogue does not contain generic placeholders like [email], [name], [order_number], #12345, example.com, or similar templated text.

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
        f"Evaluate the dialogue against all 4 criteria and respond with JSON."
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
