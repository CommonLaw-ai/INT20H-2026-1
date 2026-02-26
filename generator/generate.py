"""
generate.py — Generates a dataset of support chat dialogues using Llama 3.1 8B via Ollama.

Usage:
    python generate.py --output dataset.json
    python generate.py --output dataset.json --count 10

Requirements:
    Ollama must be running locally: https://ollama.com
    Model must be pulled: ollama pull llama3.1:8b
"""

import argparse
import json
import sys
from openai import OpenAI

from validate import validate_dialogue
from actions import classify_action, ACTIONS_LIST
from anonymize import anonymize_dataset

# ── Configuration ────────────────────────────────────────────────────────────

MODEL = "llama3.1:8b"
SEED = 42        # For deterministic results
MAX_RETRIES = 3  # Max attempts per dialogue if validation fails

# All scenario combinations to cover in the dataset
SCENARIOS = [
    {"topic": "payment_issue",      "case_type": "successful",  "description": "Customer couldn't complete a payment; agent resolves it quickly and correctly."},
    {"topic": "payment_issue",      "case_type": "agent_error", "description": "Customer has a double charge; agent gives incorrect information about refund timelines and dismisses follow-up questions."},
    {"topic": "technical_error",    "case_type": "successful",  "description": "Customer reports app crashing; agent walks through troubleshooting and fixes the issue."},
    {"topic": "technical_error",    "case_type": "problematic", "description": "Customer reports login errors repeatedly; agent keeps asking for the same info without resolving anything."},
    {"topic": "account_access",     "case_type": "successful",  "description": "Customer locked out of account; agent verifies identity and restores access promptly."},
    {"topic": "account_access",     "case_type": "conflictual", "description": "Customer frustrated about being locked out for 3 days; agent is dismissive and unhelpful, customer escalates angrily."},
    {"topic": "tariff_question",    "case_type": "successful",  "description": "Customer asks about plan differences; agent explains clearly and helps choose the best option."},
    {"topic": "tariff_question",    "case_type": "agent_error", "description": "Customer asks about pricing; agent gives wrong prices, customer notices the discrepancy. Customer formally thanks agent but the issue remains unresolved — hidden dissatisfaction."},
    {"topic": "refund",             "case_type": "problematic", "description": "Customer requests a refund for a service they never used; agent stalls with vague answers and no resolution. Customer says 'okay, thanks' but is clearly not satisfied — hidden dissatisfaction."},
    {"topic": "refund",             "case_type": "conflictual", "description": "Customer demands refund angrily after being charged twice; agent uses rude tone and refuses without proper explanation."},
]

SYSTEM_PROMPT = f"""You are a realistic dialogue generator for a support chat dataset.
Generate a natural, realistic conversation between a Customer and a Support Agent.
The dialogue must match the given scenario exactly.

Rules:
- Write in English only.
- Each message must start with either "Customer:" or "Agent:".
- The conversation should have 6–12 exchanges total.
- Make it realistic — include natural phrasing, hesitation, follow-up questions.
- For "hidden dissatisfaction" cases: the customer formally thanks the agent at the end,
  but the core problem was never actually solved (the agent gave wrong info, stalled, or ignored the real issue).
- For "agent_error" cases: include at least one clear mistake by the agent
  (wrong information, ignoring a question, unnecessary escalation, or rude tone).
- For "conflictual" cases: show genuine emotional tension, frustration, and conflict.
- Do NOT include stage directions, pauses, or actions in parentheses such as (pause), (sighs), (checks account), (long pause), (typing), etc.
- SECURITY: The agent must NEVER ask for full card number, CVV/CVC, full PIN, or full password.
  The agent MAY ask for: last 4 digits of card, transaction date, transaction amount, country, merchant name, or a screenshot of an error.
- The agent must perform exactly one of the following actions during the conversation, naturally woven into the dialogue:
{ACTIONS_LIST}
- Output ONLY the dialogue lines, no introductions or commentary.
"""


def build_user_prompt(scenario: dict) -> str:
    return (
        f"Generate a support chat dialogue with the following scenario:\n"
        f"- Topic: {scenario['topic']}\n"
        f"- Case type: {scenario['case_type']}\n"
        f"- Description: {scenario['description']}\n\n"
        f"Write the dialogue now:"
    )


def parse_dialogue(raw_text: str) -> list[dict]:
    """Parse raw LLM output into a list of message dicts."""
    messages = []
    for line in raw_text.strip().splitlines():
        line = line.strip()
        if line.startswith("Customer:"):
            messages.append({"role": "customer", "text": line[len("Customer:"):].strip()})
        elif line.startswith("Agent:"):
            messages.append({"role": "agent", "text": line[len("Agent:"):].strip()})
    return messages


def check_ollama(client: OpenAI) -> None:
    """Check if Ollama is running and the model is available."""
    try:
        models = client.models.list()
        available = [m.id for m in models.data]
        if MODEL not in available:
            print(f"ERROR: Model '{MODEL}' not found in Ollama.")
            print(f"  Run: ollama pull {MODEL}")
            print(f"  Available models: {', '.join(available) or 'none'}")
            sys.exit(1)
    except Exception:
        print("ERROR: Cannot connect to Ollama.")
        print("  Make sure Ollama is running: https://ollama.com")
        print("  Start it with: ollama serve")
        sys.exit(1)


def _call_llm(client: OpenAI, scenario: dict, attempt: int) -> list[dict]:
    """Single LLM call for dialogue generation. Uses a different seed per attempt."""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": build_user_prompt(scenario)},
        ],
        temperature=0.7,
        seed=SEED + attempt,  # different seed per retry for variation
        max_tokens=2048,
    )
    raw_text = response.choices[0].message.content
    return parse_dialogue(raw_text)


def generate_dialogue(client: OpenAI, scenario: dict, index: int) -> dict:
    """Call Ollama API and return a validated structured dialogue entry."""
    label = f"[{scenario['case_type']}] {scenario['topic']}"
    print(f"  Generating dialogue {index + 1}/{len(SCENARIOS)}: {label}")

    last_messages = None

    for attempt in range(MAX_RETRIES):
        print(f"    Attempt {attempt + 1}/{MAX_RETRIES} ...", end=" ", flush=True)

        messages = _call_llm(client, scenario, attempt)
        last_messages = messages
        print(f"generated ({len(messages)} messages), validating ...", end=" ", flush=True)

        valid, reason = validate_dialogue(client, MODEL, scenario, messages)

        if valid:
            print("OK", end=" ", flush=True)
            action = classify_action(client, MODEL, messages)
            print(f"→ action: {action}")
            return {
                "id": index + 1,
                "topic": scenario["topic"],
                "case_type": scenario["case_type"],
                "agent_action": action,
                "messages": messages,
            }
        else:
            print(f"FAILED — {reason}")

    print(f"    WARNING: using best-effort result after {MAX_RETRIES} failed attempts.", end=" ", flush=True)
    action = classify_action(client, MODEL, last_messages)
    print(f"→ action: {action}")
    return {
        "id": index + 1,
        "topic": scenario["topic"],
        "case_type": scenario["case_type"],
        "agent_action": action,
        "messages": last_messages,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate support chat dataset using Llama 3.1 8B via Ollama.")
    parser.add_argument("--output", default="dataset.json", help="Output JSON file path (default: dataset.json)")
    parser.add_argument("--count",  type=int, default=len(SCENARIOS),
                        help=f"Number of dialogues to generate (default: {len(SCENARIOS)}, max: {len(SCENARIOS)})")
    args = parser.parse_args()

    count = min(args.count, len(SCENARIOS))
    scenarios_to_run = SCENARIOS[:count]

    client = OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama",  # required by the client but not used by Ollama
    )

    check_ollama(client)

    print(f"Generating {count} dialogue(s) using {MODEL} via Ollama...")
    print(f"Output file: {args.output}\n")

    dataset = []
    for i, scenario in enumerate(scenarios_to_run):
        try:
            entry = generate_dialogue(client, scenario, i)
            dataset.append(entry)
        except Exception as e:
            print(f"ERROR generating dialogue {i + 1}: {e}")
            print(f"  Skipping dialogue {i + 1} and continuing...")

    print("\nCleaning dataset (stage directions, placeholders) ...")
    dataset, replacements = anonymize_dataset(dataset)
    print(f"  {replacements} placeholder(s) replaced.")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"\nDataset saved: {args.output} ({len(dataset)} dialogues)")


if __name__ == "__main__":
    main()