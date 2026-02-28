"""
generate.py — Generates a dataset of support chat dialogues using Llama 3.1 8B via Ollama.

Usage:
    python generate.py --output dataset.json
    python generate.py --output dataset.json --count 20
    python generate.py --output dataset.json --policy company_policy.txt
    python generate.py --output dataset.json --topics payment_issue,refund --case-types successful,conflictual
    python generate.py --output dataset.json --count 30 --vary-descriptions

Requirements:
    Ollama must be running locally: https://ollama.com
    Model must be pulled: ollama pull llama3.1:8b
"""

import argparse
import json
import random
import sys
from itertools import product
from pathlib import Path
from openai import OpenAI

from validate import validate_dialogue
from actions import classify_action, ACTIONS_LIST
from anonymize import anonymize_dataset

# ── Configuration ────────────────────────────────────────────────────────────

MODEL = "llama3.1:8b"
SEED = 42
MAX_RETRIES = 3

# Support escalation contacts
SUPPORT_EMAIL = "support@company.com"
SUPPORT_PHONE = "+1-800-123-4567"
SUPPORT_HOURS = "Monday–Friday, 9:00–18:00"

# ── Available topics and case types ──────────────────────────────────────────

TOPICS = [
    "payment_issue",
    "technical_error",
    "account_access",
    "tariff_question",
    "refund",
    "subscription_cancellation",
    "delivery_problem",
    "wrong_charge",
    "feature_request",
    "password_reset",
]

CASE_TYPES = [
    "successful",
    "agent_error",
    "problematic",
    "conflictual",
    "hidden_dissatisfaction",
]

# Hint injected into description-generation prompt for each case type
CASE_TYPE_HINTS = {
    "successful":             "The issue is resolved correctly and the customer is satisfied.",
    "agent_error":            "The agent makes at least one clear mistake (wrong info, ignores a question, wrong procedure).",
    "problematic":            "The agent fails to resolve the issue — asks for the same info repeatedly or stalls.",
    "conflictual":            "The customer becomes genuinely frustrated and emotional; tension escalates.",
    "hidden_dissatisfaction": "The customer formally thanks the agent at the end, but the core issue was never actually solved.",
}

# ── Scenario building ─────────────────────────────────────────────────────────

def generate_description(client: OpenAI, topic: str, case_type: str, seed: int) -> str:
    """Ask the LLM to write a one-sentence scenario description for the given topic + case_type."""
    hint = CASE_TYPE_HINTS[case_type]
    prompt = (
        f"Write a single sentence describing a realistic customer support chat scenario.\n"
        f"Topic: {topic.replace('_', ' ')}\n"
        f"Case type: {case_type.replace('_', ' ')} — {hint}\n"
        f"Rules:\n"
        f"- One sentence only, no bullet points, no quotes.\n"
        f"- Be specific: mention what the customer's concrete problem is and what happens.\n"
        f"- Do NOT repeat the case type label in the sentence.\n"
        f"Description:"
    )
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.9,
        seed=seed,
        max_tokens=120,
    )
    return response.choices[0].message.content.strip().splitlines()[0].strip()


def build_scenarios(
    client: OpenAI,
    topics: list[str],
    case_types: list[str],
    count: int,
    vary_descriptions: bool,
    rng: random.Random,
) -> list[dict]:
    """
    Build a list of scenario dicts with LLM-generated descriptions.

    - count <= combinations  →  sample without replacement, one description per pair
    - count > combinations   →  repeat pairs; generate a fresh description per repeat
                                if vary_descriptions=True, otherwise reuse the cached one
    """
    all_pairs = list(product(topics, case_types))

    if count <= len(all_pairs):
        chosen_pairs = rng.sample(all_pairs, count)
    else:
        full_rounds = count // len(all_pairs)
        remainder   = count % len(all_pairs)
        chosen_pairs = all_pairs * full_rounds + rng.sample(all_pairs, remainder)
        rng.shuffle(chosen_pairs)

    print(f"Building {len(chosen_pairs)} scenario(s) "
          f"({len(topics)} topic(s) × {len(case_types)} case type(s))...")

    scenarios: list[dict] = []
    description_cache: dict[tuple, str] = {}

    for idx, (topic, case_type) in enumerate(chosen_pairs):
        cache_key = (topic, case_type)
        use_cache = cache_key in description_cache and not vary_descriptions

        if use_cache:
            description = description_cache[cache_key]
        else:
            seed = SEED + idx
            print(
                f"  [{idx + 1}/{len(chosen_pairs)}] "
                f"Generating description: {topic} / {case_type} ...",
                end=" ", flush=True,
            )
            description = generate_description(client, topic, case_type, seed)
            description_cache[cache_key] = description
            short = description[:80] + ("..." if len(description) > 80 else "")
            print(f'"{short}"')

        scenarios.append({
            "topic":       topic,
            "case_type":   case_type,
            "description": description,
        })

    return scenarios


# ── System prompt ─────────────────────────────────────────────────────────────

def build_system_prompt(policy_text: str | None = None) -> str:
    policy_section = ""
    if policy_text:
        policy_section = f"""
== COMPANY POLICY DOCUMENT ==
The agent must follow this policy when answering questions about plans, refunds, procedures, etc.
Use only the information from this document — do not invent rules or prices not listed here.

{policy_text.strip()}
== END OF POLICY DOCUMENT ==
"""

    return f"""You are a realistic dialogue generator for a support chat dataset.
Generate a natural, realistic conversation between a Customer and a Support Agent.
The dialogue must match the given scenario exactly.
{policy_section}
Rules:
- Write in English only.
- Each message must start with either "Customer:" or "Agent:".
- The conversation should have 6–12 exchanges total.
- Make it realistic — include natural phrasing, hesitation, follow-up questions.
- For "hidden_dissatisfaction" cases: the customer formally thanks the agent at the end,
  but the core problem was never actually solved (agent gave wrong info, stalled, or ignored real issue).
- For "agent_error" cases: include at least one clear mistake by the agent
  (wrong information, ignoring a question, unnecessary escalation).
- For "conflictual" cases: show genuine emotional tension and frustration from the CUSTOMER side.
  The agent must remain polite at all times even under pressure.

TONE & POLITENESS (always applies, no exceptions):
- The agent must ALWAYS be respectful, calm, and professional — even in conflictual or problematic scenarios.
- The agent must NEVER use angry, rude, sarcastic, condescending, or dismissive language.
- The agent may be unhelpful due to incompetence or wrong info, but never due to bad attitude.
- The agent must address the customer warmly (e.g., "I understand your frustration", "I'm sorry to hear that").

SECURITY (always applies, no exceptions):
- The agent must NEVER ask for: full card number, CVV/CVC, full PIN, full password,
  Social Security Number, or any other sensitive credential in full.
- The agent MAY ask for: last 4 digits of card, transaction date, transaction amount,
  country, merchant name, email address used for registration, or a screenshot of an error.
- If a customer volunteers sensitive data (e.g. full card number), the agent must immediately
  ask them NOT to share it and explain it is not needed.

ESCALATION — when agent cannot resolve the issue:
- If the agent is unable to resolve a problem after reasonable attempts, they MUST offer to escalate.
- The escalation message must include:
    Email: {SUPPORT_EMAIL}
    Phone: {SUPPORT_PHONE} (available {SUPPORT_HOURS})
- Encourage the customer to use these contacts for issues requiring human specialists.

ACTIONS:
- The agent must perform exactly one of the following actions during the conversation,
  naturally woven into the dialogue:
{ACTIONS_LIST}

- Do NOT include stage directions, pauses, or actions in parentheses such as (pause), (sighs),
  (checks account), (long pause), (typing), etc.
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


# ── Parsing ───────────────────────────────────────────────────────────────────

def parse_dialogue(raw_text: str) -> list[dict]:
    messages = []
    for line in raw_text.strip().splitlines():
        line = line.strip()
        if line.startswith("Customer:"):
            messages.append({"role": "customer", "text": line[len("Customer:"):].strip()})
        elif line.startswith("Agent:"):
            messages.append({"role": "agent", "text": line[len("Agent:"):].strip()})
    return messages


# ── Ollama helpers ────────────────────────────────────────────────────────────

def check_ollama(client: OpenAI) -> None:
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


def _call_llm(client: OpenAI, scenario: dict, attempt: int, system_prompt: str) -> list[dict]:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": build_user_prompt(scenario)},
        ],
        temperature=0.7,
        seed=SEED + attempt,
        max_tokens=2048,
    )
    return parse_dialogue(response.choices[0].message.content)


def generate_dialogue(
    client: OpenAI,
    scenario: dict,
    index: int,
    total: int,
    system_prompt: str,
) -> dict:
    label = f"[{scenario['case_type']}] {scenario['topic']}"
    print(f"  Generating dialogue {index + 1}/{total}: {label}")

    last_messages = None

    for attempt in range(MAX_RETRIES):
        print(f"    Attempt {attempt + 1}/{MAX_RETRIES} ...", end=" ", flush=True)
        messages = _call_llm(client, scenario, attempt, system_prompt)
        last_messages = messages
        print(f"generated ({len(messages)} messages), validating ...", end=" ", flush=True)

        valid, reason = validate_dialogue(client, MODEL, scenario, messages)
        if valid:
            print("OK", end=" ", flush=True)
            action = classify_action(client, MODEL, messages)
            print(f"→ action: {action}")
            return {
                "id":           index + 1,
                "topic":        scenario["topic"],
                "case_type":    scenario["case_type"],
                "description":  scenario["description"],
                "agent_action": action,
                "messages":     messages,
            }
        else:
            print(f"FAILED — {reason}")

    print(f"    WARNING: using best-effort result after {MAX_RETRIES} failed attempts.", end=" ", flush=True)
    action = classify_action(client, MODEL, last_messages)
    print(f"→ action: {action}")
    return {
        "id":           index + 1,
        "topic":        scenario["topic"],
        "case_type":    scenario["case_type"],
        "description":  scenario["description"],
        "agent_action": action,
        "messages":     last_messages,
    }


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    default_count = len(TOPICS) * len(CASE_TYPES)

    parser = argparse.ArgumentParser(
        description="Generate support chat dataset using Llama 3.1 8B via Ollama.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available topics:
  {', '.join(TOPICS)}

Available case types:
  {', '.join(CASE_TYPES)}

Examples:
  # All combinations (default: {default_count} dialogues)
  python generate.py --output dataset.json

  # 30 dialogues — pairs repeat with fresh LLM descriptions each time
  python generate.py --output dataset.json --count 30 --vary-descriptions

  # Only specific topics and case types
  python generate.py --topics payment_issue,refund --case-types successful,conflictual

  # With a company policy document
  python generate.py --policy company_policy.txt
""",
    )
    parser.add_argument("--output",  default="dataset.json",
                        help="Output JSON file (default: dataset.json)")
    parser.add_argument("--count",   type=int, default=None,
                        help="Number of dialogues to generate (default: all topic×case_type combinations)")
    parser.add_argument("--topics",  default=None,
                        help=f"Comma-separated topics (default: all {len(TOPICS)})")
    parser.add_argument("--case-types", dest="case_types", default=None,
                        help=f"Comma-separated case types (default: all {len(CASE_TYPES)})")
    parser.add_argument("--policy",  default=None, metavar="FILE",
                        help="Plain-text company policy document to inject into the system prompt.")
    parser.add_argument("--vary-descriptions", dest="vary_descriptions", action="store_true",
                        help="Generate a fresh LLM description for each repeated topic/case_type pair.")
    parser.add_argument("--seed",    type=int, default=SEED,
                        help=f"Random seed (default: {SEED})")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    # Resolve and validate topics / case types
    topics     = [t.strip() for t in args.topics.split(",")]     if args.topics     else TOPICS
    case_types = [c.strip() for c in args.case_types.split(",")] if args.case_types else CASE_TYPES

    bad_topics = [t for t in topics     if t not in TOPICS]
    bad_cases  = [c for c in case_types if c not in CASE_TYPES]
    if bad_topics:
        print(f"ERROR: Unknown topic(s): {', '.join(bad_topics)}")
        print(f"  Valid: {', '.join(TOPICS)}")
        sys.exit(1)
    if bad_cases:
        print(f"ERROR: Unknown case type(s): {', '.join(bad_cases)}")
        print(f"  Valid: {', '.join(CASE_TYPES)}")
        sys.exit(1)

    count = args.count if args.count is not None else len(topics) * len(case_types)

    # Load optional policy document
    policy_text = None
    if args.policy:
        policy_path = Path(args.policy)
        if not policy_path.exists():
            print(f"ERROR: Policy file not found: {args.policy}")
            sys.exit(1)
        policy_text = policy_path.read_text(encoding="utf-8")
        print(f"Policy document loaded: {args.policy} ({len(policy_text)} chars)\n")

    system_prompt = build_system_prompt(policy_text)

    client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    check_ollama(client)

    # Phase 1 — build scenario list with LLM-generated descriptions
    print()
    scenarios = build_scenarios(client, topics, case_types, count, args.vary_descriptions, rng)

    # Phase 2 — generate dialogues
    print(f"\nGenerating {len(scenarios)} dialogue(s) using {MODEL} via Ollama...")
    print(f"Output file: {args.output}\n")

    dataset = []
    for i, scenario in enumerate(scenarios):
        try:
            entry = generate_dialogue(client, scenario, i, len(scenarios), system_prompt)
            dataset.append(entry)
        except Exception as e:
            print(f"ERROR generating dialogue {i + 1}: {e}")
            print("  Skipping and continuing...")

    # Phase 3 — clean up
    print("\nCleaning dataset (stage directions, placeholders) ...")
    dataset, replacements = anonymize_dataset(dataset)
    print(f"  {replacements} placeholder(s) replaced.")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"\nDataset saved: {args.output} ({len(dataset)} dialogues)")


if __name__ == "__main__":
    main()
