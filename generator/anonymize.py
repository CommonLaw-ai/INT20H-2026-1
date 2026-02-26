"""
anonymize.py — Post-processing service that replaces placeholder emails and phone
numbers in a generated dialogue dataset with realistic fake values.

Detects:
  - Placeholder emails:  [email], [email address], user@example.com, test@test.com, etc.
  - Placeholder phones:  [phone], [phone number], 555-xxxx, (555) xxx-xxxx, +1-800-xxx-xxxx, etc.
  - Bare numeric placeholders that look like order/account numbers: #12345, #00000, etc.

Each unique placeholder within a single dialogue gets the same replacement
(consistent within one conversation), but different dialogues get different values.

Usage:
    python anonymize.py --input dataset.json --output dataset_clean.json
    python anonymize.py --input dataset.json                      # overwrites in place
"""

import argparse
import copy
import json
import re
import sys
from faker import Faker

fake = Faker("en_US")
Faker.seed(0)  # reproducible output

# ── Regex patterns ────────────────────────────────────────────────────────────

# Bracket-style placeholders:  [email], [email address], [phone number], etc.
RE_BRACKET_EMAIL = re.compile(r'\[e-?mail(?:\s+address)?\]', re.IGNORECASE)
RE_BRACKET_PHONE = re.compile(r'\[phone(?:\s+number)?\]', re.IGNORECASE)

# Generic example/test emails:  anything@example.com, user@test.com, foo@email.com
RE_GENERIC_EMAIL = re.compile(
    r'\b[\w.+-]+@(?:example|test|email|sample|placeholder|domain)\.\w{2,}\b',
    re.IGNORECASE,
)

# Real-looking but obviously fake US phone patterns:
#   (555) 000-1234 / 555-123-4567 / +1-555-xxx-xxxx / 1-800-xxx-xxxx
RE_FAKE_PHONE = re.compile(
    r'(?:\+?1[-.\s]?)?'           # optional country code
    r'(?:\(?\b(?:555|800|888|877|866|844|833)\b\)?)'  # obviously fake/toll-free prefix
    r'[-.\s]?\d{3}[-.\s]?\d{4}\b',
    re.IGNORECASE,
)

# Repeated-digit order numbers that models love to invent: #12345, #1234, #00000, #11111
RE_FAKE_ORDER = re.compile(r'#(\d)\1{3,}|#1234\b|#12345\b|#98765\b|#00001\b', re.IGNORECASE)


# ── Replacement helpers ───────────────────────────────────────────────────────

def _fresh_email() -> str:
    return fake.email()


def _fresh_phone() -> str:
    # e.g. "415-782-3901" — realistic US number
    return fake.phone_number()


def _fresh_order() -> str:
    # e.g. "#473829"
    return f"#{fake.numerify('######')}"


# ── Core substitution ─────────────────────────────────────────────────────────

def _replace_in_text(text: str, memo: dict) -> str:
    """
    Replace all placeholder emails and phones in a single text string.

    memo — shared dict for one dialogue so the same placeholder always maps
           to the same replacement value within the conversation.
    """

    def substitute(pattern: re.Pattern, key_prefix: str, generator) -> callable:
        def replacer(match: re.Match) -> str:
            original = match.group(0)
            key = f"{key_prefix}:{original.lower()}"
            if key not in memo:
                memo[key] = generator()
            return memo[key]
        return replacer

    text = RE_BRACKET_EMAIL.sub(substitute(RE_BRACKET_EMAIL, "email", _fresh_email), text)
    text = RE_BRACKET_PHONE.sub(substitute(RE_BRACKET_PHONE, "phone", _fresh_phone), text)
    text = RE_GENERIC_EMAIL.sub(substitute(RE_GENERIC_EMAIL, "email", _fresh_email), text)
    text = RE_FAKE_PHONE.sub(substitute(RE_FAKE_PHONE, "phone", _fresh_phone), text)
    text = RE_FAKE_ORDER.sub(substitute(RE_FAKE_ORDER, "order", _fresh_order), text)

    return text


def anonymize_dialogue(dialogue: dict) -> tuple[dict, int]:
    """
    Process one dialogue entry.

    Returns:
        (anonymized_dialogue, replacements_count)
    """
    result = copy.deepcopy(dialogue)
    memo: dict = {}
    count = 0

    for message in result.get("messages", []):
        original = message["text"]
        replaced = _replace_in_text(original, memo)
        if replaced != original:
            count += len(re.findall(r'\S+', replaced)) - len(re.findall(r'\S+', original)) + 1
            # simpler: just count memo keys added during this message
        message["text"] = replaced

    count = len(memo)  # number of unique placeholders replaced in this dialogue
    return result, count


def anonymize_dataset(dataset: list[dict]) -> tuple[list[dict], int]:
    """
    Process the full dataset. Each dialogue gets its own memo (independent replacements).

    Returns:
        (anonymized_dataset, total_replacements_across_all_dialogues)
    """
    clean = []
    total = 0
    for dialogue in dataset:
        anonymized, count = anonymize_dialogue(dialogue)
        clean.append(anonymized)
        total += count
    return clean, total


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Replace placeholder emails and phone numbers with realistic fake values."
    )
    parser.add_argument("--input",  required=True, help="Path to input dataset JSON file")
    parser.add_argument("--output", default=None,  help="Path to output JSON file (default: overwrite input)")
    args = parser.parse_args()

    output_path = args.output or args.input

    try:
        with open(args.input, "r", encoding="utf-8") as f:
            dataset = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: File not found: {args.input}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON in {args.input}: {e}")
        sys.exit(1)

    print(f"Anonymizing {len(dataset)} dialogue(s) from {args.input} ...")

    clean_dataset, total = anonymize_dataset(dataset)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(clean_dataset, f, ensure_ascii=False, indent=2)

    print(f"Done. {total} placeholder(s) replaced.")
    print(f"Output saved: {output_path}")


if __name__ == "__main__":
    main()
