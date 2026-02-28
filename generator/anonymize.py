"""
anonymize.py — Post-processing service that cleans up generated dialogue datasets.

Does two things:
  1. Removes stage directions in parentheses: (pause), (sighs), (checks account), etc.
  2. Replaces placeholder emails/phones/usernames/account numbers with realistic fake values.

Detects:
  - Stage directions:     (pause), (sighs), (typing), (checks account), etc.
  - Placeholder emails:   [email], [email address], [customer email], [user@company.com], user@example.com, etc.
  - Placeholder phones:   [phone], [phone number], 555-xxxx, (555) xxx-xxxx, +1-800-xxx-xxxx, etc.
  - Placeholder names:    [username], [name], [customer name], [full name], etc.
  - Fake order numbers:   #12345, #1234, #00000, #11111, etc.
  - Fake account numbers: 1234567, 0000000, 1111111 (7-digit sequences of repeating/sequential digits)

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

# Bracket-style placeholders:  [email], [email address], [customer email], [support@company.com], [phone number], [username], [name], etc.
RE_BRACKET_EMAIL    = re.compile(r'\[(?:e-?mail(?:\s+address)?|customer\s*e-?mail|[\w.+-]+@[\w.-]+)\]', re.IGNORECASE)
RE_BRACKET_PHONE    = re.compile(r'\[phone(?:\s+number)?\]', re.IGNORECASE)
RE_BRACKET_USERNAME = re.compile(r'\[(?:user\s*name|user|customer\s*name|full\s*name|name)\]', re.IGNORECASE)

# Markdown email links: [thull@example.com](mailto:thull@example.com) — replace whole thing with just the fake email
RE_MARKDOWN_EMAIL = re.compile(
    r'\[[\w.+-]+@[\w.-]+\]\(mailto:[\w.+-]+@[\w.-]+\)',
    re.IGNORECASE,
)

# Markdown URL links with example/placeholder domains: [click here](https://example.com/link)
RE_MARKDOWN_URL = re.compile(
    r'\[([^\]]+)\]\(https?://(?:example|test|placeholder|sample)\.(?:com|org|net)[^\)]*\)',
    re.IGNORECASE,
)

# Bare URLs with example/placeholder domains: https://example.com/anything
RE_EXAMPLE_URL = re.compile(
    r'https?://(?:example|test|placeholder|sample)\.(?:com|org|net)\S*',
    re.IGNORECASE,
)

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

# Fake account numbers: 7-digit sequences of repeating or sequential digits (1234567, 0000000, 1111111)
RE_FAKE_ACCOUNT = re.compile(r'\b(?:1234567|7654321|0000000|1111111|2222222|3333333|9999999)\b')

# Stage directions in parentheses: (pause), (sighs), (checks account), (long pause), etc.
# Only matches known action/emotion verbs — avoids stripping legitimate content like (Essential) or (#473829)
RE_STAGE_DIRECTION = re.compile(
    r'\s*\(\s*(?:'
    r'pause|paus\w*|'
    r'sigh\w*|laugh\w*|chuckl\w*|'
    r'check\w*|look\w*|review\w*|search\w*|verify\w*|pull\w*|pull up\w*|'
    r'wait\w*|hold\w*|think\w*|hesitat\w*|'
    r'nod\w*|shrug\w*|'
    r'typ\w*|click\w*|scroll\w*|'
    r'long pause|brief pause|short pause|moment of silence|'
    r'after a (?:pause|moment)|takes? a (?:deep )?breath'
    r')[^)]*\)',
    re.IGNORECASE,
)


# ── Stage direction removal ───────────────────────────────────────────────────

def remove_stage_directions(text: str) -> str:
    """Remove parenthesized stage directions like (pause), (sighs), (checks account)."""
    return RE_STAGE_DIRECTION.sub('', text).strip()


# ── Replacement helpers ───────────────────────────────────────────────────────

def _fresh_email() -> str:
    return fake.email()


def _fresh_phone() -> str:
    # e.g. "415-782-3901" — realistic US number
    return fake.phone_number()


def _fresh_order() -> str:
    # e.g. "#473829"
    return f"#{fake.numerify('######')}"


def _fresh_url() -> str:
    # e.g. "https://support.acme-corp.com/help/billing"
    slug = fake.slug()
    return f"https://support.{fake.domain_name()}/{slug}"


def _fresh_username() -> str:
    # e.g. "jsmith92" or "mike_jones"
    return fake.user_name()


def _fresh_account() -> str:
    # e.g. "4829103" — realistic-looking 7-digit account number
    return fake.numerify('#######')


# ── Core substitution ─────────────────────────────────────────────────────────

def _replace_in_text(text: str, memo: dict) -> str:
    """
    Clean a single message text:
      1. Remove stage directions: (pause), (sighs), etc.
      2. Replace placeholder emails, phones, and order numbers.

    memo — shared dict for one dialogue so the same placeholder always maps
           to the same replacement value within the conversation.
    """
    text = remove_stage_directions(text)

    def substitute(pattern: re.Pattern, key_prefix: str, generator) -> callable:
        def replacer(match: re.Match) -> str:
            original = match.group(0)
            key = f"{key_prefix}:{original.lower()}"
            if key not in memo:
                memo[key] = generator()
            return memo[key]
        return replacer

    text = RE_BRACKET_EMAIL.sub(substitute(RE_BRACKET_EMAIL,       "email",    _fresh_email),    text)
    text = RE_BRACKET_PHONE.sub(substitute(RE_BRACKET_PHONE,       "phone",    _fresh_phone),    text)
    text = RE_BRACKET_USERNAME.sub(substitute(RE_BRACKET_USERNAME, "username", _fresh_username), text)
    text = RE_MARKDOWN_EMAIL.sub(substitute(RE_MARKDOWN_EMAIL,     "email",    _fresh_email),    text)
    text = RE_MARKDOWN_URL.sub(lambda m: m.group(1), text)   # [text](https://example.com) → text
    text = RE_EXAMPLE_URL.sub(substitute(RE_EXAMPLE_URL,           "url",      _fresh_url),      text)
    text = RE_GENERIC_EMAIL.sub(substitute(RE_GENERIC_EMAIL,       "email",    _fresh_email),    text)
    text = RE_FAKE_PHONE.sub(substitute(RE_FAKE_PHONE,             "phone",    _fresh_phone),    text)
    text = RE_FAKE_ORDER.sub(substitute(RE_FAKE_ORDER,             "order",    _fresh_order),    text)
    text = RE_FAKE_ACCOUNT.sub(substitute(RE_FAKE_ACCOUNT,         "account",  _fresh_account),  text)

    return text


def anonymize_dialogue(dialogue: dict) -> tuple[dict, int]:
    """
    Process one dialogue entry.

    Returns:
        (anonymized_dialogue, replacements_count)
    """
    result = copy.deepcopy(dialogue)
    memo: dict = {}

    for message in result.get("messages", []):
        message["text"] = _replace_in_text(message["text"], memo)

    return result, len(memo)


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
