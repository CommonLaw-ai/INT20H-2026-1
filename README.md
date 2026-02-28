# Support Chat Dataset Generator & Analyzer

A two-part pipeline for generating and analyzing synthetic customer support chat dialogues for AI training datasets.

## Overview

```
generator/generate_v2.py  →  dataset.json  →  analyzer/analyze.py  →  analysis_results.json
```

**Generator** uses Llama 3.1 8B (via Ollama) to produce realistic support dialogues across multiple topics and case types. Each dialogue is validated by an LLM-as-a-judge before being saved.

**Analyzer** reads the generated dataset and evaluates each dialogue — detecting hidden dissatisfaction, scoring agent quality, and identifying agent mistakes.

---

## Requirements

```bash
pip install -r requirements.txt
```

Generator also requires [Ollama](https://ollama.com) running locally:
```bash
ollama pull llama3.1:8b
ollama serve
```

---

## Generator

**File:** `generator/generate_v2.py`

Generates a dataset of support chat dialogues using a 3-phase pipeline:

1. **Scenario building** — LLM generates a unique one-sentence description for each topic × case_type combination
2. **Dialogue generation** — LLM generates a realistic dialogue per scenario, with up to 3 retry attempts at increasing temperature if validation fails
3. **Post-processing** — removes stage directions `(pause)`, `(sighs)` etc. and replaces placeholder emails/phones with realistic fake values

### Dialogue properties

Each dialogue includes:
- `topic` — support request topic (e.g. `payment_issue`, `account_access`)
- `case_type` — scenario type (e.g. `successful`, `conflictual`)
- `description` — LLM-generated scenario summary
- `agent_action` — classified action performed by the agent (e.g. `resolved`, `escalated_to_human`)
- `messages` — list of messages, last one marked with `"is_last": true`

### Topics

`payment_issue`, `technical_error`, `account_access`, `tariff_question`, `refund`, `subscription_cancellation`, `delivery_problem`, `wrong_charge`, `feature_request`, `password_reset`

### Case types

| Case type | Description |
|-----------|-------------|
| `successful` | Issue resolved, customer satisfied |
| `agent_error` | Agent gives wrong info, ignores a question, or contradicts themselves |
| `problematic` | Agent stalls, asks for the same info repeatedly, no progress |
| `conflictual` | Customer shows genuine anger and frustration |
| `hidden_dissatisfaction` | Customer formally thanks the agent, but the core issue was never solved |

### Agent actions

`resolved`, `refund_initiated`, `account_unlocked`, `info_provided`, `follow_up_scheduled`, `escalated_to_human`, `unresolved`, `customer_left`

### Usage

```bash
# Generate all topic × case_type combinations (50 dialogues by default)
python generator/generate_v2.py --output dataset.json

# Generate a specific number of dialogues
python generator/generate_v2.py --output dataset.json --count 20

# Filter by topic and case type
python generator/generate_v2.py --topics payment_issue,refund --case-types successful,agent_error

# Generate with varied descriptions for repeated pairs
python generator/generate_v2.py --count 100 --vary-descriptions

# Inject a company policy document into the system prompt
python generator/generate_v2.py --policy company_policy.txt
```

### Validation

Every generated dialogue is checked against 3 criteria before being accepted:
1. **Complete** — dialogue has a natural closing, does not end on an unanswered question
2. **Scenario match** — dialogue content matches the declared `case_type`
3. **Security** — agent never requests sensitive data (full card number, CVV, PIN)

If validation fails, the generator retries up to 3 times with increasing temperature. If all attempts fail, the best-effort result is saved with a warning.

### Determinism

Results are deterministic by default (`--seed 42`). Pass `--seed <N>` for a different reproducible run.

---

## Analyzer

**File:** `analyzer/analyze.py`

Reads `dataset.json` and analyzes each dialogue using an LLM. Supports two backends:

- **Local** (default) — Qwen 2.5 3B via HuggingFace Transformers, runs on CPU/GPU/MPS
- **Groq** — Llama 3.3 70B via Groq API (faster, requires free API key from [console.groq.com](https://console.groq.com))

### Output per dialogue

| Field | Description |
|-------|-------------|
| `intent` | Detected support topic (`payment_issue`, `technical_error`, `account_access`, `tariff_question`, `refund`, `other`) |
| `satisfaction` | Customer satisfaction: `satisfied`, `neutral`, or `unsatisfied` |
| `quality_score` | Agent quality score 1–5 |
| `agent_mistakes` | List of detected mistakes: `ignored_question`, `incorrect_info`, `rude_tone`, `no_resolution`, `unnecessary_escalation` |

The analyzer is specifically designed to detect **hidden dissatisfaction** — cases where the customer says "thank you" but the underlying problem was never actually solved.

### Usage

```bash
# Local backend (downloads model on first run)
python analyzer/analyze.py --input dataset.json --output analysis_results.json

# Groq backend (fast, requires GROQ_API_KEY env variable)
export GROQ_API_KEY=<your-key>
python analyzer/analyze.py --backend groq --input dataset.json

# Cache model on Google Drive (for Colab)
python analyzer/analyze.py --model-cache /content/drive/MyDrive/hf_models --input dataset.json
```

---

## Example dialogues

### Successful — `tariff_question`
> Customer confused about a $20 bill increase after switching plans; agent explains the overage fee clearly.

```
Customer: Hi, I'm a bit confused — I recently changed from an unlimited data plan to a 5GB
          monthly limit and now my bill is $20 more than usual.
Agent:    I'd be happy to help you understand your recent increase. Can you confirm the
          date of your plan change?
Customer: It was just last month, on the 15th. Before that my bills were always around $50.
Agent:    I see that your data usage increased significantly after switching. Did you notice
          any changes in your internet usage recently?
Customer: Honestly, no. But I do have a bunch of kids who use my phone too.
Agent:    That makes sense. The 5GB limit might not be enough for them. It looks like you
          went over by about 3 GB last month — that added an extra $10 on top of the
          $10 data plan increase.
Customer: Okay, that makes more sense now. Thanks for explaining it to me!
Agent:    You're welcome! Is there anything else I can help with today?
Customer: No, thank you so much for your time. My bill should be okay from now on.
Agent:    Great! Have a wonderful day!
Customer: You too!
```
**agent_action:** `info_provided` | **case_type:** `successful`

---

### Agent error — `payment_issue`
> Card declined; agent incorrectly insists the address is the cause despite customer confirming it is correct.

```
Customer: I'm trying to make a payment on my account but it says it's been declined.
Agent:    Can you please provide your order number and the last 4 digits of your card?
Customer: It's order #LR1234. The last four are 5678.
Agent:    Can you double-check that your address is up to date on the card used?
          Sometimes our system gets confused if it's not current.
Customer: I've checked and my address is most definitely current.
Agent:    Well, sometimes even with correct info, we still need users to verify certain
          details. Do you think if I guided you through updating your billing information,
          our system should allow the payment to process next time?
Customer: Unfortunately not. Like I said, my address is current and has been all along.
Agent:    Okay. This might have been an isolated incident, don't worry about it!
          If you need further assistance, please contact our support email.
Customer: I'll make sure of it. Thanks for nothing.
```
**agent_action:** `customer_left` | **case_type:** `agent_error`

---

## Project structure

```
├── dataset.json              # Generated dialogue dataset
├── analysis_results.json     # Analyzer output
├── requirements.txt
├── generator/
│   ├── generate_v2.py        # Main generator
│   ├── validate.py           # LLM-as-a-judge validator
│   ├── actions.py            # Agent action classifier
│   └── anonymize.py          # Post-processor (stage directions + placeholder replacement)
└── analyzer/
    └── analyze.py            # Dialogue quality analyzer
```
