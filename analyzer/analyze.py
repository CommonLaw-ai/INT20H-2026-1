#!/usr/bin/env python3
"""
analyze.py — Analyzes customer support chat dialogs using a local HuggingFace LLM.

Reads a dataset of support chat dialogs and for each one determines:
  - intent:         topic of the support request
  - satisfaction:   customer satisfaction level (detects hidden dissatisfaction)
  - quality_score:  agent quality score (1-5)
  - agent_mistakes: list of mistakes made by the agent

Results are saved as a JSON array (one object per dialog).

--- LOCAL BACKEND (default) ---
Requirements:
  pip install transformers torch accelerate pydantic

Usage:
    python analyzer/analyze.py --input dataset.json --output analysis_results.json

Google Drive / Colab usage (model cached on Drive, reused across sessions):
    from google.colab import drive
    drive.mount('/content/drive')
    !python analyzer/analyze.py \\
        --model-cache /content/drive/MyDrive/hf_models \\
        --input dataset.json

--- GROQ BACKEND (fast, requires free API key from console.groq.com) ---
Requirements:
  pip install groq pydantic

Usage:
    export GROQ_API_KEY=<your-key>
    python analyzer/analyze.py --backend groq --input dataset.json
"""

import json
import re
import argparse
import sys
from pathlib import Path
from typing import List, Literal

from pydantic import BaseModel, ValidationError


DEFAULT_LOCAL_MODEL = "Qwen/Qwen2.5-3B-Instruct"
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = """You are an expert customer support quality analyst. Analyze support chat \
dialogs and evaluate both the customer's experience and the agent's performance.

For each dialog determine:

1. intent — Primary reason for the support request (pick exactly one):
   - payment_issue:   problems with payments, billing errors, failed transactions
   - technical_error: app crashes, bugs, technical malfunctions
   - account_access:  login problems, account locked, password issues
   - tariff_question: questions about plans, pricing, features
   - refund:          refund requests, chargeback issues
   - other:           anything not covered above

2. satisfaction — Customer's REAL satisfaction level (pick exactly one):
   - satisfied:   issue was genuinely resolved and the customer is happy
   - neutral:     issue partially resolved or customer is indifferent
   - unsatisfied: customer is dissatisfied

   CRITICAL — detect HIDDEN dissatisfaction:
   Mark as "unsatisfied" when the customer formally says "thanks" or "okay" BUT the
   underlying problem was NOT actually solved, they received vague/incorrect info,
   or resolution was deferred without a clear commitment.

3. quality_score — Agent performance integer from 1 to 5:
   - 5: excellent — resolved completely, empathetic, clear communication
   - 4: good — resolved with minor shortcomings
   - 3: average — partial resolution or noticeable errors
   - 2: poor — significant mistakes, incomplete resolution
   - 1: very poor — failed to help, rude, or worsened the situation

4. agent_mistakes — Array of mistakes the agent made (empty array if none):
   - "ignored_question":       agent did not address a specific customer question
   - "incorrect_info":         agent provided wrong or misleading information
   - "rude_tone":              agent was impolite, dismissive, or unprofessional
   - "no_resolution":          agent failed to resolve the customer's core issue
   - "unnecessary_escalation": agent escalated when it was not needed

Be critical and analytical. A polite conversation ending does NOT mean satisfaction —
look at whether the actual problem was genuinely solved.

Respond ONLY with a valid JSON object matching this exact schema (no extra text):
{
  "intent": "<one of the intent values>",
  "satisfaction": "<satisfied|neutral|unsatisfied>",
  "quality_score": <integer 1-5>,
  "agent_mistakes": [<zero or more mistake strings>]
}"""


class DialogAnalysis(BaseModel):
    intent: Literal[
        "payment_issue", "technical_error", "account_access",
        "tariff_question", "refund", "other",
    ]
    satisfaction: Literal["satisfied", "neutral", "unsatisfied"]
    quality_score: Literal[1, 2, 3, 4, 5]
    agent_mistakes: List[
        Literal[
            "ignored_question", "incorrect_info", "rude_tone",
            "no_resolution", "unnecessary_escalation",
        ]
    ]


def format_dialog(dialog: dict) -> str:
    lines = []
    for msg in dialog["messages"]:
        role = "Customer" if msg["role"] == "customer" else "Support Agent"
        lines.append(f"{role}: {msg['text']}")
    return "\n".join(lines)


def extract_json(text: str) -> dict:
    """Extract first JSON object from text, handling markdown code fences."""
    # Strip markdown code fences if present
    text = re.sub(r"```(?:json)?", "", text).strip()

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Find the outermost {...} block
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    raise ValueError(f"No valid JSON found in model output:\n{text[:300]}")


# ---------------------------------------------------------------------------
# Local (HuggingFace transformers) backend
# ---------------------------------------------------------------------------

def load_local_pipeline(model_name: str, cache_dir: str | None):
    """Load a HuggingFace model and return a text-generation pipeline."""
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    except ImportError:
        print(
            "ERROR: 'transformers' and 'torch' are required for the local backend.\n"
            "Install with:  pip install transformers torch accelerate",
            file=sys.stderr,
        )
        sys.exit(1)

    # Detect best available device
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.bfloat16
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float16
    else:
        device = "cpu"
        dtype = torch.float32

    print(f"Loading model '{model_name}' on {device.upper()} (this may take a while on first run)...")
    if cache_dir:
        print(f"  Model cache directory: {cache_dir}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

    # device_map="auto" is only reliable for CUDA multi-GPU; use explicit device otherwise
    load_kwargs = dict(cache_dir=cache_dir, dtype=dtype, low_cpu_mem_usage=True)
    if device == "cuda":
        load_kwargs["device_map"] = "auto"
    if device == "mps":
        # "eager" disables the MPS flash-attention kernel that causes nan/inf with float16
        load_kwargs["attn_implementation"] = "eager"

    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)

    if device != "cuda":
        model = model.to(device)

    # Patch the model's built-in generation_config to avoid conflicts
    model.generation_config.do_sample = False
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None
    model.generation_config.max_new_tokens = 300
    model.generation_config.max_length = None

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        return_full_text=False,
    )
    print(f"Model loaded on {device.upper()}.\n")
    return pipe


def analyze_dialog_local(pipe, dialog: dict) -> dict:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": "Analyze this customer support dialog:\n\n" + format_dialog(dialog),
        },
    ]

    for attempt in range(1, 4):
        output = pipe(messages, do_sample=False, max_new_tokens=300)
        # pipeline with return_full_text=False returns just the new tokens
        response_text = output[0]["generated_text"]

        try:
            data = extract_json(response_text)
            analysis = DialogAnalysis(**data)
            return {
                "id": dialog["id"],
                "intent": analysis.intent,
                "satisfaction": analysis.satisfaction,
                "quality_score": analysis.quality_score,
                "agent_mistakes": list(analysis.agent_mistakes),
            }
        except (ValueError, ValidationError) as exc:
            if attempt == 3:
                raise RuntimeError(
                    f"Failed to get valid JSON after 3 attempts. Last error: {exc}"
                ) from exc
            # Add error feedback and retry
            messages.append({"role": "assistant", "content": response_text})
            messages.append({
                "role": "user",
                "content": (
                    f"Your previous response was not valid JSON: {exc}\n"
                    "Please respond ONLY with the JSON object, nothing else."
                ),
            })


# ---------------------------------------------------------------------------
# Groq API backend
# ---------------------------------------------------------------------------

def load_groq_client():
    try:
        from groq import Groq
    except ImportError:
        print(
            "ERROR: 'groq' package is required for the groq backend.\n"
            "Install with:  pip install groq",
            file=sys.stderr,
        )
        sys.exit(1)
    return Groq()


def analyze_dialog_groq(client, model: str, dialog: dict) -> dict:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": "Analyze this customer support dialog:\n\n" + format_dialog(dialog),
            },
        ],
        response_format={"type": "json_object"},
        temperature=0,
        seed=42,
    )
    raw = response.choices[0].message.content
    data = json.loads(raw)
    analysis = DialogAnalysis(**data)
    return {
        "id": dialog["id"],
        "intent": analysis.intent,
        "satisfaction": analysis.satisfaction,
        "quality_score": analysis.quality_score,
        "agent_mistakes": list(analysis.agent_mistakes),
    }


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def analyze_dataset(
    input_path: str,
    output_path: str,
    backend: str,
    model: str,
    model_cache: str | None,
) -> None:
    with open(input_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    print(f"Loaded {len(dataset)} dialogs from '{input_path}'")
    print(f"Backend: {backend} | Model: {model}\n")

    if backend == "local":
        pipe = load_local_pipeline(model, model_cache)

        def analyze_fn(dialog: dict) -> dict:
            return analyze_dialog_local(pipe, dialog)
    else:
        client = load_groq_client()

        def analyze_fn(dialog: dict) -> dict:
            return analyze_dialog_groq(client, model, dialog)

    results = []
    for i, dialog in enumerate(dataset, 1):
        dialog_id = dialog["id"]
        print(f"  [{i}/{len(dataset)}] Analyzing dialog ID {dialog_id}...", end=" ", flush=True)
        try:
            result = analyze_fn(dialog)
        except Exception as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            raise
        results.append(result)
        print(
            f"intent={result['intent']}, "
            f"satisfaction={result['satisfaction']}, "
            f"score={result['quality_score']}, "
            f"mistakes={result['agent_mistakes']}"
        )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nDone. Results saved to '{output_path}'")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze customer support chat dialogs using a local or Groq LLM"
    )
    parser.add_argument(
        "--input", default="dataset.json",
        help="Path to input dataset JSON (default: dataset.json)",
    )
    parser.add_argument(
        "--output", default="analysis_results.json",
        help="Path to output results JSON (default: analysis_results.json)",
    )
    parser.add_argument(
        "--backend", choices=["local", "groq"], default="local",
        help="Inference backend: 'local' (HuggingFace, default) or 'groq' (API)",
    )
    parser.add_argument(
        "--model", default=None,
        help=(
            f"Model name. Defaults: local='{DEFAULT_LOCAL_MODEL}', "
            f"groq='{DEFAULT_GROQ_MODEL}'"
        ),
    )
    parser.add_argument(
        "--model-cache", default=None, metavar="DIR",
        help=(
            "Directory to cache/load the HuggingFace model weights. "
            "Point to a Google Drive folder in Colab to persist across sessions. "
            "Example: /content/drive/MyDrive/hf_models"
        ),
    )
    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"Error: Input file '{args.input}' not found.", file=sys.stderr)
        sys.exit(1)

    model = args.model or (
        DEFAULT_LOCAL_MODEL if args.backend == "local" else DEFAULT_GROQ_MODEL
    )

    analyze_dataset(
        input_path=args.input,
        output_path=args.output,
        backend=args.backend,
        model=model,
        model_cache=args.model_cache,
    )


if __name__ == "__main__":
    main()
