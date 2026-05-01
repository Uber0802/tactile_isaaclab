"""Generate per-task language paraphrases via an LLM.

Reads task identifiers from the AnyTouch2 npy directory and asks the model
to produce N natural-language instructions per task that vary in lexical
choice, syntactic structure, and specificity. Output is a JSON file that
`anytouch2_to_h5.py --instructions_json ...` consumes verbatim.

Currently uses Google Gemini (google-genai SDK).

Requires:
    pip install google-genai
    export GEMINI_API_KEY=...

Usage:
    python scripts/generate_instructions_via_llm.py \\
        --data_dir /mnt/tank/uber/AnyTouch2/tactile_dataset \\
        --output /mnt/tank/uber/Tactile-Reward/instructions.json
"""
from __future__ import annotations

import os
import re
import json
import argparse
from typing import List

from pydantic import BaseModel, Field
from google import genai
from google.genai import types


SYSTEM_PROMPT = """You write natural-language task descriptions for robotic manipulation.

You will receive a list of task identifiers in snake_case from the AnyTouch2 dataset \
(a bimanual robot equipped with shear-force tactile sensors on both hands).

For each task, produce N distinct one-sentence instructions that vary in:
  1. Lexical choice (synonyms, alternative verbs)
  2. Syntactic structure (imperative vs. gerund vs. declarative)
  3. Specificity (how much detail is included)

Each instruction must:
  - Be a single concise English sentence a human would naturally say to a robot.
  - Stay faithful to the task semantics (do not invent or change the action).
  - Avoid technical jargon and avoid repeating the snake_case identifier verbatim.

Return a JSON array matching the requested schema."""


class TaskInstructions(BaseModel):
    task: str
    instructions: List[str]


def discover_tasks(data_dir: str):
    pattern = re.compile(r"^(.+)__(\d+)\.npy$")
    tasks = set()
    for f in os.listdir(data_dir):
        m = pattern.match(f)
        if m:
            tasks.add(m.group(1))
    return sorted(tasks)


def build_user_prompt(tasks: list[str], n_per_task: int) -> str:
    lines = [
        f"Generate {n_per_task} paraphrases for each of the following "
        f"tactile manipulation tasks:",
        "",
    ]
    lines.extend(f"- {t}" for t in tasks)
    lines.append("")
    lines.append(
        "Return a JSON array; each element has a `task` field (the original "
        f"identifier verbatim) and an `instructions` array of {n_per_task} strings."
    )
    return "\n".join(lines)


def main(args):
    if args.from_metadata:
        import h5py
        with h5py.File(args.from_metadata, "r") as h5:
            tasks = sorted(h5.keys())
    else:
        tasks = discover_tasks(args.data_dir)
    if not tasks:
        raise SystemExit(f"no tasks discovered under {args.data_dir}")
    print(f"discovered {len(tasks)} tasks")

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit(
            "GEMINI_API_KEY (or GOOGLE_API_KEY) not set. "
            "Get one at https://aistudio.google.com/apikey"
        )
    client = genai.Client(api_key=api_key)

    user_prompt = build_user_prompt(tasks, args.n_per_task)

    print(f"calling {args.model} (thinking enabled, JSON-schema constrained)...")
    response = client.models.generate_content(
        model=args.model,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=list[TaskInstructions],
            # thinking_budget=-1 lets the model decide; bigger numbers force more thinking.
            thinking_config=types.ThinkingConfig(thinking_budget=-1),
            max_output_tokens=args.max_tokens,
        ),
    )

    text = response.text
    if not text:
        raise SystemExit(f"empty response: {response}")

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as e:
        raise SystemExit(f"failed to parse JSON: {e}\nresponse start: {text[:500]!r}")

    if not isinstance(parsed, list):
        raise SystemExit(f"expected JSON array, got {type(parsed).__name__}: {text[:200]}")

    instructions: dict[str, list[str]] = {
        item["task"]: item["instructions"]
        for item in parsed
        if isinstance(item, dict) and "task" in item and "instructions" in item
    }

    missing = [t for t in tasks if t not in instructions]
    if missing:
        print(f"WARNING: missing {len(missing)} tasks (first few): {missing[:5]}")
    short = [t for t, v in instructions.items() if len(v) < args.n_per_task]
    if short:
        print(f"WARNING: {len(short)} tasks returned < {args.n_per_task} paraphrases: "
              f"{short[:5]}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as fh:
        json.dump(instructions, fh, indent=2, ensure_ascii=False)
    print(f"wrote {len(instructions)} task entries to {args.output}")
    if response.usage_metadata:
        u = response.usage_metadata
        print(f"usage: prompt={u.prompt_token_count}, "
              f"output={u.candidates_token_count}, "
              f"thoughts={getattr(u, 'thoughts_token_count', 0) or 0}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir",
                    default="/mnt/tank/uber/AnyTouch2/tactile_dataset",
                    help="Directory containing AnyTouch2 *.npy files (used for task discovery).")
    ap.add_argument("--from_metadata", default=None,
                    help="Optional metadata H5 to read task names from instead.")
    ap.add_argument("--output",
                    default="/mnt/tank/uber/Tactile-Reward/instructions.json")
    ap.add_argument("--n_per_task", type=int, default=5,
                    help="Paraphrases per task.")
    ap.add_argument("--model", default="gemini-2.5-pro",
                    help="e.g. gemini-2.5-pro, gemini-2.5-flash.")
    ap.add_argument("--max_tokens", type=int, default=16000)
    args = ap.parse_args()
    main(args)
