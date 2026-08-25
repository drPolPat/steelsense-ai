"""Evaluation harness for the SteelSense AI agent.

Runs the fixed query set in queries.json against the live agent, then
grades each answer against its expected_characteristics using a separate,
cheaper model as an LLM judge -- deliberately not the model that answered,
to avoid self-grading bias. See docs/evals.md for methodology notes and a
summary of results.

Usage:
    python -m evals.run_evals
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import anthropic
from dotenv import load_dotenv

from src.backend.agent.agent import ask

load_dotenv()

JUDGE_MODEL = os.environ.get("EVAL_JUDGE_MODEL", "claude-haiku-4-5")
QUERIES_PATH = Path(__file__).parent / "queries.json"
RESULTS_PATH = Path(__file__).parent / "results.json"

JUDGE_SYSTEM_PROMPT = """You are grading answers from a structural-health-monitoring \
assistant against a checklist of expected characteristics. For each checklist item, \
decide whether the answer satisfies it. Be strict but fair: vague or generic coverage \
of a claim that should include a specific number or a specific named finding does not \
count as met. Base your judgment only on the answer text provided, not on outside \
knowledge of what the "right" answer should be."""


def _item_key(i: int) -> str:
    return f"item_{i + 1}"


def _judge_output_schema(n: int) -> dict:
    # A fixed-length JSON array with a strict item count isn't supported by
    # structured outputs (minItems/maxItems must be 0 or 1), so each
    # checklist item gets its own named, required object property instead.
    item_schema = {
        "type": "object",
        "properties": {"met": {"type": "boolean"}, "reason": {"type": "string"}},
        "required": ["met", "reason"],
        "additionalProperties": False,
    }
    keys = [_item_key(i) for i in range(n)]
    return {
        "type": "json_schema",
        "schema": {
            "type": "object",
            "properties": {key: item_schema for key in keys},
            "required": keys,
            "additionalProperties": False,
        },
    }


def grade_answer(
    client: anthropic.Anthropic, query: str, answer: str, characteristics: list[str]
) -> list[dict]:
    checklist = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(characteristics))
    prompt = f"""Question asked: {query}

Assistant's answer:
\"\"\"
{answer}
\"\"\"

Checklist -- grade each item, one result per numbered item:
{checklist}"""

    response = client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=2048,
        system=JUDGE_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
        output_config={"format": _judge_output_schema(len(characteristics))},
    )
    text = next(block.text for block in response.content if block.type == "text")
    parsed = json.loads(text)
    return [parsed[_item_key(i)] for i in range(len(characteristics))]


def run() -> dict:
    queries = json.loads(QUERIES_PATH.read_text(encoding="utf-8"))
    client = anthropic.Anthropic()

    results = []
    for spec in queries:
        print(f"Running: {spec['id']} ...")
        answer = ask(spec["query"])
        grades = grade_answer(client, spec["query"], answer, spec["expected_characteristics"])

        checklist = [
            {"characteristic": c, "met": g["met"], "reason": g["reason"]}
            for c, g in zip(spec["expected_characteristics"], grades)
        ]
        met_count = sum(item["met"] for item in checklist)

        results.append(
            {
                "id": spec["id"],
                "category": spec["category"],
                "query": spec["query"],
                "answer": answer,
                "checklist": checklist,
                "met": met_count,
                "total": len(checklist),
                "passed": met_count == len(checklist),
            }
        )
        print(f"  {met_count}/{len(checklist)} characteristics met")

    summary = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "judge_model": JUDGE_MODEL,
        "n_queries": len(results),
        "n_passed": sum(r["passed"] for r in results),
        "results": results,
    }
    RESULTS_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n{summary['n_passed']}/{summary['n_queries']} queries fully passed all characteristics.")
    print(f"Full results written to {RESULTS_PATH}")
    return summary


if __name__ == "__main__":
    run()
