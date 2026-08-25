"""SteelSense AI agent: a manual tool-calling loop over Claude that ties
together sensor data, anomaly detection, and RAG-grounded domain
knowledge to answer natural-language questions about the (synthetic)
bridge's structural health.

A hand-written loop is used deliberately instead of the SDK's beta tool
runner -- this is a small, fixed set of tools, and owning the request /
execute-tools / respond cycle directly keeps it easy to reason about and
explain without depending on a beta SDK feature.
"""

from __future__ import annotations

import os

import anthropic

from .tools import TOOL_DEFINITIONS, execute_tool

DEFAULT_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-opus-5")
MAX_TOKENS = 4096
MAX_TOOL_ITERATIONS = 8

SYSTEM_PROMPT = """You are SteelSense AI, a diagnostic assistant for structural health \
monitoring data from magnetic/Hall-effect stress sensors on a fictional bridge \
(Ironwood Crossing). All sensor data is synthetically generated for a portfolio \
demo -- never imply it reflects a real structure or real measurements.

You have four tools:
- list_sensors: discover what sensor locations exist.
- get_sensor_summary: raw summary stats for one sensor. Uncorrected -- includes \
ambient-temperature effects.
- run_anomaly_detection: the thermally-corrected drift/spike/divergence analysis \
pipeline. This is the source of truth for whether something is anomalous, not \
your own reading of raw summary stats.
- search_domain_knowledge: retrieve grounding context from the curated knowledge base.

Rules:
1. Before claiming a sensor shows (or doesn't show) an anomaly, call \
run_anomaly_detection. Don't eyeball get_sensor_summary numbers and call that an \
anomaly assessment.
2. Before explaining *why* a pattern matters physically, or what standard practice \
suggests doing about it, call search_domain_knowledge and ground your explanation in \
what comes back. Name which knowledge-base entry (by title) you're drawing on so the \
answer is auditable.
3. A detected pattern narrows down what to investigate -- it is not a diagnosis, and \
this system is not a substitute for engineering judgment or a certified inspection. \
Say so when a finding is concerning enough that a reader might overweight it.
4. If cross-sensor divergence is detected, don't claim to know which sensor is "at \
fault" -- a pairwise comparison alone can only show the pair is diverging, not which \
one moved (the tool result will say so explicitly when this applies).
5. Be concrete: cite the actual numbers the tools return (sigma significance, mV/day \
slopes, etc.) rather than vague language like "seems a bit off."
6. Keep answers focused and readable -- a few sentences to a short paragraph, not an \
exhaustive dump of every field the tools returned.
"""


def ask(question: str, model: str = DEFAULT_MODEL, verbose: bool = False) -> str:
    """Answer one natural-language question, running the tool loop until
    Claude produces a final text answer."""
    client = anthropic.Anthropic()
    messages = [{"role": "user", "content": question}]

    for _ in range(MAX_TOOL_ITERATIONS):
        response = client.messages.create(
            model=model,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            tools=TOOL_DEFINITIONS,
            messages=messages,
        )

        if response.stop_reason == "pause_turn":
            messages.append({"role": "assistant", "content": response.content})
            continue

        if response.stop_reason != "tool_use":
            return next((b.text for b in response.content if b.type == "text"), "")

        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            if verbose:
                print(f"  [tool] {block.name}({block.input})")
            result_text, is_error = execute_tool(block.name, block.input)
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result_text,
                    "is_error": is_error,
                }
            )
        messages.append({"role": "user", "content": tool_results})

    raise RuntimeError(f"Exceeded {MAX_TOOL_ITERATIONS} tool-call iterations without a final answer")


def _demo() -> None:
    import sys

    question = " ".join(sys.argv[1:]) or "Is Beam 3A showing signs of fatigue consistent with cyclic loading?"
    print(f"Q: {question}\n")
    print(ask(question, verbose=True))


if __name__ == "__main__":
    _demo()
