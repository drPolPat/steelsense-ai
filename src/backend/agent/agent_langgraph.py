"""LangGraph implementation of the SteelSense AI agent.

A genuine alternative to the hand-written tool-use loop in agent.py, not
a simplified demo -- same four tools (delegating to the exact same
tools.execute_tool dispatcher agent.py's manual loop calls), same system
prompt, same default model. What differs is how the tool-calling loop
itself is built: LangGraph's standard StateGraph + ToolNode +
tools_condition pattern, wired explicitly here rather than hidden behind
the one-line `create_react_agent` prebuilt helper, so the state machine
is something you can actually read and reason about.

See docs/architecture.md for a side-by-side comparison of this against
the manual implementation, and which one this project actually uses by
default and why.

Run directly for a quick comparison against agent.py:
    python -m src.backend.agent.agent_langgraph "<question>"

Or select it for the API via:
    AGENT_IMPLEMENTATION=langgraph uvicorn src.backend.api.main:app
"""

from __future__ import annotations

from .. import _native_ext_shims

_native_ext_shims.install()  # must run before any langgraph/langchain_anthropic import

from langchain_anthropic import ChatAnthropic  # noqa: E402
from langchain_core.messages import HumanMessage, SystemMessage  # noqa: E402
from langchain_core.tools import tool  # noqa: E402
from langgraph.errors import GraphRecursionError  # noqa: E402
from langgraph.graph import END, START, MessagesState, StateGraph  # noqa: E402
from langgraph.graph.state import CompiledStateGraph  # noqa: E402
from langgraph.prebuilt import ToolNode, tools_condition  # noqa: E402

from . import tools as _tools  # noqa: E402
from .agent import DEFAULT_MODEL, MAX_TOKENS, MAX_TOOL_ITERATIONS, SYSTEM_PROMPT  # noqa: E402

# Each manual-loop "iteration" (one model call + one tool-execution round)
# is roughly two graph steps (the agent node, then the tools node), plus
# some headroom -- kept in the same ballpark as agent.py's
# MAX_TOOL_ITERATIONS so both implementations give up around the same
# point rather than one being far more patient than the other.
RECURSION_LIMIT = 2 * MAX_TOOL_ITERATIONS + 4


def _call_tool(name: str, **kwargs) -> str:
    """Delegate to the exact same dispatcher agent.py's manual loop uses,
    so both agents run identical business logic and only the loop
    construction differs."""
    result, _is_error = _tools.execute_tool(name, kwargs)
    return result


@tool
def list_sensors() -> str:
    """List every sensor location on the structure, with a short
    description of where each one is mounted. Call this first if you're
    not sure what sensor locations exist."""
    return _call_tool("list_sensors")


@tool
def get_sensor_summary(sensor_id: str, days: int | None = None) -> str:
    """Get structured summary statistics (mean/std/min/max/latest reading
    and a simple linear trend) for one sensor location. This is
    uncorrected and includes ambient-temperature effects -- use
    run_anomaly_detection for a proper drift/spike/divergence assessment,
    not this.

    Args:
        sensor_id: Sensor id or location name, e.g. 'Beam 3A' or 'beam-3a'.
        days: Restrict the summary to the last N days. Omit for the full series.
    """
    return _call_tool("get_sensor_summary", sensor_id=sensor_id, days=days)


@tool
def run_anomaly_detection(sensor_id: str | None = None) -> str:
    """Run the thermally-corrected drift / spike / cross-sensor-
    divergence detection pipeline -- the source of truth for whether a
    sensor is showing an anomaly. Pass a sensor_id to analyze one
    location, or omit it to analyze and rank every sensor by concern
    score (use that form for 'which sensor is most concerning' style
    questions).

    Args:
        sensor_id: Sensor id or location name. Omit to analyze all sensors.
    """
    return _call_tool("run_anomaly_detection", sensor_id=sensor_id)


@tool
def search_domain_knowledge(query: str, k: int = 3) -> str:
    """Retrieve relevant domain-knowledge excerpts (magnetic sensing
    principles, steel fatigue mechanisms, SHM standards, and diagnostic
    heuristics) to ground an explanation. Always call this before
    explaining *why* a pattern matters physically or what standard
    practice says to do about it.

    Args:
        query: Natural-language search query.
        k: Number of results to return (default 3).
    """
    return _call_tool("search_domain_knowledge", query=query, k=k)


TOOLS = [list_sensors, get_sensor_summary, run_anomaly_detection, search_domain_knowledge]

_graph_cache: dict[str, CompiledStateGraph] = {}


def _build_graph(model: str) -> CompiledStateGraph:
    llm = ChatAnthropic(model=model, max_tokens=MAX_TOKENS).bind_tools(TOOLS)

    def call_model(state: MessagesState) -> dict:
        return {"messages": [llm.invoke(state["messages"])]}

    graph = StateGraph(MessagesState)
    graph.add_node("agent", call_model)
    graph.add_node("tools", ToolNode(TOOLS))
    graph.add_edge(START, "agent")
    # Standard LangGraph ReAct wiring: after the model responds, route to
    # the tools node if it made tool calls, otherwise end.
    graph.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    return graph.compile()


def _get_graph(model: str) -> CompiledStateGraph:
    if model not in _graph_cache:
        _graph_cache[model] = _build_graph(model)
    return _graph_cache[model]


def ask(question: str, model: str = DEFAULT_MODEL, verbose: bool = False) -> str:
    """Answer one natural-language question by running the compiled
    StateGraph until it reaches END."""
    graph = _get_graph(model)
    initial_state = {
        "messages": [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=question),
        ]
    }

    try:
        result = graph.invoke(initial_state, config={"recursion_limit": RECURSION_LIMIT})
    except GraphRecursionError as exc:
        raise RuntimeError(
            f"Exceeded {RECURSION_LIMIT} graph steps without a final answer"
        ) from exc

    if verbose:
        for message in result["messages"]:
            for call in getattr(message, "tool_calls", None) or []:
                print(f"  [tool] {call['name']}({call['args']})")

    return _extract_text(result["messages"][-1].content)


def _extract_text(content: str | list) -> str:
    """Claude Opus 5's extended thinking makes AIMessage.content a list of
    blocks (thinking + text) rather than a plain string -- pull out just
    the text, mirroring agent.py's `b.type == "text"` filtering over the
    raw Anthropic SDK's content blocks."""
    if isinstance(content, str):
        return content
    parts = [block.get("text", "") for block in content if isinstance(block, dict) and block.get("type") == "text"]
    return "\n".join(parts)


def _demo() -> None:
    import sys

    question = " ".join(sys.argv[1:]) or "Is Beam 3A showing signs of fatigue consistent with cyclic loading?"
    print(f"Q: {question}\n")
    print(ask(question, verbose=True))


if __name__ == "__main__":
    _demo()
