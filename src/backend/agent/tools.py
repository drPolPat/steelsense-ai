"""Tool definitions and dispatch for the SteelSense AI agent.

Each tool wraps a piece of the pipeline built in earlier stages -- sensor
summaries and anomaly detection (src/backend/data), and domain-knowledge
retrieval (src/backend/rag) -- behind the JSON schema Claude's tool-calling
expects. Tool outputs are the same small structured dictionaries those
modules already produce; nothing here reformats or dumps raw signal data
to the model.
"""

from __future__ import annotations

import json
from functools import lru_cache

from ..data.analysis import analyze_all_sensors, analyze_sensor
from ..data.ingestion import list_sensors as _list_sensors
from ..data.ingestion import resolve_sensor_id, summarize_sensor
from ..rag.retrieval import KnowledgeBase


@lru_cache(maxsize=1)
def _knowledge_base() -> KnowledgeBase:
    return KnowledgeBase()


def warm_up() -> None:
    """Force every lazily-built, cached resource a tool call would otherwise
    build on first use -- the Chroma knowledge base (which downloads its
    embedding model on first use) and the sensor data/analysis caches.
    Intended to be called once at process startup so the first real request
    isn't the one paying for it."""
    _knowledge_base()
    analyze_all_sensors()


def _resolve_or_error(sensor_id: str) -> str:
    resolved = resolve_sensor_id(sensor_id)
    if resolved is None:
        known = ", ".join(s["location"] for s in _list_sensors())
        raise ValueError(f"Unknown sensor {sensor_id!r}. Known sensor locations: {known}")
    return resolved


def list_sensors_tool() -> dict:
    return {
        "sensors": [
            {"sensor_id": s["sensor_id"], "location": s["location"], "description": s["description"]}
            for s in _list_sensors()
        ]
    }


def get_sensor_summary_tool(sensor_id: str, days: int | None = None) -> dict:
    return summarize_sensor(_resolve_or_error(sensor_id), days=days)


def run_anomaly_detection_tool(sensor_id: str | None = None) -> dict:
    if sensor_id:
        return analyze_sensor(_resolve_or_error(sensor_id))
    return analyze_all_sensors()


def search_domain_knowledge_tool(query: str, k: int = 3) -> dict:
    chunks = _knowledge_base().retrieve(query, k=k)
    return {
        "results": [
            {"title": c.title, "category": c.category, "text": c.text, "source_note": c.source_note}
            for c in chunks
        ]
    }


TOOL_DEFINITIONS = [
    {
        "name": "list_sensors",
        "description": (
            "List every sensor location on the structure, with a short "
            "description of where each one is mounted. Call this first if "
            "you're not sure what sensor locations exist."
        ),
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "get_sensor_summary",
        "description": (
            "Get structured summary statistics (mean/std/min/max/latest "
            "reading and a simple linear trend) for one sensor location. "
            "This is uncorrected and includes ambient-temperature effects -- "
            "use run_anomaly_detection for a proper drift/spike/divergence "
            "assessment, not this."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "sensor_id": {
                    "type": "string",
                    "description": "Sensor id or location name, e.g. 'Beam 3A' or 'beam-3a'.",
                },
                "days": {
                    "type": "integer",
                    "description": "Restrict the summary to the last N days. Omit for the full series.",
                },
            },
            "required": ["sensor_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "run_anomaly_detection",
        "description": (
            "Run the thermally-corrected drift / spike / cross-sensor-"
            "divergence detection pipeline -- the source of truth for "
            "whether a sensor is showing an anomaly. Pass a sensor_id to "
            "analyze one location, or omit it to analyze and rank every "
            "sensor by concern score (use that form for 'which sensor is "
            "most concerning' style questions)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "sensor_id": {
                    "type": "string",
                    "description": "Sensor id or location name. Omit to analyze all sensors.",
                },
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "search_domain_knowledge",
        "description": (
            "Retrieve relevant domain-knowledge excerpts (magnetic sensing "
            "principles, steel fatigue mechanisms, SHM standards, and "
            "diagnostic heuristics) to ground an explanation. Always call "
            "this before explaining *why* a pattern matters physically or "
            "what standard practice says to do about it."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural-language search query."},
                "k": {"type": "integer", "description": "Number of results to return (default 3)."},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
]

_HANDLERS = {
    "list_sensors": list_sensors_tool,
    "get_sensor_summary": get_sensor_summary_tool,
    "run_anomaly_detection": run_anomaly_detection_tool,
    "search_domain_knowledge": search_domain_knowledge_tool,
}


def execute_tool(name: str, tool_input: dict) -> tuple[str, bool]:
    """Dispatch a tool call by name. Returns (json_result_str, is_error)."""
    handler = _HANDLERS.get(name)
    if handler is None:
        return json.dumps({"error": f"Unknown tool: {name}"}), True
    try:
        result = handler(**tool_input)
    except Exception as exc:
        return json.dumps({"error": str(exc)}), True
    return json.dumps(result), False
