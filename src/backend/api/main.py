"""FastAPI application for SteelSense AI.

A thin HTTP layer over the pipeline built in earlier stages: sensor
metadata/readings and anomaly detection (src/backend/data) back the
dashboard's chart panel, and the tool-calling agent (src/backend/agent)
backs the chat panel. No business logic lives here -- routes validate
input, call into those modules, and shape the response.
"""

from __future__ import annotations

import os

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ..agent.agent import ask
from ..data.analysis import analyze_all_sensors, analyze_sensor
from ..data.ingestion import get_sensor_readings, list_sensors, resolve_sensor_id

load_dotenv()  # picks up ANTHROPIC_API_KEY etc. from a local .env for `ask()`, called per-request

app = FastAPI(
    title="SteelSense AI API",
    description=(
        "LLM-grounded diagnostic API over synthetic structural health "
        "monitoring data. Portfolio project -- not a real deployment; all "
        "data is synthetic."
    ),
    version="0.1.0",
)

_DEFAULT_ORIGINS = "http://localhost:5173,http://localhost:3000"
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("ALLOWED_ORIGINS", _DEFAULT_ORIGINS).split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)


def _resolve_or_404(sensor_id: str) -> str:
    resolved = resolve_sensor_id(sensor_id)
    if resolved is None:
        raise HTTPException(status_code=404, detail=f"Unknown sensor: {sensor_id!r}")
    return resolved


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str


class SensorInfo(BaseModel):
    sensor_id: str
    location: str
    description: str
    baseline_mv: float


class ReadingPoint(BaseModel):
    timestamp: str
    reading_mv: float
    temperature_c: float


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/sensors", response_model=list[SensorInfo])
def get_sensors() -> list[dict]:
    return list_sensors()


@app.get("/api/sensors/{sensor_id}/readings", response_model=list[ReadingPoint])
def get_readings(sensor_id: str, days: int | None = None) -> list[dict]:
    resolved = _resolve_or_404(sensor_id)
    df = get_sensor_readings(resolved)
    if days is not None:
        cutoff = df["timestamp"].max() - pd.Timedelta(days=days)
        df = df[df["timestamp"] >= cutoff]
    return [
        {
            "timestamp": row.timestamp.isoformat(),
            "reading_mv": row.reading_mv,
            "temperature_c": row.temperature_c,
        }
        for row in df.itertuples()
    ]


@app.get("/api/sensors/{sensor_id}/analysis")
def get_sensor_analysis(sensor_id: str) -> dict:
    return analyze_sensor(_resolve_or_404(sensor_id))


@app.get("/api/analysis")
def get_all_analysis() -> dict:
    return analyze_all_sensors()


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> dict:
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")
    try:
        answer = ask(request.question)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"agent error: {exc}") from exc
    return {"answer": answer}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.backend.api.main:app", host="0.0.0.0", port=8000, reload=True)
