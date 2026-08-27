# SteelSense AI backend -- see docs/architecture.md for why this is
# containerized. Two stages: build the Python environment in one image,
# then copy only the finished venv + app code into a clean runtime image
# so build tools and pip's cache never end up in what actually ships.

# ---- builder ----
FROM python:3.12-slim AS builder

WORKDIR /app

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ---- runtime ----
FROM python:3.12-slim

# Non-root: the app never needs to write anywhere except its own home
# directory (the RAG layer's embedding-model cache lands under
# ~/.cache/chroma on first use), so it doesn't need root at all.
RUN groupadd --system appuser \
    && useradd --system --gid appuser --create-home --home-dir /home/appuser appuser

COPY --from=builder /opt/venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH" \
    HOME=/home/appuser \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Only what the running API needs -- not the frontend, evals, or docs.
# ingestion.py resolves its data directory relative to its own file
# location (three parents up), so this layout must be preserved as-is.
COPY src/__init__.py ./src/__init__.py
COPY src/backend ./src/backend
COPY data/sample ./data/sample

RUN chown -R appuser:appuser /app
USER appuser

# Pre-fetch the RAG layer's ~80MB embedding model at build time instead of
# leaving it to download on first use. Without this, the container's first
# boot takes ~70s of blocking work inside FastAPI's startup event (see
# agent/tools.warm_up) before uvicorn will answer *any* request, including
# /api/health -- easy to mistake for the app being broken. Runs as appuser
# so the cache lands under the same $HOME the app uses at runtime.
RUN python -c "import chromadb; c = chromadb.EphemeralClient(); col = c.create_collection('warmup'); col.add(ids=['1'], documents=['warm up the embedding model'])"

EXPOSE 8000

# Secrets (ANTHROPIC_API_KEY, etc.) are read from the environment at
# runtime by the app itself -- nothing above bakes any secret into a
# layer, and none should ever be added via ARG/ENV here.
#
# Plain uvicorn, no --reload (that's dev-only, not production) and
# deliberately no gunicorn worker pool: the RAG knowledge base and
# analysis caches are built in-process (agent/tools.warm_up), so N
# workers would mean N independent copies of that in-memory state rather
# than anything shared -- not a win at this app's scale. Shell form so
# ${PORT}, which most PaaS platforms (Railway, Render, Heroku-likes)
# inject at runtime, is respected; falls back to 8000 for plain
# `docker run` / compose.
CMD uvicorn src.backend.api.main:app --host 0.0.0.0 --port ${PORT:-8000}
