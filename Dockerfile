# syntax=docker/dockerfile:1.7
# Backend service: FastAPI + WebSocket + DQN inference.
# Frontend (frontend/) is deployed separately to Vercel.

FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH=/opt/venv/bin:$PATH

COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /usr/local/bin/uv

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY backend ./backend
COPY main.py ./
RUN uv sync --frozen --no-dev

# Telemetry DB defaults to /data so a mounted Railway volume persists it
# across deploys (override with RL_FILLER_DB to relocate).
RUN mkdir -p /data
ENV PORT=8000 \
    RL_FILLER_DB=/data/telemetry.db
EXPOSE 8000

CMD ["sh", "-c", "uv run uvicorn backend.api:app --host 0.0.0.0 --port ${PORT}"]
