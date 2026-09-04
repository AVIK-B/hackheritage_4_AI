FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-orch.txt pyproject.toml ./
RUN pip install --upgrade pip \
    && pip install -r requirements-orch.txt

COPY src ./src
COPY prompts ./prompts

RUN mkdir -p /data/audit

EXPOSE 8500

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8500/healthz || exit 1

CMD ["uvicorn", "src.orchestration.app:app", "--host", "0.0.0.0", "--port", "8500"]
