# syntax=docker/dockerfile:1
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=.,target=/app-disk \
    export TMPDIR=/app-disk && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "apis.main:app", "--host", "0.0.0.0", "--port", "8000"]