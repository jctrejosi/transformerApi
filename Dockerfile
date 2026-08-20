FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

# Render inyecta la variable PORT (por defecto 10000).
# Se usa ${PORT:-8000} para que siga funcionando en local.
CMD ["sh", "-c", "uvicorn apis.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
