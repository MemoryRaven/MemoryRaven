FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Minimal deps for psycopg/asyncpg + health
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install package
COPY pyproject.toml /app/pyproject.toml
COPY README.md /app/README.md
COPY src /app/src

RUN pip install --no-cache-dir -U pip \
 && pip install --no-cache-dir -e . \
 && pip install --no-cache-dir fastapi uvicorn[standard] asyncpg redis qdrant-client pydantic-settings cachetools

EXPOSE 8088

CMD ["python", "-m", "memory_empire.memory_service.server"]
