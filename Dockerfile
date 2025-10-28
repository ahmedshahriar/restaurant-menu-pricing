# syntax=docker/dockerfile:1.7
FROM python:3.11.9-slim AS app

# Metadata
ARG BUILD_DATE
ARG VCS_REF
LABEL org.opencontainers.image.title="restaurant-menu-pricing" \
      org.opencontainers.image.description="UberEats menu price prediction" \
      org.opencontainers.image.version="0.1.0" \
      org.opencontainers.image.authors="ahmedshahriar <ahmed.shahriar.sakib@gmail.com>" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.revision="${VCS_REF}"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    POETRY_VERSION=2.2.1 \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1

WORKDIR /app

# OS deps (lean, but enough for builds & xgboost/lightgbm openmp)
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential git curl ca-certificates libgomp1 \
  && rm -rf /var/lib/apt/lists/*

# Poetry
RUN python -m pip install "poetry==${POETRY_VERSION}" \
    && poetry --version
RUN poetry config installer.max-workers 20

# Layer-caching: lockfiles first
COPY pyproject.toml poetry.lock* ./

# Install deps (include dev so `dotenv` CLI exists for your Poe tasks)
RUN poetry install --without dev --no-ansi --no-root --no-cache && \
    rm -rf ~/.cache/pypoetry/cache/ && \
    rm -rf ~/.cache/pypoetry/artifacts/

# Now copy the rest of the code
COPY . .

# Put venv on PATH
ENV PATH="/app/.venv/bin:$PATH"

# Idle by default; exec in to run tasks
CMD ["bash","-lc","sleep infinity"]
