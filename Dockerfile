# syntax=docker/dockerfile:1.7
FROM python:3.11.9-slim AS app

# Build-time argument for version, default if not supplied
ARG APP_VERSION="0.0.0"

# Label the image with version (to inspect it later)
LABEL org.opencontainers.image.version=${APP_VERSION} \
      org.opencontainers.image.title="restaurant-menu-pricing" \
      org.opencontainers.image.description="UberEats menu price prediction"

# Set up environment vars for Python runtime behaviour
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    POETRY_VERSION=2.2.1 \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1 \
    PATH="/app/.venv/bin:$PATH" \
    LANG=C.UTF-8

# Arguments for non-root user creation
ARG APP_USER=appuser
ARG APP_UID=1000
ARG APP_GID=1000

# Create group and non-root user
RUN groupadd -g ${APP_GID} ${APP_USER} \
    && useradd -m -u ${APP_UID} -g ${APP_GID} -s /bin/bash ${APP_USER}

WORKDIR /app

# Install system dependencies (with caching mount)
RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && apt-get install -y --no-install-recommends \
      build-essential git curl ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN python -m pip install "poetry==${POETRY_VERSION}" && poetry --version
RUN poetry config installer.max-workers 20

# Copy lockfiles first so dependency layer can be cached
COPY pyproject.toml poetry.lock* ./

# Install runtime dependencies (no dev deps, no root)
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/root/.cache/pypoetry \
    poetry install --without dev --no-ansi --no-root

# Copy application source code
COPY . .

# Fix permissions so the non-root user can run things
RUN chown -R ${APP_UID}:${APP_GID} /app

# Switch to non-root user for running the app
USER ${APP_USER}

# Default command (for build success check / override in production)
CMD ["bash","-lc","poetry --version && echo 'Image built successfully. Override CMD to run tasks or services.'"]
