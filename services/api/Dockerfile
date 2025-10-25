FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl build-essential \
  && rm -rf /var/lib/apt/lists/* && useradd -m -u 10001 appuser

WORKDIR /app

# Poetry 2 (PEP 621 support)
RUN pip install --upgrade pip && pip install "poetry>=2,<3"

# Copy manifests only (better layer caching)
COPY services/api/pyproject.toml  ./

# Install dependencies ONLY (no project install yet)
RUN poetry config virtualenvs.create false \
 && poetry install --without dev --no-root --no-interaction --no-ansi

# Now copy the app source
COPY services/api/ /app/

ENV PYTHONPATH=/app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8000/api/v1/healthz || exit 1

USER appuser
CMD ["gunicorn", "app.main:app", "-k", "uvicorn.workers.UvicornWorker", "-c", "gunicorn.conf.py"]
