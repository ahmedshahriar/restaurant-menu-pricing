#!/usr/bin/env bash
set -euo pipefail

# Load .env locally (skip in CI)
if [[ -z "${GITHUB_ACTIONS:-}" && -f .env ]]; then
  set -a; source .env; set +a
fi

need(){ [[ -n "${!1:-}" ]] || { echo "❌ Missing env: $1" >&2; exit 1; }; }
need AZURE_RESOURCE_GROUP
need MODEL_ENDPOINT_NAME

# Show the endpoint scoring URI
az ml online-endpoint show \
  -n "$MODEL_ENDPOINT_NAME" \
  -g "$AZURE_RESOURCE_GROUP" \
  --query scoring_uri -o tsv
