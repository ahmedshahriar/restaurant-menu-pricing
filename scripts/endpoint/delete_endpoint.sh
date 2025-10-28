#!/usr/bin/env bash
set -euo pipefail

# Load .env locally (skip in CI)
if [[ -z "${GITHUB_ACTIONS:-}" && -f .env ]]; then
  set -a; source .env; set +a
fi

need(){ [[ -n "${!1:-}" ]] || { echo "❌ Missing env: $1" >&2; exit 1; }; }
need AZURE_RESOURCE_GROUP

# Positional override or fallback to env
ENDPOINT_NAME="${1:-${MODEL_ENDPOINT_NAME:-}}"
if [[ -z "${ENDPOINT_NAME}" ]]; then
  echo "❌ Missing endpoint name."
  echo "Usage: poetry poe delete-endpoint [endpoint-name]"
  exit 1
fi

# No-op if not found
if ! az ml online-endpoint show -n "$ENDPOINT_NAME" -g "$AZURE_RESOURCE_GROUP" >/dev/null 2>&1; then
  echo "ℹ️ Endpoint '$ENDPOINT_NAME' not found in rg '$AZURE_RESOURCE_GROUP' — nothing to delete."
  exit 0
fi

echo "🗑️  Deleting endpoint: $ENDPOINT_NAME (rg=$AZURE_RESOURCE_GROUP)"
az ml online-endpoint delete -n "$ENDPOINT_NAME" -g "$AZURE_RESOURCE_GROUP" -y
echo "✅ Deleted endpoint: $ENDPOINT_NAME"
