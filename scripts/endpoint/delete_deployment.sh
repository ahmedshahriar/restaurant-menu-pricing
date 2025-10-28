#!/usr/bin/env bash
set -euo pipefail

# Load .env locally (skip in CI)
if [[ -z "${GITHUB_ACTIONS:-}" && -f .env ]]; then
  set -a; source .env; set +a
fi

need(){ [[ -n "${!1:-}" ]] || { echo "❌ Missing env: $1" >&2; exit 1; }; }

need AZURE_RESOURCE_GROUP
need MODEL_ENDPOINT_NAME

# --- positional argument ---
DEPLOYMENT_NAME="${1:-}"

if [[ -z "$DEPLOYMENT_NAME" ]]; then
  echo "❌ Missing deployment name."
  echo "Usage: poetry poe delete-deployment blue"
  echo "Available options: blue | green"
  exit 1
fi

echo "🗑️  Deleting deployment: $DEPLOYMENT_NAME ..."
az ml online-deployment delete \
  -e "$MODEL_ENDPOINT_NAME" \
  -n "$DEPLOYMENT_NAME" \
  -g "$AZURE_RESOURCE_GROUP" \
  -y
echo "✅ Deployment '$DEPLOYMENT_NAME' deleted."
