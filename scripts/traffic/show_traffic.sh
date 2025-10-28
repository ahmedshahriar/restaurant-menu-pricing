#!/usr/bin/env bash
set -euo pipefail

# Local env load (skip in CI)
if [[ -z "${GITHUB_ACTIONS:-}" && -f .env ]]; then
  set -a; source .env; set +a
fi

need() { [[ -n "${!1:-}" ]] || { echo "❌ Missing env: $1" >&2; exit 1; }; }

need AZURE_RESOURCE_GROUP
need MODEL_ENDPOINT_NAME
# optional workspace
WS_ARGS=()
if [[ -n "${AZURE_ML_WORKSPACE_NAME:-}" ]]; then
  WS_ARGS=(--workspace-name "$AZURE_ML_WORKSPACE_NAME")
fi

az ml online-endpoint show \
  -n "$MODEL_ENDPOINT_NAME" -g "$AZURE_RESOURCE_GROUP" "${WS_ARGS[@]}" \
  --query traffic -o json
