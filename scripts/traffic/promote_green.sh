#!/usr/bin/env bash
set -euo pipefail

# Local env load (skip in CI)
if [[ -z "${GITHUB_ACTIONS:-}" && -f .env ]]; then set -a; source .env; set +a; fi

need(){ [[ -n "${!1:-}" ]] || { echo "❌ Missing env: $1" >&2; exit 1; }; }
need AZURE_RESOURCE_GROUP
need MODEL_ENDPOINT_NAME

# Optional workspace
WS_ARGS=()
[[ -n "${AZURE_ML_WORKSPACE_NAME:-}" ]] && WS_ARGS=(--workspace-name "$AZURE_ML_WORKSPACE_NAME")

# Resolve the green deployment's resource ID (what autoscale should target)
GREEN_RESOURCE_ID="$(az ml online-deployment show \
  -e "$MODEL_ENDPOINT_NAME" -n green -g "$AZURE_RESOURCE_GROUP" "${WS_ARGS[@]}" \
  --query id -o tsv 2>/dev/null || true)"

if [[ -z "$GREEN_RESOURCE_ID" ]]; then
  echo "❌ Green deployment not found; cannot verify/attach autoscale."
  exit 1
fi

AS_NAME="${MODEL_ENDPOINT_NAME}-green-autoscale"
attached=$(
  az monitor autoscale list -g "$AZURE_RESOURCE_GROUP" \
    --query "[?name=='$AS_NAME' && to_string(enabled)=='true' && targetResourceUri=='$GREEN_RESOURCE_ID'] | length(@)" \
    -o tsv 2>/dev/null || echo 0
)

if [[ "${attached:-0}" -eq 0 ]]; then
  echo "ℹ️ Re-attaching autoscale to green..."
  poetry poe apply-autoscale-green
else
  echo "ℹ️ Autoscale already attached to green."
fi

# Smoke before cutover
poetry poe smoke-green  # uses existing green smoke runner

# 100% → green (adjust to canary if desired)
GREEN=100
BLUE=0

echo "🔁 Setting traffic: green=${GREEN}% blue=${BLUE}%"
az ml online-endpoint update \
  -n "$MODEL_ENDPOINT_NAME" -g "$AZURE_RESOURCE_GROUP" "${WS_ARGS[@]}" \
  --traffic "green=${GREEN} blue=${BLUE}" --only-show-errors

# Show final state
bash ./scripts/traffic/show_traffic.sh
