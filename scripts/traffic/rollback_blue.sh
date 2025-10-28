#!/usr/bin/env bash
set -euo pipefail

# Local env load (skip in CI)
if [[ -z "${GITHUB_ACTIONS:-}" && -f .env ]]; then set -a; source .env; set +a; fi

need(){ [[ -n "${!1:-}" ]] || { echo "❌ Missing env: $1" >&2; exit 1; }; }
need AZURE_RESOURCE_GROUP
need MODEL_ENDPOINT_NAME

# Optional workspace args
WS_ARGS=()
[[ -n "${AZURE_ML_WORKSPACE_NAME:-}" ]] && WS_ARGS=(--workspace-name "$AZURE_ML_WORKSPACE_NAME")

# Resolve the blue deployment's resource ID (what autoscale should target)
BLUE_RESOURCE_ID="$(az ml online-deployment show \
  -e "$MODEL_ENDPOINT_NAME" -n blue -g "$AZURE_RESOURCE_GROUP" "${WS_ARGS[@]}" \
  --query id -o tsv 2>/dev/null || true)"

if [[ -z "$BLUE_RESOURCE_ID" ]]; then
  echo "❌ Blue deployment not found; cannot verify/attach autoscale."
  exit 1
fi

# Ensure autoscale for blue is attached (exists, enabled, targets BLUE_RESOURCE_ID)
AS_NAME="${MODEL_ENDPOINT_NAME}-blue-autoscale"
attached=$(
  az monitor autoscale list -g "$AZURE_RESOURCE_GROUP" \
    --query "[?name=='$AS_NAME' && to_string(enabled)=='true' && targetResourceUri=='$BLUE_RESOURCE_ID'] | length(@)" \
    -o tsv 2>/dev/null || echo 0
)

if [[ "${attached:-0}" -eq 0 ]]; then
  echo "ℹ️ Re-attaching autoscale to blue..."
  poetry poe apply-autoscale-blue
else
  echo "ℹ️ Autoscale already attached to blue."
fi

# Rollback to blue 100%
BLUE=100
GREEN=0
echo "🔁 Setting traffic: blue=${BLUE}% green=${GREEN}%"
az ml online-endpoint update \
  -n "$MODEL_ENDPOINT_NAME" -g "$AZURE_RESOURCE_GROUP" "${WS_ARGS[@]}" \
  --traffic "blue=${BLUE} green=${GREEN}" --only-show-errors

bash ./scripts/traffic/show_traffic.sh
