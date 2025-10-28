#!/usr/bin/env bash
set -euo pipefail
# Local env load (skip in CI)
if [[ -z "${GITHUB_ACTIONS:-}" && -f .env ]]; then set -a; source .env; set +a; fi

# Check required env vars
need(){ [[ -n "${!1:-}" ]] || { echo "❌ Missing env: $1" >&2; exit 1; }; }
need AZURE_RESOURCE_GROUP; need MODEL_ENDPOINT_NAME

usage(){ echo "Usage: $0 GREEN_PERCENT [BLUE_PERCENT]"; exit 2; }
GREEN=${1:-}; [[ -n "$GREEN" ]] || usage
if ! [[ "$GREEN" =~ ^[0-9]+$ && "$GREEN" -ge 0 && "$GREEN" -le 100 ]]; then usage; fi
BLUE=${2:-$((100-GREEN))}
if ! [[ "$BLUE" =~ ^[0-9]+$ && $((GREEN+BLUE)) -eq 100 ]]; then usage; fi

# Optional workspace args
WS_ARGS=(); [[ -n "${AZURE_ML_WORKSPACE_NAME:-}" ]] && WS_ARGS=(--workspace-name "$AZURE_ML_WORKSPACE_NAME")

# Set the traffic split
echo "🔁 Setting traffic: green=${GREEN}% blue=${BLUE}%"
az ml online-endpoint update \
  -n "$MODEL_ENDPOINT_NAME" -g "$AZURE_RESOURCE_GROUP" "${WS_ARGS[@]}" \
  --traffic "green=${GREEN} blue=${BLUE}" --only-show-errors

echo "✅ New split:"
az ml online-endpoint show \
  -n "$MODEL_ENDPOINT_NAME" -g "$AZURE_RESOURCE_GROUP" "${WS_ARGS[@]}" \
  --query traffic -o json
