#!/usr/bin/env bash
set -euo pipefail

DRY_RUN="${DRY_RUN:-0}"   # set DRY_RUN=1 to preview
WAIT_SEC="${WAIT_SEC:-30}"
STEPS_STR="${STEPS:-10 25 50 100}"
SMOKE_CMD="${SMOKE_CMD:-poetry poe smoke-green}"

run() { echo "+ $*"; [[ "$DRY_RUN" -eq 1 ]] || eval "$@"; }
maybe_sleep() { [[ "$DRY_RUN" -eq 1 ]] && echo "[dry-run] sleep $1" || sleep "$1"; }

# Local env load (skip in CI)
if [[ -z "${GITHUB_ACTIONS:-}" && -f .env ]]; then set -a; source .env; set +a; fi

need(){ [[ -n "${!1:-}" ]] || { echo "❌ Missing env: $1" >&2; exit 1; }; }
need AZURE_RESOURCE_GROUP; need MODEL_ENDPOINT_NAME

WS_ARGS=()
[[ -n "${AZURE_ML_WORKSPACE_NAME:-}" ]] && WS_ARGS=(--workspace-name "$AZURE_ML_WORKSPACE_NAME")

# Parse steps
read -r -a STEPS <<< "$STEPS_STR"

# Preflight (read-only)
GREEN_ID="$(az ml online-deployment show -e "$MODEL_ENDPOINT_NAME" -n green -g "$AZURE_RESOURCE_GROUP" "${WS_ARGS[@]}" --query id -o tsv 2>/dev/null || true)"
[[ -n "$GREEN_ID" ]] || { echo "❌ Green deployment not found"; exit 1; }

AS_NAME="${MODEL_ENDPOINT_NAME}-green-autoscale"
attached="$(
  az monitor autoscale list -g "$AZURE_RESOURCE_GROUP" \
    --query "[?name=='$AS_NAME' && enabled==\`true\` && targetResourceUri=='$GREEN_ID'] | length(@)" \
    -o tsv 2>/dev/null || echo 0
)"

BASE_GREEN="$(az ml online-endpoint show -n "$MODEL_ENDPOINT_NAME" -g "$AZURE_RESOURCE_GROUP" "${WS_ARGS[@]}" --query 'traffic.green' -o tsv 2>/dev/null || echo 0)"

echo "=== Canary plan ==="
echo "DRY_RUN=${DRY_RUN} | steps: ${STEPS[*]} | wait: ${WAIT_SEC}s"
echo "green_id: $GREEN_ID"
echo "autoscale attached to green: $([[ "$attached" -gt 0 ]] && echo yes || echo no)"
echo "current traffic: green=${BASE_GREEN}% blue=$((100-${BASE_GREEN}))%"
echo "===================="

# Ensure autoscale attached
if [[ "$attached" -eq 0 ]]; then
  echo "ℹ️ Re-attaching autoscale to green..."
  run poetry poe apply-autoscale-green
fi

# Rollback point
rollback() {
  echo "↩️ Rolling back to green=${BASE_GREEN}% ..."
  run bash ./scripts/traffic/set_traffic.sh "$BASE_GREEN" || true
  run bash ./scripts/traffic/show_traffic.sh || true
}
trap '[[ "$DRY_RUN" -eq 1 ]] || rollback' ERR

# Canary loop
for g in "${STEPS[@]}"; do
  echo "🚦 Step → green=${g}% (blue=$((100-g))%)"
  run bash ./scripts/traffic/set_traffic.sh "$g"
  run bash ./scripts/traffic/show_traffic.sh || true

  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[dry-run] would run: $SMOKE_CMD"
  else
    echo "🧪 Smoke @ ${g}% ..."
    eval "$SMOKE_CMD"
  fi

  [[ "$g" -lt 100 ]] && maybe_sleep "$WAIT_SEC"
done

trap - ERR
echo "✅ Canary complete."
run bash ./scripts/traffic/show_traffic.sh || true
