#!/usr/bin/env bash
set -euo pipefail

# Load local env (skip in CI)
if [[ -z "${GITHUB_ACTIONS:-}" && -f .env ]]; then set -a; source .env; set +a; fi

need(){ [[ -n "${!1:-}" ]] || { echo "❌ Missing env: $1" >&2; exit 1; }; }
need AZURE_RESOURCE_GROUP
need MODEL_ENDPOINT_NAME

WS_ARGS=()
[[ -n "${AZURE_ML_WORKSPACE_NAME:-}" ]] && WS_ARGS=(--workspace-name "$AZURE_ML_WORKSPACE_NAME")

echo "=== endpoint traffic ==="
az ml online-endpoint show \
  -n "$MODEL_ENDPOINT_NAME" -g "$AZURE_RESOURCE_GROUP" "${WS_ARGS[@]}" \
  --query "{traffic: traffic, auth_mode: auth_mode}" -o json || true

for d in blue green; do
  echo -e "\n=== $d ==="
  az ml online-deployment show \
    -e "$MODEL_ENDPOINT_NAME" -n "$d" -g "$AZURE_RESOURCE_GROUP" "${WS_ARGS[@]}" \
    --query "{name:name,state:provisioning_state,instance_type:instance_type,instance_count:instance_count}" -o json \
    || echo '{"status":"not-found"}'

  AS_NAME="${MODEL_ENDPOINT_NAME}-${d}-autoscale"

  # Use `list` (flattened shape) and select the one by name.
  AS_QUERY="[?name=='$AS_NAME']|[0].{enabled:enabled,min:profiles[0].capacity.minimum,max:profiles[0].capacity.maximum,target:targetResourceUri}"

  if as_json="$(az monitor autoscale list -g "$AZURE_RESOURCE_GROUP" --query "$AS_QUERY" -o json 2>/dev/null)"; then
    if [[ -n "$as_json" && "$as_json" != "null" ]]; then
      echo -e "\n=== autoscale status ==="
      echo "$as_json"
    else
      echo '{"autoscale":"not-configured"}'
    fi
  else
    echo '{"autoscale":"not-configured"}'
  fi
done
