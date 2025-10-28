#!/usr/bin/env bash
set -euo pipefail

# -----------------------------
# Flags
# -----------------------------
DRY_RUN="false"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run|-n) DRY_RUN="true"; shift ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

# -----------------------------
# Env loading (local only)
# -----------------------------
if [[ -z "${GITHUB_ACTIONS:-}" && -f .env ]]; then
  # shellcheck disable=SC1091
  set -a; source .env; set +a
fi

# -----------------------------
# Required env vars
# -----------------------------
req() { [[ -n "${!1:-}" ]] || { echo "❌ Missing env: $1" >&2; exit 1; }; }
req AZURE_RESOURCE_GROUP
req MODEL_ENDPOINT_NAME

# Optional: workspace (if az defaults already set, you can omit)
WORKSPACE_NAME="${AZURE_ML_WORKSPACE_NAME:-}"

DEPLOYMENT_NAME="green"  # fixed for this script
AUTOSCALE_TEMPLATE_PATH="${AUTOSCALE_TEMPLATE_PATH:-infrastructure/azure/scale/autoscale.json}"

# Use AZURE_LOCATION instead of LOCATION; derive from target if not provided
AZURE_LOCATION="${AZURE_LOCATION:-}"

# Default tags come from the script (can be overridden via TAGS_JSON)
TAGS_JSON_DEFAULT='{"env":"uber-eats","service":"ml-inference","owner":"ahmedshahriar"}'
TAGS_JSON="${TAGS_JSON:-$TAGS_JSON_DEFAULT}"

# -----------------------------
# Azure CLI sanity checks
# -----------------------------
command -v az >/dev/null || { echo "❌ Azure CLI (az) not found"; exit 1; }
if [[ -z "${GITHUB_ACTIONS:-}" ]]; then
  az account show >/dev/null 2>&1 || { echo "❌ Not logged in. Run: az login"; exit 1; }
fi

# Helper to append workspace args when provided
# Initialize workspace args BEFORE any use to avoid 'unbound variable'
ws_args=()
if [[ -n "$WORKSPACE_NAME" ]]; then
  ws_args=(--workspace-name "$WORKSPACE_NAME")
fi

# -----------------------------
# Resolve target deployment & location
# -----------------------------
echo "🔍 Resolving ML online deployment..."
DEPLOYMENT_ID="$(az ml online-deployment show \
  --endpoint "$MODEL_ENDPOINT_NAME" \
  --name "$DEPLOYMENT_NAME" \
  -g "$AZURE_RESOURCE_GROUP" "${ws_args[@]}" \
  --query id -o tsv)"

[[ -n "$DEPLOYMENT_ID" ]] || { echo "❌ Could not resolve deployment id"; exit 1; }

if [[ -z "$AZURE_LOCATION" ]]; then
  AZURE_LOCATION="$(az resource show --ids "$DEPLOYMENT_ID" --query location -o tsv || true)"
  [[ -n "$AZURE_LOCATION" ]] || { echo "❌ Could not derive location; set AZURE_LOCATION env"; exit 1; }
fi

# -----------------------------
# Compose autoscale name (endpoint-deployment-autoscale)
# -----------------------------
AUTOSCALE_NAME="${MODEL_ENDPOINT_NAME}-${DEPLOYMENT_NAME}-autoscale"
AUTOSCALE_NAME="${AUTOSCALE_NAME// /-}"

echo "✅ Target deployment: $DEPLOYMENT_ID"
echo "📍 Location: $AZURE_LOCATION"
echo "🏷️  Tags: ${TAGS_JSON}"
echo "🧩 Autoscale resource name: $AUTOSCALE_NAME"
echo "📄 Template: $AUTOSCALE_TEMPLATE_PATH"

# -----------------------------
# What-if
# -----------------------------
echo "🧪 What-if preview..."
az deployment group what-if \
  --resource-group "$AZURE_RESOURCE_GROUP" \
  --template-file "$AUTOSCALE_TEMPLATE_PATH" \
  --parameters targetResourceUri="$DEPLOYMENT_ID" \
               autoscaleName="$AUTOSCALE_NAME" \
               location="$AZURE_LOCATION" \
               tags="$TAGS_JSON" \
  --only-show-errors

# -----------------------------
# Apply
# -----------------------------
if [[ "$DRY_RUN" == "true" ]]; then
  echo "⏩ Dry-run; skipping apply."
  exit 0
fi

echo "🚀 Applying autoscale configuration..."
az deployment group create \
  --resource-group "$AZURE_RESOURCE_GROUP" \
  --template-file "$AUTOSCALE_TEMPLATE_PATH" \
  --parameters targetResourceUri="$DEPLOYMENT_ID" \
               autoscaleName="$AUTOSCALE_NAME" \
               location="$AZURE_LOCATION" \
               tags="$TAGS_JSON" \
  --only-show-errors

# -----------------------------
# Verification (two checks)
# -----------------------------
echo "🔎 Verifying autoscale resource..."
az monitor autoscale show \
  --resource-group "$AZURE_RESOURCE_GROUP" \
  --name "$AUTOSCALE_NAME" \
  --only-show-errors \
  --query "{name:name,enabled:enabled,profiles:length(profiles)}"

echo "🔎 Verifying deployment still healthy..."
az ml online-deployment show \
  --endpoint "$MODEL_ENDPOINT_NAME" \
  --name "$DEPLOYMENT_NAME" \
  --query "{provisioning_state:provisioning_state,traffic_weight:traffic_weight,scaleSettings:scaleSettings}" \
  -o json

echo "✅ Autoscale attached & verified."
