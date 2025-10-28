#!/usr/bin/env bash
set -euo pipefail

# ============================================================
#  Deploy Azure ML Online Endpoint + Green Deployment
#  - Loads .env locally (skipped in GitHub Actions)
#  - Idempotent create/update
#  - --dry-run prints the plan and exits
#  - Env vars required:
#      AZURE_RESOURCE_GROUP, AZURE_SUBSCRIPTION_ID,
#      MODEL_ENDPOINT_NAME, BEST_MODEL_REGISTRY_NAME, AZURE_UAMI_NAME
#  - Optional:
#      AZURE_ML_WORKSPACE  (workspace name, if not already defaulted in CLI)
# ============================================================

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
  echo "🔹 Loading environment from .env"
  # shellcheck disable=SC1091
  set -a; source .env; set +a
fi

# -----------------------------
# Required env vars
# -----------------------------
need() { [[ -n "${!1:-}" ]] || { echo "❌ Missing env: $1" >&2; exit 1; }; }
need AZURE_RESOURCE_GROUP
need AZURE_SUBSCRIPTION_ID
need MODEL_ENDPOINT_NAME
need BEST_MODEL_REGISTRY_NAME
need AZURE_UAMI_NAME

# Optional: workspace (if az defaults already set, you can omit)
WORKSPACE_NAME="${AZURE_ML_WORKSPACE_NAME:-}"

# Constants / paths
DEPLOYMENT_NAME="green"
ENDPOINT_YML="infrastructure/azure/endpoints/endpoint.yml"
DEPLOYMENT_YML="infrastructure/azure/endpoints/green/deployment-green.yml"
UAMI_RESOURCE_ID="/subscriptions/${AZURE_SUBSCRIPTION_ID}/resourceGroups/${AZURE_RESOURCE_GROUP}/providers/Microsoft.ManagedIdentity/userAssignedIdentities/${AZURE_UAMI_NAME}"

# -----------------------------
# Azure sanity
# -----------------------------
command -v az >/dev/null || { echo "❌ az not found"; exit 1; }
if [[ -z "${GITHUB_ACTIONS:-}" ]]; then
  az account show >/dev/null 2>&1 || { echo "❌ Not logged in (run 'az login')"; exit 1; }
fi

# Helper to append workspace args when provided
# Initialize workspace args BEFORE any use to avoid 'unbound variable'
ws_args=()
if [[ -n "$WORKSPACE_NAME" ]]; then
  ws_args=(--workspace-name "$WORKSPACE_NAME")
fi

# -----------------------------
# Resolve numeric "latest" model version
# -----------------------------
echo "🔎 Resolving latest version for model: ${BEST_MODEL_REGISTRY_NAME}"
# If the model lives in the workspace registry (normal case):
MODEL_VERSION="$(az ml model list \
  -g "$AZURE_RESOURCE_GROUP" \
  "${ws_args[@]}" \
  -n "$BEST_MODEL_REGISTRY_NAME" \
  --query "max_by(@, &to_number(version)).version" -o tsv || true)"

if [[ -z "$MODEL_VERSION" ]]; then
  echo "❌ Could not resolve latest version for model '$BEST_MODEL_REGISTRY_NAME'."
  echo "   Ensure the model exists in workspace '${WORKSPACE_NAME:-<default>}'"
  exit 1
fi

MODEL_REF="azureml:${BEST_MODEL_REGISTRY_NAME}:${MODEL_VERSION}"

echo "📄 Specs:"
echo "  • $ENDPOINT_YML"
echo "  • $DEPLOYMENT_YML"
echo "🔐 UAMI:   $UAMI_RESOURCE_ID"
echo "🎯 Endpoint: $MODEL_ENDPOINT_NAME    Deployment: $DEPLOYMENT_NAME"
echo "📦 Model:  $MODEL_REF (resolved from @latest)"
echo

# -----------------------------
# Dry-run: show intent/state
# -----------------------------
if [[ "$DRY_RUN" == "true" ]]; then
  echo "🧪 DRY-RUN: no changes will be applied."
  if az ml online-endpoint show -n "$MODEL_ENDPOINT_NAME" -g "$AZURE_RESOURCE_GROUP" "${ws_args[@]}" >/dev/null 2>&1; then
    echo "  • Endpoint exists"
  else
    echo "  • Endpoint will be created"
  fi
  if az ml online-deployment show -e "$MODEL_ENDPOINT_NAME" -n "$DEPLOYMENT_NAME" -g "$AZURE_RESOURCE_GROUP" "${ws_args[@]}" >/dev/null 2>&1; then
    echo "  • Deployment exists"
  else
    echo "  • Deployment will be created"
  fi
  echo "  • Identity to attach: $UAMI_RESOURCE_ID"
  echo "  • Model to use:       $MODEL_REF"
  exit 0
fi

# -----------------------------
# Create/Update endpoint (override name + identity)
# -----------------------------
echo "🚀 Ensuring endpoint exists: $MODEL_ENDPOINT_NAME"
if az ml online-endpoint show -n "$MODEL_ENDPOINT_NAME" -g "$AZURE_RESOURCE_GROUP" "${ws_args[@]}" >/dev/null 2>&1; then
  echo "ℹ️  Endpoint exists → skipping all modifications (no update attempted)"
else
  echo "ℹ️  Endpoint not found → creating"
  az ml online-endpoint create \
    -f "$ENDPOINT_YML" \
    -g "$AZURE_RESOURCE_GROUP" \
    "${ws_args[@]}" \
    --set name="$MODEL_ENDPOINT_NAME" \
          identity.type=user_assigned \
          identity.user_assigned_identities[0].resource_id="$UAMI_RESOURCE_ID" \
    --only-show-errors
fi

# -----------------------------
# Create/Update deployment (override endpoint_name + identity + model)
# -----------------------------
echo "📦 Creating/updating deployment: $DEPLOYMENT_NAME"
az ml online-deployment create \
  -f "$DEPLOYMENT_YML" \
  -g "$AZURE_RESOURCE_GROUP" \
  -n "$DEPLOYMENT_NAME" \
  -e "$MODEL_ENDPOINT_NAME" \
  "${ws_args[@]}" \
  --set endpoint_name="$MODEL_ENDPOINT_NAME" \
        model="$MODEL_REF" \
  --only-show-errors

# -----------------------------
# Verification
# -----------------------------
echo "🔎 Verifying endpoint..."
az ml online-endpoint show \
  -n "$MODEL_ENDPOINT_NAME" -g "$AZURE_RESOURCE_GROUP" "${ws_args[@]}" \
  --query "{name:name,auth_mode:auth_mode,provisioning_state:provisioning_state,traffic:traffic}" -o json

echo "🔎 Verifying deployment..."
az ml online-deployment show \
  -e "$MODEL_ENDPOINT_NAME" -n "$DEPLOYMENT_NAME" -g "$AZURE_RESOURCE_GROUP" "${ws_args[@]}" \
  --query "{name:name,provisioning_state:provisioning_state,scaleSettings:scaleSettings,image:image,model:model}" -o json

echo "✅ Green deployment is ready (model ${MODEL_VERSION})."
