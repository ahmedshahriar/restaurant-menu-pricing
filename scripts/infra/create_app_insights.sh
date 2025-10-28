#!/usr/bin/env bash
set -euo pipefail

: "${AZURE_RESOURCE_GROUP:?Set AZURE_RESOURCE_GROUP}"
: "${AZURE_ML_WORKSPACE_NAME:?Set AZURE_ML_WORKSPACE_NAME}"

RG="$AZURE_RESOURCE_GROUP"
WS="$AZURE_ML_WORKSPACE_NAME"
SUB="$(az account show --query id -o tsv)"

echo "🔎 Reading AML workspace location..."
REGION="$(az ml workspace show -g "$RG" -n "$WS" --query location -o tsv)"


if [[ -z "${REGION}" ]]; then
  echo "❌ Could not determine workspace region. Check RG/WS names and your access."
  exit 1
fi
echo "✅ Workspace region: $REGION"

echo "🧩 Ensuring resource provider Microsoft.Insights is registered..."
if [[ "$(az provider show -n Microsoft.Insights --query registrationState -o tsv)" != "Registered" ]]; then
  az provider register -n Microsoft.Insights --wait
fi
echo "✅ Provider registered."

# Derive names (adjust if needed)
LAW_NAME="${WS}-${REGION}-log"
AI_NAME="${WS}-${REGION}-ai" # avoid collisions; or: "${WS}-${REGION}-ai"

echo "🔎 Checking Log Analytics workspace: $LAW_NAME"
if ! az monitor log-analytics workspace show -g "$RG" -n "$LAW_NAME" >/dev/null 2>&1; then
  echo "➕ Creating Log Analytics workspace: $LAW_NAME in $REGION"
  az monitor log-analytics workspace create -g "$RG" -n "$LAW_NAME" -l "$REGION" >/dev/null
else
  echo "✅ Log Analytics exists."
fi

LAW_ID="$(az monitor log-analytics workspace show -g "$RG" -n "$LAW_NAME" --query id -o tsv)"

# Try to use any existing AI already linked to this LAW to avoid duplicates
EXISTING_AI_ID="$(az monitor app-insights component show -g "$RG" -a "$AI_NAME" --query id -o tsv 2>/dev/null || true)"

if [[ -z "${EXISTING_AI_ID}" ]]; then
  echo "🔎 Looking for an existing App Insights in RG linked to this LAW..."
  EXISTING_AI_ID="$(az monitor app-insights component list -g "$RG" \
    --query "[?workspaceResourceId=='${LAW_ID}'].id | [0]" -o tsv || true)"
fi

if [[ -z "${EXISTING_AI_ID}" ]]; then
  echo "➕ Creating Application Insights: $AI_NAME in $REGION (workspace-based)"
  az monitor app-insights component create \
    -g "$RG" -a "$AI_NAME" -l "$REGION" \
    --kind web \
    --workspace "$LAW_ID" >/dev/null
  AI_ID="/subscriptions/$SUB/resourceGroups/$RG/providers/Microsoft.Insights/components/$AI_NAME"
else
  echo "✅ Reusing existing Application Insights: $EXISTING_AI_ID"
  AI_ID="$EXISTING_AI_ID"
fi

echo "🔗 Attaching Application Insights to AML workspace..."
# Preferred: ML CLI update
if az ml workspace update -g "$RG" -w "$WS" --set "application_insights=$AI_ID" >/dev/null 2>&1; then
  echo "✅ Workspace updated via az ml."
else
  echo "ℹ️  Falling back to generic ARM patch..."
  az resource update \
    --ids "/subscriptions/${SUB}/resourceGroups/${RG}/providers/Microsoft.MachineLearningServices/workspaces/${WS}" \
    --set "properties.applicationInsights=$AI_ID" >/dev/null
  echo "✅ Workspace updated via az resource."
fi

echo "🧪 Verifying linkage..."
LINKED_AI_ID="$(az ml workspace show -g "$RG" -w "$WS" --query application_insights -o tsv)"
if [[ "$LINKED_AI_ID" == "$AI_ID" && -n "$LINKED_AI_ID" ]]; then
  echo "🎉 Done. AML workspace is now linked to Application Insights:"
  echo " $LINKED_AI_ID"
else
  echo "❌ Verification failed. Expected $AI_ID but workspace shows: $LINKED_AI_ID"
  exit 1
fi
