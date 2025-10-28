#!/usr/bin/env bash
set -euo pipefail

# Load .env locally (skip in CI)
if [[ -z "${GITHUB_ACTIONS:-}" && -f .env ]]; then
  set -a; source .env; set +a
fi

need(){ [[ -n "${!1:-}" ]] || { echo "❌ Missing env: $1" >&2; exit 1; }; }

# Positional override or fallback to env
RESOURCE_GROUP="${1:-${AZURE_RESOURCE_GROUP:-}}"
if [[ -z "${RESOURCE_GROUP}" ]]; then
  echo "❌ Missing resource group name."
  echo "Usage: poetry poe delete-rg [resource-group]"
  exit 1
fi

# Use subscription if provided (optional)
if [[ -n "${AZURE_SUBSCRIPTION_ID:-}" ]]; then
  az account set --subscription "$AZURE_SUBSCRIPTION_ID" >/dev/null
fi

# No-op if RG not found
if ! az group show --name "$RESOURCE_GROUP" >/dev/null 2>&1; then
  echo "ℹ️ Resource group '$RESOURCE_GROUP' not found — nothing to delete."
  exit 0
fi

echo "🗑️  Deleting resource group: $RESOURCE_GROUP"
az group delete --name "$RESOURCE_GROUP" --yes --no-wait

echo "⏳ Waiting for deletion to complete…"
az group wait --name "$RESOURCE_GROUP" --deleted

echo "✅ Deleted resource group: $RESOURCE_GROUP"
