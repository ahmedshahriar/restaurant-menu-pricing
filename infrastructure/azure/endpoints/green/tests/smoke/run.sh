#!/usr/bin/env bash
set -euo pipefail

# ============================================================
#  Smoke test for Azure ML Online Endpoint (Green deployment)
# ============================================================

# Load env vars (local only)
if [[ -z "${GITHUB_ACTIONS:-}" && -f .env ]]; then
  echo "🔹 Loading environment from .env"
  # shellcheck disable=SC1091
  set -a; source .env; set +a
fi

# Required env vars
req() { [[ -n "${!1:-}" ]] || { echo "❌ Missing env: $1" >&2; exit 1; }; }
req MODEL_ENDPOINT_NAME
req AZURE_RESOURCE_GROUP

# Optional parameters
MAX_LATENCY_MS="${SMOKE_MAX_LATENCY_MS:-4000}"
AZ_OUTPUT="${AZ_OUTPUT:-json}"
VERBOSE="${VERBOSE:-0}"

# Files to test
FILES=(
  "infrastructure/azure/endpoints/green/tests/smoke/payload.json"
  "infrastructure/azure/endpoints/green/samples/serving_input_example.json"
)

command -v az >/dev/null || { echo "❌ Azure CLI not found"; exit 1; }
HAS_JQ=1
command -v jq >/dev/null || HAS_JQ=0

# ============================================================
#  Function to test one payload file
# ============================================================
test_payload() {
  local req_file="$1"

  if [[ ! -f "$req_file" ]]; then
    echo "⚠️  Skipping missing file: $req_file"
    return
  fi

  echo "🚀 Invoking endpoint [$MODEL_ENDPOINT_NAME] with payload: $req_file"

  local start_ms end_ms latency_ms resp status
  start_ms=$(($(date +%s)*1000))
  set +e
  resp="$(az ml online-endpoint invoke \
    -n "$MODEL_ENDPOINT_NAME" \
    --deployment green \
    --request-file "$req_file" \
    --output "$AZ_OUTPUT" 2>&1)"
  status=$?
  set -e
  end_ms=$(($(date +%s)*1000))
  latency_ms=$(( end_ms - start_ms ))

  [[ "$VERBOSE" == "1" ]] && echo "↩️  Response: $resp"

  if [[ $status -ne 0 ]]; then
    echo "❌ Invocation failed (exit $status)"
    echo "$resp"
    exit $status
  fi

  if [[ -z "$resp" ]]; then
    echo "❌ Empty response"
    exit 1
  fi

  if [[ $HAS_JQ -eq 1 ]]; then
    echo "$resp" | jq -e . >/dev/null || { echo "❌ Response is not valid JSON"; exit 1; }
  fi

  if (( latency_ms > MAX_LATENCY_MS )); then
    echo "❌ Latency ${latency_ms}ms exceeds threshold ${MAX_LATENCY_MS}ms"
    exit 1
  fi

  echo "✅ Smoke test passed for $req_file (latency=${latency_ms}ms)"
}

# ============================================================
#  Run smoke tests
# ============================================================
echo "🧪 Starting smoke tests for endpoint: $MODEL_ENDPOINT_NAME"
for f in "${FILES[@]}"; do
  test_payload "$f"
done

echo "🎉 All smoke tests passed successfully."
