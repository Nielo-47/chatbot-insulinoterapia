#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV_PYTHON="$ROOT_DIR/backend/.venv/bin/python"

if [[ ! -x "$VENV_PYTHON" ]]; then
  echo "Python venv not found at $VENV_PYTHON"
  echo "Create it and install dependencies first, for example:"
  echo "  cd backend && python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
  exit 1
fi

cd "$ROOT_DIR"

# Config modules call require() at import time; load the repo .env so tests
# run without the caller exporting everything by hand. Values already present
# in the environment win over .env.
if [[ -f "$ROOT_DIR/.env" ]]; then
  while IFS='=' read -r key value || [[ -n "$key" ]]; do
    case "$key" in
      ''|\#*) continue ;;
    esac
    if [[ -z "${!key+x}" ]]; then
      export "$key=$value"
    fi
  done < "$ROOT_DIR/.env"
fi

# PocketBase values are read at import time by the config modules. The real
# ones live in .env; provide safe fallbacks so the suite runs without the real
# container (integration tests use in-memory fakes and HS256 test tokens).
export POCKETBASE_URL="${POCKETBASE_URL:-http://pocketbase:8090}"
export POCKETBASE_SUPERUSER_EMAIL="${POCKETBASE_SUPERUSER_EMAIL:-admin@test.internal}"
export POCKETBASE_SUPERUSER_PASSWORD="${POCKETBASE_SUPERUSER_PASSWORD:-test-password}"

echo "Running backend integration tests (no external services required)..."
"$VENV_PYTHON" -m unittest backend.test.integration.test_conversation_cache -v
"$VENV_PYTHON" -m unittest backend.test.integration.test_repository_integration -v
"$VENV_PYTHON" -m unittest backend.test.integration.test_conversation_service_integration -v
"$VENV_PYTHON" -m unittest backend.test.integration.test_api_endpoints -v

echo "Backend integration tests completed."
