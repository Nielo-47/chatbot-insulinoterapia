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
# ones live in .env; provide safe fallbacks so the unit suite runs without
# them (unit tests use stubs, never the real container).
export POCKETBASE_URL="${POCKETBASE_URL:-http://pocketbase:8090}"
export POCKETBASE_SUPERUSER_EMAIL="${POCKETBASE_SUPERUSER_EMAIL:-admin@test.internal}"
export POCKETBASE_SUPERUSER_PASSWORD="${POCKETBASE_SUPERUSER_PASSWORD:-test-password}"

echo "Running backend unit tests..."
"$VENV_PYTHON" -m unittest backend.test.unit.test_auth -v
"$VENV_PYTHON" -m unittest backend.test.unit.test_rate_limit -v
"$VENV_PYTHON" -m unittest backend.test.unit.test_conversation_service -v
"$VENV_PYTHON" -m unittest backend.test.unit.test_query_processor -v

echo "Backend unit tests completed."
