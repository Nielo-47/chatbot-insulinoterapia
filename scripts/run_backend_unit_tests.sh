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

echo "Running backend unit tests..."
"$VENV_PYTHON" -m unittest backend.test.unit.test_auth -v
"$VENV_PYTHON" -m unittest backend.test.unit.test_conversation_service -v
"$VENV_PYTHON" -m unittest backend.test.unit.test_query_processor -v

echo "Backend unit tests completed."
