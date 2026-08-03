#!/usr/bin/env bash
set -euo pipefail

# Audit the pinned backend dependencies for known vulnerabilities.
# Exits non-zero if any vulnerability has an available fix that is not pinned.

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV_PYTHON="$ROOT_DIR/backend/.venv/bin/python"

if [[ ! -x "$VENV_PYTHON" ]]; then
  echo "Python venv not found at $VENV_PYTHON"
  echo "Create it and install dependencies first, for example:"
  echo "  cd backend && python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
  exit 1
fi

if ! "$VENV_PYTHON" -m pip show pip-audit >/dev/null 2>&1; then
  echo "pip-audit not installed in the venv — installing..."
  "$VENV_PYTHON" -m pip install pip-audit
fi

cd "$ROOT_DIR"

echo "Auditing pinned backend dependencies (backend/requirements.txt)..."
"$VENV_PYTHON" -m pip_audit -r backend/requirements.txt

echo "Dependency audit completed."
