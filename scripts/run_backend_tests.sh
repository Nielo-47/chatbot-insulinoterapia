#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV_PYTHON="$ROOT_DIR/backend/.venv/bin/python"
POSTGRES_USER="${POSTGRES_USER:-chatbot}"
POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-chatbot}"
POSTGRES_PORT="${POSTGRES_PORT:-5432}"
POSTGRES_TEST_DB="${POSTGRES_TEST_DB:-chatbot_test}"
TEST_DATABASE_URL="${TEST_DATABASE_URL:-postgresql+psycopg://${POSTGRES_USER}:${POSTGRES_PASSWORD}@localhost:${POSTGRES_PORT}/${POSTGRES_TEST_DB}}"

if [[ ! -x "$VENV_PYTHON" ]]; then
  echo "Python venv not found at $VENV_PYTHON"
  echo "Create it and install dependencies first, for example:"
  echo "  cd backend && python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
  exit 1
fi

cd "$ROOT_DIR"

echo "Starting PostgreSQL container for integration tests..."
docker compose up -d postgres >/dev/null

echo "Ensuring dedicated test database exists (${POSTGRES_TEST_DB})..."
docker compose exec -T postgres psql -U "$POSTGRES_USER" -d postgres -c "CREATE DATABASE ${POSTGRES_TEST_DB};" >/dev/null 2>&1 || true

echo "Running schema integration tests (PostgreSQL container)..."
TEST_DATABASE_URL="$TEST_DATABASE_URL" \
  "$VENV_PYTHON" -m unittest backend.test.test_db_schema -v

echo "Running cache adapter tests..."
"$VENV_PYTHON" -m unittest backend.test.test_conversation_cache -v

echo "Running repository integration tests..."
TEST_DATABASE_URL="$TEST_DATABASE_URL" \
  "$VENV_PYTHON" -m unittest backend.test.test_repository_integration -v

echo "Running service integration tests..."
TEST_DATABASE_URL="$TEST_DATABASE_URL" \
  "$VENV_PYTHON" -m unittest backend.test.test_conversation_service_integration -v

echo "Running API endpoint tests..."
TEST_DATABASE_URL="$TEST_DATABASE_URL" \
  "$VENV_PYTHON" -m unittest backend.test.test_api_endpoints -v

echo "Running conversation service unit tests..."
"$VENV_PYTHON" -m unittest backend.test.test_conversation_service -v

echo "All backend tests completed."
