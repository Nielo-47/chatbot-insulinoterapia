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

echo "Waiting for PostgreSQL to be ready..."
until docker compose exec -T postgres pg_isready -U "$POSTGRES_USER" -d "${POSTGRES_DB:-chatbot}" >/dev/null 2>&1; do
  sleep 1
done

echo "Ensuring dedicated test database exists (${POSTGRES_TEST_DB})..."
docker compose exec -T postgres psql -U "$POSTGRES_USER" -d postgres -c "CREATE DATABASE ${POSTGRES_TEST_DB};" >/dev/null 2>&1 || true

echo "Running backend integration tests..."
TEST_DATABASE_URL="$TEST_DATABASE_URL" \
  "$VENV_PYTHON" -m unittest backend.test.integration.test_db_schema -v
"$VENV_PYTHON" -m unittest backend.test.integration.test_conversation_cache -v
TEST_DATABASE_URL="$TEST_DATABASE_URL" \
  "$VENV_PYTHON" -m unittest backend.test.integration.test_repository_integration -v
TEST_DATABASE_URL="$TEST_DATABASE_URL" \
  "$VENV_PYTHON" -m unittest backend.test.integration.test_conversation_service_integration -v
TEST_DATABASE_URL="$TEST_DATABASE_URL" \
  "$VENV_PYTHON" -m unittest backend.test.integration.test_api_endpoints -v

echo "Backend integration tests completed."
