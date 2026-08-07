#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV_PYTHON="$ROOT_DIR/backend/.venv/bin/python"
POSTGRES_USER="${POSTGRES_USER:-chatbot}"
POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-chatbot}"
POSTGRES_PORT="${POSTGRES_PORT:-5432}"
POSTGRES_TEST_DB="${POSTGRES_TEST_DB:-chatbot_test}"
TEST_DATABASE_URL="${TEST_DATABASE_URL:-postgresql+psycopg://${POSTGRES_USER}:${POSTGRES_PASSWORD}@localhost:${POSTGRES_PORT}/${POSTGRES_TEST_DB}}"
TEST_POSTGRES_CONTAINER="diabetes-chatbot-test-postgres"

if [[ ! -x "$VENV_PYTHON" ]]; then
  echo "Python venv not found at $VENV_PYTHON"
  echo "Create it and install dependencies first, for example:"
  echo "  cd backend && python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
  exit 1
fi

cd "$ROOT_DIR"

# Config modules call require() at import time; load the repo .env so tests
# run without the caller exporting everything by hand. Values already present
# in the environment (e.g. TEST_DATABASE_URL) win over .env.
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

# Supabase values are read at import time by the config modules. The real ones
# live in .env; provide safe fallbacks so the suite runs without them (tests
# that exercise Supabase use stubs and a test-only JWKS URL).
export SUPABASE_URL="${SUPABASE_URL:-https://test.supabase.co}"
export SUPABASE_JWKS_URL="${SUPABASE_JWKS_URL:-https://test.supabase.co/auth/v1/.well-known/jwks.json}"

cleanup() {
  docker rm -f "$TEST_POSTGRES_CONTAINER" >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "Starting temporary PostgreSQL container for integration tests..."
docker rm -f "$TEST_POSTGRES_CONTAINER" >/dev/null 2>&1 || true
# The production compose file intentionally does not expose postgres to the
# host; for tests only, spin up a dedicated container bound to loopback.
docker run -d --name "$TEST_POSTGRES_CONTAINER" \
  -p "127.0.0.1:${POSTGRES_PORT}:5432" \
  -e POSTGRES_USER="$POSTGRES_USER" \
  -e POSTGRES_PASSWORD="$POSTGRES_PASSWORD" \
  postgres:16-alpine >/dev/null

echo "Waiting for PostgreSQL to be ready..."
until docker exec "$TEST_POSTGRES_CONTAINER" pg_isready -U "$POSTGRES_USER" -d postgres >/dev/null 2>&1; do
  sleep 1
done

echo "Ensuring dedicated test database exists (${POSTGRES_TEST_DB})..."
docker exec "$TEST_POSTGRES_CONTAINER" psql -U "$POSTGRES_USER" -d postgres -c "CREATE DATABASE ${POSTGRES_TEST_DB};" >/dev/null 2>&1 || true

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
