#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

services=(redis postgres qdrant neo4j backend ui)

echo "== Compose services =="
docker compose ps

echo
echo "== Container health/status =="
for service in "${services[@]}"; do
  cid="$(docker compose ps -q "$service" || true)"
  if [[ -z "$cid" ]]; then
    echo "$service: not-created"
    continue
  fi

  status="$(docker inspect --format '{{.State.Status}}' "$cid")"
  health="$(docker inspect --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}n/a{{end}}' "$cid")"
  echo "$service: status=$status health=$health"
done

echo
echo "== HTTP probes =="
backend_url="http://localhost:${BACKEND_PORT:-8000}/health"
ui_url="http://localhost:${UI_PORT:-3000}/"

if curl -fsS "$backend_url" >/dev/null; then
  echo "backend probe: ok ($backend_url)"
else
  echo "backend probe: failed ($backend_url)"
fi

if curl -fsS "$ui_url" >/dev/null; then
  echo "ui probe: ok ($ui_url)"
else
  echo "ui probe: failed ($ui_url)"
fi

echo
echo "Tip: if you see orphan warnings, run: docker compose down --remove-orphans"
