#!/bin/sh
# Entrypoint wrapper for the authentik-server / authentik-worker containers.
#
# The ngrok public URL is only known once the tunnel is up, so it cannot be
# interpolated at `docker compose up` time. This wrapper discovers it from the
# ngrok agent's local inspection API (http://ngrok:4040/api/tunnels) and exports
# it as AUTHENTIK_APP_EXTERNAL_HOST before launching the real process, so the
# !Env references in ./blueprints/authentik.yaml resolve.
#
# An explicitly provided AUTHENTIK_APP_EXTERNAL_HOST always wins over discovery.
# If no tunnel URL is found within the retry budget the process still starts, but
# without the variable (the blueprint provider/application/outpost entries then
# stay skipped and are created on a later restart).
#
# Blueprints are only re-applied automatically when their file content changes,
# so on restart with a rotated ngrok URL the provider/outpost would keep the
# stale host. Once the URL is known we therefore force `apply_blueprint` for
# ./blueprints/authentik.yaml; its !Env references resolve to the value above.
set -eu

if [ -n "${AUTHENTIK_APP_EXTERNAL_HOST:-}" ]; then
    exec dumb-init -- ak "$@"
fi

NGROK_API_URL="${NGROK_API_URL:-http://ngrok:4040}"
MAX_ATTEMPTS="${NGROK_DISCOVERY_MAX_ATTEMPTS:-60}"

URL=""
attempt=0
while [ "$attempt" -lt "$MAX_ATTEMPTS" ]; do
    URL=$(python3 - "$NGROK_API_URL" 2>/dev/null <<'PY' || true
import json
import sys
import urllib.request

try:
    with urllib.request.urlopen(sys.argv[1] + "/api/tunnels", timeout=3) as resp:
        data = json.load(resp)
except Exception:
    raise SystemExit(1)

for tunnel in data.get("tunnels", []):
    public_url = tunnel.get("public_url", "")
    if public_url.startswith("https://"):
        sys.stdout.write(public_url.rstrip("/"))
        raise SystemExit(0)
raise SystemExit(1)
PY
)
    [ -n "$URL" ] && break
    attempt=$((attempt + 1))
    sleep 2
done

if [ -n "$URL" ]; then
    echo "AUTHENTIK_APP_EXTERNAL_HOST=$URL" >&2
    export AUTHENTIK_APP_EXTERNAL_HOST="$URL"
    # Force the blueprint re-apply so the provider/outpost hosts follow the
    # tunnel. Best-effort: on first boot the migrations may not be done yet, in
    # which case the regular startup apply creates the objects instead.
    if ! apply_output=$(ak apply_blueprint custom/authentik.yaml 2>&1); then
        echo "WARNING: forced blueprint re-apply failed: $apply_output" >&2
    fi
else
    echo "WARNING: no ngrok public URL after $MAX_ATTEMPTS attempts; starting without AUTHENTIK_APP_EXTERNAL_HOST" >&2
fi

exec dumb-init -- ak "$@"
