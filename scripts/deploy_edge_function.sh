#!/usr/bin/env bash
# Deploy the delete-account Supabase edge function to the hosted project and
# verify it responds.
#
# Requires the Supabase CLI (https://supabase.com/docs/guides/cli) and a logged
# in session (supabase login) plus a linked project (supabase link). Both are
# interactive: they need your Supabase personal access token and DB password.
#
# The function is deployed with gateway-level verify_jwt = true; the backend
# forwards the caller's access token, so only verified users reach the handler.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SUPABASE_DIR="$ROOT_DIR/supabase"

# Prefer an explicit SUPABASE_BIN, then PATH, then the npm user-global prefix.
if [[ -n "${SUPABASE_BIN:-}" ]]; then
  CLI="$SUPABASE_BIN"
elif command -v supabase >/dev/null 2>&1; then
  CLI="$(command -v supabase)"
elif [[ -x "$HOME/.npm-global/bin/supabase" ]]; then
  CLI="$HOME/.npm-global/bin/supabase"
else
  echo "supabase CLI not found. Install it first, for example:"
  echo "  npm install -g supabase"
  exit 1
fi

# Derive the project ref from SUPABASE_URL in .env (https://<ref>.supabase.co).
if [[ -f "$ROOT_DIR/.env" ]]; then
  while IFS='=' read -r key value || [[ -n "$key" ]]; do
    case "$key" in
      SUPABASE_URL) SUPABASE_URL="${value//\"}" ;;
    esac
  done < "$ROOT_DIR/.env"
fi
if [[ -z "${SUPABASE_URL:-}" ]]; then
  echo "SUPABASE_URL not found in $ROOT_DIR/.env"
  exit 1
fi
PROJECT_REF="$(python3 -c "import re,sys; m=re.match(r'https://([^./]+)\\.supabase\\.co', sys.argv[1]); print(m.group(1) if m else '')" "$SUPABASE_URL")"
if [[ -z "$PROJECT_REF" ]]; then
  echo "Could not parse project ref from SUPABASE_URL=$SUPABASE_URL"
  exit 1
fi

echo "Using supabase CLI: $CLI"
echo "Project ref: $PROJECT_REF"

"$CLI" login

# Confirm the CLI is logged in (throws if not).
"$CLI" projects list >/dev/null

# Link to the project if not linked yet (stores ref in supabase/.temp).
if [[ ! -f "$SUPABASE_DIR/.temp/project-ref" ]] || [[ "$(cat "$SUPABASE_DIR/.temp/project-ref")" != "$PROJECT_REF" ]]; then
  echo "Linking project (enter your database password when prompted)..." >&2
  "$CLI" link --project-ref "$PROJECT_REF"
fi

echo "Deploying delete-account edge function..."
"$CLI" functions deploy delete-account --project-ref "$PROJECT_REF"

# Verify the function is live: any HTTP status other than 404 means it exists
# (401/403 for invalid tokens, 405 for bad methods).
echo "Verifying function endpoint..."
HTTP_CODE="$(curl -sS -o /dev/null -w '%{http_code}' -X POST \
  "https://$PROJECT_REF.supabase.co/functions/v1/delete-account" \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer not-a-real-token' \
  -d '{}' --max-time 30)"
echo "HTTP $HTTP_CODE from /functions/v1/delete-account"
if [[ "$HTTP_CODE" == "404" ]]; then
  echo "ERROR: endpoint still returns 404; deployment did not take effect." >&2
  exit 1
fi

echo "delete-account edge function is deployed."
