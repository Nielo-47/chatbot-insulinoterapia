"""PocketBase (auth + database) configuration.

Authentication is delegated to the self-hosted PocketBase container. The
frontend authenticates via pocketbase-js and presents the resulting JWT access
token as a Bearer header. The backend validates that token by forwarding it to
PocketBase's own token-introspection route (see
pb_migrations/1756000003_introspect_route.js) — PocketBase signs auth tokens
with a per-record key component, so verification has to happen inside
PocketBase itself. App data (conversations/messages) lives in PocketBase
collections and is read and written through the superuser API by the
repositories.
"""

from .env import require, get_str

# Internal PocketBase URL (e.g. http://pocketbase:8090 inside the compose
# network). Used for all superuser API calls (data access, account deletion,
# username lookups) and for token introspection.
POCKETBASE_URL = require("POCKETBASE_URL")

# Superuser credentials. The backend owns app-data access, so it holds the
# superuser account (created at first boot by the pocketbase entrypoint).
POCKETBASE_SUPERUSER_EMAIL = require("POCKETBASE_SUPERUSER_EMAIL")
POCKETBASE_SUPERUSER_PASSWORD = require("POCKETBASE_SUPERUSER_PASSWORD")

# Optional request timeout for PocketBase API calls.
POCKETBASE_TIMEOUT_SECONDS = int(get_str("POCKETBASE_TIMEOUT_SECONDS", "10"))
