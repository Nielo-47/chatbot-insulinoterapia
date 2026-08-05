"""Authentik integration configuration.

All authentication is delegated to Authentik (forward-auth proxy). The backend
never sees passwords or issues tokens; it only consumes the identity headers
that the trusted nginx proxy injects from Authentik's auth subrequest.
"""

from .env import get_str

# Public base URL of the Authentik instance (e.g. https://auth.example.com).
# Used by the Authentik Admin API client for account deletion. Optional at
# import time so the backend can boot without it; when empty, account deletion
# fails closed (502).
AUTHENTIK_BASE_URL = get_str("AUTHENTIK_BASE_URL", "")

# The app's public URL as registered on the Authentik outpost provider
# (external_host). Used to build redirects back to the app.
AUTHENTIK_APP_EXTERNAL_HOST = get_str("AUTHENTIK_APP_EXTERNAL_HOST", "")

# Service-account token for the Authentik Admin API (account deletion only).
# Optional at import time so the backend can boot without it, but account
# deletion fails closed (502) when it is not configured.
AUTHENTIK_ADMIN_API_TOKEN = get_str("AUTHENTIK_ADMIN_API_TOKEN", "")
