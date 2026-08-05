"""Authentik Admin API client.

Used only for account deletion (the user's own account, on request). Login,
session and identity handling are delegated to Authentik's forward-auth proxy
and never touch this module.

The client talks to Authentik's REST API with a service-account token. Uses the
stdlib (urllib) so no new runtime dependency is required.
"""

import json
import logging
import urllib.error
import urllib.parse
import urllib.request
from typing import Optional

from backend.src.config.authentik import AUTHENTIK_ADMIN_API_TOKEN, AUTHENTIK_BASE_URL

logger = logging.getLogger(__name__)


class AuthentikError(Exception):
    """Raised when the Authentik Admin API cannot be reached or rejects a call."""


class AuthentikAdminClient:
    def __init__(
        self,
        base_url: Optional[str] = None,
        api_token: Optional[str] = None,
        timeout: int = 10,
    ) -> None:
        self.base_url = (base_url or AUTHENTIK_BASE_URL).rstrip("/")
        self.api_token = api_token if api_token is not None else AUTHENTIK_ADMIN_API_TOKEN
        self.timeout = timeout

    def _request(self, method: str, path: str) -> Optional[dict]:
        if not self.api_token:
            raise AuthentikError("AUTHENTIK_ADMIN_API_TOKEN is not configured")
        url = f"{self.base_url}{path}"
        request = urllib.request.Request(url, method=method)
        request.add_header("Authorization", f"Bearer {self.api_token}")
        request.add_header("Accept", "application/json")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read()
                return json.loads(raw) if raw else None
        except urllib.error.HTTPError as exc:
            logger.error("Authentik admin API %s %s failed with HTTP %s", method, path, exc.code)
            raise AuthentikError(f"Authentik API returned HTTP {exc.code}") from exc
        except OSError as exc:
            logger.error("Authentik admin API %s %s unreachable: %s", method, path, exc)
            raise AuthentikError(str(exc)) from exc

    def find_user_pk(self, username: str) -> Optional[str]:
        """Return the Authentik user's integer primary key for a username."""
        query = urllib.parse.urlencode({"username": username})
        try:
            data = self._request("GET", f"/api/v3/core/users/?{query}")
        except AuthentikError:
            return None
        if not data:
            return None
        for user in data.get("results", []):
            if user.get("username") == username:
                return str(user.get("pk"))
        return None

    def delete_user(self, username: str) -> bool:
        """Delete an Authentik user by username. Returns False when not found or on API error."""
        pk = self.find_user_pk(username)
        if pk is None:
            return False
        try:
            # Authentik returns 204 No Content on success.
            self._request("DELETE", f"/api/v3/core/users/{pk}/")
            return True
        except AuthentikError:
            return False
