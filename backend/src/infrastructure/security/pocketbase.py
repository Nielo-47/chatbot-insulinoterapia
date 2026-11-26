"""PocketBase Auth integration.

Two distinct responsibilities live here:

1. ``verify_access_token`` — validation of the access token the frontend
   presents as a Bearer header. PocketBase signs auth tokens with
   (record.tokenKey + collection secret), so they cannot be verified offline
   with shared key material. Instead the token is forwarded to PocketBase's
   own introspection route (``/api/linachat/token-introspect``, registered by
   the 1756000003 migration): PocketBase validates it with its own keys —
   including expiry and revocation via tokenKey changes — and answers with
   the authenticated record's id/email, or 401 for any invalid token.

2. ``PocketBaseAccountDeletionClient`` — revokes a user by deleting their
   ``users`` record via the superuser API. CascadeDelete relations remove the
   user's conversations/messages in the same operation. Returns False on any
   failure so callers can fail closed.

``PocketBaseUserDirectory`` resolves the cosmetic display username from the
user record as a fallback (the introspection response usually already carries
the email), with a small TTL cache to avoid a lookup per request.
"""

import json
import logging
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Optional

from backend.src.config.pocketbase import POCKETBASE_TIMEOUT_SECONDS, POCKETBASE_URL
from backend.src.infrastructure.pocketbase import get_pocketbase_client

logger = logging.getLogger(__name__)

USERS_COLLECTION = "users"

_INTROSPECTION_PATH = "/api/linachat/token-introspect"


def _fallback_username(user_id: str) -> str:
    # Derive a stable cosmetic name from the record id when no email exists.
    return f"user_{user_id[:8]}" if user_id else "user"


class PocketBaseTokenError(Exception):
    """Raised when a presented access token cannot be verified."""


def verify_access_token(token: str) -> Dict[str, Any]:
    """Validate the presented PocketBase token via PocketBase itself.

    Fail closed: expired, malformed, revoked and unknown tokens all surface
    as a 401 from the introspection route; network problems are also mapped
    to ``PocketBaseTokenError`` so callers reject the request instead of
    guessing. Returns ``{"id": <record id>, "email": <email or "">}``.
    """
    request = urllib.request.Request(
        f"{POCKETBASE_URL.rstrip('/')}{_INTROSPECTION_PATH}",
        headers={"Authorization": f"Bearer {token}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=POCKETBASE_TIMEOUT_SECONDS) as response:
            payload = json.loads(response.read())
    except urllib.error.HTTPError as exc:
        logger.info("Access token rejected by introspection: HTTP %s", exc.code)
        raise PocketBaseTokenError(f"introspection returned HTTP {exc.code}") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        logger.warning("Token introspection unreachable: %s", type(exc).__name__)
        raise PocketBaseTokenError("introspection unavailable") from exc

    if not isinstance(payload, dict) or not payload.get("valid"):
        logger.info("Access token rejected: introspection reported invalid")
        raise PocketBaseTokenError("token is not valid")

    user_id = str(payload.get("id") or "")
    if not user_id:
        logger.info("Access token rejected: introspection returned no subject")
        raise PocketBaseTokenError("introspection returned no subject id")

    return {"id": user_id, "email": str(payload.get("email") or "")}


class PocketBaseUserDirectory:
    """Resolve the display username for a user id, cached briefly."""

    _CACHE_TTL_SECONDS = 300
    _CACHE_MAX_ENTRIES = 512

    def __init__(self, client=None) -> None:
        self._client = client or get_pocketbase_client()
        self._cache: Dict[str, tuple[str, float]] = {}
        self._lock = threading.Lock()

    def resolve_username(self, user_id: str) -> str:
        now = time.monotonic()
        with self._lock:
            cached = self._cache.get(user_id)
            if cached and cached[1] > now:
                return cached[0]

        username: Optional[str] = None
        try:
            record = self._client.get_record(USERS_COLLECTION, user_id)
            email = str(record.get("email", "")).strip()
            if email:
                username = email
        except Exception as exc:  # noqa: BLE001 - username is cosmetic only
            logger.info("Username lookup failed for %s: %s", type(exc).__name__, type(exc))

        resolved = username or _fallback_username(user_id)

        with self._lock:
            if len(self._cache) >= self._CACHE_MAX_ENTRIES:
                self._cache.clear()
            self._cache[user_id] = (resolved, now + self._CACHE_TTL_SECONDS)
        return resolved


class PocketBaseAccountDeletionClient:
    """Revoke a PocketBase user by deleting their ``users`` record."""

    def __init__(self, client=None) -> None:
        self._client = client or get_pocketbase_client()

    def delete_user(self, user_id: str) -> bool:
        """Delete the user record matching ``user_id``.

        CascadeDelete relations remove conversations and messages along with
        the account. Returns True on success; callers fail closed on False.
        """
        try:
            return self._client.delete_record(USERS_COLLECTION, user_id)
        except Exception as exc:  # noqa: BLE001 - surfaced as failure to caller
            logger.error("PocketBase user deletion failed for %s: %s", user_id, type(exc).__name__)
            return False
