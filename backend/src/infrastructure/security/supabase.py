"""Supabase Auth integration.

Two distinct responsibilities live here:

1. ``verify_access_token`` — stateless verification of the access token the
   frontend presents as a Bearer header. The backend never sees passwords or
   runs any token-exchange with Supabase; it validates the token signature
   against the project's public JWKS endpoint (``SUPABASE_JWKS_URL``) using the
   signing algorithm the JWKS advertises (RS256 or ES256), plus the audience
   and the role. Private signing keys never leave Supabase Auth.

2. ``SupabaseAccountDeletionClient`` — revokes a Supabase Auth user by calling
   the ``delete-account`` edge function (@supabase/server on the edge), which
   holds the secret key. The backend never stores that key; it only forwards
   the caller's access token so the edge function can authenticate them and
   delete their own account. Uses the stdlib (urllib) so no extra runtime
   dependency is required.
"""

import json
import logging
import uuid
import urllib.error
import urllib.request
from typing import Any, Optional

import jwt

from backend.src.config.supabase import SUPABASE_JWKS_URL, SUPABASE_URL

logger = logging.getLogger(__name__)

SUPABASE_AUTH_AUDIENCE = "authenticated"
SUPABASE_AUTH_ROLE = "authenticated"

# Lazily fetches and caches the project's public signing keys from the JWKS
# endpoint; access tokens are verified against those public keys with the
# algorithm the JWKS advertises (RS256 or ES256).
jwks_client = jwt.PyJWKClient(SUPABASE_JWKS_URL)


class SupabaseTokenError(Exception):
    """Raised when a presented access token cannot be verified."""


def verify_access_token(token: str) -> dict[str, Any]:
    """Verify a Supabase access token and return its claims.

    Fail closed: expired, malformed, wrong-audience, anon-role, unsigned and
    tokens signed by any key other than the project's current signing keys are
    all rejected. Returns the payload dict on success.
    """
    try:
        signing_key = jwks_client.get_signing_key_from_jwt(token)
        claims = jwt.decode(
            token,
            signing_key.key,
            algorithms=[signing_key.algorithm_name],
            audience=SUPABASE_AUTH_AUDIENCE,
        )
    except jwt.PyJWTError as exc:
        logger.info("Access token verification failed: %s", type(exc).__name__)
        raise SupabaseTokenError(str(exc)) from exc

    sub = claims.get("sub")
    role = claims.get("role")
    if not sub or role != SUPABASE_AUTH_ROLE:
        logger.info("Access token rejected: missing sub or non-authenticated role")
        raise SupabaseTokenError("Token is not an authenticated-user access token")

    return claims


class SupabaseAccountDeletionClient:
    """Revoke a Supabase Auth user via the ``delete-account`` edge function."""

    def __init__(self, base_url: Optional[str] = None, timeout: int = 10) -> None:
        self.base_url = (base_url or SUPABASE_URL).rstrip("/")
        self.timeout = timeout

    def delete_user(self, user_id: uuid.UUID, access_token: str) -> bool:
        """Revoke the Auth user matching ``user_id``.

        The caller's access token is forwarded unchanged; the edge function
        verifies it and refuses to delete any account other than the token's
        own ``sub``. Returns True on success. The local profile row is removed
        by the ``on_auth_user_deleted`` trigger (and explicitly by the auth
        service), cascading to conversations/messages.
        """
        url = f"{self.base_url}/functions/v1/delete-account"
        body = json.dumps({"user_id": str(user_id)}).encode("utf-8")
        request = urllib.request.Request(url, data=body, method="POST")
        request.add_header("Content-Type", "application/json")
        request.add_header("Authorization", f"Bearer {access_token}")
        request.add_header("Accept", "application/json")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return response.status in (200, 201, 204)
        except urllib.error.HTTPError as exc:
            logger.error("delete-account edge function failed with HTTP %s", exc.code)
            return False
        except OSError as exc:
            logger.error("delete-account edge function unreachable: %s", exc)
            return False
