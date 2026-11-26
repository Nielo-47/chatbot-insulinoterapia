import logging
from typing import Callable, Optional

from backend.src.domain.models import AuthenticatedPrincipal

logger = logging.getLogger(__name__)


class AccountDeletionClientLike:
    """Structural contract: revoke the auth account for a user id."""

    def delete_user(self, user_id: str) -> bool: ...


class AuthenticationService:
    """Identity resolution and account lifecycle, delegated to PocketBase Auth.

    This service contains no credential handling at all: passwords, sessions
    and token issuance are owned by PocketBase. The backend only verifies the
    JWT access token locally (see infra/security/pocketbase.py) and reads its
    ``id`` claim — the PocketBase record id, which doubles as the conversations
    relation target. There is no local profile row anymore: the display name
    lives on the ``users`` record itself.
    """

    def __init__(
        self,
        account_deletion_client: AccountDeletionClientLike,
        username_resolver: Optional[Callable[[str], str]] = None,
    ):
        self._account_deletion_client = account_deletion_client
        self._username_resolver = username_resolver

    def resolve_principal_from_identity(self, user_id: str, username: str) -> AuthenticatedPrincipal:
        """Map a verified token subject to the application principal.

        No provisioning step is needed: the users record IS the profile.
        """
        if not user_id:
            raise ValueError("user id must not be empty")
        return AuthenticatedPrincipal(id=user_id, username=username)

    def resolve_username(self, user_id: str) -> str:
        """Resolve the cosmetic display name for a user id."""
        fallback = f"user_{user_id[:8]}" if user_id else "user"
        if self._username_resolver is None:
            return fallback
        try:
            resolved = self._username_resolver(user_id)
        except Exception as exc:  # noqa: BLE001 - username is cosmetic only
            logger.info("Username resolution failed for %s: %s", user_id, type(exc).__name__)
            return fallback
        return resolved or fallback

    def delete_account(self, user_id: str) -> bool:
        """Revoke the auth account.

        CascadeDelete relations remove conversations/messages with it. If the
        revocation fails the account is kept untouched (fail closed).
        """
        if not self._account_deletion_client.delete_user(user_id):
            return False
        return True


def build_authentication_service(
    account_deletion_client: AccountDeletionClientLike,
    username_resolver: Optional[Callable[[str], str]] = None,
) -> AuthenticationService:
    return AuthenticationService(
        account_deletion_client=account_deletion_client,
        username_resolver=username_resolver,
    )
