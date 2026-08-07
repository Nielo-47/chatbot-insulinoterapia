import logging
import uuid
from typing import Protocol

from backend.src.application.contracts.repositories import ProfilesRepositoryLike
from backend.src.domain.models import AuthenticatedPrincipal

logger = logging.getLogger(__name__)


class AccountDeletionClientLike(Protocol):
    def delete_user(self, user_id: uuid.UUID, access_token: str) -> bool: ...


class AuthenticationService:
    """Identity resolution and account lifecycle, delegated to Supabase Auth.

    This service contains no credential handling at all: passwords, sessions
    and token issuance are owned by Supabase Auth. The backend only verifies
    the JWT access token (see infra/security/supabase.py) and maps its ``sub``
    claim (a UUID) to the local ``profiles`` row used as the conversations FK.
    """

    def __init__(
        self,
        profiles_repository: ProfilesRepositoryLike,
        account_deletion_client: AccountDeletionClientLike,
    ):
        self.profiles_repository = profiles_repository
        self._account_deletion_client = account_deletion_client

    def resolve_principal_from_identity(self, sub: str, username: str) -> AuthenticatedPrincipal:
        """Resolve a Supabase identity to a local principal, provisioning on first login."""
        user_id = uuid.UUID(sub)
        self.profiles_repository.get_or_create_profile(user_id, username)
        return AuthenticatedPrincipal(id=user_id, username=username)

    def delete_supabase_user(self, user_id: uuid.UUID, access_token: str) -> bool:
        """Revoke the Auth user and remove the local profile row.

        The Auth user is revoked first (via the ``delete-account`` edge
        function, authenticated with the caller's access token) so access stops
        immediately; only then is the local profile deleted
        (conversations/messages cascade). If the revocation fails, the local
        account is kept untouched (fail closed). The ``on_auth_user_deleted``
        trigger is a safety net for deletions done outside this path.
        """
        if not self._account_deletion_client.delete_user(user_id, access_token):
            return False
        if not self.profiles_repository.delete_profile(user_id):
            logger.warning("Profile row for %s already gone after Supabase user deletion", user_id)
        return True


def build_authentication_service(
    profiles_repository: ProfilesRepositoryLike,
    account_deletion_client: AccountDeletionClientLike,
) -> AuthenticationService:
    return AuthenticationService(
        profiles_repository=profiles_repository,
        account_deletion_client=account_deletion_client,
    )
