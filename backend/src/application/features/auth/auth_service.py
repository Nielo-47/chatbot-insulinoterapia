from typing import Protocol

from backend.src.application.contracts.repositories import UsersRepositoryLike
from backend.src.domain.models import AuthenticatedPrincipal


class AuthentikAdminClientLike(Protocol):
    def delete_user(self, username: str) -> bool: ...


class AuthenticationService:
    """Identity resolution and account lifecycle, delegated to Authentik.

    This service contains no credential handling at all: passwords, sessions,
    tokens and login throttling are all owned by Authentik (forward-auth). The
    backend only maps the identity that the trusted proxy presents to a local
    integer user id (used as the conversations FK).
    """

    def __init__(
        self,
        users_repository: UsersRepositoryLike,
        authentik_admin_client: AuthentikAdminClientLike,
    ):
        self.users_repository = users_repository
        self._authentik_admin_client = authentik_admin_client

    def resolve_principal_from_identity(self, sub: str, username: str) -> AuthenticatedPrincipal:
        """Resolve an Authentik identity to a local principal, provisioning on first login."""
        user_id, _ = self.users_repository.get_or_create_user_by_sub(sub, username)
        return AuthenticatedPrincipal(id=user_id, username=username)

    def delete_user(self, user_id: int) -> bool:
        """Remove the local user row (conversations/messages cascade)."""
        return self.users_repository.delete_user_by_id(user_id)

    def delete_authentik_user(self, username: str) -> bool:
        """Revoke the account in Authentik via the Admin API."""
        return self._authentik_admin_client.delete_user(username)


def build_authentication_service(
    users_repository: UsersRepositoryLike,
    authentik_admin_client: AuthentikAdminClientLike,
) -> AuthenticationService:
    return AuthenticationService(
        users_repository=users_repository,
        authentik_admin_client=authentik_admin_client,
    )
