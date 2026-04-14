from typing import Any, Callable, Optional

from backend.src.domain.models import AuthenticatedPrincipal
from backend.src.infrastructure.repositories.users_repository import UsersRepository


class AuthenticationService:
    def __init__(
        self,
        users_repository: UsersRepository,
        verify_password: Callable[[str, str], bool],
        create_access_token: Callable[..., str],
        decode_access_token: Callable[[str], dict[str, Any]],
    ):
        self.users_repository = users_repository
        self._verify_password = verify_password
        self._create_access_token = create_access_token
        self._decode_access_token = decode_access_token

    def authenticate_credentials(self, username: str, password: str) -> Optional[AuthenticatedPrincipal]:
        user = self.users_repository.get_user_by_username(username)
        if user is None:
            return None
        if not self._verify_password(password, user.hashed_password):
            return None
        return AuthenticatedPrincipal(id=user.id, username=user.username)

    def resolve_principal_from_token(self, token: str) -> Optional[AuthenticatedPrincipal]:
        try:
            payload = self._decode_access_token(token)
            user_id = int(payload.get("sub", ""))
        except Exception:
            return None

        user = self.users_repository.get_user_by_id(user_id)
        if user is None:
            return None
        return AuthenticatedPrincipal(id=user.id, username=user.username)

    def issue_access_token(self, principal: AuthenticatedPrincipal) -> str:
        return self._create_access_token(user_id=principal.id, username=principal.username)

    def delete_user(self, user_id: int) -> bool:
        return self.users_repository.delete_user_by_id(user_id)


def build_authentication_service(
    users_repository: UsersRepository,
    verify_password: Callable[[str, str], bool],
    create_access_token: Callable[..., str],
    decode_access_token: Callable[[str], dict[str, Any]],
) -> AuthenticationService:
    return AuthenticationService(
        users_repository=users_repository,
        verify_password=verify_password,
        create_access_token=create_access_token,
        decode_access_token=decode_access_token,
    )
