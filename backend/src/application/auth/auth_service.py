from dataclasses import dataclass
from typing import Optional

from backend.src.application.auth.auth_primitives import create_access_token, decode_access_token, verify_password
from backend.src.infrastructure.repositories.users_repository import UsersRepository


@dataclass(frozen=True)
class AuthenticatedPrincipal:
    id: int
    username: str


class AuthenticationService:
    def __init__(self, users_repository: UsersRepository):
        self.users_repository = users_repository

    def authenticate_credentials(self, username: str, password: str) -> Optional[AuthenticatedPrincipal]:
        user = self.users_repository.get_user_by_username(username)
        if user is None:
            return None
        if not verify_password(password, user.hashed_password):
            return None
        return AuthenticatedPrincipal(id=user.id, username=user.username)

    def resolve_principal_from_token(self, token: str) -> Optional[AuthenticatedPrincipal]:
        try:
            payload = decode_access_token(token)
            user_id = int(payload.get("sub", ""))
        except Exception:
            return None

        user = self.users_repository.get_user_by_id(user_id)
        if user is None:
            return None
        return AuthenticatedPrincipal(id=user.id, username=user.username)

    def issue_access_token(self, principal: AuthenticatedPrincipal) -> str:
        return create_access_token(user_id=principal.id, username=principal.username)


def build_authentication_service() -> AuthenticationService:
    return AuthenticationService(users_repository=UsersRepository())