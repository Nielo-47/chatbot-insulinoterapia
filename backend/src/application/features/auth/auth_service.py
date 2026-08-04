import os
import uuid
from typing import Any, Callable, Optional

from backend.src.application.contracts.repositories import UsersRepositoryLike
from backend.src.domain.models import AuthenticatedPrincipal
from backend.src.infrastructure.security import rate_limit
from backend.src.infrastructure.security.password import hash_password

# Dummy hash used to equalize timing for non-existent users. It is a real
# PBKDF2-SHA256 hash (same algorithm/iteration count as real users) so that
# verify_password() runs the full verification work and the response timing
# is indistinguishable whether or not the account exists. A syntactically
# invalid value (e.g. a different algorithm) would short-circuit in
# verify_password() and reintroduce a timing-based account enumeration vector.
_DUMMY_HASH = hash_password(os.urandom(16).hex())


class AuthenticationService:
    def __init__(
        self,
        users_repository: UsersRepositoryLike,
        verify_password: Callable[[str, str], bool],
        create_access_token: Callable[..., str],
        decode_access_token: Callable[[str], dict[str, Any]],
    ):
        self.users_repository = users_repository
        self._verify_password = verify_password
        self._create_access_token = create_access_token
        self._decode_access_token = decode_access_token

    def authenticate_credentials(
        self,
        username: str,
        password: str,
        client_ip: Optional[str] = None,
    ) -> Optional[AuthenticatedPrincipal]:
        # Lockout identity is keyed on (normalized username, ip) so a single
        # source cannot permanently lock an account, while distributed attacks
        # are still tracked per pair.
        identity = rate_limit.build_lockout_identity(username, client_ip)

        # IP rate limiting first (fail closed: deny when Redis is down)
        if client_ip is not None:
            is_allowed, _ = rate_limit.check_rate_limit(client_ip)
            if not is_allowed:
                return None

        # Account lockout check before any credential work (fail closed)
        is_locked, _ = rate_limit.check_account_lockout(identity)
        if is_locked:
            return None

        user = self.users_repository.get_user_by_username(username)

        # Even if user doesn't exist, perform password check to prevent timing attacks
        stored_hash = user.hashed_password if user else _DUMMY_HASH

        if not self._verify_password(password, stored_hash):
            # Record failed login for every failure (existing or not) so lockout
            # behavior does not reveal whether the account exists.
            rate_limit.record_failed_login(identity)
            return None

        # Successful login
        if user is not None:
            rate_limit.clear_failed_login_attempts(identity)

            # Reset IP rate limit on successful login
            if client_ip is not None:
                rate_limit.reset_rate_limit(client_ip)

        return AuthenticatedPrincipal(id=user.id, username=user.username) if user else None

    def resolve_principal_from_token(self, token: str) -> Optional[AuthenticatedPrincipal]:
        try:
            # Check if token is blacklisted
            payload = self._decode_access_token(token)
            jti = payload.get("jti")
            if jti and rate_limit.is_token_blacklisted(jti):
                return None
                
            user_id = int(payload.get("sub", ""))
        except Exception:
            return None

        user = self.users_repository.get_user_by_id(user_id)
        if user is None:
            return None
        return AuthenticatedPrincipal(id=user.id, username=user.username)

    def issue_access_token(self, principal: AuthenticatedPrincipal) -> str:
        # Generate a unique JWT ID (jti) for token revocation
        jti = str(uuid.uuid4())
        return self._create_access_token(user_id=principal.id, username=principal.username, jti=jti)

    def delete_user(self, user_id: int) -> bool:
        # Unlock account when user is deleted
        user = self.users_repository.get_user_by_id(user_id)
        if user is not None:
            rate_limit.unlock_account(user.username)
        return self.users_repository.delete_user_by_id(user_id)

    def confirm_password(self, user_id: int, password: str) -> bool:
        """Verify the account password before a destructive action.

        When the user does not exist a dummy verification still runs so the
        response timing does not reveal whether the account exists.
        """
        user = self.users_repository.get_user_by_id(user_id)
        stored_hash = user.hashed_password if user else _DUMMY_HASH
        return user is not None and self._verify_password(password, stored_hash)

    def logout_token(self, token: str) -> bool:
        """Blacklist a token (for logout)."""
        try:
            payload = self._decode_access_token(token)
            jti = payload.get("jti")
            exp = payload.get("exp")
            
            if jti and exp:
                import time
                expires_in = exp - int(time.time())
                if expires_in > 0:
                    return rate_limit.blacklist_token(jti, expires_in)
            return False
        except Exception:
            return False


def build_authentication_service(
    users_repository: UsersRepositoryLike,
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
