import uuid
from typing import Any, Callable, Optional

from backend.src.application.contracts.repositories import UsersRepositoryLike
from backend.src.domain.models import AuthenticatedPrincipal
from backend.src.infrastructure.security import rate_limit


class AccountLockedException(Exception):
    """Raised when account is locked due to too many failed login attempts."""
    def __init__(self, remaining_seconds: int):
        self.remaining_seconds = remaining_seconds
        super().__init__(f"Account is locked. Try again in {remaining_seconds} seconds.")


class RateLimitExceededException(Exception):
    """Raised when IP rate limit is exceeded."""
    def __init__(self, remaining_seconds: int):
        self.remaining_seconds = remaining_seconds
        super().__init__(f"Too many requests. Try again in {remaining_seconds} seconds.")


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
        # Check rate limiting first (by IP)
        if client_ip is not None:
            is_allowed, _ = rate_limit.check_rate_limit(client_ip)
            if not is_allowed:
                remaining = rate_limit.get_rate_limit_remaining_seconds(client_ip)
                raise RateLimitExceededException(remaining)

        user = self.users_repository.get_user_by_username(username)
        
        # Even if user doesn't exist, perform password check to prevent timing attacks
        # Use a dummy hash for non-existent users
        stored_hash = user.hashed_password if user else "$dummy$1$dummy$dummy"
        
        if not self._verify_password(password, stored_hash):
            # Record failed login if user exists
            if user is not None:
                is_locked, remaining = rate_limit.check_account_lockout(user.id)
                if is_locked:
                    raise AccountLockedException(remaining)
                
                attempts = rate_limit.record_failed_login(user.id)
                if attempts >= 5:
                    raise AccountLockedException(rate_limit.LOCKOUT_DURATION_SECONDS)
            
            return None

        # Successful login
        if user is not None:
            # Clear failed attempts
            rate_limit.clear_failed_login_attempts(user.id)
            
            # Reset rate limit on successful login
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
        rate_limit.unlock_account(user_id)
        return self.users_repository.delete_user_by_id(user_id)

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
