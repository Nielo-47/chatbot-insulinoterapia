import unittest
from types import SimpleNamespace
from unittest.mock import patch
from typing import Any, Optional

import jwt

from backend.src.application.contracts.repositories import UsersRepositoryLike
from backend.src.application.features.auth.auth_service import AuthenticationService
from backend.src.application.features.auth.auth_primitives import (
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)
from backend.src.infrastructure.security import rate_limit as rate_limit_module


def _make_user(username: str = "alice", user_id: int = 7, hashed_password: Optional[str] = None) -> SimpleNamespace:
    if hashed_password is None:
        hashed_password = hash_password("secret-password")
    return SimpleNamespace(id=user_id, username=username, hashed_password=hashed_password)


class AuthTests(unittest.TestCase):
    def test_password_hash_round_trip(self) -> None:
        hashed = hash_password("secret-password")

        self.assertTrue(verify_password("secret-password", hashed))
        self.assertFalse(verify_password("wrong-password", hashed))

    def test_access_token_round_trip(self) -> None:
        with patch("backend.src.infrastructure.security.token.JWT_SECRET_KEY", "token-secret-value-long-enough-32-bytes"):
            with patch("backend.src.infrastructure.security.token.JWT_ALGORITHM", "HS256"):
                with patch("backend.src.infrastructure.security.token.JWT_ISSUER", "test-issuer"):
                    with patch("backend.src.infrastructure.security.token.JWT_AUDIENCE", "test-audience"):
                        token = create_access_token(7, "alice", expires_minutes=5)
                        payload = decode_access_token(token)

        self.assertEqual(payload["sub"], "7")
        self.assertEqual(payload["username"], "alice")
        self.assertEqual(payload["iss"], "test-issuer")
        self.assertEqual(payload["aud"], "test-audience")
        self.assertIn("jti", payload)  # JWT ID should be present

    def test_expired_token_is_rejected(self) -> None:
        with patch("backend.src.infrastructure.security.token.JWT_SECRET_KEY", "token-secret-value-long-enough-32-bytes"):
            with patch("backend.src.infrastructure.security.token.JWT_ALGORITHM", "HS256"):
                with patch("backend.src.infrastructure.security.token.JWT_ISSUER", "test-issuer"):
                    with patch("backend.src.infrastructure.security.token.JWT_AUDIENCE", "test-audience"):
                        token = create_access_token(7, "alice", expires_minutes=-1)

        with patch("backend.src.infrastructure.security.token.JWT_SECRET_KEY", "token-secret-value-long-enough-32-bytes"):
            with patch("backend.src.infrastructure.security.token.JWT_ALGORITHM", "HS256"):
                with patch("backend.src.infrastructure.security.token.JWT_ISSUER", "test-issuer"):
                    with patch("backend.src.infrastructure.security.token.JWT_AUDIENCE", "test-audience"):
                        with self.assertRaises(jwt.ExpiredSignatureError):
                            decode_access_token(token)

    def test_delete_user_delegates_to_repository(self) -> None:
        class UsersRepositoryStub(UsersRepositoryLike):
            def __init__(self) -> None:
                self.deleted_user_id: Optional[int] = None

            def get_user_by_id(self, user_id: int) -> Optional[Any]:
                return _make_user()

            def delete_user_by_id(self, user_id: int) -> bool:
                self.deleted_user_id = user_id
                return True

        users_repository = UsersRepositoryStub()
        service = AuthenticationService(
            users_repository=users_repository,
            verify_password=verify_password,
            create_access_token=create_access_token,
            decode_access_token=decode_access_token,
        )

        deleted = service.delete_user(12)

        self.assertTrue(deleted)
        self.assertEqual(users_repository.deleted_user_id, 12)

    def test_token_contains_jti(self) -> None:
        """Test that tokens have a unique JWT ID for revocation."""
        with patch("backend.src.infrastructure.security.token.JWT_SECRET_KEY", "token-secret-value-long-enough-32-bytes"):
            with patch("backend.src.infrastructure.security.token.JWT_ALGORITHM", "HS256"):
                with patch("backend.src.infrastructure.security.token.JWT_ISSUER", "test-issuer"):
                    with patch("backend.src.infrastructure.security.token.JWT_AUDIENCE", "test-audience"):
                        token = create_access_token(7, "alice")
                        payload = decode_access_token(token)

        # Each token should have a unique jti
        self.assertIsNotNone(payload.get("jti"))
        self.assertIsInstance(payload["jti"], str)
        self.assertTrue(len(payload["jti"]) > 0)

    # ---------- authenticate_credentials ----------

    def test_authenticate_success_returns_principal(self) -> None:
        class UsersRepositoryStub(UsersRepositoryLike):
            def get_user_by_username(self, username: str) -> Optional[Any]:
                return _make_user()

        with patch.object(rate_limit_module, "check_rate_limit", return_value=(True, 4)), patch.object(
            rate_limit_module, "check_account_lockout", return_value=(False, None)
        ), patch.object(rate_limit_module, "clear_failed_login_attempts") as clear_attempts, patch.object(
            rate_limit_module, "reset_rate_limit"
        ) as reset_attempts:
            service = AuthenticationService(
                users_repository=UsersRepositoryStub(),
                verify_password=verify_password,
                create_access_token=create_access_token,
                decode_access_token=decode_access_token,
            )
            principal = service.authenticate_credentials("alice", "secret-password", client_ip="10.0.0.1")

        self.assertIsNotNone(principal)
        self.assertEqual(principal.username, "alice")
        clear_attempts.assert_called_once()
        reset_attempts.assert_called_once()

    def test_authenticate_wrong_password_returns_none(self) -> None:
        class UsersRepositoryStub(UsersRepositoryLike):
            def get_user_by_username(self, username: str) -> Optional[Any]:
                return _make_user()

        with patch.object(rate_limit_module, "check_rate_limit", return_value=(True, 4)), patch.object(
            rate_limit_module, "check_account_lockout", return_value=(False, None)
        ), patch.object(rate_limit_module, "record_failed_login") as record_failed:
            service = AuthenticationService(
                users_repository=UsersRepositoryStub(),
                verify_password=verify_password,
                create_access_token=create_access_token,
                decode_access_token=decode_access_token,
            )
            principal = service.authenticate_credentials("alice", "wrong-password", client_ip="10.0.0.1")

        self.assertIsNone(principal)
        record_failed.assert_called_once()

    def test_authenticate_nonexistent_user_returns_none_and_records(self) -> None:
        class UsersRepositoryStub(UsersRepositoryLike):
            def get_user_by_username(self, username: str) -> Optional[Any]:
                return None

        with patch.object(rate_limit_module, "check_rate_limit", return_value=(True, 4)), patch.object(
            rate_limit_module, "check_account_lockout", return_value=(False, None)
        ), patch.object(rate_limit_module, "record_failed_login") as record_failed:
            service = AuthenticationService(
                users_repository=UsersRepositoryStub(),
                verify_password=verify_password,
                create_access_token=create_access_token,
                decode_access_token=decode_access_token,
            )
            principal = service.authenticate_credentials("ghost", "whatever", client_ip="10.0.0.1")

        self.assertIsNone(principal)
        record_failed.assert_called_once()  # anti-enumeration: identical behavior for unknown users

    def test_authenticate_rate_limited_returns_none(self) -> None:
        class UsersRepositoryStub(UsersRepositoryLike):
            def get_user_by_username(self, username: str) -> Optional[Any]:
                return _make_user()

        with patch.object(rate_limit_module, "check_rate_limit", return_value=(False, 0)), patch.object(
            rate_limit_module, "check_account_lockout", return_value=(False, None)
        ), patch.object(rate_limit_module, "record_failed_login") as record_failed:
            service = AuthenticationService(
                users_repository=UsersRepositoryStub(),
                verify_password=verify_password,
                create_access_token=create_access_token,
                decode_access_token=decode_access_token,
            )
            principal = service.authenticate_credentials("alice", "secret-password", client_ip="10.0.0.1")

        self.assertIsNone(principal)
        record_failed.assert_not_called()

    def test_authenticate_locked_out_returns_none(self) -> None:
        class UsersRepositoryStub(UsersRepositoryLike):
            def get_user_by_username(self, username: str) -> Optional[Any]:
                return _make_user()

        with patch.object(rate_limit_module, "check_rate_limit", return_value=(True, 4)), patch.object(
            rate_limit_module, "check_account_lockout", return_value=(True, 300)
        ), patch.object(rate_limit_module, "record_failed_login") as record_failed:
            service = AuthenticationService(
                users_repository=UsersRepositoryStub(),
                verify_password=verify_password,
                create_access_token=create_access_token,
                decode_access_token=decode_access_token,
            )
            principal = service.authenticate_credentials("alice", "secret-password", client_ip="10.0.0.1")

        self.assertIsNone(principal)
        record_failed.assert_not_called()

    def test_authenticate_without_client_ip_still_works(self) -> None:
        class UsersRepositoryStub(UsersRepositoryLike):
            def get_user_by_username(self, username: str) -> Optional[Any]:
                return _make_user()

        with patch.object(rate_limit_module, "check_rate_limit", return_value=(True, 4)) as check_rate, patch.object(
            rate_limit_module, "check_account_lockout", return_value=(False, None)
        ):
            service = AuthenticationService(
                users_repository=UsersRepositoryStub(),
                verify_password=verify_password,
                create_access_token=create_access_token,
                decode_access_token=decode_access_token,
            )
            principal = service.authenticate_credentials("alice", "secret-password")

        self.assertIsNotNone(principal)
        check_rate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
