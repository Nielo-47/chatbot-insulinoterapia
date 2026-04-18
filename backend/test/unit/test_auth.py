import unittest
from unittest.mock import patch
from typing import Any, Optional

import jwt

from backend.src.application.contracts.repositories import UsersRepositoryLike
from backend.src.config import Config
from backend.src.application.features.auth.auth_service import AuthenticationService
from backend.src.application.features.auth.auth_primitives import (
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)


class AuthTests(unittest.TestCase):
    def test_password_hash_round_trip(self) -> None:
        hashed = hash_password("secret-password")

        self.assertTrue(verify_password("secret-password", hashed))
        self.assertFalse(verify_password("wrong-password", hashed))

    def test_access_token_round_trip(self) -> None:
        with patch.object(Config, "JWT_SECRET_KEY", "token-secret-value-long-enough-32-bytes"):
            token = create_access_token(7, "alice", expires_minutes=5)
            payload = decode_access_token(token)

        self.assertEqual(payload["sub"], "7")
        self.assertEqual(payload["username"], "alice")

    def test_expired_token_is_rejected(self) -> None:
        with patch.object(Config, "JWT_SECRET_KEY", "token-secret-value-long-enough-32-bytes"):
            token = create_access_token(7, "alice", expires_minutes=-1)

        with patch.object(Config, "JWT_SECRET_KEY", "token-secret-value-long-enough-32-bytes"):
            with self.assertRaises(jwt.ExpiredSignatureError):
                decode_access_token(token)

    def test_delete_user_delegates_to_repository(self) -> None:
        class UsersRepositoryStub(UsersRepositoryLike):
            def __init__(self) -> None:
                self.deleted_user_id: Optional[int] = None

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


if __name__ == "__main__":
    unittest.main()
