import unittest
from unittest.mock import patch

import jwt

from backend.src.auth import create_access_token, decode_access_token, hash_password, verify_password
from backend.src.config import Config


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


if __name__ == "__main__":
    unittest.main()