import base64
import hashlib
import hmac
import os

from backend.src.config.security import AUTH_PASSWORD_ITERATIONS, AUTH_PASSWORD_SALT_BYTES

PASSWORD_HASH_ALGORITHM = "pbkdf2_sha256"


def _encode_base64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _decode_base64(encoded: str) -> bytes:
    padding = "=" * (-len(encoded) % 4)
    return base64.urlsafe_b64decode(encoded + padding)


def hash_password(password: str) -> str:
    salt = os.urandom(AUTH_PASSWORD_SALT_BYTES)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        AUTH_PASSWORD_ITERATIONS,
    )
    return f"{PASSWORD_HASH_ALGORITHM}${AUTH_PASSWORD_ITERATIONS}${_encode_base64(salt)}${_encode_base64(digest)}"


def verify_password(password: str, hashed_password: str) -> bool:
    try:
        algorithm, iterations_text, salt_text, digest_text = hashed_password.split("$", 3)
        if algorithm != PASSWORD_HASH_ALGORITHM:
            return False

        iterations = int(iterations_text)
        salt = _decode_base64(salt_text)
        expected_digest = _decode_base64(digest_text)
        actual_digest = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt,
            iterations,
        )
        return hmac.compare_digest(actual_digest, expected_digest)
    except (ValueError, TypeError):
        return False
