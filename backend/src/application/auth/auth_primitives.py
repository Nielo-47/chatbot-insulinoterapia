from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import jwt

from backend.src.config import Config
from backend.src.infrastructure.security.password import hash_password, verify_password


def create_access_token(user_id: int, username: str, expires_minutes: Optional[int] = None) -> str:
    now = datetime.now(timezone.utc)
    expire_delta = timedelta(
        minutes=expires_minutes
        if expires_minutes is not None
        else Config.JWT_ACCESS_TOKEN_EXPIRE_MINUTES
    )
    payload: dict[str, Any] = {
        "sub": str(user_id),
        "username": username,
        "iat": int(now.timestamp()),
        "exp": int((now + expire_delta).timestamp()),
    }
    return jwt.encode(payload, Config.JWT_SECRET_KEY, algorithm=Config.JWT_ALGORITHM)


def decode_access_token(token: str) -> dict[str, Any]:
    return jwt.decode(token, Config.JWT_SECRET_KEY, algorithms=[Config.JWT_ALGORITHM])
