from datetime import datetime, timedelta, timezone
from typing import Any, Optional
import uuid

import jwt

from backend.src.config.security import (
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES,
    JWT_ALGORITHM,
    JWT_AUDIENCE,
    JWT_ISSUER,
    JWT_SECRET_KEY,
)


def create_access_token(
    user_id: int,
    username: str,
    expires_minutes: Optional[int] = None,
    jti: Optional[str] = None,
) -> str:
    now = datetime.now(timezone.utc)
    expire_delta = timedelta(minutes=expires_minutes if expires_minutes is not None else JWT_ACCESS_TOKEN_EXPIRE_MINUTES)
    payload: dict[str, Any] = {
        "sub": str(user_id),
        "username": username,
        "iat": int(now.timestamp()),
        "exp": int((now + expire_delta).timestamp()),
        "iss": JWT_ISSUER,
        "aud": JWT_AUDIENCE,
        "jti": jti or str(uuid.uuid4()),
    }
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


def decode_access_token(token: str) -> dict[str, Any]:
    return jwt.decode(
        token,
        JWT_SECRET_KEY,
        algorithms=[JWT_ALGORITHM],
        audience=JWT_AUDIENCE,
        issuer=JWT_ISSUER,
    )
