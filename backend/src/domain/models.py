from dataclasses import dataclass
from datetime import datetime
from typing import Optional
import uuid


@dataclass(frozen=True)
class AuthenticatedPrincipal:
    id: uuid.UUID
    username: str


@dataclass(frozen=True)
class Profile:
    user_id: uuid.UUID
    username: str
    created_at: Optional[datetime] = None


__all__ = [
    "AuthenticatedPrincipal",
    "Profile",
]
