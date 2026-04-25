from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass(frozen=True)
class AuthenticatedPrincipal:
    id: int
    username: str


@dataclass(frozen=True)
class User:
    id: int
    username: str
    hashed_password: str
    created_at: Optional[datetime] = None


@dataclass(frozen=True)
class Conversation:
    id: int
    user_id: int
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


@dataclass(frozen=True)
class Message:
    id: Optional[int]
    conversation_id: int
    role: str
    content: str
    created_at: Optional[datetime] = None

