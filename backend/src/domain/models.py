"""Domain models shared across the application layers.

User ids are PocketBase record ids (strings of at most 15 lowercase
alphanumeric characters); the ``users`` auth record doubles as the profile
(username lives on the record) and the conversations relation points at it.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class AuthenticatedPrincipal:
    id: str
    username: str


__all__ = [
    "AuthenticatedPrincipal",
]
