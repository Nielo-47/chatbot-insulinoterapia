"""Authentication primitives - consolidated token and password functions.

This module re-exports functions from the infrastructure layer for convenience.
All token operations should use the centralized implementation in 
backend.src.infrastructure.security.token
"""

from backend.src.infrastructure.security.password import hash_password, verify_password

# Re-export token functions from centralized location
from backend.src.infrastructure.security.token import create_access_token, decode_access_token

__all__ = [
    "create_access_token",
    "decode_access_token",
    "hash_password",
    "verify_password",
]
