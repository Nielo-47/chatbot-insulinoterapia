import os

from .env import require, require_int

AUTH_PASSWORD_ITERATIONS = require_int("AUTH_PASSWORD_ITERATIONS")
AUTH_PASSWORD_SALT_BYTES = require_int("AUTH_PASSWORD_SALT_BYTES")

JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "dev-local-jwt-secret-please-change-32-plus")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
JWT_ACCESS_TOKEN_EXPIRE_MINUTES = require_int("JWT_ACCESS_TOKEN_EXPIRE_MINUTES")