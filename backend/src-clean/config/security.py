from .env import require, require_int, get_int, get_str

AUTH_PASSWORD_ITERATIONS = get_int("AUTH_PASSWORD_ITERATIONS", 240000)  # OWASP recommended
AUTH_PASSWORD_SALT_BYTES = get_int("AUTH_PASSWORD_SALT_BYTES", 16)

JWT_SECRET_KEY = require("JWT_SECRET_KEY")
JWT_ALGORITHM = require("JWT_ALGORITHM")
JWT_ACCESS_TOKEN_EXPIRE_MINUTES = require_int("JWT_ACCESS_TOKEN_EXPIRE_MINUTES")

# Rate limiting configuration
LOGIN_RATE_LIMIT = get_str("LOGIN_RATE_LIMIT", "5/minute")  # per IP
LOGIN_RATE_LIMIT_BLOCK_DURATION_SECONDS = get_int("LOGIN_RATE_LIMIT_BLOCK_DURATION", 900)  # 15 minutes

# Account lockout configuration
MAX_LOGIN_ATTEMPTS = get_int("MAX_LOGIN_ATTEMPTS", 5)  # Lock after 5 failed attempts
LOCKOUT_DURATION_SECONDS = get_int("LOCKOUT_DURATION_SECONDS", 900)  # 15 minutes lockout

# JWT claims
JWT_ISSUER = get_str("JWT_ISSUER", "diabetes-chatbot")
JWT_AUDIENCE = get_str("JWT_AUDIENCE", "diabetes-chatbot-api")
