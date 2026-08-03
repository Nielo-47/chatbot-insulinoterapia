from .env import require, require_int, get_int, get_str, get_bool

AUTH_PASSWORD_ITERATIONS = get_int("AUTH_PASSWORD_ITERATIONS", 240000)  # OWASP recommended
AUTH_PASSWORD_SALT_BYTES = get_int("AUTH_PASSWORD_SALT_BYTES", 16)

JWT_SECRET_KEY = require("JWT_SECRET_KEY")
JWT_ALGORITHM = require("JWT_ALGORITHM")
JWT_ACCESS_TOKEN_EXPIRE_MINUTES = require_int("JWT_ACCESS_TOKEN_EXPIRE_MINUTES")

# Trusted reverse proxies for real client IP resolution. Client-IP headers
# (X-Real-IP, X-Forwarded-For) are honored ONLY when the request's direct peer
# matches one of these entries (IP or CIDR); otherwise the peer address is used.
# Empty means never trust forwarded headers.
TRUSTED_PROXY_IPS = [
    item.strip() for item in get_str("TRUSTED_PROXY_IPS", "").split(",") if item.strip()
]

# Rate limiting configuration
LOGIN_RATE_LIMIT = get_str("LOGIN_RATE_LIMIT", "5/minute")  # per IP
LOGIN_RATE_LIMIT_BLOCK_DURATION_SECONDS = get_int("LOGIN_RATE_LIMIT_BLOCK_DURATION", 900)  # 15 minutes

# Account lockout configuration
MAX_LOGIN_ATTEMPTS = get_int("MAX_LOGIN_ATTEMPTS", 5)  # Lock after 5 failed attempts
LOCKOUT_DURATION_SECONDS = get_int("LOCKOUT_DURATION_SECONDS", 900)  # 15 minutes lockout

# Query rate limiting configuration
QUERY_RATE_LIMIT = get_int("QUERY_RATE_LIMIT", 30)  # max queries per window per user
QUERY_RATE_WINDOW_SECONDS = get_int("QUERY_RATE_WINDOW_SECONDS", 60)  # 1 minute window

# JWT claims
JWT_ISSUER = get_str("JWT_ISSUER", "diabetes-chatbot")
JWT_AUDIENCE = get_str("JWT_AUDIENCE", "diabetes-chatbot-api")

# Session cookie (httpOnly; keeps the JWT out of localStorage/JS-accessible storage)
AUTH_COOKIE_NAME = get_str("AUTH_COOKIE_NAME", "access_token")
AUTH_COOKIE_HTTPONLY = get_bool("AUTH_COOKIE_HTTPONLY", True)
AUTH_COOKIE_SECURE = get_bool("AUTH_COOKIE_SECURE", True)  # HTTPS only; set false for plain-HTTP dev
AUTH_COOKIE_SAMESITE = get_str("AUTH_COOKIE_SAMESITE", "lax")  # Lax blocks CSRF on cross-site POST/DELETE
AUTH_COOKIE_PATH = get_str("AUTH_COOKIE_PATH", "/")
AUTH_COOKIE_DOMAIN = get_str("AUTH_COOKIE_DOMAIN", "")
