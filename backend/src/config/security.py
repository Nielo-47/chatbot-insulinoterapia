from .env import get_str, get_int

# Trusted reverse proxies for real client IP resolution and forwarded identity
# headers. Client-IP headers (X-Real-IP, X-Forwarded-For) and Authentik identity
# headers (X-authentik-*) are honored ONLY when the request's direct peer
# matches one of these entries (IP or CIDR); otherwise the peer address is used.
# Empty means never trust forwarded headers.
TRUSTED_PROXY_IPS = [
    item.strip() for item in get_str("TRUSTED_PROXY_IPS", "").split(",") if item.strip()
]

# Query rate limiting configuration (app-level quota, not authentication)
QUERY_RATE_LIMIT = get_int("QUERY_RATE_LIMIT", 30)  # max queries per window per user
QUERY_RATE_WINDOW_SECONDS = get_int("QUERY_RATE_WINDOW_SECONDS", 60)  # 1 minute window
