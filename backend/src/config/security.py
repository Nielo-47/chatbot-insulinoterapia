from .env import get_int

# Query rate limiting configuration (app-level quota, not authentication)
QUERY_RATE_LIMIT = get_int("QUERY_RATE_LIMIT", 30)  # max queries per window per user
QUERY_RATE_WINDOW_SECONDS = get_int("QUERY_RATE_WINDOW_SECONDS", 60)  # 1 minute window
