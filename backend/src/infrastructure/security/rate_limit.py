"""Per-user query rate limiting using Redis.

Login throttling, account lockout and token blacklisting live in Supabase
Auth; what remains here is the application-level query quota (a functional
cap, not an authentication control).

Fail-closed policy: Redis is a hard dependency of the whole application
(docker-compose deployment), so when Redis is unavailable every throttle check
denies the request instead of failing open, and every cleanup / best-effort
operation degrades safely instead of raising.
"""

import logging
import uuid

import redis

from backend.src.config.infrastructure import CHAT_CACHE_REDIS_URL
from backend.src.config.security import QUERY_RATE_LIMIT, QUERY_RATE_WINDOW_SECONDS

logger = logging.getLogger(__name__)


def _get_redis_client() -> redis.Redis:
    """Get Redis client for rate limiting."""
    return redis.from_url(CHAT_CACHE_REDIS_URL, decode_responses=True)


# ==================== Query Rate Limiting ====================

QUERY_LIMIT_PREFIX = "ratelimit:query:"


def _get_query_limit_key(user_id: uuid.UUID) -> str:
    """Get Redis key for per-user query rate limiting."""
    return f"{QUERY_LIMIT_PREFIX}{user_id}"


def check_query_rate_limit(user_id: uuid.UUID) -> tuple[bool, int]:
    """
    Check if a user has exceeded the query rate limit.

    Returns:
        tuple: (is_allowed, remaining_attempts)
    """
    key = _get_query_limit_key(user_id)

    try:
        client = _get_redis_client()
        current = client.get(key)
        if current is None:
            client.setex(key, QUERY_RATE_WINDOW_SECONDS, 1)
            return True, QUERY_RATE_LIMIT - 1

        count = int(current)
        if count >= QUERY_RATE_LIMIT:
            return False, 0

        # Increment counter
        client.incr(key)
        return True, QUERY_RATE_LIMIT - (count + 1)
    except redis.RedisError as e:
        logger.warning("Redis error in query rate limiting: %s", e)
        # Fail closed: deny the request when Redis is unavailable.
        return False, 0


def get_query_rate_limit_remaining_seconds(user_id: uuid.UUID) -> int:
    """Get remaining seconds until the query rate limit resets."""
    key = _get_query_limit_key(user_id)

    try:
        client = _get_redis_client()
        ttl = client.ttl(key)
        return max(0, ttl)
    except redis.RedisError:
        return 0
