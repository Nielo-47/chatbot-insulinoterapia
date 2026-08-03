"""Rate limiting and account lockout functionality using Redis.

Fail-closed policy: Redis is a hard dependency of the whole application
(docker-compose deployment), so when Redis is unavailable every throttling /
lockout check denies the request instead of failing open, and every cleanup /
best-effort operation degrades safely instead of raising.
"""

import logging
import time
from typing import Optional

import redis

from backend.src.config.infrastructure import CHAT_CACHE_REDIS_URL, TOKEN_BLACKLIST_ENABLED, TOKEN_BLACKLIST_PREFIX
from backend.src.config.security import (
    LOCKOUT_DURATION_SECONDS,
    MAX_LOGIN_ATTEMPTS,
    QUERY_RATE_LIMIT,
    QUERY_RATE_WINDOW_SECONDS,
)

logger = logging.getLogger(__name__)


def _get_redis_client() -> redis.Redis:
    """Get Redis client for rate limiting."""
    return redis.from_url(CHAT_CACHE_REDIS_URL, decode_responses=True)


# ==================== Rate Limiting ====================

RATE_LIMIT_PREFIX = "ratelimit:login:"
RATE_LIMIT_WINDOW_SECONDS = 300  # 5 minutes window


def _get_rate_limit_key(ip_address: str) -> str:
    """Get Redis key for rate limiting by IP."""
    return f"{RATE_LIMIT_PREFIX}{ip_address}"


def check_rate_limit(ip_address: str) -> tuple[bool, int]:
    """
    Check if an IP address has exceeded the rate limit.

    Returns:
        tuple: (is_allowed, remaining_attempts)
    """
    key = _get_rate_limit_key(ip_address)

    try:
        client = _get_redis_client()
        current = client.get(key)
        if current is None:
            # First attempt - allow it, set the counter
            client.setex(key, RATE_LIMIT_WINDOW_SECONDS, 1)
            return True, 4  # 5 total - 1 used = 4 remaining

        count = int(current)
        if count >= 5:  # 5 attempts per 5 minutes
            return False, 0

        # Increment counter
        client.incr(key)
        return True, 4 - count
    except redis.RedisError as e:
        logger.warning("Redis error in rate limiting: %s", e)
        # Fail closed: deny the request when Redis is unavailable.
        return False, 0


def reset_rate_limit(ip_address: str) -> None:
    """Reset rate limit for an IP address after successful login."""
    key = _get_rate_limit_key(ip_address)

    try:
        client = _get_redis_client()
        client.delete(key)
    except redis.RedisError as e:
        logger.warning("Redis error resetting rate limit: %s", e)


def get_rate_limit_remaining_seconds(ip_address: str) -> int:
    """Get remaining seconds until rate limit resets."""
    key = _get_rate_limit_key(ip_address)

    try:
        client = _get_redis_client()
        ttl = client.ttl(key)
        return max(0, ttl)
    except redis.RedisError:
        return 0


# ==================== Query Rate Limiting ====================

QUERY_LIMIT_PREFIX = "ratelimit:query:"


def _get_query_limit_key(user_id: int) -> str:
    """Get Redis key for per-user query rate limiting."""
    return f"{QUERY_LIMIT_PREFIX}{user_id}"


def check_query_rate_limit(user_id: int) -> tuple[bool, int]:
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


def get_query_rate_limit_remaining_seconds(user_id: int) -> int:
    """Get remaining seconds until the query rate limit resets."""
    key = _get_query_limit_key(user_id)

    try:
        client = _get_redis_client()
        ttl = client.ttl(key)
        return max(0, ttl)
    except redis.RedisError:
        return 0


# ==================== Account Lockout ====================

LOCKOUT_PREFIX = "lockout:"
LOCKOUT_USER_PREFIX = f"{LOCKOUT_PREFIX}user:"
LOCKOUT_ATTEMPTS_PREFIX = f"{LOCKOUT_PREFIX}attempts:"


def build_lockout_identity(username: str, ip_address: Optional[str]) -> str:
    """Build a normalized identity keyed by (username, ip) for lockout tracking.

    Keying on username + IP prevents a single source from locking an account
    (brute-force protection) while still allowing a distributed attack to be
    tracked per pair.
    """
    user_part = (username or "").strip().lower()
    ip_part = (ip_address or "unknown").strip()
    return f"{user_part}|{ip_part}"


def _get_lockout_key(identity: str) -> str:
    return f"{LOCKOUT_USER_PREFIX}{identity}"


def _get_attempts_key(identity: str) -> str:
    return f"{LOCKOUT_ATTEMPTS_PREFIX}{identity}"


def check_account_lockout(identity: str) -> tuple[bool, Optional[int]]:
    """
    Check if an account is locked out for the given (username, ip) identity.

    Returns:
        tuple: (is_locked, remaining_seconds)
    """
    key = _get_lockout_key(identity)

    try:
        client = _get_redis_client()
        locked_until = client.get(key)
        if locked_until is None:
            return False, None

        locked_until_ts = int(locked_until)
        current_ts = int(time.time())

        if current_ts >= locked_until_ts:
            # Lockout expired, remove it
            client.delete(key)
            return False, None

        remaining = locked_until_ts - current_ts
        return True, remaining
    except redis.RedisError as e:
        logger.warning("Redis error checking account lockout: %s", e)
        # Fail closed: treat the account as locked when Redis is unavailable.
        return True, 0


def record_failed_login(identity: str) -> int:
    """
    Record a failed login attempt for a (username, ip) identity.

    Returns:
        int: Number of consecutive failed attempts after this one
    """
    attempts_key = _get_attempts_key(identity)

    try:
        client = _get_redis_client()

        # First check if already locked
        locked_until = client.get(_get_lockout_key(identity))
        if locked_until is not None:
            return MAX_LOGIN_ATTEMPTS  # Already locked

        # Get current failed attempts count
        attempts = client.get(attempts_key)
        if attempts is None:
            attempts = 0
        else:
            attempts = int(attempts)

        attempts += 1

        if attempts >= MAX_LOGIN_ATTEMPTS:
            # Lock the account
            lockout_until = int(time.time()) + LOCKOUT_DURATION_SECONDS
            client.setex(_get_lockout_key(identity), LOCKOUT_DURATION_SECONDS, lockout_until)
            # Reset attempts counter
            client.delete(attempts_key)
            logger.warning("Login identity %r locked out after %d failed login attempts", identity, attempts)
            return attempts

        # Increment failed attempts counter with expiry
        client.setex(attempts_key, LOCKOUT_DURATION_SECONDS, attempts)
        return attempts

    except redis.RedisError as e:
        logger.warning("Redis error recording failed login: %s", e)
        # Fail closed: report that the limit has been reached.
        return MAX_LOGIN_ATTEMPTS


def clear_failed_login_attempts(identity: str) -> None:
    """Clear failed login attempts after successful login."""
    try:
        client = _get_redis_client()
        client.delete(_get_attempts_key(identity))
    except redis.RedisError as e:
        logger.warning("Redis error clearing failed login attempts: %s", e)


def unlock_account(username: str) -> None:
    """Manually unlock a user account by removing all lockout state for that username."""
    try:
        client = _get_redis_client()
    except redis.RedisError as e:
        logger.warning("Redis error unlocking account: %s", e)
        return

    patterns = [
        f"{LOCKOUT_USER_PREFIX}{username.strip().lower()}|*",
        f"{LOCKOUT_ATTEMPTS_PREFIX}{username.strip().lower()}|*",
    ]

    try:
        for pattern in patterns:
            for key in client.scan_iter(match=pattern, count=100):
                client.delete(key)
        logger.info("Account %r unlocked", username)
    except redis.RedisError as e:
        logger.warning("Redis error unlocking account: %s", e)


# ==================== Token Blacklist ====================

def is_token_blacklisted(token_jti: str) -> bool:
    """Check if a token is blacklisted (for logout)."""
    if not TOKEN_BLACKLIST_ENABLED:
        return False

    try:
        client = _get_redis_client()
    except redis.RedisError as e:
        logger.warning("Redis error checking token blacklist: %s", e)
        # Fail closed: treat tokens as blacklisted when Redis is unavailable.
        return True

    key = f"{TOKEN_BLACKLIST_PREFIX}{token_jti}"

    try:
        return client.exists(key) > 0
    except redis.RedisError as e:
        logger.warning("Redis error checking token blacklist: %s", e)
        # Fail closed: treat tokens as blacklisted when Redis is unavailable.
        return True


def blacklist_token(token_jti: str, expires_in_seconds: int) -> bool:
    """
    Add a token to the blacklist.

    Args:
        token_jti: The JWT ID (jti) claim
        expires_in_seconds: How long until the token naturally expires

    Returns:
        bool: True if successfully blacklisted
    """
    if not TOKEN_BLACKLIST_ENABLED:
        return True  # No-op if disabled

    try:
        client = _get_redis_client()
    except redis.RedisError as e:
        logger.warning("Redis error blacklisting token: %s", e)
        return False

    key = f"{TOKEN_BLACKLIST_PREFIX}{token_jti}"

    try:
        # Set with TTL matching token expiration
        client.setex(key, expires_in_seconds, "1")
        return True
    except redis.RedisError as e:
        logger.warning("Redis error blacklisting token: %s", e)
        return False
