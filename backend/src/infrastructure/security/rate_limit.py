"""Rate limiting and account lockout functionality using Redis."""

import logging
import time
from typing import Optional

import redis

from backend.src.config.infrastructure import CHAT_CACHE_REDIS_URL, TOKEN_BLACKLIST_ENABLED, TOKEN_BLACKLIST_PREFIX
from backend.src.config.security import (
    LOCKOUT_DURATION_SECONDS,
    LOGIN_RATE_LIMIT_BLOCK_DURATION_SECONDS,
    MAX_LOGIN_ATTEMPTS,
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
    client = _get_redis_client()
    key = _get_rate_limit_key(ip_address)
    
    try:
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
        logger.warning(f"Redis error in rate limiting: {e}")
        # On Redis error, allow the request (fail open)
        return True, 999


def reset_rate_limit(ip_address: str) -> None:
    """Reset rate limit for an IP address after successful login."""
    client = _get_redis_client()
    key = _get_rate_limit_key(ip_address)
    
    try:
        client.delete(key)
    except redis.RedisError as e:
        logger.warning(f"Redis error resetting rate limit: {e}")


def get_rate_limit_remaining_seconds(ip_address: str) -> int:
    """Get remaining seconds until rate limit resets."""
    client = _get_redis_client()
    key = _get_rate_limit_key(ip_address)
    
    try:
        ttl = client.ttl(key)
        return max(0, ttl)
    except redis.RedisError:
        return 0


# ==================== Account Lockout ====================

LOCKOUT_PREFIX = "lockout:user:"


def _get_lockout_key(user_id: int) -> str:
    """Get Redis key for user lockout."""
    return f"{LOCKOUT_PREFIX}{user_id}"


def check_account_lockout(user_id: int) -> tuple[bool, Optional[int]]:
    """
    Check if a user account is locked out.
    
    Returns:
        tuple: (is_locked, remaining_seconds)
    """
    client = _get_redis_client()
    key = _get_lockout_key(user_id)
    
    try:
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
        logger.warning(f"Redis error checking account lockout: {e}")
        # On Redis error, fail open (allow)
        return False, None


def record_failed_login(user_id: int) -> int:
    """
    Record a failed login attempt.
    
    Returns:
        int: Number of consecutive failed attempts after this one
    """
    client = _get_redis_client()
    key = _get_lockout_key(user_id)
    
    # First check if already locked
    locked_until = client.get(key)
    if locked_until is not None:
        return MAX_LOGIN_ATTEMPTS  # Already locked
    
    # Use a separate counter for failed attempts
    attempts_key = f"{LOCKOUT_PREFIX}attempts:{user_id}"
    
    try:
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
            client.setex(key, LOCKOUT_DURATION_SECONDS, lockout_until)
            # Reset attempts counter
            client.delete(attempts_key)
            logger.warning(f"User {user_id} locked out after {attempts} failed login attempts")
            return attempts
        
        # Increment failed attempts counter with expiry
        client.setex(attempts_key, LOCKOUT_DURATION_SECONDS, attempts)
        return attempts
        
    except redis.RedisError as e:
        logger.warning(f"Redis error recording failed login: {e}")
        return 0


def clear_failed_login_attempts(user_id: int) -> None:
    """Clear failed login attempts after successful login."""
    client = _get_redis_client()
    attempts_key = f"{LOCKOUT_PREFIX}attempts:{user_id}"
    
    try:
        client.delete(attempts_key)
    except redis.RedisError as e:
        logger.warning(f"Redis error clearing failed login attempts: {e}")


def unlock_account(user_id: int) -> None:
    """Manually unlock a user account."""
    client = _get_redis_client()
    key = _get_lockout_key(user_id)
    attempts_key = f"{LOCKOUT_PREFIX}attempts:{user_id}"
    
    try:
        client.delete(key)
        client.delete(attempts_key)
        logger.info(f"User {user_id} account unlocked")
    except redis.RedisError as e:
        logger.warning(f"Redis error unlocking account: {e}")


# ==================== Token Blacklist ====================

def is_token_blacklisted(token_jti: str) -> bool:
    """Check if a token is blacklisted (for logout)."""
    if not TOKEN_BLACKLIST_ENABLED:
        return False
    
    client = _get_redis_client()
    key = f"{TOKEN_BLACKLIST_PREFIX}{token_jti}"
    
    try:
        return client.exists(key) > 0
    except redis.RedisError as e:
        logger.warning(f"Redis error checking token blacklist: {e}")
        return False


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
    
    client = _get_redis_client()
    key = f"{TOKEN_BLACKLIST_PREFIX}{token_jti}"
    
    try:
        # Set with TTL matching token expiration
        client.setex(key, expires_in_seconds, "1")
        return True
    except redis.RedisError as e:
        logger.warning(f"Redis error blacklisting token: {e}")
        return False