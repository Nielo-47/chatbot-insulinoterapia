import time
import unittest
from unittest.mock import patch

import redis

from backend.src.infrastructure.security import rate_limit
from backend.src.infrastructure.security.rate_limit import (
    check_query_rate_limit,
    get_query_rate_limit_remaining_seconds,
)


class FakeRedis:
    """Minimal in-memory fake covering the Redis surface used by rate_limit."""

    def __init__(self) -> None:
        self.store: dict[str, object] = {}
        self.ttls: dict[str, float] = {}

    def get(self, key: str):
        return self.store.get(key)

    def setex(self, key: str, seconds: int, value) -> None:
        self.store[key] = value
        self.ttls[key] = time.time() + seconds

    def incr(self, key: str) -> str:
        self.store[key] = str(int(self.store.get(key, "0")) + 1)
        return self.store[key]

    def ttl(self, key: str) -> int:
        if key not in self.store:
            return -2
        return max(0, int(self.ttls[key] - time.time()))


class QueryRateLimitFailClosedTests(unittest.TestCase):
    """When Redis is unavailable every check must deny (fail closed)."""

    def _patch_redis_unavailable(self):
        return patch.object(
            rate_limit,
            "_get_redis_client",
            side_effect=redis.exceptions.ConnectionError("redis down"),
        )

    def test_check_query_rate_limit_fails_closed_on_redis_error(self) -> None:
        with self._patch_redis_unavailable():
            is_allowed, remaining = check_query_rate_limit(7)

        self.assertFalse(is_allowed)
        self.assertEqual(remaining, 0)

    def test_get_remaining_seconds_fails_closed_on_redis_error(self) -> None:
        with self._patch_redis_unavailable():
            remaining = get_query_rate_limit_remaining_seconds(7)

        self.assertEqual(remaining, 0)


class QueryRateLimitBehaviorTests(unittest.TestCase):
    def test_check_query_rate_limit_first_attempt_allowed(self) -> None:
        fake = FakeRedis()
        with patch.object(rate_limit, "_get_redis_client", return_value=fake):
            is_allowed, remaining = check_query_rate_limit(1)

        self.assertTrue(is_allowed)
        self.assertEqual(remaining, rate_limit.QUERY_RATE_LIMIT - 1)

    def test_check_query_rate_limit_remaining_decreases(self) -> None:
        fake = FakeRedis()
        with patch.object(rate_limit, "_get_redis_client", return_value=fake):
            _, remaining = check_query_rate_limit(7)
            _, remaining2 = check_query_rate_limit(7)

        self.assertEqual(remaining, rate_limit.QUERY_RATE_LIMIT - 1)
        self.assertEqual(remaining2, rate_limit.QUERY_RATE_LIMIT - 2)

    def test_check_query_rate_limit_blocks_when_exceeded(self) -> None:
        fake = FakeRedis()
        with patch.object(rate_limit, "_get_redis_client", return_value=fake):
            for _ in range(rate_limit.QUERY_RATE_LIMIT):
                check_query_rate_limit(8)
            is_allowed, remaining = check_query_rate_limit(8)

        self.assertFalse(is_allowed)
        self.assertEqual(remaining, 0)

    def test_query_limits_are_per_user(self) -> None:
        """Each user has an independent budget in the same window."""
        fake = FakeRedis()
        with patch.object(rate_limit, "_get_redis_client", return_value=fake):
            for _ in range(rate_limit.QUERY_RATE_LIMIT):
                check_query_rate_limit(9)

            is_allowed, remaining = check_query_rate_limit(10)

        self.assertTrue(is_allowed)
        self.assertEqual(remaining, rate_limit.QUERY_RATE_LIMIT - 1)

    def test_remaining_seconds_reflects_window(self) -> None:
        fake = FakeRedis()
        fake.store[rate_limit._get_query_limit_key(11)] = "1"
        fake.ttls[rate_limit._get_query_limit_key(11)] = time.time() + 42
        with patch.object(rate_limit, "_get_redis_client", return_value=fake):
            remaining = get_query_rate_limit_remaining_seconds(11)

        self.assertGreater(remaining, 0)
        self.assertLessEqual(remaining, 42)

    def test_remaining_seconds_zero_for_unknown_user(self) -> None:
        fake = FakeRedis()
        with patch.object(rate_limit, "_get_redis_client", return_value=fake):
            remaining = get_query_rate_limit_remaining_seconds(12)

        self.assertEqual(remaining, 0)


if __name__ == "__main__":
    unittest.main()
