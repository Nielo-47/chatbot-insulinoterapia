import time
import unittest
import fnmatch
from unittest.mock import patch

import redis

from backend.src.infrastructure.security import rate_limit
from backend.src.infrastructure.security.rate_limit import (
    check_rate_limit,
    check_query_rate_limit,
    check_account_lockout,
    record_failed_login,
    clear_failed_login_attempts,
    is_token_blacklisted,
    blacklist_token,
    reset_rate_limit,
    build_lockout_identity,
    unlock_account,
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

    def delete(self, key: str) -> int:
        if key in self.store:
            del self.store[key]
            return 1
        return 0

    def exists(self, key: str) -> int:
        return 1 if key in self.store else 0

    def scan_iter(self, match: str, count: int = 100):
        for key in list(self.store.keys()):
            if fnmatch.fnmatch(str(key), match):
                yield str(key)


class RateLimitFailClosedTests(unittest.TestCase):
    """When Redis is unavailable every check must deny (fail closed)."""

    def _patch_redis_unavailable(self):
        return patch.object(
            rate_limit,
            "_get_redis_client",
            side_effect=redis.exceptions.ConnectionError("redis down"),
        )

    def test_check_rate_limit_fails_closed_on_redis_error(self) -> None:
        with self._patch_redis_unavailable():
            is_allowed, remaining = check_rate_limit("10.0.0.9")

        self.assertFalse(is_allowed)
        self.assertEqual(remaining, 0)

    def test_check_query_rate_limit_fails_closed_on_redis_error(self) -> None:
        with self._patch_redis_unavailable():
            is_allowed, remaining = check_query_rate_limit(7)

        self.assertFalse(is_allowed)
        self.assertEqual(remaining, 0)

    def test_check_account_lockout_fails_closed_on_redis_error(self) -> None:
        with self._patch_redis_unavailable():
            is_locked, remaining = check_account_lockout("alice|10.0.0.9")

        self.assertTrue(is_locked)
        self.assertEqual(remaining, 0)

    def test_record_failed_login_fails_closed_on_redis_error(self) -> None:
        with self._patch_redis_unavailable():
            attempts = record_failed_login("alice|10.0.0.9")

        self.assertGreaterEqual(attempts, 1)

    def test_is_token_blacklisted_fails_closed_on_redis_error(self) -> None:
        with self._patch_redis_unavailable():
            result = is_token_blacklisted("some-jti")

        self.assertTrue(result)

    def test_blacklist_token_fails_safely_on_redis_error(self) -> None:
        with self._patch_redis_unavailable():
            result = blacklist_token("some-jti", 300)

        self.assertFalse(result)

    def test_clear_and_reset_do_not_raise_on_redis_error(self) -> None:
        with self._patch_redis_unavailable():
            clear_failed_login_attempts("alice|10.0.0.9")
            reset_rate_limit("10.0.0.9")
            unlock_account("alice")


class RateLimitBehaviorTests(unittest.TestCase):
    def test_check_rate_limit_first_attempt_allowed(self) -> None:
        fake = FakeRedis()
        with patch.object(rate_limit, "_get_redis_client", return_value=fake):
            is_allowed, remaining = check_rate_limit("10.0.0.1")

        self.assertTrue(is_allowed)
        self.assertEqual(remaining, 4)

    def test_check_rate_limit_blocks_when_exceeded(self) -> None:
        fake = FakeRedis()
        fake.store[rate_limit._get_rate_limit_key("10.0.0.1")] = "5"
        with patch.object(rate_limit, "_get_redis_client", return_value=fake):
            is_allowed, remaining = check_rate_limit("10.0.0.1")

        self.assertFalse(is_allowed)
        self.assertEqual(remaining, 0)

    def test_check_rate_limit_blocks_at_6th_attempt(self) -> None:
        fake = FakeRedis()
        with patch.object(rate_limit, "_get_redis_client", return_value=fake):
            results = [check_rate_limit("10.0.0.2") for _ in range(6)]

        self.assertTrue(all(result[0] for result in results[:5]))
        self.assertFalse(results[5][0])

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
            is_allowed, _ = check_query_rate_limit(8)

        self.assertFalse(is_allowed)

    def test_account_lockout_reports_locked(self) -> None:
        fake = FakeRedis()
        identity = build_lockout_identity("Alice", "10.0.0.3")
        fake.store[rate_limit._get_lockout_key(identity)] = str(int(time.time()) + 600)
        with patch.object(rate_limit, "_get_redis_client", return_value=fake):
            is_locked, remaining = check_account_lockout(identity)

        self.assertTrue(is_locked)
        self.assertGreater(remaining, 0)

    def test_account_lockout_cleared_when_expired(self) -> None:
        fake = FakeRedis()
        identity = build_lockout_identity("bob", "10.0.0.4")
        fake.store[rate_limit._get_lockout_key(identity)] = str(int(time.time()) - 10)
        with patch.object(rate_limit, "_get_redis_client", return_value=fake):
            is_locked, _ = check_account_lockout(identity)

        self.assertFalse(is_locked)

    def test_record_failed_login_locks_after_max_attempts(self) -> None:
        fake = FakeRedis()
        identity = build_lockout_identity("carol", "10.0.0.5")
        with patch.object(rate_limit, "_get_redis_client", return_value=fake):
            attempts = []
            for _ in range(rate_limit.MAX_LOGIN_ATTEMPTS):
                attempts.append(record_failed_login(identity))

        self.assertEqual(attempts[-1], rate_limit.MAX_LOGIN_ATTEMPTS)
        self.assertIsNotNone(fake.store.get(rate_limit._get_lockout_key(identity)))
        self.assertNotIn(rate_limit._get_attempts_key(identity), fake.store)

    def test_clear_failed_login_attempts_removes_counter(self) -> None:
        fake = FakeRedis()
        identity = build_lockout_identity("dave", "10.0.0.6")
        fake.store[rate_limit._get_attempts_key(identity)] = "3"
        with patch.object(rate_limit, "_get_redis_client", return_value=fake):
            clear_failed_login_attempts(identity)

        self.assertNotIn(rate_limit._get_attempts_key(identity), fake.store)

    def test_build_lockout_identity_normalizes_username(self) -> None:
        self.assertEqual(
            build_lockout_identity("  Alice ", "10.0.0.7"),
            build_lockout_identity("alice", "10.0.0.7"),
        )

    def test_unlock_account_removes_all_user_keys(self) -> None:
        fake = FakeRedis()
        fake.store[rate_limit._get_lockout_key("eve|10.0.0.8")] = "12345"
        fake.store[rate_limit._get_attempts_key("eve|10.0.0.9")] = "2"
        fake.store[rate_limit._get_attempts_key("other|10.0.0.9")] = "1"
        with patch.object(rate_limit, "_get_redis_client", return_value=fake):
            unlock_account("EVE")

        self.assertNotIn(rate_limit._get_lockout_key("eve|10.0.0.8"), fake.store)
        self.assertNotIn(rate_limit._get_attempts_key("eve|10.0.0.9"), fake.store)
        self.assertIn(rate_limit._get_attempts_key("other|10.0.0.9"), fake.store)

    def test_is_token_blacklisted_disabled_returns_false(self) -> None:
        fake = FakeRedis()
        with patch.object(rate_limit, "TOKEN_BLACKLIST_ENABLED", 0), patch.object(
            rate_limit, "_get_redis_client", return_value=fake
        ):
            result = is_token_blacklisted("any-jti")

        self.assertFalse(result)

    def test_is_token_blacklisted_returns_true_when_present(self) -> None:
        fake = FakeRedis()
        fake.store["token:blacklist:known-jti"] = "1"
        with patch.object(rate_limit, "_get_redis_client", return_value=fake):
            result = is_token_blacklisted("known-jti")

        self.assertTrue(result)

    def test_blacklist_token_sets_key(self) -> None:
        fake = FakeRedis()
        with patch.object(rate_limit, "_get_redis_client", return_value=fake):
            result = blacklist_token("new-jti", 300)

        self.assertTrue(result)
        self.assertEqual(fake.store.get("token:blacklist:new-jti"), "1")


if __name__ == "__main__":
    unittest.main()
