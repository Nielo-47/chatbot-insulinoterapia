import json
import unittest
from unittest.mock import patch

from backend.src.cache.conversation_cache import ConversationCache


class FakeRedisClient:
    def __init__(self) -> None:
        self.store = {}
        self.ping_called = False
        self.set_calls = 0
        self.delete_calls = 0

    def ping(self) -> None:
        self.ping_called = True

    def get(self, key: str):
        return self.store.get(key)

    def set(self, key: str, value: str, ex: int):
        self.store[key] = value
        self.set_calls += 1

    def delete(self, key: str):
        self.store.pop(key, None)
        self.delete_calls += 1


class ConversationCacheTests(unittest.TestCase):
    def test_cache_round_trip_and_invalidate(self) -> None:
        fake_client = FakeRedisClient()

        with patch("redis.Redis.from_url", return_value=fake_client):
            cache = ConversationCache("redis://localhost:6379/1", ttl_seconds=60, key_prefix="chat:test")

        messages = [{"role": "user", "content": "oi"}, {"role": "assistant", "content": "olá"}]
        cache.set_messages(1, messages)

        self.assertEqual(fake_client.set_calls, 1)
        self.assertEqual(cache.get_messages(1), messages)

        cache.invalidate(1)

        self.assertEqual(fake_client.delete_calls, 1)
        self.assertIsNone(cache.get_messages(1))

    def test_cache_falls_back_when_redis_unavailable(self) -> None:
        with patch("redis.Redis.from_url", side_effect=RuntimeError("redis down")):
            cache = ConversationCache("redis://localhost:6379/1", ttl_seconds=60, key_prefix="chat:test")

        self.assertIsNone(cache.get_messages(1))
        cache.set_messages(1, [{"role": "user", "content": "oi"}])
        cache.invalidate(1)
        self.assertIsNone(cache.get_messages(1))


if __name__ == "__main__":
    unittest.main()