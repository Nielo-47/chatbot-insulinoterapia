import unittest
from unittest.mock import patch

from backend.src.infrastructure.repositories.messages_repository import MessagesRepository


class InMemoryCache:
    def __init__(self) -> None:
        self.messages = {}
        self.invalidated = []

    def get_messages(self, conversation_id: int):
        return self.messages.get(conversation_id)

    def set_messages(self, conversation_id: int, messages):
        self.messages[conversation_id] = messages

    def invalidate(self, conversation_id: int):
        self.invalidated.append(conversation_id)
        self.messages.pop(conversation_id, None)


class MessagesRepositoryCacheModeTests(unittest.TestCase):
    def test_guest_mode_uses_cache_without_db(self) -> None:
        cache = InMemoryCache()
        repository = MessagesRepository(cache=cache)

        with patch.object(
            __import__("backend.src.infrastructure.repositories.messages_repository", fromlist=["AUTH_ENABLED"]),
            "AUTH_ENABLED",
            False,
        ):
            with patch("backend.src.infrastructure.repositories.messages_repository.get_db_session") as db_session:
                repository.add_message(7, "user", "oi")
                repository.add_message(7, "assistant", "olá", sources=[{"path": "doc.md"}])

                self.assertFalse(db_session.called)
                self.assertEqual(
                    repository.list_recent_messages(7, limit=10),
                    [
                        {"role": "user", "content": "oi", "sources": []},
                        {"role": "assistant", "content": "olá", "sources": [{"path": "doc.md"}]},
                    ],
                )
                self.assertEqual(repository.count_messages(7), 2)

                cleared = repository.clear_conversation(7)

        self.assertEqual(cleared, 2)
        self.assertEqual(repository.count_messages(7), 0)
        self.assertEqual(cache.invalidated, [7])


if __name__ == "__main__":
    unittest.main()