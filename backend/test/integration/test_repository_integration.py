import unittest

from backend.src.infrastructure.repositories.conversations_repository import ConversationsRepository
from backend.src.infrastructure.repositories.messages_repository import MessagesRepository
from backend.test.integration.fake_pb_client import FakePocketBaseClient


class RepositoryIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = FakePocketBaseClient()
        self.conversations = ConversationsRepository(self.client)
        self.messages = MessagesRepository(client=self.client, cache=_NoopCache())

    def test_create_conversation_and_messages(self) -> None:
        user_id = "useraaaaaaaaaaaaa"
        conversation_id = self.conversations.get_or_create_conversation_id(user_id)

        self.messages.add_message(conversation_id, "user", "oi")
        self.messages.add_message(conversation_id, "assistant", "olá", sources=["doc-a.md", "doc-b.md"])

        self.assertEqual(
            self.conversations.get_conversation_id_by_user(user_id), conversation_id
        )
        self.assertEqual(self.messages.count_messages(conversation_id), 2)
        self.assertEqual(
            self.messages.list_recent_messages(conversation_id, limit=10),
            [
                {"role": "user", "content": "oi", "sources": []},
                {
                    "role": "assistant",
                    "content": "olá",
                    "sources": [
                        {"path": "doc-a.md", "page": None, "content": None},
                        {"path": "doc-b.md", "page": None, "content": None},
                    ],
                },
            ],
        )

    def test_clear_conversation_removes_only_messages(self) -> None:
        user_id = "userbbbbbbbbbbbbb"
        conversation_id = self.conversations.get_or_create_conversation_id(user_id)

        self.messages.add_message(conversation_id, "user", "primeira")
        self.messages.add_message(conversation_id, "assistant", "resposta")

        cleared = self.messages.clear_conversation(conversation_id)

        self.assertEqual(cleared, 2)
        self.assertEqual(self.messages.count_messages(conversation_id), 0)
        self.assertEqual(
            self.conversations.get_conversation_id_by_user(user_id), conversation_id
        )

    def test_list_recent_messages_respects_limit_and_order(self) -> None:
        user_id = "userccccccccccccc"
        conversation_id = self.conversations.get_or_create_conversation_id(user_id)

        for i in range(5):
            self.messages.add_message(conversation_id, "user", f"msg-{i}")

        history = self.messages.list_recent_messages(conversation_id, limit=3)

        self.assertEqual([m["content"] for m in history], ["msg-2", "msg-3", "msg-4"])

    def test_get_or_create_is_unique_per_user(self) -> None:
        user_id = "userddddddddddddd"
        first = self.conversations.get_or_create_conversation_id(user_id)
        second = self.conversations.get_or_create_conversation_id(user_id)

        self.assertEqual(first, second)
        self.assertEqual(len(self.client.collections["conversations"]), 1)


class _NoopCache:
    """Cache double: always misses, records invalidations."""

    def __init__(self) -> None:
        self.invalidated: list[str] = []

    def get_messages(self, conversation_id):
        return None

    def set_messages(self, conversation_id, messages):
        pass

    def invalidate(self, conversation_id):
        self.invalidated.append(conversation_id)


if __name__ == "__main__":
    unittest.main()
