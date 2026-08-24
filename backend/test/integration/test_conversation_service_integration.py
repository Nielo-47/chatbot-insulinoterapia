import unittest
from typing import Dict, List, Optional

from backend.src.application.features.chat.conversation_service import ConversationService
from backend.src.infrastructure.repositories.conversations_repository import ConversationsRepository
from backend.src.infrastructure.repositories.messages_repository import MessagesRepository
from backend.test.integration.fake_pb_client import FakePocketBaseClient


class TrackingCache:
    def __init__(self) -> None:
        self.store: Dict[str, List[Dict[str, object]]] = {}
        self.get_calls = 0
        self.set_calls = 0
        self.invalidate_calls = 0

    def get_messages(self, conversation_id: str) -> Optional[List[Dict[str, object]]]:
        self.get_calls += 1
        return self.store.get(conversation_id)

    def set_messages(self, conversation_id: str, messages: List[Dict[str, object]]) -> None:
        self.set_calls += 1
        self.store[conversation_id] = list(messages)

    def invalidate(self, conversation_id: str) -> None:
        self.invalidate_calls += 1
        self.store.pop(conversation_id, None)


class ConversationServiceIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = FakePocketBaseClient()
        self.conversations = ConversationsRepository(self.client)
        self.cache = TrackingCache()
        self.messages = MessagesRepository(client=self.client, cache=self.cache)
        self.service = ConversationService(
            conversations_repository=self.conversations,
            messages_repository=self.messages,
        )

    def _new_user(self, user_id: str) -> str:
        return user_id

    def test_add_messages_and_cache_warmup(self) -> None:
        user_id = self._new_user("useraaaaaaaaaaaaa")

        self.service.add_message(user_id, "user", "oi")
        self.service.add_message(user_id, "assistant", "olá")

        first_read = self.service.get_conversation(user_id)
        second_read = self.service.get_conversation(user_id)

        self.assertEqual(
            first_read,
            [
                {"role": "user", "content": "oi", "sources": []},
                {"role": "assistant", "content": "olá", "sources": []},
            ],
        )
        self.assertEqual(second_read, first_read)
        self.assertEqual(self.cache.set_calls, 1)
        self.assertGreaterEqual(self.cache.get_calls, 2)
        self.assertEqual(self.cache.invalidate_calls, 2)

    def test_reset_conversation_clears_messages_and_invalidates_cache(self) -> None:
        user_id = self._new_user("userbbbbbbbbbbbbb")

        self.service.add_message(user_id, "user", "primeira")
        self.service.add_message(user_id, "assistant", "resposta")
        self.service.get_conversation(user_id)

        was_cleared = self.service.reset_conversation(user_id)

        self.assertTrue(was_cleared)
        self.assertEqual(self.service.count_messages(user_id), 0)
        self.assertEqual(self.cache.invalidate_calls, 3)
        self.assertEqual(self.service.get_conversation(user_id), [])

    def test_purge_user_data_invalidates_cache_while_account_exists(self) -> None:
        user_id = "userccccccccccccc"
        # The users record exists independently of the conversation; create it
        # so the CascadeDelete simulation below has something to delete.
        self.client.create_record("users", {"id": user_id, "email": "carol@example.com"})

        self.service.add_message(user_id, "user", "pergunta")
        self.service.add_message(user_id, "assistant", "resposta")

        conversation_id = self.conversations.get_conversation_id_by_user(user_id)
        self.assertIsNotNone(conversation_id)
        assert conversation_id is not None

        # purge_user_data runs before the account is revoked, so the cached PII
        # is dropped while the conversation record still exists and can be resolved.
        self.service.purge_user_data(user_id)
        self.assertEqual(self.cache.invalidate_calls, 3)
        self.assertEqual(self.cache.get_messages(conversation_id), None)

        # The PocketBase CascadeDelete relation performs the equivalent of the
        # old profile delete: removing the user removes everything below it.
        deleted_account = self.client.delete_record("users", user_id)
        self.assertTrue(deleted_account)
        self.assertEqual(self.messages.count_messages(conversation_id), 0)


if __name__ == "__main__":
    unittest.main()
