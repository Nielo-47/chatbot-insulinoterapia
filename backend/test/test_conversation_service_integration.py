import unittest
from typing import Dict, List, Optional

from backend.src.infrastructure.data.models import Base
from backend.src.infrastructure.repositories.conversations_repository import ConversationsRepository
from backend.src.infrastructure.repositories.messages_repository import MessagesRepository
from backend.src.infrastructure.repositories.users_repository import UsersRepository
from backend.src.application.features.chat.conversation_service import ConversationService
from backend.test.db_test_utils import bind_session_to_schema, create_isolated_test_engine, drop_isolated_schema


class TrackingCache:
    def __init__(self) -> None:
        self.store: Dict[int, List[Dict[str, str]]] = {}
        self.get_calls = 0
        self.set_calls = 0
        self.invalidate_calls = 0

    def get_messages(self, conversation_id: int) -> Optional[List[Dict[str, str]]]:
        self.get_calls += 1
        return self.store.get(conversation_id)

    def set_messages(self, conversation_id: int, messages: List[Dict[str, str]]) -> None:
        self.set_calls += 1
        self.store[conversation_id] = list(messages)

    def invalidate(self, conversation_id: int) -> None:
        self.invalidate_calls += 1
        self.store.pop(conversation_id, None)


class ConversationServiceIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.engine, cls.schema_name = create_isolated_test_engine()
        cls.schema_engine = bind_session_to_schema(cls.engine, cls.schema_name)
        Base.metadata.create_all(bind=cls.schema_engine)

    def setUp(self) -> None:
        Base.metadata.drop_all(bind=self.schema_engine)
        Base.metadata.create_all(bind=self.schema_engine)

        self.users = UsersRepository()
        self.conversations = ConversationsRepository()
        self.cache = TrackingCache()
        self.messages = MessagesRepository(cache=self.cache)
        self.service = ConversationService(
            users_repository=self.users,
            conversations_repository=self.conversations,
            messages_repository=self.messages,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        drop_isolated_schema(cls.engine, cls.schema_name)
        cls.engine.dispose()

    def test_add_messages_and_cache_warmup(self) -> None:
        user_id = self.users.get_or_create_user_id("alice", "hashed-password")

        self.service.add_message(user_id, "user", "oi")
        self.service.add_message(user_id, "assistant", "olá")

        first_read = self.service.get_conversation(user_id)
        second_read = self.service.get_conversation(user_id)

        self.assertEqual(first_read, [{"role": "user", "content": "oi"}, {"role": "assistant", "content": "olá"}])
        self.assertEqual(second_read, first_read)
        self.assertEqual(self.cache.set_calls, 1)
        self.assertGreaterEqual(self.cache.get_calls, 2)
        self.assertEqual(self.cache.invalidate_calls, 2)

    def test_reset_conversation_clears_messages_and_invalidates_cache(self) -> None:
        user_id = self.users.get_or_create_user_id("bob", "hashed-password")

        self.service.add_message(user_id, "user", "primeira")
        self.service.add_message(user_id, "assistant", "resposta")
        self.service.get_conversation(user_id)

        was_cleared = self.service.reset_conversation(user_id)

        self.assertTrue(was_cleared)
        self.assertEqual(self.service.count_messages(user_id), 0)
        self.assertEqual(self.cache.invalidate_calls, 3)
        self.assertEqual(self.service.get_conversation(user_id), [])

    def test_delete_user_removes_all_data(self) -> None:
        user_id = self.users.get_or_create_user_id("carol", "hashed-password")

        self.service.add_message(user_id, "user", "pergunta")
        self.service.add_message(user_id, "assistant", "resposta")

        conversation_id = self.conversations.get_conversation_id_by_user(user_id)
        self.assertIsNotNone(conversation_id)
        assert conversation_id is not None

        deleted = self.service.delete_user(user_id)

        self.assertTrue(deleted)
        self.assertIsNone(self.users.get_user_by_id(user_id))
        self.assertIsNone(self.conversations.get_conversation_id_by_user(user_id))
        self.assertEqual(self.messages.count_messages(conversation_id), 0)
        self.assertEqual(self.cache.invalidate_calls, 2)


if __name__ == "__main__":
    unittest.main()