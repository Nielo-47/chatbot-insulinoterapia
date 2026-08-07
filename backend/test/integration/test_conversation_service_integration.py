import unittest
import uuid
from typing import Dict, List, Optional

from backend.src.infrastructure.data.models import Base
from backend.src.infrastructure.repositories.conversations_repository import ConversationsRepository
from backend.src.infrastructure.repositories.messages_repository import MessagesRepository
from backend.src.infrastructure.repositories.profiles_repository import ProfilesRepository
from backend.src.application.features.chat.conversation_service import ConversationService
from backend.test.integration.db_test_utils import (
    bind_session_to_schema,
    create_isolated_test_engine,
    drop_isolated_schema,
)


class TrackingCache:
    def __init__(self) -> None:
        self.store: Dict[uuid.UUID, List[Dict[str, object]]] = {}
        self.get_calls = 0
        self.set_calls = 0
        self.invalidate_calls = 0

    def get_messages(self, conversation_id: uuid.UUID) -> Optional[List[Dict[str, object]]]:
        self.get_calls += 1
        return self.store.get(conversation_id)

    def set_messages(self, conversation_id: uuid.UUID, messages: List[Dict[str, object]]) -> None:
        self.set_calls += 1
        self.store[conversation_id] = list(messages)

    def invalidate(self, conversation_id: uuid.UUID) -> None:
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

        self.profiles = ProfilesRepository()
        self.conversations = ConversationsRepository()
        self.cache = TrackingCache()
        self.messages = MessagesRepository(cache=self.cache)
        self.service = ConversationService(
            conversations_repository=self.conversations,
            messages_repository=self.messages,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        drop_isolated_schema(cls.engine, cls.schema_name)
        cls.engine.dispose()

    def _new_user(self, username: str) -> uuid.UUID:
        user_id, _ = self.profiles.get_or_create_profile(uuid.uuid4(), username)
        return user_id

    def test_add_messages_and_cache_warmup(self) -> None:
        user_id = self._new_user("alice")

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
        user_id = self._new_user("bob")

        self.service.add_message(user_id, "user", "primeira")
        self.service.add_message(user_id, "assistant", "resposta")
        self.service.get_conversation(user_id)

        was_cleared = self.service.reset_conversation(user_id)

        self.assertTrue(was_cleared)
        self.assertEqual(self.service.count_messages(user_id), 0)
        self.assertEqual(self.cache.invalidate_calls, 3)
        self.assertEqual(self.service.get_conversation(user_id), [])

    def test_purge_user_data_invalidates_cache_before_profile_delete(self) -> None:
        user_id = self._new_user("carol")

        self.service.add_message(user_id, "user", "pergunta")
        self.service.add_message(user_id, "assistant", "resposta")

        conversation_id = self.conversations.get_conversation_id_by_user(user_id)
        self.assertIsNotNone(conversation_id)
        assert conversation_id is not None

        # purge_user_data runs before the account is revoked, so the cached PII
        # is dropped while the conversation row still exists and can be resolved.
        self.service.purge_user_data(user_id)
        self.assertEqual(self.cache.invalidate_calls, 3)
        self.assertEqual(self.cache.get_messages(conversation_id), None)

        deleted = self.profiles.delete_profile(user_id)

        self.assertTrue(deleted)
        self.assertIsNone(self.profiles.get_profile_by_id(user_id))
        self.assertIsNone(self.conversations.get_conversation_id_by_user(user_id))
        self.assertEqual(self.messages.count_messages(conversation_id), 0)


if __name__ == "__main__":
    unittest.main()
