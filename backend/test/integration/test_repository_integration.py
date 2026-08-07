import unittest
import uuid

from backend.src.infrastructure.data.models import Base
from backend.src.infrastructure.repositories.conversations_repository import ConversationsRepository
from backend.src.infrastructure.repositories.messages_repository import MessagesRepository
from backend.src.infrastructure.repositories.profiles_repository import ProfilesRepository
from backend.test.integration.db_test_utils import (
    bind_session_to_schema,
    create_isolated_test_engine,
    drop_isolated_schema,
)


class RepositoryIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.engine, cls.schema_name = create_isolated_test_engine()
        cls.schema_engine = bind_session_to_schema(cls.engine, cls.schema_name)
        cls.profiles = ProfilesRepository()
        cls.conversations = ConversationsRepository()
        cls.messages = MessagesRepository()

        Base.metadata.create_all(bind=cls.schema_engine)

    def setUp(self) -> None:
        Base.metadata.drop_all(bind=self.schema_engine)
        Base.metadata.create_all(bind=self.schema_engine)

    @classmethod
    def tearDownClass(cls) -> None:
        drop_isolated_schema(cls.engine, cls.schema_name)
        cls.engine.dispose()

    def test_create_profile_conversation_and_messages(self) -> None:
        user_id = uuid.uuid4()
        user_id, created = self.profiles.get_or_create_profile(user_id, "alice")
        conversation_id = self.conversations.get_or_create_conversation_id(user_id)
        profile = self.profiles.get_profile_by_id(user_id)

        self.messages.add_message(conversation_id, "user", "oi")
        self.messages.add_message(conversation_id, "assistant", "olá", sources=["doc-a.md", "doc-b.md"])

        self.assertTrue(created)
        self.assertIsNotNone(profile)
        assert profile is not None
        self.assertEqual(profile.user_id, user_id)
        self.assertEqual(profile.username, "alice")
        self.assertEqual(self.conversations.get_conversation_id_by_user(user_id), conversation_id)
        self.assertEqual(self.messages.count_messages(conversation_id), 2)
        self.assertEqual(
            self.messages.list_recent_messages(conversation_id, limit=10),
            [
                {"role": "user", "content": "oi", "sources": []},
                {
                    "role": "assistant",
                    "content": "olá",
                    "sources": [
                        {"path": "doc-a.md", "page": None, "excerpt": None},
                        {"path": "doc-b.md", "page": None, "excerpt": None},
                    ],
                },
            ],
        )

    def test_clear_conversation_removes_only_messages(self) -> None:
        user_id, _ = self.profiles.get_or_create_profile(uuid.uuid4(), "bob")
        conversation_id = self.conversations.get_or_create_conversation_id(user_id)

        self.messages.add_message(conversation_id, "user", "primeira")
        self.messages.add_message(conversation_id, "assistant", "resposta")

        cleared = self.messages.clear_conversation(conversation_id)

        self.assertEqual(cleared, 2)
        self.assertEqual(self.messages.count_messages(conversation_id), 0)
        self.assertEqual(self.conversations.get_conversation_id_by_user(user_id), conversation_id)

    def test_delete_profile_cascades_conversation_and_messages(self) -> None:
        user_id, _ = self.profiles.get_or_create_profile(uuid.uuid4(), "carol")
        conversation_id = self.conversations.get_or_create_conversation_id(user_id)

        self.messages.add_message(conversation_id, "user", "pergunta")
        self.messages.add_message(conversation_id, "assistant", "resposta")

        deleted = self.profiles.delete_profile(user_id)

        self.assertTrue(deleted)
        self.assertIsNone(self.profiles.get_profile_by_id(user_id))
        self.assertIsNone(self.conversations.get_conversation_id_by_user(user_id))
        self.assertEqual(self.messages.count_messages(conversation_id), 0)

    def test_duplicate_user_id_returns_same_profile(self) -> None:
        user_id = uuid.uuid4()
        first_id, first_created = self.profiles.get_or_create_profile(user_id, "dana")
        second_id, second_created = self.profiles.get_or_create_profile(user_id, "dana")

        self.assertEqual(first_id, second_id)
        self.assertTrue(first_created)
        self.assertFalse(second_created)

    def test_get_profile_by_id_returns_persisted_profile(self) -> None:
        user_id, _ = self.profiles.get_or_create_profile(uuid.uuid4(), "eve")

        profile = self.profiles.get_profile_by_id(user_id)

        self.assertIsNotNone(profile)
        assert profile is not None
        self.assertEqual(profile.user_id, user_id)
        self.assertEqual(profile.username, "eve")

    def test_delete_profile_returns_false_for_missing_profile(self) -> None:
        deleted = self.profiles.delete_profile(uuid.uuid4())

        self.assertFalse(deleted)


if __name__ == "__main__":
    unittest.main()
