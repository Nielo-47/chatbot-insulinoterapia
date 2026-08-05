import unittest

from backend.src.infrastructure.data.models import Base
from backend.src.infrastructure.repositories.conversations_repository import ConversationsRepository
from backend.src.infrastructure.repositories.messages_repository import MessagesRepository
from backend.src.infrastructure.repositories.users_repository import UsersRepository
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
        cls.users = UsersRepository()
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

    def test_create_user_conversation_and_messages(self) -> None:
        sub = "authentik-sub-alice"
        user_id, created = self.users.get_or_create_user_by_sub(sub, "alice")
        conversation_id = self.conversations.get_or_create_conversation_id(user_id)
        user = self.users.get_user_by_sub(sub)

        self.messages.add_message(conversation_id, "user", "oi")
        self.messages.add_message(conversation_id, "assistant", "olá", sources=["doc-a.md", "doc-b.md"])

        self.assertTrue(created)
        self.assertIsNotNone(user)
        assert user is not None
        self.assertEqual(user.id, user_id)
        self.assertEqual(user.username, "alice")
        self.assertEqual(user.authentik_sub, sub)
        self.assertEqual(self.conversations.get_conversation_id_by_user(user_id), conversation_id)
        self.assertEqual(self.messages.count_messages(conversation_id), 2)
        self.assertEqual(
            self.messages.list_recent_messages(conversation_id, limit=10),
            [
                {"role": "user", "content": "oi", "sources": []},
                {"role": "assistant", "content": "olá", "sources": ["doc-a.md", "doc-b.md"]},
            ],
        )

    def test_clear_conversation_removes_only_messages(self) -> None:
        user_id, _ = self.users.get_or_create_user_by_sub("authentik-sub-bob", "bob")
        conversation_id = self.conversations.get_or_create_conversation_id(user_id)

        self.messages.add_message(conversation_id, "user", "primeira")
        self.messages.add_message(conversation_id, "assistant", "resposta")

        cleared = self.messages.clear_conversation(conversation_id)

        self.assertEqual(cleared, 2)
        self.assertEqual(self.messages.count_messages(conversation_id), 0)
        self.assertEqual(self.conversations.get_conversation_id_by_user(user_id), conversation_id)

    def test_delete_user_cascades_conversation_and_messages(self) -> None:
        sub = "authentik-sub-carol"
        user_id, _ = self.users.get_or_create_user_by_sub(sub, "carol")
        conversation_id = self.conversations.get_or_create_conversation_id(user_id)

        self.messages.add_message(conversation_id, "user", "pergunta")
        self.messages.add_message(conversation_id, "assistant", "resposta")

        deleted = self.users.delete_user_by_id(user_id)

        self.assertTrue(deleted)
        self.assertIsNone(self.users.get_user_by_sub(sub))
        self.assertIsNone(self.conversations.get_conversation_id_by_user(user_id))
        self.assertEqual(self.messages.count_messages(conversation_id), 0)

    def test_duplicate_sub_returns_same_user(self) -> None:
        first_id, first_created = self.users.get_or_create_user_by_sub("authentik-sub-dana", "dana")
        second_id, second_created = self.users.get_or_create_user_by_sub("authentik-sub-dana", "dana")

        self.assertEqual(first_id, second_id)
        self.assertTrue(first_created)
        self.assertFalse(second_created)

    def test_get_user_by_id_returns_persisted_user(self) -> None:
        user_id, _ = self.users.get_or_create_user_by_sub("authentik-sub-eve", "eve")

        user = self.users.get_user_by_id(user_id)

        self.assertIsNotNone(user)
        assert user is not None
        self.assertEqual(user.username, "eve")
        self.assertEqual(user.authentik_sub, "authentik-sub-eve")

    def test_legacy_username_row_is_adopted_preserving_conversations(self) -> None:
        """A pre-migration user (authentik_sub NULL, password era) keeps its
        conversation history when the same username first signs in via Authentik."""
        from sqlalchemy.orm import Session

        from backend.src.infrastructure.data.db_client import SessionLocal
        from backend.src.infrastructure.data.models import User as UserModel

        with Session(SessionLocal()) as db:
            legacy = UserModel(username="frank", authentik_sub=None)
            db.add(legacy)
            db.flush()
            legacy_id = legacy.id

        conversation_id = self.conversations.get_or_create_conversation_id(legacy_id)
        self.messages.add_message(conversation_id, "user", "historico antigo")
        self.messages.add_message(conversation_id, "assistant", "resposta antiga")

        user_id, created = self.users.get_or_create_user_by_sub("authentik-sub-frank", "frank")

        self.assertEqual(user_id, legacy_id)
        self.assertTrue(created)
        self.assertEqual(self.messages.count_messages(conversation_id), 2)
        adopted = self.users.get_user_by_sub("authentik-sub-frank")
        self.assertIsNotNone(adopted)
        assert adopted is not None
        self.assertEqual(adopted.id, legacy_id)
        self.assertEqual(adopted.username, "frank")

    def test_delete_user_by_id_returns_false_for_missing_user(self) -> None:
        deleted = self.users.delete_user_by_id(9999)

        self.assertFalse(deleted)


if __name__ == "__main__":
    unittest.main()
