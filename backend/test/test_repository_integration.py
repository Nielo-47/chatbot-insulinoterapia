import unittest

from backend.src.db.models import Base
from backend.src.repositories.conversations_repository import ConversationsRepository
from backend.src.repositories.messages_repository import MessagesRepository
from backend.src.repositories.users_repository import UsersRepository
from backend.test.db_test_utils import bind_session_to_schema, create_isolated_test_engine, drop_isolated_schema


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
        user_id = self.users.get_or_create_user_id("alice", "hashed-password")
        conversation_id = self.conversations.get_or_create_conversation_id(user_id)

        self.messages.add_message(conversation_id, "user", "oi")
        self.messages.add_message(conversation_id, "assistant", "olá")

        self.assertEqual(self.users.get_user_id("alice"), user_id)
        self.assertEqual(self.conversations.get_conversation_id_by_user(user_id), conversation_id)
        self.assertEqual(self.messages.count_messages(conversation_id), 2)
        self.assertEqual(
            self.messages.list_recent_messages(conversation_id, limit=10),
            [
                {"role": "user", "content": "oi"},
                {"role": "assistant", "content": "olá"},
            ],
        )

    def test_clear_conversation_removes_only_messages(self) -> None:
        user_id = self.users.get_or_create_user_id("bob", "hashed-password")
        conversation_id = self.conversations.get_or_create_conversation_id(user_id)

        self.messages.add_message(conversation_id, "user", "primeira")
        self.messages.add_message(conversation_id, "assistant", "resposta")

        cleared = self.messages.clear_conversation(conversation_id)

        self.assertEqual(cleared, 2)
        self.assertEqual(self.messages.count_messages(conversation_id), 0)
        self.assertEqual(self.conversations.get_conversation_id_by_user(user_id), conversation_id)
        self.assertEqual(self.users.get_user_id("bob"), user_id)

    def test_delete_user_cascades_conversation_and_messages(self) -> None:
        user_id = self.users.get_or_create_user_id("carol", "hashed-password")
        conversation_id = self.conversations.get_or_create_conversation_id(user_id)

        self.messages.add_message(conversation_id, "user", "pergunta")
        self.messages.add_message(conversation_id, "assistant", "resposta")

        deleted = self.users.delete_user_by_id(user_id)

        self.assertTrue(deleted)
        self.assertIsNone(self.users.get_user_id("carol"))
        self.assertIsNone(self.conversations.get_conversation_id_by_user(user_id))
        self.assertEqual(self.messages.count_messages(conversation_id), 0)

    def test_duplicate_username_returns_same_user(self) -> None:
        first_id = self.users.get_or_create_user_id("dana", "hash-1")
        second_id = self.users.get_or_create_user_id("dana", "hash-2")

        self.assertEqual(first_id, second_id)


if __name__ == "__main__":
    unittest.main()