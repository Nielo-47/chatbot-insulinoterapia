import unittest
import uuid
from typing import Any, Dict, List, Optional

from backend.src.application.contracts.repositories import (
    ConversationsRepositoryLike,
    MessagesRepositoryLike,
)
from backend.src.application.features.chat.conversation_service import ConversationService


class InMemoryConversationsRepository(ConversationsRepositoryLike):
    def __init__(self) -> None:
        self._by_user: Dict[uuid.UUID, uuid.UUID] = {}
        self.touched_ids: List[uuid.UUID] = []
        self._summaries: Dict[uuid.UUID, str] = {}

    def get_conversation_id_by_user(self, user_id: uuid.UUID) -> Optional[uuid.UUID]:
        return self._by_user.get(user_id)

    def get_or_create_conversation_id(self, user_id: uuid.UUID) -> uuid.UUID:
        if user_id not in self._by_user:
            self._by_user[user_id] = uuid.uuid4()
        return self._by_user[user_id]

    def touch_conversation(self, conversation_id: uuid.UUID) -> None:
        self.touched_ids.append(conversation_id)

    def get_summary(self, conversation_id: uuid.UUID) -> Optional[str]:
        return self._summaries.get(conversation_id)

    def update_summary(self, conversation_id: uuid.UUID, summary: str) -> None:
        self._summaries[conversation_id] = summary


class InMemoryMessagesRepository(MessagesRepositoryLike):
    def __init__(self) -> None:
        self._messages: Dict[uuid.UUID, List[Dict[str, str]]] = {}
        self.invalidated: List[uuid.UUID] = []

    def add_message(self, conversation_id: uuid.UUID, role: str, content: str, sources: Optional[List[str]] = None) -> None:
        self._messages.setdefault(conversation_id, []).append({"role": role, "content": content})

    def list_recent_messages(self, conversation_id: uuid.UUID, limit: int) -> List[Dict[str, str]]:
        all_messages = self._messages.get(conversation_id, [])
        return list(all_messages[-limit:])

    def count_messages(self, conversation_id: uuid.UUID) -> int:
        return len(self._messages.get(conversation_id, []))

    def clear_conversation(self, conversation_id: uuid.UUID) -> int:
        cleared = len(self._messages.get(conversation_id, []))
        self._messages[conversation_id] = []
        return cleared

    def invalidate_cache(self, conversation_id: uuid.UUID) -> None:
        self.invalidated.append(conversation_id)


class InMemoryConversationCache:
    def __init__(self) -> None:
        self._store: Dict[uuid.UUID, List[Dict[str, str]]] = {}

    def get_messages(self, conversation_id: uuid.UUID) -> Optional[List[Dict[str, str]]]:
        return self._store.get(conversation_id)

    def set_messages(self, conversation_id: uuid.UUID, messages: List[Dict[str, str]]) -> None:
        self._store[conversation_id] = list(messages)

    def invalidate(self, conversation_id: uuid.UUID) -> None:
        self._store.pop(conversation_id, None)


class ConversationServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.conversations_repo = InMemoryConversationsRepository()
        self.messages_repo = InMemoryMessagesRepository()

        self.service = ConversationService(
            conversations_repository=self.conversations_repo,
            messages_repository=self.messages_repo,
        )

    def test_single_conversation_per_user(self) -> None:
        user_id = uuid.uuid4()
        self.service.add_message(user_id, "user", "oi")
        self.service.add_message(user_id, "assistant", "ola")
        self.service.add_message(user_id, "user", "tudo bem?")

        history = self.service.get_conversation(user_id)
        self.assertEqual(len(history), 3)

        # Re-adding messages to the same user should not create a second conversation.
        self.service.add_message(user_id, "assistant", "tudo certo")
        history = self.service.get_conversation(user_id)
        self.assertEqual(len(history), 4)

        # touch_conversation should be called for each write path.
        self.assertEqual(len(self.conversations_repo.touched_ids), 4)
        self.assertEqual(len(set(self.conversations_repo.touched_ids)), 1)

    def test_reset_conversation(self) -> None:
        user_id = uuid.uuid4()
        self.service.add_message(user_id, "user", "primeira")
        self.service.add_message(user_id, "assistant", "resposta")

        was_cleared = self.service.reset_conversation(user_id)
        history = self.service.get_conversation(user_id)

        self.assertTrue(was_cleared)
        self.assertEqual(history, [])

        # reset_conversation should also touch conversation metadata.
        self.assertGreaterEqual(len(self.conversations_repo.touched_ids), 3)

    def test_purge_user_data_invalidates_cached_conversation(self) -> None:
        user_id = uuid.uuid4()
        self.service.add_message(user_id, "user", "dado pessoal")

        self.service.purge_user_data(user_id)

        conversation_id = self.conversations_repo.get_conversation_id_by_user(user_id)
        self.assertIsNotNone(conversation_id)
        self.assertEqual(self.messages_repo.invalidated, [conversation_id])

    def test_purge_user_data_is_noop_without_conversation(self) -> None:
        self.service.purge_user_data(uuid.uuid4())

        self.assertEqual(self.messages_repo.invalidated, [])


if __name__ == "__main__":
    unittest.main()
