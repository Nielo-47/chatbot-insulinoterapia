import unittest
from typing import Any, Dict, List, Optional, cast

from backend.src.services.conversation_service import ConversationService


class InMemoryUsersRepository:
    def __init__(self) -> None:
        self._ids: Dict[str, int] = {}
        self._next = 1

    def get_user_id(self, username: str) -> Optional[int]:
        return self._ids.get(username)

    def get_or_create_user_id(self, username: str, hashed_password: str) -> int:
        if username not in self._ids:
            self._ids[username] = self._next
            self._next += 1
        return self._ids[username]


class InMemoryConversationsRepository:
    def __init__(self) -> None:
        self._by_user: Dict[int, int] = {}
        self._next = 1
        self.touched_ids: List[int] = []

    def get_conversation_id_by_user(self, user_id: int) -> Optional[int]:
        return self._by_user.get(user_id)

    def get_or_create_conversation_id(self, user_id: int) -> int:
        if user_id not in self._by_user:
            self._by_user[user_id] = self._next
            self._next += 1
        return self._by_user[user_id]

    def touch_conversation(self, conversation_id: int) -> None:
        self.touched_ids.append(conversation_id)


class InMemoryMessagesRepository:
    def __init__(self) -> None:
        self._messages: Dict[int, List[Dict[str, str]]] = {}

    def add_message(self, conversation_id: int, role: str, content: str) -> None:
        self._messages.setdefault(conversation_id, []).append({"role": role, "content": content})

    def list_recent_messages(self, conversation_id: int, limit: int) -> List[Dict[str, str]]:
        all_messages = self._messages.get(conversation_id, [])
        return list(all_messages[-limit:])

    def count_messages(self, conversation_id: int) -> int:
        return len(self._messages.get(conversation_id, []))

    def clear_conversation(self, conversation_id: int) -> int:
        cleared = len(self._messages.get(conversation_id, []))
        self._messages[conversation_id] = []
        return cleared


class InMemoryConversationCache:
    def __init__(self) -> None:
        self._store: Dict[int, List[Dict[str, str]]] = {}

    def get_messages(self, conversation_id: int) -> Optional[List[Dict[str, str]]]:
        return self._store.get(conversation_id)

    def set_messages(self, conversation_id: int, messages: List[Dict[str, str]]) -> None:
        self._store[conversation_id] = list(messages)

    def invalidate(self, conversation_id: int) -> None:
        self._store.pop(conversation_id, None)


class ConversationServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.users_repo = InMemoryUsersRepository()
        self.conversations_repo = InMemoryConversationsRepository()
        self.messages_repo = InMemoryMessagesRepository()
        self.cache_repo = InMemoryConversationCache()

        self.service = ConversationService(
            users_repository=cast(Any, self.users_repo),
            conversations_repository=cast(Any, self.conversations_repo),
            messages_repository=cast(Any, self.messages_repo),
            cache=cast(Any, self.cache_repo),
        )

    def test_single_conversation_per_user(self) -> None:
        session_id = "abc"
        self.service.add_message(session_id, "user", "oi")
        self.service.add_message(session_id, "assistant", "ola")
        self.service.add_message(session_id, "user", "tudo bem?")

        history = self.service.get_conversation(session_id)
        self.assertEqual(len(history), 3)

        # Re-adding messages to the same session should not create a second conversation.
        self.service.add_message(session_id, "assistant", "tudo certo")
        history = self.service.get_conversation(session_id)
        self.assertEqual(len(history), 4)

        # touch_conversation should be called for each write path.
        self.assertEqual(len(self.conversations_repo.touched_ids), 4)
        self.assertEqual(len(set(self.conversations_repo.touched_ids)), 1)

    def test_reset_conversation(self) -> None:
        session_id = "xyz"
        self.service.add_message(session_id, "user", "primeira")
        self.service.add_message(session_id, "assistant", "resposta")

        was_cleared = self.service.reset_conversation(session_id)
        history = self.service.get_conversation(session_id)

        self.assertTrue(was_cleared)
        self.assertEqual(history, [])

        # reset_conversation should also touch conversation metadata.
        self.assertGreaterEqual(len(self.conversations_repo.touched_ids), 3)


if __name__ == "__main__":
    unittest.main()
