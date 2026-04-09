import unittest

from backend.src.services.conversation_service import ConversationService


class InMemoryUsersRepository:
    def __init__(self):
        self._ids = {}
        self._next = 1

    def get_user_id(self, username: str):
        return self._ids.get(username)

    def get_or_create_user_id(self, username: str, hashed_password: str):
        if username not in self._ids:
            self._ids[username] = self._next
            self._next += 1
        return self._ids[username]


class InMemoryConversationsRepository:
    def __init__(self):
        self._by_user = {}
        self._next = 1

    def get_conversation_id_by_user(self, user_id: int):
        return self._by_user.get(user_id)

    def get_or_create_conversation_id(self, user_id: int):
        if user_id not in self._by_user:
            self._by_user[user_id] = self._next
            self._next += 1
        return self._by_user[user_id]

    def touch_conversation(self, conversation_id: int):
        return None


class InMemoryMessagesRepository:
    def __init__(self):
        self._messages = {}

    def add_message(self, conversation_id: int, role: str, content: str):
        self._messages.setdefault(conversation_id, []).append({"role": role, "content": content})

    def list_recent_messages(self, conversation_id: int, limit: int):
        all_messages = self._messages.get(conversation_id, [])
        return list(all_messages[-limit:])

    def count_messages(self, conversation_id: int):
        return len(self._messages.get(conversation_id, []))

    def clear_conversation(self, conversation_id: int):
        cleared = len(self._messages.get(conversation_id, []))
        self._messages[conversation_id] = []
        return cleared


class InMemoryConversationCache:
    def __init__(self):
        self._store = {}

    def get_messages(self, conversation_id: int):
        return self._store.get(conversation_id)

    def set_messages(self, conversation_id: int, messages):
        self._store[conversation_id] = list(messages)

    def invalidate(self, conversation_id: int):
        self._store.pop(conversation_id, None)


class ConversationServiceTests(unittest.TestCase):
    def setUp(self):
        self.service = ConversationService(
            users_repository=InMemoryUsersRepository(),
            conversations_repository=InMemoryConversationsRepository(),
            messages_repository=InMemoryMessagesRepository(),
            cache=InMemoryConversationCache(),
        )

    def test_single_conversation_per_user(self):
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

    def test_reset_conversation(self):
        session_id = "xyz"
        self.service.add_message(session_id, "user", "primeira")
        self.service.add_message(session_id, "assistant", "resposta")

        was_cleared = self.service.reset_conversation(session_id)
        history = self.service.get_conversation(session_id)

        self.assertTrue(was_cleared)
        self.assertEqual(history, [])


if __name__ == "__main__":
    unittest.main()
