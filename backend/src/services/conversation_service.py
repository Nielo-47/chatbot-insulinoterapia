import hashlib
from typing import Dict, List, Optional

from backend.src.cache.conversation_cache import ConversationCache
from backend.src.config import Config
from backend.src.repositories.conversations_repository import ConversationsRepository
from backend.src.repositories.messages_repository import MessagesRepository
from backend.src.repositories.users_repository import UsersRepository


class ConversationService:
    def __init__(
        self,
        users_repository: UsersRepository,
        conversations_repository: ConversationsRepository,
        messages_repository: MessagesRepository,
        cache: ConversationCache,
    ):
        self.users_repository = users_repository
        self.conversations_repository = conversations_repository
        self.messages_repository = messages_repository
        self.cache = cache

    def _session_username(self, session_id: str) -> str:
        return f"{Config.TEMP_USER_PREFIX}:{session_id}"

    def _placeholder_password_hash(self, username: str) -> str:
        source = f"{username}:{Config.AUTH_PLACEHOLDER_PASSWORD_HASH}"
        return hashlib.sha256(source.encode("utf-8")).hexdigest()

    def _resolve_user_id(self, session_id: str, create_if_missing: bool) -> Optional[int]:
        username = self._session_username(session_id)
        if create_if_missing:
            return self.users_repository.get_or_create_user_id(
                username=username,
                hashed_password=self._placeholder_password_hash(username),
            )
        return self.users_repository.get_user_id(username)

    def _resolve_conversation_id(self, session_id: str, create_if_missing: bool) -> Optional[int]:
        user_id = self._resolve_user_id(session_id=session_id, create_if_missing=create_if_missing)
        if user_id is None:
            return None
        if create_if_missing:
            return self.conversations_repository.get_or_create_conversation_id(user_id)
        return self.conversations_repository.get_conversation_id_by_user(user_id)

    def ensure_conversation(self, session_id: str) -> None:
        if not session_id:
            return
        self._resolve_conversation_id(session_id=session_id, create_if_missing=True)

    def get_conversation(self, session_id: str, limit: Optional[int] = None) -> List[Dict[str, str]]:
        if not session_id:
            return []

        conversation_id = self._resolve_conversation_id(session_id=session_id, create_if_missing=True)
        if conversation_id is None:
            return []

        cached = self.cache.get_messages(conversation_id)
        if cached is not None:
            return cached

        history_limit = limit or Config.CONVERSATION_HISTORY_LIMIT
        messages = self.messages_repository.list_recent_messages(conversation_id=conversation_id, limit=history_limit)
        self.cache.set_messages(conversation_id=conversation_id, messages=messages)
        return messages

    def add_message(self, session_id: str, role: str, content: str) -> None:
        if not session_id:
            return

        clean_content = str(content).strip()
        if not clean_content:
            return

        conversation_id = self._resolve_conversation_id(session_id=session_id, create_if_missing=True)
        if conversation_id is None:
            return

        self.messages_repository.add_message(conversation_id=conversation_id, role=role, content=clean_content)
        self.conversations_repository.touch_conversation(conversation_id=conversation_id)
        self.cache.invalidate(conversation_id=conversation_id)

    def count_messages(self, session_id: str) -> int:
        if not session_id:
            return 0

        conversation_id = self._resolve_conversation_id(session_id=session_id, create_if_missing=True)
        if conversation_id is None:
            return 0

        return self.messages_repository.count_messages(conversation_id=conversation_id)

    def reset_conversation(self, session_id: str) -> bool:
        if not session_id:
            return False

        conversation_id = self._resolve_conversation_id(session_id=session_id, create_if_missing=False)
        if conversation_id is None:
            return False

        self.messages_repository.clear_conversation(conversation_id=conversation_id)
        self.conversations_repository.touch_conversation(conversation_id=conversation_id)
        self.cache.invalidate(conversation_id=conversation_id)
        return True

    def replace_with_summary(self, session_id: str, summary: str) -> None:
        if not session_id:
            return

        clean_summary = str(summary).strip()
        if not clean_summary:
            return

        conversation_id = self._resolve_conversation_id(session_id=session_id, create_if_missing=True)
        if conversation_id is None:
            return

        self.messages_repository.clear_conversation(conversation_id=conversation_id)
        self.messages_repository.add_message(
            conversation_id=conversation_id,
            role="assistant",
            content=clean_summary,
        )
        self.conversations_repository.touch_conversation(conversation_id=conversation_id)
        self.cache.invalidate(conversation_id=conversation_id)


def build_conversation_service() -> ConversationService:
    return ConversationService(
        users_repository=UsersRepository(),
        conversations_repository=ConversationsRepository(),
        messages_repository=MessagesRepository(),
        cache=ConversationCache(
            redis_url=Config.CHAT_CACHE_REDIS_URL,
            ttl_seconds=Config.CHAT_CACHE_TTL_SECONDS,
            key_prefix=Config.CHAT_CACHE_KEY_PREFIX,
        ),
    )
