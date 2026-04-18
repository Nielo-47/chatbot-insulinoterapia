import asyncio
import logging
from typing import Any, Callable, Coroutine, Dict, List, Optional

from backend.src.application.contracts.repositories import (
    ConversationsRepositoryLike,
    MessagesRepositoryLike,
    UsersRepositoryLike,
)
from backend.src.config.conversation import (
    CONVERSATION_HISTORY_LIMIT,
    SUMMARIZE_MAX_MESSAGES,
    SUMMARIZER_MAX_TOKENS,
    SUMMARIZER_TEMPERATURE,
)
from backend.src.config.prompts import SUMMARY_PROMPT, SYSTEM_PROMPT


class ConversationService:
    def __init__(
        self,
        users_repository: UsersRepositoryLike,
        conversations_repository: ConversationsRepositoryLike,
        messages_repository: MessagesRepositoryLike,
        summary_call_llm: Optional[Callable[..., Coroutine[Any, Any, str]]] = None,
    ):
        self.users_repository = users_repository
        self.conversations_repository = conversations_repository
        self.messages_repository = messages_repository
        self.summary_call_llm = summary_call_llm
        self.sessions_summarized: set[int] = set()

    def _resolve_conversation_id(self, user_id: int, create_if_missing: bool) -> Optional[int]:
        if user_id is None:
            return None
        if create_if_missing:
            return self.conversations_repository.get_or_create_conversation_id(user_id)
        return self.conversations_repository.get_conversation_id_by_user(user_id)

    def ensure_conversation(self, user_id: int) -> None:
        if user_id is None:
            return
        self._resolve_conversation_id(user_id=user_id, create_if_missing=True)

    def get_conversation(self, user_id: int, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        if user_id is None:
            return []

        conversation_id = self._resolve_conversation_id(user_id=user_id, create_if_missing=True)
        if conversation_id is None:
            return []

        history_limit = limit or CONVERSATION_HISTORY_LIMIT
        return self.messages_repository.list_recent_messages(conversation_id=conversation_id, limit=history_limit)

    def add_message(
        self,
        user_id: int,
        role: str,
        content: str,
        sources: Optional[List[str]] = None,
    ) -> None:
        if user_id is None:
            return

        clean_content = str(content).strip()
        if not clean_content:
            return

        conversation_id = self._resolve_conversation_id(user_id=user_id, create_if_missing=True)
        if conversation_id is None:
            return

        self.messages_repository.add_message(
            conversation_id=conversation_id,
            role=role,
            content=clean_content,
            sources=sources,
        )
        self.conversations_repository.touch_conversation(conversation_id=conversation_id)

    def count_messages(self, user_id: int) -> int:
        if user_id is None:
            return 0

        conversation_id = self._resolve_conversation_id(user_id=user_id, create_if_missing=True)
        if conversation_id is None:
            return 0

        return self.messages_repository.count_messages(conversation_id=conversation_id)

    def reset_conversation(self, user_id: int) -> bool:
        if user_id is None:
            return False

        conversation_id = self._resolve_conversation_id(user_id=user_id, create_if_missing=False)
        if conversation_id is None:
            return False

        self.messages_repository.clear_conversation(conversation_id=conversation_id)
        self.conversations_repository.touch_conversation(conversation_id=conversation_id)
        return True

    def delete_user(self, user_id: int) -> bool:
        if user_id is None:
            return False

        return self.users_repository.delete_user_by_id(user_id)

    def replace_with_summary(self, user_id: int, summary: str) -> None:
        if user_id is None:
            return

        clean_summary = str(summary).strip()
        if not clean_summary:
            return

        conversation_id = self._resolve_conversation_id(user_id=user_id, create_if_missing=True)
        if conversation_id is None:
            return

        self.messages_repository.clear_conversation(conversation_id=conversation_id)
        self.messages_repository.add_message(
            conversation_id=conversation_id,
            role="assistant",
            content=clean_summary,
        )
        self.conversations_repository.touch_conversation(conversation_id=conversation_id)

    def store_summary(self, user_id: int, summary: str) -> None:
        """Store summary in DB without clearing messages - preserves conversation history."""
        if user_id is None:
            return

        clean_summary = str(summary).strip()
        if not clean_summary:
            return

        conversation_id = self._resolve_conversation_id(user_id=user_id, create_if_missing=True)
        if conversation_id is None:
            return

        self.conversations_repository.update_summary(conversation_id, clean_summary)
        logging.getLogger(__name__).info("User %s summary stored (messages preserved)", user_id)

    def get_summary(self, user_id: int) -> Optional[str]:
        """Get stored summary for a user's conversation."""
        if user_id is None:
            return None

        conversation_id = self._resolve_conversation_id(user_id=user_id, create_if_missing=False)
        if conversation_id is None:
            return None

        return self.conversations_repository.get_summary(conversation_id)

    def summarize_session(self, user_id: int, max_messages: Optional[int] = None) -> str:
        if user_id is None:
            return ""

        if self.summary_call_llm is None:
            return ""

        msgs = self.get_conversation(user_id=user_id)
        if not msgs:
            return ""

        max_messages = max_messages or SUMMARIZE_MAX_MESSAGES
        recent = msgs[-max_messages:]
        history_lines = []
        for message in recent:
            role = message.get("role", "")
            content = str(message.get("content", "")).strip()
            if content:
                history_lines.append(f"{role.upper()}: {content}")

        summary_prompt = SUMMARY_PROMPT.format(history="\n".join(history_lines))

        try:
            summary = asyncio.run(
                self.summary_call_llm(
                    prompt=summary_prompt,
                    system_prompt=SYSTEM_PROMPT,
                    temperature=SUMMARIZER_TEMPERATURE,
                    max_tokens=SUMMARIZER_MAX_TOKENS,
                )
            )
            summary = summary.strip()
            if summary:
                self.store_summary(user_id=user_id, summary=summary)
                logging.getLogger(__name__).info("User %s conversation summarized (history preserved)", user_id)
                return summary
        except Exception as e:
            logging.getLogger(__name__).warning("Error summarizing user %s: %s", user_id, e)

        return ""

    def consume_summarized(self, user_id: int) -> bool:
        was_summarized = user_id in self.sessions_summarized
        if was_summarized:
            self.sessions_summarized.discard(user_id)
        return was_summarized
