import json
from typing import Any, Dict, List, Protocol

from sqlalchemy import delete, func, select

from backend.src.config.infrastructure import CHAT_CACHE_KEY_PREFIX, CHAT_CACHE_REDIS_URL, CHAT_CACHE_TTL_SECONDS
from backend.src.infrastructure.data import ConversationCache
from backend.src.infrastructure.data.models import Message
from backend.src.infrastructure.data.db_client import get_db_session


class ConversationCacheLike(Protocol):
    def get_messages(self, conversation_id: int) -> List[Dict[str, Any]] | None: ...

    def set_messages(self, conversation_id: int, messages: List[Dict[str, Any]]) -> None: ...

    def invalidate(self, conversation_id: int) -> None: ...


class MessagesRepository:
    def __init__(self, cache: ConversationCacheLike | None = None):
        self.cache = cache or ConversationCache(
            redis_url=CHAT_CACHE_REDIS_URL,
            ttl_seconds=CHAT_CACHE_TTL_SECONDS,
            key_prefix=CHAT_CACHE_KEY_PREFIX,
        )

    def add_message(
        self,
        conversation_id: int,
        role: str,
        content: str,
        sources: List[dict] | None = None,
    ) -> None:
        serialized_sources = json.dumps(sources or [])
        with get_db_session() as db:
            db.add(
                Message(conversation_id=conversation_id, role=role, content=content, sources_json=serialized_sources)
            )
        self.cache.invalidate(conversation_id)

    def list_recent_messages(self, conversation_id: int, limit: int) -> List[Dict[str, Any]]:
        cached = self.cache.get_messages(conversation_id)
        if cached is not None:
            return cached[-limit:] if limit > 0 else cached

        with get_db_session() as db:
            stmt = (
                select(Message.role, Message.content, Message.sources_json)
                .where(Message.conversation_id == conversation_id)
                .order_by(Message.created_at.desc(), Message.id.desc())
                .limit(limit)
            )
            rows = db.execute(stmt).all()

        rows = list(reversed(rows))
        messages: List[Dict[str, Any]] = []
        for role, content, sources_json in rows:
            try:
                raw_sources = json.loads(sources_json) if sources_json else []
            except json.JSONDecodeError:
                raw_sources = []
            if not isinstance(raw_sources, list):
                raw_sources = []

            # Normalize to structured format (dicts with path/page/excerpt)
            structured_sources: List[Dict[str, Any]] = []
            for src in raw_sources:
                if isinstance(src, dict):
                    structured_sources.append(src)
                elif isinstance(src, str):
                    # Legacy format: just a path string
                    structured_sources.append({"path": src, "page": None, "excerpt": None})
                else:
                    continue

            # Filter out entries without a path
            sources_list = [s for s in structured_sources if s.get("path")]
            messages.append(
                {
                    "role": role,
                    "content": content,
                    "sources": sources_list,
                }
            )
        self.cache.set_messages(conversation_id, messages)
        return messages

    def count_messages(self, conversation_id: int) -> int:
        with get_db_session() as db:
            stmt = select(func.count(Message.id)).where(Message.conversation_id == conversation_id)
            return db.execute(stmt).scalar_one()

    def clear_conversation(self, conversation_id: int) -> int:
        with get_db_session() as db:
            count_stmt = select(func.count(Message.id)).where(Message.conversation_id == conversation_id)
            total = db.execute(count_stmt).scalar_one()
            stmt = delete(Message).where(Message.conversation_id == conversation_id)
            db.execute(stmt)
        self.cache.invalidate(conversation_id)
        return int(total)
