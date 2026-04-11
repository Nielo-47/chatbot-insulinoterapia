from typing import Dict, List

from sqlalchemy import delete, func, select

from backend.src.config import Config
from backend.src.infrastructure.data import ConversationCache
from backend.src.infrastructure.data.models import Message
from backend.src.infrastructure.data.db_client import get_db_session


class MessagesRepository:
    def __init__(self, cache: ConversationCache | None = None):
        self.cache = cache or ConversationCache(
            redis_url=Config.CHAT_CACHE_REDIS_URL,
            ttl_seconds=Config.CHAT_CACHE_TTL_SECONDS,
            key_prefix=Config.CHAT_CACHE_KEY_PREFIX,
        )

    def add_message(self, conversation_id: int, role: str, content: str) -> None:
        with get_db_session() as db:
            db.add(Message(conversation_id=conversation_id, role=role, content=content))
        self.cache.invalidate(conversation_id)

    def list_recent_messages(self, conversation_id: int, limit: int) -> List[Dict[str, str]]:
        cached = self.cache.get_messages(conversation_id)
        if cached is not None:
            return cached[-limit:] if limit > 0 else cached

        with get_db_session() as db:
            stmt = (
                select(Message.role, Message.content)
                .where(Message.conversation_id == conversation_id)
                .order_by(Message.created_at.desc(), Message.id.desc())
                .limit(limit)
            )
            rows = db.execute(stmt).all()

        rows = list(reversed(rows))
        messages = [{"role": role, "content": content} for role, content in rows]
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
