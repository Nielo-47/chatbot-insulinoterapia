from typing import Dict, List

from sqlalchemy import delete, func, select

from backend.src.db.models import Message
from backend.src.db.session import get_db_session


class MessagesRepository:
    def add_message(self, conversation_id: int, role: str, content: str) -> None:
        with get_db_session() as db:
            db.add(Message(conversation_id=conversation_id, role=role, content=content))

    def list_recent_messages(self, conversation_id: int, limit: int) -> List[Dict[str, str]]:
        with get_db_session() as db:
            stmt = (
                select(Message.role, Message.content)
                .where(Message.conversation_id == conversation_id)
                .order_by(Message.created_at.desc(), Message.id.desc())
                .limit(limit)
            )
            rows = db.execute(stmt).all()

        rows = list(reversed(rows))
        return [{"role": role, "content": content} for role, content in rows]

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
            return int(total)
