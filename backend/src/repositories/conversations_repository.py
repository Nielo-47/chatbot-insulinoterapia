from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from backend.src.db.models import Conversation
from backend.src.db.session import get_db_session


class ConversationsRepository:
    def get_conversation_id_by_user(self, user_id: int) -> Optional[int]:
        with get_db_session() as db:
            stmt = select(Conversation.id).where(Conversation.user_id == user_id)
            return db.execute(stmt).scalar_one_or_none()

    def get_or_create_conversation_id(self, user_id: int) -> int:
        with get_db_session() as db:
            existing_stmt = select(Conversation.id).where(Conversation.user_id == user_id)
            existing_id = db.execute(existing_stmt).scalar_one_or_none()
            if existing_id is not None:
                return existing_id

            conversation = Conversation(user_id=user_id)
            db.add(conversation)
            try:
                db.flush()
                return conversation.id
            except IntegrityError:
                db.rollback()
                return db.execute(existing_stmt).scalar_one()

    def touch_conversation(self, conversation_id: int) -> None:
        with get_db_session() as db:
            conversation = db.get(Conversation, conversation_id)
            if conversation is None:
                return
            conversation.updated_at = datetime.now(timezone.utc)
            db.add(conversation)
