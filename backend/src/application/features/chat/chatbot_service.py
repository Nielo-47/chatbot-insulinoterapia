from typing import Any, Dict, List, Optional
import uuid

from backend.src.application.features.chat.conversation_service import ConversationService
from backend.src.application.features.chat.query_processor import QueryProcessor, QueryMode


class ChatbotService:
    """Application facade coordinating conversation management and RAG queries."""

    def __init__(
        self,
        conversation_service: ConversationService,
        query_processor: QueryProcessor,
    ):
        self._conversation = conversation_service
        self._query = query_processor

    def get_history(self, user_id: uuid.UUID) -> List[Dict[str, Any]]:
        return self._conversation.get_conversation(user_id)

    def end_session(self, user_id: uuid.UUID) -> bool:
        return self._conversation.reset_conversation(user_id)

    def summarize_session(self, user_id: uuid.UUID, max_messages: Optional[int] = None) -> str:
        return self._conversation.summarize_session(user_id, max_messages=max_messages)

    def purge_user_data(self, user_id: uuid.UUID) -> None:
        """Purge cached user data (e.g. Redis message cache) ahead of account deletion."""
        return self._conversation.purge_user_data(user_id)

    async def chat(
        self,
        query: str,
        user_id: uuid.UUID,
        mode: QueryMode = "hybrid",
        session_id: Optional[str] = None,
        **query_params,
    ) -> Dict[str, Any]:
        return await self._query.query(
            query=query,
            user_id=user_id,
            mode=mode,
            session_id=session_id,
            **query_params,
        )
