import uuid
from typing import Any, Dict, List, Literal, Optional, Protocol

QueryMode = Literal["local", "global", "hybrid", "naive", "mix", "bypass"]


class RAGRuntimeContract(Protocol):
    def query_data(
        self,
        query: str,
        mode: QueryMode,
        conversation_history: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        max_total_tokens: int = 12000,
        top_k: int = 10,
    ) -> Any: ...


class ConversationServiceContract(Protocol):
    def get_conversation(self, user_id: uuid.UUID, limit: Optional[int] = None) -> List[Dict[str, Any]]: ...

    def add_message(self, user_id: uuid.UUID, role: str, content: str, sources: Optional[List[Dict[str, Any]]] = None) -> None: ...

    def get_summary(self, user_id: uuid.UUID) -> Optional[str]: ...

    def store_summary(self, user_id: uuid.UUID, summary: str) -> None: ...
