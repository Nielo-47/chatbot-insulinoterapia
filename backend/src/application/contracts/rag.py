from __future__ import annotations

from typing import Any, Protocol

from backend.src.domain.query import QueryMode


class RAGGateway(Protocol):
    async def initialize(self, call_llm) -> None: ...

    def query_data(
        self,
        query: str,
        mode: QueryMode,
        conversation_history: list[dict[str, str]],
        system_prompt: str | None = None,
        max_total_tokens: int = 12000,
        top_k: int = 10,
    ) -> Any: ...
