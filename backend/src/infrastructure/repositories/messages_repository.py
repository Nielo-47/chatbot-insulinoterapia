import json
from typing import Any, Dict, List, Optional, Protocol

from backend.src.config.infrastructure import CHAT_CACHE_KEY_PREFIX, CHAT_CACHE_REDIS_URL, CHAT_CACHE_TTL_SECONDS
from backend.src.infrastructure.data import ConversationCache
from backend.src.infrastructure.pocketbase import PocketBaseClient, get_pocketbase_client

MESSAGES_COLLECTION = "messages"


class ConversationCacheLike(Protocol):
    def get_messages(self, conversation_id: str) -> List[Dict[str, Any]] | None: ...

    def set_messages(self, conversation_id: str, messages: List[Dict[str, Any]]) -> None: ...

    def invalidate(self, conversation_id: str) -> None: ...


class MessagesRepository:
    """PocketBase-backed message storage with the Redis read cache in front."""

    def __init__(
        self,
        client: Optional[PocketBaseClient] = None,
        cache: ConversationCacheLike | None = None,
    ):
        self._client = client or get_pocketbase_client()
        self.cache = cache or ConversationCache(
            redis_url=CHAT_CACHE_REDIS_URL,
            ttl_seconds=CHAT_CACHE_TTL_SECONDS,
            key_prefix=CHAT_CACHE_KEY_PREFIX,
        )

    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        sources: List[dict] | None = None,
    ) -> None:
        serialized_sources = json.dumps(sources or [])
        self._client.create_record(
            MESSAGES_COLLECTION,
            {
                "conversation": conversation_id,
                "role": role,
                "content": content,
                "sources_json": serialized_sources,
            },
        )
        self.cache.invalidate(conversation_id)

    @staticmethod
    def _parse_sources(sources_json: Any) -> List[Dict[str, Any]]:
        try:
            raw_sources = json.loads(sources_json) if sources_json else []
        except (json.JSONDecodeError, TypeError):
            raw_sources = []
        if not isinstance(raw_sources, list):
            return []

        # Normalize to structured format (dicts with path/page/content)
        structured_sources: List[Dict[str, Any]] = []
        for src in raw_sources:
            if isinstance(src, dict):
                structured_sources.append(src)
            elif isinstance(src, str):
                # Legacy format: just a path string
                structured_sources.append({"path": src, "page": None, "content": None})

        # Filter out entries without a path
        return [s for s in structured_sources if s.get("path")]

    @staticmethod
    def _to_history_entry(record: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "role": record.get("role", ""),
            "content": record.get("content", ""),
            "sources": MessagesRepository._parse_sources(record.get("sources_json")),
        }

    def list_recent_messages(self, conversation_id: str, limit: int) -> List[Dict[str, Any]]:
        cached = self.cache.get_messages(conversation_id)
        if cached is not None:
            return cached[-limit:] if limit > 0 else cached

        records = self._client.list_records(
            MESSAGES_COLLECTION,
            filter_expr=f'conversation = "{conversation_id}"',
            sort="-created",
        )
        if limit > 0 and len(records) > limit:
            records = records[:limit]

        messages = [self._to_history_entry(record) for record in reversed(records)]
        self.cache.set_messages(conversation_id, messages)
        return messages

    def count_messages(self, conversation_id: str) -> int:
        return self._client.count_records(
            MESSAGES_COLLECTION,
            filter_expr=f'conversation = "{conversation_id}"',
        )

    def clear_conversation(self, conversation_id: str) -> int:
        # Invalidate the cache FIRST so a partial failure can never leave the
        # Redis view out of sync with what remains stored.
        self.cache.invalidate(conversation_id)
        records = self._client.list_records(
            MESSAGES_COLLECTION,
            filter_expr=f'conversation = "{conversation_id}"',
        )
        total = 0
        for record in records:
            self._client.delete_record(MESSAGES_COLLECTION, str(record["id"]))
            total += 1
        return total

    def invalidate_cache(self, conversation_id: str) -> None:
        self.cache.invalidate(conversation_id)
