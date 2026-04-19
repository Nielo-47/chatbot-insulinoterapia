import json
import logging
import os
from typing import Any, Dict, List, Optional

import redis
from redis import Redis

logger = logging.getLogger(__name__)


class ConversationCache:
    def __init__(self, redis_url: str, ttl_seconds: int, key_prefix: str):
        self._ttl_seconds = ttl_seconds
        self._key_prefix = key_prefix
        self._enabled = True
        self._client: Optional[Redis] = None

        try:
            self._client = redis.Redis.from_url(redis_url, decode_responses=True)
            self._client.ping()
        except Exception as e:
            logger.warning("Redis cache unavailable, falling back to DB-only reads: %s", e)
            self._enabled = False

    def _messages_key(self, conversation_id: int) -> str:
        return f"{self._key_prefix}:{conversation_id}:messages"

    def get_messages(self, conversation_id: int) -> Optional[List[Dict[str, Any]]]:
        if not self._enabled or self._client is None:
            return None

        try:
            raw = self._client.get(self._messages_key(conversation_id))
            if not raw:
                return None
            data = json.loads(str(raw))
            if not isinstance(data, list):
                return None
            cleaned: List[Dict[str, Any]] = []
            for item in data:
                if not isinstance(item, dict):
                    continue
                role = str(item.get("role", "")).strip()
                content = str(item.get("content", "")).strip()
                if role and content:
                    raw_sources = item.get("sources", [])
                    if not isinstance(raw_sources, list):
                        raw_sources = []
                    # Normalize sources to structured dicts (handle legacy string format)
                    structured_sources: List[Dict[str, Any]] = []
                    for src in raw_sources:
                        if isinstance(src, dict):
                            structured_sources.append(src)
                        elif isinstance(src, str):
                            structured_sources.append({"path": src, "page": None, "excerpt": None})
                        else:
                            continue
                    source_count = item.get("source_count", len(structured_sources))
                    try:
                        source_count = int(source_count)
                    except (TypeError, ValueError):
                        source_count = len(structured_sources)
                    cleaned.append(
                        {
                            "role": role,
                            "content": content,
                            "sources": structured_sources,
                            "source_count": source_count,
                        }
                    )
            return cleaned
        except Exception as e:
            logger.warning("Failed to read conversation cache: %s", e)
            return None

    def set_messages(self, conversation_id: int, messages: List[Dict[str, Any]]) -> None:
        if not self._enabled or self._client is None:
            return

        try:
            self._client.set(
                self._messages_key(conversation_id),
                json.dumps(messages),
                ex=self._ttl_seconds,
            )
        except Exception as e:
            logger.warning("Failed to update conversation cache: %s", e)

    def invalidate(self, conversation_id: int) -> None:
        if not self._enabled or self._client is None:
            return

        try:
            self._client.delete(self._messages_key(conversation_id))
        except Exception as e:
            logger.warning("Failed to invalidate conversation cache: %s", e)


def init_semantic_cache() -> None:
    """Initialize Redis semantic cache for LLM responses."""
    from backend.src.config.infrastructure import CHAT_CACHE_REDIS_URL

    from backend.src.config.env import require, require_float

    redis_url = require("CHAT_CACHE_REDIS_URL")
    score_threshold = require_float("SEMANTIC_CACHE_THRESHOLD")

    try:
        from langchain_core.globals import set_llm_cache
        from langchain_redis import RedisSemanticCache
        from langchain_openai import OpenAIEmbeddings

        cache = RedisSemanticCache(
            redis_url=redis_url,
            embeddings=OpenAIEmbeddings(),
            distance_threshold=score_threshold,
        )
        set_llm_cache(cache)
    except Exception as e:
        logger.warning("Could not initialize semantic cache: %s", e)
