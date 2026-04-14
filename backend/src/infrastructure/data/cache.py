import json
import logging
from typing import Dict, List, Optional

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

    def get_messages(self, conversation_id: int) -> Optional[List[Dict[str, str]]]:
        if not self._enabled or self._client is None:
            return None

        try:
            raw = self._client.get(self._messages_key(conversation_id))
            if not raw:
                return None
            data = json.loads(str(raw))
            if not isinstance(data, list):
                return None
            cleaned = []
            for item in data:
                if not isinstance(item, dict):
                    continue
                role = str(item.get("role", "")).strip()
                content = str(item.get("content", "")).strip()
                if role and content:
                    cleaned.append({"role": role, "content": content})
            return cleaned
        except Exception as e:
            logger.warning("Failed to read conversation cache: %s", e)
            return None

    def set_messages(self, conversation_id: int, messages: List[Dict[str, str]]) -> None:
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