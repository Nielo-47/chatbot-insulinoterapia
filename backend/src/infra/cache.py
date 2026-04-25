import json
import logging
from typing import TypedDict

from langchain_community.cache import RedisSemanticCache
from langchain_community.vectorstores import Redis
from langchain_core.globals import set_llm_cache
from langchain_core.documents import Document

from backend.src.core.config.infrastructure import CHAT_CACHE_REDIS_URL, SEMANTIC_CACHE_THRESHOLD, CACHE_INDEX_NAME
from backend.src.infra.embeddings import Embeddings

# from redis import Redis

logger = logging.getLogger(__name__)


class ChatbotCachedMessage(TypedDict):
    content: str
    sources: list[Document]


class SemanticCache:
    def __init__(self):
        self.embeddings = Embeddings()

        self.store = Redis(
            redis_url=CHAT_CACHE_REDIS_URL,
            embedding=self.embeddings,
            index_name=CACHE_INDEX_NAME,
        )

    def setup_global_cache(self) -> None:
        """Initialize and set the global LLM cache."""
        try:
            semantic_cache = RedisSemanticCache(
                redis_url=CHAT_CACHE_REDIS_URL,
                embedding=self.embeddings,
                score_threshold=SEMANTIC_CACHE_THRESHOLD,
            )

            set_llm_cache(semantic_cache)
        except Exception as e:
            print(f"Failed to initialize global cache: {e}")

    def check_match(self, query: str) -> ChatbotCachedMessage | None:
        """Check for a cached response matching the query."""
        try:
            match = self.store.similarity_search_limit_score(query, k=1, score_threshold=SEMANTIC_CACHE_THRESHOLD)
            if not match:
                return None

            return ChatbotCachedMessage(
                content=match[0].page_content,
                sources=[
                    Document(page_content=doc.page_content, metadata=doc.metadata)
                    for doc in match[0].metadata.get("source_documents", [])
                ],
            )

        except Exception as e:
            logger.info(f"Semantic cache lookup failed: {e}")
            return

    def save(self, query: str, response: str, sources: list[Document]):
        self.store.add_texts(texts=[response], metadatas=[{"query": query, "sources": json.dumps(sources)}])
