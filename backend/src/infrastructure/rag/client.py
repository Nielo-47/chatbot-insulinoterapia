import asyncio
import logging
from typing import Any, Awaitable, Callable, Dict, List, Literal, Optional

from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc

from backend.src.config.infrastructure import OPENROUTER_API_KEY, OPENROUTER_BASE_URL
from backend.src.config.prompts import SYSTEM_PROMPT
from backend.src.config.rag import (
    EMBED_MODEL,
    EMBEDDING_DIM,
    EMBEDDING_FALLBACK_MODEL,
    EMBEDDING_FALLBACK_RETRIES,
    EMBEDDING_PRIMARY_RETRIES,
    EMBEDDING_TIMEOUT_SECONDS,
    LLM_MODEL,
    MAX_TOKENS,
    RAG_QUERY_MAX_TOKENS,
    RAG_QUERY_TEMPERATURE,
    RAG_WORKING_DIR,
    KV_STORAGE,
    VECTOR_STORAGE,
    GRAPH_STORAGE,
)
from backend.src.infrastructure.rag.cleaner import extract_sources
from backend.src.infrastructure.rag.resilient_embeddings import EmbeddingProviderConfig, build_embedding_callable

logger = logging.getLogger(__name__)

QueryMode = Literal["local", "global", "hybrid", "naive", "mix", "bypass"]


class RAGRuntime:
    def __init__(self):
        self.rag: Optional[LightRAG] = None

    async def initialize(
        self,
        call_llm: Callable[..., Awaitable[str]],
    ) -> None:
        logger.debug("Initializing RAG...")

        async def openrouter_model_complete(prompt, system_prompt=None, history_messages=None, **kwargs) -> str:
            return await call_llm(
                prompt=prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                temperature=RAG_QUERY_TEMPERATURE,
                max_tokens=RAG_QUERY_MAX_TOKENS,
            )

        embed_func = build_embedding_callable(
            primary=EmbeddingProviderConfig(
                name="openrouter",
                base_url=OPENROUTER_BASE_URL,
                api_key=OPENROUTER_API_KEY,
                model=EMBED_MODEL,
            ),
            fallback=EmbeddingProviderConfig(
                name="openrouter",
                base_url=OPENROUTER_BASE_URL,
                api_key=OPENROUTER_API_KEY,
                model=EMBEDDING_FALLBACK_MODEL,
            ),
            embedding_dim=EMBEDDING_DIM,
            timeout_seconds=EMBEDDING_TIMEOUT_SECONDS,
            primary_retries=EMBEDDING_PRIMARY_RETRIES,
            fallback_retries=EMBEDDING_FALLBACK_RETRIES,
        )

        self.rag = LightRAG(
            working_dir=RAG_WORKING_DIR,
            kv_storage=KV_STORAGE,
            vector_storage=VECTOR_STORAGE,
            graph_storage=GRAPH_STORAGE,
            llm_model_func=openrouter_model_complete,
            llm_model_name=LLM_MODEL,
            enable_llm_cache=False,
            embedding_func=EmbeddingFunc(
                embedding_dim=EMBEDDING_DIM,
                max_token_size=MAX_TOKENS,
                func=embed_func,
            ),
        )

        logger.debug("Initializing RAG storages...")
        await self.rag.initialize_storages()
        logger.debug("RAG initialization complete")

    async def query_data(
        self,
        query: str,
        mode: QueryMode,
        conversation_history: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        max_total_tokens: int = 12000,
        top_k: int = 10,
    ) -> Any:
        if self.rag is None:
            raise RuntimeError("RAG is not initialized")

        try:
            rag_data = await self.rag.aquery_data(
                query,
                param=QueryParam(
                    mode=mode,
                    user_prompt=system_prompt or SYSTEM_PROMPT,
                    max_total_tokens=max_total_tokens,
                    top_k=top_k,
                    enable_rerank=False,
                    conversation_history=conversation_history,
                ),
            )
            logger.warning("RAG RAW OUTPUT: %r", rag_data)
            if not rag_data or not isinstance(rag_data, dict):
                logger.error("RAG returned invalid data: %r", rag_data)
            elif rag_data.get("status") != "success":
                logger.error(
                    "RAG FAILURE: status=%r, message=%r, metadata=%r",
                    rag_data.get("status"),
                    rag_data.get("message"),
                    rag_data.get("metadata"),
                )
            sources, source_count = extract_sources(rag_data)
            return {
                "rag_data": rag_data,
                "sources": sources,
                "source_count": source_count,
            }
        except Exception as e:
            logger.exception("Exception during RAG query_data: %s", e)
            return {
                "rag_data": {"status": "error", "message": str(e)},
                "sources": [],
                "source_count": 0,
            }
