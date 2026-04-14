import logging
from typing import Any, Awaitable, Callable, Dict, List, Literal, Optional

import numpy as np
from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc
from openai import AsyncOpenAI

from backend.src.config.prompts import SYSTEM_PROMPT
from backend.src.config.rag import EMBED_MODEL, EMBEDDING_DIM, LLM_MODEL, MAX_TOKENS, RAG_WORKING_DIR
from backend.src.infrastructure.rag.cleaner import extract_sources

logger = logging.getLogger(__name__)

QueryMode = Literal["local", "global", "hybrid", "naive", "mix", "bypass"]


class RAGRuntime:
    def __init__(self, embed_api_key: str, embed_base_url: str):
        self.embed_api_key = embed_api_key
        self.embed_base_url = embed_base_url
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
                max_tokens=500,
            )

        async def tei_embed_func(texts: List[str]) -> np.ndarray:
            if isinstance(texts, str):
                texts = [texts]
            logger.debug("tei_embed_func called with %d text(s)", len(texts))
            logger.debug("Using embed base_url: %s", self.embed_base_url)
            try:
                client = AsyncOpenAI(api_key=self.embed_api_key, base_url=self.embed_base_url)

                logger.debug("Calling embeddings.create with model=%s", EMBED_MODEL)
                response = await client.embeddings.create(
                    model=EMBED_MODEL,
                    input=texts,
                )
                logger.debug("Embeddings received: %d embeddings", len(response.data))
                return np.array([item.embedding for item in response.data])
            except Exception as e:
                logger.exception("Embedding failed: %s: %s", type(e).__name__, e)
                raise

        self.rag = LightRAG(
            working_dir=RAG_WORKING_DIR,
            llm_model_func=openrouter_model_complete,
            llm_model_name=LLM_MODEL,
            enable_llm_cache=False,
            embedding_func=EmbeddingFunc(
                embedding_dim=EMBEDDING_DIM,
                max_token_size=MAX_TOKENS,
                func=tei_embed_func,
            ),
        )

        logger.debug("Initializing RAG storages...")
        await self.rag.initialize_storages()
        logger.debug("RAG initialization complete")

    def query_data(
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

        rag_data = self.rag.query_data(
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
        sources, source_count = extract_sources(rag_data)

        return {
            "rag_data": rag_data,
            "sources": sources,
            "source_count": source_count,
        }
