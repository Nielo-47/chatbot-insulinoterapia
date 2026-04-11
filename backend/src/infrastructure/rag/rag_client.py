from typing import Any, Awaitable, Callable, Dict, List, Literal, Optional

import numpy as np
from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc
from openai import OpenAI

from backend.src.config import Config


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
        print("[DEBUG] Initializing RAG...")

        async def openrouter_model_complete(prompt, system_prompt=None, history_messages=None, **kwargs) -> str:
            return await call_llm(
                prompt=prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                max_tokens=500,
            )

        async def tei_embed_func(texts: List[str]) -> np.ndarray:
            print(f"[DEBUG] tei_embed_func called with {len(texts) if isinstance(texts, list) else 1} text(s)")
            print(f"[DEBUG] Using embed base_url: {self.embed_base_url}")

            try:
                client = OpenAI(api_key=self.embed_api_key, base_url=self.embed_base_url)

                if isinstance(texts, str):
                    texts = [texts]

                print(f"[DEBUG] Calling embeddings.create with model={Config.EMBED_MODEL}")
                response = client.embeddings.create(
                    model=Config.EMBED_MODEL,
                    input=texts,
                )
                print(f"[DEBUG] Embeddings received: {len(response.data)} embeddings")
                return np.array([item.embedding for item in response.data])
            except Exception as e:
                print(f"[ERROR] Embedding failed: {type(e).__name__}: {e}")
                import traceback

                traceback.print_exc()
                raise

        self.rag = LightRAG(
            working_dir=Config.RAG_WORKING_DIR,
            llm_model_func=openrouter_model_complete,
            llm_model_name=Config.LLM_MODEL,
            enable_llm_cache=False,
            embedding_func=EmbeddingFunc(
                embedding_dim=Config.EMBEDDING_DIM,
                max_token_size=Config.MAX_TOKENS,
                func=tei_embed_func,
            ),
        )

        print("[DEBUG] Initializing RAG storages...")
        await self.rag.initialize_storages()
        print("[DEBUG] RAG initialization complete")

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

        return self.rag.query_data(
            query,
            param=QueryParam(
                mode=mode,
                user_prompt=system_prompt or Config.SYSTEM_PROMPT,
                max_total_tokens=max_total_tokens,
                top_k=top_k,
                enable_rerank=False,
                conversation_history=conversation_history,
            ),
        )
