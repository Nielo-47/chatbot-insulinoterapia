import logging
from typing import List

from pydantic import SecretStr
from langchain_openai import OpenAIEmbeddings

from backend.src.core.config.infrastructure import (
    EMBEDDING_DIM,
    EMBEDDING_FALLBACK_MODEL,
    EMBEDDING_MODEL,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
)

logger = logging.getLogger(__name__)


class Embeddings(OpenAIEmbeddings):
    """
    A generic embeddings client that behaves like OpenAIEmbeddings
    but gracefully falls back to a secondary client if the primary API fails.
    """

    def __init__(
        self,
        model: str = EMBEDDING_MODEL,
        fallback_model: str = EMBEDDING_FALLBACK_MODEL,
        dimensions: int = EMBEDDING_DIM,
        **kwargs,
    ):
        self.dimensions: int = dimensions

        shared_kwargs = {
            "api_key": OPENROUTER_API_KEY,
            "base_url": OPENROUTER_BASE_URL,
            "dimensions": dimensions,
        }

        super().__init__(
            model=model,
            dimensions=dimensions,
            **shared_kwargs,
            **kwargs,
        )

        self._fallback_client = OpenAIEmbeddings(
            model=fallback_model,
            **shared_kwargs,
            **kwargs,
        )

    def __log_fallback(self, error: Exception, method_name: str) -> None:
        logger.warning(f"Primary {method_name} failed with error: {error}. Falling back to secondary model.")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        try:
            return super().embed_documents(texts)
        except Exception as e:
            self.__log_fallback(e, "embed_documents")
            return self._fallback_client.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        try:
            return super().embed_query(text)
        except Exception as e:
            self.__log_fallback(e, "embed_query")
            return self._fallback_client.embed_query(text)

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        try:
            return await super().aembed_documents(texts)
        except Exception as e:
            self.__log_fallback(e, "aembed_documents")
            return await self._fallback_client.aembed_documents(texts)

    async def aembed_query(self, text: str) -> List[float]:
        try:
            return await super().aembed_query(text)
        except Exception as e:
            self.__log_fallback(e, "aembed_query")
            return await self._fallback_client.aembed_query(text)
