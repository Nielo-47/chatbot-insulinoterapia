import logging
from typing import List, Any

from langchain_core.embeddings import Embeddings as BaseLangchainEmbeddings
from langchain_openai import OpenAIEmbeddings

from backend.src.core.config.infrastructure import (
    EMBEDDING_DIM,
    EMBEDDING_FALLBACK_MODEL,
    EMBEDDING_MODEL,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    RETRIES_PER_MODEL,
)

logger = logging.getLogger(__name__)


class Embeddings(BaseLangchainEmbeddings):
    """
    Handles interactions with embedding models, supporting primary and fallback models,
    and automatic retries.
    """

    def __init__(
        self,
        model: str = EMBEDDING_MODEL,
        fallback_model: str = EMBEDDING_FALLBACK_MODEL,
        dimensions: int = EMBEDDING_DIM,
        retries: int = RETRIES_PER_MODEL,
        **kwargs: Any,
    ):
        self.dimensions = dimensions
        self._retries = retries

        self._clients = [
            self._create_client(model, dimensions, **kwargs),
            self._create_client(fallback_model, dimensions, **kwargs),
        ]

    def _create_client(self, model_name: str, dimensions: int, **kwargs: Any) -> OpenAIEmbeddings:
        """Instantiates an OpenAIEmbeddings client for a specific model."""
        return OpenAIEmbeddings(
            model=model_name,
            dimensions=dimensions,
            api_key=OPENROUTER_API_KEY,
            base_url=OPENROUTER_BASE_URL,
            **kwargs,
        )

    async def _execute_with_fallbacks(self, is_query: bool, data: Any) -> List[Any]:
        """Iterates through available models, returning the first successful response."""
        for client in self._clients:
            result = await self._execute_with_retries(client, is_query, data)
            if result is not None:
                return result

            logger.warning(f"Exhausted all retries for model '{client.model}'. Moving to fallback.")

        logger.error("All models failed to generate embeddings.")
        raise RuntimeError("Retry attempts exhausted on all models.")

    async def _execute_with_retries(self, client: OpenAIEmbeddings, is_query: bool, data: Any) -> List[Any] | None:
        """Attempts to generate embeddings with a specific model, retrying on failure."""
        for attempt in range(1, self._retries + 1):
            try:
                if is_query:
                    return await client.aembed_query(data)

                return await client.aembed_documents(data)

            except Exception as e:
                logger.warning(f"Model '{client.model}' failed (Attempt {attempt}/{self._retries}). Error: {e}")

        return None

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generates embeddings for a list of documents."""
        return await self._execute_with_fallbacks(is_query=False, data=texts)

    async def aembed_query(self, text: str) -> List[float]:
        """Generates an embedding for a single query string."""
        return await self._execute_with_fallbacks(is_query=True, data=text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError("This class is async-only. Please use `aembed_documents`.")

    def embed_query(self, text: str) -> List[float]:
        raise NotImplementedError("This class is async-only. Please use `aembed_query`.")
