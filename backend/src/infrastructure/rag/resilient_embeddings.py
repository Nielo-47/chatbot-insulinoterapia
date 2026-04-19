import asyncio
import logging
from dataclasses import dataclass
from typing import Awaitable, Callable, List, Optional, Sequence

import numpy as np
from openai import APITimeoutError, APIConnectionError, APIStatusError, AsyncOpenAI, RateLimitError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EmbeddingProviderConfig:
    name: str
    base_url: str
    api_key: str
    model: str


@dataclass(frozen=True)
class EmbeddingResilienceConfig:
    embedding_dim: int
    timeout_seconds: float
    primary_retries: int
    fallback_retries: int
    primary: EmbeddingProviderConfig
    fallback: EmbeddingProviderConfig


def _normalize_texts(texts: Sequence[str] | str) -> List[str]:
    if isinstance(texts, str):
        return [texts]
    return [str(text) for text in texts]


def _is_retryable_error(error: Exception) -> bool:
    if isinstance(error, (asyncio.TimeoutError, APITimeoutError, APIConnectionError, RateLimitError)):
        return True

    if isinstance(error, APIStatusError):
        status_code = getattr(error, "status_code", None)
        return status_code in {408, 429, 500, 502, 503, 504}

    # OpenRouter sometimes returns response.data = None
    if isinstance(error, TypeError) and "NoneType" in str(error):
        return True

    # Our own check for null response data
    if isinstance(error, ValueError) and "null data" in str(error).lower():
        return True

    message = str(error).lower()
    return any(
        token in message
        for token in (
            "timeout",
            "timed out",
            "connection aborted",
            "connection reset",
            "temporarily unavailable",
            "service unavailable",
            "rate limit",
            "server error",
        )
    )


async def _call_embedding_provider(
    provider: EmbeddingProviderConfig,
    texts: List[str],
    embedding_dim: int,
    timeout_seconds: float,
) -> np.ndarray:
    client = AsyncOpenAI(
        api_key=provider.api_key,
        base_url=provider.base_url,
        timeout=timeout_seconds,
    )

    logger.debug("Calling embedding provider %s with model=%s", provider.name, provider.model)
    response = await client.embeddings.create(model=provider.model, input=texts)
    if response.data is None:
        raise ValueError(f"{provider.name} returned null data")
    embeddings = np.array([item.embedding for item in response.data])

    if embeddings.size == 0:
        raise ValueError(f"{provider.name} returned no embeddings")

    for index, embedding in enumerate(embeddings):
        if len(embedding) != embedding_dim:
            raise ValueError(
                f"{provider.name} returned embedding of size {len(embedding)} for item {index}; expected {embedding_dim}"
            )

    return embeddings


async def embed_with_fallback(texts: Sequence[str] | str, config: EmbeddingResilienceConfig) -> np.ndarray:
    normalized_texts = _normalize_texts(texts)
    last_error: Optional[Exception] = None

    for attempt in range(config.primary_retries + 1):
        try:
            return await _call_embedding_provider(
                config.primary,
                normalized_texts,
                config.embedding_dim,
                config.timeout_seconds,
            )
        except Exception as error:
            last_error = error
            logger.warning(
                "Embedding provider %s failed on attempt %d/%d: %s: %s",
                config.primary.name,
                attempt + 1,
                config.primary_retries + 1,
                type(error).__name__,
                error,
            )
            if attempt < config.primary_retries and _is_retryable_error(error):
                await asyncio.sleep(2**attempt)
                continue
            break

    for attempt in range(config.fallback_retries + 1):
        try:
            logger.warning(
                "Falling back to embedding provider %s after primary failure",
                config.fallback.name,
            )
            return await _call_embedding_provider(
                config.fallback,
                normalized_texts,
                config.embedding_dim,
                config.timeout_seconds,
            )
        except Exception as error:
            last_error = error
            logger.warning(
                "Embedding provider %s failed on attempt %d/%d: %s: %s",
                config.fallback.name,
                attempt + 1,
                config.fallback_retries + 1,
                type(error).__name__,
                error,
            )
            if attempt < config.fallback_retries and _is_retryable_error(error):
                await asyncio.sleep(2**attempt)
                continue
            break

    if last_error is not None:
        raise last_error

    raise RuntimeError("Embedding providers failed without raising a specific error")


def build_embedding_callable(
    *,
    primary: EmbeddingProviderConfig,
    fallback: EmbeddingProviderConfig,
    embedding_dim: int,
    timeout_seconds: float,
    primary_retries: int,
    fallback_retries: int,
) -> Callable[[Sequence[str] | str], Awaitable[np.ndarray]]:
    config = EmbeddingResilienceConfig(
        embedding_dim=embedding_dim,
        timeout_seconds=timeout_seconds,
        primary_retries=primary_retries,
        fallback_retries=fallback_retries,
        primary=primary,
        fallback=fallback,
    )

    async def _embed(texts: Sequence[str] | str) -> np.ndarray:
        return await embed_with_fallback(texts, config)

    return _embed
